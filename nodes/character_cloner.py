import os
import json
import torch
import folder_paths
import comfy.sd
import comfy.utils
import server
from aiohttp import web
from PIL import Image, ImageOps
import numpy as np
import traceback

from .character_creator_v2 import CharacterCreatorV2
from ..utils import (
    load_character_info, save_config, build_face_details, 
    character_dir, sheets_dir, faces_dir, MAIN_DIRS, EMOTIONS
)



def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class CharacterCloner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "widget_data": ("STRING", {"default": "{}"}), 
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("character", "positive_prompt", "negative_prompt", "sheets_path", "faces_path", "face_details")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    def process(self, widget_data="{}", unique_id=None):
        try:
            data = json.loads(widget_data)
        except:
            data = {}

        # 1. Parse Data
        character_name = data.get("character", "Unknown")
        info = data.get("character_info", {})
        source_images = data.get("source_images", []) # List of filenames in input dir

        # 2. Construct Prompts (Reuse V2 logic)
        positive_prompt, negative_prompt = CharacterCreatorV2.construct_prompt(info)
        
        # 4. Process Images
        # Load all source images, make a grid
        images_tensors = []
        if source_images:
            for img_obj in source_images:
                # Handle both string (filename only) and dict (name, subfolder, type)
                if isinstance(img_obj, dict):
                    img_name = img_obj.get("name")
                    subfolder = img_obj.get("subfolder", "")
                    img_type = img_obj.get("type", "input")
                else:
                    img_name = img_obj
                    subfolder = ""
                    img_type = "input"

                if not img_name: continue

                # Manuall resolve path to avoid TypeError
                if img_type == "input":
                    base_dir = folder_paths.get_input_directory()
                elif img_type == "temp":
                    base_dir = folder_paths.get_temp_directory()
                else:
                    base_dir = folder_paths.get_output_directory()
                
                if subfolder:
                    img_path = os.path.join(base_dir, subfolder, img_name)
                else:
                    img_path = os.path.join(base_dir, img_name)

                if img_path and os.path.exists(img_path):
                    i = Image.open(img_path)
                    i = ImageOps.exif_transpose(i)
                    if i.mode != "RGB": i = i.convert("RGB")
                    images_tensors.append(i)
        
        if images_tensors:
            # Create a simple grid: standard collage
            # Find closest square grid
            count = len(images_tensors)
            cols = int(np.ceil(np.sqrt(count)))
            rows = int(np.ceil(count / cols))
            
            # Find max dimensions for tiles
            max_w = max(img.width for img in images_tensors)
            max_h = max(img.height for img in images_tensors)
            
            # Resize all to fit in max_w, max_h (contain) or just use max dimensions?
            # User wants "one large image". Let's preserve individual aspect ratios but fit in grid cells?
            # Or just stitch them? Stitching varies wildy if sizes differ.
            # Best approach for "Character Sheet" feel: Resize all to same height, tile horizontally? 
            # Or use standard grid logic:
            
            grid_w = cols * max_w
            grid_h = rows * max_h
            grid = Image.new("RGB", (grid_w, grid_h), "black")
            
            for idx, img in enumerate(images_tensors):
                r = idx // cols
                c = idx % cols
                
                # Center image in checking
                x = c * max_w + (max_w - img.width) // 2
                y = r * max_h + (max_h - img.height) // 2
                grid.paste(img, (x, y))
            
            final_image = pil2tensor(grid)

        else:
            # Empty image
            final_image = torch.zeros((1, 512, 512, 3))

        # 5. Paths
        character_path = character_dir(character_name)
        sheets_path = sheets_dir(character_name)
        faces_path = faces_dir(character_name)
        face_details = build_face_details(info)

        # 6. Save Config (if character name is valid)
        if character_name and character_name != "Unknown":
            # Just ensure folder exists
            os.makedirs(character_path, exist_ok=True)
            # We don't necessarily overwrite config unless user explicitly saved?
            # CharacterCreatorV2 saves on process. We stick to that pattern.
            config = {
                "character_info": info,
                "folder_structure": { "main_directories": MAIN_DIRS, "emotions": EMOTIONS },
                "character_path": character_path,
                "config_version": "2.0"
            }
            save_config(character_name, config)

        return (final_image, positive_prompt, negative_prompt, sheets_path, faces_path, face_details)


# --------------------------------------------------------------------------------
# API: Download Model Logic
# --------------------------------------------------------------------------------
import threading
import time
import requests

DOWNLOAD_STATUS = {
    "status": "idle", # idle, downloading, completed, error
    "progress": 0,
    "current_file": "",
    "total_size": 0,
    "downloaded_size": 0,
    "error": ""
}

def download_file(url, dest_path, file_label="File"):
    global DOWNLOAD_STATUS
    try:
        DOWNLOAD_STATUS["status"] = "downloading"
        DOWNLOAD_STATUS["current_file"] = file_label
        DOWNLOAD_STATUS["progress"] = 0
        DOWNLOAD_STATUS["error"] = ""
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        DOWNLOAD_STATUS["total_size"] = total_size
        DOWNLOAD_STATUS["downloaded_size"] = 0
        
        block_size = 1024 * 1024 # 1MB
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                DOWNLOAD_STATUS["downloaded_size"] += len(data)
                if total_size > 0:
                    DOWNLOAD_STATUS["progress"] = int((DOWNLOAD_STATUS["downloaded_size"] / total_size) * 100)
                    
        DOWNLOAD_STATUS["status"] = "completed"
        DOWNLOAD_STATUS["progress"] = 100
        print(f"[{file_label}] Download completed: {dest_path}")
        
    except Exception as e:
        DOWNLOAD_STATUS["status"] = "error"
        DOWNLOAD_STATUS["error"] = str(e)
        print(f"[{file_label}] Download error: {e}")

if server:
    @server.PromptServer.instance.routes.get("/vnccs/cloner_download_status")
    async def cloner_download_status(request):
        return web.json_response(DOWNLOAD_STATUS)

    @server.PromptServer.instance.routes.post("/vnccs/cloner_download_model")
    async def cloner_download_model(request):
        global DOWNLOAD_STATUS
        if DOWNLOAD_STATUS["status"] == "downloading":
             return web.Response(status=409, text="Download already in progress")
        
        # Determine what to download? 
        # For now, default to Qwen2.5-VL-7B (better license/performance) or the Qwen3 one from before?
        # User had "QwenVL GGUF".
        # Let's download a small but good one. Qwen2-VL-2B or 7B?
        # Qwen2-VL-2B-Instruct-Q4_K_M.gguf is ~1.5GB. 
        # Qwen2-VL-7B-Instruct-Q4_K_M.gguf is ~4.5GB.
        # Let's replicate the structure we looked for: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
        
        # NOTE: Using a specific hugginface link.
        # Qwen2-VL-7B-Instruct-GGUF/Qwen2-VL-7B-Instruct-Q4_K_M.gguf
        # Need separate mmproj? Qwen2-VL usually bundles logic or needs specific mmproj.
        # ComfyUI-GGUF usually needs both if using llama-cpp-python < ???
        # Newer llama-cpp-python handles split weights if named correctly?
        
        # Simplest path: Qwen2-VL-2B-Instruct-Q4_K_M.gguf + mmproj-Qwen2-VL-2B-Instruct-f16.gguf?
        # Let's try Qwen2-VL-2B-Instruct-Q4_K_M.gguf (Lightweight for auto-gen)
        # URL: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF/resolve/main/qwen2-vl-2b-instruct-q4_k_m.gguf
        # Wait, Qwen-VL-Chat-Int4 is old.
        
        # Let's use a known working fallback: Qwen2-VL-7B-Instruct-Q4_K_M.gguf
        # Repo: https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF
        # File: Qwen2-VL-7B-Instruct-Q4_K_M.gguf (4.79 GB)
        
        # If user wants it, we download it.
        # Destination: models/LLM
        base_path = folder_paths.models_dir
        llm_dir = os.path.join(base_path, "LLM")
        
        url_model = "https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
        dest_model = os.path.join(llm_dir, "Qwen2-VL-7B-Instruct-Q4_K_M.gguf")
        
        # Threaded download
        def run_sequences():
            download_file(url_model, dest_model, "Qwen2-VL-7B-Instruct-Q4_K_M.gguf")
            # If completed, check mmproj? usually contained in later GGUF or we download it separately?
            # llava-style clip?
            # bartowski usually provides separate mmproj if needed?
            # Most modern GGUF Qwen2VL require 'mmproj-Qwen2-VL-7B-Instruct-f16.gguf'
            # Let's download that too.
            if DOWNLOAD_STATUS["status"] == "completed":
                url_proj = "https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
                dest_proj = os.path.join(llm_dir, "mmproj-Qwen2-VL-7B-Instruct-f16.gguf")
                download_file(url_proj, dest_proj, "mmproj-Qwen2-VL-7B-Instruct-f16.gguf")

        t = threading.Thread(target=run_sequences)
        t.start()
        
        return web.json_response({"status": "started"})


    @server.PromptServer.instance.routes.post("/vnccs/cloner_auto_generate")
    async def cloner_auto_generate(request):
        try:
            # 0. Dynamic Import
            try:
                from llama_cpp import Llama
            except ImportError:
                return web.Response(status=500, text="llama_cpp module not found. Please install llama-cpp-python.")
            
            # Select Handler for Qwen
            HandlerCls = None
            try:
                from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                HandlerCls = Qwen2VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    HandlerCls = Llava15ChatHandler # Fallback?
                except:
                    pass
            
            # NOTE: Relaxed Check because updated library might have different names
            # If specifically looking for Qwen2.5/VL, ensure we have it.
            
            post = await request.json()
            image_data = post.get("image_name")
            
            if not image_data:
                return web.Response(status=400, text="No image provided")
            
            # ... (Image Resolution Code same as before) ...
            # 1. Locate Image
            # Handle dict vs string
            if isinstance(image_data, dict):
                img_name = image_data.get("name")
                subfolder = image_data.get("subfolder", "")
                img_type = image_data.get("type", "input")
            else:
                img_name = image_data
                subfolder = ""
                img_type = "input"

            if img_type == "input": base_dir = folder_paths.get_input_directory()
            elif img_type == "temp": base_dir = folder_paths.get_temp_directory()
            else: base_dir = folder_paths.get_output_directory()
            
            if subfolder: image_path = os.path.join(base_dir, subfolder, img_name)
            else: image_path = os.path.join(base_dir, img_name)

            if not image_path or not os.path.exists(image_path):
                return web.Response(status=404, text=f"Image {img_name} not found")

            # 2. Locate Model
            base_path = folder_paths.models_dir
            model_path = None
            
            # Update search names for the one we download
            possible_names = [
                "Qwen2-VL-7B-Instruct-Q4_K_M.gguf", 
                "qwen2-vl-7b-instruct-q4_k_m.gguf",
                "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
            ]
            
            search_dirs = [os.path.join(base_path, "LLM"), os.path.join(base_path, "llm"), base_path]
            
            for d in search_dirs:
                if not os.path.exists(d): continue
                for n in possible_names:
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        model_path = p
                        break
                if model_path: break
            
            if not model_path:
                 # Return specific JSON error for frontend to trigger download
                 return web.json_response({
                     "error": "MODEL_MISSING",
                     "message": "No QwenVL GGUF model found.",
                     "model_name": "Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
                 }, status=404)

            # 3. Locate MMProj (Vision Adapter)
            mmproj_path = None
            if model_path:
                model_dir = os.path.dirname(model_path)
                # If specific known model, look for specific mmproj
                if "Qwen2-VL-7B" in os.path.basename(model_path):
                     cand = os.path.join(model_dir, "mmproj-Qwen2-VL-7B-Instruct-f16.gguf")
                     if os.path.exists(cand): mmproj_path = cand
                
                if not mmproj_path:
                    # Flexible Search
                    for f in os.listdir(model_dir):
                        if "mmproj" in f and f.endswith(".gguf"):
                            mmproj_path = os.path.join(model_dir, f)
                            break
            
            # 4. Inference
            system_prompt = "You are a character description assistant. Analyze the image and extract the character's physical attributes into a JSON format."
            
            # Convert formatted image to base64 data URI
            with open(image_path, "rb") as f:
                import base64
                b64 = base64.b64encode(f.read()).decode("utf-8")
                img_uri = f"data:image/png;base64,{b64}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze the character. Output JSON with keys: sex, age (int), race, skin_color, hair, eyes, face, body, additional_details, aesthetics (style tags), nsfw (boolean)."},
                    {"type": "image_url", "image_url": {"url": img_uri}}
                ]}
            ]

            # Init Handler
            if not HandlerCls: HandlerCls = Llava15ChatHandler # Fallback safety
            
            try:
                chat_handler = HandlerCls(clip_model_path=mmproj_path, verbose=False)
            except Exception as e:
                # If mmproj missing but required
                return web.json_response({"error": "MMPROJ_MISSING", "message": str(e)}, status=500)
            
            # Init Model
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1, 
                chat_handler=chat_handler,
                verbose=False
            )
            
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.1
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # 5. Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            try:
                data = json.loads(content)
                return web.json_response(data)
            except:
                # If naive parse fails, try to return basic dict with full text
                return web.json_response({"additional_details": content})

        except Exception as e:
            traceback.print_exc()
            return web.Response(status=500, text=str(e))
