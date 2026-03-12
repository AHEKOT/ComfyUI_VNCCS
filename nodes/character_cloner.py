import os
import json
import shutil
import urllib.request
import subprocess
import platform
import sys
import importlib.util
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

# VNCCS Installer (REMOVED: User requested Qwen2)
# Reverted to manual update instructions if needed.

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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "*")
    RETURN_NAMES = ("character", "positive_prompt", "negative_prompt", "sheets_path", "faces_path", "face_details", "background")
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
            # Smart Grid: Minimize aspect ratio difference from 1.0 (Square)
            count = len(images_tensors)
            max_w = max(img.width for img in images_tensors)
            max_h = max(img.height for img in images_tensors)

            best_cols = 1
            best_rows = count
            best_diff = float('inf')

            # Naively try all column counts
            for c in range(1, count + 1):
                r = int(np.ceil(count / c))
                # Grid dimensions if we use this layout (assuming max_w/max_h cells)
                w_total = c * max_w
                h_total = r * max_h
                
                ratio = w_total / h_total  # width / height
                
                # Symmetric score: penalize deviation from 1.0 equally for tall vs wide
                # e.g. ratio 0.5 -> 2.0; ratio 2.0 -> 2.0
                symmetric_ratio = ratio if ratio >= 1 else 1 / ratio
                
                # Tie-breaker: Prefer square grid of CELLS (c approx r)
                # This helps when 1x4 (ratio 0.5) and 2x2 (ratio 2.0) have same symmetric score.
                # 2x2 is "structurally" squarer.
                grid_diff = abs(c - r)
                
                # Weighted score: Main priority is image aspect, secondary is grid shape
                score = symmetric_ratio + (grid_diff * 0.01)

                if score < best_diff:
                    best_diff = score
                    best_cols = c
                    best_rows = r
            
            cols = best_cols
            rows = best_rows
            
            grid_w = cols * max_w
            grid_h = rows * max_h
            grid = Image.new("RGB", (grid_w, grid_h), "black")
            
            for idx, img in enumerate(images_tensors):
                r = idx // cols
                c = idx % cols
                
                # Center image in cell
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

        # Get background color
        background_color = info.get("background_color", "Green")

        return (final_image, positive_prompt, negative_prompt, sheets_path, faces_path, face_details, background_color)


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
        
        # User requested Revert to Qwen2-VL (We use Qwen2.5-VL as the best "Qwen2" variant available GGUF)
        # Using bartowski's build which is reliable for comfy/llama-cpp
        
        base_path = folder_paths.models_dir
        llm_dir = os.path.join(base_path, "LLM")
        
        # Target: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
        # Repo: bartowski/Qwen2.5-VL-7B-Instruct-GGUF
        
        url_model = "https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        dest_model = os.path.join(llm_dir, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
        
        # Target: mmproj
        url_proj = "https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
        dest_proj = os.path.join(llm_dir, "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")

        def run_sequences():
            # 1. Download Model
            download_file(url_model, dest_model, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
            
            # 2. Download Vision Projector
            if DOWNLOAD_STATUS["status"] == "completed":
                download_file(url_proj, dest_proj, "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")

        t = threading.Thread(target=run_sequences)
        t.start()
        
        return web.json_response({"status": "started"})


    @server.PromptServer.instance.routes.post("/vnccs/cloner_auto_generate")
    async def cloner_auto_generate(request):
        import sys
        import llama_cpp
        import llama_cpp.llama_chat_format
        
        # DEBUG INFO
        lib_ver = getattr(llama_cpp, "__version__", "unknown")
        py_path = sys.executable
        available_handlers = dir(llama_cpp.llama_chat_format)
        
        print(f"[VNCCS] Auto-Gen Debug: Ver={lib_ver}, Py={py_path}")
        
        try:
            # 0. Dynamic Import Check
            # Try to find ANY QwenVL handler
            HandlerCls = None
            
            # 1. Try Qwen2.5 (Newest)
            if hasattr(llama_cpp.llama_chat_format, "Qwen25VLChatHandler"):
                HandlerCls = llama_cpp.llama_chat_format.Qwen25VLChatHandler
            
            # 2. Try Qwen2 (Standard)
            elif hasattr(llama_cpp.llama_chat_format, "Qwen2VLChatHandler"):
                HandlerCls = llama_cpp.llama_chat_format.Qwen2VLChatHandler
                
            # 3. Try Qwen (Old/Other)
            elif hasattr(llama_cpp.llama_chat_format, "Qwen25VLChatHandler"):
                 HandlerCls = llama_cpp.llama_chat_format.Qwen25VLChatHandler
            
            # If still nothing, inspect for partial matches?
            if not HandlerCls:
                # Fallback scan
                for attr in available_handlers:
                    if "Qwen" in attr and "VL" in attr and "Handler" in attr:
                        HandlerCls = getattr(llama_cpp.llama_chat_format, attr)
                        break
            
            if not HandlerCls:
                 msg = f"No QwenVL handlers found. Lib Version: {lib_ver}. Available: { [h for h in available_handlers if 'Handler' in h] }"
                 return web.json_response({
                     "error": "DEPENDENCY_MISSING", 
                     "message": msg,
                     "model_name": f"llama-cpp-python {lib_ver}"
                 }, status=500)
            
            # Proceed with HandlerCls...

            
            post = await request.json()
            image_data = post.get("image_name")
            
            if not image_data:
                return web.Response(status=400, text="No image provided")
            
            # ... (Image Resolution Code same as before) ...
            # 1. Locate Image
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
            
            # Search for Qwen2.5 / Qwen2
            possible_names = [
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "Qwen2-VL-7B-Instruct-Q4_K_M.gguf", 
                "qwen2-vl-7b-instruct-q4_k_m.gguf"
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
                 # Return specific JSON error for frontend to trigger download of Qwen2.5
                 return web.json_response({
                     "error": "MODEL_MISSING",
                     "message": "No QwenVL GGUF model found.",
                     "model_name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
                 }, status=404)

            # 3. Locate MMProj (Vision Adapter)
            mmproj_path = None
            if model_path:
                model_dir = os.path.dirname(model_path)
                # Check for 2.5/2 specific projectors
                cands = [
                    "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
                    "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
                ]
                for c in cands:
                    cp = os.path.join(model_dir, c)
                    if os.path.exists(cp):
                        mmproj_path = cp
                        break
                
                if not mmproj_path:
                    # Flexible Search
                    for f in os.listdir(model_dir):
                        if "mmproj" in f and f.endswith(".gguf") and "Qwen" in f:
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

             # 4. Initialize Llama
            try:
                # Select Handler based on model name
                HandlerCls = None
                
                # Check for Qwen2VL handler (newer llama-cpp-python)
                try:
                    from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                    if "Qwen" in model_path:
                        HandlerCls = Qwen2VLChatHandler
                        print("[VNCCS] Using Qwen2VLChatHandler")
                except ImportError:
                    pass

                # Fallback to Llava1.5
                if not HandlerCls:
                     from llama_cpp.llama_chat_format import Llava15ChatHandler
                     HandlerCls = Llava15ChatHandler
                     print("[VNCCS] Using Llava15ChatHandler (Fallback)")

                # Debug print
                print(f"[VNCCS] Loading Model: {model_path}")
                print(f"[VNCCS] Loading MMProj: {mmproj_path}")

                # Ensure Handler
                if not HandlerCls:
                    raise ImportError("No suitable ChatHandler found for Qwen/Llava.")

                chat_handler = HandlerCls(clip_model_path=mmproj_path, verbose=False)
                
                llm = llama_cpp.Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=4096, # Safe default for VL
                    n_gpu_layers=-1, # Auto
                    verbose=False
                )
                # ... Inference code ...
                if not llm:
                    raise RuntimeError("Failed to initialize Llama model.")

                # 5. Run Inference
                # 5. Run Inference
                # Explicit Instruction for JSON
                prompt_instruction = """Analyze the image and strictly output valid JSON. 
Use Danbooru-style tags for descriptions.

Keys:
- sex (string: 'male' or 'female')
- age (int: estimated number)
- race (string: e.g. 'human', 'elf', 'cyborg')
- skin_color (string: e.g. 'pale skin', 'tan skin', 'dark skin')
- hair (string: comma-separated tags for color and style, e.g. 'blue hair, long hair, ponytail')
- eyes (string: comma-separated tags for color and shape, e.g. 'green eyes, tsurime')
- face (string: tags for features, e.g. 'blush', 'scars', 'makeup')
- body (string: tags for build, e.g. 'slim', 'muscular', 'tall')
- additional_details (string: tags for clothing, accessories, pose, e.g. 'wearing suit, sitting, holding sword')
- aesthetics (string: high quality tags e.g. 'masterpiece, best quality, anime style')
- nsfw (boolean)

Structure the response as a raw JSON object. Do not output the word 'tag' as a value. DESCRIBE the character."""

                # Helper for Base64 with Resizing (Max 512px)
                import base64
                import io
                from PIL import Image

                with Image.open(image_path) as img:
                    # Convert to RGB to avoid alpha issues with JPEG
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    # Resize if needed
                    max_size = 512
                    width, height = img.size
                    if width > max_size or height > max_size:
                        if width > height:
                            new_width = max_size
                            new_height = int(height * (max_size / width))
                        else:
                            new_height = max_size
                            new_width = int(width * (max_size / height))
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        print(f"[VNCCS] Resized input image to {new_width}x{new_height}")

                    # Save to buffer
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                messages = [
                    {"role": "system", "content": "You are a character description specialist. Analyze the image and output valid JSON only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} 
                    ]}
                ]
                
                print(f"[VNCCS] Starting Inference...")
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2
                )
                print(f"[VNCCS] Inference Complete. Processing Response...")
                
                content = response["choices"][0]["message"]["content"]
                print(f"[VNCCS] Raw LLM Output: {content}")
                
                # 6. Robust JSON Extraction
                if not content or not content.strip():
                     print("[VNCCS] Error: Empty response from LLM")
                     return web.json_response({"additional_details": "Error: Empty response from LLM. check console."})

                data = None

                # Attempt 1: json_repair (if installed)
                try:
                    import json_repair
                    data = json_repair.loads(content)
                    print(f"[VNCCS] json_repair result type: {type(data)}")
                except ImportError:
                    print("[VNCCS] json_repair not installed.")
                except Exception as e:
                    print(f"[VNCCS] json_repair failed: {e}")

                # Normalize data (handle list of dicts)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]

                # Attempt 2: Standard JSON (if Attempt 1 failed or returned non-dict)
                if not isinstance(data, dict):
                    print("[VNCCS] Fallback to standard JSON parsing...")
                    try:
                        import json
                        json_str = content
                        if "```json" in content:
                            json_str = content.split("```json")[1].split("```")[0]
                        elif "```" in content:
                            json_str = content.split("```")[1].split("```")[0]
                        
                        data = json.loads(json_str.strip())
                        print("[VNCCS] Standard JSON parse success.")
                    except Exception as e:
                        print(f"[VNCCS] Standard JSON parse failed: {e}")

                # Final Check: If valid dict, return it. Else, raw content.
                if isinstance(data, dict):
                    # Ensure keys exist? Frontend handles missing keys.
                    print(f"[VNCCS] Final JSON Keys: {list(data.keys())}")
                    return web.json_response(data)
                else:
                    print("[VNCCS] Failed to extract JSON. Returning raw content.")
                    return web.json_response({"additional_details": content})

            except Exception as e:
                import traceback
                print(f"[VNCCS] CRITICAL ERROR IN INFERENCE:")
                traceback.print_exc()
                
                # Detect specific Qwen/Llama errors
                err_msg = str(e)
                err_code = "INFERENCE_ERROR"

                # Only trigger "Missing Model" dialog if it's actually about files
                if "mmproj" in err_msg.lower() or "not found" in err_msg.lower() or "no file" in err_msg.lower():
                     err_code = "MMPROJ_MISSING"
                
                return web.json_response({
                    "error": err_code, 
                    "message": f"Engine Error: {err_msg}",
                    "model_name": "Qwen2VL (Check Console)"
                }, status=500)

        except Exception as e:
            traceback.print_exc()
            return web.Response(status=500, text=str(e))
