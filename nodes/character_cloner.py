from ..operation_logger import log_event, log_stage, logged_operation

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

from ..utils import (
    load_character_info, load_config, config_path, save_config,
    character_dir, sheets_dir, MAIN_DIRS, EMOTIONS,
    safe_join_under, safe_relative_path, privileged_route, file_fingerprint,
    character_storage_lock,
)
from .preview_runtime import run_wizard_job
from .generator_context import serialized_generator

MAX_SOURCE_IMAGES = 1
MAX_GRID_PIXELS = 16 * 1024 * 1024


def _source_image_path(image):
    data = image if isinstance(image, dict) else {"name": image}
    kind = data.get("type", "input")
    if kind not in {"input", "temp", "output"}:
        raise ValueError("Unknown source image type")
    directory = getattr(folder_paths, f"get_{kind}_directory")()
    parts = []
    if data.get("subfolder"):
        parts.append(safe_relative_path(data["subfolder"], "subfolder"))
    parts.append(safe_relative_path(data.get("name"), "image_name"))
    return safe_join_under(directory, *parts)

try:
    from .qwen_vl import get_qwen_vl_chat_handler, parse_wizard_json
    from .qwen_vl import _ensure_qwen_vl_assets, QWEN_VL_MODEL_FILENAME, _start_qwen_vl_download, qwen_vl_download_status
except Exception:
    from nodes.qwen_vl import get_qwen_vl_chat_handler, parse_wizard_json
    from nodes.qwen_vl import _ensure_qwen_vl_assets, QWEN_VL_MODEL_FILENAME, _start_qwen_vl_download, qwen_vl_download_status

# VNCCS Installer (REMOVED: User requested Qwen2)
# Reverted to manual update instructions if needed.

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def _background_rgb(value, default=(255, 255, 255)):
    normalized = str(value or "").strip().lower()
    if normalized == "green":
        return (0, 255, 0)
    if normalized == "blue":
        return (0, 0, 255)
    if normalized.startswith("#") and len(normalized) == 7:
        try:
            return tuple(int(normalized[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    return default

def _composite_pil_alpha(image, background_color=None):
    if image.mode not in ("RGBA", "LA") and not (image.mode == "P" and "transparency" in image.info):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    if not np.any(alpha < 255):
        return rgba.convert("RGB")
    bg = Image.new("RGBA", rgba.size, (*_background_rgb(background_color), 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")

def _emit_cloner_validation_error(unique_id, message):
    if server is None or not unique_id:
        return
    try:
        server.PromptServer.instance.send_sync(
            "vnccs.character_cloner.validation_error",
            {
                "node_id": str(unique_id),
                "code": "SOURCE_IMAGE_REQUIRED",
                "message": message,
            },
        )
    except Exception as exc:
        log_event('warning', component='Cloner', level='warning', message=f'Failed to send validation error event: {exc}', error=str(exc))

class CharacterCloner:
    @classmethod
    def IS_CHANGED(cls, widget_data="{}", **kwargs):
        data = json.loads(widget_data)
        paths = [config_path(data.get("character", "Unknown"))]
        paths.extend(_source_image_path(image) for image in data.get("source_images", []))
        return json.dumps([file_fingerprint(path) for path in paths])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "widget_data": ("STRING", {"default": "{}"}), 
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "*")
    RETURN_NAMES = ("character", "sheets_path", "background")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @logged_operation("Cloner", "import_reference")
    @serialized_generator
    def process(self, widget_data="{}", unique_id=None):
        try:
            data = json.loads(widget_data)
        except:
            data = {}

        # 1. Parse Data
        character_name = data.get("character", "Unknown")
        info = data.get("character_info", {})
        info_owner = str(info.get("name", "") or "").strip()
        if info_owner and info_owner != str(character_name):
            raise ValueError(
                f"Character Cloner received metadata for '{info_owner}' while '{character_name}' is selected. "
                "Reload the selected character before generating."
            )
        source_images = data.get("source_images", []) # List of filenames in input dir
        if not isinstance(source_images, list) or len(source_images) > MAX_SOURCE_IMAGES:
            raise ValueError("Character Cloner accepts only one reference image. Remove extra images or upload a replacement.")
        background_color = info.get("background_color", "White")
        log_event("reference_inputs", character=character_name, images=len(source_images), background=background_color)

        # 4. Process Images
        # Load the validated single reference image
        images_tensors = []
        source_pixels = 0
        if source_images:
            for img_obj in source_images:
                # Handle both string (filename only) and dict (name, subfolder, type)
                if isinstance(img_obj, dict):
                    img_name = img_obj.get("name")
                else:
                    img_name = img_obj

                if not img_name: continue

                try:
                    img_path = _source_image_path(img_obj)
                except ValueError:
                    continue

                if img_path and os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as opened:
                            source_pixels += opened.width * opened.height
                            if source_pixels > MAX_GRID_PIXELS:
                                raise ValueError(f"Character references exceed {MAX_GRID_PIXELS:,} source pixels. Use fewer or smaller images.")
                            i = ImageOps.exif_transpose(opened)
                            i = _composite_pil_alpha(i, background_color)
                            images_tensors.append(i)
                    except ValueError:
                        raise
                    except Exception as exc:
                        log_event('warning', component='Cloner', level='warning', message=f"Failed to load source image '{img_name}': {exc}", error=str(exc))
        
        if images_tensors:
            final_image = pil2tensor(images_tensors[0])

        else:
            message = "Upload a character image in Character Cloner first."
            _emit_cloner_validation_error(unique_id, message)
            raise ValueError(message)

        # 5. Paths
        character_path = character_dir(character_name)
        sheets_path = sheets_dir(character_name)

        # 6. Save Config (if character name is valid)
        if character_name and character_name != "Unknown":
            # Just ensure folder exists
            with character_storage_lock(character_path):
                os.makedirs(character_path, exist_ok=True)
                config = load_config(character_name)
                if not isinstance(config, dict):
                    if os.path.exists(config_path(character_name)):
                        raise ValueError(f"Cannot read the existing configuration for '{character_name}'; refusing to overwrite it.")
                    config = {}
                config.setdefault("folder_structure", {"main_directories": MAIN_DIRS, "emotions": EMOTIONS})
                config.setdefault("config_version", "2.0")
                config.update({
                    "character_info": {**config.get("character_info", {}), **info, "name": character_name},
                    "character_path": character_path,
                })
                if not save_config(character_name, config):
                    raise OSError(f"Character Cloner could not save the configuration for '{character_name}'.")

        # Get background color
        background_color = info.get("background_color", "Green")

        return (final_image, sheets_path, background_color)


if server:
    @server.PromptServer.instance.routes.get("/vnccs/cloner_download_status")
    async def cloner_download_status(request):
        return await qwen_vl_download_status(request)

    @server.PromptServer.instance.routes.post("/vnccs/cloner_download_model")
    @privileged_route
    async def cloner_download_model(request):
        response = _start_qwen_vl_download(require_mmproj=True)
        if response.status == 409:
            return web.Response(status=409, text="Download already in progress")
        return web.json_response({"status": "started"})

    @logged_operation("Cloner", "wizard")
    def _cloner_auto_generate_response(post):
        try:
            import llama_cpp
            import llama_cpp.llama_chat_format
        except ImportError as error:
            return web.json_response({"error": "DEPENDENCY_MISSING", "message": str(error), "model_name": "llama-cpp-python"}, status=500)
        
        lib_ver = getattr(llama_cpp, "__version__", "unknown")
        
        try:
            try:
                HandlerCls = get_qwen_vl_chat_handler(llama_cpp)
            except RuntimeError as exc:
                return web.json_response({
                    "error": "DEPENDENCY_MISSING",
                    "message": f"{exc} Lib Version: {lib_ver}",
                    "model_name": f"llama-cpp-python {lib_ver}",
                }, status=500)
            
            # Proceed with HandlerCls...

            
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
            
            try:
                image_parts = []
                if subfolder:
                    image_parts.append(safe_relative_path(subfolder, "subfolder"))
                image_parts.append(safe_relative_path(img_name, "image_name"))
                image_path = safe_join_under(base_dir, *image_parts)
            except ValueError as e:
                return web.Response(status=400, text=str(e))

            if not image_path or not os.path.exists(image_path):
                return web.Response(status=404, text=f"Image {img_name} not found")

            try:
                model_path, mmproj_path = _ensure_qwen_vl_assets(allow_download=False)
            except Exception as e:
                return web.json_response({
                    "error": "MODEL_MISSING" if isinstance(e, FileNotFoundError) else "MODEL_INVALID",
                    "message": str(e),
                    "model_name": QWEN_VL_MODEL_FILENAME
                }, status=500)
            
             # 4. Initialize Llama
            try:
                log_event("wizard_handler", level="debug", handler=HandlerCls.__name__)

                log_stage("loading_wizard_model", component="Cloner", model=os.path.basename(model_path))
                log_event("wizard_projector", level="debug", model=os.path.basename(mmproj_path))

                chat_handler = HandlerCls(clip_model_path=mmproj_path, enable_thinking=False, verbose=False)
                
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
                prompt_instruction = """Analyze the character in the image and strictly output valid JSON.
Extract visible physical character traits using concise comma-separated tags.
Describe colors and physical traits in your own words; do not choose from presets or a closed list.

Keys:
- sex (string: 'male' or 'female')
- age (int: estimated number)
- race (string: e.g. 'human', 'elf', 'cyborg')
- skin_color (string: describe the actual visible skin color, including unusual or multiple colors)
- hair (string: comma-separated tags for color and style, e.g. 'blue hair, long hair, ponytail')
- eyes (string: comma-separated tags for color and shape, e.g. 'green eyes, tsurime')
- face (string: clearly visible facial features, e.g. 'freckles', 'facial scar', 'makeup')
- body (string: tags for build, e.g. 'slim', 'muscular', 'tall')
- additional_details (string: physical character traits not covered by the other fields, e.g. 'monster arm', 'extra limbs', 'tail', 'wings', 'body markings')
- aesthetics (string: visible art style, e.g. 'anime style, illustration, flat color')
- nsfw (boolean)

Rules:
- Describe only features clearly visible in the image. Do not invent traits or copy the examples.
- Use an empty string for absent, hidden, or uncertain traits. Do not fill a field just to avoid an empty value.
- Determine skin_color from exposed skin, not clothing, background, or assumed human skin tones.
- Preserve unusual skin colors as drawn. Red or pink skin across the face or body is skin_color, not blush.
- Use "pale skin" only for unusually pale/very light skin, never as a generic default.
- Add blush only when distinct localized cheek blush is visible against the surrounding skin color.
- Different colors on exposed body parts can be character traits; do not assume they are gloves or tights.
- additional_details contains only physical character features that do not fit race, skin_color, hair, eyes, face, or body.
- Do not include clothing, footwear, wearable accessories, held objects, pose, actions, facial expressions, camera framing, or background in character trait fields.

Return all keys in a raw JSON object. Do not output the word 'tag' as a value."""

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
                        log_event("diagnostic", component="Cloner", level="debug", message=f'Resized input image to {new_width}x{new_height}')

                    # Save to buffer
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                messages = [
                    {"role": "system", "content": "You extract visible physical character identity, with no outfit or pose descriptions. Output valid JSON only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} 
                    ]}
                ]
                
                log_stage("describing_reference", component="Cloner")
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2
                )
                log_stage("parsing_description", component="Cloner")
                
                content = response["choices"][0]["message"]["content"]
                log_event("wizard_output", component="Cloner", level="debug", output=content)
                
                # 6. Robust JSON Extraction
                if not content or not content.strip():
                     log_event("warning", component="Cloner", level="warning", message='Error: Empty response from LLM')
                     return web.json_response({"error": "INVALID_RESPONSE", "message": "The image wizard returned an empty response. Please try again."}, status=502)

                data = parse_wizard_json(content)

                # Only structured character traits may reach the character fields.
                if isinstance(data, dict):
                    # Ensure keys exist? Frontend handles missing keys.
                    log_event("traits_ready", component="Cloner", fields=len(data))
                    return web.json_response(data)
                else:
                    log_event("warning", component="Cloner", level="warning", message='Failed to extract character JSON.')
                    return web.json_response({"error": "INVALID_RESPONSE", "message": "The image wizard did not return a character JSON object. Please try again."}, status=502)

            except Exception as e:
                log_event('exception', component='Cloner', level='debug', exc_info=True, message='Operation failed', error=str(e))
                
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
            log_event('exception', component='Cloner', level='debug', exc_info=True, message='Operation failed', error=str(e))
            return web.Response(status=500, text=str(e))


    @server.PromptServer.instance.routes.post("/vnccs/cloner_auto_generate")
    @privileged_route
    async def cloner_auto_generate(request):
        try:
            post = await request.json()
        except (ValueError, TypeError):
            return web.json_response({"error": "Invalid JSON request"}, status=400)
        if not isinstance(post, dict):
            return web.json_response({"error": "Request must be an object"}, status=400)
        return await run_wizard_job(_cloner_auto_generate_response, post, "cloner")


NODE_CLASS_MAPPINGS = {
    "CharacterCloner": CharacterCloner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterCloner": "VNCCS Character Cloner",
}
