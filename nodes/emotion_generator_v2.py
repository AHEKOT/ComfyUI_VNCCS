
import os
import json
import re
import torch
import glob
from itertools import chain

try:
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, ensure_costume_structure, EMOTIONS,
        apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
        sheets_dir, load_costume_info, list_costumes
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, ensure_costume_structure, EMOTIONS,
        apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
        sheets_dir, load_costume_info, list_costumes
    )

# --- ComfyUI Server Imports ---
try:
    import server
    from aiohttp import web
except ImportError:
    print("VNCCS Warning: Running outside ComfyUI environment. API routes will not be registered.")
    server = None
    web = None

# --------------------------------------------------------------------
# Helper to get emotion config path
def get_custom_node_path():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Helper function to load the emotions JSON
def load_emotions_data():
    """Load emotions.json from the emotions-config folder."""
    config_path = os.path.join(get_custom_node_path(), "emotions-config", "emotions.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"emotions.json not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

# --------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------

if server:
    @server.PromptServer.instance.routes.get("/vnccs/get_emotions")
    async def get_emotions_config(request):
        try:
            data = load_emotions_data()
            return web.json_response(data)
        except Exception as e:
            return web.Response(status=500, text=f"Error loading emotions.json: {e}")

    @server.PromptServer.instance.routes.get("/vnccs/get_character_costumes")
    async def get_character_costumes(request):
        character = request.rel_url.query.get("character", "")
        if not character:
            return web.json_response([])
        
        costumes = list_costumes(character)
        return web.json_response(costumes)

    @server.PromptServer.instance.routes.get("/vnccs/get_character_sheet_preview")
    async def get_character_sheet_preview(request):
        character = request.rel_url.query.get("character", "")
        if not character:
             return web.Response(status=404)
        
        try:
            # 1. Find the Sheet
            # Try Naked/neutral first
            sheet_dir_path = os.path.join(character_dir(character), "Sheets", "Naked", "neutral")
            
            if not os.path.exists(sheet_dir_path):
                 found = False
                 base = os.path.join(character_dir(character), "Sheets")
                 if os.path.exists(base):
                     # Sort costumes to be deterministic
                     costumes = sorted(os.listdir(base))
                     for costume in costumes:
                         path = os.path.join(base, costume, "neutral")
                         if os.path.isdir(path):
                             sheet_dir_path = path
                             found = True
                             break
                 if not found:
                     return web.Response(status=404, text="Sheet not found")

            # 2. Find the best file (highest index)
            pattern = os.path.join(sheet_dir_path, "sheet_neutral_*.png")
            files = glob.glob(pattern)
            if not files:
                 return web.Response(status=404, text="No sheet images found")
            
            def get_index(f):
                m = re.search(r'(\d+)', os.path.basename(f))
                return int(m.group(1)) if m else 0
            
            files.sort(key=get_index)
            best_file = files[-1]

            # 3. Load and Crop
            import cv2
            from PIL import Image
            import io

            # Using Pillow for simple grid cropping
            img = Image.open(best_file)
            w, h = img.size
            
            # Layout: 2 Rows, 6 Columns
            # We want Index 11 (The last one) -> Row 1 (2nd row), Col 5 (6th col)
            
            item_w = w // 6
            item_h = h // 2
            
            row = 1
            col = 5
            
            left = col * item_w
            upper = row * item_h
            right = left + item_w
            lower = upper + item_h
            
            crop = img.crop((left, upper, right, lower))
            
            img_byte_arr = io.BytesIO()
            crop.save(img_byte_arr, format='PNG')
            return web.Response(body=img_byte_arr.getvalue(), content_type='image/png')
            
        except Exception as e:
            return web.Response(status=500)

    @server.PromptServer.instance.routes.get("/vnccs/get_emotion_image")
    async def get_emotion_image(request):
        name = request.rel_url.query.get("name", "")
        if not name or ".." in name or "/" in name or "\\" in name:
            return web.Response(status=400)
            
        from urllib.parse import unquote
        name = unquote(name).strip() 
        
        # Absolute path resolution logic
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        image_path = os.path.join(root_dir, "emotions-config", "images", f"{name}.png")
        
        if not os.path.exists(image_path):
            return web.Response(status=404)
            
        return web.FileResponse(image_path)


class EmotionGeneratorV2:
    
    EMOTIONS_DATA = None
    SAFE_NAME_MAP = None

    @classmethod
    def __init__(cls):
        cls._setup_emotions_data()

    @classmethod
    def _setup_emotions_data(cls):
        if cls.SAFE_NAME_MAP is not None:
            return

        try:
            config_path = os.path.join(get_custom_node_path(), "emotions-config", "emotions.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            safe_name_map = {}
            for category, emotion_list in data.items():
                for emotion in emotion_list:
                    if 'safe_name' in emotion and emotion['safe_name']:
                        safe_name_map[emotion['safe_name']] = {
                                "key": emotion['key'],
                                "description": emotion['description'],
                                "natural_prompt": emotion.get('natural_prompt', ''),
                                "category": category
                        }
            cls.SAFE_NAME_MAP = safe_name_map
        except Exception as e:
            print(f"[VNCCS] ERROR: Failed to load emotions data: {e}")
            cls.SAFE_NAME_MAP = {}

    @classmethod
    def INPUT_TYPES(cls):
        characters = list_characters()
        if not characters:
            characters = ["Character Name"]

        return {
            "required": {
                "prompt_style": (["SDXL Style", "QWEN Style"], {"default": "SDXL Style"}),
                "character": (characters, {"default": characters[0] if characters else "Character Name"}),
                # JSON lists passed as strings from frontend
                "costumes_data": ("STRING", {"default": "[]", "multiline": False}),
                "emotions_data": ("STRING", {"default": "[]", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "MASK")
    RETURN_NAMES = ("images", "emotions_out", "face_output_paths", "sheet_output_paths", "positive_prompt",
                    "negative_prompt", "seed", "masks")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False, True)
    FUNCTION = "generate_emotions_v2"
    CATEGORY = "VNCCS"

    def generate_emotions_v2(self, prompt_style, character, costumes_data, emotions_data):
        
        try:
            selected_costumes = json.loads(costumes_data)
        except:
            selected_costumes = []

        try:
            selected_emotions = json.loads(emotions_data)
        except:
            selected_emotions = []

        # --- SETUP ---
        if self.SAFE_NAME_MAP is None:
            self._setup_emotions_data()
            
        character_path = character_dir(character)
        info = load_character_info(character)
        sheets_dir_path = os.path.join(character_path, "Sheets")
        
        images = []
        emotions_out = []
        face_output_paths = []
        sheet_output_paths = []
        masks = []
        
        # Helper to get info
        if info:
            aesthetics = info.get("aesthetics", "")
            background_color = info.get("background_color", "blue")
            sex = info.get("sex", "")
            age = info.get("age", 18)
            race = info.get("race", "")
            eyes = info.get("eyes", "")
            hair = info.get("hair", "")
            face_features = info.get("face", "")
            body = info.get("body", "")
            skin_color = info.get("skin_color", "")
            additional_details = info.get("additional_details", "")
            lora_prompt = info.get("lora_prompt", "")
            
            negative_prompt = info.get("negative_prompt", "") 
            negative_prompt = negative_prompt + ", (facial droplet), (water drop), (water), (water droplets), (water drops)"
            
            config_seed = info.get("seed", 0)
            seed = generate_seed(config_seed)
            base_negative_prompt = negative_prompt
            positive_prompt = aesthetics
        else:
             seed = 0
             base_negative_prompt = ""
             positive_prompt = ""
             print(f"Character info not found for {character}")

        # --- GENERATION LOOP ---
        for costume in selected_costumes:
            # Check if valid costume dir
            # Note: selected_costumes comes from frontend which lists them via API
            
            costume_info = load_costume_info(character, costume)
            
            # Load neutral sheet to get the base image
            img_tensor, mask_tensor = load_character_sheet(character, costume, "neutral", with_mask=True)
            
            if img_tensor is None:
                print(f"Failed to load image for costume {costume}")
                continue

            for emotion_key in selected_emotions:
                
                emotion_details_data = self.SAFE_NAME_MAP.get(emotion_key)
                if not emotion_details_data:
                    print(f"Warning: Unknown emotion key {emotion_key}")
                    emotion_description = "unknown emotion"
                    natural_prompt = ""
                else:
                    emotion_description = emotion_details_data['description']
                    natural_prompt = emotion_details_data.get('natural_prompt', '')

                # Path construction
                face_dir = os.path.join(character_path, "Faces", costume, emotion_key)
                sheet_dir = os.path.join(character_path, "Sheets", costume, emotion_key)
                
                face_output_path = os.path.join(face_dir, f"face_{emotion_key}_")
                sheet_output_path = os.path.join(sheet_dir, f"sheet_{emotion_key}_")
                
                # Prompt Building (Same as V1)
                positive_prompt = f"{aesthetics}"
                if background_color: positive_prompt += f", {background_color} background"
                if race: positive_prompt += f", ({race} race:1.0)"
                if hair: positive_prompt += f", ({hair} hair:1.0)"
                if eyes: positive_prompt += f", ({eyes} eyes:1.0)"
                if body: positive_prompt += f", ({body} body:1.0)"
                if skin_color: positive_prompt += f", ({skin_color} skin:1.0)"
                if additional_details: positive_prompt += f", ({additional_details})"
                
                positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
                negative_prompt = f"{base_negative_prompt}, {gender_negative}"
                positive_prompt = append_age(positive_prompt, age, sex)
                
                if lora_prompt: positive_prompt += f", {lora_prompt}"

                face_details = build_face_details(info)
                
                if prompt_style == "QWEN Style":
                    # Format: SDXL Styled tags: ***description***. Emotion description: ***natural_prompt***. ***face_details***
                    emotion_text = f"Change emotion: {emotion_key}. \r\n SDXL Styled emotion tags: {emotion_description}. \r\n Emotion description: {natural_prompt} \r\n Character face details: {face_details}"
                else:
                    # SDXL Style (Original logic)
                    emotion_text = f"({emotion_key}, {emotion_description}), {face_details}"
                
                # Mask handling
                curr_mask_tensor = mask_tensor
                if curr_mask_tensor is None:
                    h, w = img_tensor.shape[2], img_tensor.shape[3]
                    curr_mask_tensor = torch.ones((1, h, w), dtype=torch.float32)

                images.append(img_tensor)
                emotions_out.append(emotion_text)
                masks.append(curr_mask_tensor)
                
                if prompt_style == "SDXL Style":
                    # SDXL workflow expects 12 face paths per sheet (one for each of the 12 sprites)
                    for _ in range(12):
                        face_output_paths.append(face_output_path)
                else:
                    # QWEN (Step 3) workflow typically handles single images or does its own mapping
                    face_output_paths.append(face_output_path)
                    
                sheet_output_paths.append(sheet_output_path)

        # Return results even if no images (user may not have connected image input)
        # But still return valid emotion data
        if not images:
            # Return empty lists for images/masks but keep emotion/prompt data valid
            return [], emotions_out, face_output_paths, sheet_output_paths, positive_prompt, negative_prompt, seed, []

        return images, emotions_out, face_output_paths, sheet_output_paths, positive_prompt, negative_prompt, seed, masks

NODE_CLASS_MAPPINGS = {
    "EmotionGeneratorV2": EmotionGeneratorV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmotionGeneratorV2": "VNCCS Emotion Studio"
}
