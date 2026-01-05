import os
import json
import re
import torch  # Assuming torch is used by load_character_sheet and for mask creation


# --- VNCCS Utility Imports (CRITICAL: Added back the relative import block) ---
try:
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, ensure_costume_structure, EMOTIONS,
        apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
        sheets_dir, load_costume_info
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ..utils import (
        base_output_dir, character_dir, list_characters,
        load_character_info, ensure_costume_structure, EMOTIONS,
        apply_sex, append_age, generate_seed, build_face_details, load_character_sheet,
        sheets_dir, load_costume_info
    )

# ----------------------------------------------------------------------------


# --- ComfyUI Server Imports (Conditional for Linting/Execution) ---
try:
    import server
    from aiohttp import web
    import sys
except ImportError:
    # Mocking for local testing/linting outside ComfyUI
    print("VNCCS Warning: Running outside ComfyUI environment. API routes will not be registered.")


    class MockPromptServer:
        def __init__ (self): self.routes = self

        def get (self, path): return lambda func: func


    class MockServer:
        instance = MockPromptServer()


    class MockWebResponse:
        def __init__ (self, status=200, text=""): pass

        def json_response (self, data): return None


    class MockWeb:
        Response = MockWebResponse
        json_response = lambda self, data: None


    server = MockServer()
    web = MockWeb()


# --------------------------------------------------------------------


# --- File Path and API Endpoint ---

# Helper function to get the path to the custom node (Used for loading emotions.json)
def get_custom_node_path ():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# Helper function to load the emotions JSON (used by both API and node execution)
def load_emotions_data ():
    """
    Loads the emotions.json file, counts keys, and checks for safe_name uniqueness.
    Patches non-unique safe_names and warns the user.
    """
    # Assumes config_path is already defined via get_custom_node_path()
    config_path = os.path.join(get_custom_node_path(), "emotions-config", "emotions.json")
    if not os.path.exists(config_path):
        print(f"VNCCS ERROR: emotions.json not found at {config_path}")
        raise FileNotFoundError(f"emotions.json not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- 1. Count the number of keys (Categories) ---
    num_categories = len(data)

    # --- 2. Check and Patch Unique safe_names ---
    seen_safe_names = {}  # Stores {safe_name: count}
    duplicate_found = False
    numnames = 0
    for category, emotion_list in data.items():
        if not isinstance(emotion_list, list):
            continue  # Skip if not a list of emotions

        for emotion in emotion_list:
            numnames += 1
            safe_name = emotion.get('safe_name')

            if not safe_name:
                continue  # Skip if safe_name is missing

            if safe_name in seen_safe_names:
                # Duplicate found!
                duplicate_found = True
                seen_safe_names[safe_name] += 1

                # PATCH: Append the count to the safe_name for uniqueness
                new_safe_name = f"{safe_name}_{seen_safe_names[safe_name]}"

                print(f"[ VNCCS ] WARNING: DUPLICATE safe_name '{safe_name}' found in category '{category}'.")
                print(f"[ VNCCS ] WARNING: Temporarily patched to '{new_safe_name}' for execution.")

                # Update the emotion object with the patched name
                emotion['safe_name'] = new_safe_name
            else:
                seen_safe_names[safe_name] = 1

    # Notify the user to manually fix the JSON file
    if duplicate_found:
        print("\n\n************************************************************************")
        print("!!! VNCCS CONFIG ALERT: Duplicate safe_names were found in emotions.json. !!!")
        print("!!! Please manually edit emotions.json to ensure every safe_name is unique, !!!")
        print("!!! or your queue selector will show incorrect/unpredictable items.         !!!")
        print("************************************************************************\n")

    print("[VNCCS] INFO: ✅ " + str(numnames) + " great emotions loaded from " + str(num_categories) + " categories.")

    return data


# API endpoint to serve the emotions.json file (used by JS frontend)
@server.PromptServer.instance.routes.get("/vnccs/get_emotions")
async def get_emotions_config (request):
    try:
        data = load_emotions_data()
        return web.json_response(data)
    except Exception as e:
        return web.Response(status=500, text=f"Error loading emotions.json: {e}")


# --- Node Class Definition ---

class EmotionGenerator:
    
    EMOTIONS_DATA = None
    SAFE_NAME_MAP = None
    
    @classmethod
    def __init__(cls):
        # Trigger the single-time data setup
        cls._setup_emotions_data()

    @classmethod
    def _setup_emotions_data(cls):
        """
        Calls load_emotions_data ONCE, which triggers the single print, 
        and caches the result.
        """
        # Check if setup has already run
        if cls.SAFE_NAME_MAP is not None:
            return

        try:
            # 🛑 CRITICAL: Call the external function. It will execute and 
            # print its messages directly to the CLI once.
            emotions_data = load_emotions_data() 
            
            # Build and cache the safe_name_map for all future calls
            safe_name_map = {}
            for category, emotion_list in emotions_data.items():
                for emotion in emotion_list:
                    if 'safe_name' in emotion and emotion['safe_name']:
                        safe_name_map[emotion['safe_name']] = {
                                "key": emotion['key'],
                                "description": emotion['description'],
                                "category": category
                        }
            cls.SAFE_NAME_MAP = safe_name_map
            print("✅ VNCCS Emotion Data Cached. Ready for execution.")

        except Exception as e:
            print(f"[VNCCS] ERROR: Failed to load emotions data during startup: {e}")
            cls.SAFE_NAME_MAP = {}

    @classmethod
    def INPUT_TYPES (cls):
        characters = list_characters()
        if not characters:
            characters = ["Character Name"]

        return {
                "required": {
                        # CORRECTED: Use dynamic list_characters() for the dropdown
                        "character"        : (characters,
                                              {"default": characters[0] if characters else "Character Name"}),

                        # CRITICAL: Placeholder for the JS controller (must be STRING)
                        "emotion_selector" : ("STRING", {"default": "loading..."}),

                        # The primary widget for user-queued entries (Character: safe_name\n...)
                        "selected_emotions": ("STRING", {
                                "default"  : "",
                                "multiline": True,
                        }),
                }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "MASK")
    RETURN_NAMES = ("images", "emotions_out", "face_output_paths", "sheet_output_paths", "positive_prompt",
                    "negative_prompt", "seed", "masks")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False, True)
    FUNCTION = "generate_emotions"
    CATEGORY = "VNCCS"

    def generate_emotions (self, character, emotion_selector, selected_emotions):
        
        # --- DEBUG LOGGING ---
        print(f"\n--- VNCCS Emotion Generator Inputs ---")
        print(f"Character: {character}")
        print(f"Queued Emotions (Raw): {selected_emotions.strip().replace('\n', ' | ')}")
        print(f"------------------------------------\n")

        character_path = character_dir(character)
        info = load_character_info(character)
        sheets_dir = os.path.join(character_path, "Sheets")
        costumes = [d for d in os.listdir(sheets_dir) if os.path.isdir(os.path.join(sheets_dir, d))] if os.path.exists(
            sheets_dir) else []
        images = []
        emotions_out = []
        face_output_paths = []
        sheet_output_paths = []
        masks = []
    
        # --- NEW: Load safe_name from cache ---
        safe_name_map = self.SAFE_NAME_MAP 
        if safe_name_map is None:
            # Fail-safe: if setup was skipped or failed silently
            self._setup_emotions_data()
            safe_name_map = self.SAFE_NAME_MAP

        # --- VARIABLE ASSIGNMENT ---
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
            
            # Negative prompt logic 
            negative_prompt = info.get("negative_prompt", "") 
            negative_prompt = negative_prompt + ", (facial droplet), (water drop), (water), (water droplets), (water drops)"
            
            config_seed = info.get("seed", 0)
            seed = generate_seed(config_seed)
            base_negative_prompt = negative_prompt
            positive_prompt = aesthetics # Initialize here to avoid UnboundLocalError
        else:
            print(f"[EmotionGenerator] Character '{character}' not found") 
            info = {}
            aesthetics = background_color = sex = race = eyes = hair = face_features = body = skin_color = additional_details = negative_prompt = lora_prompt = ""
            positive_prompt = ""
            base_negative_prompt = ""
            age = 18
            seed = generate_seed(0)
        
        # --- NEW: Adapt the emotion list from the selected_emotions queue ---
        queued_entries = selected_emotions.strip().split('\n')
        
        # This list will contain the 'safe_name' which the original code used as the 'emotion_key'.
        emotions_list = []
        for entry in queued_entries:
            # Match the "Name: safe-expression" format
            match = re.match(r"([^:]+):\s*(\S+)", entry.strip())
            if match:
                safe_name = match.group(2).strip()
                if safe_name in safe_name_map:
                    emotions_list.append(safe_name)
                else:
                    print(f"Skipping unknown emotion safe_name: {safe_name}")
            else:
                print(f"Skipping unparseable emotion entry: {entry}")
        # --- END NEW ADAPTATION ---

        for costume in costumes: 
            costume_info = load_costume_info(character, costume)
            head = costume_info.get("head", "")
            face_wear = costume_info.get("face", "")
            top = costume_info.get("top", "")
            bottom = costume_info.get("bottom", "")
            shoes = costume_info.get("shoes", "")

            neutral_dir = os.path.join(sheets_dir, costume, "neutral")
            if not os.path.exists(neutral_dir):
                print(f"Folder {neutral_dir} does not exist")
                continue

            img_tensor, mask_tensor = load_character_sheet(character, costume, "neutral", with_mask=True)
            
            if img_tensor is None:
                print(f"Failed to load image for costume {costume}")
                continue

            # The emotion_key is now the 'safe_name'
            for emotion_key in emotions_list: 
                
                # --- NEW: Get the true emotion description ---
                emotion_details = safe_name_map.get(emotion_key)
                emotion_description = emotion_details['description']
                # --- END NEW ---

                # Path construction uses the emotion_key (safe_name)
                face_dir = os.path.join(character_path, "Faces", costume, emotion_key)
                os.makedirs(face_dir, exist_ok=True)
                sheet_dir = os.path.join(character_path, "Sheets", costume, emotion_key)
                os.makedirs(sheet_dir, exist_ok=True)
                face_output_path = os.path.join(face_dir, f"face_{emotion_key}_")
                sheet_output_path = os.path.join(sheet_dir, f"sheet_{emotion_key}_")
                
                # --- RESTORED ORIGINAL PROMPT BUILDER LOGIC (INCLUDING UTILITY CALLS) ---
                positive_prompt = f"{aesthetics}"
                if background_color:
                    positive_prompt += f", {background_color} background"
                if race:
                    positive_prompt += f", ({race} race:1.0)"
                if hair:
                    positive_prompt += f", ({hair} hair:1.0)"
                if eyes:
                    positive_prompt += f", ({eyes} eyes:1.0)"
                if body:
                    positive_prompt += f", ({body} body:1.0)"
                if skin_color:
                    positive_prompt += f", ({skin_color} skin:1.0)"
                if additional_details:
                    positive_prompt += f", ({additional_details})"
                
                # Original utility calls
                positive_prompt_after_sex, gender_negative = apply_sex(sex, positive_prompt, "")
                positive_prompt = positive_prompt_after_sex
                negative_prompt = f"{base_negative_prompt}, {gender_negative}"

                positive_prompt = append_age(positive_prompt, age, sex)
                
                if lora_prompt:
                    positive_prompt += f", {lora_prompt}"

                # Original logic for emotion_text: using the description now, as requested.
                face_details = build_face_details(info)
                emotion_text = f"({emotion_key}, {emotion_description}), {face_details}" # Adapted to use description
                
                if mask_tensor is None:
                    h, w = img_tensor.shape[2], img_tensor.shape[3]
                    mask_tensor = torch.ones((1, h, w), dtype=torch.float32)

                images.append(img_tensor)
                emotions_out.append(emotion_text)
                masks.append(mask_tensor)
                for _ in range(12):
                    face_output_paths.append(face_output_path)
                sheet_output_paths.append(sheet_output_path)

        print("VNCCS DEBUG SETTINGS:")
        print("----------------------------------")
        print("seed:", seed)
        print("----------------------------------")
        print("positive_prompt:", positive_prompt)
        print("----------------------------------")
        print("negative_prompt:", negative_prompt)
        print("----------------------------------")
        print("emotion prompt:", emotions_out)

        if not images:
            return [], [], [], [], "", "", seed, []

        # Return lists directly as in original code
        return images, emotions_out, face_output_paths, sheet_output_paths, positive_prompt, negative_prompt, seed, masks
NODE_CLASS_MAPPINGS = {
    "EmotionGenerator": EmotionGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmotionGenerator": "VNCCS Emotion Generator"
}

NODE_CATEGORY_MAPPINGS = {
    "EmotionGenerator": "VNCCS"
}
