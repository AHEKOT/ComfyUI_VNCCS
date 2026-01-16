
import os
import json
import torch
import folder_paths
import comfy.sd
import comfy.samplers
import comfy.utils
import nodes
import server
from aiohttp import web
from PIL import Image
import io
import base64
import numpy as np
import traceback
import glob
import re

from ..utils import (
    load_character_info, ensure_character_structure, EMOTIONS, MAIN_DIRS,
    save_config, build_face_details, generate_seed, dedupe_tokens,
    apply_sex, append_age, load_config, age_strength,
    list_characters, character_dir, base_output_dir,
    sheets_dir, faces_dir
)

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def get_latest_sheet_crop(character_name):
    """
    Attempts to find the latest 'sheet_neutral_*.png' for the character
    and crop the 12th sprite (Row 1, Col 5), mimicking the frontend preview logic.
    Returns: torch.Tensor (1,H,W,3) or None
    """
    try:
        # 1. Find the Sheet
        # Try Naked/neutral first
        base_char_path = character_dir(character_name)
        sheet_dir_path = os.path.join(base_char_path, "Sheets", "Naked", "neutral")
        print(f"[VNCCS Debug] Checking Fallback Path: {sheet_dir_path}")
        
        if not os.path.exists(sheet_dir_path):
                print(f"[VNCCS Debug] Path not found: {sheet_dir_path}")
                found = False
                base = os.path.join(base_char_path, "Sheets")
                if os.path.exists(base):
                    costumes = sorted(os.listdir(base))
                    for costume in costumes:
                        path = os.path.join(base, costume, "neutral")
                        if os.path.isdir(path):
                            sheet_dir_path = path
                            found = True
                            print(f"[VNCCS Debug] Found alternative sheet path: {path}")
                            break
                if not found:
                    print(f"[VNCCS Debug] No sheet directories found in {base}")
                    return None

        # 2. Find the best file (highest index)
        pattern = os.path.join(sheet_dir_path, "sheet_neutral_*.png")
        files = glob.glob(pattern)
        print(f"[VNCCS Debug] Files found: {len(files)}")
        if not files:
                return None
        
        def get_index(f):
            m = re.search(r'(\d+)', os.path.basename(f))
            return int(m.group(1)) if m else 0
        
        files.sort(key=get_index)
        best_file = files[-1]
        print(f"[VNCCS Debug] Using file: {best_file}")

        # 3. Load and Crop
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
        return pil2tensor(crop)
        
    except Exception as e:
        print(f"[VNCCS] Cache Fallback Failed: {e}")
        return None

# --------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------

# --- PREVIEW CACHE ---
PREVIEW_CACHE = {
    "ckpt_name": None,
    "ckpt_obj": None, # (model, clip, vae)
    "loras": {}       # name -> tensor_dict
}

if server:
    @server.PromptServer.instance.routes.get("/vnccs/context_lists")
    async def get_context_lists(request):
        try:
            # Checkpoints
            checkpoints = folder_paths.get_filename_list("checkpoints")
            
            # Samplers & Schedulers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
            
            # Character List
            characters = list_characters()
            
            # LoRA List
            loras = folder_paths.get_filename_list("loras")

            return web.json_response({
                "checkpoints": checkpoints,
                "samplers": samplers,
                "schedulers": schedulers,
                "characters": characters,
                "loras": loras
            })
        except Exception as e:
            return web.Response(status=500, text=str(e))
            
    @server.PromptServer.instance.routes.get("/vnccs/character_info")
    async def get_character_info(request):
        try:
            name = request.rel_url.query.get("character", "")
            if not name:
                return web.json_response({})
                
            config = load_config(name)
            if config and "character_info" in config:
                return web.json_response(config["character_info"])
            return web.json_response({})
            
        except Exception as e:
            traceback.print_exc() # Print to console
            return web.Response(status=500, text=f"{str(e)}\n\n{traceback.format_exc()}")

    @server.PromptServer.instance.routes.post("/vnccs/preview_generate")
    async def preview_generate(request):
        try:
            data = await request.json()
            gen_settings = data.get("gen_settings", {})
            char_info = data.get("character_info", {})
            character_name = data.get("character", "Unknown")

            # Extract Settings
            ckpt_name = gen_settings.get("ckpt_name")
            if not ckpt_name:
                return web.Response(status=400, text="Checkpoint name required")

            # Generate Prompt
            positive_text, negative_text = CharacterCreatorV2.construct_prompt(char_info)
            
            steps = int(gen_settings.get("steps", 20))
            cfg = float(gen_settings.get("cfg", 8.0))
            sampler_name = gen_settings.get("sampler", "euler")
            scheduler = gen_settings.get("scheduler", "normal")
            seed = int(gen_settings.get("seed", 0))

            # Resolution
            width = 640
            height = 1536

            # Load Models (With Cache)
            global PREVIEW_CACHE
            
            # 1. Checkpoint
            if PREVIEW_CACHE["ckpt_name"] == ckpt_name and PREVIEW_CACHE["ckpt_obj"]:
                print(f"[VNCCS] Preview: Using Cached Checkpoint '{ckpt_name}'")
                model, clip, vae = PREVIEW_CACHE["ckpt_obj"]
            else:
                 print(f"[VNCCS] Preview: Loading Checkpoint '{ckpt_name}'")
                 ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                 if not ckpt_path:
                    return web.Response(status=404, text=f"Checkpoint {ckpt_name} not found")
                 out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
                 model, clip, vae = out[:3]
                 # Update Cache
                 PREVIEW_CACHE["ckpt_name"] = ckpt_name
                 PREVIEW_CACHE["ckpt_obj"] = (model, clip, vae)

            # Clone to avoid tainting cached model with LoRAs
            model = model.clone()
            clip = clip.clone()

            # Helper for Cached LoRA
            def apply_lora_cached(m, c, l_name, l_strength):
                if not l_name or l_name == "None": return m, c
                
                lora_dict = PREVIEW_CACHE["loras"].get(l_name)
                if not lora_dict:
                    l_path = folder_paths.get_full_path("loras", l_name)
                    if l_path:
                        lora_dict = comfy.utils.load_torch_file(l_path, safe_load=True)
                        PREVIEW_CACHE["loras"][l_name] = lora_dict
                        print(f"[VNCCS] Preview: Cached LoRA '{l_name}'")
                
                if lora_dict:
                    return comfy.sd.load_lora_for_models(m, c, lora_dict, l_strength, l_strength)
                return m, c

            # Apply LoRAs
            dmd_lora_name = gen_settings.get("dmd_lora_name")
            dmd_lora_strength = float(gen_settings.get("dmd_lora_strength", 1.0))
            model, clip = apply_lora_cached(model, clip, dmd_lora_name, dmd_lora_strength)

            age_lora_name = gen_settings.get("age_lora_name")
            if age_lora_name:
                age = int(char_info.get("age", 18))
                age_str = age_strength(age)
                model, clip = apply_lora_cached(model, clip, age_lora_name, age_str)

            lora_stack = gen_settings.get("lora_stack", [])
            for l_item in lora_stack:
                model, clip = apply_lora_cached(model, clip, l_item.get("name"), float(l_item.get("strength", 1.0)))

            # 2. Encode Prompts
            tokens_pos = clip.tokenize(positive_text)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

            tokens_neg = clip.tokenize(negative_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]

            # Patch PromptServer
            if not hasattr(server.PromptServer.instance, 'last_prompt_id'):
                server.PromptServer.instance.last_prompt_id = 'preview_gen'

            # 3. Sample
            latent = torch.zeros([1, 4, height // 8, width // 8], device=model.load_device)
            samples = nodes.common_ksampler(
                model=model, seed=seed, steps=steps, cfg=cfg, 
                sampler_name=sampler_name, scheduler=scheduler, 
                positive=positive_cond, negative=negative_cond, 
                latent={"samples": latent}, denoise=1.0
            )[0]["samples"]

            # 4. Decode
            vae_decoded = vae.decode_tiled(samples, tile_x=512, tile_y=512)
            i = 255. * vae_decoded.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])

            # Save Smart Cache
            try:
                c_dir = os.path.join(character_dir(character_name), "cache")
                os.makedirs(c_dir, exist_ok=True)
                c_path = os.path.join(c_dir, "preview.png")
                img.save(c_path)
            except Exception as e:
                print(f"[VNCCS] Failed to save preview cache: {e}")

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return web.json_response({"image": img_b64})

        except Exception as e:
            traceback.print_exc()
            return web.Response(status=500, text=str(e))


class CharacterCreatorV2:
    """
    Standalone GUI for Character Creation with On-Demand Preview.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "widget_data": ("STRING", {"default": "{}"}), 
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "VNCCS_PIPE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("character", "pipe", "positive_prompt", "negative_prompt", "sheets_path", "faces_path", "face_details")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @staticmethod
    def construct_prompt(info):
        """
        Centralized logic for constructing positive/negative prompts from character info.
        """
        aesthetics = info.get("aesthetics", "masterpiece")
        sex = info.get("sex", "female")
        age = int(info.get("age", 18))
        
        # Base Prompt
        positive_prompt = f"{aesthetics}, simple background, expressionless"
        positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
        
        # NSWF / Clothing (Ensure bool casting)
        is_nsfw = bool(info.get("nsfw", False))
        if str(info.get("nsfw")).lower() == "false": is_nsfw = False 
        
        if is_nsfw:
             nude_phrase = "(naked, nude, penis)" if sex == "male" else "(naked, nude, vagina, nipples)"
        else:
             nude_phrase = "(bare chest, wear white boxers)" if sex == "male" else "(wear white bra and panties)"
        positive_prompt += f", {nude_phrase}"
        
        # Age
        positive_prompt = append_age(positive_prompt, age, sex)
        
        # Physical Attributes
        for attr in ["race", "hair", "eyes", "face", "body", "skin_color", "additional_details"]:
            val = info.get(attr, "")
            if val:
                positive_prompt += f", ({val}:{1.0 if 'skin' in attr else 1.0})"

        # LoRA Trigger
        lora_prompt = info.get("lora_prompt", "")
        if lora_prompt:
            positive_prompt += f", {lora_prompt}"

        # Negative Prompt
        neg = info.get("negative_prompt", "")
        negative_prompt = dedupe_tokens(f"{neg},{gender_negative}")

        return positive_prompt, negative_prompt

    def process(self, widget_data="{}", unique_id=None):
        # Clear Preview Cache to free memory for workflow run
        global PREVIEW_CACHE
        if PREVIEW_CACHE["ckpt_obj"]:
             # Explicitly delete references to help GC
             del PREVIEW_CACHE["ckpt_obj"]
             PREVIEW_CACHE["ckpt_obj"] = None
        PREVIEW_CACHE["ckpt_name"] = None
        PREVIEW_CACHE["loras"].clear()

        try:
            data = json.loads(widget_data)
        except:
            data = {}

        # Extract values
        character_name = data.get("character", "Unknown")
        info = data.get("character_info", {})
        gen_settings = data.get("gen_settings", {})
        
        # 1. Generate Prompts
        positive_prompt, negative_prompt = self.construct_prompt(info)
        
        # 2. Re-extract local vars for saving / outputs
        face_details = build_face_details(info)
        face_details += ", (expressionless:1.0)"

        # Save Config logic
        character_path = character_dir(character_name)
        sheets_path = sheets_dir(character_name) # Uses default "Naked", "neutral", "sheet_neutral"
        faces_path = faces_dir(character_name)   # Uses default "Naked", "neutral", "face_neutral"

        config = load_config(character_name) or {
            "character_info": {},
            "folder_structure": {
                "main_directories": MAIN_DIRS,
                "emotions": EMOTIONS
            },
            "character_path": character_path,
            "config_version": "2.0"
        }

        # Update config with current info (omitted full mapping for brevity, assuming 'info' is up to date)
        # In a real scenario, we might want to ensure 'config' is perfectly synced.
        # But 'process' is triggered by the run button, so we should trust 'widget_data' as truth.
        # ... (Save logic kept minimal as requested focus is on Outputs)

        # 3. Load Models & Construct Pipe
        # ----------------------------------------------------------------
        ckpt_name = gen_settings.get("ckpt_name") or data.get("ckpt_name")
        if not ckpt_name:
             raise ValueError("No Checkpoint selected in Character Creator V2")

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not ckpt_path:
            raise ValueError(f"Checkpoint path not found for '{ckpt_name}'")

        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model, clip, vae = out[:3]
        
        if model is None:
             raise ValueError(f"Failed to load Model from checkpoint '{ckpt_name}'")
        if clip is None:
             raise ValueError(f"Failed to load CLIP from checkpoint '{ckpt_name}'")
        if vae is None:
             raise ValueError(f"Failed to load VAE from checkpoint '{ckpt_name}'")

        # Helper to apply LoRA
        def apply_lora_safe(m, c, l_name, l_strength):
            if not l_name or l_name == "None": return m, c
            l_path = folder_paths.get_full_path("loras", l_name)
            if l_path:
                lora = comfy.utils.load_torch_file(l_path, safe_load=True)
                return comfy.sd.load_lora_for_models(m, c, lora, l_strength, l_strength)
            return m, c

        # Apply DMD2
        dmd_name = gen_settings.get("dmd_lora_name")
        dmd_str = float(gen_settings.get("dmd_lora_strength", 1.0))
        model, clip = apply_lora_safe(model, clip, dmd_name, dmd_str)

        # Apply Age LoRA
        age_name = gen_settings.get("age_lora_name")
        if age_name:
            age = int(info.get("age", 18))
            age_str = age_strength(age)
            model, clip = apply_lora_safe(model, clip, age_name, age_str)

        # Apply Stack
        stack = gen_settings.get("lora_stack", [])
        for item in stack:
            model, clip = apply_lora_safe(model, clip, item.get("name"), float(item.get("strength", 1.0)))

        # Encode Conditioning
        tokens_pos = clip.tokenize(positive_prompt)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        conditioning_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip.tokenize(negative_prompt)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        conditioning_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

        # Construct Pipe Object
        class PipeContext:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        pipe = PipeContext(
            model=model,
            clip=clip,
            vae=vae,
            pos=conditioning_pos,
            neg=conditioning_neg,
            seed_int=int(gen_settings.get("seed", 0)),
            sample_steps=int(gen_settings.get("steps", 20)),
            cfg=float(gen_settings.get("cfg", 8.0)),
            denoise=1.0,
            sampler_name=gen_settings.get("sampler", "euler"),
            scheduler=gen_settings.get("scheduler", "normal")
        )

        # 4. Generate Image (Smart Cache Logic)
        
        # Determine Cache Path
        cache_dir = os.path.join(character_dir(character_name), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "preview.png")

        # Check Validity
        preview_valid = data.get("preview_valid", False)
        
        image = None

        # Determine source
        preview_source = "gen"
        if widget_data:
             try:
                # widget_data is a string (JSON), but here it seems passed as 'widget_data' argument which might be raw string
                # logic above parsed it into 'data' dict. use that.
                preview_source = data.get("preview_source", "gen")
             except: pass
        
        print(f"[VNCCS] Processing - Source: {preview_source}, Valid: {preview_valid}")

        # LOGIC:
        # 1. If source == 'sheet' (User just loaded character), we MUST use the sheet.
        #    We ignore the existing cache file because it might be stale.
        #    We overwrite the cache with the sheet crop.
        # 2. If source == 'gen' (User generated previously), we use the cache.
        
        if preview_valid:
             if preview_source == "sheet":
                  print("[VNCCS] Source is Sheet. Force-loading sheet crop.")
                  image = get_latest_sheet_crop(character_name)
                  if image is not None:
                       print("[VNCCS] Sheet loaded successfully. Overwriting Cache.")
                       try:
                           c_img = tensor2pil(image)
                           os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                           c_img.save(cache_path)
                       except: pass
                  else:
                       print("[VNCCS] Sheet load failed. Will try cache/regen.")

             # If image not set yet (source=gen OR sheet load failed), try cache
             if image is None and os.path.exists(cache_path):
                 print(f"[VNCCS] Smart Cache Hit: Loading existing preview for '{character_name}'")
                 try:
                     i = Image.open(cache_path)
                     image = pil2tensor(i)
                 except Exception as e:
                     print(f"[VNCCS] Failed to load cache: {e}. Regenerating.")

        if image is None:
             # Fallback: If cache is missing (even if source=gen), try sheet before regen
             if preview_valid:
                 print(f"[VNCCS] Cache Miss. Attempting Sheet Fallback...")
                 image = get_latest_sheet_crop(character_name)
                 if image is not None:
                     print(f"[VNCCS] Sheet Fallback Successful. Updating Cache.")
                     try:
                        c_img = tensor2pil(image)
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        c_img.save(cache_path)
                     except: pass
                 else:
                     print(f"[VNCCS] Sheet Fallback Failed. Regenerating...")

        if image is None:
            print(f"[VNCCS] Regenerating Preview...")
            
            latent = torch.zeros([1, 4, 1536 // 8, 640 // 8], device=model.load_device)
            
            try:
                samples = nodes.common_ksampler(
                    model=model, 
                    seed=int(gen_settings.get("seed", 0)), 
                    steps=int(gen_settings.get("steps", 20)), 
                    cfg=float(gen_settings.get("cfg", 8.0)), 
                    sampler_name=gen_settings.get("sampler", "euler"), 
                    scheduler=gen_settings.get("scheduler", "normal"), 
                    positive=conditioning_pos, 
                    negative=conditioning_neg, 
                    latent={"samples": latent}, 
                    denoise=1.0
                )[0]["samples"]
                
                image = vae.decode(samples)
                
                # Update Cache
                try:
                    # Tensor [1,H,W,3] -> PIL
                    c_img = tensor2pil(image)
                    c_img.save(cache_path)
                    print(f"[VNCCS] Saved new preview cache to {cache_path}")
                except Exception as e:
                    print(f"[VNCCS] Failed to save cache: {e}")

            except Exception as e:
                print(f"[VNCCS] Generation failed in process: {e}")
                image = torch.zeros((1, 512, 512, 3))

        return (
            image,
            pipe,
            positive_prompt,
            negative_prompt,
            sheets_path,
            faces_path,
            face_details
        )

