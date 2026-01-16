
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

from ..utils import (
    base_output_dir, character_dir, list_characters,
    load_character_info, ensure_character_structure, EMOTIONS, MAIN_DIRS,
    save_config, build_face_details, generate_seed, dedupe_tokens,
    apply_sex, append_age, load_config
)

# --------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------

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
            return web.json_response({"image": base64_image})
            
        except Exception as e:
            traceback.print_exc() # Print to console
            return web.Response(status=500, text=f"{str(e)}\n\n{traceback.format_exc()}")

    @server.PromptServer.instance.routes.post("/vnccs/preview_generate")
    async def preview_generate(request):
        try:
            json_data = await request.json()
            
            # Inputs
            ckpt_name = json_data.get("ckpt_name")
            char_info = json_data.get("character_info", {})
            
            # DEBUG
            print(f"[VNCCS] Received Preview Request. Info: {char_info}")
            
            # Generate Prompt ON BACKEND to match Node Process
            positive_text, negative_text = CharacterCreatorV2.construct_prompt(char_info)
            
            steps = int(json_data.get("steps", 20))
            cfg = float(json_data.get("cfg", 8.0))
            sampler_name = json_data.get("sampler", "euler") # Fixed key 'sampler'
            scheduler = json_data.get("scheduler", "normal")
            seed = int(json_data.get("seed", 0))
            
            # Additional LoRA Application for Preview?
            lora_name = json_data.get("lora_name")
            lora_strength = float(json_data.get("lora_strength", 1.0))

            # Resolution
            width = 640
            height = 1536

            if not ckpt_name:
                return web.Response(status=400, text="Checkpoint name required")

            # 1. Load Checkpoint (Cached)
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if not ckpt_path:
                return web.Response(status=404, text=f"Checkpoint {ckpt_name} not found")

            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model, clip, vae = out[:3]
            
            # 1.5 Apply LoRA if selected
            if lora_name:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, lora_strength, lora_strength)

            # 2. Encode Prompts
            # Positive
            tokens_pos = clip.tokenize(positive_text)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

            # Negative
            tokens_neg = clip.tokenize(negative_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]

            # Patch PromptServer for 'last_prompt_id' error
            if not hasattr(server.PromptServer.instance, 'last_prompt_id'):
                server.PromptServer.instance.last_prompt_id = 'preview_gen'

            # 3. Sample
            latent = torch.zeros([1, 4, height // 8, width // 8], device=model.load_device)
            
            # Common KSampler logic
            samples = nodes.common_ksampler(
                model=model, 
                seed=seed, 
                steps=steps, 
                cfg=cfg, 
                sampler_name=sampler_name, 
                scheduler=scheduler, 
                positive=positive_cond, 
                negative=negative_cond, 
                latent={"samples": latent}, 
                denoise=1.0
            )[0]["samples"]

            # 4. Decode (Tiled for efficiency to prevent OOM)
            vae_decoded = vae.decode_tiled(samples, tile_x=512, tile_y=512) # [1, H, W, 3]

            # 5. Convert to Base64
            # Tensor to PIL
            i = 255. * vae_decoded.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return web.json_response({"image": img_b64})

        except Exception as e:
            import traceback
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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("preview_image", "positive_prompt", "negative_prompt", "face_details", "character_name", "seed", "model_name", "lora_name", "lora_strength")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    def process(self, widget_data="{}", unique_id=None):
        try:
            data = json.loads(widget_data)
        except:
            data = {}

        # Extract values for output execution
        character_name = data.get("character", "Unknown")
        info = data.get("character_info", {})
        
    @staticmethod
    def construct_prompt(info):
        """
        Centralized logic for constructing positive/negative prompts from character info.
        Identical to V1 logic.
        """
        aesthetics = info.get("aesthetics", "masterpiece")
        sex = info.get("sex", "female")
        age = int(info.get("age", 18))
        
        # Base Prompt
        positive_prompt = f"{aesthetics}, simple background, expressionless"
        positive_prompt, gender_negative = apply_sex(sex, positive_prompt, "")
        
        # NSFW / Clothing
        is_nsfw = info.get("nsfw")
        # DEBUG PRINT
        print(f"[VNCCS] Constructing Prompt. Sex: {sex}, NSFW: {is_nsfw} (Type: {type(is_nsfw)})")
        
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
        try:
            data = json.loads(widget_data)
        except:
            data = {}

        # Extract values
        character_name = data.get("character", "Unknown")
        info = data.get("character_info", {})
        
        # Generate Prompts using Shared Logic
        positive_prompt, negative_prompt = self.construct_prompt(info)
        
        # Re-extract local vars for saving
        # (Save logic remains the same, extracting from 'info')
        
        face_details = build_face_details(info)
        face_details += ", (expressionless:1.0)"

        # --- SAVE CONFIG LOGIC (V1 Parity) ---
        # Ensure data persists when workflow runs
        character_path = character_dir(character_name)
        
        # Load existing or create template
        config = load_config(character_name) or {
            "character_info": {},
            "folder_structure": {
                "main_directories": MAIN_DIRS,
                "emotions": EMOTIONS
            },
            "character_path": character_path,
            "config_version": "2.0"
        }
        
        # Update with current widget data
        config["character_info"].update({
            "name": character_name,
            "sex": sex,
            "age": age,
            "aesthetics": aesthetics,
            "race": info.get("race", ""),
            "hair": info.get("hair", ""),
            "eyes": info.get("eyes", ""),
            "face": info.get("face", ""),
            "body": info.get("body", ""),
            "skin_color": info.get("skin_color", ""),
            "additional_details": info.get("additional_details", ""),
            "background_color": info.get("background_color", ""),
            "nsfw": info.get("nsfw", False),
            "negative_prompt": neg,
            "lora_prompt": lora_prompt,
            "seed": int(info.get("seed", 0))
        })
        
        # Preserve costumes
        if "costumes" not in config:
            config["costumes"] = {}

        save_config(character_name, config)
        # -------------------------------------

        dummy_img = torch.zeros((1, 512, 512, 3))

        return (
            dummy_img, 
            positive_prompt, 
            negative_prompt, 
            face_details, 
            character_name, 
            str(info.get("seed", 0)),
            data.get("ckpt_name", "")
        )

