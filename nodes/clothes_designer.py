
import os
import json
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
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
    character_dir, base_output_dir, load_character_info, save_costume_info,
    load_costume_info, list_costumes, ensure_costume_structure, 
    sheets_dir, faces_dir, dedupe_tokens
)

class ClothesDesigner:
    """
    VNCCS Clothes Designer Node
    Wraps standard ComfyUI nodes for GGUF loading, Sampling, and VAE decoding.
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
    RETURN_NAMES = ("character", "pipe", "positive_prompt", "negative_prompt", "sheets_path", "faces_path", "costume_info")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @staticmethod
    def construct_prompt(data):
        active_tab = data.get("activeTab", "generate")
        
        if active_tab == "clone" and data.get("clone_image"):
             bg_col = data.get("gen_settings", {}).get("background_color", "Green")
             hex_col = "00FF00" if bg_col == "Green" else "0000FF"
             return f"Put clothes, footwear and accessories from Picture 3 to character on Picture 1\nsolid {bg_col.lower()} ({hex_col}) background", ""

        info = data.get("costume_info", {})
        parts = []
        for k in ["top", "bottom", "head", "shoes", "face"]:
            v = info.get(k, "").strip()
            if v: parts.append(v)
            
        clothes_desc = "\n".join(parts)
        bg_col = data.get("gen_settings", {}).get("background_color", "Green")
        if bg_col not in ["Green", "Blue"]: bg_col = "Green"
        hex_col = "00FF00" if bg_col == "Green" else "0000FF"
        
        positive_prompt = f"Dress the character in the following clothes, keeping the body proportions the same:\n{clothes_desc}\nsolid {bg_col.lower()} ({hex_col}) background"
        negative_prompt = "bad quality, worst quality, (naked, nude, nipple, penis, vagina:2.0)"
        return positive_prompt, negative_prompt

    def get_naked_sheet_crop(self, character_name):
        try:
            base = character_dir(character_name)
            path = os.path.join(base, "Sheets", "Naked", "neutral")
            files = glob.glob(os.path.join(path, "sheet_neutral_*.png"))
            if not files: return None
            def get_idx(f):
                m = re.search(r'(\d+)', os.path.basename(f))
                return int(m.group(1)) if m else 0
            files.sort(key=get_idx)
            best = files[-1]
            img = Image.open(best)
            w, h = img.size
            if w > 0 and h > 0:
                item_w = w // 6; item_h = h // 2
                left = 5 * item_w; upper = 1 * item_h
                crop = img.crop((left, upper, left+item_w, upper+item_h))
                return torch.from_numpy(np.array(crop).astype(np.float32) / 255.0).unsqueeze(0)
            return None
        except: return None

    def process(self, widget_data="{}", unique_id=None):
        try:
            if isinstance(widget_data, str):
                data = json.loads(widget_data)
            else: data = widget_data
        except:
            data = {}

        character_name = data.get("character", "Unknown")
        costume_name = data.get("costume", "Naked")
        gen_settings = data.get("gen_settings", {})
        active_tab = data.get("activeTab", "generate")
        
        # 0. Cache Check
        import hashlib
        cache_dir = os.path.join(character_dir(character_name), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_info_path = os.path.join(cache_dir, "preview_info.json")
        cache_img_path = os.path.join(cache_dir, "preview.png")

        # Stable Hash Calculation
        try:
            if isinstance(widget_data, str):
                msg_obj = json.loads(widget_data)
            else:
                msg_obj = widget_data
            
            # Canonical JSON representation
            canonical_str = json.dumps(msg_obj, sort_keys=True, separators=(',', ':'))
            input_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
        except:
             input_hash = "INVALID"

        if os.path.exists(cache_info_path) and os.path.exists(cache_img_path):
            try:
                with open(cache_info_path, "r") as f:
                    cached_info = json.load(f)
                
                cached_hash = cached_info.get("hash")
                
                if cached_hash == input_hash:
                    # Cache Hit
                    print(f"[ClothesDesigner] Cache Hit! Loading preview from {cache_img_path}")
                    try:
                        CACHE_IMG = Image.open(cache_img_path)
                        image_np = np.array(CACHE_IMG).astype(np.float32) / 255.0
                        if image_np.shape[-1] == 4: image_np = image_np[..., :3]
                        
                        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                        
                        pos, neg = self.construct_prompt(data)
                        sheet_path = sheets_dir(character_name, costume_name, "neutral") 
                        face_path = faces_dir(character_name, costume_name, "neutral")
                        costume_json_str = json.dumps(data.get("costume_info", {}), indent=2)
                        
                        class PipeContext:
                            def __init__(self): 
                                self.seed_int = int(gen_settings.get("seed", 0))
                                self.steps = int(gen_settings.get("steps", 4))
                                self.cfg = float(gen_settings.get("cfg", 1.0))

                        pipe = PipeContext()
                        return (image_tensor, pipe, pos, neg, sheet_path, face_path, costume_json_str)
                    except Exception as e:
                        print(f"[ClothesDesigner] Cache Load Error: {e}")
            except: pass

        # 0. Validation
        def has_base_sheet(c):
             p1 = os.path.join(character_dir(c), "Sheets", "Naked", "neutral")
             p2 = os.path.join(character_dir(c), "Sheets", "Original", "neutral")
             has_naked = glob.glob(os.path.join(p1, "sheet_neutral_*.png"))
             has_orig = glob.glob(os.path.join(p2, "sheet_neutral_*.png"))
             return bool(has_naked or has_orig)

        if not has_base_sheet(character_name):
             raise ValueError(f"Character '{character_name}' is incomplete. Missing 'Naked' or 'Original' sheets.")

        # 1. Prompt
        positive_prompt, negative_prompt = self.construct_prompt(data)
        
        # 2. Paths
        sheet_path = sheets_dir(character_name, costume_name, "neutral") 
        face_path = faces_dir(character_name, costume_name, "neutral")
        
        # 3. Load Models (WRAPPER CALLS)
        unet_name = gen_settings.get("unet_name")
        clip_name = gen_settings.get("clip_name")
        vae_name = gen_settings.get("vae_name")
        
        if not all([unet_name, clip_name, vae_name]):
            raise ValueError("All model fields (UNET, CLIP, VAE) must be set.")
            
        print(f"[ClothesDesigner] Loading UNET: {unet_name}")
        if "UnetLoaderGGUF" not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError("UnetLoaderGGUF node not found. Please install ComfyUI-GGUF.")
        
        gguf_loader = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        model, = gguf_loader.load_unet(unet_name)
        
        print(f"[ClothesDesigner] Loading CLIP: {clip_name}")
        clip_loader = nodes.CLIPLoader()
        clip, = clip_loader.load_clip(clip_name, type="qwen_image")
        
        print(f"[ClothesDesigner] Loading VAE: {vae_name}")
        vae_loader = nodes.VAELoader()
        vae, = vae_loader.load_vae(vae_name)

        # 4. LoRAs
        lora_loader = nodes.LoraLoader()
        def safe_load_lora(m, c, name, strength):
            if not name or name == "None": return m, c
            print(f"[ClothesDesigner] Loading LoRA: {name} (s={strength})")
            return lora_loader.load_lora(m, c, name, strength_model=strength, strength_clip=strength)

        model, clip = safe_load_lora(model, clip, gen_settings.get("lightning_lora"), gen_settings.get("lightning_lora_strength", 1.0))
        model, clip = safe_load_lora(model, clip, gen_settings.get("service_lora"), gen_settings.get("service_lora_strength", 1.0))
        for item in gen_settings.get("lora_stack", []):
             model, clip = safe_load_lora(model, clip, item.get("name"), item.get("strength", 1.0))

        # 5. VNCCS Qwen Encoder (Subclass defined above in previous tool call, assuming standard import logic)
        from .vnccs_qwen_encoder import VNCCS_QWEN_Encoder, node_helpers
        
        class SafeQwenEncoder(VNCCS_QWEN_Encoder):
            def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, target_size=1024, upscale_method="lanczos", crop_method="center", instruction="", image1_name="Picture 1", image2_name="Picture 2", image3_name="Picture 3", weight1=1.0, weight2=1.0, weight3=1.0, vl_size=384, latent_image_index=1, qwen_2511=True):
                ref_latents = []
                input_images = [image1, image2, image3]
                names = [image1_name, image2_name, image3_name]
                vl_images = []
                template_prefix = "<|im_start|>system\n"
                template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                instruction_content = ""
                if instruction == "":
                    instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
                else:
                    if template_prefix in instruction: instruction = instruction.split(template_prefix)[1]
                    if template_suffix in instruction: instruction = instruction.split(template_suffix)[0]
                    if "{}" in instruction: instruction = instruction.replace("{}", "")
                    instruction_content = instruction
                
                llama_template = template_prefix + instruction_content + template_suffix
                image_prompt = ""

                def pad_to_square(img_tensor):
                    B, H, W, C = img_tensor.shape
                    if H == W: return img_tensor
                    max_dim = max(H, W)
                    padded = torch.zeros((B, max_dim, max_dim, C), dtype=img_tensor.dtype, device=img_tensor.device)
                    y_off = (max_dim - H) // 2
                    x_off = (max_dim - W) // 2
                    padded[:, y_off:y_off+H, x_off:x_off+W, :] = img_tensor
                    return padded

                for i, image in enumerate(input_images):
                    if image is not None:
                        if image.shape[-1] == 4: image = image[..., :3]
                        
                        # Fix: Pass Clone Image (Index 2) "AS IS" without resizing/cropping
                        if i == 2:
                            processed_ref = image
                        else:
                            processed_ref = self._process_image(image, target_size, upscale_method, crop_method)
                            
                        ref_latents.append(vae.encode(processed_ref[:, :, :, :3]))
                        image_sq = pad_to_square(image)
                        processed_vl = self._process_image(image_sq, vl_size, upscale_method, crop_method)
                        vl_images.append(processed_vl)
                        image_prompt += "{}: <|vision_start|><|image_pad|><|vision_end|>".format(names[i])
                        
                tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
                conditioning = clip.encode_from_tokens_scheduled(tokens)
                try:
                    if qwen_2511:
                        method = "index_timestep_zero"
                        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": method})
                except Exception: pass
                
                conditioning_full_ref = conditioning
                if len(ref_latents) > 0:
                    weights_list = [weight1, weight2, weight3]
                    ref_latents_weighted = [ (w ** 2) * latent for w, latent in zip(weights_list[:len(ref_latents)], ref_latents) ]
                    ref_latents_full = [latent for latent, w in zip(ref_latents_weighted, weights_list[:len(ref_latents)]) if w > 0]
                    conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents_full}, append=True)
                
                conditioning_negative = [(torch.zeros_like(cond[0]), cond[1]) for cond in conditioning_full_ref]
                if len(ref_latents) >= latent_image_index: samples = ref_latents[latent_image_index - 1]
                else: samples = torch.zeros(1, 4, 128, 128)
                latent_out = {"samples": samples}
                return (conditioning_full_ref, conditioning_negative, latent_out)

        encoder = SafeQwenEncoder()
        
        ref_image = self.get_naked_sheet_crop(character_name)
        if ref_image is None: ref_image = torch.zeros((1, 512, 512, 3))
        
        depth_img = None
        if "AIO_Preprocessor" in nodes.NODE_CLASS_MAPPINGS:
            try:
                 if not hasattr(server.PromptServer.instance, "last_prompt_id"):
                     server.PromptServer.instance.last_prompt_id = "manual_execution"
                 aio = nodes.NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
                 args = {"image": ref_image, "preprocessor": "DepthAnythingV2Preprocessor", "resolution": 512}
                 out = getattr(aio, aio.FUNCTION)(**args)
                 depth_img = out[0] if isinstance(out, (list, tuple)) else out
            except Exception as e:
                 print(f"[ClothesDesigner] Depth Gen Failed: {e}")
        if depth_img is None: depth_img = torch.zeros_like(ref_image)

        # Load Clone Image (Image 3)
        clone_image_tensor = None
        if active_tab == "clone" and data.get("clone_image"):
             try:
                 c_info = data["clone_image"]
                 c_name = c_info.get("name")
                 c_sub = c_info.get("subfolder", "")
                 c_type = c_info.get("type", "input")
                 
                 # Standard ComfyUI Image Load logic
                 if c_sub: 
                     image_path = os.path.join(folder_paths.get_input_directory(), c_sub, c_name)
                 else: 
                     image_path = folder_paths.get_annotated_filepath(c_name)
                 
                 i = Image.open(image_path)
                 
                 # Convert to Tensor (H,W,C)
                 i = torch.from_numpy(np.array(i).astype(np.float32) / 255.0).unsqueeze(0)
                 
                 # Strip Alpha if present
                 if i.shape[-1] == 4: i = i[..., :3]
                 
                 clone_image_tensor = i
             except Exception as e:
                 print(f"[ClothesDesigner] Failed to load clone image: {e}")

        pos_cond, neg_cond, empty_latent = encoder.encode(
            clip=clip, prompt=positive_prompt, vae=vae,
            image1=ref_image, image2=depth_img, image3=clone_image_tensor,
            target_size=1344,
            crop_method="pad",
            vl_size=384,
            qwen_2511=True
        )
        
        # 6. Pipe Construct
        class PipeContext:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)
                    
        pipe = PipeContext(
            model=model, clip=clip, vae=vae,
            pos=pos_cond, neg=neg_cond,
            seed_int=int(gen_settings.get("seed", 0)),
            sample_steps=int(gen_settings.get("steps", 4)),
            cfg=float(gen_settings.get("cfg", 1.0)),
            denoise=1.0,
            sampler_name=gen_settings.get("sampler", "euler"),
            scheduler=gen_settings.get("scheduler", "simple") 
        )

        # 7. Sampling (Standard KSampler)
        # nodes.KSampler.sample -> common_ksampler
        k_sampler = nodes.KSampler()
        # Create latent image input (empty)
        # VNCCS_QWEN_Encoder returns 'latent' which is a dict {"samples": tensor} or just the tensor?
        # Checking qwen encoder source: returns (..., ..., latent_out) where latent_out = {"samples": samples}
        # KSampler expects 'latent_image' which is {"samples": ...}
        
        print("[ClothesDesigner] Sampling...")
        latent_result = k_sampler.sample(
            model=model, seed=pipe.seed_int, steps=pipe.sample_steps, 
            cfg=pipe.cfg, sampler_name=pipe.sampler_name, scheduler=pipe.scheduler,
            positive=pos_cond, negative=neg_cond, latent_image=empty_latent, denoise=1.0
        )[0] # KSampler returns (LATENT,)

        # 8. Decode (Standard VAE Decode Tiled)
        print("[ClothesDesigner] VAE Decoding (Tiled)...")
        vae_decode = nodes.VAEDecodeTiled()
        image, = vae_decode.decode(vae=vae, samples=latent_result, tile_size=512)

        # Cache for UI preview
        try:
             c_dir = os.path.join(character_dir(character_name), "cache")
             os.makedirs(c_dir, exist_ok=True)
             c_path = os.path.join(c_dir, "preview.png")
             i_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
             i_pil.save(c_path)
             
             # Save Cache Info
             c_info_path = os.path.join(c_dir, "preview_info.json")
             with open(c_info_path, "w") as f:
                 json.dump({
                     "hash": input_hash,
                     "widget_data": data # Save parsed data or original string
                 }, f)
                 
             server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": unique_id, "character": character_name})
        except: pass

        costume_json_str = json.dumps(data.get("costume_info", {}), indent=2)
        return (image, pipe, positive_prompt, negative_prompt, sheet_path, face_path, costume_json_str)

# --- API ---

@server.PromptServer.instance.routes.get("/vnccs/list_costumes")
async def vnccs_list_costumes(request):
    character = request.rel_url.query.get("character", "")
    if not character: return web.json_response([])
    try:
        data = list_costumes(character)
        return web.json_response(data)
    except Exception as e:
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.get("/vnccs/get_costume")
async def vnccs_get_costume(request):
    character = request.rel_url.query.get("character", "")
    costume = request.rel_url.query.get("costume", "")
    if not character or not costume: return web.json_response({})
    try:
        data = load_costume_info(character, costume)
        return web.json_response(data)
    except Exception as e:
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.post("/vnccs/save_costume")
async def vnccs_save_costume(request):
    try:
        data = await request.json()
        character = data.get("character")
        costume = data.get("costume")
        info = data.get("info", {})
        if not character or not costume: return web.Response(status=400)
        ensure_costume_structure(character, costume)
        save_costume_info(character, costume, info)
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.get("/vnccs/get_preview")
async def vnccs_get_preview(request):
    character = request.rel_url.query.get("character", "")
    costume = request.rel_url.query.get("costume", "Naked")
    if not character: return web.Response(status=404)
    
    def get_latest_sheet_path(char, cost):
        try:
             path = os.path.join(character_dir(char), "Sheets", cost, "neutral")
             files = glob.glob(os.path.join(path, "sheet_neutral_*.png"))
             if not files: return None
             def get_idx(f):
                 m = re.search(r'(\d+)', os.path.basename(f))
                 return int(m.group(1)) if m else 0
             files.sort(key=get_idx)
             return files[-1]
        except: return None

    naked_sheet = get_latest_sheet_path(character, "Naked")
    original_sheet = get_latest_sheet_path(character, "Original")
    if not naked_sheet and not original_sheet:
         return web.Response(status=400, text="Character Incomplete.")

    target_file = get_latest_sheet_path(character, costume)
    if not target_file:
        c_path = os.path.join(character_dir(character), "cache", "preview.png")
        if os.path.exists(c_path): target_file = c_path
    if not target_file: target_file = naked_sheet if naked_sheet else original_sheet

    if target_file and os.path.exists(target_file):
         is_sheet = "Sheets" in target_file or "sheet_neutral" in os.path.basename(target_file)
         if is_sheet:
             try:
                 img = Image.open(target_file)
                 w, h = img.size
                 item_w = w // 6; item_h = h // 2
                 crop = img.crop((5 * item_w, 1 * item_h, 6 * item_w, 2 * item_h))
                 buffered = io.BytesIO()
                 crop.save(buffered, format="PNG")
                 return web.Response(body=buffered.getvalue(), content_type="image/png")
             except: pass
         
         with open(target_file, "rb") as f:
             return web.Response(body=f.read(), content_type="image/png")
             
    return web.Response(status=404)

@server.PromptServer.instance.routes.post("/vnccs/clothes_preview")
async def clothes_preview(request):
    try:
        # Re-use the ClothesDesigner class logic to generate a preview
        # This ensures we use the exact same pipeline (Standard Nodes)
        data = await request.json()
        
        # We need to create a dummy instance or static method to run the generation
        # But `process` is an instance method and returns a lot of stuff.
        # Let's extract the core logic to a helper or just instantiate ClothesDesigner.
        
        # Create canonical string to match process() hashing logic
        widget_data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        designer = ClothesDesigner()
        # Run process. This will do the standard node calls.
        # It also saves to cache/preview.png
        # process() returns (image_tensor, pipe, ...)
        
        # CAUTION: process() raises errors if models aren't found.
        image_tensor, _, _, _, _, _, _ = designer.process(widget_data=widget_data_str, unique_id="api_preview")
        
        # Convert tensor to base64 for immediate frontend display
        i_pil = Image.fromarray(np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        i_pil.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return web.json_response({"image": b64})
    except Exception as e:
        traceback.print_exc()
        return web.Response(status=500, text=str(e))
