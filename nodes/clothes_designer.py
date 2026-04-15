
import os
import json
import torch
import folder_paths
import nodes
import server
from aiohttp import web
from PIL import Image
import io
import numpy as np
import traceback
import glob
import re

from ..utils import (
    character_dir, save_costume_info,
    load_costume_info, list_costumes, ensure_costume_structure,
    sheets_dir, faces_dir
)


class PipeContext:
    def __init__(self, source=None, **updates):
        self.model = getattr(source, "model", None) if source is not None else None
        self.clip = getattr(source, "clip", None) if source is not None else None
        self.vae = getattr(source, "vae", None) if source is not None else None
        self.pos = getattr(source, "pos", None) if source is not None else None
        self.neg = getattr(source, "neg", None) if source is not None else None
        self.seed_int = getattr(source, "seed_int", getattr(source, "seed", 0)) if source is not None else 0
        self.sample_steps = getattr(source, "sample_steps", getattr(source, "steps", 0)) if source is not None else 0
        self.cfg = getattr(source, "cfg", 0.0) if source is not None else 0.0
        self.denoise = getattr(source, "denoise", 1.0) if source is not None else 1.0
        self.sampler_name = getattr(source, "sampler_name", None) if source is not None else None
        self.scheduler = getattr(source, "scheduler", None) if source is not None else None
        for key, value in updates.items():
            setattr(self, key, value)

class ClothesDesigner:
    """
    VNCCS Clothes Designer Node
    Wraps standard ComfyUI nodes for GGUF loading, Sampling, and VAE decoding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "pipe": ("VNCCS_PIPE",),
            },
            "hidden": {
                "widget_data": ("STRING", {"default": "{}"}), 
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "VNCCS_PIPE", "STRING", "STRING", "*")
    RETURN_NAMES = ("character", "sheet", "pipe", "sheets_path", "faces_path", "background")
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

    @staticmethod
    def get_cache_paths(character, costume):
        def clean_name(n):
            return "".join([c for c in n if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        safe_costume = clean_name(costume)
        cache_dir = os.path.join(character_dir(character), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        img_path = os.path.join(cache_dir, f"preview_{safe_costume}.png")
        info_path = os.path.join(cache_dir, f"preview_info_{safe_costume}.json")
        return img_path, info_path

    def get_naked_sheet_crop(self, character_name):
        try:
            base = character_dir(character_name)
            path = os.path.join(base, "Sheets", "Naked", "neutral")
            files = glob.glob(os.path.join(path, "sheet_neutral_*.png"))
            if not files: return None, None
            def get_idx(f):
                m = re.search(r'(\d+)', os.path.basename(f))
                return int(m.group(1)) if m else 0
            files.sort(key=get_idx)
            best = files[-1]
            img = Image.open(best)
            w, h = img.size
            if w > 0 and h > 0:
                # Full sheet
                image_np = np.array(img).astype(np.float32) / 255.0
                if image_np.shape[-1] == 4: image_np = image_np[..., :3]
                full_sheet = torch.from_numpy(image_np).unsqueeze(0)

                # Crop
                item_w = w // 6; item_h = h // 2
                left = 5 * item_w; upper = 1 * item_h
                crop_img = img.crop((left, upper, left+item_w, upper+item_h))
                crop_np = np.array(crop_img).astype(np.float32) / 255.0
                if crop_np.shape[-1] == 4: crop_np = crop_np[..., :3]
                crop_tensor = torch.from_numpy(crop_np).unsqueeze(0)
                
                return crop_tensor, full_sheet
            return None, None
        except: return None, None

    def process(self, pipe=None, widget_data="{}", unique_id=None):
        # CRITICAL FIX: Ensure PromptServer has last_prompt_id for preview system
        if not hasattr(server.PromptServer.instance, "last_prompt_id"):
             server.PromptServer.instance.last_prompt_id = "vnccs_api_preview"

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

        if pipe is None:
            raise ValueError("Clothes Designer requires an incoming VNCCS pipe from Control Center.")

        model = getattr(pipe, "model", None)
        clip = getattr(pipe, "clip", None)
        vae = getattr(pipe, "vae", None)
        if model is None or clip is None or vae is None:
            raise ValueError("Incoming VNCCS pipe is missing model, clip, or vae.")
        
        # 0. Cache Check
        import hashlib
        cache_img_path, cache_info_path = self.get_cache_paths(character_name, costume_name)

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

        # 3. VNCCS Qwen Encoder using model objects from the incoming pipe
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
        
        ref_image, full_naked_sheet = self.get_naked_sheet_crop(character_name)
        if ref_image is None: 
            ref_image = torch.zeros((1, 512, 512, 3))
            full_naked_sheet = torch.zeros((1, 512, 512, 3))
        
        depth_img = None
        # Skip Depth Map in Clone mode
        if active_tab != "clone":
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
        else:
            print("[ClothesDesigner] Clone Mode: Skipping Depth Map")
        # Disable Depth Map but remain compatible with encoder input.   
        depth_img = None
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
            target_size=1024,
            crop_method="disabled",
            vl_size=384,
            qwen_2511=True
        )
        
        seed_int = int(gen_settings.get("seed", 0)) or int(getattr(pipe, "seed_int", getattr(pipe, "seed", 0)) or 0)
        sample_steps = int(getattr(pipe, "sample_steps", getattr(pipe, "steps", 0)) or 4)
        cfg = float(getattr(pipe, "cfg", 1.0) or 1.0)
        denoise = float(getattr(pipe, "denoise", 1.0) or 1.0)
        sampler_name = getattr(pipe, "sampler_name", None) or "euler"
        scheduler = getattr(pipe, "scheduler", None) or "normal"

        out_pipe = PipeContext(
            source=pipe,
            model=model, clip=clip, vae=vae,
            pos=pos_cond, neg=neg_cond,
            seed_int=seed_int,
            sample_steps=sample_steps,
            cfg=cfg,
            denoise=denoise,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )

        # 4. Sampling using the incoming Control Center pipe configuration
        k_sampler = nodes.KSampler()

        print("[ClothesDesigner] Sampling...")
        latent_result = k_sampler.sample(
            model=model, seed=out_pipe.seed_int, steps=out_pipe.sample_steps,
            cfg=out_pipe.cfg, sampler_name=out_pipe.sampler_name, scheduler=out_pipe.scheduler,
            positive=pos_cond, negative=neg_cond, latent_image=empty_latent, denoise=out_pipe.denoise
        )[0]

        def normalize_decode_input(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            if isinstance(value, dict):
                return {k: normalize_decode_input(v) for k, v in value.items()}
            if isinstance(value, list):
                return [normalize_decode_input(v) for v in value]
            if isinstance(value, tuple):
                return tuple(normalize_decode_input(v) for v in value)
            return value

        latent_for_decode = normalize_decode_input(latent_result)

        # 5. Decode
        print("[ClothesDesigner] VAE Decoding...")
        try:
            vae_decode_node = nodes.NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
            decode_kwargs = dict(vae=vae, samples=latent_for_decode, tile_size=512, overlap=64)
            # Newer ComfyUI versions added temporal params; pass them if accepted
            import inspect
            sig = inspect.signature(vae_decode_node.decode)
            if "temporal_size" in sig.parameters:
                decode_kwargs["temporal_size"] = 0
                decode_kwargs["temporal_overlap"] = 0
            image, = vae_decode_node.decode(**decode_kwargs)
        except Exception as e:
            print(f"[ClothesDesigner] VAEDecodeTiled failed ({e}), falling back to VAEDecode...")
            vae_decode_node = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]()
            image, = vae_decode_node.decode(vae=vae, samples=latent_for_decode)

        # Cache for UI preview
        try:
             c_img_path, c_info_path = self.get_cache_paths(character_name, costume_name)
             i_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
             i_pil.save(c_img_path)
             
             # Save Cache Info
             with open(c_info_path, "w") as f:
                 json.dump({
                     "hash": input_hash,
                     "widget_data": data # Save parsed data or original string
                 }, f)
                 
             print(f"[ClothesDesigner] Sending Preview Update Event: ID={unique_id}, Char={character_name}")
             server.PromptServer.instance.send_sync("vnccs.preview.updated", {"node_id": str(unique_id), "character": character_name})
        except Exception as e:
             print(f"[ClothesDesigner] Failed to send preview update: {e}")
             traceback.print_exc()

        bg_color = gen_settings.get("background_color", "Green")
        return (image, full_naked_sheet, out_pipe, sheet_path, face_path, bg_color)

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

    force_cache = request.rel_url.query.get("force_cache", "") == "true"
    
    target_file = None
    
    # helper logic duplicated/inlined since we can't easily call instance static method from here without instance
    # actually we can use ClothesDesigner.get_cache_paths
    cache_file, _ = ClothesDesigner.get_cache_paths(character, costume)

    # If force_cache is true, try cache first
    if force_cache and os.path.exists(cache_file):
        target_file = cache_file

    # Otherwise, try costume sheet
    if not target_file:
         target_file = get_latest_sheet_path(character, costume)
    
    # Fallback to cache if sheet not found (legacy behavior)
    if not target_file:
         if os.path.exists(cache_file): target_file = cache_file
         
    # Final fallback to base sheets
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

