
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
    sheets_dir, load_character_info
)
from .vnccs_control_center import _apply_lora_standard, _apply_lora_nunchaku

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _latest_image_file(files):
    files = [path for path in files if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS]
    if not files:
        return None
    return max(files, key=lambda path: (os.path.getmtime(path), path))


def get_latest_sprite_path(character, costume="Naked"):
    try:
        base = os.path.join(character_dir(character), "Sprites", costume)
        if not os.path.isdir(base):
            return None

        direct_files = [
            os.path.join(base, filename)
            for filename in os.listdir(base)
            if os.path.isfile(os.path.join(base, filename))
        ]
        best = _latest_image_file(direct_files)
        if best:
            return best

        nested_files = []
        for root, _dirs, filenames in os.walk(base):
            for filename in filenames:
                nested_files.append(os.path.join(root, filename))
        return _latest_image_file(nested_files)
    except Exception:
        return None


def get_latest_sheet_path(character, costume="Naked"):
    try:
        path = os.path.join(character_dir(character), "Sheets", costume, "neutral")
        files = glob.glob(os.path.join(path, "sheet_neutral_*.png"))
        if not files:
            return None

        def get_idx(filename):
            match = re.search(r"(\d+)", os.path.basename(filename))
            return int(match.group(1)) if match else 0

        files.sort(key=get_idx)
        return files[-1]
    except Exception:
        return None


def crop_sheet_preview(sheet_path):
    img = Image.open(sheet_path)
    w, h = img.size
    item_w = w // 6
    item_h = h // 2
    return img.crop((5 * item_w, item_h, 6 * item_w, 2 * item_h))


def _validate_clothes_wizard_gguf(path, file_label="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_label} was not written: {path}")

    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise ValueError(f"{file_label} is too small to be a valid GGUF file ({size} bytes)")

    with open(path, "rb") as file:
        magic = file.read(4)
    if magic != b"GGUF":
        raise ValueError(f"{file_label} is not a valid GGUF file (magic={magic!r})")


def _find_clothes_wizard_model():
    base_path = folder_paths.models_dir
    possible_names = [
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
        "qwen2-vl-7b-instruct-q4_k_m.gguf",
    ]
    search_dirs = [os.path.join(base_path, "LLM"), os.path.join(base_path, "llm"), base_path]

    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in possible_names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def _parse_clothes_wizard_json(content):
    data = None
    try:
        import json_repair
        data = json_repair.loads(content)
    except Exception:
        data = None

    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        try:
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_str:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0]
            else:
                match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
            data = json.loads(json_str.strip())
        except Exception:
            data = None

    if not isinstance(data, dict):
        return None

    result = {}
    for key in ["top", "bottom", "shoes", "head", "face"]:
        value = data.get(key, "")
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        result[key] = value.strip()
    return result


class PipeContext:
    def __init__(self, source=None, **updates):
        s = source
        self.model = getattr(s, "model", None) if s is not None else None
        self.clip = getattr(s, "clip", None) if s is not None else None
        self.vae = getattr(s, "vae", None) if s is not None else None
        self.pos = getattr(s, "pos", None) if s is not None else None
        self.neg = getattr(s, "neg", None) if s is not None else None
        self.seed_int = getattr(s, "seed_int", getattr(s, "seed", 0)) if s is not None else 0
        self.sample_steps = getattr(s, "sample_steps", getattr(s, "steps", 0)) if s is not None else 0
        self.cfg = getattr(s, "cfg", 0.0) if s is not None else 0.0
        self.denoise = getattr(s, "denoise", 1.0) if s is not None else 1.0
        self.sampler_name = getattr(s, "sampler_name", None) if s is not None else None
        self.scheduler = getattr(s, "scheduler", None) if s is not None else None
        self.loader_type = getattr(s, "loader_type", None) if s is not None else None
        self.nunchaku_kind = getattr(s, "nunchaku_kind", None) if s is not None else None
        self.nunchaku_settings = getattr(s, "nunchaku_settings", None) if s is not None else None
        self.model_entry = getattr(s, "model_entry", None) if s is not None else None
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

    RETURN_TYPES = ("IMAGE", "STRING", "*")
    RETURN_NAMES = ("character", "sheets_path", "background")
    FUNCTION = "process"
    CATEGORY = "VNCCS"

    @staticmethod
    def _find_breasts_desc(char_info):
        """Search all string fields in character info for a breast/chest description."""
        # Prioritise 'body' field, then check the rest
        fields = ["body"] + [k for k in char_info if k != "body"]
        for key in fields:
            val = char_info.get(key)
            if not isinstance(val, str):
                continue
            m = re.search(r'[a-zA-Z\s]*(?:breasts?|flat\s+chest)[a-zA-Z\s]*', val, re.IGNORECASE)
            if m:
                return m.group(0).strip().strip(",").strip()
        return None

    @staticmethod
    def construct_prompt(data):
        active_tab = data.get("activeTab", "generate")
        character_name = data.get("character", "")

        # Determine breasts suffix for female characters
        breasts_suffix = ""
        if character_name:
            char_info = load_character_info(character_name) or {}
            sex = char_info.get("sex") or char_info.get("gender") or "female"
            if sex == "female":
                desc = ClothesDesigner._find_breasts_desc(char_info)
                if desc:
                    breasts_suffix = f", Girl with {desc}"

        if active_tab == "clone" and data.get("clone_image"):
            bg_col = data.get("gen_settings", {}).get("background_color", "Green")
            hex_col = "00FF00" if bg_col == "Green" else "0000FF"
            return (
                f"Put clothes, footwear and accessories from Picture 3 to character on Picture 1\n"
                f"solid {bg_col.lower()} ({hex_col}) background{breasts_suffix}",
                ""
            )

        info = data.get("costume_info", {})
        parts = []
        for k in ["top", "bottom", "head", "shoes", "face"]:
            v = info.get(k, "").strip()
            if v: parts.append(v)

        clothes_desc = "\n".join(parts)
        bg_col = data.get("gen_settings", {}).get("background_color", "Green")
        if bg_col not in ["Green", "Blue"]: bg_col = "Green"
        hex_col = "00FF00" if bg_col == "Green" else "0000FF"

        positive_prompt = (
            f"Dress the character:\n{clothes_desc}\n"
            f"solid {bg_col.lower()} ({hex_col}) background{breasts_suffix}"
        )
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
            sprite_path = get_latest_sprite_path(character_name, "Naked") or get_latest_sprite_path(character_name, "Original")
            if sprite_path:
                img = Image.open(sprite_path).convert("RGB")
                image_np = np.array(img).astype(np.float32) / 255.0
                sprite_tensor = torch.from_numpy(image_np).unsqueeze(0)
                return sprite_tensor, sprite_tensor

            best = get_latest_sheet_path(character_name, "Naked") or get_latest_sheet_path(character_name, "Original")
            if not best:
                return None, None

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
        def has_base_body(c):
             return bool(
                 get_latest_sprite_path(c, "Naked")
                 or get_latest_sprite_path(c, "Original")
                 or get_latest_sheet_path(c, "Naked")
                 or get_latest_sheet_path(c, "Original")
             )

        if not has_base_body(character_name):
             raise ValueError(f"Character '{character_name}' is incomplete. Missing 'Naked' or 'Original' sprites.")

        # 1. Prompt
        positive_prompt, negative_prompt = self.construct_prompt(data)
        
        # 2. Paths
        sheet_path = sheets_dir(character_name, costume_name, "neutral") 

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
            image1=ref_image, image2=None, image3=clone_image_tensor,
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

        # Apply LoRA temporarily for generation only — output pipe keeps the original model
        lora_name = gen_settings.get("lora_name", "none") or "none"
        lora_strength = float(gen_settings.get("lora_strength", 1.0) or 1.0)
        gen_model = model
        gen_clip = clip
        if lora_name != "none":
            full_path = folder_paths.get_full_path("loras", lora_name)
            if full_path and os.path.exists(full_path):
                print(f"[ClothesDesigner] Applying LoRA for generation: {lora_name} (strength={lora_strength})")
                loader_type = getattr(pipe, "loader_type", "standard") or "standard"
                if loader_type == "nunchaku":
                    gen_model = _apply_lora_nunchaku(
                        gen_model, full_path, lora_strength,
                        settings=getattr(pipe, "nunchaku_settings", None) or {},
                        model_entry=getattr(pipe, "model_entry", None),
                    )
                else:
                    gen_model, gen_clip = _apply_lora_standard(gen_model, gen_clip, full_path, lora_strength)
            else:
                print(f"[ClothesDesigner] LoRA not found: '{lora_name}', skipping.")

        # 4. Sampling using the incoming Control Center pipe configuration
        k_sampler = nodes.KSampler()

        print("[ClothesDesigner] Sampling...")
        latent_result = k_sampler.sample(
            model=gen_model, seed=out_pipe.seed_int, steps=out_pipe.sample_steps,
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
            with torch.inference_mode():
                image, = vae_decode_node.decode(**decode_kwargs)
        except Exception as e:
            print(f"[ClothesDesigner] VAEDecodeTiled failed ({e}), falling back to VAEDecode...")
            vae_decode_node = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]()
            with torch.inference_mode():
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
        return (image, sheet_path, bg_color)

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

@server.PromptServer.instance.routes.post("/vnccs/clothes_wizard")
async def vnccs_clothes_wizard(request):
    try:
        try:
            import llama_cpp
        except Exception as e:
            return web.json_response({
                "error": "DEPENDENCY_MISSING",
                "message": f"llama-cpp-python is required for Clothes Wizard: {e}",
                "model_name": "llama-cpp-python",
            }, status=500)

        post = await request.json()
        user_description = str(post.get("description", "")).strip()
        if not user_description:
            return web.Response(status=400, text="No clothes description provided")

        model_path = _find_clothes_wizard_model()
        if not model_path:
            return web.json_response({
                "error": "MODEL_MISSING",
                "message": "No Qwen GGUF model found.",
                "model_name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            }, status=404)

        try:
            _validate_clothes_wizard_gguf(model_path, os.path.basename(model_path))
        except Exception as e:
            return web.json_response({
                "error": "MODEL_INVALID",
                "message": f"Qwen GGUF model file is invalid or incomplete: {e}",
                "model_name": os.path.basename(model_path),
            }, status=422)

        system_prompt = (
            "You are a professional anime/game character costume designer. "
            "Convert broad outfit ideas into concrete, visual clothing prompts. "
            "Output valid JSON only."
        )
        user_prompt = f"""
Expand this abstract clothing idea into detailed outfit parts:
{user_description}

Return a raw JSON object with exactly these string keys:
- top
- bottom
- shoes
- head
- face (ONLY wearable face items/accessories)

Rules:
- Each value must be a detailed visual description suitable for image generation.
- Do not repeat the same abstract phrase from the user.
- Describe materials, colors, shape, trims, accessories, fit, and distinctive details.
- If a category is not needed, use an empty string.
- Keep descriptions clothing-focused.
- The "face" field is NOT for facial expression, makeup, blush, eyeshadow, lipstick, skin, cheeks, or facial features.
- Use "face" only for wearable/accessory items placed on the face, such as glasses, sunglasses, goggles, mask, veil, eyepatch, respirator, scarf over mouth, piercings, stickers, or temporary tattoos.
- If there is no wearable face item, set "face" to an empty string.
- Do not describe the body, pose, background, camera, quality tags, nudity, sex acts, facial expression, makeup, blush, eyeshadow, lipstick, skin, or cheeks.

Example for "Santa Claus costume":
{{
  "top": "red velvet Santa coat with thick white fur trim on cuffs, hem and front opening, black leather belt with square gold buckle, long sleeves, festive winter fabric texture",
  "bottom": "matching red velvet trousers with white fur cuffs, fitted but comfortable costume pants",
  "shoes": "black polished leather boots with rounded toes and folded cuffs",
  "head": "red Santa hat with white fur brim and white pom-pom, slightly tilted",
  "face": ""
}}
"""

        print(f"[ClothesDesigner] Clothes Wizard loading model: {model_path}")
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            temperature=0.35,
        )

        content = response["choices"][0]["message"]["content"]
        print(f"[ClothesDesigner] Clothes Wizard raw output: {content}")
        parsed = _parse_clothes_wizard_json(content or "")
        if parsed is None:
            return web.json_response({
                "error": "PARSE_ERROR",
                "message": "Failed to parse Clothes Wizard JSON output.",
                "raw": content or "",
            }, status=500)

        return web.json_response(parsed)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({
            "error": "INFERENCE_ERROR",
            "message": f"Engine Error: {e}",
            "model_name": "Qwen2VL (Check Console)",
        }, status=500)

@server.PromptServer.instance.routes.get("/vnccs/get_preview")
async def vnccs_get_preview(request):
    character = request.rel_url.query.get("character", "")
    costume = request.rel_url.query.get("costume", "Naked")
    if not character: return web.Response(status=404)

    naked_sprite = get_latest_sprite_path(character, "Naked")
    original_sprite = get_latest_sprite_path(character, "Original")
    naked_sheet = get_latest_sheet_path(character, "Naked")
    original_sheet = get_latest_sheet_path(character, "Original")
    if not naked_sprite and not original_sprite and not naked_sheet and not original_sheet:
         return web.Response(status=400, text="Character Incomplete.")

    force_cache = request.rel_url.query.get("force_cache", "") == "true"
    
    target_file = None
    
    # helper logic duplicated/inlined since we can't easily call instance static method from here without instance
    # actually we can use ClothesDesigner.get_cache_paths
    cache_file, _ = ClothesDesigner.get_cache_paths(character, costume)

    # If force_cache is true, try cache first
    if force_cache and os.path.exists(cache_file):
        target_file = cache_file

    # Otherwise, try costume sprites first.
    if not target_file:
         target_file = get_latest_sprite_path(character, costume)

    # For the base character preview, Original is the direct SFW fallback for Naked.
    if not target_file and costume == "Naked":
         target_file = original_sprite

    # Legacy fallback: costume sheet.
    if not target_file:
         target_file = get_latest_sheet_path(character, costume)
    
    # Fallback to cache if sheet not found (legacy behavior)
    if not target_file:
         if os.path.exists(cache_file): target_file = cache_file
         
    # Final fallback to base sprites, then legacy sheets.
    if not target_file:
         target_file = naked_sprite or original_sprite or naked_sheet or original_sheet

    if target_file and os.path.exists(target_file):
         is_sheet = "Sheets" in target_file or "sheet_neutral" in os.path.basename(target_file)
         if is_sheet:
             try:
                 crop = crop_sheet_preview(target_file)
                 buffered = io.BytesIO()
                 crop.save(buffered, format="PNG")
                 return web.Response(body=buffered.getvalue(), content_type="image/png")
             except: pass
         
         with open(target_file, "rb") as f:
             return web.Response(body=f.read(), content_type="image/png")
             
    return web.Response(status=404)
