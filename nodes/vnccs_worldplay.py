import os
import torch
import folder_paths
import sys
import json
import logging
import numpy as np

# Add background-data/worldplay to path
current_dir = os.path.dirname(os.path.abspath(__file__))
worldplay_path = os.path.abspath(os.path.join(current_dir, "../background-data"))
if worldplay_path not in sys.path:
    sys.path.append(worldplay_path)

from worldplay.inference.pipeline_worldplay import WanPipeline
from worldplay.models.dits.arwan_w_action import WanTransformer3DModel
from worldplay.utils.trajectory import parse_pose_string, generate_camera_trajectory_local, pose_to_input, pose_string_to_json
import diffusers
from huggingface_hub import hf_hub_download, snapshot_download
from comfy.utils import ProgressBar

class VNCCS_LoadWorldPlay5B:
    @classmethod
    def INPUT_TYPES(s):
        loras = folder_paths.get_filename_list("loras")
        loras.insert(0, "None")
        return {
            "required": {
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "lora": (loras, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }


    RETURN_TYPES = ("WORLDPLAY_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/WorldPlay"

    def load_model(self, precision, lora, lora_strength):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

        # Define base model path with support for extra_model_paths.yaml
        
        # 1. Register 'worldplay' folder type if not present
        if "worldplay" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("worldplay", os.path.join(folder_paths.models_dir, "worldplay"))
        
        # 2. Get all candidate paths (default + extra paths)
        candidate_paths = folder_paths.get_folder_paths("worldplay")
        if not candidate_paths:
            # Fallback if something went wrong, though add_model_folder_path ensures at least one
            candidate_paths = [os.path.join(folder_paths.models_dir, "worldplay")]
            
        # 3. Find first path that contains models, or default to first candidate
        models_base_path = candidate_paths[0]
        for path in candidate_paths:
            # Check for a marker file
            if os.path.exists(os.path.join(path, "wan_transformer", "config.json")):
                models_base_path = path
                logging.info(f"Found existing WorldPlay models at {models_base_path}")
                break
        
        logging.info(f"Using WorldPlay models path: {models_base_path}")
        print(f"Using WorldPlay models path: {models_base_path}")
        os.makedirs(models_base_path, exist_ok=True)
        
        # 5 Major steps: Transformer Config, Transformer Weights, Distilled, VAE, TextEnc
        pbar = ProgressBar(5)

        def download_file_if_missing(repo_id, filename, local_dir):
             # HuggingFace uses POSIX paths (forward slashes) even on Windows
             # Convert to OS-native path for local file check
             local_filename = filename.replace("/", os.sep)
             target_path = os.path.join(local_dir, local_filename)
             if not os.path.exists(target_path):
                logging.info(f"Downloading {filename} from {repo_id}...")
                print(f"Downloading {filename} from {repo_id} to {target_path}...")
                try:
                    # Pass original filename (POSIX) to HF API
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
                except Exception as e:
                    logging.error(f"Failed to download {filename}: {e}")
                    raise e
        
        # 1. Download WorldPlay Transformer
        # Target: models/worldplay/wan_transformer
        download_file_if_missing("tencent/HY-WorldPlay", "wan_transformer/config.json", models_base_path)
        pbar.update(1)
        
        download_file_if_missing("tencent/HY-WorldPlay", "wan_transformer/diffusion_pytorch_model.safetensors", models_base_path)
        pbar.update(1)
        
        # 2. Distilled Model Checkpoint
        # We DO NOT auto-download the 49GB model.pt file as it's too large and unexpected.
        # User should have either model.pt or model_inference.safetensors
        # If neither exists, we will raise formatted error during loading phase.
        
        # 3. VAE from diffusers-compatible Wan2.1 repo
        diffusers_repo = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        wan_base_dir = os.path.join(models_base_path, "Wan2.1-Diffusers")
        vae_target_dir = os.path.join(wan_base_dir, "vae")
        vae_config_file = os.path.join(vae_target_dir, "config.json")
        
        # Download VAE files (~500MB)
        if not os.path.exists(vae_config_file):
            print(f"Downloading VAE from {diffusers_repo}...")
            os.makedirs(vae_target_dir, exist_ok=True)
            try:
                hf_hub_download(repo_id=diffusers_repo, filename="vae/config.json", local_dir=wan_base_dir)
                hf_hub_download(repo_id=diffusers_repo, filename="vae/diffusion_pytorch_model.safetensors", local_dir=wan_base_dir)
                print(f"VAE downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download VAE: {e}")
                raise RuntimeError(f"Failed to download VAE from {diffusers_repo}. Error: {e}")
        # 4. Text Encoder (~11GB) and Tokenizer from Wan-AI/Wan2.1-T2V-1.3B
        # This is the pre-converted BF16 encoder-only version (much smaller than full 50GB model)
        text_encoder_file = os.path.join(models_base_path, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_dir = os.path.join(models_base_path, "google", "umt5-xxl")
        
        # Download Text Encoder (~11GB)
        if not os.path.exists(text_encoder_file):
            print(f"Downloading Text Encoder (~11GB)...")
            try:
                hf_hub_download(
                    repo_id="Wan-AI/Wan2.1-T2V-1.3B", 
                    filename="models_t5_umt5-xxl-enc-bf16.pth", 
                    local_dir=models_base_path
                )
                print(f"Text Encoder downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download Text Encoder: {e}")
                raise RuntimeError(f"Failed to download Text Encoder. Error: {e}")
        
        # Download Tokenizer from same repo's google/umt5-xxl folder  
        tokenizer_config = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config):
            print(f"Downloading Tokenizer...")
            os.makedirs(tokenizer_dir, exist_ok=True)
            try:
                for f in ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"]:
                    hf_hub_download(
                        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
                        filename=f"google/umt5-xxl/{f}",
                        local_dir=models_base_path
                    )
                print(f"Tokenizer downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download Tokenizer: {e}")
                raise RuntimeError(f"Failed to download Tokenizer. Error: {e}")
        pbar.update(1)

        # Internal Fixed Paths
        transformer_path = os.path.join(models_base_path, "wan_transformer")
        ckpt_safetensors = os.path.join(models_base_path, "wan_distilled_model", "model_inference.safetensors")
        ckpt_pt = os.path.join(models_base_path, "wan_distilled_model", "model.pt")
        vae_path = vae_target_dir

        logging.info(f"Loading models from {models_base_path}")
        
        # Load VAE
        print("Loading VAE...")
        try:
            vae = diffusers.AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype, local_files_only=True)
        except Exception as e:
            logging.error(f"Failed to load VAE from {vae_path}: {e}")
            raise e
        
        # Load Text Encoder from .pth file and Tokenizer
        print("Loading Text Encoder...")
        try:
            # Load tokenizer from local directory
            import transformers
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            
            # Load text encoder - need to load the .pth directly
            # First create the model structure, then load weights
            from transformers import UMT5EncoderModel, UMT5Config
            
            # Create config - UMT5-XXL parameters
            config = UMT5Config(
                vocab_size=250112,
                d_model=4096,
                d_kv=64,
                d_ff=10240,
                num_layers=24,
                num_heads=64,
                relative_attention_num_buckets=32,
                relative_attention_max_distance=128,
                dropout_rate=0.1,
                feed_forward_proj="gated-gelu",
                is_encoder_decoder=False,
            )
            text_encoder = UMT5EncoderModel(config)
            
            # Load weights from .pth file
            state_dict = torch.load(text_encoder_file, map_location="cpu")
            text_encoder.load_state_dict(state_dict, strict=False)
            
            # FORCE BF16/FP16 for Text Encoder to save VRAM
            # T5-XXL is too huge for fp32
            t5_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            text_encoder = text_encoder.to(dtype=t5_dtype)
            
            # Explicit GC
            del state_dict
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Failed to load Text Encoder: {e}")
            raise e
        pbar.update(1)

        # Load Transformer
        transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)
        # CRITICAL: Add action parameters to match WorldPlay architecture
        transformer.add_discrete_action_parameters()
        
        # Load WorldPlay fine-tuned weights
        # Prefer optimized safetensors (19GB) over original checkpoint (49GB)
        if os.path.exists(ckpt_safetensors):
            logging.info(f"Loading optimized weights from {ckpt_safetensors}")
            print(f"Loading WorldPlay weights from {ckpt_safetensors}...")
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_safetensors)
            # Already cleaned during extraction, load directly
            try:
                missing, unexpected = transformer.load_state_dict(state_dict, strict=True)
                logging.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            except Exception as e:
                logging.warning(f"Strict loading failed: {e}. Trying strict=False.")
                missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
                logging.warning(f"Loaded with strict=False. Missing: {len(missing)}")
        elif os.path.exists(ckpt_pt):
            # Fallback to original .pt file (slower, larger)
            logging.info(f"Loading checkpoint from {ckpt_pt} (consider running extract_worldplay_weights.py)")
            print(f"Loading WorldPlay weights from {ckpt_pt} (this is slow, consider extracting to safetensors)...")
            state_dict = torch.load(ckpt_pt, map_location="cpu")
            
            # Extract 'generator' key if present (FSDP checkpoint format)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            
            # Clean prefix keys (model., _fsdp_wrapped_module.)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if new_key.startswith("model."):
                    new_key = new_key[6:]
                if new_key.startswith("_fsdp_wrapped_module."):
                    new_key = new_key[21:]
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            
            try:
                missing, unexpected = transformer.load_state_dict(state_dict, strict=True)
                logging.info(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            except Exception as e:
                logging.warning(f"Strict loading failed: {e}. Trying strict=False.")
                missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
                logging.warning(f"Loaded with strict=False. Missing: {len(missing)}")
        
        # Create Scheduler
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0, 
            use_dynamic_shifting=False 
        )

        # Assemble Pipeline
        pipe = WanPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler
        )
        # Enable CPU offload to save VRAM (fixes OOM)
        # This keeps models in RAM and only moves active component to GPU
        try:
            import accelerate
            pipe.enable_model_cpu_offload()
            logging.info("Enabled model CPU offloading")
        except ImportError:
            logging.warning("Accelerate not found, handling device manually (risk of OOM)")
            pipe.to(device=device, dtype=dtype)
        except Exception as e:
            logging.error(f"Failed to enable CPU offload: {e}")
            pipe.to(device=device, dtype=dtype)
        
        # Load LoRA if specified
        if lora is not None and lora != "None":
            lora_path = folder_paths.get_full_path("loras", lora)
            if lora_path and os.path.exists(lora_path):
                logging.info(f"Loading LoRA: {lora_path} with strength {lora_strength}")
                print(f"Loading LoRA: {lora_path} with strength {lora_strength}")
                try:
                    # WanPipeline uses unpack_weights logic or standard diffusers logic?
                    # Diffusers default: pipe.load_lora_weights(path, adapter_name="default")
                    
                    # NOTE: fuse_lora might be needed for efficiency or correctness?
                    # pipe.load_lora_weights(lora_path)
                    # pipe.fuse_lora(lora_scale=lora_strength)
                    
                    # Standard load does not fuse by default but attaches adapters.
                    # We can use 'scale' arg in load_lora_weights? No, that's not standard.
                    # Standard is cross_attention_kwargs={"scale": strength} during inference.
                    # BUT my generate node takes 'model' (pipe). 
                    # If I load it here, I need to know how to pass strength.
                    # WorldPlay Pipeline might NOT support cross_attention_kwargs scale fully unless fused.
                    
                    # Let's try explicit fuse first as it's cleaner for single-inference.
                    pipe.load_lora_weights(lora_path)
                    pipe.fuse_lora(lora_scale=lora_strength)
                    
                    logging.info("LoRA fused successfully.")
                except Exception as e:
                    logging.error(f"Failed to load LoRA: {e}")
                    print(f"Failed to load LoRA: {e}")
            else:
                 logging.warning(f"LoRA path not found: {lora_path}")
        
        return (pipe,)

class VNCCS_WorldPlay5B:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WORLDPLAY_MODEL",),
                "image": ("IMAGE",), # Added image input
                "trajectory": ("STRING", {"multiline": True, "default": "w-96"}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "turbo": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot of..."}),
                # Reference default negative prompt (translated/copied from generate.py)
                "negative_prompt": ("STRING", {"multiline": True, "default": "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走"}),
                "offload_scheme": (["sequential", "model", "none"], {"default": "sequential"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "images")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/WorldPlay"

    def generate(self, model, image, trajectory, width, height, num_frames, steps, guidance_scale, seed, turbo, prompt="", negative_prompt="", offload_scheme="sequential"):
        # Single source of truth for trajectory
        final_trajectory = trajectory
        
        # AUTO-CALCULATE num_frames from trajectory
        # Trajectory defines rigid duration. We must respect it.
        if isinstance(final_trajectory, str):
            try:
                pose_data = pose_string_to_json(final_trajectory)
                num_latents = len(pose_data)
                # Formula: latents * 4 - 3 = frames (because of temporal VAE with stride 4)
                actual_num_frames = num_latents * 4 - 3
                
                if actual_num_frames != num_frames:
                    print(f"[VNCCS] Overriding num_frames: User={num_frames} -> Trajectory={actual_num_frames}")
                    logging.warning(f"Overriding num_frames from {num_frames} to {actual_num_frames} to match trajectory length.")
                    num_frames = actual_num_frames
            except Exception as e:
                logging.error(f"Failed to parse trajectory for frame count: {e}")
                # Fallthrough to manual num_frames, might crash in pose_to_input
        
        pipe = model
        device = pipe.device
        dtype = pipe.transformer.dtype
        
        # Default chunk settings
        CHUNK_SIZE = 4 
        
        # Enable Native CPU Offloading (Diffusers + Accelerate)
        import comfy.model_management as mm
        
        try:
            if offload_scheme == "sequential":
                logging.info("Enabling Sequential CPU Offloading for WorldPlay Pipeline...")
                dev = mm.get_torch_device()
                gpu_id = dev.index if dev.index is not None else 0
                pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
            
            elif offload_scheme == "model":
                logging.info("Enabling Model CPU Offloading for WorldPlay Pipeline...")
                pipe.enable_model_cpu_offload(device=mm.get_torch_device())
                
            else: # "none"
                logging.info("CPU Offloading Disabled. Model should be on GPU.")
                # pipe.to(device) # Should already be managed or on device
                
        except Exception as e:
            logging.warning(f"Failed to enable Offloading ({offload_scheme}): {e}")
            pass 
        FIRST_CHUNK_SIZE = 1 # Reference default
        CONTEXT_WINDOW = 16 # User requested & Reference default
        
        # Calculate latent dimensions
        vae_scale_temporal = 4 # Wan2.1 default
        num_latent_frames = (num_frames - 1) // vae_scale_temporal + 1
        
        # Calculate needed chunks
        import math
        needed_chunks = math.ceil(max(0, num_latent_frames - FIRST_CHUNK_SIZE) / CHUNK_SIZE) + 1
        
        # Trajectory Parsing
        w2c, K, action = pose_to_input(final_trajectory, num_latent_frames)
        w2c = w2c.to(device=device, dtype=dtype).unsqueeze(0) # (1, T, 4, 4)
        K = K.to(device=device, dtype=dtype).unsqueeze(0) # (1, T, 3, 3)
        action = action.to(device=device, dtype=dtype).unsqueeze(0)
        
        torch.manual_seed(seed)
        
        # Prepare Image for Pipeline
        # ComfyUI Image is [B, H, W, C] in range [0, 1]
        # Pipeline expects [C, H, W] in range [-1, 1] (or filepath)
        
        # Take first image from batch
        source_image = image[0] # [H, W, C]
        source_image = source_image.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
        
        # Resize/Crop to target W/H
        import torch.nn.functional as F
        # Calculate scale to fill target dimensions
        h, w = source_image.shape[2], source_image.shape[3]
        scale = max(width / w, height / h)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h != h or new_w != w:
             source_image = F.interpolate(source_image, size=(new_h, new_w), mode="bicubic", align_corners=False)
        
        # Center Crop
        crop_h = (new_h - height) // 2
        crop_w = (new_w - width) // 2
        if crop_h > 0 or crop_w > 0:
            source_image = source_image[:, :, crop_h:crop_h+height, crop_w:crop_w+width]
            
        # Normalize to [-1, 1]
        source_image = (source_image * 2.0) - 1.0
        
        # Remove batch dim to match pipeline expectation [C, H, W]
        prepared_image = source_image.squeeze(0)
        
        # Enable VAE Slicing/Tiling to save memory
        try:
            # Try tiling first (most effective for large images)
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
                logging.info("Enabled VAE tiling")
            # Fallback to slicing
            elif hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
                logging.info("Enabled VAE slicing")
        except Exception as e:
            logging.warning(f"Failed to enable VAE optimization: {e}")
        
        # temp_img_path is no longer needed
        temp_img_path = None
        
        all_video_frames = []
        
        try:
            # Clear pipeline context from previous runs if any
            if hasattr(pipe, 'ctx'):
                pipe.ctx = None
            
            # Initialize accumulated latents
            current_latents = None
                 
            # Loop Chunks
            for chunk_i in range(needed_chunks):
                
                # Careful: The pipeline expects FULL pose data for the chunk context to be handled internally?
                # Actually pipeline Line 807: self.ctx["viewmats"][:, start_idx:end_idx, ...] = viewmats
                # It assigns the PASSED 'viewmats' to the context slice.
                # So we MUST pass only the CURRENT chunk's viewmats.
                
                start_idx = chunk_i * CHUNK_SIZE
                end_idx = start_idx + CHUNK_SIZE
                
                # Handle boundaries
                # Note: Reference generate.py just slices [start:end]. 
                # If end > length, it just takes what remains. 
                # Pipeline handles index logic.
                
                curr_w2c = w2c[:, start_idx:end_idx, ...]
                curr_K = K[:, start_idx:end_idx, ...]
                curr_action = action[:, start_idx:end_idx, ...]
                
                # Call Pipeline
                # Pass 'latents=current_latents' so valid history exists for AR
                current_latents = pipe(
                    image_path=prepared_image, # Pass prepared Tensor directly
                    latents=current_latents,   # Pass accumulated latents
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    prompt=prompt if chunk_i==0 else None, # Prompt only needed first time
                    negative_prompt=negative_prompt,
                    viewmats=curr_w2c,
                    Ks=curr_K,
                    action=curr_action,
                    chunk_i=chunk_i,
                    few_step=turbo,
                    first_chunk_size=CHUNK_SIZE, 
                    context_window_length=CONTEXT_WINDOW,
                    output_type="latent", # Return latents so we can decode or accumulate
                    return_dict=False
                )
                
                
                # Ideally we decode `chunk_size` times for standard chunks.
                # But for the very first chunk?
                # generate.py uses `range(4)` regardless.
                # Let's stick to CHUNK_SIZE=4 iteration target, but stop if None returned.
                
                for _ in range(4): # Try to decode up to 4 frames for this chunk
                    video_frame = pipe.decode_next_latent(output_type="np")
                    if video_frame is not None:
                        all_video_frames.append(torch.from_numpy(video_frame))
                    else:
                        break # No more frames to decode for now
        
        finally:
            if temp_img_path and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            # Cleanup context
            pipe.ctx = None
            pipe._kv_cache = None
            if hasattr(pipe, 'dist_vae'):
                 pipe.dist_vae = None
                 
        if not all_video_frames:
             return (torch.zeros((1, 1, height, width, 3)), torch.zeros((1, height, width, 3)))

        # all_video_frames is list of [1, H, W, C] (from np)
        # Cat to [T, H, W, C]
        video_tensor = torch.cat(all_video_frames, dim=0) 
        
        # Ensure it's [T, H, W, C]
        # video_frame from decode_next_latent (np) is [1, H, W, C] usually?
        # Actually pipeline postprocess: [B, H, W, C]. B=1. 
        # So yes, cat dim 0 gives [T, H, W, C].
        
        return (video_tensor, video_tensor[0].unsqueeze(0))

class VNCCS_RoomCoverageTrajectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coverage_mode": (["full_360", "hemisphere", "orbital", "custom"], {"default": "full_360"}),
                "num_chunks": ("INT", {"default": 6, "min": 1, "max": 24}), # Multiplier for length
            },
            "optional": {
                "custom_trajectory": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "generate_trajectory"
    CATEGORY = "VNCCS/WorldPlay"
    
    def generate_trajectory(self, coverage_mode, num_chunks, custom_trajectory):
        if coverage_mode == "custom":
            return (custom_trajectory,)
            
        # 1 chunk = 16 frames = 4 latents? 
        # Actually in WorldPlay readme:
        # w-96 corresponds to 96 latents.
        # Each chunk is usually small.
        # Let's assume standard unit of movement per chunk.
        
        # Defined patterns
        # We construct a string like "w-16, right-8, ..."
        
        if coverage_mode == "full_360":
            # Rotate around and move forward to cover room
            # "w-16, right-8" repeated covers ground.
            # Then look up/down.
            
            # Simple 360 spin with forward motion:
            # (Move 16, Turn 8) * num_chunks
            # This is a spiraling outward or just walking in circle?
            # "w-16" move forward 16 units. "right-8" yaw right 8 units.
            parts = []
            for _ in range(num_chunks):
                parts.append("w-16")
                parts.append("right-8")
                
            # Add some look up/down at the end to see ceiling/floor
            parts.append("up-8")
            parts.append("right-16") # Pan ceiling
            parts.append("down-16") 
            parts.append("right-16") # Pan floor
            
            trajectory = ", ".join(parts)
            
        elif coverage_mode == "hemisphere":
            # Looking down and spinning
            # "down-16" (look down), then "right-32" (spin)
            trajectory = "down-8, right-{}".format(num_chunks * 16)
            
        elif coverage_mode == "orbital":
            # Orbiting an object center
            # Move right/left while rotating opposite to keep center in view
            # "d-16" (move right), "left-8" (rotate left)
            parts = []
            for _ in range(num_chunks):
                parts.append("d-16")
                parts.append("left-4") 
            trajectory = ", ".join(parts)
            
        else:
            trajectory = "w-96"

        return (trajectory,)

class VNCCS_VideoFrameOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",), # Renamed from images
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "min_frames": ("INT", {"default": 24, "min": 1, "max": 1000}),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("optimized_video", "optimized_images")
    FUNCTION = "optimize"
    CATEGORY = "VNCCS/WorldPlay"
    
    def optimize(self, video, threshold, min_frames):
        # video: [T, H, W, C] (Standard Comfy tensor batch)
        # Logic remains the same, just operating on the tensor
        images = video # Alias for internal logic
        
        if len(images) <= min_frames:
            return (images, images)
        
        keep_indices = [0]
        last_kept = images[0]
        
        for i in range(1, len(images)):
            curr = images[i]
            # Mean absolute diff
            diff = torch.mean(torch.abs(curr - last_kept))
            
            if diff > threshold:
                keep_indices.append(i)
                last_kept = curr
                
        if len(keep_indices) < min_frames:
             # If too few, linear interpolation of indices
             # Use numpy for linspace
             target_indices = np.linspace(0, len(images)-1, min_frames, dtype=int)
             keep_indices = sorted(list(set(target_indices)))
             
        optimized = images[keep_indices]
        return (optimized, optimized)

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldPlay5B": VNCCS_LoadWorldPlay5B,
    "VNCCS_WorldPlay5B": VNCCS_WorldPlay5B,
    "VNCCS_RoomCoverageTrajectory": VNCCS_RoomCoverageTrajectory,
    "VNCCS_VideoFrameOptimizer": VNCCS_VideoFrameOptimizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldPlay5B": "Load WorldPlay 5B Model",
    "VNCCS_WorldPlay5B": "WorldPlay 5B Sampler",
    "VNCCS_RoomCoverageTrajectory": "WorldPlay Room Coverage Trajectory",
    "VNCCS_VideoFrameOptimizer": "WorldPlay Video Frame Optimizer"
}
