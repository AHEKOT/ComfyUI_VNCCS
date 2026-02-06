import html

# import json
from typing import Any, Callable, Dict, List, Optional, Union

import os
import time
import regex as re
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from ..models.utils import (
    shard_latents_dim_across_sp,
    select_mem_frames_wan,
)
from ..utils.retrieval_context import generate_points_in_sphere
# from distributed.parallel_state import (
#     get_sp_parallel_rank,
#     get_sp_parallel_local_rank,
#     get_sp_world_size,
# )
def get_sp_world_size(): return 1
def get_sp_parallel_rank(): return 0
def get_sp_parallel_local_rank(): return 0

# from distributed.communication_op import sequence_model_parallel_all_gather
def sequence_model_parallel_all_gather(x, *args, **kwargs): return x

from .par_vae_tools import distribute_data_to_gpus, gather_results
from .helper import CHUNK_SIZE

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = (
            self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        if os.getenv("WORLD_SIZE", "0") != "0":
            self.parallel_run = True
        else:
            self.parallel_run = False
        self.par_vae_encode = False
        self.par_vae_decode = self.parallel_run and True

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}"
            )

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError(
                "`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def init_kv_cache(self):
        self._kv_cache = []
        self._kv_cache_neg = []
        transformer_num_layers = len(self.transformer.blocks)
        for i in range(transformer_num_layers):
            self._kv_cache.append({"k": None, "v": None})
            self._kv_cache_neg.append({"k": None, "v": None})

        self.use_kv_cache = True

    def resize_center_crop_to_tensor(self, img_path, target_size):
        """
        Read image -> Resize -> Center crop -> Convert to torch.tensor -> Normalize to [-1, 1]
        :param img_path: Path to the image
        :param target_size: (width, height)
        :return: torch.Tensor, shape [C, H, W], range [-1, 1]
        """
        img = Image.open(img_path).convert("RGB")
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # --- Step 1: proportional resize ---
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # --- Step 2: center crop ---
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))

        # --- Step 3: to numpy，and then tensor ---
        img_np = np.array(img).astype(np.float32)  # [H, W, C], 0~255
        img_np = img_np / 127.5 - 1.0  # normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]

        return img_tensor

    def init_decode_state(self, all_latents, chunk_i, generated_num=None):
        """
        Initialize the decoding state and set all latents required for decoding.
        """
        # Limit decoding to what has been generated so far
        limit = generated_num if generated_num is not None else all_latents.shape[2]
        self._decode_state["total_latents"] = limit
        
        self._decode_state["all_latents"] = all_latents
        
        # Only reset index if starting fresh (chunk 0)
        if chunk_i == 0:
            self._decode_state["current_latent_idx"] = 0
            
        self._decode_state["chunk_i"] = chunk_i
        
        # Determine which VAE to use for metadata
        if self._decode_state["par_vae_decode"]:
            active_vae = self.dist_vae.pipe
        else:
            active_vae = self.vae

        if self._decode_state["latents_mean"] is None:
            latents_mean = (
                torch.tensor(active_vae.config.latents_mean)
                .view(1, active_vae.config.z_dim, 1, 1, 1)
                .to(all_latents.device, all_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(
                active_vae.config.latents_std
            ).view(1, active_vae.config.z_dim, 1, 1, 1).to(
                all_latents.device, all_latents.dtype
            )

            self._decode_state["latents_mean"] = latents_mean
            self._decode_state["latents_std"] = latents_std

    def decode_next_latent(self, output_type="np"):
        """
        Decode the next latents and return the decoded frames.
        If all latents have been decoded, return None.
        """

        current_idx = self._decode_state["current_latent_idx"]
        total_latents = self._decode_state["total_latents"]

        if current_idx >= total_latents:
            return None  # all latents have been decoded

        # Get the current latent to be decoded.
        all_latents = self._decode_state["all_latents"]
        chunk_i = self._decode_state["chunk_i"]
        current_latent = all_latents[
            :, :, current_idx : current_idx + 1, :, :
        ]  # [B, C, 1, H, W]
        
        # DEBUG: Log if slicing fails
        if current_latent.shape[2] == 0:
            print(f"[Pipeline] decode_next_latent: Sliced Empty Tensor! Idx={current_idx}, Total={total_latents}, Shape={all_latents.shape}")
            return None

        sp_world_size = self._decode_state["sp_world_size"]
        rank_in_sp_group = self._decode_state["rank_in_sp_group"]
        latents_mean = self._decode_state["latents_mean"]
        latents_std = self._decode_state["latents_std"]

        # Normalize latents
        current_latent = current_latent.to(self.vae.dtype)

        # Apply scaling/shifting logic matching __call__ (Fix for WorldPlay VAE)
        # Robust access for FrozenDict which might not support attribute access for keys
        def get_cfg(key, default=None):
            val = getattr(self.vae.config, key, None)
            if val is None and hasattr(self.vae.config, "get"):
                val = self.vae.config.get(key, None)
            return val if val is not None else default

        if get_cfg("shift_factor") is not None:
            kln_pre_scale = get_cfg("kln_pre_scale", 1.0)
            kln_pre_shift = get_cfg("kln_pre_shift", 0.0)
            current_latent = current_latent / kln_pre_scale + kln_pre_shift
        else:
            scaling_factor = get_cfg("scaling_factor", 1.0)
            current_latent = current_latent / scaling_factor

        latents_mean = latents_mean.to(current_latent.dtype)
        latents_std = latents_std.to(current_latent.dtype)
        current_latent = current_latent / latents_std + latents_mean

        if self._decode_state["par_vae_decode"]:
            local_rank_in_sp_group_vae = self.ctx["local_rank_in_sp_group"]

            decode_input = distribute_data_to_gpus(
                current_latent,
                dim=4,
                rank=rank_in_sp_group,
                local_rank=local_rank_in_sp_group_vae,
                num_gpus=sp_world_size,
                dtype=current_latent.dtype,
            )
            video = self.dist_vae.pipe.decode(
                decode_input,
                return_dict=False,
                is_first_chunk=(current_idx == 0 and chunk_i == 0),
            )
            video = gather_results(
                video[0],
                dim=4,
                world_size=sp_world_size,
            )
        else:
            # Single-GPU Standard VAE decode
            # Fix: WanVAE Tiling crashes on T=1 inputs (torch.cat list empty).
            # Since we are decoding frame-by-frame, tiling is unnecessary and buggy.
            was_tiling = getattr(self.vae, "use_tiling", False)
            if was_tiling:
                self.vae.disable_tiling()
            
            try:
                video = self.vae.decode(current_latent, return_dict=False)[0]
            finally:
                if was_tiling:
                    self.vae.enable_tiling()

        # Post-process video frame
        # (This part is usually done via VideoProcessor, but we do a quick denorm here if needed)
        # Actually VideoProcessor is used on the full video usually.
        # But the original code had: video = (video * 0.5 + 0.5).clamp(0, 1) right here? 
        # Wait, let me check where line 466 went.
        video = (video * 0.5 + 0.5).clamp(0, 1)

        video = self.video_processor.postprocess_video(video, output_type=output_type)

        torch.cuda.synchronize()

        # Update the index
        self._decode_state["current_latent_idx"] += 1

        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        few_step=False,
        first_chunk_size=1,
        image_path=None,
        viewmats=None,
        Ks=None,
        action=None,
        use_memory=True,
        context_window_length=None,  # mean the context length for each chunk
        chunk_i: int = 0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor`
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if chunk_i == 0:
            # Set default context window length for memory-efficient generation on 16GB cards
            if context_window_length is None:
                context_window_length = 12
            # print("init run")
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                negative_prompt,
                height,
                width,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
                guidance_scale_2,
            )

            if num_frames % self.vae_scale_factor_temporal != 1:
                logger.warning(
                    "`num_frames - 1` has to be divisible by %d. "
                    "Rounding to the nearest number.",
                    self.vae_scale_factor_temporal,
                )
                num_frames = (
                    num_frames
                    // self.vae_scale_factor_temporal
                    * self.vae_scale_factor_temporal
                    + 1
                )
            num_frames = max(num_frames, 1)

            if self.config.boundary_ratio is not None and guidance_scale_2 is None:
                guidance_scale_2 = guidance_scale

            self._guidance_scale = guidance_scale
            self._guidance_scale_2 = guidance_scale_2
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            device = self._execution_device
            
            # SANITIZE DEVICE: Prevent 'meta' device usage which causes "Cannot copy out of meta tensor"
            # This happens with accelerate sequential cpu offload sometimes.
            if str(device) == "meta" or device.type == "meta":
                print(f"[WorldPlay] Warning: Pipeline device is 'meta'. forcing to 'cuda' or 'cpu'.")
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            # Aggressive device forcing to avoid meta tensor issues with accelerate
            if device is None or str(device) == "meta" or (hasattr(device, "type") and device.type == "meta"):
                 # Check self.device as a backup but sanitize it too
                 device = self.device if (self.device and str(self.device) != "meta") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            
            if str(device) == "meta":
                 device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                 
            self.points_local = generate_points_in_sphere(50000, 8.0, device=device)

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            transformer_dtype = (
                self.transformer.dtype
                if self.transformer is not None
                else self.transformer_2.dtype
            )
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

            # 4. Prepare timesteps
            if not few_step:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                zeros_t = torch.zeros(1).to(timesteps.device).to(timesteps.dtype)
                timesteps = torch.cat([timesteps, zeros_t])
            else:
                sigmas = self.scheduler.sigmas
                timesteps = torch.tensor(
                    [1000.0000, 960.0000, 888.8889, 727.2728, 0.0]
                ).to(sigmas.device)
                sigmas = timesteps / 1000.0

            # 5. Prepare latent variables
            # Transformer in_channels is 48 (16 noise + 32 conditions), but we only generate noise for 16 channels.
            num_channels_latents = 16
            # num_channels_latents = (
            #    self.transformer.config.in_channels
            #    if self.transformer is not None
            #    else self.transformer_2.config.in_channels
            # )
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )

            self._num_timesteps = len(timesteps)
            chunk_size = CHUNK_SIZE
            n_latent = latents.shape[2]

            # set the noise level for the inference stage
            stabilization_level = 15

            sp_world_size = 1
            rank_in_sp_group = 0
            local_rank_in_sp_group = 0
            if self.parallel_run:
                sp_world_size = get_sp_world_size()
                rank_in_sp_group = get_sp_parallel_rank()
                local_rank_in_sp_group = get_sp_parallel_local_rank()

            first_image_condition = None
            if image_path is not None:
                if isinstance(image_path, torch.Tensor):
                    # Direct tensor input (C, H, W) normalized [-1, 1]
                    first_image = image_path.to(device).to(self.vae.dtype)
                else:
                    # Path string
                    first_image = self.resize_center_crop_to_tensor(image_path, (width, height)) # (C H W)

                # Ensure dimensions are [B, C, T, H, W]
                # If input is [C, H, W] -> [1, C, 1, H, W]
                if first_image.ndim == 3:
                     first_image = first_image.unsqueeze(0).unsqueeze(2) 
                
                # VAE Encoding with OOM protection (slicing should be enabled on VAE externally)
                first_image_condition = self.vae.encode(first_image, return_dict=False)[0].sample()
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(device, self.vae.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(device, self.vae.dtype)
                first_image_condition = (first_image_condition - latents_mean) * latents_std


            if chunk_i == 0:
                self.init_kv_cache()

                self._decode_state = {
                    "current_latent_idx": 0,
                    "total_latents": None,
                    "device": device,
                    "transformer_dtype": transformer_dtype,
                    "vae": self.vae,
                    "video_processor": self.video_processor,
                    "parallel_run": self.parallel_run,
                    "par_vae_decode": self.par_vae_decode,
                    "sp_world_size": sp_world_size,
                    "rank_in_sp_group": rank_in_sp_group,
                    "latents_mean": None,
                    "latents_std": None,
                    "is_initialized": False,
                }


            self.ctx = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
                "attention_kwargs": attention_kwargs,
                "transformer_dtype": (
                    self.transformer.dtype
                    if self.transformer is not None
                    else self.transformer_2.dtype
                ),
                # Denoising scheduler and state
                "timesteps": timesteps,
                "num_inference_steps": num_inference_steps,
                "latents": latents,
                "first_image_condition": first_image_condition,
                "chunk_size": chunk_size,
                "first_chunk_size": first_chunk_size,
                "n_latent": n_latent,
                "viewmats": torch.zeros(
                    1, n_latent, 4, 4, device=device, dtype=transformer_dtype
                ),
                "Ks": torch.zeros(
                    1, n_latent, 3, 3, device=device, dtype=transformer_dtype
                ),
                "action": torch.zeros(
                    1, n_latent, device=device, dtype=transformer_dtype
                ),
                "use_memory": use_memory,
                "context_window_length": context_window_length,
                "kv_cache": self._kv_cache,
                "kv_cache_neg": self._kv_cache_neg,
                "use_kv_cache": self.use_kv_cache,
                # Distributed related
                "sp_world_size": sp_world_size,
                "rank_in_sp_group": rank_in_sp_group,
                "local_rank_in_sp_group": local_rank_in_sp_group,
                # Other auxiliary parameters
                "device": device,
                "batch_size": batch_size,
                "few_step": few_step,
                "sigmas": sigmas if few_step else None,
                "stabilization_level": stabilization_level,
            }

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # ==========================================================
        if self.ctx is None and chunk_i > 0:
            raise ValueError(
                "When chunk_i > 0, the context self.ctx cannot be empty. Please ensure the call starts from chunk_i = 0."
            )
        prompt_embeds = self.ctx["prompt_embeds"]
        negative_prompt_embeds = self.ctx["negative_prompt_embeds"]
        # Extract device immediately for guards
        device = self.ctx["device"]
        guidance_scale = self.ctx["guidance_scale"]
        guidance_scale_2 = self.ctx["guidance_scale_2"]
        attention_kwargs = self.ctx["attention_kwargs"]
        transformer_dtype = self.ctx["transformer_dtype"]

        timesteps = self.ctx["timesteps"]
        num_inference_steps = self.ctx["num_inference_steps"]
        latents = self.ctx["latents"]
        first_image_condition = self.ctx["first_image_condition"]

        chunk_size = self.ctx["chunk_size"]
        first_chunk_size = self.ctx["first_chunk_size"]
        n_latent = self.ctx["n_latent"]
        stabilization_level = self.ctx["stabilization_level"]

        # GUARD: Ensure inputs are not meta tensors from node/previous run offload
        viewmats = viewmats.to(device)
        Ks = Ks.to(device)
        action = action.to(device)

        if chunk_i == 0:
            start_idx = 0
            end_idx = first_chunk_size
        else:
            start_idx = first_chunk_size + (chunk_i - 1) * chunk_size
            end_idx = start_idx + chunk_size

        self.ctx["viewmats"][:, start_idx:end_idx, ...] = viewmats
        self.ctx["Ks"][:, start_idx:end_idx, ...] = Ks
        self.ctx["action"][:, start_idx:end_idx, ...] = action

        viewmats = self.ctx["viewmats"]
        Ks = self.ctx["Ks"]
        action = self.ctx["action"]

        use_memory = self.ctx["use_memory"]
        context_window_length = self.ctx["context_window_length"]
        self._kv_cache = self.ctx["kv_cache"]
        self._kv_cache_neg = self.ctx["kv_cache_neg"]
        self.use_kv_cache = self.ctx["use_kv_cache"]

        sp_world_size_vae = self.ctx["sp_world_size"]
        rank_in_sp_group_vae = self.ctx["rank_in_sp_group"]
        local_rank_in_sp_group_vae = self.ctx["local_rank_in_sp_group"]

        batch_size = self.ctx["batch_size"]
        few_step = self.ctx["few_step"]
        sigmas = self.ctx["sigmas"]


        # Prepare separate condition and mask tensors for the 48-channel input
        # latents initialized above (16ch)
        full_cond_latents = torch.zeros_like(latents)
        full_mask_latents = torch.zeros_like(latents)
        
        if first_image_condition is not None:
            # Assuming first_image_condition is [1, 16, 1, H, W]
            # Place it at the first frame index
            full_cond_latents[:, :, :1] = first_image_condition
            full_mask_latents[:, :, :1] = 1.0 # Mask is 1 where condition exists
        
        # Restore the attributes of the pipeline
        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._num_timesteps = len(timesteps)
        self.num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        selected_frame_indices = None
        if chunk_i == 0:
            already_generate_num = first_chunk_size
            generate_latent_num = first_chunk_size
            if first_image_condition is not None:
                # Expand first_image_condition if missing dims
                if first_image_condition.ndim == 4: # [C, T, H, W] -> [1, C, T, H, W]
                     first_image_condition = first_image_condition.unsqueeze(0)
                
                latents[:, :, :1] = first_image_condition
            latents_curr = latents[:, :, :already_generate_num].to(device=device, dtype=prompt_embeds.dtype)
            
            print(f"[Pipeline] Chunk 0 Init: first_chunk_size={first_chunk_size}")
            print(f"[Pipeline] Latents Shape: {latents.shape}")
            print(f"[Pipeline] Latents Curr Shape: {latents_curr.shape}")
            print(f"[Pipeline] Generate Latent Num: {generate_latent_num}")
            
        else:
            if not few_step:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                zeros_t = torch.zeros(1).to(timesteps.device).to(timesteps.dtype)
                timesteps = torch.cat([timesteps, zeros_t])

            already_generate_num = chunk_i * chunk_size + first_chunk_size
            latents_curr = latents[:, :, :already_generate_num].to(device=device, dtype=prompt_embeds.dtype)
            generate_latent_num = chunk_size

            # add memory selection here
            current_frame_idx = chunk_i * chunk_size  # the current frame index to generate
            if (
                use_memory
                and (current_frame_idx >= context_window_length)
                and current_frame_idx < n_latent
            ):
                selected_frame_indices = select_mem_frames_wan(
                    viewmats[0].cpu().detach().float().numpy(),
                    current_frame_idx,
                    memory_frames=context_window_length,
                    temporal_context_size=12,
                    pred_latent_size=4,
                    device=device,
                    points_local=self.points_local,
                )
                # STRICT VRAM LIMIT: Clamp selection to context_window_length
                # This guarantees attention input never exceeds user-defined budget
                if len(selected_frame_indices) > context_window_length:
                     selected_frame_indices = selected_frame_indices[-context_window_length:]

            elif use_memory:
                selected_frame_indices = list(range(0, current_frame_idx))

        # 6. Denoising loop
        self._num_timesteps = len(timesteps) - 1
        with self.progress_bar(total=num_inference_steps if not few_step else 4) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt: continue

                current_model = self.transformer
                
                # Calculate spatial size (tokens per frame) dynamically
                p_t, p_h, p_w = current_model.config.patch_size
                spatial_size = (latents.shape[3] // p_h) * (latents.shape[4] // p_w)
                
                # PREPARE INPUT WINDOW (History + Current)
                # Following bi_rollout: NO KV CACHE in the denoising loop, reprocess full window
                if chunk_i == 0:
                    current_indices = list(range(already_generate_num))
                    target_indices = current_indices
                    t_now = torch.full((batch_size, len(target_indices)), t, device=device, dtype=timesteps.dtype)
                    timestep_input = t_now
                    generate_rope_start = 0
                else:
                    current_indices = list(range(already_generate_num - chunk_size, already_generate_num))
                    now_chunk_size = len(current_indices)
                    
                    # Fix history timesteps at 14 (stabilization_level - 1)
                    t_now = torch.full((batch_size, now_chunk_size), t, device=device, dtype=timesteps.dtype)
                    t_ctx = torch.full((batch_size, len(selected_frame_indices)), 14, device=device, dtype=timesteps.dtype)
                    timestep_input = torch.cat([t_ctx, t_now], dim=1)
                    
                    target_indices = selected_frame_indices + current_indices
                    generate_rope_start = len(selected_frame_indices) * spatial_size
                
                # Slice inputs according to the window
                latent_model_input = latents_curr[:, :, target_indices].clone() # Clone to avoid in-place update of main latents_curr here
                cond_input = full_cond_latents[:, :, target_indices]
                mask_input = full_mask_latents[:, :, target_indices]
                
                viewmats_input = viewmats[:, target_indices]
                Ks_input = Ks[:, target_indices]
                action_input = action[:, target_indices]
                
                now_window_latent_num = len(target_indices)
                generate_rope_end = now_window_latent_num * spatial_size

                # CFG expansion
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                    cond_input = torch.cat([cond_input] * 2)
                    mask_input = torch.cat([mask_input] * 2)
                    timestep_input = torch.cat([timestep_input] * 2)
                    
                    viewmats_guidance = torch.cat([viewmats_input] * 2)
                    Ks_guidance = torch.cat([Ks_input] * 2)
                    action_guidance = torch.cat([action_input] * 2)
                else:
                    viewmats_guidance = viewmats_input
                    Ks_guidance = Ks_input
                    action_guidance = action_input

                # Convert to transformer dtype & device
                latent_model_input = latent_model_input.to(device=device, dtype=transformer_dtype)
                cond_input = cond_input.to(device=device, dtype=transformer_dtype)
                mask_input = mask_input.to(device=device, dtype=transformer_dtype)
                timestep_flattened = timestep_input.to(device=device, dtype=transformer_dtype).flatten()
                
                # 48ch Concatenation [Noise, Cond, Mask]
                if current_model.config.in_channels == 48:
                    latent_model_final = torch.cat([latent_model_input, cond_input, mask_input], dim=1)
                else:
                    latent_model_final = latent_model_input

                # Clear memory before heavy forward pass
                if i % 10 == 0: 
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                torch.cuda.nvtx.range_push("noise_pred")
                noise_pred = current_model(
                    hidden_states=latent_model_final,
                    timestep=timestep_flattened,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    current_start=generate_rope_start,
                    current_end=generate_rope_end,
                    kv_cache=None, # NO KV CACHE for strict alignment
                    is_cache=False,
                    window_frames=now_window_latent_num,
                    viewmats=viewmats_guidance.to(device=device, dtype=transformer_dtype),
                    Ks=Ks_guidance.to(device=device, dtype=transformer_dtype),
                    action=action_guidance.to(device=device, dtype=transformer_dtype).flatten(),
                )[0]
                torch.cuda.nvtx.range_pop()

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # SLICE NOISE PRED: 48ch model might output 48ch, but we only need 16ch noise pred
                if noise_pred.shape[1] > latents.shape[1]:
                    noise_pred = noise_pred[:, :latents.shape[1]]

                # Scheduler step on the window
                latent_model_single = latent_model_input.chunk(2)[0] if self.do_classifier_free_guidance else latent_model_input
                
                latent_model_step = self.scheduler.step(
                    noise_pred, t, latent_model_single, return_dict=False
                )[0]
                
                # Update current chunk in the main latents tensor
                chunk_len = len(current_indices)
                latents[:, :, current_indices] = latent_model_step[:, :, -chunk_len:].to(latents.dtype)
                
                # Also update latents_curr so the next timestep sees the updated frames
                latents_curr[:, :, current_indices] = latents[:, :, current_indices].to(latents_curr.dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > self.num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()


                    
            # Memory Cleanup after loop
            del full_cond_latents
            del full_mask_latents
            if chunk_i > 0:
                # Use get to avoid NameError if these weren't defined due to conditional branch
                if 'cond_cache' in locals(): del cond_cache
                if 'mask_cache' in locals(): del mask_cache
                if 'latent_model_cache' in locals(): del latent_model_cache
            
            # Global cleanup
            if 'noise_pred' in locals(): del noise_pred
            if 'noise_pred_uncond' in locals(): del noise_pred_uncond
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            latents[:, :, :already_generate_num, :, :] = latents_curr

        # ==========================================================
        # Save the state updated during this call back to self.ctx
        self.ctx["latents"] = latents
        self.ctx["kv_cache"] = self._kv_cache
        self.ctx["kv_cache_neg"] = self._kv_cache_neg
        # ==========================================================

        # Automatically initialize the decoding state (for subsequent frame-by-frame decoding)
        # Note: Must be initialized here because the latents will be modified later
        if output_type == "latent":
            # Pass already_generate_num to limit decoding to valid frames
            # And prevent resetting the decoding index if chunk_i > 0
            limit_frames = locals().get("already_generate_num", latents.shape[2])
            self.init_decode_state(latents, chunk_i, generated_num=limit_frames)

        torch.cuda.synchronize()
        vae_decode_begin_time = time.time()
        if not output_type == "latent":
            if not self.par_vae_decode:
                if not "video_list" in locals():
                    video_list = []
                    video_list = []
                
                # Decode the *current chunk's* latents?
                # Wait, 'latents' variable here is likely 'latents_curr' (the denoised chunk).
                # We should decode 'latents_curr' and append.
                
                # Check if we should reconstruct full latents first or decode chunk by chunk.
                # If we decode chunk by chunk, we avoid OOM.
                
                # Prepare latents for VAE
                latents_for_vae = latents_curr.clone() # Use current denoised chunk
                
                if self.vae.config.shift_factor is not None:
                     latents_for_vae = latents_for_vae / self.vae.config.kln_pre_scale + self.vae.config.kln_pre_shift
                else:
                    latents_for_vae = latents_for_vae / self.vae.config.scaling_factor

                latents_for_vae = latents_for_vae.to(self.vae.dtype)
                
                # Standardize
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents_for_vae.device, latents_for_vae.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                     1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents_for_vae.device, latents_for_vae.dtype)
                
                latents_for_vae = latents_for_vae / latents_std + latents_mean
                
                # Decode
                video_chunk = self.vae.decode(
                    latents_for_vae, return_dict=False, is_first_chunk=chunk_i == 0
                )[0]
                
                video_chunk = self.video_processor.postprocess_video(
                    video_chunk, output_type=output_type
                )
                video_list.append(video_chunk)

            # END OF CHUNK LOOP
            
        # Concatenate all video chunks
        if output_type == "latent":
             return latents, None

        if len(video_list) > 0:
             # video_chunk is [B, C, F, H, W] or similar?
             # postprocess_video usually returns [B, F, H, W, C] (NHWC) or [B, C, F, H, W]?
             # If tensor, typically NCHW or similar.
             # Diffusers video processor often returns PT tensor [B, C, F, H, W].
             
             # Concatenate along Frame dimension.
             # If [B, C, F, H, W], dim=2.
             # If [B, F, H, W, C], dim=1.
             
             # Let's assume standard layout from VAE decode.
             video = torch.cat(video_list, dim=2) 
        else:
             video = None # Should not happen


        torch.cuda.synchronize()
        print("Vae decode time", time.time() - vae_decode_begin_time)
        torch.cuda.nvtx.range_pop()

        if not return_dict:
            return (
                video,
                latents,
            )

        return WanPipelineOutput(frames=video)
