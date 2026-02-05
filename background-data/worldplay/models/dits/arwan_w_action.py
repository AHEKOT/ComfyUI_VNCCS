# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from platform import processor
from typing import Any, Dict, Optional, Tuple, Union, List

import os
import torch
import json
import datetime
import torch.nn as nn
import torch.nn.functional as F

from ..camera_rope import prope_qkv
# Mocks for distributed functions
def get_sp_world_size(): return 1
def get_sp_parallel_rank(): return 0
def all_to_all_sp(tensor, *args, **kwargs): return tensor
def sequence_model_parallel_all_gather(tensor, *args, **kwargs): return tensor

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from sageattention import sageattn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CausalCameraPRopeWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        is_cache: Optional[bool] = False,
        idx: Optional[int] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                x1, x2 = x[..., 0], x[..., 1]
                
                # Dynamic matching of RoPE sequence length
                seq_len = x1.shape[-2]
                rope_len = freqs_cos.shape[-2]
                
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                
                # If RoPE is longer (context window padding), slice it
                if rope_len > seq_len:
                    cos = cos[:, :, :seq_len, :]
                    sin = sin[:, :, :seq_len, :]
                # if RoPE is shorter (e.g. repeated sequence), repeat it
                elif seq_len > rope_len:
                    repeats = seq_len // rope_len
                    cos = cos.repeat(1, 1, repeats, 1)
                    sin = sin.repeat(1, 1, repeats, 1)

                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query_rope = apply_rotary_emb(query, *rotary_emb)
            key_rope = apply_rotary_emb(key, *rotary_emb)

        else:
            query_rope = query
            key_rope = key

        pad_size = 0
        if os.getenv("WORLD_SIZE", "0") != "0":
            world_size = get_sp_world_size()
            head_num = query.shape[1]
            if head_num % world_size != 0:
                # pad head_num
                pad_size = world_size - head_num % world_size
                query = torch.nn.functional.pad(query, (0, 0, 0, 0, 0, pad_size))
                key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, pad_size))
                value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, pad_size))
                query_rope = torch.nn.functional.pad(
                    query_rope, (0, 0, 0, 0, 0, pad_size)
                )
                key_rope = torch.nn.functional.pad(key_rope, (0, 0, 0, 0, 0, pad_size))

            query = all_to_all_sp(query, scatter_dim=1, gather_dim=2)
            key = all_to_all_sp(key, scatter_dim=1, gather_dim=2)
            value = all_to_all_sp(value, scatter_dim=1, gather_dim=2)
            query_rope = all_to_all_sp(query_rope, scatter_dim=1, gather_dim=2)
            key_rope = all_to_all_sp(key_rope, scatter_dim=1, gather_dim=2)

        value_rope = value

        # Dynamic patch calculation based on spatial shape (e.g. 60x60)
        # We infer the number of spatial patches by dividing seqlen by number of cameras
        query_prope, key_prope, value_prope, apply_fn_o = prope_qkv(
            query,
            key,
            value,
            viewmats=viewmats,
            Ks=Ks,
            # Pass None to trigger automatic shape inference in prope_qkv
            patches_x=None,
            patches_y=None,
        )  # [batch, num_heads, seqlen, head_dim]


        kv_cache_return = {}
        if kv_cache is not None:
            cache_key = kv_cache["k"]
            cache_value = kv_cache["v"]
        else:
            cache_key = None
            cache_value = None
        
        # CPU Offloading Logic: Move cache slice to GPU if needed
        device = query.device
        if cache_key is not None and cache_key.device != device:
             cache_key = cache_key.to(device)
        if cache_value is not None and cache_value.device != device:
             cache_value = cache_value.to(device)


        if cache_value is not None and not is_cache:
            cache_key_rope, cache_key_prope = cache_key.chunk(2, dim=-1)
            cache_value_rope, cache_value_prope = cache_value.chunk(2, dim=-1)

            key_rope = torch.cat([cache_key_rope, key_rope], dim=-2)
            value_rope = torch.cat([cache_value_rope, value_rope], dim=-2)

            key_prope = torch.cat([cache_key_prope, key_prope], dim=-2)
            value_prope = torch.cat([cache_value_prope, value_prope], dim=-2)

        if is_cache:
            kv_cache_return["k"] = torch.cat([key_rope, key_prope], dim=-1)
            kv_cache_return["v"] = torch.cat([value_rope, value_prope], dim=-1)

        # INCREMENTAL CONCATENATION to minimize peak VRAM
        # Doing cat([A, B]) then del A, B is safer than cat for all Q, K, V at once.
        query_all = torch.cat([query_rope, query_prope], dim=0)
        del query_rope, query_prope, query
        
        key_all = torch.cat([key_rope, key_prope], dim=0)
        del key_rope, key_prope, key
        
        value_all = torch.cat([value_rope, value_prope], dim=0)
        del value_rope, value_prope, value
        
        # Save dtype for later use
        query_dtype = query_all.dtype

        # FORCE SDPA with Chunking for maximum stability on 16GB VRAM
        # SageAttention causes OOM due to large quantization buffers.
        # Even standard SDPA can OOM if it falls back to math backend (approx 20GB for full ctx).
        # We chunk queries to reduce peak memory usage.
        
        q_len = query_all.shape[-2]
        # Chunk size: 256 tokens seems safe (reduces 20GB -> ~5GB peak)
        SDPA_CHUNK_SIZE = 256
        
        if q_len > SDPA_CHUNK_SIZE:
            # Pre-allocate output tensor to avoid list accumulation and cat spike
            hidden_states_all = torch.empty_like(query_all)
            
            for i in range(0, q_len, SDPA_CHUNK_SIZE):
                q_chunk = query_all[..., i : i + SDPA_CHUNK_SIZE, :]
                
                # In-place write
                hidden_states_all[..., i : i + SDPA_CHUNK_SIZE, :] = F.scaled_dot_product_attention(
                    q_chunk, key_all, value_all, dropout_p=0.0
                )
                
                # Explicit cleanup
                del q_chunk
        else:
            hidden_states_all = F.scaled_dot_product_attention(
                query_all, key_all, value_all, dropout_p=0.0
            )

        hidden_states_all = hidden_states_all.transpose(
            1, 2
        )  # [batch * 2, seqlen, per_sp_num_heads, head_dim]
        
        # Cleanup SDPA inputs
        del query_all, key_all, value_all

        hidden_states_rope, hidden_states_prope = hidden_states_all.chunk(2, dim=0)
        
        # Cleanup large combined tensor
        del hidden_states_all
        hidden_states_prope = apply_fn_o(hidden_states_prope.transpose(1, 2)).transpose(
            1, 2
        )

        if os.getenv("WORLD_SIZE", "0") != "0":
            hidden_states_rope = all_to_all_sp(
                hidden_states_rope, scatter_dim=1, gather_dim=2
            )
            hidden_states_prope = all_to_all_sp(
                hidden_states_prope, scatter_dim=1, gather_dim=2
            )

        if pad_size != 0:
            hidden_states_rope, _ = hidden_states_rope.split(
                [hidden_states_rope.shape[2] - pad_size, pad_size], dim=2
            )
            hidden_states_prope, _ = hidden_states_prope.split(
                [hidden_states_prope.shape[2] - pad_size, pad_size], dim=2
            )

        hidden_states_prope = hidden_states_prope.flatten(2, 3)
        hidden_states_prope = hidden_states_prope.to(query_dtype)
        hidden_states_prope = attn.to_out_prope[0](hidden_states_prope)

        hidden_states_rope = hidden_states_rope.flatten(2, 3)
        hidden_states_rope = hidden_states_rope.to(query_dtype)
        hidden_states_rope = attn.to_out[0](hidden_states_rope)
        hidden_states_rope = attn.to_out[1](hidden_states_rope)

        # In-place addition to save memory
        hidden_states_rope.add_(hidden_states_prope)
        hidden_states = hidden_states_rope

        return hidden_states, kv_cache_return


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                x1, x2 = x[..., 0], x[..., 1]
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            assert False
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # SP 16 Pad

        assert attention_mask is None

        hidden_states = sageattn(
            query, key, value, tensor_layout="HND", is_causal=False
        )
        # hidden_states = F.scaled_dot_product_attention(
        #    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_seq_len, in_features)
            )
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


# add the discrete action
class WanActionTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        action_embed_dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.action_embed_dim = action_embed_dim
        self.dim = dim
        self.time_freq_dim = time_freq_dim

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

    def forward(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        action = self.timesteps_proj(action)
        action_embedder_dtype = next(iter(self.action_embedder.parameters())).dtype
        if (
            action.dtype != action_embedder_dtype
            and action_embedder_dtype != torch.int8
        ):
            action = action.to(action_embedder_dtype)
        action = self.action_embedder(action).type_as(encoder_hidden_states)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        temb = temb + action

        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        ppf = max(1, ppf) # Fix for single-frame latent when p_t > 1
        pph = max(1, pph)
        ppw = max(1, ppw)

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CausalCameraPRopeWanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        kv_cache,
        is_cache,
        idx,
        viewmats,
        Ks,
        context_frames_list=None,
        current_start=0,
        current_end=None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # Turn bidrectional self-attention to causal self-attention

        # design special adaln for diffusion forcing
        if hidden_states.shape[0] != temb.shape[0]:
            # print("Diffusion forcing trainning! Adjust the AdaLN operation!")
            latent_length = temb.shape[0] // hidden_states.shape[0]  # latent length
            token_length = hidden_states.shape[1] // latent_length
            if os.getenv("WORLD_SIZE", "0") != "0":
                token_length = token_length * get_sp_world_size()
            shift_msa = shift_msa.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            scale_msa = scale_msa.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            gate_msa = gate_msa.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            c_shift_msa = c_shift_msa.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            c_scale_msa = c_scale_msa.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            c_gate_msa = c_gate_msa.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            shift_msa = shift_msa.repeat_interleave(token_length, dim=1)
            scale_msa = scale_msa.repeat_interleave(token_length, dim=1)
            gate_msa = gate_msa.repeat_interleave(token_length, dim=1)
            c_shift_msa = c_shift_msa.repeat_interleave(token_length, dim=1)
            c_scale_msa = c_scale_msa.repeat_interleave(token_length, dim=1)
            c_gate_msa = c_gate_msa.repeat_interleave(token_length, dim=1)

            if os.getenv("WORLD_SIZE", "0") != "0":
                world_size = get_sp_world_size()
                rank = get_sp_parallel_rank()
                shift_msa = shift_msa.chunk(world_size, dim=1)[rank]
                scale_msa = scale_msa.chunk(world_size, dim=1)[rank]
                gate_msa = gate_msa.chunk(world_size, dim=1)[rank]
                c_shift_msa = c_shift_msa.chunk(world_size, dim=1)[rank]
                c_scale_msa = c_scale_msa.chunk(world_size, dim=1)[rank]
                c_gate_msa = c_gate_msa.chunk(world_size, dim=1)[rank]

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output, kv_cache_return = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            is_cache=is_cache,
            idx=idx,
            viewmats=viewmats,
            Ks=Ks,
            context_frames_list=context_frames_list,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        return hidden_states, kv_cache_return


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanActionTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
            action_embed_dim=8,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False
        self.inner_dim = inner_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        current_start=None,
        current_end=None,
        kv_cache: dict = None,
        is_cache: bool = False,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        window_frames: int = None,
        context_frames_list: List[int] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_states = hidden_states.to(encoder_hidden_states.dtype)
        viewmats = (
            viewmats.to(encoder_hidden_states.dtype) if viewmats is not None else None
        )
        Ks = Ks.to(encoder_hidden_states.dtype) if Ks is not None else None
        action = action.to(encoder_hidden_states.dtype) if action is not None else None
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Fix for single-frame latent when p_t > 1
        # Ensures RoPE is generated for at least one temporal patch
        post_patch_num_frames = max(1, post_patch_num_frames)


        ## if hidden_states is smaller than the window size, we need to pad it
        if post_patch_num_frames < window_frames:
            padding = window_frames - post_patch_num_frames
            # pad hidden_state to the window size in the dim=2: [B, C, T, H, W] -> [B, C, T + padding, H, W]
            padding_hidden_states = F.pad(
                hidden_states, (0, 0, 0, 0, 0, padding), value=0.0
            )  # [B, C, T + padding, H, W]
        elif post_patch_num_frames == window_frames:
            padding_hidden_states = hidden_states.clone()
        else:
            # if hidden_states is larger than the window size, we need to slice it
            padding_hidden_states = hidden_states[:, :, :window_frames, :, :]

        # 得到所有的rope参数
        rotary_emb = self.rope(padding_hidden_states)

        hidden_states = self.patch_embedding(hidden_states)  # [B, C, T, H, W] -> [B, C', T', H', W']
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                action, timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))  # [B, 9216] -> [B, 6, 1536]

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        if is_cache:
            _kv_cache_now = []
            transformer_num_layers = 30
            for i in range(transformer_num_layers):
                _kv_cache_now.append({"k": None, "v": None})

        rotary_emb0 = rotary_emb[0][:, :, current_start:current_end, :]
        rotary_emb1 = rotary_emb[1][:, :, current_start:current_end, :]
        del rotary_emb

        # SP
        if os.getenv("WORLD_SIZE", "0") != "0":
            world_size = get_sp_world_size()
            rank = get_sp_parallel_rank()
            hidden_states = hidden_states.chunk(chunks=world_size, dim=1)[rank]
            rotary_emb0 = rotary_emb0.chunk(chunks=world_size, dim=2)[rank]
            rotary_emb1 = rotary_emb1.chunk(chunks=world_size, dim=2)[rank]
        rotary_emb = (rotary_emb0, rotary_emb1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for idx, block in enumerate(self.blocks):
                hidden_states, t_kv = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    kv_cache[idx] if kv_cache is not None else None,
                    is_cache,
                    idx,
                    viewmats,
                    Ks,
                    context_frames_list=context_frames_list,
                )
                if is_cache:
                    _kv_cache_now[idx] = t_kv
        else:
            for idx, block in enumerate(self.blocks):
                hidden_states, t_kv = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    kv_cache[idx] if kv_cache is not None else None,
                    is_cache,
                    idx,
                    viewmats,
                    Ks,
                    context_frames_list=context_frames_list,
                )
                if is_cache:
                    _kv_cache_now[idx] = t_kv

        if is_cache:
            return _kv_cache_now

        if os.getenv("WORLD_SIZE", "0") != "0":
            hidden_states = sequence_model_parallel_all_gather(
                hidden_states.contiguous(), dim=1
            )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if shift.shape[0] != hidden_states.shape[0]:
            # Diffusion forcing training, adjust the AdaLN operation
            latent_length = temb.shape[0] // hidden_states.shape[0]
            token_length = hidden_states.shape[1] // latent_length
            shift = shift.reshape(hidden_states.shape[0], -1, hidden_states.shape[2])
            scale = scale.reshape(hidden_states.shape[0], -1, hidden_states.shape[2])
            # operate on the hidden_states
            hidden_states = (
                self.norm_out(hidden_states.float())
                * (1 + scale.repeat_interleave(token_length, dim=1))
                + shift.repeat_interleave(token_length, dim=1)
            ).type_as(hidden_states)
        else:
            hidden_states = (
                self.norm_out(hidden_states.float()) * (1 + scale) + shift
            ).type_as(hidden_states)

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def add_discrete_action_parameters(self):
        # Determine target dtype and device from existing parameters
        target_dtype = self.dtype
        target_device = self.device

        self.condition_embedder.action_embedder = TimestepEmbedding(
            in_channels=self.condition_embedder.time_freq_dim,
            time_embed_dim=self.condition_embedder.dim,
        ).to(device=target_device, dtype=target_dtype)
        
        nn.init.zeros_(self.condition_embedder.action_embedder.linear_2.weight)

        if self.condition_embedder.action_embedder.linear_2.bias is not None:
            nn.init.zeros_(self.condition_embedder.action_embedder.linear_2.bias)

        # prope
        for block in self.blocks:
            block.attn1.to_out_prope = torch.nn.ModuleList(
                [
                    torch.nn.Linear(block.attn1.inner_dim, self.inner_dim, bias=True).to(device=target_device, dtype=target_dtype),
                ]
            )
            nn.init.zeros_(block.attn1.to_out_prope[0].weight)

            if block.attn1.to_out_prope[0].bias is not None:
                nn.init.zeros_(block.attn1.to_out_prope[0].bias)
