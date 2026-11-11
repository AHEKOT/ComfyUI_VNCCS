"""Single-device WorldMirror V2 model and ComfyUI quantization.

Adapted from AHEKOT/ComfyUI_HYWorld2 and AHEKOT/VNCCS_World.
Copyright (c) 2025 MiuProject. Distributed under the MIT license in ../LICENSE.
"""

from __future__ import annotations

import collections.abc
import logging
import math
import random
from functools import partial
from itertools import repeat
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint


# models/layers/mlp.py

class _CameraMlp(nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer: Callable[..., nn.Module]=nn.GELU, drop: float=0.0, bias: bool=True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.inference_chunk_size = 0

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        chunk_size = int(getattr(self, 'inference_chunk_size', 0) or 0)
        if self.training or chunk_size <= 0 or x.dim() < 3 or (x.shape[-2] <= chunk_size):
            return self._forward_impl(x)
        return torch.cat([self._forward_impl(chunk) for chunk in torch.split(x, chunk_size, dim=-2)], dim=-2)

class MlpFP32(_CameraMlp):

    @staticmethod
    def map_to_args_to_float(args, kwargs):
        args = tuple((torch.float32 if isinstance(arg, torch.dtype) else arg for arg in args))
        kwargs = dict(kwargs)
        for key in kwargs:
            if key == 'dtype':
                kwargs[key] = torch.float32
        return (args, kwargs)

    def to(self, *args, **kwargs):
        self.fc1 = self.fc1.to(*args, **kwargs)
        args, kwargs = self.map_to_args_to_float(args, kwargs)
        self.fc2 = self.fc2.to(*args, **kwargs)
        return self

    def forward_infer(self, x):
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.erf(x * 2 ** (-0.5)))
        x = self.fc2(x.float())
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_infer(x)

# models/layers/patch_embed.py

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)

class PatchEmbed(nn.Module):

    def __init__(self, img_size: Union[int, Tuple[int, int]]=224, patch_size: Union[int, Tuple[int, int]]=16, in_chans: int=3, embed_dim: int=768, norm_layer: Optional[Callable]=None, flatten_embedding: bool=True) -> None:
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f'Input image height {H} is not a multiple of patch height {patch_H}'
        assert W % patch_W == 0, f'Input image width {W} is not a multiple of patch width: {patch_W}'
        x = self.proj(x)
        H, W = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

class PatchEmbed_Mlp(PatchEmbed):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten_embedding=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten_embedding)
        self.proj = nn.Sequential(PixelUnshuffle(patch_size), Permute((0, 2, 3, 1)), Mlp(in_chans * patch_size ** 2, 4 * embed_dim, embed_dim), Permute((0, 3, 1, 2)))

class PixelUnshuffle(nn.Module):

    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        if input.numel() == 0:
            C, H, W = input.shape[-3:]
            assert H and W and (H % self.downscale_factor == W % self.downscale_factor == 0)
            return input.view(*input.shape[:-3], C * self.downscale_factor ** 2, H // self.downscale_factor, W // self.downscale_factor)
        else:
            return F.pixel_unshuffle(input, self.downscale_factor)

class Permute(torch.nn.Module):
    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self):
        return f'Permute{self.dims}'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and (not isinstance(x, str)):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.inference_chunk_size = 0

    def _forward_impl(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        size = self.inference_chunk_size
        if not self.training and size > 0 and x.ndim >= 3 and x.shape[-2] > size:
            return torch.cat([self._forward_impl(part) for part in x.split(size, dim=-2)], dim=-2)
        return self._forward_impl(x)

# models/layers/swiglu_ffn.py

class SwiGLUFFN(nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer: Callable[..., nn.Module]=None, drop: float=0.0, bias: bool=True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
SwiGLU = SwiGLUFFN

class SwiGLUFFNFused(SwiGLU):

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer: Callable[..., nn.Module]=None, drop: float=0.0, bias: bool=True) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features, out_features=out_features, bias=bias)

# models/layers/drop_path.py

def drop_path(x, drop_prob: float=0.0, training: bool=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# models/layers/layer_scale.py

class LayerScale(nn.Module):

    def __init__(self, dim: int, init_values: Union[float, Tensor]=1e-05, inplace: bool=False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        gamma = self.gamma.to(x)
        return x.mul_(gamma) if self.inplace else x * gamma

# models/layers/norm_rope.py

class PositionGetter:

    def __init__(self) -> None:
        self.position_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            self.position_cache[key] = torch.cartesian_prod(y_coords, x_coords)
        cached_positions = self.position_cache[key]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class NormalizedRotaryPositionEmbedding2D(nn.Module):

    def __init__(self, *, head_dim: int, base: float=100.0, normalize_coords: Literal['min', 'max', 'separate']='separate', shift_coords: Union[float, None]=None, jitter_coords: Union[float, None]=None, rescale_coords: Union[float, None]=None, dtype: Union[torch.dtype, None]=None, device: Union[torch.device, None]=None, **ignored_kwargs) -> None:
        super().__init__()
        if len(ignored_kwargs) > 0:
            pass
        if head_dim % 4 != 0:
            raise ValueError('head_dim must be divisible by 4 for 2D RoPE')
        self.head_dim = head_dim
        self.base = base
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        quarter_dim = head_dim // 4
        self.register_buffer('periods', torch.empty(quarter_dim, device=device, dtype=dtype), persistent=True)
        self._init_periods()

    def _init_periods(self) -> None:
        quarter_dim = self.periods.shape[0]
        half_dim = self.head_dim // 2
        exponents = 2 * torch.arange(quarter_dim, device=self.periods.device, dtype=self.dtype) / half_dim
        periods = self.base ** exponents
        self.periods.data.copy_(periods)

    def _get_sincos_for_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        dd = {'device': device, 'dtype': dtype}
        if self.normalize_coords == 'max':
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_hw
            coords_w = torch.arange(0.5, W, **dd) / max_hw
        elif self.normalize_coords == 'min':
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_hw
            coords_w = torch.arange(0.5, W, **dd) / min_hw
        elif self.normalize_coords == 'separate':
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f'Unknown normalize_coords: {self.normalize_coords}')
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        if self.training:
            if self.shift_coords is not None:
                shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
                coords += shift_hw[None, :]
            if self.jitter_coords is not None:
                jitter_max = np.log(self.jitter_coords)
                jitter_hw = torch.empty(2, **dd).uniform_(-jitter_max, jitter_max).exp()
                coords *= jitter_hw[None, :]
            if self.rescale_coords is not None:
                rescale_max = np.log(self.rescale_coords)
                rescale_hw = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
                coords *= rescale_hw
        periods = self.periods.to(device=device, dtype=dtype)
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = torch.cat((angles, angles), dim=-1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return (sin, cos)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, 'Feature dimension must be even'
        assert positions.ndim == 3 and positions.shape[-1] == 2, 'Positions must have shape (batch_size, n_tokens, 2)'
        B, _, N, C_head = tokens.shape
        if C_head != self.head_dim:
            raise ValueError(f"Head dim {C_head} doesn't match configured {self.head_dim}")
        H = int(positions[..., 0].max().item() + 1)
        W = int(positions[..., 1].max().item() + 1)
        sin, cos = self._get_sincos_for_grid(H, W, tokens.device, tokens.dtype)
        indices = (positions[..., 0] * W + positions[..., 1]).long()
        flat_indices = indices.view(-1)
        gathered_sin = sin[flat_indices].view(B, 1, N, C_head)
        gathered_cos = cos[flat_indices].view(B, 1, N, C_head)
        return tokens * gathered_cos + _rotate_half(tokens) * gathered_sin

# models/layers/attention.py

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=True, proj_bias: bool=True, attn_drop: float=0.0, proj_drop: float=0.0, norm_layer: nn.Module=nn.LayerNorm, qk_norm: bool=False, fused_attn: bool=True, rope=None) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def _compute_qkv(self, x: Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = (self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype))
        return (q, k, v, B, N, C)

    def _apply_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

    def _project_output(self, x: Tensor, B: int, N: int, C: int) -> Tensor:
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, pos=None) -> Tensor:
        q, k, v, B, N, C = self._compute_qkv(x)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = self._apply_attention(q, k, v)
        return self._project_output(x, B, N, C)

class MemEffAttention(Attention):

    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if attn_bias is not None:
            raise AssertionError('xFormers is required for using nested tensors')
        return super().forward(x)

# models/layers/block.py

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)

class Block(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=True, proj_bias: bool=True, ffn_bias: bool=True, drop: float=0.0, attn_drop: float=0.0, init_values=None, drop_path: float=0.0, act_layer: Callable[..., nn.Module]=nn.GELU, norm_layer: Callable[..., nn.Module]=nn.LayerNorm, attn_class: Callable[..., nn.Module]=Attention, ffn_layer: Callable[..., nn.Module]=Mlp, qk_norm: bool=False, fused_attn: bool=True, rope=None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, fused_attn=fused_attn, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:

        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x

def drop_add_residual_stochastic_depth(x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float=0.0, pos=None) -> Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]
    if pos is not None:
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


NestedTensorBlock = Block

# models/layers/vision_transformer.py
log = logging.getLogger(__name__)

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class BlockChunk(nn.ModuleList):

    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class DinoVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True, proj_bias=True, drop_path_rate=0.0, drop_path_uniform=False, init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU, block_fn=NestedTensorBlock, ffn_layer='mlp', block_chunks=1, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1, qk_norm=False):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_reentrant = False
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        if ffn_layer == 'mlp':
            log.info('using MLP layer as FFN')
            ffn_layer = Mlp
        elif ffn_layer == 'swiglufused' or ffn_layer == 'swiglu':
            log.info('using SwiGLU layer as FFN')
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == 'identity':
            log.info('using Identity layer as FFN')

            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError
        blocks_list = [block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer, init_values=init_values, qk_norm=qk_norm) for i in range(depth)]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-06)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-06)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed.to(x)
        pos_embed = self.pos_embed.to(device=x.device, dtype=torch.float32)
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs['scale_factor'] = (sx, sy)
        else:
            kwargs['size'] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode='bicubic', antialias=self.interpolate_antialias, **kwargs)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x).unsqueeze(0), x)
        x = torch.cat((self.cls_token.to(x).expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.to(x).expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            else:
                x = blk(x)
        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append({'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks})
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            else:
                x = blk(x)
        x_norm = self.norm(x)
        return {'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks}

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = ([], len(self.blocks))
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = ([], 0, len(self.blocks[-1]))
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def get_intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence]=1, reshape: bool=False, return_class_token: bool=False, norm=True) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous() for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret['x_norm_clstoken'])

def init_weights_vit_timm(module: nn.Module, name: str=''):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention), num_register_tokens=num_register_tokens, **kwargs)
    return model

def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention), num_register_tokens=num_register_tokens, **kwargs)
    return model

def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention), num_register_tokens=num_register_tokens, **kwargs)
    return model

def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention), num_register_tokens=num_register_tokens, **kwargs)
    return model

# models/utils/grid.py

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float=100) -> torch.Tensor:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    assert embed_dim % 2 == 0
    device = pos_grid.device
    pos_flat = pos_grid.reshape(-1, grid_dim)
    compute_dtype = pos_grid.dtype if pos_grid.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    omega = torch.arange(embed_dim // 4, dtype=compute_dtype, device=device)
    omega /= embed_dim / 4.0
    omega = 1.0 / omega_0 ** omega
    pos_x = pos_flat[:, 0].reshape(-1)
    pos_y = pos_flat[:, 1].reshape(-1)
    out_x = torch.einsum('m,d->md', pos_x, omega)
    out_y = torch.einsum('m,d->md', pos_y, omega)
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.to(pos_grid.dtype).view(H, W, embed_dim)

def create_uv_grid(width: int, height: int, aspect_ratio: float=None, dtype: torch.dtype=None, device: torch.device=None) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing='xy')
    uv_grid = torch.stack((uu, vv), dim=-1)
    return uv_grid

# models/utils/rotation.py

def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r), two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r), two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rotmat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    quat_by_rijk = torch.stack([torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1), torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1), torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1), torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1)], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    out = standardize_quaternion(out)
    return out

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)

# models/utils/camera_utils.py

def camera_params_to_vector(ext, intr, image_hw=None):
    R = ext[..., :3, :3]
    t = ext[..., :3, 3]
    q = rotmat_to_quat(R)
    h, w = image_hw
    fov_v = 2.0 * torch.atan(h * 0.5 / intr[..., 1, 1])
    fov_u = 2.0 * torch.atan(w * 0.5 / intr[..., 0, 0])
    vec = torch.stack([t[..., 0], t[..., 1], t[..., 2], q[..., 0], q[..., 1], q[..., 2], q[..., 3], fov_v, fov_u], dim=-1).float()
    return vec

def extrinsics_to_vector(ext):
    R = ext[..., :3, :3]
    t = ext[..., :3, 3]
    q = rotmat_to_quat(R)
    vec = torch.stack([t[..., 0], t[..., 1], t[..., 2], q[..., 0], q[..., 1], q[..., 2], q[..., 3]], dim=-1).float()
    return vec

def vector_to_extrinsics(cam_vec):
    q = cam_vec[..., 3:7]
    t = cam_vec[..., :3]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    return ext

def vector_to_camera_matrices(cam_vec, image_hw=None, build_intr=True):
    intr = None
    t = cam_vec[..., 0:3]
    q = cam_vec[..., 3:7]
    fov_v = cam_vec[..., 7]
    fov_u = cam_vec[..., 8]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    if build_intr:
        h, w = image_hw
        fy = h * 0.5 / torch.tan(fov_v * 0.5)
        fx = w * 0.5 / torch.tan(fov_u * 0.5)
        shape = cam_vec.shape[:-1] + (3, 3)
        intr = torch.zeros(shape, device=cam_vec.device, dtype=cam_vec.dtype)
        intr[..., 0, 0] = fx
        intr[..., 1, 1] = fy
        intr[..., 0, 2] = w * 0.5
        intr[..., 1, 2] = h * 0.5
        intr[..., 2, 2] = 1.0
    return (ext, intr)

# models/utils/priors.py

def normalize_poses(extrinsics, padding=0.1, return_stats=False):
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    for i in range(B):
        if torch.isnan(extrinsics[i]).any() or torch.isinf(extrinsics[i]).any():
            print(f'Warning: dataset sample has NaN/Inf in extrinsics')
            extrinsics[i] = torch.nan_to_num(extrinsics[i], nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    normalized_extrinsics = extrinsics.clone()
    if return_stats:
        stats = {'scale_factors': torch.zeros(B, device=device), 'translation_vectors': torch.zeros(B, 3, device=device)}
    for b in range(B):
        positions = extrinsics[b, :, :3, 3]
        valid_mask = torch.isfinite(positions).all(dim=1)
        if valid_mask.sum() == 0:
            print(f'Warning: Batch {b} has no valid camera positions')
            normalized_extrinsics[b, :, :3, 3] = 0.5
            if return_stats:
                stats['scale_factors'][b] = 1.0
                stats['translation_vectors'][b] = 0.0
            continue
        valid_positions = positions[valid_mask]
        if valid_positions.shape[0] > 10:
            min_pos = torch.quantile(valid_positions, 0.05, dim=0)
            max_pos = torch.quantile(valid_positions, 0.95, dim=0)
        else:
            min_pos = torch.min(valid_positions, dim=0)[0]
            max_pos = torch.max(valid_positions, dim=0)[0]
        pos_range = max_pos - min_pos
        eps = torch.maximum(torch.tensor(1e-06, device=device), torch.abs(max_pos) * 1e-06)
        pos_range = torch.maximum(pos_range, eps)
        scale_factor = torch.max(pos_range)
        scale_factor = torch.clamp(scale_factor, min=1e-06, max=1000000.0)
        center = (min_pos + max_pos) / 2.0
        actual_scale = scale_factor / (1 - 2 * padding)
        normalized_positions = (positions - center) / actual_scale + 0.5
        normalized_positions = torch.clamp(normalized_positions, 0.0, 1.0)
        invalid_mask = ~torch.isfinite(positions).all(dim=1)
        if invalid_mask.any():
            normalized_positions[invalid_mask] = 0.5
        normalized_extrinsics[b, :, :3, 3] = normalized_positions
        if return_stats:
            stats['scale_factors'][b] = actual_scale
            stats['translation_vectors'][b] = center
    assert torch.isfinite(normalized_extrinsics).all(), 'Output contains non-finite values'
    if return_stats:
        return (normalized_extrinsics, stats)
    return normalized_extrinsics

def normalize_depth(depth, eps=1e-06, min_percentile=1, max_percentile=99):
    B, S, H, W = depth.shape
    depth = depth.flatten(0, 1)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=1000000.0, neginf=0.0)
    normalized_list = []
    for i in range(depth.shape[0]):
        depth_img = depth[i]
        depth_flat = depth_img.flatten()
        non_zero_mask = depth_flat > 0
        if non_zero_mask.sum() > 0:
            values_to_use = depth_flat[non_zero_mask]
        else:
            values_to_use = depth_flat
        if values_to_use.numel() > 100:
            depth_min = torch.quantile(values_to_use, min_percentile / 100.0)
            depth_max = torch.quantile(values_to_use, max_percentile / 100.0)
        else:
            depth_min = values_to_use.min()
            depth_max = values_to_use.max()
        if depth_max == depth_min:
            depth_max = depth_min + 1.0
        scale = torch.abs(depth_max - depth_min)
        eps_val = max(eps, scale.item() * eps)
        depth_norm_img = (depth_img - depth_min) / (depth_max - depth_min + eps_val)
        depth_norm_img = torch.clamp(depth_norm_img, 0.0, 1.0)
        normalized_list.append(depth_norm_img)
    depth_norm = torch.stack(normalized_list)
    return depth_norm.reshape(B, S, H, W)

# models/heads/dense_head.py

class _BaseDPTHead(nn.Module):

    def __init__(self, dim_in: int, patch_size: int=14, features: int=256, out_channels: List[int]=[256, 512, 1024, 1024], pos_embed: bool=True, down_ratio: int=1, gradient_checkpoint: bool=False, _cast_pos_embed_dtype: bool=True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.gradient_checkpoint = gradient_checkpoint
        self._cast_pos_embed_dtype = _cast_pos_embed_dtype
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0), nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0), nn.Identity(), nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1)])
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        head_features_1 = features
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float=0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        if self._cast_pos_embed_dtype:
            pos_embed = pos_embed.to(x.dtype)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)
        return out

    def _extract_fused_features(self, token_list: List[torch.Tensor], B: int, S: int, H: int, W: int, patch_start_idx: int, frame_start: int=None, frame_end: int=None) -> torch.Tensor:
        ph = H // self.patch_size
        pw = W // self.patch_size
        feats = []
        for proj, resize, tokens in zip(self.projects, self.resize_layers, token_list):
            patch_tokens = tokens[:, :, patch_start_idx:]
            if frame_start is not None and frame_end is not None:
                patch_tokens = patch_tokens[:, frame_start:frame_end]
            patch_tokens = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)
            feat = patch_tokens.permute(0, 2, 1).reshape(B * S, patch_tokens.shape[-1], ph, pw)
            feat = proj(feat)
            if self.pos_embed:
                feat = self._apply_pos_embed(feat, W, H)
            feat = resize(feat)
            feats.append(feat)
        fused = checkpoint(self.scratch_forward, feats, use_reentrant=False) if self.gradient_checkpoint else self.scratch_forward(feats)
        _interpolate_fn = lambda t: custom_interpolate(t, size=(int(ph * self.patch_size / self.down_ratio), int(pw * self.patch_size / self.down_ratio)), mode='bilinear', align_corners=True)
        fused = checkpoint(_interpolate_fn, fused, use_reentrant=False) if self.gradient_checkpoint else _interpolate_fn(fused)
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)
        return fused

class DPTHead(_BaseDPTHead):

    def __init__(self, dim_in: int, patch_size: int=14, output_dim: int=4, activation: str='inv_log+expp1', features: int=256, out_channels: List[int]=[256, 512, 1024, 1024], pos_embed: bool=True, down_ratio: int=1, is_gsdpt: bool=False, enable_depth_mask: bool=False, gradient_checkpoint: bool=False) -> None:
        super().__init__(dim_in=dim_in, patch_size=patch_size, features=features, out_channels=out_channels, pos_embed=pos_embed, down_ratio=down_ratio, gradient_checkpoint=gradient_checkpoint)
        self.activation = activation
        self.is_gsdpt = is_gsdpt
        self.enable_depth_mask = enable_depth_mask
        self.frames_chunk_size = 2
        head_features_2 = 32
        conv2_in_channels = features // 2
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0))
        if self.is_gsdpt:
            self.input_merger = nn.Sequential(nn.Conv2d(3, conv2_in_channels, 7, 1, 3), nn.ReLU())

    def to(self, *args, **kwargs):
        self.norm = self.norm.to(*args, **kwargs)
        self.projects = self.projects.to(*args, **kwargs)
        self.resize_layers = self.resize_layers.to(*args, **kwargs)
        if self.is_gsdpt:
            self.input_merger = self.input_merger.to(*args, **kwargs)
        for key in ('layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn', 'refinenet1', 'refinenet2', 'refinenet3', 'refinenet4', 'output_conv1'):
            if not hasattr(self.scratch, key):
                continue
            setattr(self.scratch, key, getattr(self.scratch, key).to(*args, **kwargs))
        args, kwargs = MlpFP32.map_to_args_to_float(args, kwargs)
        self.scratch.output_conv2 = self.scratch.output_conv2.to(*args, **kwargs)
        return self

    def forward(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _, H, W = images.shape
        if frames_chunk_size is None:
            frames_chunk_size = getattr(self, 'frames_chunk_size', 2)
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx)
        assert frames_chunk_size > 0
        preds_chunks = []
        conf_chunks = []
        gs_chunks = []
        depth_mask_chunks = []
        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)
            if self.is_gsdpt:
                if self.enable_depth_mask:
                    gs, preds, conf, depth_mask = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end)
                    gs_chunks.append(gs)
                    preds_chunks.append(preds)
                    conf_chunks.append(conf)
                    depth_mask_chunks.append(depth_mask)
                else:
                    gs, preds, conf = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end)
                    gs_chunks.append(gs)
                    preds_chunks.append(preds)
                    conf_chunks.append(conf)
            elif self.enable_depth_mask:
                preds, conf, depth_mask = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end)
                preds_chunks.append(preds)
                conf_chunks.append(conf)
                depth_mask_chunks.append(depth_mask)
            else:
                preds, conf = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end)
                preds_chunks.append(preds)
                conf_chunks.append(conf)
        if self.is_gsdpt:
            if self.enable_depth_mask:
                return (torch.cat(gs_chunks, dim=1), torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1), torch.cat(depth_mask_chunks, dim=1))
            return (torch.cat(gs_chunks, dim=1), torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1))
        elif self.enable_depth_mask:
            return (torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1), torch.cat(depth_mask_chunks, dim=1))
        else:
            return (torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1))

    def _forward_impl(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frame_start: int=None, frame_end: int=None) -> torch.Tensor:
        if frame_start is not None and frame_end is not None:
            images = images[:, frame_start:frame_end].contiguous()
        B, S, _, H, W = images.shape
        fused = self._extract_fused_features(token_list, B, S, H, W, patch_start_idx, frame_start, frame_end)
        if self.is_gsdpt:
            out = self.scratch.output_conv2(fused.float().contiguous())
            if self.enable_depth_mask:
                preds, conf, depth_mask = self.activate_head(out, activation=self.activation)
            else:
                preds, conf = self.activate_head(out, activation=self.activation)
            preds = preds.reshape(B, S, *preds.shape[1:])
            conf = conf.reshape(B, S, *conf.shape[1:])
            img_flat = images.reshape(B * S, -1, H, W)
            img_feat = self.input_merger(img_flat)
            fused = fused + img_feat
            fused = fused.reshape(B, S, *fused.shape[1:]).contiguous()
            if self.enable_depth_mask:
                depth_mask = depth_mask.reshape(B, S, *depth_mask.shape[1:])
                return (fused, preds, conf, depth_mask)
            return (fused, preds, conf)
        else:
            out = self.scratch.output_conv2(fused.float().contiguous())
            if self.enable_depth_mask:
                preds, conf, depth_mask = self.activate_head(out, activation=self.activation)
                preds = preds.reshape(B, S, *preds.shape[1:])
                conf = conf.reshape(B, S, *conf.shape[1:])
                depth_mask = depth_mask.reshape(B, S, *depth_mask.shape[1:])
                return (preds, conf, depth_mask)
            else:
                preds, conf = self.activate_head(out, activation=self.activation)
                preds = preds.reshape(B, S, *preds.shape[1:])
                conf = conf.reshape(B, S, *conf.shape[1:])
                return (preds, conf)

    def activate_head(self, out_head: torch.Tensor, activation: str='inv_log+expp1') -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_depth_mask:
            act_attr, act_conf, act_depth_mask = activation.split('+') if '+' in activation else (activation, 'expp1', 'linear')
            feat = out_head.permute(0, 2, 3, 1)
            attr, conf, depth_mask = (feat[..., :-2], feat[..., -2], feat[..., -1])
        else:
            act_attr, act_conf = activation.split('+') if '+' in activation else (activation, 'expp1')
            feat = out_head.permute(0, 2, 3, 1)
            attr, conf = (feat[..., :-1], feat[..., -1])
        attr_activations = {'norm_exp': lambda x: x / x.norm(dim=-1, keepdim=True).clamp(min=1e-08) * torch.expm1(x.norm(dim=-1, keepdim=True)), 'norm': lambda x: x / x.norm(dim=-1, keepdim=True), 'exp': torch.exp, 'relu': F.relu, 'inv_log': self._apply_inverse_log_transform, 'xy_inv_log': lambda x: torch.cat([x[..., :2] * self._apply_inverse_log_transform(x[..., 2:]), self._apply_inverse_log_transform(x[..., 2:])], dim=-1), 'sigmoid': torch.sigmoid, 'linear': lambda x: x}
        if act_attr not in attr_activations:
            raise ValueError(f'Unknown attribute activation: {act_attr}')
        attr_out = attr_activations[act_attr](attr)
        conf_activations = {'expp1': lambda c: 1 + c.exp(), 'expp0': torch.exp, 'sigmoid': torch.sigmoid}
        if act_conf not in conf_activations:
            raise ValueError(f'Unknown confidence activation: {act_conf}')
        conf_out = conf_activations[act_conf](conf)
        if self.enable_depth_mask:
            depth_mask_activations = {'sigmoid': torch.sigmoid, 'linear': lambda x: x}
            if act_depth_mask not in depth_mask_activations:
                raise ValueError(f'Unknown depth mask activation: {act_depth_mask}')
            depth_mask_out = depth_mask_activations[act_depth_mask](depth_mask)
            return (attr_out, conf_out, depth_mask_out)
        else:
            return (attr_out, conf_out)

    def _apply_inverse_log_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sign(input_tensor) * torch.expm1(torch.abs(input_tensor))

def _make_fusion_block(features: int, size: int=None, has_residual: bool=True, groups: int=1) -> nn.Module:
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), deconv=False, bn=False, expand=False, align_corners=True, size=size, has_residual=has_residual, groups=groups)

def _make_scratch(in_shape: List[int], out_shape: int, groups: int=1, expand: bool=False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

class ResidualConvUnit(nn.Module):

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {'scale_factor': 2}
        elif size is None:
            modifier = {'size': self.size}
        else:
            modifier = {'size': size}
        output = custom_interpolate(output, **modifier, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def custom_interpolate(x: torch.Tensor, size: Tuple[int, int]=None, scale_factor: float=None, mode: str='bilinear', align_corners: bool=True) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=input_elements // INT_MAX + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

# models/heads/camera_head.py

class CameraHead(nn.Module):

    def __init__(self, dim_in: int=2048, trunk_depth: int=4, num_heads: int=16, mlp_ratio: int=4, init_values: float=0.01, trans_act: str='linear', quat_act: str='linear', fl_act: str='relu', block_fn: nn.Module=Block):
        super().__init__()
        self.out_dim = 9
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.depth = trunk_depth
        self.refine_net = nn.Sequential(*[block_fn(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values) for _ in range(trunk_depth)])
        self.token_norm = nn.LayerNorm(dim_in)
        self.out_norm = nn.LayerNorm(dim_in)
        self.init_token = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        self.param_embed = nn.Linear(self.out_dim, dim_in)
        self.adapt_norm_gen = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adapt_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-06)
        self.param_predictor = MlpFP32(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.out_dim, drop=0)

    def to(self, *args, **kwargs):
        self.refine_net = self.refine_net.to(*args, **kwargs)
        self.token_norm = self.token_norm.to(*args, **kwargs)
        self.out_norm = self.out_norm.to(*args, **kwargs)
        self.adapt_norm_gen = self.adapt_norm_gen.to(*args, **kwargs)
        self.adapt_norm = self.adapt_norm.to(*args, **kwargs)
        self.param_predictor = self.param_predictor.to(*args, **kwargs)
        args, kwargs = MlpFP32.map_to_args_to_float(args, kwargs)
        self.init_token = nn.Parameter(self.init_token.to(*args, **kwargs))
        self.param_embed = self.param_embed.to(*args, **kwargs)
        return self

    def forward(self, feat_seq: list, steps: int=4) -> list:
        latest_feat = feat_seq[-1]
        cam_tokens = latest_feat[:, :, 0]
        cam_tokens = self.token_norm(cam_tokens)
        b, seq_len, feat_dim = cam_tokens.shape
        curr_pred = None
        pred_seq = []
        for step in range(steps):
            if curr_pred is None:
                net_input = self.param_embed(self.init_token.to(cam_tokens).expand(b, seq_len, -1))
            else:
                curr_pred = curr_pred.detach()
                net_input = self.param_embed(curr_pred)
            net_input = net_input.to(cam_tokens.dtype)
            norm_shift, norm_scale, norm_gate = self.adapt_norm_gen(net_input).chunk(3, dim=-1)
            mod_cam_feat = norm_gate * self.apply_adaptive_modulation(self.adapt_norm(cam_tokens), norm_shift, norm_scale)
            mod_cam_feat = mod_cam_feat + cam_tokens
            proc_feat = self.refine_net(mod_cam_feat)
            param_delta = self.param_predictor(self.out_norm(proc_feat))
            if curr_pred is None:
                curr_pred = param_delta
            else:
                curr_pred = curr_pred + param_delta
            activated_params = self.apply_camera_parameter_activation(curr_pred)
            pred_seq.append(activated_params)
        return pred_seq

    def apply_camera_parameter_activation(self, params: torch.Tensor) -> torch.Tensor:
        trans_vec = params[..., :3]
        quat_vec = params[..., 3:7]
        fl_vec = params[..., 7:]
        trans_vec = self.apply_parameter_activation(trans_vec, self.trans_act)
        quat_vec = self.apply_parameter_activation(quat_vec, self.quat_act)
        fl_vec = self.apply_parameter_activation(fl_vec, self.fl_act)
        activated_params = torch.cat([trans_vec, quat_vec, fl_vec], dim=-1)
        return activated_params

    def apply_parameter_activation(self, tensor: torch.Tensor, act_type: str) -> torch.Tensor:
        if act_type == 'linear':
            return tensor
        elif act_type == 'inv_log':
            return self.apply_inverse_logarithm_transform(tensor)
        elif act_type == 'exp':
            return torch.exp(tensor)
        elif act_type == 'relu':
            return F.relu(tensor)
        else:
            raise ValueError(f'Unknown activation_type: {act_type}')

    def apply_inverse_logarithm_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.expm1(torch.abs(x))

    def apply_adaptive_modulation(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

# models/models/visual_transformer.py
logger = logging.getLogger(__name__)
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class VisualGeometryTransformer(nn.Module):

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_register_tokens=4, block_fn=Block, qkv_bias=True, proj_bias=True, ffn_bias=True, patch_embed='dinov2_vitl14_reg', qk_norm=True, rope_base=100.0, normalized_rope=False, rope_normalize_coords='separate', rope_shift_coords=None, rope_jitter_coords=None, rope_rescale_coords=None, init_values=0.01, enable_cond=False, sampling_strategy='uniform', fixed_patch_embed=False, condition_strategy=['token', 'pow3r', 'token'], intermediate_idxs: List[int]=[4, 11, 17, 23]):
        super().__init__()
        self.enable_cond = enable_cond
        self.sampling_strategy = sampling_strategy
        self.cond_methods = condition_strategy
        self.intermediate_idxs = intermediate_idxs
        self.depth = depth
        self.patch_size = patch_size
        self.patch_embed = self._init_patch_embedding_module(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim, is_fixed=fixed_patch_embed)
        if self.enable_cond:
            self._init_cond_embeddings(embed_dim, img_size, patch_size, num_register_tokens)
        self._init_rotary_position_embedding(rope_base, normalized_rope, embed_dim // num_heads, rope_normalize_coords, rope_shift_coords, rope_jitter_coords, rope_rescale_coords)
        self._init_transformer_blocks(block_fn, embed_dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, init_values, qk_norm)
        self._init_learnable_tokens(embed_dim, num_register_tokens)
        if self.enable_cond:
            self.patch_start_idx = 1 + num_register_tokens + 1 + 1
        else:
            self.patch_start_idx = 1 + num_register_tokens
        for name, value in (('_resnet_mean', _RESNET_MEAN), ('_resnet_std', _RESNET_STD)):
            self.register_buffer(name, torch.tensor(value, device="cpu", dtype=torch.float32).reshape(1, 1, 3, 1, 1), persistent=False)
        self.use_reentrant = False

    def _init_patch_embedding_module(self, patch_embed_type, img_size, patch_size, num_reg_tokens, interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0, init_values=1.0, embed_dim=1024, is_fixed=False, in_chans=3):
        if 'conv' in patch_embed_type:
            if 'mlp' in patch_embed_type:
                patch_embed_module = PatchEmbed_Mlp(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                patch_embed_module = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            vit_models = {'dinov2_vitl14_reg': vit_large, 'dinov2_vitb14_reg': vit_base, 'dinov2_vits14_reg': vit_small, 'dinov2_vitg2_reg': vit_giant2}
            patch_embed_module = vit_models[patch_embed_type](img_size=img_size, patch_size=patch_size, num_register_tokens=num_reg_tokens, interpolate_antialias=interpolate_antialias, interpolate_offset=interpolate_offset, block_chunks=block_chunks, init_values=init_values)
            if hasattr(patch_embed_module, 'mask_token'):
                patch_embed_module.mask_token.requires_grad_(False)
        if is_fixed:
            for param in patch_embed_module.parameters():
                param.requires_grad_(False)
        return patch_embed_module

    def _init_cond_embeddings(self, embed_dim, img_size, patch_size, num_reg_tokens):
        assert self.cond_methods is not None
        assert self.cond_methods[0] == 'token'
        if self.cond_methods[0] == 'token':
            self.pose_embed = nn.Sequential(nn.Linear(7, embed_dim, bias=True), nn.SiLU(), nn.Linear(embed_dim, embed_dim, bias=True))
        else:
            raise NotImplementedError
        if self.cond_methods[1] == 'pow3r':
            self.depth_embed = self._init_patch_embedding_module('conv+mlp', img_size, patch_size, num_reg_tokens, embed_dim=embed_dim, in_chans=1)
        else:
            raise NotImplementedError
        if self.cond_methods[2] == 'token':
            self.ray_embed = nn.Sequential(nn.Linear(4, embed_dim, bias=True), nn.SiLU(), nn.Linear(embed_dim, embed_dim, bias=True))
        else:
            raise NotImplementedError

    def _init_rotary_position_embedding(self, rope_base, normalized_rope, head_dim, rope_normalize_coords, rope_shift_coords, rope_jitter_coords, rope_rescale_coords):
        print('[INFO] Using normalized RoPE!')
        if head_dim % 4 != 0:
            raise ValueError('RoPE requires head_dim divisible by 4 (embed_dim must be divisible by 4*num_heads)')
        self.rope = NormalizedRotaryPositionEmbedding2D(head_dim=head_dim, base=rope_base, normalize_coords=rope_normalize_coords, shift_coords=rope_shift_coords, jitter_coords=rope_jitter_coords, rescale_coords=rope_rescale_coords) if rope_base > 0 else None
        self.pos_getter = PositionGetter() if self.rope is not None else None

    def _init_transformer_blocks(self, block_fn, embed_dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, init_values, qk_norm):
        self.frame_blocks = nn.ModuleList([block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, init_values=init_values, qk_norm=qk_norm, rope=self.rope) for _ in range(self.depth)])
        self.global_blocks = nn.ModuleList([block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, init_values=init_values, qk_norm=qk_norm, rope=self.rope) for _ in range(self.depth)])

    def _init_learnable_tokens(self, embed_dim, num_reg_tokens):
        self.cam_token = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, 2, num_reg_tokens, embed_dim))
        nn.init.normal_(self.cam_token, std=1e-06)
        nn.init.normal_(self.reg_token, std=1e-06)

    def forward(self, images: torch.Tensor, priors: List | None=None, cond_flags: List[int]=[0, 0, 0], ctx_frames: int=None, enable_bf16=False, sp_size: int=1, sp_group: torch._C._distributed_c10d.ProcessGroup=None) -> Tuple[List[torch.Tensor], int]:
        depth_maps, ray_dirs, poses = priors if priors is not None else (None, None, None)
        if ctx_frames is not None:
            for var_name in ['images', 'depth_maps', 'ray_dirs', 'poses']:
                var = locals()[var_name]
                if var is not None:
                    locals()[var_name] = var[:, :ctx_frames].clone()
        b, seq_len, ch, h, w = images.shape
        if ch != 3:
            raise ValueError(f'Expected 3 input channels, got {ch}')
        with torch.amp.autocast('cuda', enabled=not enable_bf16 and images.is_cuda, dtype=torch.bfloat16):
            images = (images - self._resnet_mean.to(images)) / self._resnet_std.to(images)
            images = images.reshape(b * seq_len, ch, h, w)
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens['x_norm_patchtokens']
        _, patch_count, embed_dim = patch_tokens.shape
        cam_tokens = expand_and_flatten_special_tokens(self.cam_token.to(patch_tokens), b, seq_len)
        reg_tokens = expand_and_flatten_special_tokens(self.reg_token.to(patch_tokens), b, seq_len)
        if self.enable_cond:
            pose_tokens, depth_tokens, ray_tokens = self._process_conditioning(depth_maps, ray_dirs, poses, b, seq_len, patch_count, embed_dim, images, cond_flags)
            patch_tokens = patch_tokens + depth_tokens
            all_tokens = torch.cat([cam_tokens, reg_tokens, pose_tokens, ray_tokens, patch_tokens], dim=1)
        else:
            all_tokens = torch.cat([cam_tokens, reg_tokens, patch_tokens], dim=1)
        _, patch_count, embed_dim = all_tokens.shape
        pos_emb = None
        if self.rope is not None:
            pos_emb = self.pos_getter(b * seq_len, h // self.patch_size, w // self.patch_size, device=images.device)
            if self.patch_start_idx > 0:
                pos_emb = pos_emb + 1
                special_pos = torch.zeros(b * seq_len, self.patch_start_idx, 2, device=images.device, dtype=pos_emb.dtype)
                pos_emb = torch.cat([special_pos, pos_emb], dim=1)
        _, patch_count, embed_dim = all_tokens.shape
        token_shape = (b, seq_len, patch_count, embed_dim)
        with torch.amp.autocast('cuda', enabled=not enable_bf16 and images.is_cuda, dtype=torch.bfloat16):
            outputs = []
            global_tokens = None
            for idx in range(self.depth):
                local_tokens = self._process_attention_blocks(tokens=all_tokens if global_tokens is None else global_tokens, b=b, seq_len=seq_len, patch_count=patch_count, embed_dim=embed_dim, block_idx=idx, blocks=self.frame_blocks, block_type='frame', pos=pos_emb)
                global_tokens = self._process_attention_blocks(tokens=local_tokens, b=b, seq_len=seq_len, patch_count=patch_count, embed_dim=embed_dim, block_idx=idx, blocks=self.global_blocks, block_type='global', pos=pos_emb)
                if idx in self.intermediate_idxs:
                    combined_out = torch.cat([local_tokens, global_tokens], dim=-1)
                    outputs.append(combined_out)
            if idx in self.intermediate_idxs:
                combined_out = torch.cat([local_tokens, global_tokens], dim=-1)
                outputs.append(combined_out)
        return (outputs, self.patch_start_idx)

    def _process_conditioning(self, depth_maps, ray_dirs, poses, b, seq_len, patch_count, embed_dim, images, cond_flags):
        h, w = images.shape[-2:]
        use_poses = cond_flags[0] == 1 and poses is not None
        if use_poses:
            poses = poses.reshape(b * seq_len, -1)
            pose_tokens = self.pose_embed(poses).unsqueeze(1)
        else:
            pose_tokens = torch.zeros((b * seq_len, 1, embed_dim), device=images.device, dtype=images.dtype)
        use_depth = cond_flags[1] == 1 and depth_maps is not None
        if use_depth:
            depth_maps = depth_maps.reshape(b * seq_len, 1, h, w)
            depth_tokens = self.depth_embed(depth_maps).reshape(b * seq_len, patch_count, embed_dim)
        else:
            depth_tokens = torch.zeros((b * seq_len, patch_count, embed_dim), device=images.device, dtype=images.dtype)
        use_rays = cond_flags[2] == 1 and ray_dirs is not None
        if use_rays:
            ray_dirs = ray_dirs.reshape(b * seq_len, -1)
            ray_tokens = self.ray_embed(ray_dirs).unsqueeze(1)
        else:
            ray_tokens = torch.zeros((b * seq_len, 1, embed_dim), device=images.device, dtype=images.dtype)
        return (pose_tokens, depth_tokens, ray_tokens)

    def _process_attention_blocks(self, tokens, b, seq_len, patch_count, embed_dim, block_idx, blocks, block_type, pos=None):
        token_shape = (b, seq_len, patch_count, embed_dim)
        if block_type == 'frame':
            target_shape = (b * seq_len, patch_count, embed_dim)
            pos_target_shape = (b * seq_len, patch_count, 2) if pos is not None else None
        else:
            target_shape = (b, seq_len * patch_count, embed_dim)
            pos_target_shape = (b, seq_len * patch_count, 2) if pos is not None else None
        if tokens.shape != target_shape:
            tokens = tokens.reshape(*target_shape)
        if pos is not None and pos.shape != pos_target_shape:
            pos = pos.reshape(*pos_target_shape)
        if self.training:
            tokens = checkpoint(blocks[block_idx], tokens, pos=pos, use_reentrant=self.use_reentrant)
        else:
            tokens = blocks[block_idx](tokens, pos=pos)
        return tokens.reshape(*token_shape)

def expand_and_flatten_special_tokens(token_tensor, b, seq_len):
    first_frame_tokens = token_tensor[:, 0:1, ...].expand(b, 1, *token_tensor.shape[2:])
    remaining_frame_tokens = token_tensor[:, 1:, ...].expand(b, seq_len - 1, *token_tensor.shape[2:])
    combined_tokens = torch.cat([first_frame_tokens, remaining_frame_tokens], dim=1)
    return combined_tokens.reshape(b * seq_len, *combined_tokens.shape[2:])


class _GaussianParameters(nn.Module):
    """The inference-only parameter head from V2's GaussianSplatRenderer."""

    def __init__(self, feature_dim=256):
        super().__init__()
        self.gs_head = nn.Sequential(nn.Conv2d(feature_dim // 2, feature_dim, 3, padding=1, bias=False),
                                     nn.ReLU(True), nn.Conv2d(feature_dim, 12, 1))
        self.gs_param_chunk_size = 1

    def forward(self, features, images, predictions):
        batch, count, channels, height, width = features.shape
        features = features.reshape(batch * count, channels, height, width)
        features = features.to(next(self.gs_head.parameters()).dtype)
        size = self.gs_param_chunk_size
        raw = torch.cat([self.gs_head(part) for part in features.split(size)], 0)
        params = raw.permute(0, 2, 3, 1).reshape(batch, count * height * width, 12).float()
        quats, scales, opacities, residual, weights = params.split((4, 3, 1, 3, 1), dim=-1)
        rgb = images.permute(0, 1, 3, 4, 2).reshape(batch, count * height * width, 3).float()
        depth = predictions["gs_depth"].reshape(batch * count, height, width).float()
        poses = predictions["camera_poses"].reshape(batch * count, 4, 4)
        intrinsics = predictions["camera_intrs"].reshape(batch * count, 3, 3)
        points, _, _ = depth_to_world_coords_points(depth, poses, intrinsics)
        return {
            "means": points.reshape(batch, count * height * width, 3),
            "gs_feats": params,
            "quats": quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            "scales": scales.exp().clamp_max(0.3),
            "opacities": opacities.squeeze(-1).sigmoid(),
            "sh": ((rgb - 0.5) / 0.28209479177387814 + residual).unsqueeze(-2),
            "residual_sh": residual.unsqueeze(-2),
            "weights": weights.squeeze(-1).sigmoid(),
        }


class _WorldMirror(nn.Module):
    """V2 prediction heads with runtime-selectable inference settings."""

    def __init__(self, config):
        super().__init__()
        if (config.get("model_size") != "large" or config.get("patch_size") != 14
                or config.get("normalized_rope") is not True):
            raise ValueError("Expected the HY-WorldMirror-2.0 large checkpoint with normalized RoPE.")
        self.visual_geometry_transformer = VisualGeometryTransformer(
            img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
            mlp_ratio=4.0, num_register_tokens=4, block_fn=Block,
            normalized_rope=True, rope_normalize_coords="separate",
            enable_cond=True, sampling_strategy="uniform", patch_embed="dinov2_vitl14_reg",
            fixed_patch_embed=True, condition_strategy=["token", "pow3r", "token"],
            intermediate_idxs=[4, 11, 17, 23],
        )
        self.cam_head = CameraHead(dim_in=2048, block_fn=Block)
        self.depth_head = DPTHead(dim_in=2048, output_dim=3, patch_size=14,
                                  activation="exp+expp1+linear", enable_depth_mask=True)
        self.pts_head = DPTHead(dim_in=2048, output_dim=4, patch_size=14, activation="inv_log+expp1")
        self.norm_head = DPTHead(dim_in=2048, output_dim=4, patch_size=14, activation="norm+expp1")
        self.gs_head = DPTHead(dim_in=2048, output_dim=3, patch_size=14, features=256,
                              is_gsdpt=True, activation="exp+expp1+linear", enable_depth_mask=True)
        self.gs_renderer = _GaussianParameters()
        self.enable_pts = True
        self.enable_norm = True
        self.enable_gs = True
        for head in (self.depth_head, self.pts_head, self.norm_head, self.gs_head):
            head.frames_chunk_size = 1
        self.enable_bf16 = False
        for layer in self.modules():
            if isinstance(layer, Mlp):
                layer.inference_chunk_size = 8192

    def get_dtype(self):
        return self.visual_geometry_transformer.cam_token.dtype

    def forward(self, views, cond_flags):
        images = views["img"]
        height, width = images.shape[-2:]
        poses = views.get("camera_poses")
        poses = extrinsics_to_vector(normalize_poses(poses[..., :3, :]))[..., :7] if poses is not None else None
        k = views.get("camera_intrs")
        rays = torch.stack((k[..., 0, 0] / width, k[..., 1, 1] / height,
                            k[..., 0, 2] / width, k[..., 1, 2] / height), -1) if k is not None else None
        depths = normalize_depth(views["depthmap"]) if "depthmap" in views else None
        if self.enable_bf16:
            images = images.bfloat16()
            poses = poses.bfloat16() if poses is not None else None
            rays = rays.bfloat16() if rays is not None else None
            depths = depths.bfloat16() if depths is not None else None
        tokens, patch_start = self.visual_geometry_transformer(
            images, (depths, rays, poses), cond_flags=cond_flags, enable_bf16=self.enable_bf16)
        if not self.enable_bf16:
            tokens = [token.float() for token in tokens]
        camera = self.cam_head(tokens)[-1]
        ext, intrinsics = vector_to_camera_matrices(camera, image_hw=(height, width))
        w2c = torch.eye(4, device=ext.device, dtype=ext.dtype).expand(*ext.shape[:-2], 4, 4).clone()
        w2c[..., :3, :] = ext
        depth, confidence, mask = self.depth_head(tokens, images=images, patch_start_idx=patch_start)
        result = {"depth": depth, "depth_conf": confidence, "depth_mask": mask.sigmoid(),
                  "depth_mask_logits": mask, "camera_params": camera,
                  "camera_poses": torch.linalg.inv(w2c.float()), "camera_intrs": intrinsics}
        if self.enable_pts:
            result["pts3d"], result["pts3d_conf"] = self.pts_head(tokens, images=images, patch_start_idx=patch_start)
        if self.enable_norm:
            result["normals"], result["normals_conf"] = self.norm_head(tokens, images=images, patch_start_idx=patch_start)
        if self.enable_gs:
            features, result["gs_depth"], result["gs_depth_conf"], mask = self.gs_head(
                tokens, images=images, patch_start_idx=patch_start)
            result["gs_depth_mask_logits"] = mask
            result["gs_depth_mask"] = mask.sigmoid()
            result["splats"] = self.gs_renderer(features, images, result)
        return result


def depth_to_camera_coords(depthmap, camera_intrinsics):
    """
    Convert depth map to 3D camera coordinates.

    Args:
        depthmap (BxHxW tensor): Batch of depth maps
        camera_intrinsics (Bx3x3 tensor): Camera intrinsics matrix for each camera

    Returns:
        X_cam (BxHxWx3 tensor): 3D points in camera coordinates
        valid_mask (BxHxW tensor): Mask indicating valid depth pixels
    """
    B, H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype

    # Ensure intrinsics are float
    camera_intrinsics = camera_intrinsics.float()

    # Extract focal lengths and principal points
    fx = camera_intrinsics[:, 0, 0]  # (B,)
    fy = camera_intrinsics[:, 1, 1]  # (B,)
    cx = camera_intrinsics[:, 0, 2]  # (B,)
    cy = camera_intrinsics[:, 1, 2]  # (B,)

    # Generate pixel grid
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij'
    )

    # Reshape for broadcasting: (1, H, W)
    u_grid = u_grid.unsqueeze(0)
    v_grid = v_grid.unsqueeze(0)

    # Compute 3D camera coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    z_cam = depthmap  # (B, H, W)
    x_cam = (u_grid - cx.view(B, 1, 1)) * z_cam / fx.view(B, 1, 1)
    y_cam = (v_grid - cy.view(B, 1, 1)) * z_cam / fy.view(B, 1, 1)

    # Stack to form (B, H, W, 3)
    X_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # Valid depth mask
    valid_mask = depthmap > 0.0

    return X_cam, valid_mask


def depth_to_world_coords_points(
    depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of depth maps to world coordinates.

    Args:
        depth_map (torch.Tensor): (B, H, W) Depth map
        extrinsic (torch.Tensor): (B, 4, 4) Camera extrinsic matrix (camera-to-world transformation)
        intrinsic (torch.Tensor): (B, 3, 3) Camera intrinsic matrix

    Returns:
        world_coords_points (torch.Tensor): (B, H, W, 3) World coordinates
        camera_points (torch.Tensor): (B, H, W, 3) Camera coordinates
        point_mask (torch.Tensor): (B, H, W) Valid depth mask
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask (B, H, W)
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates (B, H, W, 3)
    camera_points, _ = depth_to_camera_coords(depth_map, intrinsic)

    # Apply extrinsic matrix (camera -> world)
    R_cam_to_world = extrinsic[:, :3, :3]   # (B, 3, 3)
    t_cam_to_world = extrinsic[:, :3, 3]    # (B, 3)

    # Transform (B, H, W, 3) x (B, 3, 3)^T + (B, 3) -> (B, H, W, 3)
    world_coords_points = torch.einsum('bhwi,bji->bhwj', camera_points, R_cam_to_world) + t_cam_to_world[:, None, None, :]

    return world_coords_points, camera_points, point_mask


def _prepare_bf16(model):
    critical = [layer.fc2 for layer in model.modules() if isinstance(layer, MlpFP32)]
    critical.extend(getattr(model, name).scratch.output_conv2 for name in
                    ("depth_head", "pts_head", "norm_head", "gs_head") if hasattr(model, name))
    model.to(dtype=torch.bfloat16)
    for layer in critical:
        layer.to(dtype=torch.float32)

    def cast_leaf(layer, args):
        dtype = next((p.dtype for p in layer.parameters(recurse=False)), None)
        return tuple(value.to(dtype) if torch.is_tensor(value) and value.is_floating_point()
                     and dtype is not None else value for value in args)

    for layer in model.modules():
        if not any(layer.children()) and any(p.dtype == torch.bfloat16 for p in layer.parameters(recurse=False)):
            layer.register_forward_pre_hook(cast_leaf)
    model.enable_bf16 = True


def _quantize_mirror(model, precision, device):
    import comfy.model_management as management
    from comfy.ops import mixed_precision_ops
    from comfy.quant_ops import (QuantizedTensor, TensorCoreConvRotW4A4Layout,
                                 TensorWiseINT8Layout)

    layout, quant_format = {
        "convrot4": (TensorCoreConvRotW4A4Layout, "convrot_w4a4"),
        "convrot8": (TensorWiseINT8Layout, "int8_tensorwise"),
    }[precision]
    if not callable(getattr(layout, "quantize", None)):
        raise RuntimeError(f"WorldMirror {precision} requires ComfyUI's Comfy Kitchen quantization backend.")
    operations = mixed_precision_ops(compute_dtype=torch.bfloat16)
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, nn.Linear) and module.weight.dtype == torch.bfloat16]
    if device.type == "cuda" and layers:
        management.free_memory(max(module.weight.nbytes for _, module in layers) * 8, device)
    converted = 0
    for name, module in layers:
        management.throw_exception_if_processing_interrupted()
        if module.in_features < 64 or module.in_features % (64 if precision == "convrot4" else 4):
            continue
        group_size = 256
        while module.in_features % group_size:
            group_size //= 4
        kwargs = {"convrot_groupsize": group_size}
        if precision == "convrot8":
            kwargs.update(is_weight=True, per_channel=True, convrot=True)
        weight = module.weight.detach().to(device)
        qdata, params = layout.quantize(weight, **kwargs)
        quant_weight = QuantizedTensor(qdata, layout.__name__, params).to(module.weight.device)
        linear = operations.Linear(module.in_features, module.out_features,
                                   bias=module.bias is not None, device="meta")
        linear.weight = nn.Parameter(quant_weight, requires_grad=False)
        linear.bias = module.bias
        linear.quant_format = quant_format
        linear.layout_type = layout.__name__
        linear.train(module.training)
        for hook in module._forward_pre_hooks.values():
            linear.register_forward_pre_hook(hook)
        model.set_submodule(name, linear)
        converted += 1
        del weight, qdata, quant_weight
    if not converted:
        raise RuntimeError(f"WorldMirror {precision} did not quantize any compatible Linear layers.")
    print(f"[VNCCS WorldMirror V2] {precision}: quantized {converted} bf16 Linear layers with Comfy Kitchen", flush=True)


def _native_mirror_ops(model):
    from comfy.ops import manual_cast

    for name, layer in list(model.named_modules()):
        if type(layer) is nn.Linear:
            native = manual_cast.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None,
                                         device="meta", dtype=layer.weight.dtype)
        elif type(layer) is nn.Conv2d:
            native = manual_cast.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                         layer.stride, layer.padding, layer.dilation, layer.groups,
                                         layer.bias is not None, layer.padding_mode,
                                         device="meta", dtype=layer.weight.dtype)
        elif type(layer) is nn.ConvTranspose2d:
            native = manual_cast.ConvTranspose2d(
                layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding,
                layer.output_padding, layer.groups, layer.bias is not None, layer.dilation,
                layer.padding_mode, device="meta", dtype=layer.weight.dtype,
            )
        elif type(layer) is nn.LayerNorm and layer.elementwise_affine:
            native = manual_cast.LayerNorm(layer.normalized_shape, layer.eps, bias=layer.bias is not None,
                                           device="meta", dtype=layer.weight.dtype)
        else:
            continue
        native.weight, native.bias = layer.weight, layer.bias
        native.train(layer.training)
        for hook in layer._forward_pre_hooks.values():
            native.register_forward_pre_hook(hook)
        model.set_submodule(name, native)


def _load_mirror_state(model, state):
    # Hugging Face safetensors stores shared RoPE buffers only once.
    buffers = list(model.named_buffers(remove_duplicate=False))
    shared = {id(buffer): state[name] for name, buffer in buffers if name in state}
    for name, buffer in buffers:
        if name not in state and id(buffer) in shared:
            state[name] = shared[id(buffer)]
    model.load_state_dict(state, strict=True, assign=True)
