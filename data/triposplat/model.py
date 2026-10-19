"""Neural network components vendored from VAST-AI-Research/TripoSplat."""

from typing import Optional
import math
import re

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DINOv3 ViT-H/16+
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DinoV3PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=1280):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV3RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, base: float = 100.0):
        super().__init__()
        inv_freq = 1.0 / (base ** torch.arange(0, 1, 4.0 / dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, height: int, width: int, device: torch.device, dtype: torch.dtype):
        coords_h = torch.arange(0.5, height, dtype=torch.float32, device=device) / height
        coords_w = torch.arange(0.5, width, dtype=torch.float32, device=device) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = (2.0 * coords - 1.0).flatten(0, 1)
        angles = (2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]).flatten(1, 2).tile(2)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DinoV3Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: tuple = (True, False, True)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        q_bias, k_bias, v_bias = qkv_bias
        self.q_proj = nn.Linear(dim, dim, bias=q_bias)
        self.k_proj = nn.Linear(dim, dim, bias=k_bias)
        self.v_proj = nn.Linear(dim, dim, bias=v_bias)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                num_prefix_tokens: int = 0) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if num_prefix_tokens > 0:
            q_pre, q_pat = q.split((num_prefix_tokens, N - num_prefix_tokens), dim=-2)
            k_pre, k_pat = k.split((num_prefix_tokens, N - num_prefix_tokens), dim=-2)
            q = torch.cat((q_pre, q_pat * cos + _rotate_half(q_pat) * sin), dim=-2)
            k = torch.cat((k_pre, k_pat * cos + _rotate_half(k_pat) * sin), dim=-2)
        else:
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, N, C))


class DinoV3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DinoV3Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: tuple = (True, False, True), layerscale_init: float = 1.0,
                 mlp_bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = DinoV3Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.ls1 = nn.Parameter(torch.ones(dim) * layerscale_init)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = DinoV3MLP(dim, int(dim * mlp_ratio), bias=mlp_bias)
        self.ls2 = nn.Parameter(torch.ones(dim) * layerscale_init)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                num_prefix_tokens: int = 0) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x), cos, sin, num_prefix_tokens=num_prefix_tokens)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DinoV3ViT(nn.Module):
    def __init__(self, hidden_size: int = 1280, num_heads: int = 20, num_layers: int = 32,
                 patch_size: int = 16, num_register_tokens: int = 4,
                 intermediate_size: int = 5120, layerscale_init: float = 1.0,
                 query_bias: bool = True, key_bias: bool = False, value_bias: bool = True,
                 mlp_bias: bool = True, rope_theta: float = 100.0, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.patch_embed = DinoV3PatchEmbed(patch_size=patch_size, embed_dim=hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size))
        self.rope = DinoV3RotaryEmbedding2D(dim=hidden_size // num_heads, base=rope_theta)
        qkv_bias = (query_bias, key_bias, value_bias)
        self.blocks = nn.ModuleList([
            DinoV3Block(hidden_size, num_heads, mlp_ratio=intermediate_size / hidden_size,
                        qkv_bias=qkv_bias, layerscale_init=layerscale_init,
                        mlp_bias=mlp_bias, eps=layer_norm_eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    @property
    def device(self) -> torch.device:
        return self.cls_token.device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pixel_values.shape
        x = self.patch_embed(pixel_values)
        hp, wp = H // self.patch_size, W // self.patch_size
        cos, sin = self.rope(hp, wp, x.device, x.dtype)
        x = torch.cat([self.cls_token.expand(B, -1, -1),
                        self.register_tokens.expand(B, -1, -1), x], dim=1)
        num_prefix = 1 + self.num_register_tokens
        for block in self.blocks:
            x = block(x, cos, sin, num_prefix_tokens=num_prefix)
        return self.norm(x)

    def load_safetensors(self, path: str) -> None:
        state_dict = safetensors.torch.load_file(path)
        our_sd = self.state_dict()
        loaded = {}
        for hf_key in state_dict:
            k = (hf_key
                 .replace("embeddings.patch_embeddings.", "patch_embed.proj.")
                 .replace("embeddings.cls_token", "cls_token")
                 .replace("embeddings.mask_token", "mask_token")
                 .replace("embeddings.register_tokens", "register_tokens"))
            m = re.match(r"layer\.(\d+)\.(.+)", k)
            if m:
                rest = m.group(2)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    rest = rest.replace(f"attention.{proj}", f"attn.{proj}")
                rest = (rest.replace("layer_scale1.lambda1", "ls1")
                            .replace("layer_scale2.lambda1", "ls2"))
                k = f"blocks.{m.group(1)}.{rest}"
            if k in our_sd:
                assert state_dict[hf_key].shape == our_sd[k].shape, \
                    f"Shape mismatch {k}: {state_dict[hf_key].shape} vs {our_sd[k].shape}"
                loaded[k] = state_dict[hf_key]
        check_sd = {k: v for k, v in our_sd.items() if k != "mask_token"}
        missing = set(check_sd) - set(loaded)
        unexpected = set(loaded) - set(check_sd)
        if missing:
            raise KeyError(f"[DINOv3] Missing keys: {missing}")
        if unexpected:
            raise KeyError(f"[DINOv3] Unexpected keys: {unexpected}")
        self.load_state_dict(loaded, strict=True)


# ---------------------------------------------------------------------------
# Flux2 VAE Encoder
# ---------------------------------------------------------------------------

class Flux2ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_shortcut=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0) if use_shortcut else None

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = F.silu(self.norm2(self.conv1(h)))
        h = self.conv2(h)
        return h + (self.conv_shortcut(x) if self.conv_shortcut is not None else x)


class Flux2Downsampler(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Flux2Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Identity()])

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x).reshape(B, C, H * W).transpose(1, 2)
        q = self.to_q(h).reshape(B, -1, 1, C).permute(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, -1, 1, C).permute(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, -1, 1, C).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = self.to_out[0](out.permute(0, 2, 1, 3).reshape(B, -1, C))
        return x + out.transpose(1, 2).reshape(B, C, H, W)


class Flux2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, 3, 1, 1)
        self.down_0_resnets = nn.ModuleList([Flux2ResnetBlock(128, 128), Flux2ResnetBlock(128, 128)])
        self.down_0_sampler = Flux2Downsampler(128)
        self.down_1_resnets = nn.ModuleList([Flux2ResnetBlock(128, 256, use_shortcut=True), Flux2ResnetBlock(256, 256)])
        self.down_1_sampler = Flux2Downsampler(256)
        self.down_2_resnets = nn.ModuleList([Flux2ResnetBlock(256, 512, use_shortcut=True), Flux2ResnetBlock(512, 512)])
        self.down_2_sampler = Flux2Downsampler(512)
        self.down_3_resnets = nn.ModuleList([Flux2ResnetBlock(512, 512), Flux2ResnetBlock(512, 512)])
        self.mid_attn = Flux2Attention(512)
        self.mid_resnets = nn.ModuleList([Flux2ResnetBlock(512, 512), Flux2ResnetBlock(512, 512)])
        self.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6)
        self.conv_out = nn.Conv2d(512, 64, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for r in self.down_0_resnets: x = r(x)
        x = self.down_0_sampler(x)
        for r in self.down_1_resnets: x = r(x)
        x = self.down_1_sampler(x)
        for r in self.down_2_resnets: x = r(x)
        x = self.down_2_sampler(x)
        for r in self.down_3_resnets: x = r(x)
        x = self.mid_resnets[0](x)
        x = self.mid_attn(x)
        x = self.mid_resnets[1](x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Flux2VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Flux2Encoder()
        self.quant_conv = nn.Conv2d(64, 64, 1, 1, 0)
        self.bn = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True)

    def load_safetensors(self, path: str):
        sd = safetensors.torch.load_file(path)
        remapped = {}
        for k, v in sd.items():
            # Skip the decoder half of a full Flux2-VAE ckpt — we only need the encoder.
            if k.startswith(("decoder.", "post_quant_conv.")):
                continue
            # Comfy / diffusers-style naming → our flattened naming.
            m = re.match(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", k)
            if m:
                remapped[f"encoder.down_{m.group(1)}_resnets.{m.group(2)}.{m.group(3)}"] = v
                continue
            m = re.match(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.+)", k)
            if m:
                remapped[f"encoder.down_{m.group(1)}_sampler.{m.group(2)}"] = v
                continue
            m = re.match(r"encoder\.mid_block\.resnets\.(\d+)\.(.+)", k)
            if m:
                remapped[f"encoder.mid_resnets.{m.group(1)}.{m.group(2)}"] = v
                continue
            m = re.match(r"encoder\.mid_block\.attentions\.0\.(.+)", k)
            if m:
                remapped[f"encoder.mid_attn.{m.group(1)}"] = v
                continue
            remapped[k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing:
            raise KeyError(f"[VAE] Missing keys: {missing}")
        if unexpected:
            raise KeyError(f"[VAE] Unexpected keys: {unexpected}")

    def encode(self, images, deterministic: bool = True, generator: torch.Generator = None):
        moments = self.quant_conv(self.encoder(images))
        mean, logvar = moments.chunk(2, dim=1)
        if deterministic:
            latents = mean
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            latents = mean + torch.exp(0.5 * logvar) * noise
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(B, C * 4, H // 2, W // 2)
        bn_mean = self.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn.eps).to(latents.device, latents.dtype)
        return ((latents - bn_mean) / bn_std).to(torch.float32).flatten(2).transpose(1, 2).contiguous()

        return rgba


# ---------------------------------------------------------------------------
# Shared transformer helpers
# ---------------------------------------------------------------------------

class LayerNorm32(nn.LayerNorm):
    def forward(self, x):
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x):
        origin_dtype = x.dtype
        return (F.normalize(x.float(), dim=-1) * self.gamma.float() * self.scale).to(origin_dtype)


def apply_rotary_emb(hidden_states, freqs):
    x_rotated = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    x_rotated = x_rotated * freqs
    x_out = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1)
    return x_out.type_as(hidden_states)


def clamp_mul(x, f):
    f_t = f.tanh()
    return x * f_t + x.detach() * (f - f_t)


def scaled_dot_product_attention(qkv=None, q=None, k=None, v=None, kv=None):
    if qkv is not None:
        q, k, v = qkv.unbind(dim=2)
    elif kv is not None:
        k, v = kv.unbind(dim=2)
    q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    return F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class RePo3DRotaryEmbedding(nn.Module):
    def __init__(self, model_channels, num_heads, head_dim, repo_hidden_ratio=0.125, max_freq=16.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(repo_hidden_size, 3 * num_heads, bias=False)
        self.dim_0 = 2 * (head_dim // 6)
        self.dim_1 = 2 * (head_dim // 6)
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        dims = [self.dim_0, self.dim_1, self.dim_2]
        freqs_list = []
        for d in dims:
            freq_dim = d // 2
            freqs_list.append(torch.linspace(1.0, float(max_freq), steps=freq_dim, dtype=torch.float32))
        self.freqs_0 = nn.Parameter(freqs_list[0])
        self.freqs_1 = nn.Parameter(freqs_list[1])
        self.freqs_2 = nn.Parameter(freqs_list[2])

    def forward(self, hidden_states):
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1).float()  # fp32 needed for torch.polar → complex64
        return torch.polar(torch.ones_like(ang), ang).type(torch.complex64)


class PcdAbsolutePositionEmbedder(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def _freqs(self, device):
        freqs_2exp = torch.arange(self.max_res, dtype=torch.float32, device=device)
        res_dim = max(0, self.freq_dim - self.max_res)
        freqs_res = (torch.arange(res_dim, dtype=torch.float32, device=device) / max(res_dim, 1) * self.max_res
                     if res_dim > 0 else torch.empty(0, device=device))
        freqs = torch.cat([freqs_2exp, freqs_res], dim=0)[:self.freq_dim]
        return torch.pow(2.0, freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        *dims, D = x.shape
        out = torch.outer(x.reshape(-1), self._freqs(x.device)) * 2 * torch.pi
        out = torch.cat([out.sin(), out.cos()], dim=-1).reshape(*dims, -1)
        if out.shape[-1] < self.channels:
            out = torch.cat([out, torch.zeros(*dims, self.channels - out.shape[-1],
                                              device=out.device, dtype=out.dtype)], dim=-1)
        return out.to(orig_dtype)


class PcdAbsolutePositionEmbedderV2(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def _freqs(self, device):
        logs = torch.linspace(0.0, float(self.max_res), steps=self.freq_dim, dtype=torch.float32, device=device)
        return torch.pow(2.0, logs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        N, D = x.shape
        ang = x.unsqueeze(-1) * self._freqs(x.device) * torch.pi
        embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1],
                                                   device=embed.device, dtype=embed.dtype)], dim=-1)
        return embed.to(orig_dtype)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class FeedForwardNet(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0, channels_out=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels if channels_out is None else channels_out),
        )

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, channels: int, inner_channels: int, channels_out: Optional[int] = None,
                 mlp_layer_num: int = 2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(nn.Linear(channels if i == 0 else inner_channels, inner_channels))
            layers.append(nn.GELU(approximate="tanh"))
        layers.append(nn.Linear(inner_channels, channels if channels_out is None else channels_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RopeMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads, ctx_channels=None, type="self",
                 attn_mode="full", qkv_bias=True, qk_rms_norm=False, use_rope=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        self.use_rope = use_rope
        if self._type == "self":
            self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(channels, channels, bias=qkv_bias)
            self.kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if self.qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, context=None, rope_emb=None):
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            if self.use_rope:
                q = apply_rotary_emb(q, rope_emb)
                k = apply_rotary_emb(k, rope_emb)
        else:
            q = self.q(x).reshape(B, L, self.num_heads, self.head_dim)
            if context is None:
                raise ValueError("Context must be provided for cross attention")
            kv = self.kv(context).reshape(B, context.shape[1], 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        h = scaled_dot_product_attention(q=q, k=k, v=v)
        return self.out(h.reshape(B, L, C))


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads, ctx_channels=None, type="self",
                 attn_mode="full", qkv_bias=True, qk_rms_norm=False):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"]
        assert attn_mode == "full"
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, context=None):
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = torch.stack([q, k, v], dim=2)
            h = scaled_dot_product_attention(qkv=qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x).reshape(B, L, self.num_heads, -1)
            kv = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q=q, k=k, v=v)
            else:
                h = scaled_dot_product_attention(q=q, kv=kv)
        return self.to_out(h.reshape(B, L, -1))


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0, attn_mode="full",
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False, qkv_bias=True,
                 modulation=True, share_mod=False, use_shift_table=False):
        super().__init__()
        self.modulation = modulation
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = RopeMultiHeadAttention(channels, num_heads=num_heads, type="self",
                                            attn_mode=attn_mode, qkv_bias=qkv_bias,
                                            use_rope=use_rope, qk_rms_norm=qk_rms_norm)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if modulation:
            if not share_mod:
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))
            self.shift_table = nn.Parameter(torch.randn(1, 6 * channels) / channels ** 0.5) if use_shift_table else None

    def forward(self, x, mod=None, rotary_emb=None):
        if self.modulation:
            if not self.share_mod:
                mod = self.adaLN_modulation(mod)
            if hasattr(self, 'shift_table') and self.shift_table is not None:
                mod = mod + self.shift_table.type(mod.dtype)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
            h = self.norm1(x)
            h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            h = self.attn(h, rope_emb=rotary_emb)
            x = x + h * gate_msa.unsqueeze(1)
            h = self.norm2(x)
            h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            x = x + self.mlp(h) * gate_mlp.unsqueeze(1)
        else:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Quasi-random sampling utilities
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)


@torch.no_grad()
def sample_probs(probs, counts, algo="systematic", generator: torch.Generator = None):
    batch_shape = counts.shape
    B = counts.numel()
    P = probs.size(-1)
    device = probs.device
    probs = probs.view(B, P)
    counts = counts.view(B)

    probs = probs.to(torch.float32).clamp_min_(0)
    row_sums = probs.sum(1, keepdim=True)
    zero_mask = row_sums.eq(0)
    probs = probs / row_sums.clamp_min_(1)
    if zero_mask.any():
        probs = probs.clone()
        probs[zero_mask.expand_as(probs)] = 1.0 / P

    counts = counts.to(device=device, dtype=torch.long)
    out = torch.zeros(B, P, dtype=torch.long, device=device)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    unique_n, inv = counts.unique(sorted=False, return_inverse=True)
    for i, n in enumerate(unique_n.tolist()):
        if n == 0:
            continue
        rows = (inv == i).nonzero(as_tuple=False).squeeze(1)
        r = rows.numel()
        U0 = torch.rand(r, 1, device=device, generator=generator) / float(n)
        grid = torch.arange(n, device=device, dtype=torch.float32)[None, :] / float(n)
        us = (U0 + grid).clamp(max=1.0 - 1e-12)
        cdf_rows = cdf.index_select(0, rows)
        idx = torch.searchsorted(cdf_rows, us).clamp_max(probs.size(1) - 1)
        buf = torch.zeros(r, P, dtype=torch.float32, device=device)
        buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
        out.index_copy_(0, rows, buf.to(torch.long))

    return out.view(*batch_shape, P)


# ---------------------------------------------------------------------------
# VAE decoders
# ---------------------------------------------------------------------------

class LevelEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def level_embedding(t, dim, max_period=1024):
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None] * 2 * torch.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        emb = self.level_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0, share_mod=False,
                 qk_rms_norm_cross=True, qkv_bias=True):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads,
                                              type="cross", attn_mode="full", qkv_bias=qkv_bias,
                                              qk_rms_norm=qk_rms_norm_cross)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def forward(self, x, mod, context):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.cross_attn(h, context) * gate_msa.unsqueeze(1)
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(h) * gate_mlp.unsqueeze(1)
        return x


class ModulatedCrossOnlyTransformerBase(nn.Module):
    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads=None,
                 num_head_channels=64, mlp_ratio=4.0, share_mod=False, additional_level_embed=False,
                 qk_rms_norm_cross=True):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm_cross = qk_rms_norm_cross

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.l_embedder = LevelEmbedder(model_channels)
        self.l_embedder2 = LevelEmbedder(model_channels, max_period=100) if additional_level_embed else None
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        if cond_channels is not None:
            self.blocks = nn.ModuleList([
                ModulatedTransformerCrossOnlyBlock(
                    model_channels, ctx_channels=cond_channels, num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, qk_rms_norm_cross=self.qk_rms_norm_cross,
                    share_mod=self.share_mod)
                for _ in range(num_blocks)
            ])

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x, l, cond, l2=None):
        h = self.input_layer(x)
        l_emb = self.l_embedder(l)
        if self.l_embedder2 is not None and l2 is not None:
            l_emb = l_emb + self.l_embedder2(l2)
        if self.share_mod:
            l_emb = self.adaLN_modulation(l_emb)
        for block in self.blocks:
            h = block(h, l_emb, cond)
        return h


class OctreeProbabilityFixedlenDecoder(ModulatedCrossOnlyTransformerBase):
    def __init__(self, model_channels, cond_channels, num_blocks, num_heads=None,
                 num_head_channels=64, mlp_ratio=4.0, share_mod=False,
                 additional_level_embed=False, qk_rms_norm_cross=True, *,
                 no_norm=False):
        super().__init__(
            in_channels=model_channels, model_channels=model_channels,
            cond_channels=cond_channels, num_blocks=num_blocks,
            num_heads=num_heads, num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio, share_mod=share_mod,
            additional_level_embed=additional_level_embed,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        self.out_proj = nn.Linear(self.model_channels, 8)
        self.no_norm = no_norm
        self.in_proj = nn.Linear(3, self.model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderV2(channels=model_channels, in_channels=3)

    def forward(self, x, l, cond, l2=None):
        d = self.dtype
        B, L, C = x.shape
        h = self.in_proj(x.to(d)) + self.pos_embedder(x.reshape(-1, 3)).reshape(B, L, -1).to(d)
        if l2 is not None:
            l2 = torch.log2(l2)
        h = super().forward(h, l, cond.to(d), l2)
        h = F.layer_norm(h.float(), h.shape[-1:]).to(d) if not self.no_norm else h / (1 + 2 * self.num_blocks) ** 0.5
        logits = self.out_proj(h)
        return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}

    @staticmethod
    def sample(
        model,
        cond,
        num_points,
        level,
        temperature=1.0,
        algo="systematic",
        generator: torch.Generator = None,
        callback=None,
    ):
        B = cond.shape[0]
        device = cond.device
        child_offset = torch.tensor([[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]],
                                    dtype=torch.long, device=device)
        prev_coords_int = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
        prev_counts = torch.full((B, 1), num_points, dtype=torch.long, device=device)
        prev_log_probs = torch.zeros(B, 1, dtype=torch.float32, device=device)
        batch_indices_range = torch.arange(B, device=device).unsqueeze(1)
        num_tensor = torch.full((B,), num_points, dtype=torch.long, device=device)

        for lv in range(1, level + 1):
            res_p = 1 << (lv - 1)
            res = 1 << lv
            parent_coords_norm = (prev_coords_int.to(torch.float32) + 0.5) / res_p
            res_tensor = torch.full((B,), res, dtype=torch.long, device=device)
            pred_logits = model(parent_coords_norm, res_tensor, cond, num_tensor)["logits"] / temperature
            pred_probs = torch.softmax(pred_logits, dim=-1)
            pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
            sampled = sample_probs(
                pred_probs,
                prev_counts,
                algo=algo,
                generator=generator,
            ).flatten(1, 2)
            pred_log_probs = pred_log_probs.flatten(1, 2)
            prev_log_probs_expanded = prev_log_probs.repeat_interleave(8, dim=1)
            child_coords_int = (prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]).flatten(1, 2)
            mask = sampled > 0
            max_valid = mask.sum(dim=1).max().item()
            scatter_indices = mask.cumsum(dim=1) - 1
            valid_scatter_indices = scatter_indices[mask]
            valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
            next_prev_coords_int = torch.zeros(B, max_valid, 3, dtype=child_coords_int.dtype, device=device)
            next_prev_coords_int[valid_batch_indices, valid_scatter_indices] = child_coords_int[mask]
            next_prev_counts = torch.zeros(B, max_valid, dtype=sampled.dtype, device=device)
            next_prev_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]
            next_prev_log_probs = torch.zeros(B, max_valid, dtype=prev_log_probs.dtype, device=device)
            next_prev_log_probs[valid_batch_indices, valid_scatter_indices] = (prev_log_probs_expanded + pred_log_probs)[mask]
            prev_coords_int = next_prev_coords_int
            prev_counts = next_prev_counts
            prev_log_probs = next_prev_log_probs
            if callback is not None:
                callback("octree", lv, level)

        res = 1 << level
        prev_log_probs = torch.repeat_interleave(prev_log_probs.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(B, num_points)
        coords_int = torch.repeat_interleave(prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(B, num_points, -1)
        jitter = torch.rand(
            coords_int.shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        coords_norm = (coords_int.to(torch.float32) + jitter) / res
        return {"points": coords_norm, "log_probs": prev_log_probs}


class TransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0, attn_mode="full",
                 qk_rms_norm=True, qk_rms_norm_cross=True, qkv_bias=True):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(channels, num_heads=num_heads, type="self",
                                             attn_mode=attn_mode, qkv_bias=qkv_bias,
                                             qk_rms_norm=qk_rms_norm)
        self.cross_attn = MultiHeadAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads,
                                              type="cross", attn_mode="full", qkv_bias=qkv_bias,
                                              qk_rms_norm=qk_rms_norm_cross)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class TransformerBase(nn.Module):
    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads=None,
                 num_head_channels=64, mlp_ratio=4.0, attn_mode="full", window_num=None,
                 qk_rms_norm=True, qk_rms_norm_cross=True):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.input_layer = nn.Linear(in_channels, model_channels)
        if cond_channels is not None:
            self.blocks = nn.ModuleList([
                TransformerCrossBlock(model_channels, ctx_channels=cond_channels,
                                     num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                     attn_mode="full", qk_rms_norm=qk_rms_norm,
                                     qk_rms_norm_cross=qk_rms_norm_cross)
                for _ in range(num_blocks)
            ])

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x, cond=None, l=None, cond2=None):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h, cond)
        return h


class FixedlenDecoder(TransformerBase):
    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads=None,
                 num_head_channels=64, mlp_ratio=4.0, attn_mode="full", window_num=None,
                 qk_rms_norm=True, qk_rms_norm_cross=True):
        super().__init__(in_channels=model_channels, model_channels=model_channels,
                         cond_channels=cond_channels, num_blocks=num_blocks,
                         num_heads=num_heads, num_head_channels=num_head_channels,
                         mlp_ratio=mlp_ratio, attn_mode=attn_mode, window_num=window_num,
                         qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross)
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderV2(channels=model_channels, in_channels=3)

    def forward(self, x=None, cond=None):
        pcd = x["points"]
        d = self.dtype
        B, L, C = pcd.shape
        h = self.in_proj(pcd.to(d)) + self.pos_embedder(pcd.reshape(-1, 3)).reshape(B, L, -1).to(d)
        return super().forward(h, cond.to(d))


class ElasticGaussianFixedlenDecoder(FixedlenDecoder):
    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads=None,
                 num_head_channels=64, mlp_ratio=4.0, attn_mode="full", window_num=None,
                 *, no_norm=False, representation_config=None,
                 use_learned_offset_scale=True, use_per_offset=True,
                 qk_rms_norm=True, qk_rms_norm_cross=True):
        self.rep_config = representation_config
        self.use_learned_offset_scale = use_learned_offset_scale
        self.use_per_offset = use_per_offset
        self.out_channels = self._calc_layout()
        super().__init__(in_channels=in_channels, model_channels=model_channels,
                         cond_channels=cond_channels, num_blocks=num_blocks,
                         num_heads=num_heads, num_head_channels=num_head_channels,
                         mlp_ratio=mlp_ratio, attn_mode=attn_mode, window_num=window_num,
                         qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross)
        self.out_proj = nn.Linear(model_channels, self.out_channels)
        self.no_norm = no_norm
        self._build_perturbation()

    def _calc_layout(self):
        ng = self.rep_config['num_gaussians']
        self.layout = {
            '_xyz':         {'shape': (ng, 3),    'size': ng * 3},
            '_features_dc': {'shape': (ng, 1, 3), 'size': ng * 3},
            '_scaling':     {'shape': (ng, 3),    'size': ng * 3},
            '_rotation':    {'shape': (ng, 4),    'size': ng * 4},
            '_opacity':     {'shape': (ng, 1),    'size': ng},
        }
        if self.use_learned_offset_scale and self.use_per_offset:
            self.layout['_offset_scale'] = {'shape': (ng, 1), 'size': ng}
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        return start

    def _build_perturbation(self):
        ng = self.rep_config['num_gaussians']
        perturbation = torch.tensor([hammersley_sequence(3, i, ng) for i in range(ng)]).float()
        perturbation = torch.atanh((perturbation * 2 - 1) / self.rep_config['perturbe_size'])
        self.register_buffer('points_offset_perturbation', perturbation)
        if self.use_learned_offset_scale:
            base = torch.tensor(self.rep_config['offset_scale'])
            self.register_buffer('base_offset_scale', torch.log(torch.exp(base) - 1.0))

    def _get_offset(self, h):
        B = h.shape[0]
        if self.use_learned_offset_scale:
            r = self.layout['_offset_scale']['range']
            _offset_scale = F.softplus(
                h[:, :, r[0]:r[1]].reshape(B, -1, *self.layout['_offset_scale']['shape'])
                + self.base_offset_scale)

        r = self.layout['_xyz']['range']
        offset = h[:, :, r[0]:r[1]].reshape(B, -1, *self.layout['_xyz']['shape'])
        offset = offset * self.rep_config['lr']['_xyz']
        if self.rep_config['perturb_offset']:
            offset = offset + self.points_offset_perturbation
        offset = torch.tanh(offset) * 0.5 * self.rep_config['perturbe_size']
        offset = offset * (_offset_scale if self.use_learned_offset_scale else self.rep_config['offset_scale'])
        return offset

    def forward(self, x=None, cond=None):
        h = super().forward(x, cond)
        h = F.layer_norm(h.float(), h.shape[-1:]).to(h.dtype) if not self.no_norm else h / (1 + 3 * self.num_blocks) ** 0.5
        return {"features": self.out_proj(h)}


# ---------------------------------------------------------------------------
# Flow matching denoiser
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LatentSeqMMFlowModel(nn.Module):
    def __init__(self, q_token_length, in_channels, model_channels, cond_channels,
                 out_channels, num_blocks, num_refiner_blocks=2, num_heads=None,
                 num_head_channels=64, cam_channels=None, cond2_channels=None,
                 mlp_ratio=4, share_mod=True, qk_rms_norm=False, use_shift_table=False):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.cond2_channels = cond2_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_refiner_blocks = num_refiner_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.use_shift_table = use_shift_table

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels) if cond2_channels is not None else None

        sobol_seq = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(q_token_length)
        self.register_buffer("pos_pe", sobol_seq.unsqueeze(0), persistent=False)
        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
        self.noise_repo_layers = nn.ModuleList([
            RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
            for _ in range(num_refiner_blocks)])
        self.context_repo_layers = nn.ModuleList([
            RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
            for _ in range(num_refiner_blocks)])
        self.repo_layers = nn.ModuleList([
            RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
            for _ in range(num_blocks)])

        block_kwargs = dict(num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, attn_mode='full',
                            use_rope=True, qk_rms_norm=self.qk_rms_norm,
                            use_shift_table=self.use_shift_table)
        self.noise_refiner = nn.ModuleList([
            UnifiedTransformerBlock(model_channels, modulation=True, share_mod=self.share_mod, **block_kwargs)
            for _ in range(num_refiner_blocks)])
        self.context_refiner = nn.ModuleList([
            UnifiedTransformerBlock(model_channels, modulation=False, **block_kwargs)
            for _ in range(num_refiner_blocks)])
        if self.cam_channels is not None:
            self.cam_refiner = MLP(self.cam_channels, model_channels, model_channels,
                                   mlp_layer_num=num_refiner_blocks)
        self.blocks = nn.ModuleList([
            UnifiedTransformerBlock(model_channels, modulation=True, share_mod=self.share_mod, **block_kwargs)
            for _ in range(num_blocks)])
        self.shift_table = nn.Parameter(torch.randn(1, 2, model_channels) / model_channels**0.5) if use_shift_table else None
        self.out_layer = nn.Linear(model_channels, out_channels)
        if cam_channels is not None:
            self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_safetensors(self, path: str) -> None:
        self.load_state_dict(safetensors.torch.load_file(path), strict=True)

    def prepare_condition(self, cond):
        d = self.dtype
        feat1 = cond['feature1'].to(d)
        feat2 = cond['feature2'].to(d) if self.cond_embedder2 is not None else None
        h_cond = self.cond_embedder(feat1)
        if feat2 is not None:
            h_cond = h_cond + self.cond_embedder2(feat2)
        for i, block in enumerate(self.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=self.context_repo_layers[i](h_cond))
        return h_cond

    def forward(self, x_t, t, cond):
        d = self.dtype
        z = x_t['latent'].to(d)
        if isinstance(cond, dict) and "prepared_context" in cond:
            h_cond = cond["prepared_context"].to(device=z.device, dtype=d)
        else:
            h_cond = self.prepare_condition(cond)

        h_x = self.input_layer(z)
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb) if self.share_mod else t_emb

        h_x = h_x + self.pos_embedder(self.pos_pe).to(d)

        for i, block in enumerate(self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=self.noise_repo_layers[i](h_x))

        if self.cam_channels is not None:
            cam = x_t.get('camera').to(d)
            h_cam = self.cam_refiner(cam)

        h = torch.cat([h_x, h_cond], dim=1)
        if self.cam_channels is not None:
            h = torch.cat([h, h_cam], dim=1)

        for i, block in enumerate(self.blocks):
            h = block(h, mod=t_mod, rotary_emb=self.repo_layers[i](h))

        h_x = F.layer_norm(h[:, :z.shape[1]].float(), h.shape[-1:]).type(d)
        if self.cam_channels is not None:
            h_cam = F.layer_norm(h[:, -cam.shape[1]:].float(), h.shape[-1:]).type(d)

        if self.use_shift_table:
            shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
            h_x = h_x * (1 + scale) + shift
            if self.cam_channels is not None:
                h_cam = h_cam * (1 + scale) + shift

        out = {'latent': self.out_layer(h_x)}
        if self.cam_channels is not None:
            out['camera'] = self.cam_out_layer(h_cam)
        return out


# ---------------------------------------------------------------------------
# OctreeGaussianDecoder
# ---------------------------------------------------------------------------

class OctreeGaussianDecoder(nn.Module):
    _MAX_VOXEL_LEVEL = 8

    def __init__(self, octree_args: dict, gs_args: dict):
        super().__init__()
        self.octree = OctreeProbabilityFixedlenDecoder(**octree_args)
        self.gs     = ElasticGaussianFixedlenDecoder(**gs_args)

    def load_safetensors(self, path: str) -> None:
        self.load_state_dict(safetensors.torch.load_file(path), strict=True)

    @property
    def gaussians_per_point(self) -> int:
        return self.gs.rep_config['num_gaussians']

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        num_gaussians: int,
        generator: torch.Generator = None,
        callback=None,
    ):
        from .triposplat import _build_gaussians  # local import: avoid model.py ↔ triposplat.py cycle
        num_decoder_tokens = max(1, num_gaussians // self.gaussians_per_point)
        points_pred = OctreeProbabilityFixedlenDecoder.sample(
            self.octree, latent,
            num_points=num_decoder_tokens, level=self._MAX_VOXEL_LEVEL,
            temperature=1.0, algo='systematic',
            generator=generator,
            callback=callback,
        )
        if callback is not None:
            callback("gaussian", 0, 1)
        pred = self.gs(x=points_pred, cond=latent)
        if callback is not None:
            callback("gaussian", 1, 1)
        return _build_gaussians(self.gs, points_pred, pred)[0]
