"""ControlNet-LLLite for Anima (DiT) — ComfyUI port (v2 architecture).

Adapted from kohya-ss/sd-scripts. The on-disk weight format is the v2
named-key format (per-module key prefix = lllite_name, shared encoder under
``lllite_conditioning1.*``, depth embedding split per-module as
``{name}.depth_embed``); legacy ``lllite_modules.*`` files are rejected.

Differences vs. the sd-scripts reference (``networks/control_net_lllite_anima.py``):
  * No dependency on ``library.utils`` — uses stdlib logging.
  * Module discovery filters the LLM-Adapter sub-tree by class identity in
    addition to the path-based check (ComfyUI ships two distinct ``Attention``
    classes that share the bare class name).
  * ``LLLiteModuleDiT`` keeps a ``restore()`` method (and an idempotent
    ``apply_to()``); ComfyUI patches/unpatches the original Linear around
    every sampler call via ``set_model_unet_function_wrapper``.
  * Forward pass casts ``x`` and ``cond_emb`` to the LLLite parameter dtype
    so autocast / mixed-precision flows that hand us a different dtype than
    the LLLite weights still work.
  * CFG batch-size and sequence-length mismatches fall back to identity
    instead of asserting, so a slightly-off cond image cannot abort sampling.
  * The training-side ``AnimaControlNetLLLiteWrapper`` is omitted; ComfyUI
    integrates via ``model_function_wrapper`` in nodes.py instead.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Class names of the modules that LLLite injects into. The LLM-Adapter uses
# a different ``Attention`` class with the same bare name; we filter it by
# path (``llm_adapter`` in the qualified name) and by the ``is_selfattn``
# attribute presence.
TARGET_ATTENTION_CLASS = "Attention"
TARGET_MLP_CLASS = "GPT2FeedForward"
LLM_ADAPTER_NAME = "llm_adapter"

LLLITE_ARCH_VERSION = "2"


# ----------------------------------------------------------------------------
# target_layers: atomic specifiers and presets
# ----------------------------------------------------------------------------

ATOMIC_SPECIFIERS: Tuple[str, ...] = (
    "self_attn_q_pre",
    "self_attn_kv_pre",
    "cross_attn_q_pre",
    "mlp_fc1_pre",
)

PRESETS: dict = {
    "self_attn_q":            ("self_attn_q_pre",),
    "self_attn_qkv":          ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q":  ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}


def parse_target_layers(spec: str) -> Tuple[str, ...]:
    """Resolve a ``target_layers`` spec to a canonical atomic tuple.

    Accepts a preset name (``"self_attn_qkv"``) or a comma-separated list of
    atomic specifiers (``"self_attn_q_pre,mlp_fc1_pre"``). Returns the atomics
    in ``ATOMIC_SPECIFIERS`` order with duplicates removed.
    """
    if not isinstance(spec, str):
        raise TypeError(f"target_layers must be str, got {type(spec).__name__}")
    spec = spec.strip()
    if not spec:
        raise ValueError("target_layers spec is empty")

    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(
                f"unknown target_layers atomic specifier(s): {bad}. "
                f"valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}"
            )

    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


# ----------------------------------------------------------------------------
# Conditioning1 trunk (v2)
# ----------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


ASPP_DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1
        self.proj = nn.Sequential(
            nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        assert cond_in_channels >= 1, f"cond_in_channels must be >= 1, got {cond_in_channels}"
        ch_half = cond_dim // 2

        self.cond_in_channels = cond_in_channels
        self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        h = self.out_norm(h)
        return h


# ----------------------------------------------------------------------------
# LLLite module (v2: FiLM + SiLU + 5D path + depth embedding)
# ----------------------------------------------------------------------------

class LLLiteModuleDiT(nn.Module):
    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        # Wrap in a list so the original Linear is not registered as a submodule
        # and its weights stay out of state_dict.
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        # FiLM: cond_local -> (gamma, beta), zero-init for identity at start.
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.cond_emb: Optional[torch.Tensor] = None
        self.org_forward = None

        # Set by the parent ControlNetLLLiteDiT after construction.
        self.layer_idx: int = -1
        self._depth_embeds_ref: List[nn.Parameter] = []

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layouts:
        #   self/cross attention q/k/v: (B, S, D) — already flattened in the Anima block
        #   mlp.layer1:                 (B, T, H, W, D) — passed un-flattened
        # Flatten the 5D case to 3D for the LLLite path and reshape on exit.
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        cx = self.cond_emb  # (B_c, S, cond_emb_dim)

        # Broadcast cond_emb to the runtime batch (CFG cond+uncond, multi-cond).
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self.org_forward(x.reshape(orig_shape) if is_5d else x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            return self.org_forward(x.reshape(orig_shape) if is_5d else x)

        # Run the LLLite mini-MLP in its own parameter dtype, then cast the
        # correction back to ``x``'s dtype before adding. Robust to autocast
        # flows where x and LLLite weights have different dtypes.
        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        # Per-module depth embedding (zero-init so it's a no-op at train start).
        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            if depth_e.dtype != param_dtype or depth_e.device != x.device:
                depth_e = depth_e.to(device=x.device, dtype=param_dtype)
            cond_local = cx + depth_e
        else:
            cond_local = cx

        h = F.silu(self.down(x_proc))

        gb = self.cond_to_film(cond_local)
        gamma, beta = gb.chunk(2, dim=-1)

        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        y = self.org_forward(x + out)

        if is_5d:
            # org Linear out_features may differ from in_features — recover with -1.
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


# ----------------------------------------------------------------------------
# ControlNetLLLiteDiT
# ----------------------------------------------------------------------------

class ControlNetLLLiteDiT(nn.Module):
    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,
        cond_resblocks: int = 1,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        inpaint_masked_input: bool = False,
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.target_atomics = atomics
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        # 4ch (RGB+mask) inpainting metadata. `inpaint_masked_input` records the training-time
        # RGB-masking policy for cond_image preparation; it does not alter the forward pass here.
        self.cond_in_channels = cond_in_channels
        self.inpaint_masked_input = inpaint_masked_input

        self.conditioning1 = _Conditioning1(
            cond_dim, cond_emb_dim, cond_resblocks,
            use_aspp=use_aspp, aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)

        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]

        aspp_info = f"aspp={'on' + str(list(self.aspp_dilations)) if use_aspp else 'off'}"
        inpaint_info = (
            f", inpaint=on(masked_input={inpaint_masked_input})" if cond_in_channels != 3 else ""
        )
        logger.info(
            "ControlNet-LLLite (Anima v%s): created %d modules for target=%r "
            "(atomics=%s), cond_in_channels=%d, cond_dim=%d, cond_resblocks=%d, %s, "
            "cond_emb_dim=%d, mlp_dim=%d%s",
            LLLITE_ARCH_VERSION, n, target_layers, list(atomics),
            cond_in_channels, cond_dim, cond_resblocks, aspp_info, cond_emb_dim, mlp_dim,
            inpaint_info,
        )

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: Tuple[str, ...]) -> bool:
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
            return False
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
            return False  # cross_attn K,V live in text-embedding space

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        atomics: Tuple[str, ...],
        dropout: Optional[float],
        multiplier: float,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        want_mlp_fc1 = "mlp_fc1_pre" in atomics
        any_attn = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"))

        for name, module in dit.named_modules():
            if LLM_ADAPTER_NAME in name:
                continue
            cls = module.__class__.__name__

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                # The Anima-block Attention exposes is_selfattn; the LLM-Adapter
                # Attention does not — skip the latter even if path filter misses.
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                for child_name, child in module.named_children():
                    if not isinstance(child, nn.Linear):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(
                        LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                child = getattr(module, "layer1", None)
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        """cond_image: (B, 3, H*16, W*16) in [-1, 1]; ``None`` clears."""
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()


# ----------------------------------------------------------------------------
# Save / load (named-key format; legacy lllite_modules.* is rejected)
# ----------------------------------------------------------------------------

_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _from_saved_state_dict(lllite: "ControlNetLLLiteDiT", weights_sd: dict) -> dict:
    """Rewrite a v2 named-key state dict back to the internal layout."""
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX):]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[: -len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(
                f"depth_embed slices missing for module idx(es) {missing}"
            )
        out[_INTERNAL_DEPTH_KEY] = torch.stack(
            [depth_slices[i] for i in range(n_modules)], dim=0
        )

    return out


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {file} appear to be in a legacy ControlNet-LLLite weight format "
            f"(keys starting with '{_INTERNAL_MODULES_PREFIX}'). The current code uses a "
            f"named-key format (per-module key prefix = lllite_name, e.g. "
            f"'lllite_dit_blocks_0_self_attn_q_proj.down.weight'). Re-train with the current codebase."
        )

    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    logger.info("loaded LLLite weights from %s: %s", file, info)
    return info


def read_lllite_metadata(file: str) -> dict:
    if os.path.splitext(file)[1] != ".safetensors":
        return {}
    from safetensors import safe_open
    with safe_open(file, framework="pt") as f:
        meta = f.metadata()
    return meta or {}
