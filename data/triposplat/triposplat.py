"""Validated, ComfyUI-aware TripoSplat inference runtime."""

import io
import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from PIL import Image, ImageFilter

from .model import (
    DinoV3ViT, Flux2VAEEncoder,
    OctreeProbabilityFixedlenDecoder, ElasticGaussianFixedlenDecoder,
    LatentSeqMMFlowModel, OctreeGaussianDecoder,
)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

_SERIALIZE_CHUNK_SIZE = 65536
_QUATERNION_EPSILON = 1e-12


class GaussianValidationError(ValueError):
    """Raised when decoded Gaussian attributes cannot be serialized safely."""


class Gaussian:
    def __init__(self, aabb: list, sh_degree: int = 0, mininum_kernel_size: float = 0.0,
                 scaling_bias: float = 0.01, opacity_bias: float = 0.1,
                 scaling_activation: str = "exp", device='cuda'):
        self.sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

        if scaling_activation == "exp":
            self._scaling_activation = torch.exp
            self._inverse_scaling_activation = torch.log
        elif scaling_activation == "softplus":
            self._scaling_activation = F.softplus
            self._inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self._opacity_activation = torch.sigmoid
        self._inverse_opacity_activation = lambda x: torch.log(x / (1 - x))

        self.scale_bias = self._inverse_scaling_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros(4, device=self.device)
        self.rots_bias[0] = 1
        self.opacity_bias_val = self._inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)

        self._storage = {}

    def _get_store(self, name):
        return self._storage.get(name)

    def _set_store(self, name, value):
        self._storage[name] = value

    @property
    def _xyz(self):
        return self._get_store("_xyz")
    @_xyz.setter
    def _xyz(self, value):
        if value is None:
            self._set_store("_xyz", None); self._set_store("xyz", None); return
        self._set_store("_xyz", value)
        self._set_store("xyz", value * self.aabb[None, 3:] + self.aabb[None, :3])

    @property
    def get_xyz(self):
        return self._get_store("xyz")

    @property
    def _features_dc(self):
        return self._get_store("_features_dc")
    @_features_dc.setter
    def _features_dc(self, value):
        self._set_store("_features_dc", value)

    @property
    def _opacity(self):
        return self._get_store("_opacity")
    @_opacity.setter
    def _opacity(self, value):
        if value is None:
            self._set_store("_opacity", None); self._set_store("opacity", None); return
        self._set_store("_opacity", value)
        self._set_store("opacity", self._opacity_activation(value + self.opacity_bias_val))

    @property
    def get_opacity(self):
        return self._get_store("opacity")

    @property
    def _scaling(self):
        return self._get_store("_scaling")
    @_scaling.setter
    def _scaling(self, value):
        if value is None:
            self._set_store("_scaling", None); self._set_store("scaling", None); return
        self._set_store("_scaling", value)
        s = self._scaling_activation(value + self.scale_bias)
        s = torch.square(s) + self.mininum_kernel_size ** 2
        self._set_store("scaling", torch.sqrt(s))

    @property
    def get_scaling(self):
        return self._get_store("scaling")

    @property
    def _rotation(self):
        return self._get_store("_rotation")
    @_rotation.setter
    def _rotation(self, value):
        self._set_store("_rotation", value)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        dc = self._features_dc
        for i in range(dc.shape[1] * dc.shape[2]):
            l.append(f'f_dc_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def validation_report(self) -> dict:
        tensors = {
            "xyz": self.get_xyz,
            "features_dc": self._features_dc,
            "opacity_logits": self._opacity,
            "opacity": self.get_opacity,
            "scaling_logits": self._scaling,
            "scaling": self.get_scaling,
            "rotation": self._rotation,
        }
        errors = []
        counts = {}
        expected = None
        ranges = {}
        for name, value in tensors.items():
            if value is None or not isinstance(value, torch.Tensor):
                errors.append(f"{name} is missing")
                continue
            count = int(value.shape[0]) if value.ndim else 0
            counts[name] = count
            if expected is None:
                expected = count
            elif count != expected:
                errors.append(f"{name} has {count:,} rows; expected {expected:,}")
            finite = torch.isfinite(value)
            invalid = int((~finite).sum().item())
            if invalid:
                errors.append(f"{name} contains {invalid:,} NaN/Inf value(s)")
            elif value.numel():
                as_float = value.detach().float()
                ranges[name] = [float(as_float.min().item()), float(as_float.max().item())]

        if not expected:
            errors.append("Gaussian contains no splats")
        if self.get_scaling is not None:
            non_positive = int((self.get_scaling <= 0).sum().item())
            if non_positive:
                errors.append(f"scaling contains {non_positive:,} non-positive value(s)")
        if self._rotation is not None:
            rotation = self._rotation.detach().float() + self.rots_bias.detach().float()[None, :]
            norms = torch.linalg.vector_norm(rotation, dim=-1)
            invalid_norms = int(((~torch.isfinite(norms)) | (norms <= _QUATERNION_EPSILON)).sum().item())
            if invalid_norms:
                errors.append(f"rotation contains {invalid_norms:,} invalid quaternion(s)")

        return {
            "valid": not errors,
            "gaussians": int(expected or 0),
            "errors": errors,
            "ranges": ranges,
            "counts": counts,
        }

    def validate(self) -> dict:
        report = self.validation_report()
        if not report["valid"]:
            raise GaussianValidationError("invalid Gaussian output: " + "; ".join(report["errors"]))
        return report

    _DEFAULT_TRANSFORM = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    def _transform(self, transform=None):
        if transform is None:
            transform = self._DEFAULT_TRANSFORM
        transform = np.array(transform, dtype=np.float32)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise GaussianValidationError("Gaussian transform must be a finite 3×3 matrix")
        if not np.allclose(transform @ transform.T, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(transform),
            1.0,
            atol=1e-4,
        ):
            raise GaussianValidationError(
                "Gaussian export transform must be a proper rotation matrix; "
                "apply position/scale through the scene exporter"
            )
        return transform

    def _transformed_xyz_rot(self, xyz, rotation, transform):
        xyz = np.matmul(xyz, transform.T)
        R_mat = _quat_to_matrix(rotation)
        R_mat = np.matmul(transform, R_mat)
        rotation = _matrix_to_quat(R_mat)
        return xyz, rotation

    def _iter_ply_payload(
        self,
        transform=None,
        callback=None,
        chunk_size=_SERIALIZE_CHUNK_SIZE,
        _validated_report=None,
    ):
        report = _validated_report or self.validate()
        transform = self._transform(transform)
        names = self.construct_list_of_attributes()
        dtype_full = [(name, "<f4") for name in names]
        count = report["gaussians"]
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            xyz = self.get_xyz[start:end].detach().float().cpu().numpy()
            rotation = (
                self._rotation[start:end].detach().float()
                + self.rots_bias.detach().float()[None, :]
            ).cpu().numpy()
            xyz, rotation = self._transformed_xyz_rot(xyz, rotation, transform)
            arrays = (
                xyz,
                np.zeros_like(xyz),
                self._features_dc[start:end]
                .detach()
                .float()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy(),
                (
                    self._opacity[start:end].detach().float()
                    + self.opacity_bias_val.detach().float()
                ).cpu().numpy(),
                torch.log(self.get_scaling[start:end].detach().float()).cpu().numpy(),
                rotation,
            )
            elements = np.empty(end - start, dtype=np.dtype(dtype_full))
            column = 0
            for array in arrays:
                array = np.asarray(array, dtype=np.float32).reshape(end - start, -1)
                for index in range(array.shape[1]):
                    elements[names[column]] = array[:, index]
                    column += 1
            yield elements.tobytes()
            if callback is not None:
                callback(end, count)

    def to_ply_bytes(self, transform=None, callback=None) -> bytes:
        report = self.validate()
        count = report["gaussians"]
        output = io.BytesIO()
        output.write(_binary_ply_header(count, self.construct_list_of_attributes()))
        for payload in self._iter_ply_payload(
            transform=transform,
            callback=callback,
            _validated_report=report,
        ):
            output.write(payload)
        return output.getvalue()

    def _splat_arrays(self, transform=None, _validated_report=None):
        if _validated_report is None:
            self.validate()
        transform = self._transform(transform)
        xyz = self.get_xyz.detach().float().cpu().numpy()
        rotation = (
            self._rotation.detach().float() + self.rots_bias.detach().float()[None, :]
        ).cpu().numpy()
        xyz, rotation = self._transformed_xyz_rot(xyz, rotation, transform)
        scale = self.get_scaling.detach().float().cpu().numpy()
        opacity = self.get_opacity.detach().float().cpu().numpy()
        f_dc = self._features_dc.detach().float().cpu().numpy()
        rotation = _normalize_quaternions(rotation, "SPLAT rotation")
        order = np.argsort(-opacity[:, 0] * np.prod(scale, axis=-1), kind="stable")
        return xyz, rotation, scale, opacity, f_dc, order

    def _iter_splat_payload(
        self,
        transform=None,
        callback=None,
        chunk_size=_SERIALIZE_CHUNK_SIZE,
        _validated_report=None,
    ):
        xyz, rotation, scale, opacity, f_dc, order = self._splat_arrays(
            transform=transform,
            _validated_report=_validated_report,
        )
        count = len(order)
        C0 = 0.28209479177387814
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            indices = order[start:end]
            chunk_xyz = xyz[indices].astype("<f4", copy=False)
            chunk_scale = scale[indices].astype("<f4", copy=False)
            rgb = np.clip((f_dc[indices, 0, :] * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
            alpha = np.clip(opacity[indices, 0:1] * 255, 0, 255).astype(np.uint8)
            rgba = np.concatenate([rgb, alpha], axis=1)
            rot_u8 = np.clip(rotation[indices] * 128 + 128, 0, 255).astype(np.uint8)
            packed = np.empty((end - start, 32), dtype=np.uint8)
            packed[:, 0:12] = chunk_xyz.view(np.uint8).reshape(-1, 12)
            packed[:, 12:24] = chunk_scale.view(np.uint8).reshape(-1, 12)
            packed[:, 24:28] = rgba
            packed[:, 28:32] = rot_u8
            yield packed.tobytes()
            if callback is not None:
                callback(end, count)

    def to_splat_bytes(self, transform=None, callback=None) -> bytes:
        output = io.BytesIO()
        for payload in self._iter_splat_payload(transform=transform, callback=callback):
            output.write(payload)
        return output.getvalue()

    def save_ply(self, path, transform=None, callback=None, _validated_report=None):
        report = _validated_report or self.validate()
        count = report["gaussians"]
        with open(path, 'wb') as handle:
            handle.write(_binary_ply_header(count, self.construct_list_of_attributes()))
            for payload in self._iter_ply_payload(
                transform=transform,
                callback=callback,
                _validated_report=report,
            ):
                handle.write(payload)

    def save_splat(self, path, transform=None, callback=None, _validated_report=None):
        with open(path, 'wb') as handle:
            for payload in self._iter_splat_payload(
                transform=transform,
                callback=callback,
                _validated_report=_validated_report,
            ):
                handle.write(payload)


def _binary_ply_header(num_vertices, attributes) -> bytes:
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {num_vertices}\n"
    for name in attributes:
        header += f"property float {name}\n"
    header += "end_header\n"
    return header.encode('ascii')


def _normalize_quaternions(q, label="quaternion"):
    q = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    invalid = (~np.isfinite(q).all(axis=-1)) | (~np.isfinite(norms[:, 0])) | (
        norms[:, 0] <= _QUATERNION_EPSILON
    )
    if invalid.any():
        raise GaussianValidationError(
            f"{label} contains {int(invalid.sum()):,} invalid quaternion(s)"
        )
    return q / norms


def _quat_to_matrix(q):
    q = _normalize_quaternions(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)
    return R


def _matrix_to_quat(R):
    R = np.asarray(R, dtype=np.float32)
    if R.ndim != 3 or R.shape[1:] != (3, 3) or not np.isfinite(R).all():
        raise GaussianValidationError("rotation matrix batch must contain finite 3×3 matrices")
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((R.shape[0], 4), dtype=R.dtype)
    positive = trace > 0
    s = np.sqrt(np.maximum(trace + 1, _QUATERNION_EPSILON)) * 2
    q[positive, 0] = 0.25 * s[positive]
    q[positive, 1] = (R[positive, 2, 1] - R[positive, 1, 2]) / s[positive]
    q[positive, 2] = (R[positive, 0, 2] - R[positive, 2, 0]) / s[positive]
    q[positive, 3] = (R[positive, 1, 0] - R[positive, 0, 1]) / s[positive]
    remaining = ~positive
    m01 = remaining & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
    s1 = np.sqrt(np.maximum(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], 0)) * 2
    s1 = np.maximum(s1, _QUATERNION_EPSILON)
    q[m01, 0] = (R[m01, 2, 1] - R[m01, 1, 2]) / s1[m01]
    q[m01, 1] = 0.25 * s1[m01]
    q[m01, 2] = (R[m01, 0, 1] + R[m01, 1, 0]) / s1[m01]
    q[m01, 3] = (R[m01, 0, 2] + R[m01, 2, 0]) / s1[m01]
    m11 = remaining & ~m01 & (R[:, 1, 1] >= R[:, 2, 2])
    s2 = np.sqrt(np.maximum(1 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], 0)) * 2
    s2 = np.maximum(s2, _QUATERNION_EPSILON)
    q[m11, 0] = (R[m11, 0, 2] - R[m11, 2, 0]) / s2[m11]
    q[m11, 1] = (R[m11, 0, 1] + R[m11, 1, 0]) / s2[m11]
    q[m11, 2] = 0.25 * s2[m11]
    q[m11, 3] = (R[m11, 1, 2] + R[m11, 2, 1]) / s2[m11]
    m21 = remaining & ~m01 & ~m11
    s3 = np.sqrt(np.maximum(1 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], 0)) * 2
    s3 = np.maximum(s3, _QUATERNION_EPSILON)
    q[m21, 0] = (R[m21, 1, 0] - R[m21, 0, 1]) / s3[m21]
    q[m21, 1] = (R[m21, 0, 2] + R[m21, 2, 0]) / s3[m21]
    q[m21, 2] = (R[m21, 1, 2] + R[m21, 2, 1]) / s3[m21]
    q[m21, 3] = 0.25 * s3[m21]
    return _normalize_quaternions(q, "matrix-derived rotation")


def _build_gaussians(decoder: ElasticGaussianFixedlenDecoder, points_pred: dict, pred: dict):
    x = points_pred
    offset = decoder._get_offset(pred['features'])
    h = pred["features"]
    ret = []
    for i in range(h.shape[0]):
        g = Gaussian(
            sh_degree=0,
            aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
            mininum_kernel_size=decoder.rep_config['filter_kernel_size_3d'],
            scaling_bias=decoder.rep_config['scaling_bias'],
            opacity_bias=decoder.rep_config['opacity_bias'],
            scaling_activation=decoder.rep_config['scaling_activation'],
            device=x['points'].device,
        )
        _x = x["points"][i, :, None, :]
        for k, v in decoder.layout.items():
            if k == '_xyz':
                setattr(g, k, (offset[i] + _x).flatten(0, 1))
            elif k in ('_xyz_center', '_offset_scale'):
                continue
            else:
                feats = h[i][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                setattr(g, k, feats * decoder.rep_config['lr'][k])
        ret.append(g)
    return ret


# ---------------------------------------------------------------------------
# Euler flow sampler
# ---------------------------------------------------------------------------

def _ensure_finite_tensor(label: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value)
    if bool(finite.all().item()):
        return
    invalid = int((~finite).sum().item())
    raise FloatingPointError(f"{label} contains {invalid:,} NaN/Inf value(s)")


def _ensure_finite_mapping(label: str, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            _ensure_finite_tensor(f"{label}.{key}", value)


class FlowEulerCfgSampler:
    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def _get_batch_size(self, x_t):
        return next(iter(x_t.values())).shape[0] if isinstance(x_t, dict) else x_t.shape[0]

    def _get_device(self, x_t):
        return next(iter(x_t.values())).device if isinstance(x_t, dict) else x_t.device

    def _inference_model(self, model, x_t, t, cond=None):
        batch = self._get_batch_size(x_t)
        device = self._get_device(x_t)
        t_scaled = torch.tensor([1000 * t] * batch, device=device, dtype=torch.float32)
        if isinstance(cond, dict):
            for k, v in cond.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 1 and batch > 1:
                    cond[k] = v.repeat(batch, *([1] * (len(v.shape) - 1)))
        elif cond is not None and cond.shape[0] == 1 and batch > 1:
            cond = cond.repeat(batch, *([1] * (len(cond.shape) - 1)))
        return model(x_t, t_scaled, cond)

    def _cfg_prediction(self, model, x_t, t, cond, neg_cond, guidance_scale):
        # Diffusers-style convention: guidance_scale == 1 (or <= 1, or None) means no CFG —
        # only the conditional pass runs, halving the per-step cost. > 1 enables CFG and
        # blends as `pred = s * cond + (1 - s) * uncond = s * cond - (s - 1) * uncond`.
        pred_v = self._inference_model(model, x_t, t, cond)
        if isinstance(guidance_scale, dict):
            if not any(s > 1 for s in guidance_scale.values()):
                return pred_v
            neg_pred_v = self._inference_model(model, x_t, t, neg_cond)
            for key in pred_v:
                s = guidance_scale.get(key, 1.0)
                if s > 1:
                    pred_v[key] = s * pred_v[key] - (s - 1) * neg_pred_v[key]
            return pred_v
        if guidance_scale is None or guidance_scale <= 1:
            return pred_v
        neg_pred_v = self._inference_model(model, x_t, t, neg_cond)
        for key in pred_v:
            pred_v[key] = guidance_scale * pred_v[key] - (guidance_scale - 1) * neg_pred_v[key]
        return pred_v

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps=50, shift=1.0,
               guidance_scale=None, show_progress=False, callback=None):
        sample = noise
        t_seq = shift * np.linspace(1, 0, steps + 1) / (1 + (shift - 1) * np.linspace(1, 0, steps + 1))
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))
        iterator = t_pairs
        for i, (t, t_prev) in enumerate(iterator):
            x_t = {k: v.clone() for k, v in sample.items()} if isinstance(sample, dict) else sample.clone()
            pred_v = self._cfg_prediction(model, x_t, t, cond, neg_cond, guidance_scale)
            dt = t - t_prev
            if isinstance(sample, dict):
                for key in sample:
                    sample[key] = sample[key] - pred_v[key] * dt
            else:
                sample = sample - pred_v * dt
            if isinstance(sample, dict):
                _ensure_finite_mapping(f"flow sample step {i + 1}", sample)
            else:
                _ensure_finite_tensor(f"flow sample step {i + 1}", sample)
            if callback is not None:
                callback(i + 1, steps)
        return sample


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------

def _place(m, device, dtype):
    if device is not None or dtype is not None:
        m = m.to(device=device, dtype=dtype)
    return m.eval()


def load_dinov3(path: str, device=None, dtype=None) -> DinoV3ViT:
    m = DinoV3ViT()
    m.load_safetensors(path)
    return _place(m, device, dtype)


def load_vae_encoder(path: str, device=None, dtype=None) -> Flux2VAEEncoder:
    m = Flux2VAEEncoder()
    m.load_safetensors(path)
    return _place(m, device, dtype)


FLOW_MODEL_ARGS = dict(
    q_token_length=8192, in_channels=16, cam_channels=5, out_channels=16,
    model_channels=1024, cond_channels=1280, cond2_channels=128,
    num_refiner_blocks=2, num_blocks=24, num_heads=16, mlp_ratio=4,
    qk_rms_norm=True, share_mod=True, use_shift_table=True,
)


def load_flow_model(path: str, device=None, dtype=None) -> LatentSeqMMFlowModel:
    m = LatentSeqMMFlowModel(**FLOW_MODEL_ARGS)
    m.load_safetensors(path)
    return _place(m, device, dtype)


OCTREE_DECODER_ARGS = dict(
    model_channels=1024, cond_channels=16,
    num_blocks=4, num_heads=16, mlp_ratio=4, share_mod=True,
)

GS_DECODER_ARGS = dict(
    in_channels=3, model_channels=1024, cond_channels=16,
    attn_mode="full", num_blocks=16, num_heads=16, mlp_ratio=4,
    use_learned_offset_scale=True, use_per_offset=True,
    representation_config=dict(
        lr=dict(_xyz=1.0, _features_dc=1.0, _opacity=1.0, _scaling=1.0, _rotation=0.1),
        perturb_offset=True, perturbe_size=1.5, offset_scale=0.05, num_gaussians=32,
        filter_kernel_size_3d=0.0009, scaling_bias=0.004, opacity_bias=0.1,
        scaling_activation="softplus",
    ),
)


def load_decoder(path: str, device=None, dtype=None) -> OctreeGaussianDecoder:
    m = OctreeGaussianDecoder(OCTREE_DECODER_ARGS, GS_DECODER_ARGS)
    m.load_safetensors(path)
    return _place(m, device, dtype)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

_CANVAS_SIZE = 1024
_IMAGE_PATCH_SIZE = 16
_MAX_PREPROCESS_PIXELS = 4096 * 4096
_MAX_PREPROCESS_SIDE = 16384
_ALPHA_BBOX_THRESHOLD = 8


def _image_to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        return Image.open(image)
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            assert t.shape[0] == 1, (
                f"batched image input is not supported (got B={t.shape[0]}); "
                "pass one image at a time"
            )
            t = t[0]
        arr = (t.clamp(0, 1) * 255).to(torch.uint8).numpy()
        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
        return Image.fromarray(arr, mode=mode)
    raise TypeError(f"unsupported image type: {type(image)}")


def _conditioning_canvas_size(
    image_size: tuple[int, int],
    requested_size: int,
    prevent_upscale: bool,
) -> int:
    size = int(requested_size)
    if size < _IMAGE_PATCH_SIZE or size % _IMAGE_PATCH_SIZE:
        raise ValueError(
            f"conditioning resolution must be at least {_IMAGE_PATCH_SIZE} "
            f"and divisible by {_IMAGE_PATCH_SIZE}"
        )
    if not prevent_upscale:
        return size
    native_short_side = min(int(image_size[0]), int(image_size[1]))
    if native_short_side >= size:
        return size
    # Both conditioning encoders produce one aligned token per 16×16 input
    # patch. Keep the native short-side resolution without inventing pixels,
    # while preserving an exact shared token grid.
    return max(
        _IMAGE_PATCH_SIZE,
        (native_short_side // _IMAGE_PATCH_SIZE) * _IMAGE_PATCH_SIZE,
    )


def _safe_preprocess_scale(width: int, height: int, target_short_side: int) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("input image dimensions must be positive")
    requested = target_short_side / min(width, height)
    side_cap = _MAX_PREPROCESS_SIDE / max(width, height)
    pixel_cap = (_MAX_PREPROCESS_PIXELS / float(width * height)) ** 0.5
    return min(requested, side_cap, pixel_cap)


def _has_usable_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image.getchannel(3), dtype=np.uint8)
    transparent = int((alpha < 250).sum())
    minimum = max(8, int(alpha.size * 0.001))
    return transparent >= minimum


def _foreground_bbox(alpha: np.ndarray, threshold: int = _ALPHA_BBOX_THRESHOLD) -> list[int]:
    if alpha.ndim != 2:
        raise ValueError("foreground alpha mask must be two-dimensional")
    mask = alpha >= int(threshold)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError(
            "TripoSplat preprocessing produced an empty foreground mask. "
            "Use a clearer silhouette or provide a valid alpha channel."
        )
    # PIL's right/lower crop bounds are exclusive.
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def preprocess_image(
    image,
    rmbg=None,
    erode_radius: int = 0,
    canvas_size: int = _CANVAS_SIZE,
    prevent_upscale: bool = False,
    remove_background: bool = True,
) -> Image.Image:
    image = _image_to_pil(image)
    size = _conditioning_canvas_size(
        image.size,
        requested_size=canvas_size,
        prevent_upscale=prevent_upscale,
    )
    w, h = image.size
    s = _safe_preprocess_scale(w, h, size)
    image = image.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
    if remove_background:
        has_real_alpha = _has_usable_alpha(image)
        if not has_real_alpha:
            if rmbg is None:
                raise ValueError("TripoSplat background removal requires a loaded BiRefNet model")
            image = rmbg.remove_background(image.convert("RGB"))
    else:
        # Disabling background removal only bypasses BiRefNet. Preserve an
        # existing alpha channel exactly; RGB inputs naturally receive a fully
        # opaque alpha channel when converted to RGBA.
        image = image.convert("RGBA")
    if erode_radius < 0:
        raise ValueError("mask erosion radius cannot be negative")
    if erode_radius > 0:
        image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(2 * erode_radius + 1)))
    alpha = np.array(image.getchannel(3))
    bbox = _foreground_bbox(alpha)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    half = max(1.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * 1.2)
    crop = [
        int(np.floor(cx - half)),
        int(np.floor(cy - half)),
        int(np.ceil(cx + half)),
        int(np.ceil(cy + half)),
    ]
    image = image.crop(crop)
    image = image.resize((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    bg.paste(image, mask=image.split()[3])
    return bg





@torch.no_grad()
def encode_image(image: Image.Image, dinov3: DinoV3ViT, vae_encoder: Flux2VAEEncoder,
                 generator: torch.Generator = None) -> dict:
    device = next(dinov3.parameters()).device
    img_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = img_tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = img_tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_normed = (img_tensor - mean) / std
    dinov3_dtype = next(dinov3.parameters()).dtype
    vae_dtype    = next(vae_encoder.parameters()).dtype
    dinov3_feat = dinov3(pixel_values=img_normed.to(dinov3_dtype))
    dinov3_feat = F.layer_norm(dinov3_feat.float(), dinov3_feat.shape[-1:])
    vae_feat = vae_encoder.encode(img_tensor.to(vae_dtype) * 2 - 1,
                                  deterministic=False, generator=generator)
    # pad 5 zero tokens so feature2's token length matches feature1's (cls + 4 registers + patches)
    zero_reg = torch.zeros(vae_feat.shape[0], 5, vae_feat.shape[2],
                           dtype=vae_feat.dtype, device=vae_feat.device)
    vae_feat = torch.cat([zero_reg, vae_feat], dim=1)
    if dinov3_feat.shape[1] != vae_feat.shape[1]:
        raise RuntimeError(
            "conditioning encoder token grids do not match: "
            f"DINO={dinov3_feat.shape[1]}, Flux VAE={vae_feat.shape[1]}"
        )
    result = {'feature1': dinov3_feat, 'feature2': vae_feat}
    _ensure_finite_mapping("image conditioning", result)
    return result


@torch.no_grad()
def sample_latent(flow_model: LatentSeqMMFlowModel, cond: dict,
                  steps: int = 50, guidance_scale: float = 7.0, shift: float = 3.0,
                  generator: torch.Generator = None,
                  show_progress: bool = False, callback=None) -> dict:
    device = flow_model.device
    if isinstance(guidance_scale, dict):
        cfg_enabled = any(float(value) > 1 for value in guidance_scale.values())
    else:
        cfg_enabled = guidance_scale is not None and float(guidance_scale) > 1
    prepared_cond = {"prepared_context": flow_model.prepare_condition(cond)}
    _ensure_finite_mapping("prepared conditional context", prepared_cond)
    prepared_neg_cond = None
    if cfg_enabled:
        neg_cond = {k: torch.zeros_like(v) for k, v in cond.items()}
        prepared_neg_cond = {"prepared_context": flow_model.prepare_condition(neg_cond)}
        _ensure_finite_mapping("prepared negative context", prepared_neg_cond)
    noise = {'latent': torch.randn(1, flow_model.q_token_length, flow_model.in_channels,
                                   device=device, generator=generator)}
    if flow_model.cam_channels is not None:
        noise['camera'] = torch.randn(1, 1, flow_model.cam_channels,
                                      device=device, generator=generator)
    sampler = FlowEulerCfgSampler()
    result = sampler.sample(
        flow_model,
        noise,
        cond=prepared_cond,
        neg_cond=prepared_neg_cond,
        steps=steps,
        guidance_scale=guidance_scale,
        shift=shift,
        show_progress=show_progress,
        callback=callback,
    )
    _ensure_finite_mapping("flow result", result)
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def component_dtypes(device) -> dict[str, torch.dtype]:
    target = torch.device(device)
    if target.type == "cpu":
        encoder_dtype = model_dtype = torch.float32
    elif target.type == "mps":
        encoder_dtype = model_dtype = torch.float16
    else:
        supports_bfloat16 = False
        try:
            supports_bfloat16 = bool(torch.cuda.is_bf16_supported())
        except (AttributeError, RuntimeError):
            pass
        encoder_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
        model_dtype = torch.float16
    return {
        "dinov3": encoder_dtype,
        "vae_encoder": encoder_dtype,
        "rmbg": model_dtype,
        "flow_model": model_dtype,
        "decoder": model_dtype,
    }


class TripoSplatPipeline:
    _COMPONENT_NAMES = ("dinov3", "vae_encoder", "flow_model", "decoder")

    def __init__(self, ckpt_path: str, decoder_path: str, dinov3_path: str,
                 flux2_vae_encoder_path: str, device: str = "cuda",
                 load_device: str = None):
        self._device = torch.device(device)
        self._load_device = torch.device(load_device) if load_device is not None else self._device
        dtypes = component_dtypes(self._device)
        self.dinov3      = load_dinov3      (dinov3_path,             device=self._load_device, dtype=dtypes["dinov3"])
        self.vae_encoder = load_vae_encoder (flux2_vae_encoder_path,  device=self._load_device, dtype=dtypes["vae_encoder"])
        self.flow_model  = load_flow_model  (ckpt_path,               device=self._load_device, dtype=dtypes["flow_model"])
        self.decoder     = load_decoder     (decoder_path,            device=self._load_device, dtype=dtypes["decoder"])

    def _activate(self, *names: str) -> None:
        active = set(names)
        unknown = active.difference(self._COMPONENT_NAMES)
        if unknown:
            raise ValueError(f"unknown TripoSplat component(s): {', '.join(sorted(unknown))}")
        for name in self._COMPONENT_NAMES:
            module = getattr(self, name)
            target = self._device if name in active else self._load_device
            if next(module.parameters()).device != target:
                module.to(device=target)

    def _offload(self, *names: str) -> None:
        if self._load_device == self._device:
            return
        for name in names:
            getattr(self, name).to(device=self._load_device)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def preprocess_image(self, image, canvas_size=1024, prevent_upscale=False, remove_background=False):
        if remove_background:
            raise ValueError("Background Creator preserves the complete input image")
        return preprocess_image(image, canvas_size=canvas_size, prevent_upscale=prevent_upscale, remove_background=False)

    def encode_image(self, image: Image.Image, generator: torch.Generator = None) -> dict:
        self._activate("dinov3", "vae_encoder")
        try:
            return encode_image(image, self.dinov3, self.vae_encoder, generator=generator)
        finally:
            self._offload("dinov3", "vae_encoder")

    def sample_latent(self, cond: dict, steps: int = 50, guidance_scale: float = 7.0,
                      shift: float = 3.0, generator: torch.Generator = None,
                      show_progress: bool = False, callback=None) -> dict:
        self._activate("flow_model")
        try:
            return sample_latent(self.flow_model, cond, steps=steps, guidance_scale=guidance_scale,
                                 shift=shift, generator=generator,
                                 show_progress=show_progress, callback=callback)
        finally:
            self._offload("flow_model")

    def decode_latent(
        self,
        latent: torch.Tensor,
        num_gaussians: int = 262144,
        generator: torch.Generator = None,
        callback=None,
    ):
        count = self._validate_num_gaussians(int(num_gaussians))
        self._activate("decoder")
        try:
            result = self.decoder.decode(
                latent,
                num_gaussians=count,
                generator=generator,
                callback=callback,
            )
            result.last_validation_report = result.validate()
            return result
        finally:
            self._offload("decoder")

    _NUM_GAUSSIANS_MIN = 32768
    # 524K and 1.05M are experimental extensions above the upstream 262K
    # ceiling. The decoder is shape-dynamic, but 16K/32K decoder tokens
    # significantly increase full-attention VRAM and runtime requirements.
    _NUM_GAUSSIANS_MAX = 1048576

    def _validate_num_gaussians(self, n: int) -> int:
        if not self._NUM_GAUSSIANS_MIN <= n <= self._NUM_GAUSSIANS_MAX:
            raise ValueError(
                f"num_gaussians must be in "
                f"[{self._NUM_GAUSSIANS_MIN}, {self._NUM_GAUSSIANS_MAX}], got {n}"
            )
        gpp = self.decoder.gaussians_per_point
        if n % gpp == 0:
            return n
        rounded = ((n + gpp // 2) // gpp) * gpp
        if not self._NUM_GAUSSIANS_MIN <= rounded <= self._NUM_GAUSSIANS_MAX:
            raise ValueError(f"rounded num_gaussians={rounded} is outside the supported range")
        print(f"[TripoSplatPipeline] num_gaussians={n} is not a multiple of {gpp}; rounding to {rounded}")
        return rounded

    @torch.no_grad()
    def run(self, image, seed: int = 42, steps: int = 20, guidance_scale: float = 3.0,
            shift: float = 3.0, num_gaussians=262144, erode_radius: int = 0,
            show_progress: bool = False, callback=None,
            conditioning_resolution: int = _CANVAS_SIZE, prevent_upscale: bool = False,
            remove_background: bool = True):
        """
        Args:
            image: Input image. Accepts a file path / PIL.Image / torch.Tensor
                (`[1,H,W,C]` or `[H,W,C]`, float in `[0, 1]`, optional alpha
                channel as the 4th channel).
            seed: RNG seed for the VAE encoder's stochastic latent sampling and
                the initial flow-matching noise. Same seed → same output.
            steps: Number of Euler integrator steps in the flow-matching sampler.
                More steps → better fidelity, linear runtime cost.
                Recommend: 10~20.
            guidance_scale: Classifier-free-guidance strength (diffusers
                convention). `≤ 1.0` disables CFG. Higher → more detail,
                stronger adherence to the input image; too high can cause color
                oversaturation.
                Recommend: 3.0.
            shift: Flow-matching timestep schedule shift. `1.0` gives a uniform
                schedule; `>1.0` allocates more steps to the early/high-noise end.
                Recommend: 3.0.
            num_gaussians: Target Gaussian-splat count. An `int` returns a
                single `Gaussian`. A `list` / `tuple` of ints returns a
                `list[Gaussian]`. Each count is rounded to the nearest multiple
                of 32. More gaussians → more detail but higher rendering and
                storage cost.
                Recommend: 32768~262144. 524288 and 1048576 are experimental
                and require substantially more decoder VRAM and runtime.
            erode_radius: Pixel radius used to erode the alpha matte after
                background removal, to avoid segmentation-border bleed before
                compositing on black. `0` disables; `1` is a 3×3 minimum filter.
                Recommend: 0 for thin silhouettes; use 1 only to suppress a
                visible segmentation fringe.
            conditioning_resolution: Square image resolution consumed by DINOv3
                and the Flux VAE. The released model uses 1024. Higher values are
                experimental and must be divisible by 16.
            prevent_upscale: If true, cap the conditioning canvas to the source
                image's native short side, rounded down to a 16-pixel patch grid.
            remove_background: If true, isolate the subject with an existing
                alpha channel or BiRefNet. If false, preserve the complete
                source frame as opaque conditioning evidence.
            show_progress: Print a `tqdm` progress bar over sampler steps.
            callback: Optional `fn(step, total)` invoked after each sampler step.
                Useful for external progress UIs (e.g. ComfyUI's
                `ProgressBar.update`).

        Returns:
            `(gaussian, prepared_image)` for an `int` `num_gaussians`, or
            `(list_of_gaussians, prepared_image)` for a `list` / `tuple`. The
            second element is the RGB composite the encoders actually saw —
            useful for display / debugging.
        """
        if isinstance(num_gaussians, (list, tuple)):
            counts = [self._validate_num_gaussians(n) for n in num_gaussians]
        else:
            counts = [self._validate_num_gaussians(num_gaussians)]

        gen = torch.Generator(device=self._device).manual_seed(seed)
        prepared = self.preprocess_image(
            image,
            erode_radius=erode_radius,
            canvas_size=conditioning_resolution,
            prevent_upscale=prevent_upscale,
            remove_background=remove_background,
        )
        cond = self.encode_image(prepared, generator=gen)
        out = self.sample_latent(cond, steps=steps, guidance_scale=guidance_scale, shift=shift,
                                 generator=gen, show_progress=show_progress, callback=callback)
        gaussians = [
            self.decode_latent(out['latent'], num_gaussians=n, generator=gen)
            for n in counts
        ]
        if isinstance(num_gaussians, (list, tuple)):
            return gaussians, prepared
        return gaussians[0], prepared
