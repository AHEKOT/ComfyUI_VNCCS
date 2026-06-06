import json
import os
import io
import base64
import queue
import threading
import time
import urllib.parse
import inspect
import ipaddress
import sys

import folder_paths
import comfy.sd
import comfy.utils
import requests
import server
from aiohttp import web
from huggingface_hub import hf_hub_download, hf_hub_url
from PIL import Image
import numpy as np
import traceback
import struct

try:
    from ..utils import is_absolute_path_any_os, normalize_filesystem_path, validate_privileged_request
except Exception:
    from utils import is_absolute_path_any_os, normalize_filesystem_path, validate_privileged_request

try:
    from .model_path_utils import basename_agnostic, get_full_path_agnostic
except Exception:
    from model_path_utils import basename_agnostic, get_full_path_agnostic


class AnyType(str):
    def __ne__(self, other):
        return False


any_type = AnyType("*")


class _TautologyStr(str):
    def __ne__(self, other):
        return False


class _ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return _TautologyStr(item)
        return item


_CC_CONFIG_CACHE = {}
_DOWNLOAD_STATUS = {}
_DOWNLOAD_QUEUE = queue.Queue()
_CUSTOM_LORAS_FILE = "vnccs_custom_loras.json"
_PACKAGED_CC_REPO_IDS = {"MIUProject/VNCCS_v3.0"}
_PIPELINE_LOCAL_LORAS = {"vnccs pose studio qie2511"}
_FOLDER_MAP = {
    "unet": ["unet", "diffusion_models"],
    "checkpoints": ["checkpoints"],
    "loras": ["loras"],
    "clip": ["clip"],
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "controlnet": ["controlnet"],
    "upscale_models": ["upscale_models"],
    "embeddings": ["embeddings"],
    "gguf": ["unet", "diffusion_models"],
    "diffusion_models": ["diffusion_models", "unet"],
}
_MODEL_FILE_EXTENSIONS = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin"}
_MIN_MODEL_FILE_SIZE = 1024
_DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024 * 1024
_DOWNLOAD_TIMEOUT = (10, 60)
DEFAULT_MODEL_STEPS = 4
DEFAULT_MODEL_CFG = 1.0
NUNCHAKU_DISABLED_MESSAGE = (
    "Nunchaku support is disabled in VNCCS. Use GGUF models instead."
)

def _get_node_class_mappings():
    import nodes as comfy_nodes

    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})


def _detect_nunchaku_model_kind(model_entry=None, full_path=""):
    # TECH DEBT: Nunchaku support is disabled. Delete this legacy helper after
    # old saved workflows and tests no longer carry Nunchaku state.
    entry_name = (model_entry.get("name", "") if model_entry else "").lower()
    entry_path = (model_entry.get("local_path", "") if model_entry else "").lower()
    identity = " ".join([entry_name, entry_path, full_path.lower()])
    return "qwen-image" if "qwen" in identity else "flux"


def _get_nunchaku_dependency_status(model_entry=None, full_path="", has_enabled_loras=False):
    # TECH DEBT: Nunchaku dependency probing is intentionally commented out at
    # runtime. Delete the legacy Nunchaku support once old workflow state is gone.
    # Legacy behavior checked NODE_CLASS_MAPPINGS for:
    # - NunchakuQwenImageDiTLoader / NunchakuFluxDiTLoader
    # - NunchakuQwenImageLoraLoader / NunchakuFluxLoraLoader
    return {
        "model_kind": _detect_nunchaku_model_kind(model_entry=model_entry, full_path=full_path),
        "loader_name": None,
        "lora_loader_name": None,
        "has_loader": False,
        "has_lora_loader": False,
        "has_enabled_loras": has_enabled_loras,
        "ok": False,
        "disabled": True,
        "missing": [],
        "message": NUNCHAKU_DISABLED_MESSAGE,
    }


def resolve_path(relative_path):
    if not relative_path:
        return ""
    expanded = os.path.expanduser(relative_path)
    normalized = normalize_filesystem_path(expanded)
    if is_absolute_path_any_os(expanded):
        return os.path.abspath(normalized)
    base = getattr(folder_paths, "base_path", os.getcwd())
    return os.path.abspath(os.path.join(base, normalized))


def _models_root():
    return os.path.abspath(getattr(folder_paths, "models_dir", os.path.join(getattr(folder_paths, "base_path", os.getcwd()), "models")))


def _validate_https_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("Download URL must be an absolute HTTPS URL")
    host = parsed.hostname or ""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        ip = None
    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved):
        raise ValueError("Download URL host must not be a private or local address")
    return url


def _max_download_bytes():
    raw = os.environ.get("VNCCS_MAX_MODEL_DOWNLOAD_BYTES", "")
    try:
        value = int(raw)
        return value if value > 0 else _DEFAULT_MAX_DOWNLOAD_BYTES
    except Exception:
        return _DEFAULT_MAX_DOWNLOAD_BYTES


def _validate_download_response(response, expected_name="model"):
    _validate_https_url(getattr(response, "url", ""))
    total_size = int(response.headers.get("content-length", 0) or 0)
    max_bytes = _max_download_bytes()
    if total_size > max_bytes:
        raise ValueError(
            f"{expected_name} is too large to download safely "
            f"({total_size / (1024 * 1024 * 1024):.1f} GB, limit {max_bytes / (1024 * 1024 * 1024):.1f} GB)"
        )
    return total_size, max_bytes


def _validate_model_filename(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MODEL_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(_MODEL_FILE_EXTENSIONS))
        raise ValueError(f"Unsupported model file extension '{ext}'. Allowed: {allowed}")
    return ext


def _resolve_model_download_path(local_path):
    if not local_path:
        raise ValueError("Model local_path is required")
    normalized = str(local_path).strip().replace("\\", "/")
    if is_absolute_path_any_os(normalized):
        raise ValueError("Model local_path must be relative to ComfyUI models directory")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3 or parts[0] != "models":
        raise ValueError("Model local_path must use 'models/<type>/<file>'")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Model local_path contains invalid path traversal")
    folder_type = parts[1]
    if folder_type not in _FOLDER_MAP:
        raise ValueError(f"Unsupported model folder '{folder_type}'")
    _validate_model_filename(parts[-1])
    target = os.path.abspath(os.path.join(_models_root(), *parts[1:]))
    if os.path.commonpath([_models_root(), target]) != _models_root():
        raise ValueError("Model local_path escapes ComfyUI models directory")
    return target


def _validate_downloaded_model_file(path, expected_name="model"):
    size = os.path.getsize(path)
    if size < _MIN_MODEL_FILE_SIZE:
        raise ValueError(f"{expected_name} is too small to be a valid model file ({size} bytes)")

    ext = _validate_model_filename(expected_name)
    with open(path, "rb") as handle:
        head = handle.read(256)

    if head.lstrip().lower().startswith((b"<html", b"<!doctype html", b"<?xml")):
        raise ValueError(f"{expected_name} looks like an HTML/XML response, not a model file")

    if ext == ".gguf":
        if head[:4] != b"GGUF":
            raise ValueError(f"{expected_name} is not a valid GGUF file")
        return True

    if ext == ".safetensors":
        if len(head) < 9:
            raise ValueError(f"{expected_name} is not a valid safetensors file")
        header_len = struct.unpack("<Q", head[:8])[0]
        if header_len <= 0 or header_len > size - 8 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"{expected_name} has an invalid safetensors header")
        with open(path, "rb") as handle:
            handle.seek(8)
            first_header_byte = handle.read(1)
        if first_header_byte != b"{":
            raise ValueError(f"{expected_name} has an invalid safetensors JSON header")
        return True

    if ext in {".ckpt", ".pt", ".pth", ".bin"}:
        if not (head.startswith(b"PK\x03\x04") or head[:1] == b"\x80"):
            raise ValueError(f"{expected_name} does not look like a PyTorch model checkpoint")
        return True

    return True


def get_vnccs_config():
    config_path = resolve_path("vnccs_user_config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_vnccs_config(new_data):
    config_path = resolve_path("vnccs_user_config.json")
    data = get_vnccs_config()
    data.update(new_data)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _get_custom_loras_path():
    return resolve_path(_CUSTOM_LORAS_FILE)


def _load_custom_loras():
    path = _get_custom_loras_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("lora", [])
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_custom_loras(entries):
    path = _get_custom_loras_path()
    current = _load_custom_loras()
    by_path = {
        entry.get("local_path", "").replace("\\", "/"): entry
        for entry in current
        if isinstance(entry, dict) and entry.get("local_path")
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        local_path = entry.get("local_path", "").replace("\\", "/")
        if not local_path:
            continue
        by_path[local_path] = entry

    payload = {"lora": list(by_path.values())}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _remove_custom_lora(local_path=None, name=None):
    normalized_path = (local_path or "").replace("\\", "/")
    normalized_name = (name or "").strip().lower()
    current = _load_custom_loras()
    kept = []
    removed = False

    for entry in current:
        if not isinstance(entry, dict):
            continue
        entry_path = entry.get("local_path", "").replace("\\", "/")
        entry_name = entry.get("name", "").strip().lower()
        if (normalized_path and entry_path == normalized_path) or (normalized_name and entry_name == normalized_name):
            removed = True
            continue
        kept.append(entry)

    payload = {"lora": kept}
    path = _get_custom_loras_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return removed


def _build_custom_lora_name(rel_path, used_names=None):
    used_names = used_names or set()
    normalized = rel_path.replace("\\", "/").strip("/")
    base_name = os.path.splitext(os.path.basename(normalized))[0]
    candidate = base_name
    parent = os.path.dirname(normalized)
    if candidate in used_names and parent:
        candidate = f"{base_name} ({parent})"
    if candidate in used_names:
        candidate = normalized

    suffix = 2
    unique = candidate
    while unique in used_names:
        unique = f"{candidate} [{suffix}]"
        suffix += 1
    return unique


def _build_custom_lora_entry(rel_path, used_names=None):
    normalized = rel_path.replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("LoRA path is empty")

    full_path = get_full_path_agnostic(folder_paths, "loras", normalized, require_exists=True)
    if not full_path or not os.path.exists(full_path):
        raise FileNotFoundError(f"LoRA '{normalized}' not found in ComfyUI loras folder")

    return {
        "name": _build_custom_lora_name(normalized, used_names=used_names),
        "local_path": f"models/loras/{normalized}",
        "type": "Custom",
        "kind": "Custom",
        "description": f"Custom LoRA from ComfyUI folder: {normalized}",
        "custom": True,
    }


def _merge_custom_loras(config):
    base_loras = list(config.get("lora", []))
    merged = dict(config)
    merged_loras = list(base_loras)
    existing_paths = {
        entry.get("local_path", "").replace("\\", "/")
        for entry in base_loras
        if isinstance(entry, dict) and entry.get("local_path")
    }

    for entry in _load_custom_loras():
        if not isinstance(entry, dict):
            continue
        local_path = entry.get("local_path", "").replace("\\", "/")
        if not local_path or local_path in existing_paths:
            continue
        merged_loras.append(entry)
        existing_paths.add(local_path)

    merged["lora"] = merged_loras
    return merged


def get_installed_version_info():
    registry_path = resolve_path("vnccs_installed_models.json")
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def update_installed_version(model_name, version):
    registry_path = resolve_path("vnccs_installed_models.json")
    data = get_installed_version_info()
    data[model_name] = version
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _get_packaged_cc_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "control_center.json"))


def _uses_packaged_cc_config(repo_id):
    return repo_id in _PACKAGED_CC_REPO_IDS and os.path.exists(_get_packaged_cc_path())


def _get_cc_config(repo_id):
    cached = _CC_CONFIG_CACHE.get(repo_id)
    now = time.time()
    if cached and now - cached.get("ts", 0) < 300:
        return _merge_custom_loras(cached["data"])

    if _uses_packaged_cc_config(repo_id):
        path = _get_packaged_cc_path()
    else:
        user_config = get_vnccs_config()
        hf_token = user_config.get("hf_token")
        path = hf_hub_download(
            repo_id=repo_id,
            filename="control_center.json",
            local_files_only=False,
            token=hf_token,
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    _CC_CONFIG_CACHE[repo_id] = {"ts": now, "data": data}
    return _merge_custom_loras(data)


def _find_entry(entries, name):
    if not name:
        return None
    for entry in entries:
        if entry.get("name", "").strip().lower() == name.strip().lower():
            return entry
    return None


def _find_first_entry_by_type(entries, entry_type):
    wanted = str(entry_type or "").strip().lower()
    for entry in entries or []:
        if _entry_type(entry) == wanted:
            return entry
    return None


def _selected_model_name_for_type(state, entry_type):
    selected_models = state.get("selected_models", {}) if isinstance(state, dict) else {}
    if isinstance(selected_models, dict):
        name = selected_models.get(entry_type)
        if name:
            return name
    return ""


def _custom_context_model_entry(config, state):
    models = config.get("models", []) if isinstance(config, dict) else []
    name = _selected_model_name_for_type(state, "gguf") or state.get("selected_model", "")
    return _find_entry(models, name) or _find_first_entry_by_type(models, "gguf")


def _rel_within_folder(local_path):
    parts = local_path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "models":
        return "/".join(parts[2:])
    return parts[-1]


def _find_model_on_disk(local_path):
    if not local_path:
        return "", False
    try:
        _validate_model_filename(local_path)
    except ValueError:
        return "", False

    parts = local_path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "models":
        folder_type = parts[1]
        rel_within = "/".join(parts[2:])
    elif len(parts) >= 2:
        folder_type = parts[-2]
        rel_within = parts[-1]
    else:
        folder_type = ""
        rel_within = local_path

    filename = parts[-1]
    keys = _FOLDER_MAP.get(folder_type, [folder_type] if folder_type else [])

    for key in keys:
        searches = [rel_within, filename] if rel_within != filename else [filename]
        for search in searches:
            found = get_full_path_agnostic(folder_paths, key, search, require_exists=True)
            if found:
                return found, True
            try:
                for folder in folder_paths.get_folder_paths(key) or []:
                    candidate = os.path.join(folder, search.replace("\\", os.sep).replace("/", os.sep))
                    if os.path.exists(candidate):
                        return candidate, True
            except Exception as exc:
                print(f"[VNCCS Control Center] folder_paths.get_folder_paths failed for {key}: {exc}")

    try:
        fallback = _resolve_model_download_path(local_path)
    except ValueError:
        return "", False
    return fallback, os.path.exists(fallback)


def _build_dynamic_paths(config, output_slot_names):
    all_entries = {}
    for entry in config.get("controlnet", []):
        all_entries[entry["name"]] = entry
    for entry in config.get("other", []):
        all_entries[entry["name"]] = entry

    paths = []
    for name in output_slot_names or []:
        if not name or name == "-":
            continue
        entry = all_entries.get(name)
        if not entry:
            paths.append("")
            continue
        _, exists = _find_model_on_disk(entry["local_path"])
        paths.append(_rel_within_folder(entry["local_path"]) if exists else "")
    return paths


def _normalize_meta_value(value):
    return str(value or "").strip().lower()


def _entry_kind(entry):
    return _normalize_meta_value((entry or {}).get("kind") or (entry or {}).get("Kind"))


def _entry_type(entry):
    return _normalize_meta_value((entry or {}).get("type") or (entry or {}).get("Type"))


def _lora_matches_model_kind(lora_entry, model_entry):
    if not lora_entry or not model_entry:
        return True
    lora_kind = _entry_kind(lora_entry)
    model_kind = _entry_kind(model_entry)
    return not lora_kind or not model_kind or lora_kind == model_kind


def _filter_entries_by_kind(entries, kind):
    normalized_kind = _normalize_meta_value(kind)
    entries = list(entries or [])
    if not normalized_kind:
        return entries
    matched = [entry for entry in entries if _entry_kind(entry) == normalized_kind]
    if matched:
        return matched
    generic = [entry for entry in entries if not _entry_kind(entry)]
    if generic:
        return generic
    available = sorted({_entry_kind(entry) for entry in entries if _entry_kind(entry)})
    available_text = ", ".join(available) if available else "none"
    raise RuntimeError(
        f"[VNCCS Control Center] No assets compatible with model kind '{normalized_kind}'. "
        f"Available asset kinds: {available_text}."
    )


def _load_checkpoint(full_path):
    output = comfy.sd.load_checkpoint_guess_config(
        full_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return output[0], output[1], output[2]


def _load_unet(full_path, settings=None):
    import torch

    model_options = {}
    weight_dtype = (settings or {}).get("weight_dtype", "default")
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2
    return comfy.sd.load_diffusion_model(full_path, model_options=model_options)


def _load_gguf(full_path):
    import nodes as comfy_nodes

    if "UnetLoaderGGUF" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError(
            "[VNCCS Control Center] ComfyUI-GGUF not found. Install github.com/city96/ComfyUI-GGUF"
        )

    loader = comfy_nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
    model, = loader.load_unet(basename_agnostic(full_path))
    return model


def _get_nunchaku_load_candidates(full_path):
    # TECH DEBT: Nunchaku loading is disabled. Keep this legacy helper only so
    # stale imports fail gracefully until the old code is deleted.
    normalized = os.path.abspath(full_path)
    candidates = []

    def add_candidate(path):
        if not path:
            return
        abs_path = os.path.abspath(path)
        if abs_path not in candidates:
            candidates.append(abs_path)

    if os.path.isdir(normalized):
        add_candidate(normalized)
        return candidates

    parent_dir = os.path.dirname(normalized)
    stem_dir = os.path.splitext(normalized)[0]

    for candidate_dir in (parent_dir, stem_dir):
        if not os.path.isdir(candidate_dir):
            continue
        if (
            os.path.exists(os.path.join(candidate_dir, "comfy_config.json"))
            or os.path.exists(os.path.join(candidate_dir, "config.json"))
        ):
            add_candidate(candidate_dir)

    add_candidate(normalized)
    return candidates


def _resolve_nunchaku_loader_cls(model_entry, full_path):
    # TECH DEBT: legacy Nunchaku loader resolution is disabled. Delete later.
    raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")


def _resolve_nunchaku_lora_loader_cls(model, model_entry=None):
    # TECH DEBT: legacy Nunchaku LoRA resolution is disabled. Delete later.
    raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")


def _run_nunchaku_loader(loader_cls, load_target, settings):
    # TECH DEBT: legacy Nunchaku model invocation is disabled. Delete later.
    raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")


def _load_nunchaku(full_path, settings, model_entry=None):
    # TECH DEBT: legacy Nunchaku model loading is disabled. Delete later.
    raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")


def _load_clips(clip_entries, selected_names):
    if not selected_names:
        raise RuntimeError("[VNCCS Control Center] No CLIP selected.")

    paths = []
    clip_type_str = "stable_diffusion"
    for name in selected_names:
        entry = _find_entry(clip_entries, name)
        if not entry:
            raise RuntimeError(f"[VNCCS Control Center] CLIP entry not found: '{name}'")
        full_path, exists = _find_model_on_disk(entry["local_path"])
        if not exists:
            raise RuntimeError(f"[VNCCS Control Center] CLIP not downloaded: '{name}'")
        paths.append(full_path)
        clip_type_str = entry.get("clip_type", clip_type_str)

    clip_type = getattr(
        comfy.sd.CLIPType,
        clip_type_str.upper(),
        comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    return comfy.sd.load_clip(
        ckpt_paths=paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
    )


def _load_vae(vae_entries, selected_name):
    if not selected_name:
        raise RuntimeError("[VNCCS Control Center] No VAE selected.")
    entry = _find_entry(vae_entries, selected_name)
    if not entry:
        raise RuntimeError(f"[VNCCS Control Center] VAE entry not found: '{selected_name}'")
    full_path, exists = _find_model_on_disk(entry["local_path"])
    if not exists:
        raise RuntimeError(f"[VNCCS Control Center] VAE not downloaded: '{selected_name}'")
    sd, metadata = comfy.utils.load_torch_file(full_path, return_metadata=True)
    return comfy.sd.VAE(sd=sd, metadata=metadata)


def _load_model_block(model_entry, selected_type, type_settings, config, selected_clips, selected_vae, custom_model=None):
    if selected_type == "custom":
        if custom_model is None:
            raise RuntimeError("[VNCCS Control Center] Custom model input is not connected.")
        clip = _load_clips(config.get("clip", []), selected_clips)
        vae = _load_vae(config.get("vae", []), selected_vae)
        return custom_model, clip, vae

    if not model_entry:
        raise RuntimeError("[VNCCS Control Center] No model selected.")

    full_path, exists = _find_model_on_disk(model_entry["local_path"])
    if not exists:
        raise RuntimeError(
            f"[VNCCS Control Center] Model not downloaded: '{model_entry['name']}'."
        )

    model_type = selected_type or model_entry.get("type", "")
    if model_type == "checkpoint":
        return _load_checkpoint(full_path)
    if model_type == "unet":
        model = _load_unet(full_path, type_settings.get("unet", {}))
    elif model_type == "gguf":
        model = _load_gguf(full_path)
    elif model_type == "nunchaku":
        # TECH DEBT: legacy Nunchaku branch disabled. Delete this branch later.
        raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")
    else:
        raise RuntimeError(f"[VNCCS Control Center] Unknown model type: '{model_type}'")

    clip = _load_clips(config.get("clip", []), selected_clips)
    vae = _load_vae(config.get("vae", []), selected_vae)
    return model, clip, vae


def _apply_lora_standard(model, clip, full_path, strength):
    lora_name = basename_agnostic(full_path)
    try:
        _validate_downloaded_model_file(full_path, lora_name)
        lora_sd = comfy.utils.load_torch_file(full_path, safe_load=True)
    except Exception as exc:
        raise RuntimeError(
            "[VNCCS Control Center] Failed to load LoRA "
            f"'{lora_name}'. The file appears to be invalid or incomplete; "
            "delete it and download/copy the LoRA again."
        ) from exc
    clip_strength = strength if clip is not None else 0.0
    return comfy.sd.load_lora_for_models(model, clip, lora_sd, strength, clip_strength)


def _apply_lora_nunchaku(model, full_path, strength, settings=None, model_entry=None):
    # TECH DEBT: legacy Nunchaku LoRA application is disabled. Delete later.
    raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")


def _apply_loras(model, clip, lora_states, config, model_type, type_settings=None, model_entry=None):
    entries = config.get("lora", [])
    is_nunchaku = False
    if model_type == "nunchaku":
        # TECH DEBT: legacy Nunchaku LoRA path disabled. Delete later.
        raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")
    state_by_name = {item["name"]: item for item in (lora_states or []) if item.get("name")}

    for entry in entries:
        name = entry.get("name", "")
        if not name:
            continue
        lora_type = _entry_type(entry)
        is_custom = bool(entry.get("custom"))
        is_turbo = lora_type == "turbolora"

        if not is_custom and not is_turbo:
            continue
        if not is_custom and not _lora_matches_model_kind(entry, model_entry):
            continue

        normalized_name = name.strip().lower()
        if normalized_name in _PIPELINE_LOCAL_LORAS or any(target in normalized_name for target in _PIPELINE_LOCAL_LORAS):
            print(f"[VNCCS Control Center] Deferring LoRA to downstream pipeline: {name}")
            continue
        state = state_by_name.get(name, {})
        if not state.get("auto_apply", False):
            continue
        strength = 1.0 if is_turbo else float(state.get("strength", 1.0))
        if abs(strength) < 1e-6:
            continue

        full_path, exists = _find_model_on_disk(entry["local_path"])
        if not exists:
            print(f"[VNCCS Control Center] LoRA not on disk: '{name}', skipping.")
            continue

        print(f"[VNCCS Control Center] Applying LoRA: {name} (strength={strength})")
        model, clip = _apply_lora_standard(model, clip, full_path, strength)

    return model, clip


def _download_worker_loop():
    while True:
        task = _DOWNLOAD_QUEUE.get()
        if task is None:
            break

        repo_id, model_key, target_model = task
        download_repo_id = target_model.get("hf_repo", repo_id)
        temp_path = ""

        try:
            _DOWNLOAD_STATUS[model_key] = {"status": "downloading", "message": "Initializing..."}
            url = ""
            headers = {}

            if target_model.get("url"):
                url = target_model["url"]
                if "civitai.com/models/" in url and "api/download" not in url:
                    parsed = urllib.parse.urlparse(url)
                    query = urllib.parse.parse_qs(parsed.query)
                    if "modelVersionId" in query:
                        version_id = query["modelVersionId"][0]
                        url = f"https://civitai.com/api/download/models/{version_id}"
                if "civitai.com" in url:
                    civitai_token = get_vnccs_config().get("civitai_token", "")
                    if civitai_token:
                        headers = {"Authorization": f"Bearer {civitai_token}"}
            else:
                filename = target_model["hf_path"]
                if filename.startswith(f"{download_repo_id}/"):
                    filename = filename[len(download_repo_id) + 1:]
                url = hf_hub_url(download_repo_id, filename)
                hf_token = get_vnccs_config().get("hf_token")
                if hf_token:
                    headers = {"Authorization": f"Bearer {hf_token}"}

            _validate_https_url(url)
            response = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=_DOWNLOAD_TIMEOUT)
            response.raise_for_status()

            expected_name = basename_agnostic(target_model.get("local_path", "") or target_model.get("hf_path", "") or "model")
            total_size, max_bytes = _validate_download_response(response, expected_name)
            downloaded = 0
            temp_dir = os.path.join(folder_paths.base_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            sanitized_name = "".join(ch for ch in model_key if ch.isalnum())
            temp_path = os.path.join(temp_dir, f"vnccs_{sanitized_name}.tmp")

            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"{expected_name} exceeded max download size")
                    mb_done = downloaded / (1024 * 1024)
                    if total_size > 0:
                        mb_total = total_size / (1024 * 1024)
                        _DOWNLOAD_STATUS[model_key] = {
                            "status": "downloading",
                            "message": f"{mb_done:.1f}/{mb_total:.1f} MB",
                            "progress": (downloaded / total_size) * 100,
                        }
                    else:
                        _DOWNLOAD_STATUS[model_key] = {
                            "status": "downloading",
                            "message": f"{mb_done:.1f} MB",
                            "progress": 0,
                        }

            _DOWNLOAD_STATUS[model_key]["message"] = "Validating..."
            target_abs_path = _resolve_model_download_path(target_model["local_path"])
            _validate_downloaded_model_file(temp_path, os.path.basename(target_abs_path))
            _DOWNLOAD_STATUS[model_key]["message"] = "Installing..."
            os.makedirs(os.path.dirname(target_abs_path), exist_ok=True)

            import shutil

            shutil.move(temp_path, target_abs_path)
            update_installed_version(model_key, target_model.get("version", ""))
            _DOWNLOAD_STATUS[model_key] = {"status": "success", "message": "Installed"}
        except Exception as exc:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_exc:
                    print(f"[VNCCS Control Center] Failed to remove temp download '{temp_path}': {cleanup_exc}")
            is_auth_error = False
            if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                is_auth_error = exc.response.status_code == 401

            message = str(exc)
            status = "error"
            if is_auth_error:
                status = "auth_required"
                message = "API Key Required"
            elif "404" in message or "EntryNotFoundError" in message:
                message = "File not found (404)"

            _DOWNLOAD_STATUS[model_key] = {"status": status, "message": message}
        finally:
            _DOWNLOAD_QUEUE.task_done()


threading.Thread(target=_download_worker_loop, daemon=True).start()


class VNCCSPipeProxy:
    def __init__(self, model, clip, vae):
        self.model = model
        self.clip = clip
        self.vae = vae
        self.pos = None
        self.neg = None
        self.seed_int = 0
        self.sample_steps = DEFAULT_MODEL_STEPS
        self.cfg = DEFAULT_MODEL_CFG
        self.denoise = 0.0
        self.sampler_name = None
        self.scheduler = None
        self.loader_type = None        # "standard" | "nunchaku"
        self.nunchaku_kind = None      # "flux" | "qwen-image" | None
        self.nunchaku_settings = None  # dict or None
        self.model_entry = None        # config model entry dict or None
        self.repo_id = None            # source Control Center repo id
        self.lora_entries = []         # config lora entries
        self.lora_states = []          # UI lora state


class VNCCS_ControlCenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "MIUProject/VNCCS_v3.0", "multiline": False}),
                "node_state": ("STRING", {"default": "{}"}),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("VNCCS_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "VNCCS/manager"

    def execute(self, repo_id, node_state="{}", model=None):
        pipe = _build_control_center_pipe(repo_id, node_state, custom_model=model)
        return (pipe,)


def _build_control_center_pipe(repo_id, node_state, custom_model=None):
    try:
        state = json.loads(node_state) if isinstance(node_state, str) and node_state and node_state != "{}" else (node_state or {})
    except Exception:
        state = {}

    selected_type = state.get("selected_type", "")
    selected_model = state.get("selected_model", "")
    loras = state.get("loras", [])
    type_settings = state.get("type_settings", {})
    model_params = state.get("model_params", {})

    config = _get_cc_config(repo_id)
    if selected_type == "custom":
        model_entry = _custom_context_model_entry(config, state)
    else:
        model_entry = _find_entry(config.get("models", []), selected_model)
    lora_entry_by_name = {
        entry.get("name"): entry
        for entry in config.get("lora", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    has_enabled_loras = False
    for item in loras:
        if not item.get("name") or not item.get("auto_apply", False):
            continue
        lora_entry = lora_entry_by_name.get(item.get("name"))
        if not lora_entry:
            continue
        is_turbo = _entry_type(lora_entry) == "turbolora"
        if not is_turbo and abs(float(item.get("strength", 1.0))) <= 1e-6:
            continue
        if lora_entry.get("custom") or (is_turbo and _lora_matches_model_kind(lora_entry, model_entry)):
            has_enabled_loras = True
            break

    if selected_type == "nunchaku":
        # TECH DEBT: old saved workflows may still request Nunchaku. Delete this
        # guard together with all legacy Nunchaku state after migration.
        raise RuntimeError(f"[VNCCS Control Center] {NUNCHAKU_DISABLED_MESSAGE}")

    model_kind = _entry_kind(model_entry)
    compatible_clips = _filter_entries_by_kind(config.get("clip", []), model_kind)
    compatible_vaes = _filter_entries_by_kind(config.get("vae", []), model_kind)
    all_clip_names = [entry["name"] for entry in compatible_clips]
    first_vae_name = compatible_vaes[0]["name"] if compatible_vaes else ""

    model, clip, vae = _load_model_block(
        model_entry,
        selected_type,
        type_settings,
        config,
        all_clip_names,
        first_vae_name,
        custom_model=custom_model,
    )
    model, clip = _apply_loras(
        model,
        clip,
        loras,
        config,
        selected_type,
        type_settings=type_settings,
        model_entry=model_entry,
    )

    pipe = VNCCSPipeProxy(model, clip, vae)
    pipe.repo_id = repo_id
    pipe.lora_entries = list(config.get("lora", []) or [])
    pipe.lora_states = list(loras or [])

    pipe.loader_type = "standard"
    # TECH DEBT: deprecated Nunchaku pipe fields kept as None for old node state
    # compatibility. Delete these fields after old workflow JSON is migrated.
    pipe.nunchaku_kind = None
    pipe.nunchaku_settings = None
    pipe.model_entry = model_entry

    pipe.sample_steps = int(model_params.get("steps") or DEFAULT_MODEL_STEPS)
    pipe.cfg = float(model_params.get("cfg") if model_params.get("cfg") is not None else DEFAULT_MODEL_CFG)
    if model_params.get("sampler"):
        pipe.sampler_name = model_params["sampler"]
    if model_params.get("scheduler"):
        pipe.scheduler = model_params["scheduler"]

    return pipe


@server.PromptServer.instance.routes.get("/vnccs/control_center/nunchaku_fix_status")
async def cc_nunchaku_fix_status(request):
    return web.json_response({
        "installed": False,
        "disabled": True,
        "message": "Nunchaku auto-patch is disabled in VNCCS web API.",
    })


@server.PromptServer.instance.routes.post("/vnccs/control_center/nunchaku_apply_fix")
async def cc_nunchaku_apply_fix(request):
    return web.json_response({
        "ok": False,
        "disabled": True,
        "message": "Nunchaku auto-patch is disabled. Apply upstream fixes manually if you use Nunchaku.",
    }, status=410)


@server.PromptServer.instance.routes.post("/vnccs/control_center/dependencies")
async def cc_dependencies(request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    model_type = data.get("model_type", "")
    model_name = data.get("model_name", "")
    model_path = data.get("model_path", "")
    has_enabled_loras = bool(data.get("has_enabled_loras", False))

    if model_type != "nunchaku":
        return web.json_response({"ok": True, "message": ""})

    # TECH DEBT: legacy Nunchaku dependency API is kept only for old frontend
    # code paths. Delete after stale workflows are migrated.
    return web.json_response(_get_nunchaku_dependency_status(
        model_entry={"name": model_name, "local_path": model_path},
        full_path=model_path,
        has_enabled_loras=has_enabled_loras,
    ))


@server.PromptServer.instance.routes.post("/vnccs/control_center/clothes_preview")
async def cc_clothes_preview(request):
    try:
        data = await request.json()
        repo_id = (data.get("repo_id") or "").strip()
        node_state = data.get("node_state", "{}")
        clothes_state = data.get("clothes_state", {})

        if not repo_id:
            return web.Response(status=400, text="Missing repo_id")

        pipe = _build_control_center_pipe(repo_id, node_state)

        from .clothes_designer import ClothesDesigner

        widget_data_str = json.dumps(clothes_state, sort_keys=True, separators=(",", ":"))
        designer = ClothesDesigner()
        ret = designer.process(pipe=pipe, widget_data=widget_data_str, unique_id="api_preview")
        image_tensor = ret[0]

        image_array = np.clip(255.0 * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return web.json_response({"image": b64})
    except Exception as e:
        traceback.print_exc()
        return web.Response(status=500, text=str(e))


@server.PromptServer.instance.routes.get("/vnccs/control_center/check")
async def cc_check(request):
    repo_id = request.rel_url.query.get("repo_id", "").strip()
    if not repo_id or " " in repo_id:
        return web.json_response({"error": "Invalid repo_id"}, status=400)

    force = request.rel_url.query.get("force_refresh", "").lower() == "true"
    if force:
        _CC_CONFIG_CACHE.pop(repo_id, None)

    try:
        import asyncio

        loop = asyncio.get_running_loop()
        config = await loop.run_in_executor(None, lambda: _get_cc_config(repo_id))
    except Exception as exc:
        err = str(exc)
        if "404" in err or "not found" in err.lower():
            return web.json_response({"error": "control_center.json not found in repo"}, status=404)
        return web.json_response({"error": err}, status=500)

    installed = get_installed_version_info()

    def enrich(entries, category):
        result = []
        for entry in entries:
            key = f"cc_{category}_{entry['name']}"
            active_ver = installed.get(key)
            _, on_disk = _find_model_on_disk(entry["local_path"])
            if on_disk and not active_ver:
                active_ver = entry.get("version", "")
            result.append({
                **entry,
                "status": "installed" if on_disk else "missing",
                "active_version": active_ver,
            })
        return result

    # TECH DEBT: Nunchaku entries remain in older control_center.json files, but
    # VNCCS no longer exposes or uses them. Delete this filter after the catalog
    # is cleaned up.
    visible_models = [
        entry for entry in config.get("models", [])
        if entry.get("type") != "nunchaku"
    ]
    available_types = list(dict.fromkeys(
        entry.get("type", "") for entry in visible_models if entry.get("type")
    ))

    return web.json_response({
        "name": config.get("name", ""),
        "source": "packaged" if _uses_packaged_cc_config(repo_id) else "huggingface",
        "available_types": available_types,
        "models": enrich(visible_models, "models"),
        "clip": enrich(config.get("clip", []), "clip"),
        "vae": enrich(config.get("vae", []), "vae"),
        "lora": enrich(config.get("lora", []), "lora"),
        "controlnet": enrich(config.get("controlnet", []), "controlnet"),
        "other": enrich(config.get("other", []), "other"),
    })


@server.PromptServer.instance.routes.get("/vnccs/control_center/lora_files")
async def cc_lora_files(request):
    repo_id = request.rel_url.query.get("repo_id", "").strip()
    try:
        available = folder_paths.get_filename_list("loras") or []
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    added_paths = set()
    if repo_id and " " not in repo_id:
        try:
            config = _get_cc_config(repo_id)
            added_paths = {
                _rel_within_folder(entry.get("local_path", ""))
                for entry in config.get("lora", [])
                if isinstance(entry, dict) and entry.get("local_path")
            }
        except Exception:
            added_paths = set()

    items = []
    for rel_path in sorted(set(available), key=lambda value: value.lower()):
        items.append({
            "path": rel_path,
            "label": os.path.basename(rel_path),
            "already_added": rel_path in added_paths,
        })
    return web.json_response({"items": items})


@server.PromptServer.instance.routes.post("/vnccs/control_center/custom_lora")
async def cc_add_custom_lora(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    repo_id = (data.get("repo_id") or "").strip()
    rel_path = (data.get("path") or "").strip().replace("\\", "/")
    if not repo_id or " " in repo_id:
        return web.json_response({"error": "Invalid repo_id"}, status=400)
    if not rel_path:
        return web.json_response({"error": "LoRA path is required"}, status=400)

    try:
        config = _get_cc_config(repo_id)
        existing_paths = {
            entry.get("local_path", "").replace("\\", "/")
            for entry in config.get("lora", [])
            if isinstance(entry, dict) and entry.get("local_path")
        }
        target_local_path = f"models/loras/{rel_path}"
        if target_local_path in existing_paths:
            return web.json_response({"error": "LoRA already exists in the list"}, status=409)

        used_names = {
            entry.get("name")
            for entry in config.get("lora", [])
            if isinstance(entry, dict) and entry.get("name")
        }
        entry = _build_custom_lora_entry(rel_path, used_names=used_names)
        _save_custom_loras([entry])
        _CC_CONFIG_CACHE.pop(repo_id, None)
        return web.json_response({"status": "ok", "entry": entry})
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/vnccs/control_center/custom_lora/delete")
async def cc_delete_custom_lora(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    repo_id = (data.get("repo_id") or "").strip()
    local_path = (data.get("local_path") or "").strip().replace("\\", "/")
    name = (data.get("name") or "").strip()
    if not repo_id or " " in repo_id:
        return web.json_response({"error": "Invalid repo_id"}, status=400)
    if not local_path and not name:
        return web.json_response({"error": "Custom LoRA identifier is required"}, status=400)

    try:
        removed = _remove_custom_lora(local_path=local_path, name=name)
        if not removed:
            return web.json_response({"error": "Custom LoRA not found"}, status=404)
        _CC_CONFIG_CACHE.pop(repo_id, None)
        return web.json_response({"status": "ok"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/vnccs/control_center/download")
async def cc_download(request):
    try:
        validate_privileged_request(request)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=403)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    repo_id = data.get("repo_id", "").strip()
    category = data.get("category", "models")
    name = data.get("name", "")

    if not repo_id or " " in repo_id:
        return web.json_response({"error": "Invalid repo_id"}, status=400)

    try:
        import asyncio

        loop = asyncio.get_running_loop()
        config = await loop.run_in_executor(None, lambda: _get_cc_config(repo_id))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    entry = _find_entry(config.get(category, []), name)
    if not entry:
        return web.json_response({"error": f"Entry '{name}' not found in '{category}'"}, status=404)
    try:
        _resolve_model_download_path(entry.get("local_path", ""))
        if entry.get("url"):
            _validate_https_url(entry["url"])
        elif entry.get("hf_path"):
            _validate_model_filename(entry["hf_path"])
        else:
            return web.json_response({"error": "Entry must define either url or hf_path"}, status=400)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    key = f"cc_{category}_{name}"
    _DOWNLOAD_STATUS[key] = {"status": "queued", "message": "Queued..."}
    _DOWNLOAD_QUEUE.put((repo_id, key, entry))
    return web.json_response({"status": "queued", "message": f"Download queued for {name}"})


@server.PromptServer.instance.routes.get("/vnccs/manager/status")
async def get_download_status(request):
    return web.json_response(_DOWNLOAD_STATUS)


@server.PromptServer.instance.routes.post("/vnccs/manager/save_token")
async def save_api_token(request):
    try:
        validate_privileged_request(request)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=403)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    tokens = {}
    if "token" in data:
        tokens["civitai_token"] = data["token"]
    if "civitai_token" in data:
        tokens["civitai_token"] = data["civitai_token"]
    if "hf_token" in data:
        tokens["hf_token"] = data["hf_token"]

    try:
        save_vnccs_config(tokens)
        return web.json_response({"status": "saved"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@server.PromptServer.instance.routes.get("/vnccs/module_status")
async def vnccs_module_status(request):
    import re

    custom_nodes_dir = os.path.join(folder_paths.base_path, "custom_nodes")
    modules = {
        "main": ["vnccs", "ComfyUI_VNCCS"],
        "utils": ["vnccs-utils", "ComfyUI_VNCCS_Utils"],
    }
    dependency_modules = {
        "sam3": {
            "label": "SAM3",
            "github_url": "https://github.com/yolain/ComfyUI-Easy-SAM3",
            "folders": ["comfyui-easy-sam3"],
            "nodes": [
                {"node_id": "easy sam3ModelLoader", "class_names": ["LoadSam3Model"]},
                {"node_id": "easy sam3ImageSegmentation", "class_names": ["Sam3ImageSegmentation"]},
            ],
        },
        "seedvr": {
            "label": "SeedVR",
            "github_url": "https://github.com/TencentARC/ComfyUI-SeedVR2_VideoUpscaler",
            "folders": ["ComfyUI-SeedVR2_VideoUpscaler", "comfyui-seedvr2-videoupscaler", "ComfyUI-SeedVR"],
            "nodes": [
                {"class_names": ["SeedVR2LoadDiTModel"]},
                {"class_names": ["SeedVR2LoadVAEModel"]},
                {"class_names": ["SeedVR2VideoUpscaler"]},
            ],
        },
        "gguf": {
            "label": "GGUF",
            "github_url": "https://github.com/city96/ComfyUI-GGUF",
            "folders": ["ComfyUI-GGUF"],
            "nodes": [
                {"class_names": ["UnetLoaderGGUF"]},
            ],
        },
        "impact": {
            "label": "Impact",
            "github_url": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
            "folders": ["ComfyUI-Impact-Pack", "ComfyUI-Impact-Subpack"],
            "nodes": [
                {"class_names": ["UltralyticsDetectorProvider"]},
                {"class_names": ["SAMLoader"]},
                {"class_names": ["FaceDetailer"]},
            ],
        },
    }

    def read_version(dirpath):
        pyproject = os.path.join(dirpath, "pyproject.toml")
        if not os.path.exists(pyproject):
            return None
        try:
            with open(pyproject, "r", encoding="utf-8") as handle:
                content = handle.read()
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            return match.group(1) if match else None
        except Exception:
            return None

    def comfy_node_available(spec):
        mappings = _get_node_class_mappings()
        node_id = spec.get("node_id")
        if node_id and node_id in mappings:
            return True

        for class_name in spec.get("class_names", []):
            if class_name in mappings:
                return True

        for module in list(sys.modules.values()):
            if module is None:
                continue
            for class_name in spec.get("class_names", []):
                if getattr(module, class_name, None) is not None:
                    return True
            if node_id:
                try:
                    values = vars(module).values()
                except Exception:
                    continue
                for candidate in values:
                    if not inspect.isclass(candidate) or not hasattr(candidate, "define_schema"):
                        continue
                    try:
                        schema = candidate.define_schema()
                        if getattr(schema, "node_id", None) == node_id:
                            return True
                    except Exception:
                        continue
        return False

    def dependency_status(spec):
        folders = []
        for folder in spec.get("folders", []):
            path = os.path.join(custom_nodes_dir, folder)
            if os.path.isdir(path):
                folders.append(folder)

        missing_nodes = []
        for node_spec in spec.get("nodes", []):
            if not comfy_node_available(node_spec):
                missing_nodes.append(node_spec.get("node_id") or "/".join(node_spec.get("class_names", [])))

        if not missing_nodes:
            return {
                "label": spec["label"],
                "github_url": spec.get("github_url"),
                "status": "ok",
                "folder": folders[0] if folders else None,
                "missing_nodes": [],
            }
        return {
            "label": spec["label"],
            "github_url": spec.get("github_url"),
            "status": "partial" if folders else "missing",
            "folder": folders[0] if folders else None,
            "missing_nodes": missing_nodes,
        }

    result = {}
    for key, name_variants in modules.items():
        found = []
        for name in name_variants:
            path = os.path.join(custom_nodes_dir, name)
            if os.path.isdir(path):
                found.append({"folder": name, "version": read_version(path)})

        if not found:
            result[key] = {"error": "not_found"}
        elif len(found) > 1:
            preferred = found[0]
            result[key] = {
                "version": preferred["version"],
                "folder": preferred["folder"],
                "duplicate": True,
                "duplicate_folders": [item["folder"] for item in found],
            }
        else:
            result[key] = {
                "version": found[0]["version"],
                "folder": found[0]["folder"],
                "duplicate": False,
            }

    result["dependencies"] = {
        key: dependency_status(spec)
        for key, spec in dependency_modules.items()
    }

    return web.json_response(result)


NODE_CLASS_MAPPINGS = {
    "VNCCS_ControlCenter": VNCCS_ControlCenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_ControlCenter": "VNCCS Control Center",
}
