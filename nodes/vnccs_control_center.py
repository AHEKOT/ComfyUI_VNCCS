import json
import os
import io
import base64
import queue
import threading
import time
import urllib.parse
import inspect

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
_FOLDER_MAP = {
    "unet": ["unet", "diffusion_models"],
    "checkpoints": ["checkpoints"],
    "loras": ["loras"],
    "clip": ["clip"],
    "vae": ["vae"],
    "controlnet": ["controlnet"],
    "upscale_models": ["upscale_models"],
    "embeddings": ["embeddings"],
    "gguf": ["unet", "diffusion_models"],
    "diffusion_models": ["diffusion_models", "unet"],
}


def _get_node_class_mappings():
    import nodes as comfy_nodes

    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})


def _detect_nunchaku_model_kind(model_entry=None, full_path=""):
    entry_name = (model_entry.get("name", "") if model_entry else "").lower()
    entry_path = (model_entry.get("local_path", "") if model_entry else "").lower()
    identity = " ".join([entry_name, entry_path, full_path.lower()])
    return "qwen-image" if "qwen" in identity else "flux"


def _get_nunchaku_dependency_status(model_entry=None, full_path="", has_enabled_loras=False):
    mappings = _get_node_class_mappings()
    model_kind = _detect_nunchaku_model_kind(model_entry=model_entry, full_path=full_path)

    if model_kind == "qwen-image":
        loader_name = "NunchakuQwenImageDiTLoader"
        lora_loader_name = "NunchakuQwenImageLoraLoader"
        install_hint = (
            "Install ComfyUI-nunchaku for the model loader and ComfyUI-QwenImageLoraLoader "
            "for Qwen-Image LoRA support."
        )
    else:
        loader_name = "NunchakuFluxDiTLoader"
        lora_loader_name = "NunchakuFluxLoraLoader"
        install_hint = "Install ComfyUI-nunchaku to get the required FLUX Nunchaku nodes."

    has_loader = loader_name in mappings
    has_lora_loader = lora_loader_name in mappings
    missing = []

    if not has_loader:
        missing.append(loader_name)
    if has_enabled_loras and not has_lora_loader:
        missing.append(lora_loader_name)

    return {
        "model_kind": model_kind,
        "loader_name": loader_name,
        "lora_loader_name": lora_loader_name,
        "has_loader": has_loader,
        "has_lora_loader": has_lora_loader,
        "has_enabled_loras": has_enabled_loras,
        "ok": not missing,
        "missing": missing,
        "message": "" if not missing else (
            f"Missing required dependency nodes: {', '.join(missing)}. {install_hint}"
        ),
    }


def resolve_path(relative_path):
    if not relative_path:
        return ""
    expanded = os.path.expanduser(relative_path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    base = getattr(folder_paths, "base_path", os.getcwd())
    return os.path.abspath(os.path.join(base, expanded))


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


def _get_cc_config(repo_id):
    cached = _CC_CONFIG_CACHE.get(repo_id)
    now = time.time()
    if cached and now - cached.get("ts", 0) < 300:
        return cached["data"]

    user_config = get_vnccs_config()
    hf_token = user_config.get("hf_token") or os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id=repo_id,
        filename="control_center.json",
        local_files_only=False,
        token=hf_token,
    )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    _CC_CONFIG_CACHE[repo_id] = {"ts": now, "data": data}
    return data


def _find_entry(entries, name):
    if not name:
        return None
    for entry in entries:
        if entry.get("name", "").strip().lower() == name.strip().lower():
            return entry
    return None


def _rel_within_folder(local_path):
    parts = local_path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "models":
        return "/".join(parts[2:])
    return parts[-1]


def _find_model_on_disk(local_path):
    if not local_path:
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
            try:
                found = folder_paths.get_full_path(key, search)
                if found and os.path.exists(found):
                    return found, True
            except Exception:
                pass
            try:
                for folder in folder_paths.get_folder_paths(key) or []:
                    candidate = os.path.join(folder, search)
                    if os.path.exists(candidate):
                        return candidate, True
            except Exception:
                pass

    fallback = resolve_path(local_path)
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
    model, = loader.load_unet(os.path.basename(full_path))
    return model


def _get_nunchaku_load_candidates(full_path):
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
    import nodes as comfy_nodes

    entry_name = (model_entry.get("name", "") if model_entry else "").lower()
    entry_path = (model_entry.get("local_path", "") if model_entry else "").lower()
    full_path_lower = full_path.lower()
    identity = " ".join([entry_name, entry_path, full_path_lower])

    if "qwen" in identity and "NunchakuQwenImageDiTLoader" in comfy_nodes.NODE_CLASS_MAPPINGS:
        return comfy_nodes.NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"], "qwen-image"

    if "NunchakuFluxDiTLoader" in comfy_nodes.NODE_CLASS_MAPPINGS:
        return comfy_nodes.NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"], "flux"

    raise RuntimeError(
        "[VNCCS Control Center] No compatible Nunchaku loader found. "
        "Expected NunchakuQwenImageDiTLoader or NunchakuFluxDiTLoader."
    )


def _resolve_nunchaku_lora_loader_cls(model, model_entry=None):
    import nodes as comfy_nodes

    entry_name = (model_entry.get("name", "") if model_entry else "").lower()
    entry_path = (model_entry.get("local_path", "") if model_entry else "").lower()
    model_wrapper = getattr(getattr(model, "model", None), "diffusion_model", None)
    wrapper_type_name = type(model_wrapper).__name__.lower() if model_wrapper is not None else ""
    identity = " ".join([entry_name, entry_path, wrapper_type_name])

    if "qwen" in identity and "NunchakuQwenImageLoraLoader" in comfy_nodes.NODE_CLASS_MAPPINGS:
        return comfy_nodes.NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"], "qwen-image"

    if "NunchakuFluxLoraLoader" in comfy_nodes.NODE_CLASS_MAPPINGS:
        return comfy_nodes.NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"], "flux"

    raise RuntimeError(
        "[VNCCS Control Center] No compatible Nunchaku LoRA loader found. "
        "Expected NunchakuQwenImageLoraLoader or NunchakuFluxLoraLoader."
    )


def _run_nunchaku_loader(loader_cls, load_target, settings):
    original = folder_paths.get_full_path_or_raise
    abs_target = os.path.abspath(load_target)
    base_name = os.path.basename(abs_target.rstrip("\\/"))

    def patched(folder_name, filename):
        if filename == base_name or os.path.abspath(filename) == abs_target:
            return load_target
        return original(folder_name, filename)

    folder_paths.get_full_path_or_raise = patched
    try:
        loader = loader_cls()
        signature = inspect.signature(loader.load_model)
        default_values = {
            "model_name": base_name,
            "model_path": base_name,
            "attention": settings.get("attention", "nunchaku-fp16"),
            "cache_threshold": float(settings.get("cache_threshold", 0.0)),
            "cpu_offload": settings.get("cpu_offload", "auto"),
            "device_id": int(settings.get("device_id", 0)),
            "data_type": settings.get("data_type", "bfloat16"),
            "i2f_mode": settings.get("i2f_mode", "enabled"),
            "num_blocks_on_gpu": int(settings.get("num_blocks_on_gpu", 1)),
            "use_pin_memory": settings.get("use_pin_memory", "disable"),
        }
        call_kwargs = {
            name: default_values[name]
            for name in signature.parameters
            if name != "self" and name in default_values
        }
        return loader.load_model(**call_kwargs)
    finally:
        folder_paths.get_full_path_or_raise = original


def _load_nunchaku(full_path, settings, model_entry=None):
    loader_cls, loader_kind = _resolve_nunchaku_loader_cls(model_entry, full_path)
    candidates = _get_nunchaku_load_candidates(full_path)
    errors = []

    for candidate in candidates:
        try:
            result = _run_nunchaku_loader(loader_cls, candidate, settings)
            return result[0]
        except KeyError as exc:
            errors.append(f"{candidate}: missing key {exc}")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    details = "; ".join(errors) if errors else "unknown error"
    raise RuntimeError(
        f"[VNCCS Control Center] Failed to load Nunchaku {loader_kind} model. "
        f"Tried: {', '.join(candidates)}. Details: {details}"
    )


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


def _load_model_block(model_entry, selected_type, type_settings, config, selected_clips, selected_vae):
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
        model = _load_nunchaku(full_path, type_settings.get("nunchaku", {}), model_entry=model_entry)
    else:
        raise RuntimeError(f"[VNCCS Control Center] Unknown model type: '{model_type}'")

    clip = _load_clips(config.get("clip", []), selected_clips)
    vae = _load_vae(config.get("vae", []), selected_vae)
    return model, clip, vae


def _apply_lora_standard(model, clip, full_path, strength):
    lora_sd = comfy.utils.load_torch_file(full_path, safe_load=True)
    clip_strength = strength if clip is not None else 0.0
    return comfy.sd.load_lora_for_models(model, clip, lora_sd, strength, clip_strength)


def _apply_lora_nunchaku(model, full_path, strength, settings=None, model_entry=None):
    loader_cls, loader_kind = _resolve_nunchaku_lora_loader_cls(model, model_entry=model_entry)
    original = folder_paths.get_full_path_or_raise
    base_name = os.path.basename(full_path)

    def patched(folder_name, filename):
        if folder_name == "loras" and filename == base_name:
            return full_path
        return original(folder_name, filename)

    folder_paths.get_full_path_or_raise = patched
    try:
        loader = loader_cls()
        signature = inspect.signature(loader.load_lora)
        default_values = {
            "model": model,
            "lora_name": base_name,
            "lora_strength": strength,
            "cpu_offload": (settings or {}).get("cpu_offload", "disable"),
        }
        call_kwargs = {
            name: default_values[name]
            for name in signature.parameters
            if name != "self" and name in default_values
        }
        result = loader.load_lora(**call_kwargs)
    finally:
        folder_paths.get_full_path_or_raise = original
    if not isinstance(result, (tuple, list)) or not result:
        raise RuntimeError(f"[VNCCS Control Center] Unexpected result from Nunchaku {loader_kind} LoRA loader.")
    return result[0]


def _apply_loras(model, clip, lora_states, config, model_type, type_settings=None, model_entry=None):
    entries = config.get("lora", [])
    is_nunchaku = model_type == "nunchaku"
    state_by_name = {item["name"]: item for item in (lora_states or []) if item.get("name")}

    for entry in entries:
        name = entry.get("name", "")
        if not name:
            continue
        state = state_by_name.get(name, {})
        if not state.get("enabled", True):
            continue
        strength = float(state.get("strength", 1.0))
        if abs(strength) < 1e-6:
            continue

        full_path, exists = _find_model_on_disk(entry["local_path"])
        if not exists:
            print(f"[VNCCS Control Center] LoRA not on disk: '{name}', skipping.")
            continue

        print(f"[VNCCS Control Center] Applying LoRA: {name} (strength={strength})")
        if is_nunchaku:
            model = _apply_lora_nunchaku(
                model,
                full_path,
                strength,
                settings=(type_settings or {}).get("nunchaku", {}),
                model_entry=model_entry,
            )
        else:
            model, clip = _apply_lora_standard(model, clip, full_path, strength)

    return model, clip


def _download_worker_loop():
    while True:
        task = _DOWNLOAD_QUEUE.get()
        if task is None:
            break

        repo_id, model_key, target_model = task
        download_repo_id = target_model.get("hf_repo", repo_id)

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
                hf_token = get_vnccs_config().get("hf_token") or os.environ.get("HF_TOKEN")
                if hf_token:
                    headers = {"Authorization": f"Bearer {hf_token}"}

            response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
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

            _DOWNLOAD_STATUS[model_key]["message"] = "Installing..."
            target_abs_path = resolve_path(target_model["local_path"])
            os.makedirs(os.path.dirname(target_abs_path), exist_ok=True)

            import shutil

            shutil.move(temp_path, target_abs_path)
            update_installed_version(model_key, target_model.get("version", ""))
            _DOWNLOAD_STATUS[model_key] = {"status": "success", "message": "Installed"}
        except Exception as exc:
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
        self.sample_steps = 0
        self.cfg = 0.0
        self.denoise = 0.0
        self.sampler_name = None
        self.scheduler = None


class VNCCS_ControlCenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "MIUProject/VNCCS_V2", "multiline": False}),
                "node_state": ("STRING", {"default": "{}"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = _ByPassTypeTuple(("VNCCS_PIPE", any_type))
    RETURN_NAMES = _ByPassTypeTuple(("pipe", "path"))
    FUNCTION = "execute"
    CATEGORY = "VNCCS/manager"

    def execute(self, repo_id, node_state="{}"):
        config = _get_cc_config(repo_id)
        pipe = _build_control_center_pipe(repo_id, node_state)

        try:
            state = json.loads(node_state) if node_state and node_state != "{}" else {}
        except Exception:
            state = {}

        output_slot_names = [name for name in state.get("output_slot_names", []) if name and name != "-"]
        dynamic = _build_dynamic_paths(config, output_slot_names)
        return (pipe, *dynamic)


def _build_control_center_pipe(repo_id, node_state):
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
    model_entry = _find_entry(config.get("models", []), selected_model)
    has_enabled_loras = any(
        item.get("name") and item.get("enabled", True) and abs(float(item.get("strength", 1.0))) > 1e-6
        for item in loras
    )

    if selected_type == "nunchaku" and model_entry is not None:
        dependency_status = _get_nunchaku_dependency_status(
            model_entry=model_entry,
            full_path=model_entry.get("local_path", ""),
            has_enabled_loras=has_enabled_loras,
        )
        if not dependency_status["ok"]:
            raise RuntimeError(f"[VNCCS Control Center] {dependency_status['message']}")

    all_clip_names = [entry["name"] for entry in config.get("clip", [])]
    first_vae_name = config["vae"][0]["name"] if config.get("vae") else ""

    model, clip, vae = _load_model_block(
        model_entry,
        selected_type,
        type_settings,
        config,
        all_clip_names,
        first_vae_name,
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

    if model_params.get("steps"):
        pipe.sample_steps = int(model_params["steps"])
    if model_params.get("cfg") is not None:
        pipe.cfg = float(model_params["cfg"])
    if model_params.get("sampler"):
        pipe.sampler_name = model_params["sampler"]
    if model_params.get("scheduler"):
        pipe.scheduler = model_params["scheduler"]

    return pipe


def _get_nunchaku_path():
    """Find ComfyUI-nunchaku installation in ComfyUI's custom_nodes directory."""
    comfyui_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
    custom_nodes = os.path.join(comfyui_root, "custom_nodes")
    for name in ["ComfyUI-nunchaku", "comfyui-nunchaku", "ComfyUI_nunchaku"]:
        path = os.path.join(custom_nodes, name)
        if os.path.isdir(path):
            return path
    return None


def _check_nunchaku_qwen_fix():
    print("[VNCCS Qwen Fix] Checking fix status...")
    nunchaku_path = _get_nunchaku_path()
    if not nunchaku_path:
        print("[VNCCS Qwen Fix] ComfyUI-nunchaku not found in custom_nodes")
        return {"installed": False, "nunchaku_missing": True}
    qwen_file = os.path.join(nunchaku_path, "models", "qwenimage.py")
    print(f"[VNCCS Qwen Fix] Checking file: {qwen_file}")
    if not os.path.isfile(qwen_file):
        print("[VNCCS Qwen Fix] qwenimage.py not found")
        return {"installed": False, "nunchaku_missing": False}
    with open(qwen_file, "r", encoding="utf-8") as f:
        content = f.read()
    installed = "timestep_zero_index=None" in content
    print(f"[VNCCS Qwen Fix] Fix {'is' if installed else 'is NOT'} applied")
    return {"installed": installed, "nunchaku_missing": False}


def _apply_nunchaku_qwen_fix():
    """Fetch PR #790 diff and apply it to qwenimage.py via git apply."""
    import tempfile, subprocess
    print("[VNCCS Qwen Fix] Starting fix application...")
    nunchaku_path = _get_nunchaku_path()
    if not nunchaku_path:
        print("[VNCCS Qwen Fix] ERROR: ComfyUI-nunchaku not found in custom_nodes")
        return {"ok": False, "message": "ComfyUI-nunchaku not found in custom_nodes"}
    print(f"[VNCCS Qwen Fix] nunchaku path: {nunchaku_path}")
    diff_url = "https://github.com/nunchaku-ai/ComfyUI-nunchaku/pull/790.diff"
    try:
        print(f"[VNCCS Qwen Fix] Downloading diff from {diff_url}")
        resp = requests.get(diff_url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        print(f"[VNCCS Qwen Fix] Diff downloaded ({len(resp.text)} chars)")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False, encoding="utf-8") as f:
            f.write(resp.text)
            tmp_path = f.name
        print(f"[VNCCS Qwen Fix] Saved diff to temp file: {tmp_path}")
        try:
            print(f"[VNCCS Qwen Fix] Running: git apply --whitespace=fix {tmp_path}")
            result = subprocess.run(
                ["git", "apply", "--whitespace=fix", tmp_path],
                cwd=nunchaku_path,
                capture_output=True, text=True
            )
        finally:
            os.unlink(tmp_path)
            print("[VNCCS Qwen Fix] Temp file removed")
        print(f"[VNCCS Qwen Fix] git apply exit code: {result.returncode}")
        if result.stdout:
            print(f"[VNCCS Qwen Fix] git stdout: {result.stdout}")
        if result.stderr:
            print(f"[VNCCS Qwen Fix] git stderr: {result.stderr}")
        if result.returncode != 0:
            return {"ok": False, "message": result.stderr or result.stdout}
        print("[VNCCS Qwen Fix] Fix applied successfully")
        return {"ok": True}
    except Exception as e:
        print(f"[VNCCS Qwen Fix] Exception: {e}")
        return {"ok": False, "message": str(e)}


@server.PromptServer.instance.routes.get("/vnccs/control_center/nunchaku_fix_status")
async def cc_nunchaku_fix_status(request):
    return web.json_response(_check_nunchaku_qwen_fix())


@server.PromptServer.instance.routes.post("/vnccs/control_center/nunchaku_apply_fix")
async def cc_nunchaku_apply_fix(request):
    return web.json_response(_apply_nunchaku_qwen_fix())


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

    model_entry = {"name": model_name, "local_path": model_path}
    status = _get_nunchaku_dependency_status(
        model_entry=model_entry,
        full_path=model_path,
        has_enabled_loras=has_enabled_loras,
    )
    return web.json_response(status)


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

    available_types = list(dict.fromkeys(
        entry.get("type", "") for entry in config.get("models", []) if entry.get("type")
    ))

    return web.json_response({
        "name": config.get("name", ""),
        "available_types": available_types,
        "models": enrich(config.get("models", []), "models"),
        "clip": enrich(config.get("clip", []), "clip"),
        "vae": enrich(config.get("vae", []), "vae"),
        "lora": enrich(config.get("lora", []), "lora"),
        "controlnet": enrich(config.get("controlnet", []), "controlnet"),
        "other": enrich(config.get("other", []), "other"),
    })


@server.PromptServer.instance.routes.post("/vnccs/control_center/download")
async def cc_download(request):
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

    return web.json_response(result)


NODE_CLASS_MAPPINGS = {
    "VNCCS_ControlCenter": VNCCS_ControlCenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_ControlCenter": "VNCCS Control Center",
}
