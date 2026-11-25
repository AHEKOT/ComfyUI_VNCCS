"""Shared model download and native text generation for the character wizards."""
from ..operation_logger import log_event, log_stage, logged_operation

import json
import os
import re
import threading

from aiohttp import web
from huggingface_hub import hf_hub_download
import server

from ..utils import privileged_route


QWEN_VL_MODEL_FILENAME = "Qwen3.5-4B_int8_convrot.safetensors"
QWEN_VL_MODEL_REPO_ID = "beycanai/Qwen-LM"
QWEN_VL_MODEL_REVISION = "65be86eff5aab8bf7c1fe0d9a363e1fa8433d1ca"
_QWEN_VL_MODEL_ENTRY = {
    "name": QWEN_VL_MODEL_FILENAME,
    "local_path": f"models/text_encoders/{QWEN_VL_MODEL_FILENAME}",
    "clip_type": "qwen_image",
}

_QWEN_VL_DOWNLOAD_LOCK = threading.Lock()
_QWEN_VL_DOWNLOAD_STATUS = {
    "status": "idle",
    "progress": 0,
    "current_file": "",
    "total_size": 0,
    "downloaded_size": 0,
    "error": "",
}


def parse_wizard_json(content):
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

    return data


def normalize_character_wizard_traits(data):
    result = {}
    for key in ["race", "skin_color", "hair", "eyes", "face", "body", "additional_details"]:
        value = data.get(key, "")
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        result[key] = value.strip()

    sex = str(data.get("sex", "female")).strip().lower()
    result["sex"] = "male" if sex.startswith("m") else "female"
    try:
        age = int(float(data.get("age", 18)))
    except Exception:
        age = 18
    result["age"] = max(1, min(100, age))
    return result


def _local_wizard_model():
    from . import vnccs_control_center as catalog

    def installed(entry):
        path, exists = catalog._find_model_on_disk(entry["local_path"])
        if not exists:
            return False
        try:
            return catalog._validate_downloaded_model_file(path, os.path.basename(path))
        except (OSError, ValueError):
            return False

    with open(catalog._get_packaged_cc_path(), encoding="utf-8") as handle:
        config = json.load(handle)
    qi2_installed = any(
        catalog._entry_kind(entry) == "qi2" and installed(entry)
        for entry in config.get("models", ())
    )
    if qi2_installed:
        for entry in config.get("clip", ()):
            if catalog._entry_kind(entry) == "qi2" and installed(entry):
                return entry

    path, exists = catalog._find_model_on_disk(_QWEN_VL_MODEL_ENTRY["local_path"])
    if not exists:
        raise FileNotFoundError(f"{QWEN_VL_MODEL_FILENAME} is missing from ComfyUI's text encoders.")
    catalog._validate_downloaded_model_file(path, QWEN_VL_MODEL_FILENAME)
    return _QWEN_VL_MODEL_ENTRY


def generate_wizard_text(model, system_prompt, prompt, *, max_length, temperature, image=None):
    from .character_creator_v2 import QI2_TEXT_GENERATION_DEFAULTS
    from .character_generator import _call_comfy_node
    from .vnccs_control_center import _load_clips

    clip = _load_clips([model], [model["name"]])
    generation = {
        **QI2_TEXT_GENERATION_DEFAULTS,
        "max_length": max_length,
        "sampling_mode": {**QI2_TEXT_GENERATION_DEFAULTS["sampling_mode"], "temperature": temperature},
    }
    if image is not None:
        generation["image"] = image
    return _call_comfy_node("TextGenerate", clip=clip, prompt=prompt, system_prompt=system_prompt, **generation)[0]


@logged_operation("QwenVL", "download_asset")
def _download_qwen_vl_model():
    from . import vnccs_control_center as catalog

    try:
        return _local_wizard_model()
    except (FileNotFoundError, ValueError):
        pass

    path, exists = catalog._find_model_on_disk(_QWEN_VL_MODEL_ENTRY["local_path"])
    target = path if exists else catalog._resolve_model_download_path(_QWEN_VL_MODEL_ENTRY["local_path"])
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    log_stage("download", component="QwenVL", file=QWEN_VL_MODEL_FILENAME, repository=QWEN_VL_MODEL_REPO_ID)
    path = hf_hub_download(
        repo_id=QWEN_VL_MODEL_REPO_ID,
        filename=QWEN_VL_MODEL_FILENAME,
        revision=QWEN_VL_MODEL_REVISION,
        local_dir=target_dir,
        token=False,
        force_download=exists,
    )
    catalog._validate_downloaded_model_file(path, QWEN_VL_MODEL_FILENAME)
    size = os.path.getsize(path)
    _QWEN_VL_DOWNLOAD_STATUS.update(downloaded_size=size, total_size=size)
    return _QWEN_VL_MODEL_ENTRY


def _qwen_vl_download_worker():
    try:
        model = _download_qwen_vl_model()
        _QWEN_VL_DOWNLOAD_STATUS.update(status="completed", progress=100, current_file=model["name"], error="")
    except Exception as exc:
        _QWEN_VL_DOWNLOAD_STATUS.update(status="error", error=str(exc))
        log_event("download_failed", component="QwenVL", level="error", error=str(exc))


def _start_qwen_vl_download():
    with _QWEN_VL_DOWNLOAD_LOCK:
        if _QWEN_VL_DOWNLOAD_STATUS["status"] == "downloading":
            return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS), status=409)
        try:
            model = _local_wizard_model()
        except (FileNotFoundError, ValueError):
            _QWEN_VL_DOWNLOAD_STATUS.update(
                status="downloading", progress=0, current_file=QWEN_VL_MODEL_FILENAME,
                downloaded_size=0, total_size=0, error="",
            )
            try:
                threading.Thread(target=_qwen_vl_download_worker, daemon=True).start()
            except Exception as exc:
                _QWEN_VL_DOWNLOAD_STATUS.update(status="error", error=str(exc))
                return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS), status=500)
            return web.json_response({"status": "started"})

        _QWEN_VL_DOWNLOAD_STATUS.update(
            status="completed", progress=100, current_file=model["name"],
            downloaded_size=0, total_size=0, error="",
        )
        return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS))


@server.PromptServer.instance.routes.get("/vnccs/qwen_vl_model_status")
async def qwen_vl_model_status(request):
    try:
        model = _local_wizard_model()
        return web.json_response({"ready": True, "model_name": model["name"]})
    except (FileNotFoundError, ValueError) as exc:
        return web.json_response({"ready": False, "model_name": QWEN_VL_MODEL_FILENAME, "message": str(exc)})


@server.PromptServer.instance.routes.get("/vnccs/qwen_vl_download_status")
async def qwen_vl_download_status(request):
    return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS))


@server.PromptServer.instance.routes.post("/vnccs/qwen_vl_download_model")
@privileged_route
async def qwen_vl_download_model(request):
    return _start_qwen_vl_download()
