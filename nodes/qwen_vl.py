"""Lightweight helpers for Qwen VL llama-cpp-python integration."""
from ..operation_logger import log_event, log_stage, logged_operation

import json
import os
import re
import threading

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    import server
    from aiohttp import web
except ImportError:
    server = web = None

from ..utils import get_full_path_agnostic, privileged_route

QWEN_VL_HANDLER_NAMES = ("Qwen35ChatHandler",)

def get_qwen_vl_chat_handler(llama_cpp):
    chat_format = getattr(llama_cpp, "llama_chat_format", None)
    if chat_format is None:
        try:
            import llama_cpp.llama_chat_format as chat_format
        except Exception:
            chat_format = None

    available = []
    if chat_format is not None:
        available = [name for name in dir(chat_format) if "Handler" in name]
        for name in QWEN_VL_HANDLER_NAMES:
            handler = getattr(chat_format, name, None)
            if handler is not None:
                return handler

    raise RuntimeError(
        "No Qwen3.5 chat handler found in llama-cpp-python. "
        "Qwen3.5 requires Qwen35ChatHandler; "
        "refusing to use Llava15ChatHandler because it can crash with Qwen VL GGUF/mmproj. "
        f"Available handlers: {available}"
    )

def configure_qwen_text_chat(llm):
    """Keep short wizard responses in non-thinking mode using the GGUF template."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise RuntimeError("Qwen3.5 GGUF is missing its chat template.")
    llm.chat_handler = Jinja2ChatFormatter(
        template="{% set enable_thinking = false %}" + template,
        eos_token="<|im_end|>",
        bos_token="",
    ).to_chat_handler()

QWEN_VL_MODEL_FILENAME = "Qwen3.5-4B-Q8_0.gguf"

QWEN_VL_MODEL_NAMES = [QWEN_VL_MODEL_FILENAME]

QWEN_VL_MMPROJ_NAMES = ["mmproj-Qwen3.5-4B-F16.gguf", "mmproj-Qwen3.5-4B-BF16.gguf"]

QWEN_VL_MODEL_REPO_ID = "unsloth/Qwen3.5-4B-GGUF"

QWEN_VL_MMPROJ_FILENAME = "mmproj-F16.gguf"

QWEN_VL_MODEL_REVISION = "e966ccab6d3f3c91e94d858b4a01c921a5d7ef53"

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


def _validate_gguf_file(path, file_label="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_label} was not written: {path}")

    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise ValueError(f"{file_label} is too small to be a valid GGUF file ({size} bytes)")

    with open(path, "rb") as file:
        magic = file.read(4)
    if magic != b"GGUF":
        raise ValueError(f"{file_label} is not a valid GGUF file (magic={magic!r})")

def _llm_search_dirs():
    if not folder_paths or not hasattr(folder_paths, "models_dir"):
        return []
    base_path = folder_paths.models_dir
    roots = [os.path.join(base_path, "llm"), os.path.join(base_path, "LLM"), base_path]
    return [directory for root in roots for directory in (root, os.path.join(root, "Qwen3.5-4B"))]

def _find_qwen_vl_model():
    for directory in _llm_search_dirs():
        if not os.path.isdir(directory):
            continue
        for name in QWEN_VL_MODEL_NAMES:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None

def _find_qwen_vl_mmproj(model_path):
    if not model_path:
        return None

    model_dir = os.path.dirname(model_path)
    # Generic projector names are safe only inside this model's own directory.
    directories = [os.path.join(model_dir, "Qwen3.5-4B")]
    if "qwen3.5-4b" in os.path.basename(model_dir).lower():
        directories.insert(0, model_dir)
    for directory in directories:
        for name in ("mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf"):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return path
    for name in QWEN_VL_MMPROJ_NAMES:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path

    return None

def _qwen_vl_download_dir():
    if not folder_paths or not getattr(folder_paths, "models_dir", None):
        return os.path.join("models", "llm", "Qwen3.5-4B")
    return os.path.join(folder_paths.models_dir, "llm", "Qwen3.5-4B")

def _set_qwen_vl_download_status(**updates):
    _QWEN_VL_DOWNLOAD_STATUS.update(updates)

def _reset_qwen_vl_download_status(status="idle"):
    _QWEN_VL_DOWNLOAD_STATUS.update({
        "status": status,
        "progress": 0,
        "current_file": "",
        "total_size": 0,
        "downloaded_size": 0,
        "error": "",
    })

@logged_operation("QwenVL", "download_asset")
def _download_qwen_vl_file(repo_id, filename, target_dir, revision=None):
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it or place "
            f"'{filename}' in '{target_dir}'."
        )
    os.makedirs(target_dir, exist_ok=True)
    log_stage("download", component="QwenVL", file=filename, repository=repo_id)
    _set_qwen_vl_download_status(
        status="downloading",
        current_file=filename,
        progress=0,
        total_size=0,
        downloaded_size=0,
        error="",
    )
    try:
        download_options = {}
        existing = os.path.join(target_dir, filename)
        if os.path.isfile(existing):
            try:
                _validate_gguf_file(existing, filename)
            except ValueError:
                download_options["force_download"] = True
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=target_dir,
            token=False,
            **download_options,
        )
        _validate_gguf_file(path, filename)
        downloaded_size = os.path.getsize(path)
        _set_qwen_vl_download_status(
            current_file=filename,
            progress=100,
            downloaded_size=downloaded_size,
            total_size=downloaded_size,
        )
        log_event("asset_ready", component="QwenVL", file=filename, bytes=downloaded_size)
        return path
    except Exception:
        raise

def _local_qwen_vl_assets(require_mmproj=True):
    model_path = _find_qwen_vl_model()
    if not model_path:
        raise FileNotFoundError(f"{QWEN_VL_MODEL_FILENAME} is missing from models/llm. Download it from Hugging Face to continue.")
    _validate_gguf_file(model_path, os.path.basename(model_path))
    mmproj_path = _find_qwen_vl_mmproj(model_path)
    if require_mmproj:
        if not mmproj_path:
            raise FileNotFoundError("Qwen3.5-4B vision projector is missing. Download it from Hugging Face to continue.")
        _validate_gguf_file(mmproj_path, os.path.basename(mmproj_path))
    return model_path, mmproj_path

def _ensure_qwen_vl_assets(allow_download=True, require_mmproj=True):
    try:
        return _local_qwen_vl_assets(require_mmproj)
    except (FileNotFoundError, ValueError):
        if not allow_download:
            raise

    with _QWEN_VL_DOWNLOAD_LOCK:
        try:
            assets = _local_qwen_vl_assets(require_mmproj)
        except (FileNotFoundError, ValueError):
            model_path = _find_qwen_vl_model()
            target_dir = os.path.dirname(model_path) if model_path else _qwen_vl_download_dir()
            try:
                try:
                    if model_path:
                        _validate_gguf_file(model_path, QWEN_VL_MODEL_FILENAME)
                except ValueError:
                    model_path = None
                if not model_path:
                    model_path = _download_qwen_vl_file(
                        QWEN_VL_MODEL_REPO_ID, QWEN_VL_MODEL_FILENAME, target_dir,
                        revision=QWEN_VL_MODEL_REVISION,
                    )
                if require_mmproj:
                    mmproj_path = _find_qwen_vl_mmproj(model_path)
                    try:
                        if mmproj_path:
                            _validate_gguf_file(mmproj_path, "Vision projector")
                    except ValueError:
                        mmproj_path = None
                    if not mmproj_path:
                        model_dir = os.path.dirname(model_path)
                        projector_dir = model_dir if "qwen3.5-4b" in os.path.basename(model_dir).lower() else os.path.join(model_dir, "Qwen3.5-4B")
                        _download_qwen_vl_file(
                            QWEN_VL_MODEL_REPO_ID, QWEN_VL_MMPROJ_FILENAME, projector_dir,
                            revision=QWEN_VL_MODEL_REVISION,
                        )
                assets = _local_qwen_vl_assets(require_mmproj)
            except Exception as exc:
                _set_qwen_vl_download_status(status="error", error=str(exc))
                raise RuntimeError(f"Failed to prepare Qwen3.5 assets from Hugging Face: {exc}") from exc
        _set_qwen_vl_download_status(status="completed", progress=100, current_file="Qwen3.5 assets ready", error="")
        return assets

def _qwen_vl_download_worker(require_mmproj=True):
    try:
        _ensure_qwen_vl_assets(require_mmproj=require_mmproj)
        _set_qwen_vl_download_status(status="completed", progress=100, current_file="Qwen3.5 assets ready", error="")
    except Exception as exc:
        _set_qwen_vl_download_status(status="error", error=str(exc))
        log_event("download_failed", component="QwenVL", level="error", error=str(exc))

def _start_qwen_vl_download(require_mmproj=True):
    if _QWEN_VL_DOWNLOAD_STATUS.get("status") == "downloading":
        return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS), status=409)
    try:
        model_path = _find_qwen_vl_model()
        mmproj_path = _find_qwen_vl_mmproj(model_path) if model_path else None
        if model_path and (mmproj_path or not require_mmproj):
            _validate_gguf_file(model_path, os.path.basename(model_path))
            if require_mmproj:
                _validate_gguf_file(mmproj_path, os.path.basename(mmproj_path))
            _set_qwen_vl_download_status(
                status="completed",
                progress=100,
                current_file="QwenVL assets ready",
                error="",
            )
            return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS))
    except Exception:
        pass

    _reset_qwen_vl_download_status("downloading")
    thread = threading.Thread(target=_qwen_vl_download_worker, args=(require_mmproj,), daemon=True)
    thread.start()
    return web.json_response({"status": "started"})


if server is not None and web is not None:
    @server.PromptServer.instance.routes.get("/vnccs/qwen_vl_model_status")
    async def qwen_vl_model_status(request):
        try:
            _ensure_qwen_vl_assets(allow_download=False, require_mmproj=request.rel_url.query.get("vision") != "false")
            return web.json_response({"ready": True, "model_name": QWEN_VL_MODEL_FILENAME})
        except (FileNotFoundError, ValueError) as exc:
            return web.json_response({"ready": False, "model_name": QWEN_VL_MODEL_FILENAME, "message": str(exc)})

    @server.PromptServer.instance.routes.get("/vnccs/qwen_vl_download_status")
    async def qwen_vl_download_status(request):
        return web.json_response(dict(_QWEN_VL_DOWNLOAD_STATUS))

    @server.PromptServer.instance.routes.post("/vnccs/qwen_vl_download_model")
    @privileged_route
    async def qwen_vl_download_model(request):
        require_mmproj = request.rel_url.query.get("vision") != "false"
        return _start_qwen_vl_download(require_mmproj)
