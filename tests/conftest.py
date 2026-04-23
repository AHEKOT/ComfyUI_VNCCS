"""Stubs for ComfyUI runtime modules not available outside of ComfyUI."""

import importlib.util
import os
import sys
import types


def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# ── comfy hierarchy ───────────────────────────────────────────────────────────
comfy = _make_stub("comfy")
comfy_sd = _make_stub("comfy.sd")
comfy_utils = _make_stub("comfy.utils")
comfy_samplers = _make_stub("comfy.samplers")
comfy.sd = comfy_sd
comfy.utils = comfy_utils
comfy.samplers = comfy_samplers

class _KSamplerStub:
    SAMPLERS = ["euler", "euler_a", "heun", "dpm_2", "dpm_2_ancestral"]
    SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform"]

comfy_samplers.KSampler = _KSamplerStub

# ── folder_paths ──────────────────────────────────────────────────────────────
folder_paths = _make_stub("folder_paths")
folder_paths.base_path = "/tmp/comfyui"
folder_paths.get_output_directory = lambda: "/tmp/comfyui/output"
folder_paths.get_filename_list = lambda _: []
folder_paths.get_full_path = lambda *_: None
folder_paths.get_folder_paths = lambda _: []
folder_paths.add_model_folder_path = lambda *a, **kw: None

# ── server / aiohttp ──────────────────────────────────────────────────────────
server = _make_stub("server")

class _FakePromptServer:
    instance = None
    routes = types.SimpleNamespace(get=lambda *a, **kw: (lambda f: f), post=lambda *a, **kw: (lambda f: f))
_FakePromptServer.instance = _FakePromptServer()
server.PromptServer = _FakePromptServer

aiohttp = _make_stub("aiohttp")
aiohttp.web = _make_stub("aiohttp.web")

# ── other optional deps ───────────────────────────────────────────────────────
_make_stub("requests")
_hf_hub = _make_stub("huggingface_hub")
_hf_hub.hf_hub_download = lambda *a, **kw: None
_hf_hub.hf_hub_url = lambda *a, **kw: ""

# ── nodes package with parent context for relative imports ────────────────────
# Some node files use `from ..utils import` which requires `nodes` to appear as a
# subpackage. We create a fake parent `_vnccs` so those imports resolve correctly.

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_vnccs_parent = types.ModuleType("_vnccs")
_vnccs_parent.__path__ = [_PROJECT_ROOT]
_vnccs_parent.__package__ = "_vnccs"
sys.modules["_vnccs"] = _vnccs_parent

# Load utils.py under the fake parent so `from ..utils import` resolves to it.
_utils_spec = importlib.util.spec_from_file_location(
    "_vnccs.utils", os.path.join(_PROJECT_ROOT, "utils.py")
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
_utils_mod.__package__ = "_vnccs"
sys.modules["_vnccs.utils"] = _utils_mod
sys.modules["utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# Register nodes as a namespace package under _vnccs.
_nodes_ns = types.ModuleType("_vnccs.nodes")
_nodes_ns.__path__ = [os.path.join(_PROJECT_ROOT, "nodes")]
_nodes_ns.__package__ = "_vnccs.nodes"
sys.modules["_vnccs.nodes"] = _nodes_ns
sys.modules["nodes"] = _nodes_ns


def _preload_node(basename):
    """Load nodes/<basename>.py with __package__='_vnccs.nodes' so relative imports work.

    The loaded module is registered under both '_vnccs.nodes.<basename>' and
    'nodes.<basename>' so that `from nodes.<basename> import X` finds it.
    """
    file_path = os.path.join(_PROJECT_ROOT, "nodes", f"{basename}.py")
    full_name = f"_vnccs.nodes.{basename}"
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "_vnccs.nodes"
    sys.modules[full_name] = mod
    sys.modules[f"nodes.{basename}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load node files that use `from ..utils import` (relative double-dot imports).
# Other node files use absolute imports and load fine on demand.
for _basename in ("character_creator", "dataset_generator"):
    try:
        _preload_node(_basename)
    except Exception:
        # If the file itself fails (e.g., missing optional dep), tests importing
        # it will also fail — that's the expected behaviour (or importorskip).
        pass
