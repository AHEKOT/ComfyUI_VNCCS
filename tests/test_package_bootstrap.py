"""Exercise the package entrypoint used by ComfyUI's custom node loader."""

import importlib.util
import sys
import types
from pathlib import Path


def test_namespace_nodes_placeholder_is_replaced(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    package = tmp_path / "vnccs"
    nodes = package / "nodes"
    nodes.mkdir(parents=True)
    (package / "__init__.py").write_text((project_root / "__init__.py").read_text())
    (nodes / "__init__.py").write_text(
        'NODE_CLASS_MAPPINGS = {"TestNode": object()}\n'
        'NODE_DISPLAY_NAME_MAPPINGS = {"TestNode": "Test Node"}\n'
    )

    spec = importlib.util.spec_from_file_location(
        "vnccs", package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    namespace = types.ModuleType("vnccs.nodes")
    namespace.__path__ = [str(nodes)]
    comfy_runtime = types.ModuleType("comfy")
    comfy_runtime.__file__ = "/comfy/comfy/__init__.py"
    folder_paths_runtime = types.ModuleType("folder_paths")
    folder_paths_runtime.__file__ = "/comfy/folder_paths.py"
    torch_runtime = types.ModuleType("torch")
    torch_runtime.__file__ = "/comfy/torch/__init__.py"
    monkeypatch.setitem(sys.modules, "comfy", comfy_runtime)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_runtime)
    monkeypatch.setitem(sys.modules, "torch", torch_runtime)
    monkeypatch.setitem(sys.modules, "vnccs", module)
    monkeypatch.setitem(sys.modules, "vnccs.nodes", namespace)

    spec.loader.exec_module(module)

    assert "TestNode" in module.NODE_CLASS_MAPPINGS
    assert sys.modules["vnccs.nodes"].__file__ == str(nodes / "__init__.py")


def test_package_discovery_does_not_import_nodes_without_comfyui(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    package = tmp_path / "vnccs_analysis"
    nodes = package / "nodes"
    nodes.mkdir(parents=True)
    (package / "__init__.py").write_text((project_root / "__init__.py").read_text())
    (nodes / "__init__.py").write_text('raise AssertionError("nodes package was imported")\n')

    original_find_spec = importlib.util.find_spec

    def find_spec_without_comfyui(name, *args, **kwargs):
        if name in {"comfy", "folder_paths"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "comfy", raising=False)
    monkeypatch.delitem(sys.modules, "folder_paths", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_comfyui)

    spec = importlib.util.spec_from_file_location(
        "vnccs_analysis", package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vnccs_analysis", module)
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS == {}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {}
    assert "vnccs_analysis.nodes" not in sys.modules
