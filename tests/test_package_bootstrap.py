"""Exercise the package entrypoint used by ComfyUI's custom node loader."""

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("missing", [
    (),
    ("character_presets_v2.json",),
    ("character_presets_v2.json", "character_styles.json", "anima_styles.json"),
])
def test_actual_node_loader_handles_missing_catalog_files(tmp_path, monkeypatch, missing):
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / f"vnccs_catalog_test_{len(missing)}"
    shutil.copytree(root / "nodes", package / "nodes", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("__init__.py", "utils.py", "operation_logger.py"):
        shutil.copyfile(root / name, package / name)
    template = package / "character_template"
    template.mkdir()
    for name in ("character_presets_v2.json", "character_styles.json", "anima_styles.json"):
        if name not in missing:
            shutil.copyfile(root / "character_template" / name, template / name)
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(package.name)
    assert module.NODE_DISPLAY_NAME_MAPPINGS["VNCCSChromaKey"] == "VNCCS Chroma Key"
    if missing:
        creator = module.nodes.character_creator_v2
        assert creator.race_prompt("elf, custom markings") == "elf, custom markings"
        assert not creator.SKIN_COLOR_OPTIONS
        preview = module.nodes.style_preview_test
        info, style = preview._random_character_info("qi2", "Blue", 24, 24, False, {})
        assert info["race"] == "human" and info["hair"] == "black hair, long hair"
        preview._set_observed_breast_size(info, "B-Cup")
        if "character_styles.json" in missing:
            assert style == "none"
            assert creator._character_style_prompt({"style": "saved", "style_prompt": "ink drawing"}) == "ink drawing"
        assert all(not (template / name).exists() for name in missing)


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
