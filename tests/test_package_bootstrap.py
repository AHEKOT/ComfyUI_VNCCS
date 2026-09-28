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
    monkeypatch.setitem(sys.modules, "vnccs", module)
    monkeypatch.setitem(sys.modules, "vnccs.nodes", namespace)

    spec.loader.exec_module(module)

    assert "TestNode" in module.NODE_CLASS_MAPPINGS
    assert sys.modules["vnccs.nodes"].__file__ == str(nodes / "__init__.py")
