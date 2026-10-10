"""Load Creator V2's storage logic without enabling tensor tests in CI."""

import importlib.util
import sys
import types

from conftest import _preload_node


def load_creator_storage(monkeypatch):
    with monkeypatch.context() as patch:
        if importlib.util.find_spec("torch") is None:
            patch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        return _preload_node("character_creator_v2")
