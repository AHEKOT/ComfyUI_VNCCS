"""Public Git/LFS transport fixture; production validation and storage stay real."""

import types
import sys
from pathlib import Path

import pytest


@pytest.fixture
def git_repository(monkeypatch):
    state = types.SimpleNamespace(files={}, modes={}, calls=[], lfs={}, lfs_calls=[], revision="a" * 40)

    class Repository:
        def __init__(self):
            self.files = dict(state.files)
            self.revision = state.revision.encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def head(self):
            return self.revision
        def __getitem__(self, key):
            return types.SimpleNamespace(tree=b"tree") if key == self.revision else types.SimpleNamespace(data=self.files[key.decode()])

    def clone(url, target, **kwargs):
        state.calls.append({"repo_id": url.removeprefix("https://huggingface.co/"), "target": target, **kwargs})
        return Repository()

    def lookup(getter, tree, name):
        getter(name)
        return state.modes.get(name.decode(), 0o100644), name

    class Config:
        def set(self, *args):
            pass

    modules = {"dulwich": types.SimpleNamespace(porcelain=types.SimpleNamespace(clone=clone)),
               "dulwich.config": types.SimpleNamespace(ConfigDict=Config),
               "dulwich.object_store": types.SimpleNamespace(tree_lookup_path=lookup)}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def lfs_download(**kwargs):
        state.lfs_calls.append(kwargs)
        file = Path(kwargs["cache_dir"]) / kwargs["revision"] / kwargs["filename"]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(state.lfs[kwargs["filename"]])
        return str(file)

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", lfs_download)
    return state
