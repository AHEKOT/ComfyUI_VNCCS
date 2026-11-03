"""Clothes preview model ownership across success, failure and interruption."""

import gc
import sys
import types
import weakref

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

import utils
from _vnccs.nodes import character_generator as cg, clothes_designer as cd


class Interrupted(BaseException):
    """ComfyUI cancellation deliberately bypasses except Exception."""


class Weights:
    pass


class Patcher:
    def __init__(self, parent=None):
        self.parent = parent
        self.model = parent.model if parent else Weights()


class LoadedModel:
    """ComfyUI's weak patcher/parent ownership, without GPU allocation."""

    def __init__(self, model):
        self.real_model = weakref.ref(model.model)
        self._set_model(model)

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self.finalizer = weakref.finalize(model, self._switch_parent)
            self.finalizer.atexit = False

    def _switch_parent(self):
        parent = self._parent_model()
        if parent is not None:
            self._set_model(parent)

    def is_dead(self):
        return self._model() is None and self.real_model() is not None


@pytest.fixture
def preview_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    utils.save_costume_info("Example", "Suit", {})
    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "blue").save(reference)
    monkeypatch.setattr(cd, "get_latest_sprite_path", lambda *args: str(reference))
    monkeypatch.setattr(cd, "sheets_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(cd, "_resolve_pipe_clothes_core_lora", lambda pipe: "core.safetensors")
    monkeypatch.setattr(cd, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cd, "_apply_lora_standard", lambda model, *args: (Patcher(model), None))
    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_qi2_prepare_model",
                        lambda self, model, *args: (Patcher(model), False))
    monkeypatch.setattr(cd.server.PromptServer.instance, "send_sync", lambda *args: None, raising=False)

    base = Patcher()
    other = Patcher()
    unrelated = LoadedModel(other)
    state = types.SimpleNamespace(
        base=base, other=other, loaded=[unrelated], unrelated=unrelated,
        released=[], clone_refs=[], failure=None, error_type=Interrupted,
    )

    def fail_at(stage):
        if state.failure == stage:
            try:
                raise state.error_type("generation stopped")
            except BaseException:
                # A runtime error frame can retain its traceback until cyclic GC.
                retained_traceback = sys.exc_info()[2]
                assert retained_traceback is not None
                raise

    def encode(*args, **kwargs):
        fail_at("encode")
        return "positive", "negative", {"samples": torch.zeros(1)}

    def sample(self, model, *args, **kwargs):
        state.loaded.append(LoadedModel(model))
        state.clone_refs.extend((weakref.ref(model), weakref.ref(model.parent)))
        fail_at("sample")
        return {"samples": torch.zeros(1)}

    def decode(*args, **kwargs):
        fail_at("decode")
        return (torch.zeros(1, 16, 16, 3),)

    def unload(model, *, all_devices):
        assert model is base and all_devices is True
        # Unload while the wrappers are still reachable through the failing stack.
        assert all(not item.is_dead() for item in state.loaded)
        state.loaded[:] = [item for item in state.loaded if item.real_model() is not model.model]
        state.released.append(model)

    management = types.ModuleType("comfy.model_management")
    management.unload_model_and_clones = unload
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(sys.modules["comfy"], "model_management", management, raising=False)
    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_qi2_encode", encode)
    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_qi2_sample", sample)
    monkeypatch.setattr(cd, "_call_comfy_node", decode)
    state.management = management
    state.pipe = types.SimpleNamespace(model=base, clip=object(), vae=object(), model_entry={"kind": "QI2"})
    state.data = {"character": "Example", "costume": "Suit",
                  "gen_settings": {"qi2_prompt_enhancement": False}}
    return state


@pytest.mark.parametrize("node_id", ["753", "api_preview"])
@pytest.mark.parametrize("stage", ["encode", "sample", "decode"])
@pytest.mark.parametrize("error_type", [Interrupted, RuntimeError])
def test_failed_preview_releases_only_its_model_and_can_retry(preview_runtime, node_id, stage, error_type):
    state = preview_runtime
    state.failure, state.error_type = stage, error_type

    def run_failure():
        with pytest.raises(error_type, match="generation stopped"):
            cd.ClothesDesigner().process(state.pipe, state.data, node_id)

    run_failure()
    gc.collect()
    assert state.released == [state.base]
    assert state.loaded == [state.unrelated]
    assert all(ref() is None for ref in state.clone_refs)
    assert state.pipe.model is state.base

    state.failure = None
    image, _, _ = cd.ClothesDesigner().process(state.pipe, state.data, node_id)
    gc.collect()
    assert image.shape == (1, 16, 16, 3)
    assert state.released == [state.base], "Successful previews must retain normal model reuse."
    assert all(not item.is_dead() for item in state.loaded)


def test_unload_failure_preserves_original_interruption(preview_runtime, monkeypatch):
    state = preview_runtime
    state.failure = "decode"

    def fail_unload(*args, **kwargs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(state.management, "unload_model_and_clones", fail_unload)
    with pytest.raises(Interrupted, match="generation stopped"):
        cd.ClothesDesigner().process(state.pipe, state.data, "753")


def test_missing_unload_reproduces_orphan_without_masking_interruption(preview_runtime, monkeypatch):
    state = preview_runtime
    state.failure = "decode"
    monkeypatch.delattr(state.management, "unload_model_and_clones")
    def run_failure():
        with pytest.raises(Interrupted, match="generation stopped"):
            cd.ClothesDesigner().process(state.pipe, state.data, "753")

    run_failure()
    gc.collect()
    assert any(item.is_dead() for item in state.loaded)
