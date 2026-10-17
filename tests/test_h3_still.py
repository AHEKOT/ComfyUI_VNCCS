"""H3 compatibility checks run without PyTorch; tensor checks skip narrowly."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from nodes import h3_still as h3
from nodes import vnccs_control_center as cc


@pytest.mark.parametrize("version", ["0.39.0", "0.1.0", "unknown"])
def test_native_support_accepts_git_builds_and_backports(monkeypatch, version):
    monkeypatch.setitem(sys.modules, "comfyui_version", SimpleNamespace(__version__=version))
    status = h3.h3_runtime_status()
    assert status["version"] == version
    assert status["h3_still_supported"] is True
    h3.require_h3_still_support()


@pytest.mark.parametrize("missing", ["module", "constructor", "frame_slice"])
def test_old_native_decoder_requires_update_even_with_new_version_label(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, "comfyui_version", SimpleNamespace(__version__="99.0.0"))
    class OldVAE:
        def __init__(self, still_frame=None):
            pass

        def _adaptive_decode(self, z, frames=slice(None)):
            pass

    if missing == "module":
        monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.vae", None)
    else:
        if missing == "constructor":
            OldVAE.__init__ = lambda self: None
        else:
            OldVAE._adaptive_decode = lambda self, z: None
        monkeypatch.setattr(sys.modules["comfy.ldm.minimax.vae"], "MiniMaxH3VideoVAE", OldVAE)
    assert h3.h3_runtime_status()["h3_still_supported"] is False
    with pytest.raises(RuntimeError, match=r"PR #16881.*restart.*99\.0\.0"):
        h3.require_h3_still_support()


@pytest.mark.parametrize("kind", ["MiniMaxH3", "H3"])
def test_h3_is_blocked_before_loading_models_on_old_comfyui(monkeypatch, kind):
    monkeypatch.setattr(h3, "h3_runtime_status", lambda: {"version": "0.38.0", "h3_still_supported": False})
    monkeypatch.setattr(cc, "_get_cc_config", lambda *args: {"models": [], "lora": []})
    def load(*args, **kwargs):
        pytest.fail("Unsupported H3 must stop before loading models")
    monkeypatch.setattr(cc, "_load_model_block", load)
    with pytest.raises(RuntimeError, match="Update ComfyUI"):
        cc._build_control_center_pipe("demo/repo", {"active_kind": kind, "selected_type": "custom"})


def test_module_status_exposes_running_comfyui_support(monkeypatch):
    monkeypatch.setattr(cc.web, "json_response", lambda data: data, raising=False)
    monkeypatch.setattr(cc, "_registered_node_class", lambda spec: None)
    result = asyncio.run(cc.vnccs_module_status(SimpleNamespace()))
    assert result["comfyui"] == h3.h3_runtime_status()
    assert "PR #16881" in result["comfyui"]["h3_update_message"]


@pytest.mark.parametrize("samples", [None, {}, {"samples": object()}, {"samples": SimpleNamespace(shape=(1, 24, 0, 4, 4))},
                                         {"samples": SimpleNamespace(shape=(1, 4, 2, 4, 4))}])
def test_invalid_latents_fail_before_decode(samples):
    with pytest.raises(ValueError, match="MiniMax H3"):
        h3.h3_still_samples(samples)


@pytest.mark.parametrize("nested", [False, True])
def test_first_temporal_latent_preserves_batch_and_owns_only_still_storage(nested):
    torch = pytest.importorskip("torch")
    video = torch.arange(2 * 24 * 2 * 4 * 4).reshape(2, 24, 2, 4, 4)
    audio = torch.ones(2, 32, 2, 8)
    packed = SimpleNamespace(is_nested=True, unbind=lambda: (video, audio)) if nested else video
    original = {"samples": packed, "metadata": "preserved"}
    still = h3.h3_still_samples(original)
    assert original["samples"] is packed
    assert still["metadata"] == "preserved"
    assert still["samples"].shape == (2, 24, 1, 4, 4)
    torch.testing.assert_close(still["samples"], video[:, :, :1])
    assert still["samples"].untyped_storage().nbytes() == still["samples"].numel() * still["samples"].element_size()
    assert still["samples"].untyped_storage().data_ptr() != video.untyped_storage().data_ptr()
