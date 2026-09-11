"""Download progress and atomic installation regressions, without network access."""

import queue
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from nodes import vnccs_control_center as cc


class FakeTqdm:
    def __init__(self, *, total=None, initial=0, disable=None, **kwargs):
        self.total = total
        self.n = initial
        self.disable = disable is not False

    def update(self, n=1):
        if not self.disable:
            self.n += n

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def hub_progress(monkeypatch):
    def context(**kwargs):
        existing = kwargs.pop("_tqdm_bar", None)
        return nullcontext(existing) if existing is not None else FakeTqdm(**kwargs)

    module = SimpleNamespace(
        tqdm=FakeTqdm,
        _get_progress_bar_context=context,
        constants=SimpleNamespace(DOWNLOAD_CHUNK_SIZE=10 * 1024 * 1024, HF_HUB_DISABLE_XET=False,
                                  HF_HUB_ENABLE_HF_TRANSFER=True),
        is_xet_available=lambda: True,
    )
    monkeypatch.setattr(sys.modules["huggingface_hub"], "file_download", module, raising=False)
    monkeypatch.setattr(cc, "_DOWNLOAD_STATUS", {})
    return module


@pytest.mark.parametrize("mode", ["modern", "legacy_direct", "legacy_context"])
@pytest.mark.parametrize("disabled", [False, True])
def test_reports_bytes_during_download_and_restores_hub(monkeypatch, hub_progress, mode, disabled):
    original_context = hub_progress._get_progress_bar_context
    callbacks = []

    def transfer(*, repo_id, filename, revision, token, tqdm_class=None):
        assert (repo_id, filename, revision, token) == ("public/models", "model.gguf", "pinned", False)
        if mode == "modern":
            assert hub_progress.tqdm is FakeTqdm
            bar = tqdm_class(unit="B", total=100, initial=20, disable=disabled)
        elif mode == "legacy_direct":
            bar = hub_progress.tqdm(unit="B", total=100, initial=20, disable=disabled)
        else:
            bar = hub_progress._get_progress_bar_context(total=100, initial=20, log_level=0)
        with bar:
            assert cc._DOWNLOAD_STATUS["model"]["progress"] == 20
            bar.update(30)
            assert cc._DOWNLOAD_STATUS["model"]["progress"] == 50
            assert "Downloading: 50%" in cc._DOWNLOAD_STATUS["model"]["message"]
            callbacks.append(bar)

            # A second Xet transfer bar and another thread must not replace the
            # current model's reconstructed-byte count.
            extra = (tqdm_class or hub_progress.tqdm)(unit="B", total=500, disable=disabled)
            extra.update(400)
            extra.close()
            def unrelated_download():
                if mode == "legacy_context":
                    hub_progress._get_progress_bar_context(total=10, log_level=0).update(9)
                else:
                    hub_progress.tqdm(unit="B", total=10).update(9)

            other = threading.Thread(target=unrelated_download)
            other.start()
            other.join()
            assert cc._DOWNLOAD_STATUS["model"]["progress"] == 50
            bar.update(50)
        return "/cache/model.gguf"

    def legacy(*, repo_id, filename, revision, token):
        return transfer(repo_id=repo_id, filename=filename, revision=revision, token=token)

    if mode == "legacy_direct":
        del hub_progress._get_progress_bar_context
    monkeypatch.setattr(cc, "hf_hub_download", transfer if mode == "modern" else legacy)
    assert cc._download_with_progress("model", "public/models", "model.gguf", "pinned") == "/cache/model.gguf"
    assert hub_progress.tqdm is FakeTqdm
    if mode != "legacy_direct":
        assert hub_progress._get_progress_bar_context is original_context
    final = cc._DOWNLOAD_STATUS["model"].copy()
    assert final["progress"] == 100
    callbacks[0].update(5)
    assert cc._DOWNLOAD_STATUS["model"] == final


def test_unknown_total_and_retry_do_not_invent_percent(monkeypatch, hub_progress):
    def transfer(**kwargs):
        with hub_progress._get_progress_bar_context(total=None, log_level=0) as bar:
            bar.update(1024 * 1024)
            assert cc._DOWNLOAD_STATUS["model"]["message"] == "Downloading: 1.0 MB"
            assert "progress" not in cc._DOWNLOAD_STATUS["model"]
        with hub_progress._get_progress_bar_context(total=100, initial=40, log_level=0) as bar:
            bar.update(10)
            assert cc._DOWNLOAD_STATUS["model"]["progress"] == 50
        raise OSError("Interrupted transfer")

    original_context = hub_progress._get_progress_bar_context
    monkeypatch.setattr(cc, "hf_hub_download", transfer)
    with pytest.raises(OSError, match="Interrupted transfer"):
        cc._download_with_progress("model", "public/models", "model.gguf")
    assert hub_progress.tqdm is FakeTqdm
    assert hub_progress._get_progress_bar_context is original_context


@pytest.mark.parametrize("fail", [False, True])
def test_http_progress_remains_live_and_transport_override_is_scoped(monkeypatch, hub_progress, fail):
    original_constants = hub_progress.constants
    original_xet_available = hub_progress.is_xet_available
    other_thread_state = []
    snapshots = []
    callbacks = []

    def transfer(*, tqdm_class, token, **kwargs):
        assert token is False
        assert hub_progress.is_xet_available() is False
        assert hub_progress.constants.HF_HUB_DISABLE_XET is True
        assert hub_progress.constants.HF_HUB_ENABLE_HF_TRANSFER is False
        chunk_size = hub_progress.constants.DOWNLOAD_CHUNK_SIZE
        assert chunk_size == 1024 * 1024

        def unrelated_download():
            other_thread_state.append((hub_progress.is_xet_available(),
                hub_progress.constants.HF_HUB_DISABLE_XET,
                hub_progress.constants.HF_HUB_ENABLE_HF_TRANSFER,
                hub_progress.constants.DOWNLOAD_CHUNK_SIZE))

        thread = threading.Thread(target=unrelated_download)
        thread.start()
        thread.join()
        callbacks.append((hub_progress.constants, hub_progress.is_xet_available))
        # Resume at the reported 13% and consume streamed HTTP chunks.
        with tqdm_class(unit="B", total=100 * chunk_size, initial=13 * chunk_size, disable=True) as bar:
            for _ in range(87):
                bar.update(chunk_size)
                snapshots.append(cc._DOWNLOAD_STATUS["model"]["progress"])
        if fail:
            raise OSError("Download interrupted")
        return "/cache/model.safetensors"

    monkeypatch.setattr(cc, "hf_hub_download", transfer)
    if fail:
        with pytest.raises(OSError, match="Download interrupted"):
            cc._download_with_progress("model", "public/models", "model.safetensors")
    else:
        cc._download_with_progress("model", "public/models", "model.safetensors")
    assert snapshots == pytest.approx(list(range(14, 101)))
    assert other_thread_state == [(True, False, True, 10 * 1024 * 1024)]
    assert hub_progress.constants is original_constants
    assert hub_progress.is_xet_available is original_xet_available
    constants, available = callbacks[0]
    assert constants.DOWNLOAD_CHUNK_SIZE == original_constants.DOWNLOAD_CHUNK_SIZE
    assert available() is True


@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("cached_hit", [True, False])
def test_worker_installs_only_after_validation(monkeypatch, hub_progress, tmp_path, valid, cached_hit):
    cached = tmp_path / "cache.gguf"
    cached.write_bytes(b"GGUF" + b"x" * 1024 if valid else b"<html>not a model</html>")
    target = tmp_path / "models" / "model.gguf"
    target.parent.mkdir()
    target.write_bytes(b"old model")
    tasks = queue.Queue()
    tasks.put(("public/models", "model", {"local_path": "models/unet/model.gguf", "hf_path": "model.gguf"}))
    tasks.put(None)
    monkeypatch.setattr(cc, "_DOWNLOAD_QUEUE", tasks)
    monkeypatch.setattr(cc, "_resolve_model_download_path", lambda path: str(target))
    versions = []
    monkeypatch.setattr(cc, "update_installed_version", lambda *args: versions.append(args))

    def transfer(*, tqdm_class, **kwargs):
        if cached_hit:
            return str(cached)
        with tqdm_class(unit="B", total=100, disable=True) as bar:
            bar.update(50)
            assert cc._DOWNLOAD_STATUS["model"]["progress"] == 50
            assert target.read_bytes() == b"old model"
        return str(cached)

    original_validate = cc._validate_downloaded_model_file

    def validate(path, name):
        assert cc._DOWNLOAD_STATUS["model"]["status"] == "downloading"
        assert cc._DOWNLOAD_STATUS["model"]["message"] == "Validating..."
        assert target.read_bytes() == b"old model"
        return original_validate(path, name)

    monkeypatch.setattr(cc, "hf_hub_download", transfer)
    monkeypatch.setattr(cc, "_validate_downloaded_model_file", validate)
    cc._download_worker_loop()
    assert cc._DOWNLOAD_STATUS["model"]["status"] == ("success" if valid else "error")
    assert target.read_bytes() == (cached.read_bytes() if valid else b"old model")
    assert bool(versions) == valid
    assert list(target.parent.iterdir()) == [target]
