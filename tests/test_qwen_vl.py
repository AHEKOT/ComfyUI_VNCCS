import os
import types

import pytest

from nodes.qwen_vl import get_qwen_vl_chat_handler


@pytest.mark.parametrize("vision,model,projector,active,status,workers", [
    (True, False, False, False, 200, 1),
    (True, True, False, False, 200, 1),
    (False, True, False, False, 200, 0),
    (True, True, True, False, 200, 0),
    (True, True, True, True, 409, 0),
])
def test_shared_download_start_checks_required_assets_and_active_worker(monkeypatch, vision, model, projector, active, status, workers):
    from nodes import qwen_vl

    started, validated = [], []
    monkeypatch.setattr(qwen_vl, "_QWEN_VL_DOWNLOAD_STATUS", {"status": "downloading" if active else "idle"})
    monkeypatch.setattr(qwen_vl, "_find_qwen_vl_model", lambda: "model.gguf" if model else None)
    monkeypatch.setattr(qwen_vl, "_find_qwen_vl_mmproj", lambda path: "mmproj.gguf" if projector else None)
    monkeypatch.setattr(qwen_vl, "_validate_gguf_file", lambda path, name: validated.append(path))
    monkeypatch.setattr(qwen_vl.threading, "Thread", lambda **kwargs:
                        types.SimpleNamespace(start=lambda: started.append(kwargs)))
    monkeypatch.setattr(qwen_vl.web, "json_response", lambda data, status=200:
                        types.SimpleNamespace(data=data, status=status), raising=False)
    response = qwen_vl._start_qwen_vl_download(vision)
    assert response.status == status
    assert len(started) == workers
    if workers:
        assert response.data == {"status": "started"}
        assert started[0]["args"] == (vision,)
    elif not active:
        assert response.data["status"] == "completed"
        assert validated == (["model.gguf", "mmproj.gguf"] if vision else ["model.gguf"])


def test_selects_qwen35_even_when_legacy_handlers_are_available():
    handler = object()
    llama_cpp = types.SimpleNamespace(llama_chat_format=types.SimpleNamespace(
        Qwen35ChatHandler=handler, Qwen25VLChatHandler=object(), Qwen2VLChatHandler=object(),
    ))
    assert get_qwen_vl_chat_handler(llama_cpp) is handler


@pytest.mark.parametrize("name", ["Qwen25VLChatHandler", "Qwen2VLChatHandler", "Qwen3VLChatHandler", "Llava15ChatHandler"])
def test_rejects_incompatible_handlers(name):
    llama_cpp = types.SimpleNamespace(llama_chat_format=types.SimpleNamespace(**{name: object()}))
    with pytest.raises(RuntimeError, match="Qwen3.5 requires Qwen35ChatHandler"):
        get_qwen_vl_chat_handler(llama_cpp)


def test_text_wizard_disables_thinking_in_model_template(monkeypatch):
    import sys
    from nodes.qwen_vl import configure_qwen_text_chat
    captured = {}
    handler = object()
    class Formatter:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def to_chat_handler(self):
            return handler
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", types.SimpleNamespace(Jinja2ChatFormatter=Formatter))
    llm = types.SimpleNamespace(metadata={"tokenizer.chat_template": "model template"})
    configure_qwen_text_chat(llm)
    assert llm.chat_handler is handler
    assert captured["template"] == "{% set enable_thinking = false %}model template"
    assert captured["eos_token"] == "<|im_end|>"


from nodes import qwen_vl

@pytest.mark.parametrize("content, expected", [
    ('{"hair": "black"}', {"hair": "black"}),
    ('```json\n{"top": "coat"}\n```', {"top": "coat"}),
    ('```\n{"top": "coat"}\n```', {"top": "coat"}),
    ('Response: {"top": "coat"}', {"top": "coat"}),
    ('[]', None), ('broken JSON', None), (None, None),
])
def test_wizard_json_fallback_without_repair(monkeypatch, content, expected):
    import sys
    monkeypatch.setitem(sys.modules, "json_repair", None)
    assert qwen_vl.parse_wizard_json(content) == expected


def test_wizard_json_repair_unwraps_first_object(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "json_repair", types.SimpleNamespace(loads=lambda _: [{"hair": "black"}]))
    assert qwen_vl.parse_wizard_json("repaired response") == {"hair": "black"}

def test_qwen_download_disables_hub_credentials(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\0" * (1024 * 1024))
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(model_path)

    monkeypatch.setattr(qwen_vl, "hf_hub_download", fake_download)

    result = qwen_vl._download_qwen_vl_file(
        "public/repository",
        "model.gguf",
        str(tmp_path),
        revision="pinned-revision",
    )

    assert result == str(model_path)
    assert captured == {
        "repo_id": "public/repository",
        "filename": "model.gguf",
        "revision": "pinned-revision",
        "local_dir": str(tmp_path),
        "token": False,
    }


@pytest.mark.parametrize("directory", ["llm", "LLM", "llm/Qwen3.5-4B"])
def test_qwen35_text_wizard_reuses_local_model_without_projector(tmp_path, monkeypatch, directory):
    monkeypatch.setattr(qwen_vl.folder_paths, "models_dir", str(tmp_path))
    model = tmp_path / directory / qwen_vl.QWEN_VL_MODEL_FILENAME
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF" + bytes(1024 * 1024))
    monkeypatch.setattr(qwen_vl, "hf_hub_download", lambda **kwargs: pytest.fail("Local model must not download"))
    found, projector = qwen_vl._ensure_qwen_vl_assets(allow_download=False, require_mmproj=False)
    assert os.path.samefile(found, model)
    assert projector is None


def test_qwen35_missing_model_does_not_download_without_consent(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_vl.folder_paths, "models_dir", str(tmp_path))
    legacy = tmp_path / "llm" / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    legacy.parent.mkdir()
    legacy.write_bytes(b"GGUF" + bytes(1024 * 1024))
    monkeypatch.setattr(qwen_vl, "hf_hub_download", lambda **kwargs: pytest.fail("Missing model must only prompt"))
    with pytest.raises(FileNotFoundError, match="Qwen3.5-4B-Q8_0.gguf"):
        qwen_vl._ensure_qwen_vl_assets(allow_download=False, require_mmproj=False)


@pytest.mark.parametrize("vision", [False, True])
def test_qwen35_download_uses_pinned_public_assets(tmp_path, monkeypatch, vision):
    monkeypatch.setattr(qwen_vl.folder_paths, "models_dir", str(tmp_path))
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF" + bytes(1024 * 1024))
        return str(path)
    from pathlib import Path
    monkeypatch.setattr(qwen_vl, "hf_hub_download", download)
    model, projector = qwen_vl._ensure_qwen_vl_assets(require_mmproj=vision)
    assert len(calls) == (2 if vision else 1)
    assert model.endswith("llm/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf")
    assert bool(projector) == vision
    for call in calls:
        assert call["token"] is False
        assert call["revision"] == qwen_vl.QWEN_VL_MODEL_REVISION
        assert call["repo_id"] == "unsloth/Qwen3.5-4B-GGUF"
    qwen_vl._ensure_qwen_vl_assets(allow_download=False, require_mmproj=vision)
    assert len(calls) == (2 if vision else 1)


def test_qwen35_does_not_reuse_ambiguous_legacy_projector(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_vl.folder_paths, "models_dir", str(tmp_path))
    directory = tmp_path / "llm"
    directory.mkdir()
    model = directory / qwen_vl.QWEN_VL_MODEL_FILENAME
    model.write_bytes(b"GGUF" + bytes(1024 * 1024))
    (directory / "mmproj-F16.gguf").write_bytes(b"GGUF" + bytes(1024 * 1024))
    with pytest.raises(FileNotFoundError, match="vision projector"):
        qwen_vl._ensure_qwen_vl_assets(allow_download=False)
    projector = directory / "mmproj-Qwen3.5-4B-F16.gguf"
    projector.write_bytes(b"GGUF" + bytes(1024 * 1024))
    assert qwen_vl._ensure_qwen_vl_assets(allow_download=False) == (str(model), str(projector))


def test_qwen_status_check_does_not_finish_an_active_download(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_vl.folder_paths, "models_dir", str(tmp_path))
    model = tmp_path / "llm" / qwen_vl.QWEN_VL_MODEL_FILENAME
    model.parent.mkdir()
    model.write_bytes(b"GGUF" + bytes(1024 * 1024))
    monkeypatch.setattr(qwen_vl, "_QWEN_VL_DOWNLOAD_STATUS", {"status": "downloading", "progress": 42})
    qwen_vl._ensure_qwen_vl_assets(allow_download=False, require_mmproj=False)
    assert qwen_vl._QWEN_VL_DOWNLOAD_STATUS == {"status": "downloading", "progress": 42}


def test_qwen_download_repairs_invalid_model_after_confirmation(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(qwen_vl.folder_paths, "models_dir", str(tmp_path))
    model = tmp_path / "llm" / qwen_vl.QWEN_VL_MODEL_FILENAME
    model.parent.mkdir()
    model.write_bytes(b"incomplete")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"GGUF" + bytes(1024 * 1024))
        return str(destination)
    monkeypatch.setattr(qwen_vl, "hf_hub_download", download)
    with pytest.raises(ValueError):
        qwen_vl._ensure_qwen_vl_assets(allow_download=False, require_mmproj=False)
    assert calls == []
    qwen_vl._ensure_qwen_vl_assets(require_mmproj=False)
    assert calls[0]["force_download"] is True
    assert calls[0]["token"] is False
