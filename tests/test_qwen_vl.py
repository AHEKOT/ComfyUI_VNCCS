import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nodes import qwen_vl, vnccs_control_center as catalog


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
    monkeypatch.setitem(sys.modules, "json_repair", SimpleNamespace(loads=lambda _: [{"hair": "black"}]))
    assert qwen_vl.parse_wizard_json("repaired response") == {"hair": "black"}


def _write_model(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytes(1024)
    header = json.dumps({"weight": {"dtype": "U8", "shape": [len(pixels)], "data_offsets": [0, len(pixels)]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + pixels)


@pytest.fixture
def models(tmp_path, monkeypatch):
    def find(local_path):
        path = tmp_path / local_path
        return str(path), path.is_file()

    monkeypatch.setattr(catalog, "_find_model_on_disk", find)
    monkeypatch.setattr(catalog, "_resolve_model_download_path", lambda local_path: str(tmp_path / local_path))
    monkeypatch.setattr(qwen_vl, "_QWEN_VL_DOWNLOAD_STATUS", {"status": "idle"})
    monkeypatch.setattr(qwen_vl.web, "json_response", lambda data, status=200:
                        SimpleNamespace(data=data, status=status), raising=False)
    return tmp_path


@pytest.mark.parametrize("diffusion", ["qwen_image_2.1_int8_convrot", "qwen_image_2.1_turbo_int8_convrot"])
def test_complete_qi2_is_used_before_standalone_model_and_prevents_download(models, monkeypatch, diffusion):
    _write_model(models / f"models/diffusion_models/{diffusion}.safetensors")
    _write_model(models / "models/text_encoders/qwen3vl_8b_int8_convrot.safetensors")
    _write_model(models / qwen_vl._QWEN_VL_MODEL_ENTRY["local_path"])
    monkeypatch.setattr(qwen_vl, "hf_hub_download", lambda **_: pytest.fail("Complete QI2 must not download a model"))
    ready = asyncio.run(qwen_vl.qwen_vl_model_status(None))
    assert ready.data == {"ready": True, "model_name": "QI2 Qwen3-VL 8B Text Encoder"}
    assert qwen_vl._start_qwen_vl_download().data["status"] == "completed"


@pytest.mark.parametrize("present", ["encoder", "diffusion"])
def test_incomplete_qi2_uses_one_pinned_wizard_download(models, monkeypatch, present):
    _write_model(models / ("models/text_encoders/qwen3vl_8b_int8_convrot.safetensors" if present == "encoder"
                           else "models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"))
    calls, workers = [], []

    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        _write_model(path)
        return str(path)

    monkeypatch.setattr(qwen_vl, "hf_hub_download", download)
    monkeypatch.setattr(qwen_vl.threading, "Thread", lambda *, target, daemon:
                        SimpleNamespace(start=lambda: workers.append(target)))
    assert asyncio.run(qwen_vl.qwen_vl_model_status(None)).data["ready"] is False
    assert qwen_vl._start_qwen_vl_download().data == {"status": "started"}
    assert qwen_vl._start_qwen_vl_download().status == 409
    assert len(workers) == 1
    assert not calls
    workers[0]()
    assert qwen_vl._QWEN_VL_DOWNLOAD_STATUS["status"] == "completed"
    assert calls == [{
        "repo_id": "beycanai/Qwen-LM",
        "filename": "Qwen3.5-4B_int8_convrot.safetensors",
        "revision": "65be86eff5aab8bf7c1fe0d9a363e1fa8433d1ca",
        "local_dir": str(models / "models/text_encoders"),
        "token": False,
        "force_download": False,
    }]
    assert qwen_vl._start_qwen_vl_download().data["status"] == "completed"
    assert len(calls) == 1


def test_invalid_standalone_model_is_redownloaded(models, monkeypatch):
    path = models / qwen_vl._QWEN_VL_MODEL_ENTRY["local_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete")

    def download(**kwargs):
        assert kwargs["force_download"] is True
        _write_model(path)
        return str(path)

    monkeypatch.setattr(qwen_vl, "hf_hub_download", download)
    qwen_vl._qwen_vl_download_worker()
    assert qwen_vl._QWEN_VL_DOWNLOAD_STATUS["status"] == "completed"


@pytest.mark.parametrize("vision", [False, True])
def test_wizard_invokes_native_text_generate(monkeypatch, vision):
    pytest.importorskip("torch")
    from nodes import character_generator

    model = qwen_vl._QWEN_VL_MODEL_ENTRY
    clip = object()
    image = object() if vision else None
    calls = []
    monkeypatch.setattr(catalog, "_load_clips", lambda entries, names: clip if (entries, names) == ([model], [model["name"]]) else None)
    monkeypatch.setattr(character_generator, "_call_comfy_node",
                        lambda name, **kwargs: (calls.append((name, kwargs)) or '{"hair":"black"}', ""))
    assert qwen_vl.generate_wizard_text(model, "system", "user", max_length=900, temperature=0.2, image=image) == '{"hair":"black"}'
    name, options = calls[0]
    assert name == "TextGenerate"
    assert options["clip"] is clip
    assert options["prompt"] == "user"
    assert options["system_prompt"] == "system"
    assert options["thinking"] is False
    assert options["max_length"] == 900
    assert options["sampling_mode"]["temperature"] == 0.2
    assert options.get("image") is image
