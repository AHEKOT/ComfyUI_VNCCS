"""Native background output must preserve pixels without repeating image work."""

import json
from urllib.parse import parse_qs, urlparse

import pytest

torch = pytest.importorskip("torch")

from PIL import Image
from nodes import character_generator as cg


@pytest.mark.parametrize("mode", ["character", "clothes", "clone"])
@pytest.mark.parametrize("preset", ["Native", "disabled", "balanced"])
@pytest.mark.parametrize("publish", [True, False])
def test_bg_output_reuses_normalized_batch_and_saved_previews(tmp_path, monkeypatch, mode, preset, publish):
    node = {
        "character": cg.VNCCS_CharacterGenerator,
        "clothes": cg.VNCCS_ClothesGenerator,
        "clone": cg.VNCCS_CharacterCloneGenerator,
    }[mode]()
    bypass = preset != "balanced"
    images = torch.rand(3, 8, 6, 4)
    expected = images.clone()
    if not bypass:
        expected[..., 3] *= 0.5
    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    sheets = root / "Sheets" / "Naked" / "neutral"
    sheets.mkdir(parents=True)
    monkeypatch.setattr(cg.folder_paths, "get_output_directory", lambda: str(output))
    monkeypatch.setattr(node, "_extract_pipe", lambda pipe: {"seed": 42})
    monkeypatch.setattr(node, "_find_pose_lora", lambda pipe: {})
    monkeypatch.setattr(node, "_find_clothes_lora", lambda pipe: {})
    monkeypatch.setattr(node, "_run_pose_generation", lambda poses, *args, **kwargs: poses)
    monkeypatch.setattr(node, "_run_upscaler", lambda images, *args, **kwargs: images)
    monkeypatch.setattr(node, "_run_source_upscaler", lambda images, *args, **kwargs: images)
    monkeypatch.setattr(node, "_run_remove_clothes", lambda image, *args, **kwargs: image)
    if not publish:
        monkeypatch.setattr(node, "_save_final_sprites", lambda *args, **kwargs: [])

    events, encodings, normalizations, chroma_calls = [], [], [], []
    active_bg = False
    original_emit = node._emit
    original_normalize = cg.normalize_image_batch
    original_preview = cg._tensor_to_preview_urls

    def emit(*args, **kwargs):
        nonlocal active_bg
        active_bg = args[1].endswith("bg_remove") and args[2] != "done"
        return original_emit(*args, **kwargs)

    def normalize(*args, **kwargs):
        if active_bg:
            normalizations.append(kwargs.get("stage"))
        return original_normalize(*args, **kwargs)

    def preview(images, unique_id, stage, **kwargs):
        if stage.endswith("bg_remove"):
            encodings.append(stage)
            assert not (bypass and publish), "Native previews must use the published PNG files"
        return original_preview(images, unique_id, stage, **kwargs)

    class ChromaKey:
        def chroma_key(self, batch, *args, **kwargs):
            assert not bypass, "Native and disabled must skip chroma key"
            chroma_calls.append(batch.shape[0])
            result = batch.clone()
            result[..., 3] *= 0.5
            return (result,)

    monkeypatch.setattr(node, "_emit", emit)
    monkeypatch.setattr(cg, "normalize_image_batch", normalize)
    monkeypatch.setattr(cg, "_tensor_to_preview_urls", preview)
    monkeypatch.setattr(cg, "VNCCSChromaKey", ChromaKey)
    monkeypatch.setattr(cg.server.PromptServer.instance, "send_sync", lambda name, payload: events.append(payload), raising=False)
    bg_stages = ["original_bg_remove", "naked_bg_remove"] if mode == "clone" else ["bg_remove"]
    payload = {"character_name": "Alice", "bg_remove": {"preset": preset}}

    for regenerate_index in (None, 1):
        events.clear()
        encodings.clear()
        normalizations.clear()
        active_bg = False
        if regenerate_index is not None:
            payload.update(regenerate_from=bg_stages[-1], regenerate_index=regenerate_index)
        result = node.process(images, images[:1], object(), "Pose", widget_data=json.dumps(payload),
                              sheets_path=str(sheets), unique_id="native-output")
        assert torch.equal(result[0], expected)
        if bypass and regenerate_index is None:
            assert result[0] is images
            assert normalizations == []
        if bypass:
            assert len(encodings) == (0 if publish else len(bg_stages))
        else:
            assert chroma_calls
        for stage in bg_stages:
            final = next(event for event in reversed(events) if event["stage"] == stage and event["status"] == "done")
            assert len(final["images"]) == images.shape[0]
            cached = torch.load(next(root.rglob(f"{stage}.pt")), weights_only=True)
            assert torch.equal(cached, expected)
            if bypass:
                assert all("images" not in event for event in events
                           if event["stage"] == stage and event["status"] == "running")
            for index, url in enumerate(final["images"]):
                query = parse_qs(urlparse(url).query)
                path = output / query["subfolder"][0] / query["filename"][0]
                assert path.is_file()
                if bypass and publish:
                    assert "Sprites" in path.parts
                with Image.open(path) as image:
                    assert image.mode == "RGBA"
                    assert image.tobytes() == (expected[index].numpy() * 255).astype("uint8").tobytes()

    if bypass and publish:
        def fail_save(*args, **kwargs):
            raise OSError("Sprite publication failed")

        events.clear()
        monkeypatch.setattr(node, "_save_final_sprites", fail_save)
        with pytest.raises(OSError, match="Sprite publication failed"):
            node.process(images, images[:1], object(), "Pose", widget_data=json.dumps(payload),
                         sheets_path=str(sheets), unique_id="native-output")
        assert not any(event["stage"] in bg_stages and event["status"] == "done" for event in events)
        assert events[-1]["stage"] == "error"


def test_native_bypass_keeps_raw_input_normalization_and_fast_emotion_output(monkeypatch):
    node = cg.VNCCS_CharacterGenerator()
    raw = torch.arange(64, dtype=torch.uint8).reshape(1, 4, 4, 4)
    result = node._run_bg_remove(raw, {"preset": "Native"})
    assert torch.equal(result, raw.float() / 255)

    def unexpected_normalization(*args, **kwargs):
        raise AssertionError("Already normalized Native output must pass through unchanged")

    monkeypatch.setattr(cg, "normalize_image_batch", unexpected_normalization)
    assert node._run_bg_remove(result, {"preset": "Native"}, normalized=True) is result
    assert cg.VNCCS_EmotionsGenerator()._run_emotion_bg_remove(
        result, [], None, {"preset": "Native"}, normalized=True,
    ) is result
