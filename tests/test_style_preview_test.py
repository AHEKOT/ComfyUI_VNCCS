"""Contract tests for the temporary style batch output node."""

import pytest
from PIL import Image

from conftest import _preload_node


torch = pytest.importorskip("torch")
_preload_node("character_creator_v2")
preview = _preload_node("style_preview_test")


def test_output_node_has_only_model_scale_and_background_controls():
    node = preview.VNCCSStylePreviewTest
    assert node.OUTPUT_NODE is True
    assert node.RETURN_TYPES == ()
    assert list(node.INPUT_TYPES()["required"]) == ["model", "scale_mp", "background_color"]
    assert preview._square_side(1.0) == 1024
    assert preview._square_side(4.0) == 2048


@pytest.mark.parametrize("model,background", [("Anima", "Green"), ("QI2", "Transparent")])
def test_generates_every_style_with_v2_defaults_and_saves_named_pngs(
    monkeypatch, tmp_path, model, background,
):
    calls = []
    prompts = []
    references = []
    characters = []
    styles = [{"id": "ghibli_miyazaki"}, {"id": "fortiche_arcane"}]
    monkeypatch.setattr(preview, "CHARACTER_STYLE_CATALOG", {"groups": [{"styles": styles}]})
    monkeypatch.setattr(preview.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(preview.folder_paths, "get_filename_list", lambda _: [
        "anima-base-v1.0.safetensors", "qwen_image_2.1_int8_convrot.safetensors",
    ])
    monkeypatch.setattr(preview, "load_generation_assets", lambda settings: (
        None, object(), object(), object(),
    ))
    monkeypatch.setattr(preview, "load_generation_clip", lambda settings: object())
    monkeypatch.setattr(preview, "prepare_qi2_model", lambda diffusion_model, settings: (
        diffusion_model, False,
    ))
    def capture_conditioning(_clip, _vae, positive, negative, _settings, style_reference="", character_info=None):
        prompts.append((positive, negative))
        references.append(style_reference)
        characters.append(character_info)
        return "positive", "negative", "rewritten"

    monkeypatch.setattr(preview, "encode_generation_conditioning", capture_conditioning)
    monkeypatch.setattr(preview, "validate_anima_conditioning", lambda *args: None)
    monkeypatch.setattr(preview, "create_generation_latent", lambda _model, width, height, _settings: (
        width, height,
    ))
    monkeypatch.setattr(preview, "sample_generation_latent", lambda **kwargs: calls.append(kwargs) or "sample")
    monkeypatch.setattr(preview, "decode_generation_samples", lambda *args: torch.ones(1, 8, 8, 4))

    result = preview.VNCCSStylePreviewTest().generate(model, 1.5, background)

    assert len(result["ui"]["images"]) == len(styles)
    assert [item["filename"] for item in result["ui"]["images"]] == [
        f"{model.lower()}_{style['id']}_0001.png" for style in styles
    ]
    assert all((tmp_path / "VNCCS" / "style_previews" / item["filename"]).is_file()
               for item in result["ui"]["images"])
    assert all(call["latent"][0] == call["latent"][1] for call in calls)
    assert all(call["seed"] == 0 for call in calls)
    assert all(call["qi2_turbo"] is False for call in calls)
    assert all(call["gen_settings"]["turbo_enabled"] is False for call in calls)
    assert "Hayao Miyazaki / Studio Ghibli" in references[0]
    assert "Fortiche / Arcane" in references[1]
    for (positive, _negative), reference in zip(prompts, references):
        assert (reference in positive) == (model == "Anima")
    assert "black hair" in prompts[0][0]
    assert [info["style"] for info in characters] == [style["id"] for style in styles]
    assert all(info["hair"] == "black hair, long hair" for info in characters)
    if model == "QI2":
        assert "transparent background with alpha channel" in prompts[0][0]
        with Image.open(tmp_path / "VNCCS" / "style_previews" / result["ui"]["images"][0]["filename"]) as saved:
            assert saved.mode == "RGBA"


def test_anima_rejects_transparent_background():
    with pytest.raises(ValueError, match="only with QI2"):
        preview.VNCCSStylePreviewTest().generate("Anima", 1.0, "Transparent")
