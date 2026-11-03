"""Tests for nodes/clothes_designer.py — pure logic functions."""

import os
import sys
import types
import json
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")
import torch

from _vnccs.nodes.clothes_designer import (
    ClothesDesigner,
    _resolve_pipe_clothes_core_lora,
)


# ── construct_prompt ──────────────────────────────────────────────────────────

class TestClothesDesignerConstructPrompt:
    def _data(self, **overrides):
        data = {
            "activeTab": "generate",
            "character": "",
            "costume_info": {},
            "gen_settings": {"background_color": "Green"},
        }
        data.update(overrides)
        return data

    def test_generate_tab_returns_tuple(self):
        result = ClothesDesigner.construct_prompt(self._data())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_tab_includes_green_bg(self):
        pos, _ = ClothesDesigner.construct_prompt(self._data(gen_settings={"background_color": "Green"}))
        assert "green" in pos.lower()
        assert "00FF00" in pos

    def test_generate_tab_includes_blue_bg(self):
        pos, neg = ClothesDesigner.construct_prompt(self._data(gen_settings={"background_color": "Blue"}))
        assert "blue" in pos.lower()
        assert "#0000FF" in pos
        assert "purple background" in neg

    def test_qi2_generate_tab_supports_native_transparency(self):
        pos, _ = ClothesDesigner.construct_prompt(
            self._data(gen_settings={"background_color": "Transparent"}),
            model_kind="qi2",
        )
        assert "Transparent background with alpha channel." in pos

    def test_non_qi2_transparency_falls_back_to_green(self):
        pos, _ = ClothesDesigner.construct_prompt(
            self._data(gen_settings={"background_color": "Transparent"}),
            model_kind="klein9b",
        )
        assert "#00FF00" in pos
        assert "Transparent background" not in pos

    def test_generate_tab_unknown_bg_defaults_to_green(self):
        pos, _ = ClothesDesigner.construct_prompt(self._data(gen_settings={"background_color": "Red"}))
        assert "#00FF00" in pos

    def test_generate_tab_includes_costume_parts(self):
        costume = {"top": "white shirt", "bottom": "black jeans", "shoes": "sneakers"}
        pos, _ = ClothesDesigner.construct_prompt(self._data(costume_info=costume))
        assert "white shirt" in pos
        assert "black jeans" in pos
        assert "sneakers" in pos

    def test_generate_tab_omits_empty_costume_parts(self):
        costume = {"top": "red dress", "bottom": "", "head": ""}
        pos, _ = ClothesDesigner.construct_prompt(self._data(costume_info=costume))
        assert "red dress" in pos

    def test_generate_tab_negative_prompt_not_empty(self):
        _, neg = ClothesDesigner.construct_prompt(self._data())
        assert len(neg) > 0

    def test_generate_tab_negative_contains_nsfw_block(self):
        _, neg = ClothesDesigner.construct_prompt(self._data())
        assert "naked" in neg.lower() or "nude" in neg.lower()

    def test_clone_tab_requires_an_uploaded_reference(self):
        data = self._data(activeTab="clone", clone_image=None)
        with pytest.raises(ValueError, match="Upload a clothing reference image"):
            ClothesDesigner.construct_prompt(data)

    def test_clone_tab_with_clone_image(self):
        data = self._data(
            activeTab="clone", clone_image="img.png",
            costume_info={"top": "unused costume description"},
        )
        pos, neg = ClothesDesigner.construct_prompt(data)
        assert pos.startswith("Dress character to clothes from image 2")
        assert "unused costume description" not in pos
        assert "hex #00FF00" in pos
        assert "Transfer the outfit" not in pos
        assert "Preserve the identity" not in pos
        assert "no gradient" in pos
        for constraint in (
            "background scenery", "patterned background", "shapes in background",
            "multicolored background", "textured background", "gradient background",
        ):
            assert constraint in neg

    @pytest.mark.parametrize("background,model_kind,expected", [
        ("Green", "klein9b", "flat uniform pure green background, exact RGB (0, 255, 0), hex #00FF00"),
        ("Blue", "klein9b", "flat uniform pure blue background, exact RGB (0, 0, 255), hex #0000FF"),
        (" blue ", "minimaxh3", "hex #0000FF"),
        (None, "klein9b", "hex #00FF00"),
        ("Red", "klein9b", "hex #00FF00"),
        ("Transparent", "klein9b", "hex #00FF00"),
        ("Transparent", "qi2", "Transparent background with alpha channel."),
    ])
    def test_clone_tab_respects_background_color(self, background, model_kind, expected):
        from _vnccs.nodes.clothes_designer import _clothes_background_prompt

        data = self._data(
            activeTab="clone",
            clone_image="img.png",
            gen_settings={"background_color": background},
        )
        pos, neg = ClothesDesigner.construct_prompt(data, model_kind=model_kind)
        assert expected in pos
        if "#0000FF" in expected:
            assert "purple background" in neg
            assert "violet background" in neg
        if model_kind == "qi2":
            assert "#00FF00" not in pos
            assert "#0000FF" not in pos
        else:
            assert "Transparent background" not in pos
        assert pos == (
            "Dress character to clothes from image 2\n"
            f"{_clothes_background_prompt(ClothesDesigner._effective_background_color(background, model_kind))}"
        )


class TestReferenceBackgroundPreparation:
    def test_blue_composites_transparent_pixels_to_exact_blue(self):
        image = torch.tensor([[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.5, 0.25, 1.0]]]])
        prepared = ClothesDesigner._prepare_reference_background(image, "Blue")
        assert prepared.shape[-1] == 3
        assert prepared[0, 0, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])
        assert prepared[0, 0, 1].tolist() == pytest.approx([1.0, 0.5, 0.25])

    @pytest.mark.parametrize("batched", [False, True])
    def test_qi2_flattens_rgba_on_white_before_native_encoder(self, batched):
        image = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.2, 0.4, 0.6, 0.5], [0.2, 0.4, 0.6, 1.0]]])
        if batched:
            image = image.unsqueeze(0)
        original = image.clone()
        from _vnccs.nodes.character_generator import _qi2_reference_image
        prepared = _qi2_reference_image(image)
        assert prepared.shape[-1] == 3
        expected = torch.tensor([[[1.0, 1.0, 1.0], [0.6, 0.7, 0.8], [0.2, 0.4, 0.6]]])
        torch.testing.assert_close(prepared, expected.unsqueeze(0) if batched else expected)
        assert torch.equal(image, original)


# ── get_cache_paths ───────────────────────────────────────────────────────────

class TestGetCachePaths:
    def test_returns_two_paths(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()

        img_path, info_path = ClothesDesigner.get_cache_paths("Alice", "Casual")
        assert img_path.endswith(".png")
        assert info_path.endswith(".json")

    def test_costume_name_sanitized(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()

        img_path, _ = ClothesDesigner.get_cache_paths("Alice", "My Fancy/Costume!")
        basename = os.path.basename(img_path)
        # Special characters removed; spaces → underscores
        assert "/" not in basename
        assert "!" not in basename

    def test_cache_dir_created(self, tmp_path, monkeypatch):
        import utils
        monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()

        ClothesDesigner.get_cache_paths("Alice", "Dress")
        assert os.path.isdir(tmp_path / "Alice" / "cache")


# ── Clone reference preparation ───────────────────────────────────────────────

class TestCloneReferencePreparation:
    def test_sam3_preprocessing_helpers_are_not_exposed(self):
        assert not hasattr(ClothesDesigner, "_run_clone_sam3_reference")
        assert not hasattr(ClothesDesigner, "_apply_mask_on_background")

    def test_uploaded_subfolder_and_filename_resolve_to_encoder_reference(self, tmp_path, monkeypatch):
        from _vnccs.nodes import clothes_designer as cd
        from PIL import Image

        donor = tmp_path / "clothes" / "donor.png"
        donor.parent.mkdir()
        Image.new("RGB", (16, 16), (255, 0, 0)).save(donor)
        monkeypatch.setattr(cd.folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
        metadata = {"name": "donor.png", "type": "input", "subfolder": "clothes"}
        assert cd.resolve_comfy_image_path(metadata) == str(donor)
        state = {"activeTab": "clone", "clone_image": metadata}
        first = ClothesDesigner.IS_CHANGED(widget_data=state)
        assert ClothesDesigner.IS_CHANGED(widget_data=state) == first
        Image.new("RGB", (16, 16), (0, 255, 0)).save(donor)
        assert ClothesDesigner.IS_CHANGED(widget_data=state) != first


# ── Clothes Core LoRA resolution ──────────────────────────────────────────────

class TestClothesCoreLoraResolution:
    def test_qi2_does_not_reuse_legacy_clothes_lora(self):
        pipe = types.SimpleNamespace(
            model_entry={"kind": "QI2"},
            lora_entries=[
                {
                    "name": "VNCCS Clothes Core",
                    "kind": "QIE2511",
                    "local_path": "models/loras/qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.5.safetensors",
                }
            ],
        )
        assert _resolve_pipe_clothes_core_lora(pipe) == ""

    def test_selects_only_lora_matching_klein_model_kind(self):
        pipe = types.SimpleNamespace(
            model_entry={"kind": "Klein9b"},
            lora_entries=[
                {
                    "name": "VNCCS Clothes Core",
                    "kind": "QIE2511",
                    "local_path": "models/loras/qwen/VNCCS/VNCCS_QIE2511_ClothesCore-RC3.7.safetensors",
                },
                {
                    "name": "VNCCS Clothes Core Klein9b",
                    "kind": "Klein9b",
                    "local_path": "models/loras/Klein9b/VNCCS_ClothesCoreKlein9b_V1.safetensors",
                },
            ],
        )
        assert _resolve_pipe_clothes_core_lora(pipe) == "Klein9b/VNCCS_ClothesCoreKlein9b_V1.safetensors"


# ── Costume validation ───────────────────────────────────────────────────────

class TestEditableCostumeValidation:
    @pytest.mark.parametrize("costume", ["Casual", "My Costume", "armor_01"])
    def test_accepts_editable_costumes(self, costume):
        assert ClothesDesigner._is_editable_costume(costume)

    @pytest.mark.parametrize("costume", ["", None, "Naked", "Original", "naked", "ORIGINAL"])
    def test_rejects_missing_or_base_costumes(self, costume):
        assert not ClothesDesigner._is_editable_costume(costume)


@pytest.mark.parametrize("kind,size,expected", [
    ("MiniMaxH3", None, 1536), ("MiniMaxH3", 1024, 1024),
    ("QI2", None, 1024), ("QI2", 1536, 1536), ("QI2", 2560, 2560), ("QI2", 4096, 4096),
    ("Klein9b", None, 1024), ("Klein9b", 2048, 2048),
])
@pytest.mark.parametrize("clone", [False, True])
@pytest.mark.parametrize("cfg", [None, 0.0, 1.0])
def test_preview_resolution_reaches_model_encoder(tmp_path, monkeypatch, caplog, kind, size, expected, clone, cfg):
    from _vnccs.nodes import clothes_designer as cd
    from _vnccs.nodes import character_generator as cg
    from PIL import Image
    import utils

    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    utils.save_costume_info("Alice", "Dress", {})

    reference_channels = 4 if kind == "QI2" else 3
    reference = torch.zeros((1, 96, 64, reference_channels))
    reference[..., 2] = 1.0
    if reference_channels == 4:
        reference[..., 3] = 1.0
    clone_path = tmp_path / "clone.png"
    donor = Image.new("RGBA", (80, 40), (255, 0, 0, 255))
    donor.putpixel((1, 0), (24, 71, 129, 0))
    donor.putpixel((2, 0), (75, 29, 193, 128))
    donor.save(clone_path)
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (64, 96), "blue").save(reference_path)
    monkeypatch.setattr(cd, "get_latest_sprite_path", lambda *args: str(reference_path))
    monkeypatch.setattr(cd, "sheets_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(cd, "_resolve_pipe_clothes_core_lora", lambda pipe: "clothes.safetensors")
    monkeypatch.setattr(cd, "resolve_comfy_image_path", lambda info: str(clone_path))
    monkeypatch.setattr(cd.server.PromptServer.instance, "send_sync", lambda *args: None, raising=False)
    node = cd.ClothesDesigner()
    if clone:
        def forbidden_template():
            raise AssertionError("Clone Clothes must not load the PE template.")
        monkeypatch.setattr(cd, "_qi2_edit_system_prompt", forbidden_template)
    monkeypatch.setattr(node, "get_reference_sprite", lambda *args: reference)
    monkeypatch.setattr(node, "get_cache_paths", lambda *args: (str(tmp_path / "preview.png"), str(tmp_path / "preview.json")))
    calls = {}
    encoder_latent = {"samples": torch.zeros(1), "metadata": object()}
    sampling_latent = {"samples": torch.zeros(1)}
    call_names = []
    def call(name, **kwargs):
        calls[name] = kwargs
        call_names.append(name)
        if name == "TextGenerate":
            assert not clone, "Clone Clothes must pass its fixed prompt directly to the encoder."
            return (json.dumps({
                "rewritten_prompt": "Dress the character in the image with the requested outfit precisely.",
                "wh_ratio": "", "ratio_follow": "<image1>",
            }),)
        if name == "VNCCS_Flux_Klein_Encoder":
            return "positive", "negative", {"samples": torch.zeros(1)}
        if name == "ImageScale":
            source = kwargs["image"]
            return (source[0, 0, 0].expand(1, kwargs["height"], kwargs["width"], source.shape[-1]).clone(),)
        if name == "ImageScaleToTotalPixels":
            source = kwargs["image"]
            height, width = int(source.shape[1]), int(source.shape[2])
            target_pixels = float(kwargs["megapixels"]) * 1024 * 1024
            scale = (target_pixels / (width * height)) ** 0.5
            scaled_width = round(width * scale)
            scaled_height = round(height * scale)
            return (source[0, 0, 0].expand(1, scaled_height, scaled_width, source.shape[-1]).clone(),)
        if name == "TextEncodeQwenImage21":
            return "positive", "negative", encoder_latent
        if name == "EmptyLatentImage":
            return (sampling_latent,)
        if name == "MiniMaxH3ReferenceToVideo":
            return "positive", {"samples": torch.zeros(1)}
        if name in ("KSampler", "SamplerCustomAdvanced"):
            return ({"samples": torch.zeros(1, 24, 2, 4, 4) if kind == "MiniMaxH3" else torch.zeros(1)},)
        if name in ("VAEDecodeTiled", "VAEDecode"):
            return (torch.zeros((1, 96, 64, 3)),)
        return (object(),)
    def node_class(name):
        return type(name, (), {"FUNCTION": "run", "run": staticmethod(lambda **kwargs: call(name, **kwargs))})

    mappings = {name: node_class(name) for name in (
        "TextGenerate", "VNCCS_Flux_Klein_Encoder", "ImageScale", "ImageScaleToTotalPixels",
        "TextEncodeQwenImage21", "EmptyLatentImage", "MiniMaxH3ReferenceToVideo",
        "KSampler", "SamplerCustomAdvanced", "VAEDecodeTiled", "VAEDecode", "QwenImage21Cache",
        "BasicScheduler", "BasicGuider", "RandomNoise", "KSamplerSelect", "LoraLoaderModelOnly",
    )}
    monkeypatch.setattr(cg, "comfy_nodes", types.SimpleNamespace(NODE_CLASS_MAPPINGS=mappings))
    monkeypatch.setattr(cd, "_call_comfy_node", cg._call_comfy_node)
    monkeypatch.setattr(cd, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cd, "_apply_lora_standard", lambda model, clip, path, strength: (
        call("LoraLoaderModelOnly", model=model, lora_name=path, strength_model=strength)[0], clip,
    ))
    pipe = types.SimpleNamespace(
        model=object(), clip=object(), vae=object(), audio_vae=object(), model_entry={"kind": kind},
        model_cache_key={"model": {"kind": kind, "name": "Test catalog model"}},
        cfg=cfg,
    )
    workflow = json.loads((Path(__file__).resolve().parents[1] / "workflows/VNCCS_3.2_Step2_CharacterClothes.json").read_text())
    designer = next(n for n in workflow["nodes"] if n["type"] == "ClothesDesigner")
    saved = json.loads(designer["widgets_values"][0])
    if kind == "QI2" and cfg == 1.0:
        pipe.sample_steps, pipe.sampler_name, pipe.scheduler = 8, "euler", "simple"
    data = {**saved, "character": "Alice", "costume": "Dress", "gen_settings": {
                **saved["gen_settings"],
                "target_size": size,
                "background_color": "Transparent" if kind == "QI2" else "Green",
                "lora_strength": 0 if cfg is None else 0.65,
            },
            "activeTab": "clone" if clone else "generate", "clone_image": {"name": "clone.png"} if clone else None}
    image, _, background = node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
    assert image.shape == (1, 96, 64, 3)
    if kind == "MiniMaxH3":
        encoder = calls["MiniMaxH3ReferenceToVideo"]
        width, height = encoder["width"], encoder["height"]
        scale = (expected * 1024 / (64 * 96)) ** 0.5
        assert (width, height) == (round(64 * scale) // 32 * 32, round(96 * scale) // 32 * 32)
        assert encoder["ref_images"]["ref_image_1"].shape[1:3] == (height, width)
        assert width % 32 == height % 32 == 0
        assert width * height == pytest.approx(expected * 1024, rel=0.04)
        assert width / height == pytest.approx(64 / 96, rel=0.04)
        assert encoder["length"] == 5
        assert len(encoder["ref_images"]) == (2 if clone else 1)
        assert "SamplerCustomAdvanced" in calls and "KSampler" not in calls
        assert "VAEDecode" in calls
        assert "VAEDecodeTiled" not in calls
        assert set(calls["VAEDecode"]) == {"samples", "vae"}
        assert calls["VAEDecode"]["samples"]["samples"].shape == (1, 24, 1, 4, 4)
    elif kind == "Klein9b":
        assert calls["VNCCS_Flux_Klein_Encoder"]["megapixels"] == expected / 1024
        assert calls["VNCCS_Flux_Klein_Encoder"]["align_image1"] is True
    else:
        assert background == "Alpha"
        assert "ImageScale" not in calls
        sizing = calls["ImageScaleToTotalPixels"]
        assert sizing["image"] is calls["TextEncodeQwenImage21"]["images"]["image_1"]
        assert (sizing["upscale_method"], sizing["resolution_steps"], sizing["megapixels"]) == (
            "nearest-exact", 1, expected / 1024,
        )
        assert calls["KSampler"]["latent_image"] is sampling_latent
        scale = (expected * 1024 / (96 * 64)) ** 0.5
        assert calls["EmptyLatentImage"] == {"width": round(64 * scale), "height": round(96 * scale), "batch_size": 1}
        assert calls["TextEncodeQwenImage21"]["images"]["image_1"].shape[1:3] == (96, 64)
        resolution = {1024: 1024, 1536: 1248, 2560: 1632, 4096: 2048}[expected]
        if cfg == 1.0:
            sampler = calls["KSampler"]
            assert (sampler["seed"], sampler["steps"], sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]) == (
                saved["gen_settings"]["seed"], 8, "euler", "simple", 1.0,
            )
        assert calls["TextEncodeQwenImage21"]["resolution"] == resolution
        expected_negative = node.construct_prompt(data, model_kind="qi2")[1]
        assert calls["TextEncodeQwenImage21"]["negative_prompt"] == expected_negative
        if clone:
            assert "TextGenerate" not in calls
        else:
            assert call_names.index("TextGenerate") < call_names.index("TextEncodeQwenImage21")
            enhancer = calls["TextGenerate"]
            assert enhancer["clip"] is pipe.clip
            assert enhancer["thinking"] is False
            assert enhancer["max_length"] == 2048
            assert "Attribute Disentanglement at Full Strength" in enhancer["prompt"]
            assert enhancer["image"].shape == (1, 96, 64, 3)
            assert enhancer["image"][0, 0, 0].tolist() == pytest.approx([0, 0, 1])
            assert calls["TextEncodeQwenImage21"]["prompt"].startswith("Dress the character in the image with the requested outfit precisely.")
            assert "<image1>" not in calls["TextEncodeQwenImage21"]["prompt"]
        assert "Transparent background with alpha channel." in calls["TextEncodeQwenImage21"]["prompt"]
        if clone:
            assert "patterned background" in calls["TextEncodeQwenImage21"]["negative_prompt"]
        assert "QwenImage21Cache" in calls
        assert "VAEDecode" in calls
        assert "VAEDecodeTiled" not in calls
    if kind != "QI2":
        assert "TextGenerate" not in calls
        encoder_name = "MiniMaxH3ReferenceToVideo" if kind == "MiniMaxH3" else "VNCCS_Flux_Klein_Encoder"
        assert "#00FF00" in calls[encoder_name]["prompt"]
    else:
        encoder_name = "TextEncodeQwenImage21"

    def encoder_references():
        encoder = calls[encoder_name]
        if kind == "QI2":
            return encoder["images"]
        if kind == "MiniMaxH3":
            return encoder["ref_images"]
        return {key: encoder[key] for key in ("image1", "image2", "image3") if encoder[key] is not None}

    target_key, donor_key = {
        "QI2": ("image_1", "image_2"),
        "MiniMaxH3": ("ref_image_1", "ref_image_2"),
        "Klein9b": ("image1", "image2"),
    }[kind]
    references = encoder_references()
    assert references[target_key][0, 0, 0, :3].tolist() == pytest.approx([0, 0, 1])
    assert set(references) == ({target_key, donor_key} if clone else {target_key})
    if clone:
        donor_tensor = node._pil_image_tensor(donor)
        expected_donor = cg._qi2_reference_image(donor_tensor) if kind == "QI2" else donor_tensor
        torch.testing.assert_close(references[donor_key], expected_donor)
        assert references[donor_key][0, 0, 0, :3].tolist() == pytest.approx([1, 0, 0])
        assert calls[encoder_name]["prompt"] == node.construct_prompt(data, model_kind=kind.lower())[0]
        assert calls[encoder_name]["prompt"].startswith("Dress character to clothes from image 2\n")

    assert "restore, upscale and refine character's hairs" not in calls[encoder_name]["prompt"]
    assert calls["LoraLoaderModelOnly"]["strength_model"] == 1

    sampler_name = "SamplerCustomAdvanced" if kind == "MiniMaxH3" else "KSampler"
    if kind == "MiniMaxH3":
        assert calls["BasicGuider"]["conditioning"] == "positive"
    else:
        assert calls[sampler_name]["positive"] == "positive"
        assert calls[sampler_name]["negative"] == "negative"
        assert calls[sampler_name]["cfg"] == (cfg if cfg is not None else 3.0 if kind == "QI2" else 1.0)

    node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
    assert call_names.count(encoder_name) == 1
    assert call_names.count("TextGenerate") == (1 if kind == "QI2" and not clone else 0)
    if kind in {"QI2", "MiniMaxH3"}:
        records = [record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompts"]
        encoded = [record for record in records if record.vnccs["context"] == encoder_name]
        assert len(encoded) == 1, "A cache hit must not pretend to generate another image."
        assert encoded[0].vnccs["positive"] == calls[encoder_name]["prompt"]
        assert calls[encoder_name]["prompt"] in encoded[0].getMessage()
        assert encoded[0].vnccs["component"] == "ClothesDesigner"
        assert encoded[0].vnccs["node_id"] == "123"
        if kind == "QI2":
            assert encoded[0].vnccs["negative"] == calls[encoder_name]["negative_prompt"]
            dimensions = [record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "encoder_resolution"]
            assert len(dimensions) == 1, "A cache hit must not log a new encoder call."
            record = dimensions[0]
            assert record.levelname == "INFO"
            assert record.vnccs["component"] == "ClothesDesigner"
            assert record.vnccs["node_id"] == "123"
            assert record.vnccs["encoder"] == encoder_name
            assert record.vnccs["resolution"] == resolution
            assert f"resolution={resolution}" in record.getMessage()
            if not clone:
                rewrite = next(record for record in records if record.vnccs["context"] == "Text generation")
                assert rewrite.vnccs["user"] == calls["TextGenerate"]["prompt"]
                assert rewrite.vnccs["user"] in rewrite.getMessage()
        else:
            assert "negative" not in encoded[0].vnccs
    if clone:
        monkeypatch.setattr(cd, "resolve_comfy_image_path", lambda *args, **kwargs: str(clone_path))
        old_signature = node.IS_CHANGED(widget_data=json.dumps(data))
        Image.new("RGB", (64, 96), (0, 255, 0)).save(clone_path)
        assert node.IS_CHANGED(widget_data=json.dumps(data)) != old_signature
        node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
        assert call_names.count(encoder_name) == 2
        assert encoder_references()[donor_key][0, 0, 0, :3].tolist() == pytest.approx([0, 1, 0])
        assert "TextGenerate" not in calls
    if kind == "QI2" and not clone:
        previous_calls = call_names.count("TextGenerate")
        template = cd._qi2_edit_system_prompt()
        monkeypatch.setattr(cd, "_qi2_edit_system_prompt", lambda: template + "\nUpdated edit rules.")
        node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
        assert call_names.count("TextGenerate") == previous_calls + 1
        assert call_names.count(encoder_name) == previous_calls + 1
        assert "Updated edit rules." in calls["TextGenerate"]["prompt"]

        text_calls = call_names.count("TextGenerate")
        encoder_calls = call_names.count(encoder_name)
        data["gen_settings"]["qi2_prompt_enhancement"] = False
        monkeypatch.setattr(
            cd, "_qi2_edit_system_prompt",
            lambda: pytest.fail("Disabled Prompt Enhancement must not load its template."),
        )
        node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
        assert call_names.count("TextGenerate") == text_calls
        assert call_names.count(encoder_name) == encoder_calls + 1
        assert calls[encoder_name]["prompt"] == node.construct_prompt(data, model_kind="qi2")[0]


class TestQI2EditRewriter:
    @pytest.mark.parametrize("channels", [3, 4])
    def test_prompt_enhancement_preserves_reference_pixels_and_dimensions(self, monkeypatch, channels):
        from _vnccs.nodes import clothes_designer as cd
        from _vnccs.nodes.character_generator import _qi2_reference_image

        image = torch.rand(1, 1057, 67, channels)
        original = image.clone()
        calls = []
        monkeypatch.setattr(cd, "_call_comfy_node", lambda name, **kwargs: calls.append((name, kwargs)) or ("",))
        cd._rewrite_qi2_clothes_prompt("clip", "outfit", image, "Blue", "edit rules")
        assert calls[0][0] == "TextGenerate"
        prepared = calls[0][1]["image"]
        assert prepared.shape == (1, 1057, 67, 3)
        torch.testing.assert_close(prepared, _qi2_reference_image(image))
        if channels == 3:
            assert prepared is image
        assert torch.equal(image, original)

    def test_official_template_is_bundled(self):
        from _vnccs.nodes.clothes_designer import _qi2_edit_system_prompt

        prompt = _qi2_edit_system_prompt()
        assert "Edit Prompt Enhancer" in prompt
        assert "For single-image input (N = 1), do NOT use tags" in prompt
        assert '"rewritten_prompt"' in prompt
        assert '"ratio_follow"' in prompt
        assert prompt.endswith("The user's edit instruction to rewrite is:")

    def test_missing_template_fails_explicitly(self, monkeypatch):
        from _vnccs.nodes import clothes_designer as cd

        monkeypatch.setattr(cd.os.path, "join", lambda *args: "/missing/qi2_edit_template.txt")
        with pytest.raises(RuntimeError, match="QI2 edit prompt template is missing"):
            cd._qi2_edit_system_prompt()

    @pytest.mark.parametrize("background,expected", [
        ("Green", "hex #00FF00"), ("Blue", "hex #0000FF"),
        ("Transparent", "Transparent background with alpha channel."),
    ])
    @pytest.mark.parametrize("response", [
        '{"rewritten_prompt":"Change the outfit decisively.","wh_ratio":"1:1","ratio_follow":""}',
        '<think>unfinished reasoning',
        '{"rewritten_prompt":',
        '```json\n{"rewritten_prompt":"Change the outfit decisively."}\n```',
        '',
    ])
    def test_response_preserves_application_constraints(self, monkeypatch, background, expected, response):
        from _vnccs.nodes import clothes_designer as cd
        from _vnccs.nodes.character_creator_v2 import QI2_TEXT_GENERATION_DEFAULTS

        calls = []
        def call(name, **kwargs):
            calls.append((name, kwargs))
            return (response,)
        monkeypatch.setattr(cd, "_call_comfy_node", call)
        image = torch.ones((1, 8, 4, 3))
        source = "Original clothes request."
        result = cd._rewrite_qi2_clothes_prompt(object(), source, image, background, "official edit rules")
        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == "TextGenerate"
        for key, value in QI2_TEXT_GENERATION_DEFAULTS.items():
            assert kwargs[key] == value
        assert source in kwargs["prompt"]
        assert kwargs["image"] is image
        assert expected in result
        assert "pose, framing and rendering medium" in result
        assert "<think>" not in result
        assert '"wh_ratio"' not in result
        assert result.startswith("Change the outfit decisively." if "Change the outfit decisively." in response else source)
        assert "restore, upscale and refine character's hairs" not in kwargs["prompt"]
        assert "of the image" in result
        assert "<image1>" not in result
        assert "<image2>" not in result


@pytest.mark.parametrize("size", [True, "bad", -1, 0, 511, 4097, 1024.5, float("inf"), float("nan")])
def test_resolution_rejects_invalid_values(size):
    from _vnccs.nodes.clothes_designer import _clothes_target_size
    with pytest.raises(ValueError, match="Resolution scale"):
        _clothes_target_size({"target_size": size}, "minimaxh3")


def test_legacy_sub_megapixel_resolution_is_raised_to_one_megapixel():
    from _vnccs.nodes.clothes_designer import _clothes_target_size
    assert _clothes_target_size({"target_size": 512}, "qi2") == 1024
