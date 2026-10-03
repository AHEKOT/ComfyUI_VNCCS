"""Tests for nodes/clothes_designer.py — pure logic functions."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")
import torch

from _vnccs.nodes.clothes_designer import (
    ClothesDesigner,
    PipeContext,
    _resolve_pipe_clothes_core_lora,
)


# ── _find_breasts_desc ────────────────────────────────────────────────────────

class TestFindBreastsDesc:
    def test_finds_in_body_field(self):
        info = {"body": "slim, small breasts, tall"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None
        assert "breasts" in result.lower()

    def test_finds_flat_chest(self):
        info = {"body": "flat chest, petite"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None
        assert "flat chest" in result.lower()

    def test_finds_in_other_field_if_body_empty(self):
        info = {"body": "", "additional_details": "large breasts, long legs"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None
        assert "breasts" in result.lower()

    def test_returns_none_when_no_breast_desc(self):
        info = {"body": "slim, tall", "additional_details": "holding sword"}
        assert ClothesDesigner._find_breasts_desc(info) is None

    def test_body_field_takes_priority_over_others(self):
        info = {"body": "medium breasts", "additional_details": "huge breasts reference"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert "medium" in result.lower()

    def test_case_insensitive(self):
        info = {"body": "LARGE BREASTS"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None

    def test_ignores_non_string_values(self):
        info = {"body": 42, "additional_details": "small breasts"}
        result = ClothesDesigner._find_breasts_desc(info)
        assert result is not None


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

    def test_qi2_transparency_preserves_alpha_and_cleans_hidden_rgb(self):
        image = torch.tensor([[[[1.0, 0.0, 1.0, 0.0], [0.2, 0.4, 0.6, 0.5]]]])
        prepared = ClothesDesigner._prepare_reference_background(
            image, "Transparent", preserve_transparency=True,
        )
        assert prepared.shape[-1] == 4
        assert prepared[0, 0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])
        assert prepared[0, 0, 1].tolist() == pytest.approx([0.6, 0.7, 0.8, 0.5])


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

    @pytest.mark.parametrize("costume", ["", None, "Naked", "Original"])
    def test_rejects_missing_or_base_costumes(self, costume):
        assert not ClothesDesigner._is_editable_costume(costume)


# ── PipeContext ───────────────────────────────────────────────────────────────

class TestPipeContext:
    def test_creates_empty_pipe_from_none(self):
        ctx = PipeContext(source=None)
        assert ctx.model is None
        assert ctx.clip is None
        assert ctx.vae is None
        assert ctx.seed_int == 0
        assert ctx.denoise == 1.0

    def test_copies_attrs_from_source(self):
        src = types.SimpleNamespace(
            model=object(), clip=object(), vae=object(),
            pos=object(), neg=object(),
            seed_int=42, sample_steps=20, cfg=7.0, denoise=0.8,
            sampler_name="euler", scheduler="karras",
            loader_type="standard", nunchaku_kind=None,
            nunchaku_settings=None, model_entry=None,
        )
        ctx = PipeContext(source=src)
        assert ctx.model is src.model
        assert ctx.seed_int == 42
        assert ctx.cfg == 7.0
        assert ctx.sampler_name == "euler"

    def test_updates_override_source(self):
        src = types.SimpleNamespace(
            model=object(), clip=object(), vae=object(),
            pos=object(), neg=object(),
            seed_int=1, sample_steps=10, cfg=5.0, denoise=1.0,
            sampler_name="euler", scheduler="normal",
            loader_type=None, nunchaku_kind=None,
            nunchaku_settings=None, model_entry=None,
        )
        ctx = PipeContext(source=src, seed_int=999, cfg=3.5)
        assert ctx.seed_int == 999
        assert ctx.cfg == 3.5
        # other attrs unchanged from source
        assert ctx.sample_steps == 10

    def test_falls_back_to_seed_attr(self):
        src = types.SimpleNamespace(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed=777,  # old attr name, no seed_int
            sample_steps=0, cfg=0.0, denoise=1.0,
            sampler_name=None, scheduler=None,
            loader_type=None, nunchaku_kind=None,
            nunchaku_settings=None, model_entry=None,
        )
        ctx = PipeContext(source=src)
        assert ctx.seed_int == 777

    def test_propagates_loader_type(self):
        src = types.SimpleNamespace(
            model=None, clip=None, vae=None, pos=None, neg=None,
            seed_int=0, sample_steps=0, cfg=0.0, denoise=1.0,
            sampler_name=None, scheduler=None,
            loader_type="nunchaku", nunchaku_kind="flux",
            nunchaku_settings={"precision": "fp4"}, model_entry={"name": "x"},
        )
        ctx = PipeContext(source=src)
        assert ctx.loader_type == "standard"
        assert ctx.nunchaku_kind is None
        assert ctx.nunchaku_settings is None


@pytest.mark.parametrize("kind,size,expected", [
    ("MiniMaxH3", None, 1536), ("MiniMaxH3", 1024, 1024),
    ("QI2", None, 1024), ("QI2", 1536, 1536),
    ("Klein9b", None, 1024), ("Klein9b", 2048, 2048),
])
@pytest.mark.parametrize("clone", [False, True])
def test_preview_resolution_reaches_model_encoder(tmp_path, monkeypatch, kind, size, expected, clone):
    from _vnccs.nodes import clothes_designer as cd
    from _vnccs.nodes import character_generator as cg
    from PIL import Image
    import json

    reference_channels = 4 if kind == "QI2" else 3
    reference = torch.zeros((1, 96, 64, reference_channels))
    reference[..., 2] = 1.0
    if reference_channels == 4:
        reference[..., 3] = 1.0
    clone_path = tmp_path / "clone.png"
    Image.new("RGB", (64, 96), (255, 0, 0)).save(clone_path)
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (64, 96), "blue").save(reference_path)
    monkeypatch.setattr(cd, "get_latest_sprite_path", lambda *args: str(reference_path))
    monkeypatch.setattr(cd, "sheets_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(cd, "_resolve_pipe_clothes_core_lora", lambda pipe: "" if kind == "QI2" else "clothes.safetensors")
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
        if name == cd.KLEIN_ENCODER_CLASS:
            return "positive", "negative", {"samples": torch.zeros(1)}
        if name == "ImageScale":
            return (kwargs["image"],)
        if name == "ImageScaleToTotalPixels":
            source = kwargs["image"]
            height, width = int(source.shape[1]), int(source.shape[2])
            target_pixels = float(kwargs["megapixels"]) * 1024 * 1024
            scale = (target_pixels / (width * height)) ** 0.5
            scaled_width = round(width * scale)
            scaled_height = round(height * scale)
            return (source[0, 0, 0].expand(1, scaled_height, scaled_width, source.shape[-1]).clone(),)
        if name == "TextEncodeQwenImage21":
            return "positive", "negative", {"samples": torch.zeros(1)}
        if name == "EmptyLatentImage":
            return ({"samples": torch.zeros(1)},)
        if name == "MiniMaxH3ReferenceToVideo":
            return "positive", {"samples": torch.zeros(1)}
        if name in ("KSampler", "SamplerCustomAdvanced"):
            return ({"samples": torch.zeros(1)},)
        if name in ("VAEDecodeTiled", "VAEDecode"):
            return (torch.zeros((5 if kind == "MiniMaxH3" else 1, 96, 64, 3)),)
        return (object(),)
    monkeypatch.setattr(cd, "_call_comfy_node", call)
    monkeypatch.setattr(cg, "_call_comfy_node", call)
    monkeypatch.setattr(cd, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cd, "_apply_lora_standard", lambda model, clip, path, strength: (
        call("LoraLoaderModelOnly", model=model, lora_name=path, strength_model=strength)[0], clip,
    ))
    pipe = types.SimpleNamespace(
        model=object(), clip=object(), vae=object(), audio_vae=object(), model_entry={"kind": kind},
        model_cache_key={"model": {"kind": kind, "name": "Test catalog model"}},
    )
    data = {"character": "Alice", "costume": "Dress", "gen_settings": {
                "target_size": size,
                "background_color": "Transparent" if kind == "QI2" else "Green",
            },
            "activeTab": "clone" if clone else "generate", "clone_image": {"name": "clone.png"} if clone else None}
    image, _, background = node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
    assert image.shape == (1, 96, 64, 3)
    if kind == "MiniMaxH3":
        encoder = calls["MiniMaxH3ReferenceToVideo"]
        width, height = encoder["width"], encoder["height"]
        assert width % 32 == height % 32 == 0
        assert width * height == pytest.approx(expected * 1024, rel=0.04)
        assert width / height == pytest.approx(64 / 96, rel=0.04)
        assert encoder["length"] == 5
        assert len(encoder["ref_images"]) == (2 if clone else 1)
        assert "SamplerCustomAdvanced" in calls and "KSampler" not in calls
        assert "VAEDecode" in calls
        assert "VAEDecodeTiled" not in calls
        assert set(calls["VAEDecode"]) == {"samples", "vae"}
    elif kind == "Klein9b":
        assert calls[cd.KLEIN_ENCODER_CLASS]["megapixels"] == expected / 1024
    else:
        assert background == "Alpha"
        latent = calls["EmptyLatentImage"]
        assert latent["width"] * latent["height"] == pytest.approx(expected * 1024, rel=0.04)
        assert latent["width"] / latent["height"] == pytest.approx(64 / 96, rel=0.04)
        assert calls["ImageScaleToTotalPixels"]["megapixels"] == pytest.approx(expected / 1024)
        assert calls["ImageScaleToTotalPixels"]["image"].shape[-1] == 4
        assert calls["TextEncodeQwenImage21"]["resolution"] == 1024
        assert calls["TextEncodeQwenImage21"]["negative_prompt"] == node.construct_prompt(data, model_kind="qi2")[1]
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
        encoder_name = "MiniMaxH3ReferenceToVideo" if kind == "MiniMaxH3" else cd.KLEIN_ENCODER_CLASS
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
        assert references[donor_key][0, 0, 0, :3].tolist() == pytest.approx([1, 0, 0])
        assert calls[encoder_name]["prompt"] == node.construct_prompt(data, model_kind=kind.lower())[0]
        assert calls[encoder_name]["prompt"].startswith("Dress character to clothes from image 2\n")

    sampler_name = "SamplerCustomAdvanced" if kind == "MiniMaxH3" else "KSampler"
    if kind == "MiniMaxH3":
        assert calls["BasicGuider"]["conditioning"] == "positive"
    else:
        assert calls[sampler_name]["positive"] == "positive"
        assert calls[sampler_name]["negative"] == "negative"

    node.process(pipe=pipe, widget_data=json.dumps(data), unique_id="123")
    assert call_names.count(encoder_name) == 1
    assert call_names.count("TextGenerate") == (1 if kind == "QI2" and not clone else 0)
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


class TestQI2EditRewriter:
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

    def test_different_image_sizes_are_fitted_without_cropping_or_mutating_encoder_inputs(self):
        from _vnccs.nodes.clothes_designer import _qi2_edit_image_batch

        target = torch.zeros((1, 8, 4, 4))
        target[..., 2:] = 1
        donor = torch.zeros((1, 2, 8, 3))
        donor[..., 0] = 1
        original_target, original_donor = target.clone(), donor.clone()
        batch = _qi2_edit_image_batch((target, donor))
        assert batch.shape == (2, 8, 4, 3)
        assert torch.equal(target, original_target)
        assert torch.equal(donor, original_donor)
        assert batch[0, 0, 0].tolist() == [0, 0, 1]
        assert batch[1, 0, 0].tolist() == [1, 1, 1]
        assert batch[1, 3, 0].tolist() == [1, 0, 0]
        assert batch[1, 3, -1].tolist() == [1, 0, 0]

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
        references = (image,)
        source = "Original clothes request."
        result = cd._rewrite_qi2_clothes_prompt(object(), source, references, background, "official edit rules")
        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == "TextGenerate"
        for key, value in QI2_TEXT_GENERATION_DEFAULTS.items():
            assert kwargs[key] == value
        assert source in kwargs["prompt"]
        assert kwargs["image"].shape[0] == len(references)
        assert expected in result
        assert "pose, framing and rendering medium" in result
        assert "<think>" not in result
        assert '"wh_ratio"' not in result
        assert result.startswith("Change the outfit decisively." if "Change the outfit decisively." in response else source)
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
