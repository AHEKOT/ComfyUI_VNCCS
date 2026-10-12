"""Single-image randomization, clothing modes and observed chest-size contracts."""

import json
import math
import random

import pytest
from PIL import Image

from conftest import _preload_node


torch = pytest.importorskip("torch")
creator = _preload_node("character_creator_v2")
preview = _preload_node("style_preview_test")
generator = _preload_node("character_generator")


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    calls = {"sample": [], "encoded": [], "rewrite": [], "clip_load": [], "lora": [],
             "analysis": [], "analysis_output": "B-Cup", "events": [],
             "rewrite_fields": {"hair": "The hair is black and long."}}
    image = torch.ones(1, 8, 8, 4)
    clip, rewrite_clip = object(), object()
    monkeypatch.setattr(preview, "random", random.Random(42))
    monkeypatch.setattr(preview.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(preview.folder_paths, "get_filename_list", lambda _: [
        "anima-base-v1.0.safetensors", preview.QI2_DEFAULTS["diffusion_model_name"],
    ])
    monkeypatch.setattr(preview, "load_generation_assets", lambda settings: (
        None, object(), clip, object(),
    ))

    def load_clip(settings):
        calls["clip_load"].append(dict(settings))
        return rewrite_clip

    def comfy_node(name, **kwargs):
        if name == "TextGenerate":
            if "image" in kwargs:
                calls["analysis"].append(kwargs)
                calls["events"].append("analysis")
                return (calls["analysis_output"],)
            calls["rewrite"].append(kwargs)
            calls["events"].append("rewrite")
            return (json.dumps({"fields": calls["rewrite_fields"]}),)
        assert name == "TextEncodeQwenImage21"
        calls["encoded"].append((kwargs["clip"], kwargs["prompt"]))
        return "positive", "negative", None

    def encode_prompt(encoder, text, settings):
        calls["encoded"].append((encoder, text))
        return "conditioning"

    def apply_lora(model, encoder, data, strength, clip_strength):
        calls["lora"].append((data, strength, clip_strength))
        return model, encoder

    monkeypatch.setattr(preview, "load_generation_clip", load_clip)
    monkeypatch.setattr(generator, "_call_comfy_node", comfy_node)
    monkeypatch.setattr(creator, "encode_generation_prompt", encode_prompt)
    monkeypatch.setattr(preview, "prepare_qi2_model", lambda model, settings: (
        model, bool(settings["turbo_enabled"]),
    ))
    monkeypatch.setattr(preview, "get_lora_full_path", lambda name: "/test/anima-turbo.safetensors")
    monkeypatch.setattr(preview.comfy.utils, "load_torch_file", lambda *args, **kwargs: "turbo-weights", raising=False)
    monkeypatch.setattr(preview.comfy.sd, "load_lora_for_models", apply_lora, raising=False)
    monkeypatch.setattr(preview, "create_generation_latent", lambda model, width, height, settings: (
        width, height,
    ))
    def sample(**kwargs):
        calls["sample"].append(kwargs)
        calls["events"].append("sample")
        return "sample"

    def decode(*args):
        calls["events"].append("decode")
        return image

    monkeypatch.setattr(preview, "sample_generation_latent", sample)
    monkeypatch.setattr(preview, "decode_generation_samples", decode)
    return calls, image, clip, rewrite_clip, tmp_path


def test_node_contract_preserves_old_inputs_and_adds_outputs_and_controls():
    node = preview.VNCCSStylePreviewTest
    inputs = node.INPUT_TYPES()
    assert node.OUTPUT_NODE is True
    assert node.RETURN_TYPES == ("IMAGE", "STRING")
    assert node.RETURN_NAMES == ("image", "prompt")
    assert list(inputs["required"]) == ["model", "scale_mp", "background_color"]
    assert all(f"randomize_{field}" in inputs["optional"] for field in preview._RANDOM_FIELDS)
    assert inputs["optional"]["turbo_enabled"][1]["default"] is False
    assert inputs["optional"]["age_min"][1]["default"] == 18
    assert inputs["optional"]["age_max"][1]["default"] == 40
    assert inputs["optional"]["nsfw_mode"][0] == ["SFW", "NSFW", "Random"]
    assert inputs["optional"]["framing_mode"][0] == ["Full Body", "Cowboy Shot", "Random"]
    assert inputs["optional"]["framing_mode"][1]["default"] == "Random"
    assert math.isnan(node.IS_CHANGED())
    assert preview._square_side(1.0) == 1024
    assert preview._square_side(4.0) == 2048


@pytest.mark.parametrize("model,background", [("Anima", "Green"), ("QI2", "Transparent")])
@pytest.mark.parametrize("turbo", [False, True])
def test_single_image_and_returned_prompt_only_changes_observed_size(runtime, model, background, turbo):
    calls, image, clip, rewrite_clip, output = runtime
    defaults = {f"randomize_{field}": False for field in preview._RANDOM_FIELDS}
    node = preview.VNCCSStylePreviewTest()
    result = node.generate(model, 1.5, background, turbo_enabled=turbo, **defaults)
    assert len(calls["sample"]) == len(result["ui"]["images"]) == 1
    assert result["result"][0] is image
    prompt = result["result"][1]
    if model == "QI2":
        assert calls["encoded"][0][0] is clip
        assert "Breast size: B-Cup" not in calls["encoded"][0][1]
    else:
        assert calls["encoded"][0][0] is clip
        assert "black hair, long hair" in calls["encoded"][0][1]
        assert "Create one image" not in calls["encoded"][0][1]
        assert "The hair is black and long" not in calls["encoded"][0][1]
        assert "Breast size:" not in calls["encoded"][0][1]
    assert prompt.count("Breast size: B-Cup") == 1
    assert prompt.index("Breast size: B-Cup") < prompt.index("black hair")
    assert calls["events"] == (
        ["sample", "decode", "analysis", "rewrite"] if model == "Anima"
        else ["rewrite", "sample", "decode", "analysis", "rewrite"]
    )
    assert len(calls["analysis"]) == 1
    assert calls["analysis"][0]["clip"] is rewrite_clip
    assert calls["analysis"][0]["image"].shape == (1, 8, 8, 3)
    assert "character_fields:" not in calls["analysis"][0]["prompt"]
    assert "The hair is black and long" in prompt
    assert "Hayao Miyazaki / Studio Ghibli" in prompt
    assert len(calls["rewrite"]) == (1 if model == "Anima" else 2)
    assert "Hayao Miyazaki" not in calls["rewrite"][0]["prompt"]
    filename = result["ui"]["images"][0]["filename"]
    assert filename == f"{model.lower()}_ghibli_miyazaki_0001.png"
    with Image.open(output / "VNCCS" / "style_previews" / filename) as saved:
        assert saved.mode == "RGBA"
    sample = calls["sample"][0]
    assert sample["latent"][0] == sample["latent"][1]
    assert sample["gen_settings"]["turbo_enabled"] is turbo
    assert sample["qi2_turbo"] is (turbo and model == "QI2")
    assert sample["steps"] == ((12 if model == "Anima" else 6) if turbo else (30 if model == "Anima" else 25))
    assert sample["cfg"] == (1.0 if turbo else (4.0 if model == "Anima" else 3.0))
    if model == "Anima":
        assert calls["rewrite"][0]["clip"] is rewrite_clip
        assert calls["clip_load"][0]["clip_name"] == preview.QI2_DEFAULTS["clip_name"]
        assert calls["clip_load"][0]["clip_type"] == "stable_diffusion"
        assert calls["clip_load"][1]["clip_type"] == "qwen_image"
        assert calls["lora"] == ([("turbo-weights", 1.0, 0.0)] if turbo else [])
    else:
        assert calls["rewrite"][0]["clip"] is clip
        assert len(calls["clip_load"]) == 2
        assert calls["clip_load"][0]["generation_mode"] == "qi2"
        assert calls["lora"] == []
        assert "transparent background with alpha channel" in prompt
    second = node.generate(model, 1.5, background, turbo_enabled=turbo, **defaults)
    assert second["ui"]["images"][0]["filename"].endswith("_0002.png")
    assert calls["sample"][0]["seed"] != calls["sample"][1]["seed"]


def test_random_fields_use_current_presets_and_can_be_disabled_independently(monkeypatch):
    monkeypatch.setattr(preview, "random", random.Random(13))
    styles = {style["id"] for group in preview.CHARACTER_STYLE_CATALOG["groups"] for style in group["styles"]}
    seen = {field: set() for field in preview._RANDOM_FIELDS}
    for _ in range(30):
        info, style_id = preview._random_character_info("qi2", "Blue", 24, 29, False, {})
        assert 24 <= info["age"] <= 29
        assert info["style"] == style_id and style_id in styles
        for field, groups in preview._FIELD_TAG_GROUPS.items():
            allowed = {item["tag"] for group in groups for item in preview.CHARACTER_PRESETS["tags"][group]}
            assert set(info[field].split(", ")) <= allowed
            assert info[field]
        for field in seen:
            seen[field].add(info[field])
    assert all(len(values) > 1 for values in seen.values())
    for field in preview._RANDOM_FIELDS:
        info, _ = preview._random_character_info("qi2", "Blue", 24, 29, False, {field: False})
        default = preview.CHARACTER_STYLE_CATALOG["default_style"] if field == "style" else preview._PROMPT_DEFAULTS[field]
        assert info[field] == default
    info, _ = preview._random_character_info("qi2", "Blue", 37, 37, False, {})
    assert info["age"] == 37


def test_anima_override_uses_one_exact_artist_and_survives_rewriting(runtime):
    calls, *_ = runtime
    result = preview.VNCCSStylePreviewTest().generate("Anima", 1.0, "Green", anima_style_override=True)
    prompt = result["result"][1]
    artist = prompt.rsplit("\n", 1)[1]
    assert artist.startswith("@") and artist[1:] in preview._ANIMA_ARTISTS
    assert prompt.count("@") == 1
    assert "@{" not in prompt and "|" not in prompt
    assert artist not in calls["rewrite"][0]["prompt"]
    assert calls["encoded"][0][1].startswith(f"{artist},")
    assert result["ui"]["images"][0]["filename"].startswith("anima_artist_")
    assert len(preview._ANIMA_ARTISTS) == len(set(preview._ANIMA_ARTISTS))
    assert preview._ANIMA_ARTISTS[0] == "~am314"
    assert preview._ANIMA_ARTISTS[-1] == "fukuro daizi"
    info, _ = preview._random_character_info("anima", "Green", 18, 40, True, {"style": False})
    assert info["custom_style"] == "@~am314"
    info, _ = preview._random_character_info("qi2", "Green", 18, 40, True, {})
    assert info["style"] != "custom" and "custom_style" not in info


@pytest.mark.parametrize("minimum,maximum", [(40, 18), (0, 30), (18, 101)])
def test_invalid_age_range_fails_before_model_loading(runtime, minimum, maximum):
    calls, *_ = runtime
    with pytest.raises(ValueError, match="Age range"):
        preview.VNCCSStylePreviewTest().generate("Anima", 1.0, "Green", age_min=minimum, age_max=maximum)
    assert calls["rewrite"] == calls["clip_load"] == calls["sample"] == []


def test_missing_anima_turbo_adapter_fails_explicitly(runtime, monkeypatch):
    monkeypatch.setattr(preview, "get_lora_full_path", lambda _: None)
    with pytest.raises(ValueError, match="Anima Turbo LoRA not found"):
        preview.VNCCSStylePreviewTest().generate("Anima", 1.0, "Green", turbo_enabled=True)


def test_anima_rejects_transparent_background():
    with pytest.raises(ValueError, match="only with QI2"):
        preview.VNCCSStylePreviewTest().generate("Anima", 1.0, "Transparent")


def test_qi2_rejects_missing_alpha_channel(runtime, monkeypatch):
    monkeypatch.setattr(preview, "decode_generation_samples", lambda *args: torch.ones(1, 8, 8, 3))
    with pytest.raises(RuntimeError, match="no alpha channel"):
        preview.VNCCSStylePreviewTest().generate("QI2", 1.0, "Transparent")


@pytest.mark.parametrize("mode,expected", [("SFW", "wear white bra and panties"), ("NSFW", "naked, nude")])
def test_clothing_selector_controls_generation_prompt(runtime, mode, expected):
    calls, *_ = runtime
    result = preview.VNCCSStylePreviewTest().generate(
        "QI2", 1.0, "Green", nsfw_mode=mode, randomize_sex=False, randomize_age=False,
    )
    assert expected in calls["encoded"][0][1]
    assert expected in result["result"][1]


def test_random_clothing_mode_selects_both_adult_modes_and_preserves_minors(monkeypatch):
    monkeypatch.setattr(preview, "random", random.Random(55))
    choices = {
        preview._random_character_info("qi2", "Green", 18, 40, False, {}, "Random")[0]["nsfw"]
        for _ in range(20)
    }
    assert choices == {False, True}
    adult_body_tags = {item["tag"] for item in preview.CHARACTER_PRESETS["tags"]["breast_size"]}
    for _ in range(10):
        info, _ = preview._random_character_info("qi2", "Green", 1, 17, False, {}, "Random")
        assert info["nsfw"] is False
        assert set(info["body"].split(", ")).isdisjoint(adult_body_tags)
    with pytest.raises(ValueError, match="requires adult"):
        preview._random_character_info("qi2", "Green", 17, 20, False, {}, "NSFW")
    with pytest.raises(ValueError, match="Unsupported NSFW"):
        preview._random_character_info("qi2", "Green", 18, 40, False, {}, "invalid")


def test_breast_size_section_replaces_size_tags_but_preserves_body_details(runtime, monkeypatch):
    calls, *_ = runtime
    original = preview._random_character_info

    def character(*args, **kwargs):
        info, style_id = original(*args, **kwargs)
        info["body"] = "slim waist, small_breasts, mole_on_breast"
        return info, style_id

    monkeypatch.setattr(preview, "_random_character_info", character)
    result = preview.VNCCSStylePreviewTest().generate("Anima", 1.0, "Green")
    request = json.loads(calls["rewrite"][0]["prompt"].split("character_fields:\n", 1)[1])
    assert request["body"] == "The character has slim waist, mole_on_breast. Breast size: B-Cup"
    assert "small_breasts" in calls["encoded"][0][1]
    assert "Breast size:" not in calls["encoded"][0][1]
    prompt = result["result"][1]
    assert "small_breasts" not in prompt
    assert "The character has slim waist, mole_on_breast. Breast size: B-Cup" in prompt
    assert "slim waist" in prompt and "mole_on_breast" in prompt
    assert "slim waist" in calls["encoded"][0][1] and "mole_on_breast" in calls["encoded"][0][1]


@pytest.mark.parametrize("response", [
    "Flat Chest", "flat_chest", "a cup", *[f"{letter}-Cup" for letter in "BCDEFGHIJ"],
    "Unknown", "K-Cup", "B-Cup or C-Cup", "The character has very large J-Cup breasts.",
    '<think>Consider the image.</think>\nJ-Cup', '{"fields":{"breast_size":"J-Cup"}}',
])
def test_analysis_returns_complete_model_response_without_filtering(runtime, response):
    calls, image, *_ = runtime
    calls["analysis_output"] = response
    assert preview._analyze_breast_size(image) == response


@pytest.mark.parametrize("response", [
    "The character has J-Cup breasts.\nThey have a pronounced rounded silhouette.",
    "Unknown: the image is unclear.", "B-Cup or C-Cup", '{"size":"J-Cup","details":"large"}',
    "<think>Observed volume.</think>\n**J-Cup**, with (detail:1.0).",
    "Full response.\n  Additional detail: (tag:1.0).\n  ",
])
def test_complete_model_response_is_embedded_verbatim_in_body(runtime, response):
    calls, _, _, _, output = runtime
    calls["analysis_output"] = response
    result = preview.VNCCSStylePreviewTest().generate(
        "Anima", 1.0, "Green", randomize_hair=False, randomize_body=False,
    )
    prompt = result["result"][1]
    assert f"Breast size: {response}" in prompt
    assert prompt.index(response) < prompt.index("black hair")
    assert "__VNCCS_PREVIEW_OBSERVED_BODY__" not in prompt
    assert len(list((output / "VNCCS" / "style_previews").glob("*.png"))) == 1


def test_analysis_composites_alpha_on_white_and_bounds_large_images(runtime):
    calls, *_ = runtime
    image = torch.zeros(1, 1200, 1600, 4)
    assert preview._analyze_breast_size(image) == "B-Cup"
    analyzed = calls["analysis"][0]["image"]
    assert analyzed.shape == (1, 768, 1024, 3)
    assert torch.all(analyzed == 1)
    assert calls["analysis"][0]["thinking"] is False
    assert calls["analysis"][0]["max_length"] == 512
    assert calls["analysis"][0]["sampling_mode"]["sampling_mode"] == "off"
    assert calls["clip_load"][0]["clip_type"] == "stable_diffusion"


def test_sfw_minor_has_no_chest_analysis_and_nsfw_range_fails_before_generation(runtime):
    calls, *_ = runtime
    result = preview.VNCCSStylePreviewTest().generate("QI2", 1.0, "Green", age_min=15, age_max=15)
    assert calls["analysis"] == []
    assert "Breast size:" not in result["result"][1]
    assert calls["encoded"][0][1] == result["result"][1]
    before = len(calls["sample"])
    with pytest.raises(ValueError, match="requires adult"):
        preview.VNCCSStylePreviewTest().generate("QI2", 1.0, "Green", age_min=15, age_max=15, nsfw_mode="NSFW")
    assert len(calls["sample"]) == before


@pytest.mark.parametrize("model", ["Anima", "QI2"])
@pytest.mark.parametrize("framing,expected", [("Full Body", "complete body from head to toe"),
                                             ("Cowboy Shot", "Cowboy Shot (cowboy_shot)")])
def test_framing_selector_controls_generation_and_output_prompt(runtime, model, framing, expected):
    calls, *_ = runtime
    result = preview.VNCCSStylePreviewTest().generate(
        model, 1.0, "Green", framing_mode=framing, randomize_framing=False,
    )
    generation_framing = (
        "standing, full body" if framing == "Full Body" else "cowboy_shot"
    ) if model == "Anima" else expected
    assert generation_framing in calls["encoded"][0][1]
    assert expected in result["result"][1]
    request = json.loads(calls["rewrite"][0]["prompt"].split("character_fields:\n", 1)[1])
    assert expected in request["framing"]


def test_random_framing_selector_overrides_legacy_checkbox(monkeypatch):
    monkeypatch.setattr(preview, "random", random.Random(41))
    choices = {
        preview._random_character_info(
            "qi2", "Green", 18, 40, False, {"framing": False}, framing_mode="Random",
        )[0]["framing"] for _ in range(20)
    }
    assert choices == {"full_body", "cowboy_shot"}
    legacy, _ = preview._random_character_info("qi2", "Green", 18, 40, False, {"framing": False})
    assert legacy["framing"] == "cowboy_shot"


def test_invalid_framing_fails_before_generation(runtime):
    calls, *_ = runtime
    with pytest.raises(ValueError, match="Unsupported framing mode"):
        preview.VNCCSStylePreviewTest().generate("QI2", 1.0, "Green", framing_mode="invalid")
    assert calls["clip_load"] == calls["rewrite"] == calls["sample"] == []


@pytest.mark.parametrize("nsfw_mode,clothing", [("SFW", "wear white bra and panties"), ("NSFW", "naked, nude")])
def test_anima_encodes_raw_species_and_body_before_output_rewriting(runtime, monkeypatch, nsfw_mode, clothing):
    calls, *_ = runtime
    original = preview._random_character_info

    def character(*args, **kwargs):
        info, style_id = original(*args, **kwargs)
        info.update(race="elf", body="small_breasts, slim waist", hair="black hair, long hair")
        return info, style_id

    monkeypatch.setattr(preview, "_random_character_info", character)
    result = preview.VNCCSStylePreviewTest().generate(
        "Anima", 1.0, "Green", nsfw_mode=nsfw_mode, randomize_sex=False, randomize_age=False,
        anima_style_override=True, randomize_style=False,
    )
    encoded = calls["encoded"][0][1]
    assert encoded.startswith("@~am314,")
    assert "(elf)" in encoded
    assert "small_breasts, slim waist" in encoded
    assert "black hair, long hair" in encoded
    assert clothing in encoded
    assert "Typical species features" not in encoded
    assert "The hair is black and long" not in encoded
    assert calls["events"] == ["sample", "decode", "analysis", "rewrite"]
    assert "The hair is black and long" in result["result"][1]
    assert "The character has slim waist. Breast size: B-Cup" in result["result"][1]
    assert result["result"][1].endswith("@~am314")


@pytest.mark.parametrize("model", ["Anima", "QI2"])
def test_observed_j_cup_is_inside_body_and_cannot_be_changed_by_pe(runtime, monkeypatch, caplog, model):
    calls, *_ = runtime
    calls["analysis_output"] = "J-Cup"
    calls["rewrite_fields"]["body"] = "The character has A-Cup breasts."
    original = preview._random_character_info

    def character(*args, **kwargs):
        info, style_id = original(*args, **kwargs)
        info.update(age=38, body="long torso, small_breasts, mole_on_breast", hair="black hair, long hair")
        return info, style_id

    monkeypatch.setattr(preview, "_random_character_info", character)
    result = preview.VNCCSStylePreviewTest().generate(
        model, 1.0, "Green", randomize_sex=False,
    )
    prompt = result["result"][1]
    assert "The character has long torso, mole_on_breast. Breast size: J-Cup" in prompt
    assert prompt.index("Breast size: J-Cup") < prompt.index("black hair")
    if model == "Anima":
        assert prompt.index("Breast size: J-Cup") < prompt.index("masterpiece")
    assert "A-Cup" not in prompt and "small_breasts" not in prompt
    assert prompt.count("Breast size:") == 1
    assert len(calls["sample"]) == 1
    assert calls["analysis"][0]["sampling_mode"]["sampling_mode"] == "off"
    assert calls["analysis"][0]["max_length"] == 512
    assert calls["analysis"][0]["thinking"] is False
    assert calls["analysis"][0]["use_default_template"] is True
    assert calls["analysis"][0]["mtp"] == "auto"
    analysis = [record.vnccs for record in caplog.records if record.name == "VNCCS" and record.vnccs["event"] == "body_analysis"]
    assert len(analysis) == 1 and analysis[0]["breast_size"] == "J-Cup"
    assert "analysis_output" not in caplog.text


def test_json_analysis_response_is_returned_in_full(runtime):
    calls, image, *_ = runtime
    calls["analysis_output"] = '{"fields":{"breast_size":"J-Cup"}}'
    assert preview._analyze_breast_size(image) == calls["analysis_output"]


def test_calibrated_analysis_uses_image_only_and_retains_qualified_response(runtime):
    calls, image, *_ = runtime
    response = (
        "The breasts have moderate projection relative to the ribcage. "
        "D-Cup is the closest visual category; the silhouette is fuller than C-Cup "
        "without the clearly above-medium volume of E-Cup."
    )
    calls["analysis_output"] = response
    assert preview._analyze_breast_size(image) == response
    request = calls["analysis"][0]
    assert request["sampling_mode"]["sampling_mode"] == "off"
    assert request["image"].shape == (1, 8, 8, 3)
    assert "character_fields:" not in request["prompt"]
    assert "Return exactly one category:" in request["prompt"]
    assert "Do not include explanations, descriptions, comparisons," in request["prompt"]
    assert "First give" not in request["prompt"]
    assert "ribcage and upper torso" in request["prompt"]
    assert "instead of guessing a cup label" in request["prompt"]
    for label in ["Flat Chest:", *[f"{letter}-Cup:" for letter in "ABCDEFGHIJ"]]:
        assert label in request["prompt"]
    info = {"body": "long torso, small_breasts"}
    preview._set_observed_breast_size(info, response)
    assert info["body"] == f"The character has long torso. Breast size: {response}"
