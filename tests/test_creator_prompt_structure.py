"""Ordered Creator prompts and legacy anatomy work without model dependencies."""

import json
from types import SimpleNamespace
import sys

import pytest

from creator_storage_helpers import load_creator_storage


@pytest.fixture
def creator(monkeypatch):
    return load_creator_storage(monkeypatch)


def character(**changes):
    return {
        "sex": "female", "age": 30, "race": "human", "face": "freckles",
        "hair": "black hair", "eyes": "blue eyes", "skin_color": "tan skin",
        "body": "athletic build", "breast_size": "small breasts",
        "background_color": "Blue", "aesthetics": "quality_marker",
        "additional_details": "silver earrings", "lora_prompt": "exact_trigger",
        "negative_prompt": "blurry", **changes,
    }


@pytest.mark.parametrize("mode", ["qi2"])
@pytest.mark.parametrize("sex", ["female", "male"])
@pytest.mark.parametrize("nsfw", [False, True, "false", "true"])
def test_prompt_uses_ordered_fields_and_exclusive_clothing(creator, mode, sex, nsfw):
    info = character(sex=sex, nsfw=nsfw)
    original = dict(info)
    positive, negative = creator.CharacterCreatorV2.construct_creator_prompt(info, mode)
    lines = positive.splitlines()
    adult = nsfw in (True, "true")
    labels = [
        "Draw an image of a character in the following style", "Style reference",
        "Style description", "Additional style parameters", "The image contains one character.",
        "Framing", "Background", "Gender", "Age", "Race", "Facial features",
        "Hair", "Eyes", "Skin color",
        *(["Breast size"] if sex == "female" else []),
        "Build", "NSFW" if adult else "Clothes", "Additional details", "LoRA triggers",
    ]
    assert [line.split(":", 1)[0] for line in lines] == labels
    assert "Age: 30 years-old" in lines
    assert f"Gender: {sex}" in lines
    assert "Additional style parameters: quality_marker" in lines
    assert "Build: athletic build" in lines
    assert "LoRA triggers: exact_trigger" in lines
    assert ("Breast size: small breasts" in lines) == (sex == "female")
    assert ("small breasts" in positive) == (sex == "female")
    assert "blurry" in negative
    assert info == original


@pytest.mark.parametrize("body", ["small_breasts, slim waist", "small breasts, slim waist"])
def test_legacy_breast_tags_move_out_of_build_without_mutating_input(creator, body):
    info = character(body=body, breast_size="")
    original = dict(info)
    positive, _ = creator.CharacterCreatorV2.construct_creator_prompt(info, "qi2")
    assert f"Breast size: {body.split(',')[0]}\nBuild: slim waist" in positive
    assert info == original
    overridden, _ = creator.CharacterCreatorV2.construct_creator_prompt({**info, "breast_size": "medium breasts"}, "qi2")
    assert "Breast size: medium breasts\nBuild: slim waist" in overridden
    assert "small breasts" not in overridden and "small_breasts" not in overridden


def test_empty_fields_and_custom_style_preserve_the_structure(creator):
    positive, _ = creator.CharacterCreatorV2.construct_creator_prompt(character(
        style="custom", custom_style="handmade felt illustration", face="", eyes="",
        hair="", skin_color="", body="", breast_size="", additional_details="", lora_prompt="",
    ), "qi2")
    assert "Style reference: \nStyle description: handmade felt illustration\n" in positive
    assert "Facial features: expressionless\nHair: \nEyes: \nSkin color: \nBreast size: \nBuild: \n" in positive
    assert positive.endswith("Additional details: \nLoRA triggers: ")


@pytest.mark.parametrize("sex, nsfw", [("female", False), ("female", True), ("male", False), ("male", True)])
def test_qi2_official_t2i_template_receives_ordered_character_brief(creator, monkeypatch, sex, nsfw):
    info = character(sex=sex, nsfw=nsfw)
    reference = "Style: Ink artist.\nReference: Work:1.0.\nVisual description: Fine crosshatching."
    requests = []

    def generate(name, **values):
        assert name == "TextGenerate"
        requests.append(values)
        prompt = values["prompt"]
        assert ("\nBreast size:" in prompt) == (sex == "female")
        assert ("\nNSFW:" in prompt) == nsfw
        assert ("\nClothes:" in prompt) != nsfw
        assert "Age: 30 years-old" in prompt and "quality_marker" in prompt
        assert "exact_trigger" not in prompt
        assert values["system_prompt"].startswith("# Image Prompt Rewriting Expert")
        assert '{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}' in values["system_prompt"]
        assert "VNCCS Character Creator: Visual Prompt Construction" not in values["system_prompt"]
        assert values["use_default_template"] is True
        return ('{"rewritten_prompt":"A vertical illustration of one adult character with black hair.","wh_ratio":"2:3"}',)

    monkeypatch.setitem(sys.modules, "_vnccs.nodes.character_generator", SimpleNamespace(_call_comfy_node=generate))
    final = creator.generate_creator_qi2_prompt("clip", "", reference, info)
    assert "Style reference: Work:1.0." in requests[0]["prompt"]
    assert "Style description: Fine crosshatching." in requests[0]["prompt"]
    assert final.startswith("A vertical illustration of one adult character with black hair.")
    assert reference in final
    assert "wh_ratio" not in final and "rewritten_prompt" not in final
    assert final.endswith("LoRA triggers: exact_trigger")


@pytest.mark.parametrize("mode", ["illustrious", "anima"])
def test_tag_models_restore_pre_32_prompt_format_and_model_quality_defaults(creator, mode):
    info = character(style="none", style_prompt="stale cached style")
    del info["aesthetics"]
    del info["negative_prompt"]
    original = dict(info)
    positive, negative = creator.CharacterCreatorV2.construct_creator_prompt(info, mode)
    assert positive == (
        creator.CREATOR_TAG_PROMPT_DEFAULTS[mode]["aesthetics"]
        + ", simple background, expressionless, solo, cowboy_shot, (1girl), "
          "(wear white bra and panties), 30yo, adult woman, Blue background, "
          "(human:1.0), (black hair:1.0), (blue eyes:1.0), (freckles:1.0), "
          "(athletic build, small breasts:1.0), (tan skin:1.0), (silver earrings:1.0), exact_trigger"
    )
    assert negative == creator.CREATOR_TAG_PROMPT_DEFAULTS[mode]["negative_prompt"].replace(", ", ",") + ",1boy,man,penis,dick"
    assert info == original


@pytest.mark.parametrize("mode", ["illustrious", "anima"])
def test_tag_models_allow_style_selection_and_preserve_authored_quality_tags(creator, mode):
    info = character(style="custom", custom_style="handmade felt illustration")
    positive, negative = creator.CharacterCreatorV2.construct_creator_prompt(info, mode)
    assert positive.startswith("handmade felt illustration, quality_marker, simple background")
    assert negative.startswith("blurry,")
    no_style, _ = creator.CharacterCreatorV2.construct_creator_prompt(info, mode, include_style=False)
    assert "handmade felt illustration" not in no_style
    missing_style, _ = creator.CharacterCreatorV2.construct_creator_prompt(character(), mode)
    assert missing_style.startswith("quality_marker, simple background")
    assert "Style:" not in missing_style and "Typical species features" not in missing_style


@pytest.mark.parametrize("mode", ["illustrious", "anima"])
@pytest.mark.parametrize("sex", ["male", "female"])
@pytest.mark.parametrize("nsfw", [False, True])
def test_tag_models_keep_framing_and_gender_specific_anatomy(creator, mode, sex, nsfw):
    positive, _ = creator.CharacterCreatorV2.construct_creator_prompt(character(
        sex=sex, nsfw=nsfw, framing="full_body", body="small breasts, slim waist", breast_size="medium breasts",
    ), mode)
    assert "standing, full body" in positive and "cowboy_shot" not in positive
    assert "(slim waist" in positive
    assert ("medium breasts" in positive) == (sex == "female")
    assert "small breasts" not in positive
    assert ("naked, nude" in positive) == nsfw
    assert ("wear white" in positive) != nsfw


@pytest.mark.parametrize("enabled", [False, True])
def test_qi2_enhancement_toggle_controls_only_creator_text_generation(creator, monkeypatch, enabled):
    calls = []
    prompt_server = SimpleNamespace()
    monkeypatch.setattr(creator.server.PromptServer, "instance", prompt_server)

    def dispatch(name, **values):
        assert prompt_server.last_prompt_id == "vnccs_character_creator_v2"
        calls.append((name, values))
        if name == "TextGenerate":
            return ('{"rewritten_prompt":"One character with black hair with fine strands.","wh_ratio":"2:3"}',)
        assert name == "TextEncodeQwenImage21"
        return "positive", "negative", None

    monkeypatch.setitem(sys.modules, "_vnccs.nodes.character_generator", SimpleNamespace(_call_comfy_node=dispatch))
    info = character()
    positive, negative = creator.CharacterCreatorV2.construct_creator_prompt(info, "qi2", include_style=False)
    settings = creator.normalize_gen_settings({"generation_mode": "qi2", "qi2_prompt_enhancement": enabled})
    result = creator.encode_generation_conditioning(
        "clip", "vae", positive, negative, settings,
        style_reference=creator._character_style_prompt(info), character_info=info, creator_prompt=True,
    )
    assert [name for name, _ in calls] == (["TextGenerate"] if enabled else []) + ["TextEncodeQwenImage21"]
    assert result[2] == calls[-1][1]["prompt"]
    assert creator._character_style_prompt(info) in result[2] if enabled else "Style reference: Princess Mononoke; Howl's Moving Castle." in result[2]
    assert "LoRA triggers: exact_trigger" in result[2]
    assert ("black hair with fine strands." in result[2]) == enabled
    assert creator.normalize_gen_settings({"generation_mode": "qi2"})["qi2_prompt_enhancement"] is True


def test_qi2_enhancement_disabled_in_creator_settings_does_not_change_shared_dispatch(creator, monkeypatch):
    calls = []
    monkeypatch.setattr(creator, "generate_qi2_prompt", lambda *a, **k: calls.append("legacy PE") or "legacy prompt")
    monkeypatch.setitem(sys.modules, "_vnccs.nodes.character_generator", SimpleNamespace(
        _call_comfy_node=lambda *a, **k: ("positive", "negative", None),
    ))
    result = creator.encode_generation_conditioning("clip", "vae", "source", "negative", {
        "generation_mode": "qi2", "qi2_prompt_enhancement": False,
    })
    assert calls == ["legacy PE"]
    assert result[2] == "legacy prompt"


def test_custom_creator_field_expansion_remains_available_for_restoring(creator, caplog):
    info = character(face="round face", breast_size="medium breasts", hair="black hair, ear-length hair, straight hair")
    expansions = {
        "age": "30 years-old; a fully grown adult with mature facial definition and restrained skin texture.",
        "race": "A human with ordinary human anatomy and naturally rounded ears.",
        "face": "Broad, softly curved cheeks flow into a gently rounded jaw and chin, avoiding sharp angular planes.",
        "breast_size": "Moderate chest volume projects outward in balanced, rounded contours relative to the ribcage. The selected bra follows this volume without flattening it or adding exaggerated lift.",
        "body": "Toned torso and limbs form an athletic silhouette with readable muscle contours, without bodybuilding bulk.",
        "hair": "Uniform black pigmentation is retained across smooth, straight strands. The hair ends around the ears rather than extending to the shoulders.",
        "eyes": "Clearly blue irises remain distinct from the pupils and sclera.",
        "skin_color": "An even tan complexion is maintained across the visible skin.",
        "additional_details": "Silver earrings remain attached to the ears as small persistent metallic accessories.",
        "aesthetics": "Keep the character contours readable and reproduce the specified physical details consistently.",
    }
    assert "Do NOT wrap the original tag" in creator.CREATOR_QI2_PROMPT_ENHANCER
    final = creator._creator_expanded_field_prompt(json.dumps({"fields": expansions}),
        creator._character_prompt_fields(info, "qi2"), creator._character_style_prompt(info))
    for key, label in {"age": "Age", "race": "Race", "face": "Facial features", "breast_size": "Breast size",
                       "body": "Build", "hair": "Hair", "eyes": "Eyes", "skin_color": "Skin color",
                       "additional_details": "Additional details", "aesthetics": "Additional style parameters"}.items():
        assert f"{label}: {expansions[key]}" in final
    assert "medium breasts (" not in final and "round face (" not in final
    assert "Typical species features" not in final
    assert "LoRA triggers: exact_trigger" in final
    event = next(record.vnccs for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompt_enhancement_result")
    assert event["applied"] is True
    assert set(event["changed_fields"]) == set(expansions)
    assert event["missing_fields"] == []


@pytest.mark.parametrize("response", ["", "{malformed", '{"fields":{"hair":"unused"}}'])
def test_official_creator_enhancer_reports_invalid_output_and_keeps_input(creator, monkeypatch, caplog, response):
    requests = []

    def dispatch(name, **values):
        assert name == "TextGenerate"
        requests.append(values["prompt"])
        return (response,)

    monkeypatch.setitem(sys.modules, "_vnccs.nodes.character_generator", SimpleNamespace(_call_comfy_node=dispatch))
    final = creator.generate_creator_qi2_prompt("clip", "", character_info=character(background_color="Transparent"))
    assert final == requests[0] + "\n\nLoRA triggers: exact_trigger"
    assert creator.QI2_ALPHA_BACKGROUND_PROMPT in final
    event = next(record.vnccs for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompt_enhancement_result")
    assert event["applied"] is False


@pytest.mark.parametrize("response", ['{"fields":{}}', 'not JSON'])
def test_creator_enhancer_reports_no_expansion_instead_of_claiming_success(creator, caplog, response):
    info = character()
    fields = creator._character_prompt_fields(info, "qi2")
    reference = creator._character_style_prompt(info)
    final = creator._creator_expanded_field_prompt(response, fields, reference)
    assert final == creator.CharacterCreatorV2.construct_creator_prompt(info, "qi2")[0]
    event = next(record.vnccs for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompt_enhancement_result")
    assert event["applied"] is False and event["changed_fields"] == []
    assert "age" in event["missing_fields"] and "breast_size" in event["missing_fields"]


def test_creator_enhancer_preserves_numeric_age_if_model_changes_it(creator):
    info = character(age=30)
    final = creator._creator_expanded_field_prompt('{"fields":{"age":"35 years-old; mature facial contours."}}',
        creator._character_prompt_fields(info, "qi2"))
    assert "Age: 30 years-old\n" in final
    assert "35 years-old" not in final


def test_creator_logs_readable_prompts_and_only_relevant_settings(creator, caplog):
    info = character(style="fortiche_arcane", style_label="Fortiche / Arcane")
    settings = creator.normalize_gen_settings({
        "generation_mode": "qi2", "seed": 123, "qi2_prompt_enhancement": False,
        "ckpt_name": "unrelated-checkpoint", "generation_defaults_version": 6,
        "lora_stack": [{"name": "", "strength": 1}, {"name": "Detail.safetensors", "strength": 0.25}],
    })
    creator._log_creator_generation_settings("Alice", info, settings, 792, 1408)
    message = caplog.records[-1].getMessage()
    assert "Character: Alice\n" in message
    assert "Resolution: 792 x 1408\n" in message
    assert "Seed: 123 (fixed)\n" in message
    assert "Prompt enhancement: Disabled\n" in message
    assert "LoRA: Detail.safetensors | Strength: 0.25" in message
    assert "unrelated-checkpoint" not in message and "generation_defaults_version" not in message
    assert "character_info=" not in message and "settings=" not in message and "\\n" not in message
    positive, negative = creator.CharacterCreatorV2.construct_creator_prompt(info, "qi2")
    creator.CharacterCreatorV2.log_generation_prompts("QI2 encoded preview", positive, negative)
    message = caplog.records[-1].getMessage()
    assert f"Positive prompt:\n{positive}\n\nNegative prompt:\n{negative}" in message
    assert "Style reference: Arcane.\nStyle description:" in message
    assert "positive=" not in message and "\\n" not in message


@pytest.mark.parametrize("race", ["human", "Human", "human, no ears"])
def test_creator_human_race_does_not_append_generic_species_notes(creator, race):
    info = character(race=race)
    positive, _ = creator.CharacterCreatorV2.construct_creator_prompt(info, "qi2")
    assert f"Race: {race}\n" in positive
    assert "Typical species features" not in positive
    assert "ordinary human anatomy" not in positive
    fields = creator._character_prompt_fields(info, "qi2")
    assert fields["race_features"] == ""
    fallback = creator._creator_expanded_field_prompt('{"fields":{}}', fields)
    assert f"Race: {race}\n" in fallback


def test_creator_fantasy_race_keeps_visual_anatomy_without_preset_headings(creator):
    info = character(race="elf, no tail")
    positive, _ = creator.CharacterCreatorV2.construct_creator_prompt(info, "qi2")
    assert "Race: elf, no tail, " in positive
    assert creator.RACE_PRESETS["elf"]["prompt"] in positive
    assert "Typical species features" not in positive and "Elf:" not in positive
    assert "Typical species features" in creator._qi2_character_fields(info)["race_features"]
