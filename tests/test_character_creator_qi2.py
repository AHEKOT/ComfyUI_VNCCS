import json

import pytest

from conftest import _preload_node


pytest.importorskip("torch")

creator = _preload_node("character_creator_v2")
from nodes import character_generator as generator


class _CloneableAsset:
    def __init__(self, name):
        self.name = name
        self.clone_calls = 0

    def clone(self):
        self.clone_calls += 1
        return _CloneableAsset(f"{self.name}-clone")


def test_qi2_preview_cache_never_reuses_text_encoder(monkeypatch):
    model = _CloneableAsset("model")
    first_clip = _CloneableAsset("first-clip")
    fresh_clip = _CloneableAsset("fresh-clip")
    vae = object()
    settings = {
        "generation_mode": "qi2",
        "diffusion_model_name": "qwen-model.safetensors",
        "clip_name": "qwen-clip.safetensors",
        "vae_name": "qwen-vae.safetensors",
    }
    creator.PREVIEW_CACHE.update({"asset_key": None, "asset_obj": None})
    monkeypatch.setattr(
        creator,
        "load_generation_assets",
        lambda _settings: (("qi2", "model", "clip", "vae"), model, first_clip, vae),
    )
    fresh_loads = []

    def load_fresh(_settings):
        fresh_loads.append(True)
        return fresh_clip

    monkeypatch.setattr(creator, "load_generation_clip", load_fresh)

    _model_one, clip_one, _vae_one = creator.acquire_preview_assets(settings)
    assert clip_one is first_clip
    assert creator.PREVIEW_CACHE["asset_obj"] == (model, None, vae)
    assert first_clip.clone_calls == 0

    _model_two, clip_two, _vae_two = creator.acquire_preview_assets(settings)
    assert clip_two is fresh_clip
    assert fresh_loads == [True]
    assert fresh_clip.clone_calls == 0


def test_qi2_settings_normalize_cache_and_turbo_defaults():
    normal = creator.normalize_gen_settings({"generation_mode": "qi2"})
    assert normal["steps"] == 25
    assert normal["cfg"] == 3.0
    assert normal["clip_type"] == "qwen_image"
    assert normal["qi2_cache"] == {"device": "gpu", "dtype": "int8"}

    turbo = creator.normalize_gen_settings({
        "generation_mode": "qi2",
        "turbo_enabled": True,
        "steps": 25,
        "cfg": 3.0,
        "qi2_cache": {"device": "cpu", "dtype": "int4"},
    })
    assert turbo["steps"] == 6
    assert turbo["cfg"] == 1.0
    assert turbo["qi2_cache"] == {"device": "cpu", "dtype": "int4"}


def test_qi2_prompt_uses_text_generate_without_media_then_system_encoder(monkeypatch):
    calls = []

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "TextGenerate":
            return (json.dumps({"rewritten_prompt": "A rewritten character portrait.", "wh_ratio": "2:3"}),)
        if name == "TextEncodeQwenImage21":
            return ("positive", "negative", "encoder latent")
        raise AssertionError(name)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    positive, negative, rewritten = creator.encode_generation_conditioning(
        "clip",
        "vae",
        "anime character prompt",
        "negative prompt",
        {"generation_mode": "qi2"},
    )

    assert (positive, negative, rewritten) == (
        "positive",
        "negative",
        "A rewritten character portrait.",
    )
    assert [name for name, _ in calls] == ["TextGenerate", "TextEncodeQwenImage21"]
    text_generate = calls[0][1]
    assert "# VNCCS Character Field Expansion" in text_generate["prompt"]
    assert text_generate["prompt"].endswith("User image request:\nanime character prompt")
    assert text_generate["max_length"] == 2048
    assert text_generate["sampling_mode"] == {
        "sampling_mode": "on",
        "temperature": 0.3,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "seed": 0,
        "presence_penalty": 0.0,
    }
    assert text_generate["thinking"] is False
    assert text_generate["use_default_template"] is True
    assert text_generate["mtp"] == "auto"
    assert not {"image", "video", "audio"}.intersection(text_generate)

    encoder = calls[1][1]
    assert encoder["prompt"] == "A rewritten character portrait."
    assert encoder["negative_prompt"] == "negative prompt"
    assert encoder["resolution"] == 1024
    assert encoder["images"] == {}


def test_qi2_prompt_rewriter_has_builtin_fallback(monkeypatch):
    monkeypatch.setattr(creator.os.path, "isfile", lambda _path: False)

    prompt = creator._qi2_prompt_rewriter_system_prompt()

    assert prompt.startswith("# VNCCS Character Field Expansion")
    assert '"fields"' in prompt


def test_generate_text_initializes_preview_progress_context(monkeypatch):
    prompt_server = creator.server.PromptServer.instance
    monkeypatch.delattr(prompt_server, "last_prompt_id", raising=False)

    def fake_node(name, **_kwargs):
        assert name == "TextGenerate"
        assert prompt_server.last_prompt_id == "vnccs_character_creator_v2"
        return (json.dumps({"rewritten_prompt": "Character portrait."}),)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    assert creator.generate_qi2_prompt("clip", "character") == "Character portrait."


def test_qi2_rewriter_extracts_json_after_thinking_text():
    generated = '<think>Internal reasoning.</think>\n{"rewritten_prompt":"Final description.","wh_ratio":"2:3"}'
    assert creator._qi2_rewritten_prompt(generated, "fallback") == "Final description."


def test_qi2_rewriter_preserves_required_alpha_instruction(monkeypatch):
    def fake_node(name, **_kwargs):
        assert name == "TextGenerate"
        return (json.dumps({"rewritten_prompt": "A clean character cutout:1.0"}),)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    rewritten = creator.generate_qi2_prompt(
        "clip",
        "anime character, transparent background with alpha channel",
    )

    assert rewritten == (
        "A clean character cutout\n"
        "transparent background with alpha channel"
    )


def test_qi2_character_prompt_uses_alpha_and_natural_framing():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt({
        "sex": "female",
        "age": 24,
        "framing": "cowboy_shot",
        "background_color": "Transparent",
        "race": "human",
    }, "qi2")

    assert "transparent background with alpha channel" in positive
    assert "show the complete head and body down to mid-thigh" in positive
    assert "cowboy_shot" not in positive


def test_character_prompt_removes_unit_weights_for_every_model():
    info = {
        "sex": "female",
        "age": 24,
        "background_color": "Green",
        "race": "(elf:1.0)",
        "lora_prompt": "trigger:1.0",
    }

    for mode in ("illustrious", "anima", "qi2"):
        positive, negative = creator.CharacterCreatorV2.construct_prompt({
            **info,
            "negative_prompt": "artifact:1.0",
        }, mode)
        assert ":1.0" not in positive
        assert ":1.0" not in negative


@pytest.mark.parametrize("style", list(creator.CHARACTER_STYLE_PROMPTS))
def test_qi2_keeps_every_catalog_reference_outside_pe_and_appends_it_verbatim(monkeypatch, style):
    calls = []
    rewritten_body = "A woman with blue eyes and long black hair stands against a green background."

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "TextGenerate":
            return (json.dumps({"rewritten_prompt": rewritten_body}),)
        assert name == "TextEncodeQwenImage21"
        return "positive", "negative", "latent"

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    info = {
        "style": style, "age": 30, "eyes": "blue eyes", "hair": "black hair, long hair",
        "background_color": "Green",
    }
    body, negative = creator.CharacterCreatorV2.construct_prompt(info, "qi2", include_style=False)
    reference = creator._character_style_prompt(info)
    positive_cond, negative_cond, final_prompt = creator.encode_generation_conditioning(
        "clip", "vae", body, negative, {"generation_mode": "qi2"}, style_reference=reference,
    )

    assert (positive_cond, negative_cond) == ("positive", "negative")
    assert [name for name, _ in calls] == ["TextGenerate", "TextEncodeQwenImage21"]
    pe_prompt = calls[0][1]["prompt"]
    assert reference not in pe_prompt
    assert reference.splitlines()[1] not in pe_prompt
    assert pe_prompt.endswith("User image request:\n" + body)
    assert "blue eyes" in pe_prompt
    assert creator.QI2_STYLE_REWRITE_RULES in pe_prompt
    assert final_prompt == f"{rewritten_body}\n\n{creator.QI2_STYLE_REFERENCE_HEADING}\n{reference}"
    assert final_prompt.count(reference) == 1
    assert calls[1][1]["prompt"] == final_prompt
    assert calls[1][1]["negative_prompt"] == negative


@pytest.mark.parametrize("generated", ["", "Plain rewritten body.", '{"rewritten_prompt": "JSON body:1.0"}'])
@pytest.mark.parametrize("missing_system_file", [False, True])
def test_qi2_reference_survives_fallback_and_alpha_repair(monkeypatch, generated, missing_system_file):
    reference = "Artist's Custom Style:1.0\nReference: Exact Title (1995)."
    if missing_system_file:
        monkeypatch.setattr(creator.os.path, "isfile", lambda _path: False)

    def fake_node(name, **kwargs):
        assert name == "TextGenerate"
        assert reference not in kwargs["prompt"]
        assert creator.QI2_STYLE_REWRITE_RULES in kwargs["prompt"]
        return (generated,)

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    original = "blue eyes, transparent background with alpha channel"
    final = creator.generate_qi2_prompt("clip", original, style_reference=reference)
    body, style_block = final.split("\n\n" + creator.QI2_STYLE_REFERENCE_HEADING + "\n", 1)
    assert style_block == reference
    assert final.count(reference) == 1
    assert creator.QI2_ALPHA_BACKGROUND_PROMPT in body
    assert ":1.0" not in body
    if not generated:
        assert body == original


def test_empty_custom_style_does_not_create_an_empty_reference_block(monkeypatch):
    monkeypatch.setattr(generator, "_call_comfy_node", lambda *_args, **_kwargs: (
        '{"rewritten_prompt": "A character."}',
    ))
    reference = creator._character_style_prompt({"style": "custom", "custom_style": ""})
    assert creator.generate_qi2_prompt("clip", "character", style_reference=reference) == "A character."


@pytest.mark.parametrize("mode", ["illustrious", "anima"])
def test_non_qi2_conditioning_keeps_direct_style_prompt(monkeypatch, mode):
    info = {"style": "fortiche_arcane", "age": 30, "eyes": "blue eyes"}
    positive, negative = creator.CharacterCreatorV2.construct_prompt(info, mode)
    encoded = []

    def encode(_clip, text, _settings):
        encoded.append(text)
        return text

    def unexpected_pe(*_args, **_kwargs):
        raise AssertionError("PE must not run for Anima or Illustrious")

    monkeypatch.setattr(creator, "encode_generation_prompt", encode)
    monkeypatch.setattr(creator, "generate_qi2_prompt", unexpected_pe)
    result = creator.encode_generation_conditioning(
        "clip", "vae", positive, negative, {"generation_mode": mode},
        style_reference=creator._character_style_prompt(info),
    )
    assert encoded == [positive, negative]
    assert result == (positive, negative, positive)
    assert positive.count("Style: Fortiche / Arcane.") == 1


def _cat_character_info(**changes):
    info = {
        "sex": "female", "age": 18, "framing": "cowboy_shot",
        "style": "trigger_imaishi", "race": "cat girl", "skin_color": "",
        "body": "small breasts", "face": "freckles",
        "hair": "green hair, grey hair, two-tone hair, multicolored hair",
        "eyes": "pink eyes, red eyes, heterochromia",
        "additional_details": "tan lines", "background_color": "Green",
        "nsfw": False, "aesthetics": "", "lora_prompt": "",
    }
    info.update(changes)
    return info


@pytest.mark.parametrize("missing_system_file", [False, True])
def test_qi2_structured_character_reaches_encoder_with_sources_and_expansions(monkeypatch, missing_system_file):
    info = _cat_character_info(lora_prompt="exact_model_trigger", aesthetics="quality_marker")
    reference = creator._character_style_prompt(info)
    calls = []
    if missing_system_file:
        monkeypatch.setattr(creator.os.path, "isfile", lambda _path: False)

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "TextGenerate":
            pe_input = kwargs["prompt"].rsplit("character_fields:\n", 1)[1]
            fields = json.loads(pe_input)
            # The screen's separate fields must survive without tag normalization.
            for key in ("race", "body", "face", "hair", "eyes", "additional_details"):
                assert fields[key] == info[key]
            assert fields["age"] == "18 years old"
            assert fields["skin_color"] == ""
            assert fields["background"] == "solid Green background"
            assert "mid-thigh" in fields["framing"]
            assert reference not in kwargs["prompt"]
            assert "exact_model_trigger" not in kwargs["prompt"]
            assert "quality_marker" not in kwargs["prompt"]
            expansions = {key: value for key, value in fields.items() if value}
            expansions.update({
                "race": "A humanoid cat girl with attached feline ears and a feline tail.",
                "age": "An 18-year-old young adult with adult skeletal proportions and youthful skin texture.",
                "body": "Small breasts with low volume and modest projection relative to the ribcage.",
                "additional_details": "Lighter swimsuit-covered skin contrasts with darker exposed skin.",
                # Unrequested fields and guesses for blank fields must never reach the encoder.
                "skin_color": "invented ivory complexion",
                "props": "invented sword",
                "lora_prompt": "replaced_model_trigger",
            })
            return (json.dumps({"fields": expansions}),)
        assert name == "TextEncodeQwenImage21"
        return "positive", "negative", "latent"

    monkeypatch.setattr(generator, "_call_comfy_node", fake_node)
    body, negative = creator.CharacterCreatorV2.construct_prompt(info, "qi2", include_style=False)
    result = creator.encode_generation_conditioning(
        "clip", "vae", body, negative, {"generation_mode": "qi2"},
        style_reference=reference, character_info=info,
    )
    final = result[2]
    assert result[:2] == ("positive", "negative")
    assert [name for name, _kwargs in calls] == ["TextGenerate", "TextEncodeQwenImage21"]
    assert calls[1][1]["prompt"] == final
    for key in ("race", "body", "face", "hair", "eyes", "additional_details"):
        assert f"supplied value: {info[key]}" in final
    assert "attached feline ears and a feline tail" in final
    assert "low volume and modest projection" in final
    assert "Lighter swimsuit-covered skin" in final
    assert "18 years old" in final
    assert "quality_marker" in final and "exact_model_trigger" in final
    assert "replaced_model_trigger" not in final
    assert "invented" not in final
    assert final.endswith(creator.QI2_STYLE_REFERENCE_HEADING + "\n" + reference)
    assert final.count(reference) == 1


@pytest.mark.parametrize("generated", [
    "", "A plain human character with no other details.",
    '{"rewritten_prompt":"A lossy legacy paragraph."}',
    '{"fields": {"race": "truncated',
    '{"fields": null}',
    '{"fields": []}',
    '{"fields":{"race":null,"age":18,"body":["small"],"eyes":{"color":"pink"}}}',
    '<think>{"fields":{"race":"Unfinished internal reasoning"}}',
])
def test_qi2_invalid_or_lossy_response_retains_all_source_fields(monkeypatch, generated):
    monkeypatch.setattr(generator, "_call_comfy_node", lambda *_args, **_kwargs: (generated,))
    info = _cat_character_info(background_color="Transparent")
    final = creator.generate_qi2_prompt("clip", "legacy body", character_info=info)
    for value in creator._qi2_character_fields(info).values():
        if value:
            assert value in final
    assert "Visual explanation:" not in final
    assert "plain human" not in final and "lossy legacy" not in final
    assert "Unfinished internal reasoning" not in final
    assert "legacy body" not in final


def test_qi2_partial_response_cannot_remove_attributes_or_populate_empty_fields(monkeypatch):
    response = {"fields": {
        "hair": "Two-tone hair.",  # Lost the two colors: source still preserves them.
        "race": "",  # Omitted species entirely: source still preserves it.
        "body": "",  # Omitted breast size: source still preserves it.
        "skin_color": "An invented skin tone.",
    }}
    monkeypatch.setattr(generator, "_call_comfy_node", lambda *_args, **_kwargs: (json.dumps(response),))
    info = _cat_character_info(race="cat girl, no tail, human ears absent")
    final = creator.generate_qi2_prompt("clip", "", character_info=info)
    assert "cat girl, no tail, human ears absent" in final
    assert "small breasts" in final
    assert info["hair"] in final
    assert "Two-tone hair." in final
    assert "invented skin tone" not in final


def test_qi2_discards_json_in_thinking_and_uses_final_field_object():
    response = (
        '<think>{"fields":{"race":"Discard this thought"}}</think>\n'
        '{"fields":{"race":"An elf with pointed ears."}}'
    )
    final = creator._qi2_expanded_field_prompt(response, {"race": "elf"})
    assert "An elf with pointed ears." in final
    assert "Discard this thought" not in final


@pytest.mark.parametrize("age", [18, 23, 37, 58, 76])
def test_qi2_preserves_exact_age_and_independent_body_details(age):
    info = _cat_character_info(age=age, body="small breasts, broad shoulders, muscular arms")
    fields = creator._qi2_character_fields(info)
    final = creator._qi2_expanded_field_prompt('{"fields":{}}', fields)
    assert f"Age — supplied value: {age} years old" in final
    assert "small breasts, broad shoulders, muscular arms" in final


@pytest.mark.parametrize("sex,nsfw", [("female", False), ("male", False), ("female", True), ("male", "true")])
def test_qi2_field_clothing_matches_existing_creator_policy(sex, nsfw):
    info = _cat_character_info(sex=sex, nsfw=nsfw, age=30)
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(info, "qi2", include_style=False)
    assert creator._qi2_character_fields(info)["clothing"] in positive
