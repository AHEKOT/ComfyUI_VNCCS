import json
from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "web" / "vnccs_character_creator_v2.js").read_text()
STYLE_CATALOG = json.loads(
    (Path(__file__).parents[1] / "character_template" / "character_styles.json").read_text()
)


def test_resolution_scale_is_a_one_to_four_megapixel_slider():
    assert 'resolutionSlider.type = "range"' in SOURCE
    assert "RESOLUTION_SCALE_MIN_MP = 1" in SOURCE
    assert "RESOLUTION_SCALE_MAX_MP = 4" in SOURCE
    assert "RESOLUTION_SCALE_STEP_MP = 0.1" in SOURCE
    assert "[1.3, 1344]" in SOURCE
    assert "[1.5, 1536]" in SOURCE
    assert "resolutionScaleValue(resolutionSlider.value)" in SOURCE
    assert 'resolutionLabel.textContent = "Resolution scale"' in SOURCE


def test_resolution_scale_is_persisted_for_every_generation_mode():
    assert 'illustrious: ["target_size", "ckpt_name"' in SOURCE
    assert 'anima: ["target_size", "diffusion_model_name"' in SOURCE
    assert 'qi2: ["target_size", "diffusion_model_name"' in SOURCE
    assert "profile.target_size = resolutionScaleValue(" in SOURCE
    assert "LEGACY_ANIMA_RESOLUTION_SCALES" in SOURCE
    assert "delete profile.resolution_preset" in SOURCE


def test_framing_selector_is_between_age_and_race_and_persisted():
    age = 'createSlider("Age", "age", 1, 100, 1, state.character_info)'
    framing = 'createField("Framing", "framing", "select"'
    race = 'createField("Race", "race")'

    assert SOURCE.index(age) < SOURCE.index(framing) < SOURCE.index(race)
    assert 'framing: "cowboy_shot"' in SOURCE


def test_style_selector_is_between_framing_and_race_with_custom_first():
    framing = 'createField("Framing", "framing", "select"'
    style = "createStyleField()"
    race = 'createField("Race", "race")'

    assert SOURCE.index(framing) < SOURCE.index(style) < SOURCE.index(race)
    assert 'style: DEFAULT_CHARACTER_STYLE, custom_style: ""' in SOURCE
    assert 'api.fetchApi("/vnccs/character_styles")' in SOURCE
    assert 'select.add(new Option("CUSTOM STYLE", "custom"))' in SOURCE
    assert 'customInput.style.display = normalized === "custom" ? "block" : "none"' in SOURCE


def test_style_selector_has_readable_character_focused_options():
    assert 'select.className = "vnccs-select vnccs-style-select"' in SOURCE
    assert ".vnccs-style-select {\n    font-size: 14px;" in SOURCE
    assert "font-size: 16px;\n    line-height: 1.55;" in SOURCE
    assert "group.styles.forEach(style =>" in SOURCE
    assert "group.styles.map(style => style.id)" in SOURCE

    labels = {
        style["label"]
        for group in STYLE_CATALOG["groups"]
        for style in group["styles"]
    }
    assert {
        "Soft Pastel Anime",
        "Bold Cel Anime",
        "Cinematic Anime",
        "Manga Ink & Screentone",
        "Anime Game Character Art",
        "Classical Oil Character",
    }.issubset(labels)
    assert {"Shoujo Anime", "Shonen Anime", "Seinen Anime", "Cubist Geometric"}.isdisjoint(labels)
    assert STYLE_CATALOG["aliases"]["shoujo_anime"] == "soft_pastel_anime"
    assert "Soft Pastel Anime" not in SOURCE


def test_aesthetics_defaults_do_not_force_anime():
    assert "const PROMPT_DEFAULTS_VERSION = 3;" in SOURCE
    assert 'aesthetics: "masterpiece, best quality, score_7"' in SOURCE
    assert 'aesthetics: "",' in SOURCE
    assert 'aesthetics: "masterpiece, best quality, score_7, anime"' not in SOURCE
    assert 'removePromptToken(mergedModes.anima.aesthetics, "anime")' in SOURCE
    assert 'removePromptToken(mergedModes.qi2.aesthetics, "anime")' in SOURCE


def test_character_selects_share_normal_input_height():
    assert "zoom: 1.5" not in SOURCE
    assert ".vnccs-input,\n.vnccs-select {\n    height: 34px;" in SOURCE
    assert "min-height: 34px" in SOURCE
    assert "shot_type" not in SOURCE
    assert 'createSegmentedField("Framing"' not in SOURCE
    assert '{ label: "Cowboy shot", value: "cowboy_shot" }' in SOURCE
    assert '{ label: "Full body", value: "Full_body" }' in SOURCE


def test_qi2_generation_profile_exposes_model_cache_and_viggle_controls():
    assert '["qi2", "Qwen Image 2.1"]' in SOURCE
    assert 'qi2CacheTitle.innerText = "Qwen Image 2.1 Cache"' in SOURCE
    assert '{ label: "Alpha", value: "Transparent" }' in SOURCE
    assert 'state.character_info.background_color = "Transparent"' in SOURCE
    assert '["auto", "gpu", "cpu", "off"]' in SOURCE
    assert '["default", "int8", "int4"]' in SOURCE
    assert 'renderModeLoraCards(els.qi2LoraCards, "qi2")' in SOURCE
    assert 'state.gen_settings.steps = mode === "qi2" ? 6 : 12;' in SOURCE
    assert 'const QI2_DEFAULTS = {' in SOURCE
    assert 'steps: 25, cfg: 3.0' in SOURCE
