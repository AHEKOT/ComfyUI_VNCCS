from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "web" / "vnccs_clothes_designer.js"
).read_text(encoding="utf-8")


def test_clothes_core_card_filters_by_connected_model_kind():
    assert "const getConnectedModelKind = () =>" in SOURCE
    assert "const loraMatchesConnectedKind = (entry) =>" in SOURCE
    assert "entryKind === selectedKind" in SOURCE
    assert "const findConnectedClothesCoreLora = (preferredPath = \"\") =>" in SOURCE


def test_clothes_core_sync_resolves_from_its_connected_control_center():
    assert "setClothesCoreLora();" in SOURCE
    assert "const clothesCore = options.find(isClothesCoreLora);" not in SOURCE
    assert 'if (lower.startsWith("models/loras/"))' in SOURCE


def test_resolution_scale_is_a_one_to_four_megapixel_slider():
    assert 'resolutionSlider.type = "range"' in SOURCE
    assert "RESOLUTION_SCALE_MIN_MP = 1" in SOURCE
    assert "RESOLUTION_SCALE_MAX_MP = 4" in SOURCE
    assert "RESOLUTION_SCALE_STEP_MP = 0.1" in SOURCE
    assert "[1.3, 1344]" in SOURCE
    assert "[1.5, 1536]" in SOURCE
    assert "resolutionScaleValue(resolutionSlider.value)" in SOURCE
    assert 'new Option("Auto (1024)", "")' not in SOURCE


def test_qi2_background_defaults_to_transparent_without_wrapping_controls():
    assert '{ label: "Alpha", value: "Transparent" }' in SOURCE
    assert 'state.gen_settings.background_color = "Transparent"' in SOURCE
    assert 'kind === "qi2"' in SOURCE
    assert ".vnccs-segmented-field.is-three" in SOURCE
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in SOURCE
    assert "white-space: nowrap" in SOURCE
    assert 'btn.setAttribute("aria-pressed", String(selected))' in SOURCE
