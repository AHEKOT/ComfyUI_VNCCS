from pathlib import Path
import re


SOURCE = (
    Path(__file__).resolve().parents[1] / "web" / "vnccs_character_generator.js"
).read_text(encoding="utf-8")


COMMON = (Path(__file__).resolve().parents[1] / "web" / "vnccs_common.js").read_text(encoding="utf-8")


def test_emotions_generator_hides_face_denoise_slider_for_qi2():
    assert "face_denoise: 0.55" in SOURCE
    assert 'slider.type = "range"' in SOURCE
    assert 'this.set("emotion_generation", "face_denoise", next)' in SOURCE
    assert 'return this.connectedEmotionStudioMode() !== "qi2"' in SOURCE
    assert "if (this.shouldShowEmotionDenoiseControl())" in SOURCE
    assert 'this.block("Emotion Strength", [' in SOURCE
    assert "this.faceDenoiseSlider()" in SOURCE
    assert "target_size: 2048" in SOURCE
    assert 'this.block("VNCCS BBox Extractor", [' in SOURCE
    assert 'this.resolutionScaleSlider("emotion_generation", "target_size")' in SOURCE
    assert 'this.faceDetailerNumberField("bbox_dilation", "dilation"' in SOURCE
    assert 'this.faceDetailerNumberField("feather", "feather"' in SOURCE
    assert 'textarea("emotion_generation", "qi2_prompt_template", "prompt template")' in SOURCE


def test_sam_defaults_are_disabled_and_native_hides_recovery_controls():
    assert "use_sam: false" in SOURCE
    assert "use_sam3_details_recovery: false" in SOURCE
    assert "if (!this.isNativeBgRemove())" in SOURCE
    assert 'this.block("BG Remove", this.bgRemoveFields())' in SOURCE


def test_seedvr_upscaler_exposes_resolution_controls():
    assert 'number("upscaler", "resolution", "target short edge", 16, 16384, 2)' in SOURCE
    assert 'number("upscaler", "max_resolution", "maximum edge (0 = unlimited)", 0, 16384, 2)' in SOURCE
    assert 'this.field("upscaler", "resolution", "target short edge", "number", { min: 16, max: 16384, step: 2 })' in SOURCE
    assert 'this.field("upscaler", "max_resolution", "maximum edge", "number", { min: 0, max: 16384, step: 2 })' in SOURCE


def test_pose_resolution_control_uses_clear_label():
    assert 'caption.textContent = "resolution scale"' in SOURCE
    assert 'slider.type = "range"' in SOURCE
    assert "RESOLUTION_SCALE_MIN_MP = 1" in COMMON
    assert "RESOLUTION_SCALE_MAX_MP = 4" in COMMON
    assert "RESOLUTION_SCALE_STEP_MP = 0.1" in COMMON
    assert "[1.3, 1344]" in COMMON
    assert "[1.5, 1536]" in COMMON
    assert "finiteResolutionScaleValue(slider.value)" in SOURCE
    assert '"target_size", "scale area", "select"' not in SOURCE


def test_seedvr_model_card_uses_persistent_widget_setter():
    assert 'this.set("upscaler", "model", rel);' in SOURCE
    assert "this.data.upscaler.model = rel;" not in SOURCE


def test_emotion_result_tabs_stay_single_row_and_only_show_final_stage():
    assert '["emotion_0001_bg_remove", "Emotion"]' in SOURCE
    assert 'return [`${key}_bg_remove`, label];' in SOURCE
    assert 'grid-template-columns: repeat(var(--vnccs-stage-count, 1), minmax(0, 1fr));' in SOURCE
    assert '.vnccs-pipe-root.is-emotions .vnccs-pipe-tabs' in SOURCE
    assert 'flex-wrap: nowrap;' in SOURCE


def test_clone_stage_chain_fits_two_compact_rows_without_clipping():
    def rule(selector):
        return next(body for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", SOURCE)
                    if selector in [part.strip() for part in selectors.split(",")])

    main = rule(".vnccs-pipe-root.is-clone .vnccs-pipe-main")
    chain = rule(".vnccs-pipe-chain.is-clone")
    stage = rule(".vnccs-pipe-chain.is-clone .vnccs-pipe-stage")
    actions = rule(".vnccs-pipe-chain.is-clone .vnccs-pipe-stage-actions")
    main_rows = re.search(r"grid-template-rows: minmax\(0, 1fr\) (\d+)px;", main)
    assert main_rows, "Clone stage chain needs a reserved row in the preview layout"
    track_heights = re.search(r"grid-template-rows: ([\dpx ]+);", stage).group(1)
    tracks = list(map(int, re.findall(r"(\d+)px", track_heights)))
    stage_gap = int(re.search(r"gap: (\d+)px", stage).group(1))
    stage_padding = int(re.search(r"padding: (\d+)px", stage).group(1))
    stage_border = int(re.search(r"border: (\d+)px", rule(".vnccs-pipe-stage")).group(1))
    chain_gap = int(re.search(r"gap: (\d+)px", chain).group(1))
    chain_padding = int(re.search(r"padding: (\d+)px", chain).group(1))
    chain_border = int(re.search(r"border-top: (\d+)px", rule(".vnccs-pipe-chain")).group(1))
    card_height = sum(tracks) + (len(tracks) - 1) * stage_gap + 2 * stage_padding + 2 * stage_border
    assert int(main_rows.group(1)) >= 2 * card_height + chain_gap + 2 * chain_padding + chain_border
    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in chain
    assert "grid-auto-rows: minmax(0, 1fr);" in chain
    assert "box-sizing: border-box;" in chain
    assert "min-height: 0;" in chain
    assert "overflow-y: auto;" in chain
    assert "display: grid;" in stage
    assert "padding: 4px 8px;" in stage
    assert "grid-column: 2;" in actions
    assert "grid-row: 1;" in actions


def test_emotion_tabs_resync_after_late_studio_restore():
    assert "const sourceChanged = this.syncCharacterSourceData();" in SOURCE
    assert "if (!sourceChanged && !modelChanged) return;" in SOURCE
    assert "const source = this.connectedEmotionStudioNode();" in SOURCE
    assert "this.renderPreview();" in SOURCE
    assert "this.renderChain();" in SOURCE


def test_new_qi2_emotion_generators_receive_bbox_defaults_without_migrating_saved_workflows():
    assert "bbox_threshold: 0.3" in SOURCE
    assert "bbox_dilation: 50" in SOURCE
    assert "feather: 50" in SOURCE
    assert "drop_size: 10" in SOURCE
    assert "this.qi2EmotionDefaultsPending = this.isEmotions;" in SOURCE
    assert "this._vnccsCharacterGeneratorWidget.qi2EmotionDefaultsPending = false;" in SOURCE
