import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { registerCleanup, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText } from "./vnccs_common.js";

const GENERATOR_QWEN_INSTRUCTION = "Describe the character and their key features (body shape, physical characteristics, clothing, items, accessories). Then explain how the user's text instruction should alter or modify the character. Generate a new image that meets the user's requirements while maintaining consistency with the original character where appropriate.";
const QI2_EMOTION_PROMPT_TEMPLATE = "Upscale face image.\nMake character's face emotion {emotion}\nChange only face. Keep original neck colour, clothes and hairs\nkeep character's clothes";
const QI2_EMOTION_BBOX_DEFAULTS = Object.freeze({
    bbox_threshold: 0.3,
    bbox_dilation: 50,
    feather: 50,
    drop_size: 10,
});
const RESOLUTION_SCALE_BASE = 1024;
const RESOLUTION_SCALE_MIN_MP = 1;
const RESOLUTION_SCALE_MAX_MP = 4;
const RESOLUTION_SCALE_STEP_MP = 0.1;
const RESOLUTION_SCALE_PRESETS = new Map([
    [1.3, 1344],
    [1.5, 1536],
]);

function resolutionScaleMegapixels(value) {
    const numeric = Number(value);
    const megapixels = Number.isFinite(numeric) ? numeric / RESOLUTION_SCALE_BASE : RESOLUTION_SCALE_MIN_MP;
    return Math.max(RESOLUTION_SCALE_MIN_MP, Math.min(RESOLUTION_SCALE_MAX_MP, megapixels));
}

function resolutionScaleValue(megapixels) {
    const numeric = Number(megapixels);
    const clamped = Math.max(RESOLUTION_SCALE_MIN_MP, Math.min(RESOLUTION_SCALE_MAX_MP, Number.isFinite(numeric) ? numeric : RESOLUTION_SCALE_MIN_MP));
    const stepped = Number((Math.round(clamped / RESOLUTION_SCALE_STEP_MP) * RESOLUTION_SCALE_STEP_MP).toFixed(1));
    return RESOLUTION_SCALE_PRESETS.get(stepped) ?? Math.round(stepped * RESOLUTION_SCALE_BASE);
}

function resolutionScaleText(value) {
    return `${resolutionScaleMegapixels(value).toFixed(1)} MP`;
}

const DEFAULT_DATA = {
    nsfw_enabled: true,
    emotion_pairs: [],
    common: {
        target_size: 1024,
    },
    pose_generation: {
        target_size: 1024,
        upscale_method: "lanczos",
        crop_method: "disabled",
        image1_name: "image 1",
        image2_name: "image 2",
        image3_name: "image 3",
        weight1: 1,
        weight2: 1,
        weight3: 1,
        vl_size: 384,
        background_color: "from_generator",
        latent_image_index: 1,
        instruction: GENERATOR_QWEN_INSTRUCTION,
    },
    pose_sampler: {
        inherit_pipe: true,
        seed: 0,
        steps: 20,
        cfg: 1,
        sampler_name: "euler",
        scheduler: "simple",
        denoise: 1,
    },
    vae_decode: {
        tile_size: 512,
        overlap: 64,
        temporal_size: 64,
        temporal_overlap: 8,
    },
    emotion_generation: {
        task_batch_size: 0,
        target_size: 2048,
        face_denoise: 0.55,
        use_sam: false,
        bbox_model: "bbox/face_yolov8m.pt",
        segm_model: "bbox/face_yolov8m.pt",
        sam_model: "sam_vit_b_01ec64.pth",
        sam_device_mode: "AUTO",
        guide_size: 1536,
        guide_size_for: true,
        max_size: 1536,
        inherit_pipe_sampler: true,
        sampler_name: "euler",
        scheduler: "simple",
        feather: 5,
        noise_mask: true,
        force_inpaint: true,
        bbox_threshold: 0.5,
        bbox_dilation: 10,
        qi2_prompt_template: QI2_EMOTION_PROMPT_TEMPLATE,
        bbox_crop_factor: 3,
        sam_detection_hint: "center-1",
        sam_dilation: 0,
        sam_threshold: 0.93,
        sam_bbox_expansion: 0,
        sam_mask_hint_threshold: 0.7,
        sam_mask_hint_use_negative: "False",
        drop_size: 10,
        cycle: 1,
        inpaint_model: false,
        noise_mask_feather: 20,
        tiled_encode: true,
        tiled_decode: true,
        matte_expand_radius: 8,
        matte_feather_radius: 4,
        chroma_context: 16,
    },
    remove_clothes: {
        prompt: "Dress character: White underwear",
        target_size: 1024,
        upscale_method: "lanczos",
        crop_method: "disabled",
        image1_name: "image 1",
        image2_name: "image 2",
        image3_name: "image 3",
        weight1: 1,
        weight2: 1,
        weight3: 1,
        vl_size: 384,
        background_color: "White",
        latent_image_index: 1,
        instruction: GENERATOR_QWEN_INSTRUCTION,
    },
    remove_clothes_sampler: {
        inherit_pipe: true,
        seed: 0,
        steps: 20,
        cfg: 1,
        sampler_name: "euler",
        scheduler: "simple",
        denoise: 1,
    },
    upscaler: {
        mode: "seedvr",
        model: "seedvr2_3b_fp8_e4m3fn.safetensors",
        vae: "ema_vae_fp16.safetensors",
        device: "cuda:0",
        offload_device: "cpu",
        seed: 42,
        inherit_pipe_seed: true,
        resolution: 2048,
        max_resolution: 3840,
        batch_size: 1,
        uniform_batch_size: false,
        color_correction: "lab",
        temporal_overlap: 0,
        prepend_frames: 0,
        input_noise_scale: 0,
        latent_noise_scale: 0,
        blocks_to_swap: 0,
        swap_io_components: false,
        cache_dit: true,
        attention_mode: "sdpa",
        attention_mode_manual: false,
        encode_tiled: true,
        encode_tile_size: 1024,
        encode_tile_overlap: 128,
        decode_tiled: true,
        decode_tile_size: 1024,
        decode_tile_overlap: 128,
        tile_debug: "false",
        cache_vae: false,
        enable_debug: false,
    },
    bg_remove: {
        // TODO: Decide what to do with internal RMBG later.
        use_internal_rmbg: false,
        preset: "balanced",
        use_sam3_details_recovery: false,
        use_preset_values: true,
        tolerance: 0.15,
        softness: 0.12,
        despill_strength: 0.65,
        edge_width: 3,
        matte_cleanup: 0.10,
        foreground_recover: 0.35,
        edge_decontaminate: 0.75,
        edge_choke: 0.08,
        matte_method: "guided_edge",
        screen_mode: "from_background",
        output_mode: "straight_rgba",
        sam3_model: "",
        sam3_segmentor: "image",
        sam3_device: "auto",
        sam3_precision: "bf16",
        sam3_prompt: "face, clothes, accessories, hat, boots, eyes",
        sam3_threshold: 0.40,
        sam3_add_background: "none",
        sam3_detection_limit: -1,
        sam3_erode_radius: 4,
        sam3_min_foreground_overlap: 0.55,
    },
    ui: {
        selected_preview: "pose_generation",
        user_selected_preview: false,
    },
};

const STAGES = [
    ["pose_generation", "Pose Generation"],
    ["upscaler", "Upscaler"],
    ["bg_remove", "BG Remove"],
];

const CLONE_STAGES = [
    ["original_pose_generation", "Original Pose"],
    ["original_upscaler", "Original Upscaler"],
    ["original_bg_remove", "Original BG"],
    ["remove_clothes", "Remove Clothes"],
    ["naked_pose_generation", "Naked Pose"],
    ["naked_upscaler", "Naked Upscaler"],
    ["naked_bg_remove", "Naked BG"],
];

const CLONE_SFW_STAGES = [
    ["original_pose_generation", "Original Pose"],
    ["original_upscaler", "Original Upscaler"],
    ["original_bg_remove", "Original BG"],
];

const CLOTHES_STAGES = [
    ["source_upscaler", "Source Upscaler"],
    ["pose_generation", "Pose Generation"],
    ["upscaler", "Upscaler"],
    ["bg_remove", "BG Remove"],
];

const DEFAULT_EMOTION_STAGES = [
    ["emotion_0001_bg_remove", "Emotion"],
];

const WORKFLOW_UPSCALER_DIT_MODELS = [
    "seedvr2_3b_fp16.safetensors",
    "seedvr2_3b_fp8_e4m3fn.safetensors",
    "seedvr2_7b_fp16.safetensors",
    "seedvr2_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_7b_sharp_fp16.safetensors",
    "seedvr2_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
];

const WORKFLOW_UPSCALER_VAE_MODELS = [
    "ema_vae_fp16.safetensors",
];

const SEEDVR_ATTENTION_MODES = ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"];
const SEEDVR_COLOR_CORRECTION_MODES = ["lab", "wavelet", "adain", "none"];
const NATIVE_SEEDVR_NODE_NAMES = ["SeedVR2Preprocess", "SeedVR2Conditioning", "SeedVR2PostProcessing"];
const BG_REMOVE_MODES = ["Native", "disabled", "ultra_light", "light", "balanced", "strong", "aggressive"];

const CLOTHES_CORE_LORA_LABEL = "VNCCS Clothes Core";

const CSS = `
.vnccs-pipe-root {
    width: 100%;
    height: 100%;
    display: grid;
    grid-template-columns: 290px minmax(0, 1fr);
    background: #0a0a0f;
    color: #e8e8f0;
    font-family: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: hidden;
    box-sizing: border-box;
    pointer-events: auto;
    position: relative;
}
.vnccs-pipe-settings {
    border-right: 1px solid rgba(255,143,163,0.16);
    background: #101018;
    padding: 10px 10px 64px;
    overflow-y: auto;
}
.vnccs-seedvr-cards { display:flex; flex-direction:column; gap:7px; }
.vnccs-seedvr-picker { display:flex; flex-direction:column; gap:8px; }
.vnccs-seedvr-picker-menu { display:none; flex-direction:column; gap:8px; padding:8px; border:1px solid rgba(255,143,163,.18); border-radius:10px; background:rgba(8,8,12,.48); }
.vnccs-seedvr-picker.is-open .vnccs-seedvr-picker-menu { display:flex; }
.vnccs-seedvr-card { display:flex; flex-direction:column; gap:5px; padding:10px 12px 8px; border:1px solid rgba(0,214,143,.25); border-radius:10px; background:rgba(0,214,143,.05); cursor:default; position:relative; overflow:hidden; transition:all .16s ease; }
.vnccs-seedvr-card.is-picker-head { min-height:58px; cursor:pointer; }
.vnccs-seedvr-card.is-installed { cursor:pointer; }
.vnccs-seedvr-card.is-installed:hover, .vnccs-seedvr-card.is-picker-head:hover { border-color:rgba(0,214,143,.42); background:rgba(0,214,143,.08); }
.vnccs-seedvr-card.is-selected { border-color:#ff8fa3; background:rgba(255,143,163,.12); box-shadow:0 0 0 1px rgba(255,143,163,.12) inset; }
.vnccs-seedvr-card.is-missing { opacity:.92; }
.vnccs-seedvr-card-head { display:flex; align-items:center; gap:7px; min-width:0; }
.vnccs-seedvr-card-name { flex:1; min-width:0; color:#e8e8f0; font-size:13px; font-weight:700; line-height:1.25; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.vnccs-seedvr-card-dot { width:12px; height:12px; border-radius:50%; flex:none; background:#ff627d; }
.vnccs-seedvr-card.is-installed .vnccs-seedvr-card-dot { background:#00d68f; }
.vnccs-seedvr-card-status { flex:none; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#ff627d; }
.vnccs-seedvr-card.is-installed .vnccs-seedvr-card-status { color:#00d68f; }
.vnccs-seedvr-card-desc { color:#aaa7b5; font-size:11px; line-height:1.4; }
.vnccs-seedvr-download { margin-top:2px; width:100%; padding:7px 9px; border:1px solid rgba(255,143,163,.32); border-radius:7px; background:rgba(255,143,163,.08); color:#ffb4c1; font-family:inherit; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; cursor:pointer; }
.vnccs-seedvr-download:hover { background:rgba(255,143,163,.14); }
.vnccs-seedvr-download:disabled { opacity:.45; cursor:wait; }
.vnccs-pipe-settings-open {
    position: absolute;
    left: 12px;
    bottom: 12px;
    z-index: 12;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    width: 266px;
    min-height: 38px;
    border: 1px solid rgba(255,143,163,0.46);
    border-radius: 8px;
    background: rgba(25,22,34,0.96);
    color: #ffb6c8;
    box-shadow: 0 8px 24px rgba(0,0,0,0.38);
    font-family: inherit;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
}
.vnccs-pipe-settings-open-icon {
    flex: 0 0 auto;
    font-size: 18px;
    line-height: 1;
    letter-spacing: 0;
}
.vnccs-pipe-settings-open:hover {
    border-color: rgba(255,143,163,0.82);
    background: rgba(255,143,163,0.14);
}
.vnccs-pipe-main {
    min-width: 0;
    display: grid;
    grid-template-rows: minmax(0, 1fr) 108px;
    overflow: hidden;
}
.vnccs-pipe-root.is-clone .vnccs-pipe-main {
    grid-template-rows: minmax(0, 1fr) 176px;
}
.vnccs-pipe-title {
    font-size: 10px;
    font-weight: 800;
    color: #ff8fa3;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 2px 0 10px;
}
.vnccs-pipe-block {
    border: 1px solid rgba(255,143,163,0.14);
    background: rgba(10,10,15,0.56);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}
.vnccs-pipe-block-h {
    padding: 7px 9px;
    background: rgba(26,26,38,0.95);
    color: #ffb6c8;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.vnccs-pipe-block-b {
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.vnccs-pipe-field {
    display: grid;
    grid-template-columns: 1fr;
    gap: 4px;
}
.vnccs-pipe-field-row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 7px;
}
.vnccs-pipe-label {
    color: #9898a8;
    font-size: 10px;
    font-weight: 700;
}
.vnccs-pipe-input, .vnccs-pipe-select, .vnccs-pipe-textarea {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 7px;
    background: rgba(255,255,255,0.045);
    color: #e8e8f0;
    font-family: inherit;
    font-size: 11px;
    padding: 6px 8px;
    color-scheme: dark;
}
.vnccs-pipe-slider-field {
    display: grid;
    grid-template-columns: 1fr;
    gap: 7px;
}
.vnccs-pipe-slider-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.vnccs-pipe-slider-value {
    color: #e8e8f0;
    font-size: 11px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}
.vnccs-pipe-slider {
    width: 100%;
    height: 18px;
    margin: 0;
    appearance: none;
    background: transparent;
    cursor: pointer;
}
.vnccs-pipe-slider::-webkit-slider-runnable-track {
    height: 8px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: linear-gradient(90deg, var(--zone-color) 0 var(--fill), rgba(255,255,255,0.08) var(--fill) 100%);
}
.vnccs-pipe-slider::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    margin-top: -6px;
    border-radius: 50%;
    border: 2px solid #f6f0f4;
    background: var(--zone-color);
    box-shadow: 0 0 14px var(--zone-glow);
}
.vnccs-pipe-slider::-moz-range-track {
    height: 8px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.08);
}
.vnccs-pipe-slider::-moz-range-progress {
    height: 8px;
    border-radius: 999px;
    background: var(--zone-color);
}
.vnccs-pipe-slider::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2px solid #f6f0f4;
    background: var(--zone-color);
    box-shadow: 0 0 14px var(--zone-glow);
}
.vnccs-pipe-slider-status {
    border: 1px solid var(--zone-border);
    border-radius: 7px;
    background: var(--zone-bg);
    color: var(--zone-color);
    padding: 6px 8px;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
}
.vnccs-pipe-textarea {
    min-height: 72px;
    resize: vertical;
}
.vnccs-pipe-check {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #cfcfda;
    font-size: 11px;
    cursor: pointer;
    user-select: none;
    padding: 2px 0;
}
.vnccs-pipe-mode-tabs {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 6px;
}
.vnccs-pipe-mode-tab {
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.045);
    color: #9898a8;
    border-radius: 7px;
    font-size: 10px;
    font-weight: 800;
    padding: 6px 8px;
    cursor: pointer;
}
.vnccs-pipe-mode-tab.is-selected {
    color: #ffb6c8;
    border-color: rgba(255,143,163,0.48);
    background: rgba(255,143,163,0.09);
}
.vnccs-pipe-preview {
    min-height: 0;
    padding: 12px;
    overflow: hidden;
    background: radial-gradient(circle, rgba(255,143,163,0.04) 1px, transparent 1px), #09090e;
    background-size: 20px 20px, 100% 100%;
    display: flex;
    flex-direction: column;
}
.vnccs-pipe-preview-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    color: #9898a8;
    font-size: 11px;
    flex-shrink: 0;
}
.vnccs-pipe-grid {
    flex: 1;
    min-height: 0;
    display: grid;
    gap: 8px;
    align-content: center;
    justify-content: center;
    overflow: hidden;
}
.vnccs-pipe-img {
    position: relative;
    width: 100%;
    height: 100%;
    display: block;
    min-height: 0;
    justify-self: center;
    align-self: center;
    appearance: none;
    padding: 0;
    background: #14141e;
    background-position: center;
    background-repeat: no-repeat;
    background-size: contain;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    box-sizing: border-box;
    cursor: zoom-in;
    opacity: 1;
    transition: opacity 0.12s ease;
}
.vnccs-pipe-img:hover {
    border-color: rgba(255,143,163,0.45);
}
.vnccs-pipe-img-regen {
    position: absolute;
    right: 7px;
    bottom: 7px;
    border: 1px solid rgba(255,143,163,0.48);
    background: rgba(14,14,22,0.78);
    color: #ffb6c8;
    border-radius: 7px;
    font-size: 10px;
    font-weight: 900;
    padding: 5px 7px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.12s ease, border-color 0.12s ease, background 0.12s ease;
}
.vnccs-pipe-img:hover .vnccs-pipe-img-regen,
.vnccs-pipe-img:focus-within .vnccs-pipe-img-regen {
    opacity: 1;
}
.vnccs-pipe-img-regen:hover {
    border-color: rgba(255,143,163,0.82);
    background: rgba(255,143,163,0.16);
}
.vnccs-pipe-empty {
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #5e5e70;
    font-size: 12px;
}
.vnccs-pipe-modal-backdrop {
    position: absolute;
    inset: 0;
    z-index: 30;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 18px;
    background: rgba(4,4,8,0.62);
    backdrop-filter: blur(3px);
}
.vnccs-pipe-modal {
    width: min(440px, 100%);
    border: 1px solid rgba(255,143,163,0.42);
    border-radius: 8px;
    background: rgba(24,24,34,0.98);
    box-shadow: 0 18px 48px rgba(0,0,0,0.45);
    overflow: hidden;
}
.vnccs-pipe-modal.is-settings {
    width: min(940px, 100%);
    max-height: calc(100% - 12px);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
}
.vnccs-pipe-modal.is-settings .vnccs-pipe-modal-body {
    overflow-y: auto;
    white-space: normal;
}
.vnccs-pipe-settings-intro {
    margin: 0 0 12px;
    color: #9898a8;
    line-height: 1.45;
}
.vnccs-pipe-settings-groups {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    align-items: start;
}
.vnccs-pipe-settings-group {
    min-width: 0;
    border: 1px solid rgba(255,143,163,0.16);
    border-radius: 8px;
    background: rgba(10,10,15,0.48);
    overflow: hidden;
}
.vnccs-pipe-settings-group-title {
    padding: 9px 10px;
    color: #ffb6c8;
    background: rgba(31,29,42,0.96);
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.vnccs-pipe-settings-group-fields {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    padding: 10px;
}
.vnccs-pipe-settings-field {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 4px;
}
.vnccs-pipe-settings-field.is-wide {
    grid-column: 1 / -1;
}
.vnccs-pipe-settings-field.is-check {
    grid-column: 1 / -1;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    min-height: 28px;
}
.vnccs-pipe-settings-field-label {
    color: #a6a6b6;
    font-size: 10px;
    font-weight: 700;
}
.vnccs-pipe-settings-field input:not([type="checkbox"]),
.vnccs-pipe-settings-field select,
.vnccs-pipe-settings-field textarea {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 7px;
    background: rgba(255,255,255,0.045);
    color: #e8e8f0;
    color-scheme: dark;
    font-family: inherit;
    font-size: 11px;
    padding: 7px 8px;
}
.vnccs-pipe-settings-field textarea {
    min-height: 82px;
    resize: vertical;
}
.vnccs-pipe-settings-note {
    grid-column: 1 / -1;
    color: #77778a;
    font-size: 10px;
    line-height: 1.4;
}
.vnccs-pipe-modal-actions.is-settings {
    gap: 8px;
    padding-top: 12px;
    background: rgba(24,24,34,0.98);
}
.vnccs-pipe-modal-btn.is-secondary {
    border-color: rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.055);
    color: #cfcfda;
}
.vnccs-pipe-modal-btn.is-reset {
    margin-right: auto;
}
@media (max-width: 760px) {
    .vnccs-pipe-settings-groups {
        grid-template-columns: 1fr;
    }
}
.vnccs-pipe-modal-title {
    padding: 14px 16px;
    background: #1b1b29;
    color: #ffb6c8;
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.vnccs-pipe-modal-body {
    padding: 16px;
    color: #d8d8e4;
    font-size: 12px;
    line-height: 1.45;
    white-space: pre-wrap;
}
.vnccs-pipe-modal-actions {
    display: flex;
    justify-content: flex-end;
    padding: 0 16px 16px;
}
.vnccs-pipe-modal-btn {
    border: 1px solid rgba(255,143,163,0.5);
    border-radius: 7px;
    background: rgba(255,143,163,0.14);
    color: #ffd1dc;
    padding: 7px 14px;
    font-size: 11px;
    font-weight: 900;
    cursor: pointer;
}
.vnccs-pipe-modal-btn:hover {
    border-color: rgba(255,143,163,0.82);
    background: rgba(255,143,163,0.2);
}
.vnccs-pipe-chain {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    padding: 10px 12px 12px;
    background: #111119;
    border-top: 1px solid rgba(255,143,163,0.14);
}
.vnccs-pipe-chain.is-clone {
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
.vnccs-pipe-chain.is-clothes {
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
.vnccs-pipe-chain.is-emotions {
    grid-template-columns: repeat(var(--vnccs-stage-count, 1), minmax(0, 1fr));
    grid-template-rows: minmax(0, 1fr);
    gap: calc(10px * var(--vnccs-stage-scale, 1));
    overflow: hidden;
}
.vnccs-pipe-chain.is-emotions .vnccs-pipe-stage {
    height: 100%;
    min-height: 0;
    overflow: hidden;
    padding: calc(10px * var(--vnccs-stage-scale, 1));
    gap: calc(5px * var(--vnccs-stage-scale, 1));
}
.vnccs-pipe-chain.is-emotions .vnccs-pipe-stage-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: calc(12px * var(--vnccs-stage-scale, 1));
}
.vnccs-pipe-chain.is-emotions .vnccs-pipe-stage-status {
    display: -webkit-box;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    font-size: calc(10px * var(--vnccs-stage-scale, 1));
    line-height: 1.2;
}
.vnccs-pipe-chain.is-emotions .vnccs-pipe-regen {
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: calc(4px * var(--vnccs-stage-scale, 1)) calc(7px * var(--vnccs-stage-scale, 1));
    font-size: calc(10px * var(--vnccs-stage-scale, 1));
}
.vnccs-pipe-stage {
    position: relative;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 0;
}
.vnccs-pipe-stage.is-active {
    border-color: rgba(255,143,163,0.85);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.24) inset, 0 0 18px rgba(255,143,163,0.16);
}
.vnccs-pipe-stage.is-regenerating {
    border-color: rgba(255,191,116,0.72);
    box-shadow: 0 0 0 1px rgba(255,191,116,0.18) inset, 0 0 18px rgba(255,191,116,0.12);
}
.vnccs-pipe-stage.is-done {
    border-color: rgba(0,214,143,0.45);
}
.vnccs-pipe-stage-progress {
    height: 4px;
    overflow: hidden;
    border-radius: 99px;
    background: rgba(255,255,255,0.08);
}
.vnccs-pipe-stage-progress-fill {
    height: 100%;
    width: 0%;
    border-radius: inherit;
    background: #ffc074;
    transition: width 0.45s ease;
}
.vnccs-pipe-regen-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: #ffc074;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
}
.vnccs-pipe-regen-spinner {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid rgba(255,192,116,0.25);
    border-top-color: #ffc074;
    animation: vnccs-pipe-spin 0.75s linear infinite;
}
.vnccs-pipe-stage-name {
    font-size: 12px;
    font-weight: 800;
    color: #e8e8f0;
}
.vnccs-pipe-stage-status {
    font-size: 10px;
    color: #9898a8;
    line-height: 1.35;
}
.vnccs-pipe-stage-lora {
    font-size: 10px;
    color: #9898a8;
    line-height: 1.35;
    word-break: break-word;
}
.vnccs-pipe-stage-actions {
    margin-top: auto;
    display: flex;
    justify-content: flex-end;
}
.vnccs-pipe-regen {
    border: 1px solid rgba(255,143,163,0.36);
    background: rgba(255,143,163,0.08);
    color: #ffb6c8;
    border-radius: 7px;
    font-size: 10px;
    font-weight: 800;
    padding: 4px 7px;
    cursor: pointer;
}
.vnccs-pipe-regen:hover {
    border-color: rgba(255,143,163,0.72);
    background: rgba(255,143,163,0.14);
}
.vnccs-pipe-regen:disabled {
    cursor: wait;
    opacity: 0.45;
}
@keyframes vnccs-pipe-spin {
    to { transform: rotate(360deg); }
}
.vnccs-pipe-tabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.vnccs-pipe-root.is-emotions .vnccs-pipe-preview-head {
    min-width: 0;
    gap: 8px;
}
.vnccs-pipe-root.is-emotions .vnccs-pipe-preview-label {
    flex: 0 1 24%;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-pipe-root.is-emotions .vnccs-pipe-tabs {
    flex: 1 1 auto;
    min-width: 0;
    flex-wrap: nowrap;
    overflow: hidden;
}
.vnccs-pipe-root.is-emotions .vnccs-pipe-tab {
    flex: 1 1 0;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding-inline: calc(8px * var(--vnccs-stage-scale, 1));
    font-size: calc(10px * var(--vnccs-stage-scale, 1));
}
.vnccs-pipe-tab {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: #9898a8;
    border-radius: 7px;
    font-size: 10px;
    padding: 4px 8px;
    cursor: pointer;
}
.vnccs-pipe-tab.is-selected {
    color: #ffb6c8;
    border-color: rgba(255,143,163,0.45);
}
.vnccs-pipe-viewer {
    position: absolute;
    inset: 0;
    z-index: 20;
    background: #07070b;
    display: grid;
    grid-template-rows: 42px minmax(0, 1fr);
}
.vnccs-pipe-viewer-bar {
    display: flex;
    min-width: 0;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    background: #101018;
    border-bottom: 1px solid rgba(255,143,163,0.16);
}
.vnccs-pipe-viewer-stages {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1 1 0;
    min-width: 0;
    overflow-x: auto;
}
.vnccs-pipe-viewer-btn {
    flex: 0 0 auto;
    white-space: nowrap;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.055);
    color: #e8e8f0;
    border-radius: 7px;
    font-size: 10px;
    font-weight: 800;
    padding: 5px 9px;
    cursor: pointer;
}
.vnccs-pipe-viewer-btn:focus,
.vnccs-pipe-viewer-btn:focus-visible,
.vnccs-pipe-viewer-btn:active {
    outline: none;
    background: rgba(255,143,163,0.12);
    border-color: rgba(255,143,163,0.45);
    color: #ffb6c8;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.22);
}
.vnccs-pipe-viewer-btn.is-selected {
    color: #ffb6c8;
    border-color: rgba(255,143,163,0.48);
}
.vnccs-pipe-viewer-canvas {
    position: relative;
    overflow: hidden;
    cursor: grab;
    min-height: 0;
    min-width: 0;
    touch-action: none;
}
.vnccs-pipe-viewer-canvas.is-dragging {
    cursor: grabbing;
}
.vnccs-pipe-viewer-img {
    position: absolute;
    left: 0;
    top: 0;
    display: block;
    transform-origin: 0 0;
    user-select: none;
    -webkit-user-drag: none;
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transform: translate(-100000px, -100000px) scale(1);
}
.vnccs-pipe-viewer-img.is-ready {
    opacity: 1;
    visibility: visible;
}
`;

function injectStyles() {
    if (document.getElementById("vnccs-character-generator-style")) return;
    const style = document.createElement("style");
    style.id = "vnccs-character-generator-style";
    style.textContent = CSS;
    document.head.appendChild(style);
}

function deepMerge(base, patch) {
    const out = JSON.parse(JSON.stringify(base));
    for (const [section, values] of Object.entries(patch || {})) {
        if (values && typeof values === "object" && !Array.isArray(values)) {
            out[section] = { ...(out[section] || {}), ...values };
        } else {
            out[section] = values;
        }
    }
    return out;
}

function normalizeUpscalerSettings(data) {
    const upscaler = data.upscaler;
    if (!upscaler || typeof upscaler !== "object") return;
    if (String(upscaler.mode || "").trim().toLowerCase() === "gan") upscaler.mode = "off";
    delete upscaler.gan_model;
}

function readData(node) {
    const widget = node.widgets?.find(w => w.name === "widget_data");
    try {
        const parsed = JSON.parse(widget?.value || "{}");
        const data = deepMerge(DEFAULT_DATA, parsed);
        if (
            parsed?.emotion_generation
            && parsed.emotion_generation.use_sam === undefined
            && parsed.emotion_generation.use_sam_model !== undefined
        ) {
            data.emotion_generation.use_sam = Boolean(parsed.emotion_generation.use_sam_model);
        }
        if (String(data.upscaler?.model || "").startsWith("seedvr2_ema_") || String(data.upscaler?.model || "").endsWith(".gguf")) {
            data.upscaler.model = "seedvr2_3b_fp8_e4m3fn.safetensors";
        }
        if (!SEEDVR_COLOR_CORRECTION_MODES.includes(data.upscaler?.color_correction)) {
            data.upscaler.color_correction = "lab";
        }
        for (const section of ["common", "pose_generation", "remove_clothes"]) {
            data[section].target_size = resolutionScaleValue(resolutionScaleMegapixels(data[section].target_size));
        }
        normalizeUpscalerSettings(data);
        return data;
    } catch {
        return JSON.parse(JSON.stringify(DEFAULT_DATA));
    }
}

function writeData(node, data, { notify = true } = {}) {
    const widget = node.widgets?.find(w => w.name === "widget_data");
    if (!widget) return;
    normalizeUpscalerSettings(data);
    widget.value = JSON.stringify(data);
    if (notify) widget.callback?.(widget.value);
    app.graph?.setDirtyCanvas(true, true);
}

function uniqueOptions(values) {
    return [...new Set(values.filter(Boolean))];
}

function booleanValue(value, fallback = false) {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase();
        if (["true", "1", "yes", "on"].includes(normalized)) return true;
        if (["false", "0", "no", "off"].includes(normalized)) return false;
    }
    return fallback;
}

class CharacterGeneratorWidget {
    constructor(node, options = {}) {
        this.node = node;
        this.isClone = Boolean(options.isClone);
        this.isClothes = Boolean(options.isClothes);
        this.isEmotions = Boolean(options.isEmotions);
        this.qi2EmotionDefaultsPending = this.isEmotions;
        this.title = options.title || "VNCCS Character Generator";
        this.data = readData(node);
        this.seedvrAttention = { current: null, available: SEEDVR_ATTENTION_MODES };
        this.seedvrAssets = null;
        this.seedvrDownloads = {};
        this.seedvrPollTimer = null;
        this.nativeSeedvrAvailable = null;
        this.nativeSeedvrMissing = [];
        this.seedvrUpdateModalShown = false;
        this.seedvrModelPickerOpen = false;
        this.syncCharacterSourceData();
        this.stages = this.currentStages();
        this.stageState = Object.fromEntries(this.stages.map(([key]) => [key, { status: "waiting", images: null, message: "" }]));
        const defaultPreview = this.defaultPreviewStage();
        this.selectedPreview = this.data.ui?.selected_preview || defaultPreview;
        const selectedPreviewWasInvalid = !this.stages.some(([key]) => key === this.selectedPreview);
        if (selectedPreviewWasInvalid) {
            this.selectedPreview = defaultPreview;
        }
        this.userSelectedPreview = selectedPreviewWasInvalid
            ? false
            : Boolean(this.data.ui?.user_selected_preview);
        this.viewer = null;
        this.viewerFocus = null;
        this.restoredViewer = null;
        this._saveBrowserStateTimer = null;
        this.nodeDefs = {};
        this.imageMetrics = new Map();
        this.fieldDrafts = new Map();
        this.previewLayoutFrame = null;
        this.regenerateState = null;
        this.regenerateTimer = null;
        this.restoreBrowserState();
        this.build();
        this.bindEvents();
        this.loadNodeDefs();
        this.loadSeedvrAssets();
    }

    build() {
        injectStyles();
        const root = document.createElement("div");
        root.className = "vnccs-pipe-root";
        this.root = root;
        enableMiddleMouseCanvasPan(root, this.node);
        attachHelpTooltips(root);
        this.updateModeClasses();

        this.settingsEl = document.createElement("div");
        this.settingsEl.className = "vnccs-pipe-settings";
        this.previewEl = document.createElement("div");
        this.previewEl.className = "vnccs-pipe-preview";
        this.chainEl = document.createElement("div");
        this.chainEl.className = "vnccs-pipe-chain";

        const main = document.createElement("div");
        main.className = "vnccs-pipe-main";
        main.append(this.previewEl, this.chainEl);
        this.settingsButton = document.createElement("button");
        this.settingsButton.type = "button";
        this.settingsButton.className = "vnccs-pipe-settings-open";
        const settingsIcon = document.createElement("span");
        settingsIcon.className = "vnccs-pipe-settings-open-icon";
        settingsIcon.setAttribute("aria-hidden", "true");
        settingsIcon.textContent = "⚙";
        const settingsLabel = document.createElement("span");
        settingsLabel.textContent = "Generator Settings";
        this.settingsButton.append(settingsIcon, settingsLabel);
        this.settingsButton.onclick = () => this.openGeneratorSettings();
        this.protectNativeControl(this.settingsButton);
        root.append(this.settingsEl, main, this.settingsButton);

        this.node.addDOMWidget("character_generator_ui", "ui", root, {
            serialize: false,
            hideOnZoom: false,
        });
        syncDOMWidgetWidthSoon(this.node, "character_generator_ui");

        const dataWidget = this.node.widgets?.find(w => w.name === "widget_data");
        if (dataWidget) {
            dataWidget.hidden = true;
            dataWidget.computeSize = () => [0, -4];
        }

        this.syncCharacterSourceData();
        this.syncStagesFromData();
        this.syncModelResolution();
        writeData(this.node, this.data);
        this.node._vnccsCharacterGeneratorSyncBeforeQueue = () => {
            this.syncCharacterSourceData();
            this.syncStagesFromData();
            this.syncModelResolution();
            writeData(this.node, this.data);
            return this.validateNativeSeedvr(true);
        };
        registerCleanup(this.node, () => delete this.node._vnccsCharacterGeneratorSyncBeforeQueue);

        this.renderSettings();
        this.renderPreview();
        this.renderChain();
        this.previewResizeObserver = new ResizeObserver(() => this.renderPreview());
        this.previewResizeObserver.observe(this.previewEl);
        registerCleanup(this.node, () => this.previewResizeObserver?.disconnect());
        registerCleanup(this.node, () => clearInterval(this.regenerateTimer));
        registerCleanup(this.node, () => clearInterval(this.seedvrPollTimer));
        registerCleanup(this.node, () => {
            this.closeViewer();
            this.viewer = null;
            clearTimeout(this._saveBrowserStateTimer);
            cancelAnimationFrame(this.previewLayoutFrame);
        });
        this.bindModelResolutionSync();
        if (this.isClone) {
            this.sourceSyncTimer = setInterval(() => {
                const previous = this.data.nsfw_enabled;
                const changed = this.syncCharacterSourceData();
                if (!changed && previous === this.data.nsfw_enabled) return;
                this.syncStagesFromData();
                writeData(this.node, this.data);
                this.renderSettings();
                this.renderPreview();
                this.renderChain();
            }, 500);
            registerCleanup(this.node, () => clearInterval(this.sourceSyncTimer));
        }
        if (this.restoredViewer?.open && this.currentImages().length) {
            this.openViewer(this.restoredViewer.index || 0, this.restoredViewer);
        }
    }

    bindEvents() {
        this.onStage = (event) => {
            const detail = event.detail || {};
            if (String(detail.node_id) !== String(this.node.id)) return;
            const stage = detail.stage;
            if (!this.stageState[stage] && stage !== "error") return;
            if (stage === "error") {
                for (const key of Object.keys(this.stageState)) {
                    if (this.stageState[key].status === "running") this.stageState[key].status = "error";
                }
                this.finishRegenerate();
            } else {
                const status = detail.status || "waiting";
                const previousStageState = this.stageState[stage] || {};
                const hasImages = Object.prototype.hasOwnProperty.call(detail, "images");
                const targetedRegenerate = Number.isInteger(this.regenerateState?.imageIndex)
                    && this.regenerateState?.targetStages?.includes(stage);
                if (status === "running") {
                    const continuingBatch = targetedRegenerate || (
                        previousStageState.status === "running"
                        && (Boolean(detail.append_images) || !hasImages)
                    );
                    if (!continuingBatch) this.resetStagesFrom(stage);
                    if (!continuingBatch && (stage === "pose_generation" || stage === "original_pose_generation" || stage === "source_upscaler")) {
                        this.userSelectedPreview = false;
                        if (!this.data.ui) this.data.ui = {};
                        this.data.ui.user_selected_preview = false;
                    }
                }
                let nextImages = previousStageState.images || null;
                if (hasImages && detail.replace_images) {
                    nextImages = [...(previousStageState.images || [])];
                    const previewStart = Math.max(0, Number.parseInt(detail.preview_start, 10) || 0);
                    for (const [offset, image] of (detail.images || []).entries()) {
                        nextImages[previewStart + offset] = image;
                    }
                } else if (hasImages && detail.append_images) {
                    nextImages = [...(previousStageState.images || []), ...(detail.images || [])];
                } else if (hasImages) {
                    nextImages = detail.images;
                }
                this.stageState[stage] = {
                    status,
                    images: nextImages,
                    message: detail.message || "",
                    current: detail.current,
                    total: detail.total,
                };
                if (!this.userSelectedPreview && (status === "running" || status === "done" || detail.images)) {
                    this.selectedPreview = stage;
                    this.persistUI();
                }
                this.updateRegenerateProgress(stage, status);
            }
            if (this.viewer?.open) this.syncViewerImage();
            this.renderPreview();
            this.renderChain();
            this.saveBrowserState();
        };
        api.addEventListener("vnccs.character_generator.stage", this.onStage);
        registerCleanup(this.node, () => api.removeEventListener("vnccs.character_generator.stage", this.onStage));

        if (this.isClone) {
            this.onClonerUpdated = () => {
                const previous = this.data.nsfw_enabled;
                const changed = this.syncCharacterSourceData();
                if (!changed && previous === this.data.nsfw_enabled) return;
                this.syncStagesFromData();
                writeData(this.node, this.data);
                this.renderSettings();
                this.renderPreview();
                this.renderChain();
            };
            window.addEventListener("vnccs-character-cloner-updated", this.onClonerUpdated);
            registerCleanup(this.node, () => window.removeEventListener("vnccs-character-cloner-updated", this.onClonerUpdated));
        }
        if (this.isEmotions) {
            this.onEmotionStudioModeChanged = () => {
                if (this.syncModelResolution()) writeData(this.node, this.data);
                this.renderSettings();
            };
            window.addEventListener("vnccs-emotion-studio-generation-mode-changed", this.onEmotionStudioModeChanged);
            registerCleanup(this.node, () => window.removeEventListener("vnccs-emotion-studio-generation-mode-changed", this.onEmotionStudioModeChanged));
        }
    }

    resetStagesFrom(stageKey, { preserveImages = false } = {}) {
        const start = this.stages.findIndex(([key]) => key === stageKey);
        if (start < 0) return;
        for (const [key] of this.stages.slice(start)) {
            this.stageState[key] = {
                status: "waiting",
                images: preserveImages ? (this.stageState[key]?.images || null) : null,
                message: "",
                current: undefined,
                total: undefined,
            };
        }
    }

    startRegenerate(stageKey, imageIndex = null) {
        const startIndex = this.stages.findIndex(([key]) => key === stageKey);
        const targetStages = this.stages.slice(Math.max(0, startIndex)).map(([key]) => key);
        this.regenerateState = {
            from: stageKey,
            imageIndex,
            activeStage: stageKey,
            targetStages,
            startedAt: Date.now(),
            elapsed: 0,
            sawStageEvent: false,
        };
        clearInterval(this.regenerateTimer);
        this.regenerateTimer = setInterval(() => {
            if (!this.regenerateState) return;
            this.regenerateState.elapsed = Math.floor((Date.now() - this.regenerateState.startedAt) / 1000);
            this.renderPreview();
            this.renderChain();
        }, 500);
    }

    finishRegenerate() {
        clearInterval(this.regenerateTimer);
        this.regenerateTimer = null;
        this.regenerateState = null;
        this.renderPreview();
        this.renderChain();
    }

    updateRegenerateProgress(stage, status) {
        if (!this.regenerateState) return;
        this.regenerateState.sawStageEvent = true;
        if (this.regenerateState.targetStages.includes(stage) && status === "running") {
            this.regenerateState.activeStage = stage;
        }
        const lastStage = this.regenerateState.targetStages[this.regenerateState.targetStages.length - 1];
        if (stage === lastStage && status === "done") {
            this.finishRegenerate();
        }
    }

    persistUI() {
        this.syncCharacterSourceData();
        this.syncStagesFromData();
        this.data.ui = {
            ...(this.data.ui || {}),
            selected_preview: this.selectedPreview,
            user_selected_preview: this.userSelectedPreview,
        };
        writeData(this.node, this.data);
        this.saveBrowserState();
    }

    set(section, key, value) {
        this.syncCharacterSourceData();
        this.syncModelResolution();
        if (!this.data[section] || typeof this.data[section] !== "object") this.data[section] = {};
        const rerenderBgRemove = section === "bg_remove"
            && key === "preset"
            && this.data.bg_remove.preset !== value;
        this.data[section][key] = value;
        if (section === "bg_remove" && key === "preset") {
            if (String(value).trim().toLowerCase() === "native" && !this.bgRemoveModes().includes("Native")) {
                this.data.bg_remove.preset = this.previousBgRemovePreset();
            }
            this.data.bg_remove.use_preset_values = true;
        }
        if (this.isClone && section === "common" && key === "target_size") {
            this.data.pose_generation.target_size = value;
            this.data.remove_clothes.target_size = value;
        }
        this.rememberModelResolution(key === "target_size" && section === (this.isClone ? "common" : "pose_generation"));
        writeData(this.node, this.data, { notify: false });
        this.saveBrowserState();
        if (rerenderBgRemove) this.renderSettings();
    }

    isNativeBgRemove() {
        return String(this.data.bg_remove?.preset || "").trim().toLowerCase() === "native";
    }

    previousBgRemovePreset() {
        const previous = this.data.ui?.bg_remove_previous_preset;
        return BG_REMOVE_MODES.includes(previous) && previous !== "Native" ? previous : "balanced";
    }

    bgRemoveModes() {
        return this.data.ui?.bg_remove_model_kind === "qi2"
            ? BG_REMOVE_MODES
            : BG_REMOVE_MODES.filter(mode => mode !== "Native");
    }

    generatorSettingsGroups() {
        const isQI2 = this.data.ui?.resolution_model_kind === "qi2";
        const isNativeBgRemove = this.isNativeBgRemove();
        const number = (section, key, label, min, max, step = 1, extra = {}) => ({
            section, key, label, type: "number", min, max, step, ...extra,
        });
        const text = (section, key, label, extra = {}) => ({ section, key, label, type: "text", ...extra });
        const check = (section, key, label, extra = {}) => ({ section, key, label, type: "checkbox", wide: true, ...extra });
        const select = (section, key, label, options, extra = {}) => ({
            section, key, label, type: "select", options, ...extra,
        });
        const resolutionScale = (section, key = "target_size") => ({
            section, key, label: "resolution scale", type: "resolution_scale", wide: true,
        });
        const textarea = (section, key, label, extra = {}) => ({
            section, key, label, type: "textarea", wide: true, ...extra,
        });
        const groups = [];

        if (!this.isEmotions) {
            const poseTargetSection = this.isClone ? "common" : "pose_generation";
            groups.push({
                title: isQI2 ? "Text Encode Qwen Image 2.1 · Pose Generation" : "Encoder · Pose Generation",
                fields: isQI2 ? [
                    resolutionScale(poseTargetSection),
                    select("pose_generation", "background_color", "background_color", ["from_generator", "White", "Green", "Blue"]),
                ] : [
                    resolutionScale(poseTargetSection),
                    select("pose_generation", "upscale_method", "upscale_method", ["lanczos", "bicubic", "area"]),
                    select("pose_generation", "crop_method", "crop_method", ["disabled", "pad", "center"]),
                    number("pose_generation", "latent_image_index", "latent_image_index", 1, 3, 1),
                    number("pose_generation", "vl_size", "vl_size", 256, 1024, 8),
                    select("pose_generation", "background_color", "background_color", ["from_generator", "White", "Green", "Blue"]),
                    number("pose_generation", "weight1", "weight1", 0, 2, 0.01),
                    number("pose_generation", "weight2", "weight2", 0, 2, 0.01),
                    number("pose_generation", "weight3", "weight3", 0, 2, 0.01),
                    text("pose_generation", "image1_name", "image1_name"),
                    text("pose_generation", "image2_name", "image2_name"),
                    text("pose_generation", "image3_name", "image3_name"),
                    textarea("pose_generation", "instruction", "instruction"),
                ],
            });
            groups.push({
                title: "KSampler · Pose Generation",
                fields: [
                    check("pose_sampler", "inherit_pipe", "Use sampler settings from connected pipe"),
                    number("pose_sampler", "seed", "seed", 0, Number.MAX_SAFE_INTEGER, 1),
                    number("pose_sampler", "steps", "steps", 1, 10000, 1),
                    number("pose_sampler", "cfg", "cfg", 0, 100, 0.01),
                    select("pose_sampler", "sampler_name", "sampler_name", [], { nodeName: "KSampler", inputName: "sampler_name" }),
                    select("pose_sampler", "scheduler", "scheduler", [], { nodeName: "KSampler", inputName: "scheduler" }),
                    number("pose_sampler", "denoise", "denoise", 0, 1, 0.01),
                ],
                note: "Seed, steps, CFG, sampler and scheduler are used only when pipe inheritance is disabled. Denoise is always local.",
            });
            groups.push({
                title: "VAEDecodeTiled",
                fields: [
                    number("vae_decode", "tile_size", "tile_size", 64, 4096, 8),
                    number("vae_decode", "overlap", "overlap", 0, 4096, 8),
                    number("vae_decode", "temporal_size", "temporal_size", 1, 4096, 1),
                    number("vae_decode", "temporal_overlap", "temporal_overlap", 0, 4096, 1),
                ],
            });
            if (this.isClone) {
                groups.push({
                    title: isQI2 ? "Text Encode Qwen Image 2.1 · Remove Clothes" : "Encoder · Remove Clothes",
                    fields: isQI2 ? [
                        textarea("remove_clothes", "prompt", "prompt"),
                        select("remove_clothes", "background_color", "background_color", ["White", "Green", "Blue"]),
                    ] : [
                        textarea("remove_clothes", "prompt", "prompt"),
                        select("remove_clothes", "upscale_method", "upscale_method", ["lanczos", "bicubic", "area"]),
                        select("remove_clothes", "crop_method", "crop_method", ["disabled", "pad", "center"]),
                        number("remove_clothes", "latent_image_index", "latent_image_index", 1, 3, 1),
                        number("remove_clothes", "vl_size", "vl_size", 256, 1024, 8),
                        select("remove_clothes", "background_color", "background_color", ["White", "Green", "Blue"]),
                        number("remove_clothes", "weight1", "weight1", 0, 2, 0.01),
                        number("remove_clothes", "weight2", "weight2", 0, 2, 0.01),
                        number("remove_clothes", "weight3", "weight3", 0, 2, 0.01),
                        text("remove_clothes", "image1_name", "image1_name"),
                        text("remove_clothes", "image2_name", "image2_name"),
                        text("remove_clothes", "image3_name", "image3_name"),
                        textarea("remove_clothes", "instruction", "instruction"),
                    ],
                    note: "target_size is shared with the clone pose-generation encoder.",
                });
                groups.push({
                    title: "KSampler · Remove Clothes",
                    fields: [
                        check("remove_clothes_sampler", "inherit_pipe", "Use sampler settings from connected pipe"),
                        number("remove_clothes_sampler", "seed", "seed", 0, Number.MAX_SAFE_INTEGER, 1),
                        number("remove_clothes_sampler", "steps", "steps", 1, 10000, 1),
                        number("remove_clothes_sampler", "cfg", "cfg", 0, 100, 0.01),
                        select("remove_clothes_sampler", "sampler_name", "sampler_name", [], { nodeName: "KSampler", inputName: "sampler_name" }),
                        select("remove_clothes_sampler", "scheduler", "scheduler", [], { nodeName: "KSampler", inputName: "scheduler" }),
                        number("remove_clothes_sampler", "denoise", "denoise", 0, 1, 0.01),
                    ],
                    note: "Seed, steps, CFG, sampler and scheduler are used only when pipe inheritance is disabled. Denoise is always local.",
                });
            }

            groups.push({
                title: "Generator Upscaler",
                fields: [
                    select("upscaler", "mode", "mode", ["seedvr", "off"]),
                    check("upscaler", "inherit_pipe_seed", "Use seed from connected pipe"),
                    number("upscaler", "seed", "seed", 0, Number.MAX_SAFE_INTEGER, 1),
                ],
            });
            groups.push({
                title: "Native SeedVR2",
                fields: [
                    select("upscaler", "model", "diffusion model", WORKFLOW_UPSCALER_DIT_MODELS, { nodeName: "UNETLoader", inputName: "unet_name" }),
                    select("upscaler", "vae", "VAE", WORKFLOW_UPSCALER_VAE_MODELS, { nodeName: "VAELoader", inputName: "vae_name" }),
                    number("upscaler", "resolution", "target short edge", 16, 16384, 2),
                    number("upscaler", "max_resolution", "maximum edge (0 = unlimited)", 0, 16384, 2),
                    select("upscaler", "color_correction", "color correction", SEEDVR_COLOR_CORRECTION_MODES, { nodeName: "SeedVR2PostProcessing", inputName: "color_correction_method" }),
                ],
            });
        } else {
            groups.push({
                title: "UltralyticsDetectorProvider",
                fields: isQI2 ? [
                    select("emotion_generation", "bbox_model", "bbox detector model", [], { nodeName: "UltralyticsDetectorProvider", inputName: "model_name", wide: true }),
                ] : [
                    select("emotion_generation", "bbox_model", "bbox detector model", [], { nodeName: "UltralyticsDetectorProvider", inputName: "model_name", wide: true }),
                    select("emotion_generation", "segm_model", "segmentation detector model", [], { nodeName: "UltralyticsDetectorProvider", inputName: "model_name", wide: true }),
                ],
            });
            if (isQI2) {
                groups.push({
                    title: "VNCCS BBox Extractor · QI2 Face Generation",
                    fields: [
                        resolutionScale("emotion_generation", "target_size"),
                        number("emotion_generation", "bbox_threshold", "threshold", 0, 1, 0.01),
                        number("emotion_generation", "bbox_dilation", "dilation", 0, 1024, 1),
                        number("emotion_generation", "feather", "feather", 0, 1024, 1),
                        number("emotion_generation", "drop_size", "drop_size", 1, 4096, 1),
                    ],
                    note: "The crop is encoded by Text Encode Qwen Image 2.1, generated at the selected megapixel scale, resized to the original crop bounds, and pasted back at the same coordinates. Feather controls the blend at the paste boundary.",
                });
                groups.push({
                    title: "Text Encode Qwen Image 2.1 · Emotion Prompt",
                    fields: [
                        textarea("emotion_generation", "qi2_prompt_template", "prompt template"),
                    ],
                    note: "Use {emotion} where the selected card's natural prompt and description tags should be inserted.",
                });
            } else {
                groups.push({
                    title: "SAMLoader",
                    fields: [
                        check("emotion_generation", "use_sam", "Connect SAM and segmentation detector to FaceDetailer"),
                        select("emotion_generation", "sam_model", "model_name", [], { nodeName: "SAMLoader", inputName: "model_name", wide: true }),
                        select("emotion_generation", "sam_device_mode", "device_mode", ["AUTO", "Prefer GPU", "CPU"], { nodeName: "SAMLoader", inputName: "device_mode" }),
                    ],
                });
                groups.push({
                    title: "FaceDetailer",
                    fields: [
                        number("emotion_generation", "guide_size", "guide_size", 64, 16384, 8),
                        check("emotion_generation", "guide_size_for", "guide_size_for"),
                        number("emotion_generation", "max_size", "max_size", 64, 16384, 8),
                        check("emotion_generation", "inherit_pipe_sampler", "Use sampler and scheduler from connected pipe"),
                        select("emotion_generation", "sampler_name", "sampler_name", [], { nodeName: "FaceDetailer", inputName: "sampler_name" }),
                        select("emotion_generation", "scheduler", "scheduler", [], { nodeName: "FaceDetailer", inputName: "scheduler" }),
                        number("emotion_generation", "feather", "feather", 0, 1024, 1),
                        check("emotion_generation", "noise_mask", "noise_mask"),
                        check("emotion_generation", "force_inpaint", "force_inpaint"),
                        number("emotion_generation", "bbox_threshold", "bbox_threshold", 0, 1, 0.01),
                        number("emotion_generation", "bbox_dilation", "bbox_dilation", 0, 1024, 1),
                        number("emotion_generation", "bbox_crop_factor", "bbox_crop_factor", 1, 100, 0.01),
                        select("emotion_generation", "sam_detection_hint", "sam_detection_hint", ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"], { nodeName: "FaceDetailer", inputName: "sam_detection_hint" }),
                        number("emotion_generation", "sam_dilation", "sam_dilation", 0, 1024, 1),
                        number("emotion_generation", "sam_threshold", "sam_threshold", 0, 1, 0.01),
                        number("emotion_generation", "sam_bbox_expansion", "sam_bbox_expansion", 0, 1024, 1),
                        number("emotion_generation", "sam_mask_hint_threshold", "sam_mask_hint_threshold", 0, 1, 0.01),
                        select("emotion_generation", "sam_mask_hint_use_negative", "sam_mask_hint_use_negative", ["False", "True"], { nodeName: "FaceDetailer", inputName: "sam_mask_hint_use_negative" }),
                        number("emotion_generation", "drop_size", "drop_size", 0, 4096, 1),
                        number("emotion_generation", "cycle", "cycle", 1, 100, 1),
                        check("emotion_generation", "inpaint_model", "inpaint_model"),
                        number("emotion_generation", "noise_mask_feather", "noise_mask_feather", 0, 1024, 1),
                        check("emotion_generation", "tiled_encode", "tiled_encode"),
                        check("emotion_generation", "tiled_decode", "tiled_decode"),
                    ],
                    note: "Steps and CFG come from the connected pipe. Face Detailer denoise is controlled in the main panel. Sampler and scheduler can optionally be overridden here. Seed remains per emotion item.",
                });
            }
            groups.push({
                title: isQI2 ? "VNCCS Emotion Crop Merge" : "VNCCS Emotion Matte Merge",
                fields: [
                    number("emotion_generation", "matte_expand_radius", "matte_expand_radius", 0, 256, 1),
                    number("emotion_generation", "matte_feather_radius", "matte_feather_radius", 0, 256, 1),
                    number("emotion_generation", "chroma_context", "chroma_context", 0, 1024, 1),
                ],
                note: isQI2
                    ? "The generated QI2 face crop is returned to its original coordinates before background processing."
                    : "These parameters affect only the FaceDetailer region. The original sprite alpha remains untouched elsewhere.",
            });
        }

        groups.push({
            title: "VNCCS Chroma Key",
            fields: [
                select("bg_remove", "preset", "preset", this.bgRemoveModes()),
                check("bg_remove", "use_preset_values", "Use values from selected preset"),
                number("bg_remove", "tolerance", "tolerance", 0, 1, 0.01),
                number("bg_remove", "softness", "softness", 0.001, 1, 0.01),
                number("bg_remove", "despill_strength", "despill_strength", 0, 1, 0.01),
                number("bg_remove", "edge_width", "edge_width", 0, 32, 1),
                number("bg_remove", "matte_cleanup", "matte_cleanup", 0, 1, 0.01),
                number("bg_remove", "foreground_recover", "foreground_recover", 0, 1, 0.01),
                number("bg_remove", "edge_decontaminate", "edge_decontaminate", 0, 1, 0.01),
                number("bg_remove", "edge_choke", "edge_choke", 0, 1, 0.01),
                select("bg_remove", "matte_method", "matte_method", ["chroma_soft", "guided_edge", "pymatting_if_available", "screen_matte"]),
                select("bg_remove", "screen_mode", "screen_mode", ["from_background", "auto", "green", "blue", "red"]),
                select("bg_remove", "output_mode", "output_mode", ["straight_rgba", "premultiplied_rgba"]),
                ...(!isNativeBgRemove ? [check("bg_remove", "use_sam3_details_recovery", "Use SAM3 recovery mask")] : []),
            ],
            note: "When preset values are enabled, the individual chroma parameters are retained but the preset controls processing.",
        });
        if (!isNativeBgRemove) {
            groups.push({
                title: "Easy SAM3 · Model Loader",
                fields: [
                    text("bg_remove", "sam3_model", "model (blank = managed VNCCS model)", { wide: true }),
                    select("bg_remove", "sam3_segmentor", "segmentor", ["image"], { nodeName: "LoadSam3Model", inputName: "segmentor" }),
                    select("bg_remove", "sam3_device", "device", ["auto", "cuda", "cpu", "mps"], { nodeName: "LoadSam3Model", inputName: "device" }),
                    select("bg_remove", "sam3_precision", "precision", ["bf16", "fp16", "fp32"], { nodeName: "LoadSam3Model", inputName: "precision" }),
                ],
            });
            groups.push({
                title: "Easy SAM3 · Image Segmentation / Recovery",
                fields: [
                    textarea("bg_remove", "sam3_prompt", "prompt"),
                    number("bg_remove", "sam3_threshold", "threshold", 0, 1, 0.01),
                    select("bg_remove", "sam3_add_background", "add_background", ["none", "black", "white", "green", "blue"], { nodeName: "Sam3ImageSegmentation", inputName: "add_background" }),
                    number("bg_remove", "sam3_detection_limit", "detection_limit", -1, 10000, 1),
                    number("bg_remove", "sam3_erode_radius", "recovery erode radius", 0, 256, 1),
                    number("bg_remove", "sam3_min_foreground_overlap", "minimum foreground overlap", 0, 1, 0.01),
                ],
            });
        }
        return groups;
    }

    settingsFieldOptions(field, currentValue) {
        const spec = field.nodeName && field.inputName ? this.getInputSpec(field.nodeName, field.inputName) : null;
        const nodeOptions = Array.isArray(spec?.[0]) ? spec[0] : [];
        return uniqueOptions([currentValue, ...(field.options || []), ...nodeOptions]);
    }

    createGeneratorSettingsField(field, draft) {
        if (!draft[field.section] || typeof draft[field.section] !== "object") draft[field.section] = {};
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-settings-field"
            + (field.wide ? " is-wide" : "")
            + (field.type === "checkbox" ? " is-check" : "");
        const caption = document.createElement("span");
        caption.className = "vnccs-pipe-settings-field-label";
        caption.textContent = field.label;
        let input;
        const current = draft[field.section][field.key];
        if (field.type === "checkbox") {
            input = document.createElement("input");
            input.type = "checkbox";
            input.checked = booleanValue(current, false);
            input.onchange = () => {
                draft[field.section][field.key] = input.checked;
            };
            wrap.append(input, caption);
        } else if (field.type === "resolution_scale") {
            wrap.classList.add("is-wide");
            const value = document.createElement("span");
            value.className = "vnccs-pipe-settings-field-label";
            value.textContent = resolutionScaleText(current);
            input = document.createElement("input");
            input.type = "range";
            input.min = String(RESOLUTION_SCALE_MIN_MP);
            input.max = String(RESOLUTION_SCALE_MAX_MP);
            input.step = String(RESOLUTION_SCALE_STEP_MP);
            input.value = resolutionScaleMegapixels(current).toFixed(1);
            input.oninput = () => {
                draft[field.section][field.key] = resolutionScaleValue(input.value);
                value.textContent = resolutionScaleText(draft[field.section][field.key]);
            };
            wrap.append(caption, value, input);
        } else if (field.type === "select") {
            input = document.createElement("select");
            for (const value of this.settingsFieldOptions(field, current)) {
                const option = document.createElement("option");
                option.value = String(value);
                option.textContent = String(value);
                input.appendChild(option);
            }
            input.value = String(current ?? "");
            input.onchange = () => {
                draft[field.section][field.key] = input.value;
                if (field.section === "upscaler" && field.key === "attention_mode") {
                    draft.upscaler.attention_mode_manual = true;
                }
            };
            wrap.append(caption, input);
        } else if (field.type === "textarea") {
            input = document.createElement("textarea");
            input.value = String(current ?? "");
            input.oninput = () => {
                draft[field.section][field.key] = input.value;
            };
            wrap.append(caption, input);
        } else {
            input = document.createElement("input");
            input.type = field.type || "text";
            if (field.type === "number") {
                input.min = String(field.min);
                input.max = String(field.max);
                input.step = String(field.step);
            }
            input.value = String(current ?? "");
            input.oninput = () => {
                if (field.type !== "number") {
                    draft[field.section][field.key] = input.value;
                    return;
                }
                const text = String(input.value).trim().replace(",", ".");
                const value = Number(text);
                if (!text || !Number.isFinite(value)) return;
                draft[field.section][field.key] = Math.max(field.min, Math.min(field.max, value));
            };
            if (field.type === "number") {
                input.onblur = () => { input.value = String(draft[field.section][field.key] ?? ""); };
            }
            wrap.append(caption, input);
        }
        this.protectNativeControl(input);
        return wrap;
    }

    openGeneratorSettings() {
        this.closeModal();
        const draft = JSON.parse(JSON.stringify(this.data));
        const backdrop = document.createElement("div");
        backdrop.className = "vnccs-pipe-modal-backdrop";
        const modal = document.createElement("div");
        modal.className = "vnccs-pipe-modal is-settings";
        const heading = document.createElement("div");
        heading.className = "vnccs-pipe-modal-title";
        heading.textContent = `${this.title} · Settings`;
        const body = document.createElement("div");
        body.className = "vnccs-pipe-modal-body";
        const intro = document.createElement("p");
        intro.className = "vnccs-pipe-settings-intro";
        intro.textContent = "All processing controls are grouped by the internal node that receives them. Connected MODEL, CLIP, VAE, image and conditioning inputs remain managed by the generator.";
        const groupsEl = document.createElement("div");
        groupsEl.className = "vnccs-pipe-settings-groups";
        const groups = this.generatorSettingsGroups();
        const renderGroups = () => {
            groupsEl.replaceChildren();
            for (const group of groups) {
                const groupEl = document.createElement("section");
                groupEl.className = "vnccs-pipe-settings-group";
                const title = document.createElement("div");
                title.className = "vnccs-pipe-settings-group-title";
                title.textContent = group.title;
                const fields = document.createElement("div");
                fields.className = "vnccs-pipe-settings-group-fields";
                for (const field of group.fields) {
                    fields.appendChild(this.createGeneratorSettingsField(field, draft));
                }
                if (group.note) {
                    const note = document.createElement("div");
                    note.className = "vnccs-pipe-settings-note";
                    note.textContent = group.note;
                    fields.appendChild(note);
                }
                groupEl.append(title, fields);
                groupsEl.appendChild(groupEl);
            }
        };
        renderGroups();
        body.append(intro, groupsEl);

        const actions = document.createElement("div");
        actions.className = "vnccs-pipe-modal-actions is-settings";
        const reset = document.createElement("button");
        reset.type = "button";
        reset.className = "vnccs-pipe-modal-btn is-secondary is-reset";
        reset.textContent = "Load Defaults";
        reset.title = "Restore defaults in this dialog. They are saved only after Apply.";
        reset.onclick = () => {
            for (const group of groups) {
                for (const field of group.fields) {
                    const sectionDefaults = DEFAULT_DATA[field.section];
                    if (!sectionDefaults || !(field.key in sectionDefaults)) continue;
                    draft[field.section] = draft[field.section] || {};
                    draft[field.section][field.key] = sectionDefaults[field.key];
                }
            }
            renderGroups();
        };
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "vnccs-pipe-modal-btn is-secondary";
        cancel.textContent = "Cancel";
        cancel.onclick = () => this.closeModal();
        const apply = document.createElement("button");
        apply.type = "button";
        apply.className = "vnccs-pipe-modal-btn";
        apply.textContent = "Apply";
        apply.onclick = () => {
            this.data = deepMerge(DEFAULT_DATA, draft);
            this.syncModelResolution();
            this.data.bg_remove.use_internal_rmbg = false;
            writeData(this.node, this.data);
            this.saveBrowserState();
            this.renderSettings();
            this.closeModal();
        };
        this.protectNativeControl(reset);
        this.protectNativeControl(cancel);
        this.protectNativeControl(apply);
        actions.append(reset, cancel, apply);
        modal.append(heading, body, actions);
        backdrop.appendChild(modal);
        backdrop.onclick = event => {
            if (event.target === backdrop) this.closeModal();
        };
        modal.addEventListener("keydown", event => {
            if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                this.closeModal();
            }
        }, true);
        this.root.appendChild(backdrop);
        this.modalEl = backdrop;
        requestAnimationFrame(() => modal.querySelector("input, select, textarea")?.focus({ preventScroll: true }));
    }

    snapshotStageState() {
        return {
            stageState: Object.fromEntries(
                Object.entries(this.stageState || {}).map(([key, value]) => [key, { ...(value || {}) }])
            ),
            selectedPreview: this.selectedPreview,
            userSelectedPreview: this.userSelectedPreview,
            ui: { ...(this.data.ui || {}) },
        };
    }

    restoreStageSnapshot(snapshot) {
        if (!snapshot) return;
        this.stageState = Object.fromEntries(
            Object.entries(snapshot.stageState || {}).map(([key, value]) => [key, { ...(value || {}) }])
        );
        this.selectedPreview = snapshot.selectedPreview;
        this.userSelectedPreview = snapshot.userSelectedPreview;
        this.data.ui = { ...(snapshot.ui || {}) };
        writeData(this.node, this.data);
        this.renderPreview();
        this.renderChain();
        this.saveBrowserState();
    }

    showModal(title, message) {
        this.closeModal();
        const backdrop = document.createElement("div");
        backdrop.className = "vnccs-pipe-modal-backdrop";
        const modal = document.createElement("div");
        modal.className = "vnccs-pipe-modal";
        const heading = document.createElement("div");
        heading.className = "vnccs-pipe-modal-title";
        heading.textContent = title || "Message";
        const body = document.createElement("div");
        body.className = "vnccs-pipe-modal-body";
        body.textContent = message || "";
        const actions = document.createElement("div");
        actions.className = "vnccs-pipe-modal-actions";
        const ok = document.createElement("button");
        ok.type = "button";
        ok.className = "vnccs-pipe-modal-btn";
        ok.textContent = "OK";
        ok.onclick = () => this.closeModal();
        actions.appendChild(ok);
        modal.append(heading, body, actions);
        backdrop.appendChild(modal);
        backdrop.onclick = (event) => {
            if (event.target === backdrop) this.closeModal();
        };
        this.root.appendChild(backdrop);
        this.modalEl = backdrop;
        ok.focus();
    }

    validateNativeSeedvr(showModal = false) {
        if ((this.data.upscaler?.mode || "seedvr") !== "seedvr") return true;
        // Do not block Queue while /object_info is still loading.
        if (this.nativeSeedvrAvailable !== false) return true;
        if (showModal) {
            const missing = this.nativeSeedvrMissing.length ? `\n\nMissing nodes: ${this.nativeSeedvrMissing.join(", ")}` : "";
            this.showModal(
                "ComfyUI Update Required",
                `Native SeedVR2 is not available in this ComfyUI installation. Update ComfyUI to the latest release and restart it before using SeedVR upscaling.${missing}`,
            );
        }
        return false;
    }

    closeModal() {
        this.modalEl?.remove();
        this.modalEl = null;
    }

    async responseErrorMessage(response) {
        const fallback = `Regenerate failed (${response.status})`;
        try {
            const text = await response.text();
            if (!text) return fallback;
            try {
                const parsed = JSON.parse(text);
                return parsed?.error || parsed?.message || text;
            } catch {
                return text;
            }
        } catch {
            return fallback;
        }
    }

    async regenerateFrom(stageKey, imageIndex = null) {
        if (!this.stages.some(([key]) => key === stageKey)) return;
        this.syncCharacterSourceData();
        this.syncModelResolution();
        this.syncStagesFromData();
        const beforeRegenerate = this.snapshotStageState();
        this.data.regenerate_from = stageKey;
        if (imageIndex !== null && imageIndex !== undefined) {
            this.data.regenerate_index = imageIndex;
        }
        this.selectedPreview = stageKey;
        this.userSelectedPreview = false;
        if (!this.data.ui) this.data.ui = {};
        this.data.ui.selected_preview = stageKey;
        this.data.ui.user_selected_preview = false;
        this.resetStagesFrom(stageKey, { preserveImages: Number.isInteger(imageIndex) });
        this.startRegenerate(stageKey, imageIndex);
        writeData(this.node, this.data);
        this.renderPreview();
        this.renderChain();
        try {
            const response = await api.fetchApi("/vnccs/character_generator/regenerate", {
                method: "POST",
                body: JSON.stringify({
                    unique_id: String(this.node.id ?? ""),
                    generator_type: this.node.type || this.node.comfyClass || "",
                    stage: stageKey,
                    image_index: imageIndex,
                    widget_data: this.data,
                }),
            });
            if (!response.ok) {
                throw new Error(await this.responseErrorMessage(response));
            }
            this.finishRegenerate();
        } catch (error) {
            const hadStageEvent = Boolean(this.regenerateState?.sawStageEvent);
            this.finishRegenerate();
            if (!hadStageEvent) this.restoreStageSnapshot(beforeRegenerate);
            throw error;
        } finally {
            if (this.data.regenerate_from === stageKey) {
                delete this.data.regenerate_from;
                delete this.data.regenerate_index;
                writeData(this.node, this.data);
            }
        }
    }

    controlCenterWidgetNode() {
        const graph = this.node.graph || app.graph;
        const pending = [this.node];
        const visited = new Set();
        while (pending.length) {
            const node = pending.shift();
            if (!node || visited.has(node)) continue;
            visited.add(node);
            if (node.type === "VNCCS_ControlCenter" || node.comfyClass === "VNCCS_ControlCenter") return node;
            for (const input of node.inputs || []) {
                if (input.link == null) continue;
                const link = graph?.links?.[input.link];
                const source = graph?.getNodeById?.(link?.origin_id);
                if (source) pending.push(source);
            }
        }
        return null;
    }

    rememberModelResolution(edited = false) {
        if (this.isEmotions || !this.data.ui?.resolution_model_key) return;
        const section = this.isClone ? "common" : "pose_generation";
        const key = this.data.ui.resolution_model_key;
        const previous = this.data.ui.resolution_by_model?.[key];
        const size = this.data[section].target_size;
        const changed = previous && (previous.target_size ?? previous) !== size;
        const updatedAt = edited || changed
            ? Math.max(Date.now(), (previous?.updated_at || 0) + 1)
            : previous?.updated_at || 0;
        this.data.ui.resolution_by_model = {
            ...(this.data.ui.resolution_by_model || {}),
            [key]: { target_size: size, updated_at: updatedAt },
        };
    }

    syncModelResolution(sourceId = null) {
        const source = this.controlCenterWidgetNode();
        let kind = "";
        let modelKey = "";
        if (source) {
            if (sourceId != null && String(source.id) !== String(sourceId)) return false;
            const stateWidget = source.widgets?.find(widget => widget.name === "node_state");
            let state;
            try { state = JSON.parse(stateWidget?.value || "{}"); } catch { return false; }
            kind = String(state.active_kind || "QI2").trim().toLowerCase();
            const family = state.active_kind || "QI2";
            const type = state.selected_types_by_kind?.[family] || (kind === "qi2" && state.selected_type) || "unet";
            const model = state.selected_models?.[`${family}:${type}`] || state.selected_model || "";
            modelKey = JSON.stringify([kind, type, model]);
        } else if (this.isEmotions && sourceId == null) {
            kind = this.connectedEmotionStudioMode();
        } else {
            return false;
        }
        if (!["qi2", "klein9b", "minimaxh3", "anima", "illustrious"].includes(kind)) return false;
        const previousKind = this.data.ui?.resolution_model_kind;
        const previousBgKind = this.data.ui?.bg_remove_model_kind;
        let changed = false;
        if (kind === "qi2" && this.qi2EmotionDefaultsPending) {
            this.data.emotion_generation = {
                ...(this.data.emotion_generation || {}),
                ...QI2_EMOTION_BBOX_DEFAULTS,
            };
            this.qi2EmotionDefaultsPending = false;
            changed = true;
        }
        if (!this.isEmotions && this.data.ui?.resolution_model_key !== modelKey) {
            const section = this.isClone ? "common" : "pose_generation";
            const settings = this.data[section];
            const previousKey = this.data.ui?.resolution_model_key;
            this.rememberModelResolution();
            const saved = this.data.ui?.resolution_by_model?.[modelKey];
            const savedSize = saved?.target_size ?? saved;
            if (Number.isFinite(savedSize)) {
                settings.target_size = resolutionScaleValue(resolutionScaleMegapixels(savedSize));
            } else if (previousKey || (previousKind && previousKind !== kind) || (!previousKind && Number(settings.target_size) === 1024)) {
                // Use family defaults only for a model without a saved choice.
                settings.target_size = kind === "minimaxh3" ? 1536 : 1024;
            }
            this.data.ui = { ...this.data.ui, resolution_model_key: modelKey };
            if (this.isClone) {
                this.data.pose_generation.target_size = settings.target_size;
                this.data.remove_clothes.target_size = settings.target_size;
            }
            changed = true;
        }
        this.rememberModelResolution();

        if (previousBgKind !== kind || (kind !== "qi2" && this.isNativeBgRemove())) {
            const preset = String(this.data.bg_remove?.preset || "balanced");
            if (kind === "qi2") {
                if (preset.toLowerCase() !== "native") {
                    this.data.ui.bg_remove_previous_preset = preset;
                }
                this.data.bg_remove.preset = "Native";
            } else if (preset.toLowerCase() === "native") {
                this.data.bg_remove.preset = this.previousBgRemovePreset();
            }
            this.data.ui.bg_remove_model_kind = kind;
            changed = true;
        }
        this.data.ui = { ...this.data.ui, resolution_model_kind: kind };
        return changed;
    }

    bindModelResolutionSync() {
        const sync = event => {
            const sourceChanged = this.syncCharacterSourceData();
            const modelChanged = this.syncModelResolution(event?.detail?.node_id);
            if (!sourceChanged && !modelChanged) return;
            this.syncStagesFromData();
            writeData(this.node, this.data);
            this.renderSettings();
            this.renderPreview();
            this.renderChain();
        };
        window.addEventListener("vnccs-control-center-model-changed", sync);
        // Also cover graph reconnection and loading saved workflows in any node order.
        const timer = setInterval(sync, 500);
        registerCleanup(this.node, () => {
            window.removeEventListener("vnccs-control-center-model-changed", sync);
            clearInterval(timer);
        });
    }

    syncCharacterNameFromCreator() {
        return this.syncCharacterSourceData();
    }

    syncCharacterSourceData() {
        if (this.isEmotions) return this.syncEmotionStudioSourceData();
        const matchesType = (node, type, displayName = "") => {
            const title = typeof node?.getTitle === "function" ? node.getTitle() : node?.title;
            return node?.type === type || node?.comfyClass === type || node?.constructor?.type === type || title === displayName;
        };
        const sourceType = this.isClone ? "CharacterCloner" : "CharacterCreatorV2";
        const displayName = this.isClone ? "VNCCS Character Cloner" : "VNCCS Character Creator V2";
        let source = app.graph?._nodes?.find(n => matchesType(n, sourceType, displayName));
        if (!source && this.isClone) source = app.graph?._nodes?.find(n => matchesType(n, "CharacterCreatorV2", "VNCCS Character Creator V2"));
        const widget = source?.widgets?.find(w => w.name === "widget_data");
        const liveState = this.isClone ? source?._vnccsGetClonerState?.() : null;
        if (!widget?.value && !liveState) return;
        let changed = false;
        try {
            const payload = liveState || JSON.parse(widget.value);
            if (payload?.character && this.data.character_name !== payload.character) {
                this.data.character_name = payload.character;
                changed = true;
            }
            if (this.isClone && payload?.character_info && Object.prototype.hasOwnProperty.call(payload.character_info, "nsfw")) {
                const nextNsfw = booleanValue(payload.character_info.nsfw, false);
                if (this.data.nsfw_enabled !== nextNsfw) {
                    this.data.nsfw_enabled = nextNsfw;
                    changed = true;
                }
            }
        } catch {
            // Leave the previous value if the source widget is mid-edit.
        }
        return changed;
    }

    syncEmotionStudioSourceData() {
        const source = this.connectedEmotionStudioNode();
        if (!source) return false;
        const character = source.widgets?.find(w => w.name === "character")?.value || "";
        const costumesRaw = source.widgets?.find(w => w.name === "costumes_data")?.value || "[]";
        const emotionsRaw = source.widgets?.find(w => w.name === "emotions_data")?.value || "[]";
        let costumes = [];
        let emotions = [];
        try { costumes = JSON.parse(costumesRaw); } catch { costumes = []; }
        try { emotions = JSON.parse(emotionsRaw); } catch { emotions = []; }
        const pairs = [];
        for (const costume of costumes || []) {
            for (const emotion of emotions || []) {
                pairs.push({ costume, emotion });
            }
        }
        let changed = false;
        const signature = JSON.stringify(pairs);
        if (this.data.character_name !== character) {
            this.data.character_name = character;
            changed = true;
        }
        if (JSON.stringify(this.data.emotion_pairs || []) !== signature) {
            this.data.emotion_pairs = pairs;
            changed = true;
        }
        return changed;
    }

    isCloneNsfwEnabled() {
        return !this.isClone || this.data.nsfw_enabled !== false;
    }

    currentStages() {
        if (this.isClone) return this.isCloneNsfwEnabled() ? CLONE_STAGES : CLONE_SFW_STAGES;
        if (this.isEmotions) {
            const pairs = Array.isArray(this.data.emotion_pairs) ? this.data.emotion_pairs : [];
            if (!pairs.length) return DEFAULT_EMOTION_STAGES;
            return pairs.map((pair, index) => {
                const key = `emotion_${String(index + 1).padStart(4, "0")}`;
                const label = `${pair.costume || "Costume"} / ${pair.emotion || "Emotion"}`;
                return [`${key}_bg_remove`, label];
            });
        }
        return this.isClothes ? CLOTHES_STAGES : STAGES;
    }

    defaultPreviewStage() {
        if (this.isClone) return "original_pose_generation";
        if (this.isEmotions) return this.currentStages()[0]?.[0] || "emotion_0001_bg_remove";
        return this.isClothes ? "source_upscaler" : "pose_generation";
    }

    syncStagesFromData() {
        const nextStages = this.currentStages();
        const stageCount = Math.max(1, nextStages.length);
        const stageScale = this.isEmotions
            ? Math.max(0.5, Math.min(1, 6 / stageCount))
            : 1;
        this.root?.style.setProperty("--vnccs-stage-count", String(stageCount));
        this.root?.style.setProperty("--vnccs-stage-scale", String(stageScale));
        const nextKeys = new Set(nextStages.map(([key]) => key));
        this.stages = nextStages;
        if (!this.stageState) this.stageState = {};
        for (const [key] of nextStages) {
            if (!this.stageState[key]) {
                this.stageState[key] = { status: "waiting", images: null, message: "" };
            }
        }
        if (!nextKeys.has(this.selectedPreview)) {
            this.selectedPreview = this.defaultPreviewStage();
            this.userSelectedPreview = false;
            this.data.ui = {
                ...(this.data.ui || {}),
                selected_preview: this.selectedPreview,
                user_selected_preview: false,
            };
            this.closeViewer(true);
        }
        this.updateModeClasses();
    }

    updateModeClasses() {
        const cloneNsfw = this.isClone && this.isCloneNsfwEnabled();
        this.root?.classList.toggle("is-clone", cloneNsfw);
        this.root?.classList.toggle("is-clone-sfw", this.isClone && !cloneNsfw);
        this.root?.classList.toggle("is-clothes", this.isClothes);
        this.root?.classList.toggle("is-emotions", this.isEmotions);
        this.chainEl?.classList.toggle("is-clone", cloneNsfw);
        this.chainEl?.classList.toggle("is-clone-sfw", this.isClone && !cloneNsfw);
        this.chainEl?.classList.toggle("is-clothes", this.isClothes);
        this.chainEl?.classList.toggle("is-emotions", this.isEmotions);
    }

    storageKey() {
        return `vnccs:character-generator:${this.node.type || "node"}:${this.node.id}`;
    }

    restoreBrowserState() {
        let saved = null;
        try {
            saved = JSON.parse(localStorage.getItem(this.storageKey()) || "null");
        } catch {
            saved = null;
        }
        if (!saved || saved.version !== 1) return;

        let restoredData = false;
        if (saved.data) {
            const resolutionSections = ["common", "pose_generation", "remove_clothes"];
            const resolutions = resolutionSections.map(section => this.data[section]?.target_size);
            const modelKind = this.data.ui?.resolution_model_kind;
            const modelKey = this.data.ui?.resolution_model_key;
            const modelResolutions = {
                ...(saved.data.ui?.resolution_by_model || {}),
                ...(this.data.ui?.resolution_by_model || {}),
            };
            // A refresh may load an older workflow autosave than the last slider edit.
            // Compare each model independently; equal revisions keep workflow values.
            for (const [key, profile] of Object.entries(saved.data.ui?.resolution_by_model || {})) {
                if ((profile?.updated_at || 0) > (modelResolutions[key]?.updated_at || 0)) {
                    modelResolutions[key] = profile;
                }
            }
            this.data = deepMerge(this.data, saved.data);
            resolutionSections.forEach((section, index) => {
                this.data[section].target_size = resolutions[index];
            });
            this.data.ui.resolution_model_kind = modelKind;
            this.data.ui.resolution_model_key = modelKey;
            this.data.ui.resolution_by_model = modelResolutions;
            const restoredSize = modelResolutions[modelKey]?.target_size;
            if (!this.isEmotions && Number.isFinite(restoredSize)) {
                const section = this.isClone ? "common" : "pose_generation";
                this.data[section].target_size = resolutionScaleValue(resolutionScaleMegapixels(restoredSize));
                if (this.isClone) {
                    this.data.pose_generation.target_size = this.data[section].target_size;
                    this.data.remove_clothes.target_size = this.data[section].target_size;
                }
            }
            restoredData = true;
        }
        if (this.stages.some(([key]) => key === saved.selectedPreview)) {
            this.selectedPreview = saved.selectedPreview;
        }
        this.userSelectedPreview = Boolean(saved.userSelectedPreview);

        if (saved.stageState && typeof saved.stageState === "object") {
            for (const [key] of this.stages) {
                const stage = saved.stageState[key];
                if (!stage || typeof stage !== "object") continue;
                this.stageState[key] = {
                    status: stage.status || "waiting",
                    images: Array.isArray(stage.images) ? stage.images : null,
                    message: stage.message || "",
                    current: stage.current,
                    total: stage.total,
                };
            }
        }

        if (saved.viewer && typeof saved.viewer === "object") {
            if (saved.viewer.open && this.stages.some(([key]) => key === saved.viewer.stage)) {
                this.selectedPreview = saved.viewer.stage;
            }
            if (Number.isFinite(saved.viewer.centerNormX) && Number.isFinite(saved.viewer.centerNormY)) {
                this.viewerFocus = {
                    centerNormX: saved.viewer.centerNormX,
                    centerNormY: saved.viewer.centerNormY,
                    scaleRatio: Number.isFinite(saved.viewer.scaleRatio) ? saved.viewer.scaleRatio : 1,
                };
            }
            this.restoredViewer = saved.viewer;
        }
        if (restoredData) writeData(this.node, this.data);
    }

    saveBrowserState(includeImages = true) {
        this.syncCharacterSourceData();
        this.syncStagesFromData();
        const stageState = {};
        for (const [key] of this.stages) {
            const stage = this.stageState[key] || {};
            stageState[key] = {
                status: stage.status || "waiting",
                images: includeImages ? stage.images : null,
                message: stage.message || "",
                current: stage.current,
                total: stage.total,
            };
        }
        const payload = {
            version: 1,
            data: this.data,
            selectedPreview: this.selectedPreview,
            userSelectedPreview: this.userSelectedPreview,
            stageState,
            viewer: this.serializableViewerState(),
        };
        try {
            localStorage.setItem(this.storageKey(), JSON.stringify(payload));
        } catch {
            if (!includeImages) return;
            const compactState = {};
            for (const [key] of this.stages) {
                const stage = this.stageState[key] || {};
                const images = Array.isArray(stage.images)
                    ? stage.images.filter(src => typeof src === "string" && !src.startsWith("data:"))
                    : null;
                compactState[key] = {
                    status: stage.status || "waiting",
                    images,
                    message: stage.message || "",
                    current: stage.current,
                    total: stage.total,
                };
            }
            try {
                localStorage.setItem(this.storageKey(), JSON.stringify({ ...payload, stageState: compactState }));
            } catch {
                this.saveBrowserState(false);
            }
        }
    }

    scheduleBrowserStateSave() {
        if (this._saveBrowserStateTimer) clearTimeout(this._saveBrowserStateTimer);
        this._saveBrowserStateTimer = setTimeout(() => {
            this._saveBrowserStateTimer = null;
            this.saveBrowserState();
        }, 120);
    }

    serializableViewerState() {
        if (!this.viewer?.open) return { open: false };
        const state = {
            open: true,
            stage: this.selectedPreview,
            index: this.viewer.index || 0,
        };
        const focus = this.currentViewerFocus();
        if (focus) {
            state.scaleRatio = focus.scaleRatio;
            state.centerNormX = focus.centerNormX;
            state.centerNormY = focus.centerNormY;
        }
        return state;
    }

    currentViewerFocus() {
        if (this.viewerFocus) return { ...this.viewerFocus };
        if (!this.viewer?.canvas || !this.viewer?.img || !this.viewer.scale || !this.viewer.fitScale) return null;
        const rect = this.viewerCanvasRect();
        const iw = this.viewer.img.naturalWidth || 1;
        const ih = this.viewer.img.naturalHeight || 1;
        const centerImageX = (rect.width / 2 - this.viewer.x) / this.viewer.scale;
        const centerImageY = (rect.height / 2 - this.viewer.y) / this.viewer.scale;
        return {
            scaleRatio: this.viewer.scale / this.viewer.fitScale,
            centerNormX: centerImageX / iw,
            centerNormY: centerImageY / ih,
        };
    }

    updateViewerFocus() {
        if (!this.viewer?.canvas || !this.viewer?.img || !this.viewer.scale || !this.viewer.fitScale) return;
        const rect = this.viewerCanvasRect();
        const iw = this.viewer.img.naturalWidth || 1;
        const ih = this.viewer.img.naturalHeight || 1;
        const centerImageX = (rect.width / 2 - this.viewer.x) / this.viewer.scale;
        const centerImageY = (rect.height / 2 - this.viewer.y) / this.viewer.scale;
        const focus = {
            scaleRatio: this.viewer.scale / this.viewer.fitScale,
            centerNormX: centerImageX / iw,
            centerNormY: centerImageY / ih,
        };
        this.viewerFocus = {
            scaleRatio: Math.max(1, Math.min(8, focus.scaleRatio)),
            centerNormX: Math.max(-2, Math.min(3, focus.centerNormX)),
            centerNormY: Math.max(-2, Math.min(3, focus.centerNormY)),
        };
    }

    async loadNodeDefs() {
        const names = [
            "VNCCS_QWEN_Encoder",
            "TextEncodeQwenImage21",
            "QwenImage21Cache",
            "KSampler",
            "VAEDecodeTiled",
            "UNETLoader",
            "VAELoader",
            ...NATIVE_SEEDVR_NODE_NAMES,
            "VNCCSChromaKey",
            "UltralyticsDetectorProvider",
            "SAMLoader",
            "FaceDetailer",
            "LoadSam3Model",
            "easy sam3ModelLoader",
            "Sam3ImageSegmentation",
            "easy sam3ImageSegmentation",
        ];
        let allNodeDefs = null;
        await Promise.all(names.map(async name => {
            try {
                const r = await api.fetchApi(`/object_info/${encodeURIComponent(name)}`);
                if (r.ok) {
                    const data = await r.json();
                    this.nodeDefs[name] = data?.[name];
                }
            } catch {
                // Keep static defaults when an optional internal node is unavailable.
            }
        }));
        if (names.some(name => !this.nodeDefs[name])) {
            try {
                const r = await api.fetchApi("/object_info");
                if (r.ok) allNodeDefs = await r.json();
            } catch {
                allNodeDefs = null;
            }
        }
        for (const name of names) {
            if (!this.nodeDefs[name] && allNodeDefs?.[name]) {
                this.nodeDefs[name] = allNodeDefs[name];
            }
        }
        this.nativeSeedvrMissing = NATIVE_SEEDVR_NODE_NAMES.filter(name => !this.nodeDefs[name]);
        if (this.nodeDefs.SeedVR2PostProcessing && !this.getInputSpec("SeedVR2PostProcessing", "color_correction_method")) {
            this.nativeSeedvrMissing.push("SeedVR2PostProcessing.color_correction_method");
        }
        this.nativeSeedvrAvailable = this.nativeSeedvrMissing.length === 0;
        if (!this.nativeSeedvrAvailable && !this.seedvrUpdateModalShown && (this.data.upscaler?.mode || "seedvr") === "seedvr") {
            this.seedvrUpdateModalShown = true;
            this.validateNativeSeedvr(true);
        }
        await this.loadSeedvrAttentionInfo();
        this.renderSettings();
    }

    async loadSeedvrAttentionInfo() {
        try {
            const r = await api.fetchApi("/vnccs/character_generator/seedvr_attention");
            if (!r.ok) return;
            const data = await r.json();
            const available = Array.isArray(data?.available) && data.available.length ? data.available : SEEDVR_ATTENTION_MODES;
            this.seedvrAttention = {
                current: data?.current || "sdpa",
                available: uniqueOptions([data?.current, ...available, ...SEEDVR_ATTENTION_MODES]),
            };
            const upscaler = this.data.upscaler || {};
            if (!upscaler.attention_mode_manual && (!upscaler.attention_mode || upscaler.attention_mode === "sdpa") && data?.current) {
                upscaler.attention_mode = data.current;
                this.data.upscaler = upscaler;
                writeData(this.node, this.data);
            }
        } catch {
            this.seedvrAttention = { current: null, available: SEEDVR_ATTENTION_MODES };
        }
    }

    getInputSpec(nodeName, inputName) {
        const input = this.nodeDefs[nodeName]?.input || {};
        return input.required?.[inputName] || input.optional?.[inputName] || null;
    }

    getOptions(nodeName, inputName, fallback, currentValue = null) {
        const spec = this.getInputSpec(nodeName, inputName);
        const opts = Array.isArray(spec?.[0]) ? spec[0] : fallback;
        return uniqueOptions([currentValue ?? this.data.upscaler[inputName], ...(opts || fallback || [])]);
    }

    getWorkflowModelOptions(nodeName, inputName, workflowOptions, currentValue = null) {
        const spec = this.getInputSpec(nodeName, inputName);
        const nodeOptions = Array.isArray(spec?.[0]) ? spec[0] : [];
        return uniqueOptions([currentValue, ...workflowOptions, ...nodeOptions]);
    }

    syncSelectToOptions(section, key, options) {
        const values = options || [];
        if (!values.length) return values;
        if (!values.includes(this.data[section][key])) {
            this.data[section][key] = values[0];
            writeData(this.node, this.data);
        }
        return values;
    }

    protectNativeControl(input) {
        if (!input || input._vnccsNativeControlProtected) return input;
        input._vnccsNativeControlProtected = true;
        for (const eventName of ["pointerdown", "mousedown", "mouseup", "dblclick", "touchstart", "touchend", "keydown"]) {
            // Let target handlers run before isolating the event from the graph.
            input.addEventListener(eventName, event => event.stopPropagation());
        }
        // Keep the click inside the DOM widget without cancelling the control's
        // own target-phase handler (for example the settings modal opener).
        input.addEventListener("click", event => event.stopPropagation());
        return input;
    }

    modeTabs(section, key, options) {
        const wrap = document.createElement("div");
        wrap.className = "vnccs-pipe-mode-tabs";
        for (const [value, label] of options) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "vnccs-pipe-mode-tab" + (this.data[section][key] === value ? " is-selected" : "");
            btn.textContent = label;
            btn.onclick = () => {
                this.set(section, key, value);
                this.renderSettings();
            };
            wrap.appendChild(btn);
        }
        return wrap;
    }

    field(section, key, label, type = "text", options = null) {
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-field";
        const help = {
            target_size: "Sets the generated image area from 1.0 to 4.0 megapixels while preserving aspect ratio.",
            prompt: "Prompt text used for the remove-clothes/preparation stage.",
            model: "SeedVR diffusion model used for the upscaler stage.",
            resolution: "Target size of the shortest output edge in pixels.",
            max_resolution: "Maximum size of either output edge in pixels. Set to 0 to disable the limit.",
            color_correction: "SeedVR color correction mode. Try adain, wavelet, or none if lab causes color shifts on your GPU.",
            attention_mode: "Attention backend for SeedVR. Auto-detected from installed ComfyUI packages until changed manually.",
            preset: "Strength preset for chroma/background removal.",
            use_sam3_details_recovery: "Uses Easy SAM3 to restore character details after background removal.",
            use_sam: "Passes SAM and the optional segmentation detector into FaceDetailer."
        }[key];
        setHelpText(wrap, help);
        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = label;
        let input;
        if (type === "select") {
            input = document.createElement("select");
            input.className = "vnccs-pipe-select";
            this.protectNativeControl(input);
            const optionValues = options || [];
            if (!optionValues.length) {
                const option = document.createElement("option");
                option.value = "";
                option.textContent = "No models found";
                option.disabled = true;
                input.appendChild(option);
            }
            for (const opt of optionValues) {
                const option = document.createElement("option");
                option.value = opt;
                option.textContent = opt;
                input.appendChild(option);
            }
        } else if (type === "checkbox") {
            wrap.className = "vnccs-pipe-check";
            input = document.createElement("input");
            input.type = "checkbox";
            this.protectNativeControl(input);
            input.checked = Boolean(this.data[section][key]);
            input.onchange = () => this.set(section, key, input.checked);
            for (const eventName of ["pointerdown", "mousedown", "mouseup", "dblclick", "touchstart", "touchend", "keydown"]) {
                wrap.addEventListener(eventName, event => event.stopPropagation());
            }
            wrap.onclick = (event) => {
                event.stopPropagation();
                if (event.target === input) return;
                event.preventDefault();
                input.checked = !input.checked;
                this.set(section, key, input.checked);
            };
            wrap.append(input, caption);
            return wrap;
        } else if (type === "textarea") {
            input = document.createElement("textarea");
            input.className = "vnccs-pipe-textarea";
            this.protectNativeControl(input);
        } else {
            input = document.createElement("input");
            input.className = "vnccs-pipe-input";
            input.type = type;
            if (type === "number" && options && !Array.isArray(options)) {
                for (const attribute of ["min", "max", "step"]) {
                    if (options[attribute] !== undefined) input[attribute] = options[attribute];
                }
            }
            this.protectNativeControl(input);
        }
        input.value = this.data[section][key];
        input.oninput = () => {
            let raw = input.value;
            if (type === "number") {
                if (!input.value.trim()) return;
                raw = Number(input.value);
                if (!Number.isFinite(raw)) return;
                if (options?.min !== undefined) raw = Math.max(options.min, raw);
                if (options?.max !== undefined) raw = Math.min(options.max, raw);
            }
            if (section === "upscaler" && key === "attention_mode") {
                this.data.upscaler.attention_mode_manual = true;
            }
            this.set(section, key, raw);
        };
        if (type === "number") input.onblur = () => { input.value = String(this.data[section][key]); };
        wrap.append(caption, input);
        return wrap;
    }

    resolutionScaleSlider(section, key = "target_size") {
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-slider-field";
        setHelpText(wrap, "Sets the generated image area from 1.0 to 4.0 megapixels while preserving aspect ratio.");
        const head = document.createElement("div");
        head.className = "vnccs-pipe-slider-head";
        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = "resolution scale";
        const value = document.createElement("div");
        value.className = "vnccs-pipe-slider-value";
        value.textContent = resolutionScaleText(this.data[section][key]);
        head.append(caption, value);

        const slider = document.createElement("input");
        slider.className = "vnccs-pipe-slider";
        slider.type = "range";
        slider.min = String(RESOLUTION_SCALE_MIN_MP);
        slider.max = String(RESOLUTION_SCALE_MAX_MP);
        slider.step = String(RESOLUTION_SCALE_STEP_MP);
        slider.value = resolutionScaleMegapixels(this.data[section][key]).toFixed(1);
        slider.style.setProperty("--fill", `${((Number(slider.value) - 1) / 3) * 100}%`);
        slider.setAttribute("aria-label", "Resolution scale in megapixels");
        this.protectNativeControl(slider);
        slider.oninput = () => {
            const targetSize = resolutionScaleValue(slider.value);
            value.textContent = resolutionScaleText(targetSize);
            slider.style.setProperty("--fill", `${((Number(slider.value) - 1) / 3) * 100}%`);
            this.set(section, key, targetSize);
        };
        wrap.append(head, slider);
        return wrap;
    }

    block(title, fields) {
        const block = document.createElement("div");
        block.className = "vnccs-pipe-block";
        const head = document.createElement("div");
        head.className = "vnccs-pipe-block-h";
        head.textContent = title;
        const body = document.createElement("div");
        body.className = "vnccs-pipe-block-b";
        for (const field of fields) body.appendChild(field);
        block.append(head, body);
        return block;
    }

    faceDenoiseSlider() {
        const value = Math.max(0, Math.min(1, Number(this.data.emotion_generation?.face_denoise ?? 0.55)));
        const isAnima = this.connectedEmotionStudioIsAnima();
        const weakLimit = isAnima ? 0.6 : 0.5;
        const optimalLimit = isAnima ? 0.75 : 0.65;
        const denoiseZone = (next) => next < weakLimit
            ? { status: "weak", color: "#64a8ff", border: "rgba(100,168,255,0.5)", bg: "rgba(100,168,255,0.1)", glow: "rgba(100,168,255,0.3)" }
            : (next <= optimalLimit
                ? { status: "optimal", color: "#00d68f", border: "rgba(0,214,143,0.5)", bg: "rgba(0,214,143,0.1)", glow: "rgba(0,214,143,0.28)" }
                : { status: "excessive", color: "#ff5f78", border: "rgba(255,95,120,0.58)", bg: "rgba(255,95,120,0.12)", glow: "rgba(255,95,120,0.32)" });

        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-slider-field";
        setHelpText(wrap, "Controls how strongly the face detailer redraws each emotion face. Low preserves more, high changes more.");

        const head = document.createElement("div");
        head.className = "vnccs-pipe-slider-head";
        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = "face detailer denoise";
        const valueEl = document.createElement("div");
        valueEl.className = "vnccs-pipe-slider-value";
        valueEl.textContent = value.toFixed(2);
        head.append(caption, valueEl);

        const slider = document.createElement("input");
        slider.className = "vnccs-pipe-slider";
        slider.type = "range";
        slider.min = "0";
        slider.max = "1";
        slider.step = "0.01";
        slider.value = String(value);
        this.protectNativeControl(slider);

        const status = document.createElement("div");
        status.className = "vnccs-pipe-slider-status";
        const paint = (nextValue) => {
            const next = Math.max(0, Math.min(1, Number(nextValue)));
            const nextZone = denoiseZone(next);
            slider.style.setProperty("--fill", `${next * 100}%`);
            slider.style.setProperty("--zone-color", nextZone.color);
            slider.style.setProperty("--zone-glow", nextZone.glow);
            status.style.setProperty("--zone-color", nextZone.color);
            status.style.setProperty("--zone-border", nextZone.border);
            status.style.setProperty("--zone-bg", nextZone.bg);
            valueEl.textContent = next.toFixed(2);
            status.textContent = nextZone.status;
        };
        paint(value);
        slider.oninput = () => {
            const next = Math.max(0, Math.min(1, Number(slider.value)));
            paint(next);
            this.set("emotion_generation", "face_denoise", next);
        };

        wrap.append(head, slider, status);
        return wrap;
    }

    faceDetailerNumberField(key, label, { min = 0, max = 1, step = 0.01 } = {}) {
        const draftKey = `emotion_generation.${key}`;
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-field";
        const help = {
            bbox_threshold: "Detection confidence threshold for the face bbox detector.",
            bbox_dilation: "Pixel dilation applied around detected face bounding boxes.",
            sam_dilation: "Pixel dilation applied to the SAM mask.",
            sam_threshold: "SAM mask confidence threshold.",
            sam_bbox_expansion: "Pixel expansion applied to the SAM bounding box."
        }[key];
        setHelpText(wrap, help);

        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = label;

        const input = document.createElement("input");
        input.className = "vnccs-pipe-input";
        input.type = "number";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        this.protectNativeControl(input);
        input.value = this.fieldDrafts.has(draftKey)
            ? this.fieldDrafts.get(draftKey)
            : String(this.data.emotion_generation?.[key] ?? DEFAULT_DATA.emotion_generation[key]);

        const commit = () => {
            const normalized = String(input.value).trim().replace(",", ".");
            const raw = Number(normalized);
            if (!normalized || !Number.isFinite(raw)) {
                input.value = String(this.data.emotion_generation?.[key] ?? DEFAULT_DATA.emotion_generation[key]);
                this.fieldDrafts.delete(draftKey);
                return;
            }
            const next = Math.max(min, Math.min(max, raw));
            this.fieldDrafts.delete(draftKey);
            input.value = String(next);
            this.set("emotion_generation", key, next);
        };

        input.onfocus = () => {
            this.fieldDrafts.set(draftKey, input.value);
        };
        input.oninput = () => {
            this.fieldDrafts.set(draftKey, input.value);
        };
        input.onchange = commit;
        input.onblur = commit;
        input.onkeydown = (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                commit();
                input.blur();
            } else if (event.key === "Escape") {
                event.preventDefault();
                this.fieldDrafts.delete(draftKey);
                input.value = String(this.data.emotion_generation?.[key] ?? DEFAULT_DATA.emotion_generation[key]);
                input.blur();
            }
        };

        wrap.append(caption, input);
        return wrap;
    }

    bgRemoveFields() {
        const fields = [
            this.field("bg_remove", "preset", "mode", "select", this.bgRemoveModes()),
        ];
        if (!this.isNativeBgRemove()) {
            fields.push(this.field("bg_remove", "use_sam3_details_recovery", "Use SAM3 Details Recovery", "checkbox"));
        }
        return fields;
    }

    connectedEmotionStudioMode() {
        const sourceNode = this.connectedEmotionStudioNode();
        if (!sourceNode) return false;

        const settingsWidget = sourceNode.widgets?.find(widget => widget.name === "generation_settings");
        try {
            const settings = settingsWidget?.value ? JSON.parse(settingsWidget.value) : {};
            const settingsMode = String(settings?.generation_mode || "").toLowerCase();
            if (["qi2", "anima", "illustrious"].includes(settingsMode)) return settingsMode;
        } catch (_) {
            // Fall back to the hidden mode widget below.
        }
        const modeWidget = sourceNode.widgets?.find(widget => widget.name === "generation_model");
        const mode = String(modeWidget?.value || "").toLowerCase();
        return ["qi2", "anima", "illustrious"].includes(mode) ? mode : "";
    }

    connectedEmotionStudioNode() {
        if (!this.isEmotions) return null;
        const pipeInput = (this.node.inputs || []).find(input => input.name === "pipe");
        if (!pipeInput || pipeInput.link == null) return null;
        const link = app.graph?.links?.[pipeInput.link];
        const sourceNode = app.graph?.getNodeById?.(link?.origin_id);
        if (!sourceNode || (sourceNode.type !== "EmotionGeneratorV2" && sourceNode.comfyClass !== "EmotionGeneratorV2")) return null;
        return sourceNode;
    }

    connectedEmotionStudioIsAnima() {
        return this.connectedEmotionStudioMode() === "anima";
    }

    shouldShowEmotionDenoiseControl() {
        return this.connectedEmotionStudioMode() !== "qi2";
    }

    async loadSeedvrAssets(force = false) {
        try {
            const response = await api.fetchApi(`/vnccs/character_generator/seedvr_models${force ? "?refresh=1" : ""}`);
            const config = await response.json();
            if (!response.ok || config.error) throw new Error(config.error || "Unable to load SeedVR2 models");
            this.seedvrAssets = {
                models: config.models || [],
                vae: config.vae || [],
            };
            this.renderSettings();
        } catch (error) {
            console.warn("[VNCCS] SeedVR2 model cards unavailable", error);
        }
    }

    seedvrRelativePath(entry) {
        return String(entry?.local_path || "").replace(/\\/g, "/").split("/").slice(2).join("/");
    }

    async downloadSeedvrAsset(category, entry) {
        const key = `${category}:${entry.name}`;
        this.seedvrDownloads[key] = { status: "queued", message: "Queued" };
        this.renderSettings();
        const response = await api.fetchApi("/vnccs/character_generator/seedvr_download", {
            method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
            body: JSON.stringify({ category, name: entry.name }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) {
            this.seedvrDownloads[key] = { status: "error", message: payload.error || "Download failed" };
            this.renderSettings();
            return;
        }
        if (!this.seedvrPollTimer) this.seedvrPollTimer = setInterval(async () => {
            const statusResponse = await api.fetchApi("/vnccs/character_generator/seedvr_download_status");
            this.seedvrDownloads = await statusResponse.json();
            const active = Object.values(this.seedvrDownloads).some(item => ["queued", "downloading"].includes(item?.status));
            if (!active) {
                clearInterval(this.seedvrPollTimer);
                this.seedvrPollTimer = null;
                await this.loadSeedvrAssets(true);
            } else this.renderSettings();
        }, 2000);
    }

    buildSeedvrCard(entry, { pickerHead = false } = {}) {
        const rel = this.seedvrRelativePath(entry);
        const key = `models:${entry.name}`;
        const download = this.seedvrDownloads[key] || {};
        const status = ["queued", "downloading", "error"].includes(download.status) ? download.status : entry.status;
        const installed = status === "installed";
        const ready = installed;
        const selected = this.data.upscaler.model === rel;
        const card = document.createElement("div");
        card.className = `vnccs-seedvr-card ${ready ? "is-installed" : "is-missing"}${selected ? " is-selected" : ""}${pickerHead ? " is-picker-head" : ""}`;
        const displayStatus = status || "missing";
        card.innerHTML = `<div class="vnccs-seedvr-card-head"><span class="vnccs-seedvr-card-dot"></span><span class="vnccs-seedvr-card-name" title="${entry.name}">${entry.name}</span><span class="vnccs-seedvr-card-status">${displayStatus}</span></div><div class="vnccs-seedvr-card-desc">${entry.description || ""}</div>`;
        if (pickerHead) {
            card.onclick = () => {
                this.seedvrModelPickerOpen = !this.seedvrModelPickerOpen;
                this.renderSettings();
            };
        } else if (installed) {
            card.onclick = () => {
                this.set("upscaler", "model", rel);
                this.seedvrModelPickerOpen = false;
                this.renderSettings();
            };
        } else {
            const button = document.createElement("button");
            button.className = "vnccs-seedvr-download";
            button.textContent = status === "error" ? (download.message || "Retry") : (["queued", "downloading"].includes(status) ? (download.message || "Downloading…") : "Install / Download");
            button.disabled = ["queued", "downloading"].includes(status);
            button.onclick = event => {
                event.stopPropagation();
                this.downloadSeedvrAsset("models", entry);
            };
            card.appendChild(button);
        }
        return card;
    }

    seedvrModelCards() {
        const container = document.createElement("div");
        container.className = "vnccs-seedvr-cards";
        if (!this.seedvrAssets) {
            container.textContent = "Loading model catalogue…";
            return container;
        }
        const entries = this.seedvrAssets.models || [];
        const selected = entries.find(entry => this.seedvrRelativePath(entry) === this.data.upscaler.model) || entries[0];
        if (!selected) return container;
        const picker = document.createElement("div");
        picker.className = `vnccs-seedvr-picker${this.seedvrModelPickerOpen ? " is-open" : ""}`;
        picker.appendChild(this.buildSeedvrCard(selected, { pickerHead: true }));
        const menu = document.createElement("div");
        menu.className = "vnccs-seedvr-picker-menu";
        entries.forEach(entry => menu.appendChild(this.buildSeedvrCard(entry)));
        picker.appendChild(menu);
        container.appendChild(picker);
        return container;
    }

    renderSettings() {
        this.syncCharacterSourceData();
        this.syncStagesFromData();
        this.settingsEl.innerHTML = "";
        const title = document.createElement("div");
        title.className = "vnccs-pipe-title";
        title.textContent = this.title;
        this.settingsEl.appendChild(title);
        if (this.isEmotions) {
            const qi2Emotion = this.connectedEmotionStudioMode() === "qi2";
            const count = Array.isArray(this.data.emotion_pairs) ? this.data.emotion_pairs.length : 0;
            const info = document.createElement("div");
            info.className = "vnccs-pipe-block";
            info.innerHTML = `
                <div class="vnccs-pipe-block-h">Emotion Generation</div>
                <div class="vnccs-pipe-block-b">
                    <div class="vnccs-pipe-label">character</div>
                    <div class="vnccs-pipe-empty" style="min-height:auto;padding:8px;">${this.data.character_name || "Select in Emotion Studio"}</div>
                    <div class="vnccs-pipe-label">steps</div>
                    <div class="vnccs-pipe-empty" style="min-height:auto;padding:8px;">${count} costume / emotion pair(s)</div>
                </div>`;
            this.settingsEl.appendChild(info);
            if (qi2Emotion) {
                this.settingsEl.appendChild(this.block("QI2 Face Generation", [
                    this.resolutionScaleSlider("emotion_generation", "target_size"),
                ]));
                this.settingsEl.appendChild(this.block("VNCCS BBox Extractor", [
                    this.faceDetailerNumberField("bbox_threshold", "threshold", { min: 0, max: 1, step: 0.01 }),
                    this.faceDetailerNumberField("bbox_dilation", "dilation", { min: 0, max: 1024, step: 1 }),
                    this.faceDetailerNumberField("feather", "feather", { min: 0, max: 1024, step: 1 }),
                    this.faceDetailerNumberField("drop_size", "drop_size", { min: 1, max: 4096, step: 1 }),
                ]));
            } else if (this.shouldShowEmotionDenoiseControl()) {
                this.settingsEl.appendChild(this.block("Emotion Strength", [
                    this.faceDenoiseSlider(),
                ]));
            }
            if (!qi2Emotion) {
                const faceDetailerFields = [
                    this.faceDetailerNumberField("task_batch_size", "task_batch_size (0 = auto)", { min: 0, max: 32, step: 1 }),
                    this.field("emotion_generation", "use_sam", "Use SAM", "checkbox"),
                    this.faceDetailerNumberField("bbox_threshold", "bbox_threshold", { min: 0, max: 1, step: 0.01 }),
                    this.faceDetailerNumberField("bbox_dilation", "bbox_dilation", { min: 0, max: 128, step: 1 }),
                    this.faceDetailerNumberField("sam_dilation", "sam_dilation", { min: 0, max: 128, step: 1 }),
                    this.faceDetailerNumberField("sam_threshold", "sam_threshold", { min: 0, max: 1, step: 0.01 }),
                    this.faceDetailerNumberField("sam_bbox_expansion", "sam_bbox_expansion", { min: 0, max: 128, step: 1 }),
                ];
                this.settingsEl.appendChild(this.block("Face Detailer", faceDetailerFields));
            }
            this.settingsEl.appendChild(this.block("BG Remove", this.bgRemoveFields()));
            return;
        }
        if (this.isClone) {
            this.settingsEl.appendChild(this.block("Common", [
                this.resolutionScaleSlider("common"),
            ]));
            if (this.isCloneNsfwEnabled()) {
                this.settingsEl.appendChild(this.block("Remove Clothes", [
                    this.field("remove_clothes", "prompt", "prompt", "textarea"),
                ]));
            }
        } else {
            this.settingsEl.appendChild(this.block("Pose Generation", [
                this.resolutionScaleSlider("pose_generation"),
            ]));
        }
        const upscalerFields = [
            this.modeTabs("upscaler", "mode", [["seedvr", "SeedVR"], ["off", "OFF"]]),
        ];
        if (this.data.upscaler.mode !== "off") {
            const resolutionFields = document.createElement("div");
            resolutionFields.className = "vnccs-pipe-field-row";
            resolutionFields.append(
                this.field("upscaler", "resolution", "target short edge", "number", { min: 16, max: 16384, step: 2 }),
                this.field("upscaler", "max_resolution", "maximum edge", "number", { min: 0, max: 16384, step: 2 }),
            );
            upscalerFields.push(
                this.seedvrModelCards(),
                resolutionFields,
                this.field("upscaler", "color_correction", "color correction", "select", this.getOptions("SeedVR2PostProcessing", "color_correction_method", SEEDVR_COLOR_CORRECTION_MODES, this.data.upscaler.color_correction)),
            );
        }
        this.settingsEl.appendChild(this.block("Upscaler", upscalerFields));
        this.settingsEl.appendChild(this.block("BG Remove", this.bgRemoveFields()));
    }

    renderPreview() {
        this.syncStagesFromData();
        this.previewEl.innerHTML = "";
        const head = document.createElement("div");
        head.className = "vnccs-pipe-preview-head";
        const label = document.createElement("div");
        label.className = "vnccs-pipe-preview-label";
        label.textContent = this.stages.find(([key]) => key === this.selectedPreview)?.[1] || "Results";
        const tabs = document.createElement("div");
        tabs.className = "vnccs-pipe-tabs";
        for (const [key, name] of this.stages) {
            const tab = document.createElement("button");
            tab.className = "vnccs-pipe-tab" + (key === this.selectedPreview ? " is-selected" : "");
            tab.textContent = name;
            tab.onclick = () => {
                this.selectedPreview = key;
                this.userSelectedPreview = true;
                this.persistUI();
                this.renderPreview();
            };
            tabs.appendChild(tab);
        }
        if (this.regenerateState) {
            const regen = document.createElement("div");
            regen.className = "vnccs-pipe-regen-status";
            const spinner = document.createElement("span");
            spinner.className = "vnccs-pipe-regen-spinner";
            const activeName = this.stages.find(([key]) => key === this.regenerateState.activeStage)?.[1] || "Stage";
            const itemText = Number.isInteger(this.regenerateState.imageIndex) ? ` #${this.regenerateState.imageIndex + 1}` : "";
            const text = document.createElement("span");
            text.textContent = `Regenerating ${activeName}${itemText} · ${this.formatElapsed(this.regenerateState.elapsed)}`;
            regen.append(spinner, text);
            head.append(label, regen);
        } else {
            head.append(label, tabs);
        }
        this.previewEl.appendChild(head);

        const images = this.stageState[this.selectedPreview]?.images;
        if (!images?.length) {
            const empty = document.createElement("div");
            empty.className = "vnccs-pipe-empty";
            empty.textContent = this.formatStageStatus(this.selectedPreview);
            this.previewEl.appendChild(empty);
            return;
        }
        const grid = document.createElement("div");
        grid.className = "vnccs-pipe-grid";
        const selectedState = this.stageState[this.selectedPreview] || {};
        const canRegenerateImages = selectedState.status === "done" && !this.regenerateState;
        images.forEach((src, index) => {
            const tile = document.createElement("div");
            tile.tabIndex = 0;
            tile.role = "button";
            tile.className = "vnccs-pipe-img";
            tile.style.backgroundImage = `url("${String(src).replaceAll('"', "%22")}")`;
            tile.dataset.src = src;
            tile.onclick = () => this.openViewer(index);
            tile.onkeydown = (event) => {
                if (event.target !== tile) return;
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    this.openViewer(index);
                }
            };
            if (canRegenerateImages) {
                const regen = document.createElement("button");
                regen.type = "button";
                regen.className = "vnccs-pipe-img-regen";
                regen.textContent = "Regenerate";
                regen.onclick = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    this.regenerateFrom(this.selectedPreview, index).catch((error) => {
                        console.error("[VNCCS Character Generator] Image regenerate failed:", error);
                        this.showModal("Regenerate Failed", error?.message || "Regenerate failed");
                    });
                };
                tile.appendChild(regen);
            }
            grid.appendChild(tile);
        });
        this.previewEl.appendChild(grid);
        this.schedulePreviewGridLayout(grid, images);
    }

    schedulePreviewGridLayout(grid, images) {
        if (this.previewLayoutFrame) cancelAnimationFrame(this.previewLayoutFrame);
        this.previewLayoutFrame = requestAnimationFrame(() => {
            this.previewLayoutFrame = null;
            this.layoutPreviewGrid(grid, images);
        });
        for (const src of images) this.ensureImageMetrics(src, () => this.layoutPreviewGrid(grid, images));
    }

    ensureImageMetrics(src, onReady) {
        const existing = this.imageMetrics.get(src);
        if (existing) {
            if (existing.loading && onReady) existing.callbacks.push(onReady);
            else onReady?.();
            return;
        }
        this.imageMetrics.set(src, { width: 1, height: 1, loading: true, callbacks: onReady ? [onReady] : [] });
        const img = new Image();
        img.onload = () => {
            const callbacks = this.imageMetrics.get(src)?.callbacks || [];
            this.imageMetrics.set(src, {
                width: img.naturalWidth || 1,
                height: img.naturalHeight || 1,
                loading: false,
            });
            callbacks.forEach(callback => callback?.());
        };
        img.onerror = () => {
            const callbacks = this.imageMetrics.get(src)?.callbacks || [];
            this.imageMetrics.set(src, { width: 1, height: 1, loading: false });
            callbacks.forEach(callback => callback?.());
        };
        img.src = src;
    }

    layoutPreviewGrid(grid, images) {
        if (!grid?.isConnected || !images?.length) return;
        const rect = grid.getBoundingClientRect();
        const gap = 8;
        const availableW = Math.max(1, grid.clientWidth || rect.width || 1);
        const availableH = Math.max(1, grid.clientHeight || rect.height || 1);
        const aspects = images.map(src => {
            const metrics = this.imageMetrics.get(src);
            return Math.max(0.05, Math.min(20, (metrics?.width || 1) / (metrics?.height || 1)));
        });

        let best = null;
        for (let cols = 1; cols <= images.length; cols++) {
            const rows = Math.ceil(images.length / cols);
            const cellW = (availableW - gap * (cols - 1)) / cols;
            const cellH = (availableH - gap * (rows - 1)) / rows;
            if (cellW <= 0 || cellH <= 0) continue;

            let minArea = Infinity;
            let totalArea = 0;
            for (const aspect of aspects) {
                const drawW = Math.min(cellW, cellH * aspect);
                const drawH = drawW / aspect;
                const area = drawW * drawH;
                minArea = Math.min(minArea, area);
                totalArea += area;
            }
            const score = minArea * 1000000 + totalArea;
            if (!best || score > best.score) {
                best = { cols, rows, cellW, cellH, score };
            }
        }

        if (!best) return;
        grid.style.gridTemplateColumns = `repeat(${best.cols}, ${Math.floor(best.cellW)}px)`;
        grid.style.gridTemplateRows = `repeat(${best.rows}, ${Math.floor(best.cellH)}px)`;

        [...grid.children].forEach((tile, index) => {
            const aspect = aspects[index] || 1;
            const drawW = Math.min(best.cellW, best.cellH * aspect);
            const drawH = drawW / aspect;
            tile.style.width = `${Math.max(1, Math.floor(drawW))}px`;
            tile.style.height = `${Math.max(1, Math.floor(drawH))}px`;
        });
    }

    formatStageStatus(key) {
        const state = this.stageState[key] || {};
        const status = state.status || "waiting";
        const count = Number.isFinite(state.current) && Number.isFinite(state.total)
            ? ` (${state.current}/${state.total})`
            : (state.images?.length ? ` (${state.images.length})` : "");
        if (this.regenerateState?.activeStage === key && status === "waiting") {
            return `Starting regenerate · ${this.formatElapsed(this.regenerateState.elapsed)}`;
        }
        if (this.regenerateState?.targetStages?.includes(key) && status === "waiting") {
            return "Queued for regenerate";
        }
        if (state.message) return `${state.message}${count}`;
        if (status === "running") return `Running${count}`;
        if (status === "done") return `Done${count}`;
        if (status === "error") return "Error";
        return "Waiting";
    }

    formatElapsed(seconds = 0) {
        const value = Math.max(0, Number(seconds) || 0);
        const minutes = Math.floor(value / 60);
        const secs = value % 60;
        return minutes ? `${minutes}:${String(secs).padStart(2, "0")}` : `${secs}s`;
    }

    estimateRegenerateProgress() {
        if (!this.regenerateState) return 0;
        const elapsed = Math.max(0, this.regenerateState.elapsed || 0);
        return Math.min(92, 8 + elapsed * 3);
    }

    renderChain() {
        this.syncStagesFromData();
        this.chainEl.innerHTML = "";
        this.updateModeClasses();
        this.chainEl.style.setProperty("--vnccs-stage-count", String(Math.max(1, this.stages.length)));
        for (const [key, name] of this.stages) {
            const stage = document.createElement("div");
            const status = this.stageState[key]?.status || "waiting";
            const isRegeneratingStage = this.regenerateState?.targetStages?.includes(key);
            stage.className = "vnccs-pipe-stage";
            if (status === "running") stage.classList.add("is-active");
            if (isRegeneratingStage) stage.classList.add("is-regenerating");
            if (status === "done") stage.classList.add("is-done");
            stage.onclick = () => {
                this.selectedPreview = key;
                this.userSelectedPreview = true;
                this.persistUI();
                this.renderPreview();
            };
            const n = document.createElement("div");
            n.className = "vnccs-pipe-stage-name";
            n.textContent = name;
            const s = document.createElement("div");
            s.className = "vnccs-pipe-stage-status";
            s.textContent = this.formatStageStatus(key);
            stage.append(n, s);
            if (status === "running" || this.regenerateState?.activeStage === key) {
                const progress = document.createElement("div");
                progress.className = "vnccs-pipe-stage-progress";
                const fill = document.createElement("div");
                fill.className = "vnccs-pipe-stage-progress-fill";
                const state = this.stageState[key] || {};
                const current = Number(state.current);
                const total = Number(state.total);
                if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
                    fill.style.width = `${Math.max(4, Math.min(100, (current / total) * 100))}%`;
                } else {
                    fill.style.width = `${this.estimateRegenerateProgress()}%`;
                }
                progress.appendChild(fill);
                stage.appendChild(progress);
            }
            if (key === "pose_generation" || key === "original_pose_generation" || key === "naked_pose_generation") {
                const l = document.createElement("div");
                l.className = "vnccs-pipe-stage-lora";
                const poseLora = this.data.ui?.resolution_model_kind === "klein9b"
                    ? "VNCCS Pose Studio Klein9b" : "VNCCS Pose Studio QI2";
                l.textContent = `LoRA: ${poseLora}`;
                stage.appendChild(l);
            }
            if (key === "remove_clothes" && this.data.ui?.resolution_model_kind !== "qi2") {
                const l = document.createElement("div");
                l.className = "vnccs-pipe-stage-lora";
                l.textContent = `LoRA: ${CLOTHES_CORE_LORA_LABEL}`;
                stage.appendChild(l);
            }
            if (status === "done" && !this.regenerateState) {
                const actions = document.createElement("div");
                actions.className = "vnccs-pipe-stage-actions";
                const regen = document.createElement("button");
                regen.type = "button";
                regen.className = "vnccs-pipe-regen";
                regen.textContent = "Regenerate";
                regen.onclick = (event) => {
                    event.stopPropagation();
                    this.regenerateFrom(key).catch((error) => {
                        console.error("[VNCCS Character Generator] Regenerate failed:", error);
                        this.showModal("Regenerate Failed", error?.message || "Regenerate failed");
                    });
                };
                actions.appendChild(regen);
                stage.appendChild(actions);
            }
            this.chainEl.appendChild(stage);
        }
    }

    currentImages() {
        return this.stageState[this.selectedPreview]?.images || [];
    }

    openViewer(index = 0, restored = null) {
        const images = this.currentImages();
        if (!images.length) return;
        const returnFocus = this.viewer?.returnFocus || document.activeElement;
        this.closeViewer();
        if (!restored) {
            this.userSelectedPreview = true;
            this.persistUI();
        }
        this.viewer = {
            open: true,
            index: Math.max(0, Math.min(index, images.length - 1)),
            scale: 1,
            fitScale: 1,
            x: 0,
            y: 0,
            dragging: false,
            returnFocus,
            restored,
        };
        if (restored?.open && Number.isFinite(restored.centerNormX) && Number.isFinite(restored.centerNormY)) {
            this.viewerFocus = {
                centerNormX: restored.centerNormX,
                centerNormY: restored.centerNormY,
                scaleRatio: Number.isFinite(restored.scaleRatio) ? restored.scaleRatio : 1,
            };
        } else {
            this.viewerFocus = null;
        }
        this.renderViewer();
        this.saveBrowserState();
    }

    renderViewer() {
        const stageScrollLeft = this.viewer?.stageTabs?.scrollLeft || 0;
        this.closeViewer();
        const overlay = document.createElement("div");
        overlay.className = "vnccs-pipe-viewer";
        overlay.tabIndex = -1;
        overlay.setAttribute("aria-label", "Image viewer. Press Escape to close.");
        overlay.onkeydown = (event) => {
            if (event.key !== "Escape") return;
            event.preventDefault();
            event.stopPropagation();
            this.closeViewer(true);
        };
        for (const type of ["pointerdown", "mousedown", "click", "dblclick"]) {
            overlay.addEventListener(type, (event) => {
                if (event.button !== 1) event.stopPropagation();
            });
        }
        const bar = document.createElement("div");
        bar.className = "vnccs-pipe-viewer-bar";
        const back = document.createElement("button");
        back.type = "button";
        back.className = "vnccs-pipe-viewer-btn";
        back.textContent = "BACK";
        back.title = "Close viewer (Escape)";
        back.onclick = () => this.closeViewer(true);
        bar.appendChild(back);
        const stageTabs = document.createElement("div");
        stageTabs.className = "vnccs-pipe-viewer-stages";
        for (const [key, name] of this.stages) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "vnccs-pipe-viewer-btn" + (key === this.selectedPreview ? " is-selected" : "");
            btn.textContent = name;
            btn.onclick = () => {
                this.updateViewerFocus();
                const viewerFocus = this.currentViewerFocus();
                this.selectedPreview = key;
                this.userSelectedPreview = true;
                this.persistUI();
                this.viewer.index = this.clampedViewerIndex();
                this.viewer.restored = {
                    open: true,
                    stage: key,
                    index: this.viewer.index,
                    ...(viewerFocus || {}),
                };
                this.renderViewer();
                this.renderPreview();
            };
            stageTabs.appendChild(btn);
        }
        const zoomOut = document.createElement("button");
        zoomOut.type = "button";
        zoomOut.className = "vnccs-pipe-viewer-btn";
        zoomOut.textContent = "-";
        zoomOut.setAttribute("aria-label", "Zoom out");
        zoomOut.onclick = () => this.zoomViewer(0.8);
        const zoomIn = document.createElement("button");
        zoomIn.type = "button";
        zoomIn.className = "vnccs-pipe-viewer-btn";
        zoomIn.textContent = "+";
        zoomIn.setAttribute("aria-label", "Zoom in");
        zoomIn.onclick = () => this.zoomViewer(1.25);
        bar.append(stageTabs, zoomOut, zoomIn);

        const canvas = document.createElement("div");
        canvas.className = "vnccs-pipe-viewer-canvas";
        const img = document.createElement("img");
        img.className = "vnccs-pipe-viewer-img";
        img.draggable = false;
        canvas.appendChild(img);
        overlay.append(bar, canvas);
        this.root.appendChild(overlay);
        this.viewer.overlay = overlay;
        this.viewer.canvas = canvas;
        this.viewer.img = img;
        this.viewer.stageTabs = stageTabs;
        stageTabs.scrollLeft = stageScrollLeft;
        this.viewer.fitApplied = false;

        const viewer = this.viewer;
        let fitFrame = null;
        const scheduleFit = () => {
            cancelAnimationFrame(fitFrame);
            fitFrame = requestAnimationFrame(() => {
                fitFrame = null;
                if (this.viewer === viewer && viewer.canvas === canvas) this.fitViewer();
            });
        };
        img.onload = scheduleFit;
        img.onerror = () => img.classList.remove("is-ready");
        img.decoding = "async";
        img.src = this.currentImages()[this.viewer.index] || "";
        if (img.complete && img.naturalWidth) scheduleFit();
        canvas.onwheel = (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (!event.deltaY) return;
            const factor = event.deltaY < 0 ? 1.12 : 0.88;
            this.zoomViewer(factor, event);
        };
        const finishDrag = (event) => {
            if (this.viewer !== viewer || viewer.canvas !== canvas) return;
            if (!viewer.dragging) return;
            if (event?.pointerId !== undefined && event.pointerId !== viewer.pointerId) return;
            const pointerId = viewer.pointerId;
            viewer.dragging = false;
            viewer.pointerId = null;
            canvas.classList.remove("is-dragging");
            if (canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
            if (this.viewer === viewer) {
                this.updateViewerFocus();
                this.saveBrowserState();
            }
        };
        canvas.onpointerdown = (event) => {
            if (event.button !== 0 || event.isPrimary === false || !viewer.fitApplied) return;
            event.preventDefault();
            event.stopPropagation();
            overlay.focus({ preventScroll: true });
            const point = this.viewerEventPoint(event);
            viewer.dragging = true;
            viewer.pointerId = event.pointerId;
            viewer.dragX = point.x;
            viewer.dragY = point.y;
            canvas.classList.add("is-dragging");
            canvas.setPointerCapture(event.pointerId);
        };
        canvas.onpointermove = (event) => {
            if (this.viewer !== viewer || viewer.canvas !== canvas) return;
            if (!viewer.dragging || event.pointerId !== viewer.pointerId) return;
            if (!(event.buttons & 1)) {
                finishDrag(event);
                return;
            }
            event.stopPropagation();
            const point = this.viewerEventPoint(event);
            viewer.x += point.x - viewer.dragX;
            viewer.y += point.y - viewer.dragY;
            viewer.dragX = point.x;
            viewer.dragY = point.y;
            this.applyViewerTransform();
            this.updateViewerFocus();
            this.scheduleBrowserStateSave();
        };
        canvas.onpointerup = finishDrag;
        canvas.onpointercancel = finishDrag;
        canvas.onlostpointercapture = finishDrag;
        const onVisibilityChange = () => {
            if (document.hidden) finishDrag();
        };
        window.addEventListener("pointerup", finishDrag, true);
        window.addEventListener("blur", finishDrag);
        document.addEventListener("visibilitychange", onVisibilityChange);
        const resizeObserver = new ResizeObserver(() => {
            if (this.viewer !== viewer || viewer.canvas !== canvas || !canvas.isConnected) return;
            if (!viewer.fitApplied) {
                scheduleFit();
                return;
            }
            viewer.restored = { open: true, ...this.currentViewerFocus() };
            viewer.fitApplied = false;
            scheduleFit();
        });
        resizeObserver.observe(canvas);
        viewer.dispose = () => {
            finishDrag();
            cancelAnimationFrame(fitFrame);
            img.onload = null;
            img.onerror = null;
            resizeObserver.disconnect();
            window.removeEventListener("pointerup", finishDrag, true);
            window.removeEventListener("blur", finishDrag);
            document.removeEventListener("visibilitychange", onVisibilityChange);
        };
        overlay.focus({ preventScroll: true });
    }

    closeViewer(clear = false) {
        this.viewer?.dispose?.();
        if (this.viewer) this.viewer.dispose = null;
        this.viewer?.overlay?.remove();
        if (clear) {
            const returnFocus = this.viewer?.returnFocus;
            this.viewer = null;
            this.saveBrowserState();
            if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
        }
    }

    syncViewerImage() {
        const images = this.currentImages();
        if (!this.viewer?.img) return;
        this.viewer.index = this.clampedViewerIndex();
        const src = images[this.viewer.index] || "";
        if ((this.viewer.img.getAttribute("src") || "") === src) return;
        this.viewer.restored = { open: true, ...this.currentViewerFocus() };
        this.viewer.fitApplied = false;
        this.viewer.img.classList.remove("is-ready");
        if (src) this.viewer.img.src = src;
        else this.viewer.img.removeAttribute("src");
    }

    clampedViewerIndex() {
        const images = this.currentImages();
        if (!images.length) return 0;
        return Math.max(0, Math.min(this.viewer?.index ?? 0, images.length - 1));
    }

    fitViewer() {
        if (!this.viewer?.img || !this.viewer?.canvas) return;
        if (this.viewer.fitApplied) return;
        if (!this.viewer.img.complete || !this.viewer.img.naturalWidth || !this.viewer.img.naturalHeight) return;
        if (!this.viewer.canvas.clientWidth || !this.viewer.canvas.clientHeight) return;
        const rect = this.viewerCanvasRect();
        const iw = this.viewer.img.naturalWidth || 1;
        const ih = this.viewer.img.naturalHeight || 1;
        const fit = Math.min(rect.width / iw, rect.height / ih);
        this.viewer.fitScale = fit;
        const restored = this.viewer.restored;
        if (restored?.open && Number.isFinite(restored.scaleRatio)) {
            const scaleRatio = Math.max(1, Math.min(8, restored.scaleRatio));
            this.viewer.scale = fit * scaleRatio;
            const centerNormX = Number.isFinite(restored.centerNormX)
                ? restored.centerNormX
                : (Number.isFinite(restored.centerImageX) ? restored.centerImageX / iw : 0.5);
            const centerNormY = Number.isFinite(restored.centerNormY)
                ? restored.centerNormY
                : (Number.isFinite(restored.centerImageY) ? restored.centerImageY / ih : 0.5);
            const centerImageX = Math.max(-2, Math.min(3, centerNormX)) * iw;
            const centerImageY = Math.max(-2, Math.min(3, centerNormY)) * ih;
            this.viewer.x = rect.width / 2 - centerImageX * this.viewer.scale;
            this.viewer.y = rect.height / 2 - centerImageY * this.viewer.scale;
            this.viewer.restored = null;
            this.restoredViewer = null;
            this.viewerFocus = { scaleRatio, centerNormX, centerNormY };
        } else {
            this.viewer.scale = fit;
            this.centerViewerImage(rect, iw, ih);
            this.viewerFocus = { scaleRatio: 1, centerNormX: 0.5, centerNormY: 0.5 };
        }
        this.viewer.fitApplied = true;
        this.applyViewerTransform();
        this.saveBrowserState();
    }

    viewerCanvasRect() {
        const canvas = this.viewer?.canvas;
        if (!canvas) return { width: 1, height: 1 };
        const rect = canvas.getBoundingClientRect();
        return {
            left: rect.left || 0,
            top: rect.top || 0,
            width: canvas.clientWidth || rect.width || 1,
            height: canvas.clientHeight || rect.height || 1,
            viewportWidth: rect.width || canvas.clientWidth || 1,
            viewportHeight: rect.height || canvas.clientHeight || 1,
        };
    }

    viewerEventPoint(event, rect = null) {
        rect = rect || this.viewerCanvasRect();
        if (!event) return { x: rect.width / 2, y: rect.height / 2 };
        const sx = rect.width / (rect.viewportWidth || rect.width || 1);
        const sy = rect.height / (rect.viewportHeight || rect.height || 1);
        return {
            x: (event.clientX - rect.left) * sx,
            y: (event.clientY - rect.top) * sy,
        };
    }

    centerViewerImage(rect = null, iw = null, ih = null, lockYToTop = false) {
        if (!this.viewer?.canvas || !this.viewer?.img) return;
        rect = rect || this.viewerCanvasRect();
        iw = iw || this.viewer.img.naturalWidth || 1;
        ih = ih || this.viewer.img.naturalHeight || 1;
        this.viewer.x = rect.width / 2 - (iw * this.viewer.scale) / 2;
        this.viewer.y = lockYToTop ? 0 : (rect.height - ih * this.viewer.scale) / 2;
    }

    applyViewerTransform() {
        if (!this.viewer?.img) return;
        this.viewer.img.style.width = `${this.viewer.img.naturalWidth}px`;
        this.viewer.img.style.height = `${this.viewer.img.naturalHeight}px`;
        // Keep translation in canvas pixels. Using translate() scale() lets the
        // transform stack affect the translate component in some browser paths,
        // which breaks zoom-to-cursor especially on square images.
        this.viewer.img.style.transform = `matrix(${this.viewer.scale}, 0, 0, ${this.viewer.scale}, ${this.viewer.x}, ${this.viewer.y})`;
        this.viewer.img.classList.add("is-ready");
    }

    zoomViewer(factor, event = null) {
        if (!this.viewer?.canvas || !this.viewer?.img) return;
        if (!this.viewer.fitApplied) return;
        const rect = this.viewerCanvasRect();
        const oldScale = this.viewer.scale;
        const fitScale = this.viewer.fitScale || 1;
        const minScale = fitScale;
        const maxScale = this.viewer.fitScale * 8;
        const nextScale = Math.max(minScale, Math.min(maxScale, oldScale * factor));
        if (nextScale === oldScale) return;

        const anchor = this.viewerEventPoint(event, rect);
        const anchorX = anchor.x;
        const anchorY = anchor.y;
        const imagePointX = (anchorX - this.viewer.x) / oldScale;
        const imagePointY = (anchorY - this.viewer.y) / oldScale;
        let nextX = anchorX - imagePointX * nextScale;
        let nextY = anchorY - imagePointY * nextScale;

        const iw = this.viewer.img.naturalWidth || 1;
        const ih = this.viewer.img.naturalHeight || 1;
        const fitX = (rect.width - iw * fitScale) / 2;
        const fitY = (rect.height - ih * fitScale) / 2;

        if (nextScale <= fitScale + 0.0001) {
            nextX = fitX;
            nextY = fitY;
        } else if (nextScale < fitScale * 1.6) {
            const t = 1 - ((nextScale / fitScale) - 1) / 0.6;
            const ease = Math.max(0, Math.min(1, t * t * (3 - 2 * t)));
            nextX += (fitX - nextX) * ease * 0.35;
            nextY += (fitY - nextY) * ease * 0.35;
        }

        this.viewer.x = nextX;
        this.viewer.y = nextY;
        this.viewer.scale = nextScale;
        this.applyViewerTransform();
        this.updateViewerFocus();
        this.scheduleBrowserStateSave();
    }
}

app.registerExtension({
    name: "VNCCS.CharacterGenerator",
    async setup() {
        if (app._vnccsCharacterGeneratorQueueHooked) return;
        app._vnccsCharacterGeneratorQueueHooked = true;
        const queuePrompt = app.queuePrompt;
        if (typeof queuePrompt !== "function") return;
        const originalQueuePrompt = (...args) => queuePrompt.apply(app, args);
        app.queuePrompt = async function (...args) {
            for (const node of app.graph?._nodes || []) {
                if (node.mode === 2 || node.mode === 4) continue;
                if (node._vnccsCharacterGeneratorSyncBeforeQueue?.() === false) return;
            }
            return originalQueuePrompt(...args);
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const isBaseGenerator = nodeData.name === "VNCCS_CharacterGenerator";
        const isCloneGenerator = nodeData.name === "VNCCS_CharacterCloneGenerator";
        const isClothesGenerator = nodeData.name === "VNCCS_ClothesGenerator";
        const isEmotionsGenerator = nodeData.name === "VNCCS_EmotionsGenerator";
        if (!isBaseGenerator && !isCloneGenerator && !isClothesGenerator && !isEmotionsGenerator) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.setSize([1180, 760]);
            this._vnccsCharacterGeneratorWidget = new CharacterGeneratorWidget(this, {
                isClone: isCloneGenerator,
                isClothes: isClothesGenerator,
                isEmotions: isEmotionsGenerator,
                title: isCloneGenerator
                    ? "VNCCS Character Clone Generator"
                    : (isClothesGenerator ? "VNCCS Clothes Generator" : (isEmotionsGenerator ? "VNCCS Emotions Generator" : "VNCCS Character Generator")),
            });
            syncDOMWidgetWidthSoon(this, "character_generator_ui");
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            if (this._vnccsCharacterGeneratorWidget) {
                // Loading a saved workflow must preserve its explicit values.
                this._vnccsCharacterGeneratorWidget.qi2EmotionDefaultsPending = false;
                this._vnccsCharacterGeneratorWidget.data = readData(this);
                this._vnccsCharacterGeneratorWidget.syncCharacterSourceData();
                this._vnccsCharacterGeneratorWidget.syncStagesFromData();
                this._vnccsCharacterGeneratorWidget.restoreBrowserState();
                this._vnccsCharacterGeneratorWidget.syncCharacterSourceData();
                this._vnccsCharacterGeneratorWidget.syncStagesFromData();
                this._vnccsCharacterGeneratorWidget.syncModelResolution();
                writeData(this, this._vnccsCharacterGeneratorWidget.data);
                this._vnccsCharacterGeneratorWidget.renderSettings();
                this._vnccsCharacterGeneratorWidget.renderPreview();
                this._vnccsCharacterGeneratorWidget.renderChain();
                if (this._vnccsCharacterGeneratorWidget.restoredViewer?.open && this._vnccsCharacterGeneratorWidget.currentImages().length) {
                    this._vnccsCharacterGeneratorWidget.openViewer(
                        this._vnccsCharacterGeneratorWidget.restoredViewer.index || 0,
                        this._vnccsCharacterGeneratorWidget.restoredViewer,
                    );
                }
            }
            syncDOMWidgetWidthSoon(this, "character_generator_ui");
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (serialized) {
            const widget = this._vnccsCharacterGeneratorWidget;
            if (widget) {
                widget.syncModelResolution();
                writeData(this, widget.data, { notify: false });
            }
            onSerialize?.apply(this, arguments);
            const index = this.widgets?.findIndex(item => item.name === "widget_data") ?? -1;
            if (widget && index >= 0 && Array.isArray(serialized?.widgets_values)) {
                serialized.widgets_values[index] = this.widgets[index].value;
            }
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            syncDOMWidgetWidth(this, "character_generator_ui");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "character_generator_ui"));
        };
    },
});
