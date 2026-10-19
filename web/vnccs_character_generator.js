import { app } from "../../scripts/app.js";
import { createSakuraInteractions, mountSakuraDialogMotion } from "./sakura_interactions.mjs";
import { vnccsApi as api, mediaURL, checkedJSON, storage, workflowScope, cacheIdentity, watchConnection } from "./vnccs_transport.js";
import { SAKURA_THEME_CSS, SAKURA_INTERACTION_CSS, showModal as showCommonModal, registerCleanup, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, resolutionScaleMegapixels, finiteResolutionScaleValue, resolutionScaleText, RESOLUTION_SCALE_MIN_MP, RESOLUTION_SCALE_MAX_MP, RESOLUTION_SCALE_STEP_MP } from "./vnccs_common.js";

const CHARACTER_MANNEQUIN_URL = new URL("./assets/VNCCS_Character_Mannequin.png", import.meta.url).href;
const CHARACTER_MANNEQUIN_MALE_URL = new URL("./assets/VNCCS_Character_Mannequin_Male.png", import.meta.url).href;
const CHARACTER_MANNEQUIN_ASPECT = 2 / 3;
const PREVIEW_LOAD_CONCURRENCY = 2;
const GENERATOR_QWEN_INSTRUCTION = "Describe the character and their key features (body shape, physical characteristics, clothing, items, accessories). Then explain how the user's text instruction should alter or modify the character. Generate a new image that meets the user's requirements while maintaining consistency with the original character where appropriate.";
const QI2_EMOTION_PROMPT_TEMPLATE = "Upscale face image.\nMake character's face emotion {emotion}\nChange only face. Keep original neck colour, clothes and hairs\nkeep character's clothes";
const QI2_EMOTION_BBOX_DEFAULTS = Object.freeze({
    bbox_threshold: 0.3,
    drop_size: 10,
});
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
        feather: 50,
        noise_mask: true,
        force_inpaint: true,
        bbox_threshold: 0.5,
        bbox_dilation: 50,
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
        matte_method: "screen_matte",
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

const BACKGROUND_STAGES = [["source_upscaler", "Source Upscaler"], ["background_generation", "Background Generation"], ["upscaler", "Variant Upscaler"]];

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
${SAKURA_INTERACTION_CSS}
.vnccs-pipe-root {
${SAKURA_THEME_CSS}
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
.vnccs-seedvr-picker-menu { display:flex; flex-direction:column; gap:8px; padding:8px; border:1px solid rgba(255,143,163,.18); border-radius:10px; background:rgba(8,8,12,.48); }
.vnccs-seedvr-picker.is-open .vnccs-seedvr-picker-menu { display:flex; }
.vnccs-seedvr-card { display:flex; flex-direction:column; gap:5px; padding:10px 12px 8px; border:1px solid rgba(0,214,143,.25); border-radius:10px; background:rgba(0,214,143,.05); cursor:default; position:relative; overflow:hidden; transition:all .16s ease; }
.vnccs-seedvr-card.is-picker-head { min-height:58px; cursor:pointer; }
.vnccs-seedvr-card.is-installed { cursor:pointer; }
.vnccs-seedvr-card.is-installed:hover:not(.is-selected), .vnccs-seedvr-card.is-picker-head:hover:not(.is-selected) { border-color:rgba(0,214,143,.42); background:rgba(0,214,143,.08); }
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
    min-height: 0;
    display: grid;
    grid-template-rows: minmax(0, 1fr) 108px;
    overflow: hidden;
}
.vnccs-pipe-root.is-clone .vnccs-pipe-main {
    grid-template-rows: minmax(0, 1fr) 172px;
}
.vnccs-pipe-root.is-clone-sfw .vnccs-pipe-main {
    grid-template-rows: minmax(0, 1fr) 96px;
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
.vnccs-pipe-preview {
    --vnccs-preview-accent: #ff8fa3;
    --vnccs-preview-accent-rgb: 255, 143, 163;
    --vnccs-preview-highlight-rgb: 255, 235, 244;
    --vnccs-preview-label: #ffb6c8;
    --vnccs-preview-idle-border: var(--accent-lavender);
    --vnccs-preview-idle-rgb: 107, 74, 139;
    --vnccs-preview-active-bg-rgb: 119, 57, 104;
    min-height: 0;
    padding: 12px;
    overflow: hidden;
    background: radial-gradient(circle, rgba(var(--vnccs-preview-accent-rgb), 0.04) 1px, transparent 1px), #09090e;
    background-size: 20px 20px, 100% 100%;
    display: flex;
    flex-direction: column;
}
.vnccs-pipe-preview.is-male-character {
    --vnccs-preview-accent: #63c7ff;
    --vnccs-preview-accent-rgb: 99, 199, 255;
    --vnccs-preview-highlight-rgb: 232, 248, 255;
    --vnccs-preview-label: #b9e8ff;
    --vnccs-preview-idle-border: #8bcfff;
    --vnccs-preview-idle-rgb: 49, 126, 181;
    --vnccs-preview-active-bg-rgb: 31, 116, 174;
}
.vnccs-pipe-preview-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    color: #9898a8;
    font-size: 11px;
    flex-shrink: 0;
    height: 34px;
    min-width: 0;
    gap: 8px;
}
.vnccs-pipe-preview-label { flex: 0 1 22%; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vnccs-pipe-preview-head .vnccs-pipe-tabs { flex: 0 1 auto; max-width: 78%; margin-left: auto; min-width: 0; flex-wrap: nowrap; overflow-x: auto; justify-content: flex-start; }
.vnccs-pipe-preview-head .vnccs-pipe-tab { flex-shrink: 0; }
.vnccs-pipe-grid {
    position: relative;
    flex: 1;
    min-height: 0;
    min-width: 0;
    display: grid;
    gap: 8px;
    align-content: center;
    justify-content: center;
    overflow: auto;
    scrollbar-gutter: stable;
}
.vnccs-pipe-img {
    position: relative;
    width: 0;
    height: 0;
    display: block;
    min-height: 0;
    justify-self: center;
    align-self: center;
    appearance: none;
    padding: 0;
    background: #14141e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    box-sizing: border-box;
    cursor: zoom-in;
    opacity: 1;
    overflow: hidden;
    isolation: isolate;
    transform-origin: top left;
    transition: border-color 240ms ease, box-shadow 240ms ease;
}
.vnccs-pipe-img.is-entrance-pending { opacity: 0; }
.vnccs-pipe-img-content { position: absolute; z-index: 1; inset: 4px; width: calc(100% - 8px); height: calc(100% - 8px); object-fit: contain; pointer-events: none; transition: filter 520ms ease, opacity 420ms ease, transform 520ms cubic-bezier(0.22,1,0.36,1); }
.vnccs-pipe-img.is-waiting .vnccs-pipe-img-content { filter: grayscale(1) saturate(0.15) brightness(0.62); opacity: 0.64; transform: scale(0.992); }
.vnccs-pipe-img.is-processing .vnccs-pipe-img-content { filter: none; opacity: 1; transform: scale(1); }
.vnccs-pipe-img-placeholder {
    position: absolute;
    z-index: 1;
    top: 0;
    bottom: 0;
    left: 50%;
    width: auto;
    height: 100%;
    max-width: none;
    object-fit: contain;
    pointer-events: none;
    opacity: 0;
    transform: translateX(-50%);
    filter: saturate(0.8) brightness(0.75);
    transition: opacity 320ms ease, filter 320ms ease, transform 320ms ease;
}
.vnccs-pipe-img.is-placeholder { background: radial-gradient(circle at 50% 40%, rgba(var(--vnccs-preview-idle-rgb), 0.18), #11111a 72%); }
.vnccs-pipe-img.is-placeholder .vnccs-pipe-img-placeholder { opacity: 0.26; }
.vnccs-pipe-img-placeholder-scan {
    position: absolute;
    z-index: 3;
    top: 0;
    left: 0;
    right: 0;
    height: 100%;
    pointer-events: none;
    opacity: 0;
    transform: translateY(-100%);
    mix-blend-mode: screen;
}
.vnccs-pipe-img-placeholder-scan::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    --vnccs-scan-edge-offset: clamp(10px, 4%, 23px);
    bottom: calc(0px - var(--vnccs-scan-edge-offset));
    height: clamp(20px, 8%, 46px);
    background: linear-gradient(180deg, transparent, rgba(var(--vnccs-preview-accent-rgb), 0.22) 30%, rgba(var(--vnccs-preview-highlight-rgb), 0.94) 50%, rgba(var(--vnccs-preview-accent-rgb), 0.28) 68%, transparent);
    box-shadow: 0 0 22px rgba(var(--vnccs-preview-accent-rgb), 0.22);
}
.vnccs-pipe-img.is-placeholder.is-processing .vnccs-pipe-img-placeholder {
    opacity: 0.82;
    transform: translateX(-50%) scale(1.02);
    filter: saturate(1.15) brightness(1.04) drop-shadow(0 0 12px rgba(var(--vnccs-preview-accent-rgb), 0.42));
    animation: vnccs-pipe-mannequin-develop 1800ms ease-in-out infinite;
}
.vnccs-pipe-img.is-placeholder.is-processing {
    background:
        repeating-linear-gradient(180deg, transparent 0 8px, rgba(var(--vnccs-preview-accent-rgb), 0.035) 9px 10px),
        radial-gradient(circle at 50% 40%, rgba(var(--vnccs-preview-active-bg-rgb), 0.34), #11111a 72%);
}
.vnccs-pipe-img.is-processing .vnccs-pipe-img-placeholder-scan {
    opacity: 1;
    will-change: transform, opacity;
    animation: vnccs-pipe-mannequin-scan 2200ms cubic-bezier(0.4,0,0.2,1) infinite;
}
.vnccs-pipe-img.is-clothes-revealing .vnccs-pipe-img-placeholder-scan {
    opacity: 1;
    will-change: transform, opacity;
    animation: none;
}
.vnccs-pipe-img.is-character-revealing .vnccs-pipe-img-placeholder {
    opacity: 0.82;
    filter: saturate(1.05) brightness(0.96) drop-shadow(0 0 10px rgba(var(--vnccs-preview-accent-rgb), 0.32));
}
.vnccs-pipe-img.is-character-revealing .vnccs-pipe-img-placeholder-scan {
    opacity: 1;
    will-change: transform, opacity;
    animation: none;
}
.vnccs-pipe-img.is-placeholder.is-waiting .vnccs-pipe-img-placeholder {
    opacity: 0.2;
    filter: grayscale(1) saturate(0.15) brightness(0.62);
}
.vnccs-pipe-img::before, .vnccs-pipe-img::after { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; z-index: 2; opacity: 0; transition: opacity 320ms ease, transform 320ms ease; }
.vnccs-pipe-img::before { border: 2px solid var(--vnccs-preview-idle-border); }
.vnccs-pipe-img.is-waiting::before { opacity: 0.16; }
.vnccs-pipe-img::after { border: 2px solid var(--vnccs-preview-accent); box-shadow: inset 0 0 18px rgba(var(--vnccs-preview-accent-rgb), 0.12), 0 0 14px rgba(var(--vnccs-preview-accent-rgb), 0.2); }
.vnccs-pipe-img.is-processing { border-color: var(--vnccs-preview-accent); box-shadow: 0 0 12px rgba(var(--vnccs-preview-accent-rgb), 0.16); }
.vnccs-pipe-img.is-processing::after { opacity: 1; animation: vnccs-pipe-card-process 1800ms ease-in-out infinite; }
@keyframes vnccs-pipe-card-process { 0%, 100% { transform: scale(0.996); } 50% { transform: scale(1); } }
@keyframes vnccs-pipe-mannequin-develop { 0%, 100% { transform: translateX(-50%) scale(1); } 50% { transform: translateX(-50%) scale(1.025); } }
@keyframes vnccs-pipe-mannequin-scan { 0% { transform: translateY(-100%); } 72%, 100% { transform: translateY(0); } }
.vnccs-pipe-img.is-empty { visibility: hidden; pointer-events: none; }
.vnccs-pipe-img.is-empty:focus-visible { outline-offset: -3px; }
.vnccs-pipe-img:hover {
    border-color: rgba(var(--vnccs-preview-accent-rgb), 0.45);
}
.vnccs-pipe-img-regen {
    position: absolute;
    z-index: 3;
    right: 7px;
    bottom: 7px;
    border: 1px solid rgba(var(--vnccs-preview-accent-rgb), 0.48);
    background: rgba(14,14,22,0.78);
    color: var(--vnccs-preview-label);
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
    border-color: rgba(var(--vnccs-preview-accent-rgb), 0.82);
    background: rgba(var(--vnccs-preview-accent-rgb), 0.16);
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
.vnccs-pipe-empty[hidden] { display: none; }
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
    min-width: 0;
    min-height: 0;
    box-sizing: border-box;
    overflow: hidden;
    background: #111119;
    border-top: 1px solid rgba(255,143,163,0.14);
}
.vnccs-pipe-chain.is-clone {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    grid-auto-rows: minmax(0, 1fr);
    min-height: 0;
    box-sizing: border-box;
    gap: 6px;
    padding: 6px 10px;
    overflow-y: auto;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    grid-template-rows: 20px 24px 12px 4px;
    align-content: start;
    min-height: 0;
    padding: 4px 8px;
    gap: 2px 8px;
    overflow: hidden;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-name,
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-lora { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-status { overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-name,
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-status,
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-lora {
    grid-column: 1;
    line-height: 1.2;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-status,
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-lora {
    grid-column: 1 / -1;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-lora {
    grid-row: 3;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-progress {
    grid-column: 1 / -1;
    grid-row: 4;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-actions {
    grid-column: 2;
    grid-row: 1;
    align-self: center;
    margin-top: 0;
}
.vnccs-pipe-chain.is-clone-sfw .vnccs-pipe-stage-name {
    padding-right: 82px;
}
.vnccs-pipe-chain.is-clone-sfw .vnccs-pipe-stage-actions {
    top: 6px;
    bottom: auto;
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
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
    overflow: hidden;
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
    width: 100%;
    border-radius: inherit;
    background: #ffc074;
    transform: scaleX(0);
    transform-origin: left center;
    transition: transform 360ms linear;
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
    position: absolute;
    right: 10px;
    bottom: 10px;
    margin-top: 0;
    display: flex;
    justify-content: flex-end;
}
.vnccs-pipe-chain.is-clone .vnccs-pipe-stage-actions {
    position: static;
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
    flex: 0 1 auto;
    min-width: 0;
    flex-wrap: nowrap;
    overflow-x: auto;
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

.vnccs-pipe-root :is(.vnccs-pipe-block, .vnccs-seedvr-card, .vnccs-pipe-stage) {
    border-radius: var(--radius-md);
    transition: border-color 180ms ease, background-color 180ms ease, color 180ms ease, box-shadow 180ms ease;
}
.vnccs-pipe-root :is(.vnccs-pipe-input, .vnccs-pipe-select, .vnccs-common-ui-select-trigger) { height: 34px; min-height: 34px; }
.vnccs-pipe-settings-field .vnccs-common-ui-select-trigger {
    width: 100%; padding: 7px 8px; border: 1px solid var(--border-hover);
    border-radius: var(--radius-md); background: rgba(255,255,255,0.045);
    color: var(--text-primary); font-family: var(--font); font-size: 11px;
}
.vnccs-pipe-settings-group-fields { align-items: start; }
.vnccs-pipe-settings-field { min-width: 0; }
.vnccs-pipe-settings-field:not(.is-check) > .vnccs-pipe-settings-field-label { min-height: 14px; line-height: 14px; }
.vnccs-pipe-settings-field input:not([type="checkbox"]):not([type="range"]) { height: 34px; }
.vnccs-pipe-root :is(button, [role="button"], input, textarea):focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.vnccs-pipe-root :is(.vnccs-pipe-check, .vnccs-pipe-settings-field.is-check) input[type="checkbox"] {
    appearance: none; position: relative; flex: 0 0 34px; width: 34px; height: 18px; margin: 0;
    border: 1px solid var(--border-hover); border-radius: 18px; background: rgba(255,255,255,0.07);
    cursor: pointer; transition: background-color 180ms ease, border-color 180ms ease;
}
.vnccs-pipe-root :is(.vnccs-pipe-check, .vnccs-pipe-settings-field.is-check) input[type="checkbox"]::before {
    content: ""; position: absolute; top: 2px; left: 2px; width: 12px; height: 12px;
    border-radius: 50%; background: var(--text-secondary);
    transition: transform 280ms cubic-bezier(0.22, 1, 0.36, 1), background-color 180ms ease;
}
.vnccs-pipe-root :is(.vnccs-pipe-check, .vnccs-pipe-settings-field.is-check) input[type="checkbox"]:checked {
    border-color: var(--accent-border); background: rgba(255,143,163,0.22);
}
.vnccs-pipe-root :is(.vnccs-pipe-check, .vnccs-pipe-settings-field.is-check) input[type="checkbox"]:checked::before { transform: translateX(16px); background: var(--accent); }
.vnccs-seedvr-card[data-download-status="downloading"], .vnccs-seedvr-card[data-download-status="queued"] { border-color: rgba(184,169,232,0.4); }
.vnccs-seedvr-card[data-download-status="downloading"] .vnccs-seedvr-card-status, .vnccs-seedvr-card[data-download-status="queued"] .vnccs-seedvr-card-status { color: var(--accent-lavender); }
.vnccs-seedvr-download { height: 30px; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; box-sizing: border-box; }
.vnccs-pipe-preview { position: relative; }
.vnccs-pipe-img-label { position:absolute; top:8px; left:8px; right:8px; z-index:3; width:fit-content; max-width:calc(100% - 16px); padding:5px 9px; border-radius:var(--radius-sm); background:rgba(10,10,17,.82); color:var(--text-primary); font:500 11px var(--font); overflow-wrap:anywhere; pointer-events:none; }
.vnccs-pipe-disclosure { display: grid; grid-template-rows: 0fr; opacity: 0; transition: grid-template-rows 320ms cubic-bezier(0.22, 1, 0.36, 1), opacity 180ms ease; }
.vnccs-pipe-disclosure.is-open { grid-template-rows: 1fr; opacity: 1; }
.vnccs-pipe-disclosure-clip { min-height: 0; overflow: hidden; }
.vnccs-pipe-disclosure-content { display: flex; flex-direction: column; gap: 7px; padding-top: 7px; }
.vnccs-pipe-upscaler-block .vnccs-pipe-block-b, .vnccs-seedvr-picker { gap: 0; }
.vnccs-pipe-root .vnccs-common-ui-preview-aura { z-index: 1; }
@media (prefers-reduced-motion: reduce) {
    .vnccs-pipe-root *, .vnccs-pipe-root *::before, .vnccs-pipe-root *::after { animation: none !important; transition: none !important; }
    .vnccs-pipe-img.is-waiting::before { opacity: 0.35; }
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
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Generator settings must be an object.");
        for (const [section, defaults] of Object.entries(DEFAULT_DATA)) {
            if (typeof defaults !== "object" || Array.isArray(defaults) || !(section in parsed)) continue;
            const value = parsed[section];
            if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${section} settings must be an object.`);
        }
        const data = deepMerge(DEFAULT_DATA, parsed);
        if (
            parsed?.emotion_generation
            && parsed.emotion_generation.use_sam === undefined
            && parsed.emotion_generation.use_sam_model !== undefined
        ) {
            data.emotion_generation.use_sam = booleanValue(parsed.emotion_generation.use_sam_model, false);
        }
        if (String(data.upscaler?.model || "").startsWith("seedvr2_ema_") || String(data.upscaler?.model || "").endsWith(".gguf")) {
            data.upscaler.model = "seedvr2_3b_fp8_e4m3fn.safetensors";
        }
        if (!SEEDVR_COLOR_CORRECTION_MODES.includes(data.upscaler?.color_correction)) {
            data.upscaler.color_correction = "lab";
        }
        for (const section of ["common", "pose_generation", "remove_clothes"]) {
            data[section].target_size = finiteResolutionScaleValue(resolutionScaleMegapixels(data[section].target_size));
        }
        normalizeUpscalerSettings(data);
        delete node._vnccsGeneratorStateError;
        return data;
    } catch (error) {
        node._vnccsGeneratorStateError = `Unable to restore generator settings: ${error.message}. The saved data is preserved. Reload a valid workflow before editing or running.`;
        return JSON.parse(JSON.stringify(DEFAULT_DATA));
    }
}

function writeData(node, data, { notify = true, trackChange = false } = {}) {
    if (node._vnccsGeneratorStateError) return;
    const widget = node.widgets?.find(w => w.name === "widget_data");
    if (!widget) return;
    normalizeUpscalerSettings(data);
    const value = JSON.stringify(data);
    // DOM clicks run after ComfyUI's mouseup snapshot; explicitly track user edits.
    const canvas = trackChange ? app.canvas : null;
    canvas?.emitEvent?.({ subType: "before-change" });
    try {
        widget.value = value;
        if (notify) widget.callback?.(widget.value);
        app.graph?.setDirtyCanvas(true, true);
    } finally {
        canvas?.emitEvent?.({ subType: "after-change" });
    }
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
        this.isBackground = Boolean(options.isBackground);
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
        this.fieldDrafts = new Map();
        this.previewLayoutFrame = null;
        this.previewCardEntranceFrame = null;
        this.previewLoadQueue = [];
        this.previewLoadsActive = 0;
        this._clothesRevealIndices = new Set();
        this._originalRevealIndices = new Set();
        this._nakedRevealIndices = new Set();
        this._pendingAutoPreviewStage = null;
        this.regenerateState = null;
        this.regenerateTimer = null;
        this.restoreBrowserState();
        this.build();
        this.bindEvents();
        const refreshTimer = setTimeout(() => this.refreshProgress().catch(error => console.warn("[VNCCS] Progress refresh failed", error)), 0);
        registerCleanup(this.node, () => clearTimeout(refreshTimer));
        this.loadNodeDefs();
        this.loadSeedvrAssets();
    }

    build() {
        injectStyles();
        const root = document.createElement("div");
        root.className = "vnccs-pipe-root";
        root.classList.add("vnccs-common-ui-host");
        this.root = root;
        this._ui = createSakuraInteractions(root, this.node);
        registerCleanup(this.node, () => this._ui.dispose());
        enableMiddleMouseCanvasPan(root, this.node);
        attachHelpTooltips(root);
        this.updateModeClasses();

        this.settingsEl = document.createElement("div");
        this.settingsEl.className = "vnccs-pipe-settings";
        this.settingsEl.dataset.sakuraScroll = "settings";
        this.previewEl = document.createElement("div");
        this.previewEl.className = "vnccs-pipe-preview";
        this.chainEl = document.createElement("div");
        this.chainEl.className = "vnccs-pipe-chain";
        this.chainEl.dataset.sakuraScroll = "stages";

        const main = document.createElement("div");
        main.className = "vnccs-pipe-main";
        main.append(this.previewEl, this.chainEl);
        this.settingsButton = document.createElement("button");
        this.settingsButton.type = "button";
        this.settingsButton.className = "vnccs-pipe-settings-open";
        const settingsIcon = document.createElement("span");
        settingsIcon.className = "vnccs-pipe-settings-open-icon";
        settingsIcon.setAttribute("aria-hidden", "true");
        settingsIcon.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m9 3-.8 3-2.7 1.6-3-.8-1.5 2.6 2.2 2.2v3.2L1 17l1.5 2.6 3-.8L8.2 20l.8 3h3l.8-3 2.7-1.6 3 .8 1.5-2.6-2.2-2.2v-3.2L20 9l-1.5-2.6-3 .8L12.8 6 12 3z"/><circle cx="10.5" cy="13" r="3"/></svg>';
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
            dataWidget.options ||= {};
            dataWidget.options.dynamicPrompts = false;
            dataWidget.computeSize = () => [0, -4];
        }

        this.syncCharacterSourceData();
        this.syncStagesFromData();
        this.syncModelResolution();
        this.node._vnccsCharacterGeneratorSyncBeforeQueue = () => {
            if (this.node._vnccsGeneratorStateError) {
                this.showModal("Settings restore failed", this.node._vnccsGeneratorStateError);
                return false;
            }
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
        registerCleanup(this.node, () => clearInterval(this.seedvrPollTimer));
        registerCleanup(this.node, () => {
            this.closeViewer();
            this.viewer = null;
            clearTimeout(this._saveBrowserStateTimer);
            this.previewLoadQueue = [];
            this.previewLoadsActive = 0;
            for (const tile of this.previewGrid?.children || []) {
                this.finishCharacterPreviewTransition(tile);
                this.finishClothesPreviewTransition(tile);
            }
            cancelAnimationFrame(this.previewLayoutFrame);
            this.previewLayoutFrame = null;
            cancelAnimationFrame(this.previewCardEntranceFrame);
            this.previewCardEntranceFrame = null;
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
            if (this._disposed) return;
            const detail = event.detail || {};
            if (this.isBackground && detail.background_id && detail.background_id !== this.data.background_id) return;
            if (String(detail.node_id) !== String(this.node.id)) return;
            if (this._sourceCharacter === "" || (detail.character_name != null && detail.character_name !== (this.data.character_name || ""))) return;
            const scope = this.progressScope();
            if (detail.scope && detail.scope !== scope) return;
            if (!this.acceptsProgressRequest(detail.request_id)) return;
            if (detail.scope) {
                if (this._progressScope === scope && this._progressEpoch === detail.epoch && detail.revision <= (this._progressRevision || 0)) return;
                this._progressScope = scope;
                this._progressRevision = detail.revision;
                this._progressEpoch = detail.epoch;
            }
            const stage = detail.stage;
            if (!this.stageState[stage] && stage !== "error") return;
            const status = detail.status || "waiting";
            const startsNewRun = Boolean(detail.run_id)
                && detail.run_id !== this._activeProgressRunId
                && status === "running"
                && stage === this.stages[0]?.[0]
                && Math.max(0, Number(detail.current) || 0) === 0;
            if (startsNewRun) {
                this._previewPlaceholderStage = stage;
                this._pendingAutoPreviewStage = null;
                this._clothesRevealIndices?.clear();
                this._originalRevealIndices?.clear();
                this._nakedRevealIndices?.clear();
            }
            this.observeProgressRequest(detail.request_id, detail.run_id);
            if (stage === "error") {
                this.applyProgressError(detail.message);
            } else {
                const previousStageState = this.stageState[stage] || {};
                const hasImages = Object.prototype.hasOwnProperty.call(detail, "images");
                if (detail.run_id && status === "running" && Math.max(0, Number(detail.current) || 0) === 0
                    && stage.endsWith("pose_generation") && !detail.images?.length && !previousStageState.images?.length) {
                    this._previewPlaceholderStage = stage;
                }
                const targetedRegenerate = Number.isInteger(this.regenerateState?.imageIndex)
                    && this.regenerateState?.targetStages?.includes(stage);
                if (status === "running") {
                    const continuingBatch = targetedRegenerate || (
                        previousStageState.status === "running"
                        && (Boolean(detail.append_images) || Boolean(detail.replace_images) || !hasImages)
                    );
                    if (!continuingBatch) {
                        this.resetStagesFrom(stage);
                        if (this.isClothes && stage === "pose_generation") this._clothesRevealIndices?.clear();
                        if (this.isClone && stage === "original_pose_generation") this._originalRevealIndices?.clear();
                        if (this.isClone && stage === "naked_pose_generation") this._nakedRevealIndices?.clear();
                    }
                    if (!continuingBatch && (stage === "pose_generation" || stage === "original_pose_generation" || stage === "source_upscaler")) {
                        this.userSelectedPreview = false;
                        if (!this.data.ui) this.data.ui = {};
                        this.data.ui.user_selected_preview = false;
                    }
                }
                let nextImages = startsNewRun ? null : (previousStageState.images || null);
                if (hasImages && detail.replace_images) {
                    nextImages = [...(previousStageState.images || [])];
                    const previewStart = Math.max(0, Number.parseInt(detail.preview_start, 10) || 0);
                    for (const [offset, image] of (detail.images || []).entries()) {
                        nextImages[previewStart + offset] = image;
                    }
                } else if (hasImages && detail.append_images) {
                    nextImages = [...(previousStageState.images || []), ...(detail.images || [])];
                } else if (hasImages && targetedRegenerate && detail.images?.length === 1 && previousStageState.images?.length > 1) {
                    nextImages = [...previousStageState.images];
                    nextImages[this.regenerateState.imageIndex] = detail.images[0];
                } else if (hasImages && !(status === "running" && !startsNewRun && !detail.images?.length && previousStageState.images?.length)) {
                    nextImages = detail.images;
                }
                if (this.isClothes && stage === "pose_generation" && detail.replace_images && detail.images?.length
                    && (!this.userSelectedPreview || this.selectedPreview === stage)) {
                    const previewStart = Math.max(0, Number.parseInt(detail.preview_start, 10) || 0);
                    for (let offset = 0; offset < detail.images.length; offset++) {
                        (this._clothesRevealIndices ||= new Set()).add(previewStart + offset);
                    }
                }
                if (this.isClone && stage === "original_pose_generation" && detail.replace_images && detail.images?.length
                    && (!this.userSelectedPreview || this.selectedPreview === stage)) {
                    const previewStart = Math.max(0, Number.parseInt(detail.preview_start, 10) || 0);
                    for (let offset = 0; offset < detail.images.length; offset++) {
                        (this._originalRevealIndices ||= new Set()).add(previewStart + offset);
                    }
                }
                if (this.isClone && stage === "naked_pose_generation" && detail.replace_images && detail.images?.length
                    && (!this.userSelectedPreview || this.selectedPreview === stage)) {
                    const previewStart = Math.max(0, Number.parseInt(detail.preview_start, 10) || 0);
                    for (let offset = 0; offset < detail.images.length; offset++) {
                        (this._nakedRevealIndices ||= new Set()).add(previewStart + offset);
                    }
                }
                this.stageState[stage] = {
                    status,
                    images: nextImages,
                    message: detail.message || "",
                    current: detail.current,
                    total: detail.total,
                    processing_index: detail.processing_index,
                    processing_indices: detail.processing_indices,
                    preview_aspect: detail.preview_aspect ?? previousStageState.preview_aspect,
                    labels: detail.labels ?? previousStageState.labels,
                };
                if (!this.userSelectedPreview && (status === "running" || status === "done" || detail.images)) {
                    if (stage !== this.selectedPreview && this.hasPendingPreviewReveal()) {
                        // The backend can start the next stage before the browser has decoded
                        // the last preview. Keep this stage visible until its wipe completes.
                        this._pendingAutoPreviewStage = stage;
                    } else {
                        this._pendingAutoPreviewStage = null;
                        this.selectedPreview = stage;
                        this.persistUI();
                    }
                }
                this.updateRegenerateProgress(stage, status);
            }
            if (this.viewer?.open) this.syncViewerImage();
            this.renderPreview();
            this.renderChain();
            this.saveBrowserState();
        };
        this.onExecutionInterrupted = () => {
            if (this._disposed) return;
            const regenerateStage = this.regenerateState?.activeStage;
            let changed = false;
            for (const [key, state] of Object.entries(this.stageState)) {
                if (state.status !== "running" && key !== regenerateStage) continue;
                this.stageState[key] = { ...state, status: "waiting", message: "Stopped" };
                changed = true;
            }
            if (!changed && !this.regenerateState) return;
            this._interruptedProgressRunId = this._activeProgressRunId;
            this._pendingAutoPreviewStage = null;
            this.finishRegenerate({ render: false });
            if (this.viewer?.open) this.syncViewerImage();
            this.renderPreview();
            this.renderChain();
            this.saveBrowserState();
        };
        api.addEventListener("vnccs.character_generator.stage", this.onStage);
        api.addEventListener("execution_interrupted", this.onExecutionInterrupted);
        registerCleanup(this.node, () => api.removeEventListener("vnccs.character_generator.stage", this.onStage));
        registerCleanup(this.node, () => api.removeEventListener("execution_interrupted", this.onExecutionInterrupted));
        watchConnection(this.node, () => this.refreshProgress(), registerCleanup);
        const timer = setInterval(() => {
            if (document.visibilityState !== "hidden") {
                this.refreshProgress().catch(error => console.warn("[VNCCS] Progress refresh failed", error));
            }
        }, 10000);
        registerCleanup(this.node, () => {
            this._disposed = true;
            clearInterval(timer);
            this.finishRegenerate({ render: false });
            this._configuredRegenerateRequestId = null;
        });

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
                const sourceChanged = this.syncCharacterSourceData();
                const modelChanged = this.syncModelResolution();
                if (!sourceChanged && !modelChanged) return;
                writeData(this.node, this.data);
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
            const elapsed = Math.floor((Date.now() - this.regenerateState.startedAt) / 1000);
            if (elapsed === this.regenerateState.elapsed) return;
            this.regenerateState.elapsed = elapsed;
            this.renderChain();
        }, 1000);
    }

    finishRegenerate({ render = true } = {}) {
        clearInterval(this.regenerateTimer);
        this.regenerateTimer = null;
        this.regenerateState = null;
        this._regenerateRequestPending = false;
        if (render) { this.renderPreview(); this.renderChain(); }
    }

    updateRegenerateProgress(stage, status, { render = true } = {}) {
        if (!this.regenerateState) return;
        this.regenerateState.sawStageEvent = true;
        if (this.regenerateState.targetStages.includes(stage) && status === "running") {
            this.regenerateState.activeStage = stage;
        }
        const lastStage = this.regenerateState.targetStages[this.regenerateState.targetStages.length - 1];
        if (stage === lastStage && status === "done") {
            this.finishRegenerate({ render });
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
        writeData(this.node, this.data, { notify: false, trackChange: true });
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

        if (this.isBackground) return groups.map(group => {
            if (group.fields[0]?.section !== "pose_generation") return { ...group, title: group.title.replace("Pose Generation", "Background Generation") };
            return { title: "Background Generation", fields: [resolutionScale("pose_generation")] };
        });

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
                draft[field.section][field.key] = finiteResolutionScaleValue(input.value);
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
                if (!text || !Number.isFinite(value) || (field.step >= 1 && !Number.isInteger(value))) return;
                draft[field.section][field.key] = Math.max(field.min, Math.min(field.max, value));
            };
            if (field.type === "number") {
                input.onblur = () => { input.value = String(draft[field.section][field.key] ?? ""); };
            }
            wrap.append(caption, input);
        }
        input.dataset.focusKey = `advanced-${field.section}-${field.key}`;
        input.ariaLabel = field.label;
        this.protectNativeControl(input);
        return wrap;
    }

    openGeneratorSettings() {
        this.closeModal();
        const data = this.data;
        const draft = JSON.parse(JSON.stringify(data));
        const backdrop = document.createElement("div");
        backdrop.className = "vnccs-pipe-modal-backdrop";
        const modal = document.createElement("div");
        modal.className = "vnccs-pipe-modal is-settings";
        const heading = document.createElement("div");
        heading.className = "vnccs-pipe-modal-title";
        heading.textContent = `${this.title} · Settings`;
        const body = document.createElement("div");
        body.className = "vnccs-pipe-modal-body";
        body.dataset.sakuraScroll = "advanced-settings";
        const intro = document.createElement("p");
        intro.className = "vnccs-pipe-settings-intro";
        intro.textContent = "All processing controls are grouped by the internal node that receives them. Connected MODEL, CLIP, VAE, image and conditioning inputs remain managed by the generator.";
        const groupsEl = document.createElement("div");
        groupsEl.className = "vnccs-pipe-settings-groups";
        const groups = this.generatorSettingsGroups();
        const renderGroups = () => {
            const view = this._ui?.captureView();
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
            this._ui?.syncSelects();
            if (view) this._ui.restoreView(view);
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
            if (this._disposed || this.modalEl !== backdrop || this.data !== data || this.node._vnccsGeneratorStateError) return;
            for (const group of groups) {
                for (const field of group.fields) {
                    this.data[field.section] ||= {};
                    this.data[field.section][field.key] = draft[field.section][field.key];
                }
            }
            this.syncModelResolution();
            writeData(this.node, this.data, { trackChange: true });
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
        if (this._ui) {
            this._ui.syncSelects();
            for (const control of this.root.querySelectorAll(".vnccs-common-ui-select-trigger")) this.protectNativeControl(control);
        }
        if (this._ui) mountSakuraDialogMotion(this._ui, backdrop, modal);
        requestAnimationFrame(() => modal.querySelector('input:not([hidden]), textarea, .vnccs-common-ui-select-trigger')?.focus({ preventScroll: true }));
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
        const dialog = showCommonModal(this.root, title || "Message", () => {
            const body = document.createElement("div");
            body.className = "vnccs-pipe-modal-body";
            body.textContent = message || "";
            return body;
        }, [{ text: "OK", autofocus: true, action: () => this.closeModal() }]);
        dialog.overlay.className = "vnccs-pipe-modal-backdrop";
        dialog.modal.className = "vnccs-pipe-modal";
        dialog.title.className = "vnccs-pipe-modal-title";
        dialog.actions.className = "vnccs-pipe-modal-actions";
        dialog.buttons[0].className = "vnccs-pipe-modal-btn";
        dialog.overlay.onclick = event => {
            if (event.target === dialog.overlay) this.closeModal();
        };
        this.modalEl = dialog.overlay;
    }

    validateNativeSeedvr(showModal = false) {
        if (!this.isBackground && (this.data.upscaler?.mode || "seedvr") !== "seedvr") return true;
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
        if (this._disposed) return;
        if (this.node._vnccsGeneratorStateError) {
            this.showModal("Settings restore failed", this.node._vnccsGeneratorStateError);
            return;
        }
        if (!this.stages.some(([key]) => key === stageKey)) return;
        if (this.isBackground && imageIndex === 0) { stageKey = "source_upscaler"; imageIndex = null; }
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
        this.data.ui.progress_scope = this.progressScope();
        const requestId = cacheIdentity();
        const requestData = this.data;
        this._configuredRegenerateRequestId = requestId;
        this.data.ui.progress_request_id = requestId;
        this.data.ui.selected_preview = stageKey;
        this.data.ui.user_selected_preview = false;
        this.resetStagesFrom(stageKey, { preserveImages: Number.isInteger(imageIndex) });
        this.startRegenerate(stageKey, imageIndex);
        this.regenerateState.requestId = requestId;
        writeData(this.node, this.data);
        this.renderPreview();
        this.renderChain();
        try {
            this._regenerateRequestPending = requestId;
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
            if (!this._disposed && this.data === requestData && this._regenerateRequestPending === requestId) this.finishRegenerate();
        } catch (error) {
            if (this._disposed || this.data !== requestData || this._regenerateRequestPending !== requestId) return;
            const hadStageEvent = Boolean(this.regenerateState?.sawStageEvent);
            this.finishRegenerate();
            if (!hadStageEvent) {
                beforeRegenerate.ui = { ...beforeRegenerate.ui, progress_scope: this.progressScope(),
                    progress_request_id: this.data.ui.progress_request_id };
                this.restoreStageSnapshot(beforeRegenerate);
            }
            throw error;
        } finally {
            if (!this._disposed && this.data === requestData) {
                if (this._regenerateRequestPending === requestId) this._regenerateRequestPending = false;
                if (this.data.ui.progress_request_id === requestId && this.data.regenerate_from === stageKey) {
                    delete this.data.regenerate_from;
                    delete this.data.regenerate_index;
                    writeData(this.node, this.data);
                }
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
                settings.target_size = finiteResolutionScaleValue(resolutionScaleMegapixels(savedSize));
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

    syncBackgroundSourceData() {
        const graph = this.node.graph || app.graph, pending = [this.node], visited = new Set();
        while (pending.length) {
            const source = pending.shift();
            if (!source || visited.has(source)) continue;
            visited.add(source);
            if ((source.type || source.comfyClass) === "VNCCS_BackgroundCreator") {
                try {
                    const state = JSON.parse(source.widgets?.find(widget => widget.name === "background_data")?.value || "{}");
                    const names = state.editor?.variants?.map(card => card.name) || ["Morning", "Evening", "Night"];
                    const imageId = state.selected_image_id || "";
                    const changed = this.data.background_id !== (state.background_id || "") || this.data.background_source_image_id !== imageId
                        || JSON.stringify(this.data.background_variants) !== JSON.stringify(names);
                    if (changed && this.stageState) {
                        this.stageState = {}; this.finishRegenerate({ render: false }); this.closeViewer();
                        this.selectedPreview = "source_upscaler"; this.userSelectedPreview = false;
                    }
                    this.data.background_id = state.background_id || ""; this.data.background_source_image_id = imageId;
                    this.data.background_variants = names;
                    return changed;
                } catch { return false; }
            }
            for (const input of source.inputs || []) {
                const link = input.link == null ? null : graph?.links?.[input.link];
                const upstream = graph?.getNodeById?.(link?.origin_id);
                if (upstream) pending.push(upstream);
            }
        }
        return false;
    }

    syncCharacterSourceData() {
        if (this.isBackground) return this.syncBackgroundSourceData();
        if (this.isEmotions) return this.syncEmotionStudioSourceData();
        const matchesType = (node, type, displayName = "") => {
            const title = typeof node?.getTitle === "function" ? node.getTitle() : node?.title;
            return node?.type === type || node?.comfyClass === type || node?.constructor?.type === type || title === displayName;
        };
        const sourceType = this.isClone ? "CharacterCloner" : (this.isClothes ? "ClothesDesigner" : "CharacterCreatorV2");
        const displayName = this.isClone ? "VNCCS Character Cloner" : (this.isClothes ? "VNCCS Clothes Designer" : "VNCCS Character Creator V2");
        const graph = this.node.graph || app.graph;
        const pending = [this.node];
        const visited = new Set();
        let source = null;
        while (pending.length) {
            const candidate = pending.shift();
            if (!candidate || visited.has(candidate)) continue;
            visited.add(candidate);
            if (matchesType(candidate, sourceType, displayName)
                || (this.isClone && matchesType(candidate, "CharacterCreatorV2", "VNCCS Character Creator V2"))) {
                source = candidate;
                break;
            }
            for (const input of candidate.inputs || []) {
                const link = input.link == null ? null : graph?.links?.[input.link];
                const upstream = graph?.getNodeById?.(link?.origin_id);
                if (upstream) pending.push(upstream);
            }
        }
        const widget = source?.widgets?.find(w => w.name === "widget_data");
        const liveState = this.isClone ? source?._vnccsGetClonerState?.() : null;
        if (!widget?.value && !liveState) return;
        let changed = false;
        try {
            const payload = liveState || JSON.parse(widget.value);
            const gender = payload?.gender ?? payload?.sex ?? payload?.character_info?.gender ?? payload?.character_info?.sex;
            if (["male", "female"].includes(String(gender || "").trim().toLowerCase())) {
                const normalizedGender = String(gender).trim().toLowerCase();
                if (this.data.gender !== normalizedGender) {
                    this.data.gender = normalizedGender;
                    changed = true;
                }
            }
            if (payload && Object.prototype.hasOwnProperty.call(payload, "character")) {
                changed = this.syncSourceCharacter(payload.character || "") || changed;
            }
            if (this.isClothes && payload?.costume !== undefined && this.data.costume !== payload.costume) {
                this.data.costume = payload.costume;
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

    syncSourceCharacter(character) {
        this._sourceCharacter = character;
        if (this.data.character_name === character) return false;
        this.data.character_name = character;
        if (this.stageState) {
            this.resetStagesFrom(this.stages[0]?.[0]);
            this.finishRegenerate({ render: false });
            this.closeViewer();
            this.viewer = this.viewerFocus = this.restoredViewer = null;
            this._previewPlaceholderStage = this._pendingAutoPreviewStage = null;
            if (this.previewEl && this.chainEl) {
                this.renderPreview();
                this.renderChain();
            }
        }
        return true;
    }

    mannequinURL() {
        if (this.isBackground) return "";
        const gender = this.data?.gender ?? this.data?.sex
            ?? this.data?.character_info?.gender ?? this.data?.character_info?.sex;
        return String(gender || "").trim().toLowerCase() === "male"
            ? CHARACTER_MANNEQUIN_MALE_URL : CHARACTER_MANNEQUIN_URL;
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
        for (const costume of Array.isArray(costumes) ? costumes : []) {
            for (const emotion of Array.isArray(emotions) ? emotions : []) {
                pairs.push({ costume, emotion });
            }
        }
        const signature = JSON.stringify(pairs);
        let changed = this.syncSourceCharacter(character);
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
        if (this.isBackground) return BACKGROUND_STAGES;
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
        if (this.isBackground) return "source_upscaler";
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
        this.root?.classList.toggle("is-background", this.isBackground);
        this.chainEl?.classList.toggle("is-clone", cloneNsfw);
        this.chainEl?.classList.toggle("is-clone-sfw", this.isClone && !cloneNsfw);
        this.chainEl?.classList.toggle("is-clothes", this.isClothes);
        this.chainEl?.classList.toggle("is-emotions", this.isEmotions);
    }

    progressScope() {
        return workflowScope(this.node);
    }

    prepareQueuedRun() {
        // Normal execution identity comes from server events. A per-submission
        // nonce in widget_data would invalidate ComfyUI's execution cache.
        this.data.ui ||= {};
        this.data.ui.progress_scope = this.progressScope();
        delete this.data.ui.progress_request_id;
        delete this.data.ui.progress_request_kind;
        delete this.data.regenerate_from;
        delete this.data.regenerate_index;
        writeData(this.node, this.data);
    }

    acceptsProgressRequest(requestId) {
        // Reconfiguration owns a new draft, even if it restored the old request ID.
        if (requestId && this._configuredRegenerateRequestId !== undefined && requestId !== this._configuredRegenerateRequestId) return false;
        // Only an outstanding Regenerate needs a temporary request filter.
        // A completed request must not hide normal jobs already in the queue.
        const pending = this.regenerateState?.requestId || this._regenerateRequestPending;
        return !pending || requestId === pending;
    }

    observeProgressRequest(requestId, runId) {
        this._activeProgressRequestId = requestId;
        this._activeProgressRunId = runId;
    }

    progressViewKey() {
        return JSON.stringify([
            this.stages.map(([key]) => {
                const stage = this.stageState[key] || {};
                return [key, stage.status || "waiting", stage.images || null, stage.message || "", stage.current ?? null, stage.total ?? null, stage.processing_index ?? null, stage.processing_indices ?? null, stage.preview_aspect ?? null, stage.labels ?? null];
            }),
            this.selectedPreview,
            this.regenerateState ? [this.regenerateState.from, this.regenerateState.activeStage, this.regenerateState.imageIndex] : null,
        ]);
    }

    applyProgressError(message, failedStage = null, { render = true } = {}) {
        let targets = this.stages.map(([key]) => key).filter(key => this.stageState[key]?.status === "running");
        if (!targets.length) {
            const lastDone = this.stages.map(([key]) => key).reverse().find(key => this.stageState[key]?.status === "done");
            const fallback = this.stages.some(([key]) => key === failedStage) ? failedStage : lastDone || this.stages[0]?.[0];
            if (fallback) targets = [fallback];
        }
        for (const key of targets) this.stageState[key] = {
            ...this.stageState[key], status: "error", message: message || "Generation failed. Check the server log.",
        };
        if (!this.userSelectedPreview && targets.length && this.selectedPreview !== targets[targets.length - 1]) {
            this.selectedPreview = targets[targets.length - 1];
            this.persistUI();
        }
        this.finishRegenerate({ render });
    }

    async refreshProgress() {
        const scope = this.progressScope();
        if (!scope || this._progressPending || this._disposed || this._sourceCharacter === "") return;
        this._progressPending = true;
        const requestData = this.data;
        const characterName = this.data.character_name;
        const requestId = this.data.ui?.progress_request_id;
        const epoch = this._progressEpoch;
        const revision = this._progressRevision;
        try {
            const { snapshot } = await checkedJSON(`/vnccs/character_generator/progress?scope=${encodeURIComponent(scope)}`);
            if (this._disposed || this.data !== requestData || characterName !== this.data.character_name || this._sourceCharacter === "" || scope !== this.progressScope() || requestId !== this.data.ui?.progress_request_id) return;
            if (this._progressEpoch !== epoch && snapshot?.epoch !== this._progressEpoch) return;
            if (!snapshot) {
                if (this._regenerateRequestPending || revision !== this._progressRevision) return;
                let changed = false;
                for (const state of Object.values(this.stageState)) {
                    if (state.status === "running") {
                        state.status = "error";
                        state.message = "Server progress is unavailable. Check the queue before retrying.";
                        changed = true;
                    }
                }
                if (this.regenerateState) this.finishRegenerate();
                else if (changed) { this.renderPreview(); this.renderChain(); }
                if (changed) this.saveBrowserState();
                return;
            }
            if (snapshot.scope !== scope || String(snapshot.node_id) !== String(this.node.id)) return;
            if (this.isBackground && snapshot.background_id && snapshot.background_id !== this.data.background_id) return;
            if (snapshot.character_name != null && snapshot.character_name !== (this.data.character_name || "")) return;
            if (!this.acceptsProgressRequest(snapshot.request_id)) return;
            if (snapshot.run_id && snapshot.run_id === this._interruptedProgressRunId) return;
            if (this._progressScope === scope && this._progressEpoch === snapshot.epoch && snapshot.revision < (this._progressRevision || 0)) return;
            const previousView = this.progressViewKey();
            this._progressScope = scope;
            this._progressRevision = snapshot.revision;
            this._progressEpoch = snapshot.epoch;
            const firstStageKey = this.stages[0]?.[0];
            const firstIncoming = snapshot.stages[firstStageKey];
            const startsNewRun = Boolean(snapshot.run_id)
                && snapshot.run_id !== this._activeProgressRunId
                && firstIncoming?.status === "running"
                && Math.max(0, Number(firstIncoming.current) || 0) === 0;
            if (startsNewRun) {
                this._previewPlaceholderStage = firstStageKey;
            }
            this.observeProgressRequest(snapshot.request_id, snapshot.run_id);
            let followedStage = null;
            for (const [key] of this.stages) {
                const incoming = snapshot.stages[key] || { status: "waiting", images: null, message: "" };
                const previousImages = this.stageState[key]?.images;
                this.stageState[key] = !startsNewRun && incoming.status === "running" && !incoming.images?.length && previousImages?.length
                    ? { ...incoming, images: previousImages }
                    : incoming;
                if (snapshot.run_id && incoming.status === "running" && key.endsWith("pose_generation")
                    && !incoming.images?.length && !previousImages?.length) this._previewPlaceholderStage = key;
                if (["running", "done"].includes(this.stageState[key].status)) followedStage = key;
                this.updateRegenerateProgress(key, this.stageState[key].status, { render: false });
            }
            if (!snapshot.error && !this.userSelectedPreview && followedStage && this.selectedPreview !== followedStage) {
                this.selectedPreview = followedStage;
                this.persistUI();
            }
            if (snapshot.error) this.applyProgressError(snapshot.error.message, snapshot.error.stage, { render: false });
            else if (Object.values(this.stageState).some(stage => stage.status === "error")) this.finishRegenerate({ render: false });
            if (previousView === this.progressViewKey()) return;
            if (this.viewer?.open) this.syncViewerImage();
            this.renderPreview();
            this.renderChain();
            if (!this.viewer?.open && this.restoredViewer?.open && this.stageState[this.selectedPreview]?.images?.some(Boolean)) {
                this.openViewer(this.restoredViewer.index || 0, this.restoredViewer);
            }
            this.saveBrowserState();
        } finally { this._progressPending = false; }
    }

    storageKey() {
        const scope = workflowScope(this.node);
        return scope ? `character-generator:${scope}` : null;
    }

    restoreBrowserState() {
        if (!this.storageKey()) return;
        let saved = null;
        try {
            saved = JSON.parse(storage.getItem(this.storageKey()) || "null");
        } catch {
            saved = null;
        }
        if (!saved || saved.version !== 2) return;

        // Workflow settings are authoritative; browser data is only a matching UI cache.
        if (saved.settings !== JSON.stringify(this.data)) return;
        if (this.stages.some(([key]) => key === saved.selectedPreview)) {
            this.selectedPreview = saved.selectedPreview;
        }
        this.userSelectedPreview = Boolean(saved.userSelectedPreview);

        // Transient results belong to the current server's progress snapshot.
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
    }

    saveBrowserState() {
        if (!this.storageKey()) return;
        this.syncCharacterSourceData();
        this.syncStagesFromData();
        const payload = {
            version: 2,
            settings: JSON.stringify(this.data),
            selectedPreview: this.selectedPreview,
            userSelectedPreview: this.userSelectedPreview,
            viewer: this.serializableViewerState(),
        };
        storage.setItem(this.storageKey(), JSON.stringify(payload));
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
        if (this._disposed) return;
        const names = [
            "VNCCS_QWEN_Encoder",
            "TextEncodeQwenImage21",
            "QwenImage21Cache",
            "KSampler",
            "VAEDecodeTiled",
            "UNETLoader",
            "VAELoader",
            ...NATIVE_SEEDVR_NODE_NAMES,
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
                    if (this._disposed) return;
                    this.nodeDefs[name] = data?.[name];
                }
            } catch {
                // Keep static defaults when an optional internal node is unavailable.
            }
        }));
        if (this._disposed) return;
        if (names.some(name => !this.nodeDefs[name])) {
            try {
                const r = await api.fetchApi("/object_info");
                if (r.ok) allNodeDefs = await r.json();
            } catch {
                allNodeDefs = null;
            }
        }
        if (this._disposed) return;
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
        if (!this._disposed) this.renderSettings();
    }

    async loadSeedvrAttentionInfo() {
        if (this._disposed) return;
        try {
            const r = await api.fetchApi("/vnccs/character_generator/seedvr_attention");
            if (!r.ok) return;
            const data = await r.json();
            if (this._disposed) return;
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
            if (this._disposed) return;
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
            btn.className = "vnccs-pipe-mode-tab" + (this.data[section][key] === value ? " is-active" : "");
            btn.textContent = label;
            btn.dataset.focusKey = `mode-${section}-${key}-${value}`;
            btn.onclick = () => {
                this.set(section, key, value);
                if (section === "upscaler" && key === "mode" && this.upscalerPanel?.isConnected) {
                    for (let index = 0; index < options.length; index++) {
                        wrap.children[index].classList.toggle("is-active", this.data[section][key] === options[index][0]);
                    }
                    this._ui?.syncSelection(wrap);
                    this.setDisclosureOpen(this.upscalerPanel, this.isBackground || this.data.upscaler.mode !== "off");
                } else this.renderSettings();
            };
            wrap.appendChild(btn);
        }
        this._ui?.syncSelection(wrap);
        return wrap;
    }

    disclosure(content, open) {
        const panel = document.createElement("div");
        panel.className = "vnccs-pipe-disclosure";
        const clip = document.createElement("div");
        clip.className = "vnccs-pipe-disclosure-clip";
        clip.appendChild(content);
        panel.appendChild(clip);
        this.setDisclosureOpen(panel, open);
        return panel;
    }

    setDisclosureOpen(panel, open) {
        panel.classList.toggle("is-open", open);
        panel.inert = !open;
        panel.ariaHidden = String(!open);
        const trigger = this._ui?._dropdown?.trigger;
        if (!open && trigger && panel.contains(trigger)) this._ui._closeDropdown(false);
        if (open && this._ui) {
            this._ui.syncSelects();
            for (const control of panel.querySelectorAll(".vnccs-common-ui-select-trigger")) this.protectNativeControl(control);
        }
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
                input.disabled = true;
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
            input.checked = booleanValue(this.data[section][key], false);
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
            input.dataset.focusKey = `field-${section}-${key}`;
            input.ariaLabel = label;
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
        input.dataset.focusKey = `field-${section}-${key}`;
        input.ariaLabel = label;
        input.oninput = () => {
            let raw = input.value;
            if (type === "number") {
                if (!input.value.trim()) return;
                raw = Number(input.value);
                if (!Number.isFinite(raw) || (options?.step >= 1 && !Number.isInteger(raw))) return;
                if (options?.min !== undefined) raw = Math.max(options.min, raw);
                if (options?.max !== undefined) raw = Math.min(options.max, raw);
            }
            if (section === "upscaler" && key === "attention_mode") {
                this.data.upscaler.attention_mode_manual = true;
            }
            this.set(section, key, raw);
        };
        if (type === "select") input.onchange = input.oninput;
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
            const targetSize = finiteResolutionScaleValue(slider.value);
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
            if (!normalized || !Number.isFinite(raw) || (step >= 1 && !Number.isInteger(raw))) {
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
        const graph = this.node.graph || app.graph;
        const link = graph?.links?.[pipeInput.link];
        const pending = [graph?.getNodeById?.(link?.origin_id)];
        const visited = new Set();
        while (pending.length) {
            const source = pending.shift();
            if (!source || visited.has(source)) continue;
            visited.add(source);
            if (source.type === "EmotionGeneratorV2" || source.comfyClass === "EmotionGeneratorV2") return source;
            for (const input of source.inputs || []) {
                const upstreamLink = input.link == null ? null : graph?.links?.[input.link];
                const upstream = graph?.getNodeById?.(upstreamLink?.origin_id);
                if (upstream) pending.push(upstream);
            }
        }
        return null;
    }

    connectedEmotionStudioIsAnima() {
        return this.connectedEmotionStudioMode() === "anima";
    }

    shouldShowEmotionDenoiseControl() {
        return this.connectedEmotionStudioMode() !== "qi2";
    }

    async loadSeedvrAssets(force = false) {
        if (this._disposed) return;
        try {
            const response = await api.fetchApi(`/vnccs/character_generator/seedvr_models${force ? "?refresh=1" : ""}`);
            const config = await response.json();
            if (this._disposed) return;
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
        if (this._disposed) return;
        const key = `${category}:${entry.name}`;
        this.seedvrDownloads[key] = { status: "queued", message: "Queued" };
        this.renderSettings();
        try {
            const response = await api.fetchApi("/vnccs/character_generator/seedvr_download", {
                method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                body: JSON.stringify({ category, name: entry.name }),
            });
            const payload = await response.json();
            if (this._disposed) return;
            if (!response.ok || payload.error) throw new Error(payload.error || "Download failed");
        } catch (error) {
            if (this._disposed) return;
            this.seedvrDownloads[key] = { status: "error", message: error.message || "Download failed" };
            this.renderSettings();
            return;
        }
        if (!this.seedvrPollTimer) this.seedvrPollTimer = setInterval(async () => {
            if (this._disposed || this._seedvrPollPending) return;
            this._seedvrPollPending = true;
            try {
                const statusResponse = await api.fetchApi("/vnccs/character_generator/seedvr_download_status");
                if (!statusResponse.ok) throw new Error(`HTTP ${statusResponse.status}`);
                const statuses = await statusResponse.json();
                if (this._disposed) return;
                this.seedvrDownloads = statuses;
                const active = Object.values(this.seedvrDownloads).some(item => ["queued", "downloading"].includes(item?.status));
                if (!active) {
                    clearInterval(this.seedvrPollTimer);
                    this.seedvrPollTimer = null;
                    await this.loadSeedvrAssets(true);
                } else this.renderSettings();
            } catch (error) {
                console.warn("[VNCCS] SeedVR2 download status unavailable", error);
            } finally { this._seedvrPollPending = false; }
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
        card.dataset.focusKey = `seedvr-${pickerHead ? "head" : rel}`;
        card.dataset.seedvrModel = rel;
        if (pickerHead) { card.dataset.sakuraPickerHead = ""; card.dataset.sakuraFocusDefault = ""; }
        if (pickerHead || installed) {
            card.role = "button";
            card.tabIndex = 0;
            card.ariaLabel = entry.name;
            card.setAttribute("aria-pressed", String(selected));
            card.onkeydown = event => {
                if (event.target === card && ["Enter", " "].includes(event.key)) { event.preventDefault(); card.click(); }
            };
        }
        const displayStatus = status || "missing";
        card.dataset.downloadStatus = displayStatus;
        card.ariaBusy = String(status === "queued" || status === "downloading");
        card.innerHTML = `<div class="vnccs-seedvr-card-head"><span class="vnccs-seedvr-card-dot"></span><span class="vnccs-seedvr-card-name" title="${entry.name}">${entry.name}</span><span class="vnccs-seedvr-card-status">${displayStatus}</span></div><div class="vnccs-seedvr-card-desc">${entry.description || ""}</div>`;
        if (pickerHead) {
            card.onclick = () => {
                this.seedvrModelPickerOpen = !this.seedvrModelPickerOpen;
                if (this.seedvrPickerPanel?.isConnected) this.setDisclosureOpen(this.seedvrPickerPanel, this.seedvrModelPickerOpen);
                else this.renderSettings();
            };
        } else if (installed) {
            card.onclick = () => {
                this.set("upscaler", "model", rel);
                this.seedvrModelPickerOpen = false;
                if (this.seedvrPickerPanel?.isConnected) {
                    this.setDisclosureOpen(this.seedvrPickerPanel, false);
                    const picker = this.seedvrPickerPanel.parentElement;
                    const head = this.buildSeedvrCard(entry, { pickerHead: true });
                    picker.replaceChild(head, picker.children[0]);
                    for (const option of this.seedvrPickerPanel.querySelectorAll(".vnccs-seedvr-card")) {
                        option.classList.toggle("is-selected", option.dataset.seedvrModel === rel);
                        option.setAttribute("aria-pressed", String(option.dataset.seedvrModel === rel));
                    }
                    this._ui?._playMotion(head, [{ opacity: 0.35 }, { opacity: 1 }], 180);
                    head.focus({ preventScroll: true });
                } else this.renderSettings();
            };
        } else {
            const button = document.createElement("button");
            button.className = "vnccs-seedvr-download";
            button.textContent = status === "error" ? (download.message || "Retry") : (["queued", "downloading"].includes(status) ? (download.message || "Downloading…") : "Install / Download");
            button.ariaLabel = button.textContent;
            setHelpText(button, button.textContent);
            setHelpText(card, button.textContent);
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
        picker.dataset.sakuraPicker = "seedvr";
        picker.dataset.mode = "seedvr";
        picker.dataset.sakuraFocusGroup = "seedvr";
        picker.appendChild(this.buildSeedvrCard(selected, { pickerHead: true }));
        const menu = document.createElement("div");
        menu.className = "vnccs-seedvr-picker-menu";
        menu.dataset.sakuraScroll = "seedvr-models";
        entries.forEach(entry => menu.appendChild(this.buildSeedvrCard(entry)));
        this.seedvrPickerPanel = this.disclosure(menu, this.seedvrModelPickerOpen);
        picker.appendChild(this.seedvrPickerPanel);
        container.appendChild(picker);
        return container;
    }

    renderSettings() {
        const view = this._ui?.captureView();
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
                    <div class="vnccs-pipe-empty vnccs-pipe-character" style="min-height:auto;padding:8px;"></div>
                    <div class="vnccs-pipe-label">steps</div>
                    <div class="vnccs-pipe-empty" style="min-height:auto;padding:8px;">${count} costume / emotion pair(s)</div>
                </div>`;
            info.querySelector(".vnccs-pipe-character").textContent = this.data.character_name || "Select in Emotion Studio";
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
            this.settingsEl.appendChild(this.block(this.isBackground ? "Background Generation" : "Pose Generation", [
                this.resolutionScaleSlider("pose_generation"),
            ]));
        }
        const upscalerFields = [
            this.modeTabs("upscaler", "mode", [["seedvr", "SeedVR"], ["off", "OFF"]]),
        ];
        if (this.isBackground) {
            const note = document.createElement("div"); note.className = "vnccs-pipe-settings-note";
            note.textContent = "The original is always upscaled first. This switch controls the final upscale of new variants.";
            upscalerFields.unshift(note);
        }
        {
            const content = document.createElement("div");
            content.className = "vnccs-pipe-disclosure-content";
            const resolutionFields = document.createElement("div");
            resolutionFields.className = "vnccs-pipe-field-row";
            resolutionFields.append(
                this.field("upscaler", "resolution", "target short edge", "number", { min: 16, max: 16384, step: 2 }),
                this.field("upscaler", "max_resolution", "maximum edge", "number", { min: 0, max: 16384, step: 2 }),
            );
            content.append(
                this.seedvrModelCards(),
                resolutionFields,
                this.field("upscaler", "color_correction", "color correction", "select", this.getOptions("SeedVR2PostProcessing", "color_correction_method", SEEDVR_COLOR_CORRECTION_MODES, this.data.upscaler.color_correction)),
            );
            this.upscalerPanel = this.disclosure(content, this.isBackground || this.data.upscaler.mode !== "off");
            upscalerFields.push(this.upscalerPanel);
        }
        const upscalerBlock = this.block("Upscaler", upscalerFields);
        upscalerBlock.classList.add("vnccs-pipe-upscaler-block");
        this.settingsEl.appendChild(upscalerBlock);
        if (!this.isBackground) this.settingsEl.appendChild(this.block("BG Remove", this.bgRemoveFields()));
        if (this._ui) {
            this._ui.syncSelects();
            for (const control of this.root.querySelectorAll(".vnccs-common-ui-select-trigger")) this.protectNativeControl(control);
        }
        if (view) this._ui.restoreView(view);
    }

    renderPreview() {
        this.syncStagesFromData();
        if (!this.previewGrid || this.previewGrid.parentNode !== this.previewEl) {
            this.previewEl.innerHTML = "";
            this._previewStage = this._previewTabsKey = null;
            const head = document.createElement("div");
            head.className = "vnccs-pipe-preview-head";
            this.previewLabel = document.createElement("div");
            this.previewLabel.className = "vnccs-pipe-preview-label";
            this.previewTabs = document.createElement("div");
            this.previewTabs.className = "vnccs-pipe-tabs";
            this.previewTabs.dataset.sakuraScroll = "preview-tabs";
            head.append(this.previewLabel, this.previewTabs);
            this.previewEmpty = document.createElement("div");
            this.previewEmpty.className = "vnccs-pipe-empty";
            this.previewGrid = document.createElement("div");
            this.previewGrid.className = "vnccs-pipe-grid";
            this.previewGrid.dataset.sakuraScroll = "preview-gallery";
            this.previewEl.append(head, this.previewEmpty, this.previewGrid);
        }
        const name = this.stages.find(([key]) => key === this.selectedPreview)?.[1] || "Results";
        const changedStage = this._previewStage !== this.selectedPreview;
        if (changedStage) {
            this._previewRevealAt = 0;
            if (this.selectedPreview !== "pose_generation") this._clothesRevealIndices?.clear();
            if (this.selectedPreview !== "original_pose_generation") this._originalRevealIndices?.clear();
            if (this.selectedPreview !== "naked_pose_generation") this._nakedRevealIndices?.clear();
            this.previewLabel.textContent = name;
            this._ui?._playMotion(this.previewLabel, [{ opacity: 0.25 }, { opacity: 1 }], 180);
        }
        this._previewStage = this.selectedPreview;
        const tabsKey = JSON.stringify(this.stages);
        if (this._previewTabsKey !== tabsKey) {
            const view = this._ui?.captureView();
            this.previewTabs.innerHTML = "";
            for (const [key, title] of this.stages) {
                const tab = document.createElement("button");
                tab.className = "vnccs-pipe-tab";
                tab.textContent = title;
                tab.dataset.focusKey = `preview-${key}`;
                tab.onclick = () => {
                    this._pendingAutoPreviewStage = null;
                    this.selectedPreview = key;
                    this.userSelectedPreview = true;
                    this.persistUI();
                    this.renderPreview();
                };
                this.previewTabs.appendChild(tab);
            }
            this._previewTabsKey = tabsKey;
            if (view) this._ui.restoreView(view);
        }
        const previous = this._ui?.captureSelection([...this.previewTabs.children]);
        for (const tab of this.previewTabs.children) {
            const selected = tab.dataset.focusKey === `preview-${this.selectedPreview}`;
            tab.classList.toggle("is-selected", selected);
            tab.setAttribute("aria-pressed", String(selected));
        }
        if (changedStage) this._ui?.animateSelection(previous, [...this.previewTabs.children]);

        const state = this.stageState[this.selectedPreview] || {};
        const images = state.images || [];
        const stageIndex = this.stages.findIndex(([key]) => key === this.selectedPreview);
        const originalPoseImages = this.isClone && this.selectedPreview === "naked_pose_generation"
            ? this.stageState.original_pose_generation?.images : null;
        const upstream = this.isClothes && this.selectedPreview === "pose_generation" ? []
            : originalPoseImages?.length ? originalPoseImages : this.stages.slice(0, stageIndex).reverse()
                .map(([key]) => this.stageState[key]?.images).find(images => images?.length) || [];
        const total = Math.max(0, Math.min(512, Number(state.total) || 0));
        const backgroundCount = this.isBackground && state.status === "waiting" ? (this.selectedPreview === "source_upscaler" ? 1 : (this.data.background_variants?.length ?? 3) + 1) : 0;
        const count = Math.min(512, Math.max(images.length, total, backgroundCount, state.status === "done" ? 0 : upstream.length));
        const busy = state.status === "running" || this.regenerateState?.activeStage === this.selectedPreview;
        const missingPosePreviews = this.selectedPreview.endsWith("pose_generation")
            && images.filter(Boolean).length < count;
        const showPlaceholders = !this.isBackground && !this.isEmotions && (busy || state.status === "waiting") && !upstream.some(Boolean)
            && (missingPosePreviews || this._previewPlaceholderStage === this.selectedPreview);
        const mannequinURL = this.mannequinURL();
        this.previewEl.classList.toggle("is-male-character", mannequinURL === CHARACTER_MANNEQUIN_MALE_URL);
        const targeted = Number.isInteger(this.regenerateState?.imageIndex) && this.regenerateState?.targetStages?.includes(this.selectedPreview);
        const activeIndex = targeted ? this.regenerateState.imageIndex
            : (Number.isInteger(state.processing_index) ? state.processing_index : Math.max(0, Number(state.current) || 0));
        const completed = state.processing_indices?.length
            ? Math.min(...state.processing_indices) : Math.max(0, Number(state.current) || 0);
        const statePreviewAspect = Number(state.preview_aspect);
        const fixedPreviewAspect = Number.isFinite(statePreviewAspect) && statePreviewAspect > 0
            ? statePreviewAspect : null;
        this.previewEmpty.hidden = count > 0;
        this.previewEmpty.textContent = this.formatStageStatus(this.selectedPreview);
        this.previewGrid.style.display = count ? "grid" : "none";
        while (this.previewGrid.children.length > count) {
            const tile = this.previewGrid.children[this.previewGrid.children.length - 1];
            if (tile._pendingImage) {
                const pending = tile._pendingImage;
                tile._pendingImage = null;
                pending.onload = pending.onerror = null;
                this.finishPreviewImageLoad(pending);
                pending.removeAttribute("src");
            }
            this.finishCharacterPreviewTransition(tile);
            this.finishClothesPreviewTransition(tile);
            tile.remove();
        }
        while (this.previewGrid.children.length < count) {
            const index = this.previewGrid.children.length;
            const tile = document.createElement("div");
            tile.className = "vnccs-pipe-img is-empty";
            tile.role = "button";
            tile.dataset.focusKey = `preview-image-${index}`;
            tile.style.setProperty("--wait-phase", `${-index * 180}ms`);
            tile._previewEntrancePending = true;
            const placeholder = document.createElement("img");
            placeholder.className = "vnccs-pipe-img-placeholder";
            if (mannequinURL) placeholder.src = mannequinURL;
            placeholder.alt = "";
            placeholder.draggable = false;
            const scan = document.createElement("span");
            scan.className = "vnccs-pipe-img-placeholder-scan";
            scan.ariaHidden = "true";
            if (!this.isBackground) tile.appendChild(placeholder);
            tile.appendChild(scan);
            tile.onclick = () => { if (tile.dataset.src) this.openViewer(index); };
            tile.onkeydown = event => {
                if (event.target === tile && ["Enter", " "].includes(event.key)) {
                    event.preventDefault();
                    tile.onclick();
                }
            };
            this.previewGrid.appendChild(tile);
        }
        [...this.previewGrid.children].forEach((tile, index) => {
            // The backend knows the batch dimensions before individual images decode. Keep
            // that ratio authoritative so each completed card cannot reflow its neighbours.
            if (fixedPreviewAspect) tile._previewAspect = fixedPreviewAspect;
            const placeholder = tile.querySelector(".vnccs-pipe-img-placeholder");
            if (placeholder && placeholder.src !== mannequinURL) placeholder.src = mannequinURL;
            const source = (targeted && images.length === 1
                ? (index === activeIndex ? images[0] : tile._previewSource || upstream[index])
                : images[index] || (!showPlaceholders ? upstream[index] || (busy ? tile._previewSource : null) : null));
            if (source) {
                if (this.isEmotions && tile._previewEntrancePending && !tile._previewImage) {
                    tile.classList.add("is-entrance-pending");
                }
                this.updatePreviewImage(tile, source);
            }
            else {
                this.finishCharacterPreviewTransition(tile);
                this.finishClothesPreviewTransition(tile);
                if (tile._pendingImage) {
                    const pending = tile._pendingImage;
                    tile._pendingImage = null;
                    pending.onload = pending.onerror = null;
                    this.finishPreviewImageLoad(pending);
                    pending.removeAttribute("src");
                    pending.remove();
                }
                tile._previewImage?.remove();
                tile._pendingImage = tile._previewImage = tile._previewSource = null;
                delete tile.dataset.src;
                tile._previewAspect = Number(state.preview_aspect) || tile._previewAspect || CHARACTER_MANNEQUIN_ASPECT;
            }
            const processing = busy && (Array.isArray(state.processing_indices) && !targeted
                ? state.processing_indices.includes(index)
                : index === activeIndex && (targeted || activeIndex < (total || count)));
            const waiting = busy && !targeted && !processing && index >= completed;
            tile.classList.toggle("is-processing", Boolean(processing));
            tile.classList.toggle("is-waiting", Boolean(waiting || (state.status === "waiting" && !this.regenerateState)));
            tile.setAttribute("aria-busy", String(Boolean(processing || waiting)));
            tile.ariaLabel = `${name} image ${index + 1}${processing ? ", processing" : waiting ? ", waiting" : busy && index < completed ? ", processed" : ""}`;
            if (this.isBackground) {
                let label = tile.querySelector(".vnccs-pipe-img-label");
                if (!label) { label = document.createElement("span"); label.className = "vnccs-pipe-img-label"; tile.appendChild(label); }
                label.textContent = state.labels?.[index] || (index === 0 ? "Original" : this.data.background_variants?.[index - 1] || `Variant ${index}`);
                tile.ariaLabel = `${label.textContent}${processing ? ", processing" : waiting ? ", waiting" : ""}`;
            }
            tile.tabIndex = tile.dataset.src ? 0 : -1;
            const displaysPlaceholder = Boolean(showPlaceholders && !tile.dataset.src);
            tile.classList.toggle("is-placeholder", displaysPlaceholder);
            if (placeholder && !tile.classList.contains("is-character-revealing")) {
                placeholder.style.display = displaysPlaceholder ? "" : "none";
                placeholder.style.visibility = displaysPlaceholder ? "visible" : "hidden";
                if (displaysPlaceholder) placeholder.style.opacity = "";
            }
            tile.classList.toggle("is-empty", !tile.dataset.src && !tile._previewSource && !showPlaceholders);
            const canRegenerate = state.status === "done" && !this.regenerateState && Boolean(images[index]);
            let regen = tile.querySelector(".vnccs-pipe-img-regen");
            if (canRegenerate && !regen) {
                regen = document.createElement("button");
                regen.type = "button";
                regen.className = "vnccs-pipe-img-regen";
                regen.textContent = "Regenerate";
                regen.onclick = event => {
                    event.preventDefault();
                    event.stopPropagation();
                    this.regenerateFrom(this.selectedPreview, index).catch(error => {
                        console.error("[VNCCS Character Generator] Image regenerate failed:", error);
                        this.showModal("Regenerate Failed", error?.message || "Regenerate failed");
                    });
                };
                tile.appendChild(regen);
            } else if (!canRegenerate) regen?.remove();
        });
        this.flushPreviewImages();
        this.schedulePreviewGridLayout(this.previewGrid);
        this.schedulePreviewCardEntrances();
    }

    schedulePreviewCardEntrances() {
        if (this.previewCardEntranceFrame) return;
        this.previewCardEntranceFrame = requestAnimationFrame(() => {
            this.previewCardEntranceFrame = null;
            if (!this.previewGrid?.isConnected || this._disposed) return;
            const cards = [...this.previewGrid.children].filter(tile => tile._previewEntrancePending
                && (!this.isEmotions || !tile._previewSource || tile._previewImage));
            cards.forEach((tile, index) => {
                tile._previewEntrancePending = false;
                const motion = this._ui?._playMotion(tile, [
                    { opacity: 0, transform: "translateY(10px) scale(0.97)" },
                    { opacity: 1, transform: "none" },
                ], 380);
                tile.classList.toggle("is-entrance-pending", false);
                motion?.effect?.updateTiming?.({ delay: index * 70, fill: "backwards" });
            });
        });
    }

    queuePreviewImageLoad(tile, image, src) {
        this.previewLoadQueue ||= [];
        this.previewLoadsActive ||= 0;
        this.previewLoadQueue.push({ tile, image, src });
        this.pumpPreviewImageLoads();
    }

    pumpPreviewImageLoads() {
        while (!this._disposed && this.previewLoadsActive < PREVIEW_LOAD_CONCURRENCY && this.previewLoadQueue?.length) {
            const { tile, image, src } = this.previewLoadQueue.shift();
            if (tile._pendingImage !== image || image._previewReady) continue;
            image._previewLoadStarted = true;
            this.previewLoadsActive += 1;
            image.src = src;
        }
    }

    finishPreviewImageLoad(image) {
        if (!image?._previewLoadStarted || image._previewLoadFinished) return;
        image._previewLoadFinished = true;
        this.previewLoadsActive = Math.max(0, (this.previewLoadsActive || 0) - 1);
        this.pumpPreviewImageLoads();
    }

    updatePreviewImage(tile, source) {
        const src = mediaURL(source);
        if (tile._previewSource === src) return;
        this.finishCharacterPreviewTransition(tile);
        this.finishClothesPreviewTransition(tile);
        tile._previewSource = src;
        if (tile._pendingImage) {
            const previousPending = tile._pendingImage;
            tile._pendingImage = null;
            previousPending.onload = previousPending.onerror = null;
            this.finishPreviewImageLoad(previousPending);
            previousPending.removeAttribute("src");
            previousPending.remove();
        }
        const image = document.createElement("img");
        image.className = "vnccs-pipe-img-content";
        image.alt = "";
        image.decoding = "async";
        image.style.opacity = "0";
        tile._pendingImage = image;
        tile.appendChild(image);
        const ready = async () => {
            image.onload = image.onerror = null;
            try { await image.decode?.(); } catch { /* Keep the last successful preview on a failed load. */ }
            if (!this._disposed && tile._pendingImage === image && tile.isConnected) {
                image._previewReady = true;
                this.flushPreviewImages();
            }
            this.finishPreviewImageLoad(image);
        };
        image.onload = ready;
        image.onerror = () => {
            if (tile._pendingImage === image) {
                tile._pendingImage = null;
                tile._previewSource = tile.dataset.src || null;
                image.remove();
                const index = [...(this.previewGrid?.children || [])].indexOf(tile);
                if (index >= 0) {
                    this._clothesRevealIndices?.delete(index);
                    this._originalRevealIndices?.delete(index);
                    this._nakedRevealIndices?.delete(index);
                }
                this.flushPreviewImages();
                this.flushPendingAutoPreviewStage();
            }
            this.finishPreviewImageLoad(image);
        };
        this.queuePreviewImageLoad(tile, image, src);
    }

    finishCharacterPreviewTransition(tile) {
        if (!tile?._characterRevealAnimations?.length && !tile?.classList?.contains("is-character-revealing")) return;
        const placeholder = tile.querySelector?.(".vnccs-pipe-img-placeholder");
        if (tile._previewImage) tile._previewImage.style.clipPath = "";
        if (placeholder) {
            placeholder.style.display = "none";
            placeholder.style.opacity = "0";
            placeholder.style.visibility = "hidden";
            placeholder.style.clipPath = "inset(100% 0 0 0)";
        }
        tile.classList.toggle("is-character-revealing", false);
        for (const animation of tile._characterRevealAnimations || []) {
            animation.onfinish = null;
            animation.cancel?.();
        }
        tile._characterRevealAnimations = null;
        if (placeholder) placeholder.style.clipPath = "";
        this.flushPendingAutoPreviewStage();
    }

    startCharacterPreviewTransition(tile, image, delay = 0) {
        tile.classList.toggle("is-character-revealing", true);
        image.style.opacity = "1";
        image.style.clipPath = "inset(0 0 100% 0)";
        const placeholder = tile.querySelector(".vnccs-pipe-img-placeholder");
        if (placeholder) {
            placeholder.style.display = "";
            placeholder.style.opacity = "";
            placeholder.style.visibility = "visible";
            placeholder.style.clipPath = "inset(0% 0 0 0)";
        }
        const scan = tile.querySelector(".vnccs-pipe-img-placeholder-scan");
        const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
        if (reducedMotion || !image.animate || !placeholder?.animate || !scan?.animate) {
            this.finishCharacterPreviewTransition(tile);
            return;
        }
        const options = { duration: 1100, delay, easing: "cubic-bezier(0.22, 1, 0.36, 1)", fill: "both" };
        const animations = [
            image.animate([{ clipPath: "inset(0 0 100% 0)" }, { clipPath: "inset(0 0 0% 0)" }], options),
            placeholder.animate([{ clipPath: "inset(0% 0 0 0)" }, { clipPath: "inset(100% 0 0 0)" }], options),
            scan.animate([
                { transform: "translateY(-100%)", opacity: 0.72 },
                { transform: "translateY(0)", opacity: 1 },
            ], options),
        ];
        tile._characterRevealAnimations = animations;
        let remaining = animations.length;
        for (const animation of animations) animation.onfinish = () => {
            remaining -= 1;
            if (!remaining && tile._previewImage === image) this.finishCharacterPreviewTransition(tile);
        };
    }

    finishClothesPreviewTransition(tile) {
        if (!tile?._clothesRevealBase && !tile?._clothesRevealAnimations?.length) return;
        for (const animation of tile._clothesRevealAnimations || []) {
            animation.onfinish = null;
            animation.cancel?.();
        }
        tile._clothesRevealAnimations = null;
        const previous = tile._clothesRevealBase;
        tile._clothesRevealBase = null;
        if (previous && previous !== tile._previewImage) previous.remove();
        if (tile._previewImage) {
            tile._previewImage.style.clipPath = "";
            if (tile._previewImage.naturalWidth > 0 && tile._previewImage.naturalHeight > 0) {
                tile._previewAspect = tile._previewImage.naturalWidth / tile._previewImage.naturalHeight;
            }
        }
        tile.classList.toggle("is-clothes-revealing", false);
        if (tile.isConnected) this.schedulePreviewGridLayout(this.previewGrid);
        this.flushPendingAutoPreviewStage();
    }

    hasPendingPreviewReveal() {
        const tiles = [...(this.previewGrid?.children || [])];
        if (tiles.some(tile => tile.classList.contains("is-character-revealing")
            || tile.classList.contains("is-clothes-revealing"))) return true;
        if (this.isClone && this.selectedPreview === "original_pose_generation"
            && this._originalRevealIndices?.size) return true;
        if (this.isClone && this.selectedPreview === "naked_pose_generation"
            && this._nakedRevealIndices?.size) return true;
        if (this.isClothes && this.selectedPreview === "pose_generation"
            && this._clothesRevealIndices?.size) return true;
        return tiles.some(tile => tile._pendingImage && tile.classList.contains("is-placeholder"));
    }

    flushPendingAutoPreviewStage() {
        const stage = this._pendingAutoPreviewStage;
        if (!stage || this.userSelectedPreview || this.hasPendingPreviewReveal()) return;
        this._pendingAutoPreviewStage = null;
        if (!this.stages.some(([key]) => key === stage)) return;
        this.selectedPreview = stage;
        this.persistUI();
        this.renderPreview();
    }

    startClothesPreviewTransition(tile, image, previous) {
        tile._clothesRevealBase = previous;
        tile.classList.toggle("is-clothes-revealing", true);
        image.style.opacity = "1";
        image.style.clipPath = "inset(0 0 100% 0)";
        previous.style.clipPath = "inset(0% 0 0 0)";
        const scan = tile.querySelector(".vnccs-pipe-img-placeholder-scan");
        const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
        if (reducedMotion || !image.animate || !previous.animate || !scan?.animate) {
            this.finishClothesPreviewTransition(tile);
            return;
        }
        const options = { duration: 1350, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" };
        const animations = [
            image.animate([{ clipPath: "inset(0 0 100% 0)" }, { clipPath: "inset(0 0 0% 0)" }], options),
            previous.animate([{ clipPath: "inset(0% 0 0 0)" }, { clipPath: "inset(100% 0 0 0)" }], options),
            scan.animate([
                { transform: "translateY(-100%)", opacity: 1 },
                { transform: "translateY(0)", opacity: 1 },
            ], options),
        ];
        tile._clothesRevealAnimations = animations;
        let remaining = animations.length;
        for (const animation of animations) animation.onfinish = () => {
            remaining -= 1;
            if (!remaining && tile._previewImage === image) this.finishClothesPreviewTransition(tile);
        };
    }

    flushPreviewImages() {
        let changed = false;
        for (const [index, tile] of [...(this.previewGrid?.children || [])].entries()) {
            const image = tile._pendingImage;
            if (!image) continue;
            // A later network response must not overtake an earlier card's reveal.
            if (!image._previewReady) break;
            tile._pendingImage = null;
            changed = true;
            const previous = tile._previewImage;
            const originalReveal = Boolean(this.isClone && this.selectedPreview === "original_pose_generation"
                && this._originalRevealIndices?.delete(index));
            const characterReveal = Boolean(!previous && tile.classList.contains("is-placeholder") && (
                (!this.isClone && !this.isClothes && !this.isEmotions)
                || (this.isClone && this.selectedPreview === "original_pose_generation")
            ));
            const clothesRevealRequested = Boolean(
                (this.isClothes && this.selectedPreview === "pose_generation"
                    && this._clothesRevealIndices?.delete(index))
                || originalReveal
                || (this.isClone && this.selectedPreview === "naked_pose_generation"
                    && this._nakedRevealIndices?.delete(index))
            );
            const clothesReveal = Boolean(previous && clothesRevealRequested);
            tile._previewImage = image;
            const stateAspect = Number(this.stageState?.[this.selectedPreview]?.preview_aspect);
            const hasFixedStageAspect = Number.isFinite(stateAspect) && stateAspect > 0;
            if (!clothesReveal && !hasFixedStageAspect) tile._previewAspect = image.naturalWidth / image.naturalHeight;
            for (const sibling of this.previewGrid?.children || []) {
                if (!sibling.dataset.src) sibling._previewAspect = tile._previewAspect;
            }
            tile.dataset.src = image.src;
            tile.tabIndex = 0;
            tile.classList.toggle("is-empty", false);
            tile.classList.toggle("is-placeholder", false);
            const delay = Math.max(0, (this._previewRevealAt || 0) - Date.now());
            this._previewRevealAt = Date.now() + delay + 100;
            image.style.opacity = "1";
            if (clothesReveal) {
                this.startClothesPreviewTransition(tile, image, previous);
            } else if (characterReveal) {
                this.startCharacterPreviewTransition(tile, image, delay);
            } else {
                const motion = this._ui?._playMotion(image, [{ opacity: 0, filter: "blur(4px)" }, { opacity: 1, filter: "blur(0px)" }], 420);
                motion?.effect?.updateTiming?.({ delay, fill: "backwards" });
            }
            if (previous && !clothesReveal) {
                const exit = this._ui?._playMotion(previous, [{ opacity: 1 }, { opacity: 0 }], 280, () => previous.remove());
                exit?.effect?.updateTiming?.({ delay, fill: "backwards" });
                if (!this._ui) previous.remove();
            }
        }
        if (changed && [...(this.previewGrid?.children || [])].every(tile => tile.dataset.src)) this._previewPlaceholderStage = null;
        if (changed) this.schedulePreviewGridLayout(this.previewGrid);
        if (changed) this.schedulePreviewCardEntrances();
        if (changed && this.viewer?.open) this.syncViewerImage();
    }

    schedulePreviewGridLayout(grid) {
        if (this.previewLayoutFrame) cancelAnimationFrame(this.previewLayoutFrame);
        this.previewLayoutFrame = requestAnimationFrame(() => {
            this.previewLayoutFrame = null;
            this.layoutPreviewGrid(grid);
        });
    }

    layoutPreviewGrid(grid) {
        if (!grid?.isConnected || !grid.children.length) return;
        const gap = 8;
        const availableW = Math.max(1, grid.clientWidth);
        const availableH = Math.max(1, grid.clientHeight);
        const count = grid.children.length;
        const images = [...grid.children].map(tile => tile._clothesRevealBase || tile._previewImage || (tile._pendingImage?._previewReady ? tile._pendingImage : null));
        const known = images.find(image => image?.naturalWidth > 0 && image?.naturalHeight > 0);
        const fixed = Number(this.stageState?.[this.selectedPreview]?.preview_aspect);
        const fallback = known ? known.naturalWidth / known.naturalHeight : this.isBackground ? 16 / 9 : CHARACTER_MANNEQUIN_ASPECT;
        // All cards share one frame, including pending previews and mannequins.
        const aspect = Number.isFinite(fixed) && fixed > 0 ? fixed : fallback;
        const aspects = images.map(() => aspect);
        let best = null;
        for (let cols = 1; cols <= count; cols++) {
            const rows = Math.ceil(count / cols);
            const cellW = (availableW - gap * (cols - 1)) / cols;
            const cellH = (availableH - gap * (rows - 1)) / rows;
            if (cellW <= 0 || cellH <= 0) continue;
            const sizes = aspects.map(aspect => {
                const width = Math.min(cellW, cellH * aspect);
                return { width, height: width / aspect };
            });
            const areas = sizes.map(size => size.width * size.height);
            const score = Math.min(...areas) * 1000000 + areas.reduce((sum, area) => sum + area, 0);
            if (!best || score > best.score) best = {
                cols, rows, cellW, cellH, sizes, score,
                trackW: Math.max(...sizes.map(size => size.width)),
                trackH: Math.max(...sizes.map(size => size.height)),
            };
        }
        if (!best) return;
        grid.style.gridTemplateColumns = `repeat(${best.cols}, ${best.trackW}px)`;
        grid.style.gridTemplateRows = `repeat(${best.rows}, ${best.trackH}px)`;
        const lastRowCount = count % best.cols || best.cols;
        const lastRowOffset = lastRowCount < best.cols ? (best.cols - lastRowCount) * (best.trackW + gap) / 2 : 0;
        [...grid.children].forEach((tile, index) => {
            const size = images[index] || tile._previewSource || tile.classList.contains("is-placeholder")
                ? best.sizes[index] : { width: 0, height: 0 };
            tile.style.width = `${size.width}px`;
            tile.style.height = `${size.height}px`;
            tile.style.left = `${index >= count - lastRowCount ? lastRowOffset : 0}px`;
        });
        // Offsets stay in node coordinates even during reveal motion or canvas zoom.
        const rects = [...grid.children].map(tile => ({ x: tile.offsetLeft, y: tile.offsetTop, ready: Boolean(tile.dataset.src) }));
        [...grid.children].forEach((tile, index) => {
            const previous = tile._previewRect;
            const next = rects[index];
            tile._previewRect = next;
            if (previous?.ready && ["x", "y"].some(key => Math.abs(previous[key] - next[key]) > 1)) {
                this._ui?._playMotion(tile, [{ transform: `translate(${previous.x - next.x}px, ${previous.y - next.y}px)` }, { transform: "none" }], 320);
            }
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
        const view = this._ui?.captureView();
        const fills = new Map([...this.chainEl.querySelectorAll("[data-stage-progress]")].map(fill => [fill.dataset.stageProgress, fill.style.transform]));
        const statuses = new Map([...this.chainEl.querySelectorAll("[data-stage-status]")].map(stage => [stage.dataset.focusKey, stage.dataset.stageStatus]));
        this.syncStagesFromData();
        this.chainEl.innerHTML = "";
        this.updateModeClasses();
        this.chainEl.style.setProperty("--vnccs-stage-count", String(Math.max(1, this.stages.length)));
        for (const [key, name] of this.stages) {
            const stage = document.createElement("div");
            const status = this.stageState[key]?.status || "waiting";
            const isRegeneratingStage = this.regenerateState?.targetStages?.includes(key);
            stage.className = "vnccs-pipe-stage";
            stage.dataset.focusKey = `stage-${key}`;
            stage.dataset.stageStatus = status;
            stage.role = "button";
            stage.tabIndex = 0;
            stage.setAttribute("aria-pressed", String(key === this.selectedPreview));
            stage.onkeydown = event => {
                if (event.target === stage && ["Enter", " "].includes(event.key)) { event.preventDefault(); stage.click(); }
            };
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
            const previousStatus = statuses.get(stage.dataset.focusKey);
            if (previousStatus && previousStatus !== status) this._ui?._playMotion(s, [{ opacity: 0.35 }, { opacity: 1 }], 180);
            if (status === "running" || this.regenerateState?.activeStage === key) {
                const progress = document.createElement("div");
                progress.className = "vnccs-pipe-stage-progress";
                const fill = document.createElement("div");
                fill.className = "vnccs-pipe-stage-progress-fill";
                fill.dataset.stageProgress = key;
                const state = this.stageState[key] || {};
                const current = Number(state.current);
                const total = Number(state.total);
                if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
                    fill.style.transform = `scaleX(${Math.max(0.04, Math.min(1, current / total))})`;
                } else {
                    fill.style.transform = `scaleX(${this.estimateRegenerateProgress() / 100})`;
                }
                const previous = fills.get(key);
                if (previous && previous !== fill.style.transform) this._ui?._playMotion(fill, [{ transform: previous }, { transform: fill.style.transform }], 360);
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
            if (key === "remove_clothes") {
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
        if (view) this._ui.restoreView(view);
    }

    currentImages() {
        return this.previewGrid?.isConnected && this._previewStage === this.selectedPreview
            ? [...this.previewGrid.children].map(tile => tile.dataset.src || "")
            : this.stageState[this.selectedPreview]?.images || [];
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
        const view = this._ui?.captureView();
        const replacing = !!this.viewer?.overlay?.isConnected;
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
        stageTabs.dataset.sakuraScroll = "viewer-stages";
        for (const [key, name] of this.stages) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "vnccs-pipe-viewer-btn" + (key === this.selectedPreview ? " is-selected" : "");
            btn.textContent = name;
            btn.dataset.focusKey = `viewer-stage-${key}`;
            btn.setAttribute("aria-pressed", String(key === this.selectedPreview));
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
        if (!replacing && this._ui) mountSakuraDialogMotion(this._ui, overlay, bar);
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
        img.onload = () => {
            if (this.viewer !== viewer || !img.isConnected) return;
            scheduleFit();
            this._ui?._playMotion(img, [{ opacity: 0, filter: "blur(5px)" }, { opacity: 1, filter: "blur(0px)" }], 420);
        };
        img.onerror = () => img.classList.remove("is-ready");
        img.decoding = "async";
        img.src = mediaURL(this.currentImages()[this.viewer.index] || "");
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
        if (view) this._ui.restoreView(replacing ? view : { ...view, focusKey: null, focusGroup: null, selection: null });
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
        if ((this.viewer.img.getAttribute("src") || "") === src || (src && this.viewer.img.src === src)) return;
        this.viewer.restored = { open: true, ...this.currentViewerFocus() };
        this.viewer.fitApplied = false;
        this.viewer.img.classList.remove("is-ready");
        if (src) this.viewer.img.src = mediaURL(src);
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
            const nodes = (app.graph?._nodes || []).filter(node => node.mode !== 2 && node.mode !== 4);
            for (const node of nodes) {
                if (node._vnccsCharacterGeneratorSyncBeforeQueue?.() === false) return;
            }
            for (const node of nodes) node._vnccsCharacterGeneratorWidget?.prepareQueuedRun();
            return originalQueuePrompt(...args);
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const isBaseGenerator = nodeData.name === "VNCCS_CharacterGenerator";
        const isCloneGenerator = nodeData.name === "VNCCS_CharacterCloneGenerator";
        const isClothesGenerator = nodeData.name === "VNCCS_ClothesGenerator";
        const isEmotionsGenerator = nodeData.name === "VNCCS_EmotionsGenerator";
        const isBackgroundGenerator = nodeData.name === "VNCCS_BackgroundGenerator";
        if (!isBaseGenerator && !isCloneGenerator && !isClothesGenerator && !isEmotionsGenerator && !isBackgroundGenerator) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.setSize([1180, 760]);
            this._vnccsCharacterGeneratorWidget = new CharacterGeneratorWidget(this, {
                isClone: isCloneGenerator,
                isClothes: isClothesGenerator,
                isEmotions: isEmotionsGenerator,
                isBackground: isBackgroundGenerator,
                title: isBackgroundGenerator ? "VNCCS Background Generator" : isCloneGenerator
                    ? "VNCCS Character Clone Generator"
                    : (isClothesGenerator ? "VNCCS Clothes Generator" : (isEmotionsGenerator ? "VNCCS Emotions Generator" : "VNCCS Character Generator")),
            });
            syncDOMWidgetWidthSoon(this, "character_generator_ui");
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            if (this._vnccsCharacterGeneratorWidget) {
                const widget = this._vnccsCharacterGeneratorWidget;
                if (widget._configuredRegenerateRequestId !== undefined) widget._configuredRegenerateRequestId = null;
                widget.finishRegenerate({ render: false });
                // Loading a saved workflow must preserve its explicit values.
                this._vnccsCharacterGeneratorWidget.qi2EmotionDefaultsPending = false;
                this._vnccsCharacterGeneratorWidget.data = readData(this);
                widget.closeViewer();
                widget.viewer = null;
                widget.viewerFocus = null;
                widget.restoredViewer = null;
                widget.stageState = {};
                widget.selectedPreview = widget.data.ui?.selected_preview || widget.defaultPreviewStage();
                widget.userSelectedPreview = Boolean(widget.data.ui?.user_selected_preview);
                if (this._vnccsGeneratorStateError) this._vnccsCharacterGeneratorWidget.showModal("Settings restore failed", this._vnccsGeneratorStateError);
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
