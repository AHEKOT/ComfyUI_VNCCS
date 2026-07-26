import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { registerCleanup, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText } from "./vnccs_common.js";

const GENERATOR_QWEN_INSTRUCTION = "Describe the character and their key features (body shape, physical characteristics, clothing, items, accessories). Then explain how the user's text instruction should alter or modify the character. Generate a new image that meets the user's requirements while maintaining consistency with the original character where appropriate.";

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
        qwen_2511: true,
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
        face_denoise: 0.55,
        use_sam: true,
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
        qwen_2511: true,
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
        model: "seedvr2_ema_3b-Q4_K_M.gguf",
        vae: "ema_vae_fp16.safetensors",
        gan_model: "",
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
        use_sam3_details_recovery: true,
        use_preset_values: true,
        tolerance: 0.20,
        softness: 0.16,
        despill_strength: 0.50,
        edge_width: 3,
        matte_cleanup: 0.20,
        foreground_recover: 0.35,
        edge_decontaminate: 0.70,
        edge_choke: 0.20,
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
    ["emotion_0001", "Emotion"],
    ["emotion_0001_bg_remove", "Emotion BG"],
];

const WORKFLOW_UPSCALER_DIT_MODELS = [
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b-Q4_K_M.gguf",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
];

const WORKFLOW_UPSCALER_VAE_MODELS = [
    "ema_vae_fp16.safetensors",
];

const SEEDVR_ATTENTION_MODES = ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"];
const SEEDVR_COLOR_CORRECTION_MODES = ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"];

const POSE_GENERATION_LORA_LABEL = "VNCCS Pose Studio QIE2511";
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
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    background: #101018;
    border-bottom: 1px solid rgba(255,143,163,0.16);
}
.vnccs-pipe-viewer-spacer {
    flex: 1;
}
.vnccs-pipe-viewer-btn {
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
        return data;
    } catch {
        return JSON.parse(JSON.stringify(DEFAULT_DATA));
    }
}

function writeData(node, data, { notify = true } = {}) {
    const widget = node.widgets?.find(w => w.name === "widget_data");
    if (!widget) return;
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
        this.title = options.title || "VNCCS Character Generator";
        this.data = readData(node);
        this.seedvrAttention = { current: null, available: SEEDVR_ATTENTION_MODES };
        this.ganUpscaleModels = [];
        this.syncCharacterSourceData();
        this.stages = this.currentStages();
        this.stageState = Object.fromEntries(this.stages.map(([key]) => [key, { status: "waiting", images: null, message: "" }]));
        const defaultPreview = this.defaultPreviewStage();
        this.selectedPreview = this.data.ui?.selected_preview || defaultPreview;
        if (!this.stages.some(([key]) => key === this.selectedPreview)) {
            this.selectedPreview = defaultPreview;
        }
        this.userSelectedPreview = Boolean(this.data.ui?.user_selected_preview);
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
    }

    build() {
        injectStyles();
        const root = document.createElement("div");
        root.className = "vnccs-pipe-root";
        this.root = root;
        enableMiddleMouseCanvasPan(root);
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
        writeData(this.node, this.data);
        this.node._vnccsCharacterGeneratorSyncBeforeQueue = () => {
            this.syncCharacterSourceData();
            this.syncStagesFromData();
            writeData(this.node, this.data);
        };
        registerCleanup(this.node, () => delete this.node._vnccsCharacterGeneratorSyncBeforeQueue);

        this.renderSettings();
        this.renderPreview();
        this.renderChain();
        this.previewResizeObserver = new ResizeObserver(() => this.renderPreview());
        this.previewResizeObserver.observe(this.previewEl);
        registerCleanup(this.node, () => this.previewResizeObserver?.disconnect());
        registerCleanup(this.node, () => clearInterval(this.regenerateTimer));
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
                if (status === "running") {
                    this.resetStagesFrom(stage);
                    if (stage === "pose_generation" || stage === "original_pose_generation" || stage === "source_upscaler") {
                        this.userSelectedPreview = false;
                        if (!this.data.ui) this.data.ui = {};
                        this.data.ui.user_selected_preview = false;
                    }
                }
                const previousStageState = this.stageState[stage] || {};
                const hasImages = Object.prototype.hasOwnProperty.call(detail, "images");
                this.stageState[stage] = {
                    status,
                    images: hasImages ? detail.images : (previousStageState.images || null),
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
            this.onEmotionStudioModeChanged = () => this.renderSettings();
            window.addEventListener("vnccs-emotion-studio-generation-mode-changed", this.onEmotionStudioModeChanged);
            registerCleanup(this.node, () => window.removeEventListener("vnccs-emotion-studio-generation-mode-changed", this.onEmotionStudioModeChanged));
        }
    }

    resetStagesFrom(stageKey) {
        const start = this.stages.findIndex(([key]) => key === stageKey);
        if (start < 0) return;
        for (const [key] of this.stages.slice(start)) {
            this.stageState[key] = {
                status: "waiting",
                images: null,
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
        if (!this.data[section] || typeof this.data[section] !== "object") this.data[section] = {};
        this.data[section][key] = value;
        if (section === "bg_remove" && key === "preset") {
            this.data.bg_remove.use_preset_values = true;
        }
        if (this.isClone && section === "common" && key === "target_size") {
            this.data.pose_generation.target_size = value;
            this.data.remove_clothes.target_size = value;
        }
        writeData(this.node, this.data, { notify: false });
        this.saveBrowserState();
    }

    generatorSettingsGroups() {
        const number = (section, key, label, min, max, step = 1, extra = {}) => ({
            section, key, label, type: "number", min, max, step, ...extra,
        });
        const text = (section, key, label, extra = {}) => ({ section, key, label, type: "text", ...extra });
        const check = (section, key, label, extra = {}) => ({ section, key, label, type: "checkbox", wide: true, ...extra });
        const select = (section, key, label, options, extra = {}) => ({
            section, key, label, type: "select", options, ...extra,
        });
        const textarea = (section, key, label, extra = {}) => ({
            section, key, label, type: "textarea", wide: true, ...extra,
        });
        const groups = [];

        if (!this.isEmotions) {
            const poseTargetSection = this.isClone ? "common" : "pose_generation";
            groups.push({
                title: "VNCCS QWEN Encoder · Pose Generation",
                fields: [
                    number(poseTargetSection, "target_size", "target_size", 512, 4096, 8),
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
                    check("pose_generation", "qwen_2511", "qwen_2511"),
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
                    title: "VNCCS QWEN Encoder · Remove Clothes",
                    fields: [
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
                        check("remove_clothes", "qwen_2511", "qwen_2511"),
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
                    select("upscaler", "mode", "mode", ["seedvr", "gan", "off"]),
                    check("upscaler", "inherit_pipe_seed", "Use seed from connected pipe"),
                    number("upscaler", "seed", "seed", 0, Number.MAX_SAFE_INTEGER, 1),
                ],
            });
            groups.push({
                title: "SeedVR2LoadDiTModel",
                fields: [
                    select("upscaler", "model", "model", WORKFLOW_UPSCALER_DIT_MODELS, { nodeName: "SeedVR2LoadDiTModel", inputName: "model" }),
                    select("upscaler", "device", "device", ["cuda:0", "cpu"], { nodeName: "SeedVR2LoadDiTModel", inputName: "device" }),
                    select("upscaler", "offload_device", "offload_device", ["cpu", "cuda:0"], { nodeName: "SeedVR2LoadDiTModel", inputName: "offload_device" }),
                    number("upscaler", "blocks_to_swap", "blocks_to_swap", 0, 1000, 1),
                    check("upscaler", "swap_io_components", "swap_io_components"),
                    check("upscaler", "cache_dit", "cache_model"),
                    select("upscaler", "attention_mode", "attention_mode", this.seedvrAttention.available, { nodeName: "SeedVR2LoadDiTModel", inputName: "attention_mode" }),
                    check("upscaler", "attention_mode_manual", "Keep selected attention mode (disable auto-detect)"),
                ],
            });
            groups.push({
                title: "SeedVR2LoadVAEModel",
                fields: [
                    select("upscaler", "vae", "model", WORKFLOW_UPSCALER_VAE_MODELS, { nodeName: "SeedVR2LoadVAEModel", inputName: "model" }),
                    check("upscaler", "encode_tiled", "encode_tiled"),
                    number("upscaler", "encode_tile_size", "encode_tile_size", 64, 8192, 8),
                    number("upscaler", "encode_tile_overlap", "encode_tile_overlap", 0, 8192, 8),
                    check("upscaler", "decode_tiled", "decode_tiled"),
                    number("upscaler", "decode_tile_size", "decode_tile_size", 64, 8192, 8),
                    number("upscaler", "decode_tile_overlap", "decode_tile_overlap", 0, 8192, 8),
                    select("upscaler", "tile_debug", "tile_debug", ["false", "true"], { nodeName: "SeedVR2LoadVAEModel", inputName: "tile_debug" }),
                    check("upscaler", "cache_vae", "cache_model"),
                ],
                note: "device and offload_device are shared with the DiT loader.",
            });
            groups.push({
                title: "SeedVR2VideoUpscaler",
                fields: [
                    number("upscaler", "resolution", "resolution", 64, 16384, 8),
                    number("upscaler", "max_resolution", "max_resolution", 64, 32768, 8),
                    number("upscaler", "batch_size", "batch_size", 1, 1024, 1),
                    check("upscaler", "uniform_batch_size", "uniform_batch_size"),
                    select("upscaler", "color_correction", "color_correction", SEEDVR_COLOR_CORRECTION_MODES, { nodeName: "SeedVR2VideoUpscaler", inputName: "color_correction" }),
                    number("upscaler", "temporal_overlap", "temporal_overlap", 0, 1024, 1),
                    number("upscaler", "prepend_frames", "prepend_frames", 0, 1024, 1),
                    number("upscaler", "input_noise_scale", "input_noise_scale", 0, 100, 0.001),
                    number("upscaler", "latent_noise_scale", "latent_noise_scale", 0, 100, 0.001),
                    check("upscaler", "enable_debug", "enable_debug"),
                ],
            });
            groups.push({
                title: "UpscaleModelLoader · GAN",
                fields: [
                    select("upscaler", "gan_model", "model_name", this.ganUpscaleModels, { nodeName: "UpscaleModelLoader", inputName: "model_name", wide: true }),
                ],
            });
        } else {
            groups.push({
                title: "UltralyticsDetectorProvider",
                fields: [
                    select("emotion_generation", "bbox_model", "bbox detector model", [], { nodeName: "UltralyticsDetectorProvider", inputName: "model_name", wide: true }),
                    select("emotion_generation", "segm_model", "segmentation detector model", [], { nodeName: "UltralyticsDetectorProvider", inputName: "model_name", wide: true }),
                ],
            });
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
            groups.push({
                title: "VNCCS Emotion Matte Merge",
                fields: [
                    number("emotion_generation", "matte_expand_radius", "matte_expand_radius", 0, 256, 1),
                    number("emotion_generation", "matte_feather_radius", "matte_feather_radius", 0, 256, 1),
                    number("emotion_generation", "chroma_context", "chroma_context", 0, 1024, 1),
                ],
                note: "These parameters affect only the FaceDetailer region. The original sprite alpha remains untouched elsewhere.",
            });
        }

        groups.push({
            title: "VNCCS Chroma Key",
            fields: [
                select("bg_remove", "preset", "preset", ["disabled", "ultra_light", "light", "balanced", "strong", "aggressive"]),
                check("bg_remove", "use_preset_values", "Use values from selected preset"),
                number("bg_remove", "tolerance", "tolerance", 0, 1, 0.01),
                number("bg_remove", "softness", "softness", 0.001, 1, 0.01),
                number("bg_remove", "despill_strength", "despill_strength", 0, 1, 0.01),
                number("bg_remove", "edge_width", "edge_width", 0, 32, 1),
                number("bg_remove", "matte_cleanup", "matte_cleanup", 0, 1, 0.01),
                number("bg_remove", "foreground_recover", "foreground_recover", 0, 1, 0.01),
                number("bg_remove", "edge_decontaminate", "edge_decontaminate", 0, 1, 0.01),
                number("bg_remove", "edge_choke", "edge_choke", 0, 1, 0.01),
                select("bg_remove", "matte_method", "matte_method", ["chroma_soft", "guided_edge", "pymatting_if_available"]),
                select("bg_remove", "screen_mode", "screen_mode", ["from_background", "auto", "green", "blue", "red"]),
                select("bg_remove", "output_mode", "output_mode", ["straight_rgba", "premultiplied_rgba"]),
                check("bg_remove", "use_sam3_details_recovery", "Use SAM3 recovery mask"),
            ],
            note: "When preset values are enabled, the individual chroma parameters are retained but the preset controls processing.",
        });
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
                const value = Number(String(input.value).replace(",", "."));
                if (!Number.isFinite(value)) return;
                draft[field.section][field.key] = Math.max(field.min, Math.min(field.max, value));
            };
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
        modal.onkeydown = event => {
            if (event.key === "Escape") {
                event.preventDefault();
                this.closeModal();
            }
        };
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
        this.resetStagesFrom(stageKey);
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
        const matchesType = (node, type, displayName = "") => {
            const title = typeof node?.getTitle === "function" ? node.getTitle() : node?.title;
            return node?.type === type || node?.comfyClass === type || node?.constructor?.type === type || title === displayName;
        };
        const source = app.graph?._nodes?.find(n => matchesType(n, "EmotionGeneratorV2", "VNCCS Emotion Studio"));
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
            return pairs.flatMap((pair, index) => {
                const key = `emotion_${String(index + 1).padStart(4, "0")}`;
                const label = `${pair.costume || "Costume"} / ${pair.emotion || "Emotion"}`;
                return [
                    [key, label],
                    [`${key}_bg_remove`, `${label} BG`],
                ];
            });
        }
        return this.isClothes ? CLOTHES_STAGES : STAGES;
    }

    defaultPreviewStage() {
        if (this.isClone) return "original_pose_generation";
        if (this.isEmotions) return this.currentStages()[0]?.[0] || "emotion_0001";
        return this.isClothes ? "source_upscaler" : "pose_generation";
    }

    syncStagesFromData() {
        const nextStages = this.currentStages();
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
            this.data = deepMerge(this.data, saved.data);
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
            "KSampler",
            "VAEDecodeTiled",
            "SeedVR2LoadDiTModel",
            "SeedVR2LoadVAEModel",
            "SeedVR2VideoUpscaler",
            "UpscaleModelLoader",
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
        await Promise.all([
            this.loadSeedvrAttentionInfo(),
            this.loadGanUpscaleModels(),
        ]);
        this.renderSettings();
    }

    async loadGanUpscaleModels() {
        try {
            const r = await api.fetchApi("/vnccs/character_generator/gan_upscale_models");
            if (r.ok) {
                const data = await r.json();
                this.ganUpscaleModels = uniqueOptions(Array.isArray(data?.models) ? data.models : []);
            }
        } catch {
            this.ganUpscaleModels = [];
        }
        if (!this.ganUpscaleModels.length) {
            this.ganUpscaleModels = this.getLoaderModelOptions("UpscaleModelLoader", "model_name");
        }
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

    getLoaderModelOptions(nodeName, inputName) {
        const spec = this.getInputSpec(nodeName, inputName);
        return uniqueOptions(Array.isArray(spec?.[0]) ? spec[0] : []);
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
            input.addEventListener(eventName, event => event.stopPropagation(), true);
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
            target_size: "Scales the QWEN encoder latent by total pixel area while preserving the pose aspect ratio.",
            prompt: "Prompt text used for the remove-clothes/preparation stage.",
            gan_model: "Upscale model used when GAN upscaling is selected.",
            model: "SeedVR diffusion model used for the upscaler stage.",
            resolution: "Output resolution target for SeedVR upscaling.",
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
                wrap.addEventListener(eventName, event => event.stopPropagation(), true);
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
            this.protectNativeControl(input);
        }
        input.value = this.data[section][key];
        input.oninput = () => {
            const raw = type === "number" ? Number(input.value) : input.value;
            if (section === "upscaler" && key === "attention_mode") {
                this.data.upscaler.attention_mode_manual = true;
            }
            this.set(section, key, raw);
        };
        wrap.append(caption, input);
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
            if (!Number.isFinite(raw)) {
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

    connectedEmotionStudioIsAnima() {
        if (!this.isEmotions) return false;
        const pipeInput = (this.node.inputs || []).find(input => input.name === "pipe");
        if (!pipeInput?.link) return false;
        const link = app.graph?.links?.[pipeInput.link];
        const sourceNode = app.graph?.getNodeById?.(link?.origin_id);
        if (!sourceNode || sourceNode.type !== "EmotionGeneratorV2") return false;

        const settingsWidget = sourceNode.widgets?.find(widget => widget.name === "generation_settings");
        try {
            const settings = settingsWidget?.value ? JSON.parse(settingsWidget.value) : {};
            const settingsMode = String(settings?.generation_mode || "").toLowerCase();
            if (settingsMode === "anima") return true;
            if (settingsMode === "illustrious") return false;
        } catch (_) {
            // Fall back to the hidden mode widget below.
        }
        const modeWidget = sourceNode.widgets?.find(widget => widget.name === "generation_model");
        return String(modeWidget?.value || "").toLowerCase() === "anima";
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
            this.settingsEl.appendChild(this.block("Emotion Strength", [
                this.faceDenoiseSlider(),
            ]));
            const faceDetailerFields = [
                this.field("emotion_generation", "use_sam", "Use SAM", "checkbox"),
                this.faceDetailerNumberField("bbox_threshold", "bbox_threshold", { min: 0, max: 1, step: 0.01 }),
                this.faceDetailerNumberField("bbox_dilation", "bbox_dilation", { min: 0, max: 128, step: 1 }),
                this.faceDetailerNumberField("sam_dilation", "sam_dilation", { min: 0, max: 128, step: 1 }),
                this.faceDetailerNumberField("sam_threshold", "sam_threshold", { min: 0, max: 1, step: 0.01 }),
                this.faceDetailerNumberField("sam_bbox_expansion", "sam_bbox_expansion", { min: 0, max: 128, step: 1 }),
            ];
            this.settingsEl.appendChild(this.block("Face Detailer", faceDetailerFields));
            this.settingsEl.appendChild(this.block("BG Remove", [
                this.field("bg_remove", "preset", "chroma preset", "select", ["disabled", "ultra_light", "light", "balanced", "strong", "aggressive"]),
                this.field("bg_remove", "use_sam3_details_recovery", "Use SAM3 Details Recovery", "checkbox"),
            ]));
            return;
        }
        if (this.isClone) {
            this.settingsEl.appendChild(this.block("Common", [
                this.field("common", "target_size", "scale area", "select", [1024, 1344, 1536, 2048, 768, 512]),
            ]));
            if (this.isCloneNsfwEnabled()) {
                this.settingsEl.appendChild(this.block("Remove Clothes", [
                    this.field("remove_clothes", "prompt", "prompt", "textarea"),
                ]));
            }
        } else {
            this.settingsEl.appendChild(this.block("Pose Generation", [
                this.field("pose_generation", "target_size", "scale area", "select", [1024, 1344, 1536, 2048, 768, 512]),
            ]));
        }
        const upscalerFields = [
            this.modeTabs("upscaler", "mode", [["seedvr", "SeedVR"], ["gan", "GAN"], ["off", "OFF"]]),
        ];
        if (this.data.upscaler.mode === "gan") {
            const ganOptions = this.syncSelectToOptions(
                "upscaler",
                "gan_model",
                this.ganUpscaleModels,
            );
            upscalerFields.push(
                this.field("upscaler", "gan_model", "model", "select", ganOptions),
            );
        } else if (this.data.upscaler.mode !== "off") {
            upscalerFields.push(
                this.field("upscaler", "model", "dit model", "select", this.getWorkflowModelOptions("SeedVR2LoadDiTModel", "model", WORKFLOW_UPSCALER_DIT_MODELS, this.data.upscaler.model)),
                this.field("upscaler", "resolution", "resolution", "number"),
                this.field("upscaler", "color_correction", "color correction", "select", this.getOptions("SeedVR2VideoUpscaler", "color_correction", SEEDVR_COLOR_CORRECTION_MODES, this.data.upscaler.color_correction)),
                this.field("upscaler", "attention_mode", "attention mode", "select", this.getOptions("SeedVR2LoadDiTModel", "attention_mode", this.seedvrAttention.available, this.data.upscaler.attention_mode)),
            );
        }
        this.settingsEl.appendChild(this.block("Upscaler", upscalerFields));
        this.settingsEl.appendChild(this.block("BG Remove", [
            // TODO: Decide what to do with internal RMBG later.
            this.field("bg_remove", "preset", "chroma preset", "select", ["disabled", "ultra_light", "light", "balanced", "strong", "aggressive"]),
            this.field("bg_remove", "use_sam3_details_recovery", "Use SAM3 Details Recovery", "checkbox"),
        ]));
    }

    renderPreview() {
        this.syncStagesFromData();
        this.previewEl.innerHTML = "";
        const head = document.createElement("div");
        head.className = "vnccs-pipe-preview-head";
        const label = document.createElement("div");
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
                l.textContent = `LoRA: ${POSE_GENERATION_LORA_LABEL}`;
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
    }

    currentImages() {
        return this.stageState[this.selectedPreview]?.images || [];
    }

    openViewer(index = 0, restored = null) {
        const images = this.currentImages();
        if (!images.length) return;
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
        this.closeViewer();
        const overlay = document.createElement("div");
        overlay.className = "vnccs-pipe-viewer";
        const bar = document.createElement("div");
        bar.className = "vnccs-pipe-viewer-bar";
        const back = document.createElement("button");
        back.className = "vnccs-pipe-viewer-btn";
        back.textContent = "BACK";
        back.onclick = () => this.closeViewer(true);
        bar.appendChild(back);
        for (const [key, name] of this.stages) {
            const btn = document.createElement("button");
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
            bar.appendChild(btn);
        }
        const spacer = document.createElement("div");
        spacer.className = "vnccs-pipe-viewer-spacer";
        const zoomOut = document.createElement("button");
        zoomOut.className = "vnccs-pipe-viewer-btn";
        zoomOut.textContent = "-";
        zoomOut.onclick = () => this.zoomViewer(0.8);
        const zoomIn = document.createElement("button");
        zoomIn.className = "vnccs-pipe-viewer-btn";
        zoomIn.textContent = "+";
        zoomIn.onclick = () => this.zoomViewer(1.25);
        bar.append(spacer, zoomOut, zoomIn);

        const canvas = document.createElement("div");
        canvas.className = "vnccs-pipe-viewer-canvas";
        const img = document.createElement("img");
        img.className = "vnccs-pipe-viewer-img";
        canvas.appendChild(img);
        overlay.append(bar, canvas);
        this.root.appendChild(overlay);
        this.viewer.overlay = overlay;
        this.viewer.canvas = canvas;
        this.viewer.img = img;
        this.viewer.fitApplied = false;

        const scheduleFit = () => requestAnimationFrame(() => this.fitViewer());
        img.onload = scheduleFit;
        img.decoding = "async";
        img.src = this.currentImages()[this.viewer.index] || "";
        if (img.complete && img.naturalWidth) scheduleFit();
        canvas.onwheel = (event) => {
            event.preventDefault();
            const factor = event.deltaY < 0 ? 1.12 : 0.88;
            this.zoomViewer(factor, event);
        };
        canvas.onpointerdown = (event) => {
            const point = this.viewerEventPoint(event);
            this.viewer.dragging = true;
            this.viewer.dragX = point.x;
            this.viewer.dragY = point.y;
            canvas.classList.add("is-dragging");
            canvas.setPointerCapture(event.pointerId);
        };
        canvas.onpointermove = (event) => {
            if (!this.viewer?.dragging) return;
            const point = this.viewerEventPoint(event);
            this.viewer.x += point.x - this.viewer.dragX;
            this.viewer.y += point.y - this.viewer.dragY;
            this.viewer.dragX = point.x;
            this.viewer.dragY = point.y;
            this.applyViewerTransform();
            this.updateViewerFocus();
            this.scheduleBrowserStateSave();
        };
        canvas.onpointerup = (event) => {
            if (!this.viewer) return;
            this.viewer.dragging = false;
            canvas.classList.remove("is-dragging");
            canvas.releasePointerCapture(event.pointerId);
            this.updateViewerFocus();
            this.saveBrowserState();
        };
    }

    closeViewer(clear = false) {
        this.viewer?.overlay?.remove();
        if (clear) {
            this.viewer = null;
            this.saveBrowserState();
        }
    }

    syncViewerImage() {
        const images = this.currentImages();
        if (!this.viewer?.img || !images.length) return;
        this.viewer.index = this.clampedViewerIndex();
        this.viewer.img.classList.remove("is-ready");
        this.viewer.img.src = images[this.viewer.index];
    }

    clampedViewerIndex() {
        const images = this.currentImages();
        if (!images.length) return 0;
        return Math.max(0, Math.min(this.viewer?.index ?? 0, images.length - 1));
    }

    fitViewer() {
        if (!this.viewer?.img || !this.viewer?.canvas) return;
        if (this.viewer.fitApplied) return;
        const rect = this.viewerCanvasRect();
        const iw = this.viewer.img.naturalWidth || 1;
        const ih = this.viewer.img.naturalHeight || 1;
        const fit = rect.height / ih;
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
            const centerImageX = Math.max(0, Math.min(1, centerNormX)) * iw;
            const centerImageY = Math.max(0, Math.min(1, centerNormY)) * ih;
            this.viewer.x = rect.width / 2 - centerImageX * this.viewer.scale;
            this.viewer.y = rect.height / 2 - centerImageY * this.viewer.scale;
            this.viewer.restored = null;
            this.restoredViewer = null;
            this.viewerFocus = { scaleRatio, centerNormX, centerNormY };
        } else {
            this.viewer.scale = fit;
            this.centerViewerImage(rect, iw, ih, true);
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
        const originalQueuePrompt = app.queuePrompt?.bind(app);
        if (!originalQueuePrompt) return;
        app.queuePrompt = async function (...args) {
            for (const node of app.graph?._nodes || []) {
                node._vnccsCharacterGeneratorSyncBeforeQueue?.();
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
                this._vnccsCharacterGeneratorWidget.data = readData(this);
                this._vnccsCharacterGeneratorWidget.syncCharacterSourceData();
                this._vnccsCharacterGeneratorWidget.syncStagesFromData();
                this._vnccsCharacterGeneratorWidget.restoreBrowserState();
                this._vnccsCharacterGeneratorWidget.syncCharacterSourceData();
                this._vnccsCharacterGeneratorWidget.syncStagesFromData();
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

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            syncDOMWidgetWidth(this, "character_generator_ui");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "character_generator_ui"));
        };
    },
});
