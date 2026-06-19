import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { registerCleanup, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText } from "./vnccs_common.js";

const DEFAULT_DATA = {
    nsfw_enabled: true,
    emotion_pairs: [],
    common: {
        target_size: 1024,
    },
    pose_generation: {
        target_size: 1024,
    },
    emotion_generation: {
        face_denoise: 0.55,
        anima_lllite_name: "anima-lllite-any-test-like-v2.safetensors",
        anima_lllite_strength: 1.0,
        bbox_threshold: 0.1,
        bbox_dilation: 10,
        sam_dilation: 25,
        sam_threshold: 0.93,
        sam_bbox_expansion: 0,
    },
    remove_clothes: {
        prompt: "Dress character: White underwear",
    },
    upscaler: {
        mode: "seedvr",
        model: "seedvr2_ema_3b-Q4_K_M.gguf",
        vae: "ema_vae_fp16.safetensors",
        gan_model: "2x_APISR_RRDB_GAN_generator.pth",
        device: "cuda:0",
        offload_device: "cpu",
        seed: 42,
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
    },
    ui: {
        selected_preview: "pose_generation",
        user_selected_preview: false,
    },
};

const ANIMA_LLLITE_OPTIONS = [
    "anima-lllite-any-test-like-v2.safetensors",
    "anima-lllite-inpainting-v2.safetensors",
];

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

const WORKFLOW_GAN_UPSCALER_MODELS = [
    "2x_APISR_RRDB_GAN_generator.pth",
    "4x_APISR_GRL_GAN_generator.pth",
];

const SEEDVR_ATTENTION_MODES = ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"];

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
    padding: 10px;
    overflow-y: auto;
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
        return deepMerge(DEFAULT_DATA, JSON.parse(widget?.value || "{}"));
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
        root.append(this.settingsEl, main);

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
                this.stageState[stage] = {
                    status,
                    images: Object.prototype.hasOwnProperty.call(detail, "images")
                        ? detail.images
                        : (status === "running" ? null : this.stageState[stage]?.images || null),
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
        writeData(this.node, this.data, { notify: false });
        this.saveBrowserState();
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
            return pairs.length
                ? pairs.map((pair, index) => [`emotion_${String(index + 1).padStart(4, "0")}`, `${pair.costume || "Costume"} / ${pair.emotion || "Emotion"}`])
                : DEFAULT_EMOTION_STAGES;
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
        const names = ["SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler", "UpscaleModelLoader"];
        let allNodeDefs = null;
        for (const name of names) {
            try {
                const r = await api.fetchApi(`/object_info/${encodeURIComponent(name)}`);
                if (r.ok) {
                    const data = await r.json();
                    this.nodeDefs[name] = data?.[name];
                }
            } catch {
                // Keep defaults if optional upscaler nodes are unavailable.
            }
        }
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

    protectNativeControl(input) {
        if (!input || input._vnccsNativeControlProtected) return input;
        input._vnccsNativeControlProtected = true;
        for (const eventName of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "touchstart", "touchend", "keydown"]) {
            input.addEventListener(eventName, event => event.stopPropagation(), true);
        }
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
            attention_mode: "Attention backend for SeedVR. Auto-detected from installed ComfyUI packages until changed manually.",
            use_internal_rmbg: "Uses the built-in background remover instead of relying only on chroma key.",
            preset: "Strength preset for chroma/background removal."
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
            for (const opt of options || []) {
                const option = document.createElement("option");
                option.value = opt;
                option.textContent = opt;
                input.appendChild(option);
            }
        } else if (type === "checkbox") {
            wrap.className = "vnccs-pipe-check";
            input = document.createElement("input");
            input.type = "checkbox";
            input.checked = Boolean(this.data[section][key]);
            input.onchange = () => this.set(section, key, input.checked);
            wrap.append(input, caption);
            return wrap;
        } else if (type === "textarea") {
            input = document.createElement("textarea");
            input.className = "vnccs-pipe-textarea";
        } else {
            input = document.createElement("input");
            input.className = "vnccs-pipe-input";
            input.type = type;
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
        const denoiseZone = (next) => next < 0.5
            ? { status: "weak", color: "#64a8ff", border: "rgba(100,168,255,0.5)", bg: "rgba(100,168,255,0.1)", glow: "rgba(100,168,255,0.3)" }
            : (next <= 0.65
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
        input.value = String(this.data.emotion_generation?.[key] ?? DEFAULT_DATA.emotion_generation[key]);
        input.oninput = () => {
            const raw = Number(input.value);
            const next = Number.isFinite(raw) ? Math.max(min, Math.min(max, raw)) : DEFAULT_DATA.emotion_generation[key];
            this.set("emotion_generation", key, next);
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

    animaLLLiteField() {
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-field";
        setHelpText(wrap, "Anima LLLite applied to the MODEL before FaceDetailer.");

        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = "anima controlnet";

        const input = document.createElement("select");
        input.className = "vnccs-pipe-select";
        this.protectNativeControl(input);
        for (const opt of ANIMA_LLLITE_OPTIONS) {
            const option = document.createElement("option");
            option.value = opt;
            option.textContent = opt;
            input.appendChild(option);
        }
        const current = this.data.emotion_generation?.anima_lllite_name;
        input.value = ANIMA_LLLITE_OPTIONS.includes(current) ? current : ANIMA_LLLITE_OPTIONS[0];
        input.onchange = () => this.set("emotion_generation", "anima_lllite_name", input.value);

        wrap.append(caption, input);
        return wrap;
    }

    animaControlNetStrengthSlider() {
        const value = Math.max(0, Math.min(1, Number(this.data.emotion_generation?.anima_lllite_strength ?? 1.0)));
        const effectiveStrength = (relativeValue) => 0.7 + Math.max(0, Math.min(1, Number(relativeValue))) * 0.3;
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-slider-field";
        setHelpText(wrap, "Relative scale: 0 maps to 0.70, 1 maps to 1.00; drives emotion strength and FaceDetailer denoise.");

        const head = document.createElement("div");
        head.className = "vnccs-pipe-slider-head";
        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = "emotion strenght";
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

        const status = document.createElement("div");
        status.className = "vnccs-pipe-slider-status";
        const paint = (nextValue) => {
            const next = Math.max(0, Math.min(1, Number(nextValue)));
            const effective = effectiveStrength(next);
            const fill = next * 100;
            let zone = {
                text: "WEAK",
                color: "#00b7ff",
                glow: "rgba(0,183,255,0.24)",
                border: "rgba(0,183,255,0.46)",
                bg: "rgba(0,183,255,0.1)",
            };
            if (effective >= 0.9 && effective <= 0.95) {
                zone = {
                    text: "OPTIMAL",
                    color: "#00d68f",
                    glow: "rgba(0,214,143,0.28)",
                    border: "rgba(0,214,143,0.5)",
                    bg: "rgba(0,214,143,0.1)",
                };
            } else if (effective > 0.95) {
                zone = {
                    text: "STRONG",
                    color: "#ff6aa2",
                    glow: "rgba(255,106,162,0.24)",
                    border: "rgba(255,106,162,0.48)",
                    bg: "rgba(255,106,162,0.1)",
                };
            }
            slider.style.setProperty("--fill", `${fill}%`);
            slider.style.setProperty("--zone-color", zone.color);
            slider.style.setProperty("--zone-glow", zone.glow);
            status.style.setProperty("--zone-color", zone.color);
            status.style.setProperty("--zone-border", zone.border);
            status.style.setProperty("--zone-bg", zone.bg);
            valueEl.textContent = effective.toFixed(2);
            status.textContent = zone.text;
        };
        paint(value);
        slider.oninput = () => {
            const next = Math.max(0, Math.min(1, Number(slider.value)));
            paint(next);
            this.set("emotion_generation", "anima_lllite_strength", next);
        };

        wrap.append(head, slider, status);
        return wrap;
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
            const isAnima = this.connectedEmotionStudioIsAnima();
            this.settingsEl.appendChild(this.block("Emotion Strenght", [
                isAnima ? this.animaControlNetStrengthSlider() : this.faceDenoiseSlider(),
            ]));
            const faceDetailerFields = [
                this.faceDetailerNumberField("bbox_threshold", "bbox_threshold", { min: 0, max: 1, step: 0.01 }),
                this.faceDetailerNumberField("bbox_dilation", "bbox_dilation", { min: 0, max: 128, step: 1 }),
                this.faceDetailerNumberField("sam_dilation", "sam_dilation", { min: 0, max: 128, step: 1 }),
                this.faceDetailerNumberField("sam_threshold", "sam_threshold", { min: 0, max: 1, step: 0.01 }),
                this.faceDetailerNumberField("sam_bbox_expansion", "sam_bbox_expansion", { min: 0, max: 128, step: 1 }),
            ];
            this.settingsEl.appendChild(this.block("Face Detailer", faceDetailerFields));
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
            upscalerFields.push(
                this.field("upscaler", "gan_model", "model", "select", this.getWorkflowModelOptions("UpscaleModelLoader", "model_name", WORKFLOW_GAN_UPSCALER_MODELS, this.data.upscaler.gan_model)),
            );
        } else if (this.data.upscaler.mode !== "off") {
            upscalerFields.push(
                this.field("upscaler", "model", "dit model", "select", this.getWorkflowModelOptions("SeedVR2LoadDiTModel", "model", WORKFLOW_UPSCALER_DIT_MODELS, this.data.upscaler.model)),
                this.field("upscaler", "resolution", "resolution", "number"),
                this.field("upscaler", "attention_mode", "attention mode", "select", this.getOptions("SeedVR2LoadDiTModel", "attention_mode", this.seedvrAttention.available, this.data.upscaler.attention_mode)),
            );
        }
        this.settingsEl.appendChild(this.block("Upscaler", upscalerFields));
        this.settingsEl.appendChild(this.block("BG Remove", [
            // TODO: Decide what to do with internal RMBG later.
            this.field("bg_remove", "preset", "chroma preset", "select", ["disabled", "light", "balanced", "strong", "aggressive"]),
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
