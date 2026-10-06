import { createStylePicker } from "./character_styles.mjs";
import { app } from "../../scripts/app.js";
import { vnccsApi as api, mediaURL, checkedJSON, storage, refreshPreviewImage, watchConnection } from "./vnccs_transport.js";
import { presetGroups, presetSelection } from "./character_presets.mjs";
import { createTraitInput, createQwenVLModelLoader, SAKURA_THEME_CSS, debounce, registerCleanup, injectStyles, showModal as showCommonModal, createLoadingOverlay, showMessage, generateRandomSeed, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, createRequestGuard, createControlCenterClient, resolutionScaleMegapixels, resolutionScaleValue, resolutionScaleText, RESOLUTION_SCALE_MIN_MP, RESOLUTION_SCALE_MAX_MP, RESOLUTION_SCALE_STEP_MP } from "./vnccs_common.js";

const QI2_OVERHAUL_LORA_NAME = "QI2.1/VNCCS/VNCCS_QI2_AnimeOverhaulV1.safetensors";
const QI2_OVERHAUL_TITLE = "Qwen Image2.1 Character Overhaul";
const QI2_OVERHAUL_ENTRY = {
    name: "VNCCS Overhaul QI2", type: "Helper", kind: "QI2",
    local_path: `models/loras/${QI2_OVERHAUL_LORA_NAME}`,
    description: "Overhaul for Character Creator",
};
const normalizeOverhaulStrength = value => {
    if (value == null || value === "") return 0.5;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? Math.round(Math.max(0, Math.min(1, numeric)) * 4) / 4 : 0.5;
};
const isCreatorOverhaulLora = name => /^vnccs_qi2_animeoverhaulv\d+(?:[._]\d+)*\.safetensors$/i
    .test(String(name || "").replace(/\\/g, "/").split("/").pop());

const LEGACY_ANIMA_RESOLUTION_SCALES = {
    normal: 1024,
    high: 1741,
    maximum: 2458,
};

// --- STYLES: Sakura Archive Design System ---
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.vnccs-creator-container {
${SAKURA_THEME_CSS}
    --warning: #ffaa00;
    --shadow-subtle: 0 2px 8px rgba(0,0,0,0.3);
    --shadow-elevated: 0 8px 32px rgba(0,0,0,0.5);
}

/* Main Host */
.vnccs-creator-container {
    position: relative;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 13px;
    width: 100%;
    height: 100%;
    overflow: hidden;
    box-sizing: border-box;
    padding: 12px;
    gap: 12px;
    pointer-events: none;
    zoom: 0.67;
}

/* Layout */
.vnccs-creator-top-row {
    display: grid;
    grid-template-columns: 30% 35% 35%;
    gap: 12px;
    flex: 1;
    min-height: 0;
    width: 100%;
}
.vnccs-creator-bottom-row {
    display: grid;
    grid-template-columns: 30% 35% 35%;
    gap: 12px;
    height: 80px;
    min-height: 80px;
    width: 100%;
    flex-shrink: 0;
    pointer-events: auto;
}

/* Columns */
.vnccs-creator-col {
    display: flex;
    flex-direction: column;
    background: rgba(20, 16, 30, 0.88);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 16px;
    gap: 10px;
    overflow-y: auto;
    height: 100%;
    box-sizing: border-box;
    pointer-events: auto;
    position: relative;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.vnccs-creator-col::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.5), transparent);
    border-radius: 1px;
}
.vnccs-creator-col::-webkit-scrollbar { width: 4px; }
.vnccs-creator-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* Section titles */
.vnccs-creator-section-title {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    flex-shrink: 0;
    pointer-events: auto;
    display: flex;
    align-items: center;
    gap: 8px;
}
.vnccs-creator-section-title::before {
    content: '';
    width: 3px;
    height: 12px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
}

/* Interactive elements */
.vnccs-creator-field,
.vnccs-creator-btn-row > *,
.vnccs-creator-preview-container,
.vnccs-creator-lora-item,
.vnccs-creator-textarea-wrapper,
.vnccs-creator-slider-container,
.vnccs-creator-input,
.vnccs-creator-select,
.vnccs-creator-textarea {
    pointer-events: auto;
}

/* Fields */
.vnccs-creator-field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 6px; flex-shrink: 0; }
.vnccs-creator-label {
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Inputs */
.vnccs-creator-input, .vnccs-creator-textarea {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    transition: all var(--transition);
}
.vnccs-creator-select {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    min-height: 34px;
    transition: all var(--transition);
    color-scheme: dark;
}
.vnccs-creator-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}
/* Style library uses the whole node workspace without changing its size. */
.vnccs-creator-container .vnccs-style-summary {
    display: flex; gap: 12px; align-items: center; width: 100%; padding: 10px;
    text-align: left; color: var(--text-primary); background: var(--bg-surface);
    border: 1px solid var(--accent-border); border-radius: 12px; cursor: pointer;
}
.vnccs-creator-container .vnccs-style-placeholder {
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: center; aspect-ratio: 1;
    background: linear-gradient(145deg, #292034, #17131f);
    color: var(--text-secondary); font-size: 11px; border-radius: 10px;
}
.vnccs-style-preview-image {
    position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
    border-radius: inherit;
}
.vnccs-style-summary > .vnccs-style-placeholder { width: 106.4px; flex-shrink: 0; }
.vnccs-style-details { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.vnccs-style-name { font-size: 14px; }
.vnccs-style-description, .vnccs-style-reference {
    display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    font-size: 11px; color: var(--text-secondary);
}
.vnccs-creator-container .vnccs-style-gallery {
    position: absolute; inset: 0; z-index: 100; display: flex; flex-direction: column;
    padding: 20px; gap: 12px; box-sizing: border-box; border-radius: inherit;
    background: var(--bg-primary); color: var(--text-primary); overflow: hidden;
    pointer-events: auto;
}
.vnccs-style-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; flex-shrink: 0; }
.vnccs-style-toolbar input { flex: 1; min-width: 120px; }
.vnccs-style-toolbar select { width: auto; max-width: 240px; }
.vnccs-style-heading { font-size: 18px; margin-right: auto; }
.vnccs-creator-container .vnccs-style-gallery .vnccs-creator-btn {
    flex: 0 0 auto; background: var(--bg-surface); border: 1px solid var(--accent-border);
}
.vnccs-creator-container .vnccs-style-gallery .vnccs-creator-btn:hover:not(:disabled) {
    background: var(--bg-hover); border-color: var(--accent);
}
.vnccs-style-status { color: var(--text-secondary); font-size: 12px; flex-shrink: 0; }
.vnccs-style-content { flex: 1; min-height: 0; overflow-y: auto; padding: 3px; }
.vnccs-style-size-controls { display: flex; align-items: center; gap: 12px; flex-shrink: 0; font-size: 12px; color: var(--text-secondary); }
.vnccs-style-size-slider { width: 180px; max-width: 50%; accent-color: var(--accent); cursor: pointer; }
.vnccs-style-size-value { min-width: 40px; font-variant-numeric: tabular-nums; }
.vnccs-style-preview-location { font-size: 11px; color: var(--text-secondary); overflow-wrap: anywhere; }
.vnccs-style-preview-location:empty { display: none; }
.vnccs-style-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--vnccs-style-card-size, 182px)), 1fr)); gap: 14px; }
.vnccs-style-tile { position: relative; min-width: 0; }
.vnccs-creator-container .vnccs-style-card {
    position: relative; width: 100%; aspect-ratio: 1; padding: 0; overflow: hidden;
    border-radius: 14px; border: 1px solid var(--accent-border); cursor: pointer;
    background: var(--bg-surface); color: var(--text-primary); text-align: left;
}
.vnccs-style-card > .vnccs-style-placeholder { width: 100%; height: 100%; border-radius: 0; }
.vnccs-style-editor > .vnccs-style-placeholder { width: min(100%, 220px); }
.vnccs-style-card-label {
    position: absolute; inset: auto 0 0; padding: 26px 12px 12px; font-size: 13px;
    font-weight: 600; background: linear-gradient(transparent, rgba(0,0,0,.8));
}
.vnccs-creator-container .vnccs-style-card[aria-pressed="true"] { border: 2px solid var(--accent); }
.vnccs-creator-container .vnccs-style-summary:hover,
.vnccs-creator-container .vnccs-style-card:hover { border-color: var(--accent); }
.vnccs-creator-container .vnccs-style-edit,
.vnccs-creator-container .vnccs-style-delete {
    position: absolute; top: 8px; right: 8px; border: 1px solid var(--accent-border);
    border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); cursor: pointer;
}
.vnccs-creator-container .vnccs-style-edit { right: 44px; }
.vnccs-creator-container .vnccs-style-delete {
    width: 28px; height: 28px; padding: 0; font-size: 20px; line-height: 1;
}
.vnccs-creator-container .vnccs-style-delete:hover:not(:disabled) { color: var(--error); }
.vnccs-creator-container .vnccs-style-editor { max-width: 720px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
.vnccs-creator-container .vnccs-style-editor textarea { height: auto; min-height: 160px; resize: vertical; }
.vnccs-creator-container .vnccs-style-gallery [hidden] { display: none; }
.vnccs-creator-container .vnccs-style-gallery :focus-visible,
.vnccs-creator-container .vnccs-style-summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.vnccs-creator-input,
.vnccs-creator-select {
    height: 34px;
    min-height: 34px;
}
.vnccs-creator-input:focus, .vnccs-creator-select:focus, .vnccs-creator-textarea:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255,143,163,0.04);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

/* Slider */
.vnccs-creator-slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 6px 10px;
}
.vnccs-creator-slider {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    height: 3px;
}
.vnccs-creator-slider-val {
    width: 42px;
    text-align: right;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-primary);
    background: transparent;
    border: none;
}
.vnccs-creator-slider-val:focus { outline: none; border-bottom: 1px solid var(--accent-border); }

/* Preview */
.vnccs-creator-preview-container {
    flex: 1;
    background: radial-gradient(circle, rgba(255,143,163,0.04) 1px, transparent 1px), rgba(10,10,15,0.7);
    background-size: 20px 20px, 100% 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    min-height: 0;
}
.vnccs-creator-preview-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    animation: vnccs-creator-fadein 0.4s ease;
}
.vnccs-creator-preview-img.vnccs-creator-character-preview {
    width: auto;
    max-width: none;
    height: 100%;
    flex: 0 0 auto;
}
.vnccs-creator-preview-loading {
    position: absolute;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(10, 10, 16, 0.28);
    backdrop-filter: blur(1px);
    pointer-events: none;
}
.vnccs-creator-preview-loading.is-visible {
    display: flex;
}
.vnccs-creator-preview-spinner {
    width: 34px;
    height: 34px;
    border: 2px solid rgba(255, 143, 163, 0.24);
    border-top-color: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 18px rgba(255,143,163,0.25);
    animation: vnccs-creator-spin 0.75s linear infinite;
}
.vnccs-creator-sprite-nav {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding-top: 7px;
    pointer-events: auto;
}
.vnccs-creator-sprite-nav.is-visible {
    display: flex;
}
.vnccs-creator-sprite-nav-btn {
    width: 34px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    background: linear-gradient(180deg, rgba(255,143,163,0.14), rgba(255,255,255,0.04));
    color: var(--accent);
    box-shadow: 0 0 12px rgba(255,143,163,0.12);
    cursor: pointer;
    transition: all var(--transition);
}
.vnccs-creator-sprite-nav-btn:hover {
    border-color: var(--accent);
    background: linear-gradient(180deg, rgba(255,143,163,0.22), rgba(255,255,255,0.06));
    box-shadow: 0 0 16px rgba(255,143,163,0.22);
}
.vnccs-creator-sprite-nav-btn:disabled {
    opacity: 0.55;
    cursor: default;
}
.vnccs-creator-sprite-nav-btn svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    stroke-width: 2.6;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.vnccs-creator-sprite-nav-count {
    min-width: 46px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-secondary);
}
@keyframes vnccs-creator-fadein { from { opacity: 0; } to { opacity: 1; } }
@keyframes vnccs-creator-spin { to { transform: rotate(360deg); } }
.vnccs-creator-placeholder { color: var(--text-muted); text-align: center; font-size: 11px; }

/* LoRA stack */
.vnccs-creator-lora-stack {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
}
.vnccs-creator-lora-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: rgba(255,255,255,0.02);
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    transition: border-color var(--transition);
}
.vnccs-creator-lora-item:hover { border-color: var(--border-hover); }
.vnccs-creator-lora-row { display: flex; gap: 6px; align-items: center; }

/* Buttons */
.vnccs-creator-btn-row { display: flex; gap: 8px; margin-top: auto; flex-shrink: 0; }
.vnccs-creator-btn {
    flex: 1;
    padding: 10px;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11px;
    font-family: var(--font);
    color: white;
    text-align: center;
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
}
.vnccs-creator-btn-primary {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525;
    box-shadow: 0 4px 16px rgba(255,143,163,0.25);
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.vnccs-creator-btn-primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: vnccs-creator-shimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes vnccs-creator-shimmer {
    0% { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35% { transform: translateX(120%) skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%) skewX(-15deg); opacity: 0; }
}
.vnccs-creator-btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255,143,163,0.4);
}
.vnccs-creator-container button.vnccs-creator-btn.vnccs-creator-btn-primary:not(:disabled),
.vnccs-creator-container button.vnccs-creator-btn.vnccs-creator-btn-primary:not(:disabled):hover,
.vnccs-creator-container button.vnccs-creator-btn.vnccs-creator-btn-primary:not(:disabled):focus,
.vnccs-creator-container button.vnccs-creator-btn.vnccs-creator-btn-primary:not(:disabled):focus-visible,
.vnccs-creator-container button.vnccs-creator-btn.vnccs-creator-btn-primary:not(:disabled):active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    outline: none;
}
.vnccs-creator-btn-success {
    background: rgba(0,214,143,0.15);
    color: var(--success);
    border: 1px solid rgba(0,214,143,0.3);
}
.vnccs-creator-btn-success:hover:not(:disabled) {
    background: rgba(0,214,143,0.25);
    transform: translateY(-1px);
}
.vnccs-creator-btn-danger {
    background: rgba(255,71,87,0.15);
    color: var(--error);
    border: 1px solid rgba(255,71,87,0.3);
}
.vnccs-creator-btn-danger:hover:not(:disabled) {
    background: rgba(255,71,87,0.25);
    transform: translateY(-1px);
}
.vnccs-creator-btn-disabled, .vnccs-creator-btn:disabled {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-muted) !important;
    cursor: not-allowed;
    box-shadow: none !important;
    transform: none !important;
}
.vnccs-creator-btn:focus,
.vnccs-creator-btn:focus-visible,
.vnccs-creator-segmented-btn:focus,
.vnccs-creator-segmented-btn:focus-visible,
.vnccs-creator-seed-dice-btn:focus,
.vnccs-creator-seed-dice-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-creator-btn-primary:focus:not(:disabled),
.vnccs-creator-btn-primary:focus-visible:not(:disabled),
.vnccs-creator-btn-primary:active:not(:disabled) {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 8px 28px rgba(255,143,163,0.4), 0 0 0 2px rgba(255,143,163,0.28);
}

/* Bottom textareas */
.vnccs-creator-textarea-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: rgba(20,16,30,0.88);
    padding: 8px 10px;
    border-radius: var(--radius-md);
    border: 1px solid var(--accent-border);
    position: relative;
}
.vnccs-creator-textarea-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.4), transparent);
}
.vnccs-creator-textarea-wrapper textarea {
    flex: 1;
    resize: none;
    border: none;
    background: transparent;
    padding: 4px;
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
}
.vnccs-creator-textarea-wrapper textarea:focus { outline: none; }
.vnccs-creator-textarea-label {
    font-size: 9px;
    color: var(--accent);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0 2px 4px;
}

/* Character traits */
.vnccs-creator-trait-list {
    border-top: 1px solid var(--accent-border);
    flex-shrink: 0;
}
.vnccs-creator-trait-row {
    display: grid;
    grid-template-columns: minmax(56px, 18%) minmax(0, 1fr) 44px;
    align-items: center;
    gap: 12px;
    min-height: 64px;
    padding: 10px 0;
    border-bottom: 1px solid var(--accent-border);
    box-sizing: border-box;
}
.vnccs-creator-trait-label {
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 400;
}
.vnccs-creator-trait-editor { min-width: 0; }
.vnccs-creator-trait-values {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    width: 100%;
    min-height: 44px;
    padding: 0;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-primary);
    font: inherit;
    font-size: 13px;
    text-align: left;
    cursor: pointer;
}
.vnccs-creator-trait-token {
    max-width: 100%;
    padding: 5px 8px;
    border: 1px solid rgba(255,182,200,0.22);
    border-radius: 7px;
    background: rgba(184,169,232,0.14);
    line-height: 1.4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vnccs-creator-trait-empty { color: var(--text-secondary); }
.vnccs-creator-trait-input { min-height: 44px; font-size: 13px; }
.vnccs-creator-trait-values[hidden], .vnccs-creator-trait-input[hidden] { display: none; }
.vnccs-creator-trait-add {
    width: 44px;
    height: 44px;
    padding: 0;
    background: rgba(255,255,255,0.06);
    color: var(--text-primary);
    border: 1px solid rgba(255,182,200,0.22);
    border-radius: 10px;
    font: inherit;
    font-size: 20px;
    cursor: pointer;
    transition: background var(--transition), border-color var(--transition);
}
.vnccs-creator-trait-add:hover { background: var(--bg-hover); border-color: var(--accent); }
.vnccs-creator-trait-values:hover .vnccs-creator-trait-token { border-color: var(--accent-border); }

/* Tag constructor */
.vnccs-creator-tag-grid {
    display: flex; flex-wrap: wrap; gap: 5px;
    max-height: 300px; overflow-y: auto;
    padding: 8px;
    background: rgba(10,10,15,0.6);
    border-radius: var(--radius-sm);
}
.vnccs-creator-tag-chip {
    font-family: inherit;
    text-align: left;
    padding: 4px 10px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 11px;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
    transition: all var(--transition);
}
.vnccs-creator-trait-add:focus-visible, .vnccs-creator-trait-values:focus-visible, .vnccs-creator-tag-chip:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}
.vnccs-creator-preset-description {
    width: 100%;
    min-height: 5em;
    margin: 0 0 8px;
    color: var(--text-secondary);
    line-height: 1.5;
}
.vnccs-creator-tag-chip:hover:not(.selected) { background: rgba(255,143,163,0.1); border-color: var(--accent-border); color: var(--accent-hover); }
.vnccs-creator-tag-chip.selected { background: rgba(255,143,163,0.18); color: var(--accent-hover); border-color: var(--accent); }

.vnccs-creator-tag-category {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 6px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.8px;
    width: 100%;
}

/* Custom toggle checkbox */
.vnccs-creator-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    padding: 6px 0;
    user-select: none;
}
.vnccs-creator-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-creator-toggle input {
    opacity: 0;
    width: 0; height: 0;
    position: absolute;
}
.vnccs-creator-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    transition: all 0.25s ease;
}
.vnccs-creator-toggle-thumb {
    position: absolute;
    top: 3px; left: 3px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.25s ease;
}
.vnccs-creator-toggle input:checked ~ .vnccs-creator-toggle-track {
    background: rgba(255,143,163,0.2);
    border-color: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
}
.vnccs-creator-toggle input:checked ~ .vnccs-creator-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
}
.vnccs-creator-toggle-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: color var(--transition);
}
.vnccs-creator-toggle input:checked ~ ~ .vnccs-creator-toggle-label,
.vnccs-creator-toggle-wrap:has(input:checked) .vnccs-creator-toggle-label {
    color: var(--accent-hover);
}

.vnccs-creator-segmented-field {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 4px;
    padding: 4px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: rgba(0,0,0,0.18);
    min-height: 48px;
    box-sizing: border-box;
}
.vnccs-creator-segmented-field.is-three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.vnccs-creator-segmented-btn {
    border: 0;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--text-secondary);
    font-family: var(--font);
    font-size: 14px;
    font-weight: 800;
    cursor: pointer;
    transition: all var(--transition);
}

.vnccs-creator-segmented-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}
.vnccs-creator-segmented-btn:hover:not(:disabled):not(.is-active) {
    color: var(--text-primary);
    background: rgba(255,255,255,0.045);
}

.vnccs-creator-segmented-btn.is-active {
    color: #20141a;
    background: linear-gradient(180deg, #ff9bad 0%, #ff87a0 100%);
    box-shadow: 0 10px 22px rgba(255,143,163,0.22);
}

.vnccs-creator-graphic-toggle {
    width: 100%;
    min-height: 48px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.035);
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 10px 8px 14px;
    font-family: var(--font);
    cursor: pointer;
    transition: all var(--transition);
}

.vnccs-creator-graphic-toggle:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-creator-graphic-toggle.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.14);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset, 0 12px 24px rgba(255,143,163,0.12);
}

.vnccs-creator-graphic-toggle-text {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.vnccs-creator-graphic-toggle-icon {
    width: 20px;
    height: 20px;
    border-radius: 7px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: currentColor;
    background: rgba(255,255,255,0.06);
}

.vnccs-creator-graphic-toggle-switch {
    width: 44px;
    height: 24px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    position: relative;
    flex-shrink: 0;
    transition: all var(--transition);
}

.vnccs-creator-graphic-toggle-switch::after {
    content: "";
    position: absolute;
    width: 16px;
    height: 16px;
    left: 3px;
    top: 3px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all var(--transition);
}

.vnccs-creator-graphic-toggle.is-active .vnccs-creator-graphic-toggle-switch {
    border-color: var(--accent);
    background: rgba(255,143,163,0.28);
}

.vnccs-creator-graphic-toggle.is-active .vnccs-creator-graphic-toggle-switch::after {
    transform: translateX(20px);
    background: var(--accent-hover);
}

/* Filled input highlight */
.vnccs-creator-input:not(:placeholder-shown):not([value=""]),
.vnccs-creator-input.has-value {
    border-color: rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
}

/* Preview placeholder with icon */
.vnccs-creator-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    color: var(--text-muted);
    font-size: 11px;
    letter-spacing: 0.05em;
}
.vnccs-creator-placeholder-icon {
    width: 48px;
    height: 48px;
    opacity: 0.25;
}

/* LoRA slot collapsed state */
.vnccs-creator-lora-item.is-empty {
    opacity: 0.45;
}
.vnccs-creator-lora-item.is-empty:hover {
    opacity: 1;
}

/* Button hierarchy */
.vnccs-creator-btn-generate {
    flex: 2;
}
.vnccs-creator-btn-secondary {
    flex: 1;
    font-size: 10px;
}

.vnccs-creator-tab-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 6px;
}

.vnccs-creator-tab {
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    padding: 8px 10px;
    font-family: var(--font);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all var(--transition);
}

.vnccs-creator-tab:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-creator-tab.is-active {
    border-color: var(--accent);
    color: var(--accent-hover);
    background: rgba(255,143,163,0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}

.vnccs-creator-subsection {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-creator-model-card-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.vnccs-creator-model-picker {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-creator-model-picker-menu {
    display: none;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    border: 1px solid rgba(255,143,163,0.18);
    border-radius: 10px;
    background: rgba(8,8,12,0.48);
}

.vnccs-creator-model-picker.is-open .vnccs-creator-model-picker-menu {
    display: flex;
}

.vnccs-creator-model-picker-group {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.vnccs-creator-model-picker-group-title {
    color: var(--accent-hover);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.vnccs-creator-model-card {
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: rgba(0,214,143,0.05);
    border: 1px solid rgba(0,214,143,0.25);
    border-radius: 10px;
    padding: 10px 12px 8px;
    cursor: default;
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
}

.vnccs-creator-model-card.is-picker-head {
    min-height: 58px;
}

.vnccs-creator-model-card.is-installed {
    cursor: pointer;
}

.vnccs-creator-model-card.is-installed:hover:not(.is-selected) {
    border-color: rgba(0,214,143,0.42);
    background: rgba(0,214,143,0.08);
}

.vnccs-creator-model-card.is-selected {
    border-color: var(--accent);
    background: rgba(255,143,163,0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset;
}

.vnccs-creator-model-card.is-missing {
    opacity: 0.92;
}

.vnccs-creator-model-card-top {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
}

.vnccs-creator-model-card-name {
    flex: 1;
    min-width: 0;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 700;
    line-height: 1.25;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.vnccs-creator-model-card-status {
    flex-shrink: 0;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.vnccs-creator-model-card-status.ok { color: var(--success); }
.vnccs-creator-model-card-status.missing { color: var(--error); }
.vnccs-creator-model-card-status.progress { color: var(--accent-lavender); }

.vnccs-creator-model-card-desc {
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.4;
}

.vnccs-creator-model-card-actions {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vnccs-creator-model-card-download {
    width: 100%;
    padding: 7px 9px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--accent-border);
    background: rgba(255,143,163,0.08);
    color: var(--accent-hover);
    font-family: var(--font);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
}

.vnccs-creator-model-card-download:hover {
    background: rgba(255,143,163,0.14);
}

.vnccs-creator-model-card-badge {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    background: var(--text-muted);
}

.vnccs-creator-model-card-badge.ok { background: var(--success); }
.vnccs-creator-model-card-badge.missing { background: var(--error); }
.vnccs-creator-model-card-badge.progress { background: var(--accent-lavender); }

.vnccs-creator-model-card-toggle {
    margin-left: auto;
    flex-shrink: 0;
}

.vnccs-creator-generation-fallback {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-creator-gen-param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 18px;
    padding-top: 2px;
}

.vnccs-creator-resolution-field {
    display: block;
    flex-shrink: 0;
    margin: 4px 0 12px;
}

.vnccs-creator-resolution-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.vnccs-creator-resolution-value {
    color: var(--accent-hover);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
}

.vnccs-creator-resolution-slider {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
}

.vnccs-creator-overhaul-control { padding: 0 2px; }
.vnccs-creator-overhaul-slider { display: block; width: 100%; margin: 0; height: 20px; }
.vnccs-creator-overhaul-slider:focus-visible, .vnccs-creator-overhaul-info:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}
.vnccs-creator-overhaul-heading { display: flex; align-items: center; gap: 5px; flex: 1; min-width: 0; }
.vnccs-creator-overhaul-heading .vnccs-creator-model-card-name { flex: 0 1 auto; }
.vnccs-creator-overhaul-help { display: inline-flex; flex-shrink: 0; }
.vnccs-creator-overhaul-info {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; padding: 2px;
    border: 0; border-radius: 50%; background: transparent;
    color: var(--text-secondary); cursor: help;
}
.vnccs-creator-overhaul-info:hover, .vnccs-creator-overhaul-info[aria-expanded="true"] { color: var(--accent); background: var(--accent-subtle); }
.vnccs-creator-overhaul-info svg { width: 15px; height: 15px; }
.vnccs-creator-overhaul-tooltip {
    position: fixed; inset: auto; margin: 0;
    box-sizing: border-box; width: 320px; max-width: calc(100vw - 24px);
    padding: 10px 12px; border: 1px solid var(--accent-border); border-radius: 8px;
    background: var(--bg-secondary); color: var(--text-primary); box-shadow: var(--shadow-elevated);
    font: 12px/1.5 var(--font); white-space: normal; text-transform: none; letter-spacing: normal;
}
.vnccs-creator-overhaul-ticks {
    display: flex;
    justify-content: space-between;
    color: var(--text-secondary);
    font: 10px var(--font-mono);
    padding: 0 2px;
}

.vnccs-creator-qi2-cache {
    display: none;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 12px 18px;
    padding: 10px;
    border: 1px solid rgba(255,143,163,0.18);
    border-radius: var(--radius-md);
    background: rgba(255,143,163,0.045);
}

.vnccs-creator-qi2-cache-title {
    grid-column: 1 / -1;
    color: var(--accent);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.vnccs-creator-gen-param-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
}

.vnccs-creator-gen-param-input,
.vnccs-creator-gen-param-select {
    width: 100%;
    height: 48px;
    box-sizing: border-box;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.045);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 14px;
    font-weight: 600;
    padding: 8px 12px;
    transition: all var(--transition);
    color-scheme: dark;
}

.vnccs-creator-gen-param-select {
    zoom: 1;
}

.vnccs-creator-gen-param-input:focus,
.vnccs-creator-gen-param-select:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255,143,163,0.045);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

.vnccs-creator-gen-param-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}

.vnccs-creator-seed-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 52px;
    gap: 8px;
    align-items: stretch;
}

.vnccs-creator-seed-dice-btn {
    width: 52px;
    height: 48px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.045);
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition);
}

.vnccs-creator-seed-dice-btn:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-creator-seed-dice-btn.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.16);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}

.vnccs-creator-seed-dice-btn svg {
    width: 24px;
    height: 24px;
    display: block;
}

.vnccs-creator-character-wizard-btn {
    width: 100%;
    min-height: 40px;
    margin-bottom: 8px;
    flex: 0 0 auto;
}

.vnccs-creator-container .vnccs-common-modal {
    width: min(520px, calc(100% - 48px));
    max-width: min(520px, calc(100% - 48px));
    box-sizing: border-box;
    background: rgba(26,26,38,0.96);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-family: var(--font);
    overflow: hidden;
}
.vnccs-creator-container .vnccs-common-modal-title {
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-hover);
    font-family: var(--font);
}
.vnccs-creator-container .vnccs-common-modal-btn {
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: var(--font);
    font-weight: 700;
}
.vnccs-creator-container .vnccs-common-modal-btn:focus,
.vnccs-creator-container .vnccs-common-modal-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-creator-container .vnccs-common-modal-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    border-color: transparent !important;
}
.vnccs-creator-container .vnccs-common-modal-btn-primary:hover,
.vnccs-creator-container .vnccs-common-modal-btn-primary:focus,
.vnccs-creator-container .vnccs-common-modal-btn-primary:focus-visible,
.vnccs-creator-container .vnccs-common-modal-btn-primary:active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
}
.vnccs-creator-character-wizard-modal {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
}

.vnccs-creator-character-wizard-modal-text {
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.45;
    white-space: normal;
    overflow-wrap: anywhere;
}

.vnccs-creator-character-wizard-modal textarea {
    width: 100%;
    min-height: 110px;
    box-sizing: border-box;
    resize: vertical;
}

.vnccs-creator-qwenvl-download-status {
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.45;
}

.vnccs-creator-qwenvl-download-track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,143,163,0.16);
}

.vnccs-creator-qwenvl-download-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--accent-hover));
    transition: width 0.2s ease;
}

.vnccs-creator-qwenvl-download-pct {
    color: var(--accent-hover);
    font-size: 11px;
    font-weight: 800;
    text-align: right;
}
`;

app.registerExtension({
    name: "VNCCS.CharacterCreatorV2",

    async setup() {
        const queuePrompt = app.queuePrompt;
        const origQueuePrompt = (...args) => queuePrompt.apply(app, args);
        app.queuePrompt = async function(...args) {
            const nodes = app.graph?._nodes?.filter(n => n.type === "CharacterCreatorV2") || [];
            for (const node of nodes) {
                node._randomizeSeedIfNeeded?.();
            }
            return origQueuePrompt(...args);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CharacterCreatorV2") {
            let characterStyleCatalog;
            try {
                const response = await api.fetchApi("/vnccs/character_styles");
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                characterStyleCatalog = await response.json();
                if (!Array.isArray(characterStyleCatalog.groups)) {
                    throw new Error("Style catalog has no groups");
                }
            } catch (error) {
                console.error("[VNCCS Character Creator V2] Failed to load character styles:", error);
                characterStyleCatalog = { default_style: "custom", aliases: {}, groups: [] };
            }
            const DEFAULT_CHARACTER_STYLE = characterStyleCatalog.default_style || "custom";

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1280, 800]); // Default wide 3-column
                syncDOMWidgetWidthSoon(node, "ui");

                // 1. Setup CSS
                injectStyles(STYLE, "vnccs-character-creator-v2");

                // 2. Strict Widget Cleanup
                const cleanup = () => {
                    if (!node.widgets) return;
                    for (const w of node.widgets) {
                        if (w.name !== "ui" && w.name !== "widget_data") {
                            w.hidden = true;
                            //w.computeSize = () => [0, -4]; // Cause issues sometimes?
                        }
                    }
                };
                cleanup();

                // Keep keeping them hidden
                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    cleanup();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                node._randomizeSeedIfNeeded = () => {
                    if (state.gen_settings.seed_mode === "randomize") {
                        state.gen_settings.seed = generateRandomSeed();
                        if (els.seed) els.seed.value = state.gen_settings.seed;
                        saveState();
                    }
                };

                // Override onSerialize to guarantee state is written to widget before execution
                const origSerialize = node.onSerialize;
                node.onSerialize = function (o) {
                    if (origSerialize) origSerialize.apply(this, arguments);
                    syncBackgroundForGenerationMode();

                    // Critical Sync: Ensure widget_data receives latest state
                    const w = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                    if (w) {
                        saveCurrentGenerationModeValues();
                        w.value = JSON.stringify(state);
                    } else {
                        // Should have been created, but safety net
                        console.warn("[VNCCS] widget_data missing on serialize, creating...");
                        saveCurrentGenerationModeValues();
                        node.addWidget("text", "widget_data", JSON.stringify(state), (v) => { }, { serialize: true });
                    }
                    const index = node.widgets?.findIndex(widget => widget.name === "widget_data") ?? -1;
                    if (index >= 0 && Array.isArray(o?.widgets_values)) {
                        o.widgets_values[index] = node.widgets[index].value;
                    }
                };

                // 3. State & Widget Setup
                // Ensure 'widget_data' widget exists (ComfyUI backend hidden inputs don't always create widgets automatically)
                let dataWidget = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                if (!dataWidget) {
                    // Create it manually if missing. 
                    // Type "text" is safe, we'll hide it.
                    // serialize: true is default for widgets added this way? We check opts.
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                // Ensure it's hidden (cleanup hides everything else, but let's be explicit)
                if (dataWidget) dataWidget.hidden = true;

                const beginPreviewRequest = createRequestGuard(node);
                const state = {
                    preview_valid: false, // Smart Cache Flag
                    preview_source: "gen", // "gen" or "pose" - tracks what user sees
                    sprite_preview_index: 0,
                    sprite_preview_count: 0,
                    sprite_preview_cache_bust: "",
                    sprite_preview_request_id: 0,
                    character: "",
                    prompt_modes: {
                        illustrious: {
                            aesthetics: "masterpiece, best quality",
                            negative_prompt: "bad quality, worst quality",
                        },
                        anima: {
                            aesthetics: "masterpiece, best quality, score_7",
                            negative_prompt: "bad quality, worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
                        },
                        qi2: {
                            aesthetics: "",
                            negative_prompt: "bad quality, worst quality, low quality, blurry, jpeg artifacts",
                        },
                    },
                    prompt_defaults_version: 1,
                    character_info: {
                        sex: "female", age: 18, framing: "cowboy_shot", style: DEFAULT_CHARACTER_STYLE, custom_style: "", race: "human", skin_color: "",
                        hair: "black hair, waist-length hair", eyes: "", face: "", body: "", additional_details: "",
                        nsfw: false, aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                        lora_prompt: "", background_color: "Green"
                    },
                    gen_settings: {
                        generation_mode: "illustrious",
                        target_size: 1024,
                        background_model_kind: "",
                        previous_background_color: "Green",
                        ckpt_name: "", sampler: "euler", scheduler: "normal",
                        steps: 20, cfg: 8.0, seed: 0, seed_mode: "fixed",
                        diffusion_model_name: "", clip_name: "", vae_name: "",
                        mode_settings: {
                            illustrious: {
                                target_size: 1024,
                                ckpt_name: "", sampler: "euler", scheduler: "normal",
                                steps: 20, cfg: 8.0, seed: 0, seed_mode: "fixed",
                                turbo_previous_settings: null,
                                dmd_lora_name: "", dmd_lora_strength: 1.0,
                                age_lora_name: "",
                                lora_stack: [
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 }
                                ]
                            },
                            anima: {
                                diffusion_model_name: "", clip_name: "qwen_3_06b_base.safetensors", vae_name: "qwen_image_vae.safetensors",
                                target_size: 1024,
                                sampler: "er_sde", scheduler: "simple",
                                steps: 30, cfg: 4.0, seed: 0, seed_mode: "fixed",
                                turbo_enabled: false,
                                dmd_lora_name: "anima\\anima-turbo-lora-v0.1.safetensors",
                                dmd_lora_strength: 1.0,
                                turbo_previous_settings: null,
                                lora_stack: [
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 }
                                ]
                            },
                            qi2: {
                                diffusion_model_name: "qwen_image_2.1_int8_convrot.safetensors",
                                clip_name: "qwen3vl_8b_int8_convrot.safetensors",
                                vae_name: "qwen_image_2.1_vae_bf16.safetensors",
                                target_size: 1024,
                                clip_type: "qwen_image",
                                sampler: "euler", scheduler: "simple",
                                steps: 25, cfg: 3.0, seed: 0, seed_mode: "fixed",
                                turbo_enabled: false,
                                dmd_lora_name: "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
                                dmd_lora_strength: 1.0,
                                turbo_previous_settings: null,
                                qi2_overhaul_strength: 0.5,
                                qi2_cache: { device: "gpu", dtype: "int8" },
                                lora_stack: [
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 },
                                    { name: "", strength: 1.0 }
                                ]
                            }
                        },
                        anima_defaults_applied: false,
                        generation_defaults_version: 2,
                        dmd_lora_name: "", dmd_lora_strength: 1.0,
                        age_lora_name: "",
                        lora_stack: [
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 }
                        ]
                    }
                };

                const debouncedSave = debounce(() => saveState(), 300);
                const ANIMA_TURBO_LORA_NAME = "anima\\anima-turbo-lora-v0.1.safetensors";
                const ANIMA_CLIP_NAME = "qwen_3_06b_base.safetensors";
                const ANIMA_VAE_NAME = "qwen_image_vae.safetensors";
                const QI2_TURBO_LORA_NAME = "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors";
                const QI2_MODEL_NAME = "qwen_image_2.1_int8_convrot.safetensors";
                const QI2_CLIP_NAME = "qwen3vl_8b_int8_convrot.safetensors";
                const QI2_VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors";
                const ILLUSTRIOUS_DEFAULTS = {
                    target_size: 1024,
                    ckpt_name: "", sampler: "euler", scheduler: "normal",
                    steps: 20, cfg: 8.0, seed: 0, seed_mode: "fixed",
                    turbo_previous_settings: null,
                    dmd_lora_name: "", dmd_lora_strength: 1.0,
                    age_lora_name: "",
                    lora_stack: [
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 }
                    ]
                };
                const ANIMA_DEFAULTS = {
                    diffusion_model_name: "", clip_name: ANIMA_CLIP_NAME, vae_name: ANIMA_VAE_NAME,
                    target_size: 1024,
                    sampler: "er_sde", scheduler: "simple",
                    steps: 30, cfg: 4.0, seed: 0, seed_mode: "fixed",
                    turbo_enabled: false,
                    dmd_lora_name: ANIMA_TURBO_LORA_NAME,
                    dmd_lora_strength: 1.0,
                    turbo_previous_settings: null,
                    lora_stack: [
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 }
                    ]
                };
                const QI2_DEFAULTS = {
                    diffusion_model_name: QI2_MODEL_NAME, clip_name: QI2_CLIP_NAME, vae_name: QI2_VAE_NAME,
                    target_size: 1024,
                    clip_type: "qwen_image",
                    sampler: "euler", scheduler: "simple",
                    steps: 25, cfg: 3.0, seed: 0, seed_mode: "fixed",
                    turbo_enabled: false,
                    dmd_lora_name: QI2_TURBO_LORA_NAME,
                    dmd_lora_strength: 1.0,
                    turbo_previous_settings: null,
                    qi2_overhaul_strength: 0.5,
                    qi2_cache: { device: "gpu", dtype: "int8" },
                    lora_stack: [
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 }
                    ]
                };
                const GENERATION_DEFAULTS_VERSION = 6;
                const PROMPT_DEFAULTS_VERSION = 3;
                const MODE_SETTING_KEYS = {
                    illustrious: ["target_size", "ckpt_name", "sampler", "scheduler", "steps", "cfg", "seed", "seed_mode", "dmd_lora_name", "dmd_lora_strength", "turbo_previous_settings", "age_lora_name", "lora_stack"],
                    anima: ["target_size", "diffusion_model_name", "clip_name", "vae_name", "sampler", "scheduler", "steps", "cfg", "seed", "seed_mode", "turbo_enabled", "dmd_lora_name", "dmd_lora_strength", "turbo_previous_settings", "lora_stack"],
                    qi2: ["target_size", "diffusion_model_name", "clip_name", "vae_name", "clip_type", "sampler", "scheduler", "steps", "cfg", "seed", "seed_mode", "turbo_enabled", "dmd_lora_name", "dmd_lora_strength", "turbo_previous_settings", "qi2_cache", "qi2_overhaul_strength", "lora_stack"],
                };
                const MODE_PROMPT_DEFAULTS = {
                    illustrious: {
                        aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                    },
                    anima: {
                        aesthetics: "masterpiece, best quality, score_7",
                        negative_prompt: "bad quality, worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
                    },
                    qi2: {
                        aesthetics: "",
                        negative_prompt: "bad quality, worst quality, low quality, blurry, jpeg artifacts",
                    },
                };
                const CC_REPO_ID = "MIUProject/VNCCS_v3.0";
                let TAG_DATA = null;
                const ccState = { config: null, downloadStatus: {} };
                const modelPickerOpen = {
                    illustrious: false,
                    anima: false,
                    qi2: false,
                };
                let localAssets = {
                    checkpoints: [],
                    diffusion_models: [],
                    text_encoders: [],
                    vae_models: [],
                    loras: [],
                };
                let restoredWidgetInfoCharacter = null;

                const ccNormalize = (value) => String(value || "").trim().toLowerCase();
                const ccKind = (entry) => ccNormalize(entry?.kind ?? entry?.Kind);
                const ccType = (entry) => ccNormalize(entry?.type ?? entry?.Type);
                const ccStatusKey = (cat, entry) => `cc_${cat}_${entry?.name || ""}`;
                const ccResolveStatus = (entry, cat) => {
                    const transient = new Set(["queued", "downloading", "error", "auth_required"]);
                    const dls = ccState.downloadStatus[ccStatusKey(cat, entry)] || {};
                    return transient.has(dls.status) ? dls.status : (entry?.status || "missing");
                };
                const ccRelPath = (entry) => {
                    const localPath = String(entry?.local_path || "").replace(/\\/g, "/");
                    const parts = localPath.split("/").filter(Boolean);
                    if (parts.length >= 3 && parts[0] === "models") return parts.slice(2).join("/");
                    return parts[parts.length - 1] || "";
                };
                const ccEntries = (section, kind, predicate = null) => {
                    const entries = ccState.config?.[section] || [];
                    return entries.filter(entry => {
                        const kindOk = !kind || ccKind(entry) === ccNormalize(kind);
                        return kindOk && (!predicate || predicate(entry));
                    });
                };
                const localAssetRelSet = (items) => new Set((items || []).map(item => String(item || "").replace(/\\/g, "/")));
                const localAssetHas = (key, rel) => {
                    const normalized = String(rel || "").replace(/\\/g, "/");
                    const map = {
                        ckpt_name: localAssets.checkpoints,
                        diffusion_model_name: localAssets.diffusion_models,
                        clip_name: localAssets.text_encoders,
                        vae_name: localAssets.vae_models,
                    };
                    return localAssetRelSet(map[key] || []).has(normalized);
                };
                const mergeCcAndLocalEntries = (ccList, localNames, folder, type, kind) => {
                    const localSet = localAssetRelSet(localNames);
                    const seen = new Set();
                    const merged = [];

                    (ccList || []).forEach(entry => {
                        const rel = ccRelPath(entry);
                        if (!rel) return;
                        seen.add(rel);
                        merged.push({
                            ...entry,
                            status: localSet.has(rel) ? "installed" : entry.status,
                        });
                    });

                    (localNames || []).forEach(name => {
                        const rel = String(name || "").replace(/\\/g, "/");
                        if (!rel || seen.has(rel)) return;
                        seen.add(rel);
                        merged.push({
                            name: rel,
                            type,
                            kind,
                            local_path: `models/${folder}/${rel}`,
                            status: "installed",
                            description: "Local ComfyUI model.",
                            source: "local",
                        });
                    });

                    return merged;
                };
                const ccFirstEntry = (section, kind, predicate = null) => ccEntries(section, kind, predicate)[0] || null;
                const ccHasRequiredFamilies = (config) => {
                    const models = config?.models || [];
                    const clips = config?.clip || [];
                    const vaes = config?.vae || [];
                    const hasAnima = models.some(entry => ccKind(entry) === "anima")
                        && clips.some(entry => ccKind(entry) === "anima")
                        && vaes.some(entry => ccKind(entry) === "anima");
                    const hasIllustrious = models.some(entry => ccKind(entry) === "illustrious" && ccType(entry) === "checkpoint");
                    const hasQi2 = models.some(entry => ccKind(entry) === "qi2" && ccType(entry) === "unet")
                        && clips.some(entry => ccKind(entry) === "qi2")
                        && vaes.some(entry => ccKind(entry) === "qi2");
                    return hasAnima && hasIllustrious && hasQi2;
                };
                const { fetchConfig: fetchCcConfig, startPolling: startCcPolling, stopPolling: stopCcPolling } =
                    createControlCenterClient(CC_REPO_ID, ccHasRequiredFamilies, ccState, () => renderControlCenterCards());
                const ccDownloadEntry = async (cat, entry) => {
                    if (!entry?.name) return;
                    const key = ccStatusKey(cat, entry);
                    ccState.downloadStatus[key] = { status: "queued", message: "Queued…" };
                    renderControlCenterCards();
                    try {
                        const response = await api.fetchApi("/vnccs/control_center/download", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                            body: JSON.stringify({ repo_id: CC_REPO_ID, category: cat, name: entry.name }),
                        });
                        const payload = await response.json();
                        if (!response.ok || payload.error) {
                            ccState.downloadStatus[key] = { status: "error", message: payload.error || "Download failed" };
                            renderControlCenterCards();
                            return;
                        }
                        startCcPolling();
                    } catch (error) {
                        ccState.downloadStatus[key] = { status: "error", message: String(error?.message || error) };
                        renderControlCenterCards();
                    }
                };

                const els = {};
                const cloneSettingsValue = (value) => {
                    if (Array.isArray(value)) return value.map(item => ({ ...item }));
                    if (value && typeof value === "object") return { ...value };
                    return value;
                };

                const getGenerationDefaults = (mode) => ({
                    ...(mode === "qi2" ? QI2_DEFAULTS : mode === "anima" ? ANIMA_DEFAULTS : ILLUSTRIOUS_DEFAULTS),
                    seed: 0,
                });

                const ensureLoraStack = (profile) => {
                    if (!Array.isArray(profile.lora_stack)) profile.lora_stack = [];
                    profile.lora_stack = profile.lora_stack.filter(item => !isCreatorOverhaulLora(item?.name));
                    while (profile.lora_stack.length < 5) {
                        profile.lora_stack.push({ name: "", strength: 1.0 });
                    }
                    profile.lora_stack = profile.lora_stack.slice(0, 5).map(item => ({
                        name: item?.name || "",
                        strength: item?.strength ?? 1.0,
                    }));
                };

                const getModeProfile = (mode) => {
                    const normalizedMode = (mode || "illustrious").toLowerCase();
                    if (!state.gen_settings.mode_settings) state.gen_settings.mode_settings = {};
                    const defaults = getGenerationDefaults(normalizedMode);
                    const existing = state.gen_settings.mode_settings[normalizedMode] || {};
                    const profile = { ...defaults, ...existing };
                    const legacyTargetSize = normalizedMode === "anima" && existing.target_size == null
                        ? LEGACY_ANIMA_RESOLUTION_SCALES[String(existing.resolution_preset || "").toLowerCase()]
                        : null;
                    profile.target_size = resolutionScaleValue(
                        resolutionScaleMegapixels(legacyTargetSize ?? profile.target_size ?? defaults.target_size)
                    );
                    delete profile.resolution_preset;
                    if (normalizedMode === "qi2") {
                        profile.clip_type = "qwen_image";
                        profile.qi2_overhaul_strength = normalizeOverhaulStrength(profile.qi2_overhaul_strength);
                        profile.qi2_cache = {
                            device: profile.qi2_cache?.device || "gpu",
                            dtype: profile.qi2_cache?.dtype || "int8",
                        };
                        if (profile.turbo_enabled) {
                            profile.steps = 6;
                            profile.cfg = 1.0;
                        }
                    }
                    if (["illustrious", "anima", "qi2"].includes(normalizedMode)) ensureLoraStack(profile);
                    state.gen_settings.mode_settings[normalizedMode] = profile;
                    return profile;
                };

                const saveCurrentGenerationModeValues = (mode = state.gen_settings.generation_mode) => {
                    const normalizedMode = (mode || "illustrious").toLowerCase();
                    const profile = getModeProfile(normalizedMode);
                    (MODE_SETTING_KEYS[normalizedMode] || []).forEach((key) => {
                        if (state.gen_settings[key] !== undefined) {
                            profile[key] = cloneSettingsValue(state.gen_settings[key]);
                        }
                    });
                    if (["illustrious", "anima", "qi2"].includes(normalizedMode)) ensureLoraStack(profile);
                };

                const applyGenerationProfile = (mode) => {
                    const normalizedMode = (mode || "illustrious").toLowerCase();
                    const profile = getModeProfile(normalizedMode);
                    (MODE_SETTING_KEYS[normalizedMode] || []).forEach((key) => {
                        state.gen_settings[key] = cloneSettingsValue(profile[key]);
                    });
                    if (["illustrious", "anima", "qi2"].includes(normalizedMode)) ensureLoraStack(state.gen_settings);
                };

                const syncGenerationControls = () => {
                    const g = state.gen_settings;
                    if (els.ckptSelect) {
                        ensureOption(els.ckptSelect, g.ckpt_name);
                        els.ckptSelect.value = g.ckpt_name || "";
                    }
                    ["diffusion_model_name", "clip_name", "vae_name"].forEach((key) => {
                        if (!els[key]) return;
                        ensureOption(els[key], g[key]);
                        els[key].value = g[key] || "";
                    });
                    ensureOption(els.sampler, g.sampler);
                    ensureOption(els.scheduler, g.scheduler);
                    if (els.sampler) els.sampler.value = g.sampler || "";
                    if (els.scheduler) els.scheduler.value = g.scheduler || "";
                    if (els.target_size) {
                        els.target_size.value = resolutionScaleMegapixels(g.target_size).toFixed(1);
                    }
                    if (els.target_size_value) {
                        els.target_size_value.textContent = resolutionScaleText(g.target_size);
                    }
                    syncSliderValue(els.steps, g.steps);
                    syncSliderValue(els.cfg, g.cfg);
                    if (els.seed) els.seed.value = g.seed ?? "";
                    if (els.seed_mode) {
                        const randomMode = (g.seed_mode || "fixed") === "randomize";
                        els.seed_mode.classList.toggle("is-active", randomMode);
                        els.seed_mode.title = randomMode ? "Random seed" : "Fixed seed";
                        els.seed_mode.setAttribute("aria-pressed", randomMode ? "true" : "false");
                    }
                    if (els.dmdSelect) {
                        ensureOption(els.dmdSelect, g.dmd_lora_name);
                        els.dmdSelect.value = g.dmd_lora_name || "";
                    }
                    if (els.dmdSlider) {
                        const mode = (g.generation_mode || "illustrious").toLowerCase();
                        els.dmdSlider.checked = mode === "anima" || mode === "qi2"
                            ? !!g.turbo_enabled
                            : (g.dmd_lora_strength ?? 1.0) > 0;
                    }
                    if (els.qi2CacheDevice) els.qi2CacheDevice.value = g.qi2_cache?.device || "gpu";
                    if (els.qi2CacheDtype) els.qi2CacheDtype.value = g.qi2_cache?.dtype || "int8";
                    if (els.ageSelect) els.ageSelect.value = g.age_lora_name || "";
                    if (els.loraStackSelects) {
                        ensureLoraStack(g);
                        els.loraStackSelects.forEach((ref, i) => {
                            const item = g.lora_stack[i] || { name: "", strength: 1.0 };
                            ref.sel.value = item.name || "";
                            ref.rng.value = item.strength ?? 1.0;
                            ref.sel.closest(".vnccs-creator-lora-item")?.classList.toggle("is-empty", !item.name);
                        });
                    }
                    renderControlCenterCards();
                };

                const migrateGenerationModeSettings = () => {
                    const g = state.gen_settings;
                    g.generation_mode = (g.generation_mode || "illustrious").toLowerCase();
                    const currentMode = g.generation_mode;
                    const existingModes = g.mode_settings || {};
                    g.mode_settings = {
                        illustrious: { ...getGenerationDefaults("illustrious"), ...(existingModes.illustrious || {}) },
                        anima: { ...getGenerationDefaults("anima"), ...(existingModes.anima || {}) },
                        qi2: { ...getGenerationDefaults("qi2"), ...(existingModes.qi2 || {}) },
                    };
                    ensureLoraStack(g.mode_settings.illustrious);
                    ensureLoraStack(g.mode_settings.anima);
                    ensureLoraStack(g.mode_settings.qi2);

                    if ((g.generation_defaults_version || 0) < GENERATION_DEFAULTS_VERSION || !existingModes[currentMode]) {
                        const target = g.mode_settings[currentMode];
                        (MODE_SETTING_KEYS[currentMode] || []).forEach((key) => {
                            if (g[key] !== undefined && g[key] !== "") target[key] = cloneSettingsValue(g[key]);
                        });
                        if (["illustrious", "anima", "qi2"].includes(currentMode)) ensureLoraStack(target);
                    }

                    g.generation_defaults_version = GENERATION_DEFAULTS_VERSION;
                    applyGenerationProfile(currentMode);
                };

                const saveState = (isValid = false) => {
                    syncBackgroundForGenerationMode();
                    saveCurrentGenerationModeValues();
                    state.character_info.name = state.character || state.character_info.name || "";
                    const persistData = {
                        gen_settings: state.gen_settings,
                        character: state.character
                    };
                    storage.setItem("VNCCS_V2_Settings", JSON.stringify(persistData));

                    // Mark cache validity
                    state.preview_valid = isValid;

                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w) w.value = JSON.stringify(state);
                };

                const clearPreviewHandlers = () => {
                    if (!els.previewImg) return;
                    els.previewImg.onload = null;
                    els.previewImg.onerror = null;
                };

                const setPreviewLoading = (isLoading) => {
                    els.previewLoading?.classList.toggle("is-visible", !!isLoading);
                    if (els.spritePrevBtn) els.spritePrevBtn.disabled = !!isLoading;
                    if (els.spriteNextBtn) els.spriteNextBtn.disabled = !!isLoading;
                };

                const updateSpriteNav = () => {
                    const count = Number(state.sprite_preview_count || 0);
                    const visible = state.preview_source === "pose" && count > 1;
                    els.spriteNav?.classList.toggle("is-visible", visible);
                    if (els.spriteCount) {
                        const current = count ? ((Number(state.sprite_preview_index || 0) % count + count) % count) + 1 : 0;
                        els.spriteCount.textContent = count > 1 ? `${current}/${count}` : "";
                    }
                };

                const hideSpriteNav = () => {
                    state.sprite_preview_count = 0;
                    state.sprite_preview_index = 0;
                    updateSpriteNav();
                };

                const getDefaultCharacterInfo = () => ({
                    sex: "female", age: 18, framing: "cowboy_shot", style: DEFAULT_CHARACTER_STYLE, custom_style: "", race: "human", skin_color: "",
                    hair: "black hair, waist-length hair", eyes: "", face: "", body: "", additional_details: "",
                    nsfw: false, aesthetics: "masterpiece, best quality",
                    negative_prompt: "bad quality, worst quality",
                    lora_prompt: "", background_color: "Green"
                });

                const syncCharacterFields = () => {
                    Object.keys(state.character_info).forEach(k => {
                        if (els[k]) {
                            let val = state.character_info[k];
                            if (k === "background_color" && typeof val === "string" && val) {
                                val = val.charAt(0).toUpperCase() + val.slice(1).toLowerCase();
                                state.character_info[k] = val;
                            }
                            if (els[k].range && els[k].num) {
                                els[k].range.value = val;
                                els[k].num.value = val;
                            }
                            else if (els[k].setValue) els[k].setValue(val);
                            else if (els[k].type === "checkbox") els[k].checked = !!val;
                            else els[k].value = val;
                        }
                    });
                };

                const syncBackgroundForGenerationMode = (force = false) => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    const modelChanged = force || state.gen_settings.background_model_kind !== mode;
                    const current = String(state.character_info.background_color || "Green");
                    const isAlpha = ["alpha", "transparent"].includes(current.trim().toLowerCase());
                    if (mode === "qi2" && modelChanged) {
                        if (!isAlpha) {
                            state.gen_settings.previous_background_color = current;
                        }
                        state.character_info.background_color = "Transparent";
                    } else if (mode !== "qi2" && isAlpha) {
                        const previous = String(state.gen_settings.previous_background_color || "Green");
                        state.character_info.background_color = ["Green", "Blue"].includes(previous) ? previous : "Green";
                    }
                    state.gen_settings.background_model_kind = mode;
                    els.background_color?.setValue?.(state.character_info.background_color || "Green");
                    return modelChanged || current !== state.character_info.background_color;
                };

                const clearCharacterSelection = () => {
                    beginCharacterRequest();
                    beginCharacterWizardRequest();
                    beginPreviewRequest();
                    state.character = "";
                    restoredWidgetInfoCharacter = null;
                    Object.assign(state.character_info, getDefaultCharacterInfo(), { name: "" });
                    state.preview_valid = false;
                    state.preview_source = "gen";
                    state.sprite_preview_cache_bust = "";
                    clearPreviewHandlers();
                    hideSpriteNav();
                    setPreviewLoading(false);
                    if (els.previewImg) {
                        els.previewImg.removeAttribute("src");
                        els.previewImg.style.display = "none";
                    }
                    if (els.placeholder) {
                        els.placeholder.innerText = "Create a character to begin";
                        els.placeholder.style.display = "block";
                    }
                    syncCharacterFields();
                };

                const tryCachePreview = (character) => {
                    const currentRequest = beginPreviewRequest();
                    const isCurrent = () => currentRequest() && state.character === character;
                    console.log("[VNCCS] Trying to load cached preview...");
                    const cacheUrl = mediaURL(`/vnccs/get_cached_preview?character=${encodeURIComponent(character)}&t=${Date.now()}`);
                    clearPreviewHandlers();
                    hideSpriteNav();
                    setPreviewLoading(true);

                    const loader = new Image();
                    loader.onerror = () => {
                        if (!isCurrent()) return;
                        console.warn("[VNCCS] Both pose and cache preview failed.");
                        setPreviewLoading(false);
                        els.previewImg.style.display = "none";
                        els.placeholder.innerText = "No Preview Image";
                        els.placeholder.style.display = "block";
                        els.previewImg.onerror = null;
                        state.preview_valid = false;
                        state.preview_source = "gen";
                        hideSpriteNav();
                        saveState(false);
                    };
                    loader.onload = () => {
                        if (!isCurrent()) return;
                        setPreviewLoading(false);
                        els.previewImg.src = cacheUrl;
                        els.previewImg.style.display = "block";
                        els.placeholder.style.display = "none";
                        state.preview_valid = true;
                        state.preview_source = "gen";
                        hideSpriteNav();
                        saveState(true);
                    };

                    loader.src = cacheUrl;
                };

                const spritePreviewUrl = (character, index) => {
                    const cacheBust = state.sprite_preview_cache_bust || "current";
                    return mediaURL(`/vnccs/get_character_pose_preview?character=${encodeURIComponent(character)}&index=${index}&v=${encodeURIComponent(cacheBust)}`);
                };

                const prefetchSpritePreview = (character, index) => {
                    const count = Number(state.sprite_preview_count || 0);
                    if (!character || count <= 1) return;
                    const normalized = ((Number(index || 0) % count) + count) % count;
                    const prefetch = new Image();
                    prefetch.src = spritePreviewUrl(character, normalized);
                };

                const showSpritePreview = (character, index) => {
                    const count = Number(state.sprite_preview_count || 0);
                    if (!character || count <= 0) return;
                    const currentRequest = beginPreviewRequest();
                    const isCurrent = () => currentRequest() && state.character === character;
                    const normalized = ((Number(index || 0) % count) + count) % count;
                    const requestId = Number(state.sprite_preview_request_id || 0) + 1;
                    const url = spritePreviewUrl(character, normalized);
                    state.sprite_preview_request_id = requestId;
                    state.sprite_preview_index = normalized;
                    clearPreviewHandlers();
                    setPreviewLoading(true);

                    const loader = new Image();
                    loader.onerror = () => {
                        if (!isCurrent()) return;
                        console.warn("[VNCCS] Pose preview load failed. Fallback to cache.");
                        setPreviewLoading(false);
                        hideSpriteNav();
                        tryCachePreview(character);
                    };
                    loader.onload = () => {
                        if (!isCurrent()) return;
                        clearPreviewHandlers();
                        els.previewImg.src = url;
                        els.previewImg.style.display = "block";
                        els.placeholder.style.display = "none";
                        setPreviewLoading(false);
                        state.preview_valid = true;
                        state.preview_source = "pose";
                        updateSpriteNav();
                        prefetchSpritePreview(character, normalized - 1);
                        prefetchSpritePreview(character, normalized + 1);
                        saveState(true);
                    };
                    loader.src = url;
                    updateSpriteNav();
                };

                const applyStoredPrefs = (characterOnly = false) => {
                    try {
                        const s = storage.getItem("VNCCS_V2_Settings");
                        if (!s) return false;
                        const parsed = JSON.parse(s);
                        let changedCharacter = false;
                        if (parsed.character && parsed.character !== state.character) {
                            state.character = parsed.character;
                            restoredWidgetInfoCharacter = null;
                            changedCharacter = true;
                        }
                        if (!characterOnly) {
                            if (parsed.gen_settings) {
                                Object.assign(state.gen_settings, parsed.gen_settings);
                                if (parsed.prompt_modes) state.prompt_modes = parsed.prompt_modes;
                                if (parsed.prompt_defaults_version !== undefined) state.prompt_defaults_version = parsed.prompt_defaults_version;
                            } else {
                                Object.assign(state.gen_settings, parsed);
                            }
                            ensureLoraStack(state.gen_settings);
                        }
                        return changedCharacter;
                    } catch (e) {
                        return false;
                    }
                };

                const loadState = () => {
                    beginCharacterRequest();
                    beginCharacterWizardRequest();
                    // 1. Try Widget Data (Graph Persistence)
                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w && w.value && w.value !== "{}") {
                        try {
                            const parsed = JSON.parse(w.value);
                            // Merge from Graph Data
                            if (parsed.character) state.character = parsed.character;
                            if (parsed.prompt_modes) state.prompt_modes = parsed.prompt_modes;
                            if (parsed.prompt_defaults_version !== undefined) state.prompt_defaults_version = parsed.prompt_defaults_version;
                            if (parsed.character_info) {
                                const infoName = String(parsed.character_info.name || "").trim();
                                const parsedCharacter = String(parsed.character || state.character || "").trim();
                                if (infoName && infoName === parsedCharacter) {
                                    Object.assign(state.character_info, parsed.character_info);
                                    restoredWidgetInfoCharacter = parsedCharacter;
                                } else {
                                    console.warn("[VNCCS V2] Ignoring unverified widget character_info:", infoName || "(missing name)", "current:", parsedCharacter);
                                }
                            }
                            if (parsed.gen_settings) {
                                Object.assign(state.gen_settings, parsed.gen_settings);
                                ensureLoraStack(state.gen_settings);
                            }
                            if (parsed.preview_valid !== undefined) state.preview_valid = parsed.preview_valid;

                            console.log("[VNCCS V2] Loaded state from graph widget. Character:", state.character);
                            return;
                        } catch (e) { console.error("Error loading widget data", e); }
                    }

                    // 2. Fallback to LocalStorage (Global Preferences for new nodes)
                    applyStoredPrefs(false);
                };

                const syncSliderValue = (ref, value) => {
                    if (!ref) return;
                    if (ref.range) ref.range.value = value;
                    if (ref.num) ref.num.value = value;
                };

                const ensureOption = (el, value) => {
                    if (!el || !value) return;
                    const exists = Array.from(el.options).some(opt => opt.value === value);
                    if (!exists) el.add(new Option(value, value));
                };

                const splitPromptTokens = (value) => (value || "")
                    .split(",")
                    .map(token => token.trim())
                    .filter(Boolean);

                const joinPromptTokens = (tokens) => {
                    const seen = new Set();
                    const ordered = [];
                    tokens.forEach((token) => {
                        if (!seen.has(token)) {
                            seen.add(token);
                            ordered.push(token);
                        }
                    });
                    return ordered.join(", ");
                };

                const mergePromptDefaultWithCustom = (baseText, defaultText) => {
                    const baseTokens = splitPromptTokens(baseText);
                    const defaultTokens = splitPromptTokens(defaultText);
                    const defaultSet = new Set(defaultTokens);
                    const extraTokens = baseTokens.filter(token => !defaultSet.has(token));
                    return joinPromptTokens([...defaultTokens, ...extraTokens]);
                };

                const removePromptToken = (value, tokenToRemove) => joinPromptTokens(
                    splitPromptTokens(value).filter(token => token.toLowerCase() !== tokenToRemove.toLowerCase())
                );

                const applyPromptModeToFields = (mode) => {
                    const promptState = state.prompt_modes[mode] || MODE_PROMPT_DEFAULTS[mode];
                    state.character_info.aesthetics = promptState.aesthetics;
                    state.character_info.negative_prompt = promptState.negative_prompt;
                    if (els.aesthetics) els.aesthetics.value = promptState.aesthetics;
                    if (els.negative_prompt) els.negative_prompt.value = promptState.negative_prompt;
                };

                const saveCurrentPromptModeValues = () => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (!state.prompt_modes[mode]) state.prompt_modes[mode] = { ...MODE_PROMPT_DEFAULTS[mode] };
                    state.prompt_modes[mode].aesthetics = state.character_info.aesthetics || "";
                    state.prompt_modes[mode].negative_prompt = state.character_info.negative_prompt || "";
                };

                const migratePromptModes = () => {
                    const currentMode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    const existingModes = state.prompt_modes || {};
                    const mergedModes = {
                        illustrious: { ...MODE_PROMPT_DEFAULTS.illustrious, ...(existingModes.illustrious || {}) },
                        anima: { ...MODE_PROMPT_DEFAULTS.anima, ...(existingModes.anima || {}) },
                        qi2: { ...MODE_PROMPT_DEFAULTS.qi2, ...(existingModes.qi2 || {}) },
                    };

                    if ((state.prompt_defaults_version || 0) < PROMPT_DEFAULTS_VERSION) {
                        mergedModes.illustrious.aesthetics = state.character_info.aesthetics || mergedModes.illustrious.aesthetics;
                        mergedModes.illustrious.negative_prompt = state.character_info.negative_prompt || mergedModes.illustrious.negative_prompt;
                        mergedModes.anima.aesthetics = mergePromptDefaultWithCustom(mergedModes.anima.aesthetics, MODE_PROMPT_DEFAULTS.anima.aesthetics);
                        mergedModes.anima.aesthetics = removePromptToken(mergedModes.anima.aesthetics, "anime");
                        mergedModes.anima.negative_prompt = mergePromptDefaultWithCustom(mergedModes.anima.negative_prompt, MODE_PROMPT_DEFAULTS.anima.negative_prompt);
                        mergedModes.qi2.aesthetics = mergePromptDefaultWithCustom(mergedModes.qi2.aesthetics, MODE_PROMPT_DEFAULTS.qi2.aesthetics);
                        mergedModes.qi2.aesthetics = removePromptToken(mergedModes.qi2.aesthetics, "anime");
                        mergedModes.qi2.negative_prompt = mergePromptDefaultWithCustom(mergedModes.qi2.negative_prompt, MODE_PROMPT_DEFAULTS.qi2.negative_prompt);
                    }

                    state.prompt_modes = mergedModes;
                    state.prompt_defaults_version = PROMPT_DEFAULTS_VERSION;
                    applyPromptModeToFields(currentMode);
                };


                const refreshGenerationModeUI = () => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    const isAnima = mode === "anima";
                    const isQi2 = mode === "qi2";
                    syncBackgroundForGenerationMode();
                    if (els.modeTabs) {
                        Object.entries(els.modeTabs).forEach(([key, btn]) => {
                            btn.classList.toggle("is-active", key === mode);
                        });
                    }
                    if (els.illustriousModels) els.illustriousModels.style.display = !isAnima && !isQi2 ? "flex" : "none";
                    if (els.animaModels) els.animaModels.style.display = isAnima ? "flex" : "none";
                    if (els.qi2Models) els.qi2Models.style.display = isQi2 ? "flex" : "none";
                    if (els.qi2Cache) els.qi2Cache.style.display = isQi2 ? "grid" : "none";
                    if (els.loraSection) els.loraSection.style.display = "flex";
                    if (els.dmdWrap) els.dmdWrap.style.display = "none";
                    if (els.dmdLabel) els.dmdLabel.innerText = isAnima || isQi2 ? "Turbo LoRA" : "DMD2 LoRA Model";
                    if (els.loraHeader) els.loraHeader.innerText = isQi2 ? "Qwen Image 2.1 LoRA Stack" : isAnima ? "ANIMA LoRA Stack" : "LoRa Stack";
                    if (els.ageWrap) els.ageWrap.style.display = "none";
                    if (els.animaLoraCards) els.animaLoraCards.style.display = isAnima && els.animaLoraCards.children.length ? "flex" : "none";
                    if (els.qi2LoraCards) els.qi2LoraCards.style.display = isQi2 && els.qi2LoraCards.children.length ? "flex" : "none";
                    if (els.illustriousLoraCards) els.illustriousLoraCards.style.display = !isAnima && !isQi2 && els.illustriousLoraCards.children.length ? "flex" : "none";
                };

                const setGenerationMode = (mode) => {
                    const nextMode = (mode || "illustrious").toLowerCase();
                    const currentMode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    saveCurrentPromptModeValues();
                    saveCurrentGenerationModeValues(currentMode);
                    if (state.gen_settings.generation_mode === nextMode) {
                        applyGenerationProfile(nextMode);
                        applyPromptModeToFields(nextMode);
                        syncGenerationControls();
                        refreshGenerationModeUI();
                        return;
                    }
                    state.gen_settings.generation_mode = nextMode;
                    applyGenerationProfile(nextMode);
                    applyPromptModeToFields(nextMode);
                    syncGenerationControls();
                    refreshGenerationModeUI();
                    saveState();
                };

                const setAnimaTurboMode = (enabled) => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (!["anima", "qi2"].includes(mode)) return;
                    if (enabled) {
                        if (!state.gen_settings.turbo_enabled) {
                            state.gen_settings.turbo_previous_settings = {
                                steps: state.gen_settings.steps,
                                cfg: state.gen_settings.cfg,
                            };
                        }
                        state.gen_settings.turbo_enabled = true;
                        state.gen_settings.dmd_lora_strength = 1.0;
                        state.gen_settings.steps = mode === "qi2" ? 6 : 12;
                        state.gen_settings.cfg = 1.0;
                    } else {
                        state.gen_settings.turbo_enabled = false;
                        const previous = state.gen_settings.turbo_previous_settings || {};
                        state.gen_settings.steps = previous.steps ?? (mode === "qi2" ? 25 : 30);
                        state.gen_settings.cfg = previous.cfg ?? (mode === "qi2" ? 3.0 : 4.0);
                        state.gen_settings.turbo_previous_settings = null;
                    }
                    syncGenerationControls();
                    saveState();
                };

                const FIELD_HELP = {
                    background_color: "Sets the chroma key background color for generated character sheets. Use the color that is easiest to remove in your downstream workflow.",
                    sex: "Character gender profile used for prompt defaults and pose/body synchronization.",
                    nsfw: "Allows adult-oriented prompt details and generation behavior for this character.",
                    age: "Controls the character age used for prompt building and pose/body synchronization.",
                    framing: "Chooses whether the generated character uses cowboy-shot or full-body framing.",
                    style: "Selects a visual style template that is added to the character prompt.",
                    custom_style: "Custom visual style description added to the character prompt.",
                    race: "Species presets with automatic visual descriptions in the prompt. Combine species for hybrids or add custom traits; explicit traits override preset defaults.",
                    skin_color: "Natural and fantasy skin tones. Use the preset picker or enter a custom description.",
                    body: "Body type and silhouette details, including chest/body build tags.",
                    face: "Face-specific details such as freckles, scars, makeup, or other defining features.",
                    hair: "Hair color, color pattern, length, texture, style, and face-framing details.",
                    eyes: "Iris colors, eye shapes, pupils, and other eye features.",
                    additional_details: "Extra persistent character traits that should appear across outfits and emotions.",
                    aesthetics: "Visual style notes for the character, such as mood, fashion direction, or rendering flavor.",
                    generation_mode: "Chooses the generation backend profile. Illustrious uses checkpoint-style generation; Anima uses the Qwen/Anima stack.",
                    target_size: "Sets the generated portrait area from 1.0 to 4.0 megapixels while preserving its aspect ratio.",
                    ckpt_name: "Checkpoint used for Illustrious generation.",
                    diffusion_model_name: "Diffusion model used for Anima generation.",
                    clip_name: "CLIP/text encoder used by the Anima pipeline.",
                    vae_name: "VAE used to decode generated images.",
                    steps: "Number of sampling steps. Higher values can add detail but take longer.",
                    sampler: "Sampling algorithm used to denoise the image.",
                    cfg: "Prompt guidance strength. Higher values follow the prompt harder; too high can make images brittle.",
                    scheduler: "Noise schedule used together with the sampler.",
                    seed: "Numeric seed for reproducible generation. Reuse it to get similar results.",
                    seed_mode: "Toggles fixed seed versus a fresh random seed for each generation.",
                    dmd_lora_name: "Turbo/DMD LoRA used by the active generation profile.",
                    dmd_lora_strength: "Enables or disables the selected Turbo/DMD LoRA strength.",
                    age_lora_name: "Optional age helper LoRA applied to reinforce the selected age.",
                    lora_stack: "Additional LoRAs mixed into generation for style or character refinements."
                };
                const helpFor = (key, fallback = "") => FIELD_HELP[key] || fallback;

                // 4. UI Builders
                const createTraitField = (lbl, key, targetObj = state.character_info) => {
                    const { element, input } = createTraitInput({
                        prefix: "vnccs-creator", label: lbl, key, target: targetObj,
                        save: debouncedSave, choose: input => openTagConstructor(key, input),
                    });
                    setHelpText(element, helpFor(key));
                    els[key] = input;
                    return element;
                };

                const createField = (lbl, key, type = "text", opts = [], targetObj = state.character_info) => {
                    if (type === "text" && ["hair", "eyes", "race", "skin_color", "body", "face", "additional_details"].includes(key)) {
                        return createTraitField(lbl, key, targetObj);
                    }
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-field";
                    setHelpText(wrap, helpFor(key));

                    if (type === "checkbox") {
                        const toggleWrap = document.createElement("label");
                        toggleWrap.className = "vnccs-creator-toggle-wrap";

                        const toggle = document.createElement("div");
                        toggle.className = "vnccs-creator-toggle";

                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!targetObj[key];
                        inp.onchange = (e) => {
                            targetObj[key] = e.target.checked;
                            saveState();
                        };

                        const track = document.createElement("div");
                        track.className = "vnccs-creator-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "vnccs-creator-toggle-thumb";

                        toggle.appendChild(inp);
                        toggle.appendChild(track);
                        toggle.appendChild(thumb);

                        const l = document.createElement("span");
                        l.className = "vnccs-creator-toggle-label";
                        l.innerText = lbl;

                        toggleWrap.appendChild(toggle);
                        toggleWrap.appendChild(l);

                        wrap.appendChild(toggleWrap);
                        els[key] = inp;
                        return wrap;
                    }

                    // Header Row
                    const header = document.createElement("div");
                    header.style.display = "flex";
                    header.style.alignItems = "center";
                    header.style.justifyContent = "space-between";
                    header.innerHTML = `<div class="vnccs-creator-label">${lbl}</div>`;

                    wrap.appendChild(header);

                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-creator-select";
                        inp.setAttribute("aria-label", lbl);
                        opts.forEach(option => {
                            const value = typeof option === "object" ? option.value : option;
                            const label = typeof option === "object" ? option.label : option;
                            inp.add(new Option(label, value));
                        });
                        const fallback = typeof opts[0] === "object" ? opts[0]?.value : opts[0];
                        inp.value = targetObj[key] || fallback;
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-creator-input";
                        inp.type = "number";
                        if (opts.step) inp.step = opts.step;
                        inp.value = targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = parseFloat(e.target.value); saveState(); };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-creator-input";
                        inp.value = targetObj[key] || "";
                        inp.oninput = (e) => { targetObj[key] = e.target.value; debouncedSave(); };
                    }
                    els[key] = inp; // Register for updates
                    wrap.appendChild(inp);
                    return wrap;
                };

                const createSegmentedField = (lbl, key, options, targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-field";
                    setHelpText(wrap, helpFor(key));
                    wrap.innerHTML = `<div class="vnccs-creator-label">${lbl}</div>`;
                    const segmented = document.createElement("div");
                    segmented.className = "vnccs-creator-segmented-field";
                    if (options.length === 3) segmented.classList.add("is-three");
                    const buttons = [];
                    const setValue = (value, persist = false) => {
                        let normalized = String(value || options[0]?.value || "");
                        const supportsAlpha = (state.gen_settings.generation_mode || "").toLowerCase() === "qi2";
                        if (key === "background_color" && ["alpha", "transparent"].includes(normalized.trim().toLowerCase())) {
                            const previous = state.gen_settings.previous_background_color;
                            normalized = supportsAlpha ? "Transparent" : ["Green", "Blue"].includes(previous) ? previous : "Green";
                        }
                        targetObj[key] = normalized;
                        if (
                            persist
                            && key === "background_color"
                            && (state.gen_settings.generation_mode || "").toLowerCase() === "qi2"
                            && normalized !== "Transparent"
                        ) {
                            state.gen_settings.previous_background_color = normalized;
                        }
                        buttons.forEach(({ btn, value: btnValue }) => {
                            if (key === "background_color" && btnValue === "Transparent") {
                                btn.disabled = !supportsAlpha;
                                btn.title = supportsAlpha ? "Native transparency" : "Native transparency requires Qwen Image 2.1";
                            }
                            btn.classList.toggle("is-active", btnValue === normalized);
                            btn.setAttribute("aria-pressed", btnValue === normalized ? "true" : "false");
                        });
                        if (persist) saveState();
                    };
                    options.forEach(option => {
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "vnccs-creator-segmented-btn";
                        btn.textContent = option.label;
                        btn.onclick = () => setValue(option.value, true);
                        buttons.push({ btn, value: option.value });
                        segmented.appendChild(btn);
                    });
                    wrap.appendChild(segmented);
                    els[key] = { setValue, value: targetObj[key] || options[0]?.value || "" };
                    setValue(targetObj[key] || options[0]?.value);
                    return wrap;
                };

                const createStyleField = () => {
                    const picker = createStylePicker({
                        host: container, catalog: characterStyleCatalog,
                        getInfo: () => state.character_info, save: saveState,
                        fetchApi: (...args) => api.fetchApi(...args),
                        cleanup: callback => registerCleanup(node, callback),
                        showModal: (...args) => showCommonModal(...args),
                        imageURL: mediaURL,
                        getPreviewPayload: () => {
                            saveCurrentGenerationModeValues();
                            saveState();
                            return { node_id: String(node.id), character_info: state.character_info, gen_settings: state.gen_settings };
                        },
                        listenPreview: callback => {
                            api.addEventListener("vnccs.style_preview.stage", callback);
                            registerCleanup(node, () => api.removeEventListener("vnccs.style_preview.stage", callback));
                        },
                    });
                    setHelpText(picker.root, helpFor("style"));
                    els.style = { setValue: picker.setValue };
                    els.custom_style = picker.customInput;
                    return picker.root;
                };

                const createGraphicToggle = (lbl, key, targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-field";
                    setHelpText(wrap, helpFor(key));
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "vnccs-creator-graphic-toggle";
                    btn.innerHTML = `
                        <span class="vnccs-creator-graphic-toggle-text">
                            <span class="vnccs-creator-graphic-toggle-icon" aria-hidden="true">
                                <svg viewBox="0 0 24 24" width="15" height="15" fill="none">
                                    <path d="M12 3l7 4v5c0 4.5-2.8 7.4-7 9-4.2-1.6-7-4.5-7-9V7l7-4z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                                    <path d="M9 12h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </span>
                            ${lbl}
                        </span>
                        <span class="vnccs-creator-graphic-toggle-switch" aria-hidden="true"></span>
                    `;
                    const setValue = (value, persist = false) => {
                        const enabled = !!value;
                        targetObj[key] = enabled;
                        btn.classList.toggle("is-active", enabled);
                        btn.setAttribute("aria-pressed", enabled ? "true" : "false");
                        if (persist) saveState();
                    };
                    btn.onclick = () => setValue(!targetObj[key], true);
                    wrap.appendChild(btn);
                    els[key] = { type: "graphic-toggle", setValue, get checked() { return !!targetObj[key]; } };
                    setValue(!!targetObj[key]);
                    return wrap;
                };

                const createSlider = (lbl, key, min, max, step, targetObj = state.gen_settings) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-field";
                    setHelpText(wrap, helpFor(key));
                    wrap.innerHTML = `<div class="vnccs-creator-label">${lbl}</div>`;

                    const container = document.createElement("div");
                    container.className = "vnccs-creator-slider-container";

                    const range = document.createElement("input");
                    range.type = "range"; range.className = "vnccs-creator-slider";
                    range.min = min; range.max = max; range.step = step;
                    range.value = targetObj[key];

                    const num = document.createElement("input");
                    num.type = "number"; num.className = "vnccs-creator-slider-val";
                    num.step = step;
                    num.value = targetObj[key];

                    // Sync
                    range.oninput = (e) => {
                        num.value = e.target.value;
                        targetObj[key] = parseFloat(e.target.value);
                        debouncedSave();
                    };
                    num.onchange = (e) => {
                        let v = parseFloat(e.target.value);
                        if (v < min) v = min; if (v > max) v = max;
                        num.value = v; range.value = v;
                        targetObj[key] = v;
                        saveState();
                    };

                    container.appendChild(range);
                    container.appendChild(num);
                    wrap.appendChild(container);

                    els[key] = { range, num }; // composite ref
                    return wrap;
                };

                const createCompactNumberField = (lbl, key, min, max, step, targetObj = state.gen_settings) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-gen-param-field";
                    setHelpText(wrap, helpFor(key));
                    wrap.innerHTML = `<div class="vnccs-creator-label">${lbl}</div>`;

                    const input = document.createElement("input");
                    input.type = "number";
                    input.className = "vnccs-creator-gen-param-input";
                    input.min = min;
                    input.max = max;
                    input.step = step;
                    input.value = targetObj[key];
                    input.onchange = (e) => {
                        let value = parseFloat(e.target.value);
                        if (Number.isNaN(value)) value = targetObj[key] ?? min;
                        if (value < min) value = min;
                        if (value > max) value = max;
                        e.target.value = value;
                        targetObj[key] = value;
                        saveState();
                    };

                    wrap.appendChild(input);
                    els[key] = { num: input };
                    return wrap;
                };

                const createCompactSelectField = (lbl, key, targetObj = state.gen_settings) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-gen-param-field";
                    setHelpText(wrap, helpFor(key));
                    wrap.innerHTML = `<div class="vnccs-creator-label">${lbl}</div>`;

                    const select = document.createElement("select");
                    select.className = "vnccs-creator-gen-param-select";
                    select.onchange = (e) => {
                        targetObj[key] = e.target.value;
                        saveState();
                    };

                    wrap.appendChild(select);
                    els[key] = select;
                    return wrap;
                };

                const makeFallbackSelect = (label, key, targetObj = state.gen_settings) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-field";
                    setHelpText(wrap, helpFor(key));
                    wrap.innerHTML = `<div class="vnccs-creator-label">${label}</div>`;
                    const select = document.createElement("select");
                    select.className = "vnccs-creator-select";
                    select.onchange = (e) => {
                        targetObj[key] = e.target.value;
                        saveState();
                    };
                    wrap.appendChild(select);
                    els[key] = select;
                    return wrap;
                };

                const cardStatusLabel = (status, entry, cat) => {
                    const dls = ccState.downloadStatus[ccStatusKey(cat, entry)] || {};
                    if (status === "installed") return "Installed";
                    if (status === "queued") return "Queued";
                    if (status === "downloading") return dls.message || "Downloading";
                    if (status === "auth_required") return "Key Required";
                    if (status === "error") return "Error";
                    return "Missing";
                };

                const buildAssetCard = ({ entry, cat, selectedValue, onSelect, compact = false, toggled = false, onToggle = null, pickerHead = false, onDownload = null, displayName = null, nameAccessory = null }) => {
                    const status = ccResolveStatus(entry, cat);
                    const rel = ccRelPath(entry);
                    const installed = status === "installed";
                    const selected = selectedValue && rel && selectedValue.replace(/\\/g, "/") === rel;
                    const progress = ["queued", "downloading"].includes(status);

                    const card = document.createElement("div");
                    card.className = "vnccs-creator-model-card";
                    card.classList.toggle("is-picker-head", pickerHead);
                    card.classList.toggle("is-installed", installed);
                    card.classList.toggle("is-selected", selected || toggled);
                    card.classList.toggle("is-missing", !installed);
                    if (installed || pickerHead) {
                        card.onclick = () => onSelect?.(rel, entry);
                    }

                    const top = document.createElement("div");
                    top.className = "vnccs-creator-model-card-top";

                    const badge = document.createElement("span");
                    badge.className = "vnccs-creator-model-card-badge " + (installed ? "ok" : progress ? "progress" : "missing");
                    top.appendChild(badge);

                    const name = document.createElement("div");
                    name.className = "vnccs-creator-model-card-name";
                    name.textContent = displayName || entry.name || rel || "Unknown";
                    if (nameAccessory) {
                        const heading = document.createElement("div");
                        heading.className = "vnccs-creator-overhaul-heading";
                        heading.append(name, nameAccessory);
                        top.appendChild(heading);
                    } else {
                        top.appendChild(name);
                    }

                    const statusEl = document.createElement("div");
                    statusEl.className = "vnccs-creator-model-card-status " + (installed ? "ok" : progress ? "progress" : "missing");
                    statusEl.textContent = cardStatusLabel(status, entry, cat);
                    top.appendChild(statusEl);

                    if (onToggle && installed) {
                        const toggle = document.createElement("label");
                        toggle.className = "vnccs-creator-toggle vnccs-creator-model-card-toggle";
                        const input = document.createElement("input");
                        input.type = "checkbox";
                        input.checked = !!toggled;
                        input.onchange = (event) => {
                            event.stopPropagation();
                            onToggle(event.target.checked, rel, entry);
                        };
                        input.onclick = (event) => event.stopPropagation();
                        const track = document.createElement("div");
                        track.className = "vnccs-creator-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "vnccs-creator-toggle-thumb";
                        toggle.append(input, track, thumb);
                        top.appendChild(toggle);
                    }

                    card.appendChild(top);

                    if (entry.description && !compact) {
                        const desc = document.createElement("div");
                        desc.className = "vnccs-creator-model-card-desc";
                        desc.textContent = entry.description;
                        card.appendChild(desc);
                    }

                    if (!installed) {
                        const actions = document.createElement("div");
                        actions.className = "vnccs-creator-model-card-actions";
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "vnccs-creator-model-card-download";
                        btn.textContent = status === "auth_required" ? "Enter Key in Control Center" : "Download";
                        btn.disabled = progress;
                        btn.onclick = (event) => {
                            event.stopPropagation();
                            if (!progress && status !== "auth_required") (onDownload || ccDownloadEntry)(cat, entry);
                        };
                        actions.appendChild(btn);
                        card.appendChild(actions);
                    }

                    return card;
                };

                let closeOverhaulHelp = null;
                let overhaulHelpSequence = 0;
                const buildOverhaulHelp = () => {
                    const help = document.createElement("div");
                    help.className = "vnccs-creator-overhaul-help";
                    const button = document.createElement("button");
                    button.type = "button";
                    button.className = "vnccs-creator-overhaul-info";
                    button.setAttribute("aria-label", `About ${QI2_OVERHAUL_TITLE}`);
                    button.setAttribute("aria-expanded", "false");
                    button.innerHTML = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="10" cy="10" r="8"/><path d="M10 9v5"/><circle cx="10" cy="6" r="0.8" fill="currentColor" stroke="none"/></svg>';
                    const tooltip = document.createElement("div");
                    tooltip.className = "vnccs-creator-overhaul-tooltip";
                    tooltip.id = `vnccs-overhaul-help-${node.id}-${++overhaulHelpSequence}`;
                    tooltip.setAttribute("role", "tooltip");
                    tooltip.setAttribute("popover", "auto");
                    tooltip.textContent = "Character Overhaul helps Qwen Image 2.1 follow detailed character prompts more closely, including anatomy, colors, and small identifying features. It can also influence the visual style. Start with the recommended strength of 0.5. If requested details are missing or inaccurate, try 0.75 or 1. If the result drifts too far from your chosen style, lower the strength to 0.25 or 0. A value of 0 disables this LoRA. Compare results using the same prompt and seed to judge the balance between detail accuracy and style fidelity.";
                    button.setAttribute("aria-describedby", tooltip.id);
                    button.setAttribute("aria-controls", tooltip.id);
                    let pinned = false;
                    const close = () => {
                        pinned = false;
                        if (tooltip.matches(":popover-open")) tooltip.hidePopover();
                        button.setAttribute("aria-expanded", "false");
                        if (closeOverhaulHelp === close) closeOverhaulHelp = null;
                    };
                    const show = () => {
                        if (closeOverhaulHelp && closeOverhaulHelp !== close) closeOverhaulHelp();
                        closeOverhaulHelp = close;
                        if (!tooltip.matches(":popover-open")) tooltip.showPopover();
                        button.setAttribute("aria-expanded", "true");
                        const anchor = button.getBoundingClientRect();
                        const bounds = tooltip.getBoundingClientRect();
                        // Convert screen coordinates to the node's CSS zoom scale.
                        const scale = bounds.width / tooltip.offsetWidth || 1;
                        const left = Math.max(12, Math.min(anchor.left, window.innerWidth - bounds.width - 12));
                        const below = anchor.bottom + 8;
                        const top = below + bounds.height <= window.innerHeight - 12
                            ? below : Math.max(12, anchor.top - bounds.height - 8);
                        tooltip.style.left = `${left / scale}px`;
                        tooltip.style.top = `${top / scale}px`;
                    };
                    help.onpointerenter = show;
                    help.onpointerleave = () => { if (!pinned) close(); };
                    button.onfocus = show;
                    button.onclick = event => {
                        event.stopPropagation();
                        if (pinned) close();
                        else { show(); pinned = true; }
                    };
                    help.onfocusout = event => { if (!help.contains(event.relatedTarget)) close(); };
                    help.onkeydown = event => {
                        if (event.key === "Escape") { event.stopPropagation(); close(); }
                    };
                    tooltip.ontoggle = event => {
                        if (event.newState === "closed") close();
                    };
                    help.append(button, tooltip);
                    return help;
                };

                const buildOverhaulCard = (entry) => {
                    const strength = normalizeOverhaulStrength(state.gen_settings.qi2_overhaul_strength);
                    const card = buildAssetCard({
                        entry, cat: "lora", compact: true,
                        displayName: QI2_OVERHAUL_TITLE, toggled: strength > 0, nameAccessory: buildOverhaulHelp(),
                    });
                    const control = document.createElement("div");
                    control.className = "vnccs-creator-overhaul-control";
                    const slider = document.createElement("input");
                    slider.className = "vnccs-creator-slider vnccs-creator-overhaul-slider";
                    slider.type = "range";
                    slider.min = "0";
                    slider.max = "1";
                    slider.step = "0.25";
                    slider.value = String(strength);
                    slider.setAttribute("aria-label", `${QI2_OVERHAUL_TITLE} strength`);
                    const updateValue = () => {
                        const next = normalizeOverhaulStrength(slider.value);
                        slider.value = String(next);
                        slider.setAttribute("aria-valuetext", next === 0 ? "0 — Off" : String(next));
                        card.classList.toggle("is-selected", next > 0);
                        return next;
                    };
                    updateValue();
                    slider.oninput = () => {
                        state.gen_settings.qi2_overhaul_strength = updateValue();
                        saveState();
                    };
                    const ticks = document.createElement("div");
                    ticks.className = "vnccs-creator-overhaul-ticks";
                    ticks.setAttribute("aria-hidden", "true");
                    [0, 0.25, 0.5, 0.75, 1].forEach(step => {
                        const tick = document.createElement("span");
                        tick.textContent = String(step);
                        ticks.appendChild(tick);
                    });
                    control.append(slider, ticks);
                    card.appendChild(control);
                    return card;
                };

                const selectCcAsset = (key, rel) => {
                    if (!rel) return;
                    state.gen_settings[key] = rel;
                    if (els[key]) {
                        ensureOption(els[key], rel);
                        els[key].value = rel;
                    }
                    saveState();
                    renderControlCenterCards();
                };

                const ensureAnimaDefaultAux = () => {
                    const clip = ccFirstEntry("clip", "Anima");
                    const vae = ccFirstEntry("vae", "Anima");
                    if (clip) state.gen_settings.clip_name = ccRelPath(clip);
                    else if (!state.gen_settings.clip_name) state.gen_settings.clip_name = ANIMA_CLIP_NAME;
                    if (vae) state.gen_settings.vae_name = ccRelPath(vae);
                    else if (!state.gen_settings.vae_name) state.gen_settings.vae_name = ANIMA_VAE_NAME;
                    if (els.clip_name) {
                        ensureOption(els.clip_name, state.gen_settings.clip_name);
                        els.clip_name.value = state.gen_settings.clip_name || "";
                    }
                    if (els.vae_name) {
                        ensureOption(els.vae_name, state.gen_settings.vae_name);
                        els.vae_name.value = state.gen_settings.vae_name || "";
                    }
                };

                const ensureQi2DefaultAux = () => {
                    const clip = ccFirstEntry("clip", "QI2");
                    const vae = ccFirstEntry("vae", "QI2");
                    state.gen_settings.clip_name = clip ? ccRelPath(clip) : (state.gen_settings.clip_name || QI2_CLIP_NAME);
                    state.gen_settings.vae_name = vae ? ccRelPath(vae) : (state.gen_settings.vae_name || QI2_VAE_NAME);
                    state.gen_settings.clip_type = "qwen_image";
                    state.gen_settings.qi2_cache = {
                        device: state.gen_settings.qi2_cache?.device || "gpu",
                        dtype: state.gen_settings.qi2_cache?.dtype || "int8",
                    };
                    if (els.clip_name) {
                        ensureOption(els.clip_name, state.gen_settings.clip_name);
                        els.clip_name.value = state.gen_settings.clip_name || "";
                    }
                    if (els.vae_name) {
                        ensureOption(els.vae_name, state.gen_settings.vae_name);
                        els.vae_name.value = state.gen_settings.vae_name || "";
                    }
                };

                const selectAnimaModel = (rel) => {
                    ensureAnimaDefaultAux();
                    modelPickerOpen.anima = false;
                    selectCcAsset("diffusion_model_name", rel);
                };

                const selectIllustriousModel = (rel) => {
                    modelPickerOpen.illustrious = false;
                    selectCcAsset("ckpt_name", rel);
                };

                const selectQi2Model = (rel) => {
                    ensureQi2DefaultAux();
                    modelPickerOpen.qi2 = false;
                    selectCcAsset("diffusion_model_name", rel);
                };

                const downloadAnimaBundle = async (cat, entry) => {
                    ensureAnimaDefaultAux();
                    await ccDownloadEntry(cat, entry);
                    const clip = ccFirstEntry("clip", "Anima");
                    const vae = ccFirstEntry("vae", "Anima");
                    if (clip && ccResolveStatus(clip, "clip") !== "installed") await ccDownloadEntry("clip", clip);
                    if (vae && ccResolveStatus(vae, "vae") !== "installed") await ccDownloadEntry("vae", vae);
                };

                const downloadQi2Bundle = async (cat, entry) => {
                    ensureQi2DefaultAux();
                    await ccDownloadEntry(cat, entry);
                    const clip = ccFirstEntry("clip", "QI2");
                    const vae = ccFirstEntry("vae", "QI2");
                    if (clip && ccResolveStatus(clip, "clip") !== "installed") await ccDownloadEntry("clip", clip);
                    if (vae && ccResolveStatus(vae, "vae") !== "installed") await ccDownloadEntry("vae", vae);
                };

                const setCcTurboMode = (enabled, rel) => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (mode === "anima" || mode === "qi2") {
                        state.gen_settings.dmd_lora_name = rel || state.gen_settings.dmd_lora_name || "";
                        setAnimaTurboMode(enabled);
                    } else {
                        if (enabled) {
                            if ((state.gen_settings.dmd_lora_strength || 0) <= 0) {
                                state.gen_settings.turbo_previous_settings = {
                                    steps: state.gen_settings.steps,
                                    cfg: state.gen_settings.cfg,
                                };
                            }
                            state.gen_settings.dmd_lora_name = rel || state.gen_settings.dmd_lora_name || "";
                            state.gen_settings.dmd_lora_strength = 1.0;
                            state.gen_settings.steps = 4;
                            state.gen_settings.cfg = 1.0;
                        } else {
                            state.gen_settings.dmd_lora_name = "";
                            state.gen_settings.dmd_lora_strength = 0.0;
                            const previous = state.gen_settings.turbo_previous_settings || {};
                            if (previous.steps !== undefined) state.gen_settings.steps = previous.steps;
                            if (previous.cfg !== undefined) state.gen_settings.cfg = previous.cfg;
                            state.gen_settings.turbo_previous_settings = null;
                        }
                        syncGenerationControls();
                        saveState();
                    }
                    renderControlCenterCards();
                };

                const setCcAgeLora = (enabled, rel) => {
                    state.gen_settings.age_lora_name = enabled ? rel : "";
                    if (els.ageSelect) {
                        ensureOption(els.ageSelect, state.gen_settings.age_lora_name);
                        els.ageSelect.value = state.gen_settings.age_lora_name || "";
                    }
                    saveState();
                    renderControlCenterCards();
                };

                const renderModeLoraCards = (containerEl, mode) => {
                    if (!containerEl) return;
                    containerEl.innerHTML = "";
                    const kindOk = (entry) => {
                        const kind = ccKind(entry);
                        if (mode === "anima") return kind === "anima";
                        if (mode === "qi2") return kind === "qi2";
                        return kind === "sdxl" || kind === "illustrious";
                    };
                    const turboEntries = (ccState.config?.lora || []).filter(entry => kindOk(entry) && ccType(entry) === "turbolora");
                    const ageEntries = (ccState.config?.lora || []).filter(entry => kindOk(entry) && ccType(entry) === "ageslider");

                    const addGroup = (title, entries, renderer) => {
                        if (!entries.length) return;
                        const group = document.createElement("div");
                        group.className = "vnccs-creator-subsection";
                        const label = document.createElement("div");
                        label.className = "vnccs-creator-label";
                        label.textContent = title;
                        group.appendChild(label);
                        entries.forEach(entry => group.appendChild(renderer(entry)));
                        containerEl.appendChild(group);
                    };

                    addGroup("Turbo LoRA", turboEntries, entry => {
                        const rel = ccRelPath(entry);
                        const enabled = mode === "anima" || mode === "qi2"
                            ? !!state.gen_settings.turbo_enabled && String(state.gen_settings.dmd_lora_name || "").replace(/\\/g, "/") === rel
                            : (state.gen_settings.dmd_lora_strength || 0) > 0 && String(state.gen_settings.dmd_lora_name || "").replace(/\\/g, "/") === rel;
                        return buildAssetCard({
                            entry,
                            cat: "lora",
                            selectedValue: state.gen_settings.dmd_lora_name || "",
                            compact: true,
                            toggled: enabled,
                            onSelect: () => setCcTurboMode(!enabled, rel, entry),
                            onToggle: checked => setCcTurboMode(checked, rel, entry),
                        });
                    });

                    if (mode === "qi2") {
                        const entry = (ccState.config?.lora || []).find(item =>
                            ccKind(item) === "qi2" && ccType(item) === "helper"
                            && (item.name === QI2_OVERHAUL_ENTRY.name || isCreatorOverhaulLora(ccRelPath(item)))
                        ) || QI2_OVERHAUL_ENTRY;
                        const installed = localAssetRelSet(localAssets.loras).has(ccRelPath(entry));
                        containerEl.appendChild(buildOverhaulCard(installed ? { ...entry, status: "installed" } : entry));
                    }

                    addGroup("Age LoRA", ageEntries, entry => {
                        const rel = ccRelPath(entry);
                        return buildAssetCard({
                            entry,
                            cat: "lora",
                            selectedValue: state.gen_settings.age_lora_name || "",
                            compact: true,
                            onSelect: () => setCcAgeLora(true, rel, entry),
                        });
                    });

                    const isCurrentMode = (state.gen_settings.generation_mode || "illustrious").toLowerCase() === mode;
                    containerEl.style.display = isCurrentMode && containerEl.children.length ? "flex" : "none";
                };

                const renderModelPicker = ({ containerEl, entries, cat, key, mode, emptyText, onSelect, onDownload = null }) => {
                    if (!containerEl) return;
                    containerEl.innerHTML = "";
                    const picker = document.createElement("div");
                    picker.className = "vnccs-creator-model-picker";
                    picker.classList.toggle("is-open", !!modelPickerOpen[mode]);
                    containerEl.appendChild(picker);

                    if (!entries.length) {
                        const empty = document.createElement("div");
                        empty.className = "vnccs-creator-model-card-desc";
                        empty.textContent = emptyText;
                        picker.appendChild(empty);
                        return;
                    }

                    const current = String(state.gen_settings[key] || "").replace(/\\/g, "/");
                    const selectedEntry = entries.find(entry => ccRelPath(entry) === current)
                        || entries[0];

                    picker.appendChild(buildAssetCard({
                        entry: selectedEntry,
                        cat,
                        selectedValue: ccRelPath(selectedEntry),
                        pickerHead: true,
                        onSelect: () => {
                            modelPickerOpen[mode] = !modelPickerOpen[mode];
                            renderControlCenterCards();
                        },
                        onDownload,
                    }));

                    const menu = document.createElement("div");
                    menu.className = "vnccs-creator-model-picker-menu";
                    picker.appendChild(menu);

                    const appendGroup = (title, groupEntries) => {
                        if (!groupEntries.length) return;
                        const group = document.createElement("div");
                        group.className = "vnccs-creator-model-picker-group";
                        const groupTitle = document.createElement("div");
                        groupTitle.className = "vnccs-creator-model-picker-group-title";
                        groupTitle.textContent = title;
                        group.appendChild(groupTitle);
                        groupEntries.forEach(entry => {
                            group.appendChild(buildAssetCard({
                                entry,
                                cat,
                                selectedValue: state.gen_settings[key] || "",
                                onSelect,
                                onDownload,
                            }));
                        });
                        menu.appendChild(group);
                    };

                    appendGroup("VNCCS Models", entries.filter(entry => entry.source !== "local"));
                    appendGroup("User Models", entries.filter(entry => entry.source === "local"));
                };

                const renderControlCenterCards = () => {
                    closeOverhaulHelp?.();
                    if (!els.animaModelCards && !els.illustriousModelCards && !els.qi2ModelCards) return;
                    const currentMode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    const isAnimaMode = currentMode === "anima";
                    const isQi2Mode = currentMode === "qi2";
                    if (els.animaFallback) {
                        els.animaFallback.style.display = "none";
                    }
                    [els.animaModelCards, els.qi2ModelCards].forEach(containerEl => {
                        if (containerEl) containerEl.style.display = "flex";
                    });

                    const animaModels = mergeCcAndLocalEntries(
                        ccEntries("models", "Anima", entry => ccType(entry) === "unet"),
                        localAssets.diffusion_models,
                        "diffusion_models",
                        "unet",
                        "Anima",
                    );
                    renderModelPicker({
                        containerEl: els.animaModelCards,
                        entries: animaModels,
                        cat: "models",
                        key: "diffusion_model_name",
                        mode: "anima",
                        emptyText: "No Anima diffusion models found.",
                        onSelect: rel => selectAnimaModel(rel),
                        onDownload: downloadAnimaBundle,
                    });

                    const installedAnimaDefaults = [
                        ["diffusion_model_name", animaModels, "models"],
                    ];
                    installedAnimaDefaults.forEach(([key, entries, cat]) => {
                        const current = String(state.gen_settings[key] || "").replace(/\\/g, "/");
                        const firstEntry = entries[0];
                        if (!current && firstEntry && ccState.config) {
                            selectAnimaModel(ccRelPath(firstEntry));
                        }
                    });
                    ensureAnimaDefaultAux();

                    const qi2Models = mergeCcAndLocalEntries(
                        ccEntries("models", "QI2", entry => ccType(entry) === "unet"),
                        localAssets.diffusion_models,
                        "diffusion_models",
                        "unet",
                        "QI2",
                    );
                    renderModelPicker({
                        containerEl: els.qi2ModelCards,
                        entries: qi2Models,
                        cat: "models",
                        key: "diffusion_model_name",
                        mode: "qi2",
                        emptyText: "No Qwen Image 2.1 diffusion models found.",
                        onSelect: rel => selectQi2Model(rel),
                        onDownload: downloadQi2Bundle,
                    });
                    const currentQi2 = String(state.gen_settings.diffusion_model_name || "").replace(/\\/g, "/");
                    if (isQi2Mode && !currentQi2 && qi2Models[0] && ccState.config) {
                        selectQi2Model(ccRelPath(qi2Models[0]));
                    }
                    if (isQi2Mode) ensureQi2DefaultAux();

                    if (els.animaLoraCards) {
                        if (!isAnimaMode) {
                            els.animaLoraCards.innerHTML = "";
                            els.animaLoraCards.style.display = "none";
                        } else {
                            renderModeLoraCards(els.animaLoraCards, "anima");
                        }
                    }
                    if (els.qi2LoraCards) {
                        if (!isQi2Mode) {
                            els.qi2LoraCards.innerHTML = "";
                            els.qi2LoraCards.style.display = "none";
                        } else {
                            renderModeLoraCards(els.qi2LoraCards, "qi2");
                        }
                    }
                    if (els.illustriousLoraCards) {
                        if (isAnimaMode || isQi2Mode) {
                            els.illustriousLoraCards.innerHTML = "";
                            els.illustriousLoraCards.style.display = "none";
                        } else {
                            renderModeLoraCards(els.illustriousLoraCards, "illustrious");
                        }
                    }

                    const illustriousDefaults = (ccState.config?.models || []).filter(entry => {
                        const kind = ccKind(entry);
                        return (kind === "illustrious" || kind === "sdxl") && ccType(entry) === "checkpoint";
                    });
                    const illustriousCkpts = mergeCcAndLocalEntries(
                        illustriousDefaults,
                        localAssets.checkpoints,
                        "checkpoints",
                        "checkpoint",
                        "Illustrious",
                    );
                    if (els.illustriousModelCards) {
                        els.illustriousModelCards.innerHTML = "";
                        if (illustriousCkpts.length) {
                            els.illustriousFallback.style.display = "none";
                            const firstEntry = illustriousCkpts[0];
                            if (!state.gen_settings.ckpt_name && firstEntry && ccState.config) {
                                selectCcAsset("ckpt_name", ccRelPath(firstEntry));
                            }
                            renderModelPicker({
                                containerEl: els.illustriousModelCards,
                                entries: illustriousCkpts,
                                cat: "models",
                                key: "ckpt_name",
                                mode: "illustrious",
                                emptyText: "No Illustrious checkpoints found.",
                                onSelect: rel => selectIllustriousModel(rel),
                            });
                            els.illustriousModelCards.style.display = "flex";
                        } else {
                            els.illustriousFallback.style.display = "none";
                            els.illustriousModelCards.style.display = "flex";
                            const empty = document.createElement("div");
                            empty.className = "vnccs-creator-model-card-desc";
                            empty.textContent = "No checkpoints found.";
                            els.illustriousModelCards.appendChild(empty);
                        }
                    }
                    if (els.qi2Models) els.qi2Models.style.display = isQi2Mode ? "flex" : "none";
                    if (els.animaModels) els.animaModels.style.display = isAnimaMode ? "flex" : "none";
                    if (els.illustriousModels) els.illustriousModels.style.display = !isAnimaMode && !isQi2Mode ? "flex" : "none";
                };

                const isSelectedCcAssetInstalled = (section, kind, key, predicate = null) => {
                    const selected = String(state.gen_settings[key] || "").replace(/\\/g, "/");
                    if (!selected) return false;
                    if (!ccState.config) return true;
                    const entries = ccEntries(section, kind, predicate);
                    if (!entries.length) return true;
                    const match = entries.find(entry => ccRelPath(entry) === selected);
                    if (!match) return localAssetHas(key, selected);
                    return ccResolveStatus(match, section === "models" ? "models" : section) === "installed";
                };

                // 5. Build Layout
                const container = document.createElement("div");
                container.className = "vnccs-creator-container";
                enableMiddleMouseCanvasPan(container);
                attachHelpTooltips(container);
                container.addEventListener("scroll", () => closeOverhaulHelp?.(), true);
                registerCleanup(node, () => closeOverhaulHelp?.());

                // --- TOP ROW ---
                const topRow = document.createElement("div");
                topRow.className = "vnccs-creator-top-row";

                // COL 1: LEFT (Preview)
                const colLeft = document.createElement("div");
                colLeft.className = "vnccs-creator-col";
                colLeft.innerHTML = '<div class="vnccs-creator-section-title">Character Select</div>';

                const charSel = document.createElement("select"); charSel.className = "vnccs-creator-select";
                charSel.onchange = async (e) => {
                    await loadChar(e.target.value);
                };
                els.charSelect = charSel;
                colLeft.appendChild(charSel);

                const btnRow = document.createElement("div");
                btnRow.className = "vnccs-creator-btn-row";
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-creator-btn vnccs-creator-btn-primary vnccs-creator-btn-generate";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.onclick = () => doGenerate();
                els.btnGen = btnGen;

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-creator-btn vnccs-creator-btn-success vnccs-creator-btn-secondary";
                btnNew.innerText = "NEW";
                btnNew.onclick = () => doCreate();

                const btnDel = document.createElement("button");
                btnDel.className = "vnccs-creator-btn vnccs-creator-btn-danger vnccs-creator-btn-secondary";
                btnDel.innerText = "DELETE";
                // Modal Helper — delegates to vnccs_common showModal
                const showModal = (title, contentFunc, buttons) => {
                    // Map widget-specific button classes to common classes
                    const mappedButtons = buttons.map(b => ({
                        ...b,
                        class: b.class?.includes("danger") ? "danger" : b.class?.includes("primary") ? "primary" : undefined
                    }));
                    return showCommonModal(container, title, contentFunc, mappedButtons);
                };
                const showAlertModal = (title, message) => {
                    showModal(title, () => {
                        const d = document.createElement("div");
                        d.style.whiteSpace = "pre-wrap";
                        d.style.lineHeight = "1.45";
                        d.textContent = String(message ?? "");
                        return d;
                    }, [{ text: "OK", class: "vnccs-creator-btn-primary" }]);
                };

                const applyCharacterWizardData = (data) => {
                    const textKeys = ["race", "skin_color", "body", "face", "hair", "eyes", "additional_details"];
                    if (data.sex) {
                        const sex = String(data.sex).toLowerCase().startsWith("m") ? "male" : "female";
                        state.character_info.sex = sex;
                        els.sex?.setValue?.(sex);
                    }
                    if (data.age !== undefined) {
                        const age = Math.max(1, Math.min(100, parseInt(data.age, 10) || 18));
                        state.character_info.age = age;
                        if (els.age?.range && els.age?.num) {
                            els.age.range.value = age;
                            els.age.num.value = age;
                        }
                    }
                    textKeys.forEach((key) => {
                        const value = data[key] || "";
                        state.character_info[key] = value;
                        if (els[key]?.setValue) els[key].setValue(value);
                        else if (els[key]) els[key].value = value;
                    });
                    state.preview_valid = false;
                    saveState(false);
                };

                const showCharacterWizardError = (err) => {
                    showModal("Character Wizzard Error", () => {
                        const d = document.createElement("div");
                        d.className = "vnccs-creator-character-wizard-modal";
                        const text = document.createElement("div");
                        text.className = "vnccs-creator-character-wizard-modal-text";
                        text.innerText = err?.message || err?.raw || "Failed to generate character description.";
                        d.appendChild(text);
                        return d;
                    }, [{ text: "OK", class: "vnccs-creator-btn-primary" }]);
                };

                const ensureQwenVLReady = createQwenVLModelLoader(node, showModal, () => {
                    const { overlay, modal } = showModal("Downloading QwenVL...", () => {
                        const d = document.createElement("div");
                        d.className = "vnccs-creator-character-wizard-modal";
                        d.innerHTML = `
                            <div class="vnccs-creator-qwenvl-download-status" id="vnccs-qwenvl-status">Preparing model files...</div>
                            <div class="vnccs-creator-qwenvl-download-track">
                                <div class="vnccs-creator-qwenvl-download-bar" id="vnccs-qwenvl-bar"></div>
                            </div>
                            <div class="vnccs-creator-qwenvl-download-pct" id="vnccs-qwenvl-pct">0%</div>
                        `;
                        return d;
                    }, []);

                    const statusEl = modal.querySelector("#vnccs-qwenvl-status");
                    const barEl = modal.querySelector("#vnccs-qwenvl-bar");
                    const pctEl = modal.querySelector("#vnccs-qwenvl-pct");
                    return { overlay, statusEl, barEl, pctEl, completedMessage: "QwenVL ready." };
                }, false);

                const openCharacterWizard = () => {
                    const character = state.character;
                    const currentRequest = beginCharacterWizardRequest();
                    let input;
                    showModal("Character Wizzard", () => {
                        const wrap = document.createElement("div");
                        wrap.className = "vnccs-creator-character-wizard-modal";
                        const text = document.createElement("div");
                        text.className = "vnccs-creator-character-wizard-modal-text";
                        text.innerText = "Describe the character in a broad way. The model will expand it into the creator fields and prefer tags from the tag constructor.";
                        input = document.createElement("textarea");
                        input.className = "vnccs-creator-textarea";
                        input.placeholder = "e.g. adult demon girl with long white hair, red eyes, elegant sharp face";
                        wrap.append(text, input);
                        setTimeout(() => input.focus(), 50);
                        return wrap;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "FILL FIELDS",
                            class: "vnccs-creator-btn-primary",
                            action: async (_overlay, btn) => {
                                const isCurrent = () => currentRequest() && _overlay.isConnected && state.character === character;
                                if (!isCurrent()) return false;
                                const description = input.value.trim();
                                if (!description) {
                                    input.focus();
                                    return true;
                                }
                                btn.disabled = true;
                                btn.innerText = "CHECKING MODEL...";
                                try {
                                    if (!await ensureQwenVLReady()) return true;
                                    if (!isCurrent()) return false;
                                    btn.innerText = "THINKING...";
                                    const r = await api.fetchApi("/vnccs/character_wizard", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ description, node_id: node.id })
                                    });
                                    if (!isCurrent()) return false;
                                    if (!r.ok) {
                                        let err = null;
                                        try { err = await r.json(); } catch (e) { err = { message: await r.text() }; }
                                        if (!isCurrent()) return false;
                                        showCharacterWizardError(err);
                                        return false;
                                    }
                                    const data = await r.json();
                                    if (!isCurrent()) return false;
                                    applyCharacterWizardData(data);
                                    return false;
                                } catch (e) {
                                    if (!isCurrent()) return false;
                                    showCharacterWizardError({ message: e.toString() });
                                    return true;
                                } finally {
                                    btn.disabled = false;
                                    btn.innerText = "FILL FIELDS";
                                }
                            }
                        }
                    ]);
                };

                const doCreate = () => {
                    let inpRef;
                    const { content } = showModal("New Character", () => {
                        const inp = document.createElement("input");
                        inp.className = "vnccs-creator-input";
                        inp.placeholder = "Name...";
                        inpRef = inp;
                        return inp;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "Create",
                            class: "vnccs-creator-btn-primary",
                            action: async (ol, btn) => {
                                const n = inpRef.value.trim();
                                if (!n) return true; // Keep open
                                try {
                                    await checkedJSON("/vnccs/create", { method: "POST", body: JSON.stringify({ name: n, catalog: "creator_v2" }) });
                                    const exists = Array.from(els.charSelect.options).some(o => o.value === n);
                                    if (!exists) els.charSelect.add(new Option(n, n));

                                    return !await loadChar(n);
                                } catch (e) {
                                    showAlertModal("Create Failed", e);
                                    return true;
                                }
                            }
                        }
                    ]);
                    inpRef.focus();
                };

                const doDelete = () => {
                    const charName = state.character;
                    if (!charName || charName === "None" || charName === "Unknown") {
                        showAlertModal("Delete Character", "Please select a character to delete.");
                        return;
                    }

                    showModal("Delete Character", () => {
                        const div = document.createElement("div");
                        div.style.fontSize = "14px";
                        div.style.textAlign = "center";
                        div.append("Are you sure you want to ");
                        const strong = document.createElement("b");
                        strong.textContent = "PERMANENTLY DELETE";
                        div.appendChild(strong);
                        div.appendChild(document.createElement("br"));
                        const nameEl = document.createElement("span");
                        nameEl.style.color = "#fff";
                        nameEl.style.fontWeight = "bold";
                        nameEl.textContent = `'${charName}'`;
                        div.appendChild(nameEl);
                        div.append("?");
                        div.appendChild(document.createElement("br"));
                        div.appendChild(document.createElement("br"));
                        const warning = document.createElement("span");
                        warning.style.fontSize = "12px";
                        warning.style.color = "#aaa";
                        warning.textContent = "This action cannot be undone.";
                        div.appendChild(warning);
                        return div;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "CONFIRM DELETE",
                            class: "vnccs-creator-btn-danger",
                            action: async (ol, btn) => {
                                try {
                                    btn.innerText = "DELETING...";
                                    btn.disabled = true;
                                    const r = await api.fetchApi("/vnccs/delete", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                                        body: JSON.stringify({ name: charName }),
                                    });
                                    if (r.ok) {
                                        const opts = Array.from(els.charSelect.options);
                                        const idx = opts.findIndex(o => o.value === charName);
                                        if (idx > -1) els.charSelect.remove(idx);

                                        if (els.charSelect.options.length > 0) {
                                            clearCharacterSelection();
                                            await loadChar(els.charSelect.options[0].value);
                                        } else {
                                            clearCharacterSelection();
                                        }
                                        saveState();
                                        return false;
                                    } else {
                                        const err = await r.json();
                                        showAlertModal("Delete Failed", err.error || "Unknown Error");
                                        btn.innerText = "CONFIRM DELETE";
                                        btn.disabled = false;
                                        return true;
                                    }
                                } catch (e) {
                                    showAlertModal("Delete Failed", e);
                                    return false; // Close on crash to avoid Stuck UI
                                }
                            }
                        }
                    ]);
                };

                const openTagConstructor = async (fieldKey, inputEl) => {
                    // 1. Ensure Data
                    if (!TAG_DATA) {
                        try {
                            const r = await api.fetchApi("/vnccs/get_tags?catalog=creator_v2");
                            if (r.ok) TAG_DATA = await r.json();
                            else throw new Error("Failed to load presets");
                        } catch (e) {
                            showAlertModal("Preset Catalog Error", "Error loading character presets: " + e);
                            return;
                        }
                    }

                    const allTags = presetGroups(TAG_DATA, fieldKey);
                    if (!allTags.length) {
                        showAlertModal("No Presets", "No presets found for this category.");
                        return;
                    }
                    const selected = presetSelection(inputEl.value, allTags);

                    showModal(`Choose Presets: ${fieldKey.replaceAll("_", " ")}`, (modal) => {
                        const container = document.createElement("div");
                        container.className = "vnccs-creator-tag-grid";

                        const description = document.createElement("p");
                        description.className = "vnccs-creator-preset-description";
                        description.setAttribute("aria-live", "polite");
                        description.textContent = fieldKey === "race"
                            ? "Choose a species to see its features. These descriptions are added automatically to the prompt. Combine species for hybrids; custom traits take priority."
                            : "Choose the traits you need. Custom text remains editable in the character field.";
                        const content = document.createElement("div");
                        content.append(description, container);
                        modal.style.width = "500px";

                        allTags.forEach(group => {
                            if (group.header) {
                                const h = document.createElement("div");
                                h.className = "vnccs-creator-tag-category";
                                h.style.width = "100%";
                                h.innerText = group.header;
                                container.appendChild(h);
                            }

                            group.items.forEach(item => {
                                const chip = document.createElement("button");
                                chip.type = "button";
                                chip.className = "vnccs-creator-tag-chip";
                                chip.innerText = item.label || item.tag;
                                if (item.prompt) {
                                    chip.title = item.prompt;
                                    chip.setAttribute("aria-description", item.prompt);
                                }
                                const updateChip = () => {
                                    const active = selected.has(item);
                                    chip.classList.toggle("selected", active);
                                    chip.setAttribute("aria-pressed", String(active));
                                };
                                updateChip();
                                chip.onclick = () => {
                                    selected.toggle(item);
                                    updateChip();
                                    if (item.prompt) description.textContent = item.prompt;
                                };
                                chip.onfocus = () => {
                                    if (item.prompt) description.textContent = item.prompt;
                                };
                                container.appendChild(chip);
                            });
                        });

                        return content;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "APPLY",
                            class: "vnccs-creator-btn-primary",
                            action: () => {
                                const final = selected.value();
                                inputEl.value = final;
                                // Trigger oninput so state.character_info is updated
                                inputEl.dispatchEvent(new Event('input'));
                                return false; // Close
                            }
                        }
                    ]);
                };

                btnDel.onclick = () => doDelete();

                btnRow.appendChild(btnGen);
                btnRow.appendChild(btnNew);
                btnRow.appendChild(btnDel);
                colLeft.appendChild(btnRow);

                const frame = document.createElement("div");
                frame.className = "vnccs-creator-preview-container";
                frame.innerHTML = `<div class="vnccs-creator-placeholder">
                    <svg class="vnccs-creator-placeholder-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="24" cy="18" r="8" stroke="currentColor" stroke-width="2"/>
                        <path d="M8 42c0-8.837 7.163-16 16-16s16 7.163 16 16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    No Preview
                </div>`;
                const img = document.createElement("img");
                img.className = "vnccs-creator-preview-img vnccs-creator-character-preview"; img.style.display = "none";
                frame.appendChild(img);
                const previewLoading = document.createElement("div");
                previewLoading.className = "vnccs-creator-preview-loading";
                previewLoading.innerHTML = '<div class="vnccs-creator-preview-spinner"></div>';
                frame.appendChild(previewLoading);
                els.previewImg = img; els.placeholder = frame.querySelector(".vnccs-creator-placeholder");
                els.previewLoading = previewLoading;
                colLeft.appendChild(frame);

                const spriteNav = document.createElement("div");
                spriteNav.className = "vnccs-creator-sprite-nav";
                const spritePrevBtn = document.createElement("button");
                spritePrevBtn.type = "button";
                spritePrevBtn.className = "vnccs-creator-sprite-nav-btn";
                spritePrevBtn.title = "Previous sprite";
                spritePrevBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>';
                const spriteCount = document.createElement("div");
                spriteCount.className = "vnccs-creator-sprite-nav-count";
                const spriteNextBtn = document.createElement("button");
                spriteNextBtn.type = "button";
                spriteNextBtn.className = "vnccs-creator-sprite-nav-btn";
                spriteNextBtn.title = "Next sprite";
                spriteNextBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>';
                spriteNav.append(spritePrevBtn, spriteCount, spriteNextBtn);
                els.spriteNav = spriteNav;
                els.spritePrevBtn = spritePrevBtn;
                els.spriteNextBtn = spriteNextBtn;
                els.spriteCount = spriteCount;
                els.spritePrevBtn.onclick = () => showSpritePreview(state.character, Number(state.sprite_preview_index || 0) - 1);
                els.spriteNextBtn.onclick = () => showSpritePreview(state.character, Number(state.sprite_preview_index || 0) + 1);
                colLeft.appendChild(spriteNav);

                topRow.appendChild(colLeft);

                // COL 2: CENTER (Attributes)
                const colCenter = document.createElement("div");
                colCenter.className = "vnccs-creator-col";
                colCenter.innerHTML = '<div class="vnccs-creator-section-title">Attributes</div>';

                const characterWizardBtn = document.createElement("button");
                characterWizardBtn.type = "button";
                characterWizardBtn.className = "vnccs-creator-btn vnccs-creator-btn-primary vnccs-creator-character-wizard-btn";
                characterWizardBtn.innerText = "CHARACTER WIZZARD";
                characterWizardBtn.onclick = openCharacterWizard;
                colCenter.appendChild(characterWizardBtn);

                colCenter.appendChild(createSegmentedField("Background", "background_color", [
                    { label: "Green", value: "Green" },
                    { label: "Blue", value: "Blue" },
                    { label: "Alpha", value: "Transparent" },
                ]));
                colCenter.appendChild(createSegmentedField("Gender", "sex", [
                    { label: "Male", value: "male" },
                    { label: "Female", value: "female" },
                ]));
                colCenter.appendChild(createSlider("Age", "age", 1, 100, 1, state.character_info));
                colCenter.appendChild(createField("Framing", "framing", "select", [
                    { label: "Cowboy shot", value: "cowboy_shot" },
                    { label: "Full body", value: "Full_body" },
                ]));
                colCenter.appendChild(createStyleField());
                const traitList = document.createElement("div");
                traitList.className = "vnccs-creator-trait-list";
                traitList.appendChild(createField("Race", "race"));
                traitList.appendChild(createField("Skin", "skin_color"));
                traitList.appendChild(createField("Body", "body"));
                traitList.appendChild(createField("Face", "face"));
                traitList.appendChild(createField("Hair", "hair"));
                traitList.appendChild(createField("Eyes", "eyes"));
                traitList.appendChild(createField("Details", "additional_details"));
                colCenter.appendChild(traitList);
                colCenter.appendChild(createGraphicToggle("NSFW Mode", "nsfw"));

                topRow.appendChild(colCenter);

                // COL 3: RIGHT (Generation)
                const colRight = document.createElement("div");
                colRight.className = "vnccs-creator-col";
                colRight.innerHTML = '<div class="vnccs-creator-section-title">Generation</div>';

                const tabRow = document.createElement("div");
                tabRow.className = "vnccs-creator-tab-row";
                els.modeTabs = {};
                [
                    ["qi2", "Qwen Image 2.1"],
                    ["anima", "ANIMA"],
                    ["illustrious", "Illustrious"],
                ].forEach(([value, label]) => {
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "vnccs-creator-tab";
                    btn.innerText = label;
                    btn.onclick = () => setGenerationMode(value);
                    els.modeTabs[value] = btn;
                    tabRow.appendChild(btn);
                });
                colRight.appendChild(tabRow);

                const illustriousModels = document.createElement("div");
                illustriousModels.className = "vnccs-creator-subsection";
                els.illustriousModels = illustriousModels;

                const illustriousCards = document.createElement("div");
                illustriousCards.className = "vnccs-creator-model-card-list";
                els.illustriousModelCards = illustriousCards;
                illustriousModels.appendChild(illustriousCards);

                const illustriousFallback = document.createElement("div");
                illustriousFallback.className = "vnccs-creator-generation-fallback";
                illustriousFallback.style.display = "none";
                els.illustriousFallback = illustriousFallback;
                const wrapCkpt = makeFallbackSelect("Checkpoint (SDXL)", "ckpt_name");
                els.ckptSelect = wrapCkpt.querySelector("select");
                illustriousFallback.appendChild(wrapCkpt);
                illustriousModels.appendChild(illustriousFallback);
                colRight.appendChild(illustriousModels);

                const animaModels = document.createElement("div");
                animaModels.className = "vnccs-creator-subsection";
                els.animaModels = animaModels;

                const createAnimaCardField = (slotName) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-creator-field";
                    const cards = document.createElement("div");
                    cards.className = "vnccs-creator-model-card-list";
                    wrap.appendChild(cards);
                    els[slotName] = cards;
                    animaModels.appendChild(wrap);
                };

                createAnimaCardField("animaModelCards");

                const hiddenAnimaSelects = document.createElement("div");
                hiddenAnimaSelects.className = "vnccs-creator-generation-fallback";
                hiddenAnimaSelects.style.display = "none";
                els.animaFallback = hiddenAnimaSelects;
                hiddenAnimaSelects.appendChild(makeFallbackSelect("Diffusion Model", "diffusion_model_name"));
                hiddenAnimaSelects.appendChild(makeFallbackSelect("CLIP", "clip_name"));
                hiddenAnimaSelects.appendChild(makeFallbackSelect("VAE", "vae_name"));
                animaModels.appendChild(hiddenAnimaSelects);
                colRight.appendChild(animaModels);

                const qi2Models = document.createElement("div");
                qi2Models.className = "vnccs-creator-subsection";
                qi2Models.style.display = "none";
                els.qi2Models = qi2Models;
                const qi2ModelCards = document.createElement("div");
                qi2ModelCards.className = "vnccs-creator-model-card-list";
                els.qi2ModelCards = qi2ModelCards;
                qi2Models.appendChild(qi2ModelCards);
                colRight.appendChild(qi2Models);

                const resolutionWrap = document.createElement("label");
                resolutionWrap.className = "vnccs-creator-field vnccs-creator-resolution-field";
                setHelpText(resolutionWrap, helpFor("target_size"));
                const resolutionHead = document.createElement("div");
                resolutionHead.className = "vnccs-creator-resolution-head";
                const resolutionLabel = document.createElement("div");
                resolutionLabel.className = "vnccs-creator-label";
                resolutionLabel.textContent = "Resolution scale";
                const resolutionValue = document.createElement("div");
                resolutionValue.className = "vnccs-creator-resolution-value";
                resolutionHead.append(resolutionLabel, resolutionValue);
                const resolutionSlider = document.createElement("input");
                resolutionSlider.type = "range";
                resolutionSlider.className = "vnccs-creator-resolution-slider";
                resolutionSlider.min = String(RESOLUTION_SCALE_MIN_MP);
                resolutionSlider.max = String(RESOLUTION_SCALE_MAX_MP);
                resolutionSlider.step = String(RESOLUTION_SCALE_STEP_MP);
                resolutionSlider.setAttribute("aria-label", "Resolution scale in megapixels");
                resolutionSlider.value = resolutionScaleMegapixels(state.gen_settings.target_size).toFixed(1);
                resolutionValue.textContent = resolutionScaleText(state.gen_settings.target_size);
                resolutionSlider.oninput = () => {
                    state.gen_settings.target_size = resolutionScaleValue(resolutionSlider.value);
                    resolutionValue.textContent = resolutionScaleText(state.gen_settings.target_size);
                    saveState();
                };
                els.target_size = resolutionSlider;
                els.target_size_value = resolutionValue;
                resolutionWrap.append(resolutionHead, resolutionSlider);
                colRight.appendChild(resolutionWrap);

                const genParamGrid = document.createElement("div");
                genParamGrid.className = "vnccs-creator-gen-param-grid";
                genParamGrid.appendChild(createCompactNumberField("Steps", "steps", 1, 100, 1));
                genParamGrid.appendChild(createCompactSelectField("Sampler", "sampler", state.gen_settings));
                genParamGrid.appendChild(createCompactNumberField("CFG", "cfg", 1, 20, 0.1));
                genParamGrid.appendChild(createCompactSelectField("Scheduler", "scheduler", state.gen_settings));
                colRight.appendChild(genParamGrid);

                // SEED Section (Rebalanced)
                const seedWrap = document.createElement("div"); seedWrap.className = "vnccs-creator-field";
                setHelpText(seedWrap, helpFor("seed"));
                seedWrap.innerHTML = '<div class="vnccs-creator-label">Seed</div>';

                const seedRow = document.createElement("div");
                seedRow.className = "vnccs-creator-seed-row";

                const seedInp = document.createElement("input"); seedInp.className = "vnccs-creator-gen-param-input";
                seedInp.type = "number"; seedInp.value = state.gen_settings.seed;
                seedInp.onchange = (e) => {
                    state.gen_settings.seed = parseInt(e.target.value);
                    saveState();
                };
                els.seed = seedInp;

                const seedMode = document.createElement("button");
                seedMode.type = "button";
                seedMode.className = "vnccs-creator-seed-dice-btn";
                setHelpText(seedMode, helpFor("seed_mode"));
                seedMode.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <rect x="4" y="4" width="16" height="16" rx="3.5" stroke="currentColor" stroke-width="2"/>
                        <circle cx="8.5" cy="8.5" r="1.4" fill="currentColor"/>
                        <circle cx="15.5" cy="8.5" r="1.4" fill="currentColor"/>
                        <circle cx="12" cy="12" r="1.4" fill="currentColor"/>
                        <circle cx="8.5" cy="15.5" r="1.4" fill="currentColor"/>
                        <circle cx="15.5" cy="15.5" r="1.4" fill="currentColor"/>
                    </svg>`;
                seedMode.onclick = () => {
                    state.gen_settings.seed_mode = (state.gen_settings.seed_mode || "fixed") === "randomize" ? "fixed" : "randomize";
                    syncGenerationControls();
                    saveState();
                };
                els.seed_mode = seedMode;

                seedRow.appendChild(seedInp);
                seedRow.appendChild(seedMode);

                seedWrap.appendChild(seedRow);
                colRight.appendChild(seedWrap);

                const qi2Cache = document.createElement("div");
                qi2Cache.className = "vnccs-creator-qi2-cache";
                const qi2CacheTitle = document.createElement("div");
                qi2CacheTitle.className = "vnccs-creator-qi2-cache-title";
                qi2CacheTitle.innerText = "Qwen Image 2.1 Cache";
                qi2Cache.appendChild(qi2CacheTitle);
                const qi2CacheDeviceWrap = createCompactSelectField("Device", "qi2_cache_device", {});
                const qi2CacheDtypeWrap = createCompactSelectField("Dtype", "qi2_cache_dtype", {});
                ["auto", "gpu", "cpu", "off"].forEach(value => els.qi2_cache_device.add(new Option(value, value)));
                ["default", "int8", "int4"].forEach(value => els.qi2_cache_dtype.add(new Option(value, value)));
                els.qi2_cache_device.onchange = () => {
                    state.gen_settings.qi2_cache = { ...(state.gen_settings.qi2_cache || {}), device: els.qi2_cache_device.value };
                    saveState();
                };
                els.qi2_cache_dtype.onchange = () => {
                    state.gen_settings.qi2_cache = { ...(state.gen_settings.qi2_cache || {}), dtype: els.qi2_cache_dtype.value };
                    saveState();
                };
                els.qi2Cache = qi2Cache;
                els.qi2CacheDevice = els.qi2_cache_device;
                els.qi2CacheDtype = els.qi2_cache_dtype;
                qi2Cache.appendChild(qi2CacheDeviceWrap);
                qi2Cache.appendChild(qi2CacheDtypeWrap);
                colRight.appendChild(qi2Cache);

                // --- LoRA Section ---
                const loraSection = document.createElement("div");
                loraSection.className = "vnccs-creator-subsection";
                els.loraSection = loraSection;

                const loraHeader = document.createElement("div");
                loraHeader.className = "vnccs-creator-section-title";
                loraHeader.style.marginTop = "10px";
                loraHeader.innerText = "LoRa Stack";
                els.loraHeader = loraHeader;
                loraSection.appendChild(loraHeader);

                // DMD2 LoRA
                const dmdWrap = document.createElement("div"); dmdWrap.className = "vnccs-creator-lora-item";
                els.dmdWrap = dmdWrap;
                const dmdLabel = document.createElement("div");
                dmdLabel.className = "vnccs-creator-label";
                dmdLabel.innerText = "DMD2 LoRA Model";
                els.dmdLabel = dmdLabel;
                dmdWrap.appendChild(dmdLabel);
                const dmdRow = document.createElement("div"); dmdRow.className = "vnccs-creator-lora-row";
                const dmdSel = document.createElement("select"); dmdSel.className = "vnccs-creator-select";
                dmdSel.style.flex = "2";
                dmdSel.onchange = (e) => { state.gen_settings.dmd_lora_name = e.target.value; saveState(); };
                els.dmdSelect = dmdSel;

                const dmdToggleWrap = document.createElement("label");
                dmdToggleWrap.className = "vnccs-creator-toggle";
                dmdToggleWrap.style.flexShrink = "0";
                dmdToggleWrap.style.margin = "0 4px";

                const dmdStr = document.createElement("input");
                dmdStr.type = "checkbox";
                dmdStr.checked = (state.gen_settings.dmd_lora_strength || 0) > 0;
                dmdStr.onchange = (e) => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (mode === "anima" || mode === "qi2") {
                        setAnimaTurboMode(e.target.checked);
                    } else {
                        if (e.target.checked) {
                            if ((state.gen_settings.dmd_lora_strength || 0) <= 0) {
                                state.gen_settings.turbo_previous_settings = {
                                    steps: state.gen_settings.steps,
                                    cfg: state.gen_settings.cfg,
                                };
                            }
                            state.gen_settings.dmd_lora_strength = 1.0;
                            state.gen_settings.steps = 4;
                            state.gen_settings.cfg = 1.0;
                        } else {
                            state.gen_settings.dmd_lora_strength = 0.0;
                            const previous = state.gen_settings.turbo_previous_settings || {};
                            if (previous.steps !== undefined) state.gen_settings.steps = previous.steps;
                            if (previous.cfg !== undefined) state.gen_settings.cfg = previous.cfg;
                            state.gen_settings.turbo_previous_settings = null;
                        }
                        syncGenerationControls();
                        saveState();
                    }
                };
                const dmdTrack = document.createElement("div"); dmdTrack.className = "vnccs-creator-toggle-track";
                const dmdThumb = document.createElement("div"); dmdThumb.className = "vnccs-creator-toggle-thumb";
                dmdToggleWrap.appendChild(dmdStr);
                dmdToggleWrap.appendChild(dmdTrack);
                dmdToggleWrap.appendChild(dmdThumb);

                dmdRow.appendChild(dmdSel); dmdRow.appendChild(dmdToggleWrap);
                dmdWrap.appendChild(dmdRow);
                loraSection.appendChild(dmdWrap);
                els.dmdSlider = dmdStr; // Renamed ref for logic compat, though it's an input now

                const animaLoraCards = document.createElement("div");
                animaLoraCards.className = "vnccs-creator-subsection";
                animaLoraCards.style.display = "none";
                els.animaLoraCards = animaLoraCards;
                loraSection.appendChild(animaLoraCards);

                const qi2LoraCards = document.createElement("div");
                qi2LoraCards.className = "vnccs-creator-subsection";
                qi2LoraCards.style.display = "none";
                els.qi2LoraCards = qi2LoraCards;
                loraSection.appendChild(qi2LoraCards);

                const illustriousLoraCards = document.createElement("div");
                illustriousLoraCards.className = "vnccs-creator-subsection";
                illustriousLoraCards.style.display = "none";
                els.illustriousLoraCards = illustriousLoraCards;
                loraSection.appendChild(illustriousLoraCards);

                // Age LoRA
                const ageWrap = document.createElement("div"); ageWrap.className = "vnccs-creator-lora-item";
                els.ageWrap = ageWrap;
                ageWrap.innerHTML = '<div class="vnccs-creator-label">Age LoRA (Auto Strength)</div>';
                const ageSel = document.createElement("select"); ageSel.className = "vnccs-creator-select";
                ageSel.onchange = (e) => { state.gen_settings.age_lora_name = e.target.value; saveState(); };
                els.ageSelect = ageSel;
                ageWrap.appendChild(ageSel);
                loraSection.appendChild(ageWrap);

                // Stack (5 Slots)
                const stackContainer = document.createElement("div");
                stackContainer.className = "vnccs-creator-lora-stack";
                els.loraStackSelects = [];

                for (let i = 0; i < 5; i++) {
                    const item = document.createElement("div"); item.className = "vnccs-creator-lora-item";
                    const row = document.createElement("div"); row.className = "vnccs-creator-lora-row";

                    const sel = document.createElement("select"); sel.className = "vnccs-creator-select";
                    sel.style.flex = "2";
                    const updateEmpty = () => {
                        const isEmpty = !sel.value || sel.value === "";
                        item.classList.toggle("is-empty", isEmpty);
                    };
                    sel.onchange = (e) => {
                        state.gen_settings.lora_stack[i].name = e.target.value;
                        updateEmpty();
                        saveState();
                    };

                    const rng = document.createElement("input"); rng.className = "vnccs-creator-input";
                    rng.type = "number"; rng.step = "0.05"; rng.style.flex = "1";

                    rng.onchange = (e) => {
                        state.gen_settings.lora_stack[i].strength = parseFloat(e.target.value);
                        saveState();
                    };

                    row.appendChild(sel);
                    row.appendChild(rng);
                    item.appendChild(row);
                    item.classList.add("is-empty");
                    stackContainer.appendChild(item);

                    // Ref for population
                    els.loraStackSelects.push({ sel, rng, idx: i });
                }
                loraSection.appendChild(stackContainer);
                colRight.appendChild(loraSection);

                topRow.appendChild(colRight);
                container.appendChild(topRow);

                // --- BOTTOM ROW (Prompts) ---
                const bottomRow = document.createElement("div");
                bottomRow.className = "vnccs-creator-bottom-row";

                const createText = (lbl, key) => {
                    const w = document.createElement("div"); w.className = "vnccs-creator-textarea-wrapper";
                    w.innerHTML = `<div class="vnccs-creator-textarea-label">${lbl}</div>`;
                    const t = document.createElement("textarea");
                    t.value = state.character_info[key] || "";
                    t.oninput = (e) => {
                        state.character_info[key] = e.target.value;
                        if (key === "aesthetics" || key === "negative_prompt") {
                            const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                            if (!state.prompt_modes[mode]) state.prompt_modes[mode] = { ...MODE_PROMPT_DEFAULTS[mode] };
                            state.prompt_modes[mode][key] = e.target.value;
                            state.prompt_defaults_version = PROMPT_DEFAULTS_VERSION;
                        }
                        debouncedSave();
                    };
                    w.appendChild(t);
                    els[key] = t;
                    return w;
                }
                bottomRow.appendChild(createText("Aesthetics", "aesthetics"));
                bottomRow.appendChild(createText("Negative Prompt", "negative_prompt"));
                bottomRow.appendChild(createText("LoRA Trigger", "lora_prompt"));

                container.appendChild(bottomRow);

                // Inject UI
                this.addDOMWidget("ui", "ui", container, {
                    serialize: false,
                    hideOnZoom: false,
                });
                syncDOMWidgetWidthSoon(node, "ui");

                // 6. Logic

                // EVENT LISTENER for Backend Updates
                const previewUpdateHandler = (e) => {
                    if (e.detail.node_id == node.id) {
                        const charName = e.detail.character;
                        console.log(`[VNCCS] Preview Update Event received for '${charName}' (Node ${node.id})`);
                        if (charName === state.character) {
                            console.log("[VNCCS] Character matches. Refreshing local preview...");
                            tryCachePreview(charName);
                        }
                    }
                };
                api.addEventListener("vnccs.preview.updated", previewUpdateHandler);
                registerCleanup(node, () => api.removeEventListener("vnccs.preview.updated", previewUpdateHandler));

                let previewRunning = false;
                let workflowBusy = true;
                const syncPreviewButton = () => {
                    els.btnGen.disabled = previewRunning || workflowBusy;
                    els.btnGen.innerText = previewRunning ? "GENERATING..." : workflowBusy ? "WORKFLOW BUSY..." : "GENERATE PREVIEW";
                    els.btnGen.title = workflowBusy ? "Preview is unavailable while a workflow is queued or running." : "";
                };
                const beginWorkflowStatusRequest = createRequestGuard(node);
                const refreshWorkflowBusy = async () => {
                    const current = beginWorkflowStatusRequest();
                    try {
                        const response = await api.fetchApi("/queue");
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        const queue = await response.json();
                        if (!Array.isArray(queue.queue_running) || !Array.isArray(queue.queue_pending)) throw new Error("Invalid queue response");
                        if (!current()) return;
                        workflowBusy = Boolean(queue.queue_running?.length || queue.queue_pending?.length);
                        syncPreviewButton();
                    } catch (error) {
                        if (!current()) return;
                        workflowBusy = true;
                        syncPreviewButton();
                        console.warn("[VNCCS] Could not refresh workflow status", error);
                    }
                };
                const setWorkflowBusy = busy => {
                    beginWorkflowStatusRequest();
                    workflowBusy = busy;
                    syncPreviewButton();
                };
                const workflowStatusHandler = event => {
                    const remaining = event.detail?.exec_info?.queue_remaining;
                    if (typeof remaining === "number" && Number.isFinite(remaining)) setWorkflowBusy(remaining > 0);
                    else if (event.detail == null) setWorkflowBusy(true);
                };
                const workflowStartHandler = () => setWorkflowBusy(true);
                api.addEventListener("status", workflowStatusHandler);
                api.addEventListener("execution_start", workflowStartHandler);
                api.addEventListener("reconnecting", workflowStartHandler);
                registerCleanup(node, () => {
                    api.removeEventListener("status", workflowStatusHandler);
                    api.removeEventListener("execution_start", workflowStartHandler);
                    api.removeEventListener("reconnecting", workflowStartHandler);
                });
                syncPreviewButton();
                refreshWorkflowBusy();
                watchConnection(node, async () => {
                    await refreshWorkflowBusy();
                    if (state.character && !els.btnGen?.disabled) refreshPreviewImage(els.previewImg);
                }, registerCleanup);
                registerCleanup(node, () => stopCcPolling());

                const init = async () => {
                    loadState();
                    try {
                        const r = await api.fetchApi("/vnccs/context_lists");
                        const d = await r.json();

                        const pop = (el, items, none = false) => {
                            if (!el) return; el.innerHTML = "";
                            if (none) el.add(new Option("None", ""));
                            (items || []).forEach(x => el.add(new Option(x, x)));
                        };

                        pop(els.ckptSelect, d.checkpoints);
                        pop(els.diffusion_model_name, d.diffusion_models);
                        pop(els.clip_name, d.text_encoders);
                        pop(els.vae_name, d.vae_models);
                        pop(els.sampler, d.samplers);
                        pop(els.scheduler, d.schedulers);
                        localAssets = {
                            checkpoints: d.checkpoints || [],
                            diffusion_models: d.diffusion_models || [],
                            text_encoders: d.text_encoders || [],
                            vae_models: d.vae_models || [],
                            loras: d.loras || [],
                        };
                        renderControlCenterCards();

                        // Populate LoRA selectors
                        const loras = (d.loras || []).filter(name => !isCreatorOverhaulLora(name));
                        pop(els.dmdSelect, loras, true);
                        pop(els.ageSelect, loras, true);
                        els.loraStackSelects.forEach(o => pop(o.sel, loras, true));

                        fetchCcConfig(true).catch((error) => {
                            console.warn("[VNCCS V2] Control Center config unavailable:", error);
                            renderControlCenterCards();
                        });

                        const characters = Array.isArray(d.characters) ? d.characters : [];
                        els.charSelect.innerHTML = "";
                        characters.forEach(c => els.charSelect.add(new Option(c, c)));

                        // Restore values or defaults into independent per-mode profiles.
                        const g = state.gen_settings;
                        migrateGenerationModeSettings();

                        const illustriousProfile = getModeProfile("illustrious");
                        if (!illustriousProfile.ckpt_name && els.ckptSelect.options.length > 0) {
                            illustriousProfile.ckpt_name = els.ckptSelect.options[0].value;
                        }
                        ensureOption(els.ckptSelect, illustriousProfile.ckpt_name);

                        const animaProfile = getModeProfile("anima");
                        ["diffusion_model_name", "clip_name", "vae_name"].forEach((key) => {
                            const ref = els[key];
                            if (!ref) return;
                            if (key === "clip_name" && !animaProfile[key]) {
                                animaProfile[key] = ANIMA_CLIP_NAME;
                            }
                            if (key === "vae_name" && !animaProfile[key]) {
                                animaProfile[key] = ANIMA_VAE_NAME;
                            }
                            if (key !== "diffusion_model_name" && !animaProfile[key] && ref.options.length > 0) {
                                animaProfile[key] = ref.options[0].value;
                            }
                            ensureOption(ref, animaProfile[key]);
                        });
                        const qi2Profile = getModeProfile("qi2");
                        if (!qi2Profile.diffusion_model_name) qi2Profile.diffusion_model_name = QI2_MODEL_NAME;
                        if (!qi2Profile.clip_name) qi2Profile.clip_name = QI2_CLIP_NAME;
                        if (!qi2Profile.vae_name) qi2Profile.vae_name = QI2_VAE_NAME;
                        applyGenerationProfile(g.generation_mode);
                        migratePromptModes();

                        saveCurrentGenerationModeValues();
                        syncGenerationControls();
                        refreshGenerationModeUI();

                        const selected = characters.includes(state.character) ? state.character : characters[0] || "";
                        const hasWidgetData = restoredWidgetInfoCharacter && restoredWidgetInfoCharacter === selected;
                        if (selected) await loadChar(selected, hasWidgetData);
                        else clearCharacterSelection();

                        // Sync widget state. Preview validity is set by image onload handler, not preemptively.
                        saveState();

                    } catch (e) { console.error(e); }
                };

                const beginCharacterRequest = createRequestGuard(node);
                const beginCharacterWizardRequest = createRequestGuard(node);
                const loadChar = async (n, skipInfoLoad = false) => {
                    beginCharacterWizardRequest();
                    const currentRequest = beginCharacterRequest();
                    const currentPreview = beginPreviewRequest();
                    const originalCharacter = state.character;
                    const isCurrent = () => currentRequest() && (state.character === originalCharacter || state.character === n);
                    if (!n) return false;
                    try {
                        // 1. Fetch Info (skip if restoring from widget_data)
                        if (!skipInfoLoad) {
                            const i = await checkedJSON(`/vnccs/character_info?character=${encodeURIComponent(n)}`);
                            if (!isCurrent()) return false;
                            if (!i || typeof i !== "object" || Array.isArray(i) || (i.name && i.name !== n)) {
                                throw new Error(`Invalid character metadata for '${n}'`);
                            }
                            state.character = n;
                            restoredWidgetInfoCharacter = null;
                            for (const key of Object.keys(state.character_info)) delete state.character_info[key];
                            Object.assign(state.character_info, getDefaultCharacterInfo(), { name: n }, i);
                            syncBackgroundForGenerationMode(true);
                            state.prompt_modes = {
                                illustrious: {
                                    aesthetics: state.character_info.aesthetics || MODE_PROMPT_DEFAULTS.illustrious.aesthetics,
                                    negative_prompt: state.character_info.negative_prompt || MODE_PROMPT_DEFAULTS.illustrious.negative_prompt,
                                },
                                anima: {
                                    aesthetics: MODE_PROMPT_DEFAULTS.anima.aesthetics,
                                    negative_prompt: MODE_PROMPT_DEFAULTS.anima.negative_prompt,
                                },
                                qi2: {
                                    aesthetics: MODE_PROMPT_DEFAULTS.qi2.aesthetics,
                                    negative_prompt: MODE_PROMPT_DEFAULTS.qi2.negative_prompt,
                                },
                            };
                            state.prompt_defaults_version = PROMPT_DEFAULTS_VERSION;
                            applyPromptModeToFields((state.gen_settings.generation_mode || "illustrious").toLowerCase());
                        }

                        // Commit selection only after metadata has been read successfully.
                        if (!isCurrent()) return false;
                        state.character = n;
                        if (els.charSelect) els.charSelect.value = n;
                        syncCharacterFields();
                        saveState();

                        // 2. Fetch Preview Image
                        if (!currentPreview()) return;
                        try {
                            const metaResponse = await api.fetchApi(`/vnccs/get_character_pose_preview_meta?character=${encodeURIComponent(n)}&t=${Date.now()}`);
                            const meta = metaResponse.ok ? await metaResponse.json() : {};
                            if (!isCurrent() || !currentPreview()) return;
                            const count = Number(meta.count || 0);
                            state.sprite_preview_count = count;
                            state.sprite_preview_cache_bust = `${n}:${Date.now()}`;

                            if (count > 0) {
                                const index = Math.floor(Math.random() * count);
                                showSpritePreview(n, index);
                            } else {
                                hideSpriteNav();
                                tryCachePreview(n);
                            }
                        } catch (previewError) {
                            if (!isCurrent() || !currentPreview()) return;
                            console.warn("[VNCCS] Failed to load pose preview metadata. Fallback to cache.", previewError);
                            hideSpriteNav();
                            tryCachePreview(n);
                        }

                        return isCurrent();
                    } catch (error) {
                        if (!isCurrent()) return false;
                        if (els.charSelect) els.charSelect.value = state.character;
                        showAlertModal("Character Load Failed", error.message || error);
                        return false;
                    }
                };

                const doGenerate = async () => {
                    if (els.btnGen.disabled) return;
                    if (!state.character) {
                        showAlertModal("No Character", "Create a character before generating a preview.");
                        return;
                    }
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (mode === "anima" || mode === "qi2") {
                        if (!state.gen_settings.diffusion_model_name) { showAlertModal("Missing Model", "Select Diffusion Model"); return; }
                        if (!state.gen_settings.clip_name) { showAlertModal("Missing Model", "Select CLIP"); return; }
                        if (!state.gen_settings.vae_name) { showAlertModal("Missing Model", "Select VAE"); return; }
                        const family = mode === "qi2" ? "QI2" : "Anima";
                        const familyLabel = mode === "qi2" ? "Qwen Image 2.1" : "Anima";
                        if (!isSelectedCcAssetInstalled("models", family, "diffusion_model_name", entry => ccType(entry) === "unet")) {
                            showAlertModal("Model Missing", `Download and select an installed ${familyLabel} Diffusion Model`);
                            return;
                        }
                        if (!isSelectedCcAssetInstalled("clip", family, "clip_name")) {
                            showAlertModal("Model Missing", `Download and select an installed ${familyLabel} CLIP`);
                            return;
                        }
                        if (!isSelectedCcAssetInstalled("vae", family, "vae_name")) {
                            showAlertModal("Model Missing", `Download and select an installed ${familyLabel} VAE`);
                            return;
                        }
                    } else if (!state.gen_settings.ckpt_name) {
                        showAlertModal("Missing Checkpoint", "Select Checkpoint"); return;
                    } else if (!isSelectedCcAssetInstalled("models", null, "ckpt_name", entry => {
                        const kind = ccKind(entry);
                        return (kind === "illustrious" || kind === "sdxl") && ccType(entry) === "checkpoint";
                    })) {
                        showAlertModal("Model Missing", "Download and select an installed Illustrious checkpoint");
                        return;
                    }
                    node._randomizeSeedIfNeeded();
                    saveCurrentGenerationModeValues();

                    // Show loading overlay
                    const loading = createLoadingOverlay(container, "Generating preview");

                    previewRunning = true;
                    syncPreviewButton();
                    saveState();

                    const character = state.character;
                    const currentPreview = beginPreviewRequest();
                    const isCurrent = () => currentPreview() && state.character === character;
                    try {
                        const payload = {
                            character: state.character,
                            character_info: state.character_info,
                            gen_settings: { ...state.gen_settings }
                        };
                        // Clean stack in the copy
                        payload.gen_settings.lora_stack = payload.gen_settings.lora_stack.filter(x => x.name && x.name !== "None");

                        const r = await api.fetchApi("/vnccs/preview_generate", { method: "POST", body: JSON.stringify(payload) });

                        if (!r.ok) {
                            const errText = await r.text();
                            throw new Error(`Server Error (${r.status}): ${errText}`);
                        }

                        const d = await r.json();
                        if (!isCurrent()) return;
                        if (d.image) {
                            clearPreviewHandlers();
                            setPreviewLoading(false);
                            els.previewImg.src = "data:image/png;base64," + d.image;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";
                            // Successful generation -> Cache is Valid AND source is 'gen'
                            state.preview_source = "gen";
                            hideSpriteNav();
                            saveState(true);
                        }
                    } catch (e) { if (isCurrent()) showMessage(container, "Error: " + e, true); }
                    finally {
                        loading.remove();
                        previewRunning = false;
                        syncPreviewButton();
                    }
                };



                // 7. Graph Restore Hook / Main Entry Point
                let initialized = false;
                node.onConfigure = function () {
                    syncDOMWidgetWidth(node, "ui");
                    setTimeout(() => syncDOMWidgetWidth(node, "ui"), 100);
                    if (initialized) return;
                    // Prevent double init if called multiple times
                    // But we might need to re-load state if configure happens again? 
                    // Usually configure happens once on load.

                    initialized = true;
                    // Check if widget_data has value NOW
                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w && w.value) {
                        // We have data, init will use it via loadState
                    }

                    init();
                };

                // Fallback for new nodes (onConfigure runs on add too, but just in case)
                setTimeout(() => {
                    if (!initialized) {
                        initialized = true;
                        init();
                    }
                }, 100);
            };

            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                onResize?.apply(this, arguments);
                syncDOMWidgetWidth(this, "ui");
                requestAnimationFrame(() => syncDOMWidgetWidth(this, "ui"));
            };
        }
    }
});
