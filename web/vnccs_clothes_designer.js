import { app } from "../../scripts/app.js";
import { vnccsApi as api, mediaURL, checkedJSON, storage, refreshPreviewImage, watchConnection } from "./vnccs_transport.js";
import { registerCleanup, injectStyles, showModal as showCommonModal, showMessage, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, createSpritePreviewNavigator, createRequestGuard } from "./vnccs_common.js";

const RESOLUTION_SCALE_BASE = 1024;
const RESOLUTION_SCALE_MIN_MP = 1;
const RESOLUTION_SCALE_MAX_MP = 4;
const RESOLUTION_SCALE_STEP_MP = 0.1;
const RESOLUTION_SCALE_PRESETS = new Map([
    [1.3, 1344],
    [1.5, 1536],
]);

const resolutionScaleMegapixels = value => {
    const numeric = Number(value);
    const megapixels = Number.isFinite(numeric) ? numeric / RESOLUTION_SCALE_BASE : RESOLUTION_SCALE_MIN_MP;
    return Math.max(RESOLUTION_SCALE_MIN_MP, Math.min(RESOLUTION_SCALE_MAX_MP, megapixels));
};
const resolutionScaleValue = megapixels => {
    const clamped = Math.max(
        RESOLUTION_SCALE_MIN_MP,
        Math.min(RESOLUTION_SCALE_MAX_MP, Number(megapixels) || RESOLUTION_SCALE_MIN_MP)
    );
    const stepped = Number((Math.round(clamped / RESOLUTION_SCALE_STEP_MP) * RESOLUTION_SCALE_STEP_MP).toFixed(1));
    return RESOLUTION_SCALE_PRESETS.get(stepped) ?? Math.round(stepped * RESOLUTION_SCALE_BASE);
};
const resolutionScaleText = value => `${resolutionScaleMegapixels(value).toFixed(1)} MP`;

const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.vnccs-clothes-container {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-elevated: #1a1a26;
    --bg-surface: #22222e;
    --bg-hover: #2a2a38;
    --text-primary: #e8e8f0;
    --text-secondary: #9898a8;
    --text-muted: #5e5e70;
    --accent: #ff8fa3;
    --accent-hover: #ffb6c8;
    --accent-glow: rgba(255,143,163,0.3);
    --accent-subtle: rgba(255,143,163,0.1);
    --accent-border: rgba(255,143,163,0.22);
    --accent-lavender: #b8a9e8;
    --success: #00d68f;
    --warning: #ffaa00;
    --error: #ff4757;
    --border: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --transition: 0.2s ease;
}

.vnccs-clothes-container {
    display: flex; flex-direction: column;
    background: var(--bg-primary); color: var(--text-primary);
    font-family: var(--font); font-size: 13px;
    width: 100%; height: 100%; overflow: hidden; box-sizing: border-box;
    padding: 12px; gap: 12px; pointer-events: none; zoom: 0.67;
}
.vnccs-clothes-top-row {
    display: grid; grid-template-columns: 32% minmax(0, 68%); gap: 12px;
    flex: 1; min-height: 0; width: 100%;
}
.vnccs-clothes-col {
    display: flex; flex-direction: column;
    background: rgba(20,16,30,0.88);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 16px; gap: 10px;
    overflow-y: auto; height: 100%; box-sizing: border-box; pointer-events: auto;
    position: relative; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.vnccs-clothes-col::before {
    content: '';
    position: absolute; top: 0; left: 18%; right: 18%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.5), transparent);
    border-radius: 1px;
}
.vnccs-clothes-col::-webkit-scrollbar { width: 4px; }
.vnccs-clothes-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

.vnccs-clothes-section-title {
    font-size: 10px; font-weight: 700; color: var(--accent);
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px; flex-shrink: 0;
    display: flex; align-items: center; gap: 8px;
}
.vnccs-clothes-section-title::before {
    content: ''; width: 3px; height: 12px; flex-shrink: 0;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px; box-shadow: 0 0 8px var(--accent-glow);
}

.vnccs-clothes-field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 6px; flex-shrink: 0; }
.vnccs-clothes-label {
    color: var(--text-secondary); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
}

.vnccs-clothes-input, .vnccs-clothes-textarea {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    color: var(--text-primary); border-radius: var(--radius-md);
    padding: 8px 12px; font-family: var(--font); font-size: 12px;
    width: 100%; box-sizing: border-box; transition: all var(--transition);
}
.vnccs-clothes-textarea { resize: none; min-height: 40px; }
.vnccs-clothes-select {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    color: var(--text-primary); border-radius: var(--radius-md);
    padding: 8px 12px; font-family: var(--font); font-size: 12px;
    width: 100%; box-sizing: border-box; zoom: 1.5; transition: all var(--transition);
    color-scheme: dark;
}
.vnccs-clothes-select option {
    background: #1e1e2e; color: #e8e8f0;
}
.vnccs-clothes-input:focus, .vnccs-clothes-select:focus, .vnccs-clothes-textarea:focus {
    outline: none; border-color: var(--accent-border);
    background: rgba(255,143,163,0.04);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

.vnccs-clothes-btn {
    padding: 10px; border: none; border-radius: var(--radius-md); cursor: pointer;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 11px; font-family: var(--font); color: white;
    text-align: center; flex: 1; transition: all var(--transition);
    position: relative; overflow: hidden;
}
.vnccs-clothes-btn-primary {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525; box-shadow: 0 4px 16px rgba(255,143,163,0.25);
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.vnccs-clothes-btn-primary::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: cd-shimmer 3.5s ease-in-out infinite; pointer-events: none;
}
@keyframes cd-shimmer {
    0% { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35% { transform: translateX(120%) skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%) skewX(-15deg); opacity: 0; }
}
.vnccs-clothes-btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(255,143,163,0.4); }
.vnccs-clothes-container button.vnccs-clothes-btn.vnccs-clothes-btn-primary:not(:disabled),
.vnccs-clothes-container button.vnccs-clothes-btn.vnccs-clothes-btn-primary:not(:disabled):hover,
.vnccs-clothes-container button.vnccs-clothes-btn.vnccs-clothes-btn-primary:not(:disabled):focus,
.vnccs-clothes-container button.vnccs-clothes-btn.vnccs-clothes-btn-primary:not(:disabled):focus-visible,
.vnccs-clothes-container button.vnccs-clothes-btn.vnccs-clothes-btn-primary:not(:disabled):active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    outline: none;
}
.vnccs-clothes-btn-success {
    background: rgba(0,214,143,0.15); color: var(--success); border: 1px solid rgba(0,214,143,0.3);
}
.vnccs-clothes-btn-success:hover:not(:disabled) { background: rgba(0,214,143,0.25); transform: translateY(-1px); }
.vnccs-clothes-btn-danger {
    background: rgba(255,71,87,0.15); color: var(--error); border: 1px solid rgba(255,71,87,0.3);
}
.vnccs-clothes-btn-danger:hover:not(:disabled) { background: rgba(255,71,87,0.25); transform: translateY(-1px); }
.vnccs-clothes-btn:disabled {
    background: rgba(255,255,255,0.04) !important; color: var(--text-muted) !important;
    cursor: not-allowed; box-shadow: none !important; transform: none !important;
}
.vnccs-clothes-btn:focus,
.vnccs-clothes-btn:focus-visible,
.vnccs-clothes-segmented-btn:focus,
.vnccs-clothes-segmented-btn:focus-visible,
.vnccs-clothes-seed-dice-btn:focus,
.vnccs-clothes-seed-dice-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-clothes-btn-primary:focus:not(:disabled),
.vnccs-clothes-btn-primary:focus-visible:not(:disabled),
.vnccs-clothes-btn-primary:active:not(:disabled) {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 8px 28px rgba(255,143,163,0.4), 0 0 0 2px rgba(255,143,163,0.28);
}

.vnccs-clothes-btn-row { display: flex; gap: 8px; margin-top: auto; flex-shrink: 0; flex-wrap: wrap; }
.vnccs-clothes-row { display: flex; gap: 8px; align-items: center; }

.vnccs-clothes-setup-grid {
    --setup-control-height: 58px;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 8px;
    flex-shrink: 0;
}
.vnccs-clothes-setup-grid .vnccs-clothes-field {
    margin-bottom: 0;
    min-width: 0;
}
.vnccs-clothes-setup-grid .vnccs-clothes-label {
    height: 14px;
    line-height: 14px;
}
.vnccs-clothes-resolution-field {
    display: block;
    flex-shrink: 0;
    margin: 4px 0 12px;
}
.vnccs-clothes-resolution-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.vnccs-clothes-resolution-value {
    color: var(--accent-hover);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
}
.vnccs-clothes-resolution-slider {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
}
.vnccs-clothes-segmented-field {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
    height: var(--setup-control-height);
}
.vnccs-clothes-segmented-field.is-three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}
.vnccs-clothes-segmented-btn {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: rgba(255,255,255,0.04);
    color: var(--text-secondary);
    cursor: pointer;
    font-family: var(--font);
    font-size: 11px;
    font-weight: 700;
    box-sizing: border-box;
    height: var(--setup-control-height);
    min-height: var(--setup-control-height);
    padding: 0 10px;
    min-width: 0;
    white-space: nowrap;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition);
}
.vnccs-clothes-segmented-btn:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}
.vnccs-clothes-segmented-btn.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.16);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}
.vnccs-clothes-segmented-field.is-three .vnccs-clothes-segmented-btn {
    padding-inline: 5px;
}
.vnccs-clothes-seed-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 52px;
    gap: 8px;
    align-items: stretch;
    height: var(--setup-control-height);
}
.vnccs-clothes-seed-row .vnccs-clothes-input {
    box-sizing: border-box;
    height: var(--setup-control-height);
    min-height: var(--setup-control-height);
    padding-top: 0;
    padding-bottom: 0;
}
.vnccs-clothes-seed-dice-btn {
    width: 52px;
    min-width: 52px;
    box-sizing: border-box;
    height: var(--setup-control-height);
    min-height: var(--setup-control-height);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: var(--radius-sm);
    background: rgba(255,255,255,0.045);
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition);
}
.vnccs-clothes-seed-dice-btn:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}
.vnccs-clothes-seed-dice-btn.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.16);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}
.vnccs-clothes-seed-dice-btn svg {
    width: 22px;
    height: 22px;
    display: block;
}
.vnccs-clothes-lora-card {
    height: var(--setup-control-height);
    min-height: var(--setup-control-height);
    box-sizing: border-box;
    border: 1px solid rgba(0,214,143,0.28);
    border-radius: var(--radius-md);
    background: rgba(0,214,143,0.05);
    padding: 9px 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
}
.vnccs-clothes-lora-card-top {
    display: grid;
    grid-template-columns: 9px minmax(0, 1fr) auto;
    align-items: center;
    gap: 7px;
}
.vnccs-clothes-lora-card-badge {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--success);
    box-shadow: 0 0 10px rgba(0,214,143,0.35);
}
.vnccs-clothes-lora-card-name {
    min-width: 0;
    color: var(--text-primary);
    font-size: 11px;
    font-weight: 700;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-clothes-lora-card-status {
    color: var(--success);
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.vnccs-clothes-lora-card-desc {
    color: var(--text-secondary);
    font-size: 10px;
    line-height: 1.25;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-clothes-lora-card.is-missing {
    border-color: rgba(255,71,87,0.32);
    background: rgba(255,71,87,0.05);
}
.vnccs-clothes-lora-card.is-missing .vnccs-clothes-lora-card-badge { background: var(--error); box-shadow: none; }
.vnccs-clothes-lora-card.is-missing .vnccs-clothes-lora-card-status { color: var(--error); }

/* Preview */
.vnccs-clothes-preview-container {
    flex: 1;
    background: radial-gradient(circle, rgba(255,143,163,0.04) 1px, transparent 1px), rgba(10,10,15,0.7);
    background-size: 20px 20px, 100% 100%;
    border: 1px solid var(--border); border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; position: relative; min-height: 0;
}
.vnccs-clothes-preview-img {
    width: 100%; height: 100%; object-fit: contain;
    animation: cd-fadein 0.4s ease;
}
.vnccs-clothes-preview-loading {
    position: absolute;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(10, 10, 16, 0.28);
    backdrop-filter: blur(1px);
    pointer-events: none;
}
.vnccs-clothes-preview-loading.is-visible { display: flex; }
.vnccs-clothes-preview-loading::before {
    content: '';
    width: 34px;
    height: 34px;
    border: 2px solid rgba(255, 143, 163, 0.24);
    border-top-color: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 18px rgba(255,143,163,0.25);
    animation: cd-spin 0.75s linear infinite;
}
.cd-sprite-nav {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding-top: 7px;
    pointer-events: auto;
}
.cd-sprite-nav.is-visible { display: flex; }
.cd-sprite-nav-btn {
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
.cd-sprite-nav-btn:hover {
    border-color: var(--accent);
    background: linear-gradient(180deg, rgba(255,143,163,0.22), rgba(255,255,255,0.06));
    box-shadow: 0 0 16px rgba(255,143,163,0.22);
}
.cd-sprite-nav-btn:disabled { opacity: 0.55; cursor: default; }
.cd-sprite-nav-btn svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    stroke-width: 2.6;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.cd-sprite-nav-count {
    min-width: 46px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-secondary);
}
@keyframes cd-fadein { from { opacity: 0; } to { opacity: 1; } }
@keyframes cd-spin { to { transform: rotate(360deg); } }
.vnccs-clothes-placeholder {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--text-muted); font-size: 11px; letter-spacing: 0.05em;
}
.vnccs-clothes-placeholder-icon { width: 48px; height: 48px; opacity: 0.25; }

/* Tab bar */
.cd-tab-bar {
    display: flex; border-bottom: 1px solid var(--border);
    margin: 0 -16px 12px; background: transparent;
}
.cd-tab {
    flex: 1; text-align: center; padding: 10px 8px; cursor: pointer;
    font-size: 10px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--text-muted); border-bottom: 2px solid transparent;
    transition: all var(--transition);
}
.cd-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.cd-tab:hover:not(.active) { color: var(--text-secondary); }

.cd-wizard-btn {
    width: 100%;
    margin-bottom: 10px;
    flex: 0 0 auto;
}
.vnccs-clothes-container .vnccs-common-modal {
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
.vnccs-clothes-container .vnccs-common-modal-title {
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-hover);
    font-family: var(--font);
}
.vnccs-clothes-container .vnccs-common-modal-btn {
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: var(--font);
    font-weight: 700;
}
.vnccs-clothes-container .vnccs-common-modal-btn:focus,
.vnccs-clothes-container .vnccs-common-modal-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-clothes-container .vnccs-common-modal-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    border-color: transparent !important;
}
.vnccs-clothes-container .vnccs-common-modal-btn-primary:hover,
.vnccs-clothes-container .vnccs-common-modal-btn-primary:focus,
.vnccs-clothes-container .vnccs-common-modal-btn-primary:focus-visible,
.vnccs-clothes-container .vnccs-common-modal-btn-primary:active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
}
.cd-wizard-modal {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
}
.cd-wizard-modal-text {
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.45;
    white-space: normal;
    overflow-wrap: anywhere;
}
.cd-wizard-modal textarea {
    width: 100%;
    min-height: 110px;
    box-sizing: border-box;
}
.cd-download-status {
    color: var(--text-primary);
    font-size: 13px;
}
.cd-download-track {
    height: 12px;
    background: rgba(255,255,255,0.12);
    border-radius: 4px;
    overflow: hidden;
}
.cd-download-bar {
    height: 100%;
    width: 0%;
    background: var(--success);
    transition: width 0.25s ease;
}
.cd-download-pct {
    color: var(--text-secondary);
    text-align: right;
    font-size: 12px;
}

/* Clone upload area */
.cd-upload-area {
    border: 1px dashed rgba(255,143,163,0.3);
    background: rgba(255,143,163,0.04);
    border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all var(--transition); position: relative;
    min-height: 200px; overflow: hidden;
}
.cd-upload-area:hover { border-color: var(--accent); background: rgba(255,143,163,0.08); }
.cd-upload-hint { color: var(--text-muted); font-size: 11px; text-align: center; }

/* Loading overlay */
.vnccs-clothes-loading-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(10,10,15,0.92); backdrop-filter: blur(8px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto; gap: 16px; border-radius: var(--radius-lg);
}
.vnccs-clothes-spinner {
    width: 44px; height: 44px; position: relative;
}
.vnccs-clothes-spinner::before, .vnccs-clothes-spinner::after {
    content: ''; position: absolute; inset: 0; border-radius: 50%; border: 3px solid transparent;
}
.vnccs-clothes-spinner::before {
    border-top-color: var(--accent); border-right-color: rgba(255,143,163,0.3);
    animation: cd-spin 1s linear infinite;
    box-shadow: 0 0 18px rgba(255,143,163,0.2);
}
.vnccs-clothes-spinner::after {
    inset: 7px;
    border-bottom-color: rgba(184,169,232,0.6); border-left-color: rgba(184,169,232,0.2);
    animation: cd-spin 1.4s linear infinite reverse;
}
@keyframes cd-spin { to { transform: rotate(360deg); } }
.vnccs-clothes-loading-text { color: var(--text-primary); font-size: 13px; font-weight: 600; }
.vnccs-clothes-loading-dots::after {
    content: ''; animation: cd-dots 1.5s steps(4,end) infinite;
}
@keyframes cd-dots {
    0%,20% { content: ''; } 40% { content: '.'; } 60% { content: '..'; } 80%,100% { content: '...'; }
}
`;

app.registerExtension({
    name: "VNCCS.ClothesDesigner",

    async setup() {
        const queuePrompt = app.queuePrompt;
        const origQueuePrompt = (...args) => queuePrompt.apply(app, args);
        app.queuePrompt = async function(...args) {
            const nodes = app.graph?._nodes?.filter(n => n.type === "ClothesDesigner") || [];
            for (const node of nodes) {
                node._randomizeSeedIfNeeded?.();
            }
            return origQueuePrompt(...args);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ClothesDesigner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1280, 800]);
                syncDOMWidgetWidthSoon(node, "clothes_designer_ui");

                // CSS Injections
                injectStyles(STYLE, "vnccs-clothes-designer");

                const cleanup = () => {
                    if (!node.widgets) return;
                    for (const w of node.widgets) {
                        if (w.name !== "widget_data") w.hidden = true;
                    }
                };
                cleanup();

                // Widget Data Sync
                let dataWidget = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                if (!dataWidget) {
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                dataWidget.hidden = true;

                // State
                const defaultState = {
                    character: "",
                    costume: "Naked",
                    activeTab: "generate", // 'generate' or 'clone'
                    clone_image: null,     // { name, type, subfolder }
                    selected_preview_sprite: null,
                    costume_info: {
                        top: "", bottom: "", head: "", shoes: "", face: ""
                    },
                    character_info: {
                        sex: "female",
                        age: 18
                    },
                    gen_settings: {
                        background_color: "Green",
                        background_model_kind: "",
                        previous_background_color: "Green",
                        target_size: null,
                        seed: 0,
                        seed_mode: "fixed",
                        lora_name: "none",
                        lora_strength: 1.0
                    }
                };

                // Restore State Logic
                let saved = {};
                try {
                    // Priority 1: Widget Data (Graph)
                    if (dataWidget && dataWidget.value && dataWidget.value !== "{}") {
                        saved = JSON.parse(dataWidget.value);
                    } else {
                        // Priority 2: LocalStorage (Session)
                        const ls = storage.getItem("VNCCS_ClothesDesigner_State");
                        if (ls) saved = JSON.parse(ls);
                    }
                } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error loading state", e); }

                const state = {
                    ...defaultState,
                    ...saved,
                    activeTab: saved.activeTab || "generate", // Ensure valid tab
                    costume_info: { ...defaultState.costume_info, ...(saved.costume_info || {}) },
                    character_info: { ...defaultState.character_info, ...(saved.character_info || {}) },
                    gen_settings: { ...defaultState.gen_settings, ...(saved.gen_settings || {}) }
                };
                if (state.gen_settings.target_size != null) {
                    state.gen_settings.target_size = resolutionScaleValue(
                        resolutionScaleMegapixels(state.gen_settings.target_size)
                    );
                }

                const els = {};
                let spritePreviewNavigator = null;
                const beginPreviewRequest = createRequestGuard(node);
                const beginSelectionRequest = createRequestGuard(node);
                const beginDeleteRequest = createRequestGuard(node);
                const beginClothesWizardRequest = createRequestGuard(node);

                const normalizeUploadFile = (file, prefix = "vnccs_upload") => {
                    const originalName = String(file?.name || "").trim();
                    const extMatch = originalName.match(/(\.[A-Za-z0-9]{1,8})$/);
                    const ext = extMatch ? extMatch[1] : ".png";
                    let name = originalName.replace(/[\\/]/g, "_").trim();
                    name = name.replace(/^[\s.-]+/, "");
                    name = name.replace(/\s+/g, "_");
                    name = name.replace(/[^A-Za-z0-9._-]/g, "_");
                    if (!name || !/[A-Za-z0-9]/.test(name)) {
                        name = `${prefix}_${Date.now()}${ext}`;
                    }
                    if (name !== originalName) {
                        name = `${prefix}_${name}`;
                    }
                    return name === file.name ? file : new File([file], name, {
                        type: file.type,
                        lastModified: file.lastModified,
                    });
                };

                const saveState = () => {
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                    try {
                        storage.setItem("VNCCS_ClothesDesigner_State", JSON.stringify(state));
                    } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error saving to localStorage", e); }
                };

                node._randomizeSeedIfNeeded = () => {
                    if (state.gen_settings.seed_mode === "randomize") {
                        state.gen_settings.seed = Math.floor(Math.random() * 10000000000000);
                        if (els.seed) els.seed.value = state.gen_settings.seed;
                        saveState();
                    }
                };

                const onSerialize = node.onSerialize;
                node.onSerialize = function (o) {
                    onSerialize?.apply(this, arguments);
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                    const index = node.widgets?.findIndex(widget => widget.name === "widget_data") ?? -1;
                    if (index >= 0 && Array.isArray(o?.widgets_values)) {
                        o.widgets_values[index] = node.widgets[index].value;
                    }
                };

                const pendingCostumeSaves = new Set();
                const saveCostumeToBackend = async () => {
                    if (!state.character || !state.costume) return;
                    const request = checkedJSON("/vnccs/save_costume", {
                        method: "POST",
                        body: JSON.stringify({ character: state.character, costume: state.costume, info: state.costume_info })
                    });
                    pendingCostumeSaves.add(request);
                    try { return await request; }
                    finally { pendingCostumeSaves.delete(request); }
                };

                // Modal Helper — delegates to vnccs_common showModal
                const showModal = (title, contentFunc, buttons) => {
                    const mappedButtons = buttons.map(b => ({
                        ...b,
                        class: b.class?.includes("danger") ? "danger" : b.class?.includes("primary") ? "primary" : undefined
                    }));
                    return showCommonModal(container, title, contentFunc, mappedButtons);
                };

                const showInfo = (title, msg) => {
                    showModal(title, () => {
                        const d = document.createElement("div");
                        d.innerText = msg;
                        d.style.padding = "10px 0";
                        return d;
                    }, [{ text: "OK", class: "vnccs-clothes-btn-primary" }]);
                };

                const hasSelectedEditableCostume = () => {
                    const costume = String(state.costume || "").trim();
                    return !!costume && costume !== "Naked" && costume !== "Original";
                };

                const showCreateCostumeRequired = () => {
                    showInfo("Costume Required", "Create a new costume first, then select it before generating a preview.");
                };

                const syncCostumeEditControls = () => {
                    const canEdit = hasSelectedEditableCostume();
                    if (els.wizardBtn) els.wizardBtn.disabled = !canEdit;
                    ["top", "bottom", "head", "face", "shoes"].forEach(k => {
                        if (els[k]) els[k].disabled = !canEdit;
                    });
                };

                const onValidationError = (event) => {
                    const targetId = String(event.detail?.node_id);
                    const myId = String(node.id);
                    if (targetId === myId) {
                        showInfo("Costume Required", event.detail?.message || "Create a new costume first, then select it before generating a preview.");
                    }
                };
                api.addEventListener("vnccs.clothes_designer.validation_error", onValidationError);
                registerCleanup(node, () => api.removeEventListener("vnccs.clothes_designer.validation_error", onValidationError));

                const ensureQwenVLReady = async () => {
                    const statusResponse = await api.fetchApi("/vnccs/qwen_vl_model_status?vision=false");
                    if (!statusResponse.ok) throw new Error("Failed to check Qwen3.5 model files.");
                    const modelStatus = await statusResponse.json();
                    if (modelStatus.ready) return true;
                    const approved = await new Promise(resolve => {
                        const { modal } = showModal("Qwen3.5 Model Required", () => {
                            const text = document.createElement("div");
                            text.textContent = `${modelStatus.message || modelStatus.model_name} Download the required files from Hugging Face now?`;
                            return text;
                        }, [
                            { text: "Cancel", action: () => { resolve(false); return false; } },
                            { text: "DOWNLOAD & INSTALL", class: "vnccs-clothes-btn-primary", action: () => { resolve(true); return false; } },
                        ]);
                        modal.addEventListener("keydown", event => {
                            if (event.key === "Escape") resolve(false);
                        }, true);
                    });
                    if (!approved) return false;

                    const start = await api.fetchApi("/vnccs/qwen_vl_download_model?vision=false", { method: "POST" });
                    if (!start.ok && start.status !== 409) {
                        let err;
                        try { err = await start.json(); } catch (e) { err = { error: await start.text() }; }
                        throw new Error(err?.error || err?.message || "Failed to start QwenVL download.");
                    }

                    const { overlay, modal } = showModal("Downloading Model...", () => {
                        const d = document.createElement("div");
                        d.className = "cd-wizard-modal";
                        d.innerHTML = `
                            <div class="cd-download-status" id="cd-dl-status">Starting...</div>
                            <div class="cd-download-track">
                                <div class="cd-download-bar" id="cd-dl-bar"></div>
                            </div>
                            <div class="cd-download-pct" id="cd-dl-pct">0%</div>
                        `;
                        return d;
                    }, []);

                    const statusEl = modal.querySelector("#cd-dl-status");
                    const barEl = modal.querySelector("#cd-dl-bar");
                    const pctEl = modal.querySelector("#cd-dl-pct");

                    return await new Promise((resolve, reject) => {
                        const poll = async () => {
                            try {
                                const r = await api.fetchApi("/vnccs/qwen_vl_download_status");
                                if (!r.ok) throw new Error(await r.text());
                                const d = await r.json();
                                const progress = Math.max(0, Math.min(100, Number(d.progress) || 0));
                                statusEl.innerText = d.current_file ? `Downloading ${d.current_file}...` : "Preparing model files...";
                                barEl.style.width = `${progress}%`;
                                pctEl.innerText = `${progress}%`;

                                if (d.status === "completed") {
                                    statusEl.innerText = "Download Complete!";
                                    barEl.style.width = "100%";
                                    pctEl.innerText = "100%";
                                    setTimeout(() => overlay.remove(), 450);
                                    resolve(true);
                                    return;
                                }
                                if (d.status === "error") {
                                    overlay.remove();
                                    reject(new Error(d.error || "QwenVL download failed."));
                                    return;
                                }
                                setTimeout(poll, 700);
                            } catch (e) {
                                overlay.remove();
                                reject(e);
                            }
                        };
                        poll();
                    });
                };

                const showWizardModelError = (err) => {
                    if (err?.error === "DEPENDENCY_MISSING") {
                        showModal("Dependency Missing", () => {
                            const d = document.createElement("div");
                            d.className = "cd-wizard-modal-text";
                            d.innerText = `${err.message}\n\nInstall a compatible llama-cpp-python build manually.`;
                            return d;
                        }, [{ text: "OK", class: "vnccs-clothes-btn-danger" }]);
                        return;
                    }

                    if (["MODEL_MISSING", "MODEL_INVALID", "MMPROJ_MISSING", "MMPROJ_INVALID", "MODEL_DOWNLOAD_FAILED"].includes(err?.error)) {
                        showModal("Model Missing", () => {
                            const d = document.createElement("div");
                            d.className = "cd-wizard-modal-text";
                            d.innerText = `${err.message || "Required Qwen model is missing."}\n\nDownload the Qwen model files now?`;
                            return d;
                        }, [
                            { text: "Cancel" },
                            {
                                text: "DOWNLOAD & INSTALL",
                                class: "vnccs-clothes-btn-primary",
                                action: async () => {
                                    try {
                                        const dl = await api.fetchApi("/vnccs/qwen_vl_download_model?vision=false", { method: "POST" });
                                        if (dl.ok || dl.status === 409) {
                                            ensureQwenVLReady().catch(e => showInfo("Error", String(e)));
                                            return false;
                                        }
                                        showInfo("Error", await dl.text());
                                    } catch (e) {
                                        showInfo("Error", `Download trigger failed: ${e}`);
                                    }
                                    return true;
                                }
                            }
                        ]);
                        return;
                    }

                    showInfo("Clothes Wizard Error", err?.message || err?.raw || "Failed to generate clothes description.");
                };

                const openClothesWizard = () => {
                    if (!hasSelectedEditableCostume()) {
                        showCreateCostumeRequired();
                        return;
                    }
                    const character = state.character;
                    const costume = state.costume;
                    const currentRequest = beginClothesWizardRequest();
                    let input;
                    showModal("Clothes Wizzard", () => {
                        const wrap = document.createElement("div");
                        wrap.className = "cd-wizard-modal";
                        const text = document.createElement("div");
                        text.className = "cd-wizard-modal-text";
                        text.innerText = "Describe the outfit in a broad way. The model will expand it into detailed clothing parts.";
                        input = document.createElement("textarea");
                        input.className = "vnccs-clothes-textarea";
                        input.placeholder = "e.g. Santa Claus costume";
                        wrap.append(text, input);
                        setTimeout(() => input.focus(), 50);
                        return wrap;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "FILL FIELDS",
                            class: "vnccs-clothes-btn-primary",
                            action: async (overlay, btn) => {
                                const isCurrent = () => currentRequest() && overlay.isConnected &&
                                    state.character === character && state.costume === costume;
                                if (!isCurrent()) return false;
                                const description = input.value.trim();
                                if (!description) {
                                    input.focus();
                                    return true;
                                }
                                btn.disabled = true;
                                btn.innerText = "CHECKING MODEL...";
                                try {
                                    if (!await ensureQwenVLReady()) return isCurrent();
                                    if (!isCurrent()) return false;
                                    btn.innerText = "THINKING...";
                                    const r = await api.fetchApi("/vnccs/clothes_wizard", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ description, node_id: node.id })
                                    });
                                    if (!isCurrent()) return false;
                                    if (!r.ok) {
                                        let err = null;
                                        try { err = await r.json(); } catch (e) { err = { message: await r.text() }; }
                                        showWizardModelError(err);
                                        return false;
                                    }
                                    const data = await r.json();
                                    if (!isCurrent()) return false;
                                    ["top", "bottom", "shoes", "head", "face"].forEach((key) => {
                                        state.costume_info[key] = data[key] || "";
                                        if (els[key]) {
                                            els[key].value = state.costume_info[key];
                                            if (els[key].autoResize) els[key].autoResize();
                                        }
                                    });
                                    saveState();
                                    await saveCostumeToBackend();
                                    return false;
                                } catch (e) {
                                    if (!isCurrent()) return false;
                                    showInfo("Clothes Wizard Error", e.toString());
                                    return true;
                                } finally {
                                    btn.disabled = false;
                                    btn.innerText = "FILL FIELDS";
                                }
                            }
                        }
                    ]);
                };

                const getConnectedControlCenterState = () => {
                    const resolveUpstreamNode = (startNode, inputName, maxDepth = 20) => {
                        let currentNode = startNode;
                        let currentInputName = inputName;

                        for (let depth = 0; depth < maxDepth; depth++) {
                            const input = currentNode?.inputs?.find((item) => item.name === currentInputName)
                                ?? currentNode?.inputs?.[0];
                            const linkId = input?.link;
                            if (linkId == null) return null;

                            const link = app.graph?.links?.[linkId];
                            const originId = link?.origin_id ?? link?.originId;
                            if (originId == null) return null;

                            const upstream = app.graph?.getNodeById?.(originId) ?? app.graph?._nodes_by_id?.[originId] ?? null;
                            if (!upstream) return null;

                            const comfyClass = upstream.comfyClass || upstream.type || "";
                            if (comfyClass === "VNCCS_ControlCenter") return upstream;

                            currentNode = upstream;
                            currentInputName = "pipe";
                        }

                        return null;
                    };

                    const upstream = resolveUpstreamNode(node, "pipe");
                    if (!upstream) return null;

                    const repoWidget = upstream.widgets?.find((w) => w.name === "repo_id");
                    const stateWidget = upstream.widgets?.find((w) => w.name === "node_state");
                    const repo_id = String(repoWidget?.value ?? "").trim();
                    const node_state = typeof stateWidget?.value === "string"
                        ? stateWidget.value
                        : JSON.stringify(stateWidget?.value ?? {});

                    if (!repo_id || !node_state) return null;
                    let selected_type = "";
                    try {
                        selected_type = String(JSON.parse(node_state)?.selected_type || "").toLowerCase();
                    } catch {
                        selected_type = "";
                    }
                    return { repo_id, node_state, selected_type, control_center_id: String(upstream.id) };
                };


                const getConnectedControlCenterWidget = () => {
                    let currentNode = node;
                    let currentInputName = "pipe";
                    for (let depth = 0; depth < 20; depth++) {
                        const input = currentNode?.inputs?.find((item) => item.name === currentInputName)
                            ?? currentNode?.inputs?.[0];
                        const linkId = input?.link;
                        if (linkId == null) return null;
                        const link = app.graph?.links?.[linkId];
                        const originId = link?.origin_id ?? link?.originId;
                        if (originId == null) return null;
                        const upstream = app.graph?.getNodeById?.(originId) ?? app.graph?._nodes_by_id?.[originId] ?? null;
                        if (!upstream) return null;
                        if ((upstream.comfyClass || upstream.type || "") === "VNCCS_ControlCenter") {
                            return upstream._cc_widget || null;
                        }
                        currentNode = upstream;
                        currentInputName = "pipe";
                    }
                    return null;
                };

                const normalizeLoraPath = (entry) => {
                    const raw = String(entry?.local_path || entry || "").replace(/\\/g, "/");
                    if (!raw) return "";
                    const lower = raw.toLowerCase();
                    const absoluteMarker = "/models/loras/";
                    const absoluteIndex = lower.lastIndexOf(absoluteMarker);
                    if (absoluteIndex >= 0) return raw.slice(absoluteIndex + absoluteMarker.length);
                    if (lower.startsWith("models/loras/")) return raw.slice("models/loras/".length);
                    if (lower.startsWith("loras/")) return raw.slice("loras/".length);
                    return raw;
                };

                const isClothesCoreLora = (value) => {
                    const normalized = String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
                    return normalized.includes("vnccs") && normalized.includes("clothes") && normalized.includes("core");
                };

                const getConnectedModelKind = () => {
                    const ccWidget = getConnectedControlCenterWidget();
                    try {
                        const widgetKind = ccWidget?._selectedKind?.() || ccWidget?._activeKind?.() || ccWidget?.state?.active_kind;
                        if (widgetKind) return String(widgetKind).trim();
                    } catch {
                        // Fall through to the serialized Control Center state.
                    }
                    try {
                        const connected = getConnectedControlCenterState();
                        const parsed = JSON.parse(connected?.node_state || "{}");
                        return String(parsed?.active_kind || "").trim();
                    } catch {
                        return "";
                    }
                };

                const loraMatchesConnectedKind = (entry) => {
                    const selectedKind = getConnectedModelKind().toLowerCase();
                    if (!selectedKind) return true;
                    const entryKind = String(entry?.kind ?? entry?.Kind ?? "").trim().toLowerCase();
                    return entryKind === selectedKind;
                };

                const findConnectedClothesCoreLora = (preferredPath = "") => {
                    const entries = getConnectedControlCenterWidget()?.config?.lora || [];
                    const preferred = normalizeLoraPath(preferredPath).toLowerCase();
                    const compatible = entries.filter(entry =>
                        loraMatchesConnectedKind(entry)
                        && (isClothesCoreLora(entry?.name) || isClothesCoreLora(entry?.local_path))
                    );
                    if (preferred) {
                        const exact = compatible.find(entry => normalizeLoraPath(entry).toLowerCase() === preferred);
                        if (exact) return exact;
                    }
                    return compatible[0] || null;
                };

                const setClothesCoreLora = (entryOrPath = null) => {
                    syncResolutionControl();
                    let entry = entryOrPath && typeof entryOrPath === "object" ? entryOrPath : null;
                    let rel = entry ? normalizeLoraPath(entry) : normalizeLoraPath(entryOrPath);
                    if (rel && !isClothesCoreLora(rel) && !isClothesCoreLora(entry?.name)) {
                        entry = null;
                        rel = "";
                    }

                    if (entry && !loraMatchesConnectedKind(entry)) {
                        entry = null;
                        rel = "";
                    }
                    if (!entry) {
                        entry = findConnectedClothesCoreLora(rel || state.gen_settings.lora_name);
                        rel = normalizeLoraPath(entry);
                    }

                    if (rel) {
                        state.gen_settings.lora_name = rel;
                        const strength = Number(state.gen_settings.lora_strength ?? 1.0) || 1.0;
                        state.gen_settings.lora_strength = Math.max(0, Math.min(1, strength));
                    } else {
                        state.gen_settings.lora_name = "none";
                    }
                    renderClothesCoreLoraCard(entry || rel || null);
                    saveState();
                };

                const normalizeAgeValue = (value) => {
                    const parsed = parseFloat(value);
                    if (!Number.isFinite(parsed)) return 18;
                    return Math.max(1, Math.min(100, parsed));
                };
                const parsePoseStudioValues = () => {
                    const info = state.character_info || {};
                    const age = Number(info.age);
                    const sex = String(info.sex || info.gender || "").toLowerCase();
                    const gender = sex === "male" ? 1.0 : (sex === "female" ? 0.0 : NaN);
                    return {
                        age: Number.isFinite(age) ? age : NaN,
                        gender,
                        signature: `${Number.isFinite(age) ? Math.round(age) : "?"}|${Number.isFinite(gender) ? gender : "?"}`
                    };
                };
                let poseStudioSyncRetryCount = 0;
                const patchPoseStudioSync = () => {
                    const sync = window.__vnccsPoseStudioCharacterCreatorSync;
                    if (!sync || sync._vnccsClothesDesignerPatched) return !!sync;
                    const findSourceNode = sync.findSourceNode;
                    const registerStudio = sync.registerStudio;
                    const originalFindSourceNode = typeof findSourceNode === "function"
                        ? (...args) => findSourceNode.apply(sync, args)
                        : null;
                    const originalRegisterStudio = typeof registerStudio === "function"
                        ? (...args) => registerStudio.apply(sync, args)
                        : null;

                    sync.findClothesDesignerSourceNode = () => {
                        const nodes = app.graph?._nodes || [];
                        return nodes.find(n => n?.type === "ClothesDesigner") || null;
                    };
                    sync.findSourceNode = function () {
                        return originalFindSourceNode?.() || this.findClothesDesignerSourceNode?.() || null;
                    };
                    sync.applyClothesDesignerSource = function (sourceNode, options = {}) {
                        const values = sourceNode?._vnccsGetPoseStudioValues?.();
                        if (!values) return false;
                        if (!options.initial && !options.force) {
                            const previous = this.sourceSignatures?.get(sourceNode);
                            if (previous === values.signature) return false;
                        }
                        this.sourceSignatures?.set(sourceNode, values.signature);
                        let applied = false;
                        for (const studio of this.studios || []) {
                            applied = this.applyToStudio?.(studio, values, options) || applied;
                        }
                        return applied;
                    };
                    sync.hookClothesDesignerNode = function (sourceNode) {
                        if (!sourceNode || sourceNode.type !== "ClothesDesigner") return;
                        const sourceWidget = sourceNode.widgets?.find(w => w.name === "widget_data");
                        let didHook = false;
                        if (sourceWidget && !sourceWidget._vnccsPoseStudioClothesDesignerValueHooked) {
                            let currentValue = sourceWidget.value;
                            Object.defineProperty(sourceWidget, "value", {
                                configurable: true,
                                get() {
                                    return currentValue;
                                },
                                set: (value) => {
                                    currentValue = value;
                                    queueMicrotask(() => this.applyClothesDesignerSource(sourceNode));
                                }
                            });
                            sourceWidget._vnccsPoseStudioClothesDesignerValueHooked = true;
                            didHook = true;
                        }
                        if (didHook) this.applyClothesDesignerSource(sourceNode, { initial: true });
                    };
                    sync.registerStudio = function (studio) {
                        originalRegisterStudio?.(studio);
                        const designer = this.findClothesDesignerSourceNode?.();
                        if (designer) this.applyClothesDesignerSource(designer, { force: true });
                    };
                    sync._vnccsClothesDesignerPatched = true;
                    return true;
                };
                const applyPoseStudioValues = (options = {}) => {
                    node._vnccsGetPoseStudioValues = parsePoseStudioValues;
                    if (patchPoseStudioSync()) {
                        poseStudioSyncRetryCount = 0;
                        const sync = window.__vnccsPoseStudioCharacterCreatorSync;
                        const values = parsePoseStudioValues();
                        sync?.hookClothesDesignerNode?.(node);
                        sync?.sourceSignatures?.set(node, values.signature);
                        for (const studio of sync?.studios || []) {
                            const applied = studio.applyExternalCharacterCreatorValues?.(values, options);
                            if (!applied && Number.isFinite(values.age)) {
                                const age = Math.max(1, Math.min(90, Math.round(values.age)));
                                if (studio.meshParams?.age !== age && studio.meshParams) {
                                    studio.meshParams.age = age;
                                }
                                studio._suppressNextAgeFitSync = false;
                                studio.onMeshParamsChanged?.("age");
                            }
                            if (!applied && Number.isFinite(values.gender) && studio.meshParams?.gender !== values.gender) {
                                studio.setManagerGender?.(values.gender);
                            }
                        }
                    } else if (poseStudioSyncRetryCount < 40) {
                        poseStudioSyncRetryCount += 1;
                        setTimeout(() => applyPoseStudioValues(options), 250);
                    }
                };
                const beginCharacterInfoRequest = createRequestGuard(node);
                const loadCharacterInfo = async () => {
                    const currentRequest = beginCharacterInfoRequest();
                    const character = state.character;
                    if (!character) return false;
                    try {
                        const r = await api.fetchApi(`/vnccs/character_info?character=${encodeURIComponent(character)}`);
                        if (!r.ok) throw new Error(`Character metadata request failed (${r.status})`);
                        const info = await r.json();
                        if (!currentRequest() || state.character !== character) return false;
                        state.character_info = {
                            ...state.character_info,
                            ...info,
                            age: normalizeAgeValue(info.age ?? state.character_info.age),
                            sex: String(info.sex || info.gender || state.character_info.sex || "female").toLowerCase()
                        };
                        saveState();
                        applyPoseStudioValues({ force: true });
                        return true;
                    } catch (e) {
                        console.warn("[VNCCS] ClothesDesigner: Failed to load character info", e);
                        return false;
                    }
                };

                // UI Builders
                const FIELD_HELP = {
                    character: "Character whose current body/sprite will be used as the base for clothing generation.",
                    costume: "Costume slot to edit or regenerate.",
                    top: "Upper-body clothing description, such as shirt, jacket, dress top, sleeves, and colors.",
                    bottom: "Lower-body clothing description, such as skirt, pants, shorts, belts, and colors.",
                    shoes: "Footwear description used in the clothing prompt.",
                    head: "Headwear and hair accessories, such as hats, ribbons, crowns, or headphones.",
                    face: "Face accessories, such as glasses, mask, piercings, or makeup tied to the outfit.",
                    background_color: "Sets a solid chroma key background or native transparency for Qwen Image 2.1.",
                    lora_name: "VNCCS Clothes Core LoRA used to keep outfit generation compatible with this workflow.",
                    seed: "Numeric seed for reproducible clothing previews.",
                    seed_mode: "Toggles fixed seed versus a fresh random seed for each preview."
                };
                const helpFor = (key, fallback = "") => FIELD_HELP[key] || fallback;

                const createField = (key, placeholder, multiline = true) => {
                    const wrap = document.createElement("div"); wrap.className = "vnccs-clothes-field";
                    setHelpText(wrap, helpFor(key));
                    const l = document.createElement("div"); l.className = "vnccs-clothes-label";
                    l.innerText = key.toUpperCase();
                    wrap.appendChild(l);

                    const inp = document.createElement(multiline ? "textarea" : "input");
                    inp.className = multiline ? "vnccs-clothes-textarea" : "vnccs-clothes-input";
                    if (placeholder) inp.placeholder = placeholder;

                    inp.value = state.costume_info[key] || "";

                    if (multiline) {
                        inp.style.resize = "none";
                        inp.style.overflow = "hidden";
                        const autoResize = () => {
                            inp.style.height = "auto";
                            inp.style.height = inp.scrollHeight + "px";
                        };
                        inp.addEventListener("input", autoResize);
                        setTimeout(autoResize, 10);
                        inp.autoResize = autoResize;
                    }

                    inp.oninput = (e) => {
                        state.costume_info[key] = e.target.value;
                        saveState();
                    };
                    inp.onchange = (e) => {
                        inp.oninput(e);
                        saveCostumeToBackend().catch(error => showInfo("Save Failed", error.message));
                    };
                    els[key] = inp;
                    wrap.appendChild(inp);
                    return wrap;
                };

                const createSegmentedField = (lbl, key, options) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-clothes-field";
                    setHelpText(wrap, helpFor(key));
                    const label = document.createElement("div");
                    label.className = "vnccs-clothes-label";
                    label.innerText = lbl;
                    const segmented = document.createElement("div");
                    segmented.className = "vnccs-clothes-segmented-field";
                    if (options.length === 3) segmented.classList.add("is-three");
                    segmented.setAttribute("role", "group");
                    segmented.setAttribute("aria-label", lbl);
                    const buttons = [];

                    const setValue = (value, persist = false) => {
                        const raw = String(value || options[0]?.value || "");
                        const matched = options.find(option => String(option.value).toLowerCase() === raw.toLowerCase());
                        const normalized = matched?.value || raw;
                        state.gen_settings[key] = normalized;
                        if (
                            persist
                            && key === "background_color"
                            && getConnectedModelKind().trim().toLowerCase() === "qi2"
                            && normalized !== "Transparent"
                        ) {
                            state.gen_settings.previous_background_color = normalized;
                        }
                        buttons.forEach(({ btn, value: btnValue }) => {
                            const selected = btnValue === normalized;
                            btn.classList.toggle("is-active", selected);
                            btn.setAttribute("aria-pressed", String(selected));
                        });
                        if (persist) saveState();
                    };

                    options.forEach(option => {
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "vnccs-clothes-segmented-btn";
                        btn.innerText = option.label;
                        btn.onclick = () => setValue(option.value, true);
                        buttons.push({ btn, value: option.value });
                        segmented.appendChild(btn);
                    });

                    els[key] = { setValue };
                    wrap.append(label, segmented);
                    setValue(state.gen_settings[key] || options[0]?.value);
                    return wrap;
                };

                const syncResolutionControl = () => {
                    if (!els.target_size) return;
                    const kind = getConnectedModelKind().toLowerCase().replace(/[^a-z0-9]/g, "");
                    const defaultSize = ["h3", "minimaxh3"].includes(kind) ? 1536 : 1024;
                    const size = state.gen_settings.target_size == null ? defaultSize : state.gen_settings.target_size;
                    els.target_size.value = resolutionScaleMegapixels(size).toFixed(1);
                    if (els.target_size_value) {
                        const suffix = state.gen_settings.target_size == null ? " · Auto" : "";
                        els.target_size_value.textContent = `${resolutionScaleText(size)}${suffix}`;
                    }
                };

                const syncBackgroundForModel = () => {
                    const kind = getConnectedModelKind().trim().toLowerCase();
                    if (!kind || state.gen_settings.background_model_kind === kind) return false;
                    const current = String(state.gen_settings.background_color || "Green");
                    if (kind === "qi2") {
                        if (current !== "Transparent") {
                            state.gen_settings.previous_background_color = current;
                        }
                        state.gen_settings.background_color = "Transparent";
                    } else if (current === "Transparent") {
                        const previous = String(state.gen_settings.previous_background_color || "Green");
                        state.gen_settings.background_color = ["Green", "Blue"].includes(previous) ? previous : "Green";
                    }
                    state.gen_settings.background_model_kind = kind;
                    return true;
                };

                const syncGenerationControls = () => {
                    const backgroundChanged = syncBackgroundForModel();
                    syncResolutionControl();
                    if (els.seed) els.seed.value = state.gen_settings.seed || 0;
                    if (els.seed_mode) {
                        const randomize = (state.gen_settings.seed_mode || "fixed") === "randomize";
                        els.seed_mode.classList.toggle("is-active", randomize);
                        els.seed_mode.title = randomize ? "Random seed on queue" : "Fixed seed";
                    }
                    els.background_color?.setValue?.(state.gen_settings.background_color || "Green");
                    renderClothesCoreLoraCard();
                    if (backgroundChanged) saveState();
                };

                const renderClothesCoreLoraCard = (entryOrPath = null) => {
                    const card = els.clothesCoreLoraCard;
                    if (!card) return;
                    let rel = normalizeLoraPath(entryOrPath) || state.gen_settings.lora_name || "";
                    let entry = entryOrPath && typeof entryOrPath === "object"
                        ? entryOrPath
                        : findConnectedClothesCoreLora(rel);
                    if (entry && !loraMatchesConnectedKind(entry)) entry = null;
                    if (!entry) entry = findConnectedClothesCoreLora(rel);
                    if (entry) rel = normalizeLoraPath(entry);
                    const name = entry?.name || (rel ? rel.split(/[\\/]/).pop().replace(/\.safetensors$/i, "") : "VNCCS Clothes Core");
                    const hasLora = !!rel && rel !== "none";
                    card.classList.toggle("is-missing", !hasLora);
                    card.innerHTML = "";
                    const top = document.createElement("div");
                    top.className = "vnccs-clothes-lora-card-top";
                    const badge = document.createElement("span");
                    badge.className = "vnccs-clothes-lora-card-badge";
                    const nameEl = document.createElement("div");
                    nameEl.className = "vnccs-clothes-lora-card-name";
                    nameEl.innerText = name;
                    const status = document.createElement("div");
                    status.className = "vnccs-clothes-lora-card-status";
                    status.innerText = hasLora ? "Core" : "Missing";
                    top.append(badge, nameEl, status);
                    card.appendChild(top);
                    const desc = document.createElement("div");
                    desc.className = "vnccs-clothes-lora-card-desc";
                    desc.innerText = hasLora ? rel : "Connect VNCCS Control Center with VNCCS Clothes Core.";
                    card.appendChild(desc);
                };

                const createResolutionControl = () => {
                    const resolutionWrap = document.createElement("label");
                    resolutionWrap.className = "vnccs-clothes-field vnccs-clothes-resolution-field";
                    setHelpText(resolutionWrap, "Sets the generated image area from 1.0 to 4.0 megapixels while preserving aspect ratio. Auto uses 1.5 MP for H3 and 1.0 MP for other models.");
                    const resolutionHead = document.createElement("div");
                    resolutionHead.className = "vnccs-clothes-resolution-head";
                    const resolutionLabel = document.createElement("div");
                    resolutionLabel.className = "vnccs-clothes-label";
                    resolutionLabel.textContent = "Resolution scale";
                    const resolutionValue = document.createElement("div");
                    resolutionValue.className = "vnccs-clothes-resolution-value";
                    resolutionHead.append(resolutionLabel, resolutionValue);
                    const resolutionSlider = document.createElement("input");
                    resolutionSlider.type = "range";
                    resolutionSlider.className = "vnccs-clothes-resolution-slider";
                    resolutionSlider.min = String(RESOLUTION_SCALE_MIN_MP);
                    resolutionSlider.max = String(RESOLUTION_SCALE_MAX_MP);
                    resolutionSlider.step = String(RESOLUTION_SCALE_STEP_MP);
                    resolutionSlider.setAttribute("aria-label", "Resolution scale in megapixels");
                    resolutionSlider.oninput = () => {
                        state.gen_settings.target_size = resolutionScaleValue(resolutionSlider.value);
                        resolutionValue.textContent = resolutionScaleText(state.gen_settings.target_size);
                        saveState();
                    };
                    els.target_size = resolutionSlider;
                    els.target_size_value = resolutionValue;
                    resolutionWrap.append(resolutionHead, resolutionSlider);
                    syncResolutionControl();

                    return resolutionWrap;
                };

                const createGenerationControls = () => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-clothes-setup-grid";

                    wrap.appendChild(createSegmentedField("Background", "background_color", [
                        { label: "Green", value: "Green" },
                        { label: "Blue", value: "Blue" },
                        { label: "Alpha", value: "Transparent" },
                    ]));

                    const loraWrap = document.createElement("div");
                    loraWrap.className = "vnccs-clothes-field";
                    setHelpText(loraWrap, helpFor("lora_name"));
                    loraWrap.innerHTML = '<div class="vnccs-clothes-label">VNCCS Clothes Core</div>';
                    const loraCard = document.createElement("div");
                    loraCard.className = "vnccs-clothes-lora-card";
                    els.clothesCoreLoraCard = loraCard;
                    loraWrap.appendChild(loraCard);
                    wrap.appendChild(loraWrap);

                    const seedWrap = document.createElement("div");
                    seedWrap.className = "vnccs-clothes-field";
                    setHelpText(seedWrap, helpFor("seed"));
                    seedWrap.innerHTML = '<div class="vnccs-clothes-label">Seed</div>';
                    const seedRow = document.createElement("div");
                    seedRow.className = "vnccs-clothes-seed-row";
                    const seedInp = document.createElement("input");
                    seedInp.className = "vnccs-clothes-input";
                    seedInp.type = "number";
                    seedInp.value = state.gen_settings.seed || 0;
                    seedInp.onchange = (e) => {
                        state.gen_settings.seed = Number(e.target.value) || 0;
                        saveState();
                    };
                    els.seed = seedInp;
                    const seedMode = document.createElement("button");
                    seedMode.type = "button";
                    seedMode.className = "vnccs-clothes-seed-dice-btn";
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
                    seedRow.append(seedInp, seedMode);
                    seedWrap.appendChild(seedRow);
                    wrap.appendChild(seedWrap);
                    return wrap;
                };

                // Helper to render Attributes Panel (Original Tab)
                const renderAttributes = (container) => {
                    container.innerHTML = '';
                    const wizardBtn = document.createElement("button");
                    wizardBtn.type = "button";
                    wizardBtn.className = "vnccs-clothes-btn vnccs-clothes-btn-primary cd-wizard-btn";
                    wizardBtn.innerText = "CLOTHES WIZZARD";
                    wizardBtn.onclick = openClothesWizard;
                    wizardBtn.disabled = !hasSelectedEditableCostume();
                    els.wizardBtn = wizardBtn;
                    container.appendChild(wizardBtn);
                    container.appendChild(createField("top", "e.g. White t-shirt"));
                    container.appendChild(createField("bottom", "e.g. Blue jeans"));
                    container.appendChild(createField("shoes", "e.g. Sneakers"));
                    container.appendChild(createField("head", "e.g. Hat"));
                    container.appendChild(createField("face", "Face features (e.g. glasses)"));
                };

                // Helper to render Clone Panel (New Tab)
                const renderClonePanel = (container) => {
                    container.innerHTML = '';

                    const desc = document.createElement("div");
                    desc.style.fontSize = "12px"; desc.style.color = "#aaa"; desc.style.marginBottom = "10px";
                    desc.innerText = "Upload an image to clone clothes from. The AI will attempt to transfer the outfit to your character.";
                    container.appendChild(desc);

                    // Preview / Upload Area
                    const pContainer = document.createElement("div");
                    pContainer.className = "vnccs-clothes-preview-container";
                    pContainer.style.height = "250px";
                    pContainer.style.background = "#151515";
                    pContainer.style.position = "relative";
                    pContainer.style.display = "flex";
                    pContainer.style.alignItems = "center";
                    pContainer.style.justifyContent = "center";
                    pContainer.style.border = "1px solid #444";

                    const img = document.createElement("img");
                    img.style.maxWidth = "100%"; img.style.maxHeight = "100%"; img.style.objectFit = "contain";
                    img.style.display = "none";
                    if (state.clone_image) {
                        const params = new URLSearchParams();
                        params.set("filename", state.clone_image.name || "");
                        params.set("type", state.clone_image.type || "input");
                        if (state.clone_image.subfolder) params.set("subfolder", state.clone_image.subfolder);
                        img.src = api.apiURL(`/view?${params.toString()}`);
                        img.style.display = "block";
                    }
                    pContainer.appendChild(img);

                    // Upload Button Overlay
                    const overlay = document.createElement("div");
                    overlay.style.position = "absolute"; overlay.style.inset = "0";
                    overlay.style.display = "flex"; overlay.style.alignItems = "center"; overlay.style.justifyContent = "center";
                    overlay.style.cursor = "pointer";

                    const btn = document.createElement("button");
                    btn.className = "vnccs-clothes-btn";
                    btn.style.background = "#444"; btn.style.border = "1px dashed #666";
                    btn.innerText = state.clone_image ? "REPLACE IMAGE" : "+ UPLOAD IMAGE";
                    if (state.clone_image) { btn.style.opacity = "0.8"; btn.style.fontSize = "10px"; btn.style.padding = "4px 8px"; btn.style.position = "absolute"; btn.style.bottom = "10px"; }

                    overlay.appendChild(btn);

                    const fileInp = document.createElement("input");
                    fileInp.type = "file"; fileInp.accept = "image/*"; fileInp.style.display = "none";
                    fileInp.onchange = async (e) => {
                        if (e.target.files.length) {
                            const file = normalizeUploadFile(e.target.files[0], "clone_reference");
                            try {
                                btn.innerText = "UPLOADING...";
                                const body = new FormData();
                                body.append("image", file, file.name);
                                body.append("type", "input");
                                body.append("overwrite", "true");
                                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                                if (!resp.ok) {
                                    throw new Error(await resp.text() || `Upload failed (${resp.status})`);
                                }
                                const json = await resp.json();
                                if (!json.name) {
                                    throw new Error("Upload response did not include an image name.");
                                }
                                state.clone_image = {
                                    name: json.name,
                                    type: "input",
                                    subfolder: json.subfolder || ""
                                };
                                saveState();
                                // Update View
                                renderClonePanel(container);
                            } catch (err) {
                                showInfo("Upload Failed", err.toString());
                            }
                        }
                    };

                    overlay.onclick = (e) => {
                        e.stopPropagation();
                        fileInp.click();
                    };

                    pContainer.appendChild(overlay);
                    container.appendChild(pContainer);
                };


                // --- MAIN LAYOUT ---
                const container = document.createElement("div"); container.className = "vnccs-clothes-container";
                const topRow = document.createElement("div"); topRow.className = "vnccs-clothes-top-row";

                // --- COL 1: DESIGN STUDIO ---
                const colLeft = document.createElement("div"); colLeft.className = "vnccs-clothes-col";
                colLeft.innerHTML = '<div class="vnccs-clothes-section-title">Design Studio</div>';

                // Character Select
                const charRow = document.createElement("div"); charRow.className = "vnccs-clothes-field";
                setHelpText(charRow, helpFor("character"));
                charRow.innerHTML = '<div class="vnccs-clothes-label">CHARACTER</div>';
                const charSel = document.createElement("select"); charSel.className = "vnccs-clothes-select";
                charSel.onchange = async (e) => {
                    state.character = e.target.value;
                    beginClothesWizardRequest();
                    const currentSelection = beginSelectionRequest();
                    const currentPreview = beginPreviewRequest();
                    spritePreviewNavigator?.invalidate?.();
                    if (!await loadCharacterInfo() || !currentSelection()) return;
                    if (!await loadCostumes() || !currentSelection()) return;
                    if (currentPreview()) updatePreviewImage();
                    saveState();
                };
                charRow.appendChild(charSel);
                els.charSelect = charSel;
                colLeft.appendChild(charRow);

                // Costume Select
                const costRow = document.createElement("div"); costRow.className = "vnccs-clothes-field";
                setHelpText(costRow, helpFor("costume"));
                costRow.innerHTML = '<div class="vnccs-clothes-label">COSTUME (Select to Edit)</div>';
                const costSel = document.createElement("select"); costSel.className = "vnccs-clothes-select";
                costSel.onchange = async (e) => {
                    state.costume = e.target.value;
                    beginClothesWizardRequest();
                    beginSelectionRequest();
                    const currentPreview = beginPreviewRequest();
                    spritePreviewNavigator?.invalidate?.();
                    if (!await loadCostumeInfo()) return false;
                    syncCostumeEditControls();
                    if (currentPreview()) updatePreviewImage();
                    saveState();
                };
                els.costSel = costSel;
                costRow.appendChild(costSel);
                colLeft.appendChild(costRow);

                // Action Buttons
                const actionRow = document.createElement("div"); actionRow.className = "vnccs-clothes-btn-row";
                actionRow.style.marginBottom = "10px";

                const btnNewCostume = document.createElement("button");
                btnNewCostume.className = "vnccs-clothes-btn vnccs-clothes-btn-success";
                btnNewCostume.innerText = "NEW";
                btnNewCostume.style.fontSize = "10px";
                btnNewCostume.onclick = () => {
                    showModal("New Costume Name", () => {
                        const inp = document.createElement("input"); inp.className = "vnccs-clothes-input";
                        return inp;
                    }, [{ text: "Cancel" }, {
                        text: "CREATE", class: "vnccs-clothes-btn-primary", action: async (ol, btn) => {
                            const n = ol.querySelector("input").value.trim();
                            if (n) {
                                await checkedJSON("/vnccs/save_costume", {
                                    method: "POST", body: JSON.stringify({ character: state.character, costume: n, info: {} })
                                });
                                await loadCostumes();
                                state.costume = n;
                                costSel.value = n;
                                await loadCostumeInfo();
                                syncCostumeEditControls();
                                updatePreviewImage();
                                saveState();
                                return false;
                            }
                            return true;
                        }
                    }]);
                };
                actionRow.appendChild(btnNewCostume);

                const btnDelCostume = document.createElement("button");
                btnDelCostume.className = "vnccs-clothes-btn vnccs-clothes-btn-danger";
                btnDelCostume.innerText = "DELETE";
                btnDelCostume.style.fontSize = "10px";
                btnDelCostume.onclick = () => {
                    if (!state.character || !hasSelectedEditableCostume()) {
                        showInfo("Warning", "Select an editable costume first. Base sprite sets cannot be deleted.");
                        return;
                    }
                    if (els.btnGen?.disabled) {
                        showInfo("Warning", "Wait for preview generation to finish before deleting a costume.");
                        return;
                    }
                    const character = state.character;
                    const costume = state.costume;
                    const currentRequest = beginDeleteRequest();
                    const isCurrent = () => currentRequest() && state.character === character && state.costume === costume;
                    showModal("Delete", () => {
                        const d = document.createElement("div");
                        d.innerText = `Delete "${costume}" for ${character}? This removes its settings, generated images, and preview.`;
                        return d;
                    },
                        [{ text: "Cancel" }, {
                            text: "DELETE", class: "vnccs-clothes-btn-danger", action: async () => {
                                if (!isCurrent()) return false;
                                if (els.btnGen?.disabled) {
                                    showInfo("Warning", "Wait for preview generation to finish before deleting a costume.");
                                    return true;
                                }
                                beginClothesWizardRequest();
                                let deletion;
                                const controls = [charSel, costSel, btnNewCostume, btnDelCostume, els.btnGen, els.wizardBtn,
                                    ...["top", "bottom", "head", "face", "shoes"].map(key => els[key])].filter(Boolean);
                                const disabled = controls.map(control => control.disabled);
                                controls.forEach(control => { control.disabled = true; });
                                try {
                                    // Finish any earlier field saves before deleting their costume.
                                    await Promise.allSettled([...pendingCostumeSaves]);
                                    if (!isCurrent()) return false;
                                    deletion = await checkedJSON("/vnccs/delete_costume", {
                                        method: "POST",
                                        body: JSON.stringify({ character, costume })
                                    });
                                } finally {
                                    controls.forEach((control, index) => { control.disabled = disabled[index]; });
                                }
                                if (!isCurrent()) return false;
                                const currentSelection = beginSelectionRequest();
                                beginPreviewRequest();
                                beginCostumesRequest();
                                beginCostumeInfoRequest();
                                spritePreviewNavigator?.invalidate?.();
                                spritePreviewNavigator?.hideNav();
                                els.costSel.innerHTML = "";
                                els.costSel.disabled = true;
                                resetCostumeSelection();
                                if (await loadCostumes() && currentSelection() && currentRequest()) {
                                    saveState();
                                    updatePreviewImage();
                                }
                                if (deletion.warning && currentRequest()) {
                                    showInfo("Costume Deleted", deletion.warning);
                                }
                                return false;
                            }
                        }]);
                };
                actionRow.appendChild(btnDelCostume);
                els.btnDel = btnDelCostume;
                colLeft.appendChild(actionRow);

                // Generate Button
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-clothes-btn vnccs-clothes-btn-primary";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.style.width = "100%"; btnGen.style.marginBottom = "5px";
                btnGen.style.flex = "0 0 auto"; // Prevent vertical stretching
                btnGen.onclick = async () => {
                    if (!state.character) { showInfo("Error", "Select Character"); return; }
                    if (!hasSelectedEditableCostume()) { showCreateCostumeRequired(); return; }
                    if (state.activeTab === "clone" && !state.clone_image) {
                        showInfo("Reference Required", "Upload a clothing reference image before using Clone Clothes.");
                        return;
                    }
                    if (btnGen.disabled) return;

                    setClothesCoreLora();
                    node._randomizeSeedIfNeeded();

                    const controlCenter = getConnectedControlCenterState();
                    if (!controlCenter) {
                        showInfo("Error", "Connect Clothes Designer pipe input to VNCCS Control Center.");
                        return;
                    }

                    // Show loading overlay
                    const loadingOverlay = document.createElement('div');
                    loadingOverlay.className = 'vnccs-clothes-loading-overlay';
                    loadingOverlay.innerHTML = `
                        <div class="vnccs-clothes-spinner"></div>
                        <div class="vnccs-clothes-loading-text">Generating preview<span class="vnccs-clothes-loading-dots"></span></div>
                    `;
                    container.appendChild(loadingOverlay);

                    const character = state.character;
                    const costume = state.costume;
                    const currentRequest = beginPreviewRequest();
                    const isCurrent = () => currentRequest() && state.character === character && state.costume === costume;
                    spritePreviewNavigator?.invalidate?.();
                    btnGen.innerText = "GENERATING..."; btnGen.disabled = true;
                    try {
                        await saveCostumeToBackend();
                        if (!isCurrent()) return;
                        saveState();
                        const r = await api.fetchApi("/vnccs/control_center/clothes_preview", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                ...controlCenter,
                                clothes_state: state,
                            })
                        });
                        if (r.ok) {
                            const d = await r.json();
                            if (!isCurrent()) return;
                            if (d.image) {
                                els.previewImg.src = "data:image/png;base64," + d.image;
                                els.previewImg.style.display = "block";
                                els.placeholder.style.display = "none";
                            }
                        } else {
                            const message = await r.text();
                            if (isCurrent()) showInfo("Error", message || "Failed");
                        }
                    } catch (e) { if (isCurrent()) showInfo("Error", e.toString()); }
                    finally {
                        loadingOverlay.remove();
                        btnGen.innerText = "GENERATE PREVIEW / SAVE"; btnGen.disabled = false;
                    }
                };
                els.btnGen = btnGen;
                colLeft.appendChild(btnGen);

                // Preview
                const frame = document.createElement("div"); frame.className = "vnccs-clothes-preview-container";
                frame.style.marginTop = "5px";
                frame.innerHTML = `<div class="vnccs-clothes-placeholder">
                    <svg class="vnccs-clothes-placeholder-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16 12h16M12 20l4-8h16l4 8v20H12V20z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M20 40V28h8v12" stroke="currentColor" stroke-width="2"/>
                    </svg>
                    No Preview
                </div>`;
                const pImg = document.createElement("img"); pImg.className = "vnccs-clothes-preview-img"; pImg.style.display = "none";
                pImg.onclick = () => window.open(pImg.src, "_blank");
                frame.appendChild(pImg);
                const previewLoading = document.createElement("div");
                previewLoading.className = "vnccs-clothes-preview-loading";
                frame.appendChild(previewLoading);
                els.previewImg = pImg; els.placeholder = frame.querySelector(".vnccs-clothes-placeholder");
                colLeft.appendChild(frame);
                const spriteNav = document.createElement("div");
                spriteNav.className = "cd-sprite-nav";
                const spritePrevBtn = document.createElement("button");
                spritePrevBtn.type = "button";
                spritePrevBtn.className = "cd-sprite-nav-btn";
                spritePrevBtn.title = "Previous sprite";
                spritePrevBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>';
                const spriteCount = document.createElement("div");
                spriteCount.className = "cd-sprite-nav-count";
                const spriteNextBtn = document.createElement("button");
                spriteNextBtn.type = "button";
                spriteNextBtn.className = "cd-sprite-nav-btn";
                spriteNextBtn.title = "Next sprite";
                spriteNextBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>';
                spriteNav.append(spritePrevBtn, spriteCount, spriteNextBtn);
                colLeft.appendChild(spriteNav);
                spritePreviewNavigator = createSpritePreviewNavigator({
                    node,
                    isSelectionCurrent: preview => preview.character === state.character
                        && preview.costume === (hasSelectedEditableCostume() ? state.costume : "Naked"),
                    image: pImg,
                    placeholder: els.placeholder,
                    loading: previewLoading,
                    nav: spriteNav,
                    prevButton: spritePrevBtn,
                    nextButton: spriteNextBtn,
                    countLabel: spriteCount,
                    onLoaded: (url, previewState) => {
                        // A generated preview is only a visual fallback. Keep the
                        // source sprite selection that was used to produce it so
                        // the following full workflow has the same widget_data
                        // signature and can reuse the preview cache.
                        if (url.includes("force_cache=true")) return;
                        if (previewState.count > 0) {
                            state.selected_preview_sprite = {
                                character: previewState.character,
                                costume: previewState.costume || "",
                                index: previewState.index,
                                count: previewState.count,
                            };
                        } else {
                            state.selected_preview_sprite = null;
                        }
                        saveState();
                    },
                    onMissing: () => {
                        state.selected_preview_sprite = null;
                        saveState();
                    },
                });

                topRow.appendChild(colLeft);

                // --- COL 2: MIDDLE PANEL (Tabs) ---
                const colMid = document.createElement("div"); colMid.className = "vnccs-clothes-col";
                colMid.style.paddingTop = "0"; // Reset padding for tabs

                const controlsWrap = createGenerationControls();
                controlsWrap.style.paddingTop = "16px";
                colMid.appendChild(controlsWrap);
                colMid.appendChild(createResolutionControl());

                // Tab Header
                const tabBar = document.createElement("div");
                tabBar.className = "cd-tab-bar";

                const createTab = (id, label) => {
                    const t = document.createElement("div");
                    t.className = "cd-tab" + (state.activeTab === id ? " active" : "");
                    t.innerText = label;
                    t.onclick = () => {
                        state.activeTab = id;
                        saveState();
                        refreshTabs();
                    };
                    return t;
                };

                const contentArea = document.createElement("div");
                contentArea.style.flex = "1"; contentArea.style.display = "flex"; contentArea.style.flexDirection = "column";

                const refreshTabs = () => {
                    tabBar.innerHTML = '';
                    tabBar.appendChild(createTab("generate", "GENERATE CLOTHES"));
                    tabBar.appendChild(createTab("clone", "CLONE CLOTHES"));

                    if (state.activeTab === "generate") renderAttributes(contentArea);
                    else renderClonePanel(contentArea);
                };

                refreshTabs(); // Initial Render

                colMid.appendChild(tabBar);
                colMid.appendChild(contentArea);
                topRow.appendChild(colMid);

                node._vnccsRestoreClothesState = () => {
                    try {
                        const restored = JSON.parse(dataWidget.value || "{}");
                        Object.assign(state, restored, {
                            costume_info: { ...defaultState.costume_info, ...restored.costume_info },
                            character_info: { ...defaultState.character_info, ...restored.character_info },
                            gen_settings: { ...defaultState.gen_settings, ...restored.gen_settings },
                        });
                        syncGenerationControls();
                    } catch (error) { console.warn("[VNCCS] Clothes Designer state restore failed", error); }
                };
                registerCleanup(node, () => { delete node._vnccsRestoreClothesState; });

                // Initial Load
                (async () => {
                    const isMounted = createRequestGuard(node)();
                    const currentSelection = beginSelectionRequest();
                    const currentPreview = beginPreviewRequest();
                    const r = await api.fetchApi("/vnccs/context_lists");
                    const d = await r.json();
                    if (!isMounted()) return;
                    setClothesCoreLora();
                    syncGenerationControls();
                    saveState(); // Ensure defaults are persisted immediately

                    // Char List
                    els.charSelect.innerHTML = "";
                    d.characters.forEach(c => els.charSelect.add(new Option(c, c)));
                    if (state.character) els.charSelect.value = state.character;
                    else if (d.characters.length) { state.character = d.characters[0]; els.charSelect.value = state.character; }

                    if (!currentSelection()) return;
                    if (!await loadCharacterInfo() || !currentSelection()) return;
                    if (!await loadCostumes() || !currentSelection()) return;
                    if (currentPreview()) updatePreviewImage();
                    saveState();
                })();
                container.appendChild(topRow);

                // Sync LoRA options from Control Center
                const _onLoraOptions = (e) => {
                    const options = e.detail?.options;
                    if (!Array.isArray(options)) return;
                    setClothesCoreLora();
                    syncGenerationControls();
                };
                window.addEventListener("vnccs-lora-options-updated", _onLoraOptions);
                registerCleanup(node, () => window.removeEventListener("vnccs-lora-options-updated", _onLoraOptions));

                const _onControlCenterModelChanged = () => {
                    setClothesCoreLora();
                    syncGenerationControls();
                };
                window.addEventListener("vnccs-control-center-model-changed", _onControlCenterModelChanged);
                registerCleanup(node, () => window.removeEventListener("vnccs-control-center-model-changed", _onControlCenterModelChanged));

                // Functions
                const beginCostumesRequest = createRequestGuard(node);
                const resetCostumeSelection = () => {
                    state.costume = "";
                    state.costume_info = { ...defaultState.costume_info };
                    state.selected_preview_sprite = null;
                    if (els.costSel) els.costSel.value = "";
                    if (els.btnGen) els.btnGen.disabled = true;
                    if (els.btnDel) els.btnDel.disabled = true;
                    for (const key in state.costume_info) {
                        if (els[key]) els[key].value = state.costume_info[key];
                    }
                    if (els.previewImg) els.previewImg.style.display = "none";
                    if (els.placeholder) els.placeholder.style.display = "block";
                    spritePreviewNavigator?.invalidate?.();
                    spritePreviewNavigator?.hideNav();
                    syncCostumeEditControls();
                    saveState();
                };
                const loadCostumes = async () => {
                    const currentRequest = beginCostumesRequest();
                    const c = state.character;
                    if (!c) return;
                    try {
                        const r = await api.fetchApi(`/vnccs/list_costumes?character=${encodeURIComponent(c)}`);
                        let list = await r.json();
                        if (!currentRequest() || state.character !== c) return false;
                        if (!r.ok || !Array.isArray(list)) throw new Error("Failed to load character costumes.");

                        // Filter base sprite sets from display list.
                        const displayList = list.filter(i => i !== "Naked" && i !== "Original");

                        els.costSel.innerHTML = "";
                        displayList.forEach(i => els.costSel.add(new Option(i, i)));

                        // Logic: If only Naked exists (displayList empty), prevent generation/deletion
                        if (displayList.length === 0) {
                            state.costume = "";
                            if (els.btnGen) els.btnGen.disabled = true;
                            if (els.btnDel) els.btnDel.disabled = true;
                            if (els.costSel) els.costSel.disabled = true;
                        } else {
                            if (els.btnGen) els.btnGen.disabled = true;
                            if (els.btnDel) els.btnDel.disabled = true;
                            if (els.costSel) els.costSel.disabled = false;

                            // Select default if current is Naked or invalid
                            if (state.costume === "Naked" || !displayList.includes(state.costume)) {
                                state.costume = displayList[0];
                            }
                        }

                        if (els.costSel.options.length > 0) {
                            els.costSel.value = state.costume;
                        }

                        if (!await loadCostumeInfo()) return false;
                        syncCostumeEditControls();
                        return true;
                    } catch (error) {
                        if (currentRequest() && !(state.character !== c)) showInfo("Error", error.message || String(error));
                        return false;
                    }
                };

                const beginCostumeInfoRequest = createRequestGuard(node);
                const loadCostumeInfo = async () => {
                    const currentRequest = beginCostumeInfoRequest();
                    const c = state.character;
                    const cos = state.costume;
                    if (els.btnGen) els.btnGen.disabled = true;
                    if (els.btnDel) els.btnDel.disabled = true;
                    ["top", "bottom", "head", "face", "shoes"].forEach(key => {
                        if (els[key]) els[key].disabled = true;
                    });
                    if (els.wizardBtn) els.wizardBtn.disabled = true;
                    try {
                        const r = await api.fetchApi(`/vnccs/get_costume?character=${encodeURIComponent(c)}&costume=${encodeURIComponent(cos)}`);
                        const info = await r.json();
                        if (!currentRequest() || state.character !== c || state.costume !== cos) return false;
                        if (!r.ok) throw new Error("Failed to load costume metadata.");
                        if (!info || typeof info !== "object" || Array.isArray(info)) {
                            throw new Error("Invalid costume metadata.");
                        }

                        state.costume_info = {
                            top: info.top || "",
                            bottom: info.bottom || "",
                            head: info.head || "",
                            face: info.face || "",
                            shoes: info.shoes || ""
                        };

                        for (const k in state.costume_info) {
                            if (els[k]) {
                                els[k].value = state.costume_info[k];
                                if (els[k].autoResize) els[k].autoResize();
                            }
                        }
                        const canEdit = hasSelectedEditableCostume();
                        if (els.btnGen) els.btnGen.disabled = !canEdit;
                        if (els.btnDel) els.btnDel.disabled = !canEdit;
                        syncCostumeEditControls();
                        saveState();
                        return true;
                    } catch (error) {
                        if (currentRequest() && state.character === c && state.costume === cos) {
                            resetCostumeSelection();
                            showInfo("Error", error.message || String(error));
                        }
                        return false;
                    }
                };

                const updatePreviewImage = async (forceCache = false) => {
                    const currentRequest = beginPreviewRequest();
                    const character = state.character;
                    const costume = state.costume;
                    const isCurrent = () => currentRequest() && state.character === character && state.costume === costume;
                    spritePreviewNavigator?.invalidate?.();
                    if (!state.character) return;
                    const ts = Date.now();
                    const previewCostume = hasSelectedEditableCostume() ? state.costume : "Naked";
                    let url = `/vnccs/get_preview?character=${encodeURIComponent(character)}&costume=${encodeURIComponent(previewCostume)}&ts=${ts}`;
                    if (forceCache) url += "&force_cache=true";
                    if (!forceCache) {
                        state.selected_preview_sprite = null;
                        saveState();
                    }

                    // Check validity first to show message
                    try {
                        const r = await api.fetchApi(url);
                        if (!isCurrent()) return;
                        if (!r.ok) {
                            els.previewImg.style.display = "none";
                            els.placeholder.style.display = "block";
                            spritePreviewNavigator?.hideNav();
                            state.selected_preview_sprite = null;
                            saveState();
                            return;
                        }
                    } catch (e) {
                        if (!isCurrent()) return;
                        console.warn("[VNCCS] ClothesDesigner: Error in preview update", e);
                    }

                    if (forceCache) {
                        spritePreviewNavigator?.showFallback(mediaURL(url), { character, costume: previewCostume });
                    } else {
                        await spritePreviewNavigator?.load(character, {
                            costume: previewCostume,
                            fallbackUrl: mediaURL(url),
                        });
                    }
                };

                const onPreviewUpdated = (event) => {
                    const targetId = String(event.detail?.node_id);
                    const myId = String(node.id);
                    if (targetId === myId) {
                        console.log("VNCCS Preview Update Received for Node:", myId);
                        updatePreviewImage(true);
                    }
                };
                api.addEventListener("vnccs.preview.updated", onPreviewUpdated);
                registerCleanup(node, () => api.removeEventListener("vnccs.preview.updated", onPreviewUpdated));
                watchConnection(node, () => { if (!btnGen.disabled) refreshPreviewImage(els.previewImg); }, registerCleanup);

                enableMiddleMouseCanvasPan(container);
                attachHelpTooltips(container);
                node.addDOMWidget("clothes_designer_ui", "ui", container, {
                    serialize: false,
                    hideOnZoom: false
                });
                syncDOMWidgetWidthSoon(node, "clothes_designer_ui");

            };

            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                onResize?.apply(this, arguments);
                syncDOMWidgetWidth(this, "clothes_designer_ui");
                requestAnimationFrame(() => syncDOMWidgetWidth(this, "clothes_designer_ui"));
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                this._vnccsRestoreClothesState?.();
                syncDOMWidgetWidth(this, "clothes_designer_ui");
                setTimeout(() => syncDOMWidgetWidth(this, "clothes_designer_ui"), 100);
            };
        }
    }
});
