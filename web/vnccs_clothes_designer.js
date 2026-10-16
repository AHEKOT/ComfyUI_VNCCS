import { app } from "../../scripts/app.js";
import { createSakuraInteractions } from "./sakura_interactions.mjs";
import { vnccsApi as api, mediaURL, checkedJSON, storage, refreshPreviewImage, watchConnection, workflowScope } from "./vnccs_transport.js";
import { createQwenVLModelLoader, SAKURA_THEME_CSS, SAKURA_INTERACTION_CSS, registerCleanup, injectStyles, showModal as showCommonModal, showMessage, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, createSpritePreviewNavigator, createRequestGuard, animateWizardRows, resolutionScaleMegapixels, resolutionScaleValue, resolutionScaleText, RESOLUTION_SCALE_MIN_MP, RESOLUTION_SCALE_MAX_MP, RESOLUTION_SCALE_STEP_MP, normalizeUploadFile, normalizeAgeValue } from "./vnccs_common.js";

const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
${SAKURA_INTERACTION_CSS}

.vnccs-clothes-container {
${SAKURA_THEME_CSS}
    --warning: #ffaa00;
}

.vnccs-clothes-container {
    position: relative;
    display: flex; flex-direction: column;
    background: var(--bg-primary); color: var(--text-primary);
    font-family: var(--font); font-size: 13px;
    width: 100%; height: 100%; overflow: hidden; box-sizing: border-box;
    padding: 12px; gap: 12px; pointer-events: none; zoom: 0.67;
}
.vnccs-clothes-top-row {
    display: grid; grid-template-columns: 400px minmax(0, 1fr); gap: 12px;
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
.vnccs-clothes-design-panel { overflow: visible; }
.vnccs-clothes-design-content {
    position: relative;
    z-index: 1;
    display: flex;
    flex: 1;
    min-height: 0;
    flex-direction: column;
    gap: 10px;
    overflow: hidden auto;
    padding-inline: 18px;
    box-sizing: border-box;
    opacity: 1;
    transform: none;
    filter: none;
    visibility: visible;
    transition: opacity 280ms ease, transform 360ms cubic-bezier(0.22,1,0.36,1), filter 280ms ease, visibility 0s;
}
.vnccs-clothes-design-panel.is-costume-empty .vnccs-clothes-design-content {
    opacity: 0;
    transform: translateY(12px) scale(0.985);
    filter: blur(6px);
    visibility: hidden;
    pointer-events: none;
    transition: opacity 200ms ease, transform 260ms ease, filter 200ms ease, visibility 0s 260ms;
}
.vnccs-clothes-costume-empty {
    --vnccs-costume-empty-y: 210px;
    position: absolute;
    z-index: 3;
    top: var(--vnccs-costume-empty-y);
    left: 34px;
    right: 28px;
    padding: 18px 20px;
    border: 1px solid rgba(255,143,163,0.38);
    border-radius: 16px;
    background:
        radial-gradient(circle at 12% 18%, rgba(255,143,163,0.18), transparent 42%),
        linear-gradient(135deg, rgba(38,25,43,0.96), rgba(20,18,31,0.96));
    box-shadow: 0 16px 40px rgba(0,0,0,0.34), inset 0 1px rgba(255,255,255,0.05);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transform: translate(14px, -50%) scale(0.97);
    transform-origin: left center;
    transition: opacity 220ms ease, transform 420ms cubic-bezier(0.22,1,0.36,1), visibility 0s 420ms;
}
.vnccs-clothes-design-panel.is-costume-empty .vnccs-clothes-costume-empty {
    opacity: 1;
    visibility: visible;
    transform: translate(0, -50%) scale(1);
    transition-delay: 100ms;
}
.vnccs-clothes-costume-empty-icon {
    display: inline-flex;
    width: 32px;
    height: 32px;
    margin-bottom: 10px;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255,143,163,0.34);
    border-radius: 10px;
    color: var(--accent);
    background: rgba(255,143,163,0.1);
    box-shadow: 0 0 20px rgba(255,143,163,0.12);
}
.vnccs-clothes-costume-empty-icon svg { width: 18px; height: 18px; }
.vnccs-clothes-costume-empty-title {
    color: var(--text-primary);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.vnccs-clothes-costume-empty-text {
    max-width: 430px;
    margin-top: 7px;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.55;
}
.vnccs-clothes-costume-empty-arrow {
    position: absolute;
    top: 50%;
    left: -58px;
    width: 62px;
    height: 42px;
    overflow: visible;
    color: var(--accent);
    filter: drop-shadow(0 0 8px rgba(255,143,163,0.35));
    transform: translateY(-50%);
    animation: cd-costume-empty-arrow 2200ms ease-in-out infinite;
}
.vnccs-clothes-costume-empty-arrow path {
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
}
@keyframes cd-costume-empty-arrow {
    0%, 100% { transform: translate(0, -50%); opacity: 0.7; }
    50% { transform: translate(-4px, -50%); opacity: 1; }
}

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
    width: 100%; box-sizing: border-box; transition: all var(--transition);
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
.vnccs-clothes-prompt-card {
    display: flex;
    align-items: center;
    gap: 7px;
    min-height: 42px;
    box-sizing: border-box;
    margin: 0 0 12px;
    padding: 9px 12px;
    border: 1px solid rgba(0,214,143,0.25);
    border-radius: var(--radius-md);
    background: rgba(0,214,143,0.05);
    cursor: pointer;
    transition: all var(--transition);
}
.vnccs-clothes-prompt-card[hidden] { display: none; }
.vnccs-clothes-prompt-card:hover:not(.is-selected):not(.is-disabled) {
    border-color: rgba(0,214,143,0.42);
    background: rgba(0,214,143,0.08);
}
.vnccs-clothes-prompt-card.is-selected {
    border-color: var(--accent);
    background: rgba(255,143,163,0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset;
}
.vnccs-clothes-prompt-card.is-disabled { opacity: 0.55; cursor: not-allowed; }
.vnccs-clothes-prompt-badge {
    width: 12px;
    height: 12px;
    flex-shrink: 0;
    border-radius: 50%;
    background: var(--success);
}
.vnccs-clothes-prompt-name {
    flex: 1;
    min-width: 0;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 700;
}
.vnccs-clothes-prompt-status {
    flex-shrink: 0;
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.vnccs-clothes-prompt-status.ok { color: var(--success); }
.vnccs-clothes-prompt-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-clothes-prompt-toggle input {
    position: absolute;
    width: 0;
    height: 0;
    opacity: 0;
}
.vnccs-clothes-prompt-toggle-track {
    position: absolute;
    inset: 0;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    transition: all 0.25s ease;
}
.vnccs-clothes-prompt-toggle-thumb {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.25s ease;
}
.vnccs-clothes-prompt-toggle input:checked ~ .vnccs-clothes-prompt-toggle-track {
    border-color: var(--accent);
    background: rgba(255,143,163,0.2);
    box-shadow: 0 0 8px var(--accent-glow);
}
.vnccs-clothes-prompt-toggle input:checked ~ .vnccs-clothes-prompt-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
}
.vnccs-clothes-prompt-toggle input:focus-visible ~ .vnccs-clothes-prompt-toggle-track {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

/* Preview */
.vnccs-clothes-preview-container {
    --vnccs-preview-accent: #ff8fa3;
    --vnccs-preview-accent-rgb: 255, 143, 163;
    --vnccs-preview-highlight-rgb: 255, 238, 246;
    flex: 1;
    background: radial-gradient(circle, rgba(var(--vnccs-preview-accent-rgb), 0.04) 1px, transparent 1px), rgba(10,10,15,0.7);
    background-size: 20px 20px, 100% 100%;
    border: 1px solid var(--border); border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; position: relative; min-height: 0;
}
.vnccs-clothes-preview-container.is-male-character {
    --vnccs-preview-accent: #63c7ff;
    --vnccs-preview-accent-rgb: 99, 199, 255;
    --vnccs-preview-highlight-rgb: 232, 248, 255;
}
.vnccs-clothes-preview-container.is-male-character[aria-busy="true"] {
    border-color: rgba(var(--vnccs-preview-accent-rgb), 0.62);
    box-shadow: 0 0 18px rgba(var(--vnccs-preview-accent-rgb), 0.14);
}
.vnccs-clothes-preview-img {
    position: absolute;
    top: 0;
    left: 50%;
    width: auto;
    max-width: none;
    height: 100%;
    object-fit: contain;
    transform: translateX(-50%);
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
    border: 2px solid rgba(var(--vnccs-preview-accent-rgb), 0.24);
    border-top-color: var(--vnccs-preview-accent);
    border-radius: 50%;
    box-shadow: 0 0 18px rgba(var(--vnccs-preview-accent-rgb), 0.25);
    animation: cd-spin 0.75s linear infinite;
}
.vnccs-clothes-generation-layer {
    position: absolute;
    inset: 0;
    z-index: 5;
    overflow: hidden;
    pointer-events: none;
    background: radial-gradient(circle, rgba(var(--vnccs-preview-accent-rgb), 0.04) 1px, transparent 1px), #0f0f16;
    background-size: 20px 20px, 100% 100%;
}
.vnccs-clothes-generation-layer[hidden] { display: none; }
.vnccs-clothes-generation-img {
    position: absolute;
    top: 0;
    left: 50%;
    width: auto;
    max-width: none;
    height: 100%;
    object-fit: contain;
    transform: translateX(-50%);
}
.vnccs-clothes-generation-base { animation: cd-preview-wait 1800ms ease-in-out infinite; }
.vnccs-clothes-generation-new {
    z-index: 1;
    clip-path: inset(0 0 100% 0);
}
.vnccs-clothes-generation-scan {
    position: absolute;
    z-index: 2;
    top: 0;
    left: 0;
    right: 0;
    height: 100%;
    transform: translateY(-100%);
    opacity: 0.82;
    mix-blend-mode: screen;
    will-change: transform, opacity;
    animation: cd-preview-process-scan 2200ms cubic-bezier(0.4,0,0.2,1) infinite;
}
.vnccs-clothes-generation-scan::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: clamp(18px, 6%, 42px);
    background: linear-gradient(180deg, transparent, rgba(var(--vnccs-preview-accent-rgb), 0.24) 30%, rgba(var(--vnccs-preview-highlight-rgb), 0.98) 50%, rgba(var(--vnccs-preview-accent-rgb), 0.34) 68%, transparent);
    box-shadow: 0 0 24px rgba(var(--vnccs-preview-accent-rgb), 0.3);
}
.vnccs-clothes-generation-layer.is-revealing .vnccs-clothes-generation-base { animation: none; }
.vnccs-clothes-generation-layer.is-revealing .vnccs-clothes-generation-scan { animation: none; }
@keyframes cd-preview-wait { 0%, 100% { opacity: 0.78; } 50% { opacity: 1; } }
@keyframes cd-preview-process-scan { from { transform: translateY(-100%); opacity: 0.5; } to { transform: translateY(0); opacity: 0.9; } }
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
@keyframes cd-spin { to { transform: rotate(360deg); } }
.vnccs-clothes-placeholder {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--text-muted); font-size: 11px; letter-spacing: 0.05em;
}
.vnccs-clothes-placeholder-icon { width: 48px; height: 48px; opacity: 0.25; }

/* Tab bar */
.cd-tab-bar {
    display: flex; border-bottom: 1px solid var(--border);
    margin: 0 0 12px; background: transparent;
}
.vnccs-clothes-container .cd-tab-bar.vnccs-common-ui-segmented { border-radius: var(--radius-lg); }
.vnccs-clothes-container .cd-tab-bar.vnccs-common-ui-segmented::before { border-radius: var(--radius-md); }
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

@keyframes cd-spin { to { transform: rotate(360deg); } }

.vnccs-clothes-container :is(button, [role="button"]):focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.vnccs-clothes-container :is(input, select, .vnccs-common-ui-select-trigger) { box-sizing: border-box; }
@media (prefers-reduced-motion: reduce) {
    .vnccs-clothes-container *, .vnccs-clothes-container *::before, .vnccs-clothes-container *::after {
        animation: none !important; transition: none !important;
    }
}
.vnccs-clothes-input, .vnccs-clothes-select { height: 34px; min-height: 34px; }
.cd-tab { font-family: var(--font); min-height: 34px; }
`;

app.registerExtension({
    name: "VNCCS.ClothesDesigner",

    async setup() {
        if (app._vnccsClothesQueuePatched) return;
        app._vnccsClothesQueuePatched = true;
        const queuePrompt = app.queuePrompt;
        const origQueuePrompt = (...args) => queuePrompt.apply(app, args);
        app.queuePrompt = async function(...args) {
            const nodes = app.graph?._nodes?.filter(n => n.type === "ClothesDesigner") || [];
            for (const node of nodes) {
                if ([2, 4].includes(node.mode)) continue;
                if (node._vnccsStateRestoreError) throw new Error(node._vnccsStateRestoreError);
                if (node._vnccsClothesSelectionPending) throw new Error("Clothes Designer metadata is loading. Wait before running the workflow.");
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
                    costume_revision: null,
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
                        lora_strength: 1.0,
                        qi2_prompt_enhancement: true
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
                    if (!saved || typeof saved !== "object" || Array.isArray(saved)
                        || ["costume_info", "character_info", "gen_settings"].some(key => saved[key] !== undefined
                            && (!saved[key] || typeof saved[key] !== "object" || Array.isArray(saved[key])))) {
                        throw new Error("Saved clothes state must contain objects.");
                    }
                } catch (e) {
                    node._vnccsStateRestoreError = "Cannot restore saved clothes state. Original widget data is preserved. Restore a valid workflow before editing or running.";
                    saved = {};
                    console.warn("[VNCCS] ClothesDesigner: Error loading state", e);
                }

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
                node._vnccsCostumeDraft = saved.costume_info ? { character: state.character, costume: state.costume } : null;
                let spritePreviewNavigator = null;
                const beginPreviewRequest = createRequestGuard(node);
                const beginSelectionRequest = createRequestGuard(node);
                const beginDeleteRequest = createRequestGuard(node);
                const beginClothesWizardRequest = createRequestGuard(node);
                const beginClothesUploadRequest = createRequestGuard(node);

                const saveState = () => {
                    if (node._vnccsStateRestoreError) return;
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
                    if (dataWidget && !node._vnccsStateRestoreError) dataWidget.value = JSON.stringify(state);
                    const index = node.widgets?.findIndex(widget => widget.name === "widget_data") ?? -1;
                    if (index >= 0 && Array.isArray(o?.widgets_values)) {
                        o.widgets_values[index] = node.widgets[index].value;
                    }
                };

                const pendingCostumeSaves = new Set();
                const saveCostumeToBackend = async () => {
                    if (node._vnccsStateRestoreError) throw new Error(node._vnccsStateRestoreError);
                    if (node._vnccsClothesSelectionPending) throw new Error("Clothes Designer metadata is loading. Wait before saving the costume.");
                    if (!state.character || !state.costume) return;
                    const character = state.character, costume = state.costume;
                    const info = JSON.parse(JSON.stringify(state.costume_info));
                    let revision = state.costume_revision;
                    const request = Promise.allSettled([...pendingCostumeSaves]).then(results => {
                        for (const result of results) {
                            if (result.status === "fulfilled" && result.value?.character === character && result.value.costume === costume) revision = result.value.revision;
                        }
                        return checkedJSON("/vnccs/save_costume", {
                            method: "POST", body: JSON.stringify({ character, costume, info, revision })
                        }).then(result => ({ ...result, character, costume }));
                    });
                    pendingCostumeSaves.add(request);
                    try {
                        const result = await request;
                        if (state.character === character && state.costume === costume && state.costume_revision === revision) {
                            state.costume_revision = result.revision;
                            saveState();
                        }
                        return result;
                    }
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
                    return !!costume && !["naked", "original"].includes(costume.toLowerCase());
                };

                const showCreateCostumeRequired = () => {
                    showInfo("Costume Required", "Create a new costume first, then select it before generating a preview.");
                };

                const syncCostumeEditControls = () => {
                    const busy = !!node._vnccsClothesSelectionPending || !!node._vnccsClothesPreviewRunning;
                    const canEdit = !busy && hasSelectedEditableCostume();
                    const needsCostume = !!state.character && !hasSelectedEditableCostume();
                    if (els.charSelect) els.charSelect.disabled = busy;
                    if (els.btnNew) els.btnNew.disabled = busy || !state.character;
                    if (els.btnGen) els.btnGen.disabled = !canEdit;
                    if (els.btnDel) els.btnDel.disabled = !canEdit;
                    if (els.costSel) els.costSel.disabled = busy || !els.costSel.options.length;
                    if (els.wizardBtn) els.wizardBtn.disabled = !canEdit;
                    els.promptEnhancement?.setDisabled?.(busy);
                    ["top", "bottom", "head", "face", "shoes"].forEach(k => {
                        if (els[k]) els[k].disabled = !canEdit;
                    });
                    els.designPanel?.classList.toggle("is-costume-empty", needsCostume);
                    if (els.designContent) {
                        els.designContent.inert = needsCostume;
                        els.designContent.setAttribute("aria-hidden", String(needsCostume));
                    }
                    els.costumeEmptyState?.setAttribute("aria-hidden", String(!needsCostume));
                    els.alignCostumeEmptyState?.();
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

                const ensureQwenVLReady = createQwenVLModelLoader(node, showModal, () => {
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
                    return { overlay, statusEl, barEl, pctEl, completedMessage: "Download Complete!" };
                }, false);

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
                    if (node._vnccsStateRestoreError || node._vnccsClothesSelectionPending) return;
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
                                let pendingWizardApply = null;
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
                                        if (!isCurrent()) return false;
                                        showWizardModelError(err);
                                        return false;
                                    }
                                    const data = await r.json();
                                    if (!isCurrent()) return false;
                                    const rowKeys = ["top", "bottom", "shoes", "head", "face"];
                                    const sex = String(state.character_info?.sex || state.character_info?.gender || "").trim().toLowerCase();
                                    rowKeys.forEach((key) => { state.costume_info[key] = data[key] || ""; });
                                    node._vnccsCostumeDraft = { character: state.character, costume: state.costume };
                                    saveState();
                                    pendingWizardApply = () => {
                                        rowKeys.forEach((key) => {
                                            if (!els[key]) return;
                                            els[key].value = state.costume_info[key];
                                            if (els[key].autoResize) els[key].autoResize();
                                        });
                                    };
                                    await saveCostumeToBackend();
                                    animateWizardRows(rowKeys.map(key => els._wizardRows?.[key]), pendingWizardApply,
                                        { duration: 240, gap: 30, delay: 90, accent: sex === "male" ? "#63c7ff" : "#ff8fa3" });
                                    pendingWizardApply = null;
                                    return false;
                                } catch (e) {
                                    if (!isCurrent()) return false;
                                    pendingWizardApply?.();
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
                    return { repo_id, node_state, selected_type, control_center_id: String(upstream.id), control_center_scope: workflowScope(upstream) };
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
                const isPoseStudioMounted = createRequestGuard(node)();
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
                    if (!isPoseStudioMounted() || node._vnccsStateRestoreError) return;
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
                const loadCharacterInfo = async (character = state.character) => {
                    const currentRequest = beginCharacterInfoRequest();
                    const originalCharacter = state.character;
                    if (!character) return false;
                    try {
                        const r = await api.fetchApi(`/vnccs/character_info?character=${encodeURIComponent(character)}`);
                        if (!r.ok) throw new Error(`Character metadata request failed (${r.status})`);
                        const info = await r.json();
                        if (!currentRequest() || state.character !== originalCharacter) return false;
                        if (!info || typeof info !== "object" || Array.isArray(info) || (info.name && info.name !== character)) {
                            throw new Error(`Invalid character metadata for '${character}'`);
                        }
                        const changed = character !== originalCharacter;
                        state.character_info = {
                            ...(changed ? defaultState.character_info : state.character_info),
                            ...info,
                            name: character,
                            age: normalizeAgeValue(info.age ?? (changed ? defaultState.character_info.age : state.character_info.age)),
                            sex: String(info.sex || info.gender || (changed ? defaultState.character_info.sex : state.character_info.sex) || "female").toLowerCase()
                        };
                        els.previewFrame?.classList.toggle("is-male-character", state.character_info.sex === "male");
                        state.character = character;
                        if (changed) resetCostumeSelection();
                        saveState();
                        applyPoseStudioValues({ force: true });
                        return true;
                    } catch (e) {
                        if (currentRequest() && state.character === originalCharacter) {
                            if (els.charSelect) els.charSelect.value = state.character;
                            showInfo("Character Load Failed", e.message || String(e));
                        }
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
                    seed_mode: "Toggles fixed seed versus a fresh random seed for each preview.",
                    qi2_prompt_enhancement: "Rewrites the Generate Clothes prompt with Qwen text generation before QI2 encoding. Disable to send the structured clothes prompt directly."
                };
                const helpFor = (key, fallback = "") => FIELD_HELP[key] || fallback;

                const createField = (key, placeholder, multiline = true) => {
                    const wrap = document.createElement("div"); wrap.className = "vnccs-clothes-field";
                    (wrap.dataset ||= {}).vnccsWizardField = key;
                    (els._wizardRows ||= Object.create(null))[key] = wrap;
                    setHelpText(wrap, helpFor(key));
                    const l = document.createElement("div"); l.className = "vnccs-clothes-label";
                    l.innerText = key.toUpperCase();
                    wrap.appendChild(l);

                    const inp = document.createElement(multiline ? "textarea" : "input");
                    inp.className = multiline ? "vnccs-clothes-textarea" : "vnccs-clothes-input";
                    inp.dataset.focusKey = `costume-${key}`;
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
                        if (node._vnccsClothesSelectionPending) return;
                        state.costume_info[key] = e.target.value;
                        node._vnccsCostumeDraft = { character: state.character, costume: state.costume };
                        saveState();
                    };
                    inp.onchange = (e) => {
                        inp.oninput(e);
                        saveCostumeToBackend().catch(error => showInfo("Save Failed", error.message));
                    };
                    els[key] = inp;
                    wrap.appendChild(inp);
                    wrap._vnccsWizardRevealTarget = inp;
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
                        els._ui?.syncSelection(segmented);
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

                const syncPromptEnhancementControl = () => {
                    if (!els.promptEnhancementRow || !els.promptEnhancement) return;
                    els.promptEnhancementRow.hidden = !(
                        getConnectedModelKind().trim().toLowerCase() === "qi2"
                        && state.activeTab === "generate"
                    );
                    els.promptEnhancement.setValue(state.gen_settings.qi2_prompt_enhancement !== false);
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
                    syncPromptEnhancementControl();
                    if (backgroundChanged) saveState();
                    els._ui?.syncSelects();
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

                const createPromptEnhancementControl = () => {
                    const row = document.createElement("div");
                    row.className = "vnccs-clothes-prompt-card";
                    setHelpText(row, helpFor("qi2_prompt_enhancement"));
                    const badge = document.createElement("span");
                    badge.className = "vnccs-clothes-prompt-badge";
                    const name = document.createElement("div");
                    name.className = "vnccs-clothes-prompt-name";
                    name.textContent = "Prompt Enhancement";
                    const status = document.createElement("div");
                    const toggle = document.createElement("label");
                    toggle.className = "vnccs-clothes-prompt-toggle";
                    toggle.onclick = event => event.stopPropagation();
                    const input = document.createElement("input");
                    input.type = "checkbox";
                    input.setAttribute("aria-label", "Prompt Enhancement");
                    input.setValue = enabled => {
                        input.checked = enabled;
                        row.classList.toggle("is-selected", enabled);
                        status.className = "vnccs-clothes-prompt-status" + (enabled ? " ok" : "");
                        status.textContent = enabled ? "Enabled" : "Disabled";
                    };
                    input.setDisabled = disabled => {
                        input.disabled = disabled;
                        row.classList.toggle("is-disabled", disabled);
                    };
                    input.onchange = event => {
                        state.gen_settings.qi2_prompt_enhancement = event.target.checked;
                        input.setValue(event.target.checked);
                        saveState();
                    };
                    input.onclick = event => event.stopPropagation();
                    row.onclick = () => {
                        if (input.disabled) return;
                        input.checked = !input.checked;
                        input.dispatchEvent(new Event("change"));
                    };
                    const track = document.createElement("div");
                    track.className = "vnccs-clothes-prompt-toggle-track";
                    const thumb = document.createElement("div");
                    thumb.className = "vnccs-clothes-prompt-toggle-thumb";
                    toggle.append(input, track, thumb);
                    row.append(badge, name, status, toggle);
                    els.promptEnhancementRow = row;
                    els.promptEnhancement = input;
                    syncPromptEnhancementControl();
                    return row;
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
                        const value = e.target.valueAsNumber;
                        if (!e.target.checkValidity() || !Number.isFinite(value)) {
                            e.target.value = state.gen_settings.seed;
                            return;
                        }
                        state.gen_settings.seed = value;
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
                    container.appendChild(promptEnhancementControl);
                    container.appendChild(createField("top", "e.g. White t-shirt"));
                    container.appendChild(createField("bottom", "e.g. Blue jeans"));
                    container.appendChild(createField("shoes", "e.g. Sneakers"));
                    container.appendChild(createField("head", "e.g. Hat"));
                    container.appendChild(createField("face", "Face features (e.g. glasses)"));
                };

                // Helper to render Clone Panel (New Tab)
                const renderClonePanel = (container) => {
                    const view = els._ui.captureView();
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
                    btn.disabled = !!node._vnccsClothesSelectionPending;
                    if (state.clone_image) { btn.style.opacity = "0.8"; btn.style.fontSize = "10px"; btn.style.padding = "4px 8px"; btn.style.position = "absolute"; btn.style.bottom = "10px"; }

                    overlay.appendChild(btn);

                    const fileInp = document.createElement("input");
                    fileInp.type = "file"; fileInp.accept = "image/*"; fileInp.style.display = "none";
                    fileInp.onchange = async (e) => {
                        if (node._vnccsStateRestoreError || node._vnccsClothesSelectionPending) return;
                        if (e.target.files.length) {
                            const currentRequest = beginClothesUploadRequest();
                            const character = state.character, costume = state.costume, previousImage = state.clone_image;
                            const isCurrent = () => currentRequest() && !node._vnccsStateRestoreError && !node._vnccsClothesSelectionPending
                                && state.character === character && state.costume === costume
                                && state.activeTab === "clone" && state.clone_image === previousImage;
                            try {
                                const file = normalizeUploadFile(e.target.files[0], "clone_reference");
                                btn.innerText = "UPLOADING...";
                                btn.disabled = true;
                                const body = new FormData();
                                body.append("image", file, file.name);
                                body.append("type", "input");
                                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                                if (!resp.ok) {
                                    throw new Error(await resp.text() || `Upload failed (${resp.status})`);
                                }
                                const json = await resp.json();
                                if (!isCurrent()) return;
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
                                if (isCurrent()) showInfo("Upload Failed", err.toString());
                            } finally {
                                if (currentRequest()) {
                                    btn.disabled = false;
                                    btn.innerText = state.clone_image ? "REPLACE IMAGE" : "+ UPLOAD IMAGE";
                                    fileInp.value = "";
                                }
                            }
                        }
                    };

                    overlay.onclick = (e) => {
                        e.stopPropagation();
                        fileInp.click();
                    };

                    pContainer.appendChild(overlay);
                    container.appendChild(pContainer);
                    els._ui.syncSelects();
                    els._ui.restoreView(view);
                };


                // --- MAIN LAYOUT ---
                const container = document.createElement("div"); container.className = "vnccs-clothes-container";
                container.classList.add("vnccs-common-ui-host");
                els._ui = createSakuraInteractions(container, node);
                registerCleanup(node, () => els._ui.dispose());
                const topRow = document.createElement("div"); topRow.className = "vnccs-clothes-top-row";

                // --- COL 1: DESIGN STUDIO ---
                const colLeft = document.createElement("div"); colLeft.className = "vnccs-clothes-col";
                colLeft.dataset.sakuraScroll = "character";
                colLeft.innerHTML = '<div class="vnccs-clothes-section-title">Design Studio</div>';

                // Character Select
                const charRow = document.createElement("div"); charRow.className = "vnccs-clothes-field";
                setHelpText(charRow, helpFor("character"));
                charRow.innerHTML = '<div class="vnccs-clothes-label">CHARACTER</div>';
                const charSel = document.createElement("select"); charSel.className = "vnccs-clothes-select";
                charSel.onchange = async (e) => {
                    els.cancelPreviewGeneration?.(true);
                    beginClothesUploadRequest();
                    const character = e.target.value;
                    els.charSelect.value = character;
                    beginClothesWizardRequest();
                    const currentSelection = beginSelectionRequest();
                    const currentPreview = beginPreviewRequest();
                    spritePreviewNavigator?.invalidate?.();
                    node._vnccsClothesSelectionPending = true;
                    syncCostumeEditControls();
                    try {
                        if (!await loadCharacterInfo(character) || !currentSelection()) return;
                        if (!await loadCostumes() || !currentSelection()) return;
                        if (currentPreview()) updatePreviewImage();
                        saveState();
                    } finally {
                        if (currentSelection()) {
                            node._vnccsClothesSelectionPending = false;
                            els.charSelect.value = state.character;
                            syncCostumeEditControls();
                        }
                    }
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
                    els.cancelPreviewGeneration?.(true);
                    beginClothesUploadRequest();
                    const costume = e.target.value;
                    beginClothesWizardRequest();
                    const currentSelection = beginSelectionRequest();
                    const currentPreview = beginPreviewRequest();
                    spritePreviewNavigator?.invalidate?.();
                    node._vnccsClothesSelectionPending = true;
                    syncCostumeEditControls();
                    try {
                        if (!await loadCostumeInfo(false, costume) || !currentSelection()) return false;
                        if (currentPreview()) updatePreviewImage();
                        saveState();
                    } finally {
                        if (currentSelection()) {
                            node._vnccsClothesSelectionPending = false;
                            els.costSel.value = state.costume;
                            syncCostumeEditControls();
                        }
                    }
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
                    if (node._vnccsStateRestoreError || node._vnccsClothesSelectionPending) return;
                    const character = state.character;
                    const currentRequest = beginClothesUploadRequest();
                    showModal("New Costume Name", () => {
                        const inp = document.createElement("input"); inp.className = "vnccs-clothes-input";
                        return inp;
                    }, [{ text: "Cancel" }, {
                        text: "CREATE", class: "vnccs-clothes-btn-primary", action: async (ol, btn) => {
                            if (node._vnccsStateRestoreError || node._vnccsClothesSelectionPending) return true;
                            const isCurrent = () => currentRequest() && ol.isConnected
                                && !node._vnccsStateRestoreError && !node._vnccsClothesSelectionPending
                                && state.character === character;
                            if (!isCurrent()) return false;
                            const n = ol.querySelector("input").value.trim();
                            if (n) {
                                try {
                                    await checkedJSON("/vnccs/save_costume", {
                                        method: "POST", body: JSON.stringify({ character, costume: n, info: {}, create_only: true })
                                    });
                                    if (!isCurrent() || !await loadCostumes() || !isCurrent()) return false;
                                    state.costume = n;
                                    costSel.value = n;
                                    if (!await loadCostumeInfo() || !isCurrent()) return false;
                                    syncCostumeEditControls();
                                    updatePreviewImage();
                                    saveState();
                                    return false;
                                } catch (error) {
                                    if (!isCurrent()) return false;
                                    throw error;
                                }
                            }
                            return true;
                        }
                    }]);
                };
                els.btnNew = btnNewCostume;
                actionRow.appendChild(btnNewCostume);

                const btnDelCostume = document.createElement("button");
                btnDelCostume.className = "vnccs-clothes-btn vnccs-clothes-btn-danger";
                btnDelCostume.innerText = "DELETE";
                btnDelCostume.style.fontSize = "10px";
                btnDelCostume.onclick = () => {
                    if (node._vnccsStateRestoreError) return;
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
                    const isCurrent = () => currentRequest() && !node._vnccsStateRestoreError
                        && state.character === character && state.costume === costume;
                    showModal("Delete", () => {
                        const d = document.createElement("div");
                        d.innerText = `Delete "${costume}" for ${character}? This removes its settings, generated images, and preview.`;
                        return d;
                    },
                        [{ text: "Cancel" }, {
                            text: "DELETE", class: "vnccs-clothes-btn-danger", action: async () => {
                                if (!isCurrent()) return false;
                                beginClothesUploadRequest();
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
                    if (node._vnccsStateRestoreError) return;
                    if (node._vnccsClothesSelectionPending) return;
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

                    const character = state.character;
                    const costume = state.costume;
                    const currentRequest = beginPreviewRequest();
                    const draft = JSON.stringify({ ...state, costume_revision: null });
                    const isCurrent = () => currentRequest() && !node._vnccsStateRestoreError
                        && state.character === character && state.costume === costume && JSON.stringify({ ...state, costume_revision: null }) === draft;
                    spritePreviewNavigator?.invalidate?.();
                    const generationRevision = startClothesPreviewGeneration();
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
                                await revealClothesPreview("data:image/png;base64," + d.image, state.character, generationRevision);
                            }
                        } else {
                            const message = await r.text();
                            if (isCurrent()) showInfo("Error", message || "Failed");
                        }
                    } catch (e) { if (isCurrent()) showInfo("Error", e.toString()); }
                    finally {
                        stopClothesPreviewGeneration(generationRevision);
                        btnGen.innerText = "GENERATE PREVIEW / SAVE";
                        btnGen.disabled = !!node._vnccsClothesSelectionPending || !hasSelectedEditableCostume();
                    }
                };
                els.btnGen = btnGen;
                colLeft.appendChild(btnGen);

                // Preview
                const frame = document.createElement("div"); frame.className = "vnccs-clothes-preview-container";
                frame.classList.add("vnccs-common-ui-preview-host");
                frame.classList.toggle("is-male-character", String(state.character_info?.sex || state.character_info?.gender || "").trim().toLowerCase() === "male");
                frame.style.marginTop = "5px";
                frame.innerHTML = `<div class="vnccs-clothes-placeholder">
                    <svg class="vnccs-clothes-placeholder-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16 12h16M12 20l4-8h16l4 8v20H12V20z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M20 40V28h8v12" stroke="currentColor" stroke-width="2"/>
                    </svg>
                    No Preview
                </div>`;
                const pImg = document.createElement("img"); pImg.className = "vnccs-clothes-preview-img"; pImg.style.display = "none";
                const previewAura = document.createElement("div");
                previewAura.className = "vnccs-common-ui-preview-aura";
                frame.appendChild(previewAura);
                pImg.onclick = () => window.open(pImg.src, "_blank");
                frame.appendChild(pImg);
                const generationLayer = document.createElement("div");
                generationLayer.className = "vnccs-clothes-generation-layer";
                generationLayer.hidden = true;
                generationLayer.ariaHidden = "true";
                generationLayer.innerHTML = `
                    <img class="vnccs-clothes-generation-img vnccs-clothes-generation-base" alt="">
                    <img class="vnccs-clothes-generation-img vnccs-clothes-generation-new" alt="">
                    <span class="vnccs-clothes-generation-scan"></span>`;
                frame.appendChild(generationLayer);
                const previewLoading = document.createElement("div");
                previewLoading.className = "vnccs-clothes-preview-loading";
                frame.appendChild(previewLoading);
                els.previewFrame = frame;
                els.previewImg = pImg;
                els.placeholder = frame.querySelector(".vnccs-clothes-placeholder");
                els.generationLayer = generationLayer;
                els.generationBase = generationLayer.querySelector(".vnccs-clothes-generation-base");
                els.generationNew = generationLayer.querySelector(".vnccs-clothes-generation-new");
                els.generationScan = generationLayer.querySelector(".vnccs-clothes-generation-scan");
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
                let previewGenerationRevision = 0;
                let workflowPreviewGeneration = null;
                const stopClothesPreviewGeneration = (revision, restorePlaceholder = false) => {
                    if (revision !== undefined && revision !== previewGenerationRevision) return;
                    if (revision === undefined || workflowPreviewGeneration?.revision === revision) workflowPreviewGeneration = null;
                    previewGenerationRevision += 1;
                    for (const animation of els.generationLayer?._animations || []) animation.cancel();
                    if (els.generationLayer) {
                        els.generationLayer._animations = [];
                        els.generationLayer.classList.remove("is-revealing");
                        els.generationLayer.hidden = true;
                    }
                    els.generationBase?.removeAttribute("src");
                    els.generationNew?.removeAttribute("src");
                    els.previewFrame?.removeAttribute("aria-busy");
                    spritePrevBtn.disabled = false;
                    spriteNextBtn.disabled = false;
                    const visiblePreview = els.previewImg?.style?.display !== "none" && els.previewImg?.getAttribute?.("src");
                    if (restorePlaceholder && !visiblePreview && els.placeholder) els.placeholder.style.display = "flex";
                    node._vnccsClothesPreviewRunning = false;
                    syncCostumeEditControls();
                };
                const startClothesPreviewGeneration = () => {
                    const revision = ++previewGenerationRevision;
                    node._vnccsClothesPreviewRunning = true;
                    syncCostumeEditControls();
                    const sex = String(state.character_info?.sex || state.character_info?.gender || "").trim().toLowerCase();
                    els.previewFrame?.classList.toggle("is-male-character", sex === "male");
                    for (const animation of els.generationLayer?._animations || []) animation.cancel();
                    const existing = els.previewImg?.style?.display !== "none"
                        ? els.previewImg?.getAttribute?.("src") || els.previewImg?.src : "";
                    if (existing) {
                        els.generationBase.src = existing;
                        els.generationBase.style.display = "block";
                    } else {
                        els.generationBase.removeAttribute("src");
                        els.generationBase.style.display = "none";
                    }
                    els.generationBase.style.clipPath = "inset(0% 0 0 0)";
                    els.generationNew.removeAttribute("src");
                    els.generationNew.style.clipPath = "inset(0 0 100% 0)";
                    els.generationScan.style.transform = "translateY(-100%)";
                    els.generationLayer.classList.remove("is-revealing");
                    els.generationLayer.hidden = false;
                    els.previewFrame.setAttribute("aria-busy", "true");
                    spritePrevBtn.disabled = true;
                    spriteNextBtn.disabled = true;
                    if (els.placeholder) els.placeholder.style.display = "none";
                    return revision;
                };
                const revealClothesPreview = async (src, name, revision) => {
                    if (revision !== previewGenerationRevision) return false;
                    els.generationNew.src = src;
                    try { await els.generationNew.decode?.(); } catch { /* decoded data remains available to the transition */ }
                    if (revision !== previewGenerationRevision) return false;
                    els.generationLayer.classList.add("is-revealing");
                    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
                    if (!reducedMotion && els.generationNew.animate && els.generationBase.animate && els.generationScan.animate) {
                        const options = { duration: 1350, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" };
                        const reveal = els.generationNew.animate([
                            { clipPath: "inset(0 0 100% 0)" },
                            { clipPath: "inset(0 0 0% 0)" },
                        ], options);
                        const conceal = els.generationBase.animate([
                            { clipPath: "inset(0% 0 0 0)" },
                            { clipPath: "inset(100% 0 0 0)" },
                        ], options);
                        const scan = els.generationScan.animate([
                            { transform: "translateY(-100%)", opacity: 1 },
                            { transform: "translateY(0)", opacity: 1 },
                        ], options);
                        els.generationLayer._animations = [reveal, conceal, scan];
                        await Promise.all([reveal.finished.catch(() => {}), conceal.finished.catch(() => {}), scan.finished.catch(() => {})]);
                    }
                    if (revision !== previewGenerationRevision) return false;
                    els.previewImg.src = src;
                    els.previewImg.alt = `Preview of ${name}`;
                    els.previewImg.style.display = "block";
                    els.previewImg.style.opacity = "";
                    try { await els.previewImg.decode?.(); } catch { /* keep the completed transition visible */ }
                    await new Promise(resolve => window.requestAnimationFrame ? window.requestAnimationFrame(resolve) : resolve());
                    if (revision !== previewGenerationRevision) return false;
                    els.placeholder.style.display = "none";
                    if (els._ui) els._ui._previewSource = src;
                    stopClothesPreviewGeneration(revision);
                    return true;
                };
                node._vnccsStartWorkflowPreview = () => {
                    if (!state.character) return null;
                    workflowPreviewGeneration = {
                        revision: startClothesPreviewGeneration(),
                        character: state.character,
                        costume: hasSelectedEditableCostume() ? state.costume : "Naked",
                    };
                    return workflowPreviewGeneration.revision;
                };
                node._vnccsRevealWorkflowPreview = character => {
                    const workflow = workflowPreviewGeneration;
                    const currentCostume = hasSelectedEditableCostume() ? state.costume : "Naked";
                    if (!workflow || workflow.character !== character || state.character !== character
                        || workflow.costume !== currentCostume) return null;
                    if (workflow.promise) return workflow.promise;
                    const currentRequest = beginPreviewRequest();
                    spritePreviewNavigator?.invalidate?.();
                    spritePreviewNavigator?.hideNav?.();
                    previewLoading?.classList.remove("is-visible");
                    const params = new URLSearchParams({
                        character,
                        costume: workflow.costume,
                        ts: String(Date.now()),
                        force_cache: "true",
                    });
                    const url = mediaURL(`/vnccs/get_preview?${params.toString()}`);
                    workflow.promise = (async () => {
                        const presented = await revealClothesPreview(url, character, workflow.revision);
                        return Boolean(presented && currentRequest() && state.character === character
                            && (hasSelectedEditableCostume() ? state.costume : "Naked") === workflow.costume);
                    })();
                    return workflow.promise;
                };
                const workflowPreviewStartHandler = event => {
                    const executingNode = event.detail?.node ?? event.detail;
                    if (String(executingNode) === String(node.id)) node._vnccsStartWorkflowPreview();
                };
                const workflowPreviewFailureHandler = event => {
                    const failedNode = event.detail?.node_id ?? event.detail?.node;
                    if (workflowPreviewGeneration && (failedNode == null || String(failedNode) === String(node.id))) {
                        stopClothesPreviewGeneration(workflowPreviewGeneration.revision, true);
                    }
                };
                const workflowPreviewCompleteHandler = event => {
                    const executedNode = event.detail?.node ?? event.detail;
                    if (String(executedNode) !== String(node.id) || !workflowPreviewGeneration) return;
                    node._vnccsRevealWorkflowPreview(workflowPreviewGeneration.character)?.catch(error => {
                        console.warn("[VNCCS] Workflow clothes preview transition failed", error);
                    });
                };
                api.addEventListener("executing", workflowPreviewStartHandler);
                api.addEventListener("executed", workflowPreviewCompleteHandler);
                api.addEventListener("execution_error", workflowPreviewFailureHandler);
                api.addEventListener("execution_interrupted", workflowPreviewFailureHandler);
                registerCleanup(node, () => {
                    api.removeEventListener("executing", workflowPreviewStartHandler);
                    api.removeEventListener("executed", workflowPreviewCompleteHandler);
                    api.removeEventListener("execution_error", workflowPreviewFailureHandler);
                    api.removeEventListener("execution_interrupted", workflowPreviewFailureHandler);
                    delete node._vnccsStartWorkflowPreview;
                    delete node._vnccsRevealWorkflowPreview;
                });
                els.cancelPreviewGeneration = restore => stopClothesPreviewGeneration(undefined, restore);
                registerCleanup(node, () => els.cancelPreviewGeneration?.());
                spritePreviewNavigator = createSpritePreviewNavigator({
                    node,
                    isSelectionCurrent: preview => preview.character === state.character
                        && preview.costume === (hasSelectedEditableCostume() ? state.costume : "Naked"),
                    image: pImg,
                    interactions: els._ui,
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
                const colMid = document.createElement("div"); colMid.className = "vnccs-clothes-col vnccs-clothes-design-panel";
                colMid.dataset.sakuraScroll = "design";
                colMid.style.paddingTop = "0"; // Reset padding for tabs

                const designContent = document.createElement("div");
                designContent.className = "vnccs-clothes-design-content";
                colMid.appendChild(designContent);

                const costumeEmptyState = document.createElement("div");
                costumeEmptyState.className = "vnccs-clothes-costume-empty";
                costumeEmptyState.setAttribute("role", "status");
                costumeEmptyState.setAttribute("aria-live", "polite");
                costumeEmptyState.setAttribute("aria-hidden", "true");
                costumeEmptyState.innerHTML = `
                    <svg class="vnccs-clothes-costume-empty-arrow" viewBox="0 0 62 42" aria-hidden="true">
                        <path d="M60 21H10"></path><path d="m18 12-9 9 9 9"></path>
                    </svg>
                    <span class="vnccs-clothes-costume-empty-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24"><path d="M8 4h8l1 3 3 2-2 4-2-1v8H8v-8l-2 1-2-4 3-2 1-3Z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>
                    </span>
                    <div class="vnccs-clothes-costume-empty-title">Create a costume first</div>
                    <div class="vnccs-clothes-costume-empty-text">Click NEW to create a costume. The design controls will appear here as soon as it is ready.</div>`;
                colMid.appendChild(costumeEmptyState);
                els.designPanel = colMid;
                els.designContent = designContent;
                els.costumeEmptyState = costumeEmptyState;

                const alignCostumeEmptyState = () => requestAnimationFrame(() => {
                    if (!actionRow.isConnected || !colMid.isConnected) return;
                    const actionRect = actionRow.getBoundingClientRect();
                    const panelRect = colMid.getBoundingClientRect();
                    const scaleY = panelRect.height / Math.max(colMid.offsetHeight, 1);
                    const center = (actionRect.top - panelRect.top + actionRect.height / 2) / Math.max(scaleY, 0.01);
                    if (Number.isFinite(center)) costumeEmptyState.style.setProperty("--vnccs-costume-empty-y", `${center}px`);
                });
                els.alignCostumeEmptyState = alignCostumeEmptyState;
                node._vnccsAlignCostumeEmptyState = alignCostumeEmptyState;
                registerCleanup(node, () => { delete node._vnccsAlignCostumeEmptyState; });

                const controlsWrap = createGenerationControls();
                controlsWrap.style.paddingTop = "16px";
                designContent.appendChild(controlsWrap);
                designContent.appendChild(createResolutionControl());
                const promptEnhancementControl = createPromptEnhancementControl();

                // Tab Header
                const tabBar = document.createElement("div");
                tabBar.className = "cd-tab-bar";
                tabBar.setAttribute("role", "group");
                tabBar.setAttribute("aria-label", "Clothes mode");

                const createTab = (id, label) => {
                    const t = document.createElement("button");
                    t.type = "button";
                    t.dataset.focusKey = `clothes-tab-${id}`;
                    t.className = "cd-tab" + (state.activeTab === id ? " active" : "");
                    t.innerText = label;
                    t.onclick = () => {
                        els.cancelPreviewGeneration?.(true);
                        beginClothesUploadRequest();
                        state.activeTab = id;
                        saveState();
                        refreshTabs();
                    };
                    return t;
                };

                const contentArea = document.createElement("div");
                contentArea.style.flex = "1"; contentArea.style.display = "flex"; contentArea.style.flexDirection = "column";

                const refreshTabs = () => {
                    const view = els._ui.captureView();
                    tabBar.innerHTML = '';
                    tabBar.appendChild(createTab("generate", "GENERATE CLOTHES"));
                    tabBar.appendChild(createTab("clone", "CLONE CLOTHES"));

                    if (state.activeTab === "generate") renderAttributes(contentArea);
                    else renderClonePanel(contentArea);
                    syncPromptEnhancementControl();
                    syncCostumeEditControls();
                    els._ui.syncSelection(tabBar);
                    els._ui.syncSelects();
                    els._ui.restoreView(view);
                };

                refreshTabs(); // Initial Render

                designContent.appendChild(tabBar);
                designContent.appendChild(contentArea);
                topRow.appendChild(colMid);
                syncCostumeEditControls();

                node._vnccsRestoreClothesState = () => {
                    els.cancelPreviewGeneration?.(true);
                    beginClothesUploadRequest();
                    beginClothesWizardRequest();
                    beginDeleteRequest();
                    spritePreviewNavigator?.invalidate?.();
                    if (node._vnccsClothesSelectionPending) {
                        beginSelectionRequest();
                        beginCharacterInfoRequest();
                        beginCostumeInfoRequest();
                        node._vnccsClothesSelectionPending = false;
                    }
                    try {
                        const restored = JSON.parse(dataWidget.value || "{}");
                        if (!restored || typeof restored !== "object" || Array.isArray(restored)
                            || ["costume_info", "character_info", "gen_settings"].some(key => restored[key] !== undefined
                                && (!restored[key] || typeof restored[key] !== "object" || Array.isArray(restored[key])))) {
                            throw new Error("Saved clothes state must contain objects.");
                        }
                        delete node._vnccsStateRestoreError;
                        Object.assign(state, defaultState, restored, {
                            costume_info: { ...defaultState.costume_info, ...restored.costume_info },
                            character_info: { ...defaultState.character_info, ...restored.character_info },
                            gen_settings: { ...defaultState.gen_settings, ...restored.gen_settings },
                        });
                        node._vnccsCostumeDraft = restored.costume_info ? { character: state.character, costume: state.costume } : null;
                        for (const key of Object.keys(defaultState.costume_info)) {
                            if (els[key]) els[key].value = state.costume_info[key];
                        }
                        if (els.charSelect) {
                            if (state.character && !Array.from(els.charSelect.options).some(option => option.value === state.character)) {
                                els.charSelect.add(new Option(state.character, state.character));
                            }
                            els.charSelect.value = state.character;
                        }
                        if (els.costSel) els.costSel.value = state.costume;
                        refreshTabs();
                        syncGenerationControls();
                        syncCostumeEditControls();
                    } catch (error) {
                        node._vnccsStateRestoreError = "Cannot restore saved clothes state. Original widget data is preserved. Restore a valid workflow before editing or running.";
                        console.warn("[VNCCS] Clothes Designer state restore failed", error);
                        showInfo("State Restore Failed", node._vnccsStateRestoreError);
                    }
                };
                registerCleanup(node, () => { delete node._vnccsRestoreClothesState; delete node._vnccsCostumeDraft; });

                // Initial Load
                (async () => {
                    const isMounted = createRequestGuard(node)();
                    const currentSelection = beginSelectionRequest();
                    const currentPreview = beginPreviewRequest();
                    const r = await api.fetchApi("/vnccs/context_lists");
                    const d = await r.json();
                    if (!isMounted()) return;
                    if (node._vnccsStateRestoreError) return;
                    setClothesCoreLora();
                    syncGenerationControls();
                    saveState(); // Ensure defaults are persisted immediately

                    // Char List
                    const selectedCharacter = els.charSelect.value || state.character;
                    els.charSelect.innerHTML = "";
                    d.characters.forEach(c => els.charSelect.add(new Option(c, c)));
                    if (selectedCharacter) {
                        if (!d.characters.includes(selectedCharacter)) els.charSelect.add(new Option(selectedCharacter, selectedCharacter));
                        els.charSelect.value = selectedCharacter;
                    }
                    else if (d.characters.length) { state.character = d.characters[0]; els.charSelect.value = state.character; }

                    if (!currentSelection()) return;
                    if (!await loadCharacterInfo() || !currentSelection()) return;
                    if (!await loadCostumes(true) || !currentSelection()) return;
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
                    els.cancelPreviewGeneration?.(true);
                    state.costume = "";
                    state.costume_revision = null;
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
                const loadCostumes = async (preserveDraft = false) => {
                    const currentRequest = beginCostumesRequest();
                    const c = state.character;
                    const originalInfo = state.costume_info;
                    const isCurrent = () => currentRequest() && state.character === c
                        && (preserveDraft || state.costume_info === originalInfo);
                    if (!c) return;
                    try {
                        const r = await api.fetchApi(`/vnccs/list_costumes?character=${encodeURIComponent(c)}`);
                        let list = await r.json();
                        if (!isCurrent()) return false;
                        if (!r.ok || !Array.isArray(list)) throw new Error("Failed to load character costumes.");

                        // Filter base sprite sets from display list.
                        const displayList = list.filter(i => i !== "Naked" && i !== "Original");
                        if (preserveDraft && node._vnccsCostumeDraft?.character === c
                            && node._vnccsCostumeDraft?.costume === state.costume && hasSelectedEditableCostume()
                            && !displayList.includes(state.costume)) displayList.push(state.costume);

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

                        if (!await loadCostumeInfo(preserveDraft)) return false;
                        syncCostumeEditControls();
                        return true;
                    } catch (error) {
                        if (isCurrent()) showInfo("Error", error.message || String(error));
                        return false;
                    }
                };

                const beginCostumeInfoRequest = createRequestGuard(node);
                const loadCostumeInfo = async (preserveDraft = false, costume = state.costume) => {
                    const currentRequest = beginCostumeInfoRequest();
                    const c = state.character;
                    const originalCostume = state.costume;
                    const originalInfo = state.costume_info;
                    const cos = costume;
                    const isCurrent = () => currentRequest() && state.character === c && state.costume === originalCostume
                        && (preserveDraft || state.costume_info === originalInfo);
                    if (els.btnGen) els.btnGen.disabled = true;
                    if (els.btnDel) els.btnDel.disabled = true;
                    ["top", "bottom", "head", "face", "shoes"].forEach(key => {
                        if (els[key]) els[key].disabled = true;
                    });
                    if (els.wizardBtn) els.wizardBtn.disabled = true;
                    try {
                        const r = await api.fetchApi(`/vnccs/get_costume?character=${encodeURIComponent(c)}&costume=${encodeURIComponent(cos)}`);
                        const info = await r.json();
                        if (!isCurrent()) return false;
                        if (!r.ok) throw new Error("Failed to load costume metadata.");
                        if (!info || typeof info !== "object" || Array.isArray(info)) {
                            throw new Error("Invalid costume metadata.");
                        }

                        state.costume = cos;
                        const preservingDraft = preserveDraft && node._vnccsCostumeDraft?.character === c
                            && node._vnccsCostumeDraft?.costume === cos;
                        const draftMatchesDisk = preservingDraft && Object.entries(state.costume_info).every(([key, value]) =>
                            JSON.stringify(value) === JSON.stringify(info[key] ?? defaultState.costume_info[key]));
                        if (!preservingDraft || draftMatchesDisk) {
                            state.costume_revision = r.headers?.get?.("ETag") || null;
                        }
                        state.costume_info = {
                            ...info,
                            top: info.top || "",
                            bottom: info.bottom || "",
                            head: info.head || "",
                            face: info.face || "",
                            shoes: info.shoes || "",
                            ...(preserveDraft && node._vnccsCostumeDraft?.character === c
                                && node._vnccsCostumeDraft?.costume === cos ? state.costume_info : {})
                        };

                        for (const k of ["top", "bottom", "head", "face", "shoes"]) {
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
                        if (isCurrent()) {
                            if (cos === originalCostume && !(preserveDraft && node._vnccsCostumeDraft?.character === c
                                && node._vnccsCostumeDraft?.costume === cos)) resetCostumeSelection();
                            showInfo("Error", error.message || String(error));
                        }
                        return false;
                    }
                };

                const updatePreviewImage = async (forceCache = false) => {
                    els.cancelPreviewGeneration?.(true);
                    const currentRequest = beginPreviewRequest();
                    const character = state.character;
                    const costume = state.costume;
                    const info = state.costume_info;
                    const isCurrent = () => currentRequest() && state.character === character && state.costume === costume
                        && state.costume_info === info;
                    spritePreviewNavigator?.invalidate?.();
                    if (!state.character) return;
                    const ts = Date.now();
                    const previewCostume = hasSelectedEditableCostume() ? state.costume : "Naked";
                    const selectedPreview = state.selected_preview_sprite;
                    const selectedPreviewIndex = selectedPreview?.character === character
                        && (selectedPreview.costume || "") === previewCostume ? selectedPreview.index : undefined;
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
                            els._ui.emptyPreview(els.previewImg, els.placeholder, character, "No preview available");
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
                            index: selectedPreviewIndex,
                            fallbackUrl: mediaURL(url),
                        });
                    }
                };

                const onPreviewUpdated = (event) => {
                    const targetId = String(event.detail?.node_id);
                    const myId = String(node.id);
                    if (targetId === myId) {
                        console.log("VNCCS Preview Update Received for Node:", myId);
                        const transition = node._vnccsRevealWorkflowPreview?.(event.detail?.character);
                        if (transition) transition.catch(error => {
                            console.warn("[VNCCS] Workflow clothes preview transition failed", error);
                            updatePreviewImage(true);
                        });
                        else updatePreviewImage(true);
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
                requestAnimationFrame(() => {
                    syncDOMWidgetWidth(this, "clothes_designer_ui");
                    this._vnccsAlignCostumeEmptyState?.();
                });
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
