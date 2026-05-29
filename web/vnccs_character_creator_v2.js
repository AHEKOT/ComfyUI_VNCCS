import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { debounce, registerCleanup, showModal as showCommonModal, createLoadingOverlay, showMessage, generateRandomSeed, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan } from "./vnccs_common.js";

// --- STYLES: Sakura Archive Design System ---
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
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
    --accent-glow: rgba(255, 143, 163, 0.3);
    --accent-subtle: rgba(255, 143, 163, 0.1);
    --accent-border: rgba(255, 143, 163, 0.22);
    --accent-lavender: #b8a9e8;
    --success: #00d68f;
    --warning: #ffaa00;
    --error: #ff4757;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --shadow-subtle: 0 2px 8px rgba(0,0,0,0.3);
    --shadow-elevated: 0 8px 32px rgba(0,0,0,0.5);
    --transition: 0.2s ease;
}

/* Main Host */
.vnccs-container {
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
.vnccs-top-row {
    display: grid;
    grid-template-columns: 30% 35% 35%;
    gap: 12px;
    flex: 1;
    min-height: 0;
    width: 100%;
}
.vnccs-bottom-row {
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
.vnccs-col {
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
.vnccs-col::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.5), transparent);
    border-radius: 1px;
}
.vnccs-col::-webkit-scrollbar { width: 4px; }
.vnccs-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* Section titles */
.vnccs-section-title {
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
.vnccs-section-title::before {
    content: '';
    width: 3px;
    height: 12px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
}

/* Interactive elements */
.vnccs-field,
.vnccs-btn-row > *,
.vnccs-preview-container,
.vnccs-lora-item,
.vnccs-textarea-wrapper,
.vnccs-slider-container,
.vnccs-input,
.vnccs-select,
.vnccs-textarea {
    pointer-events: auto;
}

/* Fields */
.vnccs-field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 6px; flex-shrink: 0; }
.vnccs-label {
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Inputs */
.vnccs-input, .vnccs-textarea {
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
.vnccs-select {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    zoom: 1.5;
    transition: all var(--transition);
    color-scheme: dark;
}
.vnccs-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}
.vnccs-input:focus, .vnccs-select:focus, .vnccs-textarea:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255,143,163,0.04);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

/* Slider */
.vnccs-slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 6px 10px;
}
.vnccs-slider {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    height: 3px;
}
.vnccs-slider-val {
    width: 42px;
    text-align: right;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-primary);
    background: transparent;
    border: none;
}
.vnccs-slider-val:focus { outline: none; border-bottom: 1px solid var(--accent-border); }

/* Preview */
.vnccs-preview-container {
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
.vnccs-preview-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    animation: vnccs-fadein 0.4s ease;
}
@keyframes vnccs-fadein { from { opacity: 0; } to { opacity: 1; } }
.vnccs-placeholder { color: var(--text-muted); text-align: center; font-size: 11px; }

/* LoRA stack */
.vnccs-lora-stack {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
}
.vnccs-lora-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: rgba(255,255,255,0.02);
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    transition: border-color var(--transition);
}
.vnccs-lora-item:hover { border-color: var(--border-hover); }
.vnccs-lora-row { display: flex; gap: 6px; align-items: center; }

/* Buttons */
.vnccs-btn-row { display: flex; gap: 8px; margin-top: auto; flex-shrink: 0; }
.vnccs-btn {
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
.vnccs-btn-primary {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525;
    box-shadow: 0 4px 16px rgba(255,143,163,0.25);
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.vnccs-btn-primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: vnccs-shimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes vnccs-shimmer {
    0% { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35% { transform: translateX(120%) skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%) skewX(-15deg); opacity: 0; }
}
.vnccs-btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255,143,163,0.4);
}
.vnccs-container button.vnccs-btn.vnccs-btn-primary:not(:disabled),
.vnccs-container button.vnccs-btn.vnccs-btn-primary:not(:disabled):hover,
.vnccs-container button.vnccs-btn.vnccs-btn-primary:not(:disabled):focus,
.vnccs-container button.vnccs-btn.vnccs-btn-primary:not(:disabled):focus-visible,
.vnccs-container button.vnccs-btn.vnccs-btn-primary:not(:disabled):active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    outline: none;
}
.vnccs-btn-success {
    background: rgba(0,214,143,0.15);
    color: var(--success);
    border: 1px solid rgba(0,214,143,0.3);
}
.vnccs-btn-success:hover:not(:disabled) {
    background: rgba(0,214,143,0.25);
    transform: translateY(-1px);
}
.vnccs-btn-danger {
    background: rgba(255,71,87,0.15);
    color: var(--error);
    border: 1px solid rgba(255,71,87,0.3);
}
.vnccs-btn-danger:hover:not(:disabled) {
    background: rgba(255,71,87,0.25);
    transform: translateY(-1px);
}
.vnccs-btn-disabled, .vnccs-btn:disabled {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-muted) !important;
    cursor: not-allowed;
    box-shadow: none !important;
    transform: none !important;
}
.vnccs-btn:focus,
.vnccs-btn:focus-visible,
.vnccs-segmented-btn:focus,
.vnccs-segmented-btn:focus-visible,
.vnccs-seed-dice-btn:focus,
.vnccs-seed-dice-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-btn-primary:focus:not(:disabled),
.vnccs-btn-primary:focus-visible:not(:disabled),
.vnccs-btn-primary:active:not(:disabled) {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 8px 28px rgba(255,143,163,0.4), 0 0 0 2px rgba(255,143,163,0.28);
}

/* Bottom textareas */
.vnccs-textarea-wrapper {
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
.vnccs-textarea-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.4), transparent);
}
.vnccs-textarea-wrapper textarea {
    flex: 1;
    resize: none;
    border: none;
    background: transparent;
    padding: 4px;
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
}
.vnccs-textarea-wrapper textarea:focus { outline: none; }
.vnccs-textarea-label {
    font-size: 9px;
    color: var(--accent);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0 2px 4px;
}

/* Tag constructor */
.vnccs-tag-btn {
    width: 20px; height: 20px;
    background: rgba(255,143,163,0.1);
    color: var(--accent);
    border: 1px solid var(--accent-border);
    border-radius: 6px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
    margin-left: auto;
    flex-shrink: 0;
    transition: all var(--transition);
}
.vnccs-tag-btn:hover { background: rgba(255,143,163,0.2); box-shadow: 0 0 8px var(--accent-glow); }

.vnccs-tag-grid {
    display: flex; flex-wrap: wrap; gap: 5px;
    max-height: 300px; overflow-y: auto;
    padding: 8px;
    background: rgba(10,10,15,0.6);
    border-radius: var(--radius-sm);
}
.vnccs-tag-chip {
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
.vnccs-tag-chip:hover { background: rgba(255,143,163,0.1); border-color: var(--accent-border); color: var(--accent-hover); }
.vnccs-tag-chip.selected { background: rgba(255,143,163,0.18); color: var(--accent-hover); border-color: var(--accent); }

.vnccs-tag-category {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 6px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.8px;
    width: 100%;
}

/* Custom toggle checkbox */
.vnccs-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    padding: 6px 0;
    user-select: none;
}
.vnccs-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-toggle input {
    opacity: 0;
    width: 0; height: 0;
    position: absolute;
}
.vnccs-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    transition: all 0.25s ease;
}
.vnccs-toggle-thumb {
    position: absolute;
    top: 3px; left: 3px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.25s ease;
}
.vnccs-toggle input:checked ~ .vnccs-toggle-track {
    background: rgba(255,143,163,0.2);
    border-color: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
}
.vnccs-toggle input:checked ~ .vnccs-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
}
.vnccs-toggle-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: color var(--transition);
}
.vnccs-toggle input:checked ~ ~ .vnccs-toggle-label,
.vnccs-toggle-wrap:has(input:checked) .vnccs-toggle-label {
    color: var(--accent-hover);
}

.vnccs-segmented-field {
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

.vnccs-segmented-btn {
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

.vnccs-segmented-btn:hover {
    color: var(--text-primary);
    background: rgba(255,255,255,0.045);
}

.vnccs-segmented-btn.is-active {
    color: #20141a;
    background: linear-gradient(180deg, #ff9bad 0%, #ff87a0 100%);
    box-shadow: 0 10px 22px rgba(255,143,163,0.22);
}

.vnccs-graphic-toggle {
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

.vnccs-graphic-toggle:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-graphic-toggle.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.14);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset, 0 12px 24px rgba(255,143,163,0.12);
}

.vnccs-graphic-toggle-text {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.vnccs-graphic-toggle-icon {
    width: 20px;
    height: 20px;
    border-radius: 7px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: currentColor;
    background: rgba(255,255,255,0.06);
}

.vnccs-graphic-toggle-switch {
    width: 44px;
    height: 24px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    position: relative;
    flex-shrink: 0;
    transition: all var(--transition);
}

.vnccs-graphic-toggle-switch::after {
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

.vnccs-graphic-toggle.is-active .vnccs-graphic-toggle-switch {
    border-color: var(--accent);
    background: rgba(255,143,163,0.28);
}

.vnccs-graphic-toggle.is-active .vnccs-graphic-toggle-switch::after {
    transform: translateX(20px);
    background: var(--accent-hover);
}

/* Filled input highlight */
.vnccs-input:not(:placeholder-shown):not([value=""]),
.vnccs-input.has-value {
    border-color: rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
}

/* Preview placeholder with icon */
.vnccs-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    color: var(--text-muted);
    font-size: 11px;
    letter-spacing: 0.05em;
}
.vnccs-placeholder-icon {
    width: 48px;
    height: 48px;
    opacity: 0.25;
}

/* LoRA slot collapsed state */
.vnccs-lora-item.is-empty {
    opacity: 0.45;
}
.vnccs-lora-item.is-empty:hover {
    opacity: 1;
}

/* Button hierarchy */
.vnccs-btn-generate {
    flex: 2;
}
.vnccs-btn-secondary {
    flex: 1;
    font-size: 10px;
}

.vnccs-tab-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 6px;
}

.vnccs-tab {
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

.vnccs-tab:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-tab.is-active {
    border-color: var(--accent);
    color: var(--accent-hover);
    background: rgba(255,143,163,0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}

.vnccs-subsection {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-model-card-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.vnccs-model-picker {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-model-picker-menu {
    display: none;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    border: 1px solid rgba(255,143,163,0.18);
    border-radius: 10px;
    background: rgba(8,8,12,0.48);
}

.vnccs-model-picker.is-open .vnccs-model-picker-menu {
    display: flex;
}

.vnccs-model-picker-group {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.vnccs-model-picker-group-title {
    color: var(--accent-hover);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.vnccs-model-card {
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

.vnccs-model-card.is-picker-head {
    min-height: 58px;
}

.vnccs-model-card.is-installed {
    cursor: pointer;
}

.vnccs-model-card.is-installed:hover {
    border-color: rgba(0,214,143,0.42);
    background: rgba(0,214,143,0.08);
}

.vnccs-model-card.is-selected {
    border-color: var(--accent);
    background: rgba(255,143,163,0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset;
}

.vnccs-model-card.is-missing {
    opacity: 0.92;
}

.vnccs-model-card-top {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
}

.vnccs-model-card-name {
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

.vnccs-model-card-status {
    flex-shrink: 0;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.vnccs-model-card-status.ok { color: var(--success); }
.vnccs-model-card-status.missing { color: var(--error); }
.vnccs-model-card-status.progress { color: var(--accent-lavender); }

.vnccs-model-card-desc {
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.4;
}

.vnccs-model-card-actions {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vnccs-model-card-download {
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

.vnccs-model-card-download:hover {
    background: rgba(255,143,163,0.14);
}

.vnccs-model-card-badge {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    background: var(--text-muted);
}

.vnccs-model-card-badge.ok { background: var(--success); }
.vnccs-model-card-badge.missing { background: var(--error); }
.vnccs-model-card-badge.progress { background: var(--accent-lavender); }

.vnccs-model-card-toggle {
    margin-left: auto;
    flex-shrink: 0;
}

.vnccs-generation-fallback {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-gen-param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 18px;
    padding-top: 2px;
}

.vnccs-gen-param-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
}

.vnccs-gen-param-input,
.vnccs-gen-param-select {
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

.vnccs-gen-param-select {
    zoom: 1;
}

.vnccs-gen-param-input:focus,
.vnccs-gen-param-select:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255,143,163,0.045);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

.vnccs-gen-param-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}

.vnccs-seed-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 52px;
    gap: 8px;
    align-items: stretch;
}

.vnccs-seed-dice-btn {
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

.vnccs-seed-dice-btn:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-seed-dice-btn.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.16);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}

.vnccs-seed-dice-btn svg {
    width: 24px;
    height: 24px;
    display: block;
}

.vnccs-character-wizard-btn {
    width: 100%;
    min-height: 40px;
    margin-bottom: 8px;
    flex: 0 0 auto;
}

.vnccs-container .vnccs-common-modal {
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
.vnccs-container .vnccs-common-modal-title {
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-hover);
    font-family: var(--font);
}
.vnccs-container .vnccs-common-modal-btn {
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: var(--font);
    font-weight: 700;
}
.vnccs-container .vnccs-common-modal-btn:focus,
.vnccs-container .vnccs-common-modal-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-container .vnccs-common-modal-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    border-color: transparent !important;
}
.vnccs-container .vnccs-common-modal-btn-primary:hover,
.vnccs-container .vnccs-common-modal-btn-primary:focus,
.vnccs-container .vnccs-common-modal-btn-primary:focus-visible,
.vnccs-container .vnccs-common-modal-btn-primary:active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
}
.vnccs-character-wizard-modal {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
}

.vnccs-character-wizard-modal-text {
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.45;
    white-space: normal;
    overflow-wrap: anywhere;
}

.vnccs-character-wizard-modal textarea {
    width: 100%;
    min-height: 110px;
    box-sizing: border-box;
    resize: vertical;
}
`;

app.registerExtension({
    name: "VNCCS.CharacterCreatorV2",

    async setup() {
        const origQueuePrompt = app.queuePrompt.bind(app);
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
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1280, 800]); // Default wide 3-column
                syncDOMWidgetWidthSoon(node, "ui");

                // 1. Setup CSS
                const style = document.createElement("style");
                style.innerHTML = STYLE;
                document.head.appendChild(style);

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

                const state = {
                    preview_valid: false, // Smart Cache Flag
                    preview_source: "gen", // "gen" or "pose" - tracks what user sees
                    character: "",
                    prompt_modes: {
                        illustrious: {
                            aesthetics: "masterpiece, best quality",
                            negative_prompt: "bad quality, worst quality",
                        },
                        anima: {
                            aesthetics: "masterpiece, best quality, score_7, anime",
                            negative_prompt: "bad quality, worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
                        },
                    },
                    prompt_defaults_version: 1,
                    character_info: {
                        sex: "female", age: 18, race: "human", skin_color: "",
                        hair: "black hair, long hair", eyes: "", face: "", body: "", additional_details: "",
                        nsfw: false, aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                        lora_prompt: "", background_color: "Green"
                    },
                    gen_settings: {
                        generation_mode: "illustrious",
                        ckpt_name: "", sampler: "euler", scheduler: "normal",
                        steps: 20, cfg: 8.0, seed: generateRandomSeed(), seed_mode: "fixed",
                        diffusion_model_name: "", clip_name: "", vae_name: "",
                        mode_settings: {
                            illustrious: {
                                ckpt_name: "", sampler: "euler", scheduler: "normal",
                                steps: 20, cfg: 8.0, seed: generateRandomSeed(), seed_mode: "fixed",
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
                                sampler: "er_sde", scheduler: "simple",
                                steps: 30, cfg: 4.0, seed: generateRandomSeed(), seed_mode: "fixed",
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
                const ILLUSTRIOUS_DEFAULTS = {
                    ckpt_name: "", sampler: "euler", scheduler: "normal",
                    steps: 20, cfg: 8.0, seed: generateRandomSeed(), seed_mode: "fixed",
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
                    sampler: "er_sde", scheduler: "simple",
                    steps: 30, cfg: 4.0, seed: generateRandomSeed(), seed_mode: "fixed",
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
                const GENERATION_DEFAULTS_VERSION = 3;
                const PROMPT_DEFAULTS_VERSION = 1;
                const MODE_SETTING_KEYS = {
                    illustrious: ["ckpt_name", "sampler", "scheduler", "steps", "cfg", "seed", "seed_mode", "dmd_lora_name", "dmd_lora_strength", "age_lora_name", "lora_stack"],
                    anima: ["diffusion_model_name", "clip_name", "vae_name", "sampler", "scheduler", "steps", "cfg", "seed", "seed_mode", "turbo_enabled", "dmd_lora_name", "dmd_lora_strength", "turbo_previous_settings", "lora_stack"],
                };
                const MODE_PROMPT_DEFAULTS = {
                    illustrious: {
                        aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                    },
                    anima: {
                        aesthetics: "masterpiece, best quality, score_7, anime",
                        negative_prompt: "bad quality, worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
                    },
                };
                const CC_REPO_ID = "MIUProject/VNCCS_V2";
                const CC_CACHE_KEY = `vnccs_cc_cache_${CC_REPO_ID}`;
                let TAG_DATA = null;
                let ccConfig = null;
                let ccDlStatus = {};
                let ccPollingInterval = null;
                const modelPickerOpen = {
                    illustrious: false,
                    anima: false,
                };
                let localAssets = {
                    checkpoints: [],
                    diffusion_models: [],
                    text_encoders: [],
                    vae_models: [],
                };
                let restoredWidgetInfoCharacter = null;

                const ccNormalize = (value) => String(value || "").trim().toLowerCase();
                const ccKind = (entry) => ccNormalize(entry?.kind ?? entry?.Kind);
                const ccType = (entry) => ccNormalize(entry?.type ?? entry?.Type);
                const ccStatusKey = (cat, entry) => `cc_${cat}_${entry?.name || ""}`;
                const ccResolveStatus = (entry, cat) => {
                    const transient = new Set(["queued", "downloading", "error", "auth_required"]);
                    const dls = ccDlStatus[ccStatusKey(cat, entry)] || {};
                    return transient.has(dls.status) ? dls.status : (entry?.status || "missing");
                };
                const ccRelPath = (entry) => {
                    const localPath = String(entry?.local_path || "").replace(/\\/g, "/");
                    const parts = localPath.split("/").filter(Boolean);
                    if (parts.length >= 3 && parts[0] === "models") return parts.slice(2).join("/");
                    return parts[parts.length - 1] || "";
                };
                const ccEntries = (section, kind, predicate = null) => {
                    const entries = ccConfig?.[section] || [];
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
                    return hasAnima && hasIllustrious;
                };
                const ccDownloadEntry = async (cat, entry) => {
                    if (!entry?.name) return;
                    const key = ccStatusKey(cat, entry);
                    ccDlStatus[key] = { status: "queued", message: "Queued…" };
                    renderControlCenterCards();
                    try {
                        const response = await api.fetchApi("/vnccs/control_center/download", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ repo_id: CC_REPO_ID, category: cat, name: entry.name }),
                        });
                        const payload = await response.json();
                        if (!response.ok || payload.error) {
                            ccDlStatus[key] = { status: "error", message: payload.error || "Download failed" };
                            renderControlCenterCards();
                            return;
                        }
                        startCcPolling();
                    } catch (error) {
                        ccDlStatus[key] = { status: "error", message: String(error?.message || error) };
                        renderControlCenterCards();
                    }
                };
                const fetchCcConfig = async (force = false) => {
                    if (!force && window.VNCCS_CC_REGISTRY?.[CC_REPO_ID] && ccHasRequiredFamilies(window.VNCCS_CC_REGISTRY[CC_REPO_ID])) {
                        ccConfig = window.VNCCS_CC_REGISTRY[CC_REPO_ID];
                        renderControlCenterCards();
                        return ccConfig;
                    }
                    if (!force && ccConfig && ccHasRequiredFamilies(ccConfig)) return ccConfig;
                    if (!force) {
                        try {
                            const cached = localStorage.getItem(CC_CACHE_KEY);
                            if (cached) {
                                ccConfig = JSON.parse(cached);
                                if (ccHasRequiredFamilies(ccConfig)) renderControlCenterCards();
                                else ccConfig = null;
                            }
                        } catch (_) {}
                    }

                    const url = `/vnccs/control_center/check?repo_id=${encodeURIComponent(CC_REPO_ID)}${force ? "&force_refresh=true" : ""}`;
                    const response = await api.fetchApi(url);
                    const payload = await response.json();
                    if (!response.ok || payload.error) throw new Error(payload.error || "Failed to load Control Center config");
                    ccConfig = payload;
                    window.VNCCS_CC_REGISTRY = window.VNCCS_CC_REGISTRY || {};
                    window.VNCCS_CC_REGISTRY[CC_REPO_ID] = payload;
                    localStorage.setItem(CC_CACHE_KEY, JSON.stringify(payload));
                    renderControlCenterCards();
                    return payload;
                };
                const refreshCcDownloadStatus = async () => {
                    try {
                        const response = await api.fetchApi("/vnccs/manager/status");
                        if (!response.ok) return;
                        ccDlStatus = await response.json();
                        const active = Object.values(ccDlStatus || {}).some(item => ["queued", "downloading"].includes(item?.status));
                        if (!active) {
                            stopCcPolling();
                            await fetchCcConfig(true);
                        } else {
                            renderControlCenterCards();
                        }
                    } catch (_) {}
                };
                const startCcPolling = () => {
                    if (ccPollingInterval) return;
                    ccPollingInterval = setInterval(refreshCcDownloadStatus, 2000);
                };
                const stopCcPolling = () => {
                    if (!ccPollingInterval) return;
                    clearInterval(ccPollingInterval);
                    ccPollingInterval = null;
                };

                const els = {};
                const cloneSettingsValue = (value) => {
                    if (Array.isArray(value)) return value.map(item => ({ ...item }));
                    if (value && typeof value === "object") return { ...value };
                    return value;
                };

                const getGenerationDefaults = (mode) => ({
                    ...(mode === "anima" ? ANIMA_DEFAULTS : ILLUSTRIOUS_DEFAULTS),
                    seed: generateRandomSeed(),
                });

                const ensureLoraStack = (profile) => {
                    if (!Array.isArray(profile.lora_stack)) profile.lora_stack = [];
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
                    if (normalizedMode === "illustrious" || normalizedMode === "anima") ensureLoraStack(profile);
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
                    if (normalizedMode === "illustrious" || normalizedMode === "anima") ensureLoraStack(profile);
                };

                const applyGenerationProfile = (mode) => {
                    const normalizedMode = (mode || "illustrious").toLowerCase();
                    const profile = getModeProfile(normalizedMode);
                    (MODE_SETTING_KEYS[normalizedMode] || []).forEach((key) => {
                        state.gen_settings[key] = cloneSettingsValue(profile[key]);
                    });
                    if (normalizedMode === "illustrious" || normalizedMode === "anima") ensureLoraStack(state.gen_settings);
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
                        els.dmdSlider.checked = mode === "anima"
                            ? !!g.turbo_enabled
                            : (g.dmd_lora_strength ?? 1.0) > 0;
                    }
                    if (els.ageSelect) els.ageSelect.value = g.age_lora_name || "";
                    if (els.loraStackSelects) {
                        ensureLoraStack(g);
                        els.loraStackSelects.forEach((ref, i) => {
                            const item = g.lora_stack[i] || { name: "", strength: 1.0 };
                            ref.sel.value = item.name || "";
                            ref.rng.value = item.strength ?? 1.0;
                            ref.sel.closest(".vnccs-lora-item")?.classList.toggle("is-empty", !item.name);
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
                    };
                    ensureLoraStack(g.mode_settings.illustrious);
                    ensureLoraStack(g.mode_settings.anima);

                    if ((g.generation_defaults_version || 0) < GENERATION_DEFAULTS_VERSION || !existingModes[currentMode]) {
                        const target = g.mode_settings[currentMode];
                        (MODE_SETTING_KEYS[currentMode] || []).forEach((key) => {
                            if (g[key] !== undefined && g[key] !== "") target[key] = cloneSettingsValue(g[key]);
                        });
                        if (currentMode === "illustrious" || currentMode === "anima") ensureLoraStack(target);
                    }

                    g.generation_defaults_version = GENERATION_DEFAULTS_VERSION;
                    applyGenerationProfile(currentMode);
                };

                const saveState = (isValid = false) => {
                    saveCurrentGenerationModeValues();
                    state.character_info.name = state.character || state.character_info.name || "";
                    const persistData = {
                        gen_settings: state.gen_settings,
                        character: state.character
                    };
                    localStorage.setItem("VNCCS_V2_Settings", JSON.stringify(persistData));

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

                const applyStoredPrefs = (characterOnly = false) => {
                    try {
                        const s = localStorage.getItem("VNCCS_V2_Settings");
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

                            const overridden = applyStoredPrefs(true);
                            console.log("[VNCCS V2] Loaded state from graph widget. Character:", state.character, overridden ? "(last active override)" : "");
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
                    };

                    if ((state.prompt_defaults_version || 0) < PROMPT_DEFAULTS_VERSION) {
                        mergedModes.illustrious.aesthetics = state.character_info.aesthetics || mergedModes.illustrious.aesthetics;
                        mergedModes.illustrious.negative_prompt = state.character_info.negative_prompt || mergedModes.illustrious.negative_prompt;
                        mergedModes.anima.aesthetics = mergePromptDefaultWithCustom(mergedModes.anima.aesthetics, MODE_PROMPT_DEFAULTS.anima.aesthetics);
                        mergedModes.anima.negative_prompt = mergePromptDefaultWithCustom(mergedModes.anima.negative_prompt, MODE_PROMPT_DEFAULTS.anima.negative_prompt);
                    }

                    state.prompt_modes = mergedModes;
                    state.prompt_defaults_version = PROMPT_DEFAULTS_VERSION;
                    applyPromptModeToFields(currentMode);
                };

                const applyGenerationDefaults = (mode, force = false) => {
                    const defaults = getGenerationDefaults(mode);
                    const markerKey = mode === "anima" ? "anima_defaults_applied" : "illustrious_defaults_applied";
                    if (!force && state.gen_settings[markerKey]) return;

                    state.gen_settings.mode_settings[mode] = {
                        ...defaults,
                        ...(["illustrious", "anima"].includes(mode) ? { lora_stack: cloneSettingsValue(defaults.lora_stack) } : {}),
                    };
                    if (mode === state.gen_settings.generation_mode) {
                        applyGenerationProfile(mode);
                        syncGenerationControls();
                    }
                    state.gen_settings[markerKey] = true;
                    state.gen_settings.generation_defaults_version = GENERATION_DEFAULTS_VERSION;
                };

                const refreshGenerationModeUI = () => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    const isAnima = mode === "anima";
                    if (els.modeTabs) {
                        Object.entries(els.modeTabs).forEach(([key, btn]) => {
                            btn.classList.toggle("is-active", key === mode);
                        });
                    }
                    if (els.illustriousModels) els.illustriousModels.style.display = isAnima ? "none" : "flex";
                    if (els.animaModels) els.animaModels.style.display = isAnima ? "flex" : "none";
                    if (els.loraSection) els.loraSection.style.display = "flex";
                    if (els.dmdWrap) els.dmdWrap.style.display = "none";
                    if (els.dmdLabel) els.dmdLabel.innerText = isAnima ? "Turbo LoRA" : "DMD2 LoRA Model";
                    if (els.loraHeader) els.loraHeader.innerText = isAnima ? "ANIMA LoRA Stack" : "LoRa Stack";
                    if (els.ageWrap) els.ageWrap.style.display = "none";
                    if (els.animaLoraCards) els.animaLoraCards.style.display = isAnima && els.animaLoraCards.children.length ? "flex" : "none";
                    if (els.illustriousLoraCards) els.illustriousLoraCards.style.display = !isAnima && els.illustriousLoraCards.children.length ? "flex" : "none";
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
                    if ((state.gen_settings.generation_mode || "illustrious").toLowerCase() !== "anima") return;
                    if (enabled) {
                        if (!state.gen_settings.turbo_enabled) {
                            state.gen_settings.turbo_previous_settings = {
                                steps: state.gen_settings.steps,
                                cfg: state.gen_settings.cfg,
                            };
                        }
                        state.gen_settings.turbo_enabled = true;
                        state.gen_settings.dmd_lora_strength = 1.0;
                        state.gen_settings.steps = 12;
                        state.gen_settings.cfg = 1.0;
                    } else {
                        state.gen_settings.turbo_enabled = false;
                        const previous = state.gen_settings.turbo_previous_settings || {};
                        if (previous.steps !== undefined) state.gen_settings.steps = previous.steps;
                        if (previous.cfg !== undefined) state.gen_settings.cfg = previous.cfg;
                        state.gen_settings.turbo_previous_settings = null;
                    }
                    syncGenerationControls();
                    saveState();
                };

                // 4. UI Builders
                const createField = (lbl, key, type = "text", opts = [], targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-field";

                    if (type === "checkbox") {
                        const toggleWrap = document.createElement("label");
                        toggleWrap.className = "vnccs-toggle-wrap";

                        const toggle = document.createElement("div");
                        toggle.className = "vnccs-toggle";

                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!targetObj[key];
                        inp.onchange = (e) => {
                            targetObj[key] = e.target.checked;
                            saveState();
                        };

                        const track = document.createElement("div");
                        track.className = "vnccs-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "vnccs-toggle-thumb";

                        toggle.appendChild(inp);
                        toggle.appendChild(track);
                        toggle.appendChild(thumb);

                        const l = document.createElement("span");
                        l.className = "vnccs-toggle-label";
                        l.innerText = lbl;

                        toggleWrap.appendChild(toggle);
                        toggleWrap.appendChild(l);

                        wrap.appendChild(toggleWrap);
                        els[key] = inp;
                        return wrap;
                    }

                    // Header Row: Label + Optional Tag Button
                    const header = document.createElement("div");
                    header.style.display = "flex";
                    header.style.alignItems = "center";
                    header.style.justifyContent = "space-between";
                    header.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    // Fields that support tag constructor
                    // hair, eyes, race, body, skin_color(maybe), face, details
                    const tagSupported = ["hair", "eyes", "race", "body", "face", "additional_details"].includes(key);
                    if (tagSupported && type === "text") {
                        const btn = document.createElement("div");
                        btn.className = "vnccs-tag-btn";
                        btn.innerHTML = "✎"; // Pencil or List icon
                        btn.title = "Open Tag Constructor";
                        btn.onclick = () => openTagConstructor(key, inp);
                        header.appendChild(btn);
                    }

                    wrap.appendChild(header);

                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-select";
                        opts.forEach(v => inp.add(new Option(v, v)));
                        inp.value = targetObj[key] || opts[0];
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
                        inp.type = "number";
                        if (opts.step) inp.step = opts.step;
                        inp.value = targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = parseFloat(e.target.value); saveState(); };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
                        inp.value = targetObj[key] || "";
                        inp.oninput = (e) => { targetObj[key] = e.target.value; debouncedSave(); };
                    }
                    els[key] = inp; // Register for updates
                    wrap.appendChild(inp);
                    return wrap;
                };

                const createSegmentedField = (lbl, key, options, targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-field";
                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;
                    const segmented = document.createElement("div");
                    segmented.className = "vnccs-segmented-field";
                    const buttons = [];
                    const setValue = (value, persist = false) => {
                        const normalized = String(value || options[0]?.value || "");
                        targetObj[key] = normalized;
                        buttons.forEach(({ btn, value: btnValue }) => {
                            btn.classList.toggle("is-active", btnValue === normalized);
                            btn.setAttribute("aria-pressed", btnValue === normalized ? "true" : "false");
                        });
                        if (persist) saveState();
                    };
                    options.forEach(option => {
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "vnccs-segmented-btn";
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

                const createGraphicToggle = (lbl, key, targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-field";
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "vnccs-graphic-toggle";
                    btn.innerHTML = `
                        <span class="vnccs-graphic-toggle-text">
                            <span class="vnccs-graphic-toggle-icon" aria-hidden="true">
                                <svg viewBox="0 0 24 24" width="15" height="15" fill="none">
                                    <path d="M12 3l7 4v5c0 4.5-2.8 7.4-7 9-4.2-1.6-7-4.5-7-9V7l7-4z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                                    <path d="M9 12h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </span>
                            ${lbl}
                        </span>
                        <span class="vnccs-graphic-toggle-switch" aria-hidden="true"></span>
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
                    wrap.className = "vnccs-field";
                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    const container = document.createElement("div");
                    container.className = "vnccs-slider-container";

                    const range = document.createElement("input");
                    range.type = "range"; range.className = "vnccs-slider";
                    range.min = min; range.max = max; range.step = step;
                    range.value = targetObj[key];

                    const num = document.createElement("input");
                    num.type = "number"; num.className = "vnccs-slider-val";
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
                    wrap.className = "vnccs-gen-param-field";
                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    const input = document.createElement("input");
                    input.type = "number";
                    input.className = "vnccs-gen-param-input";
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
                    wrap.className = "vnccs-gen-param-field";
                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    const select = document.createElement("select");
                    select.className = "vnccs-gen-param-select";
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
                    wrap.className = "vnccs-field";
                    wrap.innerHTML = `<div class="vnccs-label">${label}</div>`;
                    const select = document.createElement("select");
                    select.className = "vnccs-select";
                    select.onchange = (e) => {
                        targetObj[key] = e.target.value;
                        saveState();
                    };
                    wrap.appendChild(select);
                    els[key] = select;
                    return wrap;
                };

                const cardStatusLabel = (status, entry, cat) => {
                    const dls = ccDlStatus[ccStatusKey(cat, entry)] || {};
                    if (status === "installed") return "Installed";
                    if (status === "queued") return "Queued";
                    if (status === "downloading") return dls.message || "Downloading";
                    if (status === "auth_required") return "Key Required";
                    if (status === "error") return "Error";
                    return "Missing";
                };

                const buildAssetCard = ({ entry, cat, selectedValue, onSelect, compact = false, toggled = false, onToggle = null, pickerHead = false, onDownload = null }) => {
                    const status = ccResolveStatus(entry, cat);
                    const rel = ccRelPath(entry);
                    const installed = status === "installed";
                    const selected = selectedValue && rel && selectedValue.replace(/\\/g, "/") === rel;
                    const progress = ["queued", "downloading"].includes(status);

                    const card = document.createElement("div");
                    card.className = "vnccs-model-card";
                    card.classList.toggle("is-picker-head", pickerHead);
                    card.classList.toggle("is-installed", installed);
                    card.classList.toggle("is-selected", selected || toggled);
                    card.classList.toggle("is-missing", !installed);
                    if (installed || pickerHead) {
                        card.onclick = () => onSelect?.(rel, entry);
                    }

                    const top = document.createElement("div");
                    top.className = "vnccs-model-card-top";

                    const badge = document.createElement("span");
                    badge.className = "vnccs-model-card-badge " + (installed ? "ok" : progress ? "progress" : "missing");
                    top.appendChild(badge);

                    const name = document.createElement("div");
                    name.className = "vnccs-model-card-name";
                    name.textContent = entry.name || rel || "Unknown";
                    top.appendChild(name);

                    const statusEl = document.createElement("div");
                    statusEl.className = "vnccs-model-card-status " + (installed ? "ok" : progress ? "progress" : "missing");
                    statusEl.textContent = cardStatusLabel(status, entry, cat);
                    top.appendChild(statusEl);

                    if (onToggle && installed) {
                        const toggle = document.createElement("label");
                        toggle.className = "vnccs-toggle vnccs-model-card-toggle";
                        const input = document.createElement("input");
                        input.type = "checkbox";
                        input.checked = !!toggled;
                        input.onchange = (event) => {
                            event.stopPropagation();
                            onToggle(event.target.checked, rel, entry);
                        };
                        input.onclick = (event) => event.stopPropagation();
                        const track = document.createElement("div");
                        track.className = "vnccs-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "vnccs-toggle-thumb";
                        toggle.append(input, track, thumb);
                        top.appendChild(toggle);
                    }

                    card.appendChild(top);

                    if (entry.description && !compact) {
                        const desc = document.createElement("div");
                        desc.className = "vnccs-model-card-desc";
                        desc.textContent = entry.description;
                        card.appendChild(desc);
                    }

                    if (!installed) {
                        const actions = document.createElement("div");
                        actions.className = "vnccs-model-card-actions";
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "vnccs-model-card-download";
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

                const selectAnimaModel = (rel) => {
                    ensureAnimaDefaultAux();
                    modelPickerOpen.anima = false;
                    selectCcAsset("diffusion_model_name", rel);
                };

                const selectIllustriousModel = (rel) => {
                    modelPickerOpen.illustrious = false;
                    selectCcAsset("ckpt_name", rel);
                };

                const downloadAnimaBundle = async (cat, entry) => {
                    ensureAnimaDefaultAux();
                    await ccDownloadEntry(cat, entry);
                    const clip = ccFirstEntry("clip", "Anima");
                    const vae = ccFirstEntry("vae", "Anima");
                    if (clip && ccResolveStatus(clip, "clip") !== "installed") await ccDownloadEntry("clip", clip);
                    if (vae && ccResolveStatus(vae, "vae") !== "installed") await ccDownloadEntry("vae", vae);
                };

                const setCcTurboMode = (enabled, rel) => {
                    if ((state.gen_settings.generation_mode || "illustrious").toLowerCase() === "anima") {
                        state.gen_settings.dmd_lora_name = rel || state.gen_settings.dmd_lora_name || "";
                        setAnimaTurboMode(enabled);
                    } else {
                        state.gen_settings.dmd_lora_name = enabled ? rel : "";
                        state.gen_settings.dmd_lora_strength = enabled ? 1.0 : 0.0;
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
                        return kind === "sdxl" || kind === "illustrious";
                    };
                    const turboEntries = (ccConfig?.lora || []).filter(entry => kindOk(entry) && ccType(entry) === "turbolora");
                    const ageEntries = (ccConfig?.lora || []).filter(entry => kindOk(entry) && ccType(entry) === "ageslider");

                    const addGroup = (title, entries, renderer) => {
                        if (!entries.length) return;
                        const group = document.createElement("div");
                        group.className = "vnccs-subsection";
                        const label = document.createElement("div");
                        label.className = "vnccs-label";
                        label.textContent = title;
                        group.appendChild(label);
                        entries.forEach(entry => group.appendChild(renderer(entry)));
                        containerEl.appendChild(group);
                    };

                    addGroup("Turbo LoRA", turboEntries, entry => {
                        const rel = ccRelPath(entry);
                        const enabled = mode === "anima"
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

                const renderCardSection = (containerEl, entries, cat, key, emptyText) => {
                    if (!containerEl) return;
                    containerEl.innerHTML = "";
                    if (!entries.length) {
                        const empty = document.createElement("div");
                        empty.className = "vnccs-model-card-desc";
                        empty.textContent = emptyText;
                        containerEl.appendChild(empty);
                        return;
                    }
                    entries.forEach(entry => {
                        containerEl.appendChild(buildAssetCard({
                            entry,
                            cat,
                            selectedValue: state.gen_settings[key] || "",
                            onSelect: rel => selectCcAsset(key, rel),
                        }));
                    });
                };

                const renderModelPicker = ({ containerEl, entries, cat, key, mode, emptyText, onSelect, onDownload = null }) => {
                    if (!containerEl) return;
                    containerEl.innerHTML = "";
                    const picker = document.createElement("div");
                    picker.className = "vnccs-model-picker";
                    picker.classList.toggle("is-open", !!modelPickerOpen[mode]);
                    containerEl.appendChild(picker);

                    if (!entries.length) {
                        const empty = document.createElement("div");
                        empty.className = "vnccs-model-card-desc";
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
                    menu.className = "vnccs-model-picker-menu";
                    picker.appendChild(menu);

                    const appendGroup = (title, groupEntries) => {
                        if (!groupEntries.length) return;
                        const group = document.createElement("div");
                        group.className = "vnccs-model-picker-group";
                        const groupTitle = document.createElement("div");
                        groupTitle.className = "vnccs-model-picker-group-title";
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
                    if (!els.animaModelCards && !els.illustriousModelCards) return;
                    const currentMode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    const isAnimaMode = currentMode === "anima";
                    if (els.animaFallback) {
                        els.animaFallback.style.display = "none";
                    }
                    [els.animaModelCards].forEach(containerEl => {
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
                        if (!current && firstEntry && ccConfig) {
                            selectAnimaModel(ccRelPath(firstEntry));
                        }
                    });
                    ensureAnimaDefaultAux();

                    if (els.animaLoraCards) {
                        if (!isAnimaMode) {
                            els.animaLoraCards.innerHTML = "";
                            els.animaLoraCards.style.display = "none";
                        } else {
                            renderModeLoraCards(els.animaLoraCards, "anima");
                        }
                    }
                    if (els.illustriousLoraCards) {
                        if (isAnimaMode) {
                            els.illustriousLoraCards.innerHTML = "";
                            els.illustriousLoraCards.style.display = "none";
                        } else {
                            renderModeLoraCards(els.illustriousLoraCards, "illustrious");
                        }
                    }

                    const illustriousDefaults = (ccConfig?.models || []).filter(entry => {
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
                            if (!state.gen_settings.ckpt_name && firstEntry && ccConfig) {
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
                            empty.className = "vnccs-model-card-desc";
                            empty.textContent = "No checkpoints found.";
                            els.illustriousModelCards.appendChild(empty);
                        }
                    }
                };

                const isSelectedCcAssetInstalled = (section, kind, key, predicate = null) => {
                    const selected = String(state.gen_settings[key] || "").replace(/\\/g, "/");
                    if (!selected) return false;
                    if (!ccConfig) return true;
                    const entries = ccEntries(section, kind, predicate);
                    if (!entries.length) return true;
                    const match = entries.find(entry => ccRelPath(entry) === selected);
                    if (!match) return localAssetHas(key, selected);
                    return ccResolveStatus(match, section === "models" ? "models" : section) === "installed";
                };

                // 5. Build Layout
                const container = document.createElement("div");
                container.className = "vnccs-container";
                enableMiddleMouseCanvasPan(container);

                // --- TOP ROW ---
                const topRow = document.createElement("div");
                topRow.className = "vnccs-top-row";

                // COL 1: LEFT (Preview)
                const colLeft = document.createElement("div");
                colLeft.className = "vnccs-col";
                colLeft.innerHTML = '<div class="vnccs-section-title">Character Select</div>';

                const charSel = document.createElement("select"); charSel.className = "vnccs-select";
                charSel.onchange = async (e) => {
                    state.character = e.target.value;
                    restoredWidgetInfoCharacter = null;
                    await loadChar(state.character);
                    saveState(true);
                };
                els.charSelect = charSel;
                colLeft.appendChild(charSel);

                const btnRow = document.createElement("div");
                btnRow.className = "vnccs-btn-row";
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-btn vnccs-btn-primary vnccs-btn-generate";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.onclick = () => doGenerate();
                els.btnGen = btnGen;

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-btn vnccs-btn-success vnccs-btn-secondary";
                btnNew.innerText = "NEW";
                btnNew.onclick = () => doCreate();

                const btnDel = document.createElement("button");
                btnDel.className = "vnccs-btn vnccs-btn-danger vnccs-btn-secondary";
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
                        if (els[key]) els[key].value = value;
                    });
                    state.preview_valid = false;
                    saveState(false);
                };

                const showCharacterWizardError = (err) => {
                    showModal("Character Wizzard Error", () => {
                        const d = document.createElement("div");
                        d.className = "vnccs-character-wizard-modal";
                        const text = document.createElement("div");
                        text.className = "vnccs-character-wizard-modal-text";
                        text.innerText = err?.message || err?.raw || "Failed to generate character description.";
                        d.appendChild(text);
                        return d;
                    }, [{ text: "OK", class: "vnccs-btn-primary" }]);
                };

                const openCharacterWizard = () => {
                    let input;
                    showModal("Character Wizzard", () => {
                        const wrap = document.createElement("div");
                        wrap.className = "vnccs-character-wizard-modal";
                        const text = document.createElement("div");
                        text.className = "vnccs-character-wizard-modal-text";
                        text.innerText = "Describe the character in a broad way. The model will expand it into the creator fields and prefer tags from the tag constructor.";
                        input = document.createElement("textarea");
                        input.className = "vnccs-textarea";
                        input.placeholder = "e.g. adult demon girl with long white hair, red eyes, elegant sharp face";
                        wrap.append(text, input);
                        setTimeout(() => input.focus(), 50);
                        return wrap;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "FILL FIELDS",
                            class: "vnccs-btn-primary",
                            action: async (_overlay, btn) => {
                                const description = input.value.trim();
                                if (!description) {
                                    input.focus();
                                    return true;
                                }
                                btn.disabled = true;
                                btn.innerText = "THINKING...";
                                try {
                                    const r = await api.fetchApi("/vnccs/character_wizard", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ description })
                                    });
                                    if (!r.ok) {
                                        let err = null;
                                        try { err = await r.json(); } catch (e) { err = { message: await r.text() }; }
                                        showCharacterWizardError(err);
                                        return false;
                                    }
                                    const data = await r.json();
                                    applyCharacterWizardData(data);
                                    return false;
                                } catch (e) {
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
                        inp.className = "vnccs-input";
                        inp.placeholder = "Name...";
                        inpRef = inp;
                        return inp;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "Create",
                            class: "vnccs-btn-primary",
                            action: async (ol, btn) => {
                                const n = inpRef.value.trim();
                                if (!n) return true; // Keep open
                                try {
                                    await api.fetchApi(`/vnccs/create?name=${encodeURIComponent(n)}`);
                                    const exists = Array.from(els.charSelect.options).some(o => o.value === n);
                                    if (!exists) els.charSelect.add(new Option(n, n));

                                    state.character = n; // Updates internal state immediately
                                    els.charSelect.value = n; // Update UI
                                    await loadChar(n);
                                    saveState();
                                    return false; // Close
                                } catch (e) {
                                    alert("Create Failed: " + e);
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
                        alert("Please select a character to delete.");
                        return;
                    }

                    showModal("Delete Character", () => {
                        const div = document.createElement("div");
                        div.innerHTML = `
                                <div style="font-size:14px; text-align:center;">
                                    Are you sure you want to <b>PERMANENTLY DELETE</b><br/>
                                    <span style="color:#fff; font-weight:bold;">'${charName}'</span>?<br/><br/>
                                    <span style="font-size:12px; color:#aaa;">This action cannot be undone.</span>
                                </div>
                            `;
                        return div;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "CONFIRM DELETE",
                            class: "vnccs-btn-danger",
                            action: async (ol, btn) => {
                                try {
                                    btn.innerText = "DELETING...";
                                    btn.disabled = true;
                                    const r = await api.fetchApi(`/vnccs/delete?name=${encodeURIComponent(charName)}`);
                                    if (r.ok) {
                                        const opts = Array.from(els.charSelect.options);
                                        const idx = opts.findIndex(o => o.value === charName);
                                        if (idx > -1) els.charSelect.remove(idx);

                                        if (els.charSelect.options.length > 0) state.character = els.charSelect.options[0].value;
                                        else state.character = "";

                                        els.charSelect.value = state.character;
                                        await loadChar(state.character);
                                        saveState();
                                        return false;
                                    } else {
                                        const err = await r.json();
                                        alert("Delete Failed: " + (err.error || "Unknown Error"));
                                        btn.innerText = "CONFIRM DELETE";
                                        btn.disabled = false;
                                        return true;
                                    }
                                } catch (e) {
                                    alert("Delete Failed: " + e);
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
                            const r = await api.fetchApi("/vnccs/get_tags");
                            if (r.ok) TAG_DATA = await r.json();
                            else throw new Error("Failed to load tags");
                        } catch (e) {
                            alert("Error loading tag database: " + e);
                            return;
                        }
                    }

                    // 2. Determine Categories
                    // Mapping: widget_key -> [json_keys...]
                    const map = {
                        "hair": ["hair_color", "hairstyles"],
                        "eyes": ["eyes"], // special handling for nested
                        "face": ["eyes"], // special handling (nested under eyes in user update)
                        "race": ["races"],
                        "body": ["breast_size"],
                        "additional_details": ["details"]
                    };

                    const categories = map[fieldKey] || [];
                    if (!categories.length) return;

                    // 3. Collect Tags
                    const allTags = [];
                    categories.forEach(cat => {
                        let data = TAG_DATA.tags[cat];
                        if (cat === "eyes") {
                            if (fieldKey === "eyes") {
                                // Flatten eyes
                                if (TAG_DATA.tags.eyes.colors) allTags.push({ header: "Eye Colors", items: TAG_DATA.tags.eyes.colors });
                                if (TAG_DATA.tags.eyes.features) allTags.push({ header: "Eye Features", items: TAG_DATA.tags.eyes.features });
                            } else if (fieldKey === "face") {
                                // Face Characteristics
                                if (TAG_DATA.tags.eyes.face_characteristics) allTags.push({ header: "Face Characteristics", items: TAG_DATA.tags.eyes.face_characteristics });
                            }
                        } else if (data) {
                            // Arrays
                            // format header nicel
                            const h = cat.replace("_", " ").toUpperCase();
                            allTags.push({ header: h, items: data });
                        }
                    });

                    if (!allTags.length) {
                        alert("No tags found for this category.");
                        return;
                    }

                    // 4. Modal
                    // Parse current values to pre-select
                    const currentVals = inputEl.value.split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
                    const selected = new Set(currentVals);

                    showModal(`Tag Constructor: ${fieldKey}`, (modal) => {
                        const container = document.createElement("div");
                        container.className = "vnccs-tag-grid";

                        // Fix width for tag modal
                        modal.style.width = "500px";

                        allTags.forEach(group => {
                            if (group.header) {
                                const h = document.createElement("div");
                                h.className = "vnccs-tag-category";
                                h.style.width = "100%";
                                h.innerText = group.header;
                                container.appendChild(h);
                            }

                            group.items.forEach(item => {
                                const t = item.tag;
                                const chip = document.createElement("div");
                                chip.className = "vnccs-tag-chip";
                                chip.innerText = item.label || t;
                                const normTag = t.replace(/_/g, " ");
                                if (selected.has(normTag)) chip.classList.add("selected");

                                chip.onclick = () => {
                                    const useTag = t.replace(/_/g, " ");
                                    if (selected.has(useTag)) {
                                        selected.delete(useTag);
                                        chip.classList.remove("selected");
                                    } else {
                                        selected.add(useTag);
                                        chip.classList.add("selected");
                                    }
                                };
                                container.appendChild(chip);
                            });
                        });

                        return container;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "APPLY",
                            class: "vnccs-btn-primary",
                            action: () => {
                                const final = Array.from(selected).join(", ");
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
                frame.className = "vnccs-preview-container";
                frame.innerHTML = `<div class="vnccs-placeholder">
                    <svg class="vnccs-placeholder-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="24" cy="18" r="8" stroke="currentColor" stroke-width="2"/>
                        <path d="M8 42c0-8.837 7.163-16 16-16s16 7.163 16 16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    No Preview
                </div>`;
                const img = document.createElement("img");
                img.className = "vnccs-preview-img"; img.style.display = "none";
                frame.appendChild(img);
                els.previewImg = img; els.placeholder = frame.querySelector(".vnccs-placeholder");
                colLeft.appendChild(frame);

                topRow.appendChild(colLeft);

                // COL 2: CENTER (Attributes)
                const colCenter = document.createElement("div");
                colCenter.className = "vnccs-col";
                colCenter.innerHTML = '<div class="vnccs-section-title">Attributes</div>';

                const characterWizardBtn = document.createElement("button");
                characterWizardBtn.type = "button";
                characterWizardBtn.className = "vnccs-btn vnccs-btn-primary vnccs-character-wizard-btn";
                characterWizardBtn.innerText = "CHARACTER WIZZARD";
                characterWizardBtn.onclick = openCharacterWizard;
                colCenter.appendChild(characterWizardBtn);

                colCenter.appendChild(createSegmentedField("Background", "background_color", [
                    { label: "Green", value: "Green" },
                    { label: "Blue", value: "Blue" },
                ]));
                colCenter.appendChild(createSegmentedField("Gender", "sex", [
                    { label: "Male", value: "male" },
                    { label: "Female", value: "female" },
                ]));
                colCenter.appendChild(createSlider("Age", "age", 1, 100, 1, state.character_info));
                colCenter.appendChild(createField("Race", "race"));
                colCenter.appendChild(createField("Skin Color", "skin_color"));
                colCenter.appendChild(createField("Body Type", "body"));
                colCenter.appendChild(createField("Face Features", "face"));
                colCenter.appendChild(createField("Hair Style", "hair"));
                colCenter.appendChild(createField("Eye Color", "eyes"));
                colCenter.appendChild(createField("Details", "additional_details"));
                colCenter.appendChild(createGraphicToggle("NSFW Mode", "nsfw"));

                topRow.appendChild(colCenter);

                // COL 3: RIGHT (Generation)
                const colRight = document.createElement("div");
                colRight.className = "vnccs-col";
                colRight.innerHTML = '<div class="vnccs-section-title">Generation</div>';

                const tabRow = document.createElement("div");
                tabRow.className = "vnccs-tab-row";
                els.modeTabs = {};
                [
                    ["illustrious", "Illustrious"],
                    ["anima", "ANIMA"],
                ].forEach(([value, label]) => {
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "vnccs-tab";
                    btn.innerText = label;
                    btn.onclick = () => setGenerationMode(value);
                    els.modeTabs[value] = btn;
                    tabRow.appendChild(btn);
                });
                colRight.appendChild(tabRow);

                const illustriousModels = document.createElement("div");
                illustriousModels.className = "vnccs-subsection";
                els.illustriousModels = illustriousModels;

                const illustriousCards = document.createElement("div");
                illustriousCards.className = "vnccs-model-card-list";
                els.illustriousModelCards = illustriousCards;
                illustriousModels.appendChild(illustriousCards);

                const illustriousFallback = document.createElement("div");
                illustriousFallback.className = "vnccs-generation-fallback";
                illustriousFallback.style.display = "none";
                els.illustriousFallback = illustriousFallback;
                const wrapCkpt = makeFallbackSelect("Checkpoint (SDXL)", "ckpt_name");
                els.ckptSelect = wrapCkpt.querySelector("select");
                illustriousFallback.appendChild(wrapCkpt);
                illustriousModels.appendChild(illustriousFallback);
                colRight.appendChild(illustriousModels);

                const animaModels = document.createElement("div");
                animaModels.className = "vnccs-subsection";
                els.animaModels = animaModels;

                const createAnimaCardField = (slotName) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-field";
                    const cards = document.createElement("div");
                    cards.className = "vnccs-model-card-list";
                    wrap.appendChild(cards);
                    els[slotName] = cards;
                    animaModels.appendChild(wrap);
                };

                createAnimaCardField("animaModelCards");

                const hiddenAnimaSelects = document.createElement("div");
                hiddenAnimaSelects.className = "vnccs-generation-fallback";
                hiddenAnimaSelects.style.display = "none";
                els.animaFallback = hiddenAnimaSelects;
                hiddenAnimaSelects.appendChild(makeFallbackSelect("Diffusion Model", "diffusion_model_name"));
                hiddenAnimaSelects.appendChild(makeFallbackSelect("CLIP", "clip_name"));
                hiddenAnimaSelects.appendChild(makeFallbackSelect("VAE", "vae_name"));
                animaModels.appendChild(hiddenAnimaSelects);
                colRight.appendChild(animaModels);

                const genParamGrid = document.createElement("div");
                genParamGrid.className = "vnccs-gen-param-grid";
                genParamGrid.appendChild(createCompactNumberField("Steps", "steps", 1, 100, 1));
                genParamGrid.appendChild(createCompactSelectField("Sampler", "sampler", state.gen_settings));
                genParamGrid.appendChild(createCompactNumberField("CFG", "cfg", 1, 20, 0.1));
                genParamGrid.appendChild(createCompactSelectField("Scheduler", "scheduler", state.gen_settings));
                colRight.appendChild(genParamGrid);

                // SEED Section (Rebalanced)
                const seedWrap = document.createElement("div"); seedWrap.className = "vnccs-field";
                seedWrap.innerHTML = '<div class="vnccs-label">Seed</div>';

                const seedRow = document.createElement("div");
                seedRow.className = "vnccs-seed-row";

                const seedInp = document.createElement("input"); seedInp.className = "vnccs-gen-param-input";
                seedInp.type = "number"; seedInp.value = state.gen_settings.seed;
                seedInp.onchange = (e) => {
                    state.gen_settings.seed = parseInt(e.target.value);
                    saveState();
                };
                els.seed = seedInp;

                const seedMode = document.createElement("button");
                seedMode.type = "button";
                seedMode.className = "vnccs-seed-dice-btn";
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

                // --- LoRA Section ---
                const loraSection = document.createElement("div");
                loraSection.className = "vnccs-subsection";
                els.loraSection = loraSection;

                const loraHeader = document.createElement("div");
                loraHeader.className = "vnccs-section-title";
                loraHeader.style.marginTop = "10px";
                loraHeader.innerText = "LoRa Stack";
                els.loraHeader = loraHeader;
                loraSection.appendChild(loraHeader);

                // DMD2 LoRA
                const dmdWrap = document.createElement("div"); dmdWrap.className = "vnccs-lora-item";
                els.dmdWrap = dmdWrap;
                const dmdLabel = document.createElement("div");
                dmdLabel.className = "vnccs-label";
                dmdLabel.innerText = "DMD2 LoRA Model";
                els.dmdLabel = dmdLabel;
                dmdWrap.appendChild(dmdLabel);
                const dmdRow = document.createElement("div"); dmdRow.className = "vnccs-lora-row";
                const dmdSel = document.createElement("select"); dmdSel.className = "vnccs-select";
                dmdSel.style.flex = "2";
                dmdSel.onchange = (e) => { state.gen_settings.dmd_lora_name = e.target.value; saveState(); };
                els.dmdSelect = dmdSel;

                const dmdToggleWrap = document.createElement("label");
                dmdToggleWrap.className = "vnccs-toggle";
                dmdToggleWrap.style.flexShrink = "0";
                dmdToggleWrap.style.margin = "0 4px";

                const dmdStr = document.createElement("input");
                dmdStr.type = "checkbox";
                dmdStr.checked = (state.gen_settings.dmd_lora_strength || 0) > 0;
                dmdStr.onchange = (e) => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (mode === "anima") {
                        setAnimaTurboMode(e.target.checked);
                    } else {
                        state.gen_settings.dmd_lora_strength = e.target.checked ? 1.0 : 0.0;
                        saveState();
                    }
                };
                const dmdTrack = document.createElement("div"); dmdTrack.className = "vnccs-toggle-track";
                const dmdThumb = document.createElement("div"); dmdThumb.className = "vnccs-toggle-thumb";
                dmdToggleWrap.appendChild(dmdStr);
                dmdToggleWrap.appendChild(dmdTrack);
                dmdToggleWrap.appendChild(dmdThumb);

                dmdRow.appendChild(dmdSel); dmdRow.appendChild(dmdToggleWrap);
                dmdWrap.appendChild(dmdRow);
                loraSection.appendChild(dmdWrap);
                els.dmdSlider = dmdStr; // Renamed ref for logic compat, though it's an input now

                const animaLoraCards = document.createElement("div");
                animaLoraCards.className = "vnccs-subsection";
                animaLoraCards.style.display = "none";
                els.animaLoraCards = animaLoraCards;
                loraSection.appendChild(animaLoraCards);

                const illustriousLoraCards = document.createElement("div");
                illustriousLoraCards.className = "vnccs-subsection";
                illustriousLoraCards.style.display = "none";
                els.illustriousLoraCards = illustriousLoraCards;
                loraSection.appendChild(illustriousLoraCards);

                // Age LoRA
                const ageWrap = document.createElement("div"); ageWrap.className = "vnccs-lora-item";
                els.ageWrap = ageWrap;
                ageWrap.innerHTML = '<div class="vnccs-label">Age LoRA (Auto Strength)</div>';
                const ageSel = document.createElement("select"); ageSel.className = "vnccs-select";
                ageSel.onchange = (e) => { state.gen_settings.age_lora_name = e.target.value; saveState(); };
                els.ageSelect = ageSel;
                ageWrap.appendChild(ageSel);
                loraSection.appendChild(ageWrap);

                // Stack (5 Slots)
                const stackContainer = document.createElement("div");
                stackContainer.className = "vnccs-lora-stack";
                els.loraStackSelects = [];

                for (let i = 0; i < 5; i++) {
                    const item = document.createElement("div"); item.className = "vnccs-lora-item";
                    const row = document.createElement("div"); row.className = "vnccs-lora-row";

                    const sel = document.createElement("select"); sel.className = "vnccs-select";
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

                    const rng = document.createElement("input"); rng.className = "vnccs-input";
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
                bottomRow.className = "vnccs-bottom-row";

                const createText = (lbl, key) => {
                    const w = document.createElement("div"); w.className = "vnccs-textarea-wrapper";
                    w.innerHTML = `<div class="vnccs-textarea-label">${lbl}</div>`;
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
                            clearPreviewHandlers();
                            els.previewImg.src = `/vnccs/get_cached_preview?character=${encodeURIComponent(charName)}&t=${Date.now()}`;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";
                            state.preview_valid = true;
                            state.preview_source = "gen";
                            saveState(true);
                        }
                    }
                };
                api.addEventListener("vnccs.preview.updated", previewUpdateHandler);
                registerCleanup(node, () => api.removeEventListener("vnccs.preview.updated", previewUpdateHandler));
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
                        };
                        renderControlCenterCards();

                        // Populate LoRA selectors
                        const loras = d.loras || [];
                        pop(els.dmdSelect, loras, true);
                        pop(els.ageSelect, loras, true);
                        els.loraStackSelects.forEach(o => pop(o.sel, loras, true));

                        fetchCcConfig(true).catch((error) => {
                            console.warn("[VNCCS V2] Control Center config unavailable:", error);
                            renderControlCenterCards();
                        });

                        els.charSelect.innerHTML = "";
                        if (!d.characters || !d.characters.length) els.charSelect.add(new Option("None", ""));
                        else d.characters.forEach(c => els.charSelect.add(new Option(c, c)));

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
                        applyGenerationProfile(g.generation_mode);
                        migratePromptModes();

                        // Auto-generate seed if still 0 (first use or unset)
                        if ((!g.seed || g.seed === 0) && g.seed_mode !== "randomize") {
                            g.seed = generateRandomSeed();
                        }
                        saveCurrentGenerationModeValues();
                        syncGenerationControls();
                        refreshGenerationModeUI();

                        if (state.character) {
                            // Ensure it's in the list (handle potential race/latency)
                            const exists = Array.from(els.charSelect.options).some(o => o.value === state.character);
                            if (!exists) {
                                els.charSelect.add(new Option(state.character, state.character));
                            }
                            els.charSelect.value = state.character;
                        }
                        else if (d.characters && d.characters.length) {
                            state.character = d.characters[0];
                            els.charSelect.value = state.character;
                        }

                        // Only skip disk load when widget_data explicitly belongs to this character.
                        // Default state also has hair/eyes fields, so field presence is not proof of valid restored data.
                        const hasWidgetData = restoredWidgetInfoCharacter && restoredWidgetInfoCharacter === state.character;
                        if (state.character) await loadChar(state.character, hasWidgetData);

                        // Sync widget state. Preview validity is set by image onload handler, not preemptively.
                        saveState();

                    } catch (e) { console.error(e); }
                };

                const loadChar = async (n, skipInfoLoad = false) => {
                    if (!n) return;
                    try {
                        // 1. Fetch Info (skip if restoring from widget_data)
                        if (!skipInfoLoad) {
                            const r = await api.fetchApi(`/vnccs/character_info?character=${encodeURIComponent(n)}`);
                            const i = await r.json();

                            const defaultInfo = {
                                sex: "female", age: 18, race: "human", skin_color: "",
                                hair: "black hair, long hair", eyes: "", face: "", body: "", additional_details: "",
                                nsfw: false, aesthetics: "masterpiece, best quality",
                                negative_prompt: "bad quality, worst quality",
                                lora_prompt: "", background_color: "Green"
                            };

                            // Reset & Assign
                            Object.assign(state.character_info, defaultInfo);
                            Object.assign(state.character_info, i);
                            state.prompt_modes = {
                                illustrious: {
                                    aesthetics: state.character_info.aesthetics || MODE_PROMPT_DEFAULTS.illustrious.aesthetics,
                                    negative_prompt: state.character_info.negative_prompt || MODE_PROMPT_DEFAULTS.illustrious.negative_prompt,
                                },
                                anima: {
                                    aesthetics: MODE_PROMPT_DEFAULTS.anima.aesthetics,
                                    negative_prompt: MODE_PROMPT_DEFAULTS.anima.negative_prompt,
                                },
                            };
                            state.prompt_defaults_version = PROMPT_DEFAULTS_VERSION;
                            applyPromptModeToFields((state.gen_settings.generation_mode || "illustrious").toLowerCase());
                        }

                        // Update Fields from current state
                        Object.keys(state.character_info).forEach(k => {
                            if (els[k]) {
                                let val = state.character_info[k];
                                // Normalize background_color case to match dropdown options
                                if (k === "background_color" && typeof val === "string" && val) {
                                    val = val.charAt(0).toUpperCase() + val.slice(1).toLowerCase();
                                    state.character_info[k] = val; // Update state too
                                }
                                if (els[k].range && els[k].num) {
                                    // Slider Composite
                                    els[k].range.value = val;
                                    els[k].num.value = val;
                                }
                                else if (els[k].setValue) els[k].setValue(val);
                                else if (els[k].type === "checkbox") els[k].checked = !!val;
                                else els[k].value = val;
                            }
                        });

                        // 2. Fetch Preview Image
                        // Strategy: Try Sheet -> Fail -> Try Cache -> Fail -> Placeholder

                        // Function to try loading cache
                        const tryCache = () => {
                            console.log("[VNCCS] Trying to load cached preview...");
                            const cacheUrl = `/vnccs/get_cached_preview?character=${encodeURIComponent(n)}&t=${Date.now()}`;

                            els.previewImg.onerror = () => {
                                console.warn("[VNCCS] Both pose and cache preview failed.");
                                els.previewImg.style.display = "none";
                                els.placeholder.innerText = "No Preview Image";
                                els.placeholder.style.display = "block";
                                els.previewImg.onerror = null;
                                state.preview_valid = false;
                                state.preview_source = "gen";
                                saveState(false);
                            };
                            els.previewImg.onload = () => {
                                state.preview_valid = true;
                                state.preview_source = "gen";
                                saveState(true);
                            };

                            els.previewImg.src = cacheUrl;
                        };

                        // First try a random saved pose sprite.
                        els.previewImg.onerror = () => {
                            console.warn("[VNCCS] Pose preview load failed. Fallback to cache.");
                            tryCache();
                        };
                        els.previewImg.onload = () => {
                            state.preview_valid = true;
                            state.preview_source = "pose";
                            saveState(true);
                        };

                        els.previewImg.src = `/vnccs/get_character_pose_preview?character=${encodeURIComponent(n)}&t=${Date.now()}`;
                        els.previewImg.style.display = "block";
                        els.placeholder.style.display = "none";

                    } catch (e) { console.error(e); }
                };

                const doGenerate = async () => {
                    const mode = (state.gen_settings.generation_mode || "illustrious").toLowerCase();
                    if (mode === "anima") {
                        if (!state.gen_settings.diffusion_model_name) { alert("Select Diffusion Model"); return; }
                        if (!state.gen_settings.clip_name) { alert("Select CLIP"); return; }
                        if (!state.gen_settings.vae_name) { alert("Select VAE"); return; }
                        if (!isSelectedCcAssetInstalled("models", "Anima", "diffusion_model_name", entry => ccType(entry) === "unet")) {
                            alert("Download and select an installed Anima Diffusion Model");
                            return;
                        }
                        if (!isSelectedCcAssetInstalled("clip", "Anima", "clip_name")) {
                            alert("Download and select an installed Anima CLIP");
                            return;
                        }
                        if (!isSelectedCcAssetInstalled("vae", "Anima", "vae_name")) {
                            alert("Download and select an installed Anima VAE");
                            return;
                        }
                    } else if (!state.gen_settings.ckpt_name) {
                        alert("Select Checkpoint"); return;
                    } else if (!isSelectedCcAssetInstalled("models", null, "ckpt_name", entry => {
                        const kind = ccKind(entry);
                        return (kind === "illustrious" || kind === "sdxl") && ccType(entry) === "checkpoint";
                    })) {
                        alert("Download and select an installed Illustrious checkpoint");
                        return;
                    }
                    if (els.btnGen.disabled) return;

                    node._randomizeSeedIfNeeded();
                    saveCurrentGenerationModeValues();

                    // Show loading overlay
                    const loading = createLoadingOverlay(container, "Generating preview");

                    els.btnGen.innerText = "GENERATING...";
                    els.btnGen.disabled = true;
                    saveState();

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
                        if (d.image) {
                            clearPreviewHandlers();
                            els.previewImg.src = "data:image/png;base64," + d.image;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";
                            // Successful generation -> Cache is Valid AND source is 'gen'
                            state.preview_source = "gen";
                            saveState(true);
                        }
                    } catch (e) { showMessage(container, "Error: " + e, true); }
                    finally {
                        loading.remove();
                        els.btnGen.innerText = "GENERATE PREVIEW";
                        els.btnGen.disabled = false;
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
