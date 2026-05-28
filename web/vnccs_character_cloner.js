import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { showModal as showCommonModal, createLoadingOverlay, injectStyles, syncDOMWidgetWidth, syncDOMWidgetWidthSoon } from "./vnccs_common.js";

// --- STYLES: Sakura Archive Design System ---
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Variables ── */
.vnccs-cloner-container {
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
    --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-elevated: 0 8px 32px rgba(0, 0, 0, 0.5);
    --transition: 0.2s ease;
}

/* ── Container ── */
.vnccs-cloner-container {
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

/* ── Rows ── */
.vnccs-cloner-top-row {
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 12px;
    flex: 1;
    min-height: 0;
    width: 100%;
}

.vnccs-cloner-bottom-row {
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 12px;
    height: 80px;
    min-height: 80px;
    width: 100%;
    flex-shrink: 0;
}

/* ── Columns ── */
.vnccs-cloner-col {
    display: flex;
    flex-direction: column;
    background: rgba(20, 16, 30, 0.88);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 14px 16px;
    gap: 10px;
    overflow-y: auto;
    height: 100%;
    box-sizing: border-box;
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    pointer-events: auto;
}
.vnccs-cloner-col::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
    border-radius: 1px;
}
.vnccs-cloner-col::-webkit-scrollbar { width: 4px; }
.vnccs-cloner-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* ── Section Titles ── */
.vnccs-cloner-section-title {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    pointer-events: auto;
}
.vnccs-cloner-section-title::before {
    content: '';
    width: 3px;
    height: 12px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
}

/* ── Fields ── */
.vnccs-cloner-field {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 6px;
    flex-shrink: 0;
}
.vnccs-cloner-label {
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Inputs ── */
.vnccs-cloner-input, .vnccs-cloner-textarea {
    background: rgba(255, 255, 255, 0.04);
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
.vnccs-cloner-select {
    background: rgba(255, 255, 255, 0.04);
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
.vnccs-cloner-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}
.vnccs-cloner-input:focus, .vnccs-cloner-select:focus, .vnccs-cloner-textarea:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255, 143, 163, 0.04);
    box-shadow: 0 0 0 3px rgba(255, 143, 163, 0.06);
}

/* ── Slider ── */
.vnccs-cloner-slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 6px 10px;
}
.vnccs-cloner-slider {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    height: 3px;
}
.vnccs-cloner-slider-val {
    width: 42px;
    text-align: right;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-primary);
    background: transparent;
    border: none;
}
.vnccs-cloner-slider-val:focus {
    outline: none;
    border-bottom: 1px solid var(--accent-border);
}

/* ── Buttons ── */
.vnccs-cloner-btn-row {
    display: flex;
    gap: 8px;
    margin-top: auto;
    flex-shrink: 0;
}
.vnccs-cloner-btn {
    flex: 1;
    padding: 10px;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-family: var(--font);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11px;
    color: white;
    text-align: center;
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
}
.vnccs-cloner-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: #1a1525;
    box-shadow: 0 4px 16px rgba(255, 143, 163, 0.25);
}
.vnccs-cloner-btn-primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: clonerBtnShimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes clonerBtnShimmer {
    0%   { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35%  { transform: translateX(120%)  skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%)  skewX(-15deg); opacity: 0; }
}
.vnccs-cloner-btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255, 143, 163, 0.4);
}
.vnccs-cloner-btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

.vnccs-cloner-btn-success {
    background: rgba(0, 214, 143, 0.15);
    border: 1px solid rgba(0, 214, 143, 0.3);
    color: var(--success);
}
.vnccs-cloner-btn-success:hover:not(:disabled) { background: rgba(0, 214, 143, 0.25); transform: translateY(-1px); }

.vnccs-cloner-btn-danger {
    background: rgba(255, 71, 87, 0.15);
    border: 1px solid rgba(255, 71, 87, 0.3);
    color: var(--error);
}
.vnccs-cloner-btn-danger:hover:not(:disabled) { background: rgba(255, 71, 87, 0.25); transform: translateY(-1px); }

.vnccs-cloner-btn-disabled, .vnccs-cloner-btn:disabled {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-muted) !important;
    cursor: not-allowed;
    box-shadow: none !important;
    transform: none !important;
}

.vnccs-cloner-btn-upload {
    background: rgba(255, 143, 163, 0.08);
    border: 1px dashed var(--accent-border);
    color: var(--accent);
    border-radius: var(--radius-md);
    padding: 12px 24px;
    cursor: pointer;
    font-family: var(--font);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    transition: all var(--transition);
}
.vnccs-cloner-btn-upload:hover {
    background: rgba(255, 143, 163, 0.15);
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-subtle);
}

/* ── Image Grid ── */
.vnccs-cloner-img-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    min-height: 66px;
    flex-shrink: 0;
}
.vnccs-cloner-thumb-wrap {
    position: relative;
    display: inline-block;
    margin: 2px;
    border: 2px solid transparent;
    border-radius: var(--radius-sm);
    transition: all var(--transition);
}
.vnccs-cloner-thumb-wrap.is-selected {
    border-color: var(--accent);
    box-shadow: 0 0 10px rgba(255, 143, 163, 0.3);
}
.vnccs-cloner-thumb {
    width: 60px;
    height: 60px;
    object-fit: cover;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--transition);
}
.vnccs-cloner-thumb:hover { border-color: var(--accent); box-shadow: 0 0 8px var(--accent-subtle); }
.vnccs-cloner-thumb.generating {
    border: 2px solid var(--accent);
    animation: clonerPulse 1s infinite alternate;
    box-shadow: 0 0 12px var(--accent-glow);
}
@keyframes clonerPulse { from { opacity: 0.6; } to { opacity: 1; } }
.vnccs-cloner-thumb-remove {
    position: absolute;
    top: -1px;
    right: -1px;
    min-width: 20px;
    height: 20px;
    border-radius: 0 var(--radius-sm) 0 var(--radius-sm);
    background: rgba(10, 10, 15, 0.82);
    border: 1px solid rgba(255, 71, 87, 0.28);
    color: var(--error);
    cursor: pointer;
    font-size: 14px;
    line-height: 18px;
    text-align: center;
    font-weight: 700;
    transition: all var(--transition);
}
.vnccs-cloner-thumb-remove:hover {
    background: rgba(255, 71, 87, 0.2);
    border-color: var(--error);
}

/* ── Toggle Switch ── */
.vnccs-cloner-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    user-select: none;
}
.vnccs-cloner-toggle-wrap span {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}
.vnccs-cloner-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-cloner-toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
.vnccs-cloner-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    transition: all var(--transition);
}
.vnccs-cloner-toggle input:checked + .vnccs-cloner-toggle-track {
    background: rgba(255, 143, 163, 0.25);
    border-color: var(--accent-border);
    box-shadow: 0 0 8px var(--accent-subtle);
}
.vnccs-cloner-toggle-thumb {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all var(--transition);
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
.vnccs-cloner-toggle input:checked ~ .vnccs-cloner-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent-glow);
}

/* ── Creator V2 Controls ── */
.vnccs-cloner-segmented-field {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 4px;
    padding: 4px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.18);
    min-height: 48px;
    box-sizing: border-box;
}
.vnccs-cloner-segmented-btn {
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
.vnccs-cloner-segmented-btn:hover {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.045);
}
.vnccs-cloner-segmented-btn.is-active {
    color: #20141a;
    background: linear-gradient(180deg, #ff9bad 0%, #ff87a0 100%);
    box-shadow: 0 10px 22px rgba(255, 143, 163, 0.22);
}
.vnccs-cloner-graphic-toggle {
    width: 100%;
    min-height: 48px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.035);
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
.vnccs-cloner-graphic-toggle:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
}
.vnccs-cloner-graphic-toggle.is-active {
    border-color: var(--accent);
    background: rgba(255, 143, 163, 0.14);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255, 143, 163, 0.12) inset, 0 12px 24px rgba(255, 143, 163, 0.12);
}
.vnccs-cloner-graphic-toggle-text {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.vnccs-cloner-graphic-toggle-icon {
    width: 20px;
    height: 20px;
    border-radius: 7px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: currentColor;
    background: rgba(255, 255, 255, 0.06);
}
.vnccs-cloner-graphic-toggle-switch {
    width: 44px;
    height: 24px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid var(--border);
    position: relative;
    flex-shrink: 0;
    transition: all var(--transition);
}
.vnccs-cloner-graphic-toggle-switch::after {
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
.vnccs-cloner-graphic-toggle.is-active .vnccs-cloner-graphic-toggle-switch {
    border-color: var(--accent);
    background: rgba(255, 143, 163, 0.28);
}
.vnccs-cloner-graphic-toggle.is-active .vnccs-cloner-graphic-toggle-switch::after {
    transform: translateX(20px);
    background: var(--accent-hover);
}

/* ── Preview Container ── */
.vnccs-cloner-preview-container {
    flex: 1;
    background: radial-gradient(circle, rgba(255, 143, 163, 0.04) 1px, transparent 1px), rgba(10, 10, 15, 0.7);
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
.vnccs-cloner-preview-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: none;
    animation: clonerFadein 0.4s ease;
}
@keyframes clonerFadein { from { opacity: 0; } to { opacity: 1; } }
.vnccs-cloner-preview-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-muted);
    pointer-events: none;
}
.vnccs-cloner-preview-placeholder svg { opacity: 0.35; }
.vnccs-cloner-preview-placeholder-text {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.vnccs-cloner-upload-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    transition: opacity var(--transition), background var(--transition);
}
.vnccs-cloner-upload-overlay:hover {
    background: rgba(10, 10, 15, 0.28);
}

/* ── Bottom textareas ── */
.vnccs-cloner-textarea-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: rgba(20, 16, 30, 0.88);
    padding: 8px 10px;
    border-radius: var(--radius-md);
    border: 1px solid var(--accent-border);
    position: relative;
    pointer-events: auto;
}
.vnccs-cloner-textarea-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.4), transparent);
}
.vnccs-cloner-textarea-wrapper textarea {
    flex: 1;
    resize: none;
    border: none;
    background: transparent;
    padding: 4px;
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
}
.vnccs-cloner-textarea-wrapper textarea:focus { outline: none; }
.vnccs-cloner-textarea-label {
    font-size: 9px;
    color: var(--accent);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0 2px 4px;
}

.vnccs-cloner-field,
.vnccs-cloner-btn-row > *,
.vnccs-cloner-preview-container,
.vnccs-cloner-input,
.vnccs-cloner-select,
.vnccs-cloner-textarea,
.vnccs-cloner-img-list {
    pointer-events: auto;
}

/* ── Loading Overlay ── */
.vnccs-cloner-loading-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(10, 10, 15, 0.92);
    backdrop-filter: blur(10px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto; gap: 16px;
    border-radius: var(--radius-lg);
}
.vnccs-cloner-spinner {
    width: 44px;
    height: 44px;
    position: relative;
}
.vnccs-cloner-spinner::before, .vnccs-cloner-spinner::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 3px solid transparent;
}
.vnccs-cloner-spinner::before {
    border-top-color: var(--accent);
    border-right-color: rgba(255, 143, 163, 0.3);
    animation: clonerSpin 1s linear infinite;
    box-shadow: 0 0 18px rgba(255, 143, 163, 0.2);
}
.vnccs-cloner-spinner::after {
    inset: 7px;
    border-bottom-color: rgba(184, 169, 232, 0.6);
    border-left-color: rgba(184, 169, 232, 0.2);
    animation: clonerSpin 1.4s linear infinite reverse;
}
@keyframes clonerSpin { to { transform: rotate(360deg); } }
.vnccs-cloner-loading-text {
    color: var(--text-secondary);
    font-family: var(--font);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.vnccs-cloner-loading-dots::after {
    content: '';
    animation: clonerDots 1.5s steps(4, end) infinite;
}
@keyframes clonerDots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}

/* ── Modal Overrides ── */
.vnccs-cloner-container .vnccs-common-modal-overlay {
    background: rgba(5, 5, 9, 0.86);
    backdrop-filter: blur(8px);
}
.vnccs-cloner-container .vnccs-common-modal {
    width: 390px;
    max-width: calc(100% - 48px);
    padding: 18px;
    gap: 14px;
    background:
        linear-gradient(180deg, rgba(255, 143, 163, 0.055), rgba(255, 143, 163, 0.015)),
        var(--bg-secondary);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.55), 0 0 26px rgba(255, 143, 163, 0.1);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 13px;
}
.vnccs-cloner-container .vnccs-common-modal-title {
    padding-bottom: 10px;
    border-bottom: 1px solid var(--accent-border);
    color: var(--accent-hover);
    font-family: var(--font);
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.vnccs-cloner-container .vnccs-common-modal-btn-row {
    gap: 8px;
}
.vnccs-cloner-container .vnccs-common-modal-btn-row:empty {
    display: none;
}
.vnccs-cloner-container .vnccs-common-modal-btn {
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 8px 14px;
}
.vnccs-cloner-container .vnccs-common-modal-btn:hover {
    background: var(--bg-hover);
}
.vnccs-cloner-container .vnccs-common-modal-btn-primary,
.vnccs-cloner-container .vnccs-common-modal-btn-danger {
    color: #0a0a0f;
    border-color: transparent;
}
.vnccs-cloner-container .vnccs-common-modal-btn-primary {
    background: var(--accent);
}
.vnccs-cloner-container .vnccs-common-modal-btn-primary:hover {
    background: var(--accent-hover);
}
.vnccs-cloner-container .vnccs-common-modal-btn-danger {
    background: var(--error);
    color: #fff;
}
.vnccs-cloner-container .vnccs-cloner-download-modal {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
}
.vnccs-cloner-container .vnccs-cloner-download-status {
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 600;
    line-height: 1.35;
}
.vnccs-cloner-container .vnccs-cloner-download-track {
    width: 100%;
    height: 10px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.07);
}
.vnccs-cloner-container .vnccs-cloner-download-bar {
    width: 0%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--accent), var(--accent-hover));
    box-shadow: 0 0 18px var(--accent-glow);
    transition: width 0.35s ease;
}
.vnccs-cloner-container .vnccs-cloner-download-pct {
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 700;
    text-align: right;
}
.vnccs-cloner-container .vnccs-cloner-download-status.is-error {
    color: var(--error);
}

/* ── Tag Constructor ── */
.vnccs-cloner-tag-btn {
    width: 20px;
    height: 20px;
    margin-left: auto;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--accent-border);
    border-radius: 6px;
    background: rgba(255, 143, 163, 0.1);
    color: var(--accent);
    cursor: pointer;
    font-size: 12px;
    transition: all var(--transition);
}
.vnccs-cloner-tag-btn:hover {
    background: rgba(255, 143, 163, 0.2);
    box-shadow: 0 0 8px var(--accent-glow);
}
.vnccs-cloner-tag-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    max-height: 300px;
    overflow-y: auto;
    padding: 8px;
    background: rgba(10, 10, 15, 0.6);
    border-radius: var(--radius-sm);
}
.vnccs-cloner-tag-chip {
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
    border-radius: 20px;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 11px;
    user-select: none;
    transition: all var(--transition);
}
.vnccs-cloner-tag-chip:hover {
    background: rgba(255, 143, 163, 0.1);
    border-color: var(--accent-border);
    color: var(--accent-hover);
}
.vnccs-cloner-tag-chip.selected {
    background: rgba(255, 143, 163, 0.18);
    border-color: var(--accent);
    color: var(--accent-hover);
}
.vnccs-cloner-tag-category {
    width: 100%;
    margin-top: 6px;
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
`;

app.registerExtension({
    name: "VNCCS.CharacterCloner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CharacterCloner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1100, 760]); // Wider Sakura-style layout, matching Creator V2 density
                syncDOMWidgetWidthSoon(node, "ui");

                // 1. Setup CSS
                injectStyles(STYLE, "vnccs-character-cloner");

                // 2. Cleanup Widgets
                const cleanup = () => {
                    if (!node.widgets) return;
                    for (const w of node.widgets) {
                        if (w.name !== "ui" && w.name !== "widget_data") {
                            w.hidden = true;
                        }
                    }
                };
                cleanup();
                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    cleanup();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                // 3. State
                let dataWidget = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                if (!dataWidget) {
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                dataWidget.hidden = true;

                const state = {
                    character: "",
                    source_images: [], // List of filenames
                    character_info: {
                        sex: "female", age: 18, race: "human", skin_color: "",
                        hair: "", eyes: "", face: "", body: "", additional_details: "",
                        nsfw: false, aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                        lora_prompt: "", background_color: "Green"
                    }
                };
                node._vnccsGetClonerState = () => state;
                let TAG_DATA = null;

                const els = {};
                const saveState = () => {
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                    node._vnccsGetClonerState = () => state;
                    app.graph?.setDirtyCanvas(true, true);
                    window.dispatchEvent(new CustomEvent("vnccs-character-cloner-updated", {
                        detail: {
                            node_id: node.id,
                            character: state.character,
                            nsfw: !!state.character_info?.nsfw
                        }
                    }));
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
                    if (!sync || sync._vnccsClonerPatched) return !!sync;
                    const originalFindSourceNode = sync.findSourceNode?.bind(sync);
                    const originalRegisterStudio = sync.registerStudio?.bind(sync);

                    sync.findClonerSourceNode = () => {
                        const nodes = app.graph?._nodes || [];
                        return nodes.find(n => n?.type === "CharacterCloner") || null;
                    };
                    sync.findSourceNode = function () {
                        return originalFindSourceNode?.() || this.findClonerSourceNode?.() || null;
                    };
                    sync.applyClonerSource = function (sourceNode, options = {}) {
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
                    sync.hookClonerNode = function (sourceNode) {
                        if (!sourceNode || sourceNode.type !== "CharacterCloner") return;
                        const sourceWidget = sourceNode.widgets?.find(w => w.name === "widget_data");
                        let didHook = false;
                        if (sourceWidget && !sourceWidget._vnccsPoseStudioClonerValueHooked) {
                            let currentValue = sourceWidget.value;
                            Object.defineProperty(sourceWidget, "value", {
                                configurable: true,
                                get() {
                                    return currentValue;
                                },
                                set: (value) => {
                                    currentValue = value;
                                    queueMicrotask(() => this.applyClonerSource(sourceNode));
                                }
                            });
                            sourceWidget._vnccsPoseStudioClonerValueHooked = true;
                            didHook = true;
                        }
                        if (didHook) this.applyClonerSource(sourceNode, { initial: true });
                    };
                    sync.registerStudio = function (studio) {
                        originalRegisterStudio?.(studio);
                        const cloner = this.findClonerSourceNode?.();
                        if (cloner) this.applyClonerSource(cloner, { initial: true });
                    };
                    sync._vnccsClonerPatched = true;
                    return true;
                };
                const syncPoseStudioAge = (ageValue = state.character_info.age, options = {}) => {
                    state.character_info.age = normalizeAgeValue(ageValue);
                    node._vnccsGetPoseStudioValues = parsePoseStudioValues;
                    if (patchPoseStudioSync()) {
                        poseStudioSyncRetryCount = 0;
                        const sync = window.__vnccsPoseStudioCharacterCreatorSync;
                        const values = parsePoseStudioValues();
                        sync?.hookClonerNode?.(node);
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
                        }
                    } else if (poseStudioSyncRetryCount < 40) {
                        poseStudioSyncRetryCount += 1;
                        setTimeout(syncPoseStudioAge, 250);
                    }
                };
                const syncPoseStudioGender = (sexValue = state.character_info.sex) => {
                    const sex = String(sexValue || "").toLowerCase();
                    if (sex === "male" || sex === "female") state.character_info.sex = sex;
                    node._vnccsGetPoseStudioValues = parsePoseStudioValues;

                    if (patchPoseStudioSync()) {
                        const sync = window.__vnccsPoseStudioCharacterCreatorSync;
                        const values = parsePoseStudioValues();
                        sync?.hookClonerNode?.(node);
                        sync?.sourceSignatures?.set(node, values.signature);
                        for (const studio of sync?.studios || []) {
                            const applied = studio.applyExternalCharacterCreatorValues?.(values, { force: true });
                            if (!applied && Number.isFinite(values.gender) && studio.meshParams?.gender !== values.gender) {
                                studio.setManagerGender?.(values.gender);
                            }
                        }
                    } else if (poseStudioSyncRetryCount < 40) {
                        poseStudioSyncRetryCount += 1;
                        setTimeout(() => syncPoseStudioGender(state.character_info.sex), 250);
                    }
                };

                const loadState = () => {
                    if (dataWidget && dataWidget.value && dataWidget.value !== "{}") {
                        try {
                            console.log("[VNCCS] Loading State from widget_data:", dataWidget.value);
                            const parsed = JSON.parse(dataWidget.value);
                            if (Object.prototype.hasOwnProperty.call(parsed, "character")) {
                                state.character = parsed.character || "";
                            }
                            if (Array.isArray(parsed.source_images)) {
                                state.source_images = parsed.source_images;
                            }
                            if (Object.prototype.hasOwnProperty.call(parsed, "selected_idx")) {
                                state.selected_idx = parsed.selected_idx;
                            }
                            if (parsed.character_info && typeof parsed.character_info === "object") {
                                Object.assign(state.character_info, parsed.character_info);
                            }
                            node._vnccsGetClonerState = () => state;
                            // Update UI
                            updateUIFromState();
                            console.log("[VNCCS] State loaded successfully.");
                        } catch (e) {
                            console.error("[VNCCS] Failed to load state:", e);
                        }
                    }
                };

                // React to external changes (e.g. reload)
                if (dataWidget) dataWidget.callback = loadState;

                // Explicitly handle graph configuration (Restoration on Reload)
                const origConfigure = node.onConfigure;
                node.onConfigure = function () {
                    if (origConfigure) origConfigure.apply(this, arguments);
                    console.log("[VNCCS] onConfigure triggered.");
                    syncDOMWidgetWidth(node, "ui");
                    setTimeout(() => syncDOMWidgetWidth(node, "ui"), 100);
                    loadState();
                };

                // UI Builders
                const createField = (lbl, key, type = "text", opts = [], targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-cloner-field";

                    if (type === "checkbox") {
                        const toggleWrap = document.createElement("label");
                        toggleWrap.className = "vnccs-cloner-toggle-wrap";

                        const toggle = document.createElement("div");
                        toggle.className = "vnccs-cloner-toggle";

                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = e.target.checked; saveState(); };

                        const track = document.createElement("div");
                        track.className = "vnccs-cloner-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "vnccs-cloner-toggle-thumb";

                        toggle.appendChild(inp);
                        toggle.appendChild(track);
                        toggle.appendChild(thumb);

                        const labelSpan = document.createElement("span");
                        labelSpan.textContent = lbl;

                        toggleWrap.appendChild(toggle);
                        toggleWrap.appendChild(labelSpan);
                        wrap.appendChild(toggleWrap);
                        els[key] = inp;
                        return wrap;
                    }

                    const header = document.createElement("div");
                    header.style.display = "flex";
                    header.style.alignItems = "center";
                    header.style.justifyContent = "space-between";
                    header.innerHTML = `<div class="vnccs-cloner-label">${lbl}</div>`;

                    const tagSupported = ["hair", "eyes", "race", "body", "face", "additional_details"].includes(key);
                    if (tagSupported && type === "text") {
                        const btn = document.createElement("div");
                        btn.className = "vnccs-cloner-tag-btn";
                        btn.innerHTML = "✎";
                        btn.title = "Open Tag Constructor";
                        btn.onclick = () => openTagConstructor(key, inp, targetObj);
                        header.appendChild(btn);
                    }
                    wrap.appendChild(header);

                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-cloner-select";
                        opts.forEach(v => inp.add(new Option(v, v)));
                        inp.value = targetObj[key] || opts[0];
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-cloner-input";
                        inp.type = "number";
                        inp.value = targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = parseFloat(e.target.value); saveState(); };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-cloner-input";
                        inp.value = targetObj[key] || "";
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    }
                    els[key] = inp;
                    wrap.appendChild(inp);
                    return wrap;
                };

                const createSegmentedField = (lbl, key, options, targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-cloner-field";
                    wrap.innerHTML = `<div class="vnccs-cloner-label">${lbl}</div>`;

                    const segmented = document.createElement("div");
                    segmented.className = "vnccs-cloner-segmented-field";
                    const buttons = [];

                    const setValue = (value, persist = false) => {
                        const normalized = String(value || options[0]?.value || "");
                        targetObj[key] = normalized;
                        buttons.forEach(({ btn, value: btnValue }) => {
                            btn.classList.toggle("is-active", btnValue === normalized);
                            btn.setAttribute("aria-pressed", btnValue === normalized ? "true" : "false");
                        });
                        if (persist) {
                            saveState();
                            if (key === "sex" || key === "gender") syncPoseStudioGender(normalized);
                        }
                    };

                    options.forEach(option => {
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "vnccs-cloner-segmented-btn";
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
                    wrap.className = "vnccs-cloner-field";

                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "vnccs-cloner-graphic-toggle";
                    btn.innerHTML = `
                        <span class="vnccs-cloner-graphic-toggle-text">
                            <span class="vnccs-cloner-graphic-toggle-icon" aria-hidden="true">
                                <svg viewBox="0 0 24 24" width="15" height="15" fill="none">
                                    <path d="M12 3l7 4v5c0 4.5-2.8 7.4-7 9-4.2-1.6-7-4.5-7-9V7l7-4z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                                    <path d="M9 12h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </span>
                            ${lbl}
                        </span>
                        <span class="vnccs-cloner-graphic-toggle-switch" aria-hidden="true"></span>
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

                const createSlider = (lbl, key, min, max, step, targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-cloner-field";
                    wrap.innerHTML = `<div class="vnccs-cloner-label">${lbl}</div>`;

                    const container = document.createElement("div");
                    container.className = "vnccs-cloner-slider-container";

                    const range = document.createElement("input");
                    range.type = "range";
                    range.className = "vnccs-cloner-slider";
                    range.min = min;
                    range.max = max;
                    range.step = step;
                    range.value = targetObj[key];

                    const num = document.createElement("input");
                    num.type = "number";
                    num.className = "vnccs-cloner-slider-val";
                    num.min = min;
                    num.max = max;
                    num.step = step;
                    num.value = targetObj[key];

                    range.oninput = (e) => {
                        num.value = e.target.value;
                        targetObj[key] = parseFloat(e.target.value);
                        saveState();
                        if (key === "age") syncPoseStudioAge(targetObj[key]);
                    };
                    num.onchange = (e) => {
                        let v = parseFloat(e.target.value);
                        if (Number.isNaN(v)) v = min;
                        if (v < min) v = min;
                        if (v > max) v = max;
                        num.value = v;
                        range.value = v;
                        targetObj[key] = v;
                        saveState();
                        if (key === "age") syncPoseStudioAge(v);
                    };

                    container.appendChild(range);
                    container.appendChild(num);
                    wrap.appendChild(container);
                    els[key] = { range, num };
                    return wrap;
                };

                // Define renderThumbs placeholder (populated later)
                let renderThumbs = () => { };

                const updateUIFromState = () => {
                    // Fields
                    for (let k in state.character_info) {
                        if (els[k]) {
                            let val = state.character_info[k];
                            // Normalize background_color case to match dropdown options
                            if (k === "background_color" && typeof val === "string" && val) {
                                val = val.charAt(0).toUpperCase() + val.slice(1).toLowerCase();
                                state.character_info[k] = val;
                            }
                            if (els[k].setValue) els[k].setValue(val);
                            else if (els[k].range && els[k].num) {
                                els[k].range.value = val;
                                els[k].num.value = val;
                            }
                            else if (els[k].type === "checkbox") els[k].checked = val;
                            else els[k].value = val;
                        }
                    }
                    // Images
                    if (renderThumbs) renderThumbs();
                    syncPoseStudioAge();
                };

                // --- Helpers (Hoisted) ---
                // Modal Helper — delegates to vnccs_common showModal
                const showModal = (title, contentFunc, buttons) => {
                    const mappedButtons = buttons.map(b => ({
                        ...b,
                        class: b.class?.includes("danger") ? "danger" : b.class?.includes("primary") ? "primary" : undefined
                    }));
                    return showCommonModal(container, title, contentFunc, mappedButtons);
                };

                const openTagConstructor = async (fieldKey, inputEl, targetObj = state.character_info) => {
                    if (!TAG_DATA) {
                        try {
                            const r = await api.fetchApi("/vnccs/get_tags");
                            if (r.ok) TAG_DATA = await r.json();
                            else throw new Error("Failed to load tags");
                        } catch (e) {
                            showModal("Error", () => {
                                const d = document.createElement("div");
                                d.innerText = "Error loading tag database: " + e;
                                return d;
                            }, [{ text: "Close" }]);
                            return;
                        }
                    }

                    const map = {
                        hair: ["hair_color", "hairstyles"],
                        eyes: ["eyes"],
                        face: ["eyes"],
                        race: ["races"],
                        body: ["breast_size"],
                        additional_details: ["details"],
                    };
                    const categories = map[fieldKey] || [];
                    if (!categories.length) return;

                    const allTags = [];
                    categories.forEach(cat => {
                        const data = TAG_DATA?.tags?.[cat];
                        if (cat === "eyes") {
                            if (fieldKey === "eyes") {
                                if (TAG_DATA.tags.eyes.colors) allTags.push({ header: "Eye Colors", items: TAG_DATA.tags.eyes.colors });
                                if (TAG_DATA.tags.eyes.features) allTags.push({ header: "Eye Features", items: TAG_DATA.tags.eyes.features });
                            } else if (fieldKey === "face" && TAG_DATA.tags.eyes.face_characteristics) {
                                allTags.push({ header: "Face Characteristics", items: TAG_DATA.tags.eyes.face_characteristics });
                            }
                        } else if (data) {
                            allTags.push({ header: cat.replace("_", " ").toUpperCase(), items: data });
                        }
                    });

                    if (!allTags.length) {
                        showModal("No Tags", () => {
                            const d = document.createElement("div");
                            d.innerText = "No tags found for this category.";
                            return d;
                        }, [{ text: "Close" }]);
                        return;
                    }

                    const currentVals = String(inputEl.value || "").split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
                    const selected = new Set(currentVals);

                    showModal(`Tag Constructor: ${fieldKey}`, (modal) => {
                        const tagGrid = document.createElement("div");
                        tagGrid.className = "vnccs-cloner-tag-grid";
                        modal.style.width = "500px";

                        allTags.forEach(group => {
                            if (group.header) {
                                const header = document.createElement("div");
                                header.className = "vnccs-cloner-tag-category";
                                header.innerText = group.header;
                                tagGrid.appendChild(header);
                            }

                            group.items.forEach(item => {
                                const tag = item.tag;
                                const useTag = tag.replace(/_/g, " ");
                                const chip = document.createElement("div");
                                chip.className = "vnccs-cloner-tag-chip";
                                chip.innerText = item.label || tag;
                                if (selected.has(useTag)) chip.classList.add("selected");
                                chip.onclick = () => {
                                    if (selected.has(useTag)) {
                                        selected.delete(useTag);
                                        chip.classList.remove("selected");
                                    } else {
                                        selected.add(useTag);
                                        chip.classList.add("selected");
                                    }
                                };
                                tagGrid.appendChild(chip);
                            });
                        });

                        return tagGrid;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "APPLY",
                            class: "vnccs-btn-primary",
                            action: () => {
                                const final = Array.from(selected).join(", ");
                                inputEl.value = final;
                                targetObj[fieldKey] = final;
                                saveState();
                                return false;
                            }
                        }
                    ]);
                };

                const loadChar = async (name, skipInfoLoad = false) => {
                    if (!name || name === "None") {
                        state.char_preview_url = null;
                        updateUIFromState();
                        return;
                    }
                    try {
                        // Skip loading character_info if restoring from widget_data
                        if (!skipInfoLoad) {
                            const r = await api.fetchApi(`/vnccs/config?name=${encodeURIComponent(name)}`);
                            if (r.ok) {
                                const d = await r.json();
                                if (d.character_info) {
                                    Object.assign(state.character_info, d.character_info);
                                    updateUIFromState();
                                }
                            }
                        } else {
                            // Still update UI from current state
                            updateUIFromState();
                        }

                        // Load Preview Image URL Logic
                        const ts = Date.now();
                        const poseUrl = `/vnccs/get_character_pose_preview?character=${encodeURIComponent(name)}&t=${ts}`;
                        const cacheUrl = `/vnccs/get_cached_preview?character=${encodeURIComponent(name)}&t=${ts}`;

                        // Helper to find valid one
                        const checkImage = (url) => {
                            return new Promise((resolve) => {
                                const img = new Image();
                                img.onload = () => resolve(url);
                                img.onerror = () => resolve(null);
                                img.src = url;
                            });
                        };

                        let finalUrl = await checkImage(poseUrl);
                        if (!finalUrl) finalUrl = await checkImage(cacheUrl);

                        state.char_preview_url = finalUrl;
                        updateUIFromState(); // Trigger renderThumbs

                    } catch (e) {
                        console.error(e);
                        state.char_preview_url = null;
                        updateUIFromState();
                    }
                };

                const loadCharList = async () => {
                    try {
                        const r = await api.fetchApi("/vnccs/context_lists");
                        if (!r.ok) return;
                        const d = await r.json();

                        els.charSelect.innerHTML = "";
                        if (!d.characters || !d.characters.length) els.charSelect.add(new Option("None", ""));
                        else d.characters.forEach(c => els.charSelect.add(new Option(c, c)));

                        // Set Value
                        if (state.character && Array.from(els.charSelect.options).some(o => o.value === state.character)) {
                            els.charSelect.value = state.character;
                        } else if (els.charSelect.options.length > 0) {
                            state.character = els.charSelect.options[0].value;
                            els.charSelect.value = state.character;
                            saveState();
                        }

                        // Wait for loadChar to finish so preview updates
                        // Skip loading info if we already have widget_data (restoring session)
                        const hasWidgetData = state.character_info && state.character_info.hair !== undefined;
                        if (state.character) await loadChar(state.character, hasWidgetData);

                    } catch (e) { console.error(e); }
                };

                const doCreate = () => {
                    let inpRef;
                    showModal("New Character", () => {
                        const inp = document.createElement("input");
                        inp.className = "vnccs-cloner-input";
                        inp.placeholder = "Name...";
                        inpRef = inp;
                        return inp;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "Create", class: "vnccs-btn-primary",
                            action: async (ol, btn) => {
                                const n = inpRef.value.trim();
                                if (!n) return true;
                                try {
                                    await api.fetchApi(`/vnccs/create?name=${encodeURIComponent(n)}`);
                                    const exists = Array.from(els.charSelect.options).some(o => o.value === n);
                                    if (!exists) els.charSelect.add(new Option(n, n));
                                    state.character = n; els.charSelect.value = n;
                                    await loadChar(n);
                                    saveState();
                                    return false;
                                } catch (e) {
                                    showModal("Error", () => { const d = document.createElement("div"); d.innerText = "Create Failed: " + e; return d; }, [{ text: "Close" }]);
                                    return true;
                                }
                            }
                        }
                    ]);
                    inpRef.focus();
                };

                const doDelete = () => {
                    const charName = state.character;
                    if (!charName || charName === "None") return;
                    showModal("Delete Character", () => {
                        const d = document.createElement("div");
                        d.innerHTML = `Are you sure you want to delete <b>${charName}</b>?`;
                        return d;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "DELETE", class: "vnccs-btn-danger",
                            action: async (ol, btn) => {
                                try {
                                    const r = await api.fetchApi(`/vnccs/delete?name=${encodeURIComponent(charName)}`);
                                    if (r.ok) {
                                        const idx = Array.from(els.charSelect.options).findIndex(o => o.value === charName);
                                        if (idx > -1) els.charSelect.remove(idx);
                                        if (els.charSelect.options.length > 0) state.character = els.charSelect.options[0].value;
                                        else state.character = "";
                                        els.charSelect.value = state.character;
                                        await loadChar(state.character);
                                        saveState();
                                        return false;
                                    }
                                } catch (e) {
                                    showModal("Error", () => { const d = document.createElement("div"); d.innerText = "" + e; return d; }, [{ text: "Close" }]);
                                    return false;
                                }
                            }
                        }
                    ]);
                };

                // --- LAYOUT ---
                const container = document.createElement("div");
                container.className = "vnccs-cloner-container";

                // Top Row
                const topRow = document.createElement("div");
                topRow.className = "vnccs-cloner-top-row";

                // COL 1: SOURCE IMAGES
                // COL 1: SOURCE IMAGES (Left Column)
                const colSrc = document.createElement("div");
                colSrc.className = "vnccs-cloner-col";

                // --- CHARACTER SELECTOR (Moved to Left Top) ---
                colSrc.innerHTML = '<div class="vnccs-cloner-section-title">Character Select</div>';

                const charRow = document.createElement("div");
                charRow.className = "vnccs-cloner-field";

                const charSel = document.createElement("select");
                charSel.className = "vnccs-cloner-select";
                charSel.onchange = async (e) => {
                    state.character = e.target.value;
                    await loadChar(state.character);
                    saveState();
                };
                els.charSelect = charSel;
                charRow.appendChild(charSel);
                colSrc.appendChild(charRow);

                // (Preview removed: Using Native Preview Window below)

                const btnRow = document.createElement("div");
                btnRow.className = "vnccs-cloner-btn-row";

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-cloner-btn vnccs-cloner-btn-success";
                btnNew.innerText = "NEW";
                btnNew.onclick = doCreate;

                const btnDel = document.createElement("button");
                btnDel.className = "vnccs-cloner-btn vnccs-cloner-btn-danger";
                btnDel.innerText = "DEL";
                btnDel.onclick = doDelete;

                btnRow.appendChild(btnNew);
                btnRow.appendChild(btnDel);
                colSrc.appendChild(btnRow);

                // --- SOURCE IMAGES SECTION ---
                const srcHeader = document.createElement("div");
                srcHeader.className = "vnccs-cloner-section-title";
                srcHeader.innerText = "Source Images";
                srcHeader.style.marginTop = "15px";
                colSrc.appendChild(srcHeader);

                const imgList = document.createElement("div");
                imgList.className = "vnccs-cloner-img-list";
                colSrc.appendChild(imgList);

                // --- IMAGE PREVIEW/UPLOAD AREA ---
                // Container for the Large Preview
                const previewContainer = document.createElement("div");
                previewContainer.className = "vnccs-cloner-preview-container";

                // The Large Image Element
                const previewImg = document.createElement("img");
                previewImg.className = "vnccs-cloner-preview-img";
                previewContainer.appendChild(previewImg);

                // Placeholder (shown when no image)
                const previewPlaceholder = document.createElement("div");
                previewPlaceholder.className = "vnccs-cloner-preview-placeholder";
                previewPlaceholder.innerHTML = `
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="7" r="4" stroke="#ff8fa3" stroke-width="1.5"/>
                        <path d="M4 20c0-4 3.582-7 8-7s8 3 8 7" stroke="#ff8fa3" stroke-width="1.5" stroke-linecap="round"/>
                        <rect x="1" y="1" width="22" height="22" rx="4" stroke="#ff8fa3" stroke-width="1" stroke-dasharray="3 3" opacity="0.4"/>
                    </svg>
                    <div class="vnccs-cloner-preview-placeholder-text">Source Preview</div>
                `;
                previewContainer.appendChild(previewPlaceholder);

                // Overlay Controls (Upload Button) - Always visible or overlay?
                // User wants standard behaviour: Empty -> Upload Button. Filled -> Image + Mini Overlay?
                // Let's make the click on image trigger upload if empty?
                // Or keep the upload button below?
                // User said "Huge square with +upload images IS the place for preview"

                const uploadOverlay = document.createElement("div");
                uploadOverlay.className = "vnccs-cloner-upload-overlay";

                // If image exists, make overlay transparent or subtle?
                // Let's replicate the Button look but centered
                const uploadBtn = document.createElement("button");
                uploadBtn.className = "vnccs-cloner-btn vnccs-cloner-btn-upload";
                uploadBtn.innerText = "+ UPLOAD IMAGES";
                uploadOverlay.appendChild(uploadBtn);

                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.multiple = true;
                fileInput.accept = "image/*";
                fileInput.style.display = "none";
                fileInput.onchange = async (e) => {
                    if (e.target.files.length) {
                        uploadBtn.innerText = "UPLOADING...";
                        for (const file of e.target.files) {
                            try {
                                const body = new FormData();
                                body.append("image", file);
                                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                                const json = await resp.json();
                                if (json.name) {
                                    state.source_images.push({
                                        name: json.name,
                                        type: json.type || "input",
                                        subfolder: json.subfolder || ""
                                    });
                                }
                            } catch (err) {
                                showModal("Upload Error", () => { const d = document.createElement("div"); d.innerText = "Upload Failed: " + err; return d; }, [{ text: "Close" }]);
                            }
                        }
                        uploadBtn.innerText = "+ UPLOAD IMAGES";
                        saveState();
                        renderThumbs();
                    }
                };

                // Click anywhere on overlay triggers upload
                uploadOverlay.onclick = (e) => {
                    // If clicking button, it propagates. 
                    // But if we want to support clicking image to swap?
                    // For now, let's keep it simple: Button triggers input.
                    if (!previewImg.src || previewImg.style.display === "none") {
                        fileInput.click();
                    } else {
                        // If image is showing, maybe we don't want giant click area?
                        // User can still drag/drop (not implemented yet) or use button.
                        // Let's stick to button click.
                        if (e.target === uploadBtn) fileInput.click();
                        // If clicking background (not button), do nothing?
                    }
                };

                // Allow clicking button specifically even if overlay is transparent
                uploadBtn.onclick = (e) => { e.stopPropagation(); fileInput.click(); };

                previewContainer.appendChild(uploadOverlay);
                colSrc.appendChild(previewContainer);
                colSrc.appendChild(fileInput);


                // COL 2: ATTRIBUTES (Auto Gen)
                const colAttr = document.createElement("div");
                colAttr.className = "vnccs-cloner-col";

                const attrHeader = document.createElement("div");
                attrHeader.className = "vnccs-cloner-section-title";
                attrHeader.innerText = "Attributes";

                const autoGenBtn = document.createElement("button");
                autoGenBtn.className = "vnccs-cloner-btn vnccs-cloner-btn-primary";
                autoGenBtn.style.fontSize = "12px";
                autoGenBtn.style.flex = "0 0 auto"; // Prevent stretching
                autoGenBtn.innerText = "Analyze Captions (Qwen2-VL-7B)";
                autoGenBtn.onclick = async () => {
                    if (autoGenBtn.disabled) return;

                    if (!state.source_images.length) {
                        showModal("No Images", (m) => {
                            const d = document.createElement("div");
                            d.innerText = "Please upload at least one source image to analyze.";
                            return d;
                        }, [{ text: "OK", class: "vnccs-btn-primary" }]);
                        return;
                    }

                    // USE SELECTED IMAGE
                    const selIdx = (typeof state.selected_idx === 'number') ? state.selected_idx : 0;
                    const imgName = state.source_images[selIdx];

                    // Show loading overlay
                    const loading = createLoadingOverlay(container, "Analyzing image");

                    autoGenBtn.innerText = "ANALYZING...";
                    autoGenBtn.disabled = true;

                    // Visual feedback highlighting selected thumb
                    const thumbs = imgList.querySelectorAll("img");
                    let thumb = null;
                    if (thumbs[selIdx]) {
                        thumb = thumbs[selIdx];
                        thumb.classList.add("generating");
                    }

                    try {
                        const r = await api.fetchApi("/vnccs/cloner_auto_generate", {
                            method: "POST",
                            body: JSON.stringify({ image_name: imgName })
                        });

                        if (r.ok) {
                            const data = await r.json();
                            console.log("[VNCCS] Auto-Gen Success. Data:", data);

                            // Merge into state
                            Object.assign(state.character_info, data);
                            console.log("[VNCCS] State Updated:", state.character_info);

                            updateUIFromState();
                            console.log("[VNCCS] UI Updated.");
                            saveState();
                        } else {
                            console.log("[VNCCS] Auto-Gen Failed. Status:", r.status);
                            // Check for structured errors (404 for model, 500 for mmproj/other)
                            let err = null;
                            try { err = await r.json(); } catch (e) { }

                            if (err && (err.error === "MODEL_MISSING" || err.error === "MODEL_INVALID" || err.error === "MMPROJ_MISSING" || err.error === "MMPROJ_INVALID" || err.error === "DEPENDENCY_MISSING")) {

                                // CASE A: Dependency Error (llama-cpp-python)
                                if (err.error === "DEPENDENCY_MISSING") {
                                    showModal("Dependency Missing", (m) => {
                                        const d = document.createElement("div");
                                        d.innerHTML = `
                                            <div style="font-size:13px; margin-bottom:10px;">
                                                <b>Missing AI Library</b><br/><br/>
                                                ${err.message}<br/>
                                                <b>Model:</b> ${err.model_name}<br/><br/>
                                                <span style="color:#f88">Correct 'llama-cpp-python' version required.</span><br/>
                                                Please install the JamePeng fork or a compatible version manually.
                                            </div>
                                        `;
                                        return d;
                                    }, [
                                        { text: "OK", class: "vnccs-btn-danger" }
                                    ]);
                                    return;
                                }

                                // CASE B: Regular Missing Model Files
                                showModal("Model Missing", (m) => {
                                    const d = document.createElement("div");
                                    const isInvalid = err.error === "MODEL_INVALID" || err.error === "MMPROJ_INVALID";
                                    const missingMsg = err.error === "MMPROJ_MISSING"
                                        ? "The Vision Projector (mmproj) is missing."
                                        : err.error === "MMPROJ_INVALID"
                                            ? "The Vision Projector (mmproj) is invalid or incomplete."
                                            : err.error === "MODEL_INVALID"
                                                ? `Model file is invalid or incomplete: ${err.model_name || 'QwenVL'}`
                                                : `Required Model: ${err.model_name || 'QwenVL'}`;
                                    d.innerHTML = `
                                        <div style="font-size:13px; margin-bottom:10px;">
                                            <b>${missingMsg}</b><br/><br/>
                                            ${isInvalid ? "The existing file will be replaced." : "This component is required for character analysis."} Would you like to download it now?<br/>
                                            <span style="color:#aaa; font-size:11px;">Verification & Download will start automatically.</span>
                                        </div>
                                    `;
                                    return d;
                                }, [
                                    { text: "Cancel" },
                                    {
                                        text: "DOWNLOAD & INSTALL", class: "vnccs-btn-success",
                                        action: async (ol, btn) => {
                                            // Trigger Download
                                            try {
                                                const dl = await api.fetchApi("/vnccs/cloner_download_model", { method: "POST" });
                                                if (dl.status === 409) {
                                                    // already downloading
                                                    startProgressPolling();
                                                    return false;
                                                }
                                                if (dl.ok) {
                                                    ol.remove(); // Close prompt
                                                    startProgressPolling(); // View Progress
                                                    return false;
                                                }
                                            } catch (e) {
                                                showModal("Error", () => { const d = document.createElement("div"); d.innerText = "Download trigger failed: " + e; return d; }, [{ text: "Close" }]);
                                            }
                                            return true;
                                        }
                                    }
                                ]);
                                return; // Exit normally
                            }

                            // Fallback for real errors
                            const txt = err && err.message ? err.message : (await r.text());
                            showModal("Error", (m) => {
                                const d = document.createElement("div");
                                d.innerText = "Auto-Gen Failed: " + txt;
                                d.style.color = "red";
                                return d;
                            }, [{ text: "Close" }]);
                        }
                    } catch (e) {
                        showModal("Error", (m) => {
                            const d = document.createElement("div"); d.innerText = "Script Error: " + e; return d;
                        }, [{ text: "Close" }]);
                    } finally {
                        loading.remove();
                        autoGenBtn.innerText = "Analyze Captions (Qwen2-VL-7B)";
                        autoGenBtn.disabled = false;
                        if (thumb) thumb.classList.remove("generating");
                    }
                };

                // Helper: Progress Polling
                const startProgressPolling = () => {
                    const { overlay, modal } = showModal("Downloading Model...", (m) => {
                        const d = document.createElement("div");
                        d.className = "vnccs-cloner-download-modal";
                        d.innerHTML = `
                             <div class="vnccs-cloner-download-status" id="vnccs-dl-status">Starting...</div>
                             <div class="vnccs-cloner-download-track">
                                <div class="vnccs-cloner-download-bar" id="vnccs-dl-bar"></div>
                             </div>
                             <div class="vnccs-cloner-download-pct" id="vnccs-dl-pct">0%</div>
                        `;
                        return d;
                    }, []); // No buttons, auto-close

                    const statusEl = modal.querySelector("#vnccs-dl-status");
                    const barEl = modal.querySelector("#vnccs-dl-bar");
                    const pctEl = modal.querySelector("#vnccs-dl-pct");

                    const interval = setInterval(async () => {
                        try {
                            const r = await api.fetchApi("/vnccs/cloner_download_status");
                            if (r.ok) {
                                const d = await r.json();
                                if (d.status === "completed") {
                                    clearInterval(interval);
                                    statusEl.innerText = "Download Complete!";
                                    barEl.style.width = "100%";
                                    pctEl.innerText = "100%";
                                    setTimeout(() => overlay.remove(), 1000);
                                    // Maybe retry generation automatically? No, user can click.
                                } else if (d.status === "error") {
                                    clearInterval(interval);
                                    statusEl.innerText = "Error: " + d.error;
                                    statusEl.classList.add("is-error");
                                } else {
                                    statusEl.innerText = `Downloading ${d.current_file}...`;
                                    barEl.style.width = d.progress + "%";
                                    pctEl.innerText = d.progress + "%";
                                }
                            }
                        } catch (e) {
                            clearInterval(interval);
                            overlay.remove();
                        }
                    }, 1000);
                };

                // Add Header
                colAttr.appendChild(attrHeader);

                // Add Button (Below header, above fields)
                autoGenBtn.style.width = "100%";
                autoGenBtn.style.marginBottom = "10px";
                autoGenBtn.style.textAlign = "center";
                colAttr.appendChild(autoGenBtn);



                // Other fields
                // colAttr.appendChild(createField("Name", "character", "text", [], state)); // REMOVE THIS

                colAttr.appendChild(createSegmentedField("Background", "background_color", [
                    { label: "Green", value: "Green" },
                    { label: "Blue", value: "Blue" },
                ]));
                colAttr.appendChild(createSegmentedField("Gender", "sex", [
                    { label: "Male", value: "male" },
                    { label: "Female", value: "female" },
                ]));
                colAttr.appendChild(createSlider("Age", "age", 1, 100, 1, state.character_info));
                colAttr.appendChild(createField("Race", "race", "text"));
                colAttr.appendChild(createField("Skin Color", "skin_color", "text"));
                colAttr.appendChild(createField("Hair", "hair", "text"));
                colAttr.appendChild(createField("Eyes", "eyes", "text"));
                colAttr.appendChild(createField("Face", "face", "text"));
                colAttr.appendChild(createField("Body", "body", "text"));
                colAttr.appendChild(createField("Details", "additional_details", "text"));
                colAttr.appendChild(createField("Aesthetics", "aesthetics", "text"));
                colAttr.appendChild(createGraphicToggle("NSFW Mode", "nsfw"));

                // Assemble Top Row
                topRow.appendChild(colSrc);
                topRow.appendChild(colAttr);
                container.appendChild(topRow);

                // --- BOTTOM ROW (Prompts) --
                const botRow = document.createElement("div");
                botRow.className = "vnccs-cloner-bottom-row";

                // Read-only Prompts? Or generated?
                // Standard textareas
                const createTA = (lbl, key) => {
                    const w = document.createElement("div");
                    w.className = "vnccs-cloner-textarea-wrapper";
                    w.innerHTML = `<div class="vnccs-cloner-textarea-label">${lbl}</div>`;
                    const t = document.createElement("textarea");
                    t.className = "vnccs-cloner-textarea";
                    // We don't bind this to state input, but output?
                    // Actually, output prompts are generated from attributes. 
                    // Users might want to ADD to it. 
                    // For now, let's bind to lora_prompt or neg_prompt
                    if (key === "lora_prompt") {
                        t.value = state.character_info.lora_prompt;
                        t.onchange = (e) => { state.character_info.lora_prompt = e.target.value; saveState(); }
                    } else if (key === "negative_prompt") {
                        t.value = state.character_info.negative_prompt;
                        t.onchange = (e) => { state.character_info.negative_prompt = e.target.value; saveState(); }
                    }
                    // If positive, it's auto-generated... maybe just show extra prompt field?
                    // Let's stick to "LoRA Trigger / Manual Prompt" and "Negative"
                    w.appendChild(t);
                    return w;
                };

                botRow.appendChild(createTA("Extra / LoRA Prompt", "lora_prompt"));
                botRow.appendChild(createTA("Negative Prompt", "negative_prompt"));

                container.appendChild(botRow);

                // ADD WIDGET
                node.addDOMWidget("ui", "ui", container, { serialize: false, hideOnZoom: false });
                syncDOMWidgetWidthSoon(node, "ui");

                // Define Render Thumbs Logic
                renderThumbs = () => {
                    imgList.innerHTML = "";
                    const images = state.source_images;

                    // Default Selection
                    if (typeof state.selected_idx !== 'number' || state.selected_idx < 0 || state.selected_idx >= images.length) {
                        state.selected_idx = 0;
                    }

                    // Preview Logic
                    if (images.length === 0) {
                        if (state.char_preview_url) {
                            previewImg.src = state.char_preview_url;
                            previewImg.style.display = "block";
                            previewPlaceholder.style.display = "none";
                            uploadOverlay.style.opacity = "0";
                            previewContainer.onmouseenter = () => uploadOverlay.style.opacity = "1";
                            previewContainer.onmouseleave = () => uploadOverlay.style.opacity = "0";
                        } else {
                            previewImg.style.display = "none";
                            previewImg.src = "";
                            previewPlaceholder.style.display = "flex";
                            uploadOverlay.style.opacity = "1";
                            uploadOverlay.style.background = "transparent";
                            uploadBtn.style.display = "block";
                            previewContainer.onmouseenter = null;
                            previewContainer.onmouseleave = null;
                        }
                    } else {
                        // Show Selected Image
                        const imgObj = images[state.selected_idx];
                        let name = "", type = "input", sub = "";
                        if (typeof imgObj === 'string') { name = imgObj; }
                        else if (imgObj && imgObj.name) { name = imgObj.name; type = imgObj.type || "input"; sub = imgObj.subfolder || ""; }

                        if (name) {
                            const params = new URLSearchParams();
                            params.append("filename", name);
                            params.append("type", type);
                            if (sub) params.append("subfolder", sub);
                            previewImg.src = api.apiURL("/view?" + params.toString());
                            previewImg.style.display = "block";
                            previewPlaceholder.style.display = "none";
                            uploadOverlay.style.opacity = "0";
                            previewContainer.onmouseenter = () => uploadOverlay.style.opacity = "1";
                            previewContainer.onmouseleave = () => uploadOverlay.style.opacity = "0";
                        }
                    }

                    // Thumbnails Loop
                    images.forEach((imgObj, idx) => {
                        let name = "", type = "input", sub = "";
                        if (typeof imgObj === 'string') { name = imgObj; }
                        else if (imgObj && imgObj.name) { name = imgObj.name; type = imgObj.type || "input"; sub = imgObj.subfolder || ""; }
                        if (!name) return;

                        const wrap = document.createElement("div");
                        wrap.className = "vnccs-cloner-thumb-wrap";
                        if (idx === state.selected_idx) {
                            wrap.classList.add("is-selected");
                        }

                        const img = document.createElement("img");
                        const params = new URLSearchParams();
                        params.append("filename", name);
                        params.append("type", type);
                        if (sub) params.append("subfolder", sub);
                        img.src = api.apiURL("/view?" + params.toString());
                        img.className = "vnccs-cloner-thumb";

                        img.onclick = () => {
                            state.selected_idx = idx;
                            saveState();
                            renderThumbs();
                        };

                        // Delete X
                        const delBtn = document.createElement("div");
                        delBtn.innerText = "×";
                        delBtn.className = "vnccs-cloner-thumb-remove";
                        delBtn.onclick = (e) => {
                            e.stopPropagation();
                            showModal("Remove Image", () => {
                                const d = document.createElement("div");
                                d.innerText = "Are you sure you want to remove this image?";
                                return d;
                            }, [
                                { text: "Cancel" },
                                {
                                    text: "Remove",
                                    class: "vnccs-btn-danger",
                                    action: () => {
                                        state.source_images.splice(idx, 1);
                                        if (state.selected_idx >= state.source_images.length) state.selected_idx = Math.max(0, state.source_images.length - 1);
                                        saveState();
                                        renderThumbs();
                                        return false;
                                    }
                                }
                            ]);
                        };

                        wrap.appendChild(img);
                        wrap.appendChild(delBtn);
                        imgList.appendChild(wrap);
                    });
                };

                // Initialize
                loadState();
                loadCharList();
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
