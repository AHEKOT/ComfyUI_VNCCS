import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan } from "./vnccs_common.js";

// --- CSS STYLES: Sakura Archive Design System ---
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Variables ── */
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
    --error: #ff4757;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --transition: 0.2s ease;
}

/* ── Container ── */
.ems-container {
    display: flex;
    flex-direction: row;
    gap: 10px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 24px;
    padding: 10px;
    border-radius: 8px;
    width: 100%;
    height: 100%;
    max-height: 100%;
    box-sizing: border-box;
    overflow: hidden;
    background-image: radial-gradient(ellipse at 15% 0%, rgba(255, 143, 163, 0.05) 0%, transparent 55%),
                      radial-gradient(ellipse at 85% 100%, rgba(184, 169, 232, 0.04) 0%, transparent 55%);
}

/* ── Columns ── */
.ems-left-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 200px;
    overflow: hidden;
}

.ems-right-col {
    flex: 3;
    display: flex;
    flex-direction: row;
    gap: 10px;
    min-width: 300px;
    overflow: hidden;
}
.ems-selection-col {
    flex: 1 1 auto;
    min-width: 420px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow: hidden;
}

/* ── Sections ── */
.ems-section {
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    background: rgba(20, 16, 30, 0.88);
    padding: 8px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.ems-section::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
    border-radius: 1px;
    pointer-events: none;
}

/* ── Character Header ── */
.ems-char-header {
    color: var(--accent);
    padding: 4px 8px 8px;
    font-weight: 700;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.ems-char-preview-container {
    flex: 1;
    background: radial-gradient(circle, rgba(255, 143, 163, 0.04) 1px, transparent 1px), rgba(10, 8, 16, 0.6);
    background-size: 18px 18px, 100% 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
}

.ems-char-preview {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* ── Costumes ── */
.ems-costumes-header {
    font-weight: 700;
    color: var(--accent);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    padding-left: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ems-costumes-header::before {
    content: '';
    width: 3px;
    height: 10px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 6px var(--accent-glow);
    flex-shrink: 0;
}
.ems-costumes-list {
    background: rgba(10, 8, 16, 0.5);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    color: var(--text-primary);
    max-height: 120px;
    overflow-y: auto;
}
.ems-costumes-list::-webkit-scrollbar { height: 3px; width: 3px; }
.ems-costumes-list::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* ── Costume Toggle Items ── */
.ems-checkbox-item {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
}
.ems-checkbox-item span {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.em-toggle {
    position: relative;
    width: 34px;
    height: 18px;
    flex-shrink: 0;
}
.em-toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
.em-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all var(--transition);
}
.em-toggle input:checked + .em-toggle-track {
    background: rgba(255, 143, 163, 0.22);
    border-color: var(--accent-border);
    box-shadow: 0 0 8px var(--accent-subtle);
}
.em-toggle-thumb {
    position: absolute;
    top: 3px; left: 3px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all var(--transition);
}
.em-toggle input:checked ~ .em-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent-glow);
}

/* ── Emotions Grid ── */
.ems-selected-emotions-wrap {
    flex: 0 0 auto;
    background: rgba(8, 6, 14, 0.5);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    padding: 8px;
    margin-bottom: 8px;
}
.ems-selected-emotions-header {
    color: var(--accent);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ems-selected-emotions-header::before {
    content: '';
    width: 3px;
    height: 10px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 6px var(--accent-glow);
    flex-shrink: 0;
}
.ems-selected-emotions-list {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 8px;
    align-content: start;
    align-items: start;
    max-height: 178px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-border) transparent;
}
.ems-selected-emotions-list::-webkit-scrollbar { width: 4px; }
.ems-selected-emotions-list::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }
.ems-selected-emotions-empty {
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    padding: 10px;
    text-align: center;
    border: 1px dashed var(--border-hover);
    border-radius: var(--radius-sm);
}
.ems-emotions-container {
    flex: 1;
    background: rgba(8, 6, 14, 0.5);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    align-content: start;
    align-items: start;
    min-height: 200px;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-border) transparent;
}
.ems-emotions-container::-webkit-scrollbar { width: 4px; }
.ems-emotions-container::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

.ems-emotion-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    align-self: start;
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    padding: 6px;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    box-sizing: border-box;
    background: transparent;
}
.ems-emotion-item:hover {
    background: rgba(255, 143, 163, 0.06);
    border-color: var(--accent-border);
}
.ems-emotion-item.selected {
    background: rgba(255, 143, 163, 0.15);
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--accent-subtle), inset 0 0 8px rgba(255, 143, 163, 0.05);
}
.ems-emotion-item.selected .ems-emotion-label {
    color: var(--accent-hover);
    font-weight: 600;
}

.ems-emotion-img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    transition: border-color var(--transition);
}
.ems-emotion-item.selected .ems-emotion-img {
    border-color: var(--accent-border);
}
.ems-emotion-label {
    font-size: 20px;
    color: var(--text-secondary);
    text-align: center;
    margin-top: 4px;
    font-family: var(--font);
    font-weight: 500;
    word-break: break-word;
    width: 100%;
    line-height: 1.2;
}
.ems-emotion-item.compact {
    padding: 4px;
}
.ems-emotion-item.compact .ems-emotion-label {
    font-size: 12px;
    margin-top: 3px;
}

/* ── Footer / Button ── */
.ems-footer {
    margin-top: 6px;
    display: flex;
    justify-content: center;
}
.ems-btn {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525;
    border: none;
    border-radius: var(--radius-md);
    padding: 10px 32px;
    font-family: var(--font);
    font-weight: 700;
    font-size: 22px;
    letter-spacing: 0.5px;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(255, 143, 163, 0.25);
    position: relative;
    overflow: hidden;
    transition: all var(--transition);
    text-transform: uppercase;
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.ems-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 45%, rgba(255,255,255,0.28) 50%, rgba(255,255,255,0.2) 55%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: emBtnShimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes emBtnShimmer {
    0%   { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35%  { transform: translateX(120%)  skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%)  skewX(-15deg); opacity: 0; }
}
.ems-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255, 143, 163, 0.45);
}
.ems-btn:focus:not(:disabled),
.ems-btn:focus-visible:not(:disabled),
.ems-btn:active:not(:disabled) {
    outline: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 8px 30px rgba(255, 143, 163, 0.45), 0 0 0 2px rgba(255,143,163,0.28);
}
.ems-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.ems-btn.cancel {
    background: rgba(255, 71, 87, 0.15);
    color: var(--error);
    border: 1px solid rgba(255, 71, 87, 0.3);
    box-shadow: none;
}
.ems-btn.cancel::after { display: none; }
.ems-btn.cancel:hover:not(:disabled) {
    background: rgba(255, 71, 87, 0.28);
    box-shadow: 0 4px 16px rgba(255, 71, 87, 0.2);
}

/* ── Search Input ── */
.ems-search-input {
    width: 100%;
    padding: 8px 14px;
    margin-top: 6px;
    margin-bottom: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-md);
    font-size: 14px;
    font-family: var(--font);
    box-sizing: border-box;
    transition: all var(--transition);
}
.ems-search-input:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255, 143, 163, 0.03);
    box-shadow: 0 0 0 3px rgba(255, 143, 163, 0.05);
}
.ems-search-input::placeholder { color: var(--text-muted); }

/* ── Custom Select Style ── */
.em-select {
    width: 100%;
    padding: 6px 10px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md);
    font-family: var(--font);
    font-size: 13px;
    font-weight: 500;
    transition: all var(--transition);
    cursor: pointer;
}
.em-select:focus {
    outline: none;
    border-color: var(--accent-border);
    box-shadow: 0 0 0 2px rgba(255, 143, 163, 0.08);
}

/* ── Generation Panel ── */
.ems-generation-section {
    flex: 0 0 460px;
    gap: 10px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-border) transparent;
}
.ems-generation-section::-webkit-scrollbar { width: 4px; }
.ems-generation-section::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }
.ems-generation-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}
.ems-tab-row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.ems-tab,
.ems-seed-dice {
    appearance: none;
    -webkit-appearance: none;
    border: 1px solid var(--border-hover);
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    font-family: var(--font);
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: all var(--transition);
}
.ems-tab {
    min-height: 34px;
    font-size: 11px;
}
.ems-tab:hover,
.ems-tab:focus,
.ems-tab:focus-visible,
.ems-tab:active {
    outline: none;
    color: var(--text-primary);
    border-color: var(--accent-border);
    background: rgba(255, 143, 163, 0.12) !important;
}
.ems-tab.active {
    color: var(--accent-hover);
    border-color: var(--accent);
    background: rgba(255, 143, 163, 0.2) !important;
    box-shadow: inset 0 0 0 1px rgba(255, 143, 163, 0.1), 0 0 18px rgba(255, 143, 163, 0.12);
}
.ems-model-card {
    min-height: 58px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: flex-start;
    gap: 5px;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--accent-border);
    background: rgba(255, 143, 163, 0.12);
    box-sizing: border-box;
    overflow: hidden;
}
.ems-model-picker {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.ems-model-card-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.ems-model-picker-menu {
    display: none;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    border: 1px solid rgba(255,143,163,0.18);
    border-radius: 10px;
    background: rgba(8,8,12,0.48);
}
.ems-model-picker.is-open .ems-model-picker-menu {
    display: flex;
}
.ems-model-picker-group {
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.ems-model-picker-group-title {
    color: var(--accent-hover);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ems-model-card.is-installed {
    cursor: pointer;
}
.ems-model-card.is-selected {
    border-color: var(--accent);
    background: rgba(255, 143, 163, 0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset;
}
.ems-model-card.is-missing {
    opacity: 0.92;
}
.ems-model-card-top {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
}
.ems-model-title {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 0;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 800;
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ems-model-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 10px rgba(0, 214, 143, 0.38);
    flex-shrink: 0;
}
.ems-model-desc {
    margin-top: 4px;
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ems-model-status {
    color: var(--success);
    font-size: 10px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    flex-shrink: 0;
}
.ems-model-status.ok { color: var(--success); }
.ems-model-status.missing { color: var(--error); }
.ems-model-status.progress { color: var(--accent-lavender); }
.ems-model-dot.missing { background: var(--error); }
.ems-model-dot.progress { background: var(--accent-lavender); }
.ems-model-card-download {
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
.ems-model-card-download:hover {
    background: rgba(255,143,163,0.14);
}
.ems-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.ems-toggle input {
    opacity: 0;
    width: 0;
    height: 0;
    position: absolute;
}
.ems-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    transition: all var(--transition);
}
.ems-toggle-thumb {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all var(--transition);
}
.ems-toggle input:checked ~ .ems-toggle-track {
    background: rgba(255,143,163,0.2);
    border-color: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
}
.ems-toggle input:checked ~ .ems-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
}
.ems-field {
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.ems-label {
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.ems-input {
    width: 100%;
    min-height: 40px;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-sm);
    font-family: var(--font);
    font-size: 13px;
    font-weight: 700;
    box-sizing: border-box;
}
.ems-input:focus {
    outline: none;
    border-color: var(--accent-border);
    box-shadow: 0 0 0 2px rgba(255, 143, 163, 0.08);
}
.ems-seed-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 44px;
    gap: 8px;
}
.ems-seed-dice {
    min-height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
}
.ems-seed-dice svg {
    width: 20px;
    height: 20px;
}
.ems-seed-dice.active {
    color: var(--accent-hover);
    border-color: var(--accent);
    background: rgba(255, 143, 163, 0.2) !important;
}
.ems-lora-stack {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.ems-lora-card {
    min-height: 42px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 8px 10px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--accent-border);
    background: rgba(255, 143, 163, 0.12);
    box-sizing: border-box;
    cursor: pointer;
}
.ems-lora-card .ems-lora-name {
    flex: 1;
}
.ems-lora-card .ems-model-status {
    margin-left: auto;
}
.ems-lora-card .ems-model-card-download {
    width: auto;
    flex-shrink: 0;
}
.ems-lora-card.is-selected {
    border-color: var(--accent);
    background: rgba(255, 143, 163, 0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset;
}
.ems-lora-name {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 800;
}
.ems-lora-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 90px;
    gap: 8px;
    padding: 8px;
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.04);
}

/* ── Confirm Modal ── */
.ems-modal-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    border-radius: 8px;
}
.ems-modal {
    background: rgba(22, 18, 34, 0.98);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 24px 28px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 280px;
    max-width: 360px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 143, 163, 0.08);
    position: relative;
}
.ems-modal::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
    border-radius: 1px;
}
.ems-modal-text {
    font-family: var(--font);
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.6;
}
.ems-modal-text strong {
    color: var(--accent);
    font-weight: 700;
}
.ems-modal-actions {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
}
.ems-modal-btn {
    padding: 8px 20px;
    border-radius: var(--radius-md);
    font-family: var(--font);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    cursor: pointer;
    border: none;
    transition: all var(--transition);
}
.ems-modal-btn:focus,
.ems-modal-btn:focus-visible,
.ems-modal-btn:active {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.ems-modal-btn--cancel {
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-secondary);
    border: 1px solid var(--border);
}
.ems-modal-btn--cancel:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
}
.ems-modal-btn--confirm {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525;
    box-shadow: 0 4px 16px rgba(255, 143, 163, 0.25);
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.ems-modal-btn--confirm:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(255, 143, 163, 0.4);
}
.ems-modal-btn--confirm:focus,
.ems-modal-btn--confirm:focus-visible,
.ems-modal-btn--confirm:active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 6px 20px rgba(255, 143, 163, 0.4), 0 0 0 2px rgba(255,143,163,0.28);
}
`;

// Inject Styles
const styleEl = document.createElement("style");
styleEl.textContent = STYLE;
document.head.appendChild(styleEl);

// --- MAIN EXTENSION ---

app.registerExtension({
    name: "VNCCS.EmotionGeneratorV2",

    async setup() {
        const origQueuePrompt = app.queuePrompt.bind(app);
        app.queuePrompt = async function(...args) {
            const nodes = app.graph?._nodes?.filter(n => n.type === "EmotionGeneratorV2") || [];
            for (const node of nodes) {
                if (node._validateBeforeQueue && !node._validateBeforeQueue()) {
                    return; // block queue, modal already shown inside _validateBeforeQueue
                }
            }
            return origQueuePrompt(...args);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EmotionGeneratorV2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;

                // Set default size
                node.setSize([920, 650]);
                syncDOMWidgetWidthSoon(node, "emotion_ui_v2");

                // Get Widgets
                const charWidget = node.widgets.find(w => w.name === "character");
                const modelWidget = node.widgets.find(w => w.name === "generation_model");
                const generationSettingsWidget = node.widgets.find(w => w.name === "generation_settings");
                const costumesDataWidget = node.widgets.find(w => w.name === "costumes_data");
                const emotionsDataWidget = node.widgets.find(w => w.name === "emotions_data");

                // Hide data widgets
                if (costumesDataWidget) costumesDataWidget.hidden = true;
                if (emotionsDataWidget) emotionsDataWidget.hidden = true;
                if (modelWidget) modelWidget.hidden = true;
                if (generationSettingsWidget) generationSettingsWidget.hidden = true;

                const generateRandomSeed = () => Math.floor(Math.random() * 9007199254740991);
                const ANIMA_TURBO_LORA_NAME = "anima\\anima-turbo-lora-v0.1.safetensors";
                const ANIMA_CLIP_NAME = "qwen_3_06b_base.safetensors";
                const ANIMA_VAE_NAME = "qwen_image_vae.safetensors";
                const GENERATION_DEFAULTS = {
                    generation_mode: "anima",
                    ckpt_name: "",
                    diffusion_model_name: "",
                    clip_name: ANIMA_CLIP_NAME,
                    vae_name: ANIMA_VAE_NAME,
                    clip_type: "stable_diffusion",
                    sampler: "er_sde",
                    scheduler: "simple",
                    steps: 30,
                    cfg: 4.0,
                    seed: generateRandomSeed(),
                    seed_mode: "fixed",
                    turbo_enabled: false,
                    turbo_previous_settings: null,
                    dmd_lora_name: ANIMA_TURBO_LORA_NAME,
                    dmd_lora_strength: 1.0,
                    lora_stack: [
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 },
                        { name: "", strength: 1.0 }
                    ],
                    mode_settings: {
                        illustrious: {
                            ckpt_name: "", sampler: "euler", scheduler: "normal",
                            steps: 20, cfg: 8.0, seed: generateRandomSeed(), seed_mode: "fixed",
                            dmd_lora_name: "", dmd_lora_strength: 1.0,
                            lora_stack: [
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 }
                            ]
                        },
                        anima: {
                            diffusion_model_name: "", clip_name: ANIMA_CLIP_NAME, vae_name: ANIMA_VAE_NAME,
                            clip_type: "stable_diffusion", sampler: "er_sde", scheduler: "simple",
                            steps: 30, cfg: 4.0, seed: generateRandomSeed(), seed_mode: "fixed",
                            turbo_enabled: false, turbo_previous_settings: null,
                            dmd_lora_name: ANIMA_TURBO_LORA_NAME, dmd_lora_strength: 1.0,
                            lora_stack: [
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 },
                                { name: "", strength: 1.0 }
                            ]
                        }
                    }
                };

                function parseGenerationSettings() {
                    try {
                        const parsed = generationSettingsWidget?.value ? JSON.parse(generationSettingsWidget.value) : {};
                        return { ...GENERATION_DEFAULTS, ...parsed };
                    } catch (e) {
                        return { ...GENERATION_DEFAULTS };
                    }
                }

                // State
                let state = {
                    character: charWidget ? charWidget.value : "",
                    costumes: [],
                    selectedCostumes: new Set(),
                    emotions: [],
                    selectedEmotions: new Set(),
                    searchTerm: "",
                    gen: parseGenerationSettings()
                };
                let receivedSerializedConfig = false;
                let characterFetchToken = 0;
                const CC_REPO_ID = "MIUProject/VNCCS_V2";
                const CC_CACHE_KEY = `vnccs_cc_cache_${CC_REPO_ID}`;
                let ccConfig = null;
                let ccDlStatus = {};
                let ccPollingInterval = null;
                let localAssets = {
                    checkpoints: [],
                    diffusion_models: [],
                    text_encoders: [],
                    vae_models: [],
                    loras: [],
                };
                const modelPickerOpen = { illustrious: false, anima: false };

                const ccNormalize = (value) => String(value || "").trim().toLowerCase();
                const ccKind = (entry) => ccNormalize(entry?.kind ?? entry?.Kind);
                const ccType = (entry) => ccNormalize(entry?.type ?? entry?.Type);
                const ccStatusKey = (cat, entry) => `cc_${cat}_${entry?.name || ""}`;
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
                const mergeCcAndLocalEntries = (ccList, localNames, folder, type, kind) => {
                    const localSet = localAssetRelSet(localNames);
                    const seen = new Set();
                    const merged = [];
                    (ccList || []).forEach(entry => {
                        const rel = ccRelPath(entry);
                        if (!rel) return;
                        seen.add(rel);
                        merged.push({ ...entry, status: localSet.has(rel) ? "installed" : entry.status });
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
                const ccResolveStatus = (entry, cat) => {
                    const transient = new Set(["queued", "downloading", "error", "auth_required"]);
                    const dls = ccDlStatus[ccStatusKey(cat, entry)] || {};
                    return transient.has(dls.status) ? dls.status : (entry?.status || "missing");
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

                function markNodeDirty() {
                    node.setDirtyCanvas?.(true, true);
                    app.graph?.setDirtyCanvas?.(true, true);
                }

                function commitWidget(widget, value, callCallback = true) {
                    if (!widget) return;
                    widget.value = value;
                    if (callCallback) widget.callback?.(value);
                    markNodeDirty();
                }

                function persistAllState() {
                    commitWidget(charWidget, state.character, false);
                    if (styleWidget) commitWidget(styleWidget, styleSelect.value, false);
                    if (costumesDataWidget) commitWidget(costumesDataWidget, JSON.stringify(Array.from(state.selectedCostumes)), false);
                    if (emotionsDataWidget) commitWidget(emotionsDataWidget, JSON.stringify(Array.from(state.selectedEmotions)), false);
                    saveGenerationSettings(false);
                }

                function saveGenerationSettings() {
                    const callCallback = arguments.length > 0 ? arguments[0] : true;
                    const mode = (state.gen.generation_mode || "anima").toLowerCase();
                    state.gen.mode_settings = state.gen.mode_settings || {};
                    state.gen.mode_settings[mode] = { ...state.gen };
                    delete state.gen.mode_settings[mode].mode_settings;

                    commitWidget(generationSettingsWidget, JSON.stringify(state.gen), callCallback);
                    commitWidget(modelWidget, mode === "anima" ? "Anima" : "Illustrious", callCallback);
                }

                function setGenerationMode(mode) {
                    mode = mode === "illustrious" ? "illustrious" : "anima";
                    const current = (state.gen.generation_mode || "anima").toLowerCase();
                    state.gen.mode_settings = state.gen.mode_settings || {};
                    state.gen.mode_settings[current] = { ...state.gen };
                    delete state.gen.mode_settings[current].mode_settings;

                    const profile = state.gen.mode_settings[mode] || GENERATION_DEFAULTS.mode_settings[mode] || {};
                    state.gen = { ...GENERATION_DEFAULTS, ...profile, mode_settings: state.gen.mode_settings, generation_mode: mode };
                    syncGenerationControls();
                    saveGenerationSettings();
                }

                function updateGenerationValue(key, value) {
                    state.gen[key] = value;
                    saveGenerationSettings();
                }

                function setAnimaTurboMode(enabled, loraName = ANIMA_TURBO_LORA_NAME) {
                    if ((state.gen.generation_mode || "anima").toLowerCase() !== "anima") return;
                    if (enabled) {
                        if (!state.gen.turbo_enabled) {
                            state.gen.turbo_previous_settings = {
                                steps: state.gen.steps,
                                cfg: state.gen.cfg,
                            };
                        }
                        state.gen.turbo_enabled = true;
                        state.gen.dmd_lora_name = loraName || ANIMA_TURBO_LORA_NAME;
                        state.gen.dmd_lora_strength = 1.0;
                        state.gen.steps = 12;
                        state.gen.cfg = 1.0;
                    } else {
                        state.gen.turbo_enabled = false;
                        const previous = state.gen.turbo_previous_settings || {};
                        if (previous.steps !== undefined) state.gen.steps = previous.steps;
                        if (previous.cfg !== undefined) state.gen.cfg = previous.cfg;
                        state.gen.turbo_previous_settings = null;
                    }
                    syncGenerationControls();
                    saveGenerationSettings();
                }

                const cardStatusLabel = (status, entry, cat) => {
                    const dls = ccDlStatus[ccStatusKey(cat, entry)] || {};
                    if (status === "installed") return "Installed";
                    if (status === "queued") return "Queued";
                    if (status === "downloading") return dls.message || "Downloading";
                    if (status === "auth_required") return "Key Required";
                    if (status === "error") return "Error";
                    return "Missing";
                };

                const ccDownloadEntry = async (cat, entry) => {
                    if (!entry?.name) return;
                    const key = ccStatusKey(cat, entry);
                    ccDlStatus[key] = { status: "queued", message: "Queued..." };
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

                const ensureAnimaDefaultAux = () => {
                    const clip = ccFirstEntry("clip", "Anima");
                    const vae = ccFirstEntry("vae", "Anima");
                    if (clip) state.gen.clip_name = ccRelPath(clip);
                    else if (!state.gen.clip_name) state.gen.clip_name = ANIMA_CLIP_NAME;
                    if (vae) state.gen.vae_name = ccRelPath(vae);
                    else if (!state.gen.vae_name) state.gen.vae_name = ANIMA_VAE_NAME;
                };

                const selectCcAsset = (key, rel) => {
                    if (!rel) return;
                    state.gen[key] = rel;
                    saveGenerationSettings();
                    renderControlCenterCards();
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
                    if ((state.gen.generation_mode || "anima").toLowerCase() === "anima") {
                        state.gen.dmd_lora_name = rel || state.gen.dmd_lora_name || "";
                        setAnimaTurboMode(enabled, rel || state.gen.dmd_lora_name || ANIMA_TURBO_LORA_NAME);
                    } else {
                        state.gen.dmd_lora_name = enabled ? rel : "";
                        state.gen.dmd_lora_strength = enabled ? 1.0 : 0.0;
                        saveGenerationSettings();
                    }
                    renderControlCenterCards();
                };

                const buildAssetCard = ({ entry, cat, selectedValue, onSelect, compact = false, toggled = false, onToggle = null, pickerHead = false, onDownload = null }) => {
                    const status = ccResolveStatus(entry, cat);
                    const rel = ccRelPath(entry);
                    const installed = status === "installed";
                    const selected = selectedValue && rel && String(selectedValue).replace(/\\/g, "/") === rel;
                    const progress = ["queued", "downloading"].includes(status);

                    const card = document.createElement("div");
                    card.className = "ems-model-card";
                    card.classList.toggle("is-installed", installed);
                    card.classList.toggle("is-selected", selected || toggled);
                    card.classList.toggle("is-missing", !installed);
                    if (installed || pickerHead) card.onclick = () => onSelect?.(rel, entry);

                    const top = document.createElement("div");
                    top.className = "ems-model-card-top";
                    const badge = document.createElement("span");
                    badge.className = "ems-model-dot " + (installed ? "ok" : progress ? "progress" : "missing");
                    top.appendChild(badge);
                    const name = document.createElement("div");
                    name.className = "ems-model-title";
                    name.textContent = entry.name || rel || "Unknown";
                    top.appendChild(name);
                    const statusEl = document.createElement("div");
                    statusEl.className = "ems-model-status " + (installed ? "ok" : progress ? "progress" : "missing");
                    statusEl.textContent = cardStatusLabel(status, entry, cat);
                    top.appendChild(statusEl);

                    if (onToggle && installed) {
                        const toggle = document.createElement("label");
                        toggle.className = "ems-toggle";
                        const input = document.createElement("input");
                        input.type = "checkbox";
                        input.checked = !!toggled;
                        input.onchange = (event) => {
                            event.stopPropagation();
                            onToggle(event.target.checked, rel, entry);
                        };
                        input.onclick = (event) => event.stopPropagation();
                        const track = document.createElement("span");
                        track.className = "ems-toggle-track";
                        const thumb = document.createElement("span");
                        thumb.className = "ems-toggle-thumb";
                        toggle.append(input, track, thumb);
                        top.appendChild(toggle);
                    }

                    card.appendChild(top);
                    if (entry.description && !compact) {
                        const desc = document.createElement("div");
                        desc.className = "ems-model-desc";
                        desc.textContent = entry.description;
                        card.appendChild(desc);
                    }
                    if (!installed) {
                        const btn = document.createElement("button");
                        btn.type = "button";
                        btn.className = "ems-model-card-download";
                        btn.textContent = status === "auth_required" ? "Enter Key in Control Center" : "Download";
                        btn.disabled = progress;
                        btn.onclick = (event) => {
                            event.stopPropagation();
                            if (!progress && status !== "auth_required") (onDownload || ccDownloadEntry)(cat, entry);
                        };
                        card.appendChild(btn);
                    }
                    return card;
                };

                function createField(label, input) {
                    const wrap = document.createElement("label");
                    wrap.className = "ems-field";
                    const title = document.createElement("div");
                    title.className = "ems-label";
                    title.innerText = label;
                    wrap.appendChild(title);
                    wrap.appendChild(input);
                    return wrap;
                }

                function createNumberInput(key, min, max, step) {
                    const input = document.createElement("input");
                    input.className = "ems-input";
                    input.type = "number";
                    input.min = min;
                    input.max = max;
                    input.step = step;
                    input.onchange = () => updateGenerationValue(key, step < 1 ? parseFloat(input.value) : parseInt(input.value));
                    return input;
                }

                function createSelectInput(key, values) {
                    const select = document.createElement("select");
                    select.className = "ems-input";
                    values.forEach(value => {
                        const opt = document.createElement("option");
                        opt.value = value;
                        opt.innerText = value;
                        select.appendChild(opt);
                    });
                    select.onchange = () => updateGenerationValue(key, select.value);
                    return select;
                }

                const generationEls = {};

                const renderModelPicker = ({ containerEl, entries, cat, key, mode, emptyText, onSelect, onDownload = null }) => {
                    if (!containerEl) return;
                    containerEl.innerHTML = "";
                    const picker = document.createElement("div");
                    picker.className = "ems-model-picker";
                    picker.classList.toggle("is-open", !!modelPickerOpen[mode]);
                    containerEl.appendChild(picker);

                    if (!entries.length) {
                        const empty = document.createElement("div");
                        empty.className = "ems-model-desc";
                        empty.textContent = emptyText;
                        picker.appendChild(empty);
                        return;
                    }

                    const current = String(state.gen[key] || "").replace(/\\/g, "/");
                    const selectedEntry = entries.find(entry => ccRelPath(entry) === current) || entries[0];
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
                    menu.className = "ems-model-picker-menu";
                    picker.appendChild(menu);

                    const appendGroup = (title, groupEntries) => {
                        if (!groupEntries.length) return;
                        const group = document.createElement("div");
                        group.className = "ems-model-picker-group";
                        const groupTitle = document.createElement("div");
                        groupTitle.className = "ems-model-picker-group-title";
                        groupTitle.textContent = title;
                        group.appendChild(groupTitle);
                        groupEntries.forEach(entry => {
                            group.appendChild(buildAssetCard({
                                entry,
                                cat,
                                selectedValue: state.gen[key] || "",
                                onSelect,
                                onDownload,
                            }));
                        });
                        menu.appendChild(group);
                    };

                    appendGroup("VNCCS Models", entries.filter(entry => entry.source !== "local"));
                    appendGroup("User Models", entries.filter(entry => entry.source === "local"));
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
                    if (!turboEntries.length) {
                        const fallback = {
                            name: "Anima Turbo LoRA",
                            type: "turbolora",
                            kind: "Anima",
                            local_path: `models/loras/${ANIMA_TURBO_LORA_NAME}`,
                            status: (localAssets.loras || []).map(x => String(x).replace(/\\/g, "/")).includes(ANIMA_TURBO_LORA_NAME.replace(/\\/g, "/")) ? "installed" : "missing",
                            description: "Anima turbo LoRA.",
                        };
                        turboEntries.push(fallback);
                    }

                    const label = document.createElement("div");
                    label.className = "ems-label";
                    label.textContent = "Turbo LoRA";
                    containerEl.appendChild(label);
                    turboEntries.forEach(entry => {
                        const rel = ccRelPath(entry);
                        const status = ccResolveStatus(entry, "lora");
                        const installed = status === "installed";
                        const progress = ["queued", "downloading"].includes(status);
                        const enabled = mode === "anima"
                            ? !!state.gen.turbo_enabled && String(state.gen.dmd_lora_name || "").replace(/\\/g, "/") === rel
                            : (state.gen.dmd_lora_strength || 0) > 0 && String(state.gen.dmd_lora_name || "").replace(/\\/g, "/") === rel;
                        const card = document.createElement("div");
                        card.className = "ems-lora-card" + (enabled ? " is-selected" : "");
                        if (installed) {
                            card.onclick = () => setCcTurboMode(!enabled, rel);
                        }

                        const name = document.createElement("div");
                        name.className = "ems-lora-name";
                        const dot = document.createElement("span");
                        dot.className = "ems-model-dot " + (installed ? "ok" : progress ? "progress" : "missing");
                        name.appendChild(dot);
                        name.appendChild(document.createTextNode(entry.name || rel || "Turbo LoRA"));

                        const statusEl = document.createElement("div");
                        statusEl.className = "ems-model-status " + (installed ? "ok" : progress ? "progress" : "missing");
                        statusEl.textContent = cardStatusLabel(status, entry, "lora");

                        card.appendChild(name);
                        card.appendChild(statusEl);

                        if (installed) {
                            const toggle = document.createElement("label");
                            toggle.className = "ems-toggle";
                            toggle.onclick = (event) => event.stopPropagation();
                            const input = document.createElement("input");
                            input.type = "checkbox";
                            input.checked = enabled;
                            input.onchange = (event) => {
                                event.stopPropagation();
                                setCcTurboMode(event.target.checked, rel);
                            };
                            const track = document.createElement("span");
                            track.className = "ems-toggle-track";
                            const thumb = document.createElement("span");
                            thumb.className = "ems-toggle-thumb";
                            toggle.append(input, track, thumb);
                            card.appendChild(toggle);
                        } else {
                            const btn = document.createElement("button");
                            btn.type = "button";
                            btn.className = "ems-model-card-download";
                            btn.textContent = status === "auth_required" ? "Enter Key" : "Download";
                            btn.disabled = progress;
                            btn.onclick = (event) => {
                                event.stopPropagation();
                                if (!progress && status !== "auth_required") ccDownloadEntry("lora", entry);
                            };
                            card.appendChild(btn);
                        }

                        containerEl.appendChild(card);
                    });
                };

                const renderControlCenterCards = () => {
                    if (!generationEls.animaModelCards || !generationEls.illustriousModelCards) return;
                    const currentMode = (state.gen.generation_mode || "anima").toLowerCase();
                    const isAnimaMode = currentMode === "anima";

                    const animaModels = mergeCcAndLocalEntries(
                        ccEntries("models", "Anima", entry => ccType(entry) === "unet"),
                        localAssets.diffusion_models,
                        "diffusion_models",
                        "unet",
                        "Anima",
                    );
                    renderModelPicker({
                        containerEl: generationEls.animaModelCards,
                        entries: animaModels,
                        cat: "models",
                        key: "diffusion_model_name",
                        mode: "anima",
                        emptyText: "No Anima diffusion models found.",
                        onSelect: rel => selectAnimaModel(rel),
                        onDownload: downloadAnimaBundle,
                    });
                    const currentAnima = String(state.gen.diffusion_model_name || "").replace(/\\/g, "/");
                    if (!currentAnima && animaModels[0]) selectAnimaModel(ccRelPath(animaModels[0]));
                    ensureAnimaDefaultAux();

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
                    renderModelPicker({
                        containerEl: generationEls.illustriousModelCards,
                        entries: illustriousCkpts,
                        cat: "models",
                        key: "ckpt_name",
                        mode: "illustrious",
                        emptyText: "No Illustrious checkpoints found.",
                        onSelect: rel => selectIllustriousModel(rel),
                    });
                    const currentIllustrious = String(state.gen.ckpt_name || "").replace(/\\/g, "/");
                    if (!currentIllustrious && illustriousCkpts[0]) selectCcAsset("ckpt_name", ccRelPath(illustriousCkpts[0]));

                    generationEls.animaModelCards.style.display = isAnimaMode ? "flex" : "none";
                    generationEls.illustriousModelCards.style.display = isAnimaMode ? "none" : "flex";
                    if (generationEls.animaLoraCards) renderModeLoraCards(generationEls.animaLoraCards, currentMode);
                };

                function populateSelect(select, values, includeNone = false) {
                    const current = select.value;
                    select.innerHTML = "";
                    if (includeNone) {
                        const opt = document.createElement("option");
                        opt.value = "";
                        opt.innerText = "None";
                        select.appendChild(opt);
                    }
                    (values || []).forEach(value => {
                        if (value === "") return;
                        const opt = document.createElement("option");
                        opt.value = value;
                        opt.innerText = value;
                        select.appendChild(opt);
                    });
                    if ([...select.options].some(opt => opt.value === current)) {
                        select.value = current;
                    }
                }

                async function loadGenerationAssets() {
                    try {
                        const response = await api.fetchApi("/vnccs/context_lists");
                        if (!response.ok) return;
                        const data = await response.json();
                        localAssets = {
                            checkpoints: data.checkpoints || [],
                            diffusion_models: data.diffusion_models || [],
                            text_encoders: data.text_encoders || [],
                            vae_models: data.vae_models || [],
                            loras: data.loras || [],
                        };
                        populateSelect(generationEls.sampler, data.samplers || ["euler", "er_sde"]);
                        populateSelect(generationEls.scheduler, data.schedulers || ["normal", "simple"]);
                        (generationEls.loraSelects || []).forEach(select => populateSelect(select, data.loras || [], true));

                        if (!state.gen.ckpt_name && data.checkpoints?.length) state.gen.ckpt_name = data.checkpoints[0];
                        if (!state.gen.diffusion_model_name && data.diffusion_models?.length) state.gen.diffusion_model_name = data.diffusion_models[0];
                        if (!state.gen.clip_name) {
                            state.gen.clip_name = (data.text_encoders || []).includes(ANIMA_CLIP_NAME) ? ANIMA_CLIP_NAME : (data.text_encoders?.[0] || ANIMA_CLIP_NAME);
                        }
                        if (!state.gen.vae_name) {
                            state.gen.vae_name = (data.vae_models || []).includes(ANIMA_VAE_NAME) ? ANIMA_VAE_NAME : (data.vae_models?.[0] || ANIMA_VAE_NAME);
                        }
                        syncGenerationControls();
                        saveGenerationSettings();
                        try {
                            await fetchCcConfig(false);
                        } catch (error) {
                            console.warn("[VNCCS Emotion Studio] Failed to load Control Center config", error);
                            renderControlCenterCards();
                        }
                    } catch (e) {
                        console.warn("[VNCCS Emotion Studio] Failed to load generation asset lists", e);
                    }
                }

                function syncGenerationControls() {
                    const mode = (state.gen.generation_mode || "anima").toLowerCase();
                    generationEls.tabIllustrious?.classList.toggle("active", mode === "illustrious");
                    generationEls.tabAnima?.classList.toggle("active", mode === "anima");
                    if (generationEls.animaModelCards) generationEls.animaModelCards.style.display = mode === "anima" ? "flex" : "none";
                    if (generationEls.illustriousModelCards) generationEls.illustriousModelCards.style.display = mode === "anima" ? "none" : "flex";
                    if (generationEls.steps) generationEls.steps.value = state.gen.steps ?? "";
                    if (generationEls.cfg) generationEls.cfg.value = state.gen.cfg ?? "";
                    if (generationEls.sampler) generationEls.sampler.value = state.gen.sampler || "euler";
                    if (generationEls.scheduler) generationEls.scheduler.value = state.gen.scheduler || "normal";
                    if (generationEls.seed) generationEls.seed.value = state.gen.seed ?? 0;
                    if (generationEls.seedMode) generationEls.seedMode.classList.toggle("active", (state.gen.seed_mode || "fixed") === "randomize");
                    if (generationEls.loraSection) generationEls.loraSection.style.display = mode === "anima" ? "flex" : "none";
                    if (generationEls.loraRows) {
                        generationEls.loraRows.forEach((row, index) => {
                            const item = (state.gen.lora_stack || [])[index] || { name: "", strength: 1.0 };
                            const name = item.name || "";
                            if (row.name && ![...row.name.options].some(opt => opt.value === name)) {
                                const opt = document.createElement("option");
                                opt.value = name;
                                opt.innerText = name || "None";
                                row.name.appendChild(opt);
                            }
                            if (row.name) row.name.value = name || "";
                            if (row.strength) row.strength.value = item.strength ?? 1;
                        });
                    }
                }

                // Create UI Container
                const container = document.createElement("div");
                container.className = "ems-container";

                // --- LEFT COL ---
                const leftCol = document.createElement("div");
                leftCol.className = "ems-left-col";

                // Character Header
                const charSection = document.createElement("div");
                charSection.className = "ems-section";
                charSection.style.flex = "1";

                const charHeader = document.createElement("div");
                charHeader.className = "ems-char-header";
                charHeader.innerText = "Character select";

                const previewContainer = document.createElement("div");
                previewContainer.className = "ems-char-preview-container";

                const charImg = document.createElement("img");
                charImg.className = "ems-char-preview";
                charImg.style.display = "none";
                previewContainer.appendChild(charImg);

                const charPlaceholder = document.createElement("div");
                charPlaceholder.style.cssText = "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:rgba(255,143,163,0.35);pointer-events:none;";
                charPlaceholder.innerHTML = `
                    <svg width="52" height="52" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="7" r="4" stroke="#ff8fa3" stroke-width="1.5"/>
                        <path d="M4 20c0-4 3.582-7 8-7s8 3 8 7" stroke="#ff8fa3" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;font-family:'Sora',sans-serif;">Character</div>
                `;
                previewContainer.appendChild(charPlaceholder);

                // Show/hide placeholder when image loads
                charImg.onload = () => { charImg.style.display = "block"; charPlaceholder.style.display = "none"; };
                charImg.onerror = () => { charImg.style.display = "none"; charPlaceholder.style.display = "flex"; };

                charSection.appendChild(charHeader);
                charSection.appendChild(previewContainer);
                leftCol.appendChild(charSection);

                container.appendChild(leftCol);

                // --- RIGHT COL ---
                const rightCol = document.createElement("div");
                rightCol.className = "ems-right-col";

                const selectionCol = document.createElement("div");
                selectionCol.className = "ems-selection-col";

                // Costumes
                const costumesSection = document.createElement("div");
                costumesSection.className = "ems-section";

                const costumesHeader = document.createElement("div");
                costumesHeader.className = "ems-costumes-header";
                costumesHeader.innerText = "Selected costumes";
                costumesSection.appendChild(costumesHeader);

                const costumesList = document.createElement("div");
                costumesList.className = "ems-costumes-list";
                costumesSection.appendChild(costumesList);

                selectionCol.appendChild(costumesSection);

                // Emotions
                const emotionsSection = document.createElement("div");
                emotionsSection.className = "ems-section";
                emotionsSection.style.flex = "1";

                const selectedEmotionsWrap = document.createElement("div");
                selectedEmotionsWrap.className = "ems-selected-emotions-wrap";
                const selectedEmotionsHeader = document.createElement("div");
                selectedEmotionsHeader.className = "ems-selected-emotions-header";
                selectedEmotionsHeader.innerText = "Selected emotions";
                const selectedEmotionsList = document.createElement("div");
                selectedEmotionsList.className = "ems-selected-emotions-list";
                selectedEmotionsWrap.appendChild(selectedEmotionsHeader);
                selectedEmotionsWrap.appendChild(selectedEmotionsList);
                emotionsSection.appendChild(selectedEmotionsWrap);

                const emotionsGrid = document.createElement("div");
                emotionsGrid.className = "ems-emotions-container";
                emotionsSection.appendChild(emotionsGrid);

                // Search Input
                const searchInput = document.createElement("input");
                searchInput.className = "ems-search-input";
                searchInput.placeholder = "Search emotions (name or description)...";
                searchInput.oninput = (e) => {
                    state.searchTerm = e.target.value;
                    renderEmotions();
                    updateButtonState();
                };
                emotionsSection.appendChild(searchInput);

                // Footer (Select All)
                const footer = document.createElement("div");
                footer.className = "ems-footer";
                const btnAll = document.createElement("button");
                btnAll.className = "ems-btn";
                btnAll.innerText = "Select ALL";
                footer.appendChild(btnAll);
                emotionsSection.appendChild(footer);

                selectionCol.appendChild(emotionsSection);

                const generationSection = document.createElement("div");
                generationSection.className = "ems-section ems-generation-section";

                const generationHeader = document.createElement("div");
                generationHeader.className = "ems-costumes-header";
                generationHeader.innerText = "Generation";
                generationSection.appendChild(generationHeader);

                const tabRow = document.createElement("div");
                tabRow.className = "ems-tab-row";
                const tabIllustrious = document.createElement("button");
                tabIllustrious.type = "button";
                tabIllustrious.className = "ems-tab";
                tabIllustrious.innerText = "Illustrious";
                tabIllustrious.onclick = () => setGenerationMode("illustrious");
                const tabAnima = document.createElement("button");
                tabAnima.type = "button";
                tabAnima.className = "ems-tab";
                tabAnima.innerText = "Anima";
                tabAnima.onclick = () => setGenerationMode("anima");
                generationEls.tabIllustrious = tabIllustrious;
                generationEls.tabAnima = tabAnima;
                tabRow.appendChild(tabIllustrious);
                tabRow.appendChild(tabAnima);
                generationSection.appendChild(tabRow);

                const illustriousModelCards = document.createElement("div");
                illustriousModelCards.className = "ems-model-card-list";
                generationEls.illustriousModelCards = illustriousModelCards;
                generationSection.appendChild(illustriousModelCards);

                const animaModelCards = document.createElement("div");
                animaModelCards.className = "ems-model-card-list";
                generationEls.animaModelCards = animaModelCards;
                generationSection.appendChild(animaModelCards);

                const genGrid = document.createElement("div");
                genGrid.className = "ems-generation-grid";
                generationEls.steps = createNumberInput("steps", 1, 100, 1);
                generationEls.sampler = createSelectInput("sampler", ["euler", "er_sde", "dpmpp_2m", "dpmpp_sde", "dpmpp_3m_sde"]);
                generationEls.cfg = createNumberInput("cfg", 0, 20, 0.1);
                generationEls.scheduler = createSelectInput("scheduler", ["normal", "simple", "karras", "exponential", "sgm_uniform"]);
                genGrid.appendChild(createField("Steps", generationEls.steps));
                genGrid.appendChild(createField("Sampler", generationEls.sampler));
                genGrid.appendChild(createField("CFG", generationEls.cfg));
                genGrid.appendChild(createField("Scheduler", generationEls.scheduler));
                generationSection.appendChild(genGrid);

                const seedInput = document.createElement("input");
                seedInput.className = "ems-input";
                seedInput.type = "number";
                seedInput.onchange = () => updateGenerationValue("seed", parseInt(seedInput.value || "0"));
                const seedDice = document.createElement("button");
                seedDice.type = "button";
                seedDice.className = "ems-seed-dice";
                seedDice.innerHTML = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <rect x="4" y="4" width="16" height="16" rx="3.5" stroke="currentColor" stroke-width="2"/>
                    <circle cx="8.5" cy="8.5" r="1.4" fill="currentColor"/>
                    <circle cx="15.5" cy="8.5" r="1.4" fill="currentColor"/>
                    <circle cx="12" cy="12" r="1.4" fill="currentColor"/>
                    <circle cx="8.5" cy="15.5" r="1.4" fill="currentColor"/>
                    <circle cx="15.5" cy="15.5" r="1.4" fill="currentColor"/>
                </svg>`;
                seedDice.onclick = () => {
                    state.gen.seed_mode = (state.gen.seed_mode || "fixed") === "randomize" ? "fixed" : "randomize";
                    if (state.gen.seed_mode === "randomize") state.gen.seed = generateRandomSeed();
                    syncGenerationControls();
                    saveGenerationSettings();
                };
                const seedRow = document.createElement("div");
                seedRow.className = "ems-seed-row";
                seedRow.appendChild(seedInput);
                seedRow.appendChild(seedDice);
                generationEls.seed = seedInput;
                generationEls.seedMode = seedDice;
                generationSection.appendChild(createField("Seed", seedRow));

                const loraSection = document.createElement("div");
                loraSection.className = "ems-lora-stack";
                generationEls.loraSelects = [];
                generationEls.loraRows = [];
                const loraHeader = document.createElement("div");
                loraHeader.className = "ems-costumes-header";
                loraHeader.innerText = "Anima LoRA Stack";
                loraSection.appendChild(loraHeader);
                const animaLoraCards = document.createElement("div");
                animaLoraCards.className = "ems-lora-stack";
                generationEls.animaLoraCards = animaLoraCards;
                loraSection.appendChild(animaLoraCards);
                for (let i = 0; i < 5; i++) {
                    const row = document.createElement("div");
                    row.className = "ems-lora-row";
                    const name = createSelectInput(`lora_${i}`, ["", "None"]);
                    const strength = createNumberInput(`lora_strength_${i}`, 0, 2, 0.05);
                    name.onchange = () => {
                        state.gen.lora_stack = state.gen.lora_stack || [];
                        state.gen.lora_stack[i] = state.gen.lora_stack[i] || { name: "", strength: 1.0 };
                        state.gen.lora_stack[i].name = name.value === "None" ? "" : name.value;
                        saveGenerationSettings();
                    };
                    strength.onchange = () => {
                        state.gen.lora_stack = state.gen.lora_stack || [];
                        state.gen.lora_stack[i] = state.gen.lora_stack[i] || { name: "", strength: 1.0 };
                        state.gen.lora_stack[i].strength = parseFloat(strength.value || "1");
                        saveGenerationSettings();
                    };
                    name.value = "None";
                    strength.value = "1";
                    generationEls.loraSelects.push(name);
                    generationEls.loraRows.push({ name, strength });
                    row.appendChild(name);
                    row.appendChild(strength);
                    loraSection.appendChild(row);
                }
                generationEls.loraSection = loraSection;
                generationSection.appendChild(loraSection);

                rightCol.appendChild(selectionCol);
                rightCol.appendChild(generationSection);
                syncGenerationControls();
                loadGenerationAssets();
                container.appendChild(rightCol);

                // Custom Select in Header
                const charSelect = document.createElement("select");
                charSelect.className = "em-select";

                charHeader.appendChild(charSelect);

                if (charWidget) {
                    if (charWidget.options.values) {
                        charWidget.options.values.forEach(v => {
                            const opt = document.createElement("option");
                            opt.value = v;
                            opt.innerText = v;
                            if (v === charWidget.value) opt.selected = true;
                            charSelect.appendChild(opt);
                        });
                    }
                    charSelect.onchange = () => {
                        state.character = charSelect.value;
                        commitWidget(charWidget, charSelect.value, false);
                        fetchCharacterData(charSelect.value);
                    };
                    charWidget.hidden = true;
                }

                // Add Prompt Style Select (Top)
                const styleWidget = node.widgets.find(w => w.name === "prompt_style");
                const normalizePromptStyle = (value) => value === "QWEN Style" ? "Anima" : value;

                const styleContainer = document.createElement("div");
                styleContainer.className = "ems-section";
                styleContainer.style.marginBottom = "10px";
                styleContainer.style.padding = "5px 10px";

                const styleSelect = document.createElement("select");
                styleSelect.className = "em-select";

                if (styleWidget && styleWidget.options.values) {
                    const seenStyles = new Set();
                    styleWidget.options.values.forEach(v => {
                        const normalized = normalizePromptStyle(v);
                        if (seenStyles.has(normalized)) return;
                        seenStyles.add(normalized);
                        const opt = document.createElement("option");
                        opt.value = normalized;
                        opt.innerText = normalized;
                        if (normalized === normalizePromptStyle(styleWidget.value)) opt.selected = true;
                        styleSelect.appendChild(opt);
                    });
                    if (styleWidget.value !== normalizePromptStyle(styleWidget.value)) {
                        commitWidget(styleWidget, normalizePromptStyle(styleWidget.value), false);
                    }
                    // Sync
                    styleSelect.onchange = () => {
                        commitWidget(styleWidget, normalizePromptStyle(styleSelect.value));
                    };
                    styleWidget.hidden = true;
                }
                styleContainer.appendChild(styleSelect);

                // Insert Style Container at the TOP of Left Col
                // Currently Left Col has charSection. We can prepend.
                leftCol.prepend(styleContainer);


                enableMiddleMouseCanvasPan(container);
                const widget = node.addDOMWidget("emotion_ui_v2", "ui", container, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() { return undefined; },
                    setValue(v) { }
                });
                syncDOMWidgetWidthSoon(node, "emotion_ui_v2");

                const origSerialize = node.onSerialize;
                node.onSerialize = function(info) {
                    persistAllState();
                    return origSerialize?.apply(this, arguments);
                };

                // Fix layout after tab switch: ComfyUI detaches/reattaches DOM widgets
                // without triggering onResize. Use ResizeObserver on the node's canvas
                // element to detect when the widget becomes visible again and reapply sizes.
                function applySize() {
                    const [w, h] = node.size;
                    container.style.width = (w - 20) + "px";
                    container.style.height = (h - 60) + "px";
                }

                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    applySize();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                function showAlert(html) {
                    const backdrop = document.createElement("div");
                    backdrop.className = "ems-modal-backdrop";

                    const modal = document.createElement("div");
                    modal.className = "ems-modal";

                    const text = document.createElement("div");
                    text.className = "ems-modal-text";
                    text.innerHTML = html;

                    const actions = document.createElement("div");
                    actions.className = "ems-modal-actions";

                    const btnOk = document.createElement("button");
                    btnOk.className = "ems-modal-btn ems-modal-btn--confirm";
                    btnOk.innerText = "OK";
                    btnOk.onclick = () => backdrop.remove();

                    actions.appendChild(btnOk);
                    modal.appendChild(text);
                    modal.appendChild(actions);
                    backdrop.appendChild(modal);
                    container.appendChild(backdrop);
                    btnOk.focus();
                }

                node._validateBeforeQueue = function() {
                    if (state.selectedCostumes.size === 0 && state.selectedEmotions.size === 0) {
                        showAlert(`<strong>Nothing selected</strong><br>Please select at least one costume and one emotion before running.`);
                        return false;
                    }
                    if (state.selectedCostumes.size === 0) {
                        showAlert(`<strong>No costumes selected</strong><br>Please enable at least one costume.`);
                        return false;
                    }
                    if (state.selectedEmotions.size === 0) {
                        showAlert(`<strong>No emotions selected</strong><br>Please select at least one emotion.`);
                        return false;
                    }
                    return true;
                };

                function showConfirm(html, onConfirm) {
                    const backdrop = document.createElement("div");
                    backdrop.className = "ems-modal-backdrop";

                    const modal = document.createElement("div");
                    modal.className = "ems-modal";

                    const text = document.createElement("div");
                    text.className = "ems-modal-text";
                    text.innerHTML = html;

                    const actions = document.createElement("div");
                    actions.className = "ems-modal-actions";

                    const btnCancel = document.createElement("button");
                    btnCancel.className = "ems-modal-btn ems-modal-btn--cancel";
                    btnCancel.innerText = "Cancel";
                    btnCancel.onclick = () => backdrop.remove();

                    const btnOk = document.createElement("button");
                    btnOk.className = "ems-modal-btn ems-modal-btn--confirm";
                    btnOk.innerText = "Proceed";
                    btnOk.onclick = () => { backdrop.remove(); onConfirm(); };

                    actions.appendChild(btnCancel);
                    actions.appendChild(btnOk);
                    modal.appendChild(text);
                    modal.appendChild(actions);
                    backdrop.appendChild(modal);
                    container.appendChild(backdrop);
                }

                function restoreStateFromWidgets() {
                    // 1. Character
                    if (charWidget && charWidget.value) {
                        state.character = charWidget.value;
                        if (![...charSelect.options].some(opt => opt.value === charWidget.value)) {
                            const opt = document.createElement("option");
                            opt.value = charWidget.value;
                            opt.innerText = charWidget.value;
                            charSelect.appendChild(opt);
                        }
                        charSelect.value = charWidget.value;
                        fetchCharacterData(state.character);
                    }

                    // 2. Style
                    if (styleWidget && styleWidget.value) {
                        const normalizedStyle = normalizePromptStyle(styleWidget.value);
                        if (![...styleSelect.options].some(opt => opt.value === normalizedStyle)) {
                            const opt = document.createElement("option");
                            opt.value = normalizedStyle;
                            opt.innerText = normalizedStyle;
                            styleSelect.appendChild(opt);
                        }
                        styleSelect.value = normalizedStyle;
                        if (styleWidget.value !== normalizedStyle) commitWidget(styleWidget, normalizedStyle, false);
                    }
                    if (modelWidget && modelWidget.value) {
                        state.gen.generation_mode = String(modelWidget.value).toLowerCase();
                    }
                    if (generationSettingsWidget && generationSettingsWidget.value) {
                        state.gen = parseGenerationSettings();
                    }
                    syncGenerationControls();

                    // 3. Costumes & Emotions (from hidden text strings)
                    if (costumesDataWidget && costumesDataWidget.value) {
                        try {
                            const savedCostumes = JSON.parse(costumesDataWidget.value);
                            state.selectedCostumes = new Set(savedCostumes);
                            renderCostumes();
                        } catch (e) { }
                    }
                    if (emotionsDataWidget && emotionsDataWidget.value) {
                        try {
                            const savedEmotions = JSON.parse(emotionsDataWidget.value);
                            state.selectedEmotions = new Set(savedEmotions);
                            // renderEmotions is called after fetch("/vnccs/get_emotions"), need to wait?
                            // No, renderEmotions() just needs state.emotions to be populated.
                            // The fetch happens async.
                        } catch (e) { }
                    }
                }

                node._vnccsEmotionRestoreFromWidgets = () => {
                    restoreStateFromWidgets();
                    renderEmotions();
                    renderCostumes();
                    updateButtonState();
                };
                node._vnccsEmotionApplySerializedInfo = (info) => {
                    const values = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
                    if (!values) return;
                    receivedSerializedConfig = true;
                    const widgetOrder = [
                        ["generation_model", modelWidget],
                        ["generation_settings", generationSettingsWidget],
                        ["prompt_style", styleWidget],
                        ["character", charWidget],
                        ["costumes_data", costumesDataWidget],
                        ["emotions_data", emotionsDataWidget],
                    ];
                    widgetOrder.forEach(([_, widget], index) => {
                        if (widget && values[index] !== undefined) widget.value = values[index];
                    });
                };
                node._vnccsEmotionPersistAllState = persistAllState;

                // Apply size on explicit resize too
                node.onResize = function (size) {
                    syncDOMWidgetWidth(node, "emotion_ui_v2");
                    requestAnimationFrame(() => syncDOMWidgetWidth(node, "emotion_ui_v2"));
                    applySize();
                }

                // Helper
                function getFilteredEmotions() {
                    if (!state.searchTerm) return state.emotions;
                    const term = state.searchTerm.toLowerCase();
                    return state.emotions.filter(e => {
                        const nameMatch = e.safe_name.toLowerCase().includes(term);
                        const descMatch = (e.description || "").toLowerCase().includes(term);
                        return nameMatch || descMatch;
                    });
                }

                // Button Logic & Text Update
                function updateButtonState() {
                    const filtered = getFilteredEmotions();
                    if (filtered.length === 0) {
                        btnAll.innerText = "No Emotions Found";
                        btnAll.disabled = true;
                        btnAll.classList.remove("cancel");
                        return;
                    }
                    btnAll.disabled = false;

                    const allFilteredSelected = filtered.every(e => state.selectedEmotions.has(e.safe_name));

                    if (allFilteredSelected) {
                        btnAll.innerText = "Cancel Selection";
                        btnAll.classList.add("cancel");
                    } else {
                        btnAll.innerText = "Select ALL";
                        btnAll.classList.remove("cancel");
                    }
                }

                btnAll.onclick = () => {
                    const filtered = getFilteredEmotions();
                    if (filtered.length === 0) return;

                    const allFilteredSelected = filtered.every(e => state.selectedEmotions.has(e.safe_name));

                    if (allFilteredSelected) {
                        // Deselect visible
                        filtered.forEach(e => state.selectedEmotions.delete(e.safe_name));
                        renderEmotions();
                        updateEmotionsData();
                    } else {
                        // Select All Visible
                        const numEmotions = filtered.length;
                        const numCostumes = state.selectedCostumes.size;
                        const total = numEmotions * numCostumes;

                        const html = `Select <strong>${numEmotions}</strong> visible emotion(s) for <strong>${numCostumes}</strong> costume(s)?<br>Total: <strong>${total}</strong> images.`;
                        showConfirm(html, () => {
                            filtered.forEach(e => state.selectedEmotions.add(e.safe_name));
                            renderEmotions();
                            updateEmotionsData();
                        });
                    }
                };

                // Functions
                function updateCostumesData() {
                    const list = Array.from(state.selectedCostumes);
                    commitWidget(costumesDataWidget, JSON.stringify(list));
                }

                function updateEmotionsData() {
                    const list = Array.from(state.selectedEmotions);
                    commitWidget(emotionsDataWidget, JSON.stringify(list));
                    updateButtonState();
                }

                function createEmotionCard(e, compact = false) {
                    const div = document.createElement("div");
                    const selected = state.selectedEmotions.has(e.safe_name);
                    div.className = "ems-emotion-item" + (selected ? " selected" : "") + (compact ? " compact" : "");
                    div.title = e.description || "";

                    const img = document.createElement("img");
                    img.className = "ems-emotion-img";
                    img.src = `/vnccs/get_emotion_image?name=${encodeURIComponent(e.safe_name)}`;
                    img.onerror = () => { img.style.display = 'none'; };

                    const lbl = document.createElement("div");
                    lbl.className = "ems-emotion-label";
                    lbl.innerText = e.safe_name;

                    div.appendChild(img);
                    div.appendChild(lbl);

                    div.onclick = () => {
                        if (state.selectedEmotions.has(e.safe_name)) {
                            state.selectedEmotions.delete(e.safe_name);
                        } else {
                            state.selectedEmotions.add(e.safe_name);
                        }
                        renderEmotions();
                        updateEmotionsData();
                    };

                    return div;
                }

                function renderCostumes() {
                    costumesList.innerHTML = "";
                    state.costumes.forEach(c => {
                        const lbl = document.createElement("label");
                        lbl.className = "ems-checkbox-item";

                        const toggle = document.createElement("div");
                        toggle.className = "em-toggle";

                        const chk = document.createElement("input");
                        chk.type = "checkbox";
                        chk.checked = state.selectedCostumes.has(c);
                        chk.onchange = () => {
                            if (chk.checked) state.selectedCostumes.add(c);
                            else state.selectedCostumes.delete(c);
                            updateCostumesData();
                        };

                        const track = document.createElement("div");
                        track.className = "em-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "em-toggle-thumb";

                        toggle.appendChild(chk);
                        toggle.appendChild(track);
                        toggle.appendChild(thumb);

                        const span = document.createElement("span");
                        span.innerText = c;

                        lbl.appendChild(toggle);
                        lbl.appendChild(span);
                        costumesList.appendChild(lbl);
                    });
                }

                function renderEmotions() {
                    selectedEmotionsList.innerHTML = "";
                    const selectedNames = Array.from(state.selectedEmotions);
                    selectedEmotionsHeader.innerText = `Selected emotions (${selectedNames.length})`;
                    if (selectedNames.length === 0) {
                        const empty = document.createElement("div");
                        empty.className = "ems-selected-emotions-empty";
                        empty.innerText = "No emotions selected";
                        selectedEmotionsList.appendChild(empty);
                    } else {
                        const byName = new Map(state.emotions.map(e => [e.safe_name, e]));
                        selectedNames.forEach(name => {
                            const emotion = byName.get(name) || { safe_name: name, description: "" };
                            selectedEmotionsList.appendChild(createEmotionCard(emotion, true));
                        });
                    }

                    emotionsGrid.innerHTML = "";
                    const filtered = getFilteredEmotions();
                    filtered.forEach(e => {
                        emotionsGrid.appendChild(createEmotionCard(e));
                    });
                    updateButtonState();
                }

                async function fetchCharacterData(charName) {
                    if (!charName || charName === "Character Name") return;
                    const token = ++characterFetchToken;

                    // Preview (randomize to force reload from disk)
                    charImg.src = `/vnccs/get_character_pose_preview?character=${encodeURIComponent(charName)}&t=${Date.now()}`;

                    // Costumes
                    try {
                        const res = await fetch(`/vnccs/get_character_costumes?character=${encodeURIComponent(charName)}`);
                        const validCostumes = await res.json();
                        if (token !== characterFetchToken || charName !== state.character) return;
                        state.costumes = validCostumes || [];

                        // FIX: Only reset to "all" if no saved selection exists
                        // Otherwise, filter saved selection to only include valid costumes
                        if (costumesDataWidget && costumesDataWidget.value) {
                            try {
                                const saved = JSON.parse(costumesDataWidget.value);
                                state.selectedCostumes = new Set(saved.filter(c => state.costumes.includes(c)));
                            } catch (e) {
                                state.selectedCostumes = new Set(state.costumes);
                            }
                        } else {
                            state.selectedCostumes = new Set(state.costumes);
                        }

                        renderCostumes();
                        updateCostumesData();
                    } catch (e) {
                        console.error("Error fetching costumes", e);
                    }
                }

                // Initial Load
                fetch("/vnccs/get_emotions").then(async (res) => {
                    if (res.ok) {
                        const data = await res.json();
                        let flat = [];
                        for (let cat in data) {
                            data[cat].forEach(e => flat.push({ ...e, category: cat }));
                        }
                        state.emotions = flat;
                        renderEmotions();

                        // NOW restore selection state (after list loaded)
                        if (!receivedSerializedConfig) restoreStateFromWidgets();
                        // Re-render to show restored selections
                        renderEmotions();
                        renderCostumes();
                        updateButtonState();
                    }
                });

                if (state.character) {
                    setTimeout(() => {
                        if (!receivedSerializedConfig) fetchCharacterData(state.character);
                    }, 50);
                }

                // Hook callback
                if (charWidget) {
                    const originalCb = charWidget.callback;
                    charWidget.callback = function (v) {
                        state.character = v;
                        if (charSelect.value !== v) charSelect.value = v;
                        fetchCharacterData(v);
                        if (originalCb) originalCb(v);
                    };
                }

                setTimeout(() => {
                    if (!receivedSerializedConfig) restoreStateFromWidgets();
                }, 50);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                setTimeout(() => {
                    this._vnccsEmotionApplySerializedInfo?.(info);
                    this._vnccsEmotionRestoreFromWidgets?.();
                }, 0);
                syncDOMWidgetWidth(this, "emotion_ui_v2");
                setTimeout(() => syncDOMWidgetWidth(this, "emotion_ui_v2"), 100);
            };
        }
    }
});
