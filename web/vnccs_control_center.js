// web/vnccs_control_center.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText } from "./vnccs_common.js";

// Global registry cache — prevents API storms when multiple CC nodes exist
window.VNCCS_CC_REGISTRY = window.VNCCS_CC_REGISTRY || {};
window.VNCCS_CC_FETCH_PROMISES = window.VNCCS_CC_FETCH_PROMISES || {};

const INITIAL_NODE_W = 300;
const INITIAL_NODE_H = 700;

const DEFAULT_SAMPLERS = [
    "euler","euler_cfg_pp","euler_ancestral","euler_ancestral_cfg_pp",
    "heun","heunpp2","dpm_2","dpm_2_ancestral","lms","dpm_fast","dpm_adaptive",
    "dpmpp_2s_ancestral","dpmpp_sde","dpmpp_2m","dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde","dpmpp_3m_sde","ddpm","lcm","ddim","uni_pc",
];
const DEFAULT_SCHEDULERS = [
    "normal","karras","exponential","sgm_uniform","simple",
    "ddim_uniform","beta","linear_quadratic","cosine",
    "align_your_steps","gits",
];
const DEFAULT_MODEL_STEPS = 4;
const DEFAULT_MODEL_CFG = 1.0;
const DEFAULT_MODEL_SCHEDULER = "simple";
const PENDING_DEPENDENCY_INSTALLS_KEY = "vnccs-control-center-pending-dependency-installs";
const MODEL_FAMILIES = [
    { kind: "QIE2511", label: "QIE2511", defaultType: "gguf" },
    { kind: "Klein9b", label: "Flux Klein9b", defaultType: "unet" },
];

// ─── CSS injection (once per page load) ──────────────────────────────────────

function _injectVNCCSControlCenterStyles() {
    if (document.getElementById("vnccs-cc-styles")) return;
    const s = document.createElement("style");
    s.id = "vnccs-cc-styles";
    s.textContent = `
/* ── VNCCS Control Center – Sakura Design ─────────────────────────────── */

.vnccs-cc-root {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    background: #0a0a0f;
    font-family: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: hidden;
    box-sizing: border-box;
    position: relative;
    color: #e8e8f0;
}

/* Header */
.vnccs-cc-header {
    padding: 6px 10px;
    background: #12121a;
    border-bottom: 1px solid rgba(255,143,163,0.15);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}
.vnccs-cc-title {
    font-size: 11px;
    font-weight: 700;
    color: #e8e8f0;
    letter-spacing: 0.06em;
}
.vnccs-cc-header-btns {
    display: flex;
    gap: 5px;
}

/* Buttons */
.vnccs-cc-btn {
    background: rgba(255,255,255,0.04);
    color: #9898a8;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 3px 8px;
    cursor: pointer;
    font-size: 11px;
    font-family: inherit;
    transition: all 0.18s ease;
    line-height: 1.4;
}
.vnccs-cc-btn:hover {
    background: rgba(255,143,163,0.1);
    border-color: rgba(255,143,163,0.3);
    color: #ff8fa3;
}
.vnccs-cc-btn:focus,
.vnccs-cc-btn:focus-visible,
.vnccs-cc-btn:active {
    outline: none;
    background: rgba(255,143,163,0.1);
    border-color: rgba(255,143,163,0.38);
    color: #ff8fa3;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.22);
}
.vnccs-cc-btn--active {
    color: #00d68f;
    border-color: rgba(0,214,143,0.35);
}
.vnccs-cc-btn--clip-active {
    color: #b8a9e8;
    border-color: rgba(184,169,232,0.35);
}
.vnccs-cc-btn--cnet-active {
    color: #ff8fa3;
    border-color: rgba(255,143,163,0.4);
}
.vnccs-cc-btn--download-all {
    width: 100%;
    font-weight: 700;
    background: linear-gradient(135deg, #ff8fa3 0%, #ffb6c8 100%);
    color: #1a1525;
    border: none;
    padding: 6px 10px;
    font-size: 11px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.18s ease;
    font-family: inherit;
    letter-spacing: 0.05em;
}
.vnccs-cc-btn--download-all:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(255,143,163,0.4);
}
.vnccs-cc-btn--download-all:focus,
.vnccs-cc-btn--download-all:focus-visible,
.vnccs-cc-btn--download-all:active {
    outline: none;
    background: linear-gradient(135deg, #ff8fa3 0%, #ffb6c8 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 4px 16px rgba(255,143,163,0.4), 0 0 0 2px rgba(255,143,163,0.22);
}
.vnccs-cc-btn--save {
    background: rgba(0,214,143,0.1);
    color: #00d68f;
    border-color: rgba(0,214,143,0.25);
}
.vnccs-cc-btn--save:hover {
    background: rgba(0,214,143,0.2);
    border-color: rgba(0,214,143,0.5);
    color: #00d68f;
}

/* Scroll */
.vnccs-cc-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 6px 7px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,143,163,0.2) transparent;
}

/* Footer */
.vnccs-cc-footer {
    padding: 6px 7px;
    background: #12121a;
    border-top: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
}

/* States */
.vnccs-cc-empty {
    color: #5e5e70;
    text-align: center;
    margin-top: 40px;
    font-size: 11px;
}
.vnccs-cc-error {
    color: #ff4757;
    padding: 10px;
    font-size: 11px;
}

/* Collapsible block */
.vnccs-cc-block {
    border: 1px solid rgba(255,143,163,0.13);
    border-radius: 10px;
    overflow: clip;
    background: rgba(10,10,15,0.5);
}
.vnccs-cc-block-hdr {
    padding: 5px 10px;
    background: rgba(26,26,38,0.95);
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    font-weight: 700;
    color: #ff8fa3;
    user-select: none;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,143,163,0.07);
    transition: background 0.15s ease;
}
.vnccs-cc-block-hdr:hover {
    background: rgba(255,143,163,0.07);
}
.vnccs-cc-block-hdr-count {
    color: #5e5e70;
    font-weight: 400;
    font-size: 9px;
    margin-left: 4px;
    letter-spacing: 0;
}
.vnccs-cc-block-arrow {
    font-size: 9px;
    color: #5e5e70;
}
.vnccs-cc-block-body {
    padding: 4px 6px 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: rgba(8,8,12,0.5);
    overflow: visible;
}
.vnccs-cc-block-empty {
    color: #5e5e70;
    font-size: 10px;
    padding: 4px;
}
.vnccs-cc-card-grid {
    padding: 6px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    background: rgba(8,8,12,0.5);
}
.vnccs-cc-card-grid--triple {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

/* Two-column layout */
.vnccs-cc-twocol {
    display: flex;
    background: rgba(8,8,12,0.5);
}
.vnccs-cc-twocol-left {
    flex: 1;
    padding: 4px 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    border-right: 1px solid rgba(255,255,255,0.04);
    min-width: 0;
}
.vnccs-cc-twocol-left > .vnccs-cc-model-card {
    flex: 1;
    box-sizing: border-box;
}
.vnccs-cc-twocol-right,
.vnccs-cc-twocol-right2 {
    width: 130px;
    flex-shrink: 0;
    padding: 7px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(26,26,38,0.3);
}
.vnccs-cc-twocol-right2 {
    border-left: 1px solid rgba(255,255,255,0.04);
}

/* Entry rows */
.vnccs-cc-row {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 5px;
    border-radius: 6px;
    font-size: 10px;
    background: rgba(18,18,26,0.5);
    border-left: 2px solid transparent;
    transition: background 0.15s ease;
    position: relative;
    overflow: hidden;
}
.vnccs-cc-row:hover {
    background: rgba(34,34,46,0.7);
}
.vnccs-cc-row-bg {
    position: absolute;
    top: 0; left: 0; height: 100%; width: 0%;
    z-index: 0;
    pointer-events: none;
    transition: width 0.3s linear;
}
.vnccs-cc-row > *:not(.vnccs-cc-row-bg) {
    position: relative;
    z-index: 1;
}
.vnccs-cc-row--model-sel {
    border-left-color: #00d68f;
    background: rgba(0,214,143,0.06);
}
.vnccs-cc-row--clip-sel {
    border-left-color: #b8a9e8;
    background: rgba(184,169,232,0.06);
}
.vnccs-cc-row--cnet-sel {
    border-left-color: #ff8fa3;
    background: rgba(255,143,163,0.06);
}
.vnccs-cc-row-name-wrap {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
}
.vnccs-cc-row-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #e8e8f0;
}
.vnccs-cc-row-desc {
    font-size: 9px;
    color: #5e5e70;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-cc-row-tag {
    color: #5e5e70;
    font-size: 9px;
    white-space: nowrap;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.vnccs-cc-row-progress {
    color: #b8a9e8;
    font-size: 9px;
    white-space: nowrap;
}

/* Status badges */
.vnccs-cc-badge { font-size: 12px; flex-shrink: 0; }
.vnccs-cc-badge--installed   { color: #00d68f; }
.vnccs-cc-badge--missing     { color: #ff4757; }
.vnccs-cc-badge--queued      { color: #b8a9e8; }
.vnccs-cc-badge--downloading { color: #b8a9e8; }
.vnccs-cc-badge--error       { color: #ff4757; }
.vnccs-cc-badge--outdated    { color: #ffaa00; }

/* Model card */
.vnccs-cc-model-card {
    border: 1px solid rgba(0,214,143,0.25);
    border-radius: 10px;
    background: rgba(0,214,143,0.05);
    padding: 10px 12px 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    position: relative;
    overflow: hidden;
}
.vnccs-cc-model-card--placeholder {
    border-color: rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    min-height: 108px;
    justify-content: center;
}
.vnccs-cc-model-card-placeholder-text {
    font-size: 11px;
    line-height: 1.45;
    color: #9da3b8;
    text-align: center;
}
.vnccs-cc-family-tabs {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 5px;
    padding: 6px;
    margin-bottom: 6px;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    background: #111119;
}
.vnccs-cc-family-tab {
    min-width: 0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    background: #0d0d13;
    color: #8e93a8;
    padding: 8px 6px;
    font: 700 10px/1.2 inherit;
    letter-spacing: 0;
    cursor: pointer;
}
.vnccs-cc-family-tab:hover {
    border-color: rgba(255,143,163,0.3);
    color: #f4c2ce;
}
.vnccs-cc-family-tab--active {
    border-color: rgba(255,143,163,0.52);
    background: rgba(255,143,163,0.16);
    color: #fff3f6;
}
.vnccs-cc-model-tabs {
    display: grid;
    grid-template-columns: repeat(var(--vnccs-model-tab-count, 1), minmax(0, 1fr));
    gap: 6px;
    padding: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    background: rgba(8,8,12,0.5);
}
.vnccs-cc-model-tab {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    background: rgba(18,18,26,0.8);
    color: #8e93a8;
    padding: 7px 6px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.18s ease;
}
.vnccs-cc-model-tab:hover {
    border-color: rgba(255,143,163,0.28);
    color: #f4c2ce;
}
.vnccs-cc-model-tab--active {
    background: linear-gradient(135deg, rgba(255,143,163,0.22) 0%, rgba(184,169,232,0.16) 100%);
    border-color: rgba(255,143,163,0.4);
    color: #fff3f6;
    box-shadow: inset 0 0 0 1px rgba(255,143,163,0.12);
}
.vnccs-cc-model-tab--missing {
    opacity: 0.45;
}
.vnccs-cc-model-card-top {
    display: flex;
    align-items: center;
    gap: 7px;
}
.vnccs-cc-model-card-top--start {
    align-items: flex-start;
}
.vnccs-cc-model-card-kicker {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #b8a9e8;
}
.vnccs-cc-model-card-name {
    font-size: 12px;
    font-weight: 700;
    color: #e8e8f0;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-cc-model-card-name--wrap {
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    line-height: 1.25;
}
.vnccs-cc-model-card-status {
    font-size: 10px;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
}
.vnccs-cc-model-card-status--ok      { color: #00d68f; }
.vnccs-cc-model-card-status--missing  { color: #ff4757; }
.vnccs-cc-model-card-status--progress { color: #b8a9e8; }
.vnccs-cc-model-card-desc {
    font-size: 10px;
    color: #9898a8;
    line-height: 1.4;
}
.vnccs-cc-model-card-footer {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 2px;
}
.vnccs-cc-model-inline-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 10px 10px;
    border-top: 1px solid rgba(255,255,255,0.04);
    background: rgba(8,8,12,0.42);
}
.vnccs-cc-model-inline-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #ff8fa3;
    padding: 0 2px;
}
.vnccs-cc-turbo-strip {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 42px;
    padding: 0 12px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(18,18,26,0.8);
    transition: border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
    position: relative;
}
.vnccs-cc-turbo-strip.is-installed { cursor: pointer; }
.vnccs-cc-turbo-strip.is-installed:hover {
    border-color: rgba(255,143,163,0.28);
}
.vnccs-cc-turbo-strip.is-active {
    border-color: rgba(255,143,163,0.46);
    background: linear-gradient(135deg, rgba(255,143,163,0.14) 0%, rgba(0,214,143,0.08) 100%);
    box-shadow: inset 0 0 0 1px rgba(255,143,163,0.12);
}
.vnccs-cc-turbo-strip-name {
    flex: 1;
    min-width: 0;
    font-size: 10.5px;
    font-weight: 700;
    color: #e8e8f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vnccs-cc-turbo-strip-status {
    font-size: 8.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
    flex-shrink: 0;
}
.vnccs-cc-turbo-strip-status--ok { color: #00d68f; }
.vnccs-cc-turbo-strip-status--active { color: #ff8fa3; }
.vnccs-cc-turbo-strip-status--missing { color: #ff4757; }
.vnccs-cc-turbo-strip-status--progress { color: #b8a9e8; }
.vnccs-cc-toggle {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-cc-toggle input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    margin: 0;
}
.vnccs-cc-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.08);
    transition: border-color 0.16s ease, background 0.16s ease;
}
.vnccs-cc-toggle-thumb {
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 999px;
    left: 3px;
    top: 3px;
    background: #cfcfe6;
    transition: transform 0.16s ease, background 0.16s ease;
}
.vnccs-cc-toggle input:checked ~ .vnccs-cc-toggle-track {
    border-color: rgba(255,143,163,0.42);
    background: rgba(255,143,163,0.16);
}
.vnccs-cc-toggle input:checked ~ .vnccs-cc-toggle-thumb {
    transform: translateX(14px);
    background: #ff8fa3;
}
.vnccs-cc-model-card-footer--stack {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
}
.vnccs-cc-lora-sections {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 6px;
    background: rgba(8,8,12,0.5);
}
.vnccs-cc-lora-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.vnccs-cc-lora-add-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 2px;
}
.vnccs-cc-lora-add-btn {
    width: 100%;
    min-height: 30px;
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid rgba(255, 143, 163, 0.22);
    background: rgba(255, 143, 163, 0.08);
    color: #ff8fa3;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background 0.16s ease, border-color 0.16s ease, transform 0.16s ease;
}
.vnccs-cc-lora-add-btn:hover {
    background: rgba(255, 143, 163, 0.14);
    border-color: rgba(255, 143, 163, 0.36);
    transform: translateY(-1px);
}
.vnccs-cc-lora-add-note {
    font-size: 8.5px;
    color: #8d8ca1;
    line-height: 1.35;
    padding: 0 2px;
}
.vnccs-cc-lora-section-title {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #ff8fa3;
    padding: 0 2px;
}
.vnccs-cc-lora-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 6px;
}
.vnccs-cc-lora-grid--compact {
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    align-items: start;
}
.vnccs-cc-lora-card {
    gap: 5px;
    padding: 7px 8px;
    cursor: pointer;
    transition: border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
}
.vnccs-cc-lora-card:hover {
    border-color: rgba(255,143,163,0.32);
    transform: translateY(-1px);
}
.vnccs-cc-lora-card--active {
    border-color: rgba(255,143,163,0.52);
    background: linear-gradient(180deg, rgba(255,143,163,0.14) 0%, rgba(10,26,22,0.92) 100%);
    box-shadow: inset 0 0 0 1px rgba(255,143,163,0.14);
    padding-right: 72px;
}
.vnccs-cc-lora-card--has-remove {
    padding-right: 110px;
}
.vnccs-cc-lora-card--compact {
    padding: 4px 52px 4px 7px;
    gap: 2px;
    height: 56px;
    min-height: 56px;
    max-height: 56px;
    overflow: hidden;
}
.vnccs-cc-lora-card--compact.vnccs-cc-lora-card--downloadable {
    padding-right: 78px;
}
.vnccs-cc-lora-card--compact .vnccs-cc-lora-card-meta {
    gap: 1px;
}
.vnccs-cc-lora-card-top {
    display: flex;
    align-items: flex-start;
    gap: 6px;
}
.vnccs-cc-lora-card-meta {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex: 1;
    min-width: 0;
}
.vnccs-cc-lora-card-name {
    font-size: 10.5px;
    font-weight: 700;
    color: #e8e8f0;
    line-height: 1.2;
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
}
.vnccs-cc-lora-card--compact .vnccs-cc-lora-card-name {
    font-size: 10px;
}
.vnccs-cc-lora-card-status {
    font-size: 8.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.vnccs-cc-lora-card-status--active {
    color: #ff8fa3;
}
.vnccs-cc-lora-card-status--pipe {
    color: #00d68f;
}
.vnccs-cc-lora-card-status--corner {
    position: absolute;
    top: 6px;
    right: 8px;
    line-height: 1;
}
.vnccs-cc-lora-card--has-remove .vnccs-cc-lora-card-status--corner {
    right: 42px;
}
.vnccs-cc-lora-remove-btn {
    position: absolute;
    top: 4px;
    right: 6px;
    min-width: 26px;
    height: 20px;
    border-radius: 6px;
    border: 1px solid rgba(255, 71, 87, 0.24);
    background: rgba(255, 71, 87, 0.08);
    color: #ff7a86;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
    padding: 0 6px;
    line-height: 1;
}
.vnccs-cc-lora-remove-btn:hover {
    background: rgba(255, 71, 87, 0.16);
    border-color: rgba(255, 71, 87, 0.42);
}
.vnccs-cc-lora-download-btn {
    position: absolute;
    right: 7px;
    top: 5px;
    min-width: 58px;
    height: 20px;
    padding: 0 7px;
    border-radius: 6px;
    border: 1px solid rgba(255,143,163,0.32);
    background: rgba(255,143,163,0.08);
    color: #ff8fa3;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1;
    cursor: pointer;
    font-family: inherit;
}
.vnccs-cc-lora-download-btn:hover {
    background: rgba(255,143,163,0.2);
    border-color: rgba(255,143,163,0.52);
}
.vnccs-cc-lora-card-desc {
    font-size: 8.5px;
    color: #a3a3b3;
    line-height: 1.3;
    white-space: normal;
    overflow: visible;
}
.vnccs-cc-lora-card-detail-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
}
.vnccs-cc-lora-card-detail-row .vnccs-cc-lora-card-desc {
    flex: 1;
    min-width: 0;
}
.vnccs-cc-lora-card-version {
    flex-shrink: 0;
    color: #777889;
    font-size: 7.5px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    line-height: 1;
    white-space: nowrap;
}
.vnccs-cc-lora-card--compact .vnccs-cc-lora-card-desc {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    overflow: hidden;
}
.vnccs-cc-lora-card-actions {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: auto;
}
.vnccs-cc-lora-strength-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: auto;
}
.vnccs-cc-model-card-note {
    font-size: 9px;
    color: #5e5e70;
    letter-spacing: 0.03em;
}
.vnccs-cc-model-card-select {
    flex: 1;
    background: rgba(255,255,255,0.05);
    color: #e8e8f0;
    border: 1px solid rgba(0,214,143,0.2);
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color-scheme: dark;
}
.vnccs-cc-model-card-select option { background: #1e1e2e; color: #e8e8f0; }
.vnccs-cc-model-card-select:focus {
    outline: none;
    border-color: rgba(0,214,143,0.5);
}

/* Divider */
.vnccs-cc-divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 3px 0;
}

/* Params panel */
.vnccs-cc-params {
    display: grid;
    grid-template-rows: repeat(2, minmax(0, 1fr));
    gap: 12px;
    align-content: start;
}
.vnccs-cc-params--right {
    display: flex;
    flex-direction: column;
    gap: 12px;
    align-content: normal;
}
.vnccs-cc-params--right .vnccs-cc-select {
    -webkit-appearance: none;
    appearance: none;
    height: auto;
    min-height: 0;
    line-height: 1.2;
    padding: 3px 30px 3px 4px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='20' viewBox='0 0 12 20'%3E%3Cpath d='M1 8l5-5 5 5' fill='none' stroke='%23f1eef7' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M1 12l5 5 5-5' fill='none' stroke='%23f1eef7' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 8px center;
    background-size: 12px 20px;
}
.vnccs-cc-param-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.vnccs-cc-param-label {
    font-size: 9px;
    color: #5e5e70;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.vnccs-cc-input {
    width: 100%;
    background: rgba(255,255,255,0.04);
    color: #e8e8f0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 5px;
    padding: 3px 5px;
    font-size: 10px;
    box-sizing: border-box;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    transition: border-color 0.18s;
}
.vnccs-cc-input:focus {
    outline: none;
    border-color: rgba(255,143,163,0.35);
}
.vnccs-cc-select {
    width: 100%;
    background: rgba(255,255,255,0.04);
    color: #e8e8f0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 5px;
    padding: 3px 4px;
    font-size: 10px;
    box-sizing: border-box;
    transition: border-color 0.18s;
    color-scheme: dark;
}
.vnccs-cc-select option { background: #1e1e2e; color: #e8e8f0; }
.vnccs-cc-select:focus {
    outline: none;
    border-color: rgba(255,143,163,0.35);
}

.vnccs-cc-lora-slider {
    width: 100%;
    accent-color: #ff8fa3;
    flex: 1;
}
.vnccs-cc-lora-val {
    min-width: 32px;
    text-align: right;
    color: #c3a2ab;
    font-size: 9px;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Settings overlay */
.vnccs-cc-settings-overlay {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.88);
    backdrop-filter: blur(8px);
    z-index: 100;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
    box-sizing: border-box;
}
.vnccs-cc-settings-panel {
    background: rgba(14,10,22,0.97);
    border: 1px solid rgba(255,143,163,0.22);
    border-radius: 12px;
    padding: 14px;
    width: 100%;
    max-width: 340px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    max-height: 90%;
    overflow-y: auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.55);
    scrollbar-width: thin;
    scrollbar-color: rgba(255,143,163,0.2) transparent;
}
.vnccs-cc-settings-title {
    margin: 0;
    color: #ff8fa3;
    font-size: 12px;
    font-weight: 700;
    border-bottom: 1px solid rgba(255,143,163,0.15);
    padding-bottom: 8px;
    letter-spacing: 0.06em;
}
.vnccs-cc-settings-details summary {
    font-size: 10px;
    color: #9898a8;
    cursor: pointer;
    padding: 4px 0;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    list-style: none;
    transition: color 0.15s;
}
.vnccs-cc-settings-details summary:hover { color: #ff8fa3; }
.vnccs-cc-settings-field {
    margin-top: 6px;
}
.vnccs-cc-settings-label {
    font-size: 9px;
    color: #5e5e70;
    display: block;
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.vnccs-cc-settings-select {
    width: 100%;
    background: rgba(255,255,255,0.04);
    color: #e8e8f0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 5px 6px;
    font-size: 10px;
    transition: border-color 0.18s;
    color-scheme: dark;
}
.vnccs-cc-settings-select option { background: #1e1e2e; color: #e8e8f0; }
.vnccs-cc-settings-select:focus {
    outline: none;
    border-color: rgba(255,143,163,0.35);
}
.vnccs-cc-settings-input {
    width: 100%;
    background: rgba(255,255,255,0.04);
    color: #e8e8f0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 5px 7px;
    font-size: 10px;
    box-sizing: border-box;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    transition: border-color 0.18s;
}
.vnccs-cc-settings-input:focus {
    outline: none;
    border-color: rgba(255,143,163,0.35);
}
.vnccs-cc-settings-btns {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
    padding-top: 4px;
}
.vnccs-cc-deps-modal-title {
    font-size: 14px;
    font-weight: 800;
    color: #ffe8ef;
    margin-bottom: 8px;
}
.vnccs-cc-deps-modal-text {
    font-size: 11px;
    line-height: 1.45;
    color: #b8b8c8;
    margin-bottom: 12px;
}
.vnccs-cc-deps-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 14px;
}
.vnccs-cc-deps-item {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
    align-items: center;
    padding: 8px;
    border: 1px solid rgba(255,143,163,0.16);
    background: rgba(255,255,255,0.03);
    border-radius: 6px;
}
.vnccs-cc-deps-name {
    font-size: 11px;
    font-weight: 800;
    color: #f3f3fb;
}
.vnccs-cc-deps-missing {
    margin-top: 3px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 9px;
    color: #ffaa00;
    word-break: break-word;
}
.vnccs-cc-deps-actions {
    display: flex;
    gap: 6px;
    align-items: center;
}
.vnccs-cc-link-btn {
    border: 1px solid rgba(0,214,143,0.28);
    background: rgba(0,214,143,0.1);
    color: #00d68f;
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 10px;
    font-weight: 800;
    cursor: pointer;
    white-space: nowrap;
}
.vnccs-cc-link-btn:hover {
    background: rgba(0,214,143,0.16);
}

/* ── Module status strip ───────────────────────────────────────────────── */
.vnccs-cc-status-strip {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    background: #0d0d16;
    border-bottom: 1px solid rgba(255,143,163,0.08);
}
.vnccs-cc-status-pills {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    padding: 4px 10px 5px;
    min-width: 0;
}
.vnccs-cc-status-label {
    font-size: 9px;
    font-weight: 700;
    color: #5e5e70;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: 2px;
    flex-shrink: 0;
}
.vnccs-cc-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 20px;
    font-size: 9px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all 0.18s ease;
    white-space: nowrap;
}
.vnccs-cc-pill--dependency {
    padding-inline: 6px;
    max-width: 92px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vnccs-cc-pill--ok {
    background: rgba(0,214,143,0.08);
    border-color: rgba(0,214,143,0.22);
    color: #00d68f;
}
.vnccs-cc-pill--update {
    background: rgba(255,170,0,0.1);
    border-color: rgba(255,170,0,0.3);
    color: #ffaa00;
    cursor: default;
}
.vnccs-cc-pill--error {
    background: rgba(255,71,87,0.08);
    border-color: rgba(255,71,87,0.2);
    color: #ff4757;
}
.vnccs-cc-pill--loading {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.06);
    color: #5e5e70;
}
.vnccs-cc-pill--dup {
    background: rgba(255,170,0,0.08);
    border-color: rgba(255,170,0,0.25);
    color: #ffaa00;
}
.vnccs-cc-pill--partial {
    background: rgba(255,170,0,0.08);
    border-color: rgba(255,170,0,0.25);
    color: #ffaa00;
}
.vnccs-cc-pill--warning {
    background: rgba(255,170,0,0.08);
    border-color: rgba(255,170,0,0.25);
    color: #ffaa00;
}
.vnccs-cc-update-banner {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px 4px;
    background: rgba(255,170,0,0.07);
    border-top: 1px solid rgba(255,170,0,0.12);
    font-size: 9px;
    color: #ffaa00;
    font-weight: 600;
    letter-spacing: 0.04em;
}
`;

    document.head.appendChild(s);
}

// ─── Node Registration ────────────────────────────────────────────────────────

app.registerExtension({
    name: "VNCCS.ControlCenter",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "VNCCS_ControlCenter") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.setSize([INITIAL_NODE_W, INITIAL_NODE_H]);
            syncDOMWidgetWidthSoon(this, "cc_widget");
            this._cc_widget = new VNCCSControlCenterWidget(this);
            this.setDirtyCanvas(true, true);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            syncDOMWidgetWidth(this, "cc_widget");
            setTimeout(() => syncDOMWidgetWidth(this, "cc_widget"), 100);
            this._cc_widget?.restoreState();
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            syncDOMWidgetWidth(this, "cc_widget");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "cc_widget"));
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            if (this._cc_widget?.pollingInterval)
                clearInterval(this._cc_widget.pollingInterval);
            if (this._cc_widget?._onGlobalPointerUp)
                window.removeEventListener("pointerup", this._cc_widget._onGlobalPointerUp);
            if (this._cc_widget?._onRegistryUpdate)
                window.removeEventListener("vnccs-cc-registry-updated", this._cc_widget._onRegistryUpdate);
            if (this._cc_widget?._onManagerTaskCompleted)
                api.removeEventListener("cm-task-completed", this._cc_widget._onManagerTaskCompleted);
            if (this._cc_widget?._onManagerQueueStatus)
                api.removeEventListener("cm-queue-status", this._cc_widget._onManagerQueueStatus);
        };
    },
});

// ─── Widget ───────────────────────────────────────────────────────────────────

class VNCCSControlCenterWidget {
    constructor(node) {
        this.node = node;
        this.config   = null;
        this.state    = {};
        this.dlStatus = {};
        this.dependencyStatus = { ok: true, message: "" };
        this._lastDependencyModalMessage = "";
        this._dependencyRefreshTimer = null;
        this._dependencyRefreshSeq = 0;
        this._draggingLoraSlider = false;
        this._onGlobalPointerUp = null;
        this.downloadStartTimes = {};  // for fake progress bars
        this._samplers   = DEFAULT_SAMPLERS;
        this._schedulers = DEFAULT_SCHEDULERS;
        this.pollingInterval = null;
        this._dependencyInstallTasks = new Map();
        this._dependencyRestartRequired = false;
        this._restartRequiredOverlay = null;
        this._managerPolicyOverlay = null;
        this._onManagerTaskCompleted = event => this._handleManagerTaskCompleted(event);
        this._onManagerQueueStatus = event => this._handleManagerQueueStatus(event);
        api.addEventListener("cm-task-completed", this._onManagerTaskCompleted);
        api.addEventListener("cm-queue-status", this._onManagerQueueStatus);

        _injectVNCCSControlCenterStyles();
        this._buildUI();
        this._startPolling();
        this._fetchSamplerLists();
        this._fetchModuleStatus(); // once on init

        setTimeout(() => {
            const rw = this.node.widgets?.find(w => w.name === "repo_id");
            if (rw) {
                const orig = rw.callback;
                rw.callback = v => { orig?.call(rw, v); if (v?.trim()) this.fetchConfig(v.trim()); };
            }
            this.restoreState();
        }, 150);

        // Listen for other CC nodes fetching fresh config (avoid re-fetching ourselves)
        this._onRegistryUpdate = (e) => {
            if (!this.container?.isConnected) {
                window.removeEventListener("vnccs-cc-registry-updated", this._onRegistryUpdate);
                return;
            }
            const repoId = e.detail?.repo_id;
            const myRepo = this._getRepoId();
            if (repoId && repoId === myRepo && window.VNCCS_CC_REGISTRY[repoId]) {
                this.config = window.VNCCS_CC_REGISTRY[repoId];
                this._syncCustomModelInput();
                if (!this._isUserInteracting()) this._renderAll();
            }
        };
        window.addEventListener("vnccs-cc-registry-updated", this._onRegistryUpdate);

        this._onGlobalPointerUp = () => {
            this._draggingLoraSlider = false;
        };
        window.addEventListener("pointerup", this._onGlobalPointerUp);
    }

    _getModelTypeTabs() {
        // TECH DEBT: Nunchaku/NVFP entries can remain in old catalogs/state, but
        // are intentionally hidden from the Control Center UI. Delete this after
        // catalogs are cleaned and old workflow JSON is migrated.
        const activeKind = this._activeKind();
        const preferred = activeKind === "Klein9b" ? ["unet", "custom"] : ["gguf", "custom"];
        const available = new Set((this.config?.models ?? [])
            .filter(entry => this._metaKind(entry).toLowerCase() === activeKind.toLowerCase())
            .map(entry => entry.type)
            .filter(Boolean));
        available.add("custom");
        const preferredTabs = preferred.filter(type => available.has(type));
        return preferredTabs.length ? preferredTabs : Array.from(available);
    }

    _getCustomModelInputIndex() {
        return (this.node.inputs ?? []).findIndex(input => input?.name === "model");
    }

    _getCustomClipInputIndex() {
        return (this.node.inputs ?? []).findIndex(input => input?.name === "clip");
    }

    _getCustomVaeInputIndex() {
        return (this.node.inputs ?? []).findIndex(input => input?.name === "vae");
    }

    _getStoredSelectedType() {
        const kind = this._activeKind();
        const selectedForKind = this.state.selected_types_by_kind?.[kind];
        if (selectedForKind) return selectedForKind;
        return kind === "QIE2511" ? (this.state.selected_type || "") : "";
    }

    _syncCustomModelInput() {
        // Before the registry config arrives, _getModelTypeTabs() contains only
        // the synthetic CUSTOM tab. That must not make a brand-new node expose
        // external model sockets. Existing custom workflows still retain their
        // sockets because their stored state is restored before this sync.
        const selectedType = this.config ? this._getSelectedType() : this._getStoredSelectedType();
        const isCustom = selectedType === "custom";

        const sync = (name, type, getIndex, shouldShow) => {
            const inputIndex = getIndex.call(this);
            if (shouldShow) {
                if (inputIndex === -1) {
                    this.node.addInput(name, type);
                    this.node.setDirtyCanvas(true, true);
                }
                return;
            }
            if (inputIndex === -1) {
                return;
            }
            this.node.removeInput(inputIndex);
            this.node.setDirtyCanvas(true, true);
        };

        sync("model", "MODEL", this._getCustomModelInputIndex, isCustom);
        sync("clip", "CLIP", this._getCustomClipInputIndex, isCustom);
        sync("vae", "VAE", this._getCustomVaeInputIndex, isCustom);
    }

    _setSelectedType(nextType) {
        if (!nextType || this._getSelectedType() === nextType) return;
        if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
        this.state.selected_types_by_kind[this._activeKind()] = nextType;
        this.state.selected_type = nextType;

        const variants = this._visibleModelsByType(nextType);
        if (nextType !== "custom") {
            const selectedModel = this._getSelectedModelName(nextType);
            if (!variants.find(m => m.name === selectedModel) && variants.length) {
                this._setSelectedModelName(nextType, variants[0].name);
            } else if (selectedModel) {
                this.state.selected_model = selectedModel;
            }
        }

        this._syncCustomModelInput();
        this._saveState();
        this._renderAll();
        this._scheduleDependencyRefresh(true);
    }

    _scheduleDependencyRefresh(showModal = false, delay = 120, rerender = true) {
        if (this._dependencyRefreshTimer) {
            clearTimeout(this._dependencyRefreshTimer);
        }

        const refreshSeq = ++this._dependencyRefreshSeq;
        this._dependencyRefreshTimer = setTimeout(async () => {
            this._dependencyRefreshTimer = null;
            await this._refreshDependencyStatus(showModal);
            if (refreshSeq !== this._dependencyRefreshSeq) return;
            if (rerender) this._renderAll();
        }, delay);
    }

    // ── Sampler/scheduler lists ───────────────────────────────────────────────

    async _fetchSamplerLists() {
        try {
            const r = await api.fetchApi("/object_info/KSampler");
            if (!r.ok) return;
            const data = await r.json();
            const req = data?.KSampler?.input?.required;
            if (req?.sampler_name?.[0])  this._samplers   = req.sampler_name[0];
            if (req?.scheduler?.[0])     this._schedulers = req.scheduler[0];
        } catch { /* use defaults */ }
    }

    // ── State helpers ─────────────────────────────────────────────────────────

    _getRepoId()     { return (this.node.widgets?.find(w => w.name === "repo_id")?.value ?? "").trim(); }
    _getStateWidget(){ return this.node.widgets?.find(w => w.name === "node_state"); }
    _activeKind() {
        const kind = String(this.state.active_kind || "QIE2511");
        return MODEL_FAMILIES.some(entry => entry.kind === kind) ? kind : "QIE2511";
    }

    _familyDefinition(kind = this._activeKind()) {
        return MODEL_FAMILIES.find(entry => entry.kind === kind) || MODEL_FAMILIES[0];
    }

    _getSelectedType(){
        const visibleTabs = this._getModelTypeTabs();
        const selected = this.state.selected_types_by_kind?.[this._activeKind()]
            ?? (this._activeKind() === "QIE2511" ? this.state.selected_type : "");
        if (selected && visibleTabs.includes(selected)) return selected;
        const preferred = this._familyDefinition().defaultType;
        return visibleTabs.includes(preferred) ? preferred : (visibleTabs[0] || "");
    }

    _getSelectedModelName(type = this._getSelectedType()) {
        const selectedByType = this.state.selected_models ?? {};
        const familyKey = `${this._activeKind()}:${type}`;
        if (selectedByType[familyKey]) return selectedByType[familyKey];
        if (this._activeKind() === "QIE2511") return selectedByType[type] ?? this.state.selected_model ?? "";
        return "";
    }

    _setSelectedModelName(type, modelName) {
        if (!type || !modelName) return;
        if (!this.state.selected_models) this.state.selected_models = {};
        this.state.selected_models[`${this._activeKind()}:${type}`] = modelName;
        if (this._activeKind() === "QIE2511") this.state.selected_models[type] = modelName;
        if (type === this._getSelectedType()) {
            this.state.selected_model = modelName;
        }
    }

    _getSelectedModelEntry() {
        if (!this.config?.models?.length) return null;
        const selectedType = this._getSelectedType();
        if (selectedType === "custom") return this._getCustomContextModelEntry();
        const selectedModel = this._getSelectedModelName(selectedType);
        const variants = this._visibleModelsByType(selectedType);
        return variants.find(m => m.name === selectedModel) ?? variants[0] ?? null;
    }

    _getCustomContextModelEntry() {
        const contextType = this._familyDefinition().defaultType;
        const variants = this._visibleModelsByType(contextType);
        const selectedModel = this._getSelectedModelName(contextType);
        return variants.find(m => m.name === selectedModel) ?? variants[0] ?? null;
    }

    _visibleModelsByType(type) {
        // TECH DEBT: Nunchaku and NVFP/UNet entries are still present in
        // control_center.json for future use, but hidden from the UI for now.
        if (type === "custom") return [];
        const activeKind = this._activeKind().toLowerCase();
        return (this.config?.models || []).filter(m =>
            (!m.type || m.type === type) && this._metaKind(m).toLowerCase() === activeKind
        );
    }

    _metaKind(entry) {
        return String(entry?.kind ?? entry?.Kind ?? "").trim();
    }

    _metaType(entry) {
        return String(entry?.type ?? entry?.Type ?? "").trim();
    }

    _selectedKind() {
        return this._metaKind(this._getSelectedModelEntry()) || this._activeKind();
    }

    _isQwenFamily(entry = this._getSelectedModelEntry()) {
        const identity = [
            this._metaKind(entry),
            entry?.name ?? "",
            entry?.local_path ?? "",
        ].join(" ").toLowerCase();
        return identity.includes("qwen") || identity.includes("qie");
    }

    _sameKind(entry, kind = this._selectedKind()) {
        const entryKind = this._metaKind(entry);
        return !entryKind || !kind || entryKind.toLowerCase() === kind.toLowerCase();
    }

    _exactKind(entry, kind = this._selectedKind()) {
        const entryKind = this._metaKind(entry);
        return Boolean(entryKind && kind && entryKind.toLowerCase() === kind.toLowerCase());
    }

    _currentModelParams() {
        if (!this.state.model_params_by_kind) this.state.model_params_by_kind = {};
        const kind = this._activeKind();
        if (!this.state.model_params_by_kind[kind]) {
            const legacy = kind === "QIE2511" && this.state.model_params
                ? this.state.model_params
                : {};
            this.state.model_params_by_kind[kind] = {
                steps: DEFAULT_MODEL_STEPS,
                cfg: DEFAULT_MODEL_CFG,
                sampler: "euler",
                scheduler: DEFAULT_MODEL_SCHEDULER,
                ...legacy,
            };
        }
        return this.state.model_params_by_kind[kind];
    }

    _syncActiveFamilyState() {
        const kind = this._activeKind();
        const type = this._getSelectedType();
        if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
        this.state.active_kind = kind;
        this.state.selected_types_by_kind[kind] = type;
        this.state.selected_type = type;
        this.state.selected_model = this._getSelectedModelName(type);
        this.state.model_params = { ...this._currentModelParams() };
    }

    _setActiveKind(kind) {
        if (!MODEL_FAMILIES.some(entry => entry.kind === kind) || kind === this._activeKind()) return;
        this._syncActiveFamilyState();
        this.state.active_kind = kind;
        const type = this._getSelectedType();
        const variants = this._visibleModelsByType(type);
        if (variants.length && !variants.some(entry => entry.name === this._getSelectedModelName(type))) {
            this._setSelectedModelName(type, variants[0].name);
        }
        this._syncCustomModelInput();
        this._saveState();
        this._renderAll();
        this._scheduleDependencyRefresh(true);
    }

    _isTurboLora(entry) {
        return this._metaType(entry).toLowerCase() === "turbolora";
    }

    _compatibleTurboLoras() {
        const selectedKind = this._selectedKind();
        return (this.config?.lora || []).filter(entry =>
            !entry.custom && this._isTurboLora(entry) && this._exactKind(entry, selectedKind)
        );
    }

    _preferredTurboLora() {
        const entries = this._compatibleTurboLoras();
        return entries.find(entry => {
            const identity = `${entry.name || ""} ${entry.local_path || ""}`.toLowerCase();
            return identity.includes("lightning");
        }) || entries[0] || null;
    }

    _setTurboPreset(enabled) {
        const params = this._currentModelParams();

        if (enabled) {
            params.turbo_previous_settings = {
                steps: params.steps ?? DEFAULT_MODEL_STEPS,
                cfg: params.cfg ?? DEFAULT_MODEL_CFG,
            };
            params.steps = 4;
            params.cfg = 1.0;
            return;
        }

        const previous = params.turbo_previous_settings || {};
        if (previous.steps !== undefined) params.steps = previous.steps;
        if (previous.cfg !== undefined) params.cfg = previous.cfg;
        params.turbo_previous_settings = null;
    }

    _isHelperLora(entry) {
        return this._metaType(entry).toLowerCase() === "helper";
    }

    _hasEnabledLoras() {
        const selectedKind = this._selectedKind();
        const entryByName = Object.fromEntries((this.config?.lora ?? []).map(entry => [entry.name, entry]));
        return (this.state.loras ?? []).some(l => {
            if (!l?.name || l.auto_apply !== true) return false;
            const entry = entryByName[l.name];
            if (!entry) return false;
            if (this._isTurboLora(entry)) return this._exactKind(entry, selectedKind);
            if (Math.abs(Number(l.strength ?? 1)) <= 1e-6) return false;
            return entry.custom;
        });
    }

    _saveState() {
        this._syncActiveFamilyState();
        const w = this._getStateWidget();
        if (w) w.value = JSON.stringify(this.state);
        this.node.setDirtyCanvas(true, true);
        this._dispatchLoraOptions();
    }

    _dispatchLoraOptions() {
        if (!this.config?.lora) return;
        const selectedKind = this._selectedKind();
        const options = [];
        for (const entry of this.config.lora) {
            if (entry.custom || this._isTurboLora(entry) || !this._isHelperLora(entry)) continue;
            if (!this._exactKind(entry, selectedKind)) continue;
            const norm = (entry.local_path || "").replace(/\\/g, "/");
            const rel = norm.startsWith("models/loras/") ? norm.slice("models/loras/".length) : norm.split("/").pop();
            options.push(rel);
        }
        window.dispatchEvent(new CustomEvent("vnccs-lora-options-updated", { detail: { options } }));
    }

    restoreState() {
        const w = this._getStateWidget();
        if (w?.value && w.value !== "{}") {
            try { this.state = JSON.parse(w.value); } catch { this.state = {}; }
        }
        if (Array.isArray(this.state.loras)) {
            this.state.loras = this.state.loras.map(lora => {
                if (lora && typeof lora === "object" && lora.auto_apply === undefined && lora.enabled !== undefined) {
                    return { ...lora, auto_apply: lora.enabled !== false };
                }
                return lora;
            });
        }
        if (this.state.selected_model && !this.state.selected_models) {
            const selectedType = this.state.selected_type;
            this.state.selected_models = selectedType ? { [selectedType]: this.state.selected_model } : {};
        }
        this.state.active_kind = this._activeKind();
        if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
        if (!this.state.selected_types_by_kind.QIE2511 && this.state.selected_type) {
            this.state.selected_types_by_kind.QIE2511 = this.state.selected_type;
        }
        // Control Center now exposes a single pipe output.
        this.state.output_slot_names = [];
        this._syncOutputSlots();
        this._syncCustomModelInput();
        const repo = this._getRepoId();
        if (repo) this.fetchConfig(repo);
    }

    async _refreshDependencyStatus(showModal = false) {
        // TECH DEBT: Nunchaku dependency checks are disabled. Delete this no-op
        // and the commented legacy implementation after stale workflows migrate.
        this.dependencyStatus = { ok: true, message: "" };
        this._lastDependencyModalMessage = "";
        return this.dependencyStatus;

        /*
        const selectedType = this._getSelectedType();
        const selectedModelEntry = this._getSelectedModelEntry();

        if (selectedType !== "nunchaku" || !selectedModelEntry) {
            this.dependencyStatus = { ok: true, message: "" };
            return this.dependencyStatus;
        }

        try {
            const response = await api.fetchApi("/vnccs/control_center/dependencies", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_type: selectedType,
                    model_name: selectedModelEntry.name,
                    model_path: selectedModelEntry.local_path ?? "",
                    has_enabled_loras: this._hasEnabledLoras(),
                }),
            });
            this.dependencyStatus = response.ok ? await response.json() : { ok: false, message: "Dependency check failed." };
        } catch (error) {
            this.dependencyStatus = { ok: false, message: String(error?.message || error) };
        }

        if (showModal && !this.dependencyStatus.ok && this.dependencyStatus.message) {
            if (this._lastDependencyModalMessage !== this.dependencyStatus.message) {
                this._lastDependencyModalMessage = this.dependencyStatus.message;
                this.showMessage(this.dependencyStatus.message, true);
            }
        }

        if (this.dependencyStatus.ok) {
            this._lastDependencyModalMessage = "";
        }

        // Auto-modal: if Qwen model selected and fix not installed
        if (showModal && selectedType === "nunchaku" && selectedModelEntry) {
            const modelId = ((selectedModelEntry.name || "") + " " + (selectedModelEntry.local_path || "")).toLowerCase();
            if (modelId.includes("qwen")) {
                try {
                    const fixResp = await api.fetchApi("/vnccs/control_center/nunchaku_fix_status");
                    if (fixResp.ok) {
                        const fixStatus = await fixResp.json();
                        if (!fixStatus.installed && !fixStatus.nunchaku_missing) {
                            const fixKey = "__qwen_fix_modal__";
                            if (this._lastDependencyModalMessage !== fixKey) {
                                this._lastDependencyModalMessage = fixKey;
                                this._showQwenFixModal();
                            }
                        }
                    }
                } catch (_) {}
            }
        }

        return this.dependencyStatus;
        */
    }

    _showQwenFixModal() {
        // TECH DEBT: Nunchaku/Qwen fix modal is disabled. Delete this legacy
        // modal after old Nunchaku support is removed completely.
        return;

        /*
        const ov = document.createElement("div");
        ov.style.cssText = `
            position: absolute; top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.75); display:flex; align-items:center;
            justify-content:center; z-index:1000; padding:16px; box-sizing:border-box;
        `;
        const box = document.createElement("div");
        box.style.cssText = `
            background: #12121a; border: 1px solid rgba(255,143,163,0.25);
            border-radius: 10px; padding: 16px; max-width: 280px; width:100%;
            text-align:center; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        `;
        box.innerHTML = `<div style="color:#e8e8f0;font-size:12px;margin-bottom:14px;line-height:1.5;font-family:'Sora',sans-serif;">
            Qwen-Image model requires the PR&#xA0;#790 fix to work correctly.<br><br>Install it now?
        </div>`;
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:8px;justify-content:center;";
        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Later";
        cancelBtn.className = "vnccs-cc-btn";
        cancelBtn.onclick = () => ov.remove();
        const installBtn = document.createElement("button");
        installBtn.textContent = "Install Fix";
        installBtn.className = "vnccs-cc-btn vnccs-cc-btn--save";
        installBtn.onclick = async () => {
            installBtn.textContent = "Installing…";
            installBtn.disabled = true;
            try {
                const res = await api.fetchApi("/vnccs/control_center/nunchaku_apply_fix", { method: "POST" });
                const result = await res.json();
                ov.remove();
                if (result.ok) {
                    this.showMessage("Fix applied. Please restart ComfyUI to apply changes.");
                } else {
                    this.showMessage("Failed: " + (result.message || "Unknown error"), true);
                }
            } catch (e) {
                ov.remove();
                this.showMessage("Error: " + e.message, true);
            }
        };
        row.append(cancelBtn, installBtn);
        box.appendChild(row);
        ov.appendChild(box);
        ov.onclick = e => { if (e.target === ov) ov.remove(); };
        this.container.appendChild(ov);
        */
    }

    // ── Polling ───────────────────────────────────────────────────────────────

    _isUserInteracting() {
        // Skip re-render if any focusable element inside our container is active
        // (e.g. an open <select> dropdown, a focused input)
        if (this._draggingLoraSlider) return true;
        const active = document.activeElement;
        return active && this.container?.contains(active);
    }

    _startPolling() {
        if (this.pollingInterval) clearInterval(this.pollingInterval);
        this.pollingInterval = setInterval(async () => {
            try {
                const r = await api.fetchApi("/vnccs/manager/status");
                if (!r.ok) return;
                const newStatuses = await r.json();

                // Detect transitions to "success" → re-fetch config to update disk state
                let needsRefresh = false;
                for (const key in newStatuses) {
                    const s = newStatuses[key];
                    const old = this.dlStatus[key];
                    if (s?.status === "success" && old?.status !== "success") {
                        needsRefresh = true;
                        break;
                    }
                }

                this.dlStatus = newStatuses;

                if (this._isUserInteracting()) return;
                if (needsRefresh) {
                    const repo = this._getRepoId();
                    if (repo) this.fetchConfig(repo);
                } else {
                    this._renderAll();
                }
            } catch { /* ignore */ }
        }, 2000);
    }

    // ── Container layout ──────────────────────────────────────────────────────

    _buildUI() {
        // Create container first, then pass it to addDOMWidget (same pattern as Pose Studio)
        const c = document.createElement("div");
        c.className = "vnccs-cc-root";
        this.container = c;
        enableMiddleMouseCanvasPan(c);
        attachHelpTooltips(c);

        this.node.addDOMWidget("cc_widget", "cc_panel", c, {
            getValue: () => "{}", setValue: () => {}, serialize: false,
            hideOnZoom: false,
        });
        syncDOMWidgetWidthSoon(this.node, "cc_widget");

        // Hide node_state widget so it takes no space
        const stateW = this.node.widgets?.find(w => w.name === "node_state");
        if (stateW) {
            stateW.computeSize = () => [0, -4];
            stateW.type = "hidden";
            stateW.draw = () => {};
            if (stateW.element) stateW.element.style.display = "none";
        }

        this._buildStatusStrip();
        this._buildHeader();
        this._buildScroll();
        this._buildFooter();
    }

    _buildHeader() {
        const h = document.createElement("div");
        h.className = "vnccs-cc-header";

        this.statusText = document.createElement("span");
        this.statusText.className = "vnccs-cc-title";
        this.statusText.textContent = "VNCCS Control Center";

        const btns = document.createElement("div");
        btns.className = "vnccs-cc-header-btns";
        btns.appendChild(this._btn("↺", () => {
            const r = this._getRepoId();
            if (r) this.fetchConfig(r, true);
            else this.showMessage("Enter repo_id first.", true);
        }, "Refresh config"));
        btns.appendChild(this._btn("⚙", () => this._showSettings(), "Settings"));

        h.append(this.statusText, btns);
        this.container.appendChild(h);
    }

    _buildScroll() {
        this.scrollArea = document.createElement("div");
        this.scrollArea.className = "vnccs-cc-scroll";

        const empty = document.createElement("div");
        empty.className = "vnccs-cc-empty";
        empty.textContent = "Enter repo_id and click ↺ to load";
        this.scrollArea.appendChild(empty);

        this.container.appendChild(this.scrollArea);
    }

    _buildFooter() {
        const f = document.createElement("div");
        f.className = "vnccs-cc-footer";

        const b = document.createElement("button");
        b.className = "vnccs-cc-btn--download-all";
        b.textContent = "⬇ Download / Update";
        b.onclick = () => this._downloadAllMissing();

        f.appendChild(b);
        this.container.appendChild(f);
    }

    // ── Module status strip ───────────────────────────────────────────────────

    _buildStatusStrip() {
        this.statusStrip = document.createElement("div");
        this.statusStrip.className = "vnccs-cc-status-strip";

        const pills = document.createElement("div");
        pills.className = "vnccs-cc-status-pills";

        const lbl = document.createElement("span");
        lbl.className = "vnccs-cc-status-label";
        lbl.textContent = "Modules";
        pills.appendChild(lbl);

        this._pillMain = this._makePill("VNCCS", null, "loading");
        this._pillUtils = this._makePill("Utils", null, "loading");
        pills.append(this._pillMain, this._pillUtils);
        this._statusPills = pills;
        this._dependencyPills = {};

        this.statusStrip.appendChild(pills);
        this._updateBanner = null; // created lazily
        this.container.appendChild(this.statusStrip);
    }

    _makePill(name, version, state, extra = "") {
        const p = document.createElement("span");
        p.className = `vnccs-cc-pill vnccs-cc-pill--${state}`;
        const icons = { ok: "●", update: "↑", error: "✕", loading: "…", dup: "⚠", partial: "⚠", warning: "⚠" };
        p.textContent = version
            ? `${name} v${version} ${icons[state] ?? ""}${extra ? " " + extra : ""}`
            : `${name} ${icons[state] ?? ""}${extra ? " " + extra : ""}`;
        if (state === "update") p.title = `Update available: ${extra}`;
        if (state === "dup")   p.title = `Duplicate install detected: ${extra}`;
        if (state === "partial") p.title = extra || "Installed but node classes are not loaded";
        if (state === "warning") p.title = extra || "Installed with warnings";
        if (state === "error") p.title = extra || "Module not found";
        return p;
    }

    _updatePill(pillEl, name, version, state, extra = "") {
        const icons = { ok: "●", update: "↑", error: "✕", loading: "…", dup: "⚠", partial: "⚠", warning: "⚠" };
        pillEl.className = `vnccs-cc-pill vnccs-cc-pill--${state}`;
        pillEl.textContent = version
            ? `${name} v${version} ${icons[state] ?? ""}${extra ? " " + extra : ""}`
            : `${name} ${icons[state] ?? ""}${extra ? " " + extra : ""}`;
        if (state === "update") pillEl.title = `Update available: v${extra}`;
        if (state === "dup")   pillEl.title = `Duplicate install: ${extra}`;
        if (state === "partial") pillEl.title = extra || "Installed but node classes are not loaded";
        if (state === "warning") pillEl.title = extra || "Installed with warnings";
        if (state === "error") pillEl.title = extra || "Module not found";
    }

    _ensureDependencyPill(key, label) {
        if (this._dependencyPills[key]) return this._dependencyPills[key];
        const pill = this._makePill(label, null, "loading");
        pill.classList.add("vnccs-cc-pill--dependency");
        this._dependencyPills[key] = pill;
        this._statusPills?.appendChild(pill);
        return pill;
    }

    _openDependencyUrl(url) {
        if (!url) return;
        window.open(url, "_blank", "noopener,noreferrer");
    }

    async _readManagerError(response, fallback) {
        try {
            const text = await response.text();
            if (!text) return fallback;
            try {
                const data = JSON.parse(text);
                return data.error || data.message || text;
            } catch {
                return text;
            }
        } catch {
            return fallback;
        }
    }

    async _getManagerInstallPolicy() {
        try {
            const response = await api.fetchApi("/vnccs/manager/install_policy", { cache: "no-store" });
            if (!response.ok) return null;
            return await response.json();
        } catch {
            return null;
        }
    }

    _isManagerPersonalCloudError(error) {
        const message = String(error?.message || error).toLowerCase();
        return message.includes("network_mode") && message.includes("personal_cloud");
    }

    _pendingDependencyKeys(items) {
        return [...new Set((items || []).map(item => String(item?.key || "").trim()).filter(Boolean))];
    }

    _storePendingDependencyInstalls(items) {
        const keys = this._pendingDependencyKeys(items);
        if (!keys.length) return;
        try {
            sessionStorage.setItem(PENDING_DEPENDENCY_INSTALLS_KEY, JSON.stringify(keys));
        } catch { /* private browsing may disable session storage */ }
    }

    async _resumePendingDependencyInstalls(items) {
        let keys = [];
        try {
            keys = JSON.parse(sessionStorage.getItem(PENDING_DEPENDENCY_INSTALLS_KEY) || "[]");
            sessionStorage.removeItem(PENDING_DEPENDENCY_INSTALLS_KEY);
        } catch {
            try { sessionStorage.removeItem(PENDING_DEPENDENCY_INSTALLS_KEY); } catch { /* ignored */ }
        }
        if (!Array.isArray(keys) || !keys.length) return false;

        const allowedKeys = new Set(keys.filter(key => typeof key === "string"));
        const pending = (items || []).filter(item => allowedKeys.has(item.key) && item.manager_id);
        if (!pending.length) return false;

        let queued = 0;
        for (const item of pending) {
            const statusButton = document.createElement("button");
            statusButton.textContent = "Resuming…";
            if (await this._installDependency(item, statusButton, { skipPolicyCheck: true })) queued += 1;
        }
        if (queued) this.showMessage(`Resumed installation of ${queued} custom node package${queued === 1 ? "" : "s"}.`);
        return queued > 0;
    }

    async _ensureManagerInstallPolicy(items) {
        const policy = await this._getManagerInstallPolicy();
        if (!policy || policy.install_allowed) return true;
        if (!policy.security_level_allows_install) {
            this.showMessage(
                `ComfyUI-Manager security_level is '${policy.security_level}'. Registered custom-node installation requires normal or below. VNCCS will not weaken this setting automatically.`,
                true,
            );
            return false;
        }
        if (policy.requires_personal_cloud) {
            this._showManagerPolicyModal(items, policy);
            return false;
        }
        return true;
    }

    _showManagerPolicyModal(items, policy = null) {
        if (this._managerPolicyOverlay?.isConnected) return;
        const pendingItems = (items || []).filter(item => item?.manager_id);
        if (!pendingItems.length) return;

        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";
        this._managerPolicyOverlay = ov;

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";
        panel.style.maxWidth = "480px";

        const title = document.createElement("div");
        title.className = "vnccs-cc-deps-modal-title";
        title.textContent = "Allow Manager installations";

        const text = document.createElement("div");
        text.className = "vnccs-cc-deps-modal-text";
        const listener = policy?.listener && policy.listener !== "unknown" ? ` (${policy.listener})` : "";
        text.textContent = `ComfyUI-Manager blocks registered custom-node installation because this server listens on a non-local address${listener}. VNCCS can set network_mode = personal_cloud in Manager's user config, preserve a backup, restart ComfyUI, and then resume this installation automatically.\n\nThis permits registered custom-node installations for users who can access this ComfyUI server. Enable it only on a trusted, single-user deployment. security_level will not be changed.`;

        const buttons = document.createElement("div");
        buttons.className = "vnccs-cc-settings-btns";
        const cancelBtn = this._btn("Cancel", () => ov.remove());
        const enableBtn = this._btn("Enable & restart", async () => {
            enableBtn.disabled = true;
            enableBtn.textContent = "Updating…";
            try {
                const response = await api.fetchApi("/vnccs/manager/enable_personal_cloud", {
                    method: "POST",
                    headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                    body: JSON.stringify({ confirmation: "enable_personal_cloud" }),
                });
                if (!response.ok) {
                    throw new Error(await this._readManagerError(response, "Could not update ComfyUI-Manager policy."));
                }
                this._storePendingDependencyInstalls(pendingItems);
                const restarted = await this._restartComfyUIServer(enableBtn, true);
                if (restarted) ov.remove();
            } catch (error) {
                enableBtn.disabled = false;
                enableBtn.textContent = "Enable & restart";
                this.showMessage(`Could not enable Manager installations:\n${String(error?.message || error)}`, true);
            }
        });
        enableBtn.classList.add("vnccs-cc-btn--save");
        buttons.append(cancelBtn, enableBtn);
        panel.append(title, text, buttons);
        ov.appendChild(panel);
        ov.onclick = event => { if (event.target === ov) ov.remove(); };
        this.container.appendChild(ov);
    }

    async _queueLegacyDependencyInstall(item, uiId) {
        const install = {
            id: item.manager_id,
            version: item.manager_version || "latest",
            ui_id: uiId,
            title: item.label || item.key,
            selected_version: "latest",
            channel: "default",
            mode: "remote",
            skip_post_install: false,
        };
        const response = await api.fetchApi("/manager/queue/install", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(install),
        });
        if (response.status === 404 || response.status === 405) return null;
        if (!response.ok) {
            throw new Error(await this._readManagerError(response, "ComfyUI-Manager rejected the installation."));
        }

        const startResponse = await api.fetchApi("/manager/queue/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
        if (!startResponse.ok && startResponse.status !== 201) {
            throw new Error(await this._readManagerError(startResponse, "The ComfyUI-Manager queue could not be started."));
        }
        return "legacy";
    }

    async _queueDependencyInstall(item, button) {
        if (!item?.manager_id) {
            throw new Error("This package is not registered in Comfy Registry. Install it manually after reviewing its source.");
        }

        const uiId = `vnccs-dependency-${item.key}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const clientId = api.clientId || "vnccs-control-center";
        const task = {
            ui_id: uiId,
            client_id: clientId,
            kind: "install",
            params: {
                id: item.manager_id,
                version: item.manager_version || "latest",
                ui_id: uiId,
                selected_version: "latest",
                repository: item.github_url || null,
                mode: "remote",
                channel: "default",
                skip_post_install: false,
            },
        };

        const tracked = { item, button, mode: "legacy" };
        this._dependencyInstallTasks.set(uiId, tracked);
        try {
            let mode = await this._queueLegacyDependencyInstall(item, uiId);
            if (!mode) {
                mode = "modern";
                tracked.mode = mode;
                const response = await api.fetchApi("/v2/manager/queue/task", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(task),
                });
                if (!response.ok) {
                    throw new Error(await this._readManagerError(response, "ComfyUI-Manager rejected the installation."));
                }
                const startResponse = await api.fetchApi("/v2/manager/queue/start", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: "{}",
                });
                if (!startResponse.ok && startResponse.status !== 201) {
                    throw new Error(await this._readManagerError(startResponse, "The Manager queue could not be started."));
                }
            }

            if (this._dependencyInstallTasks.has(uiId)) button.textContent = "Installing…";
            return mode;
        } catch (error) {
            this._dependencyInstallTasks.delete(uiId);
            throw error;
        }
    }

    async _installDependency(item, button, options = {}) {
        const originalLabel = button.textContent;
        if (!options.skipPolicyCheck && !await this._ensureManagerInstallPolicy([item])) return false;
        button.disabled = true;
        button.textContent = "Queueing…";
        try {
            await this._queueDependencyInstall(item, button);
            return true;
        } catch (error) {
            button.disabled = false;
            button.textContent = originalLabel;
            if (this._isManagerPersonalCloudError(error)) {
                this._showManagerPolicyModal([item]);
                return false;
            }
            this.showMessage(`Could not install ${item.label || item.key}:\n${String(error?.message || error)}`, true);
            return false;
        }
    }

    async _installAllDependencies(items, button, installButtons) {
        const installable = items.filter(item => item.manager_id && item.status !== "unsupported");
        if (!installable.length) return;
        if (!await this._ensureManagerInstallPolicy(installable)) return;
        button.disabled = true;
        button.textContent = "Queueing…";
        let queued = 0;
        for (const item of installable) {
            const itemButton = installButtons.get(item.key);
            if (itemButton && await this._installDependency(item, itemButton, { skipPolicyCheck: true })) queued += 1;
        }
        button.textContent = queued ? `Installing ${queued}…` : "Install all";
        if (!queued) button.disabled = false;
    }

    _scheduleReloadAfterServerRestart(initiallyUnavailable = false) {
        let sawUnavailable = initiallyUnavailable;
        let attempts = 0;
        const probe = async () => {
            attempts += 1;
            try {
                const response = await api.fetchApi("/vnccs/module_status", { cache: "no-store" });
                if (response.ok && (sawUnavailable || attempts >= 8)) {
                    window.location.reload();
                    return;
                }
            } catch {
                sawUnavailable = true;
            }
            if (attempts < 80) setTimeout(probe, 1500);
            else this.showMessage("ComfyUI did not become available after restart. Reload the page when the server is ready.", true);
        };
        setTimeout(probe, 2500);
    }

    async _restartComfyUIServer(button, reloadAfterRestart = false) {
        button.disabled = true;
        button.textContent = "Restarting…";
        try {
            let response = await api.fetchApi("/manager/reboot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: "{}",
            });
            if (response.status === 404 || response.status === 405) {
                response = await api.fetchApi("/v2/manager/reboot", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: "{}",
                });
            }
            if (!response.ok) {
                throw new Error(await this._readManagerError(response, "ComfyUI-Manager rejected the restart."));
            }
            if (reloadAfterRestart) this._scheduleReloadAfterServerRestart(false);
            return true;
        } catch (error) {
            // A successful reboot can close the connection before fetch resolves.
            if (String(error?.message || error).toLowerCase().includes("fetch")) {
                if (reloadAfterRestart) this._scheduleReloadAfterServerRestart(true);
                return true;
            }
            button.disabled = false;
            button.textContent = "Restart server";
            this.showMessage(`Could not restart ComfyUI:\n${String(error?.message || error)}`, true);
            return false;
        }
    }

    _showRestartRequiredModal() {
        if (this._restartRequiredOverlay?.isConnected) return;
        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";
        this._restartRequiredOverlay = ov;

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";
        panel.style.maxWidth = "420px";

        const title = document.createElement("div");
        title.className = "vnccs-cc-deps-modal-title";
        title.textContent = "Restart required";

        const text = document.createElement("div");
        text.className = "vnccs-cc-deps-modal-text";
        text.textContent = "The modules were installed, but their nodes are not loaded yet. Restart the ComfyUI server before using VNCCS. Unsaved workflow changes may be lost.";

        const buttons = document.createElement("div");
        buttons.className = "vnccs-cc-settings-btns";
        const laterBtn = this._btn("Later", () => ov.remove());
        const restartBtn = this._btn("Restart server", () => this._restartComfyUIServer(restartBtn));
        restartBtn.classList.add("vnccs-cc-btn--save");
        buttons.append(laterBtn, restartBtn);
        panel.append(title, text, buttons);
        ov.appendChild(panel);
        ov.onclick = event => { if (event.target === ov) ov.remove(); };
        this.container.appendChild(ov);
    }

    _finishDependencyInstall(uiId, success, message = "Installation failed.") {
        const tracked = this._dependencyInstallTasks.get(uiId);
        if (!tracked) return;
        this._dependencyInstallTasks.delete(uiId);

        tracked.button.textContent = success ? "Restart required" : "Retry";
        tracked.button.disabled = success;
        if (!success) tracked.button.disabled = false;

        if (success) {
            this._dependencyRestartRequired = true;
        } else {
            const compatibility = tracked.item.compatibility_note ? `\n\n${tracked.item.compatibility_note}` : "";
            this.showMessage(`Could not install ${tracked.item.label || tracked.item.key}:\n${message}${compatibility}`, true);
        }
        if (this._dependencyRestartRequired && this._dependencyInstallTasks.size === 0) {
            setTimeout(() => this._showRestartRequiredModal(), 100);
        }
    }

    _handleManagerTaskCompleted(event) {
        const detail = event?.detail || {};
        const status = detail.status?.status_str || "";
        const success = status === "success" || detail.result === "success";
        const message = detail.status?.messages?.join("\n") || detail.result || "Installation failed.";
        this._finishDependencyInstall(detail.ui_id, success, message);
    }

    _handleManagerQueueStatus(event) {
        const detail = event?.detail || {};
        if (detail.status === "in_progress") {
            const tracked = this._dependencyInstallTasks.get(detail.target);
            if (tracked?.mode === "legacy") tracked.button.textContent = "Installing…";
            return;
        }
        if (detail.status !== "done") return;

        const results = detail.nodepack_result || {};
        for (const [uiId, tracked] of this._dependencyInstallTasks.entries()) {
            if (tracked.mode !== "legacy" || !Object.prototype.hasOwnProperty.call(results, uiId)) continue;
            const result = results[uiId];
            const success = result === "success" || result === "skip";
            this._finishDependencyInstall(uiId, success, String(result || "Installation failed."));
        }
    }

    _showMissingDependenciesModal(items) {
        if (!items?.length) return;
        const signature = items
            .map(item => `${item.key}:${item.status}:${(item.missing_nodes || []).join("|")}`)
            .sort()
            .join(";");
        if (this._lastMissingDependencyModalSignature === signature) return;
        this._lastMissingDependencyModalSignature = signature;

        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";
        panel.style.maxWidth = "460px";

        const title = document.createElement("div");
        title.className = "vnccs-cc-deps-modal-title";
        title.textContent = "Missing custom nodes";

        const text = document.createElement("div");
        text.className = "vnccs-cc-deps-modal-text";
        text.textContent = "VNCCS uses these custom nodes internally. Install uses the registered Comfy Registry release through ComfyUI-Manager's standard policy-compatible queue. Restart ComfyUI afterward.";

        const list = document.createElement("div");
        list.className = "vnccs-cc-deps-list";
        const installButtons = new Map();

        for (const item of items) {
            const row = document.createElement("div");
            row.className = "vnccs-cc-deps-item";

            const meta = document.createElement("div");
            const name = document.createElement("div");
            name.className = "vnccs-cc-deps-name";
            name.textContent = item.label || item.key;
            meta.appendChild(name);

            const missing = Array.isArray(item.missing_nodes) ? item.missing_nodes : [];
            if (missing.length) {
                const missingEl = document.createElement("div");
                missingEl.className = "vnccs-cc-deps-missing";
                missingEl.textContent = `missing: ${missing.join(", ")}`;
                meta.appendChild(missingEl);
            }
            if (item.compatibility_note) {
                const compatibilityEl = document.createElement("div");
                compatibilityEl.className = "vnccs-cc-deps-missing";
                compatibilityEl.textContent = item.compatibility_note;
                meta.appendChild(compatibilityEl);
            }

            row.appendChild(meta);
            const actions = document.createElement("div");
            actions.className = "vnccs-cc-deps-actions";
            if (item.manager_id) {
                const installBtn = document.createElement("button");
                installBtn.type = "button";
                installBtn.className = "vnccs-cc-link-btn";
                installBtn.textContent = "Install";
                installBtn.title = item.compatibility_note || "Install the latest registry release through ComfyUI-Manager";
                installBtn.onclick = () => this._installDependency(item, installBtn);
                actions.appendChild(installBtn);
                installButtons.set(item.key, installBtn);
            }
            if (item.github_url) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "vnccs-cc-link-btn";
                btn.textContent = "GitHub";
                btn.onclick = () => this._openDependencyUrl(item.github_url);
                actions.appendChild(btn);
            }
            row.appendChild(actions);
            list.appendChild(row);
        }

        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";
        if (this._dependencyRestartRequired) {
            const restartBtn = this._btn("Restart server", () => this._restartComfyUIServer(restartBtn));
            restartBtn.classList.add("vnccs-cc-btn--save");
            btns.appendChild(restartBtn);
        } else if (items.some(item => item.manager_id && item.status !== "unsupported")) {
            const installAllBtn = this._btn("Install all", () => this._installAllDependencies(items, installAllBtn, installButtons));
            installAllBtn.classList.add("vnccs-cc-btn--save");
            btns.appendChild(installAllBtn);
        }
        const closeBtn = this._btn("Close", () => ov.remove());
        btns.appendChild(closeBtn);

        panel.append(title, text, list, btns);
        ov.appendChild(panel);
        ov.onclick = event => { if (event.target === ov) ov.remove(); };
        this.container.appendChild(ov);
    }

    _showUpdateBanner(lines) {
        if (this._updateBanner) this._updateBanner.remove();
        if (!lines.length) return;
        const b = document.createElement("div");
        b.className = "vnccs-cc-update-banner";
        b.textContent = "↑ " + lines.join("  ·  ");
        b.title = "Updates are available. Restart ComfyUI after updating.";
        this.statusStrip.appendChild(b);
        this._updateBanner = b;
    }

    _semverGt(a, b) {
        // true if a > b (both "x.y.z" strings)
        const pa = String(a).split(".").map(Number);
        const pb = String(b).split(".").map(Number);
        for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
            const va = pa[i] ?? 0, vb = pb[i] ?? 0;
            if (va > vb) return true;
            if (va < vb) return false;
        }
        return false;
    }

    async _fetchModuleStatus() {
        const LABELS = { main: "VNCCS", utils: "Utils" };
        const PILLS  = { main: this._pillMain, utils: this._pillUtils };

        // Fetch installed module versions from the local ComfyUI backend.
        let local = {};
        try {
            const r = await api.fetchApi("/vnccs/module_status");
            if (r.ok) local = await r.json();
        } catch { /* server may not be ready yet */ }

        const updateNeeded = [];

        for (const key of ["main", "utils"]) {
            const info   = local[key] ?? {};
            const label  = LABELS[key];
            const pill   = PILLS[key];
            const locVer = info.version ?? null;

            if (info.error === "not_found") {
                this._updatePill(pill, label, null, "error", "not installed");
                continue;
            }

            if (info.duplicate) {
                this._updatePill(pill, label, locVer, "dup",
                    info.duplicate_folders?.join(" & ") ?? "");
                updateNeeded.push(`${label}: duplicate folders (${info.duplicate_folders?.join(", ")})`);
                continue;
            }

            if (!locVer) {
                this._updatePill(pill, label, null, "error", "version unknown");
                continue;
            }

            this._updatePill(pill, label, locVer, "ok");
        }

        const dependencies = local.dependencies || {};
        const missingDependencies = [];
        for (const [key, info] of Object.entries(dependencies)) {
            const label = info.label || key;
            const pill = this._ensureDependencyPill(key, label);
            const missing = Array.isArray(info.missing_nodes) ? info.missing_nodes : [];
            const loader = info.loader || {};
            const loaderDetail = loader.folder || loader.module || loader.file || "";
            const detail = missing.length ? `missing: ${missing.join(", ")}` : (info.warning || (loaderDetail ? `loader: ${loaderDetail}` : (info.folder ? `folder: ${info.folder}` : "")));
            if (info.status === "ok") {
                this._updatePill(pill, label, null, "ok");
                pill.title = detail || "Installed";
            } else if (info.status === "warning") {
                this._updatePill(pill, label, null, "warning", detail);
                pill.title = detail || "Installed with warnings";
                updateNeeded.push(`${label}: ${info.warning || "installed with warnings"}`);
            } else if (info.status === "partial") {
                this._updatePill(pill, label, null, "partial");
                pill.title = detail || "installed but not loaded";
            } else if (info.status === "unsupported") {
                this._updatePill(pill, label, null, "warning", "unsupported");
                pill.title = info.compatibility_note || "Unsupported on this platform";
                updateNeeded.push(`${label}: unsupported on this platform`);
            } else {
                this._updatePill(pill, label, null, "error");
                pill.title = detail || "not installed";
            }
            if (info.status !== "ok" && info.status !== "warning" && info.status !== "unsupported") {
                missingDependencies.push({ key, ...info });
            }
        }
        const resumedPendingInstalls = await this._resumePendingDependencyInstalls(missingDependencies);
        if (!resumedPendingInstalls) this._showMissingDependenciesModal(missingDependencies);

        this._showUpdateBanner(updateNeeded);
    }

    // ── Primitives ────────────────────────────────────────────────────────────

    _btn(label, onClick, title = "") {
        const b = document.createElement("button");
        b.className = "vnccs-cc-btn";
        b.textContent = label;
        if (title) b.title = title;
        b.onclick = onClick;
        return b;
    }

    _badge(status) {
        const cls = { installed:"installed", missing:"missing", queued:"queued",
                      downloading:"downloading", error:"error", outdated:"outdated",
                      auth_required:"error", success:"installed" };
        const lbl = { installed:"●", missing:"↓", queued:"…",
                      downloading:"⬇", error:"✕", outdated:"↑",
                      auth_required:"⚠", success:"●" };
        const span = document.createElement("span");
        span.className = `vnccs-cc-badge vnccs-cc-badge--${cls[status] ?? "missing"}`;
        span.textContent = lbl[status] ?? "?";
        return span;
    }

    _sel(id, opts, val, onChange) {
        const s = document.createElement("select");
        s.id = id;
        s.className = "vnccs-cc-select";
        opts.forEach(o => {
            const op = document.createElement("option");
            op.value = op.textContent = o;
            if (o === val) op.selected = true;
            s.appendChild(op);
        });
        s.onchange = () => onChange(s.value);
        return s;
    }

    _numInput(val, min, max, step, onChange) {
        const i = document.createElement("input");
        i.type = "number"; i.value = val; i.min = min; i.max = max; i.step = step;
        i.className = "vnccs-cc-input";
        i.onchange = () => onChange(parseFloat(i.value));
        return i;
    }

    _label(text) {
        const l = document.createElement("div");
        l.className = "vnccs-cc-param-label";
        l.textContent = text;
        return l;
    }

    // ── Config fetch ──────────────────────────────────────────────────────────

    async fetchConfig(repoId, force = false) {
        this.statusText.textContent = "Loading…";

        const cacheKey = `vnccs_cc_cache_${repoId}`;

        if (force) {
            delete window.VNCCS_CC_REGISTRY[repoId];
            localStorage.removeItem(cacheKey);
        }

        // Show cached data immediately while fetching fresh
        if (!force && window.VNCCS_CC_REGISTRY[repoId]) {
            this.config = window.VNCCS_CC_REGISTRY[repoId];
            this.statusText.textContent = this.config.name || "Control Center";
            if (!this.state.output_slot_names) this.state.output_slot_names = [];
            this._syncCustomModelInput();
            this._renderAll();
            this._dispatchLoraOptions();
        }

        // Debounce: reuse in-flight fetch for same repo
        if (!force && window.VNCCS_CC_FETCH_PROMISES[repoId]) {
            try {
                const data = await window.VNCCS_CC_FETCH_PROMISES[repoId];
                this.config = data;
                this.statusText.textContent = data.name || "Control Center";
                if (!this.state.output_slot_names) this.state.output_slot_names = [];
                this._syncCustomModelInput();
                this._renderAll();
                this._dispatchLoraOptions();
            } catch { /* already shown by the original caller */ }
            return;
        }

        const fetchPromise = (async () => {
            try {
                const url = `/vnccs/control_center/check?repo_id=${encodeURIComponent(repoId)}${force ? "&force_refresh=true" : ""}`;
                const r = await api.fetchApi(url);
                const data = await r.json();
                if (data.error) throw new Error(data.error);
                window.VNCCS_CC_REGISTRY[repoId] = data;
                return data;
            } finally {
                delete window.VNCCS_CC_FETCH_PROMISES[repoId];
            }
        })();

        window.VNCCS_CC_FETCH_PROMISES[repoId] = fetchPromise;

        try {
            const data = await fetchPromise;
            this.config = data;
            this.statusText.textContent = data.name || "Control Center";
            localStorage.setItem(cacheKey, JSON.stringify(data));
            this._syncCnetSlots(data);
            this._syncCustomModelInput();
            await this._refreshDependencyStatus(true);
            this._renderAll();
            this._dispatchLoraOptions();
            window.dispatchEvent(new CustomEvent("vnccs-cc-registry-updated", {
                detail: { repo_id: repoId }
            }));
        } catch (e) {
            this.statusText.textContent = "Error";
            this.scrollArea.innerHTML = "";
            const err = document.createElement("div");
            err.className = "vnccs-cc-error";
            err.textContent = "✕ " + e.message;
            this.scrollArea.appendChild(err);
        }
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    // dlStatus carries transient states (queued/downloading/error/auth_required).
    // "success" from dlStatus is NOT authoritative — server decides installed/missing.
    _resolveStatus(dlsStatus, serverStatus) {
        const transient = new Set(["queued", "downloading", "error", "auth_required"]);
        if (dlsStatus && transient.has(dlsStatus)) return dlsStatus;
        return serverStatus || "missing";
    }

    _isDownloadableStatus(status) {
        return status === "missing" || status === "error" || status === "outdated";
    }

    _downloadLabel(status) {
        return status === "outdated" ? "↑ Update" : "⬇ Download";
    }

    _statusLabel(status, dls = {}) {
        if (status === "installed") return "✓ Installed";
        if (status === "outdated") return "↑ Update";
        if (status === "queued") return "⏳ Queued";
        if (status === "downloading") return dls.message || "Downloading…";
        if (status === "auth_required") return "⚠ Key Required";
        return "↓ Missing";
    }

    _renderAll() {
        if (!this.config) return;
        this.scrollArea.innerHTML = "";

        // TECH DEBT: legacy Nunchaku error rendering disabled. Delete after
        // stale workflow state cannot select Nunchaku anymore.
        /*
        if (this._getSelectedType() === "nunchaku" && this.dependencyStatus && !this.dependencyStatus.ok) {
            const err = document.createElement("div");
            err.className = "vnccs-cc-error";
            err.textContent = "✕ " + this.dependencyStatus.message;
            this.scrollArea.appendChild(err);
        }
        */

        const selType = this._getSelectedType();
        const isCP    = selType === "checkpoint";
        const isCustom = selType === "custom";

        // Only variants of the selected type (set in Settings)
        const variants = isCustom ? [] : this._visibleModelsByType(selType);

        // Auto-select first variant of this type if current selection doesn't match
        if (!isCustom) {
            const selectedModel = this._getSelectedModelName(selType);
            const curOk = variants.find(m => m.name === selectedModel);
            if (!curOk && variants.length) {
                this._setSelectedModelName(selType, variants[0].name);
            } else if (curOk) {
                this._setSelectedModelName(selType, curOk.name);
            }
        }

        this.scrollArea.appendChild(this._renderFamilyTabs());

        // MODEL — 2-column block: one big card for the selected type's variants
        this._renderTwoColBlockSingle(
            "MODEL", variants.length, "models",
            () => isCustom ? this._renderCustomModelPlaceholder() : this._renderModelCard(selType, variants),
            () => this._renderModelParams(),
            () => this._renderModelSamplerParams(),
            () => this._renderModelTabs(selType),
            !isCustom,
            isCustom
        );

        // CLIP + VAE — custom mode gets these from external sockets instead.
        if (!isCP && !isCustom) {
            this._renderClipVaeBlock();
        }

        // LORA
        this._renderLoraBlock();
        this._renderCustomLoraBlock();

        const turboInline = this._buildModelTurboInlineSection();
        if (turboInline) {
            this.scrollArea?.querySelector(`.vnccs-cc-block[data-block-key="models"]`)?.appendChild(turboInline);
        }

        // CONTROLNET / OTHER
        this._renderBlock("CONTROLNET", this.config.controlnet, "controlnet",
            e => this._renderCnetEntry("controlnet", e));
        this._renderBlock("OTHER", this.config.other, "other",
            e => this._renderCnetEntry("other", e));
    }

    _renderFamilyTabs() {
        const tabs = document.createElement("div");
        tabs.className = "vnccs-cc-family-tabs";
        tabs.setAttribute("role", "tablist");
        const activeKind = this._activeKind();
        MODEL_FAMILIES.forEach(family => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "vnccs-cc-family-tab";
            button.textContent = family.label;
            button.setAttribute("role", "tab");
            button.setAttribute("aria-selected", String(family.kind === activeKind));
            if (family.kind === activeKind) button.classList.add("vnccs-cc-family-tab--active");
            button.onclick = () => this._setActiveKind(family.kind);
            tabs.appendChild(button);
        });
        return tabs;
    }

    _divider() {
        const d = document.createElement("div");
        d.className = "vnccs-cc-divider";
        return d;
    }

    // Generic single-column collapsible block
    _renderBlock(title, entries, key, rowFn) {
        const collapsed = this.state.collapsed?.[key] ?? false;
        const block = this._blockShell(title, entries?.length ?? 0, key, collapsed);
        if (!collapsed) {
            const body = document.createElement("div");
            body.className = "vnccs-cc-block-body";
            (entries ?? []).forEach(e => body.appendChild(rowFn(e)));
            if (!entries?.length) {
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                body.appendChild(empty);
            }
            block.appendChild(body);
        }
        this.scrollArea.appendChild(block);
    }

    _buildClipVaeBlock() {
        const selectedKind = this._selectedKind();
        const entries = [
            ...(this.config.clip || []).filter(entry => this._sameKind(entry, selectedKind)).map(entry => ({ ...entry, _cat: "clip" })),
            ...(this.config.vae || []).filter(entry => this._sameKind(entry, selectedKind)).map(entry => ({ ...entry, _cat: "vae" })),
        ];
        const collapsed = this.state.collapsed?.clip_vae ?? false;
        const block = this._blockShell("CLIP + VAE", entries.length, "clip_vae", collapsed);

        if (!collapsed) {
            const body = document.createElement("div");
            body.className = "vnccs-cc-card-grid";

            entries.forEach(entry => body.appendChild(this._renderClipVaeCard(entry)));

            if (!entries.length) {
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                body.appendChild(empty);
            }

            block.appendChild(body);
        }

        return block;
    }

    _renderClipVaeBlock() {
        this.scrollArea.appendChild(this._buildClipVaeBlock());
    }

    _buildTurboModelBlock() {
        const selectedKind = this._selectedKind();
        const entries = (this.config.lora || []).filter(entry =>
            !entry.custom && this._isTurboLora(entry) && this._exactKind(entry, selectedKind)
        );
        const collapsed = this.state.collapsed?.turbo_model ?? false;
        const block = this._blockShell("TURBO MODEL", entries.length, "turbo_model", collapsed);

        if (!collapsed) {
            const body = document.createElement("div");
            body.className = "vnccs-cc-lora-sections";

            const grid = document.createElement("div");
            grid.className = "vnccs-cc-lora-grid vnccs-cc-lora-grid--compact";
            entries.forEach(entry => {
                const ls = (this.state.loras ?? []).find(l => l.name === entry.name);
                grid.appendChild(this._renderLoraCard(entry, {
                    compact: true,
                    active: ls?.auto_apply === true,
                    mode: "turbo",
                }));
            });

            if (entries.length) body.appendChild(grid);
            else {
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                body.appendChild(empty);
            }
            block.appendChild(body);
        }

        return block;
    }

    _renderTurboModelBlock() {
        this.scrollArea.appendChild(this._buildTurboModelBlock());
    }

    _buildLoraBlock() {
        const selectedKind = this._selectedKind();
        const entries = (this.config.lora || []).filter(entry =>
            !entry.custom && !this._isTurboLora(entry) && this._exactKind(entry, selectedKind)
        );
        const collapsed = this.state.collapsed?.lora ?? false;
        const block = this._blockShell("LORA", entries.length, "lora", collapsed);

        if (!collapsed) {
            const body = document.createElement("div");
            body.className = "vnccs-cc-lora-sections";

            const grid = document.createElement("div");
            grid.className = "vnccs-cc-lora-grid vnccs-cc-lora-grid--compact";
            entries.forEach(entry => grid.appendChild(this._renderLoraCard(entry, { compact: true, mode: "pipe" })));
            if (entries.length) body.appendChild(grid);

            if (!entries.length) {
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                body.appendChild(empty);
            }

            block.appendChild(body);
        }

        return block;
    }

    _buildCustomLoraBlock() {
        const entries = (this.config.lora || []).filter(entry => entry.custom);
        const collapsed = this.state.collapsed?.custom_lora ?? false;
        const block = this._blockShell("CUSTOM LORAS", entries.length, "custom_lora", collapsed);

        if (!collapsed) {
            const body = document.createElement("div");
            body.className = "vnccs-cc-lora-sections";

            const activeEntries = [];
            const inactiveEntries = [];
            entries.forEach(entry => {
                const ls = (this.state.loras ?? []).find(l => l.name === entry.name)
                    ?? { name: entry.name, auto_apply: false, strength: 1.0 };
                if (ls.auto_apply === true) activeEntries.push(entry);
                else inactiveEntries.push(entry);
            });

            if (activeEntries.length) {
                const activeSection = document.createElement("div");
                activeSection.className = "vnccs-cc-lora-section";

                const activeTitle = document.createElement("div");
                activeTitle.className = "vnccs-cc-lora-section-title";
                activeTitle.textContent = `Active (${activeEntries.length})`;

                const activeGrid = document.createElement("div");
                activeGrid.className = "vnccs-cc-lora-grid";
                activeEntries.forEach(entry => activeGrid.appendChild(this._renderLoraCard(entry, { active: true, mode: "custom" })));

                activeSection.append(activeTitle, activeGrid);
                body.appendChild(activeSection);
            }

            if (inactiveEntries.length) {
                const inactiveSection = document.createElement("div");
                inactiveSection.className = "vnccs-cc-lora-section";

                const inactiveTitle = document.createElement("div");
                inactiveTitle.className = "vnccs-cc-lora-section-title";
                inactiveTitle.textContent = `Available (${inactiveEntries.length})`;

                const inactiveGrid = document.createElement("div");
                inactiveGrid.className = "vnccs-cc-lora-grid vnccs-cc-lora-grid--compact";
                inactiveEntries.forEach(entry => inactiveGrid.appendChild(this._renderLoraCard(entry, { compact: true, mode: "custom" })));

                inactiveSection.append(inactiveTitle, inactiveGrid);
                body.appendChild(inactiveSection);
            }

            const addWrap = document.createElement("div");
            addWrap.className = "vnccs-cc-lora-add-wrap";

            const addBtn = document.createElement("button");
            addBtn.className = "vnccs-cc-lora-add-btn";
            addBtn.textContent = "+ Add Custom LoRA";
            addBtn.onclick = () => this._showCustomLoraDialog();

            const addNote = document.createElement("div");
            addNote.className = "vnccs-cc-lora-add-note";
            addNote.textContent = "Select any installed LoRA from ComfyUI's standard loras folder and add it as a persistent card.";

            addWrap.append(addBtn, addNote);
            body.appendChild(addWrap);

            block.appendChild(body);
        }

        return block;
    }

    _renderLoraBlock() {
        this.scrollArea.appendChild(this._buildLoraBlock());
    }

    _renderCustomLoraBlock() {
        this.scrollArea.appendChild(this._buildCustomLoraBlock());
    }

    _replaceBlock(key, nextBlock) {
        const current = this.scrollArea?.querySelector(`.vnccs-cc-block[data-block-key="${key}"]`);
        if (current?.parentNode) {
            current.parentNode.replaceChild(nextBlock, current);
        } else if (nextBlock) {
            this.scrollArea?.appendChild(nextBlock);
        }
    }

    _refreshLoraBlock() {
        if (!this.config) return;
        this._replaceBlock("lora", this._buildLoraBlock());
        this.scrollArea?.querySelector(`.vnccs-cc-block[data-block-key="turbo_model"]`)?.remove();
        this._replaceBlock("custom_lora", this._buildCustomLoraBlock());
    }

    async _removeCustomLora(entry) {
        const repoId = this._getRepoId();
        if (!repoId) {
            this.showMessage("Load control center config first.", true);
            return;
        }

        this.showConfirm(`Remove custom LoRA \"${entry.name}\" from the list?`, async () => {
            try {
                const response = await api.fetchApi("/vnccs/control_center/custom_lora/delete", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        repo_id: repoId,
                        name: entry.name,
                        local_path: entry.local_path,
                    }),
                });
                const result = await response.json();
                if (!response.ok || result.error) throw new Error(result.error || "Failed to remove custom LoRA");

                this.state.loras = (this.state.loras || []).filter(lora => lora.name !== entry.name);
                this._saveState();
                delete window.VNCCS_CC_REGISTRY[repoId];
                await this.fetchConfig(repoId, true);
                this._refreshLoraBlock();
            } catch (error) {
                this.showMessage(String(error?.message || error), true);
            }
        });
    }

    async _showCustomLoraDialog() {
        const repoId = this._getRepoId();
        if (!repoId) {
            this.showMessage("Load control center config first.", true);
            return;
        }

        let payload;
        try {
            const response = await api.fetchApi(`/vnccs/control_center/lora_files?repo_id=${encodeURIComponent(repoId)}`);
            payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || "Failed to load LoRA list");
        } catch (error) {
            this.showMessage(String(error?.message || error), true);
            return;
        }

        const items = (payload.items || []).filter(item => !item.already_added);
        if (!items.length) {
            this.showMessage("No new LoRA files available in the ComfyUI loras folder.");
            return;
        }

        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";

        const title = document.createElement("h3");
        title.className = "vnccs-cc-settings-title";
        title.textContent = "Add Custom LoRA";
        panel.appendChild(title);

        const desc = document.createElement("div");
        desc.className = "vnccs-cc-lora-add-note";
        desc.textContent = "Choose an installed LoRA file from ComfyUI's standard loras folder. It will be saved into a separate VNCCS JSON file and stay available between sessions.";
        panel.appendChild(desc);

        const selectWrap = document.createElement("div");
        selectWrap.className = "vnccs-cc-settings-field";
        const selectLabel = document.createElement("label");
        selectLabel.className = "vnccs-cc-settings-label";
        selectLabel.textContent = "LoRA File";
        const select = document.createElement("select");
        select.className = "vnccs-cc-settings-select";
        items.forEach(item => {
            const option = document.createElement("option");
            option.value = item.path;
            option.textContent = item.path;
            select.appendChild(option);
        });
        selectWrap.append(selectLabel, select);
        panel.appendChild(selectWrap);

        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";
        const cancelBtn = this._btn("Cancel", () => ov.remove());
        const saveBtn = this._btn("Add", async () => {
            const path = select.value;
            if (!path) return;
            saveBtn.disabled = true;
            saveBtn.textContent = "Adding…";
            try {
                const response = await api.fetchApi("/vnccs/control_center/custom_lora", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ repo_id: repoId, path }),
                });
                const result = await response.json();
                if (!response.ok || result.error) throw new Error(result.error || "Failed to add custom LoRA");

                ov.remove();
                delete window.VNCCS_CC_REGISTRY[repoId];
                await this.fetchConfig(repoId, true);
                this._refreshLoraBlock();
            } catch (error) {
                saveBtn.disabled = false;
                saveBtn.textContent = "Add";
                this.showMessage(String(error?.message || error), true);
            }
        });
        saveBtn.classList.add("vnccs-cc-btn--save");
        btns.append(cancelBtn, saveBtn);
        panel.appendChild(btns);

        ov.appendChild(panel);
        ov.onclick = e => { if (e.target === ov) ov.remove(); };
        this.container.appendChild(ov);
    }

    // Two-column block with a single pre-built row (MODEL slot)
    _renderTwoColBlockSingle(title, count, key, rowFn, rightFn, rightFn2 = null, topFn = null, showEmptyLeft = true, renderLeftWhenEmpty = false) {
        const collapsed = this.state.collapsed?.[key] ?? false;
        const block = this._blockShell(title, count, key, collapsed);
        if (!collapsed) {
            if (topFn) {
                block.appendChild(topFn());
            }

            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-twocol";

            const leftContent = (count || renderLeftWhenEmpty) ? rowFn() : null;
            if (leftContent) {
                const left = document.createElement("div");
                left.className = "vnccs-cc-twocol-left";
                left.appendChild(leftContent);
                wrap.appendChild(left);
            } else if (showEmptyLeft) {
                const left = document.createElement("div");
                left.className = "vnccs-cc-twocol-left";
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                left.appendChild(empty);
                wrap.appendChild(left);
            }

            const right = document.createElement("div");
            right.className = "vnccs-cc-twocol-right";
            right.appendChild(rightFn());

            wrap.appendChild(right);

            if (rightFn2) {
                const right2 = document.createElement("div");
                right2.className = "vnccs-cc-twocol-right2";
                right2.appendChild(rightFn2());
                wrap.appendChild(right2);
            }

            block.appendChild(wrap);
        }
        this.scrollArea.appendChild(block);
    }

    // Two-column block: left=list, right=panel
    _renderTwoColBlock(title, entries, key, rowFn, rightFn) {
        const collapsed = this.state.collapsed?.[key] ?? false;
        const block = this._blockShell(title, entries?.length ?? 0, key, collapsed);
        if (!collapsed) {
            const row = document.createElement("div");
            row.className = "vnccs-cc-twocol";

            const left = document.createElement("div");
            left.className = "vnccs-cc-twocol-left";
            (entries ?? []).forEach(e => left.appendChild(rowFn(e)));
            if (!entries?.length) {
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                left.appendChild(empty);
            }

            const right = document.createElement("div");
            right.className = "vnccs-cc-twocol-right";
            right.appendChild(rightFn());

            row.append(left, right);
            block.appendChild(row);
        }
        this.scrollArea.appendChild(block);
    }

    _blockShell(title, count, key, collapsed) {
        const block = document.createElement("div");
        block.className = "vnccs-cc-block";
        block.dataset.blockKey = key;

        const hdr = document.createElement("div");
        hdr.className = "vnccs-cc-block-hdr";

        const titleWrap = document.createElement("span");
        titleWrap.textContent = title;
        const countSpan = document.createElement("span");
        countSpan.className = "vnccs-cc-block-hdr-count";
        countSpan.textContent = `(${count})`;
        titleWrap.appendChild(countSpan);

        const arrow = document.createElement("span");
        arrow.className = "vnccs-cc-block-arrow";
        arrow.textContent = collapsed ? "▶" : "▼";

        hdr.append(titleWrap, arrow);
        hdr.onclick = () => {
            if (!this.state.collapsed) this.state.collapsed = {};
            this.state.collapsed[key] = !this.state.collapsed[key];
            this._saveState(); this._renderAll();
        };
        block.appendChild(hdr);
        return block;
    }

    // ── MODEL big card ────────────────────────────────────────────────────────

    _renderModelTabs(type) {
        const tabs = this._getModelTypeTabs();
        const tabsRow = document.createElement("div");
        tabsRow.className = "vnccs-cc-model-tabs";
        tabsRow.style.setProperty("--vnccs-model-tab-count", String(Math.max(1, tabs.length)));

        tabs.forEach(tabType => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "vnccs-cc-model-tab";
            if (tabType === type) button.classList.add("vnccs-cc-model-tab--active");
            button.textContent = tabType.toUpperCase();

            const hasVariants = tabType === "custom"
                ? true
                : this._visibleModelsByType(tabType).length > 0;
            if (!hasVariants) button.classList.add("vnccs-cc-model-tab--missing");

            button.onclick = () => this._setSelectedType(tabType);
            tabsRow.appendChild(button);
        });

        return tabsRow;
    }

    _renderModelCard(type, variants) {
        const cur = variants.find(v => v.name === this._getSelectedModelName(type)) ?? variants[0];
        if (!cur) return document.createElement("div");

        const key    = `cc_models_${cur.name}`;
        const dls    = this.dlStatus[key] ?? {};
        const status = this._resolveStatus(dls.status, cur.status);

        const card = document.createElement("div");
        card.className = "vnccs-cc-model-card";
        this._applyProgressLayer(card, key, dls);

        // Top: badge + name + status label
        const top = document.createElement("div");
        top.className = "vnccs-cc-model-card-top";
        top.appendChild(this._badge(status));

        const nameEl = document.createElement("span");
        nameEl.className = "vnccs-cc-model-card-name";
        nameEl.textContent = cur.name;
        top.appendChild(nameEl);

        const statusLbl = document.createElement("span");
        statusLbl.className = "vnccs-cc-model-card-status";
        if (status === "installed") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--ok");
        } else if (status === "downloading" || status === "queued") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--progress");
        } else if (status === "outdated") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--missing");
        } else {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--missing");
        }
        top.appendChild(statusLbl);
        card.appendChild(top);

        // Description
        if (cur.description) {
            const desc = document.createElement("div");
            desc.className = "vnccs-cc-model-card-desc";
            desc.textContent = cur.description;
            card.appendChild(desc);
        }

        // Footer: variant dropdown at bottom + download button
        const footer = document.createElement("div");
        footer.className = "vnccs-cc-model-card-footer";

        if (variants.length > 1) {
            const names = variants.map(v => v.name);
            let prefixLen = 0;
            const first = names[0];
            for (let i = 0; i < first.length; i++) {
                if (names.every(n => n[i] === first[i])) prefixLen = i + 1; else break;
            }
            const sep = Math.max(first.slice(0, prefixLen).lastIndexOf("-"),
                                 first.slice(0, prefixLen).lastIndexOf("_"));
            if (sep > 0) prefixLen = sep + 1;

            const icon = s => ({ installed:"●", missing:"↓", outdated:"↑", queued:"…", downloading:"⬇", error:"✕", auth_required:"⚠" }[s] ?? "?");

            const sel = document.createElement("select");
            sel.className = "vnccs-cc-model-card-select";
            variants.forEach(v => {
                const vDls = this.dlStatus[`cc_models_${v.name}`] ?? {};
                const vSt  = this._resolveStatus(vDls.status, v.status);
                const op = document.createElement("option");
                op.value = v.name;
                op.textContent = (v.name.slice(prefixLen) || v.name) + "  " + icon(vSt);
                if (v.name === cur.name) op.selected = true;
                sel.appendChild(op);
            });
            sel.onchange = () => {
                const chosen = variants.find(v => v.name === sel.value);
                if (chosen) {
                    this._setSelectedModelName(type, chosen.name);
                    if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
                    this.state.selected_types_by_kind[this._activeKind()] = chosen.type || type;
                    this.state.selected_type  = chosen.type || type;
                    this._saveState();
                    this._renderAll();
                    this._scheduleDependencyRefresh(true);
                }
            };
            footer.appendChild(sel);
        }

        if (status === "auth_required") {
            const warning = document.createElement("span");
            warning.className = "vnccs-cc-row-progress";
            warning.textContent = "Authentication required";
            footer.appendChild(warning);
        } else if (this._isDownloadableStatus(status)) {
            footer.appendChild(this._btn(this._downloadLabel(status), () => this._downloadEntry("models", cur)));
        }

        if (footer.children.length) card.appendChild(footer);
        return card;
    }

    _renderCustomModelPlaceholder() {
        const card = document.createElement("div");
        card.className = "vnccs-cc-model-card vnccs-cc-model-card--placeholder";

        const text = document.createElement("div");
        text.className = "vnccs-cc-model-card-placeholder-text";
        text.textContent = "Pass-through mode is enabled. Connect MODEL, CLIP, and VAE to the node inputs.";
        card.appendChild(text);

        return card;
    }

    _renderClipVaeCard(entry) {
        const isClip = entry._cat === "clip";
        const cat = isClip ? "clip" : "vae";
        const key = `cc_${cat}_${entry.name}`;
        const dls = this.dlStatus[key] ?? {};
        const status = this._resolveStatus(dls.status, entry.status);

        const card = document.createElement("div");
        card.className = "vnccs-cc-model-card";
        this._applyProgressLayer(card, key, dls);

        const top = document.createElement("div");
        top.className = "vnccs-cc-model-card-top";
        top.appendChild(this._badge(status));

        const kicker = document.createElement("span");
        kicker.className = "vnccs-cc-model-card-kicker";
        kicker.textContent = isClip ? "CLIP" : "VAE";
        top.appendChild(kicker);

        const nameEl = document.createElement("span");
        nameEl.className = "vnccs-cc-model-card-name";
        nameEl.textContent = entry.name;
        top.appendChild(nameEl);

        const statusLbl = document.createElement("span");
        statusLbl.className = "vnccs-cc-model-card-status";
        if (status === "installed") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--ok");
        } else if (status === "downloading" || status === "queued") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--progress");
        } else if (status === "auth_required") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--missing");
        } else {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--missing");
        }
        top.appendChild(statusLbl);
        card.appendChild(top);

        if (entry.description) {
            const desc = document.createElement("div");
            desc.className = "vnccs-cc-model-card-desc";
            desc.textContent = entry.description;
            card.appendChild(desc);
        }

        if (this._isDownloadableStatus(status)) {
            const footer = document.createElement("div");
            footer.className = "vnccs-cc-model-card-footer";
            footer.appendChild(this._btn(this._downloadLabel(status), () => this._downloadEntry(cat, entry)));
            card.appendChild(footer);
        }

        return card;
    }

    // ── MODEL right column: sampler params ────────────────────────────────────

    _modelParamsSave(patch) {
        Object.assign(this._currentModelParams(), patch);
        this._saveState();
        if (this._isQwenFamily()) this._renderAll();
    }

    _buildModelTurboInlineSection() {
        const entries = this._compatibleTurboLoras();
        if (!entries.length) return null;

        const section = document.createElement("div");
        section.className = "vnccs-cc-model-inline-section";

        const label = document.createElement("div");
        label.className = "vnccs-cc-model-inline-label";
        label.textContent = "Turbo LoRA";
        section.appendChild(label);

        entries.forEach(entry => section.appendChild(this._renderTurboInlineStrip(entry)));
        return section;
    }

    _renderTurboInlineStrip(entry) {
        const ls = (this.state.loras ?? []).find(l => l.name === entry.name)
            ?? { name: entry.name, auto_apply: false, strength: 1.0 };
        const dls = this.dlStatus[`cc_lora_${entry.name}`] ?? {};
        const status = this._resolveStatus(dls.status, entry.status);
        const installed = status === "installed";
        const active = installed && ls.auto_apply === true;

        const row = document.createElement("div");
        row.className = "vnccs-cc-turbo-strip";
        row.classList.toggle("is-installed", installed);
        row.classList.toggle("is-active", active);
        if (installed) {
            row.onclick = () => this._selectTurboLora(entry.name, !active);
        }

        row.appendChild(this._badge(status));

        const name = document.createElement("div");
        name.className = "vnccs-cc-turbo-strip-name";
        name.textContent = entry.name;
        row.appendChild(name);

        const statusLbl = document.createElement("div");
        statusLbl.className = "vnccs-cc-turbo-strip-status";
        if (installed) {
            statusLbl.textContent = active ? "Active" : "Installed";
            statusLbl.classList.add(active ? "vnccs-cc-turbo-strip-status--active" : "vnccs-cc-turbo-strip-status--ok");
        } else if (status === "downloading" || status === "queued") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-turbo-strip-status--progress");
        } else {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-turbo-strip-status--missing");
        }
        row.appendChild(statusLbl);

        if (installed) {
            const toggle = document.createElement("label");
            toggle.className = "vnccs-cc-toggle";
            toggle.onclick = event => event.stopPropagation();
            const input = document.createElement("input");
            input.type = "checkbox";
            input.checked = active;
            input.onchange = event => {
                event.stopPropagation();
                this._selectTurboLora(entry.name, event.target.checked);
            };
            const track = document.createElement("div");
            track.className = "vnccs-cc-toggle-track";
            const thumb = document.createElement("div");
            thumb.className = "vnccs-cc-toggle-thumb";
            toggle.append(input, track, thumb);
            row.appendChild(toggle);
        } else if (status === "auth_required") {
            const warning = document.createElement("span");
            warning.className = "vnccs-cc-row-progress";
            warning.textContent = "Authentication required";
            row.appendChild(warning);
        } else if (this._isDownloadableStatus(status)) {
            const btn = this._btn(this._downloadLabel(status), () => this._downloadEntry("lora", entry));
            btn.onclick = event => {
                event.stopPropagation();
                this._downloadEntry("lora", entry);
            };
            row.appendChild(btn);
        }

        return row;
    }

    _renderModelParams() {
        const p  = this._currentModelParams();
        const mp = { steps: DEFAULT_MODEL_STEPS, cfg: DEFAULT_MODEL_CFG, ...p };

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-params";

        const field = (labelText, el) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-param-field";
            const help = {
                Steps: "Default sampling step count shared with connected VNCCS generator nodes.",
                CFG: "Default prompt guidance strength shared with connected VNCCS generator nodes."
            }[labelText];
            setHelpText(wrap, help);
            wrap.appendChild(this._label(labelText));
            wrap.appendChild(el);
            return wrap;
        };

        panel.appendChild(field("Steps",
            this._numInput(mp.steps, 1, 200, 1, v => this._modelParamsSave({ steps: v }))));
        panel.appendChild(field("CFG",
            this._numInput(mp.cfg, 0, 30, 0.5, v => this._modelParamsSave({ cfg: v }))));

        return panel;
    }

    _renderModelSamplerParams() {
        const p  = this._currentModelParams();
        const mp = { sampler: "euler", scheduler: DEFAULT_MODEL_SCHEDULER, ...p };

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-params vnccs-cc-params--right";

        const field = (labelText, el) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-param-field";
            const help = {
                Sampler: "Default sampler algorithm shared with connected VNCCS generator nodes.",
                Scheduler: "Default scheduler/noise schedule shared with connected VNCCS generator nodes."
            }[labelText];
            setHelpText(wrap, help);
            wrap.appendChild(this._label(labelText));
            wrap.appendChild(el);
            return wrap;
        };

        panel.appendChild(field("Sampler",
            this._sel("vnccs-cc-sampler", this._samplers, mp.sampler, v => this._modelParamsSave({ sampler: v }))));
        panel.appendChild(field("Scheduler",
            this._sel("vnccs-cc-scheduler", this._schedulers, mp.scheduler, v => this._modelParamsSave({ scheduler: v }))));

        return panel;
    }

    // ── CLIP + VAE entry ──────────────────────────────────────────────────────

    _renderClipVaeEntry(entry) {
        const isClip = entry._cat === "clip";
        const cat    = isClip ? "clip" : "vae";
        const dls    = this.dlStatus[`cc_${cat}_${entry.name}`] ?? {};
        const status = this._resolveStatus(dls.status, entry.status);

        const key2 = `cc_${cat}_${entry.name}`;
        const row = document.createElement("div");
        row.className = "vnccs-cc-row vnccs-cc-row--clip-sel";
        this._applyProgressLayer(row, key2, dls);

        row.appendChild(this._badge(status));

        const tag = document.createElement("span");
        tag.className = "vnccs-cc-row-tag";
        tag.textContent = isClip ? "CLIP" : "VAE";
        row.appendChild(tag);

        const nameWrap = document.createElement("div");
        nameWrap.className = "vnccs-cc-row-name-wrap";
        const name = document.createElement("span");
        name.className = "vnccs-cc-row-name";
        name.textContent = entry.name;
        nameWrap.appendChild(name);
        if (entry.description) {
            const desc = document.createElement("span");
            desc.className = "vnccs-cc-row-desc";
            desc.textContent = entry.description;
            nameWrap.appendChild(desc);
        }
        row.appendChild(nameWrap);

        if (status === "downloading") {
            const p = document.createElement("span");
            p.className = "vnccs-cc-row-progress";
            p.textContent = dls.message;
            row.appendChild(p);
        } else if (status === "queued") {
            const p = document.createElement("span");
            p.className = "vnccs-cc-row-progress";
            p.textContent = "⏳ Queued";
            row.appendChild(p);
        } else if (status === "auth_required") {
            const warning = document.createElement("span");
            warning.className = "vnccs-cc-row-progress";
            warning.textContent = "Authentication required";
            row.appendChild(warning);
        } else if (this._isDownloadableStatus(status)) {
            row.appendChild(this._btn(status === "outdated" ? "↑" : "↓", () => this._downloadEntry(cat, entry)));
        }
        return row;
    }

    // ── LORA entry ────────────────────────────────────────────────────────────

    _renderLoraCard(entry, options = {}) {
        const { active = false, compact = false, mode = "custom" } = options;
        const isPipeMode = mode === "pipe";
        const isTurboMode = mode === "turbo";
        const isInteractive = mode === "custom" || isTurboMode;
        const ls  = (this.state.loras ?? []).find(l => l.name === entry.name)
            ?? { name: entry.name, auto_apply: false, strength: 1.0 };
        const dls = this.dlStatus[`cc_lora_${entry.name}`] ?? {};
        const status = this._resolveStatus(dls.status, entry.status);

        const loraKey = `cc_lora_${entry.name}`;
        const card = document.createElement("div");
        card.className = "vnccs-cc-model-card vnccs-cc-lora-card";
        if (active && !isPipeMode) card.classList.add("vnccs-cc-lora-card--active");
        if (compact) card.classList.add("vnccs-cc-lora-card--compact");
        if (compact && this._isDownloadableStatus(status)) card.classList.add("vnccs-cc-lora-card--downloadable");
        if (entry.custom) card.classList.add("vnccs-cc-lora-card--has-remove");
        this._applyProgressLayer(card, loraKey, dls);
        if (status === "installed" && isInteractive) {
            card.onclick = () => {
                if (isTurboMode) {
                    this._selectTurboLora(entry.name, !ls.auto_apply);
                } else {
                    this._updateLora(entry.name, { auto_apply: !ls.auto_apply }, { commit: true, rerender: "lora" });
                }
            };
        }

        if (entry.custom) {
            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "vnccs-cc-lora-remove-btn";
            removeBtn.textContent = "×";
            removeBtn.title = "Remove custom LoRA from list";
            removeBtn.onclick = e => {
                e.stopPropagation();
                this._removeCustomLora(entry);
            };
            card.appendChild(removeBtn);
        }

        if (compact && this._isDownloadableStatus(status)) {
            const downloadBtn = document.createElement("button");
            downloadBtn.type = "button";
            downloadBtn.className = "vnccs-cc-lora-download-btn";
            downloadBtn.textContent = status === "outdated" ? "Update" : "Download";
            downloadBtn.title = this._downloadLabel(status);
            downloadBtn.onclick = e => {
                e.stopPropagation();
                this._downloadEntry("lora", entry);
            };
            card.appendChild(downloadBtn);
        }

        const top = document.createElement("div");
        top.className = "vnccs-cc-lora-card-top";
        top.appendChild(this._badge(status));

        const meta = document.createElement("div");
        meta.className = "vnccs-cc-lora-card-meta";

        const nameEl = document.createElement("span");
        nameEl.className = "vnccs-cc-lora-card-name";
        nameEl.textContent = entry.name;
        meta.appendChild(nameEl);

        const statusLbl = document.createElement("span");
        statusLbl.className = "vnccs-cc-lora-card-status";
        if (status === "installed") {
            const effectiveActive = !isPipeMode && ls.auto_apply === true;
            statusLbl.textContent = effectiveActive ? "Active" : "Pipe";
            statusLbl.classList.add(effectiveActive ? "vnccs-cc-lora-card-status--active" : "vnccs-cc-lora-card-status--pipe");
        } else if (status === "downloading" || status === "queued") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--progress");
        } else if (status === "outdated") {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--missing");
        } else {
            statusLbl.textContent = this._statusLabel(status, dls);
            statusLbl.classList.add("vnccs-cc-model-card-status--missing");
        }
        if ((compact && status === "installed" && (isPipeMode || ls.auto_apply !== true)) ||
            (active && status === "installed" && !isPipeMode && ls.auto_apply === true)) {
            statusLbl.classList.add("vnccs-cc-lora-card-status--corner");
            card.appendChild(statusLbl);
        } else {
            meta.appendChild(statusLbl);
        }
        top.appendChild(meta);
        card.appendChild(top);

        if (entry.description || entry.version) {
            const detailRow = document.createElement("div");
            detailRow.className = "vnccs-cc-lora-card-detail-row";

            const desc = document.createElement("div");
            desc.className = "vnccs-cc-lora-card-desc";
            desc.textContent = entry.description || "";
            detailRow.appendChild(desc);

            if (entry.version) {
                const versionEl = document.createElement("span");
                versionEl.className = "vnccs-cc-lora-card-version";
                versionEl.textContent = `v${entry.version}`;
                versionEl.title = `Version ${entry.version}`;
                detailRow.appendChild(versionEl);
            }

            card.appendChild(detailRow);
        }

        const footer = document.createElement("div");
        footer.className = active ? "vnccs-cc-lora-strength-row" : "vnccs-cc-lora-card-actions";
        footer.onclick = e => e.stopPropagation();

        if (this._isDownloadableStatus(status) && !compact) {
            const btn = this._btn(this._downloadLabel(status), () => this._downloadEntry("lora", entry));
            btn.addEventListener("click", e => e.stopPropagation());
            footer.appendChild(btn);
        } else if (status === "auth_required") {
            const warning = document.createElement("span");
            warning.className = "vnccs-cc-row-progress";
            warning.textContent = "Authentication required";
            footer.appendChild(warning);
        } else if (status === "installed" && active && !isPipeMode && !isTurboMode) {
                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = -2;
                slider.max = 2;
                slider.step = 0.05;
                slider.value = ls.strength ?? 1.0;
                slider.className = "vnccs-cc-lora-slider";

                slider.onpointerdown = () => {
                    this._draggingLoraSlider = true;
                };

                const finishSliderDrag = () => {
                    this._draggingLoraSlider = false;
                };

                slider.onpointerup = finishSliderDrag;
                slider.onpointercancel = finishSliderDrag;
                slider.onblur = () => {
                    this._draggingLoraSlider = false;
                };

                const val = document.createElement("span");
                val.className = "vnccs-cc-lora-val";
                val.textContent = parseFloat(slider.value).toFixed(2);

                slider.oninput = () => {
                    const strength = parseFloat(slider.value);
                    val.textContent = strength.toFixed(2);
                    this._updateLora(entry.name, { strength }, { commit: false, persist: false });
                };

                slider.onchange = () => {
                    const strength = parseFloat(slider.value);
                    val.textContent = strength.toFixed(2);
                    this._updateLora(entry.name, { strength }, { commit: true, rerender: null });
                };

                footer.append(slider, val);
        }

        if (footer.children.length) card.appendChild(footer);
        return card;
    }

    _updateLora(name, patch, options = {}) {
        const { commit = true, persist = true, rerender = null } = options;
        if (!this.state.loras) this.state.loras = [];
        const idx = this.state.loras.findIndex(l => l.name === name);
        idx >= 0 ? Object.assign(this.state.loras[idx], patch)
                 : this.state.loras.push({ name, auto_apply: false, strength: 1.0, ...patch });
        if (persist) this._saveState();
        if (rerender === "lora") {
            this._refreshLoraBlock();
        }
        if (commit) {
            this._scheduleDependencyRefresh(true, 120, false);
        }
    }

    _selectTurboLora(name, enabled) {
        const selectedKind = this._selectedKind();
        const turboNames = new Set((this.config?.lora || [])
            .filter(entry => !entry.custom && this._isTurboLora(entry) && this._exactKind(entry, selectedKind))
            .map(entry => entry.name));

        const wasEnabled = (this.state.loras ?? []).some(lora =>
            turboNames.has(lora?.name) && lora.auto_apply === true
        );

        if (!this.state.loras) this.state.loras = [];
        for (const turboName of turboNames) {
            const idx = this.state.loras.findIndex(lora => lora.name === turboName);
            if (idx >= 0) {
                this.state.loras[idx].auto_apply = turboName === name ? enabled : false;
                this.state.loras[idx].strength = 1.0;
            } else {
                this.state.loras.push({
                    name: turboName,
                    auto_apply: turboName === name ? enabled : false,
                    strength: 1.0,
                });
            }
        }

        if (enabled && !wasEnabled) {
            this._setTurboPreset(true);
        } else if (!enabled && wasEnabled) {
            this._setTurboPreset(false);
        }

        this._saveState();
        this._renderAll();
        this._scheduleDependencyRefresh(true, 120, false);
    }

    // ── CONTROLNET / OTHER entry ──────────────────────────────────────────────

    _renderCnetEntry(cat, entry) {
        const cnetKey = `cc_${cat}_${entry.name}`;
        const dls    = this.dlStatus[cnetKey] ?? {};
        const status = this._resolveStatus(dls.status, entry.status);

        const row = document.createElement("div");
        row.className = "vnccs-cc-row vnccs-cc-row--cnet-sel";
        this._applyProgressLayer(row, cnetKey, dls);

        row.appendChild(this._badge(status));

        const nameWrap = document.createElement("div");
        nameWrap.className = "vnccs-cc-row-name-wrap";
        const name = document.createElement("span");
        name.className = "vnccs-cc-row-name";
        name.textContent = entry.name;
        nameWrap.appendChild(name);
        if (entry.description) {
            const desc = document.createElement("span");
            desc.className = "vnccs-cc-row-desc";
            desc.textContent = entry.description;
            nameWrap.appendChild(desc);
        }
        row.appendChild(nameWrap);

        if (status === "downloading") {
            const p = document.createElement("span");
            p.className = "vnccs-cc-row-progress";
            p.textContent = dls.message;
            row.appendChild(p);
        } else if (status === "queued") {
            const p = document.createElement("span");
            p.className = "vnccs-cc-row-progress";
            p.textContent = "⏳ Queued";
            row.appendChild(p);
        } else if (status === "auth_required") {
            const warning = document.createElement("span");
            warning.className = "vnccs-cc-row-progress";
            warning.textContent = "Authentication required";
            row.appendChild(warning);
        } else if (this._isDownloadableStatus(status)) {
            row.appendChild(this._btn(status === "outdated" ? "↑" : "↓", () => this._downloadEntry(cat, entry)));
        }
        return row;
    }

    // ── Output slot management ────────────────────────────────────────────────
    // Control Center exposes only slot 0 = pipe.

    _assignSlot(name) {
        if (!this.state.output_slot_names) this.state.output_slot_names = [];
        this.state.output_slot_names.push(name);
        this.node.addOutput(name, "*");
        this.node.setDirtyCanvas(true, true);
    }

    _freeSlot(name) {
        if (!this.state.output_slot_names) return;
        const stateIdx = this.state.output_slot_names.indexOf(name);
        if (stateIdx === -1) return;
        // Find actual slot index in node.outputs (skip slot 0 = pipe)
        const outIdx = (this.node.outputs ?? []).findIndex((o, i) => i > 0 && o.name === name);
        if (outIdx >= 0) this.node.removeOutput(outIdx);
        this.state.output_slot_names.splice(stateIdx, 1);
        this.node.setDirtyCanvas(true, true);
    }

    _syncOutputSlots() {
        while ((this.node.outputs?.length ?? 0) > 1) {
            this.node.removeOutput(this.node.outputs.length - 1);
        }
        this.state.output_slot_names = [];
        this.node.setDirtyCanvas(true, true);
    }

    // Legacy no-op: older saved nodes may still carry dynamic output state.
    _syncCnetSlots(config) {
        this._syncOutputSlots();
    }

    // ── Settings overlay ──────────────────────────────────────────────────────

    _showSettings() {
        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";

        const ts   = this.state.type_settings ?? {};
        const sel  = this._getSelectedType() || "gguf";
        const gguf = ts.gguf ?? {};
        const unet = ts.unet ?? {};
        // TECH DEBT: legacy Nunchaku settings are disabled. Delete after stale
        // workflow state no longer stores type_settings.nunchaku.
        // const nun  = ts.nunchaku ?? {};

        const field = (labelText, el) => {
            const w = document.createElement("div");
            w.className = "vnccs-cc-settings-field";
            const help = {
                "Weight Dtype": "Precision mode for UNet loading. Default follows the loader; fp8 modes can reduce memory use.",
                "dequant_dtype": "Dequantization precision for GGUF models.",
                // TECH DEBT: Nunchaku settings help removed with disabled UI.
                // "CPU Offload": "Controls whether Nunchaku can offload model blocks to CPU memory.",
                // "Blocks On GPU": "How many Nunchaku blocks should stay on GPU. Higher uses more VRAM and can be faster.",
                // "Pinned Memory": "Enables pinned CPU memory for Nunchaku transfers when supported."
            }[labelText];
            setHelpText(w, help);
            const l = document.createElement("label");
            l.className = "vnccs-cc-settings-label";
            l.textContent = labelText;
            w.append(l, el); return w;
        };
        const mkSel = (id, opts, val) => {
            const s = document.createElement("select");
            s.id = id;
            s.className = "vnccs-cc-settings-select";
            opts.forEach(o => {
                const op = document.createElement("option");
                op.value = op.textContent = o;
                if (o === val) op.selected = true;
                s.appendChild(op);
            });
            return s;
        };
        const title = document.createElement("h3");
        title.className = "vnccs-cc-settings-title";
        title.textContent = "⚙ Settings";
        panel.appendChild(title);

        // UNet
        const unetDet = document.createElement("details");
        unetDet.className = "vnccs-cc-settings-details";
        unetDet.open = sel === "unet";
        const unetSum = document.createElement("summary");
        unetSum.textContent = "UNet";
        unetDet.appendChild(unetSum);
        unetDet.appendChild(field("Weight Dtype",
            mkSel("vnccs-cc-unet-dtype",
                ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                unet.weight_dtype ?? "default")));
        panel.appendChild(unetDet);

        // GGUF
        const ggufDet = document.createElement("details");
        ggufDet.className = "vnccs-cc-settings-details";
        ggufDet.open = sel === "gguf";
        const ggufSum = document.createElement("summary");
        ggufSum.textContent = "GGUF";
        ggufDet.appendChild(ggufSum);
        ggufDet.appendChild(field("dequant_dtype",
            mkSel("vnccs-cc-gguf-dequant", ["default","float32","float16","bfloat16"],
                gguf.dequant_dtype ?? "default")));
        panel.appendChild(ggufDet);

        /*
        // TECH DEBT: Nunchaku settings UI is disabled. Delete this commented
        // legacy block after old workflow JSON no longer references Nunchaku.
        const nunDet = document.createElement("details");
        nunDet.className = "vnccs-cc-settings-details";
        nunDet.open = sel === "nunchaku";
        const nunSum = document.createElement("summary");
        nunSum.textContent = "Nunchaku";
        nunDet.appendChild(nunSum);
        nunDet.appendChild(field("CPU Offload",
            mkSel("vnccs-cc-nun-offload", ["auto","enable","disable"], nun.cpu_offload ?? "auto")));
        nunDet.appendChild(field("Blocks On GPU",
            mkInput("vnccs-cc-nun-blocks", "number", 1, 60, 1, nun.num_blocks_on_gpu ?? 1)));
        nunDet.appendChild(field("Pinned Memory",
            mkSel("vnccs-cc-nun-pin", ["enable","disable"], nun.use_pin_memory ?? "disable")));

        // Qwen Fix button
        const fixField = document.createElement("div");
        fixField.className = "vnccs-cc-settings-field";
        const fixLabel = document.createElement("label");
        fixLabel.textContent = "Qwen Fix (PR #790)";
        const fixBtn = this._btn("Checking…", () => {});
        fixBtn.disabled = true;
        fixBtn.style.fontSize = "10px";
        fixField.append(fixLabel, fixBtn);
        nunDet.appendChild(fixField);

        const applyQwenFix = async () => {
            fixBtn.textContent = "Installing…";
            fixBtn.disabled = true;
            try {
                const res = await api.fetchApi("/vnccs/control_center/nunchaku_apply_fix", { method: "POST" });
                const result = await res.json();
                if (result.ok) {
                    fixBtn.textContent = "✓ Installed";
                    fixBtn.style.color = "#4caf50";
                    fixBtn.style.borderColor = "rgba(76,175,80,0.4)";
                    this.showMessage("Fix applied. Please restart ComfyUI to apply changes.");
                } else {
                    fixBtn.textContent = "Install Fix";
                    fixBtn.disabled = false;
                    this.showMessage("Failed: " + (result.message || "Unknown error"), true);
                }
            } catch (e) {
                fixBtn.textContent = "Install Fix";
                fixBtn.disabled = false;
                this.showMessage("Error: " + e.message, true);
            }
        };

        api.fetchApi("/vnccs/control_center/nunchaku_fix_status").then(async r => {
            if (!r.ok) { fixBtn.textContent = "N/A"; return; }
            const status = await r.json();
            if (status.nunchaku_missing) {
                fixBtn.textContent = "N/A";
            } else if (status.installed) {
                fixBtn.textContent = "✓ Installed";
                fixBtn.style.color = "#4caf50";
                fixBtn.style.borderColor = "rgba(76,175,80,0.4)";
            } else {
                fixBtn.textContent = "Install Fix";
                fixBtn.disabled = false;
                fixBtn.onclick = applyQwenFix;
            }
        }).catch(() => { fixBtn.textContent = "N/A"; });

        panel.appendChild(nunDet);
        */

        // Buttons
        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";

        const cancelBtn = this._btn("Cancel", () => ov.remove());

        const saveBtn = this._btn("Save", async () => {
            this.state.selected_type = this._getSelectedType();
            if (!this.state.type_settings) this.state.type_settings = {};
            this.state.type_settings.unet = {
                weight_dtype: panel.querySelector("#vnccs-cc-unet-dtype")?.value ?? "default",
            };
            this.state.type_settings.gguf = {
                dequant_dtype: panel.querySelector("#vnccs-cc-gguf-dequant")?.value ?? "default",
            };
            // TECH DEBT: Nunchaku settings persistence disabled. Delete after
            // stale workflow state no longer stores type_settings.nunchaku.
            delete this.state.type_settings.nunchaku;
            this._saveState();
            await this._refreshDependencyStatus(true);
            ov.remove();
            this._renderAll();
        });
        saveBtn.classList.add("vnccs-cc-btn--save");
        btns.append(cancelBtn, saveBtn);
        panel.appendChild(btns);

        ov.appendChild(panel);
        this.container.appendChild(ov);
    }

    // ── Downloads ─────────────────────────────────────────────────────────────

    async _downloadEntry(cat, entry) {
        const repoId = this._getRepoId();
        if (!repoId) return;
        const key = `cc_${cat}_${entry.name}`;
        // Optimistic update with start time for fake progress
        this.dlStatus[key] = { status: "queued", message: "Queued…" };
        this.downloadStartTimes[key] = Date.now();
        this._renderAll();
        try {
            const r = await api.fetchApi("/vnccs/control_center/download", {
                method: "POST",
                headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                body: JSON.stringify({ repo_id: repoId, category: cat, name: entry.name }),
            });
            const responseText = await r.text();
            let d = {};
            try { d = responseText ? JSON.parse(responseText) : {}; } catch { d = {}; }
            if (!r.ok || d.error) {
                const message = d.error || responseText || `Download request failed (${r.status})`;
                this.dlStatus[key] = { status: "error", message };
                this._renderAll();
                this.showMessage(message, true);
            } else {
                window.dispatchEvent(new CustomEvent("vnccs-cc-registry-updated", {
                    detail: { repo_id: repoId },
                }));
            }
        } catch (e) {
            const message = String(e?.message || e);
            this.dlStatus[key] = { status: "error", message };
            this._renderAll();
            this.showMessage(message, true);
        }
    }

    async _downloadAllMissing() {
        if (!this.config) { this.showMessage("Load config first.", true); return; }

        const tasks = [];

        // MODEL: only the currently selected variant
        const selModel = this._getSelectedModelName();
        const modelEntry = (this.config.models ?? []).find(e => e.name === selModel);
        if (modelEntry && this._isDownloadableStatus(modelEntry.status)) {
            tasks.push({ cat: "models", e: modelEntry });
        }

        // Family assets follow the active top-level tab. Utility assets remain global.
        const selectedKind = this._selectedKind();
        for (const cat of ["clip", "vae", "lora", "controlnet", "other"]) {
            for (const e of (this.config[cat] ?? [])) {
                if (["clip", "vae"].includes(cat) && !this._sameKind(e, selectedKind)) continue;
                if (cat === "lora" && !e.custom && !this._exactKind(e, selectedKind)) continue;
                if (this._isDownloadableStatus(e.status)) tasks.push({ cat, e });
            }
        }

        if (!tasks.length) { this.showMessage("All selected items are installed."); return; }
        this.showConfirm(`Download/update ${tasks.length} item(s)?`, async () => {
            for (const { cat, e } of tasks) {
                await this._downloadEntry(cat, e);
                await new Promise(r => setTimeout(r, 300));
            }
        });
    }

    // ── Progress bar helper ───────────────────────────────────────────────────

    _applyProgressLayer(row, key, dls) {
        const status = dls?.status;
        if (!status || status === "installed" || status === "missing") return;

        const bg = document.createElement("div");
        bg.className = "vnccs-cc-row-bg";

        if (status === "downloading") {
            const elapsed = Date.now() - (this.downloadStartTimes[key] ?? Date.now());
            const progress = dls.progress ?? Math.min((elapsed / 30000) * 100, 95);
            bg.style.width = `${progress}%`;
            bg.style.background = "rgba(0, 214, 143, 0.15)";
            bg.style.transition = "width 0.3s linear";
        } else if (status === "queued") {
            bg.style.width = "100%";
            bg.style.background = "repeating-linear-gradient(45deg, rgba(184,169,232,0.12), rgba(184,169,232,0.12) 10px, transparent 10px, transparent 20px)";
        } else if (status === "auth_required") {
            bg.style.width = "100%";
            bg.style.background = "rgba(255,170,0,0.12)";
        } else if (status === "error") {
            bg.style.width = "100%";
            bg.style.background = "rgba(255,71,87,0.1)";
        }

        row.insertBefore(bg, row.firstChild);
    }

    // ── Styled message/confirm overlays (replaces alert/confirm) ──────────────

    showMessage(text, isError = false) {
        const ov = document.createElement("div");
        ov.style.cssText = `
            position: absolute; top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.75); display:flex; align-items:center;
            justify-content:center; z-index:1000; padding:16px; box-sizing:border-box;
        `;
        const box = document.createElement("div");
        box.style.cssText = `
            background: #12121a; border: 1px solid ${isError ? "rgba(255,71,87,0.4)" : "rgba(255,143,163,0.25)"};
            border-radius: 10px; padding: 16px; max-width: 280px; width:100%;
            text-align:center; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        `;
        const message = document.createElement("div");
        message.style.cssText = "color:#e8e8f0;font-size:12px;margin-bottom:14px;line-height:1.5;font-family:'Sora',sans-serif;white-space:pre-wrap;";
        message.textContent = String(text ?? "");
        box.appendChild(message);
        const btn = document.createElement("button");
        btn.textContent = "OK";
        btn.className = "vnccs-cc-btn";
        btn.style.cssText = "padding:5px 20px;font-weight:700;color:#ff8fa3;border-color:rgba(255,143,163,0.35);";
        btn.onclick = () => ov.remove();
        box.appendChild(btn);
        ov.appendChild(box);
        ov.onclick = e => { if (e.target === ov) ov.remove(); };
        this.container.appendChild(ov);
    }

    showConfirm(text, onConfirm) {
        const ov = document.createElement("div");
        ov.style.cssText = `
            position: absolute; top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.75); display:flex; align-items:center;
            justify-content:center; z-index:1000; padding:16px; box-sizing:border-box;
        `;
        const box = document.createElement("div");
        box.style.cssText = `
            background: #12121a; border: 1px solid rgba(255,143,163,0.25);
            border-radius: 10px; padding: 16px; max-width: 280px; width:100%;
            text-align:center; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        `;
        const message = document.createElement("div");
        message.style.cssText = "color:#e8e8f0;font-size:12px;margin-bottom:14px;line-height:1.5;font-family:'Sora',sans-serif;white-space:pre-wrap;";
        message.textContent = String(text ?? "");
        box.appendChild(message);
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:8px;justify-content:center;";
        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.className = "vnccs-cc-btn";
        cancelBtn.onclick = () => ov.remove();
        const okBtn = document.createElement("button");
        okBtn.textContent = "Confirm";
        okBtn.className = "vnccs-cc-btn vnccs-cc-btn--save";
        okBtn.onclick = () => { ov.remove(); onConfirm(); };
        row.append(cancelBtn, okBtn);
        box.appendChild(row);
        ov.appendChild(box);
        this.container.appendChild(ov);
    }

}
