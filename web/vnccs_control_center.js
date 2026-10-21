import { createSakuraSelect, openSakuraDropdown, closeSakuraDropdown, playSakuraMotion, stopSakuraMotion, animateSakuraIndicator } from "./sakura_interactions.mjs";
// web/vnccs_control_center.js
import { app } from "../../scripts/app.js";
import { vnccsApi as api, storage, sessionStore, serverRegistry, workflowScope } from "./vnccs_transport.js";
import { SAKURA_INTERACTION_CSS, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, createRequestGuard } from "./vnccs_common.js";

// Global registry cache — prevents API storms when multiple CC nodes exist


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
const DEFAULT_QI2_MODEL = "Qwen Image 2.1 INT8 ConvRot";
const QI2_TURBO_MODEL_NAME = "qwen_image_2.1_turbo_int8_convrot.safetensors";
const MODEL_FAMILIES = [
    { kind: "QI2", label: "Qwen Image 2.1", defaultType: "unet", preferredTypes: ["unet", "custom"], steps: 25, cfg: 3, sampler: "euler" },
    { kind: "Klein9b", label: "Flux Klein9b", defaultType: "unet", preferredTypes: ["unet", "custom"], steps: 4, sampler: "euler" },
    { kind: "MiniMaxH3", label: "MiniMax H3", defaultType: "unet", preferredTypes: ["unet", "custom"], steps: 8, sampler: "res_multistep" },
];

// ─── CSS injection (once per page load) ──────────────────────────────────────

function _injectVNCCSControlCenterStyles() {
    if (document.getElementById("vnccs-cc-styles")) return;
    const s = document.createElement("style");
    s.id = "vnccs-cc-styles";
    s.textContent = `
${SAKURA_INTERACTION_CSS}
/* ── VNCCS Control Center – Sakura Design ─────────────────────────────── */

.vnccs-cc-root {
    --vnccs-cc-accent: #ff8fa3;
    --vnccs-cc-muted: #9898a8;
    --vnccs-cc-control-height: 28px;
    --vnccs-cc-motion: 240ms;
    --vnccs-cc-ease: cubic-bezier(0.22, 1, 0.36, 1);
    min-width: 0;
    min-height: 0;
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
    position: relative;
    overflow: hidden;
    padding: 6px 10px;
    background: #12121a;
    border-bottom: 1px solid rgba(255,143,163,0.15);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}
.vnccs-cc-title {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
    font-weight: 700;
    color: #e8e8f0;
    letter-spacing: 0.06em;
}
.vnccs-cc-header-btns {
    flex-shrink: 0;
    margin-left: 8px;
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
.vnccs-cc-btn:hover:where(:not(.vnccs-cc-btn--active):not(.vnccs-cc-btn--clip-active):not(.vnccs-cc-btn--cnet-active)) {
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
.vnccs-cc-content {
    position: relative;
    display: flex;
    flex: 1;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
}
.vnccs-cc-scroll {
    flex: 1;
    min-height: 0;
    min-width: 0;
    overflow-anchor: none;
    overflow-y: auto;
    padding: 6px 7px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,143,163,0.2) transparent;
    transition: opacity 280ms ease-out, transform 280ms ease-out;
}
.vnccs-cc-scroll > * { flex-shrink: 0; }

/* Initial catalog skeleton and refresh feedback */
.vnccs-cc-loading {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 7px;
    overflow: hidden;
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transition: opacity 180ms ease-out, visibility 0s linear 180ms;
}
.vnccs-cc-loading-intro { padding: 8px 4px; }
.vnccs-cc-loading-title { font-size: 12px; font-weight: 700; color: #e8e8f0; }
.vnccs-cc-loading-hint { margin-top: 5px; font-size: 10px; color: var(--vnccs-cc-muted); }
.vnccs-cc-loading-tabs { display: flex; gap: 6px; }
.vnccs-cc-loading-tabs > * { flex: 1; height: 30px; }
.vnccs-cc-loading-block {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 80px;
    padding: 10px;
    gap: 8px;
    border: 1px solid rgba(255,143,163,0.13);
    border-radius: 10px;
    background: rgba(26,26,38,0.3);
}
.vnccs-cc-loading-bone { border-radius: 6px; background: rgba(255,143,163,0.07); }
.vnccs-cc-loading-heading { width: 34%; height: 10px; flex-shrink: 0; }
.vnccs-cc-loading-card { flex: 1; min-height: 28px; }
.vnccs-cc-loading-line { width: 68%; height: 8px; flex-shrink: 0; }
.vnccs-cc-loading::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg, transparent 35%, rgba(255,143,163,0.08) 50%, transparent 65%);
    animation: vnccs-cc-loading-sweep 1800ms ease-in-out infinite paused;
}
.vnccs-cc-root--config-skeleton .vnccs-cc-loading { opacity: 1; visibility: visible; transition: none; }
.vnccs-cc-root--config-skeleton .vnccs-cc-loading::after { animation-play-state: running; }
.vnccs-cc-root--config-skeleton .vnccs-cc-scroll { opacity: 0; transform: translateY(6px); pointer-events: none; }
.vnccs-cc-root--config-loading .vnccs-cc-header::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--vnccs-cc-accent), transparent);
    animation: vnccs-cc-loading-sweep 1600ms ease-in-out infinite;
}
.vnccs-cc-btn--download-all:disabled { opacity: 0.5; cursor: default; }
@keyframes vnccs-cc-loading-sweep {
    from { transform: translateX(-100%); }
    to { transform: translateX(100%); }
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
    color: var(--vnccs-cc-muted);
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
    width: 100%;
    min-height: var(--vnccs-cc-control-height);
    border: 0;
    font-family: inherit;
    text-align: left;
    gap: 8px;
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
    color: var(--vnccs-cc-muted);
    font-weight: 400;
    font-size: 9px;
    margin-left: 4px;
    letter-spacing: 0;
}
.vnccs-cc-block-arrow {
    font-size: 9px;
    color: var(--vnccs-cc-muted);
    transition: transform var(--vnccs-cc-motion) var(--vnccs-cc-ease);
}
.vnccs-cc-block--collapsed .vnccs-cc-block-arrow { transform: rotate(-90deg); }
.vnccs-cc-disclosure {
    display: grid;
    grid-template-rows: 1fr;
    opacity: 1;
    transition: grid-template-rows var(--vnccs-cc-motion) var(--vnccs-cc-ease), opacity 160ms ease-out;
}
.vnccs-cc-disclosure-inner { min-height: 0; overflow: hidden; }
.vnccs-cc-block--collapsed .vnccs-cc-disclosure { grid-template-rows: 0fr; opacity: 0; }
.vnccs-cc-root > .vnccs-cc-motion-ghost { position: absolute; pointer-events: none; z-index: 3; margin: 0 !important; }
.vnccs-cc-tab-glide {
    border: 1px solid rgba(255,143,163,0.55);
    border-radius: 8px;
    background: rgba(255,143,163,0.12);
    transform-origin: top left;
}
.vnccs-cc-root :is(button, [role="button"]) {
    transition: background-color 180ms ease, border-color 180ms ease, color 180ms ease, box-shadow 180ms ease, opacity 160ms ease, transform 160ms ease;
}
.vnccs-cc-root :is(button, [role="button"]):not(.vnccs-cc-btn--download-all):active:not(:disabled) { transform: translateY(1px); }
.vnccs-cc-badge { transition: color 200ms ease; }
.vnccs-cc-row-bg--indeterminate::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(100deg, transparent 30%, rgba(184,169,232,0.15) 50%, transparent 70%);
    animation: vnccs-cc-loading-sweep 1800ms ease-in-out infinite;
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
    color: var(--vnccs-cc-muted);
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
    width: min(24%, 130px);
    box-sizing: border-box;
    min-width: 0;
    flex-shrink: 0;
    padding: 4px 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(26,26,38,0.3);
}
.vnccs-cc-twocol-right2 {
    width: min(30%, 130px);
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
.vnccs-cc-row:hover:not(.vnccs-cc-row--model-sel):not(.vnccs-cc-row--clip-sel):not(.vnccs-cc-row--cnet-sel) {
    background: rgba(34,34,46,0.7);
}
.vnccs-cc-row-bg {
    position: absolute;
    top: 0; left: 0; height: 100%; width: 100%;
    z-index: 0;
    pointer-events: none;
    transform-origin: left center;
    transition: transform 360ms linear, opacity 180ms ease, background-color 200ms ease;
}
.vnccs-cc-root :is(.vnccs-cc-model-card, .vnccs-cc-lora-card, .vnccs-cc-turbo-strip, .vnccs-cc-row) {
    transition: border-color 200ms ease, background-color 200ms ease, box-shadow 200ms ease;
}
.vnccs-cc-root [data-download-status="error"] { border-color: rgba(255,71,87,0.45); }
.vnccs-cc-root [data-download-status="downloading"],
.vnccs-cc-root [data-download-status="success"],
.vnccs-cc-root [data-download-status="queued"] { border-color: rgba(184,169,232,0.4); }
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
    color: var(--vnccs-cc-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-cc-row-tag {
    color: var(--vnccs-cc-muted);
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
.vnccs-cc-badge { display: inline-grid; place-items: center; width: 14px; font-size: 12px; flex-shrink: 0; }
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
    grid-template-columns: repeat(3, minmax(0, 1fr));
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
    font-family: inherit;
    font-size: 10px;
    font-weight: 700;
    line-height: 1.3;
    min-height: 34px;
    text-wrap: balance;
    letter-spacing: 0;
    cursor: pointer;
}
.vnccs-cc-family-tab:hover:not(.vnccs-cc-family-tab--active) {
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
    min-width: 0;
    min-height: 30px;
    font-family: inherit;
    overflow-wrap: anywhere;
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
.vnccs-cc-model-tab:hover:not(.vnccs-cc-model-tab--active) {
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
    color: var(--vnccs-cc-muted);
    border-style: dashed;
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
.vnccs-cc-cache-strip {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    padding: 9px 12px 10px;
    border-radius: 10px;
    border: 1px solid rgba(184,169,232,0.18);
    background: linear-gradient(135deg, rgba(184,169,232,0.08) 0%, rgba(18,18,26,0.8) 100%);
}
.vnccs-cc-cache-strip .vnccs-cc-select {
    min-height: 28px;
}
.vnccs-cc-turbo-strip {
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 42px;
    padding: 8px 10px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(18,18,26,0.8);
    transition: border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
    position: relative;
}
.vnccs-cc-turbo-strip.is-installed { cursor: pointer; }
.vnccs-cc-turbo-strip.is-installed:hover:not(.is-active) {
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
    white-space: normal;
    overflow-wrap: anywhere;
}
.vnccs-cc-turbo-strip-status {
    max-width: 40%;
    overflow-wrap: anywhere;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: normal;
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
    height: 24px;
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
    pointer-events: none;
    position: absolute;
    inset: 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.08);
    transition: border-color 0.16s ease, background 0.16s ease;
}
.vnccs-cc-toggle-thumb {
    pointer-events: none;
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 999px;
    left: 3px;
    top: 5px;
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
    font-size: 10px;
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
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
}
.vnccs-cc-lora-grid--compact {
    align-items: start;
}
.vnccs-cc-lora-card {
    box-sizing: border-box;
    display: grid;
    grid-template-rows: 28px 14px 28px;
    height: 96px;
    min-width: 0;
    gap: 5px;
    padding: 7px 8px;
    cursor: default;
    transition: border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
}
.vnccs-cc-lora-card:hover:not(.vnccs-cc-lora-card--active) {
    border-color: rgba(255,143,163,0.32);
}
.vnccs-cc-lora-card--active {
    border-color: rgba(255,143,163,0.52);
    background: linear-gradient(180deg, rgba(255,143,163,0.14) 0%, rgba(10,26,22,0.92) 100%);
    box-shadow: inset 0 0 0 1px rgba(255,143,163,0.14);
}
.vnccs-cc-lora-card[role="button"] { cursor: pointer; }
.vnccs-cc-lora-card--compact {
    padding: 6px 7px;
    gap: 4px;
    grid-template-rows: 28px 14px;
    height: 60px;
}
.vnccs-cc-lora-card--compact .vnccs-cc-lora-card-meta {
    gap: 1px;
}
.vnccs-cc-lora-card-top {
    display: flex;
    flex-wrap: nowrap;
    align-items: flex-start;
    gap: 6px;
    height: 28px;
    min-width: 0;
}
.vnccs-cc-lora-card-meta {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex: 1 1 0;
    min-width: 0;
}
.vnccs-cc-lora-card-tools {
    display: flex;
    flex: 0 0 auto;
    flex-wrap: nowrap;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    height: var(--vnccs-cc-control-height);
}
.vnccs-cc-lora-card--compact .vnccs-cc-lora-card-status {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.vnccs-cc-model-card-status[hidden],
.vnccs-cc-lora-card-status[hidden],
.vnccs-cc-turbo-strip-status[hidden],
.vnccs-cc-download-slot[hidden],
.vnccs-cc-download-btn[hidden] { display: none; }
.vnccs-cc-lora-card-name {
    font-size: 10.5px;
    font-weight: 700;
    color: #e8e8f0;
    line-height: 14px;
    height: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vnccs-cc-lora-card--compact .vnccs-cc-lora-card-name {
    font-size: 10px;
}
.vnccs-cc-lora-card-status {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.vnccs-cc-lora-card-status--ok {
    color: #00d68f;
}
.vnccs-cc-lora-card-status--active {
    color: #ff8fa3;
}
.vnccs-cc-lora-card-status--pipe {
    color: #00d68f;
}
.vnccs-cc-lora-card-status--progress {
    color: #b8a9e8;
}
.vnccs-cc-lora-card-status--missing {
    color: #ff4757;
}
.vnccs-cc-lora-card-status--corner {
    line-height: 1.35;
}
.vnccs-cc-lora-remove-btn {
    flex: 0 0 26px;
    min-width: 26px;
    height: 24px;
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
.vnccs-cc-download-btn {
    flex-shrink: 0;
    min-width: 58px;
    height: 24px;
    padding: 0 7px;
    border-radius: 6px;
    border: 1px solid rgba(255,143,163,0.32);
    background: rgba(255,143,163,0.08);
    color: #ff8fa3;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1;
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
}
.vnccs-cc-download-btn:hover {
    background: rgba(255,143,163,0.2);
    border-color: rgba(255,143,163,0.52);
}
.vnccs-cc-download-slot {
    display: grid;
    place-items: center end;
    flex: 0 0 80px;
    width: 80px;
    height: var(--vnccs-cc-control-height);
    min-width: 0;
    font-variant-numeric: tabular-nums;
}
.vnccs-cc-download-slot > * { grid-area: 1 / 1; min-width: 0; max-width: 100%; box-sizing: border-box; }
.vnccs-cc-download-slot > .vnccs-cc-download-btn { width: 100%; height: 100%; }
.vnccs-cc-download-slot > :not(button) {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: 1.35;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: none;
    transition: color 180ms ease;
}
.vnccs-cc-root [data-download-key] > .vnccs-cc-row-bg { top: auto; bottom: 0; height: 3px; }
.vnccs-cc-row-bg--indeterminate { animation: vnccs-cc-download-flow 1200ms linear infinite; }
@keyframes vnccs-cc-download-flow { to { background-position: 28px 0; } }
.vnccs-cc-lora-card-desc {
    font-size: 10px;
    color: #a3a3b3;
    line-height: 14px;
    height: 14px;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vnccs-cc-lora-card-detail-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 40px;
    align-items: center;
    gap: 6px;
    min-width: 0;
    height: 14px;
}
.vnccs-cc-lora-card-version {
    height: 14px;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: right;
    color: var(--vnccs-cc-muted);
    font-size: 9px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    line-height: 14px;
    white-space: nowrap;
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
    grid-template-rows: repeat(2, auto);
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
    height: var(--vnccs-cc-control-height);
    line-height: 1.2;
    padding: 3px 18px 3px 4px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='20' viewBox='0 0 12 20'%3E%3Cpath d='M1 8l5-5 5 5' fill='none' stroke='%23f1eef7' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M1 12l5 5 5-5' fill='none' stroke='%23f1eef7' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 4px center;
    background-size: 8px 16px;
}
.vnccs-cc-param-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.vnccs-cc-param-label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    line-height: 12px;
    font-size: 9px;
    color: var(--vnccs-cc-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.vnccs-cc-input {
    min-width: 0;
    height: var(--vnccs-cc-control-height);
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
    min-width: 0;
    height: var(--vnccs-cc-control-height);
    font-family: inherit;
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
    min-width: 0;
    min-height: 24px;
    margin: 0;
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
    box-sizing: border-box;
    min-width: 0;
    overflow-wrap: anywhere;
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
.vnccs-cc-settings-summary {
    width: 100%;
    border: 0;
    background: none;
    font-family: inherit;
    text-align: left;
    font-size: 10px;
    color: #9898a8;
    cursor: pointer;
    padding: 4px 0;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    transition: color 0.15s;
}
.vnccs-cc-settings-summary:hover { color: #ff8fa3; }
.vnccs-cc-settings-summary::after { content: "⌄"; float: right; transition: transform var(--vnccs-cc-motion) ease; }
.vnccs-cc-block--collapsed .vnccs-cc-settings-summary::after { transform: rotate(-90deg); }
.vnccs-cc-settings-field {
    margin-top: 6px;
}
.vnccs-cc-settings-label {
    font-size: 9px;
    color: var(--vnccs-cc-muted);
    display: block;
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.vnccs-cc-settings-select {
    min-width: 0;
    min-height: var(--vnccs-cc-control-height);
    font-family: inherit;
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
    flex-wrap: wrap;
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
    flex-wrap: wrap;
    justify-content: flex-end;
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
    color: var(--vnccs-cc-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: 2px;
    flex-shrink: 0;
}
.vnccs-cc-pill {
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
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
    white-space: normal;
}
.vnccs-cc-pill--dependency {
    padding-inline: 6px;
    max-width: 100%;
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
    color: var(--vnccs-cc-muted);
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
    overflow-wrap: anywhere;
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
.vnccs-cc-root :is(button, input, select, summary, [tabindex]):focus-visible {
    outline: 2px solid #ff8fa3;
    outline-offset: -2px;
}
.vnccs-cc-toggle input:focus-visible ~ .vnccs-cc-toggle-track {
    outline: 2px solid #ff8fa3;
    outline-offset: 2px;
}
.vnccs-cc-root button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}
.vnccs-cc-param-label :is(input, select),
.vnccs-cc-settings-label :is(input, select) { text-transform: none; letter-spacing: normal; }
.vnccs-cc-row-progress { max-width: 45%; white-space: normal; overflow-wrap: anywhere; }
@media (prefers-reduced-motion: reduce) {
    .vnccs-cc-root * { transition: none !important; animation: none !important; }
    .vnccs-cc-root :is(button, [role="button"], .vnccs-cc-scroll) { transform: none !important; }
    .vnccs-cc-root *::after { animation: none !important; transition: none !important; }
    .vnccs-cc-loading::after, .vnccs-cc-header::after, .vnccs-cc-row-bg::after { transform: none !important; }
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
            this._cc_widget?._closeDropdown(false);
            syncDOMWidgetWidth(this, "cc_widget");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "cc_widget"));
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            if (this._cc_widget) this._cc_widget._removed = true;
            this._cc_widget?._stopMotion();
            clearTimeout(this._cc_widget?._initTimer);
            if (this._cc_widget?.pollingInterval)
                clearInterval(this._cc_widget.pollingInterval);
            if (this._cc_widget?._onGlobalPointerUp)
                window.removeEventListener("pointerup", this._cc_widget._onGlobalPointerUp);
            if (this._cc_widget?._onRegistryUpdate)
                window.removeEventListener("vnccs-cc-registry-updated", this._cc_widget._onRegistryUpdate);
            for (const overlay of [...(this._cc_widget?._overlays ?? [])]) overlay.remove();
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
        this._beginModuleStatusRequest = createRequestGuard(node);
        this._beginConfigRequest = createRequestGuard(node);
        this._beginDownloadStatusRequest = createRequestGuard(node);
        this.config   = null;
        this.state    = {};
        this.dlStatus = {};
        this._draggingLoraSlider = false;
        this._onGlobalPointerUp = null;
        this._downloadRefreshPending = false;
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

        this._initTimer = setTimeout(() => {
            if (this._removed) return;
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
            if (repoId && repoId === myRepo && serverRegistry("VNCCS_CC_REGISTRY")[repoId]) {
                if (this.config === serverRegistry("VNCCS_CC_REGISTRY")[repoId]) return;
                this.config = serverRegistry("VNCCS_CC_REGISTRY")[repoId];
                this._syncCustomModelInput();
                if (e.detail.download_update) this._updateDownloadIndicators();
                else if (!this._isUserInteracting()) this._renderAll();
            }
        };
        window.addEventListener("vnccs-cc-registry-updated", this._onRegistryUpdate);

        this._onGlobalPointerUp = () => {
            this._draggingLoraSlider = false;
        };
        window.addEventListener("pointerup", this._onGlobalPointerUp);
    }

    _getModelTypeTabs() {
        // Only supported model types are selectable for the active family.
        const activeKind = this._activeKind();
        const preferred = this._familyDefinition(activeKind).preferredTypes;
        const available = new Set((this.config?.models ?? [])
            .filter(entry => this._metaKind(entry).toLowerCase() === activeKind.toLowerCase())
            .map(entry => entry.type)
            .filter(Boolean));
        available.add("custom");
        if (activeKind === "QI2") available.add("unet");
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

    _getCustomAudioVaeInputIndex() {
        return (this.node.inputs ?? []).findIndex(input => input?.name === "audio_vae");
    }

    _getStoredSelectedType() {
        const kind = this._activeKind();
        const selectedForKind = this.state.selected_types_by_kind?.[kind];
        if (selectedForKind) return selectedForKind;
        return kind === "QI2" ? (this.state.selected_type || "") : "";
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
        sync("audio_vae", "VAE", this._getCustomAudioVaeInputIndex, isCustom && this._activeKind() === "MiniMaxH3");
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
        this._renderAll({ preserveDraft: false });
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
        const rawKind = String(this.state.active_kind || "QI2");
        const compactKind = rawKind.toLowerCase().replace(/[^a-z0-9]/g, "");
        const kind = compactKind === "h3" || compactKind === "minimaxh3" ? "MiniMaxH3" : rawKind;
        return MODEL_FAMILIES.some(entry => entry.kind === kind) ? kind : "QI2";
    }

    _familyDefinition(kind = this._activeKind()) {
        return MODEL_FAMILIES.find(entry => entry.kind === kind) || MODEL_FAMILIES[0];
    }

    _getSelectedType(){
        const visibleTabs = this._getModelTypeTabs();
        const selected = this.state.selected_types_by_kind?.[this._activeKind()]
            ?? (this._activeKind() === "QI2" ? this.state.selected_type : "");
        if (selected && visibleTabs.includes(selected)) return selected;
        const preferred = this._familyDefinition().defaultType;
        return visibleTabs.includes(preferred) ? preferred : (visibleTabs[0] || "");
    }

    _getSelectedModelName(type = this._getSelectedType()) {
        const selectedByType = this.state.selected_models ?? {};
        const familyKey = `${this._activeKind()}:${type}`;
        const isCustom = this._getSelectedType() === "custom";
        const savedName = selectedByType[familyKey] || (isCustom ? selectedByType[type] : "") || this.state.selected_model;
        const variants = this._visibleModelsByType(type);
        const findVariant = name => variants.find(entry => entry.name.trim().toLowerCase() === name?.trim().toLowerCase());
        const selected = findVariant(savedName) || (!selectedByType[familyKey] && !isCustom && this._activeKind() === "QI2"
            ? findVariant(selectedByType[type]) : null);
        if (selected) return selected.name;
        if (selectedByType[familyKey]) return selectedByType[familyKey];
        if (this._activeKind() === "QI2") {
            return type === "unet" ? (variants[0]?.name || DEFAULT_QI2_MODEL) : "";
        }
        return "";
    }

    _setSelectedModelName(type, modelName) {
        if (!type || !modelName) return;
        const previousName = this._getSelectedModelName(type);
        const previous = this.config?.models?.find(entry => entry.name === previousName);
        const selected = this.config?.models?.find(entry => entry.name === modelName);
        const turboModel = this._isQi2TurboModel(selected);
        if (modelName !== previousName && (turboModel || this._isQi2TurboModel(previous))) {
            const params = this._currentModelParams();
            params.steps = turboModel ? 8 : 25;
            params.cfg = turboModel ? 1.0 : 3.0;
            params.turbo_previous_settings = null;
            const turboNames = new Set(this._compatibleTurboLoras().map(entry => entry.name));
            for (const lora of this.state.loras || []) {
                if (turboNames.has(lora.name)) lora.auto_apply = false;
            }
        }
        if (!this.state.selected_models) this.state.selected_models = {};
        this.state.selected_models[`${this._activeKind()}:${type}`] = modelName;
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
        return variants.find(m => m.name === selectedModel) ?? variants[0] ?? {
            name: `Custom ${this._activeKind()}`,
            type: "custom",
            kind: this._activeKind(),
            custom: true,
        };
    }

    _visibleModelsByType(type) {
        if (type === "custom") return [];
        const activeKind = this._activeKind().toLowerCase();
        return (this.config?.models || []).filter(m =>
            (!m.type || m.type === type) && this._metaKind(m).toLowerCase() === activeKind
        ).sort((a, b) => Number(b.name === DEFAULT_QI2_MODEL) - Number(a.name === DEFAULT_QI2_MODEL));
    }

    _metaKind(entry) {
        const value = String(entry?.kind ?? entry?.Kind ?? "").trim();
        const compact = value.toLowerCase().replace(/[^a-z0-9]/g, "");
        return compact === "h3" || compact === "minimaxh3" ? "MiniMaxH3" : value;
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
        const family = this._familyDefinition(kind);
        const turboModel = this._isQi2TurboModel(this._getSelectedModelEntry());
        this.state.model_params_by_kind[kind] = {
            steps: turboModel ? 8 : family.steps ?? DEFAULT_MODEL_STEPS,
            cfg: turboModel ? 1.0 : family.cfg ?? DEFAULT_MODEL_CFG,
            sampler: family.sampler ?? "euler",
            scheduler: DEFAULT_MODEL_SCHEDULER,
            ...(this.state.model_params_by_kind[kind] || {}),
        };
        return this.state.model_params_by_kind[kind];
    }

    _isQi2TurboModel(entry) {
        return this._metaKind(entry).toLowerCase() === "qi2"
            && String(entry?.local_path || "").replace(/\\/g, "/").split("/").pop() === QI2_TURBO_MODEL_NAME;
    }

    _syncActiveFamilyState() {
        const kind = this._activeKind();
        const type = this._getSelectedType();
        if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
        this.state.active_kind = kind;
        this.state.selected_types_by_kind[kind] = type;
        this.state.selected_type = type;
        this.state.selected_model = this._getSelectedModelName(type);
        if (this.state.selected_model) this._setSelectedModelName(type, this.state.selected_model);
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
        this._renderAll({ preserveDraft: false });
        this.scrollArea?.querySelector(`.vnccs-cc-family-tab[data-kind="${kind}"]`)?.focus({ preventScroll: true });
        this._checkH3Compatibility();
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

    _setTurboPreset(enabled) {
        const params = this._currentModelParams();

        if (enabled) {
            params.turbo_previous_settings = {
                steps: params.steps ?? DEFAULT_MODEL_STEPS,
                cfg: params.cfg ?? DEFAULT_MODEL_CFG,
            };
            params.steps = this._activeKind() === "QI2" ? 6 : 4;
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

    _saveState() {
        if (this._stateRestoreError || this._removed) return;
        this._syncActiveFamilyState();
        const w = this._getStateWidget();
        if (w) w.value = JSON.stringify(this.state);
        this.node.setDirtyCanvas(true, true);
        this._dispatchLoraOptions();
        window.dispatchEvent(new CustomEvent("vnccs-control-center-model-changed", {
            detail: { node_id: this.node.id },
        }));
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
        this._closeDropdown(false);
        const w = this._getStateWidget();
        this._stateRestoreError = null;
        if (w) {
            try {
                const state = JSON.parse(w.value === "" ? "{}" : w.value);
                const isObject = value => Boolean(value) && typeof value === "object" && !Array.isArray(value);
                if (!isObject(state)) throw new Error("State must be an object");
                for (const key of ["selected_models", "selected_types_by_kind", "model_params", "model_params_by_kind", "type_settings", "qi2_cache", "collapsed"]) {
                    if (state[key] !== undefined && !isObject(state[key])) throw new Error(`${key} must be an object`);
                }
                for (const key of ["model_params_by_kind", "type_settings"]) {
                    if (Object.values(state[key] || {}).some(value => !isObject(value))) throw new Error(`${key} entries must be objects`);
                }
                for (const key of ["selected_models", "selected_types_by_kind"]) {
                    if (Object.values(state[key] || {}).some(value => typeof value !== "string")) throw new Error(`${key} entries must be strings`);
                }
                for (const key of ["active_kind", "selected_type", "selected_model", "unsupported_model_kind"]) {
                    if (state[key] !== undefined && typeof state[key] !== "string") throw new Error(`${key} must be a string`);
                }
                if (state.loras !== undefined && (!Array.isArray(state.loras) || state.loras.some(lora =>
                    !isObject(lora) || (lora.name !== undefined && typeof lora.name !== "string")))) {
                    throw new Error("LoRA states must be objects in an array");
                }
                const isFiniteNumber = (value, integer = false) => {
                    if (typeof value === "string") {
                        const decimal = integer ? /^[+-]?\d(?:_?\d)*$/
                            : /^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:e[+-]?\d(?:_?\d)*)?$/i;
                        return decimal.test(value.trim()) && Number.isFinite(Number(value.replaceAll("_", "")));
                    }
                    return ["number", "boolean"].includes(typeof value) && Number.isFinite(Number(value));
                };
                for (const params of [state.model_params || {}, ...Object.values(state.model_params_by_kind || {})]) {
                    for (const field of ["steps", "cfg"]) {
                        const value = params[field];
                        if (value != null && !(field === "steps" && value === "") && !isFiniteNumber(value, field === "steps")) {
                            throw new Error(`${field} must be a finite number`);
                        }
                    }
                    for (const field of ["sampler", "scheduler"]) {
                        if (params[field] != null && typeof params[field] !== "string") throw new Error(`${field} must be a string`);
                    }
                }
                for (const field of ["device", "dtype"]) {
                    const value = state.qi2_cache?.[field];
                    if (value != null && typeof value !== "string") throw new Error(`qi2_cache.${field} must be a string`);
                }
                if (state.loras?.some(lora => lora.strength !== undefined && !isFiniteNumber(lora.strength))) {
                    throw new Error("LoRA strength must be a finite number");
                }
                this.state = state;
            } catch (error) {
                this._stateRestoreError = error;
                this._setConfigLoading(false);
                this.showMessage("Control Center state could not be restored. Original workflow data is preserved; repair node_state before editing.", true);
                return;
            }
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
        if (this.state.active_kind === "QIE2511" ||
            (!this.state.active_kind && String(this.state.selected_model || "").includes("2511"))) {
            this.state.unsupported_model_kind = "QIE2511";
            this.state.active_kind = "QI2";
            this.state.selected_type = "unet";
            this.state.selected_model = DEFAULT_QI2_MODEL;
        }
        this.state.active_kind = this._activeKind();
        if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
        const kind = this._activeKind();
        if (this.state.model_params && typeof this.state.model_params === "object") {
            this.state.model_params_by_kind ??= {};
            this.state.model_params_by_kind[kind] = {
                ...this.state.model_params,
                ...this.state.model_params_by_kind[kind],
            };
        }
        if (!this.state.selected_types_by_kind[kind] && this.state.selected_type) {
            this.state.selected_types_by_kind[kind] = this.state.selected_type;
        }
        // Control Center now exposes a single pipe output.
        this.state.output_slot_names = [];
        this._syncOutputSlots();
        this._syncCustomModelInput();
        this._checkH3Compatibility();
        const repo = this._getRepoId();
        if (repo) this.fetchConfig(repo);
    }

    // ── Polling ───────────────────────────────────────────────────────────────

    _isUserInteracting() {
        // Skip re-render if any focusable element inside our container is active
        // (e.g. an open dropdown or a focused input)
        if (this._draggingLoraSlider) return true;
        const active = document.activeElement;
        return active && this.container?.contains(active);
    }

    _startPolling() {
        if (this.pollingInterval) clearInterval(this.pollingInterval);
        this.pollingInterval = setInterval(async () => {
            if (this._removed || this._downloadStatusPolling || document.visibilityState === "hidden") return;
            this._downloadStatusPolling = true;
            const isCurrent = this._beginDownloadStatusRequest();
            const repoId = this._getRepoId();
            try {
                const r = await api.fetchApi(`/vnccs/manager/status?repo_id=${encodeURIComponent(repoId)}`);
                if (!r.ok) return;
                const newStatuses = await r.json();
                if (!isCurrent() || this._getRepoId() !== repoId) return;
                for (const key of this._pendingDownloads?.keys() || []) {
                    if (!["queued", "downloading"].includes(newStatuses[key]?.status)) newStatuses[key] = this.dlStatus[key];
                }
                const changedStatuses = JSON.stringify(this.dlStatus) !== JSON.stringify(newStatuses);

                // Detect transitions to "success" → re-fetch config to update disk state
                for (const key in newStatuses) {
                    const s = newStatuses[key];
                    const old = this.dlStatus[key];
                    if (s?.status === "success" && old?.status !== "success") {
                        this._downloadRefreshPending = true;
                        break;
                    }
                }

                this._downloadStatusRepoId = repoId;
                this.dlStatus = newStatuses;
                if (changedStatuses) window.dispatchEvent(new CustomEvent("vnccs-cc-registry-updated", {
                    detail: { repo_id: repoId, download_status: newStatuses },
                }));
                this._updateDownloadIndicators();

                if (this._downloadRefreshPending) {
                    const repo = this._getRepoId();
                    if (repo) await this.fetchConfig(repo, true, { preserveCards: true });
                }
            } catch { /* ignore */ }
            finally { this._downloadStatusPolling = false; }
        }, 2000);
    }

    // ── Container layout ──────────────────────────────────────────────────────

    _buildUI() {
        // Create container first, then pass it to addDOMWidget (same pattern as Pose Studio)
        const c = document.createElement("div");
        c.className = "vnccs-cc-root vnccs-common-ui-host";
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
            const node = this.node;
            const serializeValue = stateW.serializeValue;
            stateW.serializeValue = function () {
                const raw = serializeValue?.apply(this, arguments) ?? this.value;
                try {
                    const state = JSON.parse(raw);
                    if (!state || typeof state !== "object" || Array.isArray(state)) return raw;
                    return JSON.stringify({ ...state, preview_scope: workflowScope(node) });
                } catch { return raw; }
            };
            stateW.computeSize = () => [0, -4];
            stateW.type = "hidden";
            stateW.draw = () => {};
            if (stateW.element) stateW.element.style.display = "none";
        }

        this._buildStatusStrip();
        this._buildHeader();
        this._buildScroll();
        this._buildFooter();
        this._setConfigLoading(Boolean(this._getRepoId()));
    }

    _buildHeader() {
        const h = document.createElement("div");
        h.className = "vnccs-cc-header";

        this.statusText = document.createElement("span");
        this.statusText.className = "vnccs-cc-title";
        this.statusText.textContent = "VNCCS Control Center";
        this.statusText.setAttribute("role", "status");

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
        const content = document.createElement("div");
        content.className = "vnccs-cc-content";
        this.scrollArea = document.createElement("div");
        this.scrollArea.className = "vnccs-cc-scroll";
        this.scrollArea.addEventListener("scroll", () => this._closeDropdown(false), { passive: true });

        const empty = document.createElement("div");
        empty.className = "vnccs-cc-empty";
        empty.textContent = "Enter repo_id and click ↺ to load";
        this.scrollArea.appendChild(empty);

        const loading = document.createElement("div");
        loading.className = "vnccs-cc-loading";
        loading.setAttribute("aria-hidden", "true");
        const intro = document.createElement("div");
        intro.className = "vnccs-cc-loading-intro";
        for (const [name, text] of [["title", "Loading configuration"], ["hint", "Preparing models and controls…"]]) {
            const line = document.createElement("div");
            line.className = `vnccs-cc-loading-${name}`;
            line.textContent = text;
            intro.appendChild(line);
        }
        loading.appendChild(intro);
        const tabs = document.createElement("div");
        tabs.className = "vnccs-cc-loading-tabs";
        for (const family of MODEL_FAMILIES) {
            const tab = document.createElement("div");
            tab.className = "vnccs-cc-loading-bone";
            tabs.appendChild(tab);
        }
        loading.appendChild(tabs);
        for (let index = 0; index < 3; index++) {
            const block = document.createElement("div");
            block.className = "vnccs-cc-loading-block";
            for (const name of ["heading", "card", "line"]) {
                const bone = document.createElement("div");
                bone.className = `vnccs-cc-loading-bone vnccs-cc-loading-${name}`;
                block.appendChild(bone);
            }
            loading.appendChild(block);
        }
        content.append(this.scrollArea, loading);
        this.container.appendChild(content);
    }

    _setConfigLoading(loading) {
        const showSkeleton = loading && !this.config;
        this.container.classList.toggle("vnccs-cc-root--config-loading", loading);
        this.container.classList.toggle("vnccs-cc-root--config-skeleton", showSkeleton);
        this.scrollArea.setAttribute("aria-busy", String(loading));
        this.scrollArea.setAttribute("aria-hidden", String(showSkeleton));
        this.scrollArea.inert = showSkeleton;
        this.downloadAllButton.disabled = loading || !this.config;
    }

    _buildFooter() {
        const f = document.createElement("div");
        f.className = "vnccs-cc-footer";

        const b = document.createElement("button");
        b.className = "vnccs-cc-btn--download-all";
        b.textContent = "⬇ Download / Update";
        b.onclick = () => this._downloadAllMissing();
        this.downloadAllButton = b;

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
        sessionStore.setItem(PENDING_DEPENDENCY_INSTALLS_KEY, JSON.stringify(keys));
    }

    async _resumePendingDependencyInstalls(items) {
        let keys = [];
        try {
            keys = JSON.parse(sessionStore.getItem(PENDING_DEPENDENCY_INSTALLS_KEY) || "[]");
        } catch {
            keys = [];
        }
        sessionStore.removeItem(PENDING_DEPENDENCY_INSTALLS_KEY);
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
        if (this._removed) return false;
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
        this._mountOverlay(ov, panel, title.textContent);
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
                if (this._removed) {
                    this._dependencyInstallTasks.delete(uiId);
                    return null;
                }
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
        if (this._removed) return false;
        const originalLabel = button.textContent;
        if (!options.skipPolicyCheck && !await this._ensureManagerInstallPolicy([item])) return false;
        button.disabled = true;
        button.textContent = "Queueing…";
        try {
            if (!await this._queueDependencyInstall(item, button)) return false;
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

    _checkH3Compatibility() {
        if (this._removed || !this._h3Runtime) return;
        if (this._activeKind() !== "MiniMaxH3" || this._h3Runtime?.h3_still_supported === true) {
            this._h3UpdateOverlay?.remove();
            this._h3WarningShown = false;
            return;
        }
        if (this._h3Runtime.h3_still_supported !== false || this._h3WarningShown) return;
        this._h3WarningShown = true;
        this._showH3UpdateModal();
    }

    _showH3UpdateModal() {
        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";
        this._h3UpdateOverlay = ov;
        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";
        panel.style.maxWidth = "460px";
        const title = document.createElement("div");
        title.className = "vnccs-cc-deps-modal-title";
        title.textContent = "ComfyUI update required for MiniMax H3";
        const text = document.createElement("div");
        text.className = "vnccs-cc-deps-modal-text";
        text.textContent = `${this._h3Runtime.h3_update_message} Detected ComfyUI version: ${this._h3Runtime.version || "unknown"}.`;
        const buttons = document.createElement("div");
        buttons.className = "vnccs-cc-settings-btns";
        const details = document.createElement("a");
        details.className = "vnccs-cc-btn";
        details.textContent = "View required ComfyUI fix";
        details.href = "https://github.com/Comfy-Org/ComfyUI/pull/16881";
        details.target = "_blank";
        details.rel = "noopener noreferrer";
        const close = this._btn("Close", () => ov.remove());
        buttons.append(details, close);
        panel.append(title, text, buttons);
        ov.appendChild(panel);
        ov.onclick = event => { if (event.target === ov) ov.remove(); };
        this._mountOverlay(ov, panel, title.textContent);
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
        this._mountOverlay(ov, panel, title.textContent);
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
        this._mountOverlay(ov, panel, title.textContent);
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

    async _fetchLatestModuleVersion(managerId) {
        try {
            const response = await api.fetchApi(`/customnode/versions/${encodeURIComponent(managerId)}`, {
                cache: "no-store", signal: AbortSignal.timeout(8000),
            });
            if (!response.ok) return null;
            const versions = await response.json();
            if (!Array.isArray(versions)) return null;
            return versions.map(item => item?.version).filter(version => /^\d+\.\d+\.\d+$/.test(version))
                .reduce((latest, version) => !latest || this._semverGt(version, latest) ? version : latest, null);
        } catch {
            return null;
        }
    }

    async _fetchModuleStatus() {
        const isCurrent = this._beginModuleStatusRequest();
        const LABELS = { main: "VNCCS", utils: "Utils" };
        const PILLS  = { main: this._pillMain, utils: this._pillUtils };

        // Fetch installed module versions from the local ComfyUI backend.
        let local = {};
        try {
            const r = await api.fetchApi("/vnccs/module_status");
            if (r.ok) local = await r.json();
        } catch { /* server may not be ready yet */ }
        if (!isCurrent()) return;
        this._h3Runtime = local.comfyui;
        this._checkH3Compatibility();

        const latestVersions = Object.fromEntries(await Promise.all(
            Object.entries({ main: "vnccs", utils: "vnccs-utils" }).map(async ([key, managerId]) => [
                key, local[key]?.version && !local[key].error && !local[key].duplicate
                    ? await this._fetchLatestModuleVersion(managerId) : null,
            ])
        ));
        if (!isCurrent()) return;

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

            const latest = latestVersions[key];
            if (!latest || !/^\d+\.\d+\.\d+$/.test(locVer)) {
                this._updatePill(pill, label, locVer, "warning", "update check unavailable");
            } else if (this._semverGt(latest, locVer)) {
                this._updatePill(pill, label, locVer, "update", latest);
                updateNeeded.push(`${label} v${locVer} → v${latest}`);
            } else {
                this._updatePill(pill, label, locVer, "ok");
                pill.title = `No newer release in Comfy Registry (latest v${latest}).`;
            }
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
        if (!isCurrent()) return;
        if (!resumedPendingInstalls) this._showMissingDependenciesModal(missingDependencies);

        this._showUpdateBanner(updateNeeded);
    }

    // ── Primitives ────────────────────────────────────────────────────────────

    _btn(label, onClick, title = "") {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "vnccs-cc-btn";
        b.textContent = label;
        if (title) {
            b.title = title;
            b.ariaLabel = title;
        }
        b.onclick = onClick;
        return b;
    }

    _downloadButton(cat, entry, status) {
        const action = status === "outdated" ? "Update" : "Download";
        const button = this._btn(action, event => {
            event.stopPropagation();
            this._downloadEntry(cat, entry);
        }, `${action} ${entry.name}`);
        button.className = "vnccs-cc-download-btn";
        button.dataset.focusKey = `download-${cat}-${entry.name}`;
        setHelpText(button, this._statusLabel(status, this.dlStatus[`cc_${cat}_${entry.name}`]));
        return button;
    }

    _mountDownloadStatus(card, parent, cat, entry, statusClass, mode = "") {
        const slot = document.createElement("div");
        slot.className = "vnccs-cc-download-slot";
        const label = document.createElement("span");
        label.className = statusClass;
        label.dataset.focusKey = `status-${cat}-${entry.name}`;
        label.setAttribute("role", "status");
        const button = !entry.custom || entry.hf_path ? this._downloadButton(cat, entry, entry.status) : null;
        slot.appendChild(label);
        if (button) slot.appendChild(button);
        parent.appendChild(slot);
        card._downloadView = { cat, entry, slot, label, button, statusClass, mode,
            badge: card.querySelector(".vnccs-cc-badge") };
    }

    _updateDownloadCard(card, animate = true) {
        const view = card._downloadView;
        if (!view) return;
        const { cat, slot, label, button, badge, statusClass, mode } = view;
        const entry = this.config?.[cat]?.find(item => item.name === view.entry.name) || view.entry;
        view.entry = entry;
        const key = `cc_${cat}_${entry.name}`;
        const dls = this.dlStatus[key] || {};
        const status = dls.status === "success" && entry.status !== "installed" && this._downloadRefreshPending ? "success"
            : this._resolveStatus(dls.status, entry.status);
        const installed = status === "installed";
        const active = installed && cat === "lora" && mode !== "pipe" && (this.state.loras || []).some(item => item.name === entry.name && item.auto_apply);
        const text = installed ? (active ? "Active" : mode === "pipe" || mode === "custom" ? "Pipe" : "Installed")
            : status === "auth_required" ? "Key required" : this._statusLabel(status, dls, true);
        const fullText = status === "success" ? "Checking installation…" : installed ? text : this._statusLabel(status, dls);
        const downloadable = !!button && this._isDownloadableStatus(status);
        const focused = slot.contains(document.activeElement) ? document.activeElement : null;
        label.className = `${statusClass} ${statusClass}--${active ? "active" : installed ? "ok"
            : ["queued", "downloading", "success"].includes(status) ? "progress" : "missing"}`;
        if (label.textContent !== text) label.textContent = text;
        slot.hidden = installed && mode === "turbo";
        label.hidden = downloadable || slot.hidden;
        label.tabIndex = label.hidden ? -1 : 0;
        label.title = label.ariaLabel = fullText;
        setHelpText(label, fullText);
        if (button) {
            const action = status === "outdated" ? "Update" : "Download";
            if (button.textContent !== action) button.textContent = action;
            button.title = button.ariaLabel = `${action} ${entry.name}`;
            setHelpText(button, fullText);
            button.hidden = !downloadable;
            button.disabled = !downloadable;
        }
        const updated = this._badge(status);
        if (badge) {
            badge.className = updated.className;
            if (badge.textContent !== updated.textContent) badge.textContent = updated.textContent;
            badge.ariaLabel = updated.ariaLabel;
        }
        if (cat === "lora" && ["custom", "turbo"].includes(mode)) {
            if (installed) {
                card.onclick = () => mode === "turbo" ? this._selectTurboLora(entry.name, !active)
                    : this._updateLora(entry.name, { auto_apply: !active }, { commit: true, rerender: "lora" });
                this._enableCardKeyboard(card, active, entry.name);
            } else {
                card.onclick = card.onkeydown = null;
                if (card.role === "button") { card.removeAttribute("role"); card.tabIndex = -1; }
            }
            if (mode === "turbo") {
                card.classList.toggle("is-installed", installed);
                card.classList.toggle("is-active", active);
                const toggle = card.querySelector(".vnccs-cc-toggle");
                if (toggle) {
                    toggle.style.visibility = installed ? "visible" : "hidden";
                    toggle.inert = !installed;
                    const input = toggle.querySelector("input");
                    input.checked = active;
                    input.disabled = !installed;
                }
            }
        }
        if (focused?.hidden) (slot.hidden ? card : downloadable ? button : label).focus({ preventScroll: true });
        const signature = `${status}:${active}`;
        if (animate && view.signature && view.signature !== signature) {
            this._playMotion(slot, [{ opacity: 0.25 }, { opacity: 1 }], 200);
            this._playMotion(badge, [{ opacity: 0.3, transform: "scale(0.8)" }, { opacity: 1, transform: "none" }], 200);
        }
        view.signature = signature;
        view.refreshVariants?.();
        this._applyProgressLayer(card, key, { ...dls, status });
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
        span.ariaLabel = status === "success" ? "Checking installation" : this._statusLabel(status);
        return span;
    }

    _sel(id, opts, val, onChange) {
        return this._selectControl(id, opts, val, onChange);
    }

    _selectControl(id, choices, value, onChange, className = "vnccs-cc-select") {
        return createSakuraSelect(this, id, choices, value, onChange, className);
    }

    _openDropdown(trigger, options, label) {
        return openSakuraDropdown(this, trigger, options, label);
    }

    _closeDropdown(restoreFocus = true) {
        return closeSakuraDropdown(this, restoreFocus);
    }

    _numInput(val, min, max, step, onChange) {
        const i = document.createElement("input");
        val = Number(typeof val === "string" ? val.replaceAll("_", "") : val);
        i.type = "number"; i.value = val; i.min = min; i.max = max; i.step = step;
        i.className = "vnccs-cc-input";
        i.onchange = () => {
            const value = i.valueAsNumber;
            if (!i.checkValidity() || !Number.isFinite(value)) { i.value = val; return; }
            val = value;
            onChange(value);
        };
        return i;
    }

    _label(text, control) {
        const l = document.createElement("label");
        l.className = "vnccs-cc-param-label";
        l.textContent = text;
        control.dataset.focusKey ||= text;
        if (control.role === "combobox") control.ariaLabel = text;
        l.appendChild(control);
        return l;
    }

    _enableCardKeyboard(card, active, label) {
        card.role = "button";
        card.tabIndex = 0;
        card.ariaPressed = String(active);
        card.ariaLabel = label;
        card.dataset.focusKey = card.dataset.downloadKey;
        card.onkeydown = event => {
            if (event.target !== card || !["Enter", " "].includes(event.key)) return;
            event.preventDefault();
            event.stopPropagation();
            if (!event.repeat) card.onclick();
        };
    }

    // ── Config fetch ──────────────────────────────────────────────────────────

    async fetchConfig(repoId, force = false, { preserveCards = false } = {}) {
        if (this._removed) return;
        if (this._downloadStatusRepoId !== repoId) {
            this.dlStatus = {};
            this._pendingDownloads?.clear();
            this._downloadRefreshPending = false;
            this._downloadStatusRepoId = repoId;
        }
        const isCurrent = this._beginConfigRequest();
        this.statusText.textContent = "Loading…";
        this._setConfigLoading(true);

        const cacheKey = `vnccs_cc_cache_${repoId}`;

        if (force && !preserveCards) {
            delete serverRegistry("VNCCS_CC_REGISTRY")[repoId];
            storage.removeItem(cacheKey);
        }

        // Show cached data immediately while fetching fresh
        if (!force && !preserveCards && serverRegistry("VNCCS_CC_REGISTRY")[repoId]) {
            this.config = serverRegistry("VNCCS_CC_REGISTRY")[repoId];
            this.statusText.textContent = this.config.name || "Control Center";
            if (!this.state.output_slot_names) this.state.output_slot_names = [];
            this._syncCustomModelInput();
            this._renderAll();
            this._dispatchLoraOptions();
            this._setConfigLoading(true);
        }

        // Debounce: reuse in-flight fetch for same repo
        const previousPromise = serverRegistry("VNCCS_CC_FETCH_PROMISES")[repoId];
        let fetchPromise = !force && previousPromise;
        if (!fetchPromise) {
            fetchPromise = (async () => {
                try {
                    const url = `/vnccs/control_center/check?repo_id=${encodeURIComponent(repoId)}${force ? "&force_refresh=true" : ""}`;
                    const r = await api.fetchApi(url);
                    const data = await r.json();
                    if (!r.ok) throw new Error(data.error || `Catalog request failed (${r.status})`);
                    if (data.error) throw new Error(data.error);
                    if (serverRegistry("VNCCS_CC_FETCH_PROMISES")[repoId] === fetchPromise) {
                        serverRegistry("VNCCS_CC_REGISTRY")[repoId] = data;
                    }
                    return data;
                } finally {
                    if (serverRegistry("VNCCS_CC_FETCH_PROMISES")[repoId] === fetchPromise) {
                        delete serverRegistry("VNCCS_CC_FETCH_PROMISES")[repoId];
                    }
                }
            })();
            serverRegistry("VNCCS_CC_FETCH_PROMISES")[repoId] = fetchPromise;
            if (previousPromise) previousPromise.replacement = fetchPromise;
        }

        try {
            let data;
            while (true) {
                try { data = await fetchPromise; }
                catch (error) { if (!fetchPromise.replacement) throw error; }
                if (!isCurrent()) return;
                if (!fetchPromise.replacement) break;
                fetchPromise = fetchPromise.replacement;
            }
            if (!isCurrent() || serverRegistry("VNCCS_CC_REGISTRY")[repoId] !== data) return;
            this.config = data;
            this.statusText.textContent = data.name || "Control Center";
            storage.setItem(cacheKey, JSON.stringify(data));
            this._syncCnetSlots(data);
            this._syncCustomModelInput();
            if (preserveCards) {
                this._downloadRefreshPending = false;
                this._updateDownloadIndicators();
            } else this._renderAll();
            this._dispatchLoraOptions();
            window.dispatchEvent(new CustomEvent("vnccs-cc-registry-updated", {
                detail: { repo_id: repoId, download_update: preserveCards }
            }));
        } catch (e) {
            if (!isCurrent()) return;
            this.statusText.textContent = "Error";
            if (preserveCards) return;
            this.scrollArea.innerHTML = "";
            const err = document.createElement("div");
            err.className = "vnccs-cc-error";
            err.textContent = "✕ " + e.message;
            this.scrollArea.appendChild(err);
        } finally {
            if (isCurrent()) this._setConfigLoading(false);
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

    _statusLabel(status, dls = {}, compact = false) {
        if (compact) {
            if (status === "downloading") return Number.isFinite(dls.progress) ? `${Math.round(Math.max(0, Math.min(100, dls.progress)))}%` : "Loading";
            if (status === "queued") return "Queued";
            if (status === "success") return "Checking";
        }
        if (status === "installed") return "✓ Installed";
        if (status === "outdated") return "↑ Update";
        if (status === "queued") return "⏳ Queued";
        if (status === "downloading") return dls.message || "Downloading…";
        if (status === "error") return dls.message || "Download failed";
        if (status === "auth_required") return "⚠ Key Required";
        return "↓ Missing";
    }

    _playMotion(element, keyframes, duration = 240, onFinish) {
        return playSakuraMotion(this, element, keyframes, duration, onFinish);
    }

    _stopMotion() {
        return stopSakuraMotion(this);
    }

    _captureMotion() {
        const snapshot = new Map();
        if (!this.container?.clientWidth || !this.scrollArea?.getBoundingClientRect) return snapshot;
        const root = this.container.getBoundingClientRect();
        const scaleX = root.width / this.container.clientWidth || 1;
        const scaleY = root.height / this.container.clientHeight || 1;
        for (const element of this.scrollArea.querySelectorAll(".vnccs-cc-block, [data-download-key], .vnccs-cc-input[data-focus-key], .vnccs-cc-family-tab--active, .vnccs-cc-model-tab--active")) {
            if (!element.classList.contains("vnccs-cc-block") && element.closest(".vnccs-cc-block--collapsed")) continue;
            const rect = element.getBoundingClientRect();
            if (!rect.width || !rect.height || rect.bottom < root.top || rect.top > root.bottom) continue;
            const key = element.dataset.blockKey ? `block:${element.dataset.blockKey}`
                : element.dataset.downloadKey ? `card:${element.dataset.downloadKey}`
                : element.classList.contains("vnccs-cc-family-tab--active") ? "tab:family"
                : element.classList.contains("vnccs-cc-model-tab--active") ? "tab:type"
                : `param:${element.dataset.focusKey}`;
            snapshot.set(key, {
                element, x: (rect.left - root.left) / scaleX, y: (rect.top - root.top) / scaleY,
                width: rect.width / scaleX, height: rect.height / scaleY,
                signature: `${element.className}|${element.textContent}|${element.value ?? ""}`,
            });
        }
        return snapshot;
    }

    _motionGhost(previous, tab = false) {
        const ghost = tab ? document.createElement("div") : previous.element.cloneNode(true);
        ghost.removeAttribute("id");
        for (const child of ghost.querySelectorAll("[id]")) child.removeAttribute("id");
        ghost.classList.add("vnccs-cc-motion-ghost");
        if (tab) ghost.classList.add("vnccs-cc-tab-glide");
        ghost.inert = true;
        ghost.ariaHidden = "true";
        Object.assign(ghost.style, { left: `${previous.x}px`, top: `${previous.y}px`, width: `${previous.width}px`, height: `${previous.height}px` });
        this.container.appendChild(ghost);
        return ghost;
    }

    _animateView(before) {
        if (!before.size || !this.scrollArea?.animate || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
        const after = this._captureMotion();
        for (const [key, current] of after) {
            const previous = before.get(key);
            const dx = previous ? previous.x - current.x : 0;
            const dy = previous ? previous.y - current.y : 0;
            if (key.startsWith("block:")) {
                if (Math.abs(dx) + Math.abs(dy) > 1) {
                    this._playMotion(current.element, [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }]);
                }
            } else if (key.startsWith("tab:")) {
                if (previous && previous.signature !== current.signature) {
                    animateSakuraIndicator(this, previous, current);
                }
            } else if (!previous || previous.signature !== current.signature) {
                if (previous && previous.element !== current.element) {
                    const ghost = this._motionGhost(previous);
                    this._playMotion(ghost, [{ opacity: 1 }, { opacity: 0, transform: "translateY(-5px)" }], 160, () => ghost.remove());
                }
                this._playMotion(current.element, [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }]);
            } else if (Math.abs(dx) + Math.abs(dy) > 1) {
                this._playMotion(current.element, [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }]);
            }
        }
        for (const [key, previous] of before) {
            if (!key.startsWith("card:") || after.has(key)) continue;
            const ghost = this._motionGhost(previous);
            this._playMotion(ghost, [{ opacity: 1 }, { opacity: 0, transform: "translateY(-5px)" }], 160, () => ghost.remove());
        }
    }

    _renderAll(options) {
        if (!this.config) return;
        this._preserveView(() => this._renderContent(), options);
    }

    _preserveView(render, { preserveDraft = true } = {}) {
        this._closeDropdown(false);
        const before = this._captureMotion();
        const renderSequence = (this._renderSequence ?? 0) + 1;
        this._renderSequence = renderSequence;
        const previousScrollTop = this.scrollArea.scrollTop;
        const previousScrollLeft = this.scrollArea.scrollLeft;
        const active = document.activeElement;
        const focusKey = this.scrollArea.contains(active) ? active.dataset?.focusKey : null;
        const draft = preserveDraft && focusKey && active.tagName === "INPUT" && active.type === "number" ? active.value : null;
        render();
        for (const animation of [...(this._motionAnimations || [])]) {
            const target = animation.effect?.target;
            if (target && (!target.isConnected || target.classList.contains("vnccs-cc-motion-ghost"))) animation.cancel();
        }
        if (focusKey) {
            const replacement = [...this.scrollArea.querySelectorAll("[data-focus-key]")]
                .find(element => element.dataset.focusKey === focusKey);
            if (replacement && !replacement.closest?.(".vnccs-cc-disclosure")?.inert) {
                if (draft !== null) replacement.value = draft;
                replacement.focus({ preventScroll: true });
            }
        }
        this.scrollArea.scrollTop = previousScrollTop;
        this.scrollArea.scrollLeft = previousScrollLeft;
        const restoredTop = this.scrollArea.scrollTop;
        const restoredLeft = this.scrollArea.scrollLeft;
        this._animateView(before);
        requestAnimationFrame(() => {
            if (this._removed || !this.scrollArea?.isConnected || this._renderSequence !== renderSequence) return;
            // A deliberate scroll after rendering takes precedence over restoration.
            if (this.scrollArea.scrollTop !== restoredTop || this.scrollArea.scrollLeft !== restoredLeft) return;
            this.scrollArea.scrollTop = previousScrollTop;
            this.scrollArea.scrollLeft = previousScrollLeft;
        });
    }

    _renderContent() {
        this.scrollArea.innerHTML = "";

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
        if (this.state.unsupported_model_kind === "QIE2511") {
            const notice = document.createElement("div");
            notice.className = "vnccs-cc-params";
            const message = document.createElement("span");
            message.textContent = "QIE2511 is no longer supported. Choose QI2 to run this workflow.";
            const choose = this._btn("Use QI2", () => {
                delete this.state.unsupported_model_kind;
                this._saveState();
                this._renderAll();
            });
            notice.append(message, choose);
            this.scrollArea.appendChild(notice);
        }

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

        const modelBlock = this.scrollArea?.querySelector(`.vnccs-cc-block[data-block-key="models"]`);
        const modelContent = modelBlock?.querySelector(".vnccs-cc-disclosure-inner");
        const cacheInline = this._buildQI2CacheInlineSection();
        if (cacheInline) modelContent?.appendChild(cacheInline);
        const turboInline = this._buildModelTurboInlineSection();
        if (turboInline) modelContent?.appendChild(turboInline);

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
        MODEL_FAMILIES.forEach((family, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "vnccs-cc-family-tab";
            button.textContent = family.label;
            button.dataset.kind = family.kind;
            button.dataset.focusKey = `family-${family.kind}`;
            button.setAttribute("role", "tab");
            button.setAttribute("aria-selected", String(family.kind === activeKind));
            button.setAttribute("aria-label", `${family.label} model family`);
            button.tabIndex = family.kind === activeKind ? 0 : -1;
            if (family.kind === activeKind) button.classList.add("vnccs-cc-family-tab--active");
            button.onclick = () => this._setActiveKind(family.kind);
            button.onkeydown = event => {
                let nextIndex = null;
                if (event.key === "ArrowLeft") nextIndex = (index - 1 + MODEL_FAMILIES.length) % MODEL_FAMILIES.length;
                if (event.key === "ArrowRight") nextIndex = (index + 1) % MODEL_FAMILIES.length;
                if (event.key === "Home") nextIndex = 0;
                if (event.key === "End") nextIndex = MODEL_FAMILIES.length - 1;
                if (nextIndex === null) return;
                event.preventDefault();
                this._setActiveKind(MODEL_FAMILIES[nextIndex].kind);
            };
            tabs.appendChild(button);
        });
        return tabs;
    }

    // Generic single-column collapsible block
    _renderBlock(title, entries, key, rowFn) {
        const collapsed = this.state.collapsed?.[key] ?? false;
        const block = this._blockShell(title, entries?.length ?? 0, key, collapsed);
        {
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
        this.scrollArea.appendChild(this._finishBlock(block));
    }

    _buildClipVaeBlock() {
        const selectedKind = this._selectedKind();
        const entries = [
            ...(this.config.clip || []).filter(entry => this._sameKind(entry, selectedKind)).map(entry => ({ ...entry, _cat: "clip" })),
            ...(this.config.vae || []).filter(entry => this._sameKind(entry, selectedKind)).map(entry => ({ ...entry, _cat: "vae" })),
        ];
        const collapsed = this.state.collapsed?.clip_vae ?? false;
        const block = this._blockShell("CLIP + VAE", entries.length, "clip_vae", collapsed);

        {
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

        return this._finishBlock(block);
    }

    _renderClipVaeBlock() {
        this.scrollArea.appendChild(this._buildClipVaeBlock());
    }

    _buildLoraBlock() {
        const selectedKind = this._selectedKind();
        const entries = (this.config.lora || []).filter(entry =>
            !entry.custom && !this._isTurboLora(entry) && this._exactKind(entry, selectedKind)
        );
        const collapsed = this.state.collapsed?.lora ?? false;
        const block = this._blockShell("LORA", entries.length, "lora", collapsed);

        {
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

        return this._finishBlock(block);
    }

    _buildCustomLoraBlock() {
        const activeKind = this._activeKind().toLowerCase();
        const entries = (this.config.lora || []).filter(entry => {
            if (!entry.custom) return false;
            const kind = this._metaKind(entry).toLowerCase();
            return !kind || kind === "custom" || kind === activeKind;
        });
        const collapsed = this.state.collapsed?.custom_lora ?? false;
        const block = this._blockShell("CUSTOM LORAS", entries.length, "custom_lora", collapsed);

        {
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
            addBtn.type = "button";
            addBtn.dataset.focusKey = "add-custom-lora";
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

        return this._finishBlock(block);
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
        this._preserveView(() => {
            this._replaceBlock("lora", this._buildLoraBlock());
            this.scrollArea?.querySelector(`.vnccs-cc-block[data-block-key="turbo_model"]`)?.remove();
            this._replaceBlock("custom_lora", this._buildCustomLoraBlock());
        });
    }

    async _removeCustomLora(entry) {
        const repoId = this._getRepoId();
        if (!repoId) {
            this.showMessage("Load control center config first.", true);
            return;
        }
        const state = this.state;
        const kind = this._activeKind();
        const isCurrent = () => !this._removed && !this._stateRestoreError && this.state === state
            && this._getRepoId() === repoId && this._activeKind() === kind;

        this.showConfirm(`Remove custom LoRA \"${entry.name}\" from the list?`, async () => {
            if (!isCurrent()) return;
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

                delete serverRegistry("VNCCS_CC_REGISTRY")[repoId];
                if (!isCurrent()) return;
                this.state.loras = (this.state.loras || []).filter(lora => lora.name !== entry.name);
                this._saveState();
                await this.fetchConfig(repoId, true);
            } catch (error) {
                if (isCurrent()) this.showMessage(String(error?.message || error), true);
            }
        });
    }

    async _showCustomLoraDialog() {
        const repoId = this._getRepoId();
        if (!repoId) {
            this.showMessage("Load control center config first.", true);
            return;
        }
        const state = this.state;
        const kind = this._activeKind();
        const isCurrent = () => !this._removed && !this._stateRestoreError && this.state === state
            && this._getRepoId() === repoId && this._activeKind() === kind;

        let payload;
        try {
            const response = await api.fetchApi(`/vnccs/control_center/lora_files?repo_id=${encodeURIComponent(repoId)}`);
            payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || "Failed to load LoRA list");
        } catch (error) {
            if (isCurrent()) this.showMessage(String(error?.message || error), true);
            return;
        }
        if (!isCurrent()) return;

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
        desc.textContent = `Choose an installed LoRA for ${this._familyDefinition().label}. It will be saved into a separate VNCCS JSON file and stay available between sessions.`;
        panel.appendChild(desc);

        const selectWrap = document.createElement("div");
        selectWrap.className = "vnccs-cc-settings-field";
        const selectLabel = document.createElement("label");
        selectLabel.className = "vnccs-cc-settings-label";
        selectLabel.textContent = "LoRA File";
        const select = this._selectControl("vnccs-cc-lora-file", items.map(item => item.path), items[0].path, null, "vnccs-cc-settings-select");
        select.ariaLabel = "LoRA file";
        selectLabel.appendChild(select);
        selectWrap.appendChild(selectLabel);
        panel.appendChild(selectWrap);

        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";
        const cancelBtn = this._btn("Cancel", () => ov.remove());
        const saveBtn = this._btn("Add", async () => {
            if (!isCurrent()) { ov.remove(); return; }
            const path = select.value;
            if (!path) return;
            saveBtn.disabled = true;
            saveBtn.textContent = "Adding…";
            try {
                const response = await api.fetchApi("/vnccs/control_center/custom_lora", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ repo_id: repoId, path, kind }),
                });
                const result = await response.json();
                if (!response.ok || result.error) throw new Error(result.error || "Failed to add custom LoRA");

                ov.remove();
                delete serverRegistry("VNCCS_CC_REGISTRY")[repoId];
                if (!isCurrent()) return;
                await this.fetchConfig(repoId, true);
            } catch (error) {
                if (!isCurrent()) { ov.remove(); return; }
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
        this._mountOverlay(ov, panel, title.textContent);
    }

    // Two-column block with a single pre-built row (MODEL slot)
    _renderTwoColBlockSingle(title, count, key, rowFn, rightFn, rightFn2 = null, topFn = null, showEmptyLeft = true, renderLeftWhenEmpty = false) {
        const collapsed = this.state.collapsed?.[key] ?? false;
        const block = this._blockShell(title, count, key, collapsed);
        {
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
        this.scrollArea.appendChild(this._finishBlock(block));
    }

    _finishBlock(block, collapsed = Boolean(this.state.collapsed?.[block.dataset.blockKey])) {
        const disclosure = document.createElement("div");
        disclosure.className = "vnccs-cc-disclosure";
        const inner = document.createElement("div");
        inner.className = "vnccs-cc-disclosure-inner";
        for (const child of [...block.children].slice(1)) inner.appendChild(child);
        disclosure.appendChild(inner);
        block.appendChild(disclosure);
        this._setDisclosure(block, collapsed);
        return block;
    }

    _setDisclosure(block, collapsed) {
        block.classList.toggle("vnccs-cc-block--collapsed", collapsed);
        block.firstChild.ariaExpanded = String(!collapsed);
        const disclosure = block.querySelector(".vnccs-cc-disclosure");
        disclosure.inert = collapsed;
        disclosure.ariaHidden = String(collapsed);
    }

    _blockShell(title, count, key, collapsed) {
        const block = document.createElement("div");
        block.className = "vnccs-cc-block";
        block.classList.toggle("vnccs-cc-block--collapsed", collapsed);
        block.dataset.blockKey = key;

        const hdr = document.createElement("button");
        hdr.type = "button";
        hdr.className = "vnccs-cc-block-hdr";
        hdr.ariaExpanded = String(!collapsed);
        hdr.dataset.focusKey = `block-${key}`;

        const titleWrap = document.createElement("span");
        titleWrap.textContent = title;
        const countSpan = document.createElement("span");
        countSpan.className = "vnccs-cc-block-hdr-count";
        countSpan.textContent = `(${count})`;
        titleWrap.appendChild(countSpan);

        const arrow = document.createElement("span");
        arrow.className = "vnccs-cc-block-arrow";
        arrow.textContent = "▼";
        arrow.ariaHidden = "true";

        hdr.append(titleWrap, arrow);
        hdr.onclick = () => {
            this._preserveView(() => {
                this.state.collapsed ??= {};
                const collapsed = !this.state.collapsed[key];
                this.state.collapsed[key] = collapsed;
                this._setDisclosure(block, collapsed);
                if (collapsed) hdr.focus({ preventScroll: true });
                this._saveState();
            });
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
            button.ariaPressed = String(tabType === type);
            button.dataset.focusKey = `model-type-${tabType}`;

            const hasVariants = tabType === "custom"
                ? true
                : this._visibleModelsByType(tabType).length > 0;
            if (!hasVariants) button.classList.add("vnccs-cc-model-tab--missing");
            button.ariaLabel = hasVariants ? `${tabType} models` : `${tabType} (no compatible models)`;

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

        // Top: badge + name + status or download button
        const top = document.createElement("div");
        top.className = "vnccs-cc-model-card-top";
        top.appendChild(this._badge(status));

        const nameEl = document.createElement("span");
        nameEl.className = "vnccs-cc-model-card-name";
        nameEl.textContent = cur.name;
        top.appendChild(nameEl);

        this._mountDownloadStatus(card, top, "models", cur, "vnccs-cc-model-card-status");
        card.appendChild(top);

        // Description
        if (cur.description) {
            const desc = document.createElement("div");
            desc.className = "vnccs-cc-model-card-desc";
            desc.textContent = cur.description;
            card.appendChild(desc);
        }

        // Footer: variant dropdown at bottom
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

            const choices = () => variants.map(v => {
                v = this.config.models.find(entry => entry.name === v.name) || v;
                const vDls = this.dlStatus[`cc_models_${v.name}`] ?? {};
                const vSt = this._resolveStatus(vDls.status, v.status);
                return { value: v.name, label: (v.name.slice(prefixLen) || v.name) + "  " + icon(vSt) };
            });
            const sel = this._selectControl("vnccs-cc-model-variant", choices(), cur.name, null, "vnccs-cc-model-card-select");
            card._downloadView.refreshVariants = () => {
                if (this._dropdown?.trigger !== sel) sel.setChoices(choices(), sel.value);
            };
            sel.ariaLabel = "Model variant";
            sel.dataset.focusKey = "model-variant";
            setHelpText(sel, cur.name);
            sel.onchange = () => {
                const chosen = variants.find(v => v.name === sel.value);
                if (chosen) {
                    this._setSelectedModelName(type, chosen.name);
                    if (!this.state.selected_types_by_kind) this.state.selected_types_by_kind = {};
                    this.state.selected_types_by_kind[this._activeKind()] = chosen.type || type;
                    this.state.selected_type  = chosen.type || type;
                    this._saveState();
                    this._renderAll({ preserveDraft: false });
                }
            };
            footer.appendChild(sel);
        }

        if (footer.children.length) card.appendChild(footer);
        this._updateDownloadCard(card, false);
        return card;
    }

    _renderCustomModelPlaceholder() {
        const card = document.createElement("div");
        card.className = "vnccs-cc-model-card vnccs-cc-model-card--placeholder";

        const text = document.createElement("div");
        text.className = "vnccs-cc-model-card-placeholder-text";
        text.textContent = this._activeKind() === "MiniMaxH3"
            ? "Pass-through mode is enabled. Connect MODEL, CLIP, video VAE, and audio VAE to the node inputs."
            : "Pass-through mode is enabled. Connect MODEL, CLIP, and VAE to the node inputs.";
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

        this._mountDownloadStatus(card, top, cat, entry, "vnccs-cc-model-card-status");
        card.appendChild(top);

        if (entry.description) {
            const desc = document.createElement("div");
            desc.className = "vnccs-cc-model-card-desc";
            desc.textContent = entry.description;
            card.appendChild(desc);
        }

        this._updateDownloadCard(card, false);
        return card;
    }

    // ── MODEL right column: sampler params ────────────────────────────────────

    _modelParamsSave(patch) {
        Object.assign(this._currentModelParams(), patch);
        this._saveState();
        if (this._isQwenFamily()) this._renderAll();
    }

    _qi2CacheSettings() {
        const cache = this.state.qi2_cache ?? {};
        return {
            device: ["auto", "gpu", "cpu", "off"].includes(cache.device) ? cache.device : "gpu",
            dtype: ["default", "int8", "int4"].includes(cache.dtype) ? cache.dtype : "int8",
        };
    }

    _qi2CacheSave(patch) {
        this.state.qi2_cache = { ...this._qi2CacheSettings(), ...patch };
        this._saveState();
    }

    _buildQI2CacheInlineSection() {
        if (this._activeKind() !== "QI2") return null;

        const cache = this._qi2CacheSettings();
        const section = document.createElement("div");
        section.className = "vnccs-cc-model-inline-section vnccs-cc-qi2-cache-section";

        const label = document.createElement("div");
        label.className = "vnccs-cc-model-inline-label";
        label.textContent = "Qwen Image 2.1 Cache";

        const strip = document.createElement("div");
        strip.className = "vnccs-cc-cache-strip";
        const field = (labelText, control, help) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-param-field";
            setHelpText(wrap, help);
            wrap.appendChild(this._label(labelText, control));
            return wrap;
        };
        strip.append(
            field(
                "Device",
                this._sel("vnccs-cc-qi2-cache-device", ["auto", "gpu", "cpu", "off"], cache.device,
                    value => this._qi2CacheSave({ device: value })),
                "KV cache location. Auto uses spare VRAM then RAM; CPU uses prefetched RAM; Off recomputes the prefix every step."
            ),
            field(
                "Dtype",
                this._sel("vnccs-cc-qi2-cache-dtype", ["default", "int8", "int4"], cache.dtype,
                    value => this._qi2CacheSave({ dtype: value })),
                "KV cache storage precision. Default is lossless; int8 halves its size; int4 uses about one quarter."
            ),
        );
        section.append(label, strip);
        return section;
    }

    _buildModelTurboInlineSection() {
        if (this._isQi2TurboModel(this._getSelectedModelEntry())) return null;
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
        this._applyProgressLayer(row, `cc_lora_${entry.name}`, dls);
        row.classList.toggle("is-installed", installed);
        row.classList.toggle("is-active", active);
        if (installed) {
            row.onclick = () => this._selectTurboLora(entry.name, !active);
            this._enableCardKeyboard(row, active, entry.name);
        }

        row.appendChild(this._badge(status));

        const name = document.createElement("div");
        name.className = "vnccs-cc-turbo-strip-name";
        name.textContent = entry.name;
        row.appendChild(name);

        this._mountDownloadStatus(row, row, "lora", entry, "vnccs-cc-turbo-strip-status", "turbo");

        {
            const toggle = document.createElement("label");
            toggle.className = "vnccs-cc-toggle";
            toggle.onclick = event => event.stopPropagation();
            const input = document.createElement("input");
            input.type = "checkbox";
            input.checked = active;
            input.ariaLabel = `Enable ${entry.name}`;
            input.dataset.focusKey = `toggle-${entry.name}`;
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
        }

        this._updateDownloadCard(row, false);
        return row;
    }

    _renderModelParams() {
        const p  = this._currentModelParams();
        const family = this._familyDefinition();
        const turbo = this._isQi2TurboModel(this._getSelectedModelEntry());
        const steps = p.steps || (turbo ? 8 : family.steps ?? DEFAULT_MODEL_STEPS);
        const mp = {
            steps: Math.trunc(Number(typeof steps === "string" ? steps.replaceAll("_", "") : steps)),
            cfg: p.cfg ?? (turbo ? 1.0 : family.cfg ?? DEFAULT_MODEL_CFG),
        };

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
            wrap.appendChild(this._label(labelText, el));
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
        const mp = { sampler: p.sampler || this._familyDefinition().sampler || "euler",
            scheduler: p.scheduler || DEFAULT_MODEL_SCHEDULER };

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
            wrap.appendChild(this._label(labelText, el));
            return wrap;
        };

        panel.appendChild(field("Sampler",
            this._sel("vnccs-cc-sampler", this._samplers, mp.sampler, v => this._modelParamsSave({ sampler: v }))));
        panel.appendChild(field("Scheduler",
            this._sel("vnccs-cc-scheduler", this._schedulers, mp.scheduler, v => this._modelParamsSave({ scheduler: v }))));

        return panel;
    }

    // ── LORA entry ────────────────────────────────────────────────────────────

    _renderLoraCard(entry, options = {}) {
        const { active = false, compact = false, mode = "custom" } = options;
        const isPipeMode = mode === "pipe";
        const isTurboMode = mode === "turbo";
        const ls  = (this.state.loras ?? []).find(l => l.name === entry.name)
            ?? { name: entry.name, auto_apply: false, strength: 1.0 };
        const dls = this.dlStatus[`cc_lora_${entry.name}`] ?? {};
        const status = this._resolveStatus(dls.status, entry.status);

        const loraKey = `cc_lora_${entry.name}`;
        const card = document.createElement("div");
        card.className = "vnccs-cc-model-card vnccs-cc-lora-card";
        if (active && !isPipeMode) card.classList.add("vnccs-cc-lora-card--active");
        if (compact) card.classList.add("vnccs-cc-lora-card--compact");
        if (entry.custom) card.classList.add("vnccs-cc-lora-card--has-remove");
        this._applyProgressLayer(card, loraKey, dls);
        const tools = document.createElement("div");
        tools.className = "vnccs-cc-lora-card-tools";

        if (entry.custom) {
            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "vnccs-cc-lora-remove-btn";
            removeBtn.textContent = "×";
            removeBtn.title = "Remove custom LoRA from list";
            removeBtn.ariaLabel = `Remove ${entry.name} from list`;
            removeBtn.dataset.focusKey = `remove-${entry.name}`;
            removeBtn.onclick = e => {
                e.stopPropagation();
                this._removeCustomLora(entry);
            };
            tools.appendChild(removeBtn);
        }

        const top = document.createElement("div");
        top.className = "vnccs-cc-lora-card-top";
        top.appendChild(this._badge(status));

        const meta = document.createElement("div");
        meta.className = "vnccs-cc-lora-card-meta";

        const nameEl = document.createElement("span");
        nameEl.className = "vnccs-cc-lora-card-name";
        nameEl.textContent = entry.name;
        nameEl.title = entry.name;
        meta.appendChild(nameEl);

        this._mountDownloadStatus(card, tools, "lora", entry, "vnccs-cc-lora-card-status", mode);
        top.appendChild(meta);
        if (tools.children.length) top.appendChild(tools);
        card.appendChild(top);

        const detailRow = document.createElement("div");
        detailRow.className = "vnccs-cc-lora-card-detail-row";
        const desc = document.createElement("div");
        desc.className = "vnccs-cc-lora-card-desc";
        desc.textContent = desc.title = entry.description || "";
        const versionEl = document.createElement("span");
        versionEl.className = "vnccs-cc-lora-card-version";
        versionEl.textContent = versionEl.title = entry.version ? `v${entry.version}` : "";
        detailRow.append(desc, versionEl);
        card.appendChild(detailRow);

        const footer = document.createElement("div");
        footer.className = active ? "vnccs-cc-lora-strength-row" : "vnccs-cc-lora-card-actions";
        footer.onclick = e => e.stopPropagation();

        if (status === "installed" && active && !isPipeMode && !isTurboMode) {
                const strength = Number(typeof ls.strength === "string" ? ls.strength.replaceAll("_", "") : ls.strength ?? 1.0);
                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = Math.min(-2, strength);
                slider.max = Math.max(2, strength);
                slider.step = 0.05;
                slider.value = strength;
                slider.className = "vnccs-cc-lora-slider";
                slider.ariaLabel = `${entry.name} strength`;
                slider.dataset.focusKey = `strength-${entry.name}`;

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
                val.textContent = strength.toFixed(2);

                slider.oninput = () => {
                    const strength = parseFloat(slider.value);
                    val.textContent = strength.toFixed(2);
                    this._updateLora(entry.name, { strength }, { commit: false });
                };

                slider.onchange = () => {
                    const strength = parseFloat(slider.value);
                    val.textContent = strength.toFixed(2);
                    this._updateLora(entry.name, { strength }, { commit: true, rerender: null });
                };

                footer.append(slider, val);
        }

        if (footer.children.length) card.appendChild(footer);
        this._updateDownloadCard(card, false);
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
        this._renderAll({ preserveDraft: false });
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

        this._mountDownloadStatus(row, row, cat, entry, "vnccs-cc-model-card-status");
        this._updateDownloadCard(row, false);
        return row;
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
        const state = this.state;
        const repoId = this._getRepoId();
        const kind = this._activeKind();
        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";

        const ts   = this.state.type_settings ?? {};
        const unet = ts.unet ?? {};

        const field = (labelText, el) => {
            const w = document.createElement("div");
            w.className = "vnccs-cc-settings-field";
            const help = {
                "Weight Dtype": "Precision mode for UNet loading. Default follows the loader; fp8 modes can reduce memory use.",
            }[labelText];
            setHelpText(w, help);
            const l = document.createElement("label");
            l.className = "vnccs-cc-settings-label";
            l.textContent = labelText;
            l.appendChild(el);
            w.appendChild(l); return w;
        };
        const mkSel = (id, opts, val) => {
            const control = this._selectControl(id, opts, val, null, "vnccs-cc-settings-select");
            control.ariaLabel = "Weight Dtype";
            return control;
        };
        const title = document.createElement("h3");
        title.className = "vnccs-cc-settings-title";
        title.textContent = "⚙ Settings";
        panel.appendChild(title);

        // UNet
        const unetDet = document.createElement("div");
        unetDet.className = "vnccs-cc-settings-details";
        const collapsed = (this._getSelectedType() || "unet") !== "unet";
        const unetSum = document.createElement("button");
        unetSum.type = "button";
        unetSum.className = "vnccs-cc-settings-summary";
        unetSum.textContent = "UNet";
        unetSum.onclick = () => this._setDisclosure(unetDet, unetSum.ariaExpanded === "true");
        unetDet.appendChild(unetSum);
        unetDet.appendChild(field("Weight Dtype",
            mkSel("vnccs-cc-unet-dtype",
                ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                unet.weight_dtype ?? "default")));
        panel.appendChild(this._finishBlock(unetDet, collapsed));

        // Buttons
        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";

        const cancelBtn = this._btn("Cancel", () => ov.remove());

        const saveBtn = this._btn("Save", async () => {
            if (this._removed || this._stateRestoreError || this.state !== state
                || this._getRepoId() !== repoId || this._activeKind() !== kind) {
                ov.remove();
                return;
            }
            this.state.selected_type = this._getSelectedType();
            if (!this.state.type_settings) this.state.type_settings = {};
            this.state.type_settings.unet = {
                weight_dtype: panel.querySelector("#vnccs-cc-unet-dtype")?.value ?? "default",
            };
            // TECH DEBT: Nunchaku settings persistence disabled. Delete after
            // stale workflow state no longer stores type_settings.nunchaku.
            delete this.state.type_settings.nunchaku;
            this._saveState();
            ov.remove();
            this._renderAll();
        });
        saveBtn.classList.add("vnccs-cc-btn--save");
        btns.append(cancelBtn, saveBtn);
        panel.appendChild(btns);

        ov.appendChild(panel);
        this._mountOverlay(ov, panel, title.textContent);
    }

    // ── Downloads ─────────────────────────────────────────────────────────────

    async _downloadEntry(cat, entry) {
        const repoId = this._getRepoId();
        if (!repoId) return;
        const state = this.state;
        const kind = this._activeKind();
        const isCurrent = () => !this._removed && !this._stateRestoreError && this.state === state
            && this._getRepoId() === repoId && this._activeKind() === kind;
        if (!isCurrent()) return;
        const key = `cc_${cat}_${entry.name}`;
        if (["queued", "downloading"].includes(this.dlStatus[key]?.status) || this._pendingDownloads?.has(key)) return;
        this._pendingDownloads ??= new Map();
        const pending = {};
        this._pendingDownloads.set(key, pending);
        // Show the queued state while the server accepts the request.
        this._beginDownloadStatusRequest();
        this._downloadStatusRepoId = repoId;
        this.dlStatus[key] = { status: "queued", message: "Queued…" };
        this._updateDownloadIndicators();
        try {
            const r = await api.fetchApi("/vnccs/control_center/download", {
                method: "POST",
                headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                body: JSON.stringify({ repo_id: repoId, category: cat, name: entry.name }),
            });
            const responseText = await r.text();
            let d = {};
            try { d = responseText ? JSON.parse(responseText) : {}; } catch { d = {}; }
            if (!isCurrent()) return;
            if (!r.ok || d.error) {
                const message = d.error || responseText || `Download request failed (${r.status})`;
                this.dlStatus[key] = { status: "error", message };
                this._updateDownloadIndicators();
                this.showMessage(message, true);
            } else {
                window.dispatchEvent(new CustomEvent("vnccs-cc-registry-updated", {
                    detail: { repo_id: repoId },
                }));
            }
        } catch (e) {
            if (!isCurrent()) return;
            const message = String(e?.message || e);
            this.dlStatus[key] = { status: "error", message };
            this._updateDownloadIndicators();
            this.showMessage(message, true);
        } finally {
            if (this._pendingDownloads.get(key) === pending) this._pendingDownloads.delete(key);
            if (isCurrent()) this._beginDownloadStatusRequest();
        }
    }

    async _downloadAllMissing() {
        if (!this.config) { this.showMessage("Load config first.", true); return; }
        const state = this.state;
        const repoId = this._getRepoId();
        const kind = this._activeKind();
        const isCurrent = () => !this._removed && !this._stateRestoreError && this.state === state
            && this._getRepoId() === repoId && this._activeKind() === kind;

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
                if (cat === "lora" && e.custom && !e.hf_path) continue;
                if (["clip", "vae"].includes(cat) && !this._sameKind(e, selectedKind)) continue;
                if (cat === "lora" && !e.custom && !this._exactKind(e, selectedKind)) continue;
                if (this._isDownloadableStatus(e.status)) tasks.push({ cat, e });
            }
        }

        if (!tasks.length) { this.showMessage("All selected items are installed."); return; }
        this.showConfirm(`Download/update ${tasks.length} item(s)?`, async () => {
            for (const { cat, e } of tasks) {
                if (!isCurrent()) return;
                await this._downloadEntry(cat, e);
                await new Promise(r => setTimeout(r, 300));
            }
        });
    }

    // ── Progress bar helper ───────────────────────────────────────────────────

    _updateDownloadIndicators() {
        for (const card of this.scrollArea.querySelectorAll("[data-download-key]")) this._updateDownloadCard(card);
    }

    _applyProgressLayer(row, key, dls) {
        row.dataset.downloadKey = key;
        const status = dls?.status;
        row.dataset.downloadStatus = status || "idle";
        row.ariaBusy = String(["queued", "downloading", "success"].includes(status));
        let bg = row.querySelector(".vnccs-cc-row-bg");
        if (!["downloading", "queued", "success", "auth_required", "error"].includes(status)) {
            if (bg && !bg.dataset.retiring) {
                bg.dataset.retiring = "true";
                this._playMotion(bg, [{ opacity: 1 }, { opacity: 0 }], 160, () => {
                    if (!["downloading", "queued", "success", "auth_required", "error"].includes(row.dataset.downloadStatus)) bg.remove();
                });
            }
            return;
        }
        const created = !bg;
        bg ||= document.createElement("div");
        bg.dataset.retiring = "";
        this._motionByElement?.get(bg)?.cancel();
        bg.className = "vnccs-cc-row-bg";
        bg.style.width = "100%";
        bg.style.opacity = "1";
        bg.style.transform = "scaleX(1)";

        if (status === "downloading" && Number.isFinite(dls.progress)) {
            const progress = Math.max(0, Math.min(dls.progress, 100));
            bg.style.transform = `scaleX(${progress / 100})`;
            bg.style.background = "rgba(0, 214, 143, 0.7)";
        } else if (["queued", "downloading", "success"].includes(status)) {
            bg.className += " vnccs-cc-row-bg--indeterminate";
            bg.style.background = "repeating-linear-gradient(45deg, rgba(184,169,232,0.12), rgba(184,169,232,0.12) 10px, transparent 10px, transparent 20px)";
        } else if (status === "auth_required") {
            bg.style.background = "rgba(255,170,0,0.12)";
        } else if (status === "error") {
            bg.style.background = "rgba(255,71,87,0.1)";
        }

        if (created) {
            row.insertBefore(bg, row.firstChild);
            this._playMotion(bg, [{ opacity: 0 }, { opacity: 1 }], 180);
        }
    }

    // ── Styled message/confirm overlays (replaces alert/confirm) ──────────────

    _mountOverlay(overlay, panel, title) {
        this._closeDropdown(false);
        const opener = document.activeElement;
        this._overlays ??= [];
        this._overlays.push(overlay);
        panel.role = "dialog";
        panel.ariaModal = "true";
        panel.ariaLabel = title;
        panel.tabIndex = -1;
        const fields = () => [...panel.querySelectorAll("button, input, select, summary, a[href], [tabindex]")]
            .filter(element => !element.disabled && !element.closest?.(".vnccs-cc-disclosure")?.inert && element.getClientRects().length);
        const remove = overlay.remove;
        overlay.remove = () => {
            if (overlay.inert) return;
            if (this._dropdown && overlay.contains(this._dropdown.trigger)) this._closeDropdown(false);
            const wasTop = this._overlays.at(-1) === overlay;
            overlay.inert = true;
            overlay.ariaHidden = "true";
            overlay.style.pointerEvents = "none";
            this._overlays = this._overlays.filter(item => item !== overlay);
            this._playMotion(overlay, [{ opacity: 1 }, { opacity: 0 }], 140, () => remove.call(overlay));
            if (wasTop && !this._removed && opener?.isConnected) opener.focus({ preventScroll: true });
        };
        overlay.onkeydown = event => {
            if (this._overlays.at(-1) !== overlay) return;
            if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                overlay.remove();
            } else if (event.key === "Tab") {
                const controls = fields();
                const index = controls.indexOf(document.activeElement);
                if (!controls.length || (event.shiftKey ? index <= 0 : index < 0 || index === controls.length - 1)) {
                    event.preventDefault();
                    event.stopPropagation();
                    (event.shiftKey ? controls.at(-1) || panel : controls[0] || panel).focus({ preventScroll: true });
                }
            }
        };
        panel.addEventListener?.("scroll", () => this._closeDropdown(false), { passive: true });
        this.container.appendChild(overlay);
        this._playMotion(overlay, [{ opacity: 0 }, { opacity: 1 }], 180);
        this._playMotion(panel, [{ opacity: 0, transform: "translateY(8px) scale(0.98)" }, { opacity: 1, transform: "none" }], 240);
        requestAnimationFrame(() => {
            if (this._removed || !overlay.isConnected || this._overlays.at(-1) !== overlay) return;
            (fields()[0] || panel).focus({ preventScroll: true });
        });
    }

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
            max-height:100%; overflow:auto; overflow-wrap:anywhere;
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
        this._mountOverlay(ov, box, isError ? "Error" : "Message");
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
            max-height:100%; overflow:auto; overflow-wrap:anywhere;
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
        this._mountOverlay(ov, box, "Confirmation");
    }

}
