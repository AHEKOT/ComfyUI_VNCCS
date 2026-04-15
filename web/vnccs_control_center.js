// web/vnccs_control_center.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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
.vnccs-cc-twocol-right {
    width: 130px;
    flex-shrink: 0;
    padding: 7px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(26,26,38,0.3);
}
.vnccs-cc-twocol-right2 {
    width: 130px;
    flex-shrink: 0;
    padding: 7px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(26,26,38,0.3);
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
.vnccs-cc-model-card-top {
    display: flex;
    align-items: center;
    gap: 7px;
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
.vnccs-cc-model-card-select {
    flex: 1;
    background: rgba(255,255,255,0.05);
    color: #e8e8f0;
    border: 1px solid rgba(0,214,143,0.2);
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
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
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.vnccs-cc-param-field {
    display: flex;
    flex-direction: column;
    gap: 2px;
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
    transition: border-color 0.18s;
}
.vnccs-cc-select:focus {
    outline: none;
    border-color: rgba(255,143,163,0.35);
}

/* LoRA */
.vnccs-cc-lora-chk {
    cursor: pointer;
    accent-color: #ff8fa3;
    flex-shrink: 0;
}
.vnccs-cc-lora-slider {
    width: 65px;
    accent-color: #ff8fa3;
    flex-shrink: 0;
}
.vnccs-cc-lora-val {
    min-width: 32px;
    text-align: right;
    color: #5e5e70;
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
}
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
    gap: 6px;
    padding: 4px 10px;
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
            this._cc_widget = new VNCCSControlCenterWidget(this);
            this.setDirtyCanvas(true, true);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            this._cc_widget?.restoreState();
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            if (this._cc_widget?.pollingInterval)
                clearInterval(this._cc_widget.pollingInterval);
            if (this._cc_widget?._onRegistryUpdate)
                window.removeEventListener("vnccs-cc-registry-updated", this._cc_widget._onRegistryUpdate);
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
        this.downloadStartTimes = {};  // for fake progress bars
        this._samplers   = DEFAULT_SAMPLERS;
        this._schedulers = DEFAULT_SCHEDULERS;
        this.pollingInterval = null;

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
                if (!this._isUserInteracting()) this._renderAll();
            }
        };
        window.addEventListener("vnccs-cc-registry-updated", this._onRegistryUpdate);
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
    _getSelectedType(){ return this.state.selected_type || (this.config?.available_types?.[0] ?? ""); }

    _getSelectedModelEntry() {
        if (!this.config?.models?.length) return null;
        const selectedType = this._getSelectedType();
        const selectedModel = this.state.selected_model;
        const variants = this.config.models.filter(m => !m.type || m.type === selectedType);
        return variants.find(m => m.name === selectedModel) ?? variants[0] ?? null;
    }

    _hasEnabledLoras() {
        return (this.state.loras ?? []).some(l => l?.name && l.enabled !== false && Math.abs(Number(l.strength ?? 1)) > 1e-6);
    }

    _saveState() {
        const w = this._getStateWidget();
        if (w) w.value = JSON.stringify(this.state);
        this.node.setDirtyCanvas(true, true);
    }

    restoreState() {
        const w = this._getStateWidget();
        if (w?.value && w.value !== "{}") {
            try { this.state = JSON.parse(w.value); } catch { this.state = {}; }
        }
        // Rebuild dynamic output slots from saved state
        this._syncOutputSlots();
        const repo = this._getRepoId();
        if (repo) this.fetchConfig(repo);
    }

    async _refreshDependencyStatus(showModal = false) {
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
    }

    _showQwenFixModal() {
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
    }

    // ── Polling ───────────────────────────────────────────────────────────────

    _isUserInteracting() {
        // Skip re-render if any focusable element inside our container is active
        // (e.g. an open <select> dropdown, a focused input)
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

        this.node.addDOMWidget("cc_widget", "cc_panel", c, {
            getValue: () => "{}", setValue: () => {}, serialize: false,
            hideOnZoom: false,
        });

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
            const r = this._getRepoId(); r ? this.fetchConfig(r, true) : alert("Enter repo_id first.");
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
        b.textContent = "⬇ Download All Missing";
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

        this.statusStrip.appendChild(pills);
        this._updateBanner = null; // created lazily
        this.container.appendChild(this.statusStrip);
    }

    _makePill(name, version, state, extra = "") {
        const p = document.createElement("span");
        p.className = `vnccs-cc-pill vnccs-cc-pill--${state}`;
        const icons = { ok: "●", update: "↑", error: "✕", loading: "…", dup: "⚠" };
        p.textContent = version
            ? `${name} v${version} ${icons[state] ?? ""}${extra ? " " + extra : ""}`
            : `${name} ${icons[state] ?? ""}${extra ? " " + extra : ""}`;
        if (state === "update") p.title = `Update available: ${extra}`;
        if (state === "dup")   p.title = `Duplicate install detected: ${extra}`;
        if (state === "error") p.title = extra || "Module not found";
        return p;
    }

    _updatePill(pillEl, name, version, state, extra = "") {
        const icons = { ok: "●", update: "↑", error: "✕", loading: "…", dup: "⚠" };
        pillEl.className = `vnccs-cc-pill vnccs-cc-pill--${state}`;
        pillEl.textContent = version
            ? `${name} v${version} ${icons[state] ?? ""}${extra ? " " + extra : ""}`
            : `${name} ${icons[state] ?? ""}${extra ? " " + extra : ""}`;
        if (state === "update") pillEl.title = `Update available: v${extra}`;
        if (state === "dup")   pillEl.title = `Duplicate install: ${extra}`;
        if (state === "error") pillEl.title = extra || "Module not found";
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

    _parseTomlVersion(text) {
        const m = text.match(/^version\s*=\s*["']([^"']+)["']/m);
        return m ? m[1] : null;
    }

    async _fetchGitHubVersion(repo) {
        // repo: "AHEKOT/ComfyUI_VNCCS" etc.
        try {
            const url = `https://raw.githubusercontent.com/${repo}/main/pyproject.toml`;
            const r = await fetch(url, { cache: "no-store" });
            if (!r.ok) return null;
            return this._parseTomlVersion(await r.text());
        } catch { return null; }
    }

    async _fetchModuleStatus() {
        const REPOS = {
            main:  "AHEKOT/ComfyUI_VNCCS",
            utils: "AHEKOT/ComfyUI_VNCCS_Utils",
        };
        const LABELS = { main: "VNCCS", utils: "Utils" };
        const PILLS  = { main: this._pillMain, utils: this._pillUtils };

        // 1. Fetch local versions from Python
        let local = {};
        try {
            const r = await api.fetchApi("/vnccs/module_status");
            if (r.ok) local = await r.json();
        } catch { /* server may not be ready yet */ }

        // 2. Fetch GitHub versions in parallel
        const [ghMain, ghUtils] = await Promise.all([
            this._fetchGitHubVersion(REPOS.main),
            this._fetchGitHubVersion(REPOS.utils),
        ]);
        const ghVersions = { main: ghMain, utils: ghUtils };

        const updateNeeded = [];

        for (const key of ["main", "utils"]) {
            const info   = local[key] ?? {};
            const label  = LABELS[key];
            const pill   = PILLS[key];
            const ghVer  = ghVersions[key];
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

            if (ghVer && this._semverGt(ghVer, locVer)) {
                this._updatePill(pill, label, locVer, "update", ghVer);
                updateNeeded.push(`${label} v${locVer} → v${ghVer}`);
            } else {
                this._updatePill(pill, label, locVer, "ok");
            }
        }

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
            this._renderAll();
        }

        // Debounce: reuse in-flight fetch for same repo
        if (!force && window.VNCCS_CC_FETCH_PROMISES[repoId]) {
            try {
                const data = await window.VNCCS_CC_FETCH_PROMISES[repoId];
                this.config = data;
                this.statusText.textContent = data.name || "Control Center";
                if (!this.state.output_slot_names) this.state.output_slot_names = [];
                this._renderAll();
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
            await this._refreshDependencyStatus(true);
            this._renderAll();
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

    _renderAll() {
        if (!this.config) return;
        this.scrollArea.innerHTML = "";

        if (this._getSelectedType() === "nunchaku" && this.dependencyStatus && !this.dependencyStatus.ok) {
            const err = document.createElement("div");
            err.className = "vnccs-cc-error";
            err.textContent = "✕ " + this.dependencyStatus.message;
            this.scrollArea.appendChild(err);
        }

        const selType = this.state.selected_type || (this.config.available_types?.[0] ?? "");
        const isCP    = selType === "checkpoint";

        // Only variants of the selected type (set in Settings)
        const variants = (this.config.models || []).filter(m => !m.type || m.type === selType);

        // Auto-select first variant of this type if current selection doesn't match
        const curOk = variants.find(m => m.name === this.state.selected_model);
        if (!curOk && variants.length) {
            this.state.selected_model = variants[0].name;
        }

        // MODEL — 2-column block: one big card for the selected type's variants
        this._renderTwoColBlockSingle(
            "MODEL", variants.length, "models",
            () => this._renderModelCard(selType, variants),
            () => this._renderModelParams(),
            () => this._renderModelSamplerParams()
        );

        // CLIP + VAE — single column (no params)
        if (!isCP) {
            const entries = [
                ...this.config.clip.map(e => ({ ...e, _cat: "clip" })),
                { _divider: true },
                ...this.config.vae.map(e => ({ ...e, _cat: "vae" })),
            ];
            this._renderBlock("CLIP + VAE", entries, "clip_vae",
                e => e._divider ? this._divider() : this._renderClipVaeEntry(e));
        }

        // LORA
        this._renderBlock("LORA", this.config.lora, "lora", e => this._renderLoraEntry(e));

        // CONTROLNET / OTHER
        this._renderBlock("CONTROLNET", this.config.controlnet, "controlnet",
            e => this._renderCnetEntry("controlnet", e));
        this._renderBlock("OTHER", this.config.other, "other",
            e => this._renderCnetEntry("other", e));
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

    // Two-column block with a single pre-built row (MODEL slot)
    _renderTwoColBlockSingle(title, count, key, rowFn, rightFn, rightFn2 = null) {
        const collapsed = this.state.collapsed?.[key] ?? false;
        const block = this._blockShell(title, count, key, collapsed);
        if (!collapsed) {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-twocol";

            const left = document.createElement("div");
            left.className = "vnccs-cc-twocol-left";
            if (count) left.appendChild(rowFn());
            else {
                const empty = document.createElement("div");
                empty.className = "vnccs-cc-block-empty";
                empty.textContent = "No entries";
                left.appendChild(empty);
            }

            const right = document.createElement("div");
            right.className = "vnccs-cc-twocol-right";
            right.appendChild(rightFn());

            wrap.append(left, right);

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

    _renderModelCard(type, variants) {
        const cur = variants.find(v => v.name === this.state.selected_model) ?? variants[0];
        if (!cur) return document.createElement("div");

        const key    = `cc_models_${cur.name}`;
        const dls    = this.dlStatus[key] ?? {};
        const status = dls.status || cur.status || "missing";

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
            statusLbl.textContent = "✓ Installed";
            statusLbl.classList.add("vnccs-cc-model-card-status--ok");
        } else if (status === "downloading" || status === "queued") {
            statusLbl.textContent = status === "queued" ? "⏳ Queued" : (dls.message || "Downloading…");
            statusLbl.classList.add("vnccs-cc-model-card-status--progress");
        } else {
            statusLbl.textContent = "↓ Missing";
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

            const icon = s => ({ installed:"●", missing:"↓", queued:"…", downloading:"⬇", error:"✕", auth_required:"⚠" }[s] ?? "?");

            const sel = document.createElement("select");
            sel.className = "vnccs-cc-model-card-select";
            variants.forEach(v => {
                const vDls = this.dlStatus[`cc_models_${v.name}`] ?? {};
                const vSt  = vDls.status || v.status || "missing";
                const op = document.createElement("option");
                op.value = v.name;
                op.textContent = (v.name.slice(prefixLen) || v.name) + "  " + icon(vSt);
                if (v.name === cur.name) op.selected = true;
                sel.appendChild(op);
            });
            sel.onchange = () => {
                const chosen = variants.find(v => v.name === sel.value);
                if (chosen) {
                    this.state.selected_model = chosen.name;
                    this.state.selected_type  = chosen.type || type;
                    this._saveState();
                    setTimeout(async () => {
                        await this._refreshDependencyStatus(true);
                        this._renderAll();
                    }, 0);
                }
            };
            footer.appendChild(sel);
        }

        if (status === "auth_required") {
            const b = this._btn("⚠ Enter Key", () => this.showApiKeyDialog("models", cur));
            b.style.color = "#ffaa00"; b.style.borderColor = "rgba(255,170,0,0.4)";
            footer.appendChild(b);
        } else if (status === "missing" || status === "error") {
            footer.appendChild(this._btn("⬇ Download", () => this._downloadEntry("models", cur)));
        }

        if (footer.children.length) card.appendChild(footer);
        return card;
    }

    // ── MODEL right column: sampler params ────────────────────────────────────

    _modelParamsSave(patch) {
        if (!this.state.model_params) this.state.model_params = {};
        Object.assign(this.state.model_params, patch);
        this._saveState();
    }

    _renderModelParams() {
        const p  = this.state.model_params ?? {};
        const mp = { steps: 20, cfg: 3.5, ...p };

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-params";

        const field = (labelText, el) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-param-field";
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
        const p  = this.state.model_params ?? {};
        const mp = { sampler: "euler", scheduler: "karras", ...p };

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-params";

        const field = (labelText, el) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-cc-param-field";
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
        const status = dls.status || entry.status;

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
            const b = this._btn("⚠ Enter Key", () => this.showApiKeyDialog(cat, entry));
            b.style.color = "#ffaa00"; b.style.borderColor = "rgba(255,170,0,0.4)";
            row.appendChild(b);
        } else if (status === "missing" || status === "error") {
            row.appendChild(this._btn("↓", () => this._downloadEntry(cat, entry)));
        }
        return row;
    }

    // ── LORA entry ────────────────────────────────────────────────────────────

    _renderLoraEntry(entry) {
        const ls  = (this.state.loras ?? []).find(l => l.name === entry.name)
            ?? { name: entry.name, enabled: true, strength: 1.0 };
        const dls = this.dlStatus[`cc_lora_${entry.name}`] ?? {};
        const status = dls.status || entry.status;

        const loraKey = `cc_lora_${entry.name}`;
        const row = document.createElement("div");
        row.className = "vnccs-cc-row";
        this._applyProgressLayer(row, loraKey, dls);

        row.appendChild(this._badge(status));

        if (status === "missing" || status === "error" || status === "queued" || status === "downloading" || status === "auth_required") {
            const nameWrap = document.createElement("div");
            nameWrap.className = "vnccs-cc-row-name-wrap";
            const name = document.createElement("span");
            name.className = "vnccs-cc-row-name";
            name.style.color = "#5e5e70";
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
                const b = this._btn("⚠ Enter Key", () => this.showApiKeyDialog("lora", entry));
                b.style.color = "#ffaa00"; b.style.borderColor = "rgba(255,170,0,0.4)";
                row.appendChild(b);
            } else {
                row.appendChild(this._btn("↓", () => this._downloadEntry("lora", entry)));
            }
            return row;
        }

        const chk = document.createElement("input");
        chk.type = "checkbox"; chk.checked = ls.enabled;
        chk.className = "vnccs-cc-lora-chk";
        chk.onchange = () => this._updateLora(entry.name, { enabled: chk.checked }, { commit: true });

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

        const slider = document.createElement("input");
        slider.type = "range"; slider.min = -2; slider.max = 2; slider.step = 0.05;
        slider.value = ls.strength ?? 1.0;
        slider.className = "vnccs-cc-lora-slider";

        const val = document.createElement("span");
        val.className = "vnccs-cc-lora-val";
        val.textContent = parseFloat(slider.value).toFixed(2);

        slider.oninput = () => {
            const strength = parseFloat(slider.value);
            val.textContent = strength.toFixed(2);
            this._updateLora(entry.name, { strength }, { commit: false });
        };

        slider.onchange = () => {
            const strength = parseFloat(slider.value);
            val.textContent = strength.toFixed(2);
            this._updateLora(entry.name, { strength }, { commit: true });
        };

        row.append(chk, nameWrap, slider, val);
        return row;
    }

    _updateLora(name, patch, options = {}) {
        const { commit = true } = options;
        if (!this.state.loras) this.state.loras = [];
        const idx = this.state.loras.findIndex(l => l.name === name);
        idx >= 0 ? Object.assign(this.state.loras[idx], patch)
                 : this.state.loras.push({ name, enabled: true, strength: 1.0, ...patch });
        this._saveState();
        if (commit) {
            this._refreshDependencyStatus(true).then(() => this._renderAll());
        }
    }

    // ── CONTROLNET / OTHER entry ──────────────────────────────────────────────

    _renderCnetEntry(cat, entry) {
        const cnetKey = `cc_${cat}_${entry.name}`;
        const dls    = this.dlStatus[cnetKey] ?? {};
        const status = dls.status || entry.status;

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
            const b = this._btn("⚠ Enter Key", () => this.showApiKeyDialog(cat, entry));
            b.style.color = "#ffaa00"; b.style.borderColor = "rgba(255,170,0,0.4)";
            row.appendChild(b);
        } else if (status === "missing" || status === "error") {
            row.appendChild(this._btn("↓", () => this._downloadEntry(cat, entry)));
        }
        return row;
    }

    // ── Output slot management ────────────────────────────────────────────────
    // Uses node.addOutput() / node.removeOutput() so slots are truly added/removed.
    // slot 0 = pipe (always present, defined in Python RETURN_TYPES)
    // slots 1..N = dynamic paths, managed here

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
        const desired = (this.state.output_slot_names ?? []).filter(n => n && n !== "-");
        // Current dynamic slots (skip slot 0 = pipe which is always present)
        const dynamicSlots = (this.node.outputs ?? []).slice(1);
        const currentNames = dynamicSlots.map(o => o.name);

        // Never rebuild if any existing slot has active links — would destroy connections.
        const hasLinks = dynamicSlots.some(o => o.links?.length > 0);
        if (hasLinks) return;

        // If slot names already match, nothing to do.
        if (desired.length === currentNames.length && desired.every((n, i) => n === currentNames[i])) return;

        // Remove all dynamic outputs (keep slot 0 = pipe)
        while ((this.node.outputs?.length ?? 0) > 1) {
            this.node.removeOutput(this.node.outputs.length - 1);
        }
        // Re-add in order
        for (const name of desired) {
            this.node.addOutput(name, "*");
        }
        this.node.setDirtyCanvas(true, true);
    }

    // Auto-assign ALL controlnet + other entries as output slots from config
    _syncCnetSlots(config) {
        const allNames = [
            ...(config.controlnet ?? []).map(e => e.name),
            ...(config.other ?? []).map(e => e.name),
        ];
        this.state.output_slot_names = allNames;
        this._syncOutputSlots();
    }

    // ── Settings overlay ──────────────────────────────────────────────────────

    _showSettings() {
        const ov = document.createElement("div");
        ov.className = "vnccs-cc-settings-overlay";

        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";

        // All supported model types — always shown regardless of what's in the config
        const ALL_TYPES = ["gguf", "unet", "checkpoint", "nunchaku"];
        const ts   = this.state.type_settings ?? {};
        const sel  = this.state.selected_type ?? (this.config?.available_types?.[0] ?? "gguf");
        const gguf = ts.gguf ?? {};
        const unet = ts.unet ?? {};
        const nun  = ts.nunchaku ?? {};

        const field = (labelText, el) => {
            const w = document.createElement("div");
            w.className = "vnccs-cc-settings-field";
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
        const mkInput = (id, type, min, max, step, val) => {
            const i = document.createElement("input");
            i.id = id; i.type = type;
            if (min !== null) i.min = min;
            if (max !== null) i.max = max;
            if (step !== null) i.step = step;
            i.value = val;
            i.className = "vnccs-cc-settings-input";
            return i;
        };

        const title = document.createElement("h3");
        title.className = "vnccs-cc-settings-title";
        title.textContent = "⚙ Settings";
        panel.appendChild(title);

        panel.appendChild(field("Model Type", mkSel("vnccs-cc-type", ALL_TYPES, sel)));

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

        // Nunchaku
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

        // Tokens
        const tokDet = document.createElement("details");
        tokDet.className = "vnccs-cc-settings-details";
        const tokSum = document.createElement("summary");
        tokSum.textContent = "API Tokens";
        tokDet.appendChild(tokSum);
        const hfIn = mkInput("vnccs-cc-hf-tok", "password", null, null, null, "");
        hfIn.placeholder = "hf_...";
        const cvIn = mkInput("vnccs-cc-cv-tok", "password", null, null, null, "");
        cvIn.placeholder = "Civitai key...";
        tokDet.append(field("HuggingFace Token", hfIn), field("Civitai Token", cvIn));
        panel.appendChild(tokDet);

        // Buttons
        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";

        const cancelBtn = this._btn("Cancel", () => ov.remove());

        const saveBtn = this._btn("Save", async () => {
            const newType = panel.querySelector("#vnccs-cc-type")?.value ?? sel;
            this.state.selected_type = newType;
            if (!this.state.type_settings) this.state.type_settings = {};
            this.state.type_settings.unet = {
                weight_dtype: panel.querySelector("#vnccs-cc-unet-dtype")?.value ?? "default",
            };
            this.state.type_settings.gguf = {
                dequant_dtype: panel.querySelector("#vnccs-cc-gguf-dequant")?.value ?? "default",
            };
            this.state.type_settings.nunchaku = {
                cpu_offload:     panel.querySelector("#vnccs-cc-nun-offload")?.value ?? "auto",
                num_blocks_on_gpu: parseInt(panel.querySelector("#vnccs-cc-nun-blocks")?.value ?? 1, 10),
                use_pin_memory: panel.querySelector("#vnccs-cc-nun-pin")?.value ?? "disable",
            };
            const hf = hfIn.value.trim(), cv = cvIn.value.trim();
            if (hf || cv) {
                await api.fetchApi("/vnccs/manager/save_token", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(Object.fromEntries(
                        [hf && ["hf_token", hf], cv && ["civitai_token", cv]].filter(Boolean)
                    )),
                }).catch(console.error);
            }
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
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repo_id: repoId, category: cat, name: entry.name }),
            });
            const d = await r.json();
            if (d.error) {
                this.dlStatus[key] = { status: "error", message: d.error };
                this._renderAll();
            } else {
                window.dispatchEvent(new CustomEvent("vnccs-cc-registry-updated"));
            }
        } catch (e) {
            this.dlStatus[key] = { status: "error", message: String(e) };
            this._renderAll();
        }
    }

    async _downloadAllMissing() {
        if (!this.config) { this.showMessage("Load config first.", true); return; }

        const tasks = [];

        // MODEL: only the currently selected variant
        const selModel = this.state.selected_model;
        const modelEntry = (this.config.models ?? []).find(e => e.name === selModel);
        if (modelEntry && (modelEntry.status === "missing" || modelEntry.status === "error")) {
            tasks.push({ cat: "models", e: modelEntry });
        }

        // Everything else: all missing or errored
        for (const cat of ["clip", "vae", "lora", "controlnet", "other"]) {
            for (const e of (this.config[cat] ?? [])) {
                if (e.status === "missing" || e.status === "error") tasks.push({ cat, e });
            }
        }

        if (!tasks.length) { this.showMessage("All selected items are installed."); return; }
        this.showConfirm(`Download ${tasks.length} missing item(s)?`, async () => {
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
        box.innerHTML = `<div style="color:#e8e8f0;font-size:12px;margin-bottom:14px;line-height:1.5;font-family:'Sora',sans-serif;">${text}</div>`;
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
        box.innerHTML = `<div style="color:#e8e8f0;font-size:12px;margin-bottom:14px;line-height:1.5;font-family:'Sora',sans-serif;">${text}</div>`;
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

    showApiKeyDialog(cat, entry) {
        const ov = document.createElement("div");
        ov.style.cssText = `
            position: absolute; top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.85); z-index:100;
            display:flex; flex-direction:column; justify-content:center; align-items:center;
            padding:16px; box-sizing:border-box;
        `;
        const panel = document.createElement("div");
        panel.className = "vnccs-cc-settings-panel";

        const title = document.createElement("h3");
        title.className = "vnccs-cc-settings-title";
        title.textContent = "⚠ API Key Required";
        panel.appendChild(title);

        const desc = document.createElement("p");
        desc.style.cssText = "margin:0;font-size:11px;color:#9898a8;line-height:1.5;";
        desc.textContent = `"${entry.name}" requires an API token to download.`;
        panel.appendChild(desc);

        const mkField = (labelText, inputEl) => {
            const w = document.createElement("div");
            w.className = "vnccs-cc-settings-field";
            const l = document.createElement("label");
            l.className = "vnccs-cc-settings-label";
            l.textContent = labelText;
            w.append(l, inputEl);
            return w;
        };
        const hfIn = document.createElement("input");
        hfIn.type = "password"; hfIn.placeholder = "hf_...";
        hfIn.className = "vnccs-cc-settings-input";
        const cvIn = document.createElement("input");
        cvIn.type = "password"; cvIn.placeholder = "Civitai key...";
        cvIn.className = "vnccs-cc-settings-input";
        panel.appendChild(mkField("HuggingFace Token", hfIn));
        panel.appendChild(mkField("Civitai API Key", cvIn));

        const btns = document.createElement("div");
        btns.className = "vnccs-cc-settings-btns";
        const cancelBtn = this._btn("Cancel", () => ov.remove());
        const saveBtn = this._btn("Save & Retry", async () => {
            const hf = hfIn.value.trim(), cv = cvIn.value.trim();
            if (!hf && !cv) { this.showMessage("Enter at least one token.", true); return; }
            const payload = {};
            if (hf) payload.hf_token = hf;
            if (cv) payload.civitai_token = cv;
            try {
                const r = await api.fetchApi("/vnccs/manager/save_token", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (r.ok) {
                    ov.remove();
                    this._downloadEntry(cat, entry);
                } else {
                    this.showMessage("Failed to save tokens.", true);
                }
            } catch (e) {
                this.showMessage("Error: " + e.message, true);
            }
        });
        saveBtn.classList.add("vnccs-cc-btn--save");
        btns.append(cancelBtn, saveBtn);
        panel.appendChild(btns);
        ov.appendChild(panel);
        this.container.appendChild(ov);
    }
}
