import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { registerCleanup, syncDOMWidgetWidth, syncDOMWidgetWidthSoon } from "./vnccs_common.js";

const DEFAULT_DATA = {
    pose_generation: {
        target_size: 1024,
    },
    upscaler: {
        model: "seedvr2_ema_3b-Q4_K_M.gguf",
        vae: "ema_vae_fp16.safetensors",
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
        cache_dit: false,
        attention_mode: "sdpa",
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
        tolerance: 0.5,
        despill_strength: 1,
        despill_kernel_size: 3,
        despill_color: "black",
    },
    ui: {
        selected_preview: "pose_generation",
        user_selected_preview: false,
    },
};

const STAGES = [
    ["pose_generation", "Pose Generation"],
    ["upscaler", "Upscaler"],
    ["bg_remove", "BG Remove"],
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
.vnccs-pipe-input, .vnccs-pipe-select {
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
.vnccs-pipe-check {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #cfcfda;
    font-size: 11px;
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
    grid-template-columns: repeat(6, minmax(70px, 1fr));
    grid-template-rows: repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.vnccs-pipe-img {
    width: 100%;
    height: 100%;
    display: block;
    min-height: 0;
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
.vnccs-pipe-empty {
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #5e5e70;
    font-size: 12px;
}
.vnccs-pipe-chain {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    padding: 10px 12px 12px;
    background: #111119;
    border-top: 1px solid rgba(255,143,163,0.14);
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
.vnccs-pipe-stage.is-done {
    border-color: rgba(0,214,143,0.45);
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
.vnccs-pipe-tabs {
    display: flex;
    gap: 6px;
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
    if (document.getElementById("vnccs-pipeline-style")) return;
    const style = document.createElement("style");
    style.id = "vnccs-pipeline-style";
    style.textContent = CSS;
    document.head.appendChild(style);
}

function deepMerge(base, patch) {
    const out = JSON.parse(JSON.stringify(base));
    for (const [section, values] of Object.entries(patch || {})) {
        if (values && typeof values === "object" && !Array.isArray(values)) {
            out[section] = { ...(out[section] || {}), ...values };
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

function writeData(node, data) {
    const widget = node.widgets?.find(w => w.name === "widget_data");
    if (!widget) return;
    widget.value = JSON.stringify(data);
    widget.callback?.(widget.value);
    app.graph?.setDirtyCanvas(true, true);
}

function uniqueOptions(values) {
    return [...new Set(values.filter(Boolean))];
}

class PipelineWidget {
    constructor(node) {
        this.node = node;
        this.data = readData(node);
        this.stageState = Object.fromEntries(STAGES.map(([key]) => [key, { status: "waiting", images: null, message: "" }]));
        this.selectedPreview = this.data.ui?.selected_preview || "pose_generation";
        this.userSelectedPreview = Boolean(this.data.ui?.user_selected_preview);
        this.viewer = null;
        this.viewerFocus = null;
        this.restoredViewer = null;
        this._saveBrowserStateTimer = null;
        this.nodeDefs = {};
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

        this.node.addDOMWidget("pipeline_ui", "ui", root, {
            serialize: false,
            hideOnZoom: false,
        });
        syncDOMWidgetWidthSoon(this.node, "pipeline_ui");

        const dataWidget = this.node.widgets?.find(w => w.name === "widget_data");
        if (dataWidget) {
            dataWidget.hidden = true;
            dataWidget.computeSize = () => [0, -4];
        }

        this.syncCharacterNameFromCreator();
        writeData(this.node, this.data);

        this.renderSettings();
        this.renderPreview();
        this.renderChain();
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
            } else {
                const status = detail.status || "waiting";
                if (status === "running") {
                    this.resetStagesFrom(stage);
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
                if (detail.images && !this.userSelectedPreview) {
                    this.selectedPreview = stage;
                    this.persistUI();
                }
            }
            if (this.viewer?.open) this.syncViewerImage();
            this.renderPreview();
            this.renderChain();
            this.saveBrowserState();
        };
        api.addEventListener("vnccs.pipeline.stage", this.onStage);
        registerCleanup(this.node, () => api.removeEventListener("vnccs.pipeline.stage", this.onStage));
    }

    resetStagesFrom(stageKey) {
        const start = STAGES.findIndex(([key]) => key === stageKey);
        if (start < 0) return;
        for (const [key] of STAGES.slice(start)) {
            this.stageState[key] = {
                status: "waiting",
                images: null,
                message: "",
                current: undefined,
                total: undefined,
            };
        }
    }

    persistUI() {
        this.syncCharacterNameFromCreator();
        this.data.ui = {
            ...(this.data.ui || {}),
            selected_preview: this.selectedPreview,
            user_selected_preview: this.userSelectedPreview,
        };
        writeData(this.node, this.data);
        this.saveBrowserState();
    }

    set(section, key, value) {
        this.syncCharacterNameFromCreator();
        this.data[section][key] = value;
        writeData(this.node, this.data);
        this.saveBrowserState();
    }

    syncCharacterNameFromCreator() {
        const creator = app.graph?._nodes?.find(n => n.type === "CharacterCreatorV2");
        const widget = creator?.widgets?.find(w => w.name === "widget_data");
        if (!widget?.value) return;
        try {
            const payload = JSON.parse(widget.value);
            if (payload?.character) this.data.character_name = payload.character;
        } catch {
            // Leave the previous value if the creator widget is mid-edit.
        }
    }

    storageKey() {
        return `vnccs:character-sheet-pipeline:${this.node.type || "node"}:${this.node.id}`;
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
        if (STAGES.some(([key]) => key === saved.selectedPreview)) {
            this.selectedPreview = saved.selectedPreview;
        }
        this.userSelectedPreview = Boolean(saved.userSelectedPreview);

        if (saved.stageState && typeof saved.stageState === "object") {
            for (const [key] of STAGES) {
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
            if (saved.viewer.open && STAGES.some(([key]) => key === saved.viewer.stage)) {
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
        this.syncCharacterNameFromCreator();
        const stageState = {};
        for (const [key] of STAGES) {
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
            for (const [key] of STAGES) {
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
        const names = ["SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler"];
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
        this.renderSettings();
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

    field(section, key, label, type = "text", options = null) {
        const wrap = document.createElement("label");
        wrap.className = "vnccs-pipe-field";
        const caption = document.createElement("div");
        caption.className = "vnccs-pipe-label";
        caption.textContent = label;
        let input;
        if (type === "select") {
            input = document.createElement("select");
            input.className = "vnccs-pipe-select";
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
        } else {
            input = document.createElement("input");
            input.className = "vnccs-pipe-input";
            input.type = type;
        }
        input.value = this.data[section][key];
        input.oninput = () => {
            const raw = type === "number" ? Number(input.value) : input.value;
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

    renderSettings() {
        this.settingsEl.innerHTML = "";
        const title = document.createElement("div");
        title.className = "vnccs-pipe-title";
        title.textContent = "VNCCS Pipeline";
        this.settingsEl.appendChild(title);
        this.settingsEl.appendChild(this.block("Pose Generation", [
            this.field("pose_generation", "target_size", "target size", "select", [1024, 1344, 1536, 2048, 768, 512]),
        ]));
        this.settingsEl.appendChild(this.block("Upscaler", [
            this.field("upscaler", "model", "dit model", "select", this.getWorkflowModelOptions("SeedVR2LoadDiTModel", "model", WORKFLOW_UPSCALER_DIT_MODELS, this.data.upscaler.model)),
            this.field("upscaler", "vae", "vae model", "select", this.getWorkflowModelOptions("SeedVR2LoadVAEModel", "model", WORKFLOW_UPSCALER_VAE_MODELS, this.data.upscaler.vae)),
            this.field("upscaler", "device", "device", "select", this.getOptions("SeedVR2LoadDiTModel", "device", ["cuda:0", "cuda:1", "cpu", "mps"])),
            this.field("upscaler", "offload_device", "offload", "select", this.getOptions("SeedVR2LoadDiTModel", "offload_device", ["cpu", "cuda:0", "cuda:1", "mps"])),
            this.field("upscaler", "seed", "seed", "number"),
            this.field("upscaler", "resolution", "resolution", "number"),
        ]));
        this.settingsEl.appendChild(this.block("BG Remove", [
            this.field("bg_remove", "tolerance", "tolerance", "number"),
            this.field("bg_remove", "despill_strength", "despill strength", "number"),
            this.field("bg_remove", "despill_kernel_size", "kernel size", "number"),
            this.field("bg_remove", "despill_color", "despill color", "select", ["black", "limit", "interior_average", "guided_filter", "difference_trim"]),
        ]));
    }

    renderPreview() {
        this.previewEl.innerHTML = "";
        const head = document.createElement("div");
        head.className = "vnccs-pipe-preview-head";
        const label = document.createElement("div");
        label.textContent = STAGES.find(([key]) => key === this.selectedPreview)?.[1] || "Results";
        const tabs = document.createElement("div");
        tabs.className = "vnccs-pipe-tabs";
        for (const [key, name] of STAGES) {
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
        head.append(label, tabs);
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
        images.forEach((src, index) => {
            const tile = document.createElement("button");
            tile.type = "button";
            tile.className = "vnccs-pipe-img";
            tile.style.backgroundImage = `url("${String(src).replaceAll('"', "%22")}")`;
            tile.onclick = () => this.openViewer(index);
            grid.appendChild(tile);
        });
        this.previewEl.appendChild(grid);
    }

    formatStageStatus(key) {
        const state = this.stageState[key] || {};
        const status = state.status || "waiting";
        const count = Number.isFinite(state.current) && Number.isFinite(state.total)
            ? ` (${state.current}/${state.total})`
            : (state.images?.length ? ` (${state.images.length})` : "");
        if (state.message) return `${state.message}${count}`;
        if (status === "running") return `Running${count}`;
        if (status === "done") return `Done${count}`;
        if (status === "error") return "Error";
        return "Waiting";
    }

    renderChain() {
        this.chainEl.innerHTML = "";
        for (const [key, name] of STAGES) {
            const stage = document.createElement("div");
            const status = this.stageState[key]?.status || "waiting";
            stage.className = "vnccs-pipe-stage";
            if (status === "running") stage.classList.add("is-active");
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
            this.chainEl.appendChild(stage);
        }
    }

    currentImages() {
        return this.stageState[this.selectedPreview]?.images || [];
    }

    openViewer(index = 0, restored = null) {
        const images = this.currentImages();
        if (!images.length) return;
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
        for (const [key, name] of STAGES) {
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
            this.viewer.dragging = true;
            this.viewer.dragX = event.clientX;
            this.viewer.dragY = event.clientY;
            canvas.classList.add("is-dragging");
            canvas.setPointerCapture(event.pointerId);
        };
        canvas.onpointermove = (event) => {
            if (!this.viewer?.dragging) return;
            this.viewer.x += event.clientX - this.viewer.dragX;
            this.viewer.y += event.clientY - this.viewer.dragY;
            this.viewer.dragX = event.clientX;
            this.viewer.dragY = event.clientY;
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
        this.viewer.img.style.transform = `translate(${this.viewer.x}px, ${this.viewer.y}px) scale(${this.viewer.scale})`;
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

        const anchorX = event ? event.clientX - rect.left : rect.width / 2;
        const anchorY = event ? event.clientY - rect.top : rect.height / 2;
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
    name: "VNCCS.CharacterSheetPipeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_CharacterSheetPipeline") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.setSize([1180, 760]);
            this._vnccsPipelineWidget = new PipelineWidget(this);
            syncDOMWidgetWidthSoon(this, "pipeline_ui");
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            if (this._vnccsPipelineWidget) {
                this._vnccsPipelineWidget.data = readData(this);
                this._vnccsPipelineWidget.restoreBrowserState();
                this._vnccsPipelineWidget.renderSettings();
                this._vnccsPipelineWidget.renderPreview();
                this._vnccsPipelineWidget.renderChain();
                if (this._vnccsPipelineWidget.restoredViewer?.open && this._vnccsPipelineWidget.currentImages().length) {
                    this._vnccsPipelineWidget.openViewer(
                        this._vnccsPipelineWidget.restoredViewer.index || 0,
                        this._vnccsPipelineWidget.restoredViewer,
                    );
                }
            }
            syncDOMWidgetWidthSoon(this, "pipeline_ui");
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            syncDOMWidgetWidth(this, "pipeline_ui");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "pipeline_ui"));
        };
    },
});
