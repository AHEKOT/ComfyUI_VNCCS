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
    },
    bg_remove: {
        tolerance: 0.5,
        despill_strength: 1,
        despill_kernel_size: 3,
        despill_color: "black",
    },
};

const STAGES = [
    ["pose_generation", "Pose Generation"],
    ["upscaler", "Upscaler"],
    ["bg_remove", "BG Remove"],
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
    min-height: 0;
    object-fit: contain;
    background: #14141e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    box-sizing: border-box;
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
    text-transform: uppercase;
    letter-spacing: 0.08em;
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
        this.stageState = Object.fromEntries(STAGES.map(([key]) => [key, { status: "waiting", images: null }]));
        this.selectedPreview = "pose_generation";
        this.build();
        this.bindEvents();
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

        this.renderSettings();
        this.renderPreview();
        this.renderChain();
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
                this.stageState[stage] = {
                    status: detail.status || "waiting",
                    images: detail.images || this.stageState[stage]?.images || null,
                    message: detail.message || "",
                };
                if (detail.images) this.selectedPreview = stage;
            }
            this.renderPreview();
            this.renderChain();
        };
        api.addEventListener("vnccs.pipeline.stage", this.onStage);
        registerCleanup(this.node, () => api.removeEventListener("vnccs.pipeline.stage", this.onStage));
    }

    set(section, key, value) {
        this.data[section][key] = value;
        writeData(this.node, this.data);
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
            this.field("upscaler", "model", "model", "select", [
                ...uniqueOptions([
                    this.data.upscaler.model,
                    "seedvr2_ema_3b-Q4_K_M.gguf",
                    "seedvr2_ema_3b_fp16.safetensors",
                ]),
            ]),
            this.field("upscaler", "vae", "vae", "select", [
                ...uniqueOptions([
                    this.data.upscaler.vae,
                    "ema_vae_fp16.safetensors",
                ]),
            ]),
            this.field("upscaler", "device", "device", "select", ["cuda:0", "cuda:1", "cpu", "mps"]),
            this.field("upscaler", "offload_device", "offload", "select", ["cpu", "cuda:0", "cuda:1", "mps"]),
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
            empty.textContent = "Waiting for pipeline results";
            this.previewEl.appendChild(empty);
            return;
        }
        const grid = document.createElement("div");
        grid.className = "vnccs-pipe-grid";
        for (const src of images) {
            const img = document.createElement("img");
            img.className = "vnccs-pipe-img";
            img.src = src;
            grid.appendChild(img);
        }
        this.previewEl.appendChild(grid);
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
                this.renderPreview();
            };
            const n = document.createElement("div");
            n.className = "vnccs-pipe-stage-name";
            n.textContent = name;
            const s = document.createElement("div");
            s.className = "vnccs-pipe-stage-status";
            s.textContent = status;
            stage.append(n, s);
            this.chainEl.appendChild(stage);
        }
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
            this._vnccsPipelineWidget?.renderSettings();
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
