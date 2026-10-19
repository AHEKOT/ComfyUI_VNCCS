import { app } from "../../scripts/app.js";
import { vnccsApi, watchConnection } from "./vnccs_transport.js";
import { createSakuraInteractions } from "./sakura_interactions.mjs";
import { presetSelection } from "./character_presets.mjs?v=legacy-tags-1";
import { SAKURA_THEME_CSS, SAKURA_INTERACTION_CSS, SAKURA_CREATOR_CONTROL_CSS, createTraitInput, injectStyles, showModal, showMessage, registerCleanup, generateRandomSeed, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, animateWizardRows, createControlCenterClient } from "./vnccs_common.js";

const CREATOR_ZOOM = 0.67;
const GENERATION_PANEL_WIDTH = (1280 / CREATOR_ZOOM - 24) * .35;
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.vnccs-background-creator { ${SAKURA_THEME_CSS} --warning:#ffaa00; --shadow-subtle:0 2px 8px rgba(0,0,0,0.3); --shadow-elevated:0 8px 32px rgba(0,0,0,0.5); }
${SAKURA_INTERACTION_CSS}
${SAKURA_CREATOR_CONTROL_CSS}

.vnccs-background-creator { display:flex; flex-direction:column; min-height:0; padding:12px; gap:12px; background:var(--bg-primary); }
.vnccs-background-creator .bg-header { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; flex:none; }
.vnccs-background-creator .bg-header strong { color:var(--accent); font-size:17px; letter-spacing:.03em; }
.vnccs-background-creator nav { flex:none; min-width:0; }
.vnccs-background-creator button:disabled { cursor:not-allowed; }

.vnccs-background-creator .bg-step { flex:1 1 0; min-width:0; min-height:0; display:grid; gap:12px; }
.vnccs-background-creator .bg-background { grid-template-columns:minmax(360px,1fr) minmax(${240 + GENERATION_PANEL_WIDTH * .2}px,calc((100% - 24px)*.21875 + ${GENERATION_PANEL_WIDTH * .2}px)) ${GENERATION_PANEL_WIDTH * .8}px; grid-template-rows:minmax(0,1fr); }
.vnccs-background-creator .bg-panel { min-width:0; min-height:0; overscroll-behavior:contain; overflow-x:hidden; }
.vnccs-background-creator .bg-source { scroll-padding-block:16px; }
.vnccs-background-creator .bg-panel>* { flex:0 0 auto; min-width:0; }
.vnccs-background-creator .bg-panel>button { min-height:40px; }
.vnccs-background-creator .bg-panel>.bg-viewport,
.vnccs-background-creator .bg-panel>.bg-cg-list { flex:1 1 0; min-height:0; }
.vnccs-background-creator textarea { user-select:text; }
.vnccs-background-creator input,.vnccs-background-creator select,.vnccs-background-creator textarea { min-width:0; }
.vnccs-background-creator input[type=range] { min-width:0; width:100%; accent-color:var(--accent); cursor:pointer; }
.vnccs-background-creator .bg-row { display:grid; grid-auto-flow:column; grid-auto-columns:minmax(0,1fr); align-items:start; gap:10px; }
.vnccs-background-creator .bg-row>* { min-width:0; }
.vnccs-background-creator .bg-actions { grid-template-columns:minmax(0,2fr) repeat(2,minmax(0,1fr)); align-items:stretch; gap:8px; }
.vnccs-background-creator .bg-row>button { min-height:40px; }

.vnccs-background-creator .bg-wide { width:100%; flex:none; min-height:40px; }
.vnccs-background-creator .bg-panel>.bg-preview { flex:none; width:100%; height:auto; aspect-ratio:16/9; min-height:0; }
.vnccs-background-creator .bg-preview>[data-preview] { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
.vnccs-background-creator .bg-preview .vnccs-common-creator-generation-img { width:100%; left:0; transform:none; }
.vnccs-background-creator .bg-preview [data-enlarge] { position:absolute; inset:0; width:100%; height:100%; padding:0; border:0; background:transparent; cursor:zoom-in; }
.vnccs-background-creator .bg-preview>[data-empty] { padding:18px; }
.vnccs-background-creator .bg-variation-head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
.vnccs-background-creator .bg-variation-head button { padding:6px 12px; }
.vnccs-background-creator .bg-variation-strip { display:grid; grid-template-columns:32px minmax(0,1fr) 32px; gap:8px; align-items:center; }
.vnccs-background-creator .bg-variation-cards { display:flex; gap:10px; min-width:0; overflow-x:auto; overscroll-behavior-x:contain; scrollbar-width:thin; scrollbar-color:var(--accent-border) transparent; scroll-snap-type:x proximity; padding:2px; }
.vnccs-background-creator .bg-variation-item { display:flex; flex-direction:column; gap:7px; flex:0 0 calc((100% - 20px)/3); min-width:160px; scroll-snap-align:start; }
.vnccs-background-creator .bg-variation-scroll { width:32px; height:32px; }
.vnccs-background-creator .bg-variation-item>span { color:var(--text-primary); font-size:13px; font-weight:600; overflow-wrap:anywhere; }
.vnccs-background-creator .bg-variation-preview { position:relative; }
.vnccs-background-creator [data-edit-variation] { position:relative; display:block; width:100%; aspect-ratio:16/9; min-height:52px; padding:0; background:var(--bg-surface); border:2px solid color-mix(in srgb,var(--accent-lavender) 60%,var(--bg-surface)); border-radius:var(--radius-md); }
.vnccs-background-creator [data-edit-variation]:hover { border-color:var(--accent); background:var(--bg-hover); }
.vnccs-background-creator [data-edit-variation][aria-expanded=true] { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
.vnccs-background-creator [data-variation-image] { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:cover; border-radius:10px; pointer-events:none; }
.vnccs-background-creator .bg-variation-delete { position:absolute; top:8px; right:8px; border:1px solid var(--accent-border); border-radius:6px; background:var(--bg-primary); color:var(--text-primary); cursor:pointer; width:28px; height:28px; padding:0; font-size:20px; line-height:1; }
.vnccs-background-creator .bg-variation-delete:hover:not(:disabled) { color:var(--error); }
.vnccs-background-creator .selected { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }


.vnccs-background-creator .bg-variation-editor { border-top:1px solid var(--accent-border); padding-top:10px; }
.vnccs-background-creator .bg-variation-editor-head { display:flex; align-items:center; gap:12px; }
.vnccs-background-creator .bg-variation-editor-head label { display:flex; align-items:center; gap:12px; flex:1; color:var(--text-secondary); }
.vnccs-background-creator .bg-variation-editor-head input { flex:1; }
.vnccs-background-creator .bg-variation-editor-head>[data-close-variation] { flex:0 0 auto; width:auto; }
.vnccs-background-creator .bg-variation-fields { display:flex; flex-direction:column; }
.vnccs-background-creator .bg-attributes>.bg-description { flex:1 1 180px; min-height:180px; }
.vnccs-background-creator .bg-description textarea { flex:1; min-height:140px; resize:none; }
.vnccs-background-creator .bg-description textarea:focus { background:rgba(255,255,255,0.04); box-shadow:none; }
.vnccs-background-creator .bg-image-dialog { width:min(1400px,calc(100% - 32px)); max-width:min(1400px,calc(100% - 32px)); }
.vnccs-background-creator .bg-image-dialog img { display:block; width:100%; max-height:75vh; object-fit:contain; }
.vnccs-background-creator .bg-model { grid-template-columns:240px minmax(380px,1fr) 290px; }
.vnccs-background-creator .bg-model-source { gap:12px; }
.vnccs-background-creator [data-model-source] { width:100%; aspect-ratio:16/9; object-fit:contain; background:#09090f; border-radius:10px; }
.vnccs-background-creator [data-model-dimensions],.vnccs-background-creator [data-model-status] { color:var(--text-secondary); font-size:11px; }
.vnccs-background-creator .bg-viewport-panel { overflow:hidden; }
.vnccs-background-creator .bg-badge { margin-left:auto; padding:3px 7px; border-radius:20px; border:1px solid var(--accent-border); font-size:10px; font-weight:500; letter-spacing:0; color:var(--text-secondary); }
.vnccs-background-creator .bg-viewport { flex:1; position:relative; min-height:0; background:#08090e; border:1px solid var(--border); border-radius:14px; overflow:hidden; }
.vnccs-background-creator [data-viewer] { position:absolute; inset:0; }
.vnccs-background-creator [data-viewer][inert] { pointer-events:none; }
.vnccs-background-creator [data-viewer] canvas:focus-visible { outline-offset:-3px; }
.vnccs-background-creator .bg-camera-toolbar { display:grid; grid-template-columns:minmax(160px,1fr) auto auto; align-items:center; gap:8px; }
.vnccs-background-creator .bg-camera-toolbar label { display:grid; grid-template-columns:auto minmax(0,1fr) 56px; align-items:center; gap:7px; min-width:0; font-size:11px; color:var(--text-secondary); }
.vnccs-background-creator .bg-camera-toolbar input[type=number] { width:56px; height:34px; padding:7px 4px; }
.vnccs-background-creator .bg-camera-toolbar button { flex:none; height:34px; white-space:nowrap; }

.vnccs-background-creator .bg-viewer-empty { position:absolute; inset:20px; display:grid; place-items:center; text-align:center; color:var(--text-secondary); pointer-events:none; }
.vnccs-background-creator .bg-hint { font-size:10px; color:var(--text-secondary); }
.vnccs-background-creator .bg-cg-panel { overflow:hidden; }
.vnccs-background-creator .bg-cg-list { min-height:0; overflow:auto; display:flex; flex-direction:column; gap:14px; padding:2px; }
.vnccs-background-creator .bg-cg-card { flex:none; display:flex; flex-direction:column; gap:9px; text-align:left; padding:12px; text-transform:none; letter-spacing:normal; overflow-wrap:anywhere; }
.vnccs-background-creator .bg-cg-card img { width:100%; aspect-ratio:16/9; object-fit:contain; background:#08090e; border-radius:8px; }
.vnccs-background-creator .bg-cg-card span { color:var(--text-secondary); font:10px var(--font-mono); }
.vnccs-background-creator [data-cg-empty] { margin:0; color:var(--text-secondary); }
.vnccs-background-creator .bg-hint,.vnccs-background-creator .bg-model-card,.vnccs-background-creator [data-missing-assets],.vnccs-background-creator [role=status] { overflow-wrap:anywhere; }
.vnccs-background-creator .bg-announcement { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip-path:inset(50%); white-space:nowrap; }
.vnccs-background-creator .error { color:var(--error) !important; }
.vnccs-background-creator { position:relative; color:var(--text-primary); font:13px var(--font); width:100%; box-sizing:border-box; overflow:hidden; zoom:${CREATOR_ZOOM}; }
.vnccs-background-creator *, .vnccs-background-creator *::before, .vnccs-background-creator *::after { box-sizing:border-box; }
.vnccs-background-creator [hidden] { display:none !important; }

.vnccs-background-creator .bg-dialog-fields { display:flex; flex-direction:column; gap:12px; }
.vnccs-background-creator .bg-viewer-canvas { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); touch-action:none; }
.vnccs-background-creator .bg-camera-toolbar .vnccs-common-ui-select-trigger { width:auto; }
.vnccs-background-creator .vnccs-common-modal { max-height:calc(100% - 32px); overflow:auto; }
.vnccs-background-creator.bg-compact .bg-step { grid-template-columns:minmax(0,1fr); grid-template-rows:none; grid-auto-rows:max-content; overflow:auto; overscroll-behavior:contain; scroll-padding-block:16px; }
.vnccs-background-creator.bg-compact .bg-panel { height:auto; overflow:visible; }
.vnccs-background-creator.bg-compact .bg-viewport-panel { height:480px; overflow:hidden; }
.vnccs-background-creator.bg-compact .bg-cg-panel { max-height:360px; overflow:auto; }
.vnccs-background-creator.bg-compact .bg-cg-panel>.bg-cg-list { flex:0 0 auto; overflow:visible; }
.vnccs-background-creator.bg-compact .bg-camera-toolbar { grid-template-columns:repeat(2,minmax(0,1fr)); }
.vnccs-background-creator.bg-compact .bg-camera-toolbar label { grid-column:1/-1; }
@media (prefers-reduced-motion:reduce) { .vnccs-background-creator *, .vnccs-background-creator *::before, .vnccs-background-creator *::after { transition:none !important; animation:none !important; } }

`;

export const DEFAULT_EDITOR = Object.freeze({
    description: "", scene_description: "", prompt: "", attributes: { scene_type: "Interior", time: "Day" },
    variants: ["Morning", "Evening", "Night"].map(name => ({ id: name.toLowerCase(), name, time: name, weather: "", additional_details: "" })),
    aspect: "16:9", style: "Hand-painted anime", aesthetics: "", negative: "",
    cg_prompt: "Complete this rough 3D background render into a finished image. Preserve the camera angle, perspective, composition, geometry, and lighting. Reconstruct missing details and remove Gaussian splat artifacts. Do not add people or characters.",
    generation: { model_type: "qi2", steps: 25, cfg: 1, sampler: "euler", scheduler: "simple", seed: 0, resolution_mp: 2, turbo: false, prompt_enhance: true, cache_device: "gpu", cache_dtype: "int8", lora_stack: [], diffusion_model_name: "qwen_image_2.1_int8_convrot.safetensors" },
});
export const MOSS_GENERATION = Object.freeze({ model_type: "moss", steps: 25, cfg: 2.5, sampler: "er_sde", scheduler: "simple", turbo: false,
    diffusion_model_name: "moss_image_sft_int8_convrot.safetensors" });
export const MODEL_DEFAULTS = Object.freeze({ num_gaussians: 262144, steps: 20, guidance_scale: 3, seed: 0 });
export const clone = value => JSON.parse(JSON.stringify(value));

export function sourceCameraView(source) {
    const camera = clone(source);
    // Keep the predicted image-plane size while choosing a practical perspective.
    const scale = Math.tan(camera.fov * Math.PI / 360) / Math.tan(Math.PI / 6);
    camera.position = camera.position.map((value, index) => camera.target[index] + (value - camera.target[index]) * scale);
    camera.fov = 60;
    return camera;
}

const LEGACY_SCENE_LABELS = { composition: "Composition", location: "Location", era: "Era", architecture: "Architecture", lighting: "Lighting", weather: "Weather", mood: "Atmosphere", objects: "Objects and furnishings", details: "Additional details" };
const promptLine = value => String(value || "").replace(/\s+/g, " ").trim();

export function parsePrompt(text) {
    const match = /^Style:[ \t]*(.*?)\r?\nTime:[ \t]*(.*?)\r?\nInterior\/Exterior:[ \t]*(.*?)\r?\nScene description:[ \t]*([\s\S]*)$/.exec(text.trim());
    return match ? { style: match[1], time: match[2], scene_type: match[3], scene_description: match[4] } : null;
}

export function legacySceneDescription(editor) {
    if (editor.prompt) return parsePrompt(editor.prompt)?.scene_description ?? editor.prompt;
    return [editor.description, ...Object.entries(LEGACY_SCENE_LABELS).map(([key, label]) => editor.attributes?.[key] ? `${label}: ${editor.attributes[key]}.` : "")].filter(Boolean).join("\n");
}

export function editorDefaults(value = {}) {
    const editor = { ...clone(DEFAULT_EDITOR), ...value, attributes: { ...DEFAULT_EDITOR.attributes, ...value.attributes },
        generation: { ...clone(DEFAULT_EDITOR.generation), ...(value.generation?.model_type === "moss" ? MOSS_GENERATION : {}), ...value.generation } };
    if (editor.generation.model_type === "moss") editor.generation.turbo = false;
    if (!["qi2", "moss"].includes(editor.generation.model_type)) throw new Error("Unsupported background model type.");
    if (!Object.hasOwn(value, "scene_description")) {
        const parsed = value.prompt && parsePrompt(value.prompt);
        if (parsed) { editor.style = parsed.style; editor.attributes.time = parsed.time; editor.attributes.scene_type = parsed.scene_type; }
        editor.scene_description = legacySceneDescription(editor);
    }
    if (!Array.isArray(editor.variants) || editor.variants.length > 64
        || editor.variants.some(card => !card || typeof card !== "object" || !["id", "name", "time", "weather", "additional_details"].every(key => typeof card[key] === "string"))) {
        throw new Error("Background variant cards are invalid. Saved data is retained.");
    }
    const day = { id: "day", name: "Day", time: "Day", weather: "", additional_details: "" };
    editor.variants = editor.variants.filter(card => Object.keys(card).length !== 5 || Object.entries(day).some(([key, value]) => card[key] !== value));
    return editor;
}

export function composePrompt(editor) {
    const description = editor.scene_description ?? legacySceneDescription(editor);
    const body = [description, editor.aesthetics && !description.includes(editor.aesthetics) ? editor.aesthetics : ""].filter(Boolean).join("\n");
    return `Style: ${promptLine(editor.style)}\nTime: Day\nInterior/Exterior: ${promptLine(editor.attributes.scene_type)}\nScene description: ${promptLine(body)}`;
}

const isTurboModel = name => /qwen.*2[._]1.*turbo/i.test(name || "");
const modelPath = entry => String(entry.local_path || "").replace(/\\/g, "/").replace(/^models\/[^/]+\//, "");

export function readState(raw) {
    if (typeof raw !== "string" || raw.length > 256 * 1024) throw new Error("Background Creator state is too large.");
    const value = JSON.parse(raw);
    if (!value || Array.isArray(value) || ![1, 2].includes(value.schema_version)) throw new Error("Unsupported Background Creator workflow state.");
    if (value.background_name != null && (typeof value.background_name !== "string" || [...value.background_name].length > 96)) throw new Error("Invalid saved background name.");
    for (const key of ["scene_id", "background_id", "selected_image_id", "model_id", "selected_cg_id"]) {
        if (value[key] != null && (typeof value[key] !== "string" || (value[key] && !/^[a-f0-9]{32}$/.test(value[key])))) throw new Error("Invalid background asset reference.");
    }
    if (value.camera != null) {
        if (!["position", "target", "up"].every(key => Array.isArray(value.camera[key]) && value.camera[key].length === 3
            && value.camera[key].every(v => typeof v === "number" && Number.isFinite(v) && Math.abs(v) <= 1e6))
            || typeof value.camera.fov !== "number" || !Number.isFinite(value.camera.fov) || value.camera.fov < 5 || value.camera.fov > 120) throw new Error("Invalid saved background camera.");
    }
    if (value.editor != null && (typeof value.editor !== "object" || Array.isArray(value.editor))) throw new Error("Invalid background editor state.");
    if (value.camera_view_version != null && value.camera_view_version !== 1) throw new Error("Unsupported background camera view version.");
    if (value.frame != null && (![value.frame.width, value.frame.height].every(n => Number.isInteger(n) && n >= 64 && n <= 4096 && n % 16 === 0) || value.frame.width * value.frame.height > 16e6)) throw new Error("Invalid saved camera frame dimensions.");
    return { schema_version: 2, background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", step: "background",
        ...value, editor: editorDefaults(value.editor), model_settings: { ...MODEL_DEFAULTS, ...value.model_settings } };
}

export function frameDimensions(aspect, megapixels) {
    const [a, b] = String(aspect).split(":").map(Number);
    const pixels = Number(megapixels) * 1e6;
    if (!(a > 0 && b > 0 && pixels > 0)) throw new Error("Invalid frame dimensions.");
    const width = Math.sqrt(pixels * a / b), height = Math.sqrt(pixels * b / a);
    const scale = Math.min(1, 4096 / Math.max(width, height), Math.sqrt(16e6 / pixels));
    const size = { width: Math.max(64, Math.round(width * scale / 16) * 16), height: Math.max(64, Math.round(height * scale / 16) * 16) };
    while (size.width * size.height > 16e6) size[size.width >= size.height ? "width" : "height"] -= 16;
    return size;
}

export class CameraHistory {
    constructor({ onRestore }) { this.onRestore = onRestore; this.clear(); }
    clear() { this.undoStack = []; this.redoStack = []; this.pending = null; }
    begin(label, camera) { this.pending ||= { before: clone(camera) }; }
    commit(camera) {
        if (!this.pending) return false;
        const entry = { before: this.pending.before, after: clone(camera) }; this.pending = null;
        if (JSON.stringify(entry.before) === JSON.stringify(entry.after)) return false;
        this.undoStack.push(entry); if (this.undoStack.length > 200) this.undoStack.shift();
        this.redoStack = []; return true;
    }
    undo() {
        const entry = this.undoStack.pop(); if (!entry) return false;
        this.redoStack.push(entry); this.onRestore(clone(entry.before)); return true;
    }
    redo() {
        const entry = this.redoStack.pop(); if (!entry) return false;
        this.undoStack.push(entry); this.onRestore(clone(entry.after)); return true;
    }
}

// Gaussian viewport; no Factory editor, routes, geometry or scene runtime.
export class BackgroundViewer {
    static async create(host, options) {
        const [THREE, spark] = await Promise.all([import("./vendor/spark/three.module.js"), import("./vendor/spark/spark.module.js")]);
        return new BackgroundViewer(host, options, THREE, spark);
    }
    constructor(host, options, THREE, { SparkRenderer, SplatMesh }) {
        this.host = host; this.options = options; this.THREE = THREE; this.SplatMesh = SplatMesh;
        this.meshes = []; this.epoch = 0; this.disposed = false; this.events = new AbortController();
        this.canvas = document.createElement("canvas"); this.canvas.tabIndex = 0;
        this.canvas.className = "bg-viewer-canvas"; host.appendChild(this.canvas);
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.scene = new THREE.Scene(); this.scene.background = new THREE.Color("#000000");
        this.camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.0001, 100000);
        this.camera.position.set(0, 0, 2); this.target = new THREE.Vector3();
        this.spark = new SparkRenderer({ renderer: this.renderer, maxStdDev: Math.sqrt(5), minSortIntervalMs: 32, lodRenderScale: 1 });
        this.spark.onDirty = () => this.invalidate(); this.scene.add(this.spark);
        this.frame = { width: 1600, height: 900 }; this.controls = { enableDamping: false };
        this.listen("pointerdown", event => {
            if (![0, 1, 2].includes(event.button)) return;
            event.preventDefault(); event.stopPropagation(); this.canvas.focus({ preventScroll: true });
            this.canvas.setPointerCapture(event.pointerId);
            this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, pan: event.button !== 0 || event.shiftKey };
            this.beginInteraction();
        });
        this.listen("pointermove", event => {
            if (!this.drag || event.pointerId !== this.drag.id) return;
            const dx = event.clientX - this.drag.x, dy = event.clientY - this.drag.y;
            this.drag.x = event.clientX; this.drag.y = event.clientY;
            this.moveCamera(dx, dy, this.drag.pan);
        });
        for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) this.listen(type, () => { this.drag = null; this.endInteraction(); });
        this.listen("wheel", event => {
            event.preventDefault(); event.stopPropagation();
            this.beginInteraction();
            const direction = this.target.clone().sub(this.camera.position).normalize();
            const distance = Math.max(0.001, this.target.distanceTo(this.camera.position));
            const delta = direction.multiplyScalar(-Math.sign(event.deltaY) * distance * 0.1);
            this.camera.position.add(delta); this.target.add(delta); this.changed(); this.endInteraction();
        }, { passive: false });
        this.listen("contextmenu", event => event.preventDefault());
        document.addEventListener("visibilitychange", () => { if (!document.hidden) this.invalidate(); }, { signal: this.events.signal });
        this.resize();
    }
    listen(type, callback, options = {}) { this.canvas.addEventListener(type, callback, { ...options, signal: this.events.signal }); }
    beginInteraction() {
        clearTimeout(this.qualityTimer);
        if (!this.interactive) { this.interactive = true; this.resize(); }
    }
    endInteraction() {
        clearTimeout(this.qualityTimer);
        this.qualityTimer = setTimeout(() => {
            this.qualityTimer = 0;
            if (this.disposed || this.drag) return;
            this.interactive = false; this.resize();
        }, 180);
    }
    moveCamera(dx, dy, pan) {
        const T = this.THREE, position = this.camera.position, height = Math.max(1, this.canvas.getBoundingClientRect().height);
        const distance = Math.max(0.001, position.distanceTo(this.target));
        const forward = this.target.clone().sub(position).normalize(), up = new T.Vector3(0, 1, 0);
        const right = new T.Vector3().crossVectors(forward, up).normalize();
        if (pan) {
            const gain = 2 * distance * Math.tan(this.camera.fov * Math.PI / 360) / height;
            const vertical = new T.Vector3().crossVectors(right, forward).normalize();
            const delta = right.multiplyScalar(-dx * gain).addScaledVector(vertical, dy * gain);
            position.add(delta); this.target.add(delta);
        } else {
            const gain = 2 * Math.tan(this.camera.fov * Math.PI / 360) / height * 0.7;
            forward.applyAxisAngle(up, -dx * gain);
            right.crossVectors(forward, up).normalize();
            const pitch = Math.asin(T.MathUtils.clamp(forward.dot(up), -1, 1));
            const next = T.MathUtils.clamp(pitch - dy * gain, -Math.PI * 89 / 180, Math.PI * 89 / 180);
            forward.applyAxisAngle(right, next - pitch).normalize();
            this.target.copy(position).addScaledVector(forward, distance);
        }
        this.changed();
    }
    changed(emit = true) {
        const distance = Math.max(0.001, this.camera.position.distanceTo(this.target));
        this.camera.near = Math.max(0.00001, distance / 100000); this.camera.far = Math.max(100000, distance * 1000);
        this.camera.lookAt(this.target); this.camera.updateProjectionMatrix(); this.invalidate();
        if (emit) this.options.onStateChange?.({ camera: this.getCameraState() });
    }
    getCameraState() { return { position: this.camera.position.toArray(), target: this.target.toArray(), up: this.camera.up.toArray(), fov: this.camera.fov }; }
    setCameraState(camera, { emit = false } = {}) {
        this.camera.position.fromArray(camera.position); this.target.fromArray(camera.target); this.camera.up.fromArray(camera.up).normalize();
        this.camera.fov = camera.fov; this.changed(emit);
    }
    getCaptureSettings() { return { ...this.frame }; }
    setCaptureSettings(size) { this.frame = { width: size.width, height: size.height }; this.resize(); }
    setLighting(value) { this.scene.background.set(value.background || "#000000"); this.invalidate(); }
    select() { /* Background models have no editable object selection. */ }
    async setScene(scene) {
        const epoch = ++this.epoch, next = [], failures = [];
        try {
            for (const item of scene.objects) {
                const response = await vnccsApi.fetchApi(item.urls.splat, { signal: this.events.signal });
                if (!response.ok) throw new Error(`Background model request failed: HTTP ${response.status}`);
                const bytes = await response.arrayBuffer();
                if (!bytes.byteLength || bytes.byteLength % 32 || bytes.byteLength > 128 * 1024 * 1024) throw new Error("Invalid background SPLAT size");
                const view = new DataView(bytes);
                for (let offset = 0; offset < bytes.byteLength; offset += 32) {
                    for (let axis = 0; axis < 6; axis++) {
                        const value = view.getFloat32(offset + axis * 4, true);
                        if (!Number.isFinite(value) || (axis >= 3 && value <= 0)) throw new Error("Invalid background Gaussian data");
                    }
                    if (offset && offset % (32 * 16384) === 0) await new Promise(resolve => requestAnimationFrame(resolve));
                    if (this.disposed || epoch !== this.epoch) throw new Error("Background model load was superseded");
                }
                const mesh = new this.SplatMesh({ fileBytes: bytes, fileType: "splat", fileName: `${item.object_id}.splat`, lod: false, editable: false, raycastable: false });
                next.push(mesh);
                mesh.setRotationFromMatrix(new this.THREE.Matrix4().set(0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1));
                const group = new this.THREE.Group(); group.add(mesh); mesh.backgroundGroup = group;
                const transform = item.transform || {};
                group.position.fromArray(transform.position || [0, 0, 0]);
                group.rotation.set(...(transform.rotation || [0, 0, 0]).map(value => value * Math.PI / 180));
                group.scale.setScalar(transform.scale ?? 1);
                this.scene.add(group); group.visible = false;
                await mesh.initialized;
                if (this.disposed || epoch !== this.epoch) throw new Error("Background model load was superseded");
                if (!Number.isFinite(mesh.numSplats) || mesh.numSplats < 1) throw new Error("Background model contains no Gaussians");
            }
            for (const mesh of this.meshes) { this.scene.remove(mesh.backgroundGroup); mesh.dispose(); }
            this.meshes = next;
            for (const mesh of next) mesh.backgroundGroup.visible = true;
            this.invalidate();
        } catch (error) {
            for (const mesh of next) { this.scene.remove(mesh.backgroundGroup); mesh.dispose(); }
            failures.push(error.message);
            if (!this.disposed && epoch === this.epoch) this.options.onError?.(error);
        }
        return { failures };
    }
    resize() {
        if (this.disposed || this.capturing) return;
        const width = Math.max(1, this.host.clientWidth), height = Math.max(1, this.host.clientHeight);
        const scale = Math.min(width / this.frame.width, height / this.frame.height);
        const w = Math.max(1, Math.floor(this.frame.width * scale)), h = Math.max(1, Math.floor(this.frame.height * scale));
        const heavy = this.meshes.reduce((count, mesh) => count + mesh.numSplats, 0) >= 262145;
        const nativeRatio = Math.min(window.devicePixelRatio || 1, 1.5);
        const budget = this.interactive ? (heavy ? 900000 : 1250000) : 2200000;
        const ratio = Math.max(0.5, Math.min(this.interactive ? Math.min(nativeRatio, heavy ? 0.85 : 1) : nativeRatio, Math.sqrt(budget / (w * h))));
        const sizeChanged = w !== this.viewportWidth || h !== this.viewportHeight || ratio !== this.pixelRatio;
        if (sizeChanged) {
            this.canvas.style.width = `${w}px`; this.canvas.style.height = `${h}px`;
            this.renderer.setDrawingBufferSize(w, h, ratio);
            this.viewportWidth = w; this.viewportHeight = h; this.pixelRatio = ratio;
        }
        const aspect = this.frame.width / this.frame.height, projectionChanged = this.camera.aspect !== aspect;
        if (projectionChanged) { this.camera.aspect = aspect; this.camera.updateProjectionMatrix(); }
        if (sizeChanged || projectionChanged) this.invalidate();
    }
    invalidate() {
        if (this.disposed || this.capturing || this.animation) return;
        this.animation = requestAnimationFrame(time => {
            this.animation = 0;
            if (this.disposed || this.capturing || document.hidden || !this.host.isConnected || !this.host.getClientRects().length) return;
            if (this.interactive && this.lastRenderTime && time - this.lastRenderTime < 1000 / 30) { this.invalidate(); return; }
            try { this.draw(); this.lastRenderTime = time; }
            catch (error) { this.options.onError?.(error); }
        });
    }
    draw() {
        if (this.disposed || this.capturing) return;
        // Spark updates and sorts in the background through its render hook.
        this.renderer.render(this.scene, this.camera);
    }
    async waitForSort() {
        const deadline = performance.now() + 15000;
        while (this.spark.sorting || this.spark.sortDirty) {
            if (this.disposed) throw new Error("Background viewer was removed");
            if (performance.now() >= deadline) throw new Error("Background model sorting timed out");
            await new Promise(resolve => setTimeout(resolve, 16));
        }
    }
    async capturePreview({ width, height, cameraState: state } = {}) {
        if (!Number.isInteger(width) || !Number.isInteger(height) || width < 64 || height < 64 || width > 4096 || height > 4096 || width * height > 16e6) throw new Error("Invalid capture dimensions");
        const previous = this.getCameraState(), target = document.createElement("canvas");
        this.capturing = true;
        try {
            await this.waitForSort();
            if (this.disposed) throw new Error("Background viewer was removed");
            this.setCameraState(state || previous); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
            this.renderer.setDrawingBufferSize(width, height, 1); this.spark.setDirty?.();
            await this.spark.update({ scene: this.scene, camera: this.camera });
            await this.waitForSort();
            if (this.disposed) throw new Error("Background viewer was removed");
            this.renderer.render(this.scene, this.camera);
            target.width = width; target.height = height;
            const context = target.getContext("2d", { alpha: false });
            if (!context) throw new Error("Could not create the background capture canvas");
            context.drawImage(this.canvas, 0, 0, width, height);
        } finally {
            if (!this.disposed) { this.setCameraState(previous); this.capturing = false; this.viewportWidth = 0; this.resize(); }
        }
        return await new Promise((resolve, reject) => target.toBlob(blob => blob ? resolve(blob) : reject(new Error("Background capture failed")), "image/png"));
    }
    dispose() {
        if (this.disposed) return;
        this.disposed = true; ++this.epoch; this.events.abort(); cancelAnimationFrame(this.animation); clearTimeout(this.qualityTimer);
        for (const mesh of this.meshes) mesh.dispose();
        this.spark.onDirty = null; this.spark.dispose?.(); this.renderer.dispose(); this.canvas.remove();
    }
}

// Workspace widget.


const BASE = "/vnccs/background-creator";
const ASPECTS = ["16:9", "9:16", "4:3", "3:4", "1:1", "21:9", "3:2", "2:3"];
const VARIANT_FIELDS = [
    ["time", "Time", ["Morning", "Day", "Evening", "Night", "Dawn", "Sunset", "Midnight"]],
    ["weather", "Weather", ["Clear sky", "Cloudy", "Overcast", "Light rain", "Heavy rain", "Thunderstorm", "Fog", "Snow", "Blizzard"]],
    ["additional_details", "Additional Details", ["Soft natural light", "Warm interior lighting", "Lights switched off", "Wet reflective surfaces", "Snow-covered ground"]],
];
const options = values => values.map(value => `<option>${value}</option>`).join("");
const selectValue = (select, value) => {
    if (value && ![...select.options].some(option => option.value === value)) {
        const option = document.createElement("option"); option.value = value; option.textContent = value; select.appendChild(option);
    }
    select.value = value ?? "";
};
const input = (label, path, type = "text", extra = "") => `<label class="bg-field"><span>${label}</span><input type="${type}" data-setting="${path}" ${extra}></label>`;
const compactField = (label, control) => `<label class="vnccs-common-creator-gen-param-field"><span class="vnccs-common-creator-label">${label}</span>${control}</label>`;
const validNumber = control => control.value.trim() !== "" && Number.isFinite(Number(control.value)) && control.validity?.valid !== false;
const toggleCard = (label, path) => `<label class="vnccs-common-creator-model-card is-installed" data-setting-card="${path}"><span class="vnccs-common-creator-model-card-top"><span class="vnccs-common-creator-model-card-badge ok"></span><span class="vnccs-common-creator-model-card-name">${label}</span><span class="vnccs-common-creator-model-card-status" data-toggle-status></span><span class="vnccs-common-creator-toggle vnccs-common-creator-model-card-toggle"><input type="checkbox" data-setting="${path}" aria-label="${label}"><span class="vnccs-common-creator-toggle-track"></span><span class="vnccs-common-creator-toggle-thumb"></span></span></span></label>`;
const latestModel = (document, imageId) => (document.models || [document.model]).filter(model => model?.source_image_id === imageId).at(-1);
const imageCGs = (document, imageId, modelId) => (document?.cgs || []).filter(cg => cg.source_image_id
    ? cg.source_image_id === imageId : document.images.length === 1 || cg.model_id === modelId);

export class BackgroundCreatorWidget {
    constructor(node, api) {
        injectStyles(STYLE, "vnccs-background-creator");
        this.node = node; this.api = api;
        this.state = null; this.document = null; this.scene = null; this.viewer = null;
        this.epoch = 0; this.restoring = true; this.busy = false; this.disposed = false;
        this.events = new AbortController(); this.operations = Promise.resolve(); this.saveSerial = Promise.resolve(); this.dialogs = new Set();
        this.history = new CameraHistory({ onRestore: camera => this.viewer.setCameraState(camera, { emit: true }) });
        this.container = document.createElement("div");
        this.container.className = "vnccs-background-creator";
        this.container.innerHTML = `
            <header class="bg-header"><strong>VNCCS Background Creator</strong><nav aria-label="Workspace stage"><button data-step="background">01 &nbsp; Background</button><button data-step="model">02 &nbsp; Model</button></nav></header>
            <section class="bg-step bg-background" data-panel="background">
                <section class="bg-panel bg-source"><div class="vnccs-common-creator-section-title">Background select</div><select data-library aria-label="Saved background"><option value="">Loading backgrounds…</option></select>
                    <div class="bg-row bg-actions"><button class="primary" data-action="background">Generate background</button><button class="success" data-new>New</button><button class="danger" data-delete>Delete</button></div>
                    <div class="bg-preview vnccs-common-creator-preview-container"><img data-preview alt="Selected background" hidden><span class="vnccs-common-creator-placeholder" data-empty><span class="vnccs-common-creator-placeholder-title">Restoring background…</span><span class="vnccs-common-creator-placeholder-hint">Loading saved workflow settings.</span></span><button class="bg-enlarge" data-enlarge aria-label="Open full-size background preview" hidden></button><div class="vnccs-common-creator-generation-layer" data-generation-layer hidden aria-hidden="true"><img class="vnccs-common-creator-generation-img vnccs-common-creator-generation-base" alt=""><img class="vnccs-common-creator-generation-img vnccs-common-creator-generation-new" alt=""><span class="vnccs-common-creator-generation-scan"></span></div></div>
                    <button data-upload>Upload image</button>
                    <div class="bg-variation-head"><span class="vnccs-common-creator-section-title">Variants</span><button type="button" data-add-variation>Add variant</button></div>
                    <div class="bg-variation-strip"><button type="button" class="vnccs-common-creator-sprite-nav-btn bg-variation-scroll" data-scroll-variation="-1" aria-label="Scroll variants left"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg></button><div class="bg-variation-cards" data-variation-cards data-sakura-scroll="background-variants"></div><button type="button" class="vnccs-common-creator-sprite-nav-btn bg-variation-scroll" data-scroll-variation="1" aria-label="Scroll variants right"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 6 6 6-6 6"/></svg></button></div>
                <div class="bg-variation-editor" data-variation-editor hidden><div class="bg-variation-editor-head"><label><span class="vnccs-common-creator-label">Variant name</span><input class="vnccs-common-creator-input" data-variation-name aria-label="Variant name" maxlength="96"></label><button type="button" class="vnccs-common-creator-model-card-download" data-close-variation>Close</button></div><div class="bg-variation-fields" data-variation-fields></div></div>
                </section>
                <section class="bg-panel bg-attributes"><div class="vnccs-common-creator-section-title">Scene attributes</div><button class="primary bg-wide" data-wizard>Background wizard</button>
                    ${input("Style", "style", "text", 'placeholder="Anime, digital art, cel shaded"')}
                    <div class="bg-field"><span>Interior/Exterior</span><div class="bg-segments" data-trait="scene_type"><button data-value="Interior">Interior</button><button data-value="Exterior">Exterior</button></div></div>
                    <label class="bg-field bg-description"><span>Scene description</span><textarea data-setting="scene_description" rows="12" placeholder="Describe the room or landscape, composition, objects and lighting."></textarea></label>
                </section>
                <section class="bg-panel bg-generation"><div class="vnccs-common-creator-section-title">Generation</div>
                    <div class="vnccs-common-creator-tab-row" style="--tabs:2" data-model-types role="group" aria-label="Generation model type"><button class="vnccs-common-creator-tab" data-model-type="qi2">Qwen Image 2.1</button><button class="vnccs-common-creator-tab" data-model-type="moss">Moss</button></div>
                    <div class="vnccs-common-creator-subsection"><div class="vnccs-common-creator-model-card-list" data-model-picker></div>
                        ${toggleCard("Prompt Enhancement", "generation.prompt_enhance")}
                    </div>
                    <label class="bg-field"><span>Aspect ratio</span><select data-setting="aspect">${options(ASPECTS)}</select></label>
                    <label class="vnccs-common-creator-field vnccs-common-creator-resolution-field"><span class="vnccs-common-creator-resolution-head"><span class="vnccs-common-creator-label">Resolution scale</span><output class="vnccs-common-creator-resolution-value" data-resolution>2.00 MP</output></span><input class="vnccs-common-creator-resolution-slider" type="range" min="0.25" max="8" step="0.25" data-setting="generation.resolution_mp" aria-label="Resolution scale in megapixels"></label>
                    <div class="vnccs-common-creator-gen-param-grid">
                        ${compactField("Steps", '<input class="vnccs-common-creator-gen-param-input" type="number" min="1" max="150" data-setting="generation.steps">')}
                        ${compactField("Sampler", '<select class="vnccs-common-creator-gen-param-select" data-setting="generation.sampler"><option>euler</option></select>')}
                        ${compactField("CFG", '<input class="vnccs-common-creator-gen-param-input" type="number" min="0" max="30" step="0.1" data-setting="generation.cfg">')}
                        ${compactField("Scheduler", '<select class="vnccs-common-creator-gen-param-select" data-setting="generation.scheduler"><option>simple</option></select>')}
                    </div>
                    <div class="vnccs-common-creator-field"><span class="vnccs-common-creator-label">Seed</span><div class="vnccs-common-creator-seed-row"><input aria-label="Generation seed" class="vnccs-common-creator-gen-param-input" type="number" min="0" max="9007199254740991" data-setting="generation.seed"><button type="button" class="vnccs-common-creator-seed-dice-btn" data-random="generation.seed" aria-label="Randomize generation seed"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3.5" stroke="currentColor" stroke-width="2"/><circle cx="8.5" cy="8.5" r="1.4" fill="currentColor"/><circle cx="15.5" cy="8.5" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="8.5" cy="15.5" r="1.4" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.4" fill="currentColor"/></svg></button></div></div>
                    <div class="vnccs-common-creator-qi2-cache" style="display:grid" data-qwen-only><div class="vnccs-common-creator-qi2-cache-title">Qwen Image 2.1 Cache</div>
                        ${compactField("Device", `<select class="vnccs-common-creator-gen-param-select" data-setting="generation.cache_device">${options(["gpu", "auto", "cpu", "off"])}</select>`)}
                        ${compactField("Dtype", `<select class="vnccs-common-creator-gen-param-select" data-setting="generation.cache_dtype">${options(["int8", "default", "int4"])}</select>`)}
                    </div>
                    <div class="vnccs-common-creator-subsection"><div class="vnccs-common-creator-section-title">LoRA Stack</div><div class="vnccs-common-creator-subsection" data-qwen-only><span class="vnccs-common-creator-label">Turbo LoRA</span>${toggleCard("Viggle Turbo · 6 steps", "generation.turbo")}</div>
                        <div class="vnccs-common-creator-lora-stack">${[0, 1, 2, 3, 4].map(index => `<div class="vnccs-common-creator-lora-item is-empty"><div class="vnccs-common-creator-lora-row"><select class="vnccs-common-creator-select" style="flex:2" data-lora="${index}" aria-label="LoRA ${index + 1}"><option value="">None</option></select><input class="vnccs-common-creator-input" style="flex:1" type="number" min="-10" max="10" step="0.05" value="1" data-lora-strength="${index}" aria-label="LoRA ${index + 1} strength"></div></div>`).join("")}</div>
                    </div>
                </section>
            </section>
            <section class="bg-step bg-model" data-panel="model" hidden>
                <aside class="bg-panel bg-model-source"><div class="vnccs-common-creator-section-title">Source background</div><select data-library aria-label="Model background"></select><img data-model-source alt="Model source background" hidden><span class="bg-hint" data-model-empty>Create or upload a background in the Background stage.</span><span data-model-dimensions></span><div class="vnccs-common-creator-section-title">Model generation</div><div class="bg-model-card"><div class="vnccs-common-creator-model-card-top"><span class="vnccs-common-creator-model-card-badge" data-tripo-badge></span><span class="vnccs-common-creator-model-card-name">TripoSplat</span></div><span class="vnccs-common-creator-model-card-desc" data-triposplat-assets>Single-image Gaussian splat</span></div>
                    <label class="bg-field"><span>Gaussians</span><select data-model-setting="num_gaussians">${[32768, 65536, 131072, 262144, 524288, 1048576].map(count => `<option value="${count}">${Math.round(count / 1024)}K</option>`).join("")}</select></label>
                    <div class="bg-row"><label class="bg-field"><span>Steps</span><input type="number" min="1" max="100" data-model-setting="steps"></label><label class="bg-field"><span>Guidance</span><input type="number" min="1" max="20" step="0.1" data-model-setting="guidance_scale"></label></div>
                    <label class="bg-field"><span>Seed</span><input type="number" min="0" max="2147483647" data-model-setting="seed"></label><span class="vnccs-common-creator-model-card-desc">Full background preserved</span><button class="primary bg-wide" data-action="model">Create model</button><div data-model-status>Model not created</div>
                </aside>
                <section class="bg-panel bg-viewport-panel"><div class="vnccs-common-creator-section-title">Model viewport <span class="bg-badge">Raw splat</span></div><div class="bg-camera-toolbar"><label>FOV <input type="range" min="5" max="120" step="0.1" value="42" data-fov aria-label="Camera field of view"><input type="number" min="5" max="120" step="0.1" value="42" data-fov-number aria-label="Exact camera field of view"></label><button class="primary" data-save-view>Save view</button><button data-recenter>Recenter</button></div><div class="bg-viewport"><div data-viewer></div><div class="bg-viewer-empty" data-viewer-empty>Create the model to compose camera views.</div></div><div class="bg-hint">Drag to look · Middle-drag to pan · Scroll to move · Ctrl Z to undo</div></section>
                <aside class="bg-panel bg-cg-panel"><div class="vnccs-common-creator-section-title">Generated CG <span class="bg-badge" data-cg-count>0</span></div><div class="bg-cg-list" data-cgs hidden></div><p data-cg-empty>Save a camera view to generate its finished CG.</p><label class="vnccs-common-creator-textarea-wrapper"><span class="vnccs-common-creator-textarea-label">CG refinement instruction</span><textarea data-setting="cg_prompt" rows="5"></textarea></label></aside>
            </section>
            <span class="bg-announcement" role="status" aria-live="polite" aria-atomic="true"></span>
            <input type="file" accept="image/png,image/jpeg,image/webp" data-file hidden>`;
        this.host = this.container.querySelector("[data-viewer]"); this.status = this.container.querySelector("[role=status]");
        this.fov = this.container.querySelector("[data-fov]"); this.fovNumber = this.container.querySelector("[data-fov-number]");
        this.container.classList.add("vnccs-common-ui-host", "vnccs-common-creator-container");
        this.customSelects = createSakuraInteractions(this.container, node);
        this.ccState = { config: null, downloadStatus: {} };
        this.cc = createControlCenterClient("MIUProject/VNCCS_v3.0", config => config.models?.some(entry => String(entry.kind).toLowerCase() === "qi2"), this.ccState,
            () => { if (!this.disposed && this.state) { this.renderModelPicker(); this.updateControls(); } }, node);
        this.decorateControls();
        this.container._node = node;
        for (const region of this.container.querySelectorAll(".bg-background,.bg-header,.bg-model-source,.bg-cg-panel")) enableMiddleMouseCanvasPan(region, node);
        this.tooltipCleanup = attachHelpTooltips(this.container);
        for (const region of this.container.querySelectorAll(".bg-step,.bg-panel,.bg-cg-list")) region.dataset.sakuraScroll = region.className;
        this.on(this.container, "click", event => this.click(event));
        this.on(this.container.querySelector("[data-variation-cards]"), "scroll", () => this.updateVariantScroll(), { passive: true });
        for (const image of this.container.querySelectorAll("[data-preview],[data-model-source]")) this.on(image, "error", () => {
            image.hidden = true; delete image.dataset.sourceURL;
            if (image.hasAttribute("data-preview")) {
                const empty = this.container.querySelector("[data-empty]"); empty.hidden = false;
                empty.querySelector(".vnccs-common-creator-placeholder-title").textContent = "Preview unavailable";
                empty.querySelector(".vnccs-common-creator-placeholder-hint").textContent = "Select the background again to retry. Saved data is retained.";
            } else {
                const empty = this.container.querySelector("[data-model-empty]"); empty.hidden = false; empty.textContent = "Source preview unavailable. Select the background again to retry.";
            }
        });
        this.on(this.container, "input", event => this.input(event));
        this.on(this.container, "change", event => this.change(event));
        for (const type of ["change", "blur", "keyup", "pointerup", "pointercancel"]) this.on(this.container, type, () => this.commitGesture());
        this.on(this.host, "pointerdown", () => this.beginGesture(), { capture: true });
        for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) this.on(this.host, type, () => this.commitGesture());
        this.on(this.host, "wheel", () => { this.beginGesture(); clearTimeout(this.wheelTimer); this.wheelTimer = setTimeout(() => this.commitGesture(), 180); }, { capture: true, passive: true });
        this.on(this.container, "keydown", event => {
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.target.closest("input,textarea")) {
                event.preventDefault(); if (this.canEditCamera()) this.history[event.shiftKey ? "redo" : "undo"]();
            }
        });
        this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(this.container);
        this.on(window, "beforeunload", event => { if (this.unsaved) { event.preventDefault(); event.returnValue = ""; } });
        this.on(this.api, "executing", event => {
            const id = event.detail?.node ?? event.detail;
            if (String(id) !== String(this.node.id) || this.restoring || this.busy || this.state?.step !== "background") return;
            this.workflowRun = { epoch: this.epoch, backgroundId: this.state.background_id };
            clearTimeout(this.saveTimer); this.busy = true; this.activeAction = "background";
            const source = this.document?.images.find(image => image.image_id === this.state.selected_image_id);
            if (source?.settings?.source !== "upload") this.startPreviewGeneration();
            this.message("Preparing background output…"); this.updateControls();
        });
        this.on(this.api, "vnccs.background.stage", event => {
            const detail = event.detail;
            if (!this.workflowRun || String(detail?.node_id) !== String(this.node.id) || detail.background_id !== this.state?.background_id) return;
            this.message(detail.message || "Rendering background…");
        });
        this.on(this.api, "execution_error", event => {
            if (this.workflowRun && String(event.detail?.node_id ?? event.detail?.node) === String(this.node.id)) this.finishWorkflow(null, event.detail?.exception_message || "Background generation failed.");
        });
        this.on(this.api, "vnccs.background.set", event => {
            const detail = event.detail, epoch = this.epoch;
            if (!detail || !this.document || this.disposed || this.restoring || detail.background_id !== this.state?.background_id || detail.source_image_id !== this.state.selected_image_id) return;
            this.refreshGeneratedSet(detail).catch(error => { if (!this.disposed && epoch === this.epoch) this.message("Could not refresh variant previews: " + error.message, true); });
        });
        this.on(this.api, "execution_interrupted", () => {
            if (this.jobId) {
                const epoch = this.epoch;
                this.json(`/jobs/${this.jobId}/cancel`, {}).catch(error => { if (!this.disposed && epoch === this.epoch) this.message(error.message, true); });
            }
            if (this.workflowRun) this.finishWorkflow(null, "Background generation stopped.");
        });
        watchConnection(node, () => { if (!this.disposed && this.restoreFailed) return this.restore(); }, registerCleanup);
        this.updateControls();
    }

    on(target, type, callback, options = {}) { target.addEventListener(type, callback, { ...options, signal: this.events.signal }); }
    message(text, error = false) {
        if (this.disposed) return;
        this.status.textContent = text; this.status.classList.toggle("error", error);
        if (error && this.lastError !== text) showMessage(this.container, text, true);
        this.lastError = error ? text : "";
    }
    startPreviewGeneration() {
        this.stopPreviewGeneration();
        const layer = this.container.querySelector("[data-generation-layer]"), frame = layer.parentElement;
        const base = layer.querySelector(".vnccs-common-creator-generation-base"), next = layer.querySelector(".vnccs-common-creator-generation-new"), scan = layer.querySelector(".vnccs-common-creator-generation-scan");
        const image = this.container.querySelector("[data-preview]"), existing = image.hidden ? "" : image.src;
        this.previewGeneration = { epoch: this.epoch, layer, frame, base, next, scan, animations: [] };
        if (existing) base.src = existing; else base.removeAttribute("src");
        base.hidden = !existing; base.style.clipPath = "inset(0% 0 0 0)";
        next.removeAttribute("src"); next.style.clipPath = "inset(0 0 100% 0)";
        scan.style.transform = "translateY(-100%)";
        layer.classList.remove("is-revealing"); layer.hidden = false;
        frame.setAttribute("aria-busy", "true"); this.container.querySelector("[data-empty]").hidden = true;
    }
    stopPreviewGeneration() {
        const owner = this.previewGeneration;
        if (!owner) return;
        this.previewGeneration = null;
        for (const animation of owner.animations) animation.cancel();
        owner.layer.classList.remove("is-revealing"); owner.layer.hidden = true;
        owner.base.removeAttribute("src"); owner.next.removeAttribute("src"); owner.frame.removeAttribute("aria-busy");
        this.container.querySelector("[data-empty]").hidden = !this.container.querySelector("[data-preview]").hidden;
    }
    async revealGeneratedPreview(url) {
        const owner = this.previewGeneration;
        if (!owner || !url) return false;
        const current = () => !this.disposed && owner.epoch === this.epoch && this.previewGeneration === owner;
        owner.next.src = this.api.apiURL(url);
        try { await owner.next.decode?.(); } catch { return false; }
        if (!current()) return false;
        owner.layer.classList.add("is-revealing");
        if (!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches && owner.next.animate && owner.base.animate && owner.scan.animate) {
            const options = { duration: 1350, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" };
            owner.animations = [
                owner.next.animate([{ clipPath: "inset(0 0 100% 0)" }, { clipPath: "inset(0 0 0% 0)" }], options),
                owner.base.animate([{ clipPath: "inset(0% 0 0 0)" }, { clipPath: "inset(100% 0 0 0)" }], options),
                owner.scan.animate([{ transform: "translateY(-100%)", opacity: 0.65 }, { transform: "translateY(-50%)", opacity: 1, offset: 0.5 }, { transform: "translateY(0)", opacity: 0.65 }], options),
            ];
            await Promise.all(owner.animations.map(animation => animation.finished.catch(() => {})));
        }
        return current();
    }
    async request(path, options = {}) {
        const response = await this.api.fetchApi(path.startsWith("/vnccs/") ? path : BASE + path, { ...options, signal: this.events.signal });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Request failed: HTTP ${response.status}`);
        return result;
    }
    json(path, data, method = "POST") { return this.request(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }); }
    canEditCamera() { return !this.disposed && !this.restoring && !this.busy && !this.loadingModel && Boolean(this.viewer && this.scene && this.scene.model_id === this.state?.model_id); }
    beginGesture() { if (this.canEditCamera()) this.history.begin("Camera", this.viewer.getCameraState()); }
    commitGesture() { if (this.canEditCamera()) { this.sync(); this.history.commit(this.viewer.getCameraState()); } }

    updateControls() {
        const locked = this.busy || this.restoring || Boolean(this.document?._recovery || this.document?.archived);
        const hasSource = Boolean(this.document?.images.some(image => image.image_id === this.state?.selected_image_id));
        this.container.setAttribute("aria-busy", String(this.busy || this.restoring));
        for (const control of this.container.querySelectorAll("button,input,select,textarea")) control.disabled = !control.closest(".vnccs-common-message-overlay") && locked;
        for (const card of this.container.querySelectorAll("[data-model-toggle],[data-model-choice]")) {
            card.disabled = locked || (card.hasAttribute("data-model-choice") && !card.classList.contains("is-installed"));
            if (card.getAttribute("role") === "button") { card.setAttribute("aria-disabled", String(card.disabled)); card.tabIndex = card.disabled ? -1 : 0; }
        }
        for (const button of this.container.querySelectorAll("[data-download-model]")) button.disabled = locked || ["queued", "downloading"].includes(this.ccState.downloadStatus[`cc_models_${button.dataset.downloadModel}`]?.status);
        for (const control of this.container.querySelectorAll("[data-step],[data-library],[data-new],[data-enlarge]")) control.disabled = this.busy || this.restoring;
        this.container.querySelector("[data-delete]").disabled = locked || !this.document;
        this.container.querySelector('[data-action="model"]').disabled = locked || !hasSource;
        this.container.querySelector("[data-enlarge]").disabled = this.busy || this.restoring || !hasSource;
        for (const control of this.container.querySelectorAll("[data-close-preview]")) control.disabled = false;
        const turbo = this.container.querySelector('[data-setting="generation.turbo"]');
        turbo.disabled = locked || this.state?.editor.generation.model_type === "moss" || isTurboModel(this.state?.editor.generation.diffusion_model_name);
        for (const control of this.container.querySelectorAll("[data-fov],[data-fov-number],[data-save-view],[data-recenter]")) control.disabled = !this.canEditCamera();
        this.container.querySelector("[data-save-view]").disabled = !this.canEditCamera() || Boolean(this.document?._recovery || this.document?.archived);
        this.host.inert = !this.canEditCamera();
        this.container.querySelector('[data-action="background"]').textContent = this.workflowRun ? "Workflow busy..." : this.busy && this.activeAction === "background" ? "Generating..." : "Generate background";
        this.container.querySelector('[data-action="model"]').textContent = this.busy && this.activeAction === "model" ? "Creating model..." : "Create model";
        this.customSelects.syncSelects();
        this.updateVariantScroll();
    }

    invalidateRestore() { ++this.epoch; this.stopPreviewGeneration(); this.restoring = true; this.busy = false; this.activeAction = ""; this.workflowRun = null; this.unsaved = false; this.jobId = ""; clearTimeout(this.saveTimer); this.updateControls(); }
    restore() {
        const epoch = ++this.epoch; this.restoring = true; this.restoreFailed = false;
        this.operations = this.operations.catch(() => {}).then(async () => {
            if (this.disposed || epoch !== this.epoch) return;
            try {
                const widget = this.node.widgets.find(item => item.name === "background_data");
                const state = readState(widget.value), stored = JSON.parse(widget.value);
                this.state = state;
                if (this.document?.background_id !== state.background_id) { this.document = null; this.scene = null; }
                this.render();
                const [config, catalog] = await Promise.all([this.request("/config"), this.request("/backgrounds")]);
                if (this.disposed || epoch !== this.epoch) return;
                this.state = state; this.config = config; this.catalog = catalog.backgrounds;
                this.fillChoices();
                this.render();
                this.cc.fetchConfig().catch(error => { if (!this.disposed) this.message("Model catalog unavailable: " + error.message + ". Local model choices are retained.", true); });
                if (state.schema_version === 1 && state.scene_id) {
                    const document = await this.json("/legacy", { scene_id: state.scene_id, editor: state.editor });
                    if (epoch !== this.epoch || this.disposed) return;
                    this.state.schema_version = 2;
                    await this.acceptDocument(document, true, epoch);
                } else if (state.background_id) {
                    const document = await this.request(`/backgrounds/${state.background_id}`);
                    if (epoch !== this.epoch || this.disposed) return;
                    if (!stored.editor) state.editor = editorDefaults(document.editor);
                    if (!stored.model_settings) state.model_settings = { ...MODEL_DEFAULTS, ...document.model_settings };
                    await this.acceptDocument(document, false, epoch);
                }
                if (epoch !== this.epoch || this.disposed) return;
                this.restoring = false; this.render();
                this.message(this.document?._recovery ? "Recovered backup is read-only. The damaged original is preserved." : "Create or upload a background. Generation runs inside this workspace.", Boolean(this.document?._recovery));
                if (state.schema_version !== 1) this.sync();
            } catch (error) {
                if (epoch === this.epoch && !this.disposed) { this.restoreFailed = true; this.message(error.message + " Original workflow data is preserved.", true); }
            }
            this.updateControls();
        });
        return this.operations;
    }

    fillChoices() {
        const fill = (selector, choices) => {
            for (const select of this.container.querySelectorAll(selector)) {
                select.replaceChildren(...choices.map(value => { const option = document.createElement("option"); option.value = value; option.textContent = value || "None"; return option; }));
            }
        };
        fill('[data-setting="generation.sampler"]', this.config.samplers);
        fill('[data-setting="generation.scheduler"]', this.config.schedulers);
        fill("[data-lora]", ["", ...this.config.loras]);
        this.renderModelPicker();
        const missing = (this.config.triposplat_assets || []).filter(item => !item.installed);
        const tripo = this.container.querySelector("[data-triposplat-assets]");
        tripo.textContent = missing.length ? "Missing: " + missing.map(item => `${item.category}/${item.name}`).join(", ") : "Single-image Gaussian splat";
        tripo.classList.toggle("error", Boolean(missing.length));
        const tripoBadge = this.container.querySelector("[data-tripo-badge]");
        tripoBadge.classList.toggle("ok", !missing.length); tripoBadge.classList.toggle("missing", Boolean(missing.length));
    }

    renderModelPicker() {
        if (!this.state || !this.config) return;
        const container = this.container.querySelector("[data-model-picker]"), view = this.customSelects.captureView();
        const modelType = this.state.editor.generation.model_type, moss = modelType === "moss";
        const assets = this.config.model_types?.[modelType]?.assets || (moss ? [] : this.config.assets) || [];
        const local = new Set((this.config.diffusion_models || []).map(name => name.replace(/\\/g, "/")).filter(name => /moss[_-]?image/i.test(name) === moss));
        const resolve = name => {
            const matches = [...local].filter(path => path.split("/").at(-1).toLowerCase() === name.split("/").at(-1).toLowerCase());
            return local.has(name) || matches.length !== 1 ? name : matches[0];
        };
        const entries = (this.ccState.config?.models || []).filter(entry => String(entry.kind).toLowerCase() === modelType && String(entry.type).toLowerCase() === "unet");
        if (!entries.length) entries.push({ name: moss ? "MOSS-Image SFT INT8 ConvRot" : "Qwen Image 2.1 INT8 ConvRot",
            local_path: "models/diffusion_models/" + (moss ? MOSS_GENERATION : DEFAULT_EDITOR.generation).diffusion_model_name,
            status: assets.find(asset => asset.category === "diffusion_models")?.installed || (!moss && this.config.installed) ? "installed" : "missing",
            ...(moss ? { source: "local", description: "MOSS-Image SFT · Qwen3.5 9B · Flux2 VAE" } : {}) });
        this.modelEntries = entries.map(entry => ({ ...entry, local_path: "models/diffusion_models/" + resolve(modelPath(entry)), status: local.has(resolve(modelPath(entry))) ? "installed" : entry.status || "missing" }));
        const known = new Set(this.modelEntries.map(modelPath));
        for (const name of local) if (!known.has(name)) this.modelEntries.push({ name, local_path: `models/diffusion_models/${name}`, status: "installed", source: "local", description: "Local ComfyUI model." });
        const selected = resolve(this.state.editor.generation.diffusion_model_name);
        if (!this.modelEntries.some(entry => modelPath(entry) === selected)) this.modelEntries.push({ name: selected, local_path: `models/diffusion_models/${selected}`, status: "missing", source: "local", description: "Saved model is unavailable. Its selection is retained." });
        const picker = document.createElement("div"); picker.className = "vnccs-common-creator-model-picker"; picker.classList.toggle("is-open", Boolean(this.modelPickerOpen));
        const build = (entry, head = false) => {
            const status = this.ccState.downloadStatus[`cc_models_${entry.name}`]?.status || entry.status;
            const installed = status === "installed", pending = ["queued", "downloading"].includes(status);
            const card = document.createElement("div");
            if (installed || head) {
                card.setAttribute("role", "button"); card.tabIndex = 0;
                card.onkeydown = event => { if (event.target === card && ["Enter", " "].includes(event.key)) { event.preventDefault(); event.stopPropagation(); if (!event.repeat) this.click({ target: card }); } };
            }
            card.className = "vnccs-common-creator-model-card"; card.classList.toggle("is-picker-head", head); card.classList.toggle("is-installed", installed);
            card.classList.toggle("is-selected", head || modelPath(entry) === selected); card.classList.toggle("is-missing", !installed);
            card.setAttribute("aria-label", entry.name); card.setAttribute("aria-pressed", String(modelPath(entry) === selected));
            card.dataset.focusKey = `background-model-${head ? "head" : modelPath(entry)}`;
            if (head) { card.dataset.modelToggle = ""; card.setAttribute("aria-expanded", String(Boolean(this.modelPickerOpen))); }
            else { card.dataset.modelChoice = modelPath(entry); card.disabled = !installed; }
            const top = document.createElement("div"); top.className = "vnccs-common-creator-model-card-top";
            const badge = document.createElement("span"); badge.className = "vnccs-common-creator-model-card-badge " + (installed ? "ok" : pending ? "progress" : "missing");
            const name = document.createElement("span"); name.className = "vnccs-common-creator-model-card-name"; name.textContent = entry.name;
            const label = document.createElement("span"); label.className = "vnccs-common-creator-model-card-status " + (installed ? "ok" : pending ? "progress" : "missing");
            label.textContent = installed ? "Installed" : pending ? "Downloading" : status === "error" ? "Error" : "Missing";
            top.append(badge, name, label); card.appendChild(top);
            if (entry.description) { const description = document.createElement("span"); description.className = "vnccs-common-creator-model-card-desc"; description.textContent = entry.description; card.appendChild(description); }
            if (!installed && entry.source !== "local") {
                const actions = document.createElement("div"); actions.className = "vnccs-common-creator-model-card-actions";
                const download = document.createElement("button"); download.type = "button"; download.className = "vnccs-common-creator-model-card-download";
                download.textContent = "Download"; download.dataset.downloadModel = entry.name; download.disabled = pending;
                actions.appendChild(download); card.appendChild(actions);
            }
            return card;
        };
        picker.appendChild(build(this.modelEntries.find(entry => modelPath(entry) === selected), true));
        const menu = document.createElement("div"); menu.className = "vnccs-common-creator-model-picker-menu"; menu.hidden = !this.modelPickerOpen; menu.dataset.sakuraScroll = "background-model-picker";
        for (const [title, user] of [["VNCCS Models", false], ["User Models", true]]) {
            const groupEntries = this.modelEntries.filter(entry => (entry.source === "local") === user); if (!groupEntries.length) continue;
            const group = document.createElement("div"); group.className = "vnccs-common-creator-model-picker-group";
            const label = document.createElement("div"); label.className = "vnccs-common-creator-model-picker-group-title"; label.textContent = title; group.appendChild(label);
            for (const entry of groupEntries) group.appendChild(build(entry));
            menu.appendChild(group);
        }
        picker.appendChild(menu); container.replaceChildren(picker);
        const missing = assets.filter(asset => asset.category !== "diffusion_models" && !asset.installed
            && !this.ccState.config?.[asset.category === "text_encoders" ? "clip" : asset.category]?.some(entry => String(entry.kind).toLowerCase() === modelType && entry.status === "installed"));
        if (missing.length) { const hint = document.createElement("span"); hint.className = "vnccs-common-creator-model-card-desc error"; hint.textContent = "Missing: " + missing.map(asset => asset.name).join(", "); container.appendChild(hint); }
        this.customSelects.restoreView(view);
    }

    async downloadModel(name) {
        const entry = this.modelEntries.find(item => item.name === name);
        if (!entry || entry.source === "local") return;
        const bundle = [["models", entry], ...["clip", "vae"].map(category => [category, this.ccState.config?.[category]?.find(item => String(item.kind).toLowerCase() === this.state.editor.generation.model_type)])];
        for (const [category, asset] of bundle) {
            if (!asset || asset.status === "installed" || this.disposed) continue;
            const key = `cc_${category}_${asset.name}`;
            this.ccState.downloadStatus[key] = { status: "queued" }; this.renderModelPicker(); this.updateControls();
            try {
                await this.request("/vnccs/control_center/download", { method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                    body: JSON.stringify({ repo_id: "MIUProject/VNCCS_v3.0", category, name: asset.name }) });
            } catch (error) { this.ccState.downloadStatus[key] = { status: "error" }; this.renderModelPicker(); throw error; }
            this.cc.startPolling();
        }
    }

    async acceptDocument(document, adopt = true, epoch = this.epoch) {
        if (this.disposed || epoch !== this.epoch) return;
        this.document = document;
        this.catalog = (this.catalog || []).filter(item => item.background_id !== document.background_id);
        if (!document.archived) this.catalog.push(document);
        if (adopt) {
            const previousId = this.state.background_id;
            if (previousId !== document.background_id) this.activeVariationId = "";
            this.state.background_id = document.background_id; this.state.selected_image_id = document.selected_image_id;
            this.state.model_id = latestModel(document, document.selected_image_id)?.model_id || "";
            this.state.editor = editorDefaults(document.editor);
            this.state.model_settings = { ...MODEL_DEFAULTS, ...(document.model_settings || latestModel(document, document.selected_image_id)?.settings) };
            const cgs = imageCGs(document, this.state.selected_image_id, this.state.model_id);
            if (previousId !== document.background_id || !cgs.some(cg => cg.cg_id === this.state.selected_cg_id)) this.state.selected_cg_id = cgs.at(-1)?.cg_id || "";
            this.state.camera = null; this.state.frame = null;
        }
        if (this.state.background_id === document.background_id) this.state.background_name = document.name;
        if (this.state.model_id) await this.showModel(epoch);
        if (epoch !== this.epoch || this.disposed) return;
        this.render(); this.sync(); this.updateControls();
    }

    async showModel(epoch = this.epoch) {
        const modelId = this.state.model_id, backgroundId = this.state.background_id;
        const request = this.modelRequest = (this.modelRequest || 0) + 1;
        if (!modelId) return;
        const scene = await this.request(`/backgrounds/${backgroundId}/models/${modelId}`);
        if (request !== this.modelRequest || epoch !== this.epoch || modelId !== this.state.model_id || this.disposed) return;
        if (!this.viewer) {
            this.viewerPromise ||= BackgroundViewer.create(this.host, {
                cameraDragScale: 0.7,
                resolveAssetURL: path => this.api.apiURL(path), onSelectionChange: () => this.viewer.select("", { emit: false }),
                onStateChange: value => { if (this.canEditCamera()) { this.state.camera = value.camera; this.sync(); this.syncFov(value.camera); } },
                onError: error => this.message(String(error?.message || error), true),
            });
            const viewer = await this.viewerPromise;
            if (this.disposed) { viewer.dispose(); return; }
            this.viewer = viewer;
            if (epoch !== this.epoch) return;
            this.viewer.controls.enableDamping = false;
            this.viewer.canvas.setAttribute("aria-label", "Background camera. Drag to look, middle-drag to pan, scroll to move.");
        }
        this.loadingModel = true;
        try {
            if (this.scene?.model_id !== modelId) {
                const result = await this.viewer.setScene(scene);
                if (request !== this.modelRequest || epoch !== this.epoch || this.disposed) return;
                if (result.failures.length) throw new Error("The background model could not be loaded.");
                this.history.clear();
            }
            if (epoch !== this.epoch || this.disposed || modelId !== this.state.model_id) return;
            this.scene = scene;
            this.state.selected_image_id = scene.source_image_id || this.state.selected_image_id;
            const selected = this.document.cgs.find(cg => cg.cg_id === this.state.selected_cg_id && cg.model_id === modelId);
            const sourceCamera = scene.objects[0].source_camera || scene.camera;
            const oldSourceView = !this.state.camera_view_version && !selected && this.state.camera
                && ["position", "target", "up"].every(key => this.state.camera[key].every((value, index) => Math.abs(value - sourceCamera[key][index]) < 1e-7));
            const camera = oldSourceView ? sourceCameraView(sourceCamera) : this.state.camera || selected?.camera || sourceCameraView(sourceCamera);
            this.state.camera_view_version = 1;
            this.state.camera = clone(camera);
            this.viewer.setLighting({ preset: "off", background: "#000000" });
            const source = this.document.images.find(image => image.image_id === scene.source_image_id);
            const size = this.state.frame || (selected ? { width: selected.width, height: selected.height }
                : frameDimensions(source ? `${source.width}:${source.height}` : this.state.editor.aspect, this.state.editor.generation.resolution_mp));
            this.state.frame = clone(size);
            this.viewer.setCaptureSettings({ ...size, show_camera_frame: true }); this.viewer.setCameraState(camera); this.viewer.select("", { emit: false });
            this.syncFov(camera); this.container.querySelector("[data-viewer-empty]").hidden = true;
        } finally { if (request === this.modelRequest) this.loadingModel = false; }
        this.resize();
    }

    updateVariantScroll() {
        const list = this.container.querySelector("[data-variation-cards]");
        for (const button of this.container.querySelectorAll("[data-scroll-variation]")) {
            const atEdge = Number(button.dataset.scrollVariation) < 0 ? list.scrollLeft <= 1 : list.scrollLeft + list.clientWidth >= list.scrollWidth - 1;
            button.disabled = this.busy || this.restoring || atEdge;
        }
    }

    async refreshGeneratedSet(detail) {
        const epoch = this.epoch, request = this.setRequest = (this.setRequest || 0) + 1;
        await this.saveSerial.catch(() => {});
        if (this.disposed || this.restoring || epoch !== this.epoch) return;
        const document = await this.request(`/backgrounds/${detail.background_id}`);
        if (this.disposed || this.restoring || epoch !== this.epoch || request !== this.setRequest
            || detail.background_id !== this.state.background_id || detail.source_image_id !== this.state.selected_image_id
            || document.revision < this.document.revision) return;
        const unchanged = ["editor", "selected_image_id", "model_settings"].every(key => JSON.stringify(document[key]) === JSON.stringify(this.document[key]));
        this.document.generated_sets = document.generated_sets;
        // Only rebase pending edits when generation left the saved authored settings unchanged.
        if (unchanged) this.document.revision = document.revision;
        this.render();
        if (unchanged && this.unsaved) this.markEdited();
    }

    variantPreview(card) {
        for (const group of [...(this.document?.generated_sets || [])].reverse()) {
            if (group.source_image_id !== this.state.selected_image_id) continue;
            const index = group.variants.findIndex(saved => ["id", "time", "weather", "additional_details"].every(key => saved[key] === card[key]));
            if (index >= 0 && group.images[index + 1]?.url) return group.images[index + 1];
        }
        return null;
    }

    renderVariantEditor() {
        const panel = this.container.querySelector("[data-variation-editor]"), fields = panel.querySelector("[data-variation-fields]");
        const card = this.state.editor.variants.find(card => card.id === this.activeVariationId);
        panel.hidden = !card;
        if (card !== this.variantEditorCard) {
            fields.replaceChildren(); this.variantEditorCard = card;
            if (card) for (const [key, label] of VARIANT_FIELDS) {
                const control = createTraitInput({ prefix: "vnccs-common-creator", label, key, target: card, removable: true,
                    save: () => { if (this.variantEditorCard === card && !this.restoring && !this.disposed && !this.busy) this.markEdited(); },
                    choose: input => this.openVariantPresets(key, input, card),
                });
                control.input.dataset.variationField = key; control.input.maxLength = key === "time" ? 200 : key === "weather" ? 500 : 8000;
                fields.appendChild(control.element);
            }
        }
        const name = panel.querySelector("[data-variation-name]");
        if (card && document.activeElement !== name) name.value = card.name;
    }

    openVariantPresets(key, input, card) {
        const [, label, values] = VARIANT_FIELDS.find(field => field[0] === key), epoch = this.epoch;
        const items = values.map(value => ({ tag: value, label: value })), selected = presetSelection(input.value, [{ items }]);
        const close = () => { this.dialogs.delete(close); modal.overlay.remove(); };
        const modal = showModal(this.container, `Choose Presets: ${label}`, () => {
            const grid = document.createElement("div"); grid.className = "vnccs-common-creator-tag-grid";
            for (const item of items) {
                const chip = document.createElement("button"); chip.type = "button"; chip.className = "vnccs-common-creator-tag-chip"; chip.textContent = item.label;
                const update = () => { chip.classList.toggle("selected", selected.has(item)); chip.setAttribute("aria-pressed", String(selected.has(item))); };
                chip.onclick = () => { selected.toggle(item); update(); }; update(); grid.appendChild(chip);
            }
            return grid;
        }, [{ text: "Cancel", action: close }, { text: "Apply", class: "primary", action: () => {
            if (!this.disposed && !this.restoring && !this.busy && epoch === this.epoch && this.variantEditorCard === card
                && !this.document?._recovery && !this.document?.archived) {
                input.value = selected.value(); input.dispatchEvent(new Event("input"));
            }
            close();
        } }]);
        this.dialogs.add(close);
        const remove = modal.overlay.remove;
        modal.overlay.remove = () => { this.dialogs.delete(close); remove.apply(modal.overlay); };
    }

    render() {
        if (!this.state) return;
        const scroll = [...this.container.querySelectorAll(".bg-step,.bg-panel,.bg-cg-list")].map(element => [element, element.scrollTop, element.scrollLeft]);
        const view = this.customSelects.captureView();
        const editor = this.state.editor;
        const cardList = this.container.querySelector("[data-variation-cards]");
        const cardKey = JSON.stringify(editor.variants.map(card => [card.id, card.name]));
        if (cardList.dataset.cards !== cardKey) {
            cardList.replaceChildren(); cardList.dataset.cards = cardKey;
            for (const card of editor.variants) {
                const item = document.createElement("div"); item.className = "bg-variation-item";
                const name = document.createElement("span"); name.textContent = card.name;
                const preview = document.createElement("div"); preview.className = "bg-variation-preview";
                const button = document.createElement("button"); button.type = "button"; button.dataset.editVariation = card.id;
                button.setAttribute("aria-label", `Edit ${card.name} variant`);
                const image = document.createElement("img"); image.dataset.variationImage = card.id; image.hidden = true; image.decoding = "async";
                this.on(image, "error", () => { image.hidden = true; delete image.dataset.sourceURL; }); button.appendChild(image);
                const remove = document.createElement("button"); remove.type = "button"; remove.className = "bg-variation-delete";
                remove.dataset.deleteVariation = card.id; remove.textContent = "×";
                remove.setAttribute("aria-label", `Delete ${card.name} variant`);
                preview.append(button, remove); item.append(name, preview); cardList.appendChild(item);
            }
        }
        for (const button of cardList.querySelectorAll("[data-edit-variation]")) {
            button.setAttribute("aria-expanded", String(button.dataset.editVariation === this.activeVariationId));
            const card = editor.variants.find(card => card.id === button.dataset.editVariation), image = button.querySelector("[data-variation-image]"), preview = this.variantPreview(card);
            image.alt = `${card.name} variant`; image.hidden = !preview;
            if (preview && image.dataset.sourceURL !== preview.url) { image.src = this.api.apiURL(preview.url); image.dataset.sourceURL = preview.url; }
            if (!preview) { image.removeAttribute("src"); delete image.dataset.sourceURL; }
        }
        this.renderVariantEditor();
        const moss = editor.generation.model_type === "moss";
        this.container.querySelector("[data-model-types]").style.setProperty("--selected", moss ? 1 : 0);
        for (const button of this.container.querySelectorAll("[data-model-type]")) {
            const selected = button.dataset.modelType === editor.generation.model_type;
            button.classList.toggle("is-active", selected); button.setAttribute("aria-pressed", String(selected));
        }
        for (const section of this.container.querySelectorAll("[data-qwen-only]")) section.hidden = moss;
        const hasModel = Boolean(this.state.model_id && this.scene?.model_id === this.state.model_id);
        this.host.hidden = !hasModel; this.container.querySelector("[data-viewer-empty]").hidden = hasModel;
        for (const control of this.container.querySelectorAll("[data-setting]")) {
            const value = control.dataset.setting.split(".").reduce((current, key) => current?.[key], editor);
            if (control.type === "checkbox") control.checked = Boolean(value);
            else if (control.tagName === "SELECT") selectValue(control, value);
            else if (document.activeElement !== control) control.value = value ?? "";
        }
        for (const control of this.container.querySelectorAll("[data-model-setting]")) control.value = this.state.model_settings[control.dataset.modelSetting];
        for (const select of this.container.querySelectorAll("[data-library]")) {
            select.replaceChildren();
            const empty = document.createElement("option"); empty.value = ""; empty.textContent = "Select a background"; select.appendChild(empty);
            for (const item of this.catalog || []) { const option = document.createElement("option"); option.value = item.background_id; option.textContent = item.name; select.appendChild(option); }
            if (this.state.background_id && !(this.catalog || []).some(item => item.background_id === this.state.background_id)) {
                const selected = document.createElement("option"); selected.value = this.state.background_id;
                selected.textContent = this.state.background_name || "Loading saved background…"; select.appendChild(selected);
            }
            select.value = this.state.background_id;
        }
        for (const group of this.container.querySelectorAll("[data-trait]")) for (const button of group.querySelectorAll("button")) button.classList.toggle("active", editor.attributes[group.dataset.trait] === button.dataset.value);
        const source = this.document?.images.find(image => image.image_id === this.state.selected_image_id);
        const previewURL = source?.url || (this.restoring && this.state.background_id && this.state.selected_image_id
            ? `${BASE}/backgrounds/${this.state.background_id}/images/${this.state.selected_image_id}` : "");
        for (const image of this.container.querySelectorAll("[data-preview],[data-model-source],[data-large-preview]")) {
            image.hidden = !previewURL;
            if (previewURL && image.dataset.sourceURL !== previewURL) { image.src = this.api.apiURL(previewURL); image.dataset.sourceURL = previewURL; }
        }
        this.container.querySelector("[data-empty]").hidden = Boolean(previewURL);
        this.container.querySelector("[data-enlarge]").hidden = !source;
        this.container.querySelector("[data-model-empty]").hidden = Boolean(previewURL);
        if (!source) {
            this.container.querySelector(".vnccs-common-creator-placeholder-title").textContent = "Create a background";
            this.container.querySelector(".vnccs-common-creator-placeholder-hint").textContent = "Generate a preview or upload your image.";
            this.container.querySelector("[data-model-empty]").textContent = "Create or upload a background in the Background stage.";
        }
        this.container.querySelector("[data-model-dimensions]").textContent = source ? `${source.width} × ${source.height}` : "";
        this.container.querySelector("[data-model-status]").textContent = this.loadingModel ? "Loading model…" : hasModel ? "Model ready" : this.state.model_id ? "Model unavailable" : "Model not created";
        const gallery = this.container.querySelector("[data-cgs]"); gallery.replaceChildren();
        const cgs = imageCGs(this.document, this.state.selected_image_id, this.state.model_id);
        for (const cg of cgs) {
            const card = document.createElement("button"); card.className = "bg-cg-card"; card.classList.toggle("selected", cg.cg_id === this.state.selected_cg_id); card.dataset.cgId = cg.cg_id;
            card.setAttribute("aria-pressed", String(cg.cg_id === this.state.selected_cg_id));
            const title = document.createElement("strong"); title.textContent = cg.name; const image = document.createElement("img"); image.src = this.api.apiURL(cg.url); image.alt = cg.name; image.loading = "lazy"; image.decoding = "async";
            const meta = document.createElement("span"); meta.textContent = `${cg.width} × ${cg.height} · FOV ${Math.round(cg.camera.fov)}°`;
            card.append(title, image, meta); gallery.appendChild(card);
        }
        this.container.querySelector("[data-cg-count]").textContent = String(cgs.length);
        gallery.hidden = !cgs.length;
        this.container.querySelector("[data-cg-empty]").hidden = Boolean(cgs.length);
        for (const select of this.container.querySelectorAll("[data-lora]")) selectValue(select, editor.generation.lora_stack[Number(select.dataset.lora)]?.name || "");
        for (const number of this.container.querySelectorAll("[data-lora-strength]")) number.value = editor.generation.lora_stack[Number(number.dataset.loraStrength)]?.strength ?? 1;
        this.container.querySelector("[data-resolution]").textContent = `${Number(editor.generation.resolution_mp).toFixed(2)} MP`;
        this.syncPrompt(); this.renderModelPicker(); this.syncGenerationCards();
        this.setStep(this.state.step, false); this.decorateControls(); this.updateControls();
        for (const [element, top, left] of scroll) { element.scrollTop = top; element.scrollLeft = left; }
        this.customSelects.restoreView(view);
    }

    setStep(step, save = true) {
        const next = step === "model" ? "model" : "background";
        if (next !== this.state.step) this.customSelects._closeDropdown(false);
        this.state.step = next;
        for (const panel of this.container.querySelectorAll("[data-panel]")) panel.hidden = panel.dataset.panel !== this.state.step;
        for (const button of this.container.querySelectorAll("[data-step]")) button.classList.toggle("active", button.dataset.step === this.state.step);
        this.syncSegments(this.container.querySelector("nav"));
        if (save) this.sync(); this.resize(); if (this.state.step === "model") this.viewer?.invalidate();
    }
    syncFov(camera) { this.fov.value = String(camera.fov); this.fovNumber.value = String(Math.round(camera.fov * 10) / 10); }

    sync() {
        if (this.disposed || this.restoring || !this.state) return;
        if (this.viewer && this.scene?.model_id === this.state.model_id && !this.busy && !this.loadingModel) this.state.camera = this.viewer.getCameraState();
        const raw = JSON.stringify(this.state), widget = this.node.widgets.find(item => item.name === "background_data");
        if (widget && widget.value !== raw) { widget.value = raw; this.node.properties ??= {}; this.node.properties.backgroundCreator = raw; this.node.graph?.setDirtyCanvas?.(true, true); }
        return raw;
    }

    markEdited() {
        this.syncPrompt(); this.unsaved = true; this.sync();
        clearTimeout(this.saveTimer); this.saveTimer = setTimeout(() => this.persist().catch(error => { if (!this.workflowRun) this.message(error.message, true); }), 450);
    }

    syncPrompt() {
        const editor = this.state.editor;
        for (const control of this.container.querySelectorAll('[data-setting="style"],[data-setting="scene_description"]')) {
            if (document.activeElement !== control) control.value = editor[control.dataset.setting];
        }
        for (const group of this.container.querySelectorAll("[data-trait]")) {
            for (const button of group.querySelectorAll("button")) button.classList.toggle("active", editor.attributes[group.dataset.trait] === button.dataset.value);
            this.syncSegments(group);
        }
    }
    persist() {
        clearTimeout(this.saveTimer);
        const operation = this.saveSerial.catch(() => {}).then(async () => {
            if (!this.unsaved || !this.document || this.disposed || this.restoring) return;
            const id = this.document.background_id, epoch = this.epoch, editor = clone(this.state.editor), selectedImage = this.state.selected_image_id,
                modelSettings = clone(this.state.model_settings);
            const saved = await this.json(`/backgrounds/${id}`, { editor, model_settings: modelSettings, selected_image_id: selectedImage, revision: this.document.revision }, "PATCH");
            if (this.disposed || epoch !== this.epoch || this.document?.background_id !== id) return;
            this.document = saved;
            this.unsaved = JSON.stringify(editor) !== JSON.stringify(this.state.editor) || selectedImage !== this.state.selected_image_id
                || JSON.stringify(modelSettings) !== JSON.stringify(this.state.model_settings);
            if (this.unsaved) this.markEdited();
        });
        this.saveSerial = operation; return operation;
    }

    finishWorkflow(output, failure = "") {
        const preview = output?.background_preview?.[0], owner = this.workflowRun || { epoch: this.epoch, backgroundId: preview?.background_id };
        if (this.disposed || this.restoring || !owner.backgroundId || owner.backgroundId !== this.state?.background_id) return;
        this.workflowCompletion = (async () => {
            try {
                await this.saveSerial.catch(() => {});
                const document = await this.request(`/backgrounds/${owner.backgroundId}`);
                if (this.disposed || owner.epoch !== this.epoch || owner.backgroundId !== this.state.background_id) return;
                const current = preview && JSON.stringify(this.state.editor) === JSON.stringify(preview.editor)
                    && this.state.selected_image_id === preview.request_image_id
                    && JSON.stringify(this.state.model_settings) === JSON.stringify(preview.model_settings);
                this.stopPreviewGeneration();
                if (current && this.state.selected_image_id !== preview.image_id) {
                    Object.assign(this.state, { selected_image_id: preview.image_id,
                        model_id: latestModel(document, preview.image_id)?.model_id || "", selected_cg_id: "", camera: null, frame: null });
                }
                await this.acceptDocument(document, false, owner.epoch);
                if (this.disposed || owner.epoch !== this.epoch) return;
                this.unsaved = preview ? !current : this.unsaved;
                this.message(failure || (preview?.source === "upload" ? "Uploaded background passed through." : preview?.source === "cache" ? "Background preview reused from cache." : "Background generated."), Boolean(failure));
            } catch (error) { if (!this.disposed && owner.epoch === this.epoch) this.message(error.message, true); }
            finally {
                if (!this.disposed && owner.epoch === this.epoch && owner.backgroundId === this.state.background_id) {
                    this.stopPreviewGeneration(); this.workflowRun = null; this.busy = false; this.activeAction = ""; this.updateControls();
                    if (this.unsaved) this.markEdited();
                }
            }
        })();
        return this.workflowCompletion;
    }

    decorateControls() {
        const add = (selector, name) => { for (const element of this.container.querySelectorAll(selector)) element.classList.add("vnccs-common-creator-" + name); };
        for (const button of this.container.querySelectorAll("button")) {
            button.type = "button";
            if (!button.closest(".bg-segments,nav") && !button.className.includes("vnccs-common-")) {
                const action = ["primary", "success", "danger"].some(variant => button.classList.contains(variant));
                button.classList.add("vnccs-common-creator-" + (action ? "btn" : button.hasAttribute("data-upload") ? "btn-upload" : "model-card-download"));
            }
            for (const variant of ["primary", "success", "danger"]) if (button.classList.contains(variant)) button.classList.add("vnccs-common-creator-btn-" + variant);
        }
        for (const control of this.container.querySelectorAll("input,select,textarea")) {
            if (control.classList.contains("vnccs-common-creator-gen-param-input") || control.classList.contains("vnccs-common-creator-gen-param-select") || control.classList.contains("vnccs-common-creator-resolution-slider") || control.closest(".vnccs-common-creator-textarea-wrapper")) continue;
            if (control.tagName === "INPUT" && ["checkbox", "file"].includes(control.type)) continue;
            control.classList.add("vnccs-common-creator-" + (control.type === "range" ? "slider" : control.tagName === "SELECT" ? "select" : control.tagName === "TEXTAREA" ? "textarea" : "input"));
        }
        add(".bg-panel", "col"); add(".bg-model-card", "model-card"); add(".bg-field", "field");
        add("[data-wizard]", "character-wizard-btn"); add("[data-new],[data-delete]", "btn-secondary");
        for (const field of this.container.querySelectorAll(".bg-field")) for (const label of field.children) if (label.tagName === "SPAN") label.classList.add("vnccs-common-creator-label");
        for (const group of this.container.querySelectorAll(".bg-segments,nav")) {
            group.classList.add("vnccs-common-creator-segmented-field");
            if (group.children.length === 3) group.classList.add("is-three");
            for (const button of group.children) button.classList.add("vnccs-common-creator-segmented-btn");
            this.syncSegments(group);
        }
    }

    syncSegments(group) {
        const buttons = [...group.children];
        for (const button of buttons) button.classList.toggle("is-active", button.classList.contains("active"));
        group.style.setProperty("--selected", String(Math.max(0, buttons.findIndex(button => button.classList.contains("active")))));
        this.customSelects.syncSelection(group);
    }

    syncGenerationCards() {
        for (const card of this.container.querySelectorAll("[data-setting-card]")) {
            const enabled = card.querySelector("input").checked;
            card.classList.toggle("is-selected", enabled);
            const status = card.querySelector("[data-toggle-status]");
            status.classList.toggle("ok", enabled); status.textContent = enabled ? "Enabled" : "Disabled";
        }
        for (const select of this.container.querySelectorAll("[data-lora]")) select.closest(".vnccs-common-creator-lora-item").classList.toggle("is-empty", !select.value);
    }

    async dialog(title, fields, submitLabel = "Save") {
        return await new Promise(resolve => {
            const controls = {};
            const finish = value => { if (!this.dialogs.delete(close)) return; modal.overlay.remove(); resolve(value); };
            const close = () => finish(null);
            this.dialogs.add(close);
            const modal = showModal(this.container, title, () => {
                const form = document.createElement("div"); form.className = "bg-dialog-fields";
                for (const field of fields) {
                    const label = document.createElement("label"), text = document.createElement("span"); text.textContent = field.label; label.className = "bg-field";
                    const control = document.createElement(field.multiline ? "textarea" : field.options ? "select" : "input");
                    if (field.options) for (const value of field.options) { const option = document.createElement("option"); option.textContent = value; control.appendChild(option); }
                    control.value = field.value ?? ""; control.name = field.key;
                    if (!field.options && !field.multiline) control.type = field.type || "text";
                    if (field.required) control.required = true; if (field.multiline) control.rows = 5;
                    if (field.maxLength) control.maxLength = field.maxLength;
                    if (field.oninput) for (const type of ["input", "change"]) control.addEventListener(type, () => field.oninput(control.value));
                    label.append(text, control);
                    form.appendChild(label); controls[field.key] = control;
                }
                return form;
            }, [{ text: "Cancel", action: close }, { text: submitLabel, class: "primary", action: () => {
                if (!Object.values(controls).every(control => control.reportValidity())) return true;
                finish(Object.fromEntries(Object.entries(controls).map(([key, control]) => [key, control.value])));
            } }]);
            const remove = modal.overlay.remove;
            modal.overlay.remove = () => { remove.apply(modal.overlay); if (this.dialogs.delete(close)) resolve(null); };
            this.decorateControls(); this.customSelects.syncSelects();
        });
    }

    async ensureDocument() {
        if (this.document) return true;
        const epoch = this.epoch;
        const values = await this.dialog("New background", [{ key: "name", label: "Background name", value: "Background", required: true }], "Create");
        if (!values || this.disposed || epoch !== this.epoch) return false;
        const document = await this.json("/backgrounds", { name: values.name, editor: this.state.editor, model_settings: this.state.model_settings });
        if (this.disposed || epoch !== this.epoch) return false;
        await this.acceptDocument(document, true, epoch); return epoch === this.epoch && !this.disposed;
    }

    async click(event) {
        const button = event.target.closest("button,[data-model-toggle],[data-model-choice]"); if (!button || !this.container.contains(button) || button.disabled) return;
        const epoch = this.epoch;
        try {
            if (button.hasAttribute("data-scroll-variation")) {
                const list = this.container.querySelector("[data-variation-cards]");
                list.scrollBy({ left: Number(button.dataset.scrollVariation) * list.clientWidth * .8, behavior: "smooth" }); return;
            }
            if (button.hasAttribute("data-close-variation")) { this.activeVariationId = ""; this.render(); return; }
            if (button.hasAttribute("data-delete-variation")) {
                const cards = this.state.editor.variants, index = cards.findIndex(card => card.id === button.dataset.deleteVariation);
                if (index < 0) return;
                this.state.editor.variants = cards.filter(card => card.id !== button.dataset.deleteVariation);
                this.markEdited(); this.render();
                const remaining = this.container.querySelectorAll("[data-delete-variation]");
                (remaining[Math.min(index, remaining.length - 1)] || this.container.querySelector("[data-add-variation]")).focus?.();
                return;
            }
            if (button.hasAttribute("data-add-variation") || button.hasAttribute("data-edit-variation")) {
                const cards = this.state.editor.variants;
                let id = button.dataset.editVariation;
                if (!id) {
                    if (cards.length >= 64) throw new Error("A background supports up to 64 variants.");
                    const base = `variant_${generateRandomSeed()}`; id = base;
                    for (let index = 1; cards.some(card => card.id === id); index++) id = `${base}_${index}`;
                    cards.push({ id, name: `Variant ${cards.length + 1}`, time: "", weather: "", additional_details: "" });
                    this.markEdited();
                }
                this.activeVariationId = id; this.render();
                // Shared view restoration finishes on the next frame.
                requestAnimationFrame(() => {
                    if (this.disposed || epoch !== this.epoch || this.activeVariationId !== id) return;
                    const behavior = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
                    const card = [...this.container.querySelectorAll("[data-edit-variation]")].find(card => card.dataset.editVariation === id);
                    card?.scrollIntoView?.({ block: "nearest", inline: "nearest", behavior });
                    this.container.querySelector("[data-variation-editor]").scrollIntoView?.({ block: "nearest", behavior });
                });
                if (button.hasAttribute("data-add-variation")) this.container.querySelector("[data-variation-name]").focus?.({ preventScroll: true });
                return;
            }
            if (button.dataset.step) { this.setStep(button.dataset.step); return; }
            if (button.dataset.modelType) {
                const editor = this.state.editor, type = button.dataset.modelType;
                if (type === editor.generation.model_type || !["qi2", "moss"].includes(type)) return;
                editor.generation_profiles ??= {};
                editor.generation_profiles[editor.generation.model_type] = clone(editor.generation);
                editor.generation = { ...clone(DEFAULT_EDITOR.generation), ...(type === "moss" ? MOSS_GENERATION : {}), ...clone(editor.generation_profiles[type] || {}) };
                if (type === "moss") editor.generation.turbo = false;
                this.modelPickerOpen = false; this.markEdited(); this.render(); return;
            }
            if (button.hasAttribute("data-model-toggle")) { this.modelPickerOpen = !this.modelPickerOpen; this.renderModelPicker(); this.updateControls(); return; }
            if (button.dataset.modelChoice) {
                const settings = this.state.editor.generation, next = button.dataset.modelChoice;
                if (isTurboModel(next) || isTurboModel(settings.diffusion_model_name)) { settings.steps = isTurboModel(next) ? 8 : 25; settings.cfg = 1; settings.turbo = false; }
                settings.diffusion_model_name = next; this.modelPickerOpen = false; this.markEdited(); this.render(); return;
            }
            if (button.dataset.downloadModel) { await this.downloadModel(button.dataset.downloadModel); return; }
            if (button.hasAttribute("data-enlarge")) { this.openPreview(); return; }
            if (button.dataset.value) { this.state.editor.attributes[button.closest("[data-trait]").dataset.trait] = button.dataset.value; this.markEdited(); this.render(); return; }
            if (button.dataset.random) { this.state.editor.generation.seed = generateRandomSeed(); this.markEdited(); this.render(); return; }
            if (button.dataset.cgId) { const cg = this.document.cgs.find(item => item.cg_id === button.dataset.cgId); this.state.selected_cg_id = cg.cg_id; this.state.model_id = cg.model_id; this.state.camera = clone(cg.camera); this.state.frame = { width: cg.width, height: cg.height }; await this.showModel(); this.sync(); this.render(); return; }
            if (button.hasAttribute("data-new")) {
                const values = await this.dialog("New background", [{ key: "name", label: "Background name", value: "Background", required: true }], "Create");
                if (!values || this.disposed || epoch !== this.epoch) return;
                await this.persist();
                if (epoch !== this.epoch || this.disposed) return;
                const editor = { ...this.state.editor, variants: clone(DEFAULT_EDITOR.variants) };
                await this.acceptDocument(await this.json("/backgrounds", { name: values.name, editor, model_settings: this.state.model_settings }), true, epoch); return;
            }
            if (button.hasAttribute("data-delete")) {
                if (!this.document) return;
                const confirmation = await this.dialog("Remove background from selector", [{ key: "name", label: "Saved assets and workflow references will be retained. Background", value: this.document.name }], "Remove");
                if (!confirmation || epoch !== this.epoch || this.disposed) return;
                await this.persist(); if (epoch !== this.epoch || this.disposed) return;
                await this.json(`/backgrounds/${this.document.background_id}`, { revision: this.document.revision }, "DELETE");
                if (epoch !== this.epoch || this.disposed) return;
                this.catalog = this.catalog.filter(item => item.background_id !== this.document.background_id); this.document = null;
                Object.assign(this.state, { background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", camera: null }); this.render(); this.sync(); return;
            }
            if (button.hasAttribute("data-upload")) { this.container.querySelector("[data-file]").click(); return; }
            if (button.hasAttribute("data-recenter")) { this.beginGesture(); this.viewer.setCameraState(sourceCameraView(this.scene.objects[0].source_camera || this.scene.camera), { emit: true }); this.commitGesture(); return; }
            if (button.hasAttribute("data-wizard")) {
                const values = await this.dialog("Background wizard", [{ key: "description", label: "Describe your background", value: this.state.editor.scene_description, multiline: true, required: true }], "Fill scene attributes");
                if (!values || epoch !== this.epoch || this.disposed) return; this.state.editor.scene_description = values.description; this.markEdited(); await this.runJob("wizard"); return;
            }
            if (button.hasAttribute("data-save-view")) {
                const previousFrame = clone(this.state.frame || this.viewer.getCaptureSettings());
                const frameAspect = ASPECTS.reduce((best, aspect) => {
                    const ratio = value => { const [a, b] = value.split(":").map(Number); return a / b; };
                    return Math.abs(ratio(aspect) - previousFrame.width / previousFrame.height) < Math.abs(ratio(best) - previousFrame.width / previousFrame.height) ? aspect : best;
                }, this.state.editor.aspect);
                const previewAspect = aspect => { if (epoch === this.epoch && !this.disposed) this.viewer.setCaptureSettings({ ...frameDimensions(aspect, this.state.editor.generation.resolution_mp), show_camera_frame: true }); };
                const values = await this.dialog("Generate CG from this camera", [{ key: "name", label: "CG name", value: `View ${(this.document.cgs.length || 0) + 1}`, required: true }, { key: "aspect", label: "Aspect ratio", value: frameAspect, options: ASPECTS, oninput: previewAspect }, { key: "cg_prompt", label: "Refinement instruction", value: this.state.editor.cg_prompt, multiline: true, required: true }], "Generate CG");
                if (!values || this.disposed || epoch !== this.epoch) { if (!this.disposed && epoch === this.epoch) this.viewer.setCaptureSettings({ ...previousFrame, show_camera_frame: true }); return; }
                this.state.editor.cg_prompt = values.cg_prompt; this.markEdited();
                const size = frameDimensions(values.aspect, this.state.editor.generation.resolution_mp);
                this.state.frame = clone(size); this.sync();
                this.viewer.setCaptureSettings({ ...size, show_camera_frame: true });
                const camera = clone(this.viewer.getCameraState());
                this.busy = true; this.updateControls();
                const raw = await this.viewer.capturePreview({ ...size, cameraState: camera });
                if (this.disposed || epoch !== this.epoch) return;
                await this.runJob("cg", { name: values.name, camera, model_id: this.state.model_id }, raw); return;
            }
            if (button.dataset.action) await this.runJob(button.dataset.action);
        } catch (error) { if (epoch === this.epoch && !this.disposed) { this.busy = false; this.updateControls(); this.message(error.message, true); } }
    }

    openPreview() {
        if (!this.document?.images.some(image => image.image_id === this.state.selected_image_id)) return;
        const close = () => modal.overlay.remove();
        const modal = showModal(this.container, "Background preview", () => {
            const content = document.createElement("div"), image = document.createElement("img"); image.dataset.largePreview = ""; image.alt = "Full-size selected background"; content.appendChild(image);
            return content;
        }, [{ text: "Close", action: close }]);
        modal.modal.classList.add("bg-image-dialog"); modal.buttons[0].dataset.closePreview = "";
        this.dialogs.add(close);
        const remove = modal.overlay.remove;
        modal.overlay.remove = () => { this.dialogs.delete(close); remove.apply(modal.overlay); };
        this.render();
    }

    input(event) {
        if (this.restoring || this.busy || !this.state) return;
        const control = event.target;
        if (control.hasAttribute("data-variation-name")) {
            const card = this.state.editor.variants.find(card => card.id === this.activeVariationId);
            if (card) { card.name = control.value; this.markEdited(); this.render(); }
            return;
        }
        if (["number", "range"].includes(control.type) && !validNumber(control)) return;
        if (control === this.fov || control === this.fovNumber) {
            const fov = Number(control.value); if (!this.canEditCamera() || !control.value || !Number.isFinite(fov)) return;
            this.beginGesture(); this.viewer.setCameraState({ ...this.viewer.getCameraState(), fov: Math.max(5, Math.min(120, fov)) }, { emit: true }); return;
        }
        if (control.dataset.setting) {
            const keys = control.dataset.setting.split("."), last = keys.pop();
            const target = keys.reduce((value, key) => value[key] ||= {}, this.state.editor);
            target[last] = control.type === "checkbox" ? control.checked : ["number", "range"].includes(control.type) ? Number(control.value) : control.value;
            if (last === "turbo") { this.state.editor.generation.steps = control.checked ? 6 : 25; this.state.editor.generation.cfg = 1; this.render(); }
            if (last === "resolution_mp") this.container.querySelector("[data-resolution]").textContent = `${Number(control.value).toFixed(2)} MP`;
            if (control.type === "checkbox") this.syncGenerationCards();
            this.markEdited(); return;
        }
        if (control.dataset.modelSetting) { this.state.model_settings[control.dataset.modelSetting] = Number(control.value); this.markEdited(); return; }
        if (control.hasAttribute("data-lora") || control.hasAttribute("data-lora-strength")) {
            this.state.editor.generation.lora_stack = [...[...this.container.querySelectorAll("[data-lora]")].map(select => ({ name: select.value, strength: Number(this.container.querySelector(`[data-lora-strength="${select.dataset.lora}"]`).value) })), ...this.state.editor.generation.lora_stack.slice(5)];
            this.syncGenerationCards();
            this.markEdited();
        }
    }

    async change(event) {
        const control = event.target;
        if (this.restoring || this.busy) return;
        if (control.type === "number" && !validNumber(control)) {
            const value = control.dataset.setting ? control.dataset.setting.split(".").reduce((current, key) => current?.[key], this.state.editor)
                : control.dataset.modelSetting ? this.state.model_settings[control.dataset.modelSetting]
                : control.hasAttribute("data-lora-strength") ? this.state.editor.generation.lora_stack[Number(control.dataset.loraStrength)]?.strength ?? 1
                : control === this.fovNumber ? this.viewer?.getCameraState().fov : undefined;
            if (value !== undefined) { control.value = value; this.message("Enter a number within the field limits.", true); }
            return;
        }
        if (control.tagName === "SELECT" && (control.dataset.setting || control.dataset.modelSetting || control.hasAttribute("data-lora"))) { this.input(event); return; }
        let epoch = this.epoch;
        try {
            if (control.hasAttribute("data-library")) {
                const id = control.value; if (!id) return; await this.persist(); if (epoch !== this.epoch || this.disposed) return; epoch = ++this.epoch;
                this.restoring = true; this.updateControls();
                try {
                    const document = await this.request(`/backgrounds/${id}`);
                    if (epoch === this.epoch && !this.disposed) await this.acceptDocument(document, true, epoch);
                } finally {
                    if (epoch === this.epoch && !this.disposed) { this.restoring = false; this.render(); this.sync(); }
                }
            } else if (control.hasAttribute("data-file")) {
                const file = control.files?.[0]; if (!file) return;
                if (!await this.ensureDocument()) return;
                await this.persist(); if (epoch !== this.epoch || this.disposed) return; this.busy = true; this.updateControls();
                const form = new FormData(); form.append("image", file); form.append("data", JSON.stringify({ revision: this.document.revision }));
                await this.acceptDocument(await this.request(`/backgrounds/${this.document.background_id}/upload`, { method: "POST", body: form }), true, epoch);
                if (epoch !== this.epoch || this.disposed) return;
                this.busy = false; this.updateControls(); this.message("Background uploaded. Build its model when ready."); control.value = "";
            }
        } catch (error) { if (epoch === this.epoch && !this.disposed) { this.busy = false; this.updateControls(); this.message(error.message, true); } }
    }

    async runJob(action, extra = {}, raw = null) {
        if (!await this.ensureDocument() || this.disposed) return;
        const epoch = this.epoch;
        this.busy = true; this.activeAction = action;
        if (action === "background") this.startPreviewGeneration();
        this.updateControls(); this.message("Preparing " + action + "…");
        try {
            await this.persist();
            if (epoch !== this.epoch || this.disposed) return;
            if (["model", "cg"].includes(action) && !this.state.selected_image_id) throw new Error("Create or upload a background image first.");
            const id = this.state.background_id;
            const payload = { action, editor: clone(this.state.editor), revision: this.document.revision, image_id: this.state.selected_image_id,
                model_settings: clone(this.state.model_settings), ...extra };
            let result;
            if (raw) { const form = new FormData(); form.append("data", JSON.stringify(payload)); form.append("image", raw, "raw.png"); result = await this.request(`/backgrounds/${id}/jobs`, { method: "POST", body: form }); }
            else result = await this.json(`/backgrounds/${id}/jobs`, payload);
            this.jobId = result.job_id; this.updateControls();
            while (!this.disposed && epoch === this.epoch) {
                result = await this.request(`/jobs/${this.jobId}`);
                this.message(result.message || "Working…");
                if (["completed", "failed", "cancelled"].includes(result.status)) break;
                await new Promise(resolve => { this.pollTimer = setTimeout(resolve, 500); this.pollResolve = resolve; });
            }
            if (this.disposed || epoch !== this.epoch) return;
            if (result.status !== "completed") throw new Error(result.error || (result.status === "cancelled" ? "Generation stopped." : "Generation failed."));
            if (result.result.editor) {
                this.busy = false;
                animateWizardRows([...this.container.querySelectorAll(".bg-attributes .bg-field")], () => {
                    this.state.editor = editorDefaults(result.result.editor); this.markEdited(); this.render();
                });
            }
            else {
                const document = result.result.document;
                if (action === "background") {
                    await this.revealGeneratedPreview(document.images.find(image => image.image_id === document.selected_image_id)?.url);
                    if (this.disposed || epoch !== this.epoch) return;
                }
                if (action === "cg") {
                    const cg = document.cgs.at(-1);
                    Object.assign(this.state, { selected_cg_id: cg.cg_id, model_id: cg.model_id, camera: clone(cg.camera), frame: { width: cg.width, height: cg.height } });
                }
                await this.acceptDocument(document, action !== "cg", epoch);
                if (action === "background") this.state.step = "background";
                if (action === "model") this.state.step = "model";
                this.sync(); this.render();
            }
            this.message(action === "cg" ? "CG generated and saved with its camera. The selected CG is the workflow output." : action === "model" ? "Model ready. Set the camera, then save a view to generate its CG." : action === "wizard" ? "Scene attributes filled." : "Background generated and saved.");
        } finally { if (epoch === this.epoch && !this.disposed) { this.stopPreviewGeneration(); this.jobId = ""; this.busy = false; this.activeAction = ""; this.render(); } }
    }

    resize() {
        const height = `${Math.max(300, (this.node.size?.[1] || 800) - 90) / CREATOR_ZOOM}px`;
        const width = this.container.clientWidth || (this.node.size?.[0] || 1280) / CREATOR_ZOOM;
        if (height !== this.container.style.height || width !== this.layoutWidth) this.customSelects._closeDropdown(false);
        this.layoutWidth = width;
        this.container.classList.toggle("bg-compact", width < 1400);
        if (this.container.style.height !== height) this.container.style.height = height;
        this.viewer?.resize();
        this.updateVariantScroll();
    }
    dispose() {
        if (this.disposed) return;
        this.sync(); this.stopPreviewGeneration();
        this.disposed = true; ++this.epoch; this.events.abort(); clearTimeout(this.saveTimer); clearTimeout(this.wheelTimer); clearTimeout(this.pollTimer); this.pollResolve?.();
        this.cc.stopPolling();
        for (const finish of [...this.dialogs]) finish(null);
        this.observer.disconnect(); this.customSelects._closeDropdown?.(false); this.customSelects.dispose(); this.tooltipCleanup?.(); this.viewer?.dispose(); this.container.remove();
    }
}


function hideState(node) {
    if (node.outputs?.length === 1 && !node.outputs.some(output => output.name === "Prompt")) {
        node.addOutput?.("Prompt", "STRING", { shape: globalThis.LiteGraph?.GRID_SHAPE });
    }
    // Older workflow JSON can restore the removed optional socket.
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index--) {
        if (node.inputs[index].name === "image" && node.inputs[index].type === "IMAGE") node.removeInput?.(index);
    }
    const seed = node.widgets?.find(item => item.name === "seed");
    if (seed) { seed.hidden = true; seed.computeSize = () => [0, -4]; seed.draw = () => {}; if (seed.inputEl) seed.inputEl.style.display = "none"; }
    const widget = node.widgets?.find(item => item.name === "background_data");
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.element) widget.element.style.display = "none";
    if (widget.inputEl) widget.inputEl.style.display = "none";
    return widget;
}

function fit(node) {
    syncDOMWidgetWidth(node, "background_ui");
    node.backgroundCreator?.resize();
}

app.registerExtension({
    name: "VNCCS.BackgroundCreator",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_BackgroundCreator") return;
        const created = nodeType.prototype.onNodeCreated;
        const configured = nodeType.prototype.onConfigure;
        const resized = nodeType.prototype.onResize;
        const serialized = nodeType.prototype.onSerialize;
        const executed = nodeType.prototype.onExecuted;
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            if (this.backgroundCreator) return result;
            const state = hideState(this);
            this.backgroundCreator = new BackgroundCreatorWidget(this, vnccsApi);
            this.backgroundCreatorDOM = this.addDOMWidget("background_ui", "ui", this.backgroundCreator.container, {
                serialize: false, hideOnZoom: false, getMinHeight: () => 300,
            });
            if (state) {
                const beforeQueued = state.beforeQueued;
                state.beforeQueued = (...args) => {
                    beforeQueued?.apply(state, args);
                    if (this.backgroundCreator.restoring) throw new Error("Background Creator is still restoring its scene.");
                    if (this.backgroundCreator.busy) throw new Error("Wait for Background Creator to finish its current operation.");
                    this.backgroundCreator.sync();
                };
                state.serializeValue = () => this.backgroundCreator.sync() || state.value;
            }
            this.setSize?.([1280, 800]);
            fit(this);
            syncDOMWidgetWidthSoon(this, "background_ui");
            registerCleanup(this, () => this.backgroundCreator?.dispose());
            this._backgroundRestore = setTimeout(() => this.backgroundCreator?.restore(), 0);
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            this.backgroundCreator?.invalidateRestore();
            const result = configured?.apply(this, arguments);
            hideState(this);
            clearTimeout(this._backgroundRestore);
            this.backgroundCreator?.restore();
            fit(this);
            return result;
        };
        nodeType.prototype.onResize = function () {
            const result = resized?.apply(this, arguments);
            fit(this);
            return result;
        };
        nodeType.prototype.onSerialize = function (info) {
            const result = serialized?.apply(this, arguments);
            const raw = this.backgroundCreator?.sync();
            const index = this.widgets?.findIndex(item => item.name === "background_data");
            if (raw && info && index >= 0 && Array.isArray(info.widgets_values)) info.widgets_values[index] = raw;
            return result;
        };
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            if (message?.background_preview) this.backgroundCreator?.finishWorkflow(message);
            return result;
        };
        nodeType.prototype.onRemoved = function () {
            clearTimeout(this._backgroundRestore);
            this.backgroundCreator?.dispose();
            this.backgroundCreator = null;
            return removed?.apply(this, arguments);
        };
    },
});
