import { CGCharacterPanel, CG_CHARACTER_PANEL_CSS } from "./cg_character_panel.mjs?v=cg-creator-3";
import { CGCharacterEditor, loadCharacterRuntime, validateCharacters } from "./cg_characters.mjs?v=cg-creator-3";
import { STYLE_PICKER_CSS } from "./character_styles.mjs?v=style-mix-9";
import { app } from "../../scripts/app.js";
import { vnccsApi as api, storage, workflowScope, cacheIdentity, watchConnection } from "./vnccs_transport.js";
import { createSakuraInteractions, playSakuraMotion, stopSakuraMotion } from "./sakura_interactions.mjs";
import { SAKURA_THEME_CSS, SAKURA_INTERACTION_CSS, SAKURA_CREATOR_CONTROL_CSS, injectStyles, showModal, enableMiddleMouseCanvasPan, createControlCenterClient, RESOLUTION_SCALE_MIN_MP, RESOLUTION_SCALE_MAX_MP, RESOLUTION_SCALE_STEP_MP, resolutionScaleMegapixels, resolutionScaleValue, resolutionScaleText, setHelpText, attachHelpTooltips, generateRandomSeed, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, registerCleanup } from "./vnccs_common.js";

const BACKGROUND_CREATOR_ZOOM = .67;
const BACKGROUND_GENERATION_WIDTH = (1280 - 24 * BACKGROUND_CREATOR_ZOOM) * .35;
const SOURCE_PANEL_WIDTH = (2102 - 48 * BACKGROUND_CREATOR_ZOOM) * .78125 - BACKGROUND_GENERATION_WIDTH;
const LAYOUT_WIDTH = Math.ceil(SOURCE_PANEL_WIDTH + 2247 + 960 + 24 + 26);
const CREATOR_MIN_HEIGHT = 950;
const STYLE = `
.vnccs-cg-widget { position:relative; width:100%; height:100%; min-width:0; min-height:0; box-sizing:border-box; overflow:hidden; }
.vnccs-cg-creator { ${SAKURA_THEME_CSS} --warning:#ffaa00; --shadow-subtle:0 2px 8px rgba(0,0,0,0.3); --shadow-elevated:0 8px 32px rgba(0,0,0,0.5); }
${SAKURA_INTERACTION_CSS}
${SAKURA_CREATOR_CONTROL_CSS}
${CG_CHARACTER_PANEL_CSS}
${STYLE_PICKER_CSS.replaceAll(".vnccs-creator-container", ".vnccs-cg-creator")}
/* CG workspace and Utils viewport layout. */
.vnccs-cg-creator.vnccs-i3s { position:relative; display:flex; flex-direction:column; min-height:0; box-sizing:border-box; border:0; overflow:hidden; transform-origin:top left; pointer-events:auto; font-family:var(--font); }
.vnccs-cg-creator [hidden] { display:none !important; }
/* One logical composition scales as a whole with the node. */
.vnccs-cg-creator .cg-workspace { flex:1; min-height:0; display:grid; grid-template-columns:${SOURCE_PANEL_WIDTH}px 2247px 960px; gap:12px; }
.vnccs-cg-creator .bg-panel { min-width:0; min-height:0; overflow-x:hidden; overscroll-behavior:contain; }
.vnccs-cg-creator .bg-panel>* { flex:0 0 auto; min-width:0; }
.vnccs-cg-creator .bg-model-source { overflow:hidden; gap:10px; }
.vnccs-cg-creator .cg-middle { position:relative; display:grid; grid-template-rows:minmax(0,1fr) 248px; min-height:0; min-width:0; gap:12px; }
.vnccs-cg-creator .bg-generation-dock { min-width:0; min-height:248px; }
.vnccs-cg-creator .bg-generation-dock .bg-panel { overflow:hidden; gap:7px; padding:12px; }
.vnccs-cg-creator .bg-generation-dock .bg-qi2-model-panel { overflow:visible; gap:6px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-model-picker { position:relative; }
/* Floating surface copied from Sakura .vnccs-common-ui-dropdown. */
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-model-picker-menu { position:absolute; bottom:calc(100% + 8px); left:0; right:0; z-index:10; max-height:320px; overflow:auto; background:#17121d; box-shadow:0 12px 28px rgba(0,0,0,0.45),0 0 0 1px rgba(255,143,163,0.04); }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-section-title { margin:0; font-size:10px; letter-spacing:1px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-subsection { gap:7px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-field { margin:0; gap:5px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-gen-param-grid { gap:8px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-gen-param-input,.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-gen-param-select { min-height:34px; height:34px; padding:6px 10px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-seed-dice-btn { height:34px; width:34px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-model-card { padding:8px 10px; gap:4px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-model-card-desc { font-size:10px; }
.vnccs-cg-creator .bg-generation-dock .vnccs-common-creator-model-card-status { font-size:9px; }
.vnccs-cg-creator .bg-frame-settings { display:contents; }
.vnccs-cg-creator .bg-frame-settings .bg-field { margin:0; gap:4px; }
.vnccs-cg-creator .bg-frame-settings .bg-field:first-child { display:grid; grid-template-columns:88px minmax(0,1fr); align-items:center; }
.vnccs-cg-creator .bg-frame-settings .vnccs-common-creator-select { height:34px; min-height:34px; padding:6px 10px; }
.vnccs-cg-creator .bg-frame-settings .vnccs-common-creator-label { font-size:9px; }
.vnccs-cg-creator .bg-qi2-cache { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.vnccs-cg-creator .bg-qi2-cache .vnccs-common-creator-gen-param-field { display:grid; grid-template-columns:auto minmax(0,1fr); align-items:center; gap:8px; }
.vnccs-cg-creator .bg-qi2-cache .vnccs-common-creator-label { font-size:9px; white-space:nowrap; }
.vnccs-cg-creator .bg-generation-dock .bg-qi2-cache .vnccs-common-creator-gen-param-select { min-height:28px; height:28px; padding:3px 8px; font-size:12px; }
.vnccs-cg-creator [data-qi2].vnccs-common-creator-container { display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,1fr); gap:12px; padding:0; overflow:visible; zoom:1; pointer-events:auto; }
.vnccs-cg-creator [data-qi2]>.bg-qi2-contents { display:contents; }
.vnccs-cg-creator .bg-row { display:grid; grid-auto-flow:column; grid-auto-columns:minmax(0,1fr); align-items:start; gap:10px; }
.vnccs-cg-creator .bg-model-mode { align-items:center; }
.vnccs-cg-creator .bg-model-mode .vnccs-common-creator-section-title { margin:0; }
.vnccs-cg-creator .bg-model-mode:has([data-moge-refine-label]:not([hidden])) { grid-template-columns:minmax(0,1fr) minmax(0,1fr) auto; }
.vnccs-cg-creator [data-moge-refine-label]:not([hidden]) { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; font-size:11px; }
.vnccs-cg-creator [data-moge-refine-views] { margin:0; accent-color:var(--accent); }
.vnccs-cg-creator .bg-wide { width:100%; flex:none; min-height:40px; }
/* Source preview rules copied from VNCCS Background Creator; root selector adapted. */
.vnccs-cg-creator .bg-panel>.bg-preview { flex:0 0 auto; width:100%; height:auto; aspect-ratio:16/9; min-height:100px; }
.vnccs-cg-creator .bg-preview>[data-preview] { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
.vnccs-cg-creator .bg-preview [data-enlarge] { position:absolute; inset:0; width:100%; height:100%; padding:0; border:0; background:transparent; cursor:zoom-in; }
.vnccs-cg-creator .bg-preview>[data-empty] { padding:18px; }
.vnccs-cg-creator .bg-dialog-fields { display:flex; flex-direction:column; gap:12px; }
.vnccs-cg-creator .vnccs-common-modal { max-height:calc(100% - 32px); overflow:auto; }
.vnccs-cg-creator .bg-image-preview { position:absolute; inset:0; background:var(--bg-primary); }
.vnccs-cg-creator .bg-image-preview [data-close-preview] { position:absolute; right:12px; bottom:12px; z-index:1; padding:6px 16px; font-size:13px; cursor:pointer; }
.vnccs-cg-creator .bg-image-viewport { position:absolute; inset:0; min-height:0; overflow:hidden; touch-action:none; cursor:grab; }
.vnccs-cg-creator .bg-image-viewport.is-dragging { cursor:grabbing; }
.vnccs-cg-creator .bg-image-viewport:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.vnccs-cg-creator .bg-image-preview img { position:absolute; left:0; top:0; display:block; max-width:none; max-height:none; transform-origin:0 0; pointer-events:none; user-select:none; }
.vnccs-cg-creator .bg-model-card { display:flex; flex-direction:column; align-items:stretch; text-align:left; width:100%; }
.vnccs-cg-creator .vnccs-i3s__camera-frame>span { display:none; }
.vnccs-cg-creator .bg-viewport-panel { overflow:hidden; }
.vnccs-cg-creator .bg-badge { margin-left:auto; font-size:10px; font-weight:400; letter-spacing:0; color:var(--text-secondary); }
.vnccs-cg-creator .bg-viewport { box-sizing:border-box; flex:0 0 auto; position:relative; width:100%; height:auto; aspect-ratio:16/9; min-height:0; background:#08090e; border:1px solid var(--border); border-radius:var(--radius-md); overflow:hidden; }
.vnccs-cg-creator [data-viewer] { position:absolute; inset:0; }
.vnccs-cg-creator [data-viewer][inert] { pointer-events:none; }
.vnccs-cg-creator [data-viewer] canvas:focus-visible { outline-offset:-3px; }
.vnccs-cg-creator .bg-camera-toolbar { display:flex; align-items:center; flex-wrap:wrap; gap:8px; padding:8px; background:var(--bg-primary); border:1px solid var(--accent-border); border-radius:var(--radius-md); }
.vnccs-cg-creator .bg-camera-toolbar label { display:flex; align-items:center; gap:7px; flex:1; min-width:130px; font-size:11px; color:var(--text-secondary); }
.vnccs-cg-creator .bg-camera-toolbar input[type=number] { width:58px; padding:7px 4px; flex:none; }
.vnccs-cg-creator .bg-camera-toolbar input[type=range] { width:60px; flex:1; }
.vnccs-cg-creator .bg-camera-toolbar button { width:auto; flex:0 0 auto; min-height:34px; font-size:11px; padding:7px 9px; }
.vnccs-cg-creator .bg-expansion-actions:not([hidden]) { position:absolute; z-index:5; bottom:12px; left:50%; transform:translateX(-50%); display:flex; flex-wrap:wrap; justify-content:center; gap:8px; padding:8px; box-sizing:border-box; width:max-content; max-width:calc(100% - 24px); background:var(--bg-primary); border:1px solid var(--accent-border); border-radius:var(--radius-md); }
.vnccs-cg-creator .bg-expansion-actions > button { flex:0 1 auto; min-width:0; width:auto; height:auto; white-space:normal; overflow-wrap:anywhere; }
.vnccs-cg-creator .bg-expansion-compare:not([hidden]) { position:absolute; z-index:4; inset:0; display:grid; place-items:center; container-type:size; background:#000; cursor:ew-resize; }
.vnccs-cg-creator .bg-expansion-images { position:relative; width:min(100%, calc(100cqh * var(--frame-aspect))); aspect-ratio:var(--frame-aspect); }
.vnccs-cg-creator .bg-expansion-images img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
.vnccs-cg-creator [data-expansion-after] { clip-path:inset(0 0 0 var(--split, 50%)); }
.vnccs-cg-creator .bg-expansion-images::after { content:""; position:absolute; top:0; bottom:0; left:var(--split, 50%); border-left:1px solid var(--accent); pointer-events:none; }
.vnccs-cg-creator [data-expansion-split] { position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; cursor:inherit; }
.vnccs-cg-creator .bg-expansion-images:has([data-expansion-split]:focus-visible) { outline:2px solid var(--accent); outline-offset:-2px; }
.vnccs-cg-creator .bg-viewer-empty { position:absolute; inset:20px; display:grid; place-items:center; text-align:center; color:var(--text-secondary); pointer-events:none; }
.vnccs-cg-creator .bg-hint { font-size:10px; color:var(--text-secondary); }
.vnccs-cg-creator .bg-cg-panel { overflow:hidden; }
.vnccs-cg-creator .bg-cg-list { flex:1; min-height:0; overflow:auto; display:flex; flex-direction:column; gap:14px; padding:2px; }
.vnccs-cg-creator .bg-cg-entry { position:relative; flex:none; }
.vnccs-cg-creator .bg-cg-entry .bg-cg-card { width:100%; }
.vnccs-cg-creator .bg-cg-entry .bg-cg-card strong { padding-right:32px; }
.vnccs-cg-creator .bg-cg-card { flex:none; display:flex; flex-direction:column; gap:9px; text-align:left; padding:12px; }
.vnccs-cg-creator .bg-cg-card img { width:100%; height:auto; object-fit:contain; border-radius:var(--radius-sm); }
.vnccs-cg-creator .bg-cg-card img:not([src]) { visibility:hidden; }
.vnccs-cg-creator .bg-cg-card span { color:var(--text-secondary); font:10px var(--font-mono); }
.vnccs-cg-creator .bg-cg-pending-preview { position:relative; flex:none; width:100%; overflow:hidden; border-radius:var(--radius-sm); }
.vnccs-cg-creator .bg-cg-pending-preview .vnccs-common-creator-generation-img { left:0; width:100%; height:100%; transform:none; object-fit:contain; }
.vnccs-cg-creator [data-cg-empty] { color:var(--text-secondary); }
.vnccs-cg-creator .error { color:var(--error) !important; }

/* Original cards are divs; button bindings inherit their owner text color. */
.vnccs-cg-creator .bg-model-card,.vnccs-cg-creator .bg-cg-card { color:inherit; }


/* Background Creator's horizontal collection, with two rows of larger source cards. */
.vnccs-cg-creator .bg-variation-strip { display:grid; grid-template-columns:minmax(0,1fr); align-items:center; gap:8px; min-width:0; }
.vnccs-cg-creator .bg-variation-strip.is-scrollable { grid-template-columns:36px minmax(0,1fr) 36px; }
.vnccs-cg-creator .bg-variation-strip.is-scrollable .bg-variation-cards { grid-column:2; }
.vnccs-cg-creator .bg-variation-cards { grid-column:1; grid-row:1; display:grid; grid-auto-columns:calc((100% - 15px)/2); grid-template-rows:repeat(2,max-content); gap:15px; min-width:0; overflow-x:auto; overflow-y:hidden; overscroll-behavior-x:contain; scrollbar-width:thin; scrollbar-color:var(--accent-border) transparent; scroll-snap-type:x proximity; scroll-padding-inline:3px; padding:3px; }
.vnccs-cg-creator .bg-variation-cards.is-single-row { grid-template-rows:max-content; }
.vnccs-cg-creator .bg-variation-item { min-width:0; scroll-snap-align:start; }
.vnccs-cg-creator [data-source-choice] { display:block; width:100%; min-width:0; overflow:hidden; padding:0; text-align:left; background:var(--bg-surface); color:var(--text-primary); border:2px solid color-mix(in srgb,var(--accent-lavender) 60%,var(--bg-surface)); border-radius:var(--radius-md); }
.vnccs-cg-creator [data-source-choice]:hover:not(:disabled) { border-color:var(--accent); background:var(--bg-hover); }
.vnccs-cg-creator [data-source-choice][aria-pressed=true] { border-color:var(--accent); }
.vnccs-cg-creator [data-source-choice]:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.vnccs-cg-creator .bg-variation-preview { position:relative; aspect-ratio:16/9; }
.vnccs-cg-creator [data-variation-image] { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:contain; border-radius:10px; pointer-events:none; }
.vnccs-cg-creator .bg-source-name { display:block; padding:7px 9px; font-size:12px; line-height:16px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vnccs-cg-creator .bg-variation-scroll { grid-row:1; width:36px; height:44px; padding:5px; background:var(--bg-surface); color:var(--text-primary); }
.vnccs-cg-creator .bg-variation-scroll svg { width:24px; height:24px; flex:none; stroke-width:2.25; }
.vnccs-cg-creator [data-scroll-source="-1"] { grid-column:1; }
.vnccs-cg-creator [data-scroll-source="1"] { grid-column:3; }
.vnccs-cg-creator .bg-variation-scroll:hover:not(:disabled) { border-color:var(--accent); background:var(--bg-hover); color:var(--accent); }
.vnccs-cg-creator .bg-variation-scroll:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.vnccs-cg-creator .bg-variation-scroll:disabled { opacity:.35; border-color:var(--accent-border); background:var(--bg-surface); box-shadow:none; }

.vnccs-cg-creator { position:relative; background:var(--bg-primary); color:var(--text-primary); font-family:var(--font); font-size:13px; width:100%; height:100%; overflow:hidden; box-sizing:border-box; padding:12px; gap:12px; }
@media (prefers-reduced-motion: reduce) {
    .vnccs-cg-creator .vnccs-common-creator-generation-base { animation:none !important; }
    .vnccs-cg-creator .vnccs-common-creator-generation-scan { display:none; animation:none !important; }
}
`;

const MODEL_DEFAULTS = Object.freeze({ provider: "pixal3d", num_gaussians: 262144, steps: 20, guidance_scale: 3, seed: -1,
    pixal_profile: "fast", pixal_mesh_resolution: 0, pixal_texture_resolution: 0,
    worldmirror_target_size: 2044, worldmirror_apply_sky_mask: true, camera_shift_lora: "", moge_refine_views: false });
const DEFAULT_SETTINGS = Object.freeze({
    prompt: "Change the camera angle from <image2> to <image1>.",
    aspect: "16:9", resolution_mp: 2,
    generation: { steps: 25, cfg: 1, sampler: "euler", scheduler: "simple", seed: 0, turbo: false, prompt_enhance: false, cache_device: "default", cache_dtype: "default", lora_stack: [], diffusion_model_name: "qwen_image_2.1_int8_convrot.safetensors", seed_mode: "fixed", turbo_previous_settings: null },
});
const clone = value => JSON.parse(JSON.stringify(value));
const settingsDefaults = (value = {}) => ({ ...clone(DEFAULT_SETTINGS), ...value, generation: { ...clone(DEFAULT_SETTINGS.generation), ...value.generation, prompt_enhance: false } });

function readState(raw) {
    if (typeof raw !== "string" || raw.length > 256 * 1024) throw new Error("CG Creator state is too large.");
    const value = JSON.parse(raw);
    if (!value || Array.isArray(value) || value.schema_version !== 1) throw new Error("Unsupported CG Creator workflow state.");
    for (const key of ["background_id", "selected_image_id", "model_id", "selected_cg_id"]) {
        if (value[key] != null && (typeof value[key] !== "string" || (value[key] && !/^[a-f0-9]{32}$/.test(value[key])))) throw new Error("Invalid CG asset reference.");
    }
    if (value.selected_variant != null) validateVariant(value.selected_variant);
    if (value.characters != null) {
        if (!value.characters || typeof value.characters !== "object" || Array.isArray(value.characters)) throw new Error("Invalid CG character scenes.");
        for (const [id, characters] of Object.entries(value.characters)) {
            if (!/^[a-f0-9]{32}$/.test(id)) throw new Error("Invalid CG character scene.");
            validateCharacters(characters);
        }
    }
    if (value.camera != null && (!["position", "target", "up"].every(key => Array.isArray(value.camera[key]) && value.camera[key].length === 3 && value.camera[key].every(n => Number.isFinite(n) && Math.abs(n) <= 1e6))
        || !Number.isFinite(value.camera.fov) || value.camera.fov < 5 || value.camera.fov > 120)) throw new Error("Invalid saved CG camera.");
    if (value.frame != null && (![value.frame.width, value.frame.height].every(n => Number.isInteger(n) && n >= 64 && n <= 4096 && n % 16 === 0) || value.frame.width * value.frame.height > 16e6)) throw new Error("Invalid saved CG frame.");
    if (value.settings != null && (typeof value.settings !== "object" || Array.isArray(value.settings))) throw new Error("Invalid CG settings.");
    if (value.settings && Object.hasOwn(value.settings, "generation") && (!value.settings.generation || typeof value.settings.generation !== "object" || Array.isArray(value.settings.generation))) throw new Error("Invalid CG generation settings.");
    if (value.model_settings != null && (typeof value.model_settings !== "object" || Array.isArray(value.model_settings))) throw new Error("Invalid background model settings.");
    if (value.settings_revision != null && (!Number.isSafeInteger(value.settings_revision) || value.settings_revision < 0)) throw new Error("Invalid CG settings revision.");
    if (value.preview != null && (!["source", "cg", "pending"].includes(value.preview.kind)
        || value.preview.kind === "cg" && !/^[a-f0-9]{32}$/.test(value.preview.cg_id)
        || !Number.isFinite(value.preview.zoom) || value.preview.zoom < 1 || value.preview.zoom > 8
        || !Array.isArray(value.preview.center) || value.preview.center.length !== 2 || !value.preview.center.every(n => Number.isFinite(n) && Math.abs(n) <= 1e6))) throw new Error("Invalid saved CG preview.");
    if (value.scroll != null && (!value.scroll || ![value.scroll.sources, value.scroll.cgs].every(n => Number.isFinite(n) && n >= 0 && n <= 1e7))) throw new Error("Invalid saved CG scroll position.");
    if (value.job != null && (!(value.job.id ? /^[a-f0-9]{32}$/.test(value.job.id) : typeof value.job.request_id === "string" && /^[A-Za-z0-9_-]{1,160}$/.test(value.job.request_id)) || !["model", "cg", "expand"].includes(value.job.action)
        || typeof value.job.name !== "string" || value.job.name.length > 96
        || value.job.node_id != null && (typeof value.job.node_id !== "string" || !value.job.node_id || value.job.node_id.length > 64))) throw new Error("Invalid saved CG generation.");
    return { schema_version: 1, background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", ...value,
        settings: settingsDefaults(value.settings), model_settings: { ...MODEL_DEFAULTS, ...value.model_settings } };
}

function sourceCameraView(source) {
    const camera = clone(source), scale = Math.tan(camera.fov * Math.PI / 360) / Math.tan(Math.PI / 6);
    camera.position = camera.position.map((value, i) => camera.target[i] + (value - camera.target[i]) * scale);
    camera.fov = 60;
    return camera;
}

function frameDimensions(aspect, megapixels) {
    const [a, b] = String(aspect).split(":").map(Number), pixels = Number(megapixels) * 1e6;
    if (!(a > 0 && b > 0 && pixels > 0) || !Number.isFinite(pixels)) throw new Error("Invalid frame dimensions.");
    const width = Math.sqrt(pixels * a / b), height = Math.sqrt(pixels * b / a);
    const scale = Math.min(1, 4096 / Math.max(width, height), Math.sqrt(16e6 / pixels));
    const size = { width: Math.max(64, Math.round(width * scale / 16) * 16), height: Math.max(64, Math.round(height * scale / 16) * 16) };
    while (size.width * size.height > 16e6) size[size.width >= size.height ? "width" : "height"] -= 16;
    return size;
}

function validateVariant(value) {
    if (!value || typeof value !== "object" || Array.isArray(value) || !/^[a-f0-9]{32}$/.test(value.set_id) || !Number.isInteger(value.index) || value.index < 1 || value.index > 64
        || value.variant_id != null && (typeof value.variant_id !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(value.variant_id))
        || value.name != null && (typeof value.name !== "string" || value.name.length > 160)) throw new Error("Invalid background variant reference.");
    return value;
}
const variantKey = variant => variant ? `${variant.set_id}:${variant.index}` : "";
const sameSource = (record, state) => record.source_image_id === state.selected_image_id && variantKey(record.source_variant) === variantKey(state.selected_variant);
// Copied from BackgroundCreatorWidget.variantPreview; return the owning set along with its image.
function variantSource(document, imageId, card) {
    for (const group of [...(document?.generated_sets || [])].reverse()) {
        if (group.version !== "" || group.source_image_id !== imageId) continue;
        const index = group.variants.findIndex(saved => ["id", "time", "weather", "additional_details"].every(key => saved[key] === card[key]));
        if (index >= 0 && group.images[index + 1]?.url) return { group, index, image: group.images[index + 1] };
    }
    return null;
}
function sourceChoices(document) {
    const choices = [];
    for (const image of (document?.images || []).filter(image => image.image_id === document.selected_image_id)) {
        const title = "Original";
        choices.push({ key: image.image_id, image_id: image.image_id, variant: null, label: title, image });
        const latest = [...(document.generated_sets || [])].reverse().find(group => group.source_image_id === image.image_id && group.version === "");
        for (const card of document.editor?.variants || latest?.variants || []) {
            const source = variantSource(document, image.image_id, card);
            if (!source) continue;
            const { group, index } = source;
            const variant = { set_id: group.set_id, index: index + 1, ...(card.id ? { variant_id: card.id } : {}), ...(group.variants[index].name ? { name: group.variants[index].name } : {}) };
            choices.push({ key: `${image.image_id}:${variantKey(variant)}`, image_id: image.image_id, variant, label: card.name || `Variant ${index + 1}`, image: source.image });
        }
    }
    return choices;
}
function selectedSource(document, state) {
    const current = sourceChoices(document).find(choice => choice.image_id === state.selected_image_id && variantKey(choice.variant) === variantKey(state.selected_variant));
    if (current || !state.selected_variant) return current;
    // A saved workflow may pin an earlier render of the same variant.
    const variant = state.selected_variant, group = document?.generated_sets?.find(group => group.set_id === variant.set_id && group.source_image_id === state.selected_image_id);
    const image = group?.images?.[variant.index];
    return image?.url ? { key: `${state.selected_image_id}:${variantKey(variant)}`, image_id: state.selected_image_id, variant, label: variant.name || `Variant ${variant.index}`, image } : null;
}

// Resolve Utils assets relative to ComfyUI's extension mount, including proxy prefixes.
async function loadFactoryRuntime(descriptor) {
    if (descriptor?.error) throw new Error(descriptor.error);
    if (descriptor?.version !== 1 || !/^\/extensions\/[^/]+\/factory3d\/viewer\.js$/.test(descriptor.module_url)) {
        throw new Error("The installed 3D Factory viewer is unavailable.");
    }
    const root = new URL("../../", import.meta.url);
    const viewerURL = new URL(descriptor.module_url.slice(1), root);
    viewerURL.searchParams.set("v", "20261108.1");
    const [viewer, history] = await Promise.all([import(viewerURL.href), import(new URL("./editor_commands.mjs", viewerURL).href)]);
    const runtime = { Factory3DViewer: viewer.Factory3DViewer, FactoryCommandHistory: history.FactoryCommandHistory,
        viewerURL, stylesURL: new URL("./styles.css", viewerURL).href };
    if (typeof runtime.Factory3DViewer !== "function" || typeof runtime.FactoryCommandHistory !== "function") {
        throw new Error("Unsupported 3D Factory viewer interface.");
    }
    if (!document.getElementById("vnccs-3d-factory-styles")) {
        const link = document.createElement("link");
        link.id = "vnccs-3d-factory-styles"; link.rel = "stylesheet"; link.href = runtime.stylesURL;
        document.head.appendChild(link);
    }
    return runtime;
}

class CGQi2Controls {
    constructor(host, { node, api: transport, getSettings, onChange, onError = () => {}, onResolutionChange = () => {} }) {
        this.host = host; this.getSettings = getSettings; this.onChange = onChange; this.onError = onError;
        this.disposed = false; this.disabled = false; this.config = null; this.events = new AbortController();
        const api = { apiURL: value => transport.apiURL(value), user: transport.user,
            fetchApi: (path, options = {}) => transport.fetchApi(path, { ...options, signal: this.events.signal }) };

        host._node = this;
        const fallback = { resolution_mp: 2, generation: { steps: 25, cfg: 1, sampler: "euler", scheduler: "simple", seed: 0, cache_device: "default", cache_dtype: "default", lora_stack: [] } };
        const settings = () => getSettings() || fallback;
        const generation = () => settings().generation;
        const g = new Proxy({}, {
            get: (_target, key) => {
                const value = generation();
                if (key === "generation_mode") return "qi2";
                if (key === "target_size") return Number(settings().resolution_mp || 2) * 1024;
                if (key === "turbo_enabled") return !!value.turbo;
                if (key === "dmd_lora_name") return value.turbo_lora_name || QI2_TURBO_LORA_NAME;
                if (key === "dmd_lora_strength") return value.turbo ? 1 : 0;
                return value[key];
            },
            set: (_target, key, value) => {
                if (this.disposed || this.disabled) return true;
                const target = generation();
                if (key === "target_size") { settings().resolution_mp = resolutionScaleMegapixels(value); onResolutionChange(settings().resolution_mp); }
                else if (key === "turbo_enabled") target.turbo = value;
                else if (key === "dmd_lora_name") target.turbo_lora_name = value;
                else if (key !== "generation_mode" && key !== "dmd_lora_strength") target[key] = value;
                return true;
            },
        });
        const state = { gen_settings: g }, els = this.els = {}, ccState = { config: null, downloadStatus: {} };
        const CC_REPO_ID = "MIUProject/VNCCS_v3.0", modelPickerOpen = { qi2: false };
        let localAssets = { diffusion_models: [], loras: [] };
        const saveState = () => { if (!this.disposed && !this.disabled && getSettings()) onChange(); };
        const ccHasRequiredFamilies = config => (config?.models || []).some(entry => String(entry.kind || entry.Kind || "").toLowerCase() === "qi2");
        const isCcMounted = () => !this.disposed;

        const QI2_TURBO_LORA_NAME = "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors";
        const QI2_TURBO_MODEL_NAME = "qwen_image_2.1_turbo_int8_convrot.safetensors";
        const isQi2TurboModel = modelName => String(modelName || "").replace(/\\/g, "/").split("/").pop() === QI2_TURBO_MODEL_NAME;

        const ccNormalize = (value) => String(value || "").trim().toLowerCase();
        const ccKind = (entry) => ccNormalize(entry?.kind ?? entry?.Kind);
        const ccType = (entry) => ccNormalize(entry?.type ?? entry?.Type);
        const ccStatusKey = (cat, entry) => `cc_${cat}_${entry?.name || ""}`;
        const ccResolveStatus = (entry, cat) => {
            const transient = new Set(["queued", "downloading", "error", "auth_required"]);
            const dls = ccState.downloadStatus[ccStatusKey(cat, entry)] || {};
            return transient.has(dls.status) ? dls.status : (entry?.status || "missing");
        };
        const ccRelPath = (entry) => {
            const localPath = String(entry?.local_path || "").replace(/\\/g, "/");
            const parts = localPath.split("/").filter(Boolean);
            if (parts.length >= 3 && parts[0] === "models") return parts.slice(2).join("/");
            return parts[parts.length - 1] || "";
        };
        const ccEntries = (section, kind, predicate = null) => {
            const entries = ccState.config?.[section] || [];
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

        const FIELD_HELP = {
            target_size: "Sets the generated CG frame area from 1.0 to 4.0 megapixels while preserving its aspect ratio.",
            steps: "Number of sampling steps. Higher values can add detail but take longer.",
            sampler: "Sampling algorithm used to denoise the image.",
            cfg: "Prompt guidance strength. Higher values follow the prompt harder; too high can make images brittle.",
            scheduler: "Noise schedule used together with the sampler.",
            seed: "Numeric seed for reproducible generation. Reuse it to get similar results.",
            seed_mode: "Toggles fixed seed versus a fresh random seed for each generation.",
            cache_device: "KV cache location. Auto uses spare VRAM then RAM; CPU uses prefetched RAM; Off recomputes the prefix every step.",
            cache_dtype: "KV cache storage precision. Default is lossless; int8 halves its size; int4 uses about one quarter.",
            dmd_lora_name: "Turbo/DMD LoRA used by the active generation profile.",
            dmd_lora_strength: "Enables or disables the selected Turbo/DMD LoRA strength.",
            lora_stack: "Additional LoRAs mixed into generation for style refinements."
        };
        const helpFor = (key, fallback = "") => FIELD_HELP[key] || fallback;

        const createCompactNumberField = (lbl, key, min, max, step, targetObj = state.gen_settings) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-common-creator-gen-param-field";
            setHelpText(wrap, helpFor(key));
            wrap.innerHTML = `<div class="vnccs-common-creator-label">${lbl}</div>`;

            const input = document.createElement("input");
            input.type = "number";
            input.className = "vnccs-common-creator-gen-param-input";
            input.min = min;
            input.max = max;
            input.step = step;
            input.value = targetObj[key];
            input.oninput = (e) => {
                const value = e.target.valueAsNumber;
                if (!e.target.checkValidity() || !Number.isFinite(value)) {
                    return;
                }
                targetObj[key] = value;
                saveState();
            };

            input.onblur = () => { input.value = targetObj[key]; };
            wrap.appendChild(input);
            els[key] = { num: input };
            return wrap;
        };

        const createCompactSelectField = (lbl, key, targetObj = state.gen_settings) => {
            const wrap = document.createElement("div");
            wrap.className = "vnccs-common-creator-gen-param-field";
            setHelpText(wrap, helpFor(key));
            wrap.innerHTML = `<div class="vnccs-common-creator-label">${lbl}</div>`;

            const select = document.createElement("select");
            select.className = "vnccs-common-creator-gen-param-select";
            select.onchange = (e) => {
                targetObj[key] = e.target.value;
                saveState();
            };

            wrap.appendChild(select);
            els[key] = select;
            return wrap;
        };

        const cardStatusLabel = (status, entry, cat) => {
            const dls = ccState.downloadStatus[ccStatusKey(cat, entry)] || {};
            if (status === "installed") return "Installed";
            if (status === "queued") return "Queued";
            if (status === "downloading") return dls.message || "Downloading";
            if (status === "auth_required") return "Key Required";
            if (status === "error") return "Error";
            return "Missing";
        };

        const buildAssetCard = ({ entry, cat, selectedValue, onSelect, pickerHead = false, onDownload = null }) => {
            const status = ccResolveStatus(entry, cat);
            const rel = ccRelPath(entry);
            const installed = status === "installed";
            const selected = selectedValue && rel && selectedValue.replace(/\\/g, "/") === rel;
            const progress = ["queued", "downloading"].includes(status);

            const card = document.createElement("div");
            card.className = "vnccs-common-creator-model-card";
            card.dataset.focusKey = `asset-${cat}-${rel}-${pickerHead ? "head" : "option"}`;
            card.dataset.qi2Setting = cat === "models" ? "generation.diffusion_model_name" : "generation.turbo";
            if (pickerHead) card.dataset.sakuraFocusDefault = "true";
            card.classList.toggle("is-picker-head", pickerHead);
            card.classList.toggle("is-installed", installed);
            card.classList.toggle("is-selected", (cat !== "lora" || installed) && Boolean(selected));
            card.classList.toggle("is-missing", !installed);
            if (installed || pickerHead) {
                card.onclick = () => onSelect?.(rel, entry);
                card.role = "button";
                card.tabIndex = 0;
                card.ariaPressed = String(Boolean(selected));
                card.ariaLabel = entry.name || rel || "Model";
                card.onkeydown = event => {
                    if (event.target !== card || !["Enter", " "].includes(event.key)) return;
                    event.preventDefault(); event.stopPropagation();
                    if (!event.repeat) card.onclick();
                };
            }

            const top = document.createElement("div");
            top.className = "vnccs-common-creator-model-card-top";

            const badge = document.createElement("span");
            badge.className = "vnccs-common-creator-model-card-badge " + (installed ? "ok" : progress ? "progress" : "missing");
            top.appendChild(badge);

            const name = document.createElement("div");
            name.className = "vnccs-common-creator-model-card-name";
            name.textContent = entry.name || rel || "Unknown";
            top.appendChild(name);

            const statusEl = document.createElement("div");
            statusEl.className = "vnccs-common-creator-model-card-status " + (installed ? "ok" : progress ? "progress" : "missing");
            statusEl.textContent = cardStatusLabel(status, entry, cat);
            top.appendChild(statusEl);

            card.appendChild(top);

            if (entry.description) {
                const desc = document.createElement("div");
                desc.className = "vnccs-common-creator-model-card-desc";
                desc.textContent = entry.description;
                card.appendChild(desc);
            }

            if (!installed && entry.source !== "local") {
                const actions = document.createElement("div");
                actions.className = "vnccs-common-creator-model-card-actions";
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "vnccs-common-creator-model-card-download";
                btn.textContent = status === "auth_required" ? "Enter Key in Control Center" : "Download";
                btn.disabled = progress;
                btn.dataset.qi2Pending = String(progress);
                btn.onclick = (event) => {
                    event.stopPropagation();
                    if (!progress && status !== "auth_required") (onDownload || ccDownloadEntry)(cat, entry);
                };
                actions.appendChild(btn);
                card.appendChild(actions);
            }

            return card;
        };

        const renderModelPicker = ({ containerEl, entries, cat, key, mode, emptyText, onSelect, onDownload = null }) => {
            if (!containerEl) return;
            containerEl.innerHTML = "";
            const picker = document.createElement("div");
            picker.className = "vnccs-common-creator-model-picker";
            picker.dataset.mode = mode;
            picker.dataset.sakuraFocusGroup = mode;
            picker.classList.toggle("is-open", !!modelPickerOpen[mode]);
            containerEl.appendChild(picker);

            const current = String(state.gen_settings[key] || "").replace(/\\/g, "/");
            if (current && !entries.some(entry => ccRelPath(entry) === current)) {
                entries = [...entries, { name: current, local_path: `models/diffusion_models/${current}`, status: "missing", source: "local",
                    description: "Saved model is unavailable. Its selection is retained." }];
            }
            if (!entries.length) {
                const empty = document.createElement("div");
                empty.className = "vnccs-common-creator-model-card-desc";
                empty.textContent = emptyText;
                picker.appendChild(empty);
                return;
            }

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
            menu.className = "vnccs-common-creator-model-picker-menu";
            menu.dataset.sakuraScroll = `model-picker-${mode}`;
            picker.appendChild(menu);

            const appendGroup = (title, groupEntries) => {
                if (!groupEntries.length) return;
                const group = document.createElement("div");
                group.className = "vnccs-common-creator-model-picker-group";
                const groupTitle = document.createElement("div");
                groupTitle.className = "vnccs-common-creator-model-picker-group-title";
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

        const syncGenerationControls = () => {
            const before = els._ui?.captureValues([els.steps?.num, els.cfg?.num, els.seed]);
            const g = state.gen_settings;
            ensureOption(els.sampler, g.sampler);
            ensureOption(els.scheduler, g.scheduler);
            if (els.sampler) els.sampler.value = g.sampler || "";
            if (els.scheduler) els.scheduler.value = g.scheduler || "";
            if (els.cache_device) els.cache_device.value = g.cache_device === "default" ? "auto" : g.cache_device;
            if (els.cache_dtype) els.cache_dtype.value = g.cache_dtype;
            if (els.target_size) {
                els.target_size.value = resolutionScaleMegapixels(g.target_size).toFixed(1);
            }
            if (els.target_size_value) {
                els.target_size_value.textContent = resolutionScaleText(g.target_size);
            }
            syncSliderValue(els.steps, g.steps);
            syncSliderValue(els.cfg, g.cfg);
            if (els.seed) els.seed.value = g.seed ?? "";
            if (els.seed_mode) {
                const randomMode = (g.seed_mode || "fixed") === "randomize";
                els.seed_mode.classList.toggle("is-active", randomMode);
                els.seed_mode.title = randomMode ? "Random seed" : "Fixed seed";
                els.seed_mode.setAttribute("aria-pressed", randomMode ? "true" : "false");
                els.seed_mode.setAttribute("aria-label", randomMode ? "Use fixed seed" : "Use random seed");
                els.seed?.classList.toggle("is-random", randomMode);
            }
            renderControlCenterCards();
            els._ui?.animateValues(before);
            els._ui?.syncSelects();
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

        const selectQi2Model = (rel) => {
            if (!rel) return;
            const turboModel = isQi2TurboModel(rel);
            const previousTurboModel = isQi2TurboModel(state.gen_settings.diffusion_model_name);
            if (rel !== state.gen_settings.diffusion_model_name && (turboModel || previousTurboModel)) {
                state.gen_settings.steps = turboModel ? 8 : 25;
                state.gen_settings.cfg = 1.0;
                state.gen_settings.turbo_enabled = false;
                state.gen_settings.turbo_previous_settings = null;
            }
            modelPickerOpen.qi2 = false;
            selectCcAsset("diffusion_model_name", rel);
            syncGenerationControls();
        };

        const downloadQi2Bundle = async (cat, entry) => {
            await ccDownloadEntry(cat, entry);
            const clip = ccFirstEntry("clip", "QI2");
            const vae = ccFirstEntry("vae", "QI2");
            if (clip && ccResolveStatus(clip, "clip") !== "installed") await ccDownloadEntry("clip", clip);
            if (vae && ccResolveStatus(vae, "vae") !== "installed") await ccDownloadEntry("vae", vae);
        };

        const ccDownloadEntry = async (cat, entry) => {
            if (!isCcMounted() || !entry?.name) return;
            const key = ccStatusKey(cat, entry);
            ccState.downloadStatus[key] = { status: "queued", message: "Queued…" };
            renderControlCenterCards();
            try {
                const response = await api.fetchApi("/vnccs/control_center/download", {
                    method: "POST",
                    headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                    body: JSON.stringify({ repo_id: CC_REPO_ID, category: cat, name: entry.name }),
                });
                const payload = await response.json();
                if (!isCcMounted()) return;
                if (!response.ok || payload.error) {
                    ccState.downloadStatus[key] = { status: "error", message: payload.error || "Download failed" };
                    renderControlCenterCards();
                    return;
                }
                startCcPolling();
            } catch (error) {
                if (!isCcMounted()) return;
                ccState.downloadStatus[key] = { status: "error", message: String(error?.message || error) };
                renderControlCenterCards();
            }
        };

        const renderControlCenterCards = () => {
            if (this.disposed) return;
            const catalog = ccEntries("models", "QI2", entry => ccType(entry) === "unet");
            const presets = (this.config?.model_presets || []).filter(entry => !catalog.some(saved => ccRelPath(saved) === ccRelPath(entry)));
            const entries = mergeCcAndLocalEntries([...catalog, ...presets], localAssets.diffusion_models, "diffusion_models", "unet", "QI2");
            renderModelPicker({ containerEl: els.qi2ModelCards, entries, cat: "models", key: "diffusion_model_name", mode: "qi2",
                emptyText: "No Qwen Image 2.1 diffusion models found.", onSelect: selectQi2Model, onDownload: downloadQi2Bundle });
            for (const control of els.qi2ModelCards.querySelectorAll('[role="button"]')) control.dataset.qi2Setting = "generation.diffusion_model_name";
            this.setDisabled(this.disabled);
        };
        const { fetchConfig: fetchCcConfig, startPolling: startCcPolling, stopPolling: stopCcPolling } = createControlCenterClient(CC_REPO_ID, ccHasRequiredFamilies, ccState, renderControlCenterCards, this);
        const colRight = this.colRight = document.createElement("div"); colRight.className = "bg-qi2-contents";

        const qi2Models = document.createElement("div");
        qi2Models.className = "vnccs-common-creator-subsection";
        qi2Models.style.display = "flex";
        els.qi2Models = qi2Models;
        const qi2ModelCards = document.createElement("div");
        qi2ModelCards.className = "vnccs-common-creator-model-card-list";
        els.qi2ModelCards = qi2ModelCards;
        qi2Models.appendChild(qi2ModelCards);
        colRight.appendChild(qi2Models);

        const resolutionWrap = document.createElement("label");
        resolutionWrap.className = "vnccs-common-creator-field vnccs-common-creator-resolution-field";
        setHelpText(resolutionWrap, helpFor("target_size"));
        const resolutionHead = document.createElement("div");
        resolutionHead.className = "vnccs-common-creator-resolution-head";
        const resolutionLabel = document.createElement("div");
        resolutionLabel.className = "vnccs-common-creator-label";
        resolutionLabel.textContent = "Resolution scale";
        const resolutionValue = document.createElement("div");
        resolutionValue.className = "vnccs-common-creator-resolution-value";
        resolutionHead.append(resolutionLabel, resolutionValue);
        const resolutionSlider = document.createElement("input");
        resolutionSlider.type = "range";
        resolutionSlider.className = "vnccs-common-creator-resolution-slider";
        resolutionSlider.min = String(RESOLUTION_SCALE_MIN_MP);
        resolutionSlider.max = String(RESOLUTION_SCALE_MAX_MP);
        resolutionSlider.step = String(RESOLUTION_SCALE_STEP_MP);
        resolutionSlider.setAttribute("aria-label", "Resolution scale in megapixels");
        resolutionSlider.value = resolutionScaleMegapixels(state.gen_settings.target_size).toFixed(1);
        resolutionValue.textContent = resolutionScaleText(state.gen_settings.target_size);
        resolutionSlider.oninput = () => {
            state.gen_settings.target_size = resolutionScaleValue(resolutionSlider.value);
            resolutionValue.textContent = resolutionScaleText(state.gen_settings.target_size);
            saveState();
        };
        els.target_size = resolutionSlider;
        els.target_size_value = resolutionValue;
        resolutionWrap.append(resolutionHead, resolutionSlider);
        colRight.appendChild(resolutionWrap);

        const genParamGrid = document.createElement("div");
        genParamGrid.className = "vnccs-common-creator-gen-param-grid";
        genParamGrid.appendChild(createCompactNumberField("Steps", "steps", 1, 150, 1));
        genParamGrid.appendChild(createCompactSelectField("Sampler", "sampler", state.gen_settings));
        genParamGrid.appendChild(createCompactNumberField("CFG", "cfg", 0, 30, 0.1));
        genParamGrid.appendChild(createCompactSelectField("Scheduler", "scheduler", state.gen_settings));
        colRight.appendChild(genParamGrid);

        // SEED Section (Rebalanced)
        const seedWrap = document.createElement("div"); seedWrap.className = "vnccs-common-creator-field";
        setHelpText(seedWrap, helpFor("seed"));
        seedWrap.innerHTML = '<div class="vnccs-common-creator-label">Seed</div>';

        const seedRow = document.createElement("div");
        seedRow.className = "vnccs-common-creator-seed-row";

        const seedInp = document.createElement("input"); seedInp.className = "vnccs-common-creator-gen-param-input";
        seedInp.type = "number"; seedInp.value = state.gen_settings.seed;
        seedInp.oninput = (e) => {
            const value = e.target.valueAsNumber;
            if (!e.target.checkValidity() || !Number.isFinite(value)) {
                return;
            }
            state.gen_settings.seed = value;
            saveState();
        };
        seedInp.min = 0; seedInp.step = 1;
        seedInp.onblur = () => { seedInp.value = state.gen_settings.seed; };
        els.seed = seedInp;

        const seedMode = document.createElement("button");
        seedMode.type = "button";
        seedMode.className = "vnccs-common-creator-seed-dice-btn";
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
            if (state.gen_settings.seed_mode === "randomize") state.gen_settings.seed = generateRandomSeed();
            syncGenerationControls();
            saveState();
        };
        els.seed_mode = seedMode;

        seedRow.appendChild(seedInp);
        seedRow.appendChild(seedMode);

        seedWrap.appendChild(seedRow);
        colRight.appendChild(seedWrap);

        const modelPanel = document.createElement("section"); modelPanel.className = "bg-panel bg-qi2-model-panel vnccs-common-creator-col";
        const modelTitle = document.createElement("div"); modelTitle.className = "vnccs-common-creator-section-title"; modelTitle.textContent = "Qwen Image 2.1 model & frame";
        modelPanel.append(modelTitle, qi2Models, resolutionWrap);
        const frameSettings = host.querySelector("[data-frame-settings]"); if (frameSettings) modelPanel.appendChild(frameSettings);
        const cacheSettings = document.createElement("div"); cacheSettings.className = "bg-qi2-cache";
        cacheSettings.setAttribute("role", "group"); cacheSettings.setAttribute("aria-label", "Qwen Image 2.1 Cache");
        for (const [key, label, choices] of [["cache_device", "Cache device", ["auto", "gpu", "cpu", "off"]],
            ["cache_dtype", "Cache dtype", ["default", "int8", "int4"]]]) {
            cacheSettings.appendChild(createCompactSelectField(label, key));
            els[key].replaceChildren(...choices.map(value => new Option(value, value)));
        }
        modelPanel.appendChild(cacheSettings);
        const samplingPanel = document.createElement("section"); samplingPanel.className = "bg-panel bg-qi2-sampling-panel vnccs-common-creator-col";
        const samplingTitle = document.createElement("div"); samplingTitle.className = "vnccs-common-creator-section-title"; samplingTitle.textContent = "Sampling";
        samplingPanel.append(samplingTitle, genParamGrid, seedWrap);
        colRight.replaceChildren(modelPanel, samplingPanel);

        host.classList.add("vnccs-common-creator-container"); host.appendChild(colRight);
        attachHelpTooltips(host);
        for (const [key, control] of Object.entries({ steps: els.steps.num, cfg: els.cfg.num, sampler: els.sampler, scheduler: els.scheduler,
            seed: els.seed, seed_mode: els.seed_mode, cache_device: els.cache_device, cache_dtype: els.cache_dtype })) {
            control.dataset.qi2Setting = `generation.${key}`; control.setAttribute("aria-label", key.replaceAll("_", " "));
        }
        els.target_size.dataset.qi2Setting = "resolution_mp";
        this.renderControls = config => {
            localAssets = { diffusion_models: (config.diffusion_models || []), loras: config.loras || [] };
            if (!localAssets.diffusion_models.length) localAssets.diffusion_models = (config.assets || []).filter(asset => asset.category === "diffusion_models" && asset.installed).map(asset => asset.name);
            for (const [control, values] of [[els.sampler, config.samplers || []], [els.scheduler, config.schedulers || []]]) {
                control.replaceChildren(...values.map(value => new Option(value || "None", value)));
            }
            syncGenerationControls();
            if (!this.catalogStarted) { this.catalogStarted = true; fetchCcConfig().catch(error => {
                if (this.disposed) return;
                stopCcPolling(); renderControlCenterCards();
                if (!localAssets.diffusion_models.length) onError(error);
            }); }
        };
    }
    render(config) { if (!this.disposed) { this.config = config; this.renderControls(config); } }
    setDisabled(value) {
        this.disabled = Boolean(value); this.host.inert = this.disabled;
        for (const control of this.host.querySelectorAll("input,select,button")) control.disabled = this.disabled || control.dataset.qi2Pending === "true";
    }
    dispose() {
        if (this.disposed) return; this.disposed = true; this.events.abort();
        this.onRemoved?.(); this.colRight.remove();
    }
}


const BASE = "/vnccs/cg-creator";
const MODEL_NAMES = { triposplat: "TripoSplat", pixal3d: "Pixal3D", worldmirror: "WorldMirror", moge: "MoGe" };
const CAMERA_HINT = "Drag to look · Middle-drag to pan · Scroll to move · Ctrl Z to undo";
const ASPECTS = ["16:9", "9:16", "4:3", "3:4", "1:1", "21:9", "3:2", "2:3"];
const latestModel = (doc, state) => doc?.models?.filter(model => sameSource(model, state)).at(-1);
const imageCGs = doc => [...(doc?.cgs || [])];

class CGCreatorWidget {
    constructor(node, api) {
        injectStyles(STYLE, "vnccs-cg-creator"); this.node = node; this.api = api;
        this.state = null; this.document = null; this.scene = null; this.viewer = null; this.catalog = [];
        this.epoch = 0; this.restoring = true; this.busy = false; this.disposed = false;
        this.events = new AbortController(); this.operations = Promise.resolve(); this.dialogs = new Set();
        this.history = null;
        this.container = document.createElement("div"); this.container.className = "vnccs-i3s vnccs-cg-creator vnccs-common-creator-container vnccs-common-ui-host";
        this.frame = document.createElement("div"); this.frame.className = "vnccs-cg-widget"; this.frame.appendChild(this.container);
        this.container.innerHTML = `<section class="cg-workspace">
                <aside class="bg-panel bg-model-source vnccs-common-creator-col">
                    <div class="vnccs-common-creator-section-title">Background select</div>
                    <select class="vnccs-common-creator-select" data-library aria-label="Saved background"></select>
                    <div class="bg-preview vnccs-common-creator-preview-container"><img data-preview data-model-source alt="Selected background" hidden>
                        <span class="vnccs-common-creator-placeholder" data-empty><span class="vnccs-common-creator-placeholder-title">Select a background</span><span class="vnccs-common-creator-placeholder-hint">Choose a saved background and one of its variants.</span></span>
                        <button class="bg-enlarge" data-enlarge aria-label="Open full-size background preview" hidden></button></div>
                    <div class="vnccs-common-creator-section-title">Variants</div>
                    <div class="bg-variation-strip"><button type="button" class="vnccs-common-creator-sprite-nav-btn bg-variation-scroll" data-scroll-source="-1" aria-label="Scroll variants left"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 12H5m6-6-6 6 6 6"/></svg></button><div class="bg-variation-cards" data-source-choices data-sakura-scroll="cg-source-variants" role="group" aria-label="Background variants"></div><button type="button" class="vnccs-common-creator-sprite-nav-btn bg-variation-scroll" data-scroll-source="1" aria-label="Scroll variants right"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6"/></svg></button></div>
                    <div class="bg-row bg-model-mode"><div class="vnccs-common-creator-section-title">Model generation</div><select class="vnccs-common-creator-select" data-model-provider aria-label="Model generator"><option value="triposplat">TripoSplat</option><option value="pixal3d" selected>Pixal3D</option><option value="worldmirror">WorldMirror</option><option value="moge">MoGe</option></select><label data-moge-refine-label hidden title="Refine views captured from an initial MoGe scene with the CG LoRA"><input type="checkbox" data-moge-refine-views>Refine views</label></div>
                    <button class="vnccs-common-creator-model-card is-installed bg-model-card" data-model-settings aria-label="Pixal3D settings"><span class="vnccs-common-creator-model-card-top"><span class="vnccs-common-creator-model-card-badge ok"></span><span class="vnccs-common-creator-model-card-name">Pixal3D</span><span class="vnccs-common-creator-model-card-status" data-model-status>Model not created</span></span><span class="vnccs-common-creator-model-card-desc">Textured GLB mesh</span></button>
                    <button class="vnccs-common-creator-btn vnccs-common-creator-btn-primary bg-wide" data-action="model">Create model</button>
                </aside>
                <section class="cg-middle">
                <section class="bg-panel bg-viewport-panel vnccs-common-creator-col"><div class="vnccs-common-creator-section-title">Model viewport <span class="bg-badge" data-model-kind>Raw splat</span></div><div class="bg-camera-toolbar"><label>FOV <input type="range" min="5" max="120" step="0.1" value="60" class="vnccs-common-creator-slider" data-fov aria-label="Camera field of view"><input type="number" min="5" max="120" step="0.1" value="60" class="vnccs-common-creator-input" data-fov-number aria-label="Exact camera field of view"></label><button class="vnccs-common-creator-model-card-download" data-add-character>Add Character</button><button class="vnccs-common-creator-btn vnccs-common-creator-btn-primary" data-save-view>Generate CG</button><button class="vnccs-common-creator-model-card-download" data-expand-scene title="Create the model to save reconstruction frames">Expand Scene</button><button class="vnccs-common-creator-model-card-download" data-recenter>Recenter</button></div><div class="bg-viewport" data-vnccs-own-navigation><div data-viewer></div><div class="bg-viewer-empty" data-viewer-empty>Select a saved background and create its model.</div><div class="bg-expansion-compare" data-expansion-compare hidden><div class="bg-expansion-images"><img data-expansion-before alt="Captured view"><img data-expansion-after alt="QI2 reconstruction frame"><input type="range" min="0" max="100" value="50" class="vnccs-common-creator-slider" data-expansion-split aria-label="Compare captured view and QI2 frame"></div></div><div class="bg-expansion-actions" data-expansion-frame-actions hidden><button class="vnccs-common-creator-btn" data-discard-expansion-frame>Discard</button><button class="vnccs-common-creator-btn vnccs-common-creator-btn-primary" data-approve-expansion-frame>Rebuild Scene</button></div><div class="bg-expansion-actions" data-expansion-actions hidden><button class="vnccs-common-creator-btn" data-discard-expansion>Discard</button><button class="vnccs-common-creator-btn vnccs-common-creator-btn-primary" data-save-expansion>Save Scene</button></div></div><div class="bg-hint">Drag to look · Middle-drag to pan · Scroll to move · Ctrl Z to undo</div></section>
                    <div class="bg-generation-dock" data-qi2><div class="bg-frame-settings" data-frame-settings>
                    <label class="bg-field vnccs-common-creator-field"><span class="vnccs-common-creator-label">Aspect ratio</span><select class="vnccs-common-creator-select" data-aspect aria-label="Aspect ratio">${ASPECTS.map(value => `<option>${value}</option>`).join("")}</select></label>
                    </div></div>
                </section>
                <aside class="bg-panel bg-cg-panel vnccs-common-creator-col"><div class="vnccs-common-creator-section-title">Generated CG <span class="bg-badge" data-cg-count>0</span></div><div class="bg-cg-list" data-cgs></div><p data-cg-empty>Compose the camera and press Generate CG.</p></aside>
            </section>`;
        this.host = this.container.querySelector("[data-viewer]"); this.status = this.container.querySelector(".bg-hint"); this.status.setAttribute("role", "status");
        this.expansionCompare = this.container.querySelector("[data-expansion-compare]");
        this.expansionBefore = this.container.querySelector("[data-expansion-before]");
        this.expansionAfter = this.container.querySelector("[data-expansion-after]");
        this.expansionSplit = this.container.querySelector("[data-expansion-split]");
        this.on(this.expansionCompare, "pointermove", event => {
            if (this.expansionCompare.hidden || this.expansionSplit.disabled || event.pointerType === "touch") return;
            const { left, width } = this.expansionBefore.getBoundingClientRect();
            if (width <= 0) return;
            const split = Math.round(100 * Math.max(0, Math.min(1, (event.clientX - left) / width)));
            this.expansionSplit.value = String(split);
            this.expansionCompare.style.setProperty("--split", split + "%");
        });
        for (const image of [this.expansionBefore, this.expansionAfter]) {
            this.on(image, "load", () => this.updateControls());
            this.on(image, "error", () => {
                if (!this.expansionCompare.hidden) this.message("The expansion comparison could not be loaded.", true);
                this.updateControls();
            });
        }
        this.fov = this.container.querySelector("[data-fov]"); this.fovNumber = this.container.querySelector("[data-fov-number]");
        this.customSelects = createSakuraInteractions(this.container, node);
        this.canvasNavigation = enableMiddleMouseCanvasPan(this.container, node);
        this.qi2 = new CGQi2Controls(this.container.querySelector("[data-qi2]"), { node, api, getSettings: () => this.state?.settings,
            onChange: () => this.markEdited(),
            onError: error => this.message(String(error.message || error), true), onResolutionChange: () => this.updateFrame() });
        this.qi2.els._ui = this.customSelects;
        this.on(this.container, "click", event => this.click(event)); this.on(this.container, "input", event => this.input(event)); this.on(this.container, "change", event => this.change(event));
        for (const type of ["change", "blur", "keyup", "pointerup", "pointercancel"]) this.on(this.container, type, () => this.commitGesture());
        this.on(this.host, "pointerdown", () => this.beginCameraNavigation(), { capture: true });
        this.on(this.host, "pointermove", () => { if (this.canEditCamera() && this.history?.pending) this.sync(); });
        for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) this.on(this.host, type, () => this.commitGesture());
        this.on(this.host, "wheel", () => { this.beginCameraNavigation(); clearTimeout(this.wheelTimer); this.wheelTimer = setTimeout(() => this.commitGesture(), 180); }, { capture: true, passive: true });
        this.on(this.container, "keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.target.closest("input,textarea")) { event.preventDefault(); if (this.canEditCamera()) this.history[event.shiftKey ? "redo" : "undo"](); } });
        this.on(this.container.querySelector("[data-source-choices]"), "scroll", () => { this.updateSourceScroll(); if (!this.restoring) this.sync(); }, { passive: true });
        this.on(this.container.querySelector("[data-cgs]"), "scroll", () => { if (!this.restoring) this.sync(); }, { passive: true });
        this.on(window, "beforeunload", event => {
            this.commitGesture();
            if (this.draftFailed) { event.preventDefault(); event.returnValue = ""; }
        });
        watchConnection(node, () => this.refreshLibrary(), registerCleanup);
        this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(this.frame); this.updateControls();
    }

    on(target, type, callback, options = {}) { target.addEventListener(type, callback, { ...options, signal: this.events.signal }); }
    message(text, error = false, showText = false) { if (!this.disposed) { this.status.textContent = error || showText ? text : this.modelError || this.libraryError || (this.characters?.entries.size ? "Drag joints to pose · Select a joint to rotate · Right-drag to look · Ctrl Z to undo" : CAMERA_HINT); this.status.classList.toggle("error", error || Boolean(this.modelError || this.libraryError)); } }
    async request(path, options = {}) {
        const response = await this.api.fetchApi(BASE + path, { ...options, signal: this.events.signal }), result = await response.json();
        if (!response.ok) { const error = new Error(result.error || `Request failed: HTTP ${response.status}`); error.status = response.status; throw error; }
        return result;
    }
    json(path, data, method = "POST") { return this.request(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }); }
    selectedScene() { return this.scene?.model_id === this.state?.model_id && this.scene?.background_id === this.state?.background_id ? this.scene : null; }
    expansionURL(id) { return `/backgrounds/${this.state.background_id}/models/${this.state.model_id}/expansion/${id}`; }
    canEditCamera() { return !this.disposed && !this.restoring && !this.busy && !this.state?.job && !this.loadingModel && !this.previewSource && Boolean(this.viewer && this.selectedScene() && !this.scene.expansion_frame_id); }
    editorSnapshot() { return { camera: this.viewer.getCameraState(), characters: this.characters?.snapshot() || [] }; }
    beginGesture() { if (this.canEditCamera()) this.history.begin("Scene", this.editorSnapshot()); }
    beginCameraNavigation() {
        if (!this.canEditCamera()) return;
        const scene = this.selectedScene(), viewer = this.viewer;
        const provider = scene.background_settings?.provider;
        const moge = provider === "moge", worldmirror = provider === "worldmirror";
        const bounds = (moge || worldmirror) ? viewer.objects?.get(scene.objects[0]?.object_id)?.localBounds : null;
        const snapshot = this.editorSnapshot();
        let panSpeed = 1, zoomSensitivity = 0.1;
        this.worldMirrorDollyStep = 0;
        if (bounds && !bounds.isEmpty()) {
            const { min, max } = bounds;
            const { position, target } = snapshot.camera;
            const distance = Math.hypot(...target.map((value, axis) => value - position[axis]));
            // Distant geometry can stretch the bounds far beyond the foreground.
            const maxReference = moge && this.mogeUnit > 0 ? 25 * this.mogeUnit : Infinity;
            const reference = Math.min(maxReference, Math.max(max.x - min.x, max.y - min.y, max.z - min.z) / 2);
            if (worldmirror && Number.isFinite(reference) && reference > 0) {
                panSpeed = reference / Math.max(0.001, distance);
                this.worldMirrorDollyStep = reference * 0.1;
            } else if (moge) {
                const zoomDistance = Math.max(0.1, distance);
                panSpeed = Math.max(1, reference / Math.max(0.001, distance));
                zoomSensitivity *= Math.min(maxReference, Math.max(zoomDistance, reference)) / zoomDistance;
            }
        }
        viewer.controls.panSpeed = panSpeed;
        viewer.setZoomSensitivity(zoomSensitivity, { emit: false });
        viewer.options.cameraDragScale = moge ? 1.4 : 0.7;
        this.history.begin("Scene", snapshot);
    }
    dollyWorldMirrorCamera(amount, options) {
        const { position, target } = this.viewer.getCameraState();
        const forward = target.map((value, axis) => value - position[axis]);
        const distance = Math.hypot(...forward);
        if (!amount || !distance) return;
        // The look target is one unit from the source camera regardless of the splat scale.
        const travel = Math.max(-10, Math.min(10, amount)) * this.worldMirrorDollyStep;
        const move = point => point.map((value, axis) => value + forward[axis] * travel / distance);
        this.viewer.setCameraState({ position: move(position), target: move(target) }, options);
    }
    rotateWorldMirrorCamera({ yaw = 0, pitch = 0 }, { emit = true } = {}) {
        const { camera, controls } = this.viewer;
        const up = camera.up;
        const forward = controls.target.clone().sub(camera.position);
        const distance = Math.max(0.001, forward.length());
        const radians = Math.PI / 180;
        forward.normalize().applyAxisAngle(up, yaw * radians);
        const currentPitch = Math.asin(Math.max(-1, Math.min(1, forward.dot(up))));
        const nextPitch = Math.max(-89 * radians, Math.min(89 * radians, currentPitch + pitch * radians));
        const right = forward.clone().cross(up);
        if (right.lengthSq() > 1e-12) forward.applyAxisAngle(right.normalize(), nextPitch - currentPitch);
        const target = camera.position.clone().addScaledVector(forward, distance).toArray();
        this.viewer.setCameraState({ target }, { emit });
    }
    commitGesture() {
        if (this.canEditCamera()) this.sync({ sceneChanged: this.history.commit(this.editorSnapshot()) });
    }
    async ensureCharacters() {
        if (this.characters) return this.characters;
        this.characterRuntime ||= loadCharacterRuntime(this.factoryRuntime.viewerURL).catch(error => { this.characterRuntime = null; throw error; });
        const runtime = await this.characterRuntime;
        if (this.disposed) throw new Error("CG Creator was removed.");
        this.characters ||= new CGCharacterEditor(this.viewer, runtime, {
            canEdit: () => this.canEditCamera() && !this.document?._recovery && !this.document?.archived,
            onStart: () => this.beginGesture(), onChange: () => { this.sync(); this.characterPanel?.refresh(); }, onEnd: () => { this.commitGesture(); this.characterPanel?.refresh(); },
        });
        this.characterPanel ||= new CGCharacterPanel(this);
        return this.characters;
    }
    captureScene(options, mode) {
        if (options.cameraState && this.selectedScene()?.background_settings?.provider === "moge") {
            const { cameraState, ...liveOptions } = options;
            const current = this.viewer.getCameraState();
            if (Math.abs(cameraState.fov - current.fov) < 1e-8 && ["position", "target", "up"].every(key =>
                cameraState[key].every((value, axis) => Math.abs(value - current[key][axis]) < 1e-8))) {
                // The live capture camera inherits the viewport's MoGe clipping planes.
                options = liveOptions;
            }
        }
        return this.characters ? this.characters.capture(options, mode) : this.viewer.capturePreview(options);
    }
    updateControls() {
        for (const control of this.container.querySelectorAll("button,input,select,textarea")) control.disabled = this.busy || this.restoring || Boolean(this.state?.job || this.document?._recovery || this.document?.archived);
        this.container.querySelector("[data-library]").disabled = this.busy || this.restoring || Boolean(this.state?.job);
        for (const control of this.container.querySelectorAll("[data-cg-id],[data-close-preview]")) control.disabled = this.restoring || Boolean(this.backgroundTransition);
        for (const control of this.container.querySelectorAll("[data-pending-cg]")) control.disabled = this.restoring || !this.pendingCG?.url;
        for (const control of this.container.querySelectorAll("[data-fov],[data-fov-number],[data-recenter]")) control.disabled = !this.canEditCamera();
        this.container.querySelector("[data-add-character]").disabled = !this.canEditCamera() || Boolean(this.document?._recovery || this.document?.archived);
        this.container.querySelector("[data-add-character]").textContent = this.characters?.entries.size ? "Edit Character" : "Add Character";
        this.characterPanel?.refresh();
        this.container.querySelector("[data-save-view]").disabled = !this.canEditCamera() || Boolean(this.document?._recovery || this.document?.archived);
        const expand = this.container.querySelector("[data-expand-scene]");
        const savedScene = this.selectedScene();
        expand.hidden = !["moge", "worldmirror"].includes(savedScene?.background_settings?.provider);
        expand.disabled ||= !this.canEditCamera();
        if (savedScene?.can_expand) expand.removeAttribute("title");
        else expand.setAttribute("title", "Recreate this model to save its reconstruction frames");
        this.container.querySelector("[data-expansion-actions]").hidden = !this.expansionId;
        if (this.expansionId) {
            const available = "[data-discard-expansion],[data-save-expansion],[data-fov],[data-fov-number],[data-recenter]";
            for (const control of this.container.querySelectorAll("button,input,select,textarea")) {
                if (!control.matches(available)) control.disabled = true;
            }
            this.container.querySelector("[data-save-expansion]").disabled ||= !savedScene?.expansion_id;
        }
        const frameId = savedScene?.expansion_frame_id;
        const compareURL = frameId ? this.expansionURL(frameId) + "/frame/" : "";
        this.expansionCompare.hidden = !frameId;
        const frameActions = this.container.querySelector("[data-expansion-frame-actions]");
        frameActions.hidden = !frameId;
        if (compareURL !== this.expansionCompare.dataset.url) {
            this.expansionCompare.dataset.url = compareURL;
            this.expansionSplit.value = "50";
            this.expansionCompare.style.setProperty("--split", "50%");
            if (frameId) {
                const [width, height] = savedScene.reconstruction_size;
                this.expansionCompare.style.setProperty("--frame-aspect", width / height);
                this.expansionBefore.src = this.api.apiURL(BASE + compareURL + "before");
                this.expansionAfter.src = this.api.apiURL(BASE + compareURL + "after");
            } else {
                this.expansionBefore.removeAttribute("src");
                this.expansionAfter.removeAttribute("src");
            }
        }
        if (frameId) {
            for (const control of this.container.querySelectorAll("button,input,select,textarea")) {
                if (control !== this.expansionSplit && !frameActions.contains(control)) control.disabled = true;
            }
            this.container.querySelector("[data-approve-expansion-frame]").disabled ||= ![this.expansionBefore, this.expansionAfter]
                .every(image => image.complete && image.naturalWidth > 0);
        }
        this.container.querySelector('[data-action="model"]').disabled ||= !this.state?.selected_image_id;
        this.container.querySelector("[data-model-provider]").disabled ||= !this.document;
        const modelSettings = this.state?.model_settings;
        this.container.querySelector("[data-model-settings]").disabled ||= modelSettings?.provider === "moge" && modelSettings.moge_refine_views === true;
        this.qi2?.setDisabled(this.busy || this.restoring || !this.document || Boolean(frameId || this.state?.job || this.document?._recovery || this.document?.archived));
        this.updateSourceScroll();
        this.customSelects?.syncSelects();
        this.host.inert = !this.canEditCamera();
        const toolbar = this.container.querySelector(".bg-camera-toolbar"), hidden = Boolean(this.previewSource || !this.selectedScene());
        toolbar.style.visibility = hidden ? "hidden" : ""; toolbar.style.opacity = hidden ? "0" : ""; toolbar.inert = hidden;
        this.container.querySelector("[data-model-status]").textContent = this.busy && this.activeAction === "model" ? "Creating model…"
            : this.modelError ? "Model unavailable" : !this.state?.model_id ? "Model not created" : this.loadingModel || this.backgroundTransition ? "Loading model…" : "Model ready";
    }

    invalidateRestore() { this.characterPanel?.hide(); this.restoring = true; this.backgroundTransition = false; this.clearSourceLoad(); stopSakuraMotion(this.customSelects); this.clearPendingCG(); this.cgPreviewClose?.(); ++this.epoch; this.restoreFailed = false; this.busy = false; this.jobId = ""; clearTimeout(this.pollTimer); this.pollResolve?.(); this.updateControls(); }
    restore() {
        const epoch = ++this.epoch; this.restoring = true; this.restoreFailed = false; this.updateControls(); this.message("Loading saved backgrounds…");
        this.operations = this.operations.catch(() => {}).then(async () => {
            if (this.disposed || epoch !== this.epoch) return;
            try {
                const raw = this.node.widgets.find(item => item.name === "cg_data").value, state = readState(raw);
                const [config, catalog] = await Promise.all([this.request("/config"), this.request("/backgrounds")]);
                if (this.disposed || epoch !== this.epoch) return;
                this.factoryRuntime = await loadFactoryRuntime(config.factory);
                if (this.disposed || epoch !== this.epoch) return;
                this.history ??= new this.factoryRuntime.FactoryCommandHistory({ onRestore: snapshot => {
                    this.characters?.restore(snapshot.characters);
                    this.viewer.setCameraState(snapshot.camera, { emit: true }); this.sync(); this.characterPanel?.refresh(); this.updateControls();
                } });
                this.state = state; this.document = null; this.expansionId = ""; this.config = config; this.catalog = catalog.backgrounds;
                const recovered = this.restoreDraft();
                this.libraryError = (catalog.errors || []).map(item => `${item.name}: ${item.error}`).join(" ");
                if (state.job?.node_id != null && state.job.node_id !== String(this.node.id)) {
                    delete state.job; if (state.preview?.kind === "pending") delete state.preview;
                }
                if (state.background_id) {
                    const document = await this.request(`/backgrounds/${state.background_id}`);
                    if (this.disposed || epoch !== this.epoch) return;
                    await this.acceptDocument(document, false, epoch);
                }
                if (this.disposed || epoch !== this.epoch) return;
                this.render();
                const scroll = this.state.scroll;
                this.container.querySelector("[data-source-choices]").scrollLeft = scroll?.sources || 0;
                this.container.querySelector("[data-cgs]").scrollTop = scroll?.cgs || 0;
                this.restoring = false;
                if (recovered) this.retainDraft();
                this.sync({ trackChange: Boolean(recovered) });
                this.message(this.document?._recovery ? "Recovered backup is read-only. The original data is preserved." : this.catalog.length ? "Set the camera and press Generate CG." : "Create a background in VNCCS Background Creator.", Boolean(this.document?._recovery));
                this.restorePreview();
                if (this.state.job && !this.document?._recovery && !this.document?.archived) this.jobOperation = this.runJob(this.state.job.action).catch(error => { if (!this.disposed && epoch === this.epoch) this.message(error.message, true); });
            } catch (error) { if (!this.disposed && epoch === this.epoch) { this.restoreFailed = true; this.message(error.message + " Original workflow data is preserved. Restoration will retry when the connection returns or this window becomes active.", true); } }
            this.updateControls();
        }); return this.operations;
    }

    async refreshLibrary() {
        if (this.disposed || this.busy || this.restoring && !this.restoreFailed) return;
        if (this.restoreFailed || this.state?.job) { this.invalidateRestore(); return this.restore(); }
        const epoch = this.epoch, previous = this.document, id = this.state.background_id;
        const current = () => !this.disposed && epoch === this.epoch && previous === this.document
            && !this.busy && !this.state.job && !this.history?.pending;
        if (!current()) return;
        try {
            const catalog = await this.request("/backgrounds");
            if (!current()) return;
            const document = id && (catalog.backgrounds.find(item => item.background_id === id)
                || await this.request(`/backgrounds/${id}`));
            if (!current()) return;
            const names = new Map(catalog.backgrounds.map(item => [item.background_id, item.name]));
            const catalogChanged = names.size !== this.catalog.length || this.catalog.some(item => names.get(item.background_id) !== item.name);
            const libraryError = (catalog.errors || []).map(item => `${item.name}: ${item.error}`).join(" ");
            const errorChanged = this.libraryError !== libraryError;
            this.libraryError = libraryError;
            if (catalogChanged) this.catalog = catalog.backgrounds;
            const sourceURL = selectedSource(previous, this.state)?.image.url;
            const retryImage = !this.sourceLoad && sourceURL
                && this.container.querySelector("[data-model-source]").dataset.sourceURL !== this.api.apiURL(sourceURL);
            if (errorChanged && !document?._recovery) this.message(CAMERA_HINT);
            if (document && (JSON.stringify(document) !== JSON.stringify(previous) || this.modelError)) {
                await this.acceptDocument(document, false, epoch, { preserveScene: true });
            } else if (catalogChanged || retryImage) this.render();
        } catch (error) {
            if (current() && this.libraryError !== error.message) {
                this.libraryError = error.message;
                this.message(error.message, true);
            }
        }
    }

    async acceptDocument(document, adopt = true, epoch = this.epoch, { preserveScene = false } = {}) {
        if (this.disposed || epoch !== this.epoch) return;
        if (!adopt && document.background_id !== this.state.background_id) throw new Error("The response belongs to another background.");
        const previousSource = this.document?.images.find(image => image.image_id === this.document.selected_image_id);
        const currentSource = document.images.find(image => image.image_id === document.selected_image_id);
        const replacedSource = previousSource && (previousSource.image_id !== currentSource?.image_id
            || previousSource.preview_revision !== currentSource?.preview_revision || previousSource.url !== currentSource?.url);
        const staleSource = this.state.selected_image_id ? (this.state.selected_image_id !== document.selected_image_id
            || this.state.selected_variant && !sourceChoices(document).some(choice => sameSource({ source_image_id: choice.image_id, source_variant: choice.variant }, this.state)))
            : !this.state.model_id && Boolean(document.selected_image_id);
        this.document = document; document.models ||= []; document.cgs ||= []; this.catalog = this.catalog.filter(item => item.background_id !== document.background_id);
        if (!document.archived) this.catalog.push(document);
        if (adopt || !this.state.job && (replacedSource || staleSource)) {
            this.cgPreviewClose?.();
            this.state.background_id = document.background_id; this.state.selected_image_id = document.selected_image_id || document.images[0]?.image_id || "";
            this.state.selected_variant = null;
            this.state.model_id = latestModel(document, this.state)?.model_id || "";
            const savedCG = !adopt && !replacedSource && document.cgs.find(cg => cg.cg_id === this.state.selected_cg_id);
            this.state.selected_cg_id = savedCG?.cg_id || (replacedSource ? "" : document.cgs.filter(cg => sameSource(cg, this.state)).at(-1)?.cg_id || "");
            const selected = document.cgs.find(cg => cg.cg_id === this.state.selected_cg_id);
            if (selected && sameSource(selected, this.state)
                && document.models.some(model => model.model_id === selected.model_id && sameSource(model, this.state))) this.state.model_id = selected.model_id;
            this.state.camera = null; this.state.frame = null;
        } else if ((this.state.model_id || !this.state.job)
            && !document.models.some(model => model.model_id === this.state.model_id)) {
            const previous = this.state.model_id;
            const selected = document.cgs.find(cg => cg.cg_id === this.state.selected_cg_id);
            const linked = document.models.find(model => model.model_id === selected?.model_id && sameSource(model, this.state));
            this.state.model_id = (linked || latestModel(document, this.state))?.model_id || "";
            if (this.state.model_id && !linked && (selected || !previous)) {
                this.state.camera = null; this.state.frame = null;
            }
            if (this.state.model_id && this.state.characters?.[previous]) {
                this.state.characters[this.state.model_id] ??= clone(this.state.characters[previous]);
            }
        }
        this.modelError = "";
        if (adopt) return;
        if (this.state.model_id && (!preserveScene || !this.selectedScene() || !this.state.camera)) {
            try {
                const opening = this.showModel(epoch);
                if (preserveScene) this.updateControls();
                await opening;
            }
            catch (error) {
                if (this.disposed || epoch !== this.epoch) return;
                this.scene = null; this.modelError = error.message;
            }
        }
        if (this.disposed || epoch !== this.epoch) return;
        this.render(); this.sync(); this.updateControls();
    }

    async showModel(epoch = this.epoch) {
        const modelId = this.state.model_id, backgroundId = this.state.background_id;
        if (!modelId) return;
        const previous = this.modelLoad;
        let finish;
        // The shared canvas must finish its old load before a new scene can replace it.
        const load = this.modelLoad = new Promise(resolve => { finish = resolve; });
        this.loadingModel = true;
        const current = () => !this.disposed && epoch === this.epoch && load === this.modelLoad && modelId === this.state.model_id && backgroundId === this.state.background_id;
        try {
            await previous;
            if (!current()) return;
            const saved = await this.request(`/backgrounds/${backgroundId}/models/${modelId}`);
            if (!current()) return;
            this.expansionId = saved.expansion_id || "";
            const scene = this.expansionId ? await this.request(this.expansionURL(this.expansionId)) : saved;
            if (!current()) return;
            if (!this.viewer) {
                this.viewer = new this.factoryRuntime.Factory3DViewer(this.host, { cameraDragScale: 0.7, resolveAssetURL: path => this.api.apiURL(path),
                    onSelectionChange: () => this.viewer.select("", { emit: false }),
                    onStateChange: value => { if (this.canEditCamera()) { if (!this.expansionId) this.state.camera = clone(value.camera); this.syncFov(value.camera); this.sync(); } },
                    onError: error => this.message(String(error.message || error), true) });
                const updateClipPlanes = this.viewer._updateClipPlanes;
                if (updateClipPlanes) this.viewer._updateClipPlanes = (...args) => {
                    const result = updateClipPlanes.apply(this.viewer, args);
                    const near = this.mogeUnit / 10, camera = this.viewer.camera;
                    if (near > 0 && camera.near > near) {
                        camera.near = near;
                        camera.updateProjectionMatrix();
                    }
                    return result;
                };
                const rotate = this.viewer.rotateCameraFPV;
                this.viewer.rotateCameraFPV = (turn, options) => this.selectedScene()?.background_settings?.provider === "worldmirror"
                    ? this.rotateWorldMirrorCamera(turn, options) : rotate.call(this.viewer, turn, options);
                const dolly = this.viewer.dollyCamera;
                this.viewer.dollyCamera = (amount, options) => this.selectedScene()?.background_settings?.provider === "worldmirror" && this.worldMirrorDollyStep
                    ? this.dollyWorldMirrorCamera(amount, options) : dolly.call(this.viewer, amount, options);
                this.viewer.controls.enableDamping = false; this.viewer.canvas.setAttribute("aria-label", "CG camera. Drag to look, middle-drag to pan, scroll to move.");
            }
            if (!this.selectedScene() || this.scene.expansion_id !== scene.expansion_id) {
                this.scene = null;
                const result = await this.viewer.setScene(scene);
                if (!current()) return;
                if (result.failures.length) throw new Error("The model could not be loaded.");
                this.history.clear();
            }
            const selected = this.document.cgs.find(cg => cg.cg_id === this.state.selected_cg_id && cg.model_id === modelId);
            const characters = this.state.characters?.[modelId] ?? selected?.characters ?? [];
            if (characters.length) await this.ensureCharacters();
            if (!current()) return;
            this.state.selected_image_id = scene.source_image_id; this.state.selected_variant = clone(scene.source_variant || null);
            this.characters?.restore(scene.expansion_id ? [] : characters);
            const size = this.state.job && this.state.frame || frameDimensions(this.state.settings.aspect, this.state.settings.resolution_mp);
            this.state.frame = clone(size); this.viewer.setLighting({ preset: "off", background: "#000000" });
            this.viewer.setCaptureSettings({ ...size, show_camera_frame: true });
            let camera = scene.expansion_id ? null : this.state.camera || selected?.camera;
            const source = scene.objects[0]?.source_camera;
            if (camera && scene.background_settings?.provider === "worldmirror") {
                const up = (source || scene.camera)?.up;
                if (up && up.reduce((dot, value, axis) => dot + value * camera.up[axis], 0) < 0) {
                    camera = { ...camera, up };
                }
            }
            // Earlier MoGe scenes saved the source camera directly as the viewport camera.
            if (!selected && !this.state.job && scene.background_settings?.provider === "moge" && camera && source
                && Math.abs(camera.fov - source.fov) < 1e-6
                && ["position", "target", "up"].every(key => source[key]?.length === 3
                    && camera[key].every((value, axis) => Math.abs(value - source[key][axis]) < 1e-6))) {
                camera = null;
            }
            if (camera) this.viewer.setCameraState(camera);
            else this.recenterCamera(scene);
            const modelCamera = scene.background_settings?.provider === "moge"
                ? this.viewer.getModelSourceCamera?.(scene.objects[0]?.object_id) : null;
            // MoGe's source look vector represents one unit before the viewer normalizes the mesh.
            this.mogeUnit = modelCamera ? Math.hypot(...modelCamera.target.map((value, axis) => value - modelCamera.position[axis])) : 0;
            this.viewer._updateClipPlanes?.();
            const shownCamera = camera || this.viewer.getCameraState();
            if (!scene.expansion_id) this.state.camera = clone(shownCamera);
            this.viewer.select("", { emit: false }); this.syncFov(shownCamera);
            this.scene = scene; this.modelError = "";
        } finally { finish(); if (this.modelLoad === load) this.loadingModel = false; }
        this.resize();
    }

    recenterCamera(scene, emit = false) {
        const item = scene.objects[0];
        if (item.asset_kind !== "mesh") {
            const source = item.source_camera || scene.camera;
            this.viewer.setCameraState(scene.background_settings?.provider === "worldmirror" ? source : sourceCameraView(source), { emit });
            return;
        }
        const source = this.viewer.getModelSourceCamera?.(item.object_id);
        if (source) this.viewer.setCameraState(source, { emit });
        else {
            this.viewer.setCameraState({ position: [0, 0, 1], target: [0, 0, 0], up: [0, 1, 0], fov: 60 });
            this.viewer.fit(item.object_id, { emit, front: true });
        }
    }

    clearSourceLoad() {
        const incoming = this.sourceLoad; this.sourceLoad = null;
        this.resolveSource?.(false); this.resolveSource = null; this.sourceReady = null;
        if (!incoming) return;
        incoming.onload = incoming.onerror = null;
        this.customSelects._motionByElement?.get(incoming)?.cancel(); incoming.remove();
    }

    async switchBackground(document, epoch) {
        const current = () => !this.disposed && epoch === this.epoch;
        if (!current()) return;
        this.clearSourceLoad();
        this.backgroundTransition = true;
        const previous = { document: this.document, state: clone(this.state), modelError: this.modelError };
        try {
            await this.acceptDocument(document, true, epoch);
            if (!current()) return;
            this.render();
            const ready = !this.sourceReady || await this.sourceReady;
            if (!current()) return;
            if (!ready) throw new Error("The background image could not be loaded. The previous background is still selected.");
            this.backgroundTransition = false;
            this.render(); this.sync();
            void this.loadBackgroundModel(epoch);
        } catch (error) {
            if (current()) {
                this.clearSourceLoad(); this.document = previous.document; this.state = previous.state;
                this.modelError = previous.modelError;
                this.backgroundTransition = false;
                this.render(); this.sync(); this.restorePreview();
            }
            throw error;
        } finally { if (current()) this.backgroundTransition = false; }
    }

    async loadBackgroundModel(epoch) {
        const modelId = this.state.model_id;
        if (!modelId) return;
        const opening = this.showModel(epoch), load = this.modelLoad;
        const current = () => !this.disposed && epoch === this.epoch && load === this.modelLoad && this.state.model_id === modelId;
        try { await opening; }
        catch (error) {
            if (!current()) return;
            this.scene = null; this.modelError = error.message; this.message(error.message, true);
        }
        if (current()) { this.render(); this.sync(); }
    }

    render() {
        if (!this.state) return;
        const view = this.customSelects.captureView();
        const scroll = [...this.container.querySelectorAll(".bg-panel,.bg-cg-list")].map(element => [element, element.scrollTop, element.scrollLeft]);
        const fill = (selector, entries, value, empty) => {
            const select = this.container.querySelector(selector); select.replaceChildren();
            if (empty) { const option = document.createElement("option"); option.value = ""; option.textContent = empty; select.appendChild(option); }
            for (const [id, label] of entries) { const option = document.createElement("option"); option.value = id; option.textContent = label; select.appendChild(option); }
            select.value = value;
        };
        const catalog = this.document && !this.catalog.some(doc => doc.background_id === this.document.background_id) ? [...this.catalog, this.document] : this.catalog;
        fill("[data-library]", catalog.map(doc => [doc.background_id, doc.name]), this.state.background_id, "Select a background");
        const choices = sourceChoices(this.document), selected = selectedSource(this.document, this.state);
        if (selected && !choices.some(choice => choice.key === selected.key)) choices.push(selected);
        const source = selectedSource(this.document, this.state)?.image, image = this.container.querySelector("[data-model-source]");
        const sourceURL = source ? this.api.apiURL(source.url) : "", placeholder = this.container.querySelector("[data-empty]"), enlarge = this.container.querySelector("[data-enlarge]");
        if ((this.sourceLoad?.dataset.sourceURL ?? image.dataset.sourceURL) !== sourceURL) {
            this.clearSourceLoad();
            enlarge.hidden = true;
            const previous = !image.hidden && Boolean(image.src), animate = previous && !this.restoring;
            placeholder.hidden = previous;
            const title = placeholder.querySelector(".vnccs-common-creator-placeholder-title"), hint = placeholder.querySelector(".vnccs-common-creator-placeholder-hint");
            title.textContent = source ? "Loading background…" : "Select a background";
            hint.textContent = "Choose a saved background and one of its variants.";
            if (source) {
                const incoming = document.createElement("img"); this.sourceLoad = incoming;
                this.sourceReady = new Promise(resolve => { this.resolveSource = resolve; });
                incoming.dataset.preview = ""; incoming.dataset.sourceIncoming = ""; incoming.dataset.sourceURL = sourceURL;
                incoming.alt = "Selected background"; incoming.hidden = true; incoming.style.pointerEvents = "none";
                incoming.fetchPriority = "high";
                image.parentElement.appendChild(incoming);
                const current = () => !this.disposed && this.sourceLoad === incoming;
                const failure = "Background image unavailable. Select another variant or return to this window to retry.";
                const publish = () => {
                    if (!current()) return;
                    incoming.onload = incoming.onerror = null;
                    incoming.dataset.modelSource = ""; delete incoming.dataset.sourceIncoming;
                    image.replaceWith(incoming); this.sourceLoad = null; enlarge.hidden = false;
                    this.resolveSource?.(true); this.resolveSource = null;
                };
                incoming.onerror = () => {
                    if (!current()) return;
                    this.clearSourceLoad(); placeholder.hidden = previous;
                    title.textContent = "Background image unavailable"; hint.textContent = "Select another variant or return to this window to retry.";
                    if (previous) this.message(failure, true);
                };
                incoming.onload = async () => {
                    try { await incoming.decode?.(); } catch (_) { if (current()) incoming.onerror(); return; }
                    if (!current()) return;
                    incoming.hidden = false; placeholder.hidden = true;
                    if (this.status.textContent === failure) this.message(CAMERA_HINT);
                    if (animate && !this.restoring && !this.backgroundTransition) playSakuraMotion(this.customSelects, incoming, [{ opacity: 0 }, { opacity: 1 }], 240, publish);
                    else publish();
                };
                incoming.src = sourceURL;
            } else { image.hidden = true; placeholder.hidden = false; image.removeAttribute("src"); image.dataset.sourceURL = ""; }
        } else if (!this.sourceLoad && !image.hidden) enlarge.hidden = false;
        this.renderSourceCards(choices, selected?.key);
        const selectedProvider = this.state.model_settings.provider;
        const provider = Object.hasOwn(MODEL_NAMES, selectedProvider) ? selectedProvider : "triposplat";
        this.container.querySelector("[data-model-provider]").value = provider;
        this.container.querySelector("[data-moge-refine-label]").hidden = provider !== "moge" && provider !== "worldmirror";
        this.container.querySelector("[data-moge-refine-views]").checked = this.state.model_settings.moge_refine_views === true;
        const card = this.container.querySelector("[data-model-settings]");
        card.setAttribute("aria-label", `${MODEL_NAMES[provider]} settings`);
        card.querySelector(".vnccs-common-creator-model-card-name").textContent = MODEL_NAMES[provider];
        card.querySelector(".vnccs-common-creator-model-card-desc").textContent = provider === "pixal3d" || provider === "moge" ? "Textured GLB mesh"
            : provider === "worldmirror" ? "Multi-view Gaussian splat" : "Single-image Gaussian splat";
        const visible = Boolean(this.selectedScene()); this.host.hidden = !visible; this.container.querySelector("[data-viewer-empty]").hidden = visible;
        const kind = this.container.querySelector("[data-model-kind]");
        kind.hidden = !visible;
        kind.textContent = this.scene?.objects[0]?.asset_kind === "mesh" ? "Textured mesh" : "Raw splat";
        this.container.querySelector("[data-viewer-empty]").textContent = this.modelError || (latestModel(this.document, this.state)
            ? "Loading the saved model…" : "Select a saved background and create its model.");
        const gallery = this.container.querySelector("[data-cgs]");
        const cgs = imageCGs(this.document, this.state).reverse();
        const galleryKey = JSON.stringify([cgs.map(cg => [cg.cg_id, cg.name, cg.url, cg.width, cg.height]), this.pendingCG && [this.pendingCG.name, this.pendingCG.url, this.pendingCG.width, this.pendingCG.height]]);
        if (gallery.dataset.cards !== galleryKey) {
            this.galleryObserver?.disconnect();
            gallery.replaceChildren(); gallery.dataset.cards = galleryKey;
            this.galleryObserver = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries, observer) => {
                for (const { target, isIntersecting } of entries) {
                    if (!isIntersecting || !gallery.contains(target)) continue;
                    target.src = target.dataset.cgSrc;
                    observer.unobserve(target);
                }
            }) : null;
            if (this.pendingCG) {
                const pending = this.pendingCG, card = document.createElement("button"); card.className = "bg-cg-card vnccs-common-creator-model-card"; card.dataset.pendingCg = ""; card.setAttribute("aria-label", `Preview ${pending.name} while generating`);
                const title = document.createElement("strong"); title.textContent = pending.name;
                const preview = document.createElement("div"); preview.className = "bg-cg-pending-preview vnccs-common-creator-preview-container"; preview.style.aspectRatio = `${pending.width} / ${pending.height}`;
                const layer = document.createElement("div"); layer.className = "vnccs-common-creator-generation-layer uses-mannequin";
                if (pending.url) { const image = document.createElement("img"); image.className = "vnccs-common-creator-generation-img vnccs-common-creator-generation-base"; image.src = pending.url; image.alt = "Captured CG camera frame"; layer.appendChild(image); }
                const scan = document.createElement("div"); scan.className = "vnccs-common-creator-generation-scan"; layer.appendChild(scan); preview.appendChild(layer);
                const label = document.createElement("span"); label.textContent = "Generating CG…"; label.setAttribute("role", "status");
                card.append(title, preview, label); gallery.appendChild(card);
            }
            for (const cg of cgs) {
                const entry = document.createElement("div"); entry.className = "bg-cg-entry";
                const card = document.createElement("button"); card.className = "bg-cg-card vnccs-common-creator-model-card"; card.dataset.cgId = cg.cg_id; card.classList.toggle("is-selected", cg.cg_id === this.state.selected_cg_id);
                const title = document.createElement("strong"); title.textContent = cg.name;
                const image = document.createElement("img"); image.alt = cg.name; image.decoding = "async"; image.fetchPriority = "low";
                image.width = cg.width; image.height = cg.height; image.style.aspectRatio = `${cg.width} / ${cg.height}`;
                image.dataset.cgSrc = this.api.apiURL(cg.url);
                const remove = document.createElement("button"); remove.type = "button"; remove.className = "vnccs-style-delete"; remove.dataset.deleteCg = cg.cg_id; remove.textContent = "×"; remove.title = `Delete ${cg.name}`; remove.setAttribute("aria-label", remove.title);
                card.append(title, image); entry.append(card, remove); gallery.appendChild(entry);
                if (this.galleryObserver) this.galleryObserver.observe(image);
                else { image.loading = "lazy"; image.src = image.dataset.cgSrc; }
            }
        }
        for (const card of gallery.querySelectorAll("[data-cg-id]")) card.classList.toggle("is-selected", card.dataset.cgId === this.state.selected_cg_id);
        this.container.querySelector("[data-cg-count]").textContent = String(cgs.length + (this.pendingCG ? 1 : 0)); this.container.querySelector("[data-cg-empty]").hidden = Boolean(cgs.length || this.pendingCG);
        this.qi2.render(this.config); this.container.querySelector("[data-aspect]").value = this.state.settings.aspect;
        this.customSelects.syncSelects(); this.customSelects.restoreView(view); this.updateControls(); for (const [element, top, left] of scroll) { element.scrollTop = top; element.scrollLeft = left; } this.resize();
    }

    renderSourceCards(choices, selectedKey) {
        const list = this.container.querySelector("[data-source-choices]");
        const key = JSON.stringify(choices.map(choice => [choice.key, choice.label, choice.image.url]));
        if (list.dataset.cards !== key) {
            list.replaceChildren(); list.dataset.cards = key; list.classList.toggle("is-single-row", choices.length <= 2);
            for (const [index, choice] of choices.entries()) {
                const item = document.createElement("div"); item.className = "bg-variation-item";
                item.style.gridColumn = String(Math.floor(index / 4) * 2 + index % 2 + 1); item.style.gridRow = String(Math.floor(index % 4 / 2) + 1);
                const name = document.createElement("span"); name.className = "bg-source-name"; name.textContent = choice.label;
                const preview = document.createElement("div"); preview.className = "bg-variation-preview";
                const button = document.createElement("button"); button.type = "button"; button.dataset.sourceChoice = choice.key; button.dataset.focusKey = "source-" + choice.key;
                button.setAttribute("aria-label", "Select " + choice.label + " background variant");
                const image = document.createElement("img"); image.dataset.variationImage = choice.key; image.decoding = "async";
                image.alt = choice.label + " variant"; image.src = this.api.apiURL(choice.image.url);
                image.onerror = () => { image.hidden = true; };
                preview.appendChild(image); button.append(preview, name); item.appendChild(button); list.appendChild(item);
            }
        }
        for (const button of list.querySelectorAll("[data-source-choice]"))
            button.setAttribute("aria-pressed", String(button.dataset.sourceChoice === selectedKey));
        this.updateSourceScroll();
    }

    updateSourceScroll() {
        const list = this.container.querySelector("[data-source-choices]");
        const overflow = list.scrollWidth > list.clientWidth + 1;
        list.parentElement.classList.toggle("is-scrollable", overflow);
        for (const button of this.container.querySelectorAll("[data-scroll-source]")) {
            button.hidden = !overflow;
            const edge = Number(button.dataset.scrollSource) < 0 ? list.scrollLeft <= 1 : list.scrollLeft + list.clientWidth >= list.scrollWidth - 1;
            button.disabled = this.restoring || this.busy || edge;
        }
    }

    syncFov(camera) { this.fov.value = String(camera.fov); this.fovNumber.value = String(Math.round(camera.fov * 10) / 10); }
    sync({ trackChange = true, sceneChanged = false } = {}) {
        if (this.disposed || this.restoring || this.backgroundTransition || !this.state) return;
        const sceneReady = this.selectedScene() && !this.scene.expansion_frame_id && !this.expansionId && !this.busy && !this.loadingModel;
        if (this.viewer && sceneReady) this.state.camera = this.viewer.getCameraState();
        if (this.characters && sceneReady) {
            this.state.characters ??= {}; this.state.characters[this.state.model_id] = this.characters.snapshot();
        }
        const sources = this.container.querySelector("[data-source-choices]").scrollLeft, cgs = this.container.querySelector("[data-cgs]").scrollTop;
        if (sources || cgs || this.state.scroll) this.state.scroll = { sources, cgs };
        const raw = JSON.stringify(this.state), widget = this.node.widgets.find(item => item.name === "cg_data");
        const changed = widget && widget.value !== raw;
        if (changed || sceneChanged) {
            const canvas = trackChange && !this.history?.pending ? app.canvas : null;
            canvas?.emitEvent?.({ subType: "before-change" });
            try {
                if (changed) {
                    widget.value = raw;
                    this.node.properties ??= {}; this.node.properties.cgCreator = raw;
                    this.node.graph?.setDirtyCanvas?.(true, true);
                }
            } finally {
                canvas?.emitEvent?.({ subType: "after-change" });
            }
        }
        return raw;
    }
    draftKey(legacyBackground = "") {
        this.draftId ??= cacheIdentity();
        const owner = (this.node.id != null && this.node.id !== -1 && workflowScope(this.node)) || this.draftId;
        return legacyBackground ? `VNCCS.CGCreator:${owner}:${legacyBackground}` : `VNCCS.CGCreator:settings:${owner}`;
    }

    retainDraft() {
        const { settings, model_settings, settings_revision } = this.state;
        this.draftFailed = !storage.setItem(this.draftKey(), JSON.stringify({ settings, model_settings, settings_revision }));
        if (this.draftFailed) this.message("Browser recovery is unavailable. Save this workflow to keep the node settings.", true);
    }
    restoreDraft() {
        try {
            let raw = storage.getItem(this.draftKey());
            const legacy = !raw && this.state.settings_revision == null && this.state.background_id;
            if (legacy) raw = storage.getItem(this.draftKey(legacy));
            if (!raw) return;
            const draft = JSON.parse(raw);
            if (!draft?.settings || !draft.model_settings) throw new Error("Invalid recovery draft");
            const restored = readState(JSON.stringify({ ...draft, schema_version: 1 }));
            const revision = legacy ? 1 : restored.settings_revision;
            if (!(revision > (this.state.settings_revision ?? 0))) return;
            this.state.settings = restored.settings;
            this.state.model_settings = restored.model_settings;
            this.state.settings_revision = revision;
            return true;
        } catch (_) { this.message("A recovery draft could not be read. Its original bytes are preserved.", true); }
    }
    markEdited() {
        this.state.settings_revision = (this.state.settings_revision ?? 0) + 1;
        this.retainDraft();
        this.sync();
    }

    async dialog(title, fields, submitLabel = "Save", { message = "", danger = false } = {}) {
        return await new Promise(resolve => {
            const controls = {};
            const finish = value => { if (!this.dialogs.delete(close)) return; modal.overlay.remove(); resolve(value); };
            const close = () => finish(null);
            this.dialogs.add(close);
            const modal = showModal(this.container, title, () => {
                const form = document.createElement("div"); form.className = "bg-dialog-fields";
                if (message) { const description = document.createElement("div"); description.textContent = message; form.appendChild(description); }
                for (const field of fields) {
                    const label = document.createElement("label"), text = document.createElement("span"); text.textContent = field.label; label.className = "bg-field";
                    const control = document.createElement(field.multiline ? "textarea" : field.options ? "select" : "input");
                    if (field.options) for (const choice of field.options) {
                        const option = document.createElement("option");
                        option.value = choice.value ?? choice;
                        option.textContent = choice.label ?? choice;
                        control.appendChild(option);
                    }
                    control.value = field.value ?? ""; control.name = field.key;
                    if (!field.options && !field.multiline) control.type = field.type || "text";
                    if (field.required) control.required = true; if (field.multiline) control.rows = 5;
                    for (const key of ["min", "max", "step", "pattern"]) if (field[key] != null) control[key] = field[key];
                    if (field.maxLength) control.maxLength = field.maxLength;
                    if (field.oninput) for (const type of ["input", "change"]) control.addEventListener(type, () => field.oninput(control.value));
                    label.append(text, control);
                    form.appendChild(label); controls[field.key] = control;
                }
                return form;
            }, [{ text: "Cancel", action: close }, { text: submitLabel, class: danger ? "danger" : "primary", action: () => {
                if (!Object.values(controls).every(control => control.reportValidity())) return true;
                finish(Object.fromEntries(Object.entries(controls).map(([key, control]) => [key, control.value])));
            } }]);
            const remove = modal.overlay.remove;
            modal.overlay.remove = () => { remove.apply(modal.overlay); if (this.dialogs.delete(close)) resolve(null); };
            for (const control of modal.content.querySelectorAll("input,select,textarea")) control.classList.add("vnccs-common-creator-" + (control.tagName === "SELECT" ? "select" : control.tagName === "TEXTAREA" ? "textarea" : "input"));
            for (const label of modal.content.querySelectorAll(".bg-field>span")) label.classList.add("vnccs-common-creator-label");
            this.customSelects.syncSelects();
        });
    }

    async click(event) {
        const button = event.target.closest("button"); if (!button || !this.container.contains(button) || button.disabled) return; const epoch = this.epoch;
        let navigating = false, sourceFocus;
        try {
            if (button.hasAttribute("data-expand-scene")) {
                if (!this.canEditCamera() || this.expansionId) return;
                if (!this.selectedScene().can_expand) {
                    this.message("This model has no reconstruction frames. Use Create Model once, then try Expand Scene.", true);
                    return;
                }
                const camera = this.viewer.getCameraState();
                const scene = this.selectedScene();
                const [width, height] = scene.reconstruction_size;
                // WorldMirror splats and viewport cameras already share the PLY coordinate system.
                let modelCamera = camera;
                if (scene.background_settings.provider === "moge") {
                    const space = this.viewer.objects.get(scene.objects[0].object_id)?.modelSpace;
                    if (!space) throw new Error("The MoGe model camera is unavailable.");
                    const local = values => space.worldToLocal(this.viewer.camera.position.clone().fromArray(values)).toArray();
                    const position = local(camera.position);
                    const up = local(camera.position.map((value, axis) => value + camera.up[axis]))
                        .map((value, axis) => value - position[axis]);
                    const length = Math.hypot(...up);
                    modelCamera = { position, target: local(camera.target), up: up.map(value => value / length), fov: camera.fov };
                }
                const size = { width: Math.max(64, 16 * Math.round(width / 16)), height: Math.max(64, 16 * Math.round(height / 16)) };
                this.commitGesture(); this.sync();
                this.busy = true; this.updateControls();
                this.message("Capturing the current view…", false, true);
                const raw = await this.captureScene({ ...size, cameraState: camera }, { backgroundOnly: true });
                if (epoch !== this.epoch || this.disposed) return;
                await this.runJob("expand", { model_id: this.state.model_id, model_camera: modelCamera }, raw);
                return;
            }
            if (button.hasAttribute("data-approve-expansion-frame") || button.hasAttribute("data-discard-expansion-frame")) {
                const scene = this.selectedScene(), frameId = scene?.expansion_frame_id;
                if (!frameId || this.busy || this.loadingModel) return;
                if (button.hasAttribute("data-approve-expansion-frame")) {
                    await this.runJob("expand", { model_id: scene.model_id, expansion_frame_id: frameId });
                } else {
                    this.busy = true; this.updateControls();
                    await this.json(this.expansionURL(frameId) + "/frame", {}, "DELETE");
                    if (epoch !== this.epoch || this.disposed) return;
                    delete scene.expansion_frame_id;
                    this.busy = false; this.updateControls();
                    this.message("Expansion frame discarded.", false, true);
                }
                return;
            }
            if (button.hasAttribute("data-save-expansion") || button.hasAttribute("data-discard-expansion")) {
                const save = button.hasAttribute("data-save-expansion"), id = this.expansionId;
                if (!id || this.busy || this.loadingModel) return;
                navigating = this.busy = true; this.updateControls();
                const camera = save ? this.viewer.getCameraState() : this.state.camera;
                const document = await this.json(this.expansionURL(id), { save });
                if (epoch !== this.epoch || this.disposed) return;
                this.expansionId = "";
                this.state.camera = camera;
                if (save) this.state.selected_cg_id = "";
                await this.acceptDocument(document, false, epoch);
                this.message(save ? "Expanded scene saved." : "Previous scene restored.");
                return;
            }
            if (button.hasAttribute("data-scroll-source")) {
                const behavior = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
                const list = this.container.querySelector("[data-source-choices]");
                list.scrollBy({ left: Number(button.dataset.scrollSource) * list.clientWidth * .8, behavior }); return;
            }
            if (button.dataset.deleteCg) {
                const id = button.dataset.deleteCg, backgroundId = this.state.background_id;
                const cg = this.document?.cgs.find(cg => cg.cg_id === id); if (!cg) return;
                navigating = this.busy = true; this.updateControls();
                const confirmed = await this.dialog("Delete CG", [], "Delete", { message: `Delete "${cg.name}" from the CG library?`, danger: true });
                if (!confirmed || epoch !== this.epoch || this.disposed || backgroundId !== this.state.background_id) return;
                const document = await this.json(`/backgrounds/${backgroundId}/cg/${id}`, { revision: this.document.revision }, "DELETE");
                if (epoch !== this.epoch || this.disposed) return;
                if (this.previewSource?.cg_id === id) this.cgPreviewClose?.();
                this.document = document; this.catalog = this.catalog.map(item => item.background_id === backgroundId ? document : item);
                if (this.state.selected_cg_id === id) this.state.selected_cg_id = "";
                this.sync(); this.render(); return;
            }
            if (button.hasAttribute("data-source-choice")) {
                const choice = sourceChoices(this.document).find(choice => choice.key === button.dataset.sourceChoice);
                if (!choice || choice.key === selectedSource(this.document, this.state)?.key && !this.state.selected_cg_id) return;
                if (document.activeElement === button) sourceFocus = choice.key;
                this.cgPreviewClose?.();
                navigating = this.busy = true; this.updateControls(); this.state.selected_image_id = choice.image_id; this.state.selected_variant = clone(choice.variant);
                this.state.model_id = latestModel(this.document, this.state)?.model_id || ""; this.state.selected_cg_id = ""; this.state.camera = null; this.state.frame = null;
                this.modelError = "";
                this.sync(); this.render();
                await this.loadBackgroundModel(epoch); return;
            }
            if (button.hasAttribute("data-pending-cg")) { this.openPreview(this.pendingCG); return; }
            if (button.dataset.cgId) {
                const cg = this.document.cgs.find(cg => cg.cg_id === button.dataset.cgId); if (!cg) return;
                this.openPreview(cg);
                if (this.busy || this.state.job || this.document?._recovery || this.document?.archived) return;
                const model = this.document.models.find(model => model.model_id === cg.model_id);
                if (!model && this.state.model_id) {
                    this.state.selected_cg_id = cg.cg_id;
                    this.sync(); this.render(); return;
                }
                navigating = this.busy = true; this.updateControls();
                Object.assign(this.state, { selected_cg_id: cg.cg_id, model_id: model?.model_id || "", camera: clone(cg.camera), frame: { width: cg.width, height: cg.height } });
                if (cg.characters?.length || this.state.characters?.[cg.model_id]) {
                    this.state.characters ??= {}; this.state.characters[cg.model_id] = clone(cg.characters || []);
                }
                this.sync();
                if (model) await this.showModel(epoch);
                if (epoch !== this.epoch || this.disposed) return;
                this.sync(); this.render(); return;
            }
            if (button.hasAttribute("data-add-character")) {
                if (!this.canEditCamera()) return;
                if (this.characters?.entries.size) { await this.characterPanel?.show(); return; }
                this.beginGesture(); this.busy = true; this.updateControls();
                const editor = await this.ensureCharacters();
                if (this.disposed || epoch !== this.epoch) return;
                this.busy = false; editor.add(Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join(""));
                this.commitGesture(); this.updateControls(); await this.characterPanel?.show(); return;
            }
            if (button.hasAttribute("data-recenter")) { this.beginGesture(); this.recenterCamera(this.scene, true); this.commitGesture(); return; }
            if (button.hasAttribute("data-enlarge")) { this.openPreview(); return; }
            if (button.hasAttribute("data-model-settings")) {
                const provider = this.state.model_settings.provider;
                const mesh = provider === "pixal3d", worldmirror = provider === "worldmirror", moge = provider === "moge";
                const profileDefault = { value: 0, label: "Profile default" };
                const fields = worldmirror ? [
                    { key: "worldmirror_target_size", label: "Resolution", value: this.state.model_settings.worldmirror_target_size,
                        type: "number", min: 252, max: 4088, step: 14, required: true },
                    { key: "worldmirror_apply_sky_mask", label: "Sky mask", value: String(this.state.model_settings.worldmirror_apply_sky_mask), options: [
                        { value: "true", label: "On" }, { value: "false", label: "Off" },
                    ] },
                ] : mesh ? [
                    { key: "pixal_profile", label: "Profile", value: this.state.model_settings.pixal_profile, options: [
                        { value: "standard", label: "Standard" },
                        { value: "fast", label: "Fast · Base color only" },
                    ] },
                    { key: "pixal_mesh_resolution", label: "Mesh resolution", value: this.state.model_settings.pixal_mesh_resolution, options: [profileDefault, 1024, 1536, 2048] },
                    { key: "pixal_texture_resolution", label: "Texture atlas resolution", value: this.state.model_settings.pixal_texture_resolution, options: [profileDefault, 256, 512, 1024, 2048, 4096] },
                ] : moge ? [] : [
                    { key: "num_gaussians", label: "Gaussians", value: this.state.model_settings.num_gaussians, options: [32768, 65536, 131072, 262144, 524288, 1048576] },
                    { key: "steps", label: "Steps", value: this.state.model_settings.steps, type: "number", min: 1, max: 100, step: 1, required: true },
                    { key: "guidance_scale", label: "Guidance", value: this.state.model_settings.guidance_scale, type: "number", min: 1, max: 20, step: 0.1, required: true },
                ];
                if ((worldmirror || moge) && !this.state.model_settings.moge_refine_views) {
                    const value = this.state.model_settings.camera_shift_lora;
                    const loras = [...new Set([...(this.config?.loras || []), value].filter(Boolean))];
                    fields.unshift({ key: "camera_shift_lora", label: "Camera shift LoRA", value,
                        options: [{ value: "", label: "Select a LoRA" }, ...loras] });
                } else if (!worldmirror && !moge) fields.push({ key: "seed", label: "Seed", value: this.state.model_settings.seed, type: "number", min: -1, max: 2147483647, step: 1, required: true });
                const values = await this.dialog(`${MODEL_NAMES[provider] ?? MODEL_NAMES.triposplat} settings`, fields);
                if (values && epoch === this.epoch && !this.disposed) {
                    for (const [key, value] of Object.entries(values)) {
                        if (key === "worldmirror_apply_sky_mask") this.state.model_settings[key] = value === "true";
                        else if (key === "camera_shift_lora") this.state.model_settings[key] = value;
                        else if (key !== "pixal_profile") this.state.model_settings[key] = Number(value);
                        else if (value === "fast" || value === "standard") this.state.model_settings[key] = value;
                    }
                    this.markEdited();
                }
                return;
            }
            if (button.hasAttribute("data-save-view")) {
                const values = await this.dialog("Generate CG", [{ key: "name", label: "Scene name (required)", required: true, maxLength: 96, pattern: ".*\\S.*" }], "Generate");
                if (!values || epoch !== this.epoch || !this.canEditCamera()) return;
                const name = values.name.trim();
                const size = clone(this.state.frame), camera = clone(this.viewer.getCameraState());
                this.commitGesture(); this.sync(); this.busy = true; this.updateControls();
                this.pendingCG = { ...size, camera, name }; this.render();
                const raw = await this.captureScene({ ...size, cameraState: camera });
                if (epoch !== this.epoch || this.disposed) return;
                this.pendingCG.url = URL.createObjectURL(raw); this.render();
                await this.runJob("cg", { name, camera, characters: this.characters?.snapshot() || [], model_id: this.state.model_id }, raw); return;
            }
            if (button.dataset.action === "model") {
                if (this.dialogs.size) return;
                const { provider, camera_shift_lora, moge_refine_views } = this.state.model_settings;
                const usesShiftLoRA = (provider === "worldmirror" || provider === "moge") && !moge_refine_views;
                if (usesShiftLoRA && !camera_shift_lora) {
                    this.message("Select the trained Camera shift LoRA in model settings.", true);
                    return;
                }
                const existing = latestModel(this.document, this.state);
                if (existing) {
                    const confirmed = await this.dialog("Replace model?", [], "Replace model", {
                        message: "The existing model for this source will be deleted once the new model is ready.", danger: true,
                    });
                    if (!confirmed || epoch !== this.epoch || this.disposed || this.busy || this.state.job
                        || latestModel(this.document, this.state)?.model_id !== existing.model_id) return;
                }
                await this.runJob("model");
            }
        } catch (error) { if (epoch === this.epoch && !this.disposed) { this.clearPendingCG(); this.busy = false; this.render(); this.message(error.message, true); } }
        finally { if (navigating && epoch === this.epoch && !this.disposed) {
            this.busy = false; this.render();
            if (sourceFocus) [...this.container.querySelectorAll("[data-source-choice]")].find(item => item.dataset.sourceChoice === sourceFocus)?.focus?.({ preventScroll: true });
        } }
    }
    input(event) {
        if (this.restoring || this.busy || !this.state) return; const control = event.target;
        if (control.hasAttribute("data-expansion-split")) this.expansionCompare.style.setProperty("--split", control.value + "%");
        else if (control === this.fov || control === this.fovNumber) { const value = Number(control.value); if (!this.canEditCamera() || !control.value || !Number.isFinite(value)) return;
            this.beginGesture(); this.viewer.setCameraState({ ...this.viewer.getCameraState(), fov: Math.max(5, Math.min(120, value)) }, { emit: true });
        } else if (control.dataset.modelSetting && control.value !== "" && Number.isFinite(Number(control.value))) { this.state.model_settings[control.dataset.modelSetting] = Number(control.value); this.markEdited(); }
    }
    async change(event) {
        if (this.restoring || this.busy || this.state?.job) return; const control = event.target; let epoch = this.epoch;
        let navigating = false;
        try {
            if (control.hasAttribute("data-library")) { const id = control.value; if (!id || id === this.state.background_id) { control.value = this.state.background_id; return; } navigating = this.busy = true; this.updateControls(); epoch = ++this.epoch;
                await this.switchBackground(await this.request(`/backgrounds/${id}`), epoch);
            } else if (control.hasAttribute("data-model-provider")) {
                if (control.disabled || !Object.hasOwn(MODEL_NAMES, control.value)) return;
                this.state.model_settings.provider = control.value; this.markEdited(); this.render();
            } else if (control.hasAttribute("data-moge-refine-views")) {
                if (!["moge", "worldmirror"].includes(this.state.model_settings.provider) || control.disabled) return;
                this.state.model_settings.moge_refine_views = control.checked; this.markEdited(); this.updateControls();
            } else if (control.hasAttribute("data-aspect")) { this.state.settings.aspect = control.value; this.updateFrame(); this.markEdited();
            }
        } catch (error) {
            if (epoch === this.epoch && !this.disposed) {
                this.message(error.message, true);
                if (navigating && !this.selectedScene() && !this.modelError) void this.loadBackgroundModel(epoch);
            }
        }
        finally { if (navigating && epoch === this.epoch && !this.disposed) { this.busy = false; this.container.querySelector("[data-library]").value = this.state.background_id; this.updateControls(); } }
    }

    async runJob(action, extra = {}, raw = null) {
        if (!this.document) throw new Error("Select a saved background first."); const epoch = this.epoch, id = this.state.background_id;
        const savedJob = this.state.job;
        if (savedJob && savedJob.action !== action) throw new Error("Reconnect to the unfinished generation before starting another action.");
        let terminal = false;
        this.activeAction = action;
        this.busy = true; this.updateControls();
        this.message(action === "expand" ? "Submitting scene expansion…" : "Preparing " + action + "…", false, action === "expand");
        try {
            let result;
            if (savedJob) {
                this.jobId = savedJob.id || "";
                if (!this.jobId) {
                    result = await this.request(`/backgrounds/${id}/jobs/${savedJob.request_id}`);
                    if (epoch !== this.epoch || this.disposed) return;
                    if (!/^[a-f0-9]{32}$/.test(result.job_id)) throw new Error("Invalid generation response.");
                    this.jobId = savedJob.id = result.job_id; this.sync();
                }
                if (action === "cg" && this.state.frame && this.state.camera && this.viewer && this.scene?.model_id === this.state.model_id) {
                    this.pendingCG = { ...clone(this.state.frame), camera: clone(this.state.camera), name: savedJob.name };
                    this.render();
                    const preview = await this.captureScene({ ...this.state.frame, cameraState: this.state.camera });
                    if (epoch !== this.epoch || this.disposed) return;
                    this.pendingCG.url = URL.createObjectURL(preview); this.render(); this.restorePreview();
                }
            } else {
                const payload = { action, revision: this.document.revision, image_id: this.state.selected_image_id, source_variant: clone(this.state.selected_variant || null), model_settings: clone(this.state.model_settings), settings: clone(this.state.settings), ...extra };
                payload.request_id = cacheIdentity();
                this.state.job = { request_id: payload.request_id, node_id: String(this.node.id), action, name: extra.name || "" }; this.sync();
                if (action === "cg" && payload.settings.generation.seed_mode === "randomize") {
                    const seed = generateRandomSeed();
                    payload.settings.generation.seed = this.state.settings.generation.seed = seed;
                    payload.settings.generation.seed_mode = "fixed";
                    this.sync();
                }
                if (raw) { const form = new FormData(); form.append("data", JSON.stringify(payload)); form.append("image", raw, "raw.png"); result = await this.request(`/backgrounds/${id}/jobs`, { method: "POST", body: form }); } else result = await this.json(`/backgrounds/${id}/jobs`, payload);
                if (epoch !== this.epoch || this.disposed || id !== this.state.background_id) return;
                if (!/^[a-f0-9]{32}$/.test(result.job_id)) throw new Error("Invalid generation response.");
                this.jobId = result.job_id; this.state.job.id = this.jobId; this.sync();
            }
            this.updateControls();
            while (epoch === this.epoch && !this.disposed && id === this.state.background_id) {
                result = await this.request(`/jobs/${this.jobId}`); if (epoch !== this.epoch || this.disposed || id !== this.state.background_id) return;
                if (["completed", "failed", "cancelled"].includes(result.status)) break;
                if (action === "expand") this.message(result.message || "Expanding scene…", false, true);
                await new Promise(resolve => { this.pollResolve = resolve; this.pollTimer = setTimeout(resolve, 500); });
            }
            if (epoch !== this.epoch || this.disposed || id !== this.state.background_id) return; terminal = true; if (result.status !== "completed") throw new Error(result.error || "Generation stopped.");
            if (action === "expand") {
                const frameId = result.result?.expansion_frame_id;
                if (frameId && this.selectedScene()) this.scene.expansion_frame_id = frameId;
                else await this.loadBackgroundModel(epoch);
                if (epoch === this.epoch && !this.disposed) {
                    if (this.selectedScene()?.expansion_frame_id) this.message("Review the QI2 frame before rebuilding the scene.", false, true);
                    else if (this.selectedScene()?.expansion_id) this.message("Explore the expanded scene, then save or discard it.", false, true);
                }
                return;
            }
            const document = result.result.document;
            if (document.background_id !== id) throw new Error("The generation result belongs to another background.");
            const previewPending = this.pendingCG && this.previewSource === this.pendingCG;
            if (action === "cg") { const cg = document.cgs.at(-1); Object.assign(this.state, { selected_cg_id: cg.cg_id, model_id: cg.model_id, camera: clone(cg.camera), frame: { width: cg.width, height: cg.height } }); }
            else { this.state.model_id = document.models.at(-1).model_id; this.state.selected_cg_id = ""; this.state.camera = null; this.state.frame = null; }
            await this.acceptDocument(document, false, epoch); this.sync(); this.render(); this.message(action === "cg" ? "CG saved with its camera. The selected CG is the workflow output." : "Model ready. Compose the camera and press Generate CG.");
            if (previewPending && this.previewSource === this.pendingCG) this.openPreview(document.cgs.at(-1), this.state.preview);
        } catch (error) {
            if (!savedJob && !this.jobId && error.status >= 400 && error.status < 500) { terminal = true; throw error; }
            if (this.state.job && !terminal) {
                if (error.status === 404) { terminal = true; throw new Error("The saved generation is no longer available. Saved assets are preserved; select a finished result or generate again."); }
                throw new Error("Generation status is unavailable. Reconnection will retry when the connection returns or this window becomes active. " + error.message);
            }
            throw error;
        } finally { if (epoch === this.epoch && !this.disposed) { this.clearPendingCG(); if (terminal) { delete this.state.job; if (this.state.preview?.kind === "pending") delete this.state.preview; } this.jobId = ""; this.activeAction = ""; this.busy = false; this.render(); this.sync(); } }
    }
    updateFrame() {
        if (!this.state || this.restoring || this.busy) return;
        const source = selectedSource(this.document, this.state)?.image;
        const mp = this.state.settings.resolution_mp || (source ? source.width * source.height / 1e6 : 2);
        this.state.frame = frameDimensions(this.state.settings.aspect, mp);
        if (this.viewer && this.scene?.model_id === this.state.model_id) this.viewer.setCaptureSettings({ ...this.state.frame, show_camera_frame: true });
        this.resize();
        this.sync();
    }

    restorePreview() {
        const preview = this.state?.preview;
        if (!preview) return;
        const cg = preview.kind === "cg" ? this.document?.cgs.find(cg => cg.cg_id === preview.cg_id) : preview.kind === "pending" ? this.pendingCG : null;
        if (preview.kind === "cg" && !cg) { this.message("The saved preview is unavailable. Its workflow reference is preserved; select another preview or return to the model.", true); return; }
        if (preview.kind !== "pending" || cg?.url) this.openPreview(cg, preview);
    }

    openPreview(cg = null, savedView = null) {
        const source = cg || selectedSource(this.document, this.state)?.image;
        if (!source?.url) return;
        this.cgPreviewClose?.();
        const opener = document.activeElement;
        const panel = document.createElement("div"); panel.className = "bg-image-preview";
        panel.setAttribute("role", "region"); panel.setAttribute("aria-label", cg ? `${cg.name} preview` : "Background preview");
        const viewport = document.createElement("div"); viewport.className = "bg-image-viewport"; viewport.dataset.previewViewport = ""; viewport.tabIndex = 0;
        viewport.setAttribute("aria-label", "Image preview. Scroll to zoom, drag to pan.");
        const image = document.createElement("img"); image.dataset.largePreview = ""; image.alt = cg ? `Full-size ${cg.name}` : "Full-size selected background";
        image.draggable = false; image.style.visibility = "hidden"; viewport.appendChild(image);
        const close = () => {
            if (!this.dialogs.delete(close)) return;
            const returnFocus = panel.contains(document.activeElement);
            if (this.cgPreviewClose === close) { this.cgPreviewClose = null; this.previewSource = null; if (!this.restoring && !this.disposed) { delete this.state.preview; this.sync(); } }
            dispose(); panel.remove(); if (!this.disposed) this.updateControls();
            if (returnFocus && !this.restoring && !this.disposed && opener?.isConnected) opener.focus({ preventScroll: true });
        };
        const button = document.createElement("button"); button.type = "button"; button.className = "vnccs-common-modal-btn";
        button.dataset.closePreview = ""; button.textContent = "Close"; button.onclick = close;
        panel.addEventListener("keydown", event => { if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(); } });
        panel.append(viewport, button); this.host.parentElement.appendChild(panel);
        this.cgPreviewClose = close; this.previewSource = source;
        this.state.preview = { kind: cg && cg === this.pendingCG ? "pending" : cg ? "cg" : "source", ...(cg?.cg_id ? { cg_id: cg.cg_id } : {}), zoom: savedView?.zoom || 1, center: clone(savedView?.center || [.5, .5]) };
        const dispose = this.attachPreviewNavigation(viewport, image, this.state.preview, view => {
            if (this.cgPreviewClose !== close || this.restoring || this.disposed) return;
            Object.assign(this.state.preview, view); this.sync();
        });
        this.dialogs.add(close);
        image.src = this.api.apiURL(source.url); image.dataset.sourceURL = source.url;
        this.render(); this.sync();
    }

    attachPreviewNavigation(viewport, image, savedView = null, onChange = null) {
        const events = new AbortController();
        let scale = 1, fitScale = 1, x = 0, y = 0, width = 0, height = 0, ready = false, closed = false, drag = null;
        const apply = (notify = false) => {
            image.style.width = `${image.naturalWidth}px`; image.style.height = `${image.naturalHeight}px`;
            image.style.transform = `matrix(${scale},0,0,${scale},${x},${y})`; image.style.visibility = "visible";
            if (notify) onChange?.({ zoom: Math.max(1, Math.min(8, scale / fitScale)), center: [(width / 2 - x) / scale / image.naturalWidth, (height / 2 - y) / scale / image.naturalHeight] });
        };
        const fit = () => {
            if (closed || !image.complete || !image.naturalWidth || !viewport.clientWidth || !viewport.clientHeight) return;
            const ratio = ready ? scale / fitScale : savedView?.zoom || 1;
            const centerX = ready ? (width / 2 - x) / scale : image.naturalWidth * (savedView?.center[0] ?? .5);
            const centerY = ready ? (height / 2 - y) / scale : image.naturalHeight * (savedView?.center[1] ?? .5);
            width = viewport.clientWidth; height = viewport.clientHeight;
            fitScale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
            scale = fitScale * ratio; x = width / 2 - centerX * scale; y = height / 2 - centerY * scale;
            ready = true; apply();
        };
        const point = event => {
            const rect = viewport.getBoundingClientRect();
            return { x: (event.clientX - rect.left) * viewport.clientWidth / rect.width, y: (event.clientY - rect.top) * viewport.clientHeight / rect.height };
        };
        const finish = event => {
            if (!drag || event?.pointerId != null && event.pointerId !== drag.id) return;
            const id = drag.id; drag = null; viewport.classList.remove("is-dragging");
            if (viewport.hasPointerCapture(id)) viewport.releasePointerCapture(id);
        };
        viewport.onwheel = event => {
            event.preventDefault(); event.stopPropagation();
            if (!ready || !event.deltaY) return;
            const anchor = point(event), next = Math.max(fitScale, Math.min(fitScale * 8, scale * (event.deltaY < 0 ? 1.12 : .88)));
            x = anchor.x - (anchor.x - x) * next / scale; y = anchor.y - (anchor.y - y) * next / scale; scale = next;
            if (scale === fitScale) { x = (width - image.naturalWidth * scale) / 2; y = (height - image.naturalHeight * scale) / 2; }
            apply(true);
        };
        viewport.onpointerdown = event => {
            if (!ready || ![0, 1].includes(event.button) || event.isPrimary === false) return;
            event.preventDefault(); event.stopPropagation(); viewport.focus({ preventScroll: true });
            drag = { ...point(event), id: event.pointerId, buttons: event.button === 0 ? 1 : 4 };
            viewport.setPointerCapture(event.pointerId); viewport.classList.add("is-dragging");
        };
        viewport.onpointermove = event => {
            if (!drag || event.pointerId !== drag.id) return;
            if (!(event.buttons & drag.buttons)) { finish(event); return; }
            event.preventDefault(); event.stopPropagation(); const next = point(event);
            x += next.x - drag.x; y += next.y - drag.y; Object.assign(drag, next); apply(true);
        };
        viewport.onpointerup = viewport.onpointercancel = viewport.onlostpointercapture = finish;
        viewport.onauxclick = event => { if (event.button === 1) event.preventDefault(); };
        image.onload = () => { ready = false; fit(); };
        image.onerror = () => { ready = false; image.style.visibility = "hidden"; viewport.setAttribute("aria-label", "Preview image unavailable. Close this window and try again."); };
        window.addEventListener("blur", () => finish(), { signal: events.signal });
        document.addEventListener("visibilitychange", () => { if (document.hidden) finish(); }, { signal: events.signal });
        const observer = new ResizeObserver(fit); observer.observe(viewport); fit();
        return () => {
            if (closed) return;
            closed = true; finish(); events.abort(); observer.disconnect(); image.onload = image.onerror = null;
            for (const type of ["wheel", "pointerdown", "pointermove", "pointerup", "pointercancel", "lostpointercapture", "auxclick"]) viewport[`on${type}`] = null;
        };
    }

    clearPendingCG() {
        if (this.pendingCG && this.previewSource === this.pendingCG) this.cgPreviewClose?.();
        if (this.pendingCG?.url) URL.revokeObjectURL(this.pendingCG.url);
        this.pendingCG = null;
    }
    resize() {
        const width = this.frame.clientWidth || this.node.size?.[0] || LAYOUT_WIDTH;
        const scale = width / LAYOUT_WIDTH;
        const height = `${Math.max(300, (this.node.size?.[1] || CREATOR_MIN_HEIGHT) - 90) / scale}px`;
        if (height !== this.container.style.height || scale !== this.layoutScale) this.customSelects._closeDropdown?.(false);
        this.layoutScale = scale;
        this.container.style.width = `${LAYOUT_WIDTH}px`;
        this.container.style.zoom = scale;
        if (this.container.style.height !== height) this.container.style.height = height;
        let missingHeight = 0;
        for (const panel of this.container.querySelectorAll(".bg-panel")) {
            if (!panel.offsetHeight) continue;
            const bounds = panel.getBoundingClientRect(), scale = bounds.height / panel.offsetHeight;
            if (!(scale > 0)) continue;
            const style = getComputedStyle(panel), inset = parseFloat(style.paddingBottom) + parseFloat(style.borderBottomWidth);
            for (const child of panel.children) if (child.getClientRects().length) {
                missingHeight = Math.max(missingHeight, (child.getBoundingClientRect().bottom - bounds.bottom) / scale + inset);
            }
        }
        if (missingHeight > 1) {
            this.node.setSize?.([this.node.size[0], this.node.size[1] + Math.ceil(missingHeight * scale)]);
            this.container.style.height = `${(this.node.size[1] - 90) / scale}px`;
        }
        this.updateSourceScroll();
        this.viewer?.resize();
    }
    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.clearPendingCG();
        ++this.epoch; this.canvasNavigation?.(); this.events.abort(); clearTimeout(this.wheelTimer); clearTimeout(this.pollTimer); this.pollResolve?.();
        this.clearSourceLoad();
        this.galleryObserver?.disconnect();
        for (const finish of [...this.dialogs]) finish(null); this.observer.disconnect(); this.customSelects.dispose(); this.qi2.dispose(); this.characterPanel?.dispose(); this.characters?.dispose(); this.viewer?.dispose(); this.frame.remove();
    }
}


function hideState(node) {
    const widget = node.widgets?.find(item => item.name === "cg_data");
    if (!widget) return;
    widget.hidden = true; widget.computeSize = () => [0, -4]; widget.draw = () => {};
    if (widget.element) widget.element.style.display = "none";
    if (widget.inputEl) widget.inputEl.style.display = "none";
    return widget;
}

function fit(node) {
    if (node.size?.[1] < CREATOR_MIN_HEIGHT) node.setSize?.([node.size[0], CREATOR_MIN_HEIGHT]);
    syncDOMWidgetWidth(node, "cg_ui");
    node.cgCreator?.resize();
}

app.registerExtension({
    name: "VNCCS.CGCreator",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_CGCreator") return;
        const created = nodeType.prototype.onNodeCreated, configured = nodeType.prototype.onConfigure;
        const resized = nodeType.prototype.onResize, serialized = nodeType.prototype.onSerialize, removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            if (this.cgCreator) return result;
            const state = hideState(this);
            this.cgCreator = new CGCreatorWidget(this, api);
            this.cgCreatorDOM = this.addDOMWidget("cg_ui", "ui", this.cgCreator.frame, { serialize: false, hideOnZoom: false, getMinHeight: () => 300 });
            if (state) {
                const beforeQueued = state.beforeQueued;
                state.beforeQueued = (...args) => {
                    beforeQueued?.apply(state, args);
                    if (this.cgCreator.restoring) throw new Error("CG Creator is still restoring its scene.");
                    if (this.cgCreator.busy || this.cgCreator.state?.job) throw new Error("Wait for CG Creator to finish its current operation.");
                    this.cgCreator.sync({ trackChange: false });
                };
                const serializeValue = state.serializeValue;
                state.serializeValue = (...args) => {
                    const result = serializeValue?.apply(state, args);
                    return this.cgCreator.sync({ trackChange: false }) ?? result ?? state.value;
                };
            }
            this.setSize?.([LAYOUT_WIDTH, CREATOR_MIN_HEIGHT]); fit(this);
            syncDOMWidgetWidthSoon(this, "cg_ui");
            this._cgRestore = setTimeout(() => this.cgCreator?.restore(), 0);
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            this.cgCreator?.invalidateRestore();
            const result = configured?.apply(this, arguments);
            hideState(this); clearTimeout(this._cgRestore);
            this._cgRestore = setTimeout(() => this.cgCreator?.restore(), 0); fit(this); return result;
        };
        nodeType.prototype.onResize = function () { const result = resized?.apply(this, arguments); fit(this); return result; };
        nodeType.prototype.onSerialize = function (info) {
            const result = serialized?.apply(this, arguments), raw = this.cgCreator?.sync({ trackChange: false });
            const index = this.widgets?.findIndex(item => item.name === "cg_data");
            if (raw && info && index >= 0 && Array.isArray(info.widgets_values)) info.widgets_values[index] = raw;
            return result;
        };
        nodeType.prototype.onRemoved = function () { clearTimeout(this._cgRestore); this.cgCreator?.dispose(); this.cgCreator = null; return removed?.apply(this, arguments); };
    },
});
