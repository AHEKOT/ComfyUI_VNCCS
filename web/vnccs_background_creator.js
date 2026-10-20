import { createStylePicker, STYLE_PICKER_CSS } from "./character_styles.mjs?v=style-mix-9";
import { app } from "../../scripts/app.js";
import { vnccsApi, watchConnection } from "./vnccs_transport.js";
import { createSakuraInteractions } from "./sakura_interactions.mjs";
import { presetSelection } from "./character_presets.mjs?v=legacy-tags-1";
import { SAKURA_THEME_CSS, SAKURA_INTERACTION_CSS, SAKURA_CREATOR_CONTROL_CSS, RESOLUTION_SCALE_MIN_MP, RESOLUTION_SCALE_MAX_MP, RESOLUTION_SCALE_STEP_MP, createTraitInput, injectStyles, showModal, registerCleanup, createRequestGuard, generateRandomSeed, enableMiddleMouseCanvasPan, attachHelpTooltips, setHelpText, syncDOMWidgetWidth, syncDOMWidgetWidthSoon, animateWizardRows, createControlCenterClient, installResolvedSeedTransport } from "./vnccs_common.js";

const BACKGROUND_MANNEQUIN_INTERIOR_URL = new URL("./assets/VNCCS_Background_Mannequin_Interior.png", import.meta.url).href;
const BACKGROUND_MANNEQUIN_EXTERIOR_URL = new URL("./assets/VNCCS_Background_Mannequin_Exterior.png", import.meta.url).href;
const CREATOR_ZOOM = 0.67;
const CREATOR_MIN_HEIGHT = 840;
const GENERATION_PANEL_WIDTH = (1280 / CREATOR_ZOOM - 24) * .35;
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.vnccs-background-creator { ${SAKURA_THEME_CSS} --warning:#ffaa00; --shadow-subtle:0 2px 8px rgba(0,0,0,0.3); --shadow-elevated:0 8px 32px rgba(0,0,0,0.5); }
${SAKURA_INTERACTION_CSS}
${SAKURA_CREATOR_CONTROL_CSS}
${STYLE_PICKER_CSS.replaceAll(".vnccs-creator-container", ".vnccs-background-creator")}

.vnccs-background-creator { display:flex; flex-direction:column; min-height:0; padding:12px; gap:12px; background:var(--bg-primary); }
.vnccs-background-creator button:disabled { cursor:not-allowed; }

.vnccs-background-creator .bg-step { flex:1 1 0; min-width:0; min-height:0; display:grid; gap:12px; }
.vnccs-background-creator .bg-background { grid-template-columns:minmax(360px,1fr) minmax(${240 + GENERATION_PANEL_WIDTH * .2}px,calc((100% - 24px)*.21875 + ${GENERATION_PANEL_WIDTH * .2}px)) ${GENERATION_PANEL_WIDTH * .8}px; grid-template-rows:minmax(0,1fr); }
.vnccs-background-creator .bg-panel { min-width:0; min-height:0; overflow:visible; }
.vnccs-background-creator .bg-generation { gap:4px; }
.vnccs-background-creator .bg-generation .vnccs-common-creator-subsection { gap:4px; }
.vnccs-background-creator .bg-generation .vnccs-common-creator-gen-param-grid { row-gap:8px; }
.vnccs-background-creator .bg-generation .vnccs-common-creator-qi2-cache { row-gap:8px; padding:8px; }
.vnccs-background-creator .bg-generation .vnccs-common-creator-lora-stack { gap:4px; margin-top:4px; padding-top:8px; }
.vnccs-background-creator .bg-panel>* { flex:0 0 auto; min-width:0; }
.vnccs-background-creator .bg-panel>button { min-height:40px; }
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
.vnccs-background-creator .bg-variation-strip { position:relative; min-width:0; }
.vnccs-background-creator .bg-variation-cards { display:flex; gap:10px; min-width:0; overflow-x:auto; overscroll-behavior-x:contain; scrollbar-width:thin; scrollbar-color:var(--accent-border) transparent; scroll-snap-type:x proximity; padding:2px; }
.vnccs-background-creator .bg-variation-item { display:grid; grid-template-rows:18px auto 216px; align-content:start; gap:7px; flex:0 0 calc((100% - 20px)/3); min-width:220px; scroll-snap-align:start; }
.vnccs-background-creator .bg-variation-scroll { position:absolute; top:var(--variation-nav-top,80px); z-index:4; width:36px; height:36px; padding:8px; border-radius:50%; background:color-mix(in srgb,var(--bg-primary) 88%,transparent); border:1px solid color-mix(in srgb,var(--accent) 45%,var(--accent-border)); box-shadow:0 5px 18px rgba(0,0,0,.42),0 0 14px color-mix(in srgb,var(--accent) 22%,transparent); backdrop-filter:blur(8px); transform:translateY(-50%); }
.vnccs-background-creator .bg-variation-scroll[data-scroll-variation="-1"] { left:8px; }
.vnccs-background-creator .bg-variation-scroll[data-scroll-variation="1"] { right:8px; }
.vnccs-background-creator .bg-variation-scroll:hover:not(:disabled) { border-color:var(--accent); background:color-mix(in srgb,var(--bg-hover) 92%,transparent); }
.vnccs-background-creator .bg-variation-scroll:disabled { opacity:.28; }
.vnccs-background-creator .bg-variation-item>span { padding-left:4px; color:var(--text-primary); font-size:12px; line-height:18px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vnccs-background-creator .bg-variation-preview { position:relative; }
.vnccs-background-creator [data-view-variation] { position:relative; display:block; width:100%; aspect-ratio:16/9; min-height:52px; padding:0; background:var(--bg-surface); border:2px solid color-mix(in srgb,var(--accent-lavender) 60%,var(--bg-surface)); border-radius:var(--radius-md); overflow:hidden; }
.vnccs-background-creator [data-view-variation]:hover:not(:disabled) { border-color:var(--accent); background:var(--bg-hover); }
.vnccs-background-creator [data-view-variation]:disabled { opacity:1; cursor:default; }
.vnccs-background-creator [data-variation-image] { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:cover; border-radius:10px; pointer-events:none; }
.vnccs-background-creator [data-variation-image].is-placeholder { opacity:.64; filter:saturate(.9) brightness(.88); }
.vnccs-background-creator .bg-variation-delete { position:absolute; top:8px; right:8px; border:1px solid var(--accent-border); border-radius:6px; background:var(--bg-primary); color:var(--text-primary); cursor:pointer; width:28px; height:28px; padding:0; font-size:20px; line-height:1; }
.vnccs-background-creator .bg-variation-delete:hover:not(:disabled) { color:var(--error); }
.vnccs-background-creator .selected { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }

.vnccs-background-creator .bg-variation-settings { display:grid; grid-template-rows:repeat(4,54px); height:216px; padding:0 9px; border:1px solid var(--accent-border); border-radius:var(--radius-md); background:rgba(255,255,255,.025); box-sizing:border-box; }
.vnccs-background-creator .bg-variation-name { display:grid; grid-template-columns:80px minmax(0,1fr); align-items:center; gap:7px; min-width:0; height:54px; padding:7px 0; border-bottom:1px solid var(--accent-border); box-sizing:border-box; }
.vnccs-background-creator .bg-variation-name .vnccs-common-creator-label { min-width:0; font-size:9px; line-height:1.15; white-space:nowrap; }
.vnccs-background-creator .bg-variation-name input { width:100%; min-width:0; min-height:36px; height:36px; padding:7px 10px; font-size:11px; text-overflow:ellipsis; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-row { grid-template-columns:minmax(62px,30%) minmax(0,1fr) 34px; gap:7px; min-width:0; min-height:54px; height:54px; padding:7px 0; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-label { min-width:0; font-size:11px; line-height:1.15; overflow-wrap:anywhere; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-editor,
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-values { min-width:0; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-values { flex-wrap:nowrap; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-token { max-width:100%; min-width:0; padding:6px 9px; font-size:11px; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-empty { max-width:100%; font-size:11px; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vnccs-background-creator .bg-variation-settings .vnccs-common-creator-trait-add { width:34px; height:34px; font-size:18px; }
.vnccs-background-creator .bg-attributes>.bg-description { flex:1 1 0; min-height:100px; }
.vnccs-background-creator .bg-description textarea { flex:1; min-height:64px; resize:none; }
.vnccs-background-creator .bg-description textarea:focus { background:rgba(255,255,255,0.04); box-shadow:none; }
.vnccs-background-creator .bg-negative textarea { flex:none; height:64px; min-height:64px; max-height:64px; resize:none; }
.vnccs-background-creator .bg-image-dialog { width:min(1400px,calc(100% - 32px)); max-width:min(1400px,calc(100% - 32px)); height:min(900px,calc(100% - 32px)); }
.vnccs-background-creator .bg-image-viewport { position:relative; flex:1; min-height:0; overflow:hidden; touch-action:none; cursor:grab; }
.vnccs-background-creator .bg-image-viewport.is-dragging { cursor:grabbing; }
.vnccs-background-creator .bg-image-viewport:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.vnccs-background-creator .bg-image-dialog img { position:absolute; left:0; top:0; display:block; max-width:none; max-height:none; transform-origin:0 0; pointer-events:none; user-select:none; }

.vnccs-background-creator .bg-announcement { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip-path:inset(50%); white-space:nowrap; }
.vnccs-background-creator .error { color:var(--error) !important; }
.vnccs-background-creator { position:relative; color:var(--text-primary); font:13px var(--font); width:100%; box-sizing:border-box; overflow:hidden; zoom:${CREATOR_ZOOM}; }
.vnccs-background-creator *, .vnccs-background-creator *::before, .vnccs-background-creator *::after { box-sizing:border-box; }
.vnccs-background-creator [hidden] { display:none !important; }

.vnccs-background-creator .bg-dialog-fields { display:flex; flex-direction:column; gap:12px; }
.vnccs-background-creator .vnccs-common-modal { max-height:calc(100% - 32px); overflow:visible; }
.vnccs-background-creator .bg-wizard-modal { display:flex; flex-direction:column; gap:10px; width:100%; min-width:0; box-sizing:border-box; }
.vnccs-background-creator .bg-wizard-text { color:var(--text-secondary); font-size:12px; line-height:1.45; white-space:normal; overflow-wrap:anywhere; }
.vnccs-background-creator .bg-wizard-modal textarea { width:100%; min-height:110px; box-sizing:border-box; resize:vertical; }
.vnccs-background-creator .vnccs-common-modal.bg-wizard-dialog { overflow:hidden; }
.vnccs-background-creator .vnccs-common-modal.bg-error-dialog { overflow:hidden; }
.vnccs-background-creator .bg-error-text { color:var(--text-secondary); font-size:12px; line-height:1.45; white-space:normal; overflow-wrap:anywhere; }
.vnccs-background-creator .vnccs-common-creator-model-picker { position:relative; }
.vnccs-background-creator .vnccs-common-creator-model-picker-menu { position:absolute; top:calc(100% + 6px); left:0; right:0; z-index:30; max-height:320px; overflow-y:auto; background:var(--bg-primary); }
.vnccs-background-creator.bg-compact .bg-step { grid-template-columns:minmax(0,1.15fr) repeat(2,minmax(0,1fr)); }
.vnccs-background-creator.bg-compact .bg-panel { padding:10px; gap:6px; }
.vnccs-background-creator.bg-compact .bg-actions { grid-template-columns:minmax(0,2fr) repeat(2,minmax(0,1fr)); }
.vnccs-background-creator .vnccs-style-toolbar > button { padding:10px; border-radius:var(--radius-md); cursor:pointer; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; font-size:11px; font-family:var(--font); color:white; text-align:center; transition:all var(--transition); position:relative; overflow:hidden; }
.vnccs-background-creator .vnccs-style-gallery :is(input,select,textarea,.vnccs-common-ui-select-trigger) { box-sizing:border-box; min-width:0; color:var(--text-primary); background:var(--bg-surface); border:1px solid var(--accent-border); border-radius:8px; padding:8px 12px; font:12px var(--font); }
.vnccs-background-creator .vnccs-style-gallery :is(input,select,.vnccs-common-ui-select-trigger) { height:34px; }
.vnccs-background-creator .vnccs-style-gallery .vnccs-style-size-slider { width:180px; max-width:50%; height:auto; padding:0; border:0; background:transparent; }
.vnccs-background-creator .vnccs-style-summary>.vnccs-style-placeholder { width:88px; }
@media (prefers-reduced-motion:reduce) { .vnccs-background-creator *, .vnccs-background-creator *::before, .vnccs-background-creator *::after { transition:none !important; animation:none !important; } }

`;

export const DEFAULT_EDITOR = Object.freeze({
    description: "", scene_description: "", front: "", back: "", left: "", right: "", prompt: "", attributes: { scene_type: "Interior", time: "Day" },
    variants: ["Morning", "Evening", "Night"].map(name => ({ id: name.toLowerCase(), name, time: name, weather: "", additional_details: "" })),
    aspect: "16:9", style: "", aesthetics: "", negative: "",
    generation: { model_type: "qi2", steps: 25, cfg: 1, sampler: "euler", scheduler: "simple", seed: 0, resolution_mp: 2, seed_mode: "fixed", turbo: false, turbo_previous_settings: null, prompt_enhance: true, cache_device: "gpu", cache_dtype: "int8", lora_stack: [], diffusion_model_name: "qwen_image_2.1_int8_convrot.safetensors" },
});
export const MOSS_GENERATION = Object.freeze({ model_type: "moss", steps: 25, cfg: 2.5, sampler: "er_sde", scheduler: "simple", turbo: false,
    diffusion_model_name: "moss_image_sft_int8_convrot.safetensors" });
export const clone = value => JSON.parse(JSON.stringify(value));

const LEGACY_SCENE_LABELS = { composition: "Composition", location: "Location", era: "Era", architecture: "Architecture", lighting: "Lighting", weather: "Weather", mood: "Atmosphere", objects: "Objects and furnishings", details: "Additional details" };
const COMPOSITION_FIELDS = ["front", "back", "left", "right"];
const promptLine = value => String(value || "").replace(/\s+/g, " ").trim();

export function parsePrompt(text) {
    const match = /^Style:[ \t]*(.*?)\r?\nTime:[ \t]*(.*?)\r?\nInterior\/Exterior:[ \t]*(.*?)\r?\nScene description:[ \t]*([\s\S]*)$/.exec(text.trim());
    if (!match) return null;
    const parts = match[4].split(/\r?\n(Front|Back|Left|Right):[ \t]*/);
    const parsed = { style: match[1], time: match[2], scene_type: match[3], scene_description: parts[0] };
    for (let i = 1; i < parts.length; i += 2) parsed[parts[i].toLowerCase()] = parts[i + 1].trim();
    return parsed;
}

export function legacySceneDescription(editor) {
    if (editor.prompt) return parsePrompt(editor.prompt)?.scene_description ?? editor.prompt;
    return [editor.description, ...Object.entries(LEGACY_SCENE_LABELS).map(([key, label]) => editor.attributes?.[key] ? `${label}: ${editor.attributes[key]}.` : "")].filter(Boolean).join("\n");
}

export function editorDefaults(value = {}) {
    const editor = { ...clone(DEFAULT_EDITOR), ...value, attributes: { ...DEFAULT_EDITOR.attributes, ...value.attributes },
        generation: { ...clone(DEFAULT_EDITOR.generation), ...(value.generation?.model_type === "moss" ? MOSS_GENERATION : {}), ...value.generation } };
    const settings = editor.generation;
    if (!["fixed", "randomize"].includes(settings.seed_mode)) throw new Error("Unsupported background seed mode.");
    if (settings.model_type === "moss") settings.turbo = false;
    else if (isTurboModel(settings.diffusion_model_name)) {
        settings.turbo = false;
        if (!Object.hasOwn(value.generation || {}, "steps")) settings.steps = 8;
    } else if (settings.turbo) { settings.steps = 6; settings.cfg = 1; }
    if (!["qi2", "moss"].includes(editor.generation.model_type)) throw new Error("Unsupported background model type.");
    if (!Object.hasOwn(value, "scene_description")) {
        const parsed = value.prompt && parsePrompt(value.prompt);
        if (parsed) {
            editor.style = parsed.style; editor.attributes.time = parsed.time; editor.attributes.scene_type = parsed.scene_type;
            for (const key of COMPOSITION_FIELDS) if (!Object.hasOwn(value, key) && Object.hasOwn(parsed, key)) editor[key] = parsed[key];
        }
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
    const style = String(editor.style || "").replace(/^\s*Style\s*:\s*/i, "").replace(/\bVisual reference\s*:/gi, "Reference:");
    return [`Style: ${promptLine(style)}\nTime: Day\nInterior/Exterior: ${promptLine(editor.attributes.scene_type)}\nScene description: ${promptLine(body)}`,
        ...COMPOSITION_FIELDS.filter(key => editor[key]).map(key => `${key[0].toUpperCase() + key.slice(1)}: ${promptLine(editor[key])}`)].join("\n");
}

const isTurboModel = name => String(name || "").replace(/\\/g, "/").split("/").at(-1) === "qwen_image_2.1_turbo_int8_convrot.safetensors";
const modelPath = entry => String(entry.local_path || "").replace(/\\/g, "/").replace(/^models\/[^/]+\//, "");

export function readState(raw) {
    if (typeof raw !== "string" || new Blob([raw]).size > 4 * 1024 * 1024) throw new Error("Background Creator state is too large.");
    const value = JSON.parse(raw);
    if (!value || Array.isArray(value) || value.schema_version !== 2) throw new Error("Unsupported Background Creator workflow state.");
    if (value.background_name != null && (typeof value.background_name !== "string" || [...value.background_name].length > 96)) throw new Error("Invalid saved background name.");
    for (const key of ["background_id", "selected_image_id"]) {
        if (value[key] != null && (typeof value[key] !== "string" || (value[key] && !/^[a-f0-9]{32}$/.test(value[key])))) throw new Error("Invalid background asset reference.");
    }
    if (value.editor != null && (typeof value.editor !== "object" || Array.isArray(value.editor))) throw new Error("Invalid background editor state.");
    return { schema_version: 2, background_id: "", selected_image_id: "",
        ...value, editor: editorDefaults(value.editor) };
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
const compactField = (label, control) => `<label class="vnccs-common-creator-gen-param-field"><span class="vnccs-common-creator-label">${label}</span>${control}</label>`;
const validNumber = control => control.value.trim() !== "" && Number.isFinite(Number(control.value)) && control.validity?.valid !== false;
const toggleCard = (label, path) => `<label class="vnccs-common-creator-model-card is-installed" data-setting-card="${path}"><span class="vnccs-common-creator-model-card-top"><span class="vnccs-common-creator-model-card-badge ok"></span><span class="vnccs-common-creator-model-card-name">${label}</span><span class="vnccs-common-creator-model-card-status" data-toggle-status></span><span class="vnccs-common-creator-toggle vnccs-common-creator-model-card-toggle"><input type="checkbox" data-setting="${path}" aria-label="${label}"><span class="vnccs-common-creator-toggle-track"></span><span class="vnccs-common-creator-toggle-thumb"></span></span></span></label>`;
export class BackgroundCreatorWidget {
    constructor(node, api) {
        injectStyles(STYLE, "vnccs-background-creator");
        this.node = node; this.api = api;
        this.state = null; this.document = null;
        this.epoch = 0; this.restoring = true; this.busy = false; this.disposed = false;
        this.workflowBusy = true; this.beginWorkflowStatusRequest = createRequestGuard(node);
        this.events = new AbortController(); this.operations = Promise.resolve(); this.saveSerial = Promise.resolve(); this.dialogs = new Set();
        this.container = document.createElement("div");
        this.container.className = "vnccs-background-creator";
        this.container.innerHTML = `
            <section class="bg-step bg-background" data-panel="background">
                <section class="bg-panel bg-source"><div class="vnccs-common-creator-section-title">Background select</div><select data-library aria-label="Saved background"><option value="">Loading backgrounds…</option></select>
                    <div class="bg-row bg-actions"><button class="primary" data-action="background">Generate background</button><button class="success" data-new>New</button><button class="danger" data-delete>Delete</button></div>
                    <div class="bg-preview vnccs-common-creator-preview-container"><img data-preview alt="Selected background" hidden><span class="vnccs-common-creator-placeholder" data-empty><span class="vnccs-common-creator-placeholder-title">Restoring background…</span><span class="vnccs-common-creator-placeholder-hint">Loading saved workflow settings.</span></span><button class="bg-enlarge" data-enlarge aria-label="Open full-size background preview" hidden></button><div class="vnccs-common-creator-generation-layer" data-generation-layer hidden aria-hidden="true"><img class="vnccs-common-creator-generation-img vnccs-common-creator-generation-base" alt=""><img class="vnccs-common-creator-generation-img vnccs-common-creator-generation-new" alt=""><span class="vnccs-common-creator-generation-scan"></span></div></div>
                    <button data-upload>Upload image</button>
                    <div class="bg-variation-head"><span class="vnccs-common-creator-section-title">Variants</span><button type="button" data-add-variation>Add variant</button></div>
                    <div class="bg-variation-strip"><button type="button" class="vnccs-common-creator-sprite-nav-btn bg-variation-scroll" data-scroll-variation="-1" aria-label="Scroll variants left"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg></button><div class="bg-variation-cards" data-variation-cards data-sakura-scroll="background-variants"></div><button type="button" class="vnccs-common-creator-sprite-nav-btn bg-variation-scroll" data-scroll-variation="1" aria-label="Scroll variants right"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 6 6 6-6 6"/></svg></button></div>
                </section>
                <section class="bg-panel bg-attributes"><div class="vnccs-common-creator-section-title">Scene attributes</div><button class="primary bg-wide" data-wizard>Background wizard</button>
                    <div class="bg-field" data-style-picker></div>
                    <div class="bg-field"><span>Interior/Exterior</span><div class="bg-segments" data-trait="scene_type"><button data-value="Interior">Interior</button><button data-value="Exterior">Exterior</button></div></div>
                    <label class="bg-field bg-description"><span>Scene description</span><textarea data-setting="scene_description" rows="3" maxlength="40000" placeholder="Describe the overall setting, atmosphere and lighting."></textarea></label>
                    <label class="bg-field bg-description"><span>Front</span><textarea data-setting="front" rows="3" maxlength="40000" placeholder="Describe the nearest surfaces, objects, or framing elements."></textarea></label>
                    <label class="bg-field bg-description"><span>Back</span><textarea data-setting="back" rows="3" maxlength="40000" placeholder="Describe the distant environment, horizon, architecture, or backdrop."></textarea></label>
                    <label class="bg-field bg-description"><span>Left</span><textarea data-setting="left" rows="3" maxlength="40000" placeholder="Describe elements and spatial relationships on the left side."></textarea></label>
                    <label class="bg-field bg-description"><span>Right</span><textarea data-setting="right" rows="3" maxlength="40000" placeholder="Describe elements and spatial relationships on the right side."></textarea></label>
                    <label class="bg-field bg-negative"><span>Negative Prompt</span><textarea data-setting="negative" rows="2" maxlength="8000" aria-label="Negative prompt" placeholder="Optional: unwanted details or artifacts."></textarea></label>
                </section>
                <section class="bg-panel bg-generation"><div class="vnccs-common-creator-section-title">Generation</div>
                    <div class="vnccs-common-creator-tab-row" style="--tabs:2" data-model-types role="group" aria-label="Generation model type"><button class="vnccs-common-creator-tab" data-model-type="qi2">Qwen Image 2.1</button><button class="vnccs-common-creator-tab" data-model-type="moss">Moss</button></div>
                    <div class="vnccs-common-creator-subsection"><div class="vnccs-common-creator-model-card-list" data-model-picker></div>
                        ${toggleCard("Prompt Enhancement", "generation.prompt_enhance")}
                    </div>
                    <label class="bg-field"><span>Aspect ratio</span><select data-setting="aspect">${options(ASPECTS)}</select></label>
                    <label class="vnccs-common-creator-field vnccs-common-creator-resolution-field"><span class="vnccs-common-creator-resolution-head"><span class="vnccs-common-creator-label">Resolution scale</span><output class="vnccs-common-creator-resolution-value" data-resolution>2.0 MP</output></span><input class="vnccs-common-creator-resolution-slider" type="range" min="${RESOLUTION_SCALE_MIN_MP}" max="${RESOLUTION_SCALE_MAX_MP}" step="${RESOLUTION_SCALE_STEP_MP}" data-setting="generation.resolution_mp" aria-label="Resolution scale in megapixels"></label>
                    <div class="vnccs-common-creator-gen-param-grid">
                        ${compactField("Steps", '<input class="vnccs-common-creator-gen-param-input" type="number" min="1" max="150" data-setting="generation.steps">')}
                        ${compactField("Sampler", '<select class="vnccs-common-creator-gen-param-select" data-setting="generation.sampler"><option>euler</option></select>')}
                        ${compactField("CFG", '<input class="vnccs-common-creator-gen-param-input" type="number" min="0" max="30" step="0.1" data-setting="generation.cfg">')}
                        ${compactField("Scheduler", '<select class="vnccs-common-creator-gen-param-select" data-setting="generation.scheduler"><option>simple</option></select>')}
                    </div>
                    <div class="vnccs-common-creator-field"><span class="vnccs-common-creator-label">Seed</span><div class="vnccs-common-creator-seed-row"><input aria-label="Generation seed" class="vnccs-common-creator-gen-param-input" type="number" min="0" max="9007199254740991" data-setting="generation.seed"><button type="button" class="vnccs-common-creator-seed-dice-btn" data-random="generation.seed" aria-label="Use random seed" aria-pressed="false"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3.5" stroke="currentColor" stroke-width="2"/><circle cx="8.5" cy="8.5" r="1.4" fill="currentColor"/><circle cx="15.5" cy="8.5" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="8.5" cy="15.5" r="1.4" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.4" fill="currentColor"/></svg></button></div></div>
                    <div class="vnccs-common-creator-qi2-cache" style="display:grid" data-qwen-only><div class="vnccs-common-creator-qi2-cache-title">Qwen Image 2.1 Cache</div>
                        ${compactField("Device", `<select class="vnccs-common-creator-gen-param-select" data-setting="generation.cache_device">${options(["gpu", "auto", "cpu", "off"])}</select>`)}
                        ${compactField("Dtype", `<select class="vnccs-common-creator-gen-param-select" data-setting="generation.cache_dtype">${options(["int8", "default", "int4"])}</select>`)}
                    </div>
                    <div class="vnccs-common-creator-subsection"><div class="vnccs-common-creator-section-title">LoRA Stack</div><div class="vnccs-common-creator-subsection" data-turbo-lora><span class="vnccs-common-creator-label">Turbo LoRA</span>${toggleCard("Viggle Turbo · 6 steps", "generation.turbo")}</div>
                        <div class="vnccs-common-creator-lora-stack">${[0, 1, 2, 3, 4].map(index => `<div class="vnccs-common-creator-lora-item is-empty"><div class="vnccs-common-creator-lora-row"><select class="vnccs-common-creator-select" style="flex:2" data-lora="${index}" aria-label="LoRA ${index + 1}"><option value="">None</option></select><input class="vnccs-common-creator-input" style="flex:1" type="number" min="-10" max="10" step="0.05" value="1" data-lora-strength="${index}" aria-label="LoRA ${index + 1} strength"></div></div>`).join("")}</div>
                    </div>
                </section>
            </section>
            <span class="bg-announcement" role="status" aria-live="polite" aria-atomic="true"></span>
            <input type="file" accept="image/png,image/jpeg,image/webp" data-file hidden>`;
        this.status = this.container.querySelector("[role=status]");
        this.container.classList.add("vnccs-common-ui-host", "vnccs-common-creator-container");
        this.customSelects = createSakuraInteractions(this.container, node);
        this.ccState = { config: null, downloadStatus: {} };
        this.cc = createControlCenterClient("MIUProject/VNCCS_v3.0", config => config.models?.some(entry => String(entry.kind).toLowerCase() === "qi2"), this.ccState,
            () => { if (!this.disposed && this.state) { this.renderModelPicker(); this.syncGenerationCards(); this.updateControls(); } }, node);
        this.decorateControls();
        this.container._node = node;
        enableMiddleMouseCanvasPan(this.container, node);
        this.tooltipCleanup = attachHelpTooltips(this.container);
        for (const region of this.container.querySelectorAll(".bg-step,.bg-panel")) region.dataset.sakuraScroll = region.className;
        this.on(this.container, "click", event => this.click(event));
        this.on(this.container.querySelector("[data-variation-cards]"), "scroll", () => this.updateVariantScroll(), { passive: true });
        for (const image of this.container.querySelectorAll("[data-preview]")) this.on(image, "error", () => {
            image.hidden = true; delete image.dataset.sourceURL;
            const empty = this.container.querySelector("[data-empty]"); empty.hidden = false;
            empty.querySelector(".vnccs-common-creator-placeholder-title").textContent = "Preview unavailable";
            empty.querySelector(".vnccs-common-creator-placeholder-hint").textContent = "Select the background again to retry. Saved data is retained.";
        });
        this.on(this.container, "input", event => this.input(event));
        this.on(this.container, "change", event => this.change(event));
        this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(this.container);
        this.on(window, "beforeunload", event => { if (this.unsaved) { event.preventDefault(); event.returnValue = ""; } });
        this.on(this.api, "executing", event => {
            const id = event.detail?.node ?? event.detail;
            if (String(id) !== String(this.node.id) || this.restoring || this.busy) return;
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
        this.on(this.api, "status", event => {
            const remaining = event.detail?.exec_info?.queue_remaining;
            if (typeof remaining === "number" && Number.isFinite(remaining)) this.setWorkflowBusy(remaining > 0);
            else if (event.detail == null) this.setWorkflowBusy(true);
        });
        for (const type of ["execution_start", "reconnecting"]) this.on(this.api, type, () => this.setWorkflowBusy(true));
        watchConnection(node, async () => {
            if (this.disposed) return;
            const restore = this.restoreFailed ? this.restore() : null;
            await this.refreshWorkflowBusy();
            await restore;
        }, registerCleanup);
        this.updateControls();
        this.refreshWorkflowBusy();
    }

    on(target, type, callback, options = {}) { target.addEventListener(type, callback, { ...options, signal: this.events.signal }); }
    message(text, error = false, title = "Background Creator Error") {
        if (this.disposed) return;
        this.status.textContent = text; this.status.classList.toggle("error", error);
        if (error && this.lastError !== text) {
            this.errorDialog?.overlay.remove();
            const content = document.createElement("div");
            content.className = "bg-error-text"; content.textContent = text;
            this.errorDialog = showModal(this.container, title, () => content, [{ text: "OK", class: "primary" }]);
            this.errorDialog.modal.classList.add("bg-error-dialog");
            this.errorDialog.overlay.setAttribute("data-background-error", "");
        }
        this.lastError = error ? text : "";
    }
    startPreviewGeneration() {
        this.stopPreviewGeneration();
        const layer = this.container.querySelector("[data-generation-layer]"), frame = layer.parentElement;
        const base = layer.querySelector(".vnccs-common-creator-generation-base"), next = layer.querySelector(".vnccs-common-creator-generation-new"), scan = layer.querySelector(".vnccs-common-creator-generation-scan");
        const image = this.container.querySelector("[data-preview]"), existing = image.hidden ? "" : image.src;
        const mannequin = this.state?.editor?.attributes?.scene_type === "Exterior" ? BACKGROUND_MANNEQUIN_EXTERIOR_URL : BACKGROUND_MANNEQUIN_INTERIOR_URL;
        this.previewGeneration = { epoch: this.epoch, layer, frame, base, next, scan, animations: [] };
        base.src = existing || mannequin; base.hidden = false; base.style.clipPath = "inset(0% 0 0 0)";
        next.removeAttribute("src"); next.style.clipPath = "inset(0 0 100% 0)";
        scan.style.transform = "translateY(-100%)";
        layer.classList.toggle("uses-mannequin", !existing); layer.classList.remove("is-revealing"); layer.hidden = false;
        frame.setAttribute("aria-busy", "true"); this.container.querySelector("[data-empty]").hidden = true;
    }
    stopPreviewGeneration() {
        const owner = this.previewGeneration;
        if (!owner) return;
        this.previewGeneration = null;
        for (const animation of owner.animations) animation.cancel();
        owner.layer.classList.remove("is-revealing", "uses-mannequin"); owner.layer.hidden = true;
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
    setWorkflowBusy(busy) {
        this.beginWorkflowStatusRequest(); this.workflowBusy = busy; this.updateControls();
    }
    async refreshWorkflowBusy() {
        const current = this.beginWorkflowStatusRequest();
        try {
            const response = await this.api.fetchApi("/queue", { signal: this.events.signal });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const queue = await response.json();
            if (!Array.isArray(queue.queue_running) || !Array.isArray(queue.queue_pending)) throw new Error("Invalid queue response");
            if (!current() || this.disposed) return;
            this.workflowBusy = Boolean(queue.queue_running.length || queue.queue_pending.length);
        } catch (error) {
            if (!current() || this.disposed) return;
            this.workflowBusy = true;
            console.warn("[VNCCS Background Creator] Could not refresh workflow status", error);
        }
        this.updateControls();
    }
    updateControls() {
        const locked = this.busy || this.restoring || Boolean(this.document?._recovery || this.document?.archived);
        const hasSource = Boolean(this.document?.images.some(image => image.image_id === this.state?.selected_image_id));
        this.container.setAttribute("aria-busy", String(this.busy || this.restoring));
        for (const control of this.container.querySelectorAll("button,input,select,textarea")) {
            if (!control.closest("[data-background-error],[data-background-wizard]")) control.disabled = locked;
        }
        for (const button of this.container.querySelectorAll("[data-view-variation]")) {
            const card = this.state?.editor.variants.find(card => card.id === button.dataset.viewVariation);
            button.disabled = locked || !card || !this.variantPreview(card);
        }
        for (const card of this.container.querySelectorAll("[data-model-toggle],[data-model-choice]")) {
            card.disabled = locked || (card.hasAttribute("data-model-choice") && !card.classList.contains("is-installed"));
            if (card.getAttribute("role") === "button") { card.setAttribute("aria-disabled", String(card.disabled)); card.tabIndex = card.disabled ? -1 : 0; }
        }
        for (const button of this.container.querySelectorAll("[data-download-model]")) button.disabled = locked || ["queued", "downloading"].includes(this.ccState.downloadStatus[`cc_models_${button.dataset.downloadModel}`]?.status);
        for (const control of this.container.querySelectorAll("[data-library],[data-new]")) {
            const canRecover = control.hasAttribute("data-library") && this.restoreFailed && Array.isArray(this.catalog);
            control.disabled = this.busy || this.workflowBusy || this.restoring && !canRecover;
        }
        this.container.querySelector("[data-delete]").disabled = locked || this.workflowBusy || !this.document;
        this.container.querySelector("[data-enlarge]").disabled = this.busy || this.restoring || !hasSource;
        for (const control of this.container.querySelectorAll("[data-close-preview]")) control.disabled = false;
        const turbo = this.container.querySelector('[data-setting="generation.turbo"]');
        turbo.disabled = locked || this.state?.editor.generation.model_type === "moss" || isTurboModel(this.state?.editor.generation.diffusion_model_name)
            || this.turboStatus() !== "installed";
        const download = this.container.querySelector("[data-download-turbo]");
        if (download) download.disabled = locked || ["queued", "downloading"].includes(this.turboStatus());
        const generate = this.container.querySelector('[data-action="background"]');
        for (const control of [generate, this.container.querySelector("[data-wizard]")]) control.disabled = locked || this.workflowBusy;
        generate.textContent = this.workflowRun ? "Workflow busy..." : this.busy && this.activeAction === "background" ? "Generating..." : this.workflowBusy ? "Workflow busy..." : "Generate background";
        generate.title = this.workflowBusy ? "Generation is unavailable while a workflow is queued or running." : "";
        this.customSelects.syncSelects();
        this.updateVariantScroll();
    }

    invalidateRestore() { ++this.epoch; this.wizardDialog?.overlay.remove(); this.stopPreviewGeneration(); this.restoring = true; this.busy = false; this.activeAction = ""; this.workflowRun = null; this.unsaved = false; this.jobId = ""; clearTimeout(this.saveTimer); delete this.container.querySelector("[data-variation-cards]").dataset.cards; this.updateControls(); }
    restore() {
        const epoch = ++this.epoch; this.restoring = true; this.restoreFailed = false;
        this.operations = this.operations.catch(() => {}).then(async () => {
            if (this.disposed || epoch !== this.epoch) return;
            try {
                const widget = this.node.widgets.find(item => item.name === "background_data");
                const state = readState(widget.value), stored = JSON.parse(widget.value);
                this.state = state;
                if (this.document?.background_id !== state.background_id) { this.document = null; }
                this.render();
                const [config, catalog] = await Promise.all([this.request("/config"), this.request("/backgrounds")]);
                if (this.disposed || epoch !== this.epoch) return;
                this.state = state; this.config = config; this.catalog = catalog.backgrounds;
                this.fillChoices();
                this.render();
                this.cc.fetchConfig().catch(error => { if (!this.disposed) this.message("Model catalog unavailable: " + error.message + ". Local model choices are retained.", true); });
                if (state.background_id) {
                    const document = await this.request(`/backgrounds/${state.background_id}`);
                    if (epoch !== this.epoch || this.disposed) return;
                    if (!stored.editor) state.editor = editorDefaults(document.editor);
                    state.revision = document.revision;
                    await this.acceptDocument(document, false, epoch);
                }
                if (epoch !== this.epoch || this.disposed) return;
                this.restoring = false; this.render();
                const libraryErrors = [...(catalog.errors || []).map(item => `${item.name}: ${item.error}`),
                    config.styles_error ? "Style library unavailable: " + config.styles_error + ". Saved style text is retained." : ""].filter(Boolean).join("\n");
                this.message(libraryErrors || (this.document?._recovery ? "Recovered backup is read-only. The damaged original is preserved." : "Create or upload a background. Generation runs inside this workspace."), Boolean(libraryErrors || this.document?._recovery));
                this.sync();
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

    turboEntry() {
        return this.ccState.config?.lora?.find(entry => String(entry.kind).toLowerCase() === "qi2"
            && String(entry.type).toLowerCase() === "turbolora" && entry.name === "Qwen Image 2.1 Viggle Turbo") || this.config?.turbo_lora;
    }
    turboStatus() {
        const entry = this.turboEntry(), transient = this.ccState.downloadStatus[`cc_lora_${entry?.name}`]?.status;
        return ["queued", "downloading", "error"].includes(transient) ? transient : entry?.status || "missing";
    }
    renderTurboCard() {
        const control = this.container.querySelector('[data-setting="generation.turbo"]'), card = control.closest("[data-setting-card]");
        const entry = this.turboEntry(), status = this.turboStatus(), installed = status === "installed";
        card.classList.toggle("is-installed", installed); card.classList.toggle("is-missing", !installed);
        card.classList.toggle("is-selected", installed && control.checked);
        control.closest(".vnccs-common-creator-toggle").hidden = !installed;
        card.querySelector(".vnccs-common-creator-model-card-badge").className = "vnccs-common-creator-model-card-badge " + (installed ? "ok" : "missing");
        card.querySelector("[data-toggle-status]").textContent = installed ? "Installed" : status === "queued" ? "Queued" : status === "downloading" ? "Downloading" : status === "error" ? "Error" : "Missing";
        let download = card.querySelector("[data-download-turbo]");
        if (entry && !installed && !download) {
            download = document.createElement("button"); download.type = "button"; download.dataset.downloadTurbo = "";
            download.className = "vnccs-common-creator-model-card-download"; download.textContent = "Download"; card.appendChild(download);
        }
        if (download) { download.hidden = installed || !entry; download.disabled = ["queued", "downloading"].includes(status); }
    }
    renderStyles() {
        const editor = this.state.editor, choices = this.config?.styles || [];
        const key = JSON.stringify(choices);
        if (!this.stylePicker) {
            this.styleInfo = {};
            this.stylePicker = createStylePicker({ host: this.container, catalog: { default_style: "none", groups: [] },
                getInfo: () => this.styleInfo,
                save: () => {
                    if (this.restoring || this.busy || this.disposed) return;
                    const info = this.styleInfo, current = this.state.editor;
                    if (!current.style_id) current.custom_style = current.style;
                    if (info.style === "custom") {
                        delete current.style_id;
                        current.custom_style = info.custom_style ?? current.custom_style ?? ""; current.style = current.custom_style;
                    }
                    else {
                        current.style_id = info.style;
                        const choice = this.config?.styles?.find(style => style.id === info.style);
                        current.style = info.style_prompt || choice?.prompt || "";
                    }
                    this.markEdited(); this.render();
                },
                catalogURL: () => BASE + "/styles", showGenerateAll: true, allowNone: () => true,
                getPreviewPayload: () => {
                    if (this.restoring || this.busy || this.disposed) throw new Error("Background Creator is busy");
                    return { node_id: String(this.node.id), editor: clone(this.state.editor) };
                },
                listenPreview: callback => this.on(this.api, "vnccs.style_preview.stage", callback),
                fetchApi: (...args) => this.api.fetchApi(...args), imageURL: value => this.api.apiURL(value),
                cleanup: callback => { this.stylePickerCleanup = callback; }, showModal,
            });
            this.stylePicker.root.classList.add("vnccs-common-creator-field");
            this.stylePicker.customInput.classList.add("vnccs-common-creator-input");
            this.stylePicker.root.children[0].classList.add("vnccs-common-creator-label");
            const field = this.container.querySelector("[data-style-picker]");
            field.appendChild(this.stylePicker.root);
            field._vnccsWizardRevealTarget = this.stylePicker.root.querySelector(".vnccs-style-summary");
        }
        if (key !== this.styleCatalogKey) {
            this.styleCatalogKey = key;
            this.stylePicker.setCatalog({ default_style: "none", groups: [{ label: "Background styles", styles: choices.map(style => ({ ...style, description: style.description || style.prompt, user: style.id.startsWith("user_") })) }] });
        }
        const selectedStyle = editor.style_id || (editor.style ? "custom" : "none");
        Object.assign(this.styleInfo, { style: selectedStyle, custom_style: editor.custom_style ?? (editor.style_id ? "" : editor.style),
            style_label: selectedStyle === "none" ? "None" : editor.style_id || "Custom style", style_description: editor.style, style_prompt: editor.style });
        if (!this.stylePicker.customInput.contains?.(document.activeElement)) this.stylePicker.setValue(this.styleInfo.style);
    }
    randomizeSeedIfNeeded() {
        if (this.state?.editor.generation.seed_mode !== "randomize") return;
        this.state.editor.generation.seed = generateRandomSeed(); this.markEdited(); this.render();
    }
    async downloadAsset(category, asset) {
        const key = `cc_${category}_${asset.name}`;
        this.ccState.downloadStatus[key] = { status: "queued" }; this.renderModelPicker(); this.syncGenerationCards(); this.updateControls();
        try {
            await this.request("/vnccs/control_center/download", { method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                body: JSON.stringify({ repo_id: "MIUProject/VNCCS_v3.0", category, name: asset.name }) });
        } catch (error) { this.ccState.downloadStatus[key] = { status: "error" }; this.renderModelPicker(); this.syncGenerationCards(); throw error; }
        this.cc.startPolling();
    }

    async downloadModel(name) {
        const entry = this.modelEntries.find(item => item.name === name);
        if (!entry || entry.source === "local") return;
        const bundle = [["models", entry], ...["clip", "vae"].map(category => [category, this.ccState.config?.[category]?.find(item => String(item.kind).toLowerCase() === this.state.editor.generation.model_type)])];
        for (const [category, asset] of bundle) {
            if (!asset || asset.status === "installed" || this.disposed) continue;
            await this.downloadAsset(category, asset);
        }
    }

    async acceptDocument(document, adopt = true, epoch = this.epoch) {
        if (this.disposed || epoch !== this.epoch) return;
        this.document = document;
        this.catalog = (this.catalog || []).filter(item => item.background_id !== document.background_id);
        if (!document.archived) this.catalog.push(document);
        if (adopt) {
            this.state.background_id = document.background_id; this.state.selected_image_id = document.selected_image_id;
            this.state.editor = editorDefaults(document.editor);
            this.state.revision = document.revision;
            delete this.container.querySelector("[data-variation-cards]").dataset.cards;
        }
        if (this.state.background_id === document.background_id) this.state.background_name = document.name;
        if (epoch !== this.epoch || this.disposed) return;
        this.render(); this.sync(); this.updateControls();
    }

    updateVariantScroll() {
        const list = this.container.querySelector("[data-variation-cards]"), strip = list.closest(".bg-variation-strip");
        const preview = list.querySelector(".bg-variation-preview");
        if (Number.isFinite(preview?.offsetTop) && Number.isFinite(preview?.offsetHeight))
            strip.style.setProperty("--variation-nav-top", `${preview.offsetTop + preview.offsetHeight / 2}px`);
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
        const unchanged = ["editor", "selected_image_id"].every(key => JSON.stringify(document[key]) === JSON.stringify(this.document[key]));
        this.document.generated_sets = document.generated_sets;
        // Only rebase pending edits when generation left the saved authored settings unchanged.
        if (unchanged) {
            if (this.state.revision === this.document.revision) this.state.revision = document.revision;
            this.document.revision = document.revision;
            this.sync();
        }
        this.render();
        if (unchanged && this.unsaved) this.markEdited();
    }

    variantPreview(card) {
        for (const group of [...(this.document?.generated_sets || [])].reverse()) {
            if (group.version !== "" || group.source_image_id !== this.state.selected_image_id) continue;
            const index = group.variants.findIndex(saved => ["id", "time", "weather", "additional_details"].every(key => saved[key] === card[key]));
            if (index >= 0 && group.images[index + 1]?.url) return group.images[index + 1];
        }
        return null;
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
            if (!this.disposed && !this.restoring && !this.busy && epoch === this.epoch && this.state.editor.variants.includes(card)
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
        const scroll = [...this.container.querySelectorAll(".bg-step,.bg-panel")].map(element => [element, element.scrollTop, element.scrollLeft]);
        const view = this.customSelects.captureView();
        const editor = this.state.editor;
        const cardList = this.container.querySelector("[data-variation-cards]");
        const cardKey = JSON.stringify(editor.variants.map(card => card.id));
        if (cardList.dataset.cards !== cardKey) {
            cardList.replaceChildren(); cardList.dataset.cards = cardKey;
            for (const card of editor.variants) {
                const item = document.createElement("div"); item.className = "bg-variation-item"; item.dataset.variation = card.id;
                const name = document.createElement("span"); name.dataset.variationTitle = ""; name.textContent = card.name; name.title = card.name;
                const preview = document.createElement("div"); preview.className = "bg-variation-preview";
                const button = document.createElement("button"); button.type = "button"; button.dataset.viewVariation = card.id;
                const image = document.createElement("img"); image.dataset.variationImage = card.id; image.hidden = true; image.decoding = "async";
                this.on(image, "error", () => { image.hidden = true; delete image.dataset.sourceURL; }); button.appendChild(image);
                const remove = document.createElement("button"); remove.type = "button"; remove.className = "bg-variation-delete";
                remove.dataset.deleteVariation = card.id; remove.textContent = "×";
                remove.setAttribute("aria-label", `Delete ${card.name} variant`);
                const settings = document.createElement("div"); settings.className = "bg-variation-settings";
                const nameField = document.createElement("label"); nameField.className = "bg-variation-name";
                const nameLabel = document.createElement("span"); nameLabel.className = "vnccs-common-creator-label"; nameLabel.textContent = "Variant name";
                const nameInput = document.createElement("input"); nameInput.className = "vnccs-common-creator-input";
                nameInput.dataset.variationName = card.id; nameInput.setAttribute("aria-label", `${card.name} variant name`); nameInput.maxLength = 96; nameInput.value = card.name;
                nameField.append(nameLabel, nameInput); settings.appendChild(nameField);
                for (const [key, label] of VARIANT_FIELDS) {
                    const control = createTraitInput({ prefix: "vnccs-common-creator", label, key, target: card, removable: true,
                        save: () => {
                            if (!this.state.editor.variants.includes(card) || this.restoring || this.disposed || this.busy) return;
                            this.markEdited(); this.render();
                        },
                        choose: input => this.openVariantPresets(key, input, card),
                    });
                    control.input.dataset.variationField = key; control.input.dataset.variationId = card.id;
                    control.input.setAttribute("aria-label", `${card.name} ${label.toLowerCase()}`);
                    control.input.maxLength = key === "time" ? 200 : key === "weather" ? 500 : 8000;
                    settings.appendChild(control.element);
                }
                preview.append(button, remove); item.append(name, preview, settings); cardList.appendChild(item);
            }
        }
        const mannequin = editor.attributes.scene_type === "Exterior" ? BACKGROUND_MANNEQUIN_EXTERIOR_URL : BACKGROUND_MANNEQUIN_INTERIOR_URL;
        for (const button of cardList.querySelectorAll("[data-view-variation]")) {
            const card = editor.variants.find(card => card.id === button.dataset.viewVariation), item = button.closest(".bg-variation-item");
            const image = button.querySelector("[data-variation-image]"), preview = this.variantPreview(card), source = preview ? this.api.apiURL(preview.url) : mannequin;
            const title = item.querySelector("[data-variation-title]"), name = item.querySelector("[data-variation-name]");
            title.textContent = title.title = card.name;
            if (document.activeElement !== name) name.value = card.name;
            name.setAttribute("aria-label", `${card.name} variant name`);
            item.querySelector("[data-delete-variation]").setAttribute("aria-label", `Delete ${card.name} variant`);
            button.disabled = !preview;
            button.setAttribute("aria-label", preview ? `View ${card.name} variant` : `${card.name} preview will appear after generation`);
            if (preview) button.setAttribute("aria-haspopup", "dialog"); else button.removeAttribute("aria-haspopup");
            image.alt = preview ? `${card.name} variant` : ""; image.hidden = false; image.classList.toggle("is-placeholder", !preview);
            if (image.dataset.sourceURL !== source) { image.src = source; image.dataset.sourceURL = source; }
        }
        const moss = editor.generation.model_type === "moss";
        this.container.querySelector("[data-model-types]").style.setProperty("--selected", moss ? 1 : 0);
        for (const button of this.container.querySelectorAll("[data-model-type]")) {
            const selected = button.dataset.modelType === editor.generation.model_type;
            button.classList.toggle("is-active", selected); button.setAttribute("aria-pressed", String(selected));
        }
        for (const section of this.container.querySelectorAll("[data-qwen-only]")) section.hidden = moss;
        this.container.querySelector("[data-turbo-lora]").hidden = moss || isTurboModel(editor.generation.diffusion_model_name);
        for (const control of this.container.querySelectorAll("[data-setting]")) {
            const value = control.dataset.setting.split(".").reduce((current, key) => current?.[key], editor);
            if (control.type === "checkbox") control.checked = Boolean(value);
            else if (control.tagName === "SELECT") selectValue(control, value);
            else if (document.activeElement !== control) control.value = value ?? "";
        }
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
        for (const image of this.container.querySelectorAll("[data-preview],[data-large-preview]")) {
            if (image.dataset.previewVariation) continue;
            image.hidden = !previewURL;
            if (previewURL && image.dataset.sourceURL !== previewURL) { image.src = this.api.apiURL(previewURL); image.dataset.sourceURL = previewURL; }
        }
        this.container.querySelector("[data-empty]").hidden = Boolean(previewURL);
        this.container.querySelector("[data-enlarge]").hidden = !source;
        if (!source) {
            this.container.querySelector(".vnccs-common-creator-placeholder-title").textContent = "Create a background";
            this.container.querySelector(".vnccs-common-creator-placeholder-hint").textContent = "Generate a preview or upload your image.";
        }
        for (const select of this.container.querySelectorAll("[data-lora]")) selectValue(select, editor.generation.lora_stack[Number(select.dataset.lora)]?.name || "");
        for (const number of this.container.querySelectorAll("[data-lora-strength]")) {
            if (document.activeElement !== number) number.value = editor.generation.lora_stack[Number(number.dataset.loraStrength)]?.strength ?? 1;
        }
        this.renderStyles();
        const random = editor.generation.seed_mode === "randomize", dice = this.container.querySelector("[data-random]");
        dice.classList.toggle("is-active", random); dice.setAttribute("aria-pressed", String(random));
        dice.setAttribute("aria-label", random ? "Use fixed seed" : "Use random seed");
        this.container.querySelector('[data-setting="generation.seed"]').classList.toggle("is-random", random);
        this.syncResolution();
        this.syncPrompt(); this.renderModelPicker(); this.syncGenerationCards();
        this.resize(); this.decorateControls(); this.updateControls();
        for (const [element, top, left] of scroll) { element.scrollTop = top; element.scrollLeft = left; }
        this.customSelects.restoreView(view);
    }

    syncResolution() {
        const resolution = Number(this.state.editor.generation.resolution_mp);
        this.container.querySelector("[data-resolution]").textContent = `${resolution.toFixed(1)} MP`;
        const saved = resolution < RESOLUTION_SCALE_MIN_MP || resolution > RESOLUTION_SCALE_MAX_MP
            ? `Saved value: ${resolution} MP. It is retained until you move the slider. ` : "";
        setHelpText(this.container.querySelector('[data-setting="generation.resolution_mp"]'),
            `${saved}Resolution scale: ${RESOLUTION_SCALE_MIN_MP}–${RESOLUTION_SCALE_MAX_MP} MP in ${RESOLUTION_SCALE_STEP_MP} MP steps.`);
    }

    sync() {
        if (this.disposed || this.restoring || !this.state) return;
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
        for (const control of this.container.querySelectorAll('[data-setting="style"],[data-setting="scene_description"],[data-setting="front"],[data-setting="back"],[data-setting="left"],[data-setting="right"]')) {
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
            const id = this.document.background_id, epoch = this.epoch, editor = clone(this.state.editor), selectedImage = this.state.selected_image_id;
            const saved = await this.json(`/backgrounds/${id}`, { editor, selected_image_id: selectedImage, revision: this.state.revision }, "PATCH");
            if (this.disposed || epoch !== this.epoch || this.document?.background_id !== id) return;
            this.document = saved;
            this.state.revision = saved.revision; this.sync();
            this.unsaved = JSON.stringify(editor) !== JSON.stringify(this.state.editor) || selectedImage !== this.state.selected_image_id;
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
                    && this.state.selected_image_id === preview.request_image_id;
                this.stopPreviewGeneration();
                if (current && this.state.selected_image_id !== preview.image_id) {
                    this.state.selected_image_id = preview.image_id;
                }
                if (current) this.state.revision = preview.revision ?? document.revision;
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
            if (button.classList.contains("vnccs-style-summary") || button.closest(".vnccs-style-gallery")) continue;
            button.type = "button";
            if (!button.closest(".bg-segments") && !button.className.includes("vnccs-common-")) {
                const action = ["primary", "success", "danger"].some(variant => button.classList.contains(variant));
                button.classList.add("vnccs-common-creator-" + (action ? "btn" : button.hasAttribute("data-upload") ? "btn-upload" : "model-card-download"));
            }
            for (const variant of ["primary", "success", "danger"]) if (button.classList.contains(variant)) button.classList.add("vnccs-common-creator-btn-" + variant);
        }
        for (const control of this.container.querySelectorAll("input,select,textarea")) {
            if (control.closest(".vnccs-style-gallery")) continue;
            if (control.classList.contains("vnccs-common-creator-gen-param-input") || control.classList.contains("vnccs-common-creator-gen-param-select") || control.classList.contains("vnccs-common-creator-resolution-slider") || control.closest(".vnccs-common-creator-textarea-wrapper")) continue;
            if (control.tagName === "INPUT" && ["checkbox", "file"].includes(control.type)) continue;
            control.classList.add("vnccs-common-creator-" + (control.type === "range" ? "slider" : control.tagName === "SELECT" ? "select" : control.tagName === "TEXTAREA" ? "textarea" : "input"));
        }
        add(".bg-panel", "col"); add(".bg-field", "field");
        add("[data-wizard]", "character-wizard-btn"); add("[data-new],[data-delete]", "btn-secondary");
        for (const field of this.container.querySelectorAll(".bg-field")) for (const label of field.children) if (label.tagName === "SPAN") label.classList.add("vnccs-common-creator-label");
        for (const group of this.container.querySelectorAll(".bg-segments")) {
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
        this.renderTurboCard();
    }

    async dialog(title = "New Background") {
        return await new Promise(resolve => {
            let input;
            const finish = value => { if (!this.dialogs.delete(close)) return; modal.overlay.remove(); resolve(value); };
            const close = () => finish(null);
            this.dialogs.add(close);
            const modal = showModal(this.container, title, () => {
                input = document.createElement("input"); input.className = "vnccs-common-creator-input";
                input.placeholder = "Name..."; input.maxLength = 96; input.dataset.backgroundName = "";
                return input;
            }, [{ text: "Cancel", action: close }, { text: "Create", class: "primary", action: () => {
                const name = input.value.trim();
                if (!name) { input.focus(); return true; }
                finish({ name });
            } }]);
            const remove = modal.overlay.remove;
            modal.overlay.remove = () => { remove.apply(modal.overlay); if (this.dialogs.delete(close)) resolve(null); };
            this.decorateControls(); this.customSelects.syncSelects();
        });
    }

    openDeleteDialog() {
        const background = this.document;
        if (!background) return;
        const epoch = this.epoch, id = background.background_id, name = background.name;
        const close = () => modal.overlay.remove();
        const modal = showModal(this.container, "Delete Background", () => {
            const content = document.createElement("div"); content.style.fontSize = "14px"; content.style.textAlign = "center";
            const question = document.createElement("span"); question.textContent = "Are you sure you want to ";
            const permanent = document.createElement("b"); permanent.textContent = "PERMANENTLY DELETE";
            const selected = document.createElement("span"); selected.style.color = "#fff"; selected.style.fontWeight = "bold"; selected.textContent = `'${name}'`;
            const suffix = document.createElement("span"); suffix.textContent = "?";
            const warning = document.createElement("span"); warning.style.fontSize = "12px"; warning.style.color = "#aaa"; warning.textContent = "This action cannot be undone.";
            content.append(question, permanent, document.createElement("br"), selected, suffix, document.createElement("br"), document.createElement("br"), warning);
            return content;
        }, [{ text: "Cancel", action: close }, { text: "CONFIRM DELETE", class: "danger", action: async (_overlay, button) => {
            const current = () => !this.disposed && epoch === this.epoch && this.document?.background_id === id;
            if (!current()) return false;
            button.textContent = "DELETING...";
            try {
                await this.json(`/backgrounds/${id}`, {}, "DELETE");
                if (!current()) return false;
                this.catalog = this.catalog.filter(item => item.background_id !== id); this.document = null; this.unsaved = false;
                const next = this.catalog[0];
                if (next) await this.acceptDocument(next, true, epoch);
                else {
                    const editor = { ...clone(DEFAULT_EDITOR), custom_style: "", aspect: this.state.editor.aspect, generation: clone(this.state.editor.generation) };
                    Object.assign(this.state, { background_id: "", background_name: "", selected_image_id: "", editor }); delete this.state.revision;
                    this.render(); this.sync(); this.updateControls();
                }
                return false;
            } catch (error) { button.textContent = "CONFIRM DELETE"; throw error; }
        } }]);
        this.dialogs.add(close);
        const remove = modal.overlay.remove;
        modal.overlay.remove = () => { this.dialogs.delete(close); remove.apply(modal.overlay); };
    }

    async ensureDocument() {
        if (this.document) return true;
        const epoch = this.epoch;
        const values = await this.dialog("New Background");
        if (!values || this.disposed || epoch !== this.epoch) return false;
        const document = await this.json("/backgrounds", { name: values.name, editor: this.state.editor });
        if (this.disposed || epoch !== this.epoch) return false;
        await this.acceptDocument(document, true, epoch); return epoch === this.epoch && !this.disposed;
    }

    openWizard() {
        if (this.wizardDialog || this.workflowBusy || this.busy || this.restoring || this.disposed) return;
        const epoch = this.epoch;
        let input, modal;
        const isCurrent = () => !this.disposed && epoch === this.epoch && this.wizardDialog === modal && this.container.contains(modal.overlay);
        modal = showModal(this.container, "Background Wizard", () => {
            const wrap = document.createElement("div"); wrap.className = "bg-wizard-modal";
            const text = document.createElement("div"); text.className = "bg-wizard-text";
            text.textContent = "Describe the background in a broad way. The model will expand it into the scene fields while preserving the selected style and scene type.";
            input = document.createElement("textarea"); input.className = "vnccs-common-creator-textarea bg-wizard-input";
            input.maxLength = 40000; input.placeholder = "e.g. a quiet interior, open landscape, city street, or fantasy environment";
            wrap.append(text, input); return wrap;
        }, [
            { text: "Cancel" },
            { text: "FILL FIELDS", class: "primary", action: async (_overlay, button) => {
                if (!isCurrent()) return false;
                const description = input.value.trim();
                if (!description) { input.focus(); return true; }
                button.textContent = "THINKING...";
                try {
                    return !await this.runJob("wizard", description, isCurrent);
                } catch (error) {
                    if (isCurrent()) {
                        this.message(error.message || String(error), true, "Background Wizard Error");
                        const dialog = this.errorDialog, remove = dialog?.overlay.remove;
                        if (remove) dialog.overlay.remove = (...args) => {
                            const result = remove.apply(dialog.overlay, args);
                            if (isCurrent()) input.focus({ preventScroll: true });
                            return result;
                        };
                    }
                    return true;
                } finally {
                    button.textContent = "FILL FIELDS";
                }
            } },
        ]);
        modal.modal.classList.add("bg-wizard-dialog");
        modal.overlay.dataset.backgroundWizard = "";
        this.wizardDialog = modal;
        const remove = modal.overlay.remove;
        modal.overlay.remove = (...args) => {
            if (this.wizardDialog === modal) this.wizardDialog = null;
            return remove.apply(modal.overlay, args);
        };
    }

    async click(event) {
        const button = event.target.closest("button,[data-model-toggle],[data-model-choice]"); if (!button || !this.container.contains(button) || button.disabled) return;
        const epoch = this.epoch;
        try {
            if (button.hasAttribute("data-scroll-variation")) {
                const list = this.container.querySelector("[data-variation-cards]");
                const behavior = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
                list.scrollBy({ left: Number(button.dataset.scrollVariation) * list.clientWidth * .8, behavior }); return;
            }
            if (button.hasAttribute("data-delete-variation")) {
                const cards = this.state.editor.variants, index = cards.findIndex(card => card.id === button.dataset.deleteVariation);
                if (index < 0) return;
                this.state.editor.variants = cards.filter(card => card.id !== button.dataset.deleteVariation);
                this.markEdited(); this.render();
                const remaining = this.container.querySelectorAll("[data-delete-variation]");
                (remaining[Math.min(index, remaining.length - 1)] || this.container.querySelector("[data-add-variation]")).focus?.();
                return;
            }
            if (button.hasAttribute("data-add-variation")) {
                const cards = this.state.editor.variants;
                if (cards.length >= 64) throw new Error("A background supports up to 64 variants.");
                const base = `variant_${generateRandomSeed()}`; let id = base;
                for (let index = 1; cards.some(card => card.id === id); index++) id = `${base}_${index}`;
                cards.push({ id, name: `Variant ${cards.length + 1}`, time: "", weather: "", additional_details: "" });
                this.markEdited(); this.render();
                // Shared view restoration finishes on the next frame.
                requestAnimationFrame(() => {
                    if (this.disposed || epoch !== this.epoch || !this.state.editor.variants.some(card => card.id === id)) return;
                    const behavior = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
                    const card = [...this.container.querySelectorAll("[data-variation]")].find(card => card.dataset.variation === id);
                    card?.scrollIntoView?.({ block: "nearest", inline: "nearest", behavior });
                    card?.querySelector("[data-variation-name]")?.focus?.({ preventScroll: true });
                });
                return;
            }
            if (button.hasAttribute("data-view-variation")) { this.openPreview(this.state.editor.variants.find(card => card.id === button.dataset.viewVariation)); return; }
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
                if (next !== settings.diffusion_model_name && (isTurboModel(next) || isTurboModel(settings.diffusion_model_name))) { settings.steps = isTurboModel(next) ? 8 : 25; settings.cfg = 1; settings.turbo = false; settings.turbo_previous_settings = null; }
                settings.diffusion_model_name = next; this.modelPickerOpen = false; this.markEdited(); this.render(); return;
            }
            if (button.hasAttribute("data-download-turbo")) { event.preventDefault?.(); const entry = this.turboEntry(); if (entry) await this.downloadAsset("lora", entry); return; }
            if (button.dataset.downloadModel) { await this.downloadModel(button.dataset.downloadModel); return; }
            if (button.hasAttribute("data-enlarge")) { this.openPreview(); return; }
            if (button.dataset.value) { this.state.editor.attributes[button.closest("[data-trait]").dataset.trait] = button.dataset.value; this.markEdited(); this.render(); return; }
            if (button.dataset.random) {
                const settings = this.state.editor.generation;
                settings.seed_mode = settings.seed_mode === "randomize" ? "fixed" : "randomize";
                if (settings.seed_mode === "randomize") settings.seed = generateRandomSeed();
                this.markEdited(); this.render(); return;
            }
            if (button.hasAttribute("data-new")) {
                const values = await this.dialog("New Background");
                if (!values || this.disposed || epoch !== this.epoch) return;
                await this.persist();
                if (epoch !== this.epoch || this.disposed) return;
                const editor = { ...clone(DEFAULT_EDITOR), custom_style: "", aspect: this.state.editor.aspect, generation: clone(this.state.editor.generation) };
                await this.acceptDocument(await this.json("/backgrounds", { name: values.name, editor }), true, epoch); return;
            }
            if (button.hasAttribute("data-delete")) {
                this.openDeleteDialog(); return;
            }
            if (button.hasAttribute("data-upload")) { this.container.querySelector("[data-file]").click(); return; }
            if (button.hasAttribute("data-wizard")) { this.openWizard(); return; }
            if (button.dataset.action) await this.runJob(button.dataset.action);
        } catch (error) { if (epoch === this.epoch && !this.disposed) {
            this.busy = false; this.updateControls();
            if (button.isConnected && !button.disabled) button.focus({ preventScroll: true });
            this.message(error.message, true, button.hasAttribute("data-wizard") ? "Background Wizard Error" : "Background Creator Error");
        } }
    }

    openPreview(card = null) {
        const source = card ? this.variantPreview(card) : this.document?.images.find(image => image.image_id === this.state.selected_image_id);
        if (!source?.url) return;
        let image, viewport;
        const close = () => modal.overlay.remove();
        const modal = showModal(this.container, card ? `${card.name} preview` : "Background preview", () => {
            viewport = document.createElement("div"); viewport.className = "bg-image-viewport"; viewport.dataset.previewViewport = ""; viewport.tabIndex = 0;
            viewport.setAttribute("aria-label", "Image preview. Scroll to zoom, drag to pan.");
            image = document.createElement("img"); image.dataset.largePreview = ""; image.alt = card ? `${card.name} variant` : "Full-size selected background";
            if (card) image.dataset.previewVariation = card.id;
            image.draggable = false; image.style.visibility = "hidden"; viewport.appendChild(image);
            return viewport;
        }, [{ text: "Close", action: close }]);
        modal.modal.classList.add("bg-image-dialog"); modal.buttons[0].dataset.closePreview = "";
        const dispose = this.attachPreviewNavigation(viewport, image);
        this.dialogs.add(close);
        const remove = modal.overlay.remove;
        modal.overlay.remove = () => { this.dialogs.delete(close); dispose(); remove.apply(modal.overlay); };
        image.src = this.api.apiURL(source.url); image.dataset.sourceURL = source.url;
        this.render();
    }

    attachPreviewNavigation(viewport, image) {
        const events = new AbortController();
        let scale = 1, fitScale = 1, x = 0, y = 0, width = 0, height = 0, ready = false, closed = false, drag = null;
        const apply = () => {
            image.style.width = `${image.naturalWidth}px`; image.style.height = `${image.naturalHeight}px`;
            image.style.transform = `matrix(${scale},0,0,${scale},${x},${y})`; image.style.visibility = "visible";
        };
        const fit = () => {
            if (closed || !image.complete || !image.naturalWidth || !viewport.clientWidth || !viewport.clientHeight) return;
            const ratio = ready ? scale / fitScale : 1;
            const centerX = ready ? (width / 2 - x) / scale : image.naturalWidth / 2;
            const centerY = ready ? (height / 2 - y) / scale : image.naturalHeight / 2;
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
            apply();
        };
        viewport.onpointerdown = event => {
            if (!ready || event.button !== 0 || event.isPrimary === false) return;
            event.preventDefault(); event.stopPropagation(); viewport.focus({ preventScroll: true });
            drag = { ...point(event), id: event.pointerId, buttons: 1 };
            viewport.setPointerCapture(event.pointerId); viewport.classList.add("is-dragging");
        };
        viewport.onpointermove = event => {
            if (!drag || event.pointerId !== drag.id) return;
            if (!(event.buttons & drag.buttons)) { finish(event); return; }
            event.preventDefault(); event.stopPropagation(); const next = point(event);
            x += next.x - drag.x; y += next.y - drag.y; Object.assign(drag, next); apply();
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

    input(event) {
        if (this.restoring || this.busy || !this.state) return;
        const control = event.target;
        if (control.hasAttribute("data-variation-name")) {
            const card = this.state.editor.variants.find(card => card.id === control.dataset.variationName);
            if (card) {
                card.name = control.value; this.markEdited();
                const item = control.closest(".bg-variation-item"), title = item.querySelector("[data-variation-title]");
                title.textContent = title.title = card.name;
                item.querySelector("[data-delete-variation]").setAttribute("aria-label", `Delete ${card.name} variant`);
                item.querySelector("[data-view-variation]").setAttribute("aria-label", this.variantPreview(card) ? `View ${card.name} variant` : `${card.name} preview will appear after generation`);
            }
            return;
        }
        if (["number", "range"].includes(control.type) && !validNumber(control)) return;
        if (control.dataset.setting) {
            const keys = control.dataset.setting.split("."), last = keys.pop();
            const target = keys.reduce((value, key) => value[key] ||= {}, this.state.editor);
            target[last] = control.type === "checkbox" ? control.checked : ["number", "range"].includes(control.type) ? Number(control.value) : control.value;
            if (last === "style") { this.state.editor.custom_style = control.value; delete this.state.editor.style_id; }
            if (last === "turbo") {
                const settings = this.state.editor.generation;
                if (control.checked) {
                    settings.turbo_previous_settings ??= { steps: settings.steps, cfg: settings.cfg };
                    settings.steps = 6; settings.cfg = 1;
                } else {
                    settings.steps = settings.turbo_previous_settings?.steps ?? 25;
                    settings.cfg = settings.turbo_previous_settings?.cfg ?? 1;
                    settings.turbo_previous_settings = null;
                }
                this.render();
            }
            if (last === "resolution_mp") this.syncResolution();
            if (control.type === "checkbox") this.syncGenerationCards();
            this.markEdited(); return;
        }
        if (control.hasAttribute("data-lora") || control.hasAttribute("data-lora-strength")) {
            const previous = this.state.editor.generation.lora_stack;
            this.state.editor.generation.lora_stack = [...[...this.container.querySelectorAll("[data-lora]")].map(select => {
                const weight = this.container.querySelector(`[data-lora-strength="${select.dataset.lora}"]`), saved = previous[Number(select.dataset.lora)];
                return { ...saved, name: select.value, strength: validNumber(weight) ? Number(weight.value) : saved?.strength ?? 1 };
            }), ...previous.slice(5)];
            this.syncGenerationCards();
            this.markEdited();
        }
    }

    async change(event) {
        const control = event.target;
        const canRecover = control.hasAttribute("data-library") && this.restoreFailed && Array.isArray(this.catalog);
        if (this.busy || this.restoring && !canRecover) return;
        if (control.type === "number" && !validNumber(control)) {
            const value = control.dataset.setting ? control.dataset.setting.split(".").reduce((current, key) => current?.[key], this.state.editor)
                : control.hasAttribute("data-lora-strength") ? this.state.editor.generation.lora_stack[Number(control.dataset.loraStrength)]?.strength ?? 1
                : undefined;
            if (value !== undefined) { control.value = value; this.message("Enter a number within the field limits.", true); }
            return;
        }
        if (control.tagName === "SELECT" && (control.dataset.setting || control.hasAttribute("data-lora"))) { this.input(event); return; }
        let epoch = this.epoch;
        try {
            if (control.hasAttribute("data-library")) {
                const id = control.value;
                try { await this.persist(); }
                catch (error) { selectValue(control, this.state.background_id); this.render(); throw error; }
                if (epoch !== this.epoch || this.disposed) return; epoch = ++this.epoch;
                if (!id) {
                    this.document = null; this.unsaved = false;
                    this.state.background_id = ""; this.state.background_name = ""; this.state.selected_image_id = "";
                    delete this.state.revision;
                    this.render(); this.sync(); return;
                }
                this.restoring = true; this.updateControls();
                try {
                    const document = await this.request(`/backgrounds/${id}`);
                    if (epoch === this.epoch && !this.disposed) { await this.acceptDocument(document, true, epoch); this.restoreFailed = false; }
                } finally {
                    if (epoch === this.epoch && !this.disposed) { this.restoring = this.restoreFailed; this.render(); this.sync(); }
                }
            } else if (control.hasAttribute("data-file")) {
                const file = control.files?.[0]; if (!file) return;
                if (!await this.ensureDocument()) return;
                await this.persist(); if (epoch !== this.epoch || this.disposed) return; this.busy = true; this.updateControls();
                const form = new FormData(); form.append("image", file); form.append("data", JSON.stringify({ revision: this.document.revision }));
                await this.acceptDocument(await this.request(`/backgrounds/${this.document.background_id}/upload`, { method: "POST", body: form }), true, epoch);
                if (epoch !== this.epoch || this.disposed) return;
                this.busy = false; this.updateControls(); this.message("Background uploaded."); control.value = "";
            }
        } catch (error) { if (epoch === this.epoch && !this.disposed) { this.busy = false; this.updateControls(); this.message(error.message, true); } }
    }

    async runJob(action, description = "", ownsRequest = () => true) {
        if (this.workflowBusy || this.busy || this.restoring || this.disposed || !ownsRequest()) return false;
        if (!await this.ensureDocument() || this.disposed || this.workflowBusy || !ownsRequest()) return false;
        const epoch = this.epoch;
        if (action === "background") this.randomizeSeedIfNeeded();
        this.busy = true; this.activeAction = action;
        if (action === "background") this.startPreviewGeneration();
        this.updateControls(); this.message("Preparing " + action + "…");
        try {
            await this.persist();
            if (epoch !== this.epoch || this.disposed || this.workflowBusy || !ownsRequest()) return false;
            const id = this.state.background_id;
            const payload = { action, editor: clone(this.state.editor), revision: this.state.revision, image_id: this.state.selected_image_id, _vnccs_seed_resolved: true };
            if (action === "wizard") payload.description = description;
            let result = await this.json(`/backgrounds/${id}/jobs`, payload);
            if (this.disposed || epoch !== this.epoch || !ownsRequest()) return false;
            const jobId = result.job_id;
            this.jobId = jobId; this.updateControls();
            while (!this.disposed && epoch === this.epoch && ownsRequest()) {
                result = await this.request(`/jobs/${jobId}`);
                if (this.disposed || epoch !== this.epoch || !ownsRequest()) return false;
                this.message(result.message || "Working…");
                if (["completed", "failed", "cancelled"].includes(result.status)) break;
                await new Promise(resolve => { this.pollTimer = setTimeout(resolve, 500); this.pollResolve = resolve; });
            }
            if (this.disposed || epoch !== this.epoch || !ownsRequest()) return false;
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
                await this.acceptDocument(document, true, epoch);
                this.sync(); this.render();
            }
            this.message(action === "wizard" ? "Scene attributes filled." : "Background generated and saved.");
            return true;
        } finally { if (epoch === this.epoch && !this.disposed) { this.stopPreviewGeneration(); this.jobId = ""; this.busy = false; this.activeAction = ""; this.render(); } }
    }

    resize() {
        const height = `${Math.max(300, (this.node.size?.[1] || CREATOR_MIN_HEIGHT) - 90) / CREATOR_ZOOM}px`;
        const width = this.container.clientWidth || (this.node.size?.[0] || 1280) / CREATOR_ZOOM;
        if (height !== this.container.style.height || width !== this.layoutWidth) this.customSelects._closeDropdown(false);
        this.layoutWidth = width;
        this.container.classList.toggle("bg-compact", width < 1400);
        if (this.container.style.height !== height) this.container.style.height = height;
        let missingHeight = 0;
        // Measure in-flow controls only; collection overlays keep their own scrolling.
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
            this.node.setSize?.([this.node.size[0], this.node.size[1] + Math.ceil(missingHeight * CREATOR_ZOOM)]);
            this.container.style.height = `${(this.node.size[1] - 90) / CREATOR_ZOOM}px`;
        }
        this.updateVariantScroll();
    }
    dispose() {
        if (this.disposed) return;
        this.sync(); this.stopPreviewGeneration();
        this.disposed = true; ++this.epoch; this.events.abort(); clearTimeout(this.saveTimer); clearTimeout(this.pollTimer); this.pollResolve?.();
        this.cc.stopPolling();
        for (const finish of [...this.dialogs]) finish(null);
        this.wizardDialog?.overlay.remove();
        this.errorDialog?.overlay.remove();
        this.observer.disconnect(); this.stylePickerCleanup?.(); this.customSelects._closeDropdown?.(false); this.customSelects.dispose(); this.tooltipCleanup?.(); this.container.remove();
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
    if (node.size?.[1] < CREATOR_MIN_HEIGHT) node.setSize?.([node.size[0], CREATOR_MIN_HEIGHT]);
    syncDOMWidgetWidth(node, "background_ui");
    node.backgroundCreator?.resize();
}

app.registerExtension({
    name: "VNCCS.BackgroundCreator",
    setup() {
        installResolvedSeedTransport(app, ["VNCCS_BackgroundCreator"], "background_data");
        if (app._vnccsBackgroundQueuePatched || typeof app.queuePrompt !== "function") return;
        app._vnccsBackgroundQueuePatched = true;
        const queuePrompt = app.queuePrompt;
        app.queuePrompt = function (...args) {
            for (const node of app.graph?._nodes || []) {
                if (node.type !== "VNCCS_BackgroundCreator" || [2, 4].includes(node.mode)) continue;
                const widget = node.backgroundCreator;
                if (widget?.restoring || widget?.busy) throw new Error("Wait for Background Creator to finish its current operation.");
                widget?.randomizeSeedIfNeeded();
            }
            return queuePrompt.apply(this, args);
        };
    },
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
            this.setSize?.([1280, CREATOR_MIN_HEIGHT]);
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
