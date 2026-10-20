import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { presetSelection } from "../web/character_presets.mjs";
const common = fs.readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const resolutionSource = common.slice(common.indexOf("const RESOLUTION_SCALE_BASE"), common.indexOf("export function normalizeUploadFile")).replaceAll("export ", "");
const traitSource = common.slice(common.indexOf("export function createTraitInput"), common.indexOf("// ── Debounce")).replace("export ", "");
const messageSource = common.slice(common.indexOf("export function showMessage"), common.indexOf("export function createLoadingOverlay")).replace("export ", "");
const cleanupSource = common.slice(common.indexOf("export function registerCleanup"), common.indexOf("// Each loader owns a guard")).replace("export ", "");
const guardSource = common.slice(common.indexOf("export function createRequestGuard"), common.indexOf("// ── DOM Widget Width Sync")).replace("export ", "");
const transport = fs.readFileSync(new URL("../web/vnccs_transport.js", import.meta.url), "utf8");
const connectionSource = transport.slice(transport.indexOf("export function watchConnection")).replace("export ", "");
const source = fs.readFileSync(new URL("../web/vnccs_background_creator.js", import.meta.url), "utf8");
const pure = source.slice(source.indexOf("export const DEFAULT_EDITOR"), source.indexOf("// Workspace widget."));
const { DEFAULT_EDITOR, MOSS_GENERATION, clone, readState, editorDefaults, composePrompt, parsePrompt, legacySceneDescription, isTurboModel, modelPath } = vm.runInThisContext("(() => {" + pure.replaceAll("export ", "") + "\n;return { DEFAULT_EDITOR, MOSS_GENERATION, clone, readState, editorDefaults, composePrompt, parsePrompt, legacySceneDescription, isTurboModel, modelPath }; })()");
const widgetSource = source.slice(0, source.indexOf("function hideState")).replace(pure, "");
const entrySource = source.slice(source.indexOf("function hideState"));
const BASE = "/vnccs/background-creator";
const A = "a".repeat(32), B = "b".repeat(32), IMAGE = "d".repeat(32);
const SCENE_COMPOSITION = {
    front: "A broad path crosses the foreground between low planting beds.",
    back: "Distant architecture frames the horizon beneath an open sky.",
    left: "A shaded colonnade runs along the left edge of the scene.",
    right: "A reflective pool borders the right side of the composition.",
};
const state = (overrides = {}) => ({ schema_version: 2, background_id: A, selected_image_id: IMAGE, revision: 3, editor: editorDefaults(), ...overrides });
const background = (id = A, overrides = {}) => ({ schema_version: 2, background_id: id, name: "Evening room", revision: 3,
    editor: editorDefaults(), selected_image_id: IMAGE, images: [{ image_id: IMAGE, width: 1600, height: 900, url: `${BASE}/backgrounds/${id}/images/${IMAGE}` }], ...overrides });
const configuration = () => ({ installed: true, assets: [], turbo_lora: {
    name: "Qwen Image 2.1 Viggle Turbo", kind: "QI2", type: "TurboLora", status: "installed",
    local_path: "models/loras/QI2/Viggle/current.safetensors" }, loras: ["environment.safetensors"], samplers: ["euler"], schedulers: ["simple"] });
const response = (value, status = 200) => ({ ok: status < 400, status, json: async () => clone(value) });
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const dataKey = key => key.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
const abortCallbacks = new WeakMap();

// Only the DOM operations used by the widget are emulated; no browser or GPU is involved.
class Element {
    constructor(tag = "div") {
        this.tagName = tag.toUpperCase(); this.style = { setProperty(name, value) { this[name] = value; } }; this.dataset = {}; this.attributes = new Map();
        this.listeners = new Map(); this.children = []; this.className = ""; this.type = "";
        this.scrollTop = 0; this.scrollLeft = 0; this.disabled = false; this.hidden = false;
        this.classList = {
            contains: value => this.className.split(/\s+/).includes(value),
            add: (...values) => values.forEach(value => this.classList.toggle(value, true)), remove: (...values) => values.forEach(value => this.classList.toggle(value, false)),
            toggle: (value, enabled) => { const values = new Set(this.className.split(/\s+/).filter(Boolean));
                const active = enabled ?? !values.has(value); active ? values.add(value) : values.delete(value); this.className = [...values].join(" "); return active; },
        };
    }
    set innerHTML(html) {
        this.replaceChildren(); const stack = [this];
        for (const token of html.matchAll(/<\/?[a-z][^>]*>|[^<]+/gi)) {
            if (!token[0].startsWith("<")) { stack.at(-1)._text = (stack.at(-1)._text || "") + token[0]; continue; }
            if (token[0].startsWith("</")) { stack.pop(); continue; }
            const tag = /^<(\w+)/.exec(token[0])[1], child = new Element(tag);
            for (const attribute of token[0].slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) child.setAttribute(attribute[1], attribute[2] ?? "");
            stack.at(-1).appendChild(child);
            if (!["input", "img", "link", "br", "hr", "meta"].includes(tag) && !token[0].endsWith("/>")) stack.push(child);
        }
    }
    get textContent() { return (this._text || "") + this.children.map(child => child.textContent).join(""); }
    set textContent(value) { this.replaceChildren(); this._text = String(value); }
    get value() {
        if (this.tagName === "SELECT") return this._value === undefined ? this.options[0]?.value || "" : this.options.find(option => option.value === this._value)?.value || "";
        return this._value ?? (this.tagName === "OPTION" ? this.textContent : "");
    }
    set value(value) { this._value = String(value); }
    get validity() {
        const value = Number(this.value), min = this.attributes.get("min"), max = this.attributes.get("max");
        return { valid: this.value !== "" && Number.isFinite(value) && (min === undefined || value >= Number(min)) && (max === undefined || value <= Number(max)) };
    }
    get options() { return this.children.filter(child => child.tagName === "OPTION"); }
    setAttribute(name, value) {
        this.attributes.set(name, String(value));
        if (name.startsWith("data-")) this.dataset[dataKey(name.slice(5))] = String(value);
        else if (name === "aria-label") this.ariaLabel = String(value);
        else if (["class", "value", "type", "name", "id"].includes(name)) this[name === "class" ? "className" : name] = String(value);
        else if (["hidden", "disabled", "required"].includes(name)) this[name] = true;
    }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) {
        this.attributes.delete(name);
        if (name.startsWith("data-")) delete this.dataset[dataKey(name.slice(5))];
        else if (name === "src") delete this.src;
        else if (name === "class") this.className = "";
        else if (["hidden", "disabled", "required"].includes(name)) this[name] = false;
    }
    hasAttribute(name) { return name.startsWith("data-") ? Object.hasOwn(this.dataset, dataKey(name.slice(5))) : this.attributes.has(name); }
    matches(selector) {
        return selector.split(",").some(part => {
            part = part.trim();
            if (part === "[inert]") return Boolean(this.inert);
            if (part.startsWith(".")) return part.slice(1).split(".").every(value => this.classList.contains(value));
            const attribute = /^\[([^=\]]+)(?:=["']?([^\]"']+)["']?)?\]$/.exec(part);
            if (attribute) {
                const value = attribute[1].startsWith("data-") ? this.dataset[dataKey(attribute[1].slice(5))] : this.attributes.get(attribute[1]);
                return this.hasAttribute(attribute[1]) && (attribute[2] === undefined || value === attribute[2]);
            }
            return part === "*" || this.tagName.toLowerCase() === part;
        });
    }
    querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    contains(element) { return element === this || this.children.some(child => child.contains(element)); }
    appendChild(child) { if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child); child.parentElement = this; this.children.push(child); return child; }
    before(sibling) { const parent = this.parentElement; if (!parent) return; sibling.parentElement = parent; parent.children.splice(parent.children.indexOf(this), 0, sibling); }
    after(sibling) { const parent = this.parentElement; if (!parent) return; sibling.remove(); sibling.parentElement = parent; parent.children.splice(parent.children.indexOf(this) + 1, 0, sibling); }
    append(...children) { for (const child of children) this.appendChild(child); }
    replaceChildren(...children) { for (const child of this.children) child.parentElement = null; this.children = []; this.append(...children); }
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this); this.parentElement = null; this.removed = true; }
    addEventListener(type, callback, options = {}) {
        const callbacks = this.listeners.get(type) || []; callbacks.push(callback); this.listeners.set(type, callbacks);
        if (options.signal) {
            if (!abortCallbacks.has(options.signal)) {
                const callbacks = new Set(); abortCallbacks.set(options.signal, callbacks);
                options.signal.addEventListener("abort", () => { for (const callback of callbacks) callback(); }, { once: true });
            }
            abortCallbacks.get(options.signal).add(() => this.removeEventListener(type, callback));
        }
    }
    removeEventListener(type, callback) { this.listeners.set(type, (this.listeners.get(type) || []).filter(item => item !== callback)); }
    fire(type, detail = {}) {
        const event = { target: this, preventDefault() {}, stopPropagation() { this.stopped = true; }, ...detail }, results = [];
        for (let element = this; element; element = element.parentElement) {
            results.push(element[`on${type}`]?.(event));
            for (const callback of element.listeners.get(type) || []) results.push(callback(event));
            if (event.stopped) break;
        }
        return Promise.all(results);
    }
    dispatchEvent(event) { return this.fire(event.type, event); }
    getClientRects() { for (let element = this; element; element = element.parentElement) if (element.hidden) return []; return [{}]; }
    focus() { this.focused = true; }
    blur() { this.onblur?.(); }
    click() { return this.fire("click"); }
    showModal() { this.open = true; }
    close() { this.open = false; }
    reportValidity() { return true; }
}

function harness(raw = JSON.stringify(state({ background_id: "", selected_image_id: "" })), queue = { queue_running: [], queue_pending: [] }) {
    const calls = [], events = new Map(), timers = new Map(), drafts = new Map(), queued = [];
    let extension, timerId = 0;
    const window = new Element("window"), documents = new Map([[A, background()], [B, background(B)]]);
    const api = {
        addEventListener(name, callback, options = {}) {
            const callbacks = events.get(name) || []; callbacks.push(callback); events.set(name, callbacks);
            options.signal?.addEventListener("abort", () => api.removeEventListener(name, callback), { once: true });
        },
        removeEventListener(name, callback) { const callbacks = (events.get(name) || []).filter(item => item !== callback); callbacks.length ? events.set(name, callbacks) : events.delete(name); },
        apiURL: value => value,
        async fetchApi(path, options = {}) {
            calls.push({ path, options });
            if (api.handle) { const result = await api.handle(path, options); if (result !== undefined) return result; }
            if (path === "/queue") return response(queue);
            if (path === `${BASE}/config`) return response(configuration());
            if (path === `${BASE}/backgrounds`) return response({ backgrounds: [...documents.values()] });
            const match = new RegExp(`^${BASE}/backgrounds/([a-f0-9]{32})$`).exec(path);
            if (match && !options.method) return response(documents.get(match[1]));
            throw new Error(`Unexpected request: ${options.method || "GET"} ${path}`);
        },
        async queuePrompt(number, prompt, options) { queued.push({ number, prompt, options }); return { prompt_id: "import-prompt" }; },
    };
    const localStorage = { getItem: key => drafts.get(key) ?? null, setItem: (key, value) => drafts.set(key, value), removeItem: key => drafts.delete(key) };
    const context = vm.createContext({
        document: Object.assign(new Element("document"), { getElementById: () => true, createElement: tag => new Element(tag), head: new Element("head"), activeElement: null }),
        window, localStorage, storage: { ...localStorage, setItem: (key, value) => { drafts.set(key, value); return true; } },
        crypto: { randomUUID: () => "01234567-89ab-cdef-0123-456789abcdef", getRandomValues: value => value.fill(17) },
        createControlCenterClient: () => ({ async fetchConfig() {}, startPolling() { this.polling = true; }, stopPolling() { this.polling = false; } }),
        showMessage(container, text, error) {
            const overlay = new Element(); overlay.className = "vnccs-common-message-overlay";
            overlay.textContent = text; overlay.isError = error; container.appendChild(overlay); return overlay;
        },
        showModal(container, title, content, actions) {
            const overlay = new Element(), modal = new Element(); overlay.className = "vnccs-common-modal-overlay"; modal.className = "vnccs-common-modal";
            modal.setAttribute("role", "dialog");
            const heading = new Element(); heading.className = "vnccs-common-modal-title"; heading.textContent = title; modal.appendChild(heading);
            modal.appendChild(content(modal)); const buttons = actions.map(action => { const button = new Element("button"); button.type = "button"; button.className = "vnccs-common-modal-btn" + (action.class === "primary" ? " vnccs-common-modal-btn-primary" : ""); button.textContent = action.text;
                button.onclick = async () => { if (button.disabled) return; if (!action.action || !await action.action(overlay, button)) overlay.remove(); }; modal.appendChild(button); return button; });
            overlay.appendChild(modal); container.appendChild(overlay); return { overlay, modal, buttons };
        },
        SAKURA_THEME_CSS: "", SAKURA_INTERACTION_CSS: "", SAKURA_CREATOR_CONTROL_CSS: "", COMMON_CSS: "", injectStyles() {}, generateRandomSeed: () => 19,
        setHelpText() {}, registerCleanup() {}, enableMiddleMouseCanvasPan() {}, attachHelpTooltips() {},
        installResolvedSeedTransport() {}, syncDOMWidgetWidthSoon() {}, syncDOMWidgetWidth: node => { if (!node.backgroundCreatorDOM._bound) { Object.defineProperty(node.backgroundCreatorDOM, "width", { get: () => node.size[0] }); node.backgroundCreatorDOM._bound = true; } },
        animateWizardRows: (rows, apply) => apply(),
        Event: class { constructor(type, options = {}) { this.type = type; Object.assign(this, options); } }, requestAnimationFrame() {},
        URL, AbortController, FormData, Blob, console, api, vnccsApi: api, presetSelection, DEFAULT_EDITOR, MOSS_GENERATION, clone, readState, editorDefaults, composePrompt, parsePrompt, legacySceneDescription, isTurboModel, modelPath,
        ResizeObserver: class { observe() {} disconnect() { this.disconnected = true; } },
        app: { registerExtension: value => { extension = value; }, graphToPrompt: async () => context.graphPrompt },
        setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
        clearTimeout: id => timers.delete(id),
    });
    vm.runInContext(fs.readFileSync(new URL("../web/sakura_interactions.mjs", import.meta.url), "utf8").replaceAll("export ", ""), context);
    vm.runInContext(fs.readFileSync(new URL("../web/character_styles.mjs", import.meta.url), "utf8").replace(/^import .*\n/gm, "").replaceAll("export ", ""), context);
    vm.runInContext(traitSource, context);
    vm.runInContext(resolutionSource, context);
    vm.runInContext(common.slice(common.indexOf('export function setHelpText'), common.indexOf('export function attachHelpTooltips')).replace('export ', ''), context);
    vm.runInContext(messageSource + cleanupSource + guardSource + connectionSource, context);
    vm.runInContext(widgetSource.replace(/^import .*\n/gm, "").replaceAll("export ", "").replace("class BackgroundCreatorWidget", "globalThis.BackgroundCreatorWidget = class BackgroundCreatorWidget").replaceAll("import.meta.url", '"https://localhost/widget.mjs"'), context);
    vm.runInContext(entrySource.replace(/^import .*\n/gm, ""), context);
    class Node {
        constructor() { this.id = 7; this.widgets = [{ name: "seed", value: 0 }, { name: "background_data", value: raw }]; this.size = [1600, 1000]; this.graph = { setDirtyCanvas() {} }; }
        addDOMWidget(name, type, element, options) { const value = { name, element, options }; this.widgets.push(value); return value; }
        setSize(size) { this.size = size; }
        onNodeCreated() { this.created = true; }
        onConfigure() { this.configured = true; }
        onRemoved() { this.removed = true; }
        onExecuted(message) { this.executed = message; }
    }
    extension.beforeRegisterNodeDef(Node, { name: "VNCCS_BackgroundCreator" });
    const node = new Node(); node.onNodeCreated(); const widget = node.backgroundCreator;
    const fireTimers = delay => { const pending = [...timers].filter(([, timer]) => timer.delay === delay); for (const [id, timer] of pending) { timers.delete(id); timer.callback(); } };
    const emit = (name, detail) => { for (const callback of events.get(name) || []) callback({ detail }); };
    return { node, widget, calls, api, events, timers, drafts, queued, documents, context, window, data: node.widgets[1],
        emit: detail => emit("vnccs_background_creator", { node_id: "7", ...detail }), emitEvent: emit, fireTimers,
        ready: async () => { fireTimers(0); await widget.operations; },
        control: selector => widget.container.querySelector(selector),
    };
}

test("workflow state restores nested generation defaults", () => {
    const saved = state({ editor: { description: "A quiet room", generation: { steps: 31 }, attributes: { lighting: "Soft window light" } } });
    const restored = readState(JSON.stringify(saved));
    assert.equal(restored.editor.generation.steps, 31);
    assert.equal(restored.editor.generation.sampler, "euler");
    assert.equal(restored.editor.attributes.scene_type, "Interior");
    assert.equal(restored.editor.attributes.lighting, "Soft window light");
    restored.editor.generation.lora_stack.push({ name: "a", strength: 1 });
    assert.deepEqual(editorDefaults().generation.lora_stack, []);
    for (const invalid of ["{broken", "[]", "null", JSON.stringify({ schema_version: 3 }), JSON.stringify({ schema_version: 1 }), JSON.stringify(state({ background_id: "../model" })), JSON.stringify(state({ editor: [] }))]) assert.throws(() => readState(invalid));
});

test("Qwen and Moss use Creator type tabs and retain independent generation settings", async () => {
    const qwen = { ...clone(DEFAULT_EDITOR.generation), steps: 31, cfg: 1.7, diffusion_model_name: "user/rooms.safetensors",
        lora_stack: [{ name: "room.safetensors", strength: 0.6 }] };
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ generation: qwen }) }))); await h.ready();
    assert.equal(h.control("[data-model-types]").classList.contains("vnccs-common-creator-tab-row"), true);
    assert.deepEqual(h.widget.container.querySelectorAll("[data-model-type]").map(button => button.textContent), ["Qwen Image 2.1", "Moss"]);
    await h.control('[data-model-type="moss"]').click();
    assert.equal(h.widget.state.editor.generation.model_type, "moss");
    assert.equal(h.widget.state.editor.generation.sampler, "er_sde"); assert.equal(h.widget.state.editor.generation.cfg, 2.5);
    assert.equal(h.widget.state.editor.generation.diffusion_model_name, MOSS_GENERATION.diffusion_model_name);
    assert.equal(h.control('[data-model-type="moss"]').getAttribute("aria-pressed"), "true");
    assert.equal(h.control('[data-model-type="moss"]').classList.contains("is-active"), true);
    assert.equal(h.control('[data-setting="generation.turbo"]').disabled, true);
    assert.equal(h.widget.container.querySelectorAll("[data-qwen-only]").every(section => section.hidden), true);
    assert.deepEqual(clone(h.widget.state.editor.generation.lora_stack), []);
    const steps = h.control('[data-setting="generation.steps"]'); steps.value = "19"; steps.fire("input");
    await h.control('[data-model-type="qi2"]').click();
    assert.deepEqual(clone(h.widget.state.editor.generation), qwen);
    assert.equal(h.widget.container.querySelectorAll("[data-qwen-only]").every(section => !section.hidden), true);
    await h.control('[data-model-type="moss"]').click();
    assert.equal(h.widget.state.editor.generation.steps, 19);
    assert.equal(JSON.parse(h.data.value).editor.generation_profiles.qi2.steps, 31);
});

test("Moss and cached Qwen settings save and load through the selected background JSON", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.api.handle = (path, options) => {
        if (options.method !== "PATCH") return;
        const current = h.documents.get(A), data = JSON.parse(options.body);
        assert.equal(path, `${BASE}/backgrounds/${A}`);
        const saved = { ...current, ...data, revision: current.revision + 1 }; h.documents.set(A, saved); return response(saved);
    };
    await h.control('[data-model-type="moss"]').click();
    await h.widget.persist();
    const saved = h.documents.get(A);
    assert.equal(saved.editor.generation.model_type, "moss"); assert.equal(saved.editor.generation_profiles.qi2.sampler, "euler");
    const restored = state(); delete restored.editor;
    const next = harness(JSON.stringify(restored)); next.documents.set(A, saved); await next.ready();
    assert.equal(next.widget.state.editor.generation.model_type, "moss");
    assert.equal(next.control('[data-model-type="moss"]').getAttribute("aria-pressed"), "true");
    await next.control('[data-model-type="qi2"]').click();
    assert.equal(next.widget.state.editor.generation.diffusion_model_name, DEFAULT_EDITOR.generation.diffusion_model_name);
    assert.equal(next.widget.state.editor.generation.sampler, "euler");
    assert.equal(h.drafts.size, 0); assert.equal(next.drafts.size, 0);
});

test("Moss inventory and missing-asset messages use its own model family", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.config.diffusion_models = [DEFAULT_EDITOR.generation.diffusion_model_name, MOSS_GENERATION.diffusion_model_name];
    h.widget.config.model_types = { moss: { assets: [{ category: "diffusion_models", installed: true },
        { category: "text_encoders", installed: false, name: "qwen3.5_9b_w4a8_gptq.safetensors" }] } };
    await h.control('[data-model-type="moss"]').click();
    const card = h.control("[data-model-toggle]");
    assert.equal(card.classList.contains("is-installed"), true); assert.match(card.textContent, /MOSS-Image/);
    assert.equal(h.widget.modelEntries.some(entry => modelPath(entry) === DEFAULT_EDITOR.generation.diffusion_model_name), false);
    assert.match(h.control("[data-model-picker]").textContent, /Missing: qwen3\.5_9b_w4a8_gptq/);
    await h.control('[data-model-type="qi2"]').click();
    assert.equal(h.widget.modelEntries.some(entry => modelPath(entry) === MOSS_GENERATION.diffusion_model_name), false);
    assert.throws(() => readState(JSON.stringify(state({ editor: { generation: { model_type: "unknown" } } }))), /model type/);
});

test("serialization and queue preserve the latest scene fields", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.editor.scene_description = "My current environment";
    Object.assign(h.widget.state.editor, SCENE_COMPOSITION);
    const info = { widgets_values: [0, "stale"] }; h.node.onSerialize(info);
    assert.equal(JSON.parse(info.widgets_values[1]).editor.scene_description, "My current environment");
    for (const [key, value] of Object.entries(SCENE_COMPOSITION)) assert.equal(JSON.parse(info.widgets_values[1]).editor[key], value);
    h.widget.state.editor.style = "Ink painting"; h.data.beforeQueued();
    assert.equal(JSON.parse(h.data.value).editor.style, "Ink painting");
    assert.equal(JSON.parse(h.data.serializeValue()).editor.style, "Ink painting");
    for (const [key, value] of Object.entries(SCENE_COMPOSITION)) assert.equal(JSON.parse(h.data.serializeValue()).editor[key], value);
    assert.equal(h.control("header,nav,[data-panel=\"model\"]"), null);
    assert.equal(h.calls.some(call => call.path.includes("/models/")), false);
    h.widget.busy = true;
    assert.throws(() => h.data.beforeQueued(), /running|busy|generation|import|operation/i);
});

test("RUN refreshes the background preview and preserves original execution hooks", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const editor = clone(h.widget.state.editor), imageId = "8".repeat(32);
    const document = background(A, { revision: 4, selected_image_id: imageId });
    document.images.push({ image_id: imageId, width: 1600, height: 896, url: "/run-generated.png" });
    h.documents.set(A, document);
    h.emitEvent("executing", "999"); assert.equal(h.widget.busy, false);
    h.emitEvent("executing", "7"); assert.equal(h.widget.busy, true);
    assert.equal(h.control('[data-setting="scene_description"]').disabled, true);
    h.emitEvent("vnccs.background.stage", { node_id: "999", background_id: A, progress: 90, message: "Unrelated node" });
    assert.equal(h.widget.status.textContent, "Preparing background output…");
    h.emitEvent("vnccs.background.stage", { node_id: "7", background_id: A, progress: 45, message: "Rendering background" });
    assert.equal(h.widget.status.textContent, "Rendering background");
    assert.equal(h.control("progress"), null); assert.equal(h.control("[data-generation-layer]").hidden, false);
    const output = { background_preview: [{ background_id: A, image_id: imageId, request_image_id: IMAGE,
        source: "generated", editor }] };
    h.node.onExecuted(output); await h.widget.workflowCompletion;
    assert.equal(h.node.executed, output);
    assert.equal(h.control('[data-panel="model"]'), null);
    assert.equal(h.widget.state.selected_image_id, imageId);
    assert.equal(h.control("[data-preview]").src, "/run-generated.png");
    assert.equal(JSON.parse(h.data.value).selected_image_id, imageId);
    assert.equal(h.widget.document.revision, 4); assert.equal(h.widget.busy, false); assert.equal(h.widget.unsaved, false);
    assert.equal(h.control('[data-setting="scene_description"]').disabled, false);
    assert.deepEqual(h.queued, []);
});

test("cached and uploaded RUN results keep their selected background", async () => {
    for (const source of ["cache", "upload"]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        h.emitEvent("executing", "7");
        h.node.onExecuted({ background_preview: [{ background_id: A, image_id: IMAGE, request_image_id: IMAGE, source,
            editor: clone(h.widget.state.editor) }] });
        await h.widget.workflowCompletion;
        assert.equal(h.widget.state.selected_image_id, IMAGE);
        assert.match(h.widget.status.textContent, source === "cache" ? /reused from cache/ : /passed through/);
        assert.equal(h.widget.busy, false);
    }
});

test("late RUN completion cannot replace edits made after the queued request", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const queued = clone(h.widget.state.editor), document = background(A, { revision: 4 }); h.documents.set(A, document);
    h.widget.state.editor.scene_description = "Keep my newer description"; h.widget.markEdited();
    h.node.onExecuted({ background_preview: [{ background_id: A, image_id: "8".repeat(32), request_image_id: IMAGE,
        source: "generated", editor: queued }] });
    await h.widget.workflowCompletion;
    assert.equal(h.widget.state.editor.scene_description, "Keep my newer description");
    assert.equal(h.widget.state.selected_image_id, IMAGE); assert.equal(h.widget.unsaved, true);
    assert.equal(h.widget.document.revision, 4);
    assert.equal(JSON.parse(h.data.value).editor.scene_description, "Keep my newer description");
});

test("RUN errors and interruption release controls without changing the selected background", async () => {
    for (const interrupted of [false, true]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        h.emitEvent("executing", "7");
        h.emitEvent(interrupted ? "execution_interrupted" : "execution_error", { node_id: "7", exception_message: "Decoder unavailable" });
        await h.widget.workflowCompletion;
        assert.equal(h.widget.state.background_id, A); assert.equal(h.widget.state.selected_image_id, IMAGE);
        assert.equal(h.widget.busy, false);
        assert.match(h.widget.status.textContent, interrupted ? /stopped/ : /Decoder unavailable/);
        assert.equal(h.control('[data-setting="scene_description"]').disabled, false);
    }
});

test("failed or malformed restore preserves original bytes and blocks workflow execution", async () => {
    for (const raw of ["{broken", '{"schema_version":3}', JSON.stringify(state({ selected_image_id: {} }))]) {
        const h = harness(raw); await h.ready();
        assert.equal(h.data.value, raw); assert.equal(h.widget.restoring, true);
        assert.equal(h.data.serializeValue(), raw);
        assert.throws(() => h.data.beforeQueued(), /restoring/i);
    }
    const raw = JSON.stringify(state()); const h = harness(raw);
    h.api.handle = path => path === `${BASE}/backgrounds/${A}` ? response({ error: "Damaged document" }, 409) : undefined;
    await h.ready(); assert.equal(h.data.value, raw); assert.equal(h.widget.restoring, true);
    assert.match(h.widget.status.textContent, /Damaged document/);
    assert.throws(() => h.data.beforeQueued(), /restoring/i);
});

test("an obsolete restore cannot overwrite a later configured background or description", async () => {
    const h = harness(JSON.stringify(state())), first = deferred(); let intercepted = false;
    h.api.handle = path => { if (path === `${BASE}/config` && !intercepted) { intercepted = true; return first.promise; } };
    h.fireTimers(0); await settle(); assert.equal(intercepted, true);
    const saved = JSON.stringify(state({ background_id: B, editor: editorDefaults({ scene_description: "The newly selected garden" }) }));
    h.data.value = saved; h.node.onConfigure(); h.fireTimers(0);
    first.resolve(response(configuration())); await h.widget.operations;
    assert.equal(h.widget.state.background_id, B);
    assert.equal(h.widget.state.editor.scene_description, "The newly selected garden");
    assert.equal(JSON.parse(h.data.value).background_id, B);
});

test("continuous edits save silently into the selected background JSON and preserve scroll positions", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const description = h.control('[data-setting="scene_description"]'), save = deferred(), status = h.widget.status.textContent;
    description.value = "First description"; description.fire("input");
    assert.equal(h.control("[data-saved]"), null);
    assert.equal(h.widget.status.textContent, status);
    assert.equal(JSON.parse(h.data.value).editor.scene_description, "First description");
    h.api.handle = (path, options) => path === `${BASE}/backgrounds/${A}` && options.method === "PATCH" ? save.promise : undefined;
    const pending = h.widget.persist(); await settle();
    description.value = "Latest description"; description.fire("input");
    const panel = h.control(".bg-panel"), gallery = h.control(".bg-variation-cards"); panel.scrollTop = 180; gallery.scrollTop = 95;
    h.widget.render(); assert.equal(panel.scrollTop, 180); assert.equal(gallery.scrollTop, 95);
    save.resolve(response(background(A, { revision: 4, editor: editorDefaults({ scene_description: "First description" }) })));
    await pending;
    assert.equal(h.widget.state.editor.scene_description, "Latest description");
    assert.equal(h.widget.document.revision, 4); assert.equal(h.widget.unsaved, true);
    h.api.handle = (path, options) => path === `${BASE}/backgrounds/${A}` && options.method === "PATCH" ? response(background(A, { revision: 5, editor: h.widget.state.editor })) : undefined;
    await h.widget.persist(); assert.equal(h.widget.unsaved, false);
    assert.equal(h.widget.document.editor.scene_description, "Latest description");
    assert.equal(h.widget.status.textContent, status);
    assert.equal(h.drafts.size, 0);
});

test("a background JSON save conflict retains the visible edit and its document revision", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const description = h.control('[data-setting="scene_description"]'); description.value = "Do not lose this edit"; description.fire("input");
    h.api.handle = (path, options) => options.method === "PATCH" ? response({ error: "Background was modified elsewhere" }, 409) : undefined;
    await assert.rejects(h.widget.persist(), /modified elsewhere/);
    assert.equal(h.widget.document.revision, 3); assert.equal(h.widget.unsaved, true);
    assert.equal(JSON.parse(h.data.value).editor.scene_description, "Do not lose this edit");
    assert.equal(h.widget.state.editor.scene_description, "Do not lose this edit");
    assert.equal(h.control("[data-saved]"), null); assert.equal(h.drafts.size, 0);
});

test("switching backgrounds saves and restores each scene settings", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const other = background(B, { editor: editorDefaults({ scene_description: "An outdoor garden" }) });
    h.documents.set(B, other);
    h.api.handle = (path, options) => {
        if (options.method !== "PATCH") return;
        const id = path.split("/").at(-1), current = h.documents.get(id), data = JSON.parse(options.body);
        assert.equal(data.revision, current.revision);
        const saved = { ...current, ...data, revision: current.revision + 1 }; h.documents.set(id, saved);
        return response(saved);
    };
    const description = h.control('[data-setting="scene_description"]'); description.value = "My painted bedroom"; description.fire("input");
    const select = h.control("[data-library]"); select.value = B; await select.fire("change");
    assert.equal(h.documents.get(A).editor.scene_description, "My painted bedroom");
    assert.equal(h.widget.state.editor.scene_description, "An outdoor garden");
    assert.equal(h.documents.get(B).editor.scene_description, "An outdoor garden");
    select.value = A; await select.fire("change");
    assert.equal(h.widget.state.editor.scene_description, "My painted bedroom");
    assert.equal(h.drafts.size, 0);
});

test("failed background selection keeps the previous background and re-enables its controls", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const loading = deferred();
    h.api.handle = path => path === `${BASE}/backgrounds/${B}` ? loading.promise : undefined;
    const select = h.control("[data-library]"); select.value = B; const pending = select.fire("change"); await settle();
    assert.equal(h.widget.restoring, true);
    assert.equal(h.control('[data-setting="scene_description"]').disabled, true);
    loading.resolve(response({ error: "Cannot load background metadata" }, 500)); await pending;
    assert.equal(h.widget.state.background_id, A); assert.equal(select.value, A);
    assert.equal(h.widget.restoring, false);
    assert.equal(h.control('[data-setting="scene_description"]').disabled, false);
    assert.match(h.widget.status.textContent, /Cannot load background metadata/);
});

test("saved backup recovery exposes retained images and prevents editing damaged disk state", async () => {
    const h = harness(JSON.stringify(state())); h.documents.set(A, background(A, { _recovery: { source: "backup" } })); await h.ready();
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.document._recovery.source, "backup");
    assert.match(h.widget.status.textContent, /read-only|read only/i);
    assert.equal(h.control('[data-action="background"]').disabled, true);
    assert.equal(h.control("[data-upload]").disabled, true);
    assert.equal(h.control('[data-setting="scene_description"]').disabled, true);
});

function variantBackground() {
    const secondImage = "4".repeat(32), document = background();
    document.images.push({ image_id: secondImage, width: 1024, height: 768, url: "/second-background.png" });
    return { document, secondImage };
}

test("a late image upload cannot replace a newly configured background", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const upload = deferred();
    h.api.handle = path => path === `${BASE}/backgrounds/${A}/upload` ? upload.promise : undefined;
    const file = h.control("[data-file]"); file.files = [new Blob(["image"], { type: "image/png" })];
    const action = h.widget.change({ target: file }); await settle();
    assert.equal(h.calls.some(call => call.path.endsWith("/upload")), true);
    const selectedEditor = editorDefaults({ scene_description: "New background description" });
    h.data.value = JSON.stringify(state({ background_id: B, editor: selectedEditor })); h.node.onConfigure(); await h.ready();
    upload.resolve(response(background(A, { revision: 4 }))); await action;
    assert.equal(h.widget.document.background_id, B); assert.equal(h.widget.state.background_id, B);
    assert.deepEqual(h.widget.state.editor, selectedEditor); assert.equal(JSON.parse(h.data.value).background_id, B);
    assert.equal(h.widget.busy, false);
});

for (const mode of ["new", "initial"]) test(`a late ${mode} background creation cannot replace reconfigured state`, async () => {
    const initial = mode === "initial" ? state({ background_id: "", selected_image_id: "" }) : state();
    const h = harness(JSON.stringify(initial)); await h.ready(); const creation = deferred();
    h.widget.dialog = async () => ({ name: "Pending background" });
    h.api.handle = (path, options) => path === `${BASE}/backgrounds` && options.method === "POST" ? creation.promise : undefined;
    const action = mode === "new" ? h.control("[data-new]").click() : h.widget.ensureDocument(); await settle();
    assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds` && call.options.method === "POST"), true);
    h.data.value = JSON.stringify(state({ background_id: B })); h.node.onConfigure(); await h.ready();
    creation.resolve(response(background("3".repeat(32)))); await action;
    assert.equal(h.widget.document.background_id, B); assert.equal(h.widget.state.background_id, B);
    assert.equal(JSON.parse(h.data.value).background_id, B); assert.equal(h.widget.busy, false);
});

test("removal preserves unsaved edits and disposes listeners and pending work", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const description = h.control('[data-setting="scene_description"]'); description.value = "Retained on removal"; description.fire("input");
    h.node.onRemoved();
    assert.equal(h.node.removed, true); assert.equal(h.widget.disposed, true);
    assert.equal(h.widget.observer.disconnected, true);
    assert.equal(h.events.size, 0); assert.equal(h.widget.events.signal.aborted, true);
    assert.equal(h.widget.customSelects._removed, true); assert.equal(h.widget.container.removed, true);
    assert.equal(h.timers.size, 0);
    assert.equal(JSON.parse(h.data.value).editor.scene_description, "Retained on removal");
    assert.equal(h.drafts.size, 0);
});

test("live node resize retains the background controls and tracks both width and height", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const container = h.widget.container;
    h.node.size = [1500, 1050]; h.node.onResize();
    assert.equal(h.node.backgroundCreatorDOM.width, 1500); assert.equal(h.widget.container, container);
    assert.ok(Math.abs(parseFloat(h.widget.container.style.height) * 0.67 - 960) < 1e-6);
    assert.equal(h.control('[data-panel="model"]'), null);
});

const commonSource = fs.readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const creatorSource = fs.readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const clonerSource = fs.readFileSync(new URL("../web/vnccs_character_cloner.js", import.meta.url), "utf8");
const sharedControls = commonSource.match(/export const SAKURA_CREATOR_CONTROL_CSS = `([\s\S]*?)`;/)[1];

test("Background uses the exact Creator declarations for every shared control", () => {
    const normalize = css => css.replaceAll("vnccs-common-creator-", "vnccs-creator-").replaceAll("vnccs-cloner-", "vnccs-creator-").replace(/\s+/g, "");
    const rules = css => [...css.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{([^{}]*)\}/g)]
        .filter(([, selector]) => selector.includes(".vnccs-")).map(([, selector, body]) => [normalize(selector), normalize(body)]);
    const canonical = [creatorSource, clonerSource].flatMap(reference => rules(reference.slice(reference.indexOf("const STYLE = `"), reference.indexOf("`;", reference.indexOf("const STYLE = `")))
        .replace(".vnccs-creator-toggle input:checked ~ ~ .vnccs-creator-toggle-label,\n", "")));
    for (const [selector, declaration] of rules(sharedControls)) {
        const body = declaration.replace("repeat(var(--tabs,3),minmax(0,1fr))", "repeat(3,minmax(0,1fr))")
            .replace("calc((100%-(var(--tabs,3)-1)*8px)/var(--tabs,3))", "calc((100%-16px)/3)");
        assert.ok(canonical.some(([key, value]) => key === selector && value === body), `Creator component differs: ${selector}`);
    }
    assert.doesNotMatch(source, /\.bg-(?:mode|toggle|lora|seed)(?:[ >.{]|$)/);
    assert.doesNotMatch(source, /\.bg-generation (?:input|select)|input\[type=number\] \{ font-family/);
    assert.ok(source.includes("createSakuraInteractions(this.container, node)"));
    assert.ok(source.includes("createStylePicker({"));
    assert.ok(source.includes('class="bg-field bg-description"'));
    assert.ok(source.includes("showModal(this.container"));
    assert.ok(source.includes("animateWizardRows("));
    assert.ok(source.includes("captureView()") && source.includes("restoreView(view)"));
    assert.doesNotMatch(source, /factory3d\/|unicanvas\/|installCustomSelects|Factory3DViewer/);
});

test("Background mounts the Creator component structure and real shared dropdowns", async () => {
    const h = harness(); await h.ready();
    const generation = h.control(".bg-generation");
    assert.equal(generation.classList.contains("vnccs-common-creator-col"), true);
    const grid = generation.querySelector(".vnccs-common-creator-gen-param-grid");
    assert.equal(grid.children.length, 4);
    for (const field of grid.children) assert.equal(field.classList.contains("vnccs-common-creator-gen-param-field"), true);
    for (const setting of ["sampler", "scheduler", "cache_device", "cache_dtype"]) {
        const select = h.control(`[data-setting="generation.${setting}"]`);
        assert.equal(select.hidden, true);
        const trigger = select.parentElement.querySelector(".vnccs-common-ui-select-trigger");
        assert.equal(trigger.classList.contains("vnccs-common-creator-gen-param-select"), true);
        assert.equal(trigger.classList.contains("vnccs-common-creator-btn"), false);
        assert.equal(trigger.value, h.widget.state.editor.generation[setting]);
    }
    const steps = h.control('[data-setting="generation.steps"]');
    assert.equal(steps.classList.contains("vnccs-common-creator-gen-param-input"), true);
    assert.equal(steps.classList.contains("vnccs-common-creator-input"), false);
    const range = h.control('[data-setting="generation.resolution_mp"]');
    assert.equal(range.parentElement.classList.contains("vnccs-common-creator-resolution-field"), true);
    assert.equal(range.classList.contains("vnccs-common-creator-slider"), false);
    assert.equal(range.getAttribute("min"), "1");
    assert.equal(range.getAttribute("max"), "4");
    assert.equal(range.getAttribute("step"), "0.1");
    assert.equal(h.control('[data-resolution]').textContent, '2.0 MP');
    assert.equal(generation.querySelector("details"), null);
    const dice = h.control("[data-random]");
    assert.equal(dice.parentElement.classList.contains("vnccs-common-creator-seed-row"), true);
    assert.equal(dice.querySelector("svg").getAttribute("fill"), "none");
    assert.equal(dice.querySelector("rect").getAttribute("stroke"), "currentColor");
    assert.equal(dice.querySelectorAll("circle").length, 5);
    for (const key of ["prompt", "aesthetics"]) assert.equal(h.control(`[data-setting="${key}"]`), null);
    assert.equal(h.control('.bg-attributes').contains(h.control('[data-setting="negative"]')), true);
    assert.equal(h.control('[data-setting="negative"]').value, '');
    assert.equal(h.control(".bg-bottom"), null);
    const seed = h.control('[data-setting="generation.seed"]');
    await dice.click();
    assert.equal(Number(seed.value), 19); assert.equal(JSON.parse(h.data.value).editor.generation.seed, 19);
});

for (const modelType of ['qi2', 'moss']) test(`${modelType} resolution edits use tenths of a megapixel and serialize the exact selected value`, async () => {
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ generation: { model_type: modelType } }) }))); await h.ready();
    const range = h.control('[data-setting="generation.resolution_mp"]');
    for (const value of [1, 1.1, 1.2, 1.3, 2.7, 3.9, 4]) {
        range.value = value; await range.fire('input');
        assert.equal(h.widget.state.editor.generation.resolution_mp, value);
        assert.equal(JSON.parse(h.data.value).editor.generation.resolution_mp, value);
        assert.equal(h.control('[data-resolution]').textContent, `${value.toFixed(1)} MP`);
        h.widget.render();
        assert.equal(h.control('[data-resolution]').textContent, `${value.toFixed(1)} MP`);
    }
    h.node.onRemoved();
});

for (const megapixels of [.25, 1.25, 8]) test(`restoring a saved ${megapixels} MP value does not rewrite authored settings`, async () => {
    const saved = state({ editor: editorDefaults({ generation: { resolution_mp: megapixels } }) });
    const h = harness(JSON.stringify(saved)), originalDocument = JSON.stringify(h.documents.get(A)); await h.ready();
    assert.equal(h.widget.state.editor.generation.resolution_mp, megapixels);
    assert.equal(JSON.parse(h.data.value).editor.generation.resolution_mp, megapixels);
    assert.equal(JSON.stringify(h.documents.get(A)), originalDocument);
    assert.equal(h.calls.some(call => call.options.method === 'PATCH'), false);
    const range = h.control('[data-setting="generation.resolution_mp"]');
    assert.equal(range.getAttribute('step'), '0.1');
    assert.equal(range.getAttribute('max'), '4');
    if (megapixels < 1 || megapixels > 4) assert.match(range.dataset.vnccsHelp, new RegExp(`Saved value: ${megapixels} MP`));
    range.value = '1.3'; await range.fire('input');
    assert.equal(h.widget.state.editor.generation.resolution_mp, 1.3);
    assert.equal(h.control('[data-resolution]').textContent, '1.3 MP');
    assert.doesNotMatch(range.dataset.vnccsHelp, /Saved value/);
    h.node.onRemoved();
});

test("Creator cards and LoRA selectors reflect saved, edited and blocked states", async () => {
    const saved = state({ editor: editorDefaults({ generation: {
        prompt_enhance: false, turbo: true, steps: 6, cfg: 1, lora_stack: [{ name: "environment.safetensors", strength: 0.7 }],
    } }) });
    const h = harness(JSON.stringify(saved)); await h.ready();
    const enhancement = h.control('[data-setting="generation.prompt_enhance"]');
    const card = enhancement.closest("[data-setting-card]");
    assert.equal(card.classList.contains("is-selected"), false);
    assert.equal(card.querySelector("[data-toggle-status]").textContent, "Disabled");
    assert.equal(enhancement.parentElement.classList.contains("vnccs-common-creator-model-card-toggle"), true);
    enhancement.checked = true; await enhancement.fire("input");
    assert.equal(card.classList.contains("is-selected"), true);
    assert.equal(card.querySelector("[data-toggle-status]").textContent, "Enabled");
    assert.equal(JSON.parse(h.data.value).editor.generation.prompt_enhance, true);
    const turbo = h.control('[data-setting="generation.turbo"]');
    assert.equal(turbo.closest("[data-setting-card]").classList.contains("is-selected"), true);
    turbo.checked = false; await turbo.fire("input");
    assert.equal(JSON.parse(h.data.value).editor.generation.steps, 25);
    assert.equal(turbo.closest("[data-setting-card]").classList.contains("is-selected"), false);
    const lora = h.control('[data-lora="0"]'), item = lora.closest(".vnccs-common-creator-lora-item");
    assert.equal(item.classList.contains("is-empty"), false);
    const trigger = item.querySelector(".vnccs-common-ui-select-trigger");
    assert.equal(trigger.classList.contains("vnccs-common-creator-select"), true);
    assert.equal(trigger.classList.contains("vnccs-common-creator-gen-param-select"), false);
    trigger.value = ""; trigger.onchange();
    assert.equal(item.classList.contains("is-empty"), true);
    assert.equal(JSON.parse(h.data.value).editor.generation.lora_stack.length, 5);
    assert.equal(JSON.parse(h.data.value).editor.generation.lora_stack[0].name, "");
    h.widget.busy = true; h.widget.updateControls();
    assert.equal(enhancement.disabled, true); assert.equal(trigger.disabled, true);
    h.widget.busy = false; h.widget.updateControls();
    assert.equal(enhancement.disabled, false); assert.equal(trigger.disabled, false);
    const model = h.control("[data-model-toggle]");
    assert.equal(model.classList.contains("is-selected"), true);
    h.widget.config.installed = false; h.widget.config.assets = [{ category: "vae", name: "vae.safetensors", installed: false }];
    h.widget.fillChoices();
    assert.equal(h.control("[data-model-toggle]").classList.contains("is-missing"), true);
    assert.match(h.control("[data-model-picker]").textContent, /vae.safetensors/);
});

test("shared dropdown changes persist and retain restored names absent from the inventory", async () => {
    const saved = state({ editor: editorDefaults({ generation: { sampler: "restored_sampler", scheduler: "restored_scheduler", lora_stack: [{ name: "unavailable.safetensors", strength: 0.65 }] } }) });
    const h = harness(JSON.stringify(saved)); await h.ready();
    for (const [selector, value] of [['[data-setting="generation.sampler"]', "restored_sampler"], ['[data-setting="generation.scheduler"]', "restored_scheduler"], ['[data-lora="0"]', "unavailable.safetensors"]]) {
        const select = h.control(selector);
        assert.equal(select.value, value);
        assert.equal(select.parentElement.querySelector(".vnccs-common-ui-select-trigger").value, value);
    }
    const sampler = h.control('[data-setting="generation.sampler"]');
    const trigger = sampler.parentElement.querySelector(".vnccs-common-ui-select-trigger");
    trigger.value = "euler"; trigger.onchange();
    assert.equal(JSON.parse(h.data.value).editor.generation.sampler, "euler");
    assert.equal(JSON.parse(h.data.value).editor.generation.scheduler, "restored_scheduler");
    assert.equal(JSON.parse(h.data.value).editor.generation.lora_stack[0].name, "unavailable.safetensors");
    assert.equal(JSON.parse(h.data.value).editor.generation.lora_stack[0].strength, 0.65);
});

test("background actions retain Creator components without retired model controls", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    assert.equal(h.control("[data-upload]").classList.contains("vnccs-common-creator-btn-upload"), true);
    for (const selector of ['[data-action="background"]', "[data-wizard]"]) {
        assert.equal(h.control(selector).classList.contains("vnccs-common-creator-btn-primary"), true);
    }
    for (const button of h.widget.container.querySelectorAll("button")) assert.equal(button.type, "button");
    for (const selector of ['header', 'nav', '[data-step]', '[data-viewer]', '[data-cgs]', '[data-model-setting]', '[data-save-view]']) assert.equal(h.control(selector), null);
    assert.match(source, /\.bg-panel>\*\s*\{[^}]*flex:0 0 auto/);
});

test("attribute indicators follow the selected value without requiring a rerender", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const sceneType = h.control('[data-trait="scene_type"]');
    await sceneType.children[1].click();
    assert.equal(sceneType.style["--selected"], "1"); assert.equal(sceneType.style["--segment-index"], "1");
    assert.equal(sceneType.children[1].getAttribute("aria-pressed"), "true");
    assert.equal(JSON.parse(h.data.value).editor.attributes.scene_type, "Exterior");
    assert.equal(h.control('[data-trait="time"]'), null);
});

test("decorating repeatedly keeps selectors, dice, traits and dialog buttons in their own components", async () => {
    const h = harness(); await h.ready();
    const modal = new Element(); modal.className = "vnccs-common-modal";
    const confirm = new Element("button"); confirm.className = "vnccs-common-modal-btn vnccs-common-modal-btn-primary";
    modal.appendChild(confirm); h.widget.container.appendChild(modal);
    h.widget.decorateControls();
    const before = h.widget.container.querySelectorAll("button").map(button => [button, button.className]);
    h.widget.decorateControls(); h.widget.customSelects.syncSelects(); h.widget.decorateControls();
    for (const [button, classes] of before) assert.equal(button.className, classes);
    assert.equal(confirm.className, "vnccs-common-modal-btn vnccs-common-modal-btn-primary");
    assert.equal(h.control("[data-random]").className, "vnccs-common-creator-seed-dice-btn");
    for (const trigger of h.widget.container.querySelectorAll(".vnccs-common-ui-select-trigger")) {
        assert.equal(trigger.classList.contains("vnccs-common-creator-model-card-download"), false);
        assert.equal(trigger.classList.contains("vnccs-common-creator-tab"), false);
    }
});

test("all persistent scroll regions retain both axes through navigation and state refreshes", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const positions = h.widget.container.querySelectorAll("[data-sakura-scroll]").map((element, index) => {
        element.scrollTop = 120 + index; element.scrollLeft = 40 + index;
        return [element, element.scrollTop, element.scrollLeft];
    });
    h.widget.render(); await h.control('[data-trait="scene_type"]').children[1].click(); h.widget.render();
    for (const [element, top, left] of positions) { assert.equal(element.scrollTop, top); assert.equal(element.scrollLeft, left); }
});

test("empty, working and archived states expose valid actions without blank image frames", async () => {
    const empty = harness(); await empty.ready();
    assert.equal(empty.control("[data-delete]").disabled, true);
    assert.equal(empty.control('[data-action="model"]'), null);
    assert.equal(empty.control("[data-enlarge]").disabled, true);
    assert.equal(empty.control("[data-new]").disabled, false);
    empty.widget.busy = true; empty.widget.jobId = "job"; empty.widget.updateControls();
    assert.equal(empty.widget.container.getAttribute("aria-busy"), "true");
    assert.equal(empty.control("[data-new]").disabled, true);
    assert.equal(empty.control("[data-stop]"), null); assert.equal(empty.control("progress"), null); assert.equal(empty.control("footer"), null);
    const archived = harness(JSON.stringify(state())); archived.documents.set(A, background(A, { archived: true })); await archived.ready();
    assert.equal(archived.control('[data-setting="scene_description"]').disabled, true);
    assert.equal(archived.control("[data-new]").disabled, false);
});

test("image errors retain saved assets and do not turn later refreshes into broken image placeholders", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const image = h.control("[data-preview]"), original = image.src, saved = JSON.stringify(h.widget.document);
    let loads = 0;
    Object.defineProperty(image, "src", { get: () => original, set: () => { loads++; } });
    h.widget.render(); assert.equal(loads, 0);
    await image.fire("error");
    assert.equal(image.hidden, true); assert.equal(h.control("[data-empty]").hidden, false);
    assert.match(h.control(".vnccs-common-creator-placeholder-title").textContent, /unavailable/);
    assert.equal(JSON.stringify(h.widget.document), saved);
    h.widget.render(); assert.equal(loads, 1);
    assert.equal(image.hidden, false); assert.equal(h.control("[data-empty]").hidden, true);

});

test("unfinished or invalid numeric edits never replace saved parameters with zero or NaN", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    for (const selector of ['[data-setting="generation.steps"]', '[data-setting="generation.seed"]']) {
        const control = h.control(selector), previous = control.value, bytes = h.data.value;
        for (const value of ["", "Infinity", "-100"]) {
            control.value = value; await control.fire("input");
            assert.equal(h.data.value, bytes, `${selector}: ${value}`);
            await control.fire("change"); assert.equal(control.value, previous);
        }
    }
});

test("the background widget has no reconstruction runtime or 3D assets", () => {
    assert.doesNotMatch(source, /BackgroundViewer|CameraHistory|SplatMesh|SparkRenderer|vendor\/spark|TripoSplat/);
    for (const path of ["../data/triposplat/triposplat.py", "../data/triposplat/model.py", "../web/vendor/spark/spark.module.js", "../web/vendor/spark/three.module.js"]) assert.equal(fs.existsSync(new URL(path, import.meta.url)), false);
    assert.ok(source.includes('syncDOMWidgetWidth(node, "background_ui")'));
    assert.ok(source.includes('vnccs-common-creator-seed-row'));
    assert.ok(source.includes('vnccs-common-creator-gen-param-grid'));
});

test("LoRA edits retain all five slots and restored entries beyond them", async () => {
    const saved = state({ editor: editorDefaults({ generation: { lora_stack: [
        { name: "first.safetensors", strength: 1 }, { name: "second.safetensors", strength: 0.5 }, { name: "third.safetensors", strength: 0.3 }, { name: "fourth.safetensors", strength: 0.4 }, { name: "fifth.safetensors", strength: 0.5 }, { name: "retained.safetensors", strength: 0.75 },
    ] } }) });
    const h = harness(JSON.stringify(saved)); await h.ready();
    const first = h.control('[data-lora="0"]'); first.value = "environment.safetensors";
    h.widget.input({ target: first });
    assert.equal(JSON.parse(h.data.value).editor.generation.lora_stack[0].name, "environment.safetensors");
    assert.equal(h.widget.state.editor.generation.lora_stack[5].name, "retained.safetensors");
    assert.equal(JSON.parse(h.data.value).editor.generation.lora_stack[5].strength, 0.75);
});

test("pending background JSON writes guard page exit without separate browser copies", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.context.storage.setItem = () => { throw new Error("Separate recovery writes are forbidden"); };
    const description = h.control('[data-setting="scene_description"]'); description.value = "Keep this draft";
    h.widget.input({ target: description });
    assert.equal(h.widget.unsaved, true); assert.equal(h.drafts.size, 0);
    let blocked = false;
    await h.window.fire("beforeunload", { preventDefault: () => { blocked = true; } });
    assert.equal(blocked, true);
    assert.equal(JSON.parse(h.data.value).editor.scene_description, "Keep this draft");
});

test("Background stage has a growing preview and no duplicate stage or description controls", async () => {
    const h = harness(); await h.ready();
    for (const selector of ["[data-back]", "[data-open-model]", "[data-connected]", '[data-setting="description"]', "[data-thumbnails]"]) assert.equal(h.control(selector), null);
    assert.equal(h.control(".bg-generation").querySelector(".vnccs-common-creator-tab-row"), h.control("[data-model-types]"));
    assert.equal(h.control("[data-model-types]").children.length, 2);
    assert.equal(h.widget.container.querySelectorAll("[data-lora]").length, 5);
    assert.equal(h.widget.container.querySelectorAll('[data-action="model"]').length, 0);
    assert.match(source, /\.bg-panel>\.bg-preview\s*\{[^}]*flex:none;[^}]*width:100%;[^}]*height:auto;[^}]*aspect-ratio:16\/9;[^}]*min-height:0/);
});

test("legacy stored images do not create preview history or navigation in the enlarged view", async () => {
    const { document, secondImage } = variantBackground(), h = harness(JSON.stringify(state()));
    h.documents.set(A, document); await h.ready();
    assert.equal(h.control('[data-variant]'), null);
    assert.equal(h.control('[data-variant-count]'), null);
    await h.control("[data-enlarge]").click();
    const preview = h.control("[data-large-preview]");
    assert.equal(preview.src, document.images[0].url);
    assert.equal(h.control(".bg-image-dialog").classList.contains("vnccs-common-modal"), true);
    assert.equal(h.control('.bg-image-dialog').querySelector('[data-variant]'), null);
    document.selected_image_id = secondImage;
    await h.widget.acceptDocument(document, true);
    assert.equal(preview.src, "/second-background.png");
    assert.equal(h.widget.state.selected_image_id, secondImage);
    assert.equal(h.control('[data-variant-count]'), null);
    await h.control("[data-close-preview]").click();
    assert.equal(h.control("[data-large-preview]"), null); assert.equal(h.widget.dialogs.size, 0);
});

test('replacing a preview with the same image ID refreshes both images without a history counter', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const editor = clone(h.widget.state.editor);
    await h.control('[data-enlarge]').click();
    const document = clone(h.widget.document), image = document.images.find(image => image.image_id === IMAGE);
    image.url += '?v=5'; document.revision++;
    await h.widget.acceptDocument(document, true);
    assert.equal(h.control('[data-preview]').src, image.url);
    assert.equal(h.control('[data-large-preview]').src, image.url);
    assert.equal(h.control('[data-variants]'), null);
    assert.equal(h.widget.state.selected_image_id, IMAGE);
    assert.deepEqual(clone(h.widget.state.editor), editor);
    assert.equal(document.images.length, 1);
    await h.control('[data-close-preview]').click(); h.node.onRemoved();
});

test("legacy prompt data stays recoverable while scene edits use the middle panel", async () => {
    const legacy = editorDefaults({ prompt: "A quiet room\nKeep the red desk", aesthetics: "Old lighting", negative: "Old exclusions", attributes: { location: "Tokyo" } });
    const h = harness(JSON.stringify(state({ editor: legacy }))); await h.ready();
    const field = h.control('[data-setting="scene_description"]');
    assert.match(field.value, /Keep the red desk/);
    field.value += "\nMy exact camera and brushwork instructions."; await field.fire("input");
    await h.control('[data-trait="scene_type"]').children[1].click();
    const prompt = composePrompt(h.widget.state.editor);
    assert.match(prompt, /Time: Day/); assert.match(prompt, /Interior\/Exterior: Exterior/);
    assert.match(prompt, /My exact camera and brushwork instructions\./);
    for (const key of ['prompt','aesthetics','negative']) {
        assert.equal(JSON.parse(h.data.value).editor[key], legacy[key]);
    }
    for (const key of ['prompt', 'aesthetics']) assert.equal(h.control(`[data-setting="${key}"]`), null);
    assert.equal(h.control('[data-setting="negative"]').value, legacy.negative);
    const reopened = harness(h.data.value); await reopened.ready();
    assert.equal(reopened.control('[data-setting="scene_description"]').value, field.value);
    assert.equal(composePrompt(reopened.widget.state.editor), prompt);
});

test("model cards select installed catalog and local models and retain a missing saved choice", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.config.diffusion_models = ["QI2/qwen_image_2.1_int8_convrot.safetensors", "custom/room.safetensors", "qwen_image_2.1_turbo_int8_convrot.safetensors"];
    h.widget.ccState.config = { models: [
        { name: "Qwen Image 2.1", kind: "QI2", type: "unet", local_path: "models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors", status: "missing" },
        { name: "Qwen Image 2.1 Turbo", kind: "QI2", type: "unet", local_path: "models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", status: "installed" },
    ] };
    h.widget.render();
    assert.match(h.control("[data-model-toggle]").textContent, /Qwen Image 2.1/);
    assert.equal(h.control("[data-model-toggle]").classList.contains("is-installed"), true);
    await h.control("[data-model-toggle]").click(); assert.equal(h.control("[data-model-toggle]").getAttribute("aria-expanded"), "true");
    const choice = h.control('[data-model-choice="custom/room.safetensors"]'); await choice.fire("keydown", { key: "Enter" });
    assert.equal(JSON.parse(h.data.value).editor.generation.diffusion_model_name, "custom/room.safetensors");
    assert.equal(h.control("[data-model-toggle]").getAttribute("aria-expanded"), "false");
    await h.control('[data-model-choice="qwen_image_2.1_turbo_int8_convrot.safetensors"]').click();
    assert.equal(h.widget.state.editor.generation.steps, 8); assert.equal(h.widget.state.editor.generation.turbo, false);
    assert.equal(h.control('[data-setting="generation.turbo"]').disabled, true);
    assert.equal(h.control('[data-turbo-lora]').hidden, true);
    await h.control('[data-model-choice="custom/room.safetensors"]').click();
    assert.equal(h.control('[data-turbo-lora]').hidden, false);
    h.widget.busy = true; h.widget.updateControls();
    assert.equal(h.control("[data-model-toggle]").getAttribute("aria-disabled"), "true");
    h.widget.busy = false; h.widget.state.editor.generation.diffusion_model_name = "retained/absent.safetensors"; h.widget.render();
    assert.match(h.control("[data-model-toggle]").textContent, /retained\/absent/);
    assert.equal(h.control('[data-model-choice="retained/absent.safetensors"]').disabled, true);
    assert.equal(h.widget.state.editor.generation.diffusion_model_name, "retained/absent.safetensors");
});

test("model bundle download uses the shared Control Center route and preserves selection on errors", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const entry = { name: "Missing QI2", kind: "QI2", type: "unet", local_path: "models/diffusion_models/missing.safetensors", status: "missing" };
    h.widget.ccState.config = { models: [entry], clip: [{ name: "QI2 encoder", kind: "QI2", status: "missing" }], vae: [{ name: "QI2 VAE", kind: "QI2", status: "installed" }] }; h.widget.render();
    const downloads = [];
    h.api.handle = (path, options) => { if (path === "/vnccs/control_center/download") { assert.equal(options.headers["X-VNCCS-CSRF"], "1"); downloads.push(JSON.parse(options.body)); return response({ status: "queued" }); } };
    await h.control('[data-download-model="Missing QI2"]').click();
    assert.deepEqual(downloads.map(item => [item.category, item.name]), [["models", "Missing QI2"], ["clip", "QI2 encoder"]]);
    assert.equal(h.widget.cc.polling, true);
    assert.equal(h.control('[data-download-model="Missing QI2"]').disabled, true);
    h.widget.ccState.downloadStatus = {}; h.widget.render();
    const selected = h.widget.state.editor.generation.diffusion_model_name;
    h.api.handle = path => path === "/vnccs/control_center/download" ? response({ error: "Disk is full" }, 500) : undefined;
    await h.control('[data-download-model="Missing QI2"]').click();
    assert.match(h.widget.status.textContent, /Disk is full/);
    assert.equal(h.widget.state.editor.generation.diffusion_model_name, selected);
    assert.equal(h.widget.ccState.downloadStatus["cc_models_Missing QI2"].status, "error");
    h.widget.dispose(); assert.equal(h.widget.cc.polling, false);
});

test("old workflows lose only the removed IMAGE socket and retain authored scene state", async () => {
    const saved = state({ editor: editorDefaults({ prompt: "Keep this exact prompt" }) });
    const h = harness(JSON.stringify(saved)); await h.ready();
    h.node.inputs = [{ name: "image", type: "IMAGE", link: 12 }, { name: "mask", type: "MASK", link: 13 }];
    const removed = [];
    h.node.removeInput = index => { removed.push(h.node.inputs[index]); h.node.inputs.splice(index, 1); };
    h.data.value = JSON.stringify(saved); h.node.onConfigure(); await h.ready();
    assert.deepEqual(removed, [{ name: "image", type: "IMAGE", link: 12 }]);
    assert.deepEqual(h.node.inputs, [{ name: "mask", type: "MASK", link: 13 }]);
    assert.equal(h.widget.state.editor.prompt, "Keep this exact prompt");
    assert.equal(h.widget.state.selected_image_id, IMAGE);
    assert.equal(h.control("[data-connected]"), null);
    assert.equal(h.queued.length, 0);
    assert.doesNotMatch(source, /prepareConnected|connectedImagePrompt|importToken|api\.queuePrompt/);
    assert.ok(source.includes('installResolvedSeedTransport(app, ["VNCCS_BackgroundCreator"], "background_data")'));
    assert.ok(source.includes('node.type !== "VNCCS_BackgroundCreator"'));
});

test("the middle scene fields retain their order and compose the structured prompt without a bottom panel", async () => {
    const description = "A large, circular stone table sits in a spacious futuristic room with reflective water and purple flowers.";
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ style: "Anime, digital art, cel shaded", scene_description: description, attributes: { time: "Night", scene_type: "Interior" } }) })));
    await h.ready();
    assert.equal(composePrompt(h.widget.state.editor), `Style: Anime, digital art, cel shaded\nTime: Day\nInterior/Exterior: Interior\nScene description: ${description}`);
    assert.deepEqual(h.control('.bg-attributes').querySelectorAll('.bg-field').map(field => field.querySelector('.vnccs-creator-label')?.textContent || field.children[0].textContent), ["Style", "Interior/Exterior", "Scene description", "Front", "Back", "Left", "Right", "Negative Prompt"]);
    assert.equal(h.control('.bg-attributes').querySelector('[data-setting="aspect"]'), null);
    assert.equal(h.control('.bg-generation').contains(h.control('[data-setting="aspect"]')), true);
    h.widget.styleInfo.custom_style = 'Watercolor, paper texture'; h.widget.stylePicker.setValue('custom', true);
    const field = h.control('[data-setting="scene_description"]'); field.value = 'A new scene\nwith its exact details.'; await field.fire('input');
    const expected = 'Style: Watercolor, paper texture\nTime: Day\nInterior/Exterior: Interior\nScene description: A new scene with its exact details.';
    assert.equal(composePrompt(h.widget.state.editor), expected);
    const reopened = harness(h.data.value); await reopened.ready();
    assert.equal(composePrompt(reopened.widget.state.editor), expected);
    assert.equal(h.control('.bg-bottom'), null);
});

test("scene description migration retains legacy data and never resurrects cleared details", () => {
    const old = { description: "Original authored room", attributes: { architecture: "stone arches", lighting: "blue ceiling panels", location: "Tokyo" } };
    const editor = editorDefaults(old), bytes = JSON.stringify(editor);
    assert.match(editor.scene_description, /Original authored room/); assert.match(editor.scene_description, /stone arches/);
    assert.deepEqual(editorDefaults(JSON.parse(bytes)), editor);
    assert.equal(editor.description, old.description); assert.equal(editor.attributes.lighting, old.attributes.lighting);
    editor.scene_description = "";
    assert.equal(composePrompt(editor).split('\n').at(-1), 'Scene description: ');
    assert.equal(editorDefaults(editor).scene_description, '');
});

test("five equal scene fields save edits, reopen and retain deliberately empty composition", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const fields = h.control('.bg-attributes').querySelectorAll('.bg-description');
    assert.deepEqual(fields.map(field => field.querySelector('textarea').dataset.setting), ['scene_description', 'front', 'back', 'left', 'right']);
    for (const field of fields) {
        const control = field.querySelector('textarea');
        assert.equal(control.getAttribute('rows'), '3'); assert.equal(control.getAttribute('maxlength'), '40000');
        assert.equal(control.parentElement.tagName, 'LABEL');
    }
    assert.match(source, /\.bg-attributes>\.bg-description \{ flex:1 1 0; min-height:100px;/);
    for (const [key, value] of Object.entries(SCENE_COMPOSITION)) {
        const control = h.control(`[data-setting="${key}"]`); control.value = value; await control.fire('input');
        assert.equal(JSON.parse(h.data.value).editor[key], value);
    }
    assert.deepEqual(composePrompt(h.widget.state.editor).split('\n').slice(4),
        Object.entries(SCENE_COMPOSITION).map(([key, value]) => `${key[0].toUpperCase() + key.slice(1)}: ${value}`));
    const reopened = harness(h.data.value); await reopened.ready();
    for (const [key, value] of Object.entries(SCENE_COMPOSITION)) assert.equal(reopened.control(`[data-setting="${key}"]`).value, value);
    reopened.control('[data-setting="front"]').value = ''; await reopened.control('[data-setting="front"]').fire('input');
    reopened.api.handle = (path, options) => options.method === 'PATCH'
        ? response(background(A, { revision: 4, editor: JSON.parse(options.body).editor })) : undefined;
    await reopened.widget.persist();
    const saved = reopened.widget.document.editor;
    assert.equal(saved.front, ''); assert.equal(saved.back, SCENE_COMPOSITION.back);
    assert.doesNotMatch(composePrompt(saved), /\nFront:/);
    assert.equal(readState(JSON.stringify(state({ editor: saved }))).editor.front, '');
    h.node.onRemoved(); reopened.node.onRemoved();
});

test("structured composition prompts restore separate fields without changing legacy descriptions", () => {
    const editor = editorDefaults({ scene_description: 'An open courtyard.', ...SCENE_COMPOSITION });
    const prompt = composePrompt(editor);
    assert.equal(parsePrompt(prompt).scene_description, editor.scene_description);
    assert.equal(composePrompt(editorDefaults({ prompt })), prompt);
    for (const key of Object.keys(SCENE_COMPOSITION)) {
        assert.equal(editorDefaults({ prompt })[key], SCENE_COMPOSITION[key]);
        assert.equal(editorDefaults({ prompt, [key]: '' })[key], '');
        assert.equal(editorDefaults({ scene_description: 'Keep my old room.' })[key], '');
    }
});

test("variant cards open inline Creator trait fields and retain edits in the background JSON", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.api.handle = (path, options) => {
        if (options.method !== 'PATCH') return;
        const current = h.documents.get(A), data = JSON.parse(options.body);
        assert.equal(data.revision, current.revision);
        const saved = { ...current, ...data, revision: current.revision + 1 };
        h.documents.set(A, saved); return response(saved);
    };
    const cards = h.control('[data-variation-cards]');
    assert.deepEqual(cards.children.map(item => item.children[0].textContent), ["Morning", "Evening", "Night"]);
    assert.ok(cards.querySelectorAll('[data-edit-variation]').every(button => button.textContent === ""));
    assert.equal(h.control('[data-variation-editor]').hidden, true);
    assert.equal(h.control('.bg-meta'), null);
    assert.equal(h.control('[data-dimensions]'), null);
    await h.control('[data-edit-variation="morning"]').click();
    assert.equal(h.control('.vnccs-common-modal'), null);
    assert.equal(h.control('[data-variation-editor]').hidden, false);
    assert.equal(h.control('.bg-source').contains(h.control('[data-variation-editor]')), true);
    assert.equal(h.control('[data-edit-variation="morning"]').getAttribute('aria-expanded'), 'true');
    assert.equal(h.control('[data-variation-editor]').querySelectorAll('select').length, 0);
    assert.equal(h.control('[data-variation-editor]').querySelectorAll('.vnccs-common-creator-trait-row').length, 3);
    assert.equal(h.control('[data-variation-field="time"]').value, 'Morning');
    await h.control('[aria-label="Choose weather presets"]').click();
    const modal = h.control('.vnccs-common-modal');
    await modal.querySelectorAll('.vnccs-common-creator-tag-chip').find(chip => chip.textContent === 'Light rain').click();
    await modal.querySelectorAll('button').find(button => button.textContent === 'Apply').click();
    const detail = h.control('[data-variation-field="additional_details"]'); detail.value = 'Keep the blue curtains.'; await detail.fire('input');
    const saved = JSON.parse(h.data.value).editor.variants;
    assert.equal(saved[0].weather, 'Light rain'); assert.equal(saved[0].additional_details, 'Keep the blue curtains.');
    assert.ok(saved.slice(1).every(card => card.weather === ''));
    await h.widget.persist(); assert.deepEqual(h.documents.get(A).editor.variants, saved);
    const reopened = harness(h.data.value); await reopened.ready();
    assert.deepEqual(JSON.parse(reopened.data.value).editor.variants, saved);
});

test("adding, renaming and switching inline variants preserves other cards and authored settings", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const before = clone(h.widget.state.editor);
    await h.control('[data-add-variation]').click();
    assert.equal(h.control('.vnccs-common-modal'), null);
    const name = h.control('[data-variation-name]'); name.value = 'Rainy dawn'; await name.fire('input');
    for (const [key,value] of Object.entries({ time:'Dawn',weather:'Drizzle',additional_details:'Mist over the river.' })) {
        const input = h.control(`[data-variation-field="${key}"]`); input.value=value; await input.fire('input');
    }
    const editor = JSON.parse(h.data.value).editor;
    assert.equal(editor.variants.length, 4); assert.deepEqual(editor.variants.slice(0, 3), before.variants);
    assert.equal(editor.variants[3].name, 'Rainy dawn'); assert.equal(editor.scene_description, before.scene_description);
    await h.control('[data-edit-variation="evening"]').click();
    assert.equal(h.control('[data-variation-field="time"]').value, 'Evening');
    await h.control('[aria-label="Choose time presets"]').click();
    const modal = h.control('.vnccs-common-modal');
    await modal.querySelectorAll('.vnccs-common-creator-tag-chip').find(chip => chip.textContent === 'Midnight').click();
    await modal.querySelectorAll('button').find(button => button.textContent === 'Cancel').click();
    assert.deepEqual(JSON.parse(h.data.value).editor, editor);
    await h.control('[data-close-variation]').click();
    assert.equal(h.control('[data-variation-editor]').hidden, true);
    assert.deepEqual(JSON.parse(h.data.value).editor, editor);
    await h.control('[data-add-variation]').click();
    const ids = h.widget.state.editor.variants.map(card=>card.id);
    assert.equal(new Set(ids).size,ids.length,'Repeated Add clicks must not duplicate card IDs');
    h.node.onRemoved(); assert.equal(h.widget.dialogs.size, 0);
});

test('variant arrows scroll the single strip and update their edge states', async () => {
    const h = harness(JSON.stringify(state()));await h.ready();
    const list=h.control('[data-variation-cards]');list.clientWidth=500;list.scrollWidth=1200;
    list.scrollBy=({left})=>{list.scrollLeft=Math.min(700,Math.max(0,list.scrollLeft+left));h.widget.updateVariantScroll();};
    h.widget.updateVariantScroll();
    assert.equal(h.control('[data-scroll-variation="-1"]').disabled,true);
    assert.equal(h.control('[data-scroll-variation="1"]').disabled,false);
    await h.control('[data-scroll-variation="1"]').click();assert.equal(list.scrollLeft,400);
    await h.control('[data-scroll-variation="1"]').click();assert.equal(list.scrollLeft,700);
    assert.equal(h.control('[data-scroll-variation="1"]').disabled,true);
    await h.control('[data-scroll-variation="-1"]').click();assert.equal(list.scrollLeft,300);
});

test('an old preset picker cannot modify another background after restoration', async () => {
    const h=harness(JSON.stringify(state()));await h.ready();
    await h.control('[data-edit-variation="morning"]').click();
    await h.control('[aria-label="Choose weather presets"]').click();
    const modal=h.control('.vnccs-common-modal');
    await modal.querySelectorAll('.vnccs-common-creator-tag-chip').find(chip=>chip.textContent==='Snow').click();
    await h.widget.acceptDocument(h.documents.get(B),true);
    const before=h.data.value;
    await modal.querySelectorAll('button').find(button=>button.textContent==='Apply').click();
    assert.equal(h.data.value,before);
    assert.ok(h.widget.state.editor.variants.every(card=>card.weather===''));
    assert.equal(h.widget.dialogs.size,0);h.node.onRemoved();
});

test('default and custom variant deletion persists through reload, including the final card', async () => {
    const saved = state();
    saved.editor.variants.push({ id: 'rain', name: 'Rainy dawn', time: 'Dawn', weather: 'Rain', additional_details: 'Keep the curtains.', custom: true });
    const h = harness(JSON.stringify(saved)); await h.ready();
    const original = clone(h.documents.get(A));
    h.api.handle = (path, options) => {
        if (options.method !== 'PATCH') return;
        const current = h.documents.get(A), data = JSON.parse(options.body);
        assert.equal(data.revision, current.revision);
        const saved = { ...current, ...data, revision: current.revision + 1 };
        h.documents.set(A, saved); return response(saved);
    };
    for (const id of ['morning', 'rain', 'evening', 'night']) {
        const remaining = clone(h.widget.state.editor.variants.filter(card => card.id !== id));
        const remove = h.control(`[data-delete-variation="${id}"]`);
        assert.match(remove.getAttribute('aria-label'), /^Delete .+ variant$/);
        await remove.click();
        assert.equal(h.control('.vnccs-common-modal'), null, 'Deleting must not open the edit dialog');
        assert.deepEqual(JSON.parse(h.data.value).editor.variants, remaining, 'Workflow mirror updates immediately');
        await h.widget.persist();
        assert.deepEqual(h.documents.get(A).editor.variants, remaining);
        const reopened = harness(h.data.value); reopened.documents.set(A, clone(h.documents.get(A))); await reopened.ready();
        assert.deepEqual(JSON.parse(reopened.data.value).editor.variants, remaining, 'Deleted defaults must not return');
        assert.equal(reopened.control('[data-variation-cards]').children.length, remaining.length);
        reopened.node.onRemoved();
    }
    assert.deepEqual(h.documents.get(A).images, original.images);
    assert.equal(h.control('[data-add-variation]').disabled, false);
    h.widget.dialog = async () => ({ name: 'New rain', time: 'Day', weather: 'Rain', additional_details: '' });
    await h.control('[data-add-variation]').click();
    assert.equal(h.widget.state.editor.variants.length, 1);
    await h.widget.persist(); h.node.onRemoved();
});

test('variant deletion stays locked during generation and retains its draft when saving fails', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const before = clone(h.widget.state.editor.variants);
    h.widget.busy = true; h.widget.updateControls();
    await h.control('[data-delete-variation="morning"]').click();
    assert.deepEqual(clone(h.widget.state.editor.variants), before);
    h.widget.busy = false; h.widget.updateControls();
    await h.control('[data-delete-variation="morning"]').click();
    h.api.handle = (path, options) => options.method === 'PATCH' ? response({ error: 'Disk full' }, 500) : undefined;
    await assert.rejects(h.widget.persist(), /Disk full/);
    assert.deepEqual(JSON.parse(h.data.value).editor.variants, before.slice(1));
    assert.equal(h.widget.unsaved, true);
    assert.deepEqual(h.documents.get(A).editor.variants, before);
    h.node.onRemoved();
});

test('older workflows gain Prompt LIST socket without changing the image slot or its links', async () => {
    const h = harness();
    const image = { name: 'image', type: 'IMAGE', links: [23] };
    h.node.outputs = [image];
    h.node.addOutput = (name, type, options) => h.node.outputs.push({name,type,...options});
    h.node.onConfigure(); await h.ready();
    assert.equal(h.node.outputs[0], image);
    assert.deepEqual(image.links, [23]);
    assert.equal(h.node.outputs[1].name, 'Prompt'); assert.equal(h.node.outputs[1].type, 'STRING');
    h.node.onConfigure(); await h.ready();
    assert.equal(h.node.outputs.length, 2);
});

test('generation uses the existing Creator scan styles instead of footer progress controls', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    for (const selector of ['footer', 'progress', '[data-stop]']) assert.equal(h.control(selector), null);
    const creator = fs.readFileSync(new URL('../web/vnccs_character_creator_v2.js', import.meta.url), 'utf8');
    const common = fs.readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
    const rules = creator.slice(creator.indexOf('.vnccs-creator-generation-layer {'), creator.indexOf('.vnccs-creator-preview-loading {'));
    assert.ok(common.includes(rules.replaceAll('vnccs-creator-', 'vnccs-common-creator-')));
    for (const rule of creator.split('\n').filter(line => line.startsWith('@keyframes vnccs-creator-generation-'))) assert.ok(common.includes(rule.replaceAll('vnccs-creator-', 'vnccs-common-creator-')));
    const image = h.control('[data-preview]'), previous = image.src;
    h.widget.startPreviewGeneration();
    assert.equal(h.control('[data-generation-layer]').hidden, false);
    assert.equal(h.control('.vnccs-common-creator-generation-base').src, previous);
    assert.equal(h.control('.bg-preview').getAttribute('aria-busy'), 'true');
    h.widget.stopPreviewGeneration();
    assert.equal(h.control('[data-generation-layer]').hidden, true);
    assert.equal(image.src, previous); assert.equal(h.control('.bg-preview').getAttribute('aria-busy'), null);
});

test('the completed preview uses the same synchronized reveal and cleans animations on reload', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const animations = [], pending = [];
    for (const selector of ['.vnccs-common-creator-generation-new', '.vnccs-common-creator-generation-base', '.vnccs-common-creator-generation-scan']) {
        const control = h.control(selector); control.decode = async () => {};
        control.animate = (frames, options) => {
            const done = deferred(); const animation = { frames, options, finished: done.promise, cancel() { done.resolve(); this.cancelled = true; } };
            animations.push(animation); pending.push(done); return animation;
        };
    }
    h.widget.startPreviewGeneration();
    const reveal = h.widget.revealGeneratedPreview('/new-background.png'); await settle();
    assert.equal(h.control('[data-generation-layer]').classList.contains('is-revealing'), true);
    assert.equal(animations.length, 3);
    assert.deepEqual(Array.from(animations[0].frames, frame => frame.clipPath), ['inset(0 0 100% 0)', 'inset(0 0 0% 0)']);
    assert.deepEqual(Array.from(animations[1].frames, frame => frame.clipPath), ['inset(0% 0 0 0)', 'inset(100% 0 0 0)']);
    assert.deepEqual(Array.from(animations[2].frames, frame => frame.transform), ['translateY(-100%)', 'translateY(-50%)', 'translateY(0)']);
    assert.ok(animations.every(animation => animation.options.duration === 1350 && animation.options.easing === 'cubic-bezier(0.4, 0, 0.2, 1)'));
    h.node.onConfigure();
    assert.equal(await reveal, false); assert.ok(animations.every(animation => animation.cancelled));
    await h.ready(); assert.equal(h.control('[data-generation-layer]').hidden, true);
});

test('normal completion keeps the reveal visible until the new preview is installed', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const animations = [];
    for (const control of h.widget.container.querySelectorAll('.vnccs-common-creator-generation-img,.vnccs-common-creator-generation-scan')) {
        control.animate = () => { const animation = { finished: Promise.resolve(), cancel() { this.cancelled = true; } }; animations.push(animation); return animation; };
    }
    h.widget.startPreviewGeneration();
    assert.equal(await h.widget.revealGeneratedPreview('/new-background.png'), true);
    assert.equal(h.control('[data-generation-layer]').hidden, false);
    h.widget.stopPreviewGeneration();
    assert.equal(h.control('[data-generation-layer]').hidden, true);
    assert.ok(animations.every(animation => animation.cancelled));
});

test('empty and interrupted generations restore the existing preview and keep user settings', async () => {
    const h = harness(); await h.ready(); const raw = h.data.value;
    h.widget.startPreviewGeneration(); assert.equal(h.control('[data-empty]').hidden, true);
    h.widget.stopPreviewGeneration(); assert.equal(h.control('[data-empty]').hidden, false);
    assert.equal(h.data.value, raw);
    h.widget.jobId = B; h.widget.busy = true; h.widget.activeAction = 'background'; h.widget.updateControls();
    assert.equal(h.control('[data-action="background"]').textContent, 'Generating...');
    let cancelled = false;
    h.api.handle = (path, options) => { if (path === `${BASE}/jobs/${B}/cancel`) { assert.equal(options.method, 'POST'); cancelled = true; return response({}); } };
    h.emitEvent('execution_interrupted', {}); await settle();
    assert.equal(cancelled, true, 'Standard ComfyUI interruption still cancels the owned preview job');
    assert.equal(h.data.value, raw);
});

test('Background errors use the Creator modal with a title, body and primary acknowledgement', async () => {
    const h = harness(); await h.ready();
    h.widget.message('Decoder unavailable', true);
    const notification = h.control('[data-background-error]');
    assert.equal(notification.querySelector('.vnccs-common-modal-title').textContent, 'Background Creator Error');
    assert.equal(notification.querySelector('.bg-error-text').textContent, 'Decoder unavailable');
    assert.ok(notification.querySelector('.bg-error-dialog'));
    assert.ok(notification.querySelector('.vnccs-common-modal-btn-primary'));
    assert.equal(h.control('.vnccs-common-message-box'), null);
    assert.equal(h.widget.status.textContent, 'Decoder unavailable');
    assert.equal(h.control('footer'), null);
    h.widget.message('Decoder unavailable', true);
    assert.equal(h.widget.container.querySelectorAll('[data-background-error]').length, 1);
});

for (const status of ['failed', 'cancelled']) test(`${status} preview jobs restore the original image and Generate button`, async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const previous = h.control('[data-preview]').src, raw = h.data.value;
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs`) return response({ job_id: B });
        if (path === `${BASE}/jobs/${B}`) return response({ status, error: status === 'failed' ? 'Decoder failed' : undefined });
    };
    await assert.rejects(h.widget.runJob('background'), status === 'failed' ? /Decoder failed/ : /Generation stopped/);
    assert.equal(h.widget.busy, false); assert.equal(h.control('[data-generation-layer]').hidden, true);
    assert.equal(h.control('[data-action="background"]').textContent, 'Generate background');
    assert.equal(h.control('[data-action="background"]').disabled, false);
    assert.equal(h.control('[data-preview]').src, previous); assert.equal(h.data.value, raw);
});

test('reduced motion keeps the same preview handoff without animated effects', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.window.matchMedia = () => ({ matches: true });
    for (const image of h.widget.container.querySelectorAll('.vnccs-common-creator-generation-img')) image.animate = () => assert.fail('Reduced motion must not run the reveal animation');
    h.widget.startPreviewGeneration();
    assert.equal(await h.widget.revealGeneratedPreview('/new-background.png'), true);
    h.widget.stopPreviewGeneration(); assert.equal(h.control('[data-generation-layer]').hidden, true);
});

test('workflow pass-through of an uploaded background does not animate a false generation', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.document.images[0].settings = { source: 'upload' };
    h.emitEvent('executing', '7');
    assert.equal(h.widget.busy, true); assert.equal(h.control('[data-generation-layer]').hidden, true);
    assert.equal(h.control('[data-action="background"]').textContent, 'Workflow busy...');
    h.emitEvent('execution_interrupted', {}); await h.widget.workflowCompletion;
    assert.equal(h.control('[data-action="background"]').textContent, 'Generate background');
    assert.equal(h.control('[data-generation-layer]').hidden, true);
});

function savedVariantSet(editor, { source = IMAGE, id = '1'.repeat(32) } = {}) {
    return { set_id: id, version: "", source_image_id: source, variants: clone(editor.variants),
        images: [{ name: 'Original', url: '/original.png' }, ...editor.variants.map(card => ({ name: card.name, url: `/sets/${id}/${card.id}.png` }))] };
}

test('new backgrounds omit Day while explicitly saved Day settings survive', async () => {
    assert.deepEqual(editorDefaults().variants.map(card => card.name), ['Morning', 'Evening', 'Night']);
    const day = { id: 'day', name: 'Day', time: 'Day', weather: 'Rain', additional_details: 'Lit interior.' };
    const saved = editorDefaults({ variants: [day] });
    assert.deepEqual(saved.variants, [day]);
});

test('saved variant previews match card IDs, parameters and source across reopening and reordering', async () => {
    const h = harness(JSON.stringify(state())), saved = h.documents.get(A);
    const set = savedVariantSet(saved.editor); saved.generated_sets = [set]; await h.ready();
    assert.equal(h.control('[data-variation-image="morning"]').src, set.images[1].url);
    assert.equal(h.control('[data-variation-image="morning"]').hidden, false);
    h.widget.state.editor.variants.reverse(); h.widget.state.editor.variants[0].name = 'After dark'; h.widget.render();
    assert.equal(h.control('[data-variation-image="night"]').src, set.images[3].url);
    h.widget.state.editor.variants[0].weather = 'Rain'; h.widget.render();
    assert.equal(h.control('[data-variation-image="night"]').hidden, true);
    h.widget.state.selected_image_id = '9'.repeat(32); h.widget.render();
    assert.ok(h.widget.container.querySelectorAll('[data-variation-image]').every(image => image.hidden));
    h.node.onRemoved();
});

test('generated cards open their own preview while keeping inline editing and the source unchanged', async () => {
    const h = harness(JSON.stringify(state())), saved = h.documents.get(A);
    const set = savedVariantSet(saved.editor); saved.generated_sets = [set]; await h.ready();
    const raw = h.data.value, source = h.control('[data-preview]').src;
    for (const [index, card] of h.widget.state.editor.variants.entries()) {
        await h.control(`[data-edit-variation="${card.id}"]`).click();
        const image = h.control('[data-large-preview]');
        assert.ok(image, 'A generated card opens the modal preview');
        assert.equal(image.src, set.images[index + 1].url);
        assert.equal(image.alt, `${card.name} variant`);
        h.widget.render();
        assert.equal(image.src, set.images[index + 1].url, 'Rendering must not replace a variant with the base background');
        assert.equal(h.control('[data-preview]').src, source);
        assert.equal(h.data.value, raw, 'Viewing a variant does not write user settings');
        await h.control('[data-close-preview]').click();
        assert.equal(h.control('[data-large-preview]'), null);
        assert.equal(h.control('[data-variation-editor]').hidden, false);
        assert.equal(h.widget.activeVariationId, card.id);
    }
    h.widget.state.editor.variants[0].weather = 'New weather'; h.widget.render();
    await h.control('[data-edit-variation="morning"]').click();
    assert.equal(h.control('[data-large-preview]'), null, 'A card without a matching result opens only its inline settings');
    h.node.onRemoved();
    assert.equal(h.widget.dialogs.size, 0);
});

for (const removeNode of [false, true]) test(`modal navigation releases a captured pointer on ${removeNode ? 'node removal' : 'Close'}`, async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const blurListeners = h.window.listeners.get('blur')?.length || 0;
    await h.control('[data-enlarge]').click();
    const viewport = h.control('[data-preview-viewport]'), image = h.control('[data-large-preview]'), captured = new Set();
    Object.assign(viewport, { clientWidth: 600, clientHeight: 400,
        getBoundingClientRect: () => ({ left: 10, top: 20, width: 300, height: 200 }),
        setPointerCapture: id => captured.add(id), hasPointerCapture: id => captured.has(id), releasePointerCapture: id => captured.delete(id) });
    Object.assign(image, { complete: true, naturalWidth: 1600, naturalHeight: 900 }); image.onload();
    await viewport.fire('pointerdown', { button: 0, buttons: 1, pointerId: 4, clientX: 80, clientY: 100 });
    assert.equal(captured.has(4), true);
    assert.equal(viewport.classList.contains('is-dragging'), true);
    if (removeNode) h.node.onRemoved(); else await h.control('[data-close-preview]').click();
    assert.equal(captured.size, 0);
    assert.equal(viewport.classList.contains('is-dragging'), false);
    assert.equal(viewport.onwheel, null); assert.equal(viewport.onpointermove, null); assert.equal(image.onload, null);
    assert.equal(h.window.listeners.get('blur')?.length || 0, blurListeners);
    assert.equal(h.widget.dialogs.size, 0);
    if (!removeNode) h.node.onRemoved();
});

test('completed background sets refresh cards without overwriting unsaved fields or the main preview', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const original = h.control('[data-preview]').src, revision = h.widget.document.revision;
    h.widget.state.editor.scene_description = 'My unfinished scene'; h.widget.markEdited();
    const saved = h.documents.get(A); saved.generated_sets = [savedVariantSet(saved.editor)]; saved.revision++;
    h.emitEvent('vnccs.background.set', { background_id: B, source_image_id: IMAGE }); await settle();
    assert.equal(h.widget.document.revision, revision);
    h.emitEvent('vnccs.background.set', { background_id: A, source_image_id: IMAGE }); await settle();
    assert.equal(h.widget.document.revision, saved.revision);
    assert.equal(h.widget.state.editor.scene_description, 'My unfinished scene');
    assert.equal(JSON.parse(h.data.value).editor.scene_description, 'My unfinished scene');
    assert.equal(h.control('[data-preview]').src, original);
    assert.equal(h.control('[data-variation-image="morning"]').src, saved.generated_sets[0].images[1].url);
    assert.equal(h.widget.unsaved, true); h.node.onRemoved();
});

test('late set updates and concurrent authored changes cannot rebase another background draft', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const pending = deferred(); h.api.handle = path => path === `${BASE}/backgrounds/${A}` ? pending.promise : undefined;
    const refresh = h.widget.refreshGeneratedSet({ background_id: A, source_image_id: IMAGE }); await settle();
    await h.widget.acceptDocument(h.documents.get(B));
    const saved = h.documents.get(A); saved.generated_sets = [savedVariantSet(saved.editor)]; saved.revision++;
    pending.resolve(response(saved)); await refresh;
    assert.equal(h.widget.document.background_id, B);
    assert.ok(h.widget.container.querySelectorAll('[data-variation-image]').every(image => image.hidden));
    h.api.handle = null; await h.widget.acceptDocument(clone(saved));
    const priorRevision = h.widget.document.revision;
    saved.editor.scene_description = 'Another editor changed this'; saved.revision++;
    h.widget.state.editor.scene_description = 'My draft';
    await h.widget.refreshGeneratedSet({ background_id: A, source_image_id: IMAGE });
    assert.equal(h.widget.document.revision, priorRevision);
    assert.equal(h.widget.state.editor.scene_description, 'My draft');
    h.node.onRemoved();
});

test('restoration error message remains dismissible while authored controls stay locked', async () => {
    const raw = JSON.stringify(state()), h = harness(raw);
    h.api.handle = path => { if (path === `${BASE}/config`) throw new TypeError('Failed to fetch'); };
    await h.ready();
    const overlay = h.control('[data-background-error]'), ok = overlay.querySelector('button');
    assert.equal(h.widget.restoring, true);
    assert.equal(h.control('[data-action="background"]').disabled, true);
    assert.equal(h.data.value, raw);
    assert.equal(ok.disabled, false, 'Dismissing an error must not depend on backend readiness');
    await ok.click(); assert.equal(h.control('[data-background-error]'), null);
    assert.equal(h.data.value, raw); h.node.onRemoved();
});

test('New background starts with fresh variants instead of copying the previous list', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.editor.variants.push({ id: 'day', name: 'Day', time: 'Day', weather: '', additional_details: '' });
    h.widget.state.editor.variants[0].weather = 'Rain';
    h.widget.state.editor.style = 'My preferred style'; h.widget.markEdited();
    let created;
    h.widget.dialog = async () => ({ name: 'Fresh background' });
    h.api.handle = (path, options) => {
        if (options.method === 'PATCH') {
            const payload = JSON.parse(options.body), saved = { ...h.documents.get(A), ...payload, revision: 4 };
            h.documents.set(A, saved); return response(saved);
        }
        if (path === `${BASE}/backgrounds` && options.method === 'POST') {
            created = JSON.parse(options.body);
            return response(background(B, { editor: created.editor, images: [], selected_image_id: '' }));
        }
    };
    await h.control('[data-new]').click();
    assert.deepEqual(created.editor.variants.map(card => card.name), ['Morning', 'Evening', 'Night']);
    assert.ok(created.editor.variants.every(card => card.weather === '' && card.additional_details === ''));
    assert.equal(created.editor.style, 'My preferred style');
    assert.equal(h.documents.get(A).editor.variants[0].weather, 'Rain');
    assert.equal(h.widget.state.background_id, B); h.node.onRemoved();
});

test('reconnection retries a failed restoration without writes and releases the controls', async () => {
    const saved = state(); saved.editor.scene_description = 'Keep this pending scene';
    const raw = JSON.stringify(saved), h = harness(raw);
    h.api.handle = path => { if (path === `${BASE}/config`) throw new TypeError('Failed to fetch'); };
    await h.ready(); assert.equal(h.widget.restoreFailed, true);
    const ok = h.control('[data-background-error]').querySelector('button'); await ok.click();
    assert.equal(h.data.value, raw); h.api.handle = null;
    h.emitEvent('reconnecting'); h.emitEvent('status', {}); await h.widget.operations;
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.restoreFailed, false);
    assert.equal(h.control('[data-action="background"]').disabled, false);
    assert.equal(h.widget.state.editor.scene_description, 'Keep this pending scene');
    assert.equal(h.calls.some(call => ['POST', 'PATCH', 'DELETE'].includes(call.options.method)), false);
    const beforeRemoval = h.calls.length; h.node.onRemoved();
    h.emitEvent('reconnected'); await h.window.fire('focus');
    assert.equal(h.calls.length, beforeRemoval); assert.equal(h.events.size, 0);
});

test('only the untouched legacy default Day is removed from restored cards', () => {
    const plain = { id:'day',name:'Day',time:'Day',weather:'',additional_details:'' };
    const previous = { variants:[...editorDefaults().variants,plain] }, retained = clone(previous);
    assert.deepEqual(editorDefaults(previous).variants.map(card=>card.name), ['Morning','Evening','Night']);
    assert.deepEqual(previous,retained,'Normalization must not mutate the saved input');
    for (const authored of [{...plain,weather:'Rain'},{...plain,additional_details:'Lamps on'},{...plain,id:'custom-day'},{...plain,custom:{retained:true}}]) {
        assert.deepEqual(editorDefaults({variants:[authored]}).variants,[authored]);
    }
});

test('saved selection and fields render before server metadata arrives', async () => {
    const saved = state({ background_name:'My restored background' });
    saved.editor.scene_description='Keep this scene during restore';
    const raw=JSON.stringify(saved), h=harness(raw), metadata=deferred();
    h.api.handle=path=>path===`${BASE}/config`?metadata.promise:undefined;
    h.fireTimers(0);await settle();
    assert.equal(h.control('[data-library]').value,A,'The saved background must already be selected while metadata is pending');
    assert.equal(h.control('[data-library]').children.find(option=>option.value===A).textContent,'My restored background');
    assert.equal(h.control('[data-setting="scene_description"]').value,'Keep this scene during restore');
    assert.equal(h.control('[data-preview]').hidden,false);
    assert.equal(h.control('[data-preview]').src,`${BASE}/backgrounds/${A}/images/${IMAGE}`);
    assert.equal(h.control('[data-empty]').hidden,true);
    assert.equal(h.data.value,raw,'Early rendering must not write into the workflow');
    assert.equal(h.widget.restoring,true);assert.equal(h.control('[data-action="background"]').disabled,true);
    metadata.resolve(response(configuration()));await h.widget.operations;
    assert.equal(h.control('[data-library]').value,A);
    assert.equal(JSON.parse(h.data.value).background_name,'Evening room');h.node.onRemoved();
});

test('legacy workflow references stay selected while loading and empty selections stay empty', async () => {
    const h=harness(JSON.stringify(state())), metadata=deferred();
    h.api.handle=path=>path===`${BASE}/config`?metadata.promise:undefined;
    h.fireTimers(0);await settle();
    assert.equal(h.control('[data-library]').value,A);
    assert.match(h.control('[data-library]').children.find(option=>option.value===A).textContent,/Loading saved background/);
    metadata.resolve(response(configuration()));await h.widget.operations;
    assert.equal(h.widget.state.background_name,'Evening room');h.node.onRemoved();
    const blank=harness();await blank.ready();
    assert.equal(blank.widget.state.background_id,'');assert.equal(blank.control('[data-library]').value,'');
    assert.equal(blank.control('[data-preview]').hidden,true);blank.node.onRemoved();
});

test('saved display names are bounded without accepting invalid cached metadata', () => {
    for(const name of [[],{},15,'x'.repeat(97)])assert.throws(()=>readState(JSON.stringify(state({background_name:name}))),/saved background name/);
    assert.equal(readState(JSON.stringify(state({background_name:'🌄'.repeat(96)}))).background_name,'🌄'.repeat(96));
});

test('the wizard collects a separate brief before starting and saves returned fields through the normal editor path', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const editor = editorDefaults({scene_description: 'Detailed wizard scene', ...SCENE_COMPOSITION}), jobId = '1'.repeat(32);
    let payload;
    h.api.handle = (path, options) => {
        if (path.endsWith('/jobs') && options.method === 'POST') {
            payload = JSON.parse(options.body); assert.equal(payload.action, 'wizard');
            return response({job_id: jobId});
        }
        if (path === `${BASE}/jobs/${jobId}`) return response({status: 'completed', result: {editor}});
        if (options.method === 'PATCH') return response(background(A, {revision: 4, editor}));
    };
    await h.control('[data-wizard]').click();
    const overlay = h.control('[data-background-wizard]'), input = overlay.querySelector('.bg-wizard-input');
    const fill = overlay.querySelectorAll('button').at(-1);
    assert.equal(overlay.querySelector('.vnccs-common-modal-title').textContent, 'Background Wizard');
    assert.equal(h.calls.filter(call => call.path.endsWith('/jobs')).length, 0);
    await fill.click();
    assert.equal(h.control('[data-background-wizard]'), overlay);
    assert.equal(input.focused, true);
    assert.equal(h.calls.filter(call => call.path.endsWith('/jobs')).length, 0);
    input.value = 'A quiet coastal town beneath overcast skies';
    await fill.click();
    assert.equal(payload.description, 'A quiet coastal town beneath overcast skies');
    assert.equal(payload.editor.scene_description, '');
    assert.equal(h.control('[data-background-wizard]'), null);
    assert.equal(h.widget.state.editor.scene_description, 'Detailed wizard scene');
    for (const [key, value] of Object.entries(SCENE_COMPOSITION)) {
        assert.equal(h.control(`[data-setting="${key}"]`).value, value);
        assert.equal(JSON.parse(h.data.value).editor[key], value);
    }
    assert.equal(h.widget.unsaved, true);
    await h.widget.persist();
    assert.equal(h.widget.document.editor.scene_description, 'Detailed wizard scene');
    for (const [key, value] of Object.entries(SCENE_COMPOSITION)) assert.equal(h.widget.document.editor[key], value);
    assert.equal(JSON.parse(h.data.value).revision, 4);
    h.node.onRemoved();
});

test('a stale job-start response cannot redirect a new job after workflow configuration', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const firstStart = deferred(), secondPoll = deferred();
    const firstJob = '1'.repeat(32), secondJob = '2'.repeat(32);
    h.api.handle = (path, options) => {
        if (options.method === 'POST' && path === `${BASE}/backgrounds/${A}/jobs`) return firstStart.promise;
        if (options.method === 'POST' && path === `${BASE}/backgrounds/${B}/jobs`) return response({job_id: secondJob});
        if (path === `${BASE}/jobs/${secondJob}`) return secondPoll.promise;
        if (path === `${BASE}/jobs/${firstJob}`) throw new Error('The stale job must not be polled');
    };
    const first = h.widget.runJob('background'); await settle();
    h.data.value = JSON.stringify(state({background_id: B})); h.node.onConfigure(); await h.widget.operations;
    const second = h.widget.runJob('background'); await settle();
    assert.equal(h.widget.jobId, secondJob);
    firstStart.resolve(response({job_id: firstJob})); await first;
    assert.equal(h.widget.jobId, secondJob);
    secondPoll.resolve(response({status: 'completed', result: {document: background(B)}})); await second;
    assert.equal(h.widget.state.background_id, B);
    assert.equal(h.widget.busy, false);
    assert.equal(h.calls.some(call => call.path === `${BASE}/jobs/${firstJob}`), false);
    h.node.onRemoved();
});

test('a stale status response cannot update another workflow or install a poll timer', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const poll = deferred(), jobId = '1'.repeat(32);
    h.api.handle = (path, options) => path.endsWith('/jobs') && options.method === 'POST'
        ? response({job_id: jobId}) : path === `${BASE}/jobs/${jobId}` ? poll.promise : undefined;
    const pending = h.widget.runJob('background'); await settle();
    h.data.value = JSON.stringify(state({background_id: B})); h.node.onConfigure(); await h.widget.operations;
    const message = h.widget.status.textContent;
    poll.resolve(response({status: 'running', message: 'Stale progress'})); await pending;
    assert.equal(h.widget.state.background_id, B);
    assert.equal(h.widget.status.textContent, message);
    assert.equal([...h.timers.values()].some(timer => timer.delay === 500), false);
    h.node.onRemoved();
});

test('restoring a stale revision keeps workflow settings and allows saving and generation', async () => {
    const h = harness(JSON.stringify(state({revision: 2, editor: editorDefaults({scene_description: 'Workflow draft'})}))); await h.ready();
    assert.equal(h.widget.state.revision, 3); assert.equal(h.widget.document.revision, 3);
    assert.equal(h.widget.state.editor.scene_description, 'Workflow draft');
    h.api.handle = (path, options) => {
        if (options.method === 'PATCH') {
            const payload = JSON.parse(options.body);
            const saved = background(A, {revision: 4, editor: payload.editor});
            h.documents.set(A, saved); return response(saved);
        }
        if (path.endsWith('/jobs') && options.method === 'POST') return response({job_id: B});
        if (path === `${BASE}/jobs/${B}`) return response({status: 'completed', result: {document: h.documents.get(A)}});
    };
    h.widget.state.editor.scene_description = 'Keep this draft'; h.widget.markEdited();
    await h.widget.runJob('background');
    assert.equal(JSON.parse(h.data.value).editor.scene_description, 'Keep this draft');
    assert.equal(h.widget.state.revision, 4); assert.equal(h.widget.document.revision, 4);
    assert.equal(h.widget.unsaved, false);
    h.node.onRemoved();
});

test('matching workflow settings adopt the current disk revision before the first edit', async () => {
    const saved = state({revision: 2});
    saved.editor.selected_image_id = IMAGE;
    const h = harness(JSON.stringify(saved)); await h.ready();
    assert.equal(h.widget.state.revision, 3);
    assert.equal(JSON.parse(h.data.value).revision, 3);
    assert.equal(h.calls.some(call => call.options.method), false);
    h.api.handle = (path, options) => {
        if (options.method !== 'PATCH') return;
        const payload = JSON.parse(options.body);
        assert.equal(payload.revision, 3);
        return response(background(A, {revision: 4, editor: payload.editor}));
    };
    const description = h.control('[data-setting="scene_description"]');
    description.value = 'First edit after reopening'; await description.fire('input');
    await h.widget.persist();
    assert.equal(h.widget.state.revision, 4);
    assert.equal(h.widget.document.editor.scene_description, description.value);
    h.node.onRemoved();
});

test('the same legacy workflow can be reopened and edited after every disk save', async () => {
    for (const revision of [undefined, null, true, 0, -1, 1.5, '2']) {
        const saved = state({revision, editor: editorDefaults({scene_description: 'Unversioned workflow draft'})});
        const raw = JSON.stringify(saved), h = harness(raw);
        h.api.handle = (path, options) => {
            if (options.method !== 'PATCH') return;
            const payload = JSON.parse(options.body), current = h.documents.get(A);
            const document = background(A, {revision: current.revision + 1, editor: payload.editor});
            h.documents.set(A, document); return response(document);
        };
        await h.ready();
        for (let session = 0; session < 2; session++) {
            assert.equal(h.widget.state.editor.scene_description, 'Unversioned workflow draft');
            assert.equal(h.widget.restoring, false);
            assert.equal(h.control('[data-action="background"]').disabled, false);
            h.widget.state.editor.scene_description = `Session ${session}`;
            h.widget.markEdited();
            await h.widget.persist();
            assert.equal(h.documents.get(A).editor.scene_description, `Session ${session}`);
            h.data.value = raw; h.node.onConfigure(); await h.widget.operations;
        }
        h.node.onRemoved();
    }
});

test('RUN acknowledges only its committed revision even if another editor saves before the refresh', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.documents.set(A, background(A, {revision: 5, editor: editorDefaults({scene_description: 'Another editor'})}));
    h.emitEvent('executing', '7');
    h.node.onExecuted({background_preview: [{background_id: A, image_id: IMAGE, request_image_id: IMAGE,
        source: 'upload', revision: 4, editor: clone(h.widget.state.editor)}]});
    await h.widget.workflowCompletion;
    assert.equal(h.widget.state.revision, 4); assert.equal(h.widget.document.revision, 5);
    assert.equal(JSON.parse(h.data.value).revision, 4);
    h.node.onRemoved();
});

test('catalog errors stay visible without blocking a healthy selected background', async () => {
    const h = harness(JSON.stringify(state()));
    h.api.handle = path => path === `${BASE}/backgrounds` ? response({backgrounds: [background()],
        errors: [{name: 'Damaged neighbor', error: 'Cannot restore; original files are preserved'}]}) : undefined;
    await h.ready();
    assert.equal(h.widget.restoring, false); assert.match(h.widget.status.textContent, /Damaged neighbor/);
    assert.equal(h.control('[data-action="background"]').disabled, false);
    assert.equal(h.calls.some(call => call.options.method), false);
    h.node.onRemoved();
});

test('a damaged selected background retains workflow bytes and allows selecting a healthy neighbor', async () => {
    const raw = JSON.stringify(state()), h = harness(raw);
    h.api.handle = path => path === `${BASE}/backgrounds`
        ? response({backgrounds: [background(B)], errors: [{name: 'Damaged room', error: 'Original files are preserved'}]})
        : path === `${BASE}/backgrounds/${A}` ? response({error: 'Cannot restore damaged room'}, 400) : undefined;
    await h.ready();
    assert.equal(h.data.value, raw); assert.equal(h.widget.restoring, true);
    assert.equal(h.control('[data-library]').disabled, false);
    assert.equal(h.control('[data-action="background"]').disabled, true);
    h.control('[data-library]').value = B; await h.control('[data-library]').fire('change');
    assert.equal(h.widget.state.background_id, B); assert.equal(h.widget.restoreFailed, false);
    assert.equal(h.widget.restoring, false); assert.equal(h.control('[data-action="background"]').disabled, false);
    assert.equal(h.calls.some(call => call.options.method), false);
    h.node.onRemoved();
});


test('Turbo toggle restores authored Steps and CFG after save and reload', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    Object.assign(h.widget.state.editor.generation, { steps: 40, cfg: 2.5 });
    const toggle = h.control('[data-setting="generation.turbo"]');
    toggle.checked = true; await toggle.fire('input');
    assert.deepEqual(clone(h.widget.state.editor.generation.turbo_previous_settings), { steps: 40, cfg: 2.5 });
    const reopened = harness(h.data.value); await reopened.ready();
    const restored = reopened.control('[data-setting="generation.turbo"]'); restored.checked = false; await restored.fire('input');
    assert.equal(reopened.widget.state.editor.generation.steps, 40);
    assert.equal(reopened.widget.state.editor.generation.cfg, 2.5);
    h.node.onRemoved(); reopened.node.onRemoved();
});

test('selecting the current built-in Turbo preserves explicit sampling parameters', async () => {
    const name = 'qwen_image_2.1_turbo_int8_convrot.safetensors';
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ generation: { diffusion_model_name: name, steps: 10, cfg: 1.5 } }) })));
    await h.ready(); h.widget.config.diffusion_models = [name]; h.widget.render();
    assert.equal(h.control('[data-turbo-lora]').hidden, true);
    await h.control(`[data-model-choice="${name}"]`).click();
    assert.equal(h.widget.state.editor.generation.steps, 10); assert.equal(h.widget.state.editor.generation.cfg, 1.5);
    assert.equal(editorDefaults({ generation: { diffusion_model_name: name } }).generation.steps, 8);
    assert.equal(editorDefaults({ generation: { turbo: true, steps: 7, cfg: 2 } }).generation.steps, 6);
    h.node.onRemoved();
});

test('seed dice retains random mode and queue generates one displayed seed per run', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    let seed = 20; h.context.generateRandomSeed = () => ++seed;
    const app = h.context.app, submissions = [];
    h.node.type = 'VNCCS_BackgroundCreator'; app.graph = { _nodes: [h.node] };
    app.queuePrompt = function () { assert.equal(this, app); submissions.push(JSON.parse(h.data.value).editor.generation.seed); };
    // Register the same source setup used by ComfyUI.
    const setup = entrySource.slice(entrySource.indexOf('setup() {'), entrySource.indexOf('beforeRegisterNodeDef'));
    vm.runInContext('globalThis.seedExtension = {' + setup + '};', h.context); h.context.seedExtension.setup();
    await h.control('[data-random]').click();
    assert.equal(h.control('[data-random]').getAttribute('aria-pressed'), 'true');
    app.queuePrompt(); app.queuePrompt();
    assert.deepEqual(submissions, [22, 23]);
    assert.equal(JSON.parse(h.data.value).editor.generation.seed_mode, 'randomize');
    await h.control('[data-random]').click(); app.queuePrompt();
    assert.equal(submissions[2], 23); assert.equal(h.widget.state.editor.generation.seed_mode, 'fixed');
    h.node.mode = 2; app.queuePrompt(); assert.equal(submissions[3], 23);
    h.node.onRemoved();
});

test('missing Turbo uses Control Center Download and only installed adapters can be enabled', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.config.turbo_lora.status = 'missing'; h.widget.render();
    assert.equal(h.control('[data-setting="generation.turbo"]').disabled, true);
    const button = h.control('[data-download-turbo]'); assert.equal(button.hidden, false);
    let payload;
    h.api.handle = (path, options) => {
        if (path === '/vnccs/control_center/download') { payload = JSON.parse(options.body); return response({ok: true}); }
    };
    await button.click();
    assert.equal(payload.category, 'lora'); assert.equal(payload.name, 'Qwen Image 2.1 Viggle Turbo');
    assert.equal(h.control('[data-download-turbo]').disabled, true);
    h.widget.ccState.downloadStatus = {};
    h.widget.config.turbo_lora.status = 'installed'; h.widget.syncGenerationCards(); h.widget.updateControls();
    assert.equal(h.control('[data-setting="generation.turbo"]').disabled, false);
    assert.equal(h.control('[data-download-turbo]').hidden, true);
    h.node.onRemoved();
});

test('shared style cards use rendering text, persist selection and preserve old custom text', async () => {
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ style: 'My old brushwork' }) }))); await h.ready();
    const styles = [{ id: 'watercolor', label: 'Watercolor', prompt: 'Watercolor washes and paper grain' }];
    h.widget.config.styles = styles; h.widget.render();
    assert.equal(h.widget.styleInfo.style, 'custom'); assert.equal(h.widget.styleInfo.custom_style, 'My old brushwork');
    assert.ok(h.control('.vnccs-style-summary'));
    h.widget.stylePicker.setValue('watercolor', true);
    const saved = JSON.parse(h.data.value);
    assert.equal(saved.editor.style_id, 'watercolor'); assert.equal(saved.editor.style, styles[0].prompt);
    assert.equal(h.widget.styleInfo.style, 'watercolor');
    assert.match(composePrompt(saved.editor), /Style: Watercolor washes and paper grain/);
    const reopened = harness(h.data.value); await reopened.ready();
    assert.equal(reopened.widget.state.editor.style_id, 'watercolor');
    assert.equal(reopened.widget.state.editor.style, styles[0].prompt);
    reopened.widget.stylePicker.setValue('custom', true);
    assert.equal(reopened.widget.styleInfo.style, 'custom');
    assert.equal(reopened.widget.state.editor.style, 'My old brushwork');
    assert.match(composePrompt(editorDefaults({style: 'Ink wash\nReference: Alice Example'})), /^Style: Ink wash Reference: Alice Example/m);
    assert.match(composePrompt(editorDefaults({style: 'Ink wash. Visual reference: Alice Example'})), /^Style: Ink wash\. Reference: Alice Example/m);
    h.node.onRemoved(); reopened.node.onRemoved();
});

for (const custom of ['My authored ink style', '']) test(`custom style ${custom ? 'text' : 'emptiness'} survives presets, disk save and reopening`, async () => {
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ style: custom }) }))); await h.ready();
    const styles = [{ id: 'watercolor', label: 'Watercolor', prompt: 'Watercolor washes' }, { id: 'ink', label: 'Ink', prompt: 'Ink lines' }];
    h.widget.config.styles = styles; h.widget.render();
    for (const style of styles) {
        h.widget.stylePicker.setValue(style.id, true);
        assert.equal(h.widget.state.editor.custom_style, custom);
    }
    h.api.handle = (path, options) => {
        if (options.method === 'PATCH') {
            const saved = background(A, { editor: JSON.parse(options.body).editor, revision: 4 });
            h.documents.set(A, saved); return response(saved);
        }
    };
    await h.widget.persist();
    const reopened = harness(JSON.stringify(state({ editor: h.documents.get(A).editor, revision: 4 })));
    await reopened.ready();
    reopened.widget.stylePicker.setValue('custom', true);
    assert.equal(reopened.widget.state.editor.style, custom);
    assert.equal(reopened.widget.styleInfo.custom_style, custom);
    reopened.widget.styleInfo.custom_style = 'Edited custom'; reopened.widget.stylePicker.setValue('custom', true);
    assert.equal(JSON.parse(reopened.data.value).editor.custom_style, 'Edited custom');
    h.node.onRemoved(); reopened.node.onRemoved();
});

for (const field of ['queue_running', 'queue_pending']) test(`background actions respect restored ${field} and live workflow status`, async () => {
    const h = harness(JSON.stringify(state()), { queue_running: [], queue_pending: [], [field]: [[1, 'prompt']] });
    assert.equal(h.control('[data-action="background"]').disabled, true);
    await h.ready();
    for (const selector of ['[data-action="background"]', '[data-wizard]', '[data-new]', '[data-delete]', '[data-library]']) assert.equal(h.control(selector).disabled, true);
    await h.widget.runJob('background'); await h.widget.runJob('wizard');
    assert.equal(h.calls.some(call => call.options.method), false);
    h.emitEvent('status', { exec_info: { queue_remaining: 0 } });
    assert.equal(h.control('[data-action="background"]').disabled, false);
    h.emitEvent('execution_start', {}); h.emitEvent('executing', 'another-node');
    assert.equal(h.control('[data-action="background"]').disabled, true);
    h.emitEvent('status', { exec_info: { queue_remaining: 0 } });
    h.widget.busy = true; h.widget.updateControls();
    h.emitEvent('status', { exec_info: { queue_remaining: 0 } });
    assert.equal(h.control('[data-action="background"]').disabled, true, 'Idle queue must not unlock a running background job');
    h.node.onRemoved();
});

test('stale queue responses, reconnection and queue errors cannot unlock background generation', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const stale = deferred(); h.api.handle = path => path === '/queue' ? stale.promise : undefined;
    const pending = h.widget.refreshWorkflowBusy(); h.emitEvent('execution_start', {});
    stale.resolve(response({ queue_running: [], queue_pending: [] })); await pending;
    assert.equal(h.control('[data-action="background"]').disabled, true);
    h.context.console = { ...console, warn() {} };
    h.api.handle = path => path === '/queue' ? response({}, 500) : undefined;
    await h.widget.refreshWorkflowBusy(); assert.equal(h.control('[data-action="background"]').disabled, true);
    h.api.handle = path => path === '/queue' ? response({ queue_running: [], queue_pending: [] }) : undefined;
    h.emitEvent('reconnecting'); h.emitEvent('reconnected'); await settle();
    assert.equal(h.control('[data-action="background"]').disabled, false);
    const removed = deferred(); h.api.handle = path => path === '/queue' ? removed.promise : undefined;
    const last = h.widget.refreshWorkflowBusy(); h.node.onRemoved();
    removed.resolve(response({ queue_running: [[1, 'prompt']], queue_pending: [] })); await last;
    assert.equal(h.widget.workflowBusy, false); assert.equal(h.events.size, 0);
});

test('a failed style library retains the saved scene and allows ordinary edits and generation', async () => {
    const editor = editorDefaults({ style_id: 'user_kept', style: 'My saved rendering', scene_description: 'My room' });
    const h = harness(JSON.stringify(state({ editor })));
    h.api.handle = path => path === `${BASE}/config` ? response({ ...configuration(), styles: [], styles_error: 'Invalid user style library' }) : undefined;
    await h.ready();
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.restoreFailed, false);
    assert.equal(h.widget.state.editor.style, editor.style); assert.equal(h.widget.state.editor.style_id, editor.style_id);
    assert.equal(h.control('[data-preview]').hidden, false); assert.equal(h.control('[data-action="background"]').disabled, false);
    assert.match(h.widget.status.textContent, /Style library unavailable/);
    assert.equal(h.calls.some(call => call.options.method), false);
    const field = h.control('[data-setting="scene_description"]'); field.value = 'My edited room'; await field.fire('input');
    assert.equal(JSON.parse(h.data.value).editor.scene_description, 'My edited room');
    assert.equal(JSON.parse(h.data.value).editor.style, editor.style);
    h.node.onRemoved();
});

test('negative prompt is a compact field below the description and persists independently', async () => {
    const h = harness(JSON.stringify(state({ editor: editorDefaults({ negative: 'text, watermark' }) }))); await h.ready();
    const field = h.control('[data-setting="negative"]'), panel = h.control('.bg-attributes');
    assert.equal(panel.children.at(-1), field.parentElement);
    assert.equal(field.getAttribute('rows'), '2'); assert.equal(field.getAttribute('maxlength'), '8000');
    assert.equal(field.value, 'text, watermark'); assert.equal(h.control('[data-setting="aesthetics"]'), null);
    const rule = source.match(/\.bg-negative textarea \{([^}]+)\}/)[1];
    assert.match(rule, /flex:none/); assert.match(rule, /height:64px/); assert.match(rule, /max-height:64px/);
    const prompt = composePrompt(h.widget.state.editor);
    field.value = 'blur, extra text'; await field.fire('input');
    assert.equal(composePrompt(h.widget.state.editor), prompt);
    const info = { widgets_values: [0, 'stale'] }; h.node.onSerialize(info);
    assert.equal(JSON.parse(info.widgets_values[1]).editor.negative, field.value);
    h.api.handle = (path, options) => options.method === 'PATCH'
        ? response(background(A, { editor: JSON.parse(options.body).editor, revision: 4 })) : undefined;
    await h.widget.persist();
    const reopened = harness(JSON.stringify(state({ editor: h.widget.document.editor, revision: 4 }))); await reopened.ready();
    assert.equal(reopened.control('[data-setting="negative"]').value, 'blur, extra text');
    reopened.control('[data-setting="negative"]').value = ''; await reopened.control('[data-setting="negative"]').fire('input');
    assert.equal(JSON.parse(reopened.data.value).editor.negative, '');
    h.node.onRemoved(); reopened.node.onRemoved();
});

test('choosing the empty selection must clear the selected background', async () => {
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  const select=h.control('[data-library]');select.value='';await select.fire('change');
  console.log('AUDIT empty selector',JSON.stringify({shown:select.value,actual:h.widget.state.background_id,serialized:JSON.parse(h.data.value).background_id}));
  assert.equal(h.widget.state.background_id,'','The placeholder must not leave the old background active for RUN');
 } finally { h.node.onRemoved(); }
});

test('failed switching must keep displayed selection consistent with actual owner', async () => {
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  const edit=h.control('[data-setting="scene_description"]');edit.value='Authored draft';await edit.fire('input');
  h.api.handle=(path,options)=>options.method==='PATCH'?response({error:'Disk save unavailable'},500):undefined;
  await assert.rejects(h.widget.persist(),/Disk save unavailable/);
  const select=h.control('[data-library]');select.value=B;await select.fire('change');
  console.log('Rejected save owner',JSON.stringify({shown:select.value,actual:h.widget.state.background_id,unsaved:h.widget.unsaved,revision:h.widget.state.revision}));
  assert.equal(select.value,h.widget.state.background_id,'A rejected save must not display another background as selected');
 } finally { h.node.onRemoved(); }
});

test('state accepted by the form must remain reloadable', async () => {
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  const variants=Array.from({length:40},(_,i)=>({id:'v'+i,name:'Variant '+i,time:'Morning',weather:'',additional_details:'x'.repeat(8000)}));
  h.widget.state.editor=editorDefaults({variants});h.widget.sync();
  console.log('AUDIT state size',JSON.stringify({characters:h.data.value.length,cards:variants.length,perCardDetails:8000}));
  assert.doesNotThrow(()=>readState(h.data.value),'Allowed controls must not create a workflow that the same widget refuses to restore');
 } finally { h.node.onRemoved(); }
});

test('middle button must remain graph PAN in the image dialog', async () => {
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  const viewport=new Element('div'),image=new Element('img');
  Object.assign(viewport,{clientWidth:400,clientHeight:200,getBoundingClientRect:()=>({left:0,top:0,width:400,height:200}),setPointerCapture:id=>{viewport.captured=id;},hasPointerCapture:()=>false});
  Object.assign(image,{complete:true,naturalWidth:800,naturalHeight:400});
  const dispose=h.widget.attachPreviewNavigation(viewport,image);
  try {
   let stopped=false;viewport.onpointerdown({button:1,isPrimary:true,pointerId:1,clientX:100,clientY:50,preventDefault(){},stopPropagation(){stopped=true;}});
   console.log('AUDIT middle button',JSON.stringify({imageDrag:viewport.classList.contains('is-dragging'),stopped,captured:viewport.captured}));
   assert.equal(viewport.classList.contains('is-dragging'),false,'MMB must not start image drag');
  } finally {dispose();}
 } finally {h.node.onRemoved();}
});

test('a known Klein model remains offered as a Qwen text-to-image model',async()=>{
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  const foreign='flux2-klein-9b.safetensors';
  h.widget.config.diffusion_models=[DEFAULT_EDITOR.generation.diffusion_model_name,foreign];
  h.widget.ccState.config={models:[{kind:'Klein9b',type:'unet',name:'Flux Klein9b',local_path:'models/diffusion_models/'+foreign,status:'installed'}]};
  h.widget.renderModelPicker();
  const entry=h.widget.modelEntries.find(e=>modelPath(e)===foreign);
  console.log('AUDIT foreign model',JSON.stringify(entry));
  assert.ok(entry, 'The owner permits a broad installed-model picker');
 } finally {h.node.onRemoved();}
});

test('LoRA numeric drafts survive rerender while focused',async()=>{
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  const weight=h.control('[data-lora-strength="0"]');h.context.document.activeElement=weight;weight.value='';
  h.widget.render();
  console.log('AUDIT LoRA draft',JSON.stringify({focused:h.context.document.activeElement===weight,value:weight.value}));
  assert.equal(weight.value,'','A repaint must not replace the active unfinished numeric draft');
 } finally{h.node.onRemoved();}
});

test('reduced motion disables arrow-triggered smooth scrolling',async()=>{
 const h=harness(JSON.stringify(state()));await h.ready();
 try {
  h.window.matchMedia=()=>({matches:true});
  const list=h.control('[data-variation-cards]'),button=h.control('[data-scroll-variation="1"]');let options;
  list.clientWidth=400;list.scrollBy=value=>{options=value;};button.disabled=false;
  await h.widget.click({target:button});
  console.log('AUDIT reduced scroll',JSON.stringify(options));
  assert.equal(options.behavior,'auto','Reduced motion must apply to imperative scroll animation too');
 } finally{h.node.onRemoved();}
});


test('the approved full-width variant editor receives height without changing saved settings or node width', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const raw = h.data.value;
    assert.deepEqual([...h.node.size], [1280, 840]);
    for (const width of [1280, 1600]) {
        h.node.size = [width, 800]; h.node.onResize();
        assert.deepEqual([...h.node.size], [width, 840]);
        assert.equal(h.data.value, raw);
    }
    h.node.size = [1600, 1000]; h.node.onResize();
    assert.deepEqual([...h.node.size], [1600, 1000]);
});


test('height follows missing form space once at non-unit scale while an overlaid model list stays local', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.context.getComputedStyle = () => ({ paddingBottom: '16px', borderBottomWidth: '1px' });
    const panel = h.widget.container.querySelector('.bg-source');
    panel.offsetHeight = 1000;
    panel.getBoundingClientRect = () => ({ height: 500 + (h.node.size[1] - 840) / .67 * .5, bottom: 500 + (h.node.size[1] - 840) / .67 * .5 });
    for (const child of panel.children) child.getBoundingClientRect = () => ({ bottom: 495 });
    const width = h.node.size[0];
    h.widget.resize();
    assert.deepEqual([...h.node.size], [width, 845]);
    h.widget.resize();
    assert.deepEqual([...h.node.size], [width, 845]);
    const menu = h.widget.container.querySelector('.vnccs-common-creator-model-picker-menu');
    menu.getBoundingClientRect = () => ({ bottom: 100000 });
    h.widget.resize();
    assert.deepEqual([...h.node.size], [width, 845]);
});

test('Background style gallery sends the current editor to its own cover endpoint and cleans its progress listener', async () => {
    const catalog = JSON.parse(fs.readFileSync(new URL('../background_template/background_styles.json', import.meta.url), 'utf8'));
    const styles = catalog.groups.flatMap(group => group.styles);
    const editor = editorDefaults({ style_id: styles[0].id, style: styles[0].prompt, custom_style: '', scene_description: 'My environment',
        generation: { model_type: 'moss', steps: 21, resolution_mp: 2.5, lora_stack: [{ name: 'Scenery.safetensors', strength: .7 }] } });
    const h = harness(JSON.stringify(state({ editor }))); await h.ready();
    h.widget.config.styles = styles; h.widget.render();
    let release; const sent = [];
    h.api.handle = (path, options) => {
        if (path === `${BASE}/styles`) return response(structuredClone(catalog));
        if (path === `${BASE}/styles/preview`) {
            const payload = JSON.parse(options.body); sent.push(payload);
            return new Promise(resolve => { release = () => resolve(response({ saved: true, style_id: payload.style_id,
                image: `${BASE}/styles/preview?style=${payload.style_id}&v=1`, width: 1024, height: 1024 })); });
        }
    };
    await h.control('.vnccs-style-summary').onclick();
    const before = h.data.value;
    const pending = h.widget.stylePicker.generatePreviews();
    assert.equal(sent.length, 1);
    assert.equal(sent[0].node_id, '7'); assert.equal(sent[0].character_info, undefined);
    assert.equal(sent[0].editor.scene_description, 'My environment'); assert.equal(sent[0].editor.generation.model_type, 'moss');
    assert.equal(sent[0].editor.generation.resolution_mp, 2.5); assert.equal(sent[0].editor.generation.lora_stack[0].strength, .7);
    h.emitEvent('vnccs.style_preview.stage', { node_id: 'other', request_id: sent[0].request_id, status: 'queued' });
    assert.doesNotMatch(h.control('.vnccs-style-status').textContent, /^Queued/);
    h.emitEvent('vnccs.style_preview.stage', { node_id: '7', request_id: sent[0].request_id, status: 'queued' });
    assert.match(h.control('.vnccs-style-status').textContent, /^Queued/);
    await h.widget.stylePicker.generatePreviews(); release(); await pending;
    assert.equal(sent.length, 1); assert.equal(h.data.value, before);
    assert.equal(h.events.get('vnccs.style_preview.stage').length, 1);
    h.node.onRemoved(); assert.equal(h.events.has('vnccs.style_preview.stage'), false);
});

test('a failed wizard opens the Creator-style error modal and keeps authored fields', async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.editor.scene_description = ''; h.widget.markEdited();
    const before = h.data.value;
    h.api.handle = (path, options) => {
        if (options.method === 'PATCH') return response(background(A, {editor: JSON.parse(options.body).editor}));
        if (path.endsWith('/jobs') && options.method === 'POST') return response({job_id: B});
        if (path === `${BASE}/jobs/${B}`) return response({status: 'failed', error: 'Describe the background for the wizard'});
    };
    await h.control('[data-wizard]').click();
    const wizard = h.control('[data-background-wizard]'), input = wizard.querySelector('.bg-wizard-input');
    input.value = 'A moonlit garden';
    await wizard.querySelectorAll('button').at(-1).click();
    const error = h.control('[data-background-error]');
    assert.equal(error.querySelector('.vnccs-common-modal-title').textContent, 'Background Wizard Error');
    assert.equal(error.querySelector('.bg-error-text').textContent, 'Describe the background for the wizard');
    assert.equal(h.control('.vnccs-common-message-box'), null);
    assert.equal(h.widget.busy, false);
    assert.equal(h.widget.state.editor.scene_description, '');
    assert.equal(input.value, 'A moonlit garden');
    const authored = JSON.parse(before).editor;
    assert.deepEqual(JSON.parse(h.data.value).editor, authored);
    h.widget.restoring = true; h.widget.updateControls();
    assert.equal(error.querySelector('button').disabled, false);
    assert.equal(wizard.querySelector('button').disabled, false);
    await error.querySelector('button').click();
    assert.equal(h.control('[data-background-error]'), null);
    assert.equal(h.control('[data-background-wizard]'), wizard);
    assert.equal(input.focused, true);
    await wizard.querySelector('button').click();
    assert.equal(h.control('[data-background-wizard]'), null);
    h.node.onRemoved();
});

test('Background error dialogs use safe text, replace obsolete errors and are disposed with the node', async () => {
    const h = harness(); await h.ready();
    h.widget.message('<img src=x onerror=alert(1)>', true);
    assert.equal(h.control('.bg-error-text').textContent, '<img src=x onerror=alert(1)>');
    assert.equal(h.control('.bg-error-text').children.length, 0);
    const previous = h.widget.errorDialog.overlay;
    h.widget.message('Another failure', true);
    assert.equal(h.widget.container.contains(previous), false);
    assert.equal(h.widget.container.querySelectorAll('[data-background-error]').length, 1);
    const current = h.widget.errorDialog.overlay;
    h.node.onRemoved();
    assert.equal(h.widget.container.contains(current), false);
});
