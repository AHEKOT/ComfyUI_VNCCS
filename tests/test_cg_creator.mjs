import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { webcrypto } from "node:crypto";
import { FactoryCommandHistory } from "./fixtures/factory_history.mjs";
import { source, clone, readState, settingsDefaults, frameDimensions, sourceCameraView, MODEL_DEFAULTS, sourceChoices, selectedSource, sameSource, widgetSource, entrySource, connectionSource, cleanupSource, syncDOMWidgetWidth, BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH, CREATOR_MIN_HEIGHT } from "./helpers/cg_frontend.mjs";

const BASE = "/vnccs/cg-creator";
const A = "a".repeat(32), B = "b".repeat(32), MODEL = "c".repeat(32), IMAGE = "d".repeat(32), CG = "e".repeat(32);
const camera = { position: [0, 0.2, 2], target: [0, 0, 0], up: [0, 1, 0], fov: 42 };
const state = (overrides = {}) => ({ schema_version: 1, background_id: A, selected_image_id: IMAGE, model_id: MODEL,
    selected_cg_id: CG, camera: clone(camera), settings: settingsDefaults(), model_settings: clone(MODEL_DEFAULTS), ...overrides });
const background = (id = A, overrides = {}) => ({ schema_version: 2, background_id: id, name: "Evening room", revision: 3,
    cg_settings: settingsDefaults(), model_settings: clone(MODEL_DEFAULTS), selected_image_id: IMAGE,
    images: [{ image_id: IMAGE, width: 1600, height: 900, url: `${BASE}/backgrounds/${id}/images/${IMAGE}` }],
    models: [{ model_id: MODEL, source_image_id: IMAGE, settings: { provider: "pixal3d" } }], model: { model_id: MODEL, source_image_id: IMAGE },
    cgs: [{ cg_id: CG, name: "Window view", model_id: MODEL, source_image_id: IMAGE,
        width: 1600, height: 896, camera: clone(camera), url: `${BASE}/backgrounds/${id}/cg/${CG}/image` }], ...overrides });
const scene = (id = A, overrides = {}) => ({ scene_id: "f".repeat(32), model_id: MODEL, background_id: id, camera: clone(camera),
    objects: [{ object_id: B, locked: true, source_camera: clone(camera) }], ...overrides });
const configuration = () => ({ settings: settingsDefaults(), installed: true, assets: [], loras: ["environment.safetensors"], samplers: ["euler"], schedulers: ["simple"] });
const response = (value, status = 200) => ({ ok: status < 400, status, json: async () => clone(value) });
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const settle = () => new Promise(resolve => setImmediate(resolve));
const dataKey = key => key.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
const abortCallbacks = new WeakMap();

test("malformed nested profiles are retained as invalid workflow data", () => {
    for (const changes of [{ settings: { generation: null } }, { settings: { generation: [] } }, { model_settings: [] }]) {
        const raw = JSON.stringify(state(changes));
        assert.throws(() => readState(raw), /settings/i);
    }
});

test("unavailable Pose Studio leaves the background usable and Add Character retryable", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const saved = h.data.value, viewer = h.widget.viewer;
    h.context.loadCharacterRuntime = async () => { throw new Error("Pose Studio assets are unavailable."); };
    await h.control("[data-add-character]").click();
    assert.equal(h.widget.busy, false); assert.equal(h.control("[data-add-character]").disabled, false);
    assert.equal(h.data.value, saved); assert.equal(h.widget.viewer, viewer);
    assert.match(h.widget.status.textContent, /Pose Studio assets are unavailable/);
});

test("corrupt character state retains its original bytes during restoration", async () => {
    const raw = JSON.stringify(state({ characters: { [MODEL]: [{ id: B, position: [0, 0, 0], scale: -1, pose: {} }] } }));
    const h = harness(raw); await h.ready();
    assert.equal(h.data.value, raw); assert.equal(h.widget.restoreFailed, true);
    assert.equal(h.control("[data-add-character]").disabled, true);
});

test("Add Character works without randomUUID and adds only one character", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.context.crypto = { getRandomValues: values => webcrypto.getRandomValues(values) };
    const entries = new Map();
    h.widget.characters = {
        entries, snapshot: () => [...entries.values()],
        add(id) { entries.set(id, { id, position: [0, 0, 0], scale: 1,
            pose: { bones: {}, bonePositions: {}, modelRotation: [0, 0, 0] } }); },
    };
    await h.control("[data-add-character]").click();
    await h.control("[data-add-character]").click();
    assert.equal(entries.size, 1, h.widget.status.textContent);
    assert.ok([...entries.keys()].every(id => /^[a-f0-9]{32}$/.test(id)));
    assert.deepEqual(readState(h.data.value).characters[MODEL], [...entries.values()]);
    assert.equal(h.control("[data-add-character]").disabled, false);
});

// Only the DOM operations used by the widget are emulated; no browser or GPU is involved.
class Element {
    constructor(tag = "div") {
        this.tagName = tag.toUpperCase(); this.style = { setProperty(name, value) { this[name] = String(value); } }; this.dataset = {}; this.attributes = new Map();
        this.listeners = new Map(); this.children = []; this.className = ""; this.value = ""; this.type = "";
        if (["SELECT", "TEXTAREA"].includes(this.tagName)) Object.defineProperty(this, "type", { get: () => this.tagName === "SELECT" ? "select-one" : "textarea" });
        this.scrollTop = 0; this.scrollLeft = 0; this.disabled = false; this.hidden = false;
        this.classList = {
            contains: value => this.className.split(/\s+/).includes(value),
            add: value => this.classList.toggle(value, true), remove: value => this.classList.toggle(value, false),
            toggle: (value, enabled) => { const values = new Set(this.className.split(/\s+/).filter(Boolean));
                const active = enabled ?? !values.has(value); active ? values.add(value) : values.delete(value); this.className = [...values].join(" "); return active; },
        };
    }
    set innerHTML(html) {
        this.replaceChildren(); const stack = [this];
        for (const token of html.matchAll(/<\/?[a-z][^>]*>/gi)) {
            if (token[0].startsWith("</")) { stack.pop(); continue; }
            const tag = /^<(\w+)/.exec(token[0])[1], child = new Element(tag);
            for (const attribute of token[0].slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) child.setAttribute(attribute[1], attribute[2] ?? "");
            stack.at(-1).appendChild(child);
            if (!["input", "img", "link", "br", "hr", "meta"].includes(tag) && !token[0].endsWith("/>")) stack.push(child);
        }
    }
    setAttribute(name, value) {
        this.attributes.set(name, String(value));
        if (name.startsWith("data-")) this.dataset[dataKey(name.slice(5))] = String(value);
        else if (["class", "value", "type", "name", "id"].includes(name)) this[name === "class" ? "className" : name] = String(value);
        else if (["hidden", "disabled", "required"].includes(name)) this[name] = true;
    }
    hasAttribute(name) { return name.startsWith("data-") ? Object.hasOwn(this.dataset, dataKey(name.slice(5))) : this.attributes.has(name); }
    removeAttribute(name) { this.attributes.delete(name); if (name.startsWith("data-")) delete this.dataset[dataKey(name.slice(5))]; }
    matches(selector) {
        return selector.split(",").some(part => {
            part = part.trim();
            if (part === "*") return true;
            if (part.startsWith(".")) return this.classList.contains(part.slice(1));
            const attribute = /^\[([^=\]]+)(?:=["']?([^\]"']+)["']?)?\]$/.exec(part);
            if (attribute) {
                const value = attribute[1].startsWith("data-") ? this.dataset[dataKey(attribute[1].slice(5))] : this.attributes.get(attribute[1]);
                return this.hasAttribute(attribute[1]) && (attribute[2] === undefined || value === attribute[2]);
            }
            return this.tagName.toLowerCase() === part;
        });
    }
    querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    contains(element) { return element === this || this.children.some(child => child.contains(element)); }
    appendChild(child) { if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child); child.parentElement = this; this.children.push(child); return child; }
    append(...children) { for (const child of children) this.appendChild(child); }
    replaceChildren(...children) { for (const child of this.children) child.parentElement = null; this.children = []; this.append(...children); }
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this); this.parentElement = null; this.removed = true; }
    replaceWith(child) {
        const parent = this.parentElement;
        if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child);
        parent.children.splice(parent.children.indexOf(this), 1, child); child.parentElement = parent; this.parentElement = null; this.removed = true;
    }
    cloneNode(deep = false) {
        const copy = new Element(this.tagName);
        copy.style = { ...this.style }; copy.dataset = { ...this.dataset }; copy.attributes = new Map(this.attributes);
        for (const key of ["className", "value", "src", "hidden", "disabled", "textContent", "width", "height"]) copy[key] = this[key];
        if (deep) for (const child of this.children) copy.appendChild(child.cloneNode(true));
        return copy;
    }
    decode() { return Promise.resolve(); }
    getContext() { return { drawImage: bitmap => { this.bitmap = bitmap; } }; }
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
        const event = { target: this, preventDefault() {}, ...detail }, results = [];
        for (let element = this; element; element = element.parentElement) for (const callback of element.listeners.get(type) || []) results.push(callback(event));
        return Promise.all(results);
    }
    click() { return this.fire("click"); }
    showModal() { this.open = true; }
    close() { this.open = false; }
    getAnimations() { return this.animations || []; }
    reportValidity() { return (!this.required || this.value !== "") && (!this.pattern || new RegExp("^(?:" + this.pattern + ")$", "u").test(this.value)); }
}

function harness(raw = JSON.stringify(state({ background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", camera: null })), { nodeId = 7, drafts = new Map(), session = new Map(), scope = "tab:workflow" } = {}) {
    const calls = [], events = new Map(), timers = new Map(), queued = [];
    let extension, timerId = 0;
    const window = new Element("window"), documents = new Map([[A, background()], [B, background(B)]]);
    class Viewer {
        constructor(host, options) { this.options = options; this.canvas = new Element("canvas"); this.canvas.width = 1600; this.canvas.height = 900; host.appendChild(this.canvas); this.controls = {}; this.camera = clone(camera); this.frames = []; this.loads = []; }
        async setScene(value) { this.scene = value; this.loads.push(value); this.camera = { ...camera, position: [99, 99, 99] }; return { failures: [] }; }
        setLighting(settings) { this.lighting = settings; }
        setCaptureSettings(settings) { this.settings = settings; }
        setZoomSensitivity(value) { this.zoomSensitivity = value; }
        setCameraState(value, { emit = false } = {}) { this.camera = { ...this.camera, ...clone(value) }; this.frames.push(this.camera); if (emit) this.options.onStateChange({ camera: this.camera }); }
        getCameraState() { return clone(this.camera); }
        select() {}
        resize() { this.resized = true; }
        async capturePreview(options) { this.captureOptions = options; return new Blob(["png"]); }
        dispose() { this.disposed = true; }
    }
    const api = {
        addEventListener(name, callback) { const callbacks = events.get(name) || []; callbacks.push(callback); events.set(name, callbacks); },
        removeEventListener(name, callback) { const callbacks = (events.get(name) || []).filter(item => item !== callback); callbacks.length ? events.set(name, callbacks) : events.delete(name); },
        apiURL: value => value,
        async fetchApi(path, options = {}) {
            calls.push({ path, options });
            if (api.handle) { const result = await api.handle(path, options); if (result !== undefined) return result; }
            if (path === `${BASE}/config`) return response(configuration());
            if (path === `${BASE}/backgrounds`) return response({ backgrounds: [...documents.values()] });
            const match = new RegExp(`^${BASE}/backgrounds/([a-f0-9]{32})(/models/[a-f0-9]{32})?$`).exec(path);
            if (match && !options.method) {
                const document = documents.get(match[1]);
                if (!match[2]) return response(document);
                const modelId = match[2].split("/").at(-1), model = (document.models || [document.model]).find(item => item?.model_id === modelId);
                return response(scene(match[1], { model_id: modelId, source_image_id: model?.source_image_id, source_variant: model?.source_variant }));
            }
            throw new Error(`Unexpected request: ${options.method || "GET"} ${path}`);
        },
        async queuePrompt(number, prompt, options) { queued.push({ number, prompt, options }); return { prompt_id: "import-prompt" }; },
    };
    const localStorage = { getItem: key => drafts.get(key) ?? null, setItem: (key, value) => (drafts.set(key, value), true), removeItem: key => drafts.delete(key) };
    const context = vm.createContext({
        document: Object.assign(new Element("document"), { getElementById: () => true, createElement: tag => new Element(tag), head: new Element("head"), activeElement: null }),
        window, storage: localStorage, sessionStore: { getItem: key => session.get(key), setItem: (key, value) => session.set(key, value) },
        STYLE: "", injectStyles() {}, generateRandomSeed: () => 424242, workflowScope: node => scope + ":" + node.id, cacheIdentity: () => Math.random().toString(36).slice(2), crypto: { randomUUID: () => "01234567-89ab-cdef-0123-456789abcdef" },
        URL, AbortController, FormData, Blob, createImageBitmap: async () => ({ close() { this.closed = true; } }), console, api, clone, readState, settingsDefaults, frameDimensions, sourceCameraView, MODEL_DEFAULTS, sourceChoices, selectedSource, sameSource, syncDOMWidgetWidth, syncDOMWidgetWidthSoon() {}, BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH, CREATOR_MIN_HEIGHT,
        ResizeObserver: class { constructor(callback) { this.callback = callback; } observe(target) { this.target = target; } disconnect() { this.disconnected = true; } },
        createSakuraInteractions: () => ({ syncSelects() {}, captureView() { return {}; }, restoreView() {}, dispose() { this.disconnected = true; } }),
        playSakuraMotion(_owner, _element, _frames, _duration, finish) { finish?.(); }, stopSakuraMotion() {},
        CGQi2Controls: class { constructor(host, options) { this.options = options; this.host = host; this.els = {}; } render(config) { this.config = config; } setDisabled(value) { this.disabled = value; } dispose() { this.disposed = true; } },
        showModal: (container, title, createContent, actions) => {
            const overlay = new Element("div"), modal = new Element("div"), content = createContent(), buttons = [];
            for (const element of [overlay, modal]) element.animations = [{ cancel() { this.cancelled = true; } }];
            modal.setAttribute("role", "dialog"); modal.appendChild(content); overlay.appendChild(modal); container.appendChild(overlay);
            for (const action of actions) { const button = new Element("button"); button.textContent = action.text; button.onclick = () => { if (action.action?.() !== true) overlay.remove(); }; modal.appendChild(button); buttons.push(button); }
            return { overlay, modal, content, buttons };
        },
        loadFactoryRuntime: async () => ({ Factory3DViewer: Viewer, FactoryCommandHistory }),
        enableMiddleMouseCanvasPan: () => () => {},
        Factory3DViewer: Viewer, FactoryCommandHistory,
        app: { registerExtension: value => { extension = value; }, graphToPrompt: async () => { throw new Error("CG Creator must not queue the workflow"); } },
        setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
        clearTimeout: id => timers.delete(id),
    });
    vm.runInContext(cleanupSource + connectionSource, context);
    vm.runInContext(widgetSource.replace(/^import .*\n/gm, "").replace("class CGCreatorWidget", "globalThis.CGCreatorWidget = class CGCreatorWidget").replaceAll("import.meta.url", '"https://localhost/widget.mjs"'), context);
    vm.runInContext(entrySource.replace(/^import .*\n/gm, ""), context);
    class Node {
        constructor() { this.id = nodeId; this.widgets = [{ name: "cg_data", value: raw }]; this.size = [1500, 950]; this.graph = { setDirtyCanvas() {} }; }
        addDOMWidget(name, type, element, options) { const value = { name, element, options }; this.widgets.push(value); return value; }
        setSize(size) { this.size = size; }
        onNodeCreated() { this.created = true; }
        onConfigure() { this.configured = true; }
        onRemoved() { this.removed = true; }
    }
    extension.beforeRegisterNodeDef(Node, { name: "VNCCS_CGCreator" });
    const node = new Node(); node.onNodeCreated(); const widget = node.cgCreator;
    const fireTimers = delay => { const pending = [...timers].filter(([, timer]) => timer.delay === delay); for (const [id, timer] of pending) { timers.delete(id); timer.callback(); } };
    const emit = (name, detail) => { for (const callback of events.get(name) || []) callback({ detail }); };
    return { node, widget, calls, api, events, timers, drafts, queued, documents, context, window, extension, data: node.widgets[0],
        emit: detail => emit("vnccs_cg_creator", { node_id: "7", ...detail }), emitEvent: emit, fireTimers,
        ready: async ({ loadSource = true } = {}) => { fireTimers(0); await widget.operations; if (loadSource) await widget.sourceLoad?.onload?.(); },
        control: selector => widget.container.querySelector(selector),
    };
}

function generateCG(h, name = "Window view") {
    const running = h.control("[data-save-view]").click(), dialog = h.control('[role="dialog"]');
    dialog.querySelector("input").value = name;
    dialog.querySelectorAll("button").at(-1).onclick();
    return running;
}

test("CG state restores nested settings without sharing mutable defaults", () => {
    const restored = readState(JSON.stringify(state({ settings: { prompt: "Keep the room", generation: { steps: 31 } } })));
    assert.equal(restored.settings.prompt, "Keep the room"); assert.equal(restored.settings.generation.steps, 31);
    assert.equal(restored.settings.generation.sampler, "euler"); assert.equal(restored.selected_cg_id, CG);
    restored.settings.generation.lora_stack.push({ name: "environment.safetensors", strength: 1 });
    assert.deepEqual(settingsDefaults().generation.lora_stack, []);
    for (const raw of ["{broken", "[]", "null", '{"schema_version":2}', JSON.stringify(state({ model_id: "../model" })), JSON.stringify(state({ camera: {} }))]) assert.throws(() => readState(raw));
});

test("the node owns its settings across background and CG selection and refresh", async () => {
    const settings = settingsDefaults({ aspect: "9:16", resolution_mp: 3, generation: {
        diffusion_model_name: "Local models/custom_checkpoint.safetensors", steps: 39, cfg: 0, seed: 915,
    } });
    const model_settings = { ...MODEL_DEFAULTS, provider: "triposplat", pixal_profile: "standard", steps: 33 };
    const h = harness(JSON.stringify(state({ settings, model_settings })));
    const other = background(B, { model_settings: MODEL_DEFAULTS });
    h.documents.set(B, other);
    await h.ready();
    const check = () => {
        const saved = JSON.parse(h.data.serializeValue());
        assert.deepEqual(saved.settings, settings);
        assert.deepEqual(saved.model_settings, model_settings);
    };
    check();

    const select = h.control("[data-library]"); select.value = B;
    const switching = h.widget.change({ target: select }); await settle(); await h.widget.sourceLoad.onload(); await switching;
    assert.equal(h.widget.state.background_id, B);
    check();
    await h.control("[data-cg-id]").click();
    assert.equal(h.widget.state.selected_cg_id, CG);
    assert.deepEqual(clone(h.widget.state.frame), frameDimensions("9:16", 3));
    check();
    await h.window.fire("focus");
    check();
    assert.equal(h.calls.some(call => call.options.method === "PATCH"), false);
    assert.deepEqual(h.documents.get(B), other);
});

test("missing legacy node fields use initial values without importing background preferences", async () => {
    const h = harness(JSON.stringify(state({ settings: { generation: { steps: 39, cfg: 0 } }, model_settings: undefined })));
    h.documents.set(A, background(A, {
        cg_settings: settingsDefaults({ generation: { diffusion_model_name: "Other.safetensors", steps: 13, cfg: 2 } }),
        model_settings: { ...MODEL_DEFAULTS, provider: "triposplat", steps: 33 },
    }));
    await h.ready();
    const saved = JSON.parse(h.data.value);
    assert.deepEqual(saved.settings, settingsDefaults({ generation: { steps: 39, cfg: 0 } }));
    assert.deepEqual(saved.model_settings, MODEL_DEFAULTS);
});

test("node recovery preserves all edited values after reload and respects newer serialized settings", async () => {
    const raw = JSON.stringify(state()), drafts = new Map();
    const h = harness(raw, { drafts }); await h.ready();
    const generation = h.widget.state.settings.generation;
    Object.assign(generation, { diffusion_model_name: "Custom.safetensors", steps: 37, cfg: 0, seed: 153, cache_dtype: "int4" });
    h.widget.qi2.options.onChange();
    const provider = h.control("[data-model-provider]"); provider.value = "triposplat"; await provider.fire("change");
    const reloaded = harness(raw, { drafts }); await reloaded.ready();
    assert.deepEqual(clone(reloaded.widget.state.settings), clone(h.widget.state.settings));
    assert.deepEqual(clone(reloaded.widget.state.model_settings), clone(h.widget.state.model_settings));

    const newer = JSON.parse(h.data.value);
    newer.settings_revision++;
    newer.settings.generation.steps = 61;
    const reopened = harness(JSON.stringify(newer), { drafts }); await reopened.ready();
    assert.equal(reopened.widget.state.settings.generation.steps, 61);
});

test("CG Creator publishes the complete edited state to ComfyUI without a library save", async () => {
    const h = harness(JSON.stringify(state()));
    const notifications = [];
    let autosaved;
    h.context.app.canvas = { emitEvent({ subType }) {
        notifications.push(subType);
        if (subType !== "after-change") return;
        const info = { widgets_values: ["stale"] };
        h.node.onSerialize(info);
        autosaved = info.widgets_values[0];
    } };
    await h.ready();
    assert.deepEqual(notifications, []);
    Object.assign(h.widget.state.settings.generation, { cache_device: "cpu", cache_dtype: "int4",
        steps: 19, cfg: 0, sampler: "dpmpp_2m", scheduler: "normal", seed: 429, seed_mode: "randomize" });
    Object.assign(h.widget.state.model_settings, { pixal_mesh_resolution: 2048, pixal_texture_resolution: 512, seed: 735 });
    h.widget.qi2.options.onChange();
    assert.deepEqual(notifications, ["before-change", "after-change"]);
    assert.equal(autosaved, h.data.value);
    const configured = JSON.parse(autosaved);
    assert.deepEqual(configured.settings, clone(h.widget.state.settings));
    assert.deepEqual(configured.model_settings, clone(h.widget.state.model_settings));
    assert.equal(h.calls.some(call => call.options.method === "PATCH"), false);
    h.widget.beginGesture();
    h.widget.viewer.setCameraState({ ...camera, fov: 67 }, { emit: true });
    assert.deepEqual(notifications, ["before-change", "after-change"]);
    await h.window.fire("beforeunload");
    assert.deepEqual(notifications, ["before-change", "after-change", "before-change", "after-change"]);
    const reloaded = harness(autosaved, { drafts: h.drafts });
    reloaded.documents.set(A, h.documents.get(A)); await reloaded.ready();
    assert.deepEqual(clone(reloaded.widget.state.settings), configured.settings);
    assert.deepEqual(clone(reloaded.widget.state.model_settings), configured.model_settings);
    assert.equal(reloaded.widget.state.camera.fov, 67);
});

test("opening a saved TripoSplat model keeps Pixal3D selected for generation", async () => {
    for (const firstBackground of [false, true]) {
        const h = firstBackground ? harness() : harness(JSON.stringify(state()));
        const triposplat = background(B, {
            model_settings: { ...MODEL_DEFAULTS, provider: "triposplat", pixal_profile: "standard" },
            models: [{ model_id: MODEL, source_image_id: IMAGE, settings: firstBackground ? {} : { provider: "triposplat" } }],
        });
        h.documents.set(B, triposplat);
        await h.ready();
        const select = h.control("[data-library]"); select.value = B;
        const switching = h.widget.change({ target: select }); await settle();
        await h.widget.sourceLoad.onload(); await switching; await settle();
        assert.equal(h.widget.state.model_settings.provider, "pixal3d");
        assert.equal(h.widget.state.model_settings.pixal_profile, "fast");
        assert.equal(h.widget.state.model_id, MODEL);
        assert.equal(h.widget.scene.model_id, MODEL);
        const selectedCG = firstBackground ? CG : "";
        assert.equal(h.widget.state.selected_cg_id, selectedCG);
        assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds/${B}/models/${MODEL}`), true);

        const reloaded = harness(h.data.serializeValue());
        reloaded.documents.set(B, triposplat);
        await reloaded.ready();
        assert.equal(reloaded.widget.state.model_id, MODEL);
        assert.equal(reloaded.widget.state.selected_cg_id, selectedCG);
        await reloaded.control("[data-cg-id]").click();
        assert.equal(reloaded.control("[data-large-preview]").src, triposplat.cgs[0].url);
        assert.equal(reloaded.widget.state.model_id, MODEL);
    }
});

test("saved preview, collection and generation references are bounded without changing legacy workflows", () => {
    const preview = { kind: "cg", cg_id: CG, zoom: 2.4, center: [.4, .6] };
    assert.deepEqual(readState(JSON.stringify(state({ preview }))).preview, preview);
    for (const changes of [{ preview: { ...preview, zoom: 9 } }, { preview: { ...preview, center: [0] } },
        { preview: { ...preview, cg_id: "../file" } }, { scroll: { sources: -1, cgs: 0 } },
        { job: { id: "../job", action: "model", name: "" } }, { job: { request_id: B, action: "unknown", name: "" } }]) {
        assert.throws(() => readState(JSON.stringify(state(changes))), /saved CG/);
    }
    assert.equal(readState(JSON.stringify(state())).preview, undefined);
});

test("serialized preview mode and scroll survive a fresh widget and configure cannot overwrite incoming state", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    await h.control("[data-cg-id]").click();
    h.widget.state.preview.zoom = 2; h.widget.state.preview.center = [.35, .6];
    h.control("[data-source-choices]").scrollLeft = 240; h.control("[data-cgs]").scrollTop = 130;
    const info = { widgets_values: ["stale"] }; h.node.onSerialize(info);
    const reloaded = harness(info.widgets_values[0]); await reloaded.ready();
    assert.equal(reloaded.control("[data-large-preview]").src, background().cgs[0].url);
    assert.deepEqual(clone(reloaded.widget.state.preview), { kind: "cg", cg_id: CG, zoom: 2, center: [.35, .6] });
    assert.equal(reloaded.control("[data-source-choices]").scrollLeft, 240); assert.equal(reloaded.control("[data-cgs]").scrollTop, 130);
    const incoming = JSON.stringify(state({ background_id: B, preview: { kind: "source", zoom: 1.5, center: [.4, .3] } }));
    h.data.value = incoming; h.node.onConfigure(); assert.equal(h.data.value, incoming); await h.ready();
    assert.match(h.control("[data-large-preview]").src, new RegExp(B));
    assert.equal(h.widget.state.preview.kind, "source");
    h.widget.cgPreviewClose(); const closed = harness(h.data.serializeValue()); await closed.ready();
    assert.equal(closed.control("[data-large-preview]"), null); assert.equal(closed.widget.canEditCamera(), true);
});

test("gallery thumbnails load on intersection and retain their dimensions through updates", async () => {
    const h = harness(JSON.stringify(state()));
    h.context.IntersectionObserver = class {
        constructor(callback) { this.callback = callback; this.targets = new Set(); }
        observe(target) { this.targets.add(target); }
        unobserve(target) { this.targets.delete(target); }
        disconnect() { this.targets.clear(); }
    };
    await h.ready();
    const gallery = h.control("[data-cgs]"), card = h.control("[data-cg-id]"), image = card.querySelector("img");
    const observer = h.widget.galleryObserver;
    assert.equal(observer.targets.has(image), true);
    observer.callback([{ target: image, isIntersecting: false }], observer);
    assert.equal(image.src, undefined);
    observer.callback([{ target: image, isIntersecting: true }], observer);
    assert.equal(image.src, h.widget.document.cgs[0].url);
    assert.equal(observer.targets.has(image), false);
    assert.equal(image.style.aspectRatio, "1600 / 896"); gallery.scrollTop = 130;
    await h.control("[data-enlarge]").click(); h.widget.render();
    assert.equal(h.control("[data-cg-id]"), card); assert.equal(card.querySelector("img"), image); assert.equal(gallery.scrollTop, 130);
    h.widget.cgPreviewClose(); h.widget.state.selected_cg_id = ""; h.widget.render();
    assert.equal(h.control("[data-cg-id]"), card); assert.equal(card.classList.contains("is-selected"), false);
});

test("reload resumes acknowledged and unacknowledged jobs without another generation request", async () => {
    for (const action of ["model", "cg"]) for (const acknowledged of [false, true]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        const submission = deferred(), completion = deferred(), polled = deferred();
        h.api.handle = path => path.endsWith("/jobs") ? submission.promise : path === `${BASE}/jobs/${B}` ? (polled.resolve(), completion.promise) : undefined;
        const running = action === "cg" ? generateCG(h) : h.widget.runJob(action);
        await settle();
        assert.match(h.widget.state.job.request_id, /^[A-Za-z0-9_-]{1,160}$/);
        if (acknowledged) { submission.resolve(response({ job_id: B })); await polled.promise; }
        const saved = h.data.serializeValue(); assert.doesNotMatch(saved, /blob:/);
        h.node.onRemoved(); submission.resolve(response({ job_id: B })); completion.resolve(response({ status: "cancelled" })); await running;
        const reloaded = harness(saved), resumed = deferred(), resumedPoll = deferred();
        reloaded.api.handle = path => {
            if (path.includes(`/backgrounds/${A}/jobs/`)) return response({ job_id: B });
            if (path === `${BASE}/jobs/${B}`) { resumedPoll.resolve(); return resumed.promise; }
        };
        await reloaded.ready(); await resumedPoll.promise;
        assert.equal(reloaded.widget.busy, true); assert.equal(reloaded.control('[data-action="model"]').disabled, true);
        assert.throws(() => reloaded.data.beforeQueued(), /operation/);
        assert.equal(reloaded.calls.some(call => call.options.method === "POST"), false);
        if (action === "cg") { assert.ok(reloaded.control("[data-pending-cg]")); assert.ok(reloaded.widget.pendingCG.url.startsWith("blob:")); }
        resumed.resolve(response({ status: "completed", result: { document: background() } })); await reloaded.widget.jobOperation;
        assert.equal(reloaded.widget.busy, false); assert.equal(reloaded.widget.state.job, undefined);
        assert.equal(reloaded.widget.state.selected_cg_id, action === "cg" ? CG : "");
    }
});

test("generation connection failures preserve resumable state while terminal failures release it", async () => {
    for (const status of [404, 503]) {
        const job = { id: B, action: "model", name: "" }, h = harness(JSON.stringify(state({ job })));
        h.api.handle = path => path === `${BASE}/jobs/${B}` ? response({ error: "Status unavailable" }, status) : undefined;
        await h.ready(); await h.widget.jobOperation;
        assert.equal(h.widget.busy, false); assert.equal(h.widget.state.background_id, A);
        if (status === 503) {
            assert.deepEqual(clone(h.widget.state.job), job); assert.throws(() => h.data.beforeQueued(), /operation/);
            assert.equal(h.control('[data-action="model"]').disabled, true); assert.equal(h.control("[data-refresh]"), null);
            h.api.handle = path => path === `${BASE}/jobs/${B}` ? response({ status: "cancelled" }) : undefined;
            h.emitEvent("reconnected"); await h.widget.operations; await h.widget.jobOperation;
        }
        assert.equal(h.widget.state.job, undefined);
    }
});

test("duplicating a node retains its assets and settings without attaching to the original node's generation", async () => {
    const saved = state({ job: { id: B, node_id: "7", action: "cg", name: "Original node job" }, preview: { kind: "pending", zoom: 1, center: [.5, .5] } });
    const clone = harness(JSON.stringify(saved), { nodeId: 8 }); await clone.ready();
    assert.equal(clone.widget.state.job, undefined); assert.equal(clone.widget.state.preview, undefined);
    assert.equal(clone.widget.state.selected_cg_id, CG); assert.equal(clone.widget.state.background_id, A);
    assert.equal(clone.calls.some(call => call.path.includes("/jobs/")), false);
    assert.equal(clone.widget.busy, false); assert.equal(clone.widget.canEditCamera(), true);
});

test("restoring a frame-less workflow preserves the authored aspect instead of replacing it with the source aspect", async () => {
    const h = harness(JSON.stringify(state({ selected_cg_id: "", settings: settingsDefaults({ aspect: "9:16", resolution_mp: 3 }) }))); await h.ready();
    assert.equal(h.widget.state.settings.aspect, "9:16");
    assert.deepEqual(clone(h.widget.state.frame), frameDimensions("9:16", 3));
    assert.equal(JSON.parse(h.data.value).settings.aspect, "9:16");
});

test("a retired model restores the active model for the same background variant", async () => {
    const h = harness(JSON.stringify(state())), active = "9".repeat(32);
    const document = h.documents.get(A);
    document.models[0].model_id = active; document.cgs[0].model_id = active;
    await h.ready();
    assert.equal(h.widget.state.model_id, active);
    assert.equal(h.widget.scene.model_id, active);
    assert.equal(h.widget.state.selected_cg_id, CG);
    assert.deepEqual(clone(h.widget.state.camera), camera);
    assert.equal(h.calls.some(call => call.path.endsWith(`/models/${MODEL}`)), false);
});

test("an old CG keeps the active model and recovers an empty saved viewport", async () => {
    const document = background(), retired = "9".repeat(32), activeCamera = { ...camera, fov: 61 };
    document.cgs[0].model_id = retired;
    const h = harness(JSON.stringify(state({ selected_cg_id: "", camera: activeCamera })));
    h.documents.set(A, document);
    await h.ready();
    await h.control("[data-cg-id]").click();
    assert.equal(h.control("[data-large-preview]").src, document.cgs[0].url);
    h.widget.cgPreviewClose();
    assert.equal(h.widget.canEditCamera(), true);
    assert.equal(h.widget.state.model_id, MODEL);
    assert.deepEqual(h.widget.viewer.getCameraState(), activeCamera);
    assert.equal(JSON.parse(h.data.value).selected_cg_id, CG);

    const reloaded = harness(JSON.stringify(state({ model_id: "", frame: { width: 1600, height: 896 } })));
    reloaded.documents.set(A, document);
    await reloaded.ready();
    assert.equal(reloaded.widget.canEditCamera(), true);
    assert.equal(reloaded.widget.state.model_id, MODEL);
    assert.deepEqual(reloaded.widget.viewer.getCameraState(), sourceCameraView(camera));
    assert.equal(JSON.parse(reloaded.data.value).selected_cg_id, CG);
});

test("a missing 3D model does not prevent restoring a finished CG preview or its workflow output", async () => {
    const preview = { kind: "cg", cg_id: CG, zoom: 2, center: [.4, .6] }, h = harness(JSON.stringify(state({ preview })));
    h.api.handle = path => path.includes("/models/") ? response({ error: "Model file unavailable" }, 404) : undefined;
    await h.ready();
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.restoreFailed, false);
    assert.equal(h.control("[data-large-preview]").src, background().cgs[0].url);
    assert.equal(h.widget.state.selected_cg_id, CG); assert.deepEqual(clone(h.widget.state.preview), preview);
    assert.doesNotThrow(() => h.data.beforeQueued()); assert.equal(h.control("[data-save-view]").disabled, true);
    assert.match(h.widget.status.textContent, /Model file unavailable/);
});

test("a completed CG job can reconnect even when its previous 3D model is unavailable", async () => {
    const job = { id: B, node_id: "7", action: "cg", name: "Recover finished CG" }, h = harness(JSON.stringify(state({ job })));
    h.api.handle = path => path.includes("/models/") ? response({ error: "Model file unavailable" }, 404)
        : path === `${BASE}/jobs/${B}` ? response({ status: "completed", result: { document: background() } }) : undefined;
    await h.ready(); await h.widget.jobOperation;
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.state.job, undefined);
    assert.equal(h.widget.state.selected_cg_id, CG); assert.equal(h.widget.busy, false);
    assert.doesNotThrow(() => h.data.beforeQueued()); assert.equal(h.calls.some(call => call.options.method === "POST"), false);
});

test("late model asset loading cannot overwrite a more recently configured model canvas", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const oldId = "2".repeat(32), newId = "3".repeat(32), loading = deferred(), started = deferred(), viewer = h.widget.viewer;
    h.documents.get(B).models[0].model_id = newId;
    h.api.handle = path => path.includes("/models/") ? response(scene(path.includes(`/backgrounds/${B}/`) ? B : A, { model_id: path.split('/').at(-1), source_image_id: IMAGE })) : undefined;
    viewer.setScene = async scene => {
        if (scene.model_id === oldId) { started.resolve(); await loading.promise; }
        viewer.visibleModel = scene.model_id; return { failures: [] };
    };
    h.widget.state.model_id = oldId; const oldLoad = h.widget.showModel(); await started.promise;
    h.data.value = JSON.stringify(state({ background_id: B, model_id: newId, selected_cg_id: "" })); h.node.onConfigure(); h.fireTimers(0); await settle(); await settle();
    loading.resolve(); await oldLoad; await h.widget.operations;
    assert.equal(viewer.visibleModel, newId); assert.equal(h.widget.scene.model_id, newId);
    assert.equal(h.widget.state.background_id, B); assert.equal(h.widget.restoring, false);
});

test("removing a generating node preserves its persisted pending preview mode", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.job = { id: B, action: "cg", name: "Keep pending preview" };
    h.widget.pendingCG = { ...frameDimensions("16:9", 2), camera, name: "Keep pending preview", url: URL.createObjectURL(new Blob(["png"])) };
    h.widget.openPreview(h.widget.pendingCG); const saved = h.data.value;
    h.node.onRemoved(); assert.equal(h.data.value, saved); assert.equal(JSON.parse(h.data.value).preview.kind, "pending");
});

test("an archived selected background retains its name and placeholder selection cannot mislabel the current document", async () => {
    const h = harness(JSON.stringify(state())); h.documents.set(A, background(A, { archived: true })); await h.ready();
    const select = h.control("[data-library]");
    assert.equal(select.children.some(option => option.value === A && option.textContent.includes("Evening room")), true);
    select.value = ""; await h.widget.change({ target: select });
    assert.equal(select.value, A); assert.equal(h.widget.state.background_id, A);
});

test("source-camera calibration keeps image-plane framing while saved CG cameras stay exact", async () => {
    const native = { position: [0, 0, 3.35], target: [0, 0, 0], up: [0, 1, 0], fov: 15.17 }, calibrated = sourceCameraView(native);
    assert.equal(calibrated.fov, 60); assert.equal(native.fov, 15.17);
    assert.ok(Math.abs(calibrated.position[2] * Math.tan(Math.PI / 6) - 3.35 * Math.tan(native.fov * Math.PI / 360)) < 1e-12);
    const h = harness(JSON.stringify(state({ selected_cg_id: "", camera: null }))); h.documents.set(A, background(A, { cgs: [] }));
    h.api.handle = path => path.includes(`/models/${MODEL}`) ? response(scene(A, { source_image_id: IMAGE, camera: native, objects: [{ object_id: B, locked: true, source_camera: native }] })) : undefined;
    await h.ready(); assert.deepEqual(h.widget.viewer.camera, calibrated);
    const saved = { ...native, fov: 19, position: [0.4, 0.2, 2.6] };
    h.documents.set(A, background(A, { cgs: [{ ...background().cgs[0], camera: saved }] }));
    h.data.value = JSON.stringify(state({ camera: saved })); h.node.onConfigure(); await h.ready();
    assert.deepEqual(h.widget.viewer.camera, saved);
});

test("Expand Scene explains why an older model cannot be expanded before capturing or submitting", async () => {
    const h = harness(JSON.stringify(state({ selected_cg_id: "" })));
    h.api.handle = path => path.includes(`/models/${MODEL}`) ? response(scene(A, {
        background_settings: { provider: "worldmirror" },
    })) : undefined;
    await h.ready();
    const button = h.control("[data-expand-scene]");
    assert.equal(button.disabled, false);
    await button.click();
    assert.match(h.widget.status.textContent, /no reconstruction frames.*Create Model/);
    assert.equal(h.widget.viewer.captureOptions, undefined);
    assert.equal(h.calls.some(call => call.path.endsWith("/jobs")), false);
});

test("Expand Scene compares the captured and generated frames before enabling a rebuild", async () => {
    const h = harness(JSON.stringify(state({ selected_cg_id: "" })));
    const frameId = "1".repeat(32), buildId = "2".repeat(32);
    const modelURL = `${BASE}/backgrounds/${A}/models/${MODEL}`;
    const base = scene(A, { source_image_id: IMAGE, can_expand: true, reconstruction_size: [1600, 896],
        background_settings: { provider: "worldmirror" } });
    const builds = [];
    h.api.handle = (path, options) => {
        if (path === modelURL) return response(builds.length ? { ...base, expansion_id: buildId } : base);
        if (path === modelURL + "/expansion/" + buildId) return response({ ...base, expansion_id: buildId });
        if (path.endsWith("/jobs")) {
            if (options.body instanceof FormData) {
                assert.deepEqual(JSON.parse(options.body.get("data")).model_camera, h.widget.viewer.getCameraState());
                return response({ job_id: frameId });
            }
            builds.push(JSON.parse(options.body));
            return response({ job_id: buildId });
        }
        if (path === `${BASE}/jobs/${frameId}`) return response({ status: "completed", result: { expansion_frame_id: frameId } });
        if (path === `${BASE}/jobs/${buildId}`) return response({ status: "completed", result: { expansion_id: buildId } });
    };
    await h.ready();
    await h.control("[data-expand-scene]").click();
    const compare = h.control("[data-expansion-compare]"), before = h.control("[data-expansion-before]");
    const after = h.control("[data-expansion-after]"), approve = h.control("[data-approve-expansion-frame]");
    assert.equal(compare.hidden, false);
    assert.equal(h.widget.host.inert, true);
    assert.equal(h.control("[data-fov]").disabled, true);
    assert.equal(approve.disabled, true);
    assert.equal(builds.length, 0);
    assert.equal(before.src, modelURL + "/expansion/" + frameId + "/frame/before");
    assert.equal(after.src, modelURL + "/expansion/" + frameId + "/frame/after");
    const slider = h.control("[data-expansion-split]");
    before.getBoundingClientRect = () => ({ left: 100, width: 200 });
    await compare.fire("pointermove", { pointerType: "mouse", buttons: 0, clientX: 140 });
    assert.equal(compare.style["--split"], "20%");
    assert.equal(slider.value, "20");
    await compare.fire("pointermove", { pointerType: "mouse", buttons: 0, clientX: 350 });
    assert.equal(compare.style["--split"], "100%");
    slider.value = "73";
    await slider.fire("input");
    assert.equal(compare.style["--split"], "73%");
    for (const image of [before, after]) {
        Object.assign(image, { complete: true, naturalWidth: 1600 });
        await image.fire("load");
    }
    assert.equal(approve.disabled, false);
    await approve.click();
    assert.equal(builds[0].expansion_frame_id, frameId);
    assert.equal(compare.hidden, true);
    assert.equal(h.control("[data-expansion-actions]").hidden, false);
    assert.equal(h.widget.host.inert, false);
});

test("WorldMirror restores the predicted up direction and recenters to the predicted camera", async () => {
    const predicted = { position: [1, 2, 3], target: [2, 2, 3], up: [0, -1, 0], fov: 47 };
    const h = harness(JSON.stringify(state({ selected_cg_id: "", camera: null, model_settings: { provider: "worldmirror" } })));
    h.documents.set(A, background(A, { cgs: [] }));
    h.api.handle = path => path.includes(`/models/${MODEL}`) ? response(scene(A, {
        source_image_id: IMAGE, camera: predicted, background_settings: { provider: "worldmirror" },
        objects: [{ object_id: B, locked: true, source_camera: predicted }],
    })) : undefined;
    await h.ready();
    assert.deepEqual(h.widget.viewer.camera, predicted);
    const moved = { ...predicted, target: [2, 2.2, 3], fov: 60 };
    h.data.value = JSON.stringify(state({ selected_cg_id: "", camera: { ...moved, up: [0, 1, 0] } }));
    h.node.onConfigure(); await h.ready();
    assert.deepEqual(h.widget.viewer.camera, moved);
    h.widget.viewer.camera = { ...predicted, fov: 60 };
    await h.control("[data-recenter]").click();
    assert.deepEqual(h.widget.viewer.camera, predicted);
    assert.deepEqual(JSON.parse(h.data.value).camera, predicted);
});

test("WorldMirror camera translation follows splat scale and preserves the viewing direction", async () => {
    const source = { position: [0, 0, 0], target: [0, 0, 1], up: [0, -1, 0], fov: 47 };
    const h = harness(JSON.stringify(state({ selected_cg_id: "", camera: null })));
    h.api.handle = path => path.includes(`/models/${MODEL}`) ? response(scene(A, {
        source_image_id: IMAGE, camera: source, background_settings: { provider: "worldmirror" },
        objects: [{ object_id: B, locked: true, source_camera: source }],
    })) : undefined;
    await h.ready();
    for (const radius of [0.5, 100]) {
        h.widget.viewer.setCameraState(source);
        h.widget.viewer.objects = new Map([[B, { localBounds: {
            isEmpty: () => false, min: { x: -radius, y: 0, z: 0 }, max: { x: radius, y: 1, z: radius },
        } }]]);
        await h.widget.host.fire("wheel");
        assert.equal(h.widget.viewer.controls.panSpeed, radius);
        h.widget.viewer.dollyCamera(0.1, { emit: true });
        const movement = radius * 0.01;
        const moved = h.widget.viewer.getCameraState();
        assert.ok(Math.abs(moved.position[2] - movement) < 1e-10);
        assert.deepEqual(moved, { ...source, position: [0, 0, moved.position[2]], target: [0, 0, 1 + moved.position[2]] });
    }
});

test("MoGe restores and recenters its source camera in the imported mesh coordinates", async () => {
    const source = { position: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov: 47 };
    const imported = { ...source, position: [1, 2, 3], target: [1, 2, 2.5] };
    const h = harness(JSON.stringify(state({ selected_cg_id: "", camera: source, model_settings: { provider: "moge" } })));
    h.context.Factory3DViewer.prototype.getModelSourceCamera = objectId => {
        assert.equal(objectId, B);
        return imported;
    };
    h.documents.set(A, background(A, { cgs: [] }));
    h.api.handle = path => path.includes(`/models/${MODEL}`) ? response(scene(A, {
        source_image_id: IMAGE, camera: source, background_settings: { provider: "moge" },
        objects: [{ object_id: B, locked: true, asset_kind: "mesh", source_camera: source }],
    })) : undefined;
    await h.ready();
    assert.equal(h.control("[data-model-provider]").value, "moge");
    assert.equal(h.control("[data-model-settings]").disabled, false);
    assert.deepEqual(h.widget.viewer.camera, imported);
    assert.deepEqual(JSON.parse(h.data.value).camera, imported);
    await h.widget.captureScene({ width: 128, height: 64, cameraState: imported });
    assert.equal(h.widget.viewer.captureOptions.cameraState, undefined);
    const otherView = { ...imported, fov: 53 };
    await h.widget.captureScene({ width: 128, height: 64, cameraState: otherView });
    assert.equal(h.widget.viewer.captureOptions.cameraState, otherView);
    h.widget.viewer.objects = new Map([[B, { localBounds: {
        isEmpty: () => false, min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 1, z: 1 },
    } }]]);
    await h.widget.host.fire("pointerdown");
    assert.equal(h.widget.viewer.controls.panSpeed, 2);
    assert.equal(h.widget.viewer.zoomSensitivity, 0.2);
    assert.equal(h.widget.viewer.options.cameraDragScale, 1.4);
    assert.deepEqual(h.widget.viewer.camera, imported);
    const authored = { ...imported, fov: 60 };
    h.data.value = JSON.stringify({ ...JSON.parse(h.data.value), camera: authored });
    h.node.onConfigure(); await h.ready();
    assert.deepEqual(h.widget.viewer.camera, authored);
    await h.control("[data-recenter]").click();
    assert.deepEqual(JSON.parse(h.data.value).camera, imported);
});

test("MoGe and WorldMirror share the saved Refine views choice", async () => {
    const h = harness(JSON.stringify(state({ model_settings: { provider: "moge" } })));
    await h.ready();
    const choice = h.control("[data-moge-refine-views]"), label = h.control("[data-moge-refine-label]");
    assert.equal(label.hidden, false);
    assert.equal(choice.checked, false);
    choice.checked = true; await choice.fire("change");
    assert.equal(JSON.parse(h.data.value).model_settings.moge_refine_views, true);
    assert.equal(h.control("[data-model-settings]").disabled, true);
    const provider = h.control("[data-model-provider]");
    provider.value = "worldmirror"; await provider.fire("change");
    assert.equal(label.hidden, false);
    assert.equal(choice.checked, true);
    assert.equal(h.control("[data-model-settings]").disabled, false);
    let keys, jobAction;
    h.widget.dialog = async (_title, fields) => {
        keys = Array.from(fields, field => field.key);
        return null;
    };
    await h.control("[data-model-settings]").click();
    assert.deepEqual(keys, ["worldmirror_target_size", "worldmirror_apply_sky_mask"]);
    h.widget.dialog = async () => true;
    h.widget.runJob = async action => { jobAction = action; };
    await h.control('[data-action="model"]').click();
    assert.equal(jobAction, "model");
    provider.value = "pixal3d"; await provider.fire("change");
    assert.equal(label.hidden, true);
    provider.value = "moge"; await provider.fire("change");
    assert.equal(choice.checked, true);
});

test("frame dimensions support portrait and landscape within renderer limits", () => {
    for (const [aspect, ratio] of [["16:9", 16 / 9], ["9:16", 9 / 16], ["4:3", 4 / 3], ["1:1", 1]]) {
        const size = frameDimensions(aspect, 2);
        assert.equal(size.width % 16, 0); assert.equal(size.height % 16, 0);
        assert.ok(Math.abs(size.width / size.height - ratio) < 0.03);
        assert.ok(size.width >= 64 && size.height >= 64 && size.width <= 4096 && size.height <= 4096);
    }
    assert.throws(() => frameDimensions("0:9", 2)); assert.throws(() => frameDimensions("16:9", 0));
});

test("standalone CG entry registers its new node and exposes only the model workspace", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    class OldNode { onNodeCreated() {} }
    const original = OldNode.prototype.onNodeCreated;
    h.extension.beforeRegisterNodeDef(OldNode, { name: "VNCCS_BackgroundCreator" });
    assert.equal(OldNode.prototype.onNodeCreated, original);
    assert.equal(h.node.cgCreator, h.widget); assert.equal(h.node.cgCreatorDOM.options.serialize, false);
    assert.equal(h.node.cgCreatorDOM.name, "cg_ui"); assert.equal(h.data.hidden, true);
    assert.equal(h.node.size[0], LAYOUT_WIDTH);
    assert.equal(h.control("footer"), null); assert.equal(h.control(".bg-status"), null); assert.equal(h.control("progress"), null); assert.equal(h.control("[data-stop]"), null);
    assert.equal(h.control("[data-step]"), null); assert.equal(h.control("[data-upload]"), null);
    assert.equal(h.control("[data-wizard]"), null); assert.equal(h.control('[data-action="background"]'), null);
    assert.ok(h.control("[data-source-choices]")); assert.equal(h.control("[data-image]"), null); assert.ok(h.control(".bg-generation-dock")); assert.ok(h.control('[data-action="model"]')); assert.ok(h.control("[data-qi2]")); assert.ok(h.control("[data-model-settings]")); assert.equal(h.control("[data-settings]"), null); assert.equal(h.control("[data-model-setting]"), null);
    assert.ok(h.widget.container.classList.contains("vnccs-cg-creator")); assert.ok(h.calls.every(call => call.path.startsWith(BASE + "/")));
    assert.ok(h.widget.container.classList.contains("vnccs-common-ui-host"), "Shared motion copies stay positioned over their controls");
    assert.equal(h.control("[data-cg-id]").children.length, 2, "CG cards contain only the name and preview, with no FOV row");
    assert.deepEqual(h.queued, []);
});

test("TripoSplat settings use the copied VNCCS modal while QI2 controls stay inline", async () => {
    const h = harness(JSON.stringify(state({ model_settings: { ...MODEL_DEFAULTS, provider: "triposplat" } }))); await h.ready();
    const opening = h.control("[data-model-settings]").click(); await settle();
    const dialog = h.control("[role=dialog]"); assert.ok(dialog, h.widget.status.textContent);
    assert.equal(dialog.querySelector("select").type, "select-one");
    assert.equal(h.control("[data-prompt]"), null);
    dialog.querySelectorAll("button")[0].onclick(); await opening;
    assert.equal(h.control("[role=dialog]"), null); assert.ok(h.widget.qi2.config);
});

test("construction leaves configured bytes untouched until restoration completes", async () => {
    const h = harness(), original = h.data.value; assert.equal(h.widget.state, null); assert.equal(h.data.value, original);
    const savedCamera = { ...camera, fov: 61, position: [1, 2, 3] };
    h.data.value = JSON.stringify(state({ camera: savedCamera })); h.node.onConfigure(); await h.ready();
    assert.equal(h.node.created, true); assert.equal(h.node.configured, true);
    assert.deepEqual(h.widget.viewer.camera, savedCamera); assert.equal(h.widget.canEditCamera(), true);
    assert.equal(h.widget.viewer.scene.objects[0].locked, true); assert.equal(JSON.parse(h.data.value).camera.fov, 61);
});

test("FOV updates both fields, camera and serialized state throughout a single undo gesture", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    for (const value of [35, 55]) {
        h.widget.fov.value = String(value); h.widget.fov.fire("input");
        assert.equal(h.widget.viewer.camera.fov, value); assert.equal(h.widget.fovNumber.value, String(value));
        assert.equal(JSON.parse(h.data.value).camera.fov, value); assert.equal(h.widget.history.undoStack.length, 0);
    }
    h.widget.fovNumber.value = "65"; h.widget.fovNumber.fire("input"); assert.equal(h.widget.fov.value, "65");
    await h.widget.fov.fire("change"); assert.equal(h.widget.history.undoStack.length, 1);
    await h.widget.container.fire("keydown", { key: "z", ctrlKey: true }); assert.equal(h.widget.viewer.camera.fov, 42);
    await h.widget.container.fire("keydown", { key: "z", ctrlKey: true, shiftKey: true }); assert.equal(h.widget.viewer.camera.fov, 65);
    await h.control("[data-recenter]").click(); assert.deepEqual(h.widget.viewer.camera, sourceCameraView(camera));
});

test("serialization replaces collected payload and queueing rejects active generation", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); h.widget.viewer.camera = { ...camera, fov: 80 };
    const info = { widgets_values: ["stale"] }; h.node.onSerialize(info); assert.equal(JSON.parse(info.widgets_values[0]).camera.fov, 80);
    h.widget.viewer.camera = { ...camera, fov: 90 }; h.data.beforeQueued(); assert.equal(JSON.parse(h.data.serializeValue()).camera.fov, 90);
    h.widget.busy = true; assert.throws(() => h.data.beforeQueued(), /running|busy|generation|operation/i);
});

test("malformed or failed restoration retains original state and blocks workflow execution", async () => {
    for (const raw of ["{broken", '{"schema_version":2}', JSON.stringify(state({ camera: {} }))]) {
        const h = harness(raw); await h.ready(); assert.equal(h.data.value, raw); assert.equal(h.widget.restoring, true);
        assert.equal(h.data.serializeValue(), raw); assert.throws(() => h.data.beforeQueued(), /restoring/i);
    }
    const raw = JSON.stringify(state()), h = harness(raw);
    h.api.handle = path => path === `${BASE}/backgrounds/${A}` ? response({ error: "Damaged document" }, 409) : undefined;
    await h.ready(); assert.equal(h.data.value, raw); assert.equal(h.widget.restoring, true);
    assert.match(h.widget.status.textContent, /Damaged document/); assert.throws(() => h.data.beforeQueued(), /restoring/i);
});

test("late restoration cannot replace a newer configured scene or camera", async () => {
    const h = harness(JSON.stringify(state())), pending = deferred(); let intercepted = false;
    h.api.handle = path => { if (path === `${BASE}/config` && !intercepted) { intercepted = true; return pending.promise; } };
    h.fireTimers(0); await settle(); assert.equal(intercepted, true);
    const savedCamera = { ...camera, fov: 72 };
    h.data.value = JSON.stringify(state({ background_id: B, camera: savedCamera })); h.node.onConfigure(); h.fireTimers(0);
    pending.resolve(response(configuration())); await h.widget.operations;
    assert.equal(h.widget.document.background_id, B); assert.equal(h.widget.scene.background_id, B);
    assert.deepEqual(h.widget.viewer.camera, savedCamera); assert.equal(JSON.parse(h.data.value).background_id, B);
});

test("library navigation publishes the new image before the model finishes loading", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "Preserve during navigation"; h.widget.markEdited();
    const load = deferred(), model = deferred();
    h.api.handle = path => {
        if (path === `${BASE}/backgrounds/${B}`) return load.promise;
        if (path === `${BASE}/backgrounds/${B}/models/${MODEL}`) return model.promise;
    };
    const library = h.control("[data-library]"); library.value = B;
    const navigation = h.widget.change({ target: library });
    const assertLocked = async () => {
        assert.equal(h.widget.busy, true); assert.equal(h.control('[data-action="model"]').disabled, true);
        assert.equal(h.control("[data-save-view]").disabled, true);
        await h.control('[data-action="model"]').click();
        assert.equal(h.calls.some(call => call.path.endsWith("/jobs")), false);
        assert.throws(() => h.data.beforeQueued(), /operation|busy|generation/i);
    };
    await assertLocked(); await settle();
    assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds/${B}`), true); await assertLocked();
    load.resolve(response(background(B))); await settle();
    await assertLocked();
    assert.match(h.control("[data-model-source]").src, new RegExp(A));
    assert.equal(h.widget.viewer.captureOptions, undefined);
    assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds/${B}/models/${MODEL}`), false);
    await h.widget.sourceLoad.onload(); await navigation;
    assert.match(h.control("[data-model-source]").src, new RegExp(B));
    assert.equal(h.widget.busy, false); assert.equal(h.widget.document.background_id, B);
    assert.equal(h.control("[data-library]").disabled, false);
    assert.equal(h.control("[data-save-view]").disabled, true);
    assert.equal(h.widget.host.hidden, true);
    assert.equal(h.widget.loadingModel, true);
    assert.equal(h.control("[data-viewer-empty]").textContent, "Loading the saved model…");
    assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds/${B}/models/${MODEL}`), true);
    model.resolve(response(scene(B))); await settle();
    assert.equal(h.widget.scene.background_id, B);
    assert.equal(h.widget.host.hidden, false);
    assert.equal(h.control("[data-save-view]").disabled, false);
});

test("a rapid switch back cannot publish the scene that was loading", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const opening = deferred(), viewer = h.widget.viewer, setScene = viewer.setScene;
    viewer.setScene = async value => {
        const result = await setScene.call(viewer, value);
        if (value.background_id === B) await opening.promise;
        return result;
    };
    const library = h.control("[data-library]");
    const select = async id => {
        library.value = id;
        const navigation = h.widget.change({ target: library }); await settle();
        await h.widget.sourceLoad.onload(); await navigation; await settle();
    };
    await select(B);
    assert.equal(viewer.scene.background_id, B);
    await select(A);
    assert.match(h.control("[data-model-source]").src, new RegExp(A));
    assert.equal(h.widget.host.hidden, true);
    opening.resolve(); await settle();
    assert.equal(viewer.scene.background_id, A);
    assert.equal(h.widget.host.hidden, false);
});

test("a failed background image restores the previous document, camera and saved state", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const original = h.data.value, create = h.context.document.createElement;
    h.context.document.createElement = tag => {
        const element = create(tag);
        if (tag === "img") element.decode = () => element.src?.includes(B) ? Promise.reject(new Error("Missing asset")) : Promise.resolve();
        return element;
    };
    const library = h.control("[data-library]"); library.value = B;
    const navigation = h.widget.change({ target: library }); await settle(); await settle();
    await h.widget.sourceLoad.onload(); await navigation;
    assert.equal(h.widget.document.background_id, A); assert.equal(h.widget.scene.background_id, A);
    assert.deepEqual(h.widget.viewer.camera, camera); assert.equal(h.data.value, original);
    assert.equal(h.control("[data-library]").value, A);
    assert.equal(h.widget.busy, false); assert.match(h.widget.status.textContent, /image could not be loaded/);
});

test("reconfiguration and removal cancel background transitions without publishing late assets", async () => {
    for (const remove of [false, true]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        const library = h.control("[data-library]"); library.value = B;
        const navigation = h.widget.change({ target: library }); await settle(); await settle();
        const incoming = h.widget.sourceLoad, late = incoming.onload;
        if (remove) h.node.onRemoved();
        else { h.data.value = JSON.stringify(state()); h.node.onConfigure(); }
        await navigation; await late();
        assert.equal(incoming.removed, true);
        if (remove) assert.equal(h.widget.disposed, true);
        else { await h.ready(); assert.equal(h.widget.state.background_id, A); assert.equal(h.widget.scene.background_id, A); }
    }
});

test("latest edits stay in the node and recovery if the library is unavailable", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.api.handle = () => response({ error: "Library unavailable" }, 503);
    h.widget.state.settings.prompt = "First instruction"; h.widget.markEdited();
    h.widget.state.settings.prompt = "Latest instruction"; h.widget.markEdited();
    const panel = h.control(".bg-panel"), gallery = h.control(".bg-cg-list"); panel.scrollTop = 180; gallery.scrollTop = 95;
    h.widget.render(); assert.equal(panel.scrollTop, 180); assert.equal(gallery.scrollTop, 95);
    await h.window.fire("focus");
    assert.equal(JSON.parse(h.data.value).settings.prompt, "Latest instruction");
    assert.equal(JSON.parse(h.drafts.get(h.widget.draftKey())).settings.prompt, "Latest instruction");
    assert.equal(h.calls.some(call => call.options.method === "PATCH"), false);
});

test("nodes editing the same scene retain independent recovery drafts across saves and reloads", async () => {
    const drafts = new Map(), first = harness(JSON.stringify(state()), { nodeId: 7, drafts }), second = harness(JSON.stringify(state()), { nodeId: 8, drafts });
    await first.ready(); await second.ready();
    first.widget.state.settings.prompt = "First node instruction"; first.widget.markEdited();
    second.widget.state.settings.prompt = "Second node unsaved instruction"; second.widget.markEdited();
    const firstKey = first.widget.draftKey(), secondKey = second.widget.draftKey(), secondBytes = drafts.get(secondKey);
    assert.notEqual(firstKey, secondKey);
    assert.equal(drafts.has(firstKey), true); assert.equal(drafts.get(secondKey), secondBytes);
    const reloaded = harness(JSON.stringify(state()), { nodeId: 8, drafts }); await reloaded.ready();
    assert.equal(reloaded.widget.draftKey(), secondKey); assert.equal(reloaded.widget.state.settings.prompt, "Second node unsaved instruction");
    const unassignedFirst = harness(JSON.stringify(state()), { nodeId: -1, drafts }), unassignedSecond = harness(JSON.stringify(state()), { nodeId: -1, drafts });
    await unassignedFirst.ready(); await unassignedSecond.ready();
    assert.notEqual(unassignedFirst.widget.draftKey(), unassignedSecond.widget.draftKey());
});

test("separate workflows and tabs do not inherit the same node's recovery draft", async () => {
    const drafts = new Map(), first = harness(JSON.stringify(state()), { drafts, scope: "tab1:workflow1" });
    await first.ready(); first.widget.state.settings.prompt = "Private draft"; first.widget.markEdited();
    for (const scope of ["tab2:workflow1", "tab1:workflow2"]) {
        const other = harness(JSON.stringify(state()), { drafts, scope }); await other.ready();
        assert.notEqual(other.widget.draftKey(), first.widget.draftKey());
        assert.notEqual(other.widget.state.settings.prompt, "Private draft");
    }
});

test("source previews retain the displayed frame until the replacement loads and ignore stale loads", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const image = h.control("[data-model-source]"), placeholder = h.control("[data-empty]");
    const original = image.src;
    assert.equal(image.hidden, false);
    h.widget.document.images[0].url += "?revision=2"; h.widget.render();
    const obsolete = h.control("[data-source-incoming]"), oldLoad = obsolete.onload;
    assert.equal(image.hidden, false); assert.equal(image.src, original); assert.equal(placeholder.hidden, true);
    h.widget.document.images[0].url += "&revision=3"; h.widget.render();
    await oldLoad(); assert.equal(image.src, original); assert.equal(obsolete.onload, null);
    h.control("[data-source-incoming]").onerror();
    assert.equal(image.hidden, false); assert.equal(image.src, original); assert.equal(placeholder.hidden, true);
    assert.match(h.widget.status.textContent, /Background image unavailable/);
    h.widget.render(); await h.control("[data-source-incoming]").onload();
    const replacement = h.control("[data-model-source]");
    assert.equal(replacement.hidden, false); assert.equal(placeholder.hidden, true); assert.match(replacement.src, /revision=3/);
    assert.ok(replacement.parentElement.children.indexOf(replacement) < replacement.parentElement.children.indexOf(h.control("[data-enlarge]")), "The image stays behind the accessible preview button");
    assert.equal(h.widget.status.classList.contains("error"), false);
    assert.equal(image.removed, true); assert.equal(h.control("[data-source-incoming]"), null);
    h.widget.document.images[0].url += "&revision=4"; h.widget.render();
    const pending = h.control("[data-source-incoming]"), lateLoad = pending.onload;
    h.widget.dispose(); await lateLoad(); assert.equal(pending.onload, null); assert.equal(pending.onerror, null); assert.equal(pending.removed, true);
});

test("an initial source load exposes loading/errors and configure cancels its obsolete response", async () => {
    const h = harness(JSON.stringify(state())); await h.ready({ loadSource: false });
    assert.equal(h.control("[data-model-source]").hidden, true); assert.equal(h.control("[data-empty]").hidden, false);
    h.control("[data-source-incoming]").onerror();
    assert.equal(h.control("[data-model-source]").hidden, true);
    assert.match(h.control("[data-empty]").querySelector(".vnccs-common-creator-placeholder-title").textContent, /unavailable/);
    h.widget.render(); const obsolete = h.control("[data-source-incoming]"), onload = obsolete.onload;
    h.node.onConfigure(); await h.ready({ loadSource: false }); await onload();
    assert.equal(obsolete.onload, null); assert.equal(h.control("[data-model-source]").hidden, true);
    await h.control("[data-source-incoming]").onload(); assert.equal(h.control("[data-model-source]").hidden, false);
});

test("legacy browser drafts recover node edits without rewriting the background", async () => {
    for (const revision of [undefined, 1]) {
        const h = harness(JSON.stringify(state()));
        const key = h.widget.draftKey(A);
        const draft = JSON.stringify({ settings: settingsDefaults({ prompt: "Older authored instruction" }),
            model_settings: { ...MODEL_DEFAULTS, provider: "triposplat", steps: 35 }, revision });
        h.drafts.set(key, draft); await h.ready();
        assert.equal(h.widget.state.settings.prompt, "Older authored instruction");
        assert.equal(h.widget.state.model_settings.steps, 35);
        assert.equal(h.widget.state.model_settings.provider, "triposplat");
        assert.equal(h.drafts.get(key), draft);
        assert.equal(JSON.parse(h.data.value).settings_revision, 1);
        assert.equal(h.documents.get(A).revision, 3);
    }
});

test("failed restoration can retry without losing the original workflow bytes", async () => {
    const raw = JSON.stringify(state()), h = harness(raw);
    h.api.handle = path => path.endsWith("/backgrounds") ? response({ error: "Storage unavailable" }, 503) : undefined;
    await h.ready();
    assert.equal(h.data.value, raw); assert.equal(h.widget.restoreFailed, true);
    assert.equal(h.control("[data-refresh]"), null);
    assert.throws(() => h.data.beforeQueued(), /restoring/i);
    h.api.handle = undefined;
    h.emitEvent("reconnecting"); h.emitEvent("status", {}); await h.widget.operations;
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.restoreFailed, false);
    assert.equal(h.widget.state.background_id, A); assert.equal(h.widget.state.model_id, MODEL);
    assert.equal(JSON.parse(h.data.value).camera.fov, camera.fov);
    h.data.beforeQueued();
});

test("the shared connection watcher updates the library without a refresh button or resetting authored state", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    assert.equal(h.control('[data-refresh]'), null);
    h.widget.state.settings.generation.diffusion_model_name = 'User models/keep_selected.safetensors'; h.widget.markEdited();
    await h.control('[data-enlarge]').click();
    const preview = h.control('[data-large-preview]'), camera = clone(h.widget.viewer.camera), id = '2'.repeat(32);
    h.documents.set(id, background(id, { name: 'Newly saved background' }));
    await h.window.fire('focus');
    assert.ok(h.control('[data-library]').children.some(option => option.value === id));
    assert.equal(h.control('[data-library]').value, A); assert.equal(h.control('[data-large-preview]'), preview);
    assert.deepEqual(h.widget.viewer.camera, camera);
    assert.equal(h.widget.state.settings.generation.diffusion_model_name, 'User models/keep_selected.safetensors');
    assert.equal(h.calls.some(call => call.options.method === 'PATCH'), false);
    const beforeRemoval = h.calls.length; h.node.onRemoved();
    h.emitEvent('reconnected'); await h.window.fire('focus');
    assert.equal(h.calls.length, beforeRemoval); assert.equal(h.events.size, 0);
});

test("a background check leaves the viewport editable and skips rendering when nothing changed", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const catalog = deferred(), viewer = h.widget.viewer, frames = viewer.frames.length;
    h.api.handle = path => path === `${BASE}/backgrounds` ? catalog.promise : undefined;
    const renders = [];
    const render = h.widget.render;
    h.widget.render = () => { renders.push(true); render.call(h.widget); };
    const refresh = h.window.fire("focus");
    assert.equal(h.widget.busy, false);
    assert.equal(h.widget.host.inert, false);
    assert.equal(h.control("[data-recenter]").disabled, false);
    await h.widget.host.fire("pointerdown");
    const moved = { ...viewer.camera, target: [0.3, 0.2, 0] };
    viewer.setCameraState(moved, { emit: true });
    await h.widget.host.fire("pointerup");
    catalog.resolve(response({ backgrounds: [...h.documents.values()] }));
    await refresh;
    assert.equal(renders.length, 0);
    assert.equal(viewer.frames.length, frames + 1);
    assert.deepEqual(JSON.parse(h.data.value).camera, moved);
});

test("automatic library updates ignore active operations and obsolete responses", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const initial = h.calls.length; h.widget.busy = true; await h.window.fire('focus');
    assert.equal(h.calls.length, initial); h.widget.busy = false;
    const catalog = deferred(); let held = false;
    h.api.handle = path => {
        if (!held && path === `${BASE}/backgrounds`) { held = true; return catalog.promise; }
    };
    const update = h.window.fire('focus'); await settle();
    h.data.value = JSON.stringify(state({ background_id: B })); h.node.onConfigure(); await h.ready();
    const restored = h.data.value;
    catalog.resolve(response({ backgrounds: [] })); await update;
    assert.equal(h.widget.state.background_id, B); assert.equal(h.data.value, restored);
    assert.equal(h.widget.catalog.length, 2); assert.equal(h.widget.busy, false);
});

test("restoring an empty workflow clears the previous scene while retaining its recovery draft", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "Retain the old scene draft"; h.widget.markEdited(); const key = h.widget.draftKey();
    h.data.value = JSON.stringify(state({ background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", camera: null }));
    h.node.onConfigure(); await h.ready();
    assert.equal(h.widget.document, null); assert.equal(h.widget.state.background_id, "");;
    assert.equal(h.widget.host.hidden, true); assert.equal(h.control("[data-cgs]").children.length, 0);
    assert.equal(JSON.parse(h.drafts.get(key)).settings.prompt, "Retain the old scene draft");
    assert.equal(h.calls.some(call => call.options.method === "PATCH"), false);
});

test("read-only backup recovery disables generation while retaining the saved CG", async () => {
    const h = harness(JSON.stringify(state())); h.documents.set(A, background(A, { _recovery: { source: "backup" } })); await h.ready();
    assert.equal(h.widget.restoring, false); assert.match(h.widget.status.textContent, /read-only|read only/i);
    assert.equal(h.control('[data-action="model"]').disabled, true); assert.equal(h.control("[data-save-view]").disabled, true);
    assert.equal(h.control("[data-cgs]").children.length, 1);
    await h.window.fire('focus');
    assert.equal(h.calls.some(call => call.options.method === 'PATCH'), false);
    assert.match(h.widget.status.textContent, /read-only|read only/i);
});

test("a regenerated source clears stale model and variant selection while ready CGs retain their provenance", async () => {
    const oldImage = "4".repeat(32), oldModel = "6".repeat(32), oldCG = "7".repeat(32), setId = "8".repeat(32);
    const ref = { set_id: setId, index: 1, variant_id: "morning", name: "Morning" }, document = background();
    document.images[0].preview_revision = 3;
    document.images.push({ image_id: oldImage, width: 1024, height: 768, url: "/retired-background.png" });
    document.generated_sets = [{ set_id: setId, version: "", source_image_id: IMAGE,
        variants: [{ id: "morning", name: "Morning" }], images: [{ url: "/original-set.png" }, { url: "/morning.png", width: 1280, height: 720 }] }];
    document.models[0].source_variant = ref; document.model = document.models[0];
    Object.assign(document.cgs[0], { source_variant: ref, camera: { ...camera, fov: 73 } });
    document.cgs.push({ cg_id: oldCG, model_id: oldModel, source_image_id: oldImage, name: "Earlier source view",
        width: 1024, height: 768, camera: { ...camera, fov: 56 }, url: "/earlier-cg.png" });
    const h = harness(JSON.stringify(state({ selected_variant: ref }))); h.documents.set(A, document); await h.ready();
    assert.deepEqual(sourceChoices(document).map(choice => choice.label), ["Original", "Morning"]);
    assert.equal(h.control(`[data-source-choice="${oldImage}"]`), null);
    const savedCGs = clone(h.widget.document.cgs), replaced = clone(document);
    replaced.revision = 4; replaced.images = [{ ...document.images[0], preview_revision: 4, url: "/regenerated-source.png?v=4" }];
    replaced.models = []; replaced.model = null; replaced.generated_sets = [];
    h.documents.set(A, replaced); const requests = h.calls.length;
    await h.widget.acceptDocument(replaced, false); await h.widget.sourceLoad?.onload?.();
    assert.equal(h.widget.state.selected_image_id, IMAGE); assert.equal(h.widget.state.model_id, "");
    assert.equal(h.widget.state.selected_variant, null); assert.equal(h.widget.state.selected_cg_id, "");
    assert.deepEqual(sourceChoices(h.widget.document).map(choice => choice.label), ["Original"]);
    assert.equal(h.control("[data-model-source]").src, "/regenerated-source.png?v=4");
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [oldCG, CG]);
    assert.deepEqual(clone(h.widget.document.cgs), savedCGs);
    for (const [id, fov, url] of [[CG, 73, document.cgs[0].url], [oldCG, 56, "/earlier-cg.png"]]) {
        await h.control(`[data-cg-id="${id}"]`).click();
        assert.equal(h.widget.state.selected_cg_id, id); assert.equal(h.widget.state.model_id, "");
        assert.equal(h.widget.state.selected_image_id, IMAGE); assert.equal(h.widget.state.camera.fov, fov);
        assert.equal(h.control("[data-large-preview]").src, url);
        assert.equal(JSON.parse(h.data.value).camera.fov, fov);
    }
    assert.equal(h.calls.slice(requests).some(call => call.path.includes("/models/")), false);
    assert.deepEqual(clone(h.widget.document.cgs), savedCGs, "Opening finished CGs preserves their original model, source and camera");
});

test("image preview fills the model surface without modal padding, borders or control rows", () => {
    const panel = source.match(/\.vnccs-cg-creator \.bg-image-preview \{([^}]+)\}/)[1];
    assert.match(panel, /position:absolute;/); assert.match(panel, /inset:0;/);
    const image = source.match(/\.vnccs-cg-creator \.bg-image-viewport \{([^}]+)\}/)[1];
    assert.match(image, /position:absolute;/); assert.match(image, /inset:0;/);
    const controls = source.match(/\.bg-image-preview \[data-close-preview\] \{([^}]+)\}/)[1];
    assert.match(controls, /position:absolute;/);
});

test("CG preview opens inside the model viewport and cleans up on close, configure and removal", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const viewer = h.widget.viewer, viewport = h.control(".bg-viewport");
    const layout = { size: [...h.node.size], height: h.widget.container.style.height };
    assert.equal(h.control(".bg-camera-toolbar").hidden, false);
    await h.control("[data-cg-id]").click();
    assert.equal(h.control(".bg-camera-toolbar").hidden, false, "The toolbar keeps its layout row during image inspection");
    assert.equal(h.control(".bg-camera-toolbar").style.visibility, "hidden");
    assert.equal(h.control(".bg-camera-toolbar").style.opacity, "0"); assert.equal(h.control(".bg-camera-toolbar").inert, true);
    assert.deepEqual([...h.node.size], layout.size); assert.equal(h.widget.container.style.height, layout.height);
    assert.equal(h.widget.host.inert, true); assert.equal(h.control("[data-fov]").disabled, true);
    h.widget.fov.value = "90"; await h.widget.fov.fire("input");
    assert.equal(viewer.getCameraState().fov, camera.fov, "Image inspection cannot edit the model camera");
    const image = h.control("[data-large-preview]"), navigation = h.control("[data-preview-viewport]");
    const panel = h.control('.bg-image-preview');
    assert.equal(viewport.contains(panel), true);
    assert.equal(panel.attributes.get('role'), 'region'); assert.equal(panel.hasAttribute('aria-modal'), false);
    assert.equal(h.control('[role="dialog"]'), null, "Inline previews do not create a global modal");
    assert.equal(panel.getAnimations().length, 0, "The preview never translates or scales into the model window");
    assert.equal(image.src, background().cgs[0].url); assert.equal(image.alt, "Full-size Window view");
    assert.equal(h.widget.state.selected_cg_id, CG); assert.equal(h.widget.viewer, viewer);
    const saved = h.data.value;
    h.widget.cgPreviewClose();
    assert.equal(h.control(".bg-camera-toolbar").hidden, false); assert.equal(h.widget.host.inert, false);
    assert.equal(h.control(".bg-camera-toolbar").style.visibility, "");
    assert.equal(h.control(".bg-camera-toolbar").style.opacity, ""); assert.equal(h.control(".bg-camera-toolbar").inert, false);
    assert.deepEqual([...h.node.size], layout.size); assert.equal(h.widget.container.style.height, layout.height);
    assert.equal(h.control('.bg-image-preview'), null); assert.equal(h.widget.cgPreviewClose, null);
    assert.equal(image.onload, null); assert.equal(navigation.onpointermove, null); assert.equal(navigation.onwheel, null); assert.equal(h.widget.dialogs.size, 0);
    const closed = JSON.parse(h.data.value), inspected = JSON.parse(saved); delete inspected.preview;
    assert.deepEqual(closed, inspected, "Closing the image changes only its saved viewing mode");
    await h.control("[data-cg-id]").click();
    await h.control("[data-cg-id]").click(); assert.equal(h.widget.dialogs.size, 1);
    h.node.onConfigure(); assert.equal(h.control('.bg-image-preview'), null); await h.ready();
    assert.equal(h.control("[data-large-preview]").src, background().cgs[0].url, "Configure reopens the saved preview");
    await h.control("[data-cg-id]").click(); h.node.onRemoved();
    assert.equal(h.widget.dialogs.size, 0); assert.equal(h.widget.cgPreviewClose, null);
    assert.equal(h.window.listeners.get("blur")?.length || 0, 0);
});

test("background preview fills the model viewport and shares preview replacement and cleanup", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const saved = h.data.value, viewer = h.widget.viewer;
    await h.control("[data-enlarge]").click();
    assert.equal(h.control(".bg-camera-toolbar").hidden, false, "Hiding camera controls must preserve their row");
    assert.equal(h.control(".bg-camera-toolbar").style.visibility, "hidden", "Camera controls are visible only in model view");
    assert.equal(h.control(".bg-viewport").contains(h.control('.bg-image-preview')), true);
    assert.equal(h.control("[data-large-preview]").src, background().images[0].url);
    await h.control("[data-enlarge]").click();
    assert.equal(h.widget.dialogs.size, 1);
    await h.control("[data-cg-id]").click();
    assert.equal(h.widget.dialogs.size, 1);
    await h.control("[data-enlarge]").click(); h.widget.cgPreviewClose();
    assert.equal(h.control(".bg-camera-toolbar").hidden, false);
    assert.equal(h.control('.bg-image-preview'), null);
    assert.equal(h.widget.viewer, viewer); assert.equal(h.data.value, saved);
});

test("CG deletion waits for disk acknowledgement and preserves the camera and other cards", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const other = { ...background().cgs[0], cg_id: "0".repeat(32), name: "Keep" };
    h.widget.document.cgs.push(other); h.widget.render();
    const viewer = h.widget.viewer, camera = viewer.getCameraState(), pending = deferred();
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/cg/${CG}` && options.method === "DELETE") {
            assert.deepEqual(JSON.parse(options.body), { revision: 3 }); return pending.promise;
        }
    };
    await h.control(`[data-cg-id="${CG}"]`).click();
    const operation = h.control(`[data-delete-cg="${CG}"]`).click(); await settle();
    assert.equal(h.calls.some(call => call.options.method === "DELETE"), false, "Opening confirmation does not delete");
    const dialog = h.control('.bg-dialog-fields').parentElement;
    assert.equal(dialog.querySelector('.bg-dialog-fields').children[0].textContent, 'Delete "Window view" from the CG library?');
    dialog.querySelectorAll("button").at(-1).onclick(); await settle();
    assert.equal(h.widget.busy, true); assert.equal(h.control("[data-delete-cg]").disabled, true);
    assert.equal(h.control("[data-cgs]").children.length, 2, "Deletion is not optimistic");
    const saved = background(A, { revision: 4, cgs: [other] }); h.documents.set(A, saved);
    pending.resolve(response(saved)); await operation;
    assert.equal(h.widget.state.selected_cg_id, ""); assert.equal(JSON.parse(h.data.value).selected_cg_id, "");
    assert.equal(h.control('[role="dialog"]'), null); assert.equal(h.widget.viewer, viewer);
    assert.deepEqual(viewer.getCameraState(), camera);
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [other.cg_id]);
    h.node.onConfigure(); await h.ready();
    assert.equal(h.widget.state.selected_cg_id, ""); assert.equal(h.control("[data-cgs]").children.length, 1);
});

test("failed CG deletion retains the card and selected output", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const saved = h.data.value;
    h.api.handle = (path, options) => options.method === "DELETE" ? response({ error: "Scene changed elsewhere" }, 409) : undefined;
    const operation = h.control("[data-delete-cg]").click();
    h.control('[role="dialog"]').querySelectorAll("button").at(-1).onclick(); await operation;
    assert.equal(h.widget.busy, false); assert.equal(h.widget.state.selected_cg_id, CG);
    assert.equal(h.data.value, saved); assert.equal(h.control("[data-cgs]").children.length, 1);
    assert.match(h.widget.status.textContent, /Scene changed elsewhere/);
});

test("cancelling CG deletion preserves the library, workflow and camera", async () => {
    for (const dismiss of ["cancel", "close"]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        const saved = h.data.value, camera = h.widget.viewer.getCameraState();
        const operation = h.control("[data-delete-cg]").click();
        await h.control("[data-delete-cg]").click();
        assert.equal(h.control("[data-library]").disabled, true);
        assert.equal(h.widget.dialogs.size, 1, "Repeated clicks do not open another confirmation");
        const dialog = h.control('[role="dialog"]');
        if (dismiss === "cancel") dialog.querySelectorAll("button")[0].onclick();
        else dialog.parentElement.remove();
        await operation;
        assert.equal(h.calls.some(call => call.options.method === "DELETE"), false);
        assert.equal(h.data.value, saved); assert.equal(h.control("[data-cgs]").children.length, 1);
        assert.deepEqual(h.widget.viewer.getCameraState(), camera);
        assert.equal(h.widget.busy, false); assert.equal(h.control("[data-delete-cg]").disabled, false);
        assert.equal(h.widget.dialogs.size, 0);
    }
});

test("stale CG confirmations cannot delete after workflow restoration or removal", async () => {
    for (const lifecycle of ["configure", "remove"]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        const operation = h.control("[data-delete-cg]").click();
        const confirm = h.control('[role="dialog"]').querySelectorAll("button").at(-1);
        if (lifecycle === "configure") {
            h.data.value = JSON.stringify(state({ background_id: B })); h.node.onConfigure();
        } else h.node.onRemoved();
        confirm.onclick(); await operation;
        assert.equal(h.calls.some(call => call.options.method === "DELETE"), false);
    }
});

test("replacing a saved model requires confirmation even when a different generator is selected", async () => {
    const h = harness(JSON.stringify(state({ model_id: "", selected_cg_id: "" })));
    h.documents.get(A).models[0].settings.provider = "triposplat";
    await h.ready();
    assert.equal(h.widget.scene.model_id, MODEL);
    assert.equal(h.control("[data-viewer-empty]").hidden, true);
    assert.equal(h.widget.state.model_settings.provider, "pixal3d");
    const button = h.control('[data-action="model"]'), posts = () => h.calls.filter(call => call.options.method === "POST");
    const cancelled = button.click();
    await button.click();
    assert.equal(h.widget.dialogs.size, 1);
    assert.equal(posts().length, 0);
    const dialog = h.control('[role="dialog"]');
    assert.match(dialog.querySelector(".bg-dialog-fields").children[0].textContent, /deleted once the new model is ready/);
    dialog.querySelectorAll("button")[0].onclick(); await cancelled;
    assert.equal(posts().length, 0);
    assert.equal(h.widget.document.models[0].model_id, MODEL);

    h.api.handle = path => path === `${BASE}/backgrounds/${A}/jobs` ? response({ job_id: B })
        : path === `${BASE}/jobs/${B}` ? response({ status: "cancelled" }) : undefined;
    const confirmed = button.click();
    h.control('[role="dialog"]').querySelectorAll("button").at(-1).onclick(); await confirmed;
    assert.equal(posts().length, 1);
    assert.equal(JSON.parse(posts()[0].options.body).action, "model");
});

test("first model generation starts directly and uses the calibrated source camera", async () => {
    const h = harness(JSON.stringify(state({ model_id: "", selected_cg_id: "" })));
    const document = background(A, { models: [{ model_id: MODEL, source_image_id: B }], cgs: [] });
    document.images.push({ ...document.images[0], image_id: B });
    h.documents.set(A, document);
    await h.ready();
    const modelId = "2".repeat(32), model = { model_id: modelId, source_image_id: IMAGE };
    const completed = background(A, { revision: 4, models: [model], model });
    let payload;
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs`) { payload = JSON.parse(options.body); return response({ job_id: B }); }
        if (path === `${BASE}/jobs/${B}`) return response({ status: "completed", result: { document: completed } });
        if (path === `${BASE}/backgrounds/${A}/models/${modelId}`) return response(scene(A, { model_id: modelId, source_image_id: IMAGE }));
    };
    await h.control('[data-action="model"]').click();
    assert.equal(h.widget.dialogs.size, 0);
    assert.equal(payload.action, "model"); assert.equal(payload.image_id, IMAGE);
    assert.equal(h.widget.state.model_id, modelId); assert.equal(h.widget.viewer.camera.fov, 60);
    assert.equal(h.widget.busy, false); assert.equal(h.control("progress"), null); assert.equal(h.control("[data-stop]"), null); assert.deepEqual(h.queued, []);
});

test("generation has an animated transient CG card and no status panel, progress bar or Stop", async () => {
    for (const action of ["model", "cg"]) {
        const h = harness(JSON.stringify(state())), completion = deferred(), polled = deferred(); await h.ready();
        h.api.handle = (path, options) => {
            if (path === `${BASE}/backgrounds/${A}/jobs`) return response({ job_id: B });
            if (path === `${BASE}/jobs/${B}`) { polled.resolve(); return completion.promise; }
        };
        const savedOutput = h.widget.state.selected_cg_id;
        const running = action === "model" ? h.control('[data-action="model"]').click() : generateCG(h);
        if (action === "model") h.control('[role="dialog"]').querySelectorAll("button").at(-1).onclick();
        await polled.promise; await settle();
        assert.equal(h.control("progress"), null); assert.equal(h.control("[data-stop]"), null); assert.equal(h.control(".bg-job-feedback"), null); assert.equal(h.control("footer"), null);
        if (action === "cg") {
            assert.ok(h.control("[data-pending-cg]")); assert.ok(h.control(".vnccs-common-creator-generation-scan"));
            assert.equal(h.widget.pendingCG.url.startsWith("blob:"), true);
            assert.equal(JSON.parse(h.data.value).selected_cg_id, savedOutput);
            assert.doesNotMatch(h.data.value, /blob:|pendingCG/);
            const before = h.data.value;
            await h.control("[data-pending-cg]").click();
            assert.equal(h.control("[data-large-preview]").src, h.widget.pendingCG.url);
            assert.equal(h.control("[data-close-preview]").disabled, false);
            await h.control("[data-cgs]").querySelectorAll("[data-cg-id]").at(-1).click();
            assert.equal(h.control("[data-large-preview]").src, background().cgs[0].url);
            const after = JSON.parse(h.data.value), previous = JSON.parse(before); delete after.preview;
            assert.deepEqual(after, previous, "Inspection while generating changes only the saved preview");
            h.widget.cgPreviewClose();
        } else {
            assert.equal(h.control("[data-pending-cg]"), null);
            assert.equal(h.control("[data-model-status]").textContent, "Creating model…");
        }
        completion.resolve(response({ status: "failed", error: "Fixture generation failed" })); await running;
        assert.equal(h.widget.busy, false); assert.equal(h.control("[data-pending-cg]"), null); assert.equal(h.widget.pendingCG, null);
        assert.match(h.widget.status.textContent, /Fixture generation failed/);
    }
});

test("CG image preview opens before a model response and survives a missing model", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const model = deferred();
    h.api.handle = path => path.includes(`/models/${MODEL}`) ? model.promise : undefined;
    const opening = h.control("[data-cg-id]").click(); await settle();
    assert.equal(h.control("[data-large-preview]").src, background().cgs[0].url);
    assert.equal(h.control("[data-close-preview]").disabled, false);
    model.resolve(response({ error: "Historical model unavailable" }, 404)); await opening;
    assert.equal(h.control("[data-large-preview]").src, background().cgs[0].url);
    assert.match(h.widget.status.textContent, /Historical model unavailable/);
    h.widget.cgPreviewClose();
});

test("late generation completion cannot replace a reconfigured scene", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const completion = deferred();
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs` && options.method === "POST") return response({ job_id: B });
        if (path === `${BASE}/jobs/${B}`) return completion.promise;
    };
    const job = h.widget.runJob("model"); await settle();
    assert.equal(h.calls.some(call => call.path === `${BASE}/jobs/${B}`), true);
    h.data.value = JSON.stringify(state({ background_id: B })); h.node.onConfigure(); await h.ready();
    completion.resolve(response({ status: "completed", result: { document: background(A, { revision: 4 }) } })); await job;
    assert.equal(h.widget.document.background_id, B); assert.equal(h.widget.state.background_id, B);
    assert.equal(JSON.parse(h.data.value).background_id, B); assert.equal(h.widget.busy, false);
});

test("a completed job for another background is rejected before changing selected assets", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const original = h.data.value;
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs` && options.method === "POST") return response({ job_id: B });
        if (path === `${BASE}/jobs/${B}`) return response({ status: "completed", result: { document: background(B, { revision: 4 }) } });
    };
    await assert.rejects(h.widget.runJob("model"), /another background/i);
    assert.equal(h.widget.document.background_id, A); assert.equal(h.widget.state.background_id, A);
    assert.equal(h.widget.state.selected_cg_id, CG); assert.equal(h.data.value, original); assert.equal(h.widget.busy, false);
});

test("Generate CG requires a nonblank scene name and cancellation leaves the scene unchanged", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const saved = h.data.value;
    const opening = h.control("[data-save-view]").click(), dialog = h.control('[role="dialog"]'), input = dialog.querySelector("input");
    assert.equal(input.required, true); assert.equal(input.maxLength, 96); assert.equal(input.value, "");
    assert.equal(dialog.querySelector("span").textContent, "Scene name (required)");
    for (const name of ["", "   ", "\t"]) {
        input.value = name; dialog.querySelectorAll("button").at(-1).onclick(); await settle();
        assert.equal(h.control('[role="dialog"]'), dialog); assert.equal(h.widget.pendingCG, undefined);
        assert.equal(h.widget.viewer.captureOptions, undefined); assert.equal(h.calls.some(call => call.path.endsWith("/jobs")), false);
    }
    dialog.querySelectorAll("button")[0].onclick(); await opening;
    assert.equal(h.control('[role="dialog"]'), null); assert.equal(h.widget.busy, false); assert.equal(h.data.value, saved);
});

test("a name submitted after configure or removal cannot start CG generation", async () => {
    for (const action of ["configure", "remove"]) {
        const h = harness(JSON.stringify(state())); await h.ready();
        const opening = h.control("[data-save-view]").click(), dialog = h.control('[role="dialog"]');
        dialog.querySelector("input").value = "Old scene";
        if (action === "configure") { h.node.onConfigure(); await h.ready(); dialog.querySelectorAll("button").at(-1).onclick(); }
        else h.node.onRemoved();
        await opening;
        assert.equal(h.widget.viewer.captureOptions, undefined); assert.equal(h.calls.some(call => call.path.endsWith("/jobs")), false);
    }
});

test("Generate CG captures its camera and frame then selects the finished CG", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const completion = deferred(), polled = deferred(), gallery = h.control("[data-cgs]");
    const savedCamera = { ...camera, fov: 58 }, cgId = "0".repeat(32), size = frameDimensions("4:3", 2), completed = background(A, { revision: 5 });
    completed.cgs.push({ cg_id: cgId, name: "Desk", model_id: MODEL, source_image_id: IMAGE, camera: savedCamera, ...size, url: "/finished-cg.png" });
    h.widget.viewer.camera = savedCamera;
    h.widget.state.settings.aspect = "4:3"; h.widget.state.settings.prompt = "Complete the room and retain the camera."; h.widget.updateFrame(); let payload;
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs`) { assert.ok(options.body.get("image") instanceof Blob); payload = JSON.parse(options.body.get("data")); return response({ job_id: B }); }
        if (path === `${BASE}/jobs/${B}`) { polled.resolve(); return completion.promise; }
    };
    const generating = generateCG(h, "  Desk  "); await polled.promise;
    assert.equal(gallery.children[0].dataset.pendingCg, "", "The new scene starts at the top while generating");
    gallery.scrollTop = 72;
    h.documents.set(A, completed);
    completion.resolve(response({ status: "completed", progress: 100, result: { document: completed } })); await generating;
    assert.equal(payload.name, "Desk");
    assert.equal(payload.action, "cg"); assert.equal(payload.model_id, MODEL); assert.deepEqual(payload.camera, savedCamera);
    assert.equal(h.widget.viewer.captureOptions.width, size.width); assert.equal(h.widget.viewer.captureOptions.height, size.height);
    assert.deepEqual(clone(h.widget.state.frame), size); assert.equal(h.widget.state.selected_cg_id, cgId);
    assert.equal(h.widget.viewer.settings.width, size.width); assert.equal(h.widget.viewer.settings.height, size.height);
    const cards = gallery.querySelectorAll("[data-cg-id]"); assert.equal(cards.length, 2);
    assert.deepEqual(cards.map(card => card.dataset.cgId), [cgId, CG], "The completed scene stays at the top");
    assert.equal(cards[0].querySelector("img").src, "/finished-cg.png"); assert.equal(cards[0].classList.contains("is-selected"), true);
    assert.equal(gallery.scrollTop, 72, "Completion preserves gallery scroll position");
    assert.deepEqual(h.widget.document.cgs.map(cg => cg.cg_id), [CG, cgId], "Display order never mutates saved history");
    assert.equal(JSON.parse(h.data.value).selected_cg_id, cgId); assert.deepEqual(h.queued, []);
    await h.window.fire("focus"); h.node.onConfigure(); await h.ready();
    assert.deepEqual(gallery.querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [cgId, CG], "Newest-first order survives refresh and workflow reload");
    assert.equal(h.widget.state.selected_cg_id, cgId);
});

test("random CG seed resolves once before transport and matches serialized UI state", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    let resolutions = 0, payload;
    h.context.generateRandomSeed = () => { resolutions++; return 123456; };
    h.widget.state.settings.generation.seed_mode = "randomize";
    h.api.handle = (path, options) => {
        if (path === BASE + "/backgrounds/" + A + "/jobs") {
            payload = JSON.parse(options.body.get("data")); return response({ job_id: B });
        }
        if (path === BASE + "/jobs/" + B) return response({ status: "completed", result: { document: background() } });
    };
    await generateCG(h);
    assert.equal(resolutions, 1); assert.equal(payload.settings.generation.seed, 123456);
    assert.equal(payload.settings.generation.seed_mode, "fixed");
    const saved = JSON.parse(h.data.value).settings.generation;
    assert.equal(saved.seed, payload.settings.generation.seed);
    assert.equal(saved.seed_mode, "randomize");
});

test("completed CGs retain historical model and camera instead of the latest model alias", async () => {
    const historicalModel = "1".repeat(32), cgId = "0".repeat(32), size = frameDimensions("4:3", 2);
    const h = harness(JSON.stringify(state({ model_id: historicalModel }))), original = background();
    original.models.unshift({ model_id: historicalModel, source_image_id: IMAGE }); original.cgs[0].model_id = historicalModel;
    h.documents.set(A, original); await h.ready(); const savedCamera = { ...camera, fov: 66 }, completed = clone(original); completed.revision = 5;
    completed.cgs.push({ cg_id: cgId, name: "Historical view", model_id: historicalModel, source_image_id: IMAGE, camera: savedCamera, ...size, url: "/historical-cg.png" });
    h.widget.viewer.camera = savedCamera; h.widget.state.settings.aspect = "4:3"; h.widget.updateFrame();
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs`) return response({ job_id: B });
        if (path === `${BASE}/jobs/${B}`) return response({ status: "completed", result: { document: completed } });
    };
    await generateCG(h);
    assert.equal(h.widget.document.model.model_id, MODEL); assert.equal(h.widget.state.model_id, historicalModel);
    assert.equal(h.widget.scene.model_id, historicalModel); assert.equal(h.widget.viewer.camera.fov, 66);
    assert.deepEqual(clone(h.widget.state.frame), size); assert.equal(h.widget.state.selected_cg_id, cgId);
});

test("inline aspect selection immediately updates the capture frame and serialized state", async () => {
    const h = harness(JSON.stringify(state({ frame: { width: 1024, height: 768 } }))); await h.ready();
    const select = h.control("[data-aspect]"); select.value = "9:16"; await select.fire("change");
    const size = frameDimensions("9:16", 2);
    assert.equal(h.widget.viewer.settings.width, size.width); assert.equal(h.widget.viewer.settings.height, size.height);
    assert.deepEqual(clone(h.widget.state.frame), size); assert.equal(JSON.parse(h.data.value).settings.aspect, "9:16");
    assert.deepEqual(h.queued, []);
});

test("named generated variants preserve exact model, CG, preview and workflow ownership", async () => {
    const setId = "8".repeat(32), variantModel = "9".repeat(32), variantCG = "0".repeat(32), ref = { set_id: setId, index: 1, variant_id: "morning", name: "Morning" };
    const document = background();
    document.generated_sets = [{ set_id: setId, version: "", source_image_id: IMAGE, variants: [{ id: "morning", name: "Morning" }], images: [{ url: "/original-set.png" }, { url: "/morning.png", width: 1280, height: 720 }] }];
    document.models.push({ model_id: variantModel, source_image_id: IMAGE, source_variant: ref, settings: { provider: "moge" } });
    document.cgs.push({ cg_id: variantCG, name: "Morning view", model_id: variantModel, source_image_id: IMAGE, source_variant: ref, camera, width: 1280, height: 720, url: "/morning-cg.png" });
    const h = harness(JSON.stringify(state())); h.documents.set(A, document); await h.ready();
    const choices = sourceChoices(document); assert.deepEqual(choices.map(choice => choice.label), ["Original", "Morning"]);
    await h.control(`[data-source-choice="${choices[1].key}"]`).click();
    assert.equal(h.widget.state.model_id, variantModel); assert.deepEqual(clone(h.widget.state.selected_variant), ref);
    await h.control("[data-source-incoming]").onload();
    assert.equal(h.control("[data-model-source]").src, "/morning.png");
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [variantCG, CG]);
    assert.deepEqual(JSON.parse(h.data.value).selected_variant, ref);
    assert.equal(h.control(`[data-source-choice="${choices[1].key}"]`).attributes.get("aria-pressed"), "true");
    assert.equal(h.control(`[data-source-choice="${choices[0].key}"]`).querySelector("img").src, document.images[0].url);
    await h.control("[data-cg-id]").click(); assert.equal(h.widget.state.selected_cg_id, variantCG);
    await h.control(`[data-source-choice="${choices[0].key}"]`).click();
    assert.equal(h.widget.state.model_id, MODEL); assert.equal(h.widget.state.selected_variant, null);
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [variantCG, CG]);
});

test("source selections keep every card and image in the same slot", async () => {
    const doc = background(), setId = "8".repeat(32);
    doc.generated_sets = [{ set_id: setId, version: "", source_image_id: IMAGE,
        variants: ["Morning", "Evening", "Night", "Cloudy", "Rainy"].map((name, index) => ({ id: "v" + index, name })),
        images: [{ url: "/base.png" }, ...Array.from({ length: 5 }, (_, index) => ({ url: "/v" + index + ".png", width: 1280, height: 720 }))] }];
    const h = harness(JSON.stringify(state())); h.documents.set(A, doc); await h.ready();
    const keys = () => h.control("[data-source-choices]").querySelectorAll("[data-source-choice]").map(button => button.dataset.sourceChoice);
    const before = keys(), clicked = before[3], scroll = h.control("[data-source-choices]"); scroll.scrollLeft = 150;
    const cards = [...scroll.querySelectorAll("[data-source-choice]")], images = cards.map(card => card.querySelector("img").src);
    assert.equal(before.length, 6); assert.equal(before[0], IMAGE);
    await h.control('[data-source-choice="' + clicked + '"]').click();
    assert.deepEqual(keys(), before);
    assert.deepEqual([...scroll.querySelectorAll("[data-source-choice]")], cards);
    assert.deepEqual(cards.map(card => card.querySelector("img").src), images);
    assert.equal(cards[3].attributes.get("aria-pressed"), "true");
    assert.equal(cards[0].attributes.get("aria-pressed"), "false");
    await h.control("[data-source-incoming]").onload();
    assert.equal(h.control("[data-model-source]").src, "/v2.png");
    assert.equal(scroll.scrollLeft, 150);
    assert.equal(JSON.parse(h.data.value).selected_variant.index, 3);
    await h.control('[data-source-choice="' + IMAGE + '"]').click();
    await h.control("[data-source-incoming]").onload();
    assert.deepEqual(keys(), before); assert.equal(h.control("[data-model-source]").src, doc.images[0].url);
    assert.equal(cards[0].attributes.get("aria-pressed"), "true");
    const selected = JSON.stringify(h.widget.state), requests = h.calls.length;
    await cards[0].click();
    assert.equal(JSON.stringify(h.widget.state), selected); assert.equal(h.calls.length, requests);
});

test("source scroll arrows follow overflow and preserve the collection instead of paging it", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const list = h.control("[data-source-choices]"); list.clientWidth = 500; list.scrollWidth = 1100;
    h.widget.updateSourceScroll();
    const left = h.control('[data-scroll-source="-1"]'), right = h.control('[data-scroll-source="1"]');
    assert.equal(list.parentElement.classList.contains("is-scrollable"), true);
    assert.equal(left.hidden, false); assert.equal(left.disabled, true); assert.equal(right.disabled, false);
    let motion;
    list.scrollBy = value => { motion = value; };
    await right.click(); assert.deepEqual(clone(motion), { left: 400, behavior: "smooth" });
    list.scrollLeft = 600; h.widget.updateSourceScroll();
    assert.equal(left.disabled, false); assert.equal(right.disabled, true);
    list.scrollWidth = 500; list.scrollLeft = 0; h.widget.updateSourceScroll();
    assert.equal(left.hidden, true); assert.equal(right.hidden, true);
    assert.equal(list.parentElement.classList.contains("is-scrollable"), false);
});

test("malformed generated variant references never replace saved workflow bytes", () => {
    for (const selected_variant of [{ set_id: "../bad", index: 1 }, { set_id: A, index: 0 }, { set_id: A, index: 65 }, { set_id: A, index: 1, variant_id: "../../bad" }, []])
        assert.throws(() => readState(JSON.stringify(state({ selected_variant }))), /variant reference/);
});

test("removal retains pending edits and disposes timers, viewer, observer and listeners", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const viewer = h.widget.viewer;
    h.widget.state.settings.prompt = "Retained on removal"; h.widget.markEdited(); const key = h.widget.draftKey(); h.node.onRemoved();
    assert.equal(h.node.removed, true); assert.equal(h.widget.disposed, true); assert.equal(h.widget.observer.disconnected, true);
    assert.equal(viewer.disposed, true); assert.equal(h.events.size, 0); assert.equal(h.widget.events.signal.aborted, true);
    assert.equal(h.widget.customSelects.disconnected, true); assert.equal(h.widget.frame.removed, true); assert.equal(h.timers.size, 0);
    assert.equal(h.widget.container._vnccsMiddleMouseCanvasPan, undefined);
    assert.equal(JSON.parse(h.drafts.get(key)).settings.prompt, "Retained on removal");
});

test("live resize retains the viewer and follows node dimensions", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const viewer = h.widget.viewer, container = h.widget.container;
    h.node.size = [1500, 1050]; h.node.onResize();
    assert.equal(h.node.cgCreatorDOM.width, 1500); assert.equal(h.widget.viewer, viewer); assert.equal(h.widget.container, container);
    assert.equal(h.widget.container.style.height, `${(1050 - 90) / (1500 / LAYOUT_WIDTH)}px`); assert.equal(viewer.resized, true);
    for (const [width, height] of [[3200, 1900], [4100, 2200], [2900, 1800]]) {
        h.node.size = [width, height]; h.node.onResize(); h.node.cgCreatorDOM.width = 100;
        assert.equal(h.node.cgCreatorDOM.width, width);
        assert.deepEqual(h.node.size, [width, height]);
        assert.equal(h.widget.layoutScale, width / LAYOUT_WIDTH);
        assert.equal(h.widget.container.style.zoom, width / LAYOUT_WIDTH);
        assert.equal(h.widget.container.style.height, `${(height - 90) / (width / LAYOUT_WIDTH)}px`);
        assert.equal(h.widget.viewer, viewer); assert.equal(h.widget.container, container);
    }
});

test("ComfyUI width allocation stays on an unscaled mount and the full CG layout fits inside it", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const mount = h.node.cgCreatorDOM.element, viewer = h.widget.viewer, camera = clone(h.widget.state.camera);
    for (const [nodeWidth, availableWidth] of [[6000, 5976], [1500, 1476], [3200, 3176]]) {
        h.node.size = [nodeWidth, 1900]; h.node.onResize();
        mount.style.width = `${availableWidth}px`; mount.clientWidth = availableWidth;
        h.widget.observer.callback();
        assert.equal(mount.style.width, `${availableWidth}px`, "CG resize must preserve the width assigned by ComfyUI");
        assert.equal(mount.style.zoom, undefined, "ComfyUI's mount must not scale its own assigned width");
        assert.equal(h.widget.observer.target, mount);
        assert.equal(h.widget.container.parentElement, mount);
        assert.ok(Math.abs(parseFloat(h.widget.container.style.width) * h.widget.layoutScale - availableWidth) < 1,
            "The whole composition fits the available width without clipping its right panel");
        assert.equal(h.widget.viewer, viewer); assert.deepEqual(h.widget.state.camera, camera);
    }
    h.node.onConfigure(); await h.ready();
    assert.equal(h.node.cgCreatorDOM.element, mount); assert.equal(mount.style.width, "3176px");
    assert.equal(mount.style.zoom, undefined); assert.equal(h.widget.viewer, viewer);
});

test("width and height resizing follow the Background Creator DOM contract without changing the camera", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const viewport = h.control(".bg-viewport");
    const before = clone(h.widget.state.camera), viewer = h.widget.viewer;
    h.node.size = [LAYOUT_WIDTH, 1772];
    h.node.onResize();
    const initial = [...h.node.size]; h.node.onResize();
    assert.deepEqual(h.node.size, initial, "Observation does not introduce resize loops");
    h.node.size[1] = 3454; h.node.onResize();
    assert.equal(h.widget.container.style.height, `${3454 - 90}px`);
    assert.equal(h.node.size[0], LAYOUT_WIDTH);
    assert.equal(h.node.size[1], 3454);
    assert.equal(h.node.cgCreatorDOM.options.getMinHeight(), 300);
    h.node.size[1] = 950; h.node.onResize();
    assert.equal(h.node.size[0], LAYOUT_WIDTH);
    assert.equal(h.node.size[1], 950);
    h.widget.state.settings.aspect = "9:16"; h.widget.updateFrame();
    assert.equal(viewport.style.aspectRatio, undefined, "Output aspect never changes the fixed landscape viewport");
    assert.equal(h.widget.viewer, viewer);
    assert.deepEqual(h.widget.state.camera, before);
});

test("restored node sizes scale only the inner composition without transforming the ComfyUI mount", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.node.size = [LAYOUT_WIDTH / 2, 950];
    h.node.onResize();
    assert.equal(h.widget.container.style.transform, undefined);
    assert.equal(h.widget.container.style.zoom, .5);
    assert.equal(h.widget.container.style.width, `${LAYOUT_WIDTH}px`);
    assert.equal(h.widget.container.style.height, `${(950 - 90) / .5}px`);
    assert.equal(h.node.cgCreatorDOM.element, h.widget.frame);
    assert.equal(h.widget.frame.style.zoom, undefined); assert.equal(h.widget.frame.style.width, undefined);
    assert.deepEqual(h.node.size, [LAYOUT_WIDTH / 2, 950]);
});

test("resolution changes keep viewport proportions stable while updating the capture immediately", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const viewport = h.control(".bg-viewport"), viewer = h.widget.viewer, camera = clone(h.widget.state.camera);
    for (const mp of [1, 1.1, 1.7, 2, 2.8, 3.9, 4]) {
        h.widget.state.settings.resolution_mp = mp; h.widget.updateFrame();
        assert.equal(viewport.style.aspectRatio, undefined);
        assert.deepEqual(clone(h.widget.state.frame), frameDimensions("16:9", mp));
        assert.deepEqual(clone(viewer.settings), { ...frameDimensions("16:9", mp), show_camera_frame: true });
        assert.equal(JSON.parse(h.data.value).settings.resolution_mp, mp);
        assert.equal(h.widget.viewer, viewer); assert.deepEqual(h.widget.state.camera, camera);
    }
});

test("variant lookup uses the original creator's latest matching attribute signature", () => {
    const card = { id: "morning", name: "Morning", time: "Morning", weather: "Clear", additional_details: "Sunlight" };
    const group = (id, saved, url) => ({ set_id: id, version: "", source_image_id: IMAGE, variants: [saved], images: [{ url: "/original.png" }, { url }] });
    const doc = background(A, { editor: { variants: [card] }, generated_sets: [
        group(A, card, "/older-matching.png"), group(B, card, "/latest-matching.png"),
        group(CG, { ...card, weather: "Rain" }, "/different-weather.png"),
    ] });
    const selected = sourceChoices(doc)[1];
    assert.equal(selected.image.url, "/latest-matching.png"); assert.equal(selected.variant.set_id, B);
    assert.equal(selected.variant.index, 1); assert.equal(selected.variant.variant_id, "morning");
    const pinned = selectedSource(doc, state({ selected_variant: { set_id: A, index: 1 } }));
    assert.equal(pinned.image.url, "/older-matching.png"); assert.equal(pinned.variant.set_id, A);
});

test("current variant selection excludes archives while saved references retain their archived image", () => {
    const card = { id: "morning", name: "Morning", time: "Morning", weather: "Clear", additional_details: "" };
    const group = (id, version, url) => ({ set_id: id, version, source_image_id: IMAGE, variants: [card], images: [{ url: "/original.png" }, { url }] });
    const doc = background(A, { editor: { variants: [card] }, generated_sets: [
        group(B, "", "/Images/01_Morning.png?v=9"), group(CG, "V1", "/Images/V1/01_Morning.png?v=4"),
    ] });
    assert.equal(sourceChoices(doc)[1].image.url, "/Images/01_Morning.png?v=9");
    const pinned = selectedSource(doc, state({ selected_variant: { set_id: CG, index: 1, name: "Morning" } }));
    assert.equal(pinned.image.url, "/Images/V1/01_Morning.png?v=4");
    doc.generated_sets.shift();
    assert.equal(sourceChoices(doc).length, 1, "An archived set does not become the current variant");
});

test("compact dock preserves stored prompts, profiles and Qwen cache settings", async () => {
    const settings = { ...settingsDefaults(), prompt: "Keep this authored instruction", generation: {
        ...settingsDefaults().generation, cache_device: "cpu", cache_dtype: "int4",
        lora_stack: [{ name: "saved-style.safetensors", strength: 0.4 }], future_option: "retain" } };
    const h = harness(JSON.stringify(state({ settings }))); await h.ready();
    assert.equal(h.control(".cg-middle").contains(h.control(".bg-generation-dock")), true);
    assert.equal(h.control(".bg-generation-dock").contains(h.control("[data-aspect]")), true);
    assert.equal(h.control("[data-prompt]"), null);
    assert.equal(h.control(".bg-viewport").contains(h.control(".bg-camera-toolbar")), false);
    assert.equal(h.control(".bg-viewport-panel").contains(h.control(".bg-camera-toolbar")), true);
    assert.equal(h.control(".bg-model-source").contains(h.control("[data-qi2]")), false);
    assert.deepEqual(JSON.parse(h.data.value).settings.generation.lora_stack, settings.generation.lora_stack);
    assert.equal(JSON.parse(h.data.value).settings.generation.future_option, "retain");
    assert.equal(JSON.parse(h.data.value).settings.prompt, settings.prompt);
    assert.equal(JSON.parse(h.data.value).settings.generation.cache_device, "cpu");
    assert.equal(JSON.parse(h.data.value).settings.generation.cache_dtype, "int4");
    assert.equal(JSON.parse(h.data.value).settings.generation.prompt_enhance, false);
});
