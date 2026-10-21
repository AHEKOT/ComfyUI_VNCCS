import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { FactoryCommandHistory } from "./fixtures/factory_history.mjs";
import { clone, readState, settingsDefaults, frameDimensions, sourceCameraView, MODEL_DEFAULTS, sourceChoices, selectedSource, sameSource, widgetSource, entrySource, syncDOMWidgetWidth, BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH, CREATOR_MIN_HEIGHT } from "./helpers/cg_frontend.mjs";

const BASE = "/vnccs/cg-creator";
const A = "a".repeat(32), B = "b".repeat(32), MODEL = "c".repeat(32), IMAGE = "d".repeat(32), CG = "e".repeat(32);
const camera = { position: [0, 0.2, 2], target: [0, 0, 0], up: [0, 1, 0], fov: 42 };
const state = (overrides = {}) => ({ schema_version: 1, background_id: A, selected_image_id: IMAGE, model_id: MODEL,
    selected_cg_id: CG, camera: clone(camera), settings: settingsDefaults(), model_settings: clone(MODEL_DEFAULTS), ...overrides });
const background = (id = A, overrides = {}) => ({ schema_version: 2, background_id: id, name: "Evening room", revision: 3,
    cg_settings: settingsDefaults(), model_settings: clone(MODEL_DEFAULTS), selected_image_id: IMAGE,
    images: [{ image_id: IMAGE, width: 1600, height: 900, url: `${BASE}/backgrounds/${id}/images/${IMAGE}` }],
    models: [{ model_id: MODEL, source_image_id: IMAGE }], model: { model_id: MODEL, source_image_id: IMAGE },
    cgs: [{ cg_id: CG, name: "Window view", model_id: MODEL, source_image_id: IMAGE,
        width: 1600, height: 896, camera: clone(camera), url: `${BASE}/backgrounds/${id}/cg/${CG}/image` }], ...overrides });
const scene = (id = A, overrides = {}) => ({ scene_id: "f".repeat(32), model_id: MODEL, background_id: id, camera: clone(camera),
    objects: [{ object_id: B, locked: true, source_camera: clone(camera) }], ...overrides });
const configuration = () => ({ settings: settingsDefaults(), installed: true, assets: [], loras: ["environment.safetensors"], samplers: ["euler"], schedulers: ["simple"] });
const response = (value, status = 200) => ({ ok: status < 400, status, json: async () => clone(value) });
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const dataKey = key => key.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
const abortCallbacks = new WeakMap();

test("malformed nested profiles are retained as invalid workflow data", () => {
    for (const changes of [{ settings: { generation: null } }, { settings: { generation: [] } }, { model_settings: [] }]) {
        const raw = JSON.stringify(state(changes));
        assert.throws(() => readState(raw), /settings/i);
    }
});

// Only the DOM operations used by the widget are emulated; no browser or GPU is involved.
class Element {
    constructor(tag = "div") {
        this.tagName = tag.toUpperCase(); this.style = {}; this.dataset = {}; this.attributes = new Map();
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
    reportValidity() { return true; }
}

function harness(raw = JSON.stringify(state({ background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", camera: null })), { nodeId = 7, drafts = new Map(), scope = "tab:workflow" } = {}) {
    const calls = [], events = new Map(), timers = new Map(), queued = [];
    let extension, timerId = 0;
    const window = new Element("window"), documents = new Map([[A, background()], [B, background(B)]]);
    class Viewer {
        constructor(host, options) { this.options = options; this.canvas = new Element("canvas"); this.controls = {}; this.camera = clone(camera); this.frames = []; this.loads = []; }
        async setScene(value) { this.scene = value; this.loads.push(value); this.camera = { ...camera, position: [99, 99, 99] }; return { failures: [] }; }
        setLighting(settings) { this.lighting = settings; }
        setCaptureSettings(settings) { this.settings = settings; }
        setCameraState(value, { emit = false } = {}) { this.camera = clone(value); this.frames.push(this.camera); if (emit) this.options.onStateChange({ camera: this.camera }); }
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
            if (match && !match[2] && options.method === "PATCH") {
                const document = documents.get(match[1]), values = JSON.parse(options.body);
                if (values.revision !== document.revision) return response({ error: "Scene was modified elsewhere" }, 409);
                const saved = { ...document, revision: document.revision + 1, cg_settings: values.cg_settings, model_settings: values.model_settings };
                documents.set(match[1], saved); return response(saved);
            }
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
        window, storage: localStorage, STYLE: "", injectStyles() {}, generateRandomSeed: () => 424242, workflowScope: node => scope + ":" + node.id, cacheIdentity: () => Math.random().toString(36).slice(2), crypto: { randomUUID: () => "01234567-89ab-cdef-0123-456789abcdef" },
        URL, AbortController, FormData, Blob, console, api, clone, readState, settingsDefaults, frameDimensions, sourceCameraView, MODEL_DEFAULTS, sourceChoices, selectedSource, sameSource, syncDOMWidgetWidth, syncDOMWidgetWidthSoon() {}, BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH, CREATOR_MIN_HEIGHT,
        ResizeObserver: class { observe() {} disconnect() { this.disconnected = true; } },
        createSakuraInteractions: () => ({ syncSelects() {}, captureView() { return {}; }, restoreView() {}, dispose() { this.disconnected = true; } }),
        playSakuraMotion() {}, stopSakuraMotion() {},
        CGQi2Controls: class { constructor(host, options) { this.options = options; this.host = host; this.els = {}; } render(config) { this.config = config; } setDisabled(value) { this.disabled = value; } dispose() { this.disposed = true; } },
        showModal: (container, title, createContent, actions) => {
            const overlay = new Element("div"), modal = new Element("div"), content = createContent(), buttons = [];
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
        ready: async () => { fireTimers(0); await widget.operations; },
        control: selector => widget.container.querySelector(selector),
    };
}

test("CG state restores nested settings without sharing mutable defaults", () => {
    const restored = readState(JSON.stringify(state({ settings: { prompt: "Keep the room", generation: { steps: 31 } } })));
    assert.equal(restored.settings.prompt, "Keep the room"); assert.equal(restored.settings.generation.steps, 31);
    assert.equal(restored.settings.generation.sampler, "euler"); assert.equal(restored.selected_cg_id, CG);
    restored.settings.generation.lora_stack.push({ name: "environment.safetensors", strength: 1 });
    assert.deepEqual(settingsDefaults().generation.lora_stack, []);
    for (const raw of ["{broken", "[]", "null", '{"schema_version":2}', JSON.stringify(state({ model_id: "../model" })), JSON.stringify(state({ camera: {} }))]) assert.throws(() => readState(raw));
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
    assert.deepEqual(h.queued, []);
});

test("TripoSplat settings use the copied VNCCS modal while QI2 controls stay inline", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
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

test("library navigation locks generation before saving and until the new model finishes loading", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "Save before navigation"; h.widget.markEdited();
    const save = deferred(), load = deferred(), model = deferred();
    h.api.handle = (path, options) => {
        if (options.method === "PATCH") return save.promise;
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
    save.resolve(response(background(A, { revision: 4, cg_settings: h.widget.state.settings }))); await settle();
    assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds/${B}`), true); await assertLocked();
    load.resolve(response(background(B))); await settle();
    assert.equal(h.calls.some(call => call.path === `${BASE}/backgrounds/${B}/models/${MODEL}`), true); await assertLocked();
    model.resolve(response(scene(B, { source_image_id: IMAGE }))); await navigation;
    assert.equal(h.widget.busy, false); assert.equal(h.widget.document.background_id, B);
    assert.equal(h.control('[data-action="model"]').disabled, false);
});

test("settings edited during a pending disk save survive its acknowledgement and preserve scroll", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "First instruction"; h.widget.markEdited(); const pending = deferred();
    h.api.handle = (_path, options) => options.method === "PATCH" ? pending.promise : undefined;
    const save = h.widget.persist(); await settle();
    h.widget.state.settings.prompt = "Latest instruction"; h.widget.markEdited();
    const panel = h.control(".bg-panel"), gallery = h.control(".bg-cg-list"); panel.scrollTop = 180; gallery.scrollTop = 95;
    h.widget.render(); assert.equal(panel.scrollTop, 180); assert.equal(gallery.scrollTop, 95);
    pending.resolve(response(background(A, { revision: 4, cg_settings: settingsDefaults({ prompt: "First instruction" }) }))); await save;
    assert.equal(h.widget.state.settings.prompt, "Latest instruction"); assert.equal(h.widget.unsaved, true); assert.equal(h.widget.document.revision, 4);
    assert.equal(JSON.parse(h.drafts.get(h.widget.draftKey())).settings.prompt, "Latest instruction");
    const patch = h.calls.find(call => call.options.method === "PATCH");
    assert.equal(JSON.parse(patch.options.body).cg_settings.prompt, "First instruction");
    assert.ok(JSON.parse(patch.options.body).model_settings);
    h.api.handle = (_path, options) => options.method === "PATCH" ? response(background(A, { revision: 5, cg_settings: h.widget.state.settings })) : undefined;
    await h.widget.persist(); assert.equal(h.widget.unsaved, false); assert.equal(h.drafts.has(h.widget.draftKey()), false);
});

test("disk conflicts retain workflow edits and browser recovery", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "Retain this instruction"; h.widget.markEdited();
    h.api.handle = (_path, options) => options.method === "PATCH" ? response({ error: "Scene was modified elsewhere" }, 409) : undefined;
    await assert.rejects(h.widget.persist(), /modified elsewhere/);
    assert.equal(h.widget.document.revision, 3); assert.equal(h.widget.unsaved, true);
    assert.equal(JSON.parse(h.data.value).settings.prompt, "Retain this instruction");
    assert.equal(JSON.parse(h.drafts.get(h.widget.draftKey())).settings.prompt, "Retain this instruction");
});

test("nodes editing the same scene retain independent recovery drafts across saves and reloads", async () => {
    const drafts = new Map(), first = harness(JSON.stringify(state()), { nodeId: 7, drafts }), second = harness(JSON.stringify(state()), { nodeId: 8, drafts });
    await first.ready(); await second.ready();
    first.widget.state.settings.prompt = "First node instruction"; first.widget.markEdited();
    second.widget.state.settings.prompt = "Second node unsaved instruction"; second.widget.markEdited();
    const firstKey = first.widget.draftKey(), secondKey = second.widget.draftKey(), secondBytes = drafts.get(secondKey);
    assert.notEqual(firstKey, secondKey); await first.widget.persist();
    assert.equal(drafts.has(firstKey), false); assert.equal(drafts.get(secondKey), secondBytes);
    const reloaded = harness(JSON.stringify(state()), { nodeId: 8, drafts }); await reloaded.ready();
    assert.equal(reloaded.widget.draftKey(), secondKey); assert.equal(reloaded.widget.state.settings.prompt, "Second node unsaved instruction");
    assert.equal(reloaded.widget.unsaved, true);
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
        assert.equal(other.widget.unsaved, false);
        assert.notEqual(other.widget.state.settings.prompt, "Private draft");
    }
});

test("source previews hide stale pixels, expose failures and ignore replaced loads", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const image = h.control("[data-model-source]"), placeholder = h.control("[data-empty]");
    assert.equal(image.hidden, true);
    const oldLoad = image.onload;
    h.widget.document.images[0].url += "?revision=2"; h.widget.render();
    oldLoad(); assert.equal(image.hidden, true);
    image.onerror(); assert.equal(image.hidden, true); assert.equal(placeholder.hidden, false);
    assert.match(placeholder.querySelector(".vnccs-common-creator-placeholder-title").textContent, /unavailable/);
    image.onload(); assert.equal(image.hidden, false); assert.equal(placeholder.hidden, true);
    const lateLoad = image.onload;
    h.widget.dispose(); lateLoad(); assert.equal(image.onload, null); assert.equal(image.onerror, null);
});

test("a save acknowledgement cannot remove a changed recovery payload under the same key", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "Instruction being saved"; h.widget.markEdited(); const pending = deferred();
    h.api.handle = (_path, options) => options.method === "PATCH" ? pending.promise : undefined;
    const save = h.widget.persist(); await settle();
    const key = h.widget.draftKey(), replacement = JSON.stringify({ settings: settingsDefaults({ prompt: "Replacement from another editor" }), model_settings: clone(MODEL_DEFAULTS), revision: 3 });
    h.drafts.set(key, replacement);
    pending.resolve(response(background(A, { revision: 4, cg_settings: h.widget.state.settings }))); await save;
    assert.equal(h.drafts.get(key), replacement); assert.equal(h.widget.unsaved, false);
});

test("older recovery drafts remain editable and save without a revision confirmation gate", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const key = h.widget.draftKey();
    const draft = JSON.stringify({ settings: settingsDefaults({ prompt: "Older authored instruction" }), model_settings: { ...MODEL_DEFAULTS, steps: 35 }, revision: 1 });
    h.drafts.set(key, draft); h.node.onConfigure(); await h.ready();
    assert.equal(h.widget.state.settings.prompt, "Older authored instruction"); assert.equal(h.widget.state.model_settings.steps, 35);
    assert.equal(h.widget.unsaved, true); assert.equal(h.drafts.get(key), draft);
    await h.widget.persist();
    assert.equal(h.widget.document.revision, 4); assert.equal(h.widget.unsaved, false); assert.equal(h.drafts.has(key), false);
});

test("recovery drafts without revision metadata remain savable", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const key = h.widget.draftKey();
    h.drafts.set(key, JSON.stringify({ settings: settingsDefaults({ prompt: "Legacy draft" }), model_settings: MODEL_DEFAULTS }));
    h.node.onConfigure(); await h.ready();
    assert.equal(h.widget.state.settings.prompt, "Legacy draft");
    await h.widget.persist(); assert.equal(h.widget.unsaved, false); assert.equal(h.drafts.has(key), false);
});

test("failed restoration can retry without losing the original workflow bytes", async () => {
    const raw = JSON.stringify(state()), h = harness(raw);
    h.api.handle = path => path.endsWith("/backgrounds") ? response({ error: "Storage unavailable" }, 503) : undefined;
    await h.ready();
    assert.equal(h.data.value, raw); assert.equal(h.widget.restoreFailed, true);
    assert.equal(h.control("[data-refresh]").disabled, false);
    assert.throws(() => h.data.beforeQueued(), /restoring/i);
    h.api.handle = undefined;
    await h.control("[data-refresh]").click(); await h.widget.operations;
    assert.equal(h.widget.restoring, false); assert.equal(h.widget.restoreFailed, false);
    assert.equal(h.widget.state.background_id, A); assert.equal(h.widget.state.model_id, MODEL);
    assert.equal(JSON.parse(h.data.value).camera.fov, camera.fov);
    h.data.beforeQueued();
});

test("a pending save cannot overwrite settings restored by later configuration", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const pending = deferred();
    h.widget.state.settings.prompt = "Old pending instruction"; h.widget.markEdited();
    h.api.handle = (_path, options) => options.method === "PATCH" ? pending.promise : undefined;
    const save = h.widget.persist(); await settle();
    const replacement = state({ background_id: B, settings: settingsDefaults({ prompt: "New workflow instruction" }) });
    h.data.value = JSON.stringify(replacement); h.node.onConfigure(); await h.ready();
    pending.resolve(response(background(A, { revision: 4 }))); await save;
    assert.equal(h.widget.document.background_id, B); assert.equal(h.widget.state.settings.prompt, "New workflow instruction");
    assert.equal(JSON.parse(h.data.value).settings.prompt, "New workflow instruction");
});

test("restoring an empty workflow clears the previous scene while retaining its recovery draft", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.widget.state.settings.prompt = "Retain the old scene draft"; h.widget.markEdited(); const key = h.widget.draftKey();
    h.data.value = JSON.stringify(state({ background_id: "", selected_image_id: "", model_id: "", selected_cg_id: "", camera: null }));
    h.node.onConfigure(); await h.ready();
    assert.equal(h.widget.document, null); assert.equal(h.widget.state.background_id, ""); assert.equal(h.widget.unsaved, false);
    assert.equal(h.widget.host.hidden, true); assert.equal(h.control("[data-cgs]").children.length, 0);
    assert.equal(JSON.parse(h.drafts.get(key)).settings.prompt, "Retain the old scene draft");
    await h.widget.persist(); assert.equal(h.calls.some(call => call.options.method === "PATCH"), false);
});

test("read-only backup recovery disables generation while retaining the saved CG", async () => {
    const h = harness(JSON.stringify(state())); h.documents.set(A, background(A, { _recovery: { source: "backup" } })); await h.ready();
    assert.equal(h.widget.restoring, false); assert.match(h.widget.status.textContent, /read-only|read only/i);
    assert.equal(h.control('[data-action="model"]').disabled, true); assert.equal(h.control("[data-save-view]").disabled, true);
    assert.equal(h.control("[data-cgs]").children.length, 1);
});

function variantBackground() {
    const secondImage = "4".repeat(32), latest = "5".repeat(32), secondModel = "6".repeat(32), secondCG = "7".repeat(32);
    const document = background(); document.images.push({ image_id: secondImage, width: 1024, height: 768, url: "/second-background.png" });
    document.models = [{ model_id: MODEL, source_image_id: IMAGE }, { model_id: latest, source_image_id: IMAGE }, { model_id: secondModel, source_image_id: secondImage }];
    document.model = document.models.at(-1); document.cgs[0].camera.fov = 73;
    document.cgs.push({ cg_id: secondCG, model_id: secondModel, source_image_id: secondImage, name: "Second variant view",
        width: 1024, height: 768, camera: { ...camera, fov: 56 }, url: "/second-cg.png" });
    return { document, secondImage, latest, secondModel, secondCG };
}

test("image selection binds its latest model and only its own CG gallery", async () => {
    const { document, secondImage, latest, secondModel, secondCG } = variantBackground(), h = harness(JSON.stringify(state()));
    h.documents.set(A, document); await h.ready();
    await h.control(`[data-source-choice="${secondImage}"]`).click();
    assert.equal(h.widget.state.model_id, secondModel); assert.equal(h.widget.state.selected_cg_id, "");
    assert.equal(h.widget.scene.source_image_id, secondImage); assert.equal(h.widget.host.hidden, false);
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [secondCG]);
    await h.control("[data-cg-id]").click(); assert.equal(h.widget.viewer.camera.fov, 56);
    await h.control(`[data-source-choice="${IMAGE}"]`).click();
    assert.equal(h.widget.state.model_id, latest); assert.equal(h.widget.state.selected_cg_id, "");
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [CG]);
    await h.control("[data-cg-id]").click();
    assert.equal(h.widget.state.model_id, MODEL); assert.equal(h.widget.viewer.camera.fov, 73); assert.equal(h.widget.state.selected_image_id, IMAGE);
});

test("CG preview opens inside the model viewport and cleans up on close, configure and removal", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const viewer = h.widget.viewer, viewport = h.control(".bg-viewport");
    await h.control("[data-cg-id]").click();
    const image = h.control("[data-large-preview]"), navigation = h.control("[data-preview-viewport]");
    assert.equal(viewport.contains(h.control('[role="dialog"]')), true);
    assert.equal(image.src, background().cgs[0].url); assert.equal(image.alt, "Full-size Window view");
    assert.equal(h.widget.state.selected_cg_id, CG); assert.equal(h.widget.viewer, viewer);
    const saved = h.data.value;
    h.widget.cgPreviewClose();
    assert.equal(h.control('[role="dialog"]'), null); assert.equal(h.widget.cgPreviewClose, null);
    assert.equal(image.onload, null); assert.equal(navigation.onpointermove, null); assert.equal(navigation.onwheel, null); assert.equal(h.widget.dialogs.size, 0);
    assert.equal(h.data.value, saved, "Closing the image preserves the selected CG and its camera");
    await h.control("[data-cg-id]").click();
    await h.control("[data-cg-id]").click(); assert.equal(h.widget.dialogs.size, 1);
    h.node.onConfigure(); assert.equal(h.control('[role="dialog"]'), null); await h.ready();
    await h.control("[data-cg-id]").click(); h.node.onRemoved();
    assert.equal(h.widget.dialogs.size, 0); assert.equal(h.widget.cgPreviewClose, null);
    assert.equal(h.window.listeners.get("blur")?.length || 0, 0);
});

test("background preview fills the model viewport and shares preview replacement and cleanup", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const saved = h.data.value, viewer = h.widget.viewer;
    await h.control("[data-enlarge]").click();
    assert.equal(h.control(".bg-viewport").contains(h.control('[role="dialog"]')), true);
    assert.equal(h.control("[data-large-preview]").src, background().images[0].url);
    await h.control("[data-enlarge]").click();
    assert.equal(h.widget.dialogs.size, 1);
    await h.control("[data-cg-id]").click();
    assert.equal(h.widget.dialogs.size, 1);
    await h.control("[data-enlarge]").click(); h.widget.cgPreviewClose();
    assert.equal(h.control('[role="dialog"]'), null);
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
    await h.control("[data-cg-id]").click();
    const operation = h.control("[data-delete-cg]").click(); await settle();
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
    await h.control("[data-delete-cg]").click();
    assert.equal(h.widget.busy, false); assert.equal(h.widget.state.selected_cg_id, CG);
    assert.equal(h.data.value, saved); assert.equal(h.control("[data-cgs]").children.length, 1);
    assert.match(h.widget.status.textContent, /Scene changed elsewhere/);
});

test("model generation runs independently and starts at the calibrated source camera", async () => {
    const h = harness(JSON.stringify(state())); await h.ready(); const modelId = "2".repeat(32), completed = background(A, { revision: 4 });
    completed.models.push({ model_id: modelId, source_image_id: IMAGE }); completed.model = completed.models.at(-1);
    let payload;
    h.api.handle = (path, options) => {
        if (path === `${BASE}/backgrounds/${A}/jobs`) { payload = JSON.parse(options.body); return response({ job_id: B }); }
        if (path === `${BASE}/jobs/${B}`) return response({ status: "completed", result: { document: completed } });
        if (path === `${BASE}/backgrounds/${A}/models/${modelId}`) return response(scene(A, { model_id: modelId, source_image_id: IMAGE }));
    };
    await h.control('[data-action="model"]').click();
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
        const running = h.control(action === "model" ? '[data-action="model"]' : "[data-save-view]").click(); await polled.promise; await settle();
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
            assert.equal(h.data.value, before, "Inspection while generating preserves the selected output and camera");
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

test("Generate CG captures its camera and frame then selects the finished CG", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    const savedCamera = { ...camera, fov: 58 }, cgId = "0".repeat(32), size = frameDimensions("4:3", 2), completed = background(A, { revision: 5 });
    completed.cgs.push({ cg_id: cgId, name: "Desk", model_id: MODEL, source_image_id: IMAGE, camera: savedCamera, ...size, url: "/finished-cg.png" });
    h.widget.viewer.camera = savedCamera;
    h.widget.state.settings.aspect = "4:3"; h.widget.state.settings.prompt = "Complete the room and retain the camera."; h.widget.updateFrame(); let payload;
    h.api.handle = (path, options) => {
        if (options.method === "PATCH") return response(background(A, { revision: 4, cg_settings: h.widget.state.settings }));
        if (path === `${BASE}/backgrounds/${A}/jobs`) { assert.ok(options.body.get("image") instanceof Blob); payload = JSON.parse(options.body.get("data")); return response({ job_id: B }); }
        if (path === `${BASE}/jobs/${B}`) return response({ status: "completed", progress: 100, result: { document: completed } });
    };
    await h.control("[data-save-view]").click();
    assert.equal(payload.action, "cg"); assert.equal(payload.model_id, MODEL); assert.deepEqual(payload.camera, savedCamera);
    assert.equal(h.widget.viewer.captureOptions.width, size.width); assert.equal(h.widget.viewer.captureOptions.height, size.height);
    assert.deepEqual(clone(h.widget.state.frame), size); assert.equal(h.widget.state.selected_cg_id, cgId);
    assert.equal(h.widget.viewer.settings.width, size.width); assert.equal(h.widget.viewer.settings.height, size.height);
    const cards = h.control("[data-cgs]").querySelectorAll("[data-cg-id]"); assert.equal(cards.length, 2);
    assert.equal(cards.at(-1).querySelector("img").src, "/finished-cg.png"); assert.equal(cards.at(-1).classList.contains("is-selected"), true);
    assert.equal(JSON.parse(h.data.value).selected_cg_id, cgId); assert.deepEqual(h.queued, []);
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
    await h.control("[data-save-view]").click();
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
        if (options.method === "PATCH") return response({ ...original, revision: 4, cg_settings: h.widget.state.settings });
        if (path === `${BASE}/backgrounds/${A}/jobs`) return response({ job_id: B });
        if (path === `${BASE}/jobs/${B}`) return response({ status: "completed", result: { document: completed } });
    };
    await h.control("[data-save-view]").click();
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
    document.models.push({ model_id: variantModel, source_image_id: IMAGE, source_variant: ref });
    document.cgs.push({ cg_id: variantCG, name: "Morning view", model_id: variantModel, source_image_id: IMAGE, source_variant: ref, camera, width: 1280, height: 720, url: "/morning-cg.png" });
    const h = harness(JSON.stringify(state())); h.documents.set(A, document); await h.ready();
    const choices = sourceChoices(document); assert.deepEqual(choices.map(choice => choice.label), ["Original", "Morning"]);
    await h.control(`[data-source-choice="${choices[1].key}"]`).click();
    assert.equal(h.widget.state.model_id, variantModel); assert.deepEqual(clone(h.widget.state.selected_variant), ref);
    assert.equal(h.control("[data-model-source]").src, "/morning.png");
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [variantCG]);
    assert.deepEqual(JSON.parse(h.data.value).selected_variant, ref);
    assert.equal(h.control(`[data-source-choice="${choices[1].key}"]`).attributes.get("aria-pressed"), "true");
    assert.equal(h.control(`[data-source-choice="${choices[0].key}"]`).querySelector("img").src, document.images[0].url);
    await h.control("[data-cg-id]").click(); assert.equal(h.widget.state.selected_cg_id, variantCG);
    await h.control(`[data-source-choice="${choices[0].key}"]`).click();
    assert.equal(h.widget.state.model_id, MODEL); assert.equal(h.widget.state.selected_variant, null);
    assert.deepEqual(h.control("[data-cgs]").querySelectorAll("[data-cg-id]").map(card => card.dataset.cgId), [CG]);
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
    assert.equal(h.control("[data-model-source]").src, "/v2.png");
    assert.equal(scroll.scrollLeft, 150);
    assert.equal(JSON.parse(h.data.value).selected_variant.index, 3);
    await h.control('[data-source-choice="' + IMAGE + '"]').click();
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
    assert.equal(h.widget.customSelects.disconnected, true); assert.equal(h.widget.container.removed, true); assert.equal(h.timers.size, 0);
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

test("restored node sizes use the Background Creator zoom layout without transforms", async () => {
    const h = harness(JSON.stringify(state())); await h.ready();
    h.node.size = [LAYOUT_WIDTH / 2, 950];
    h.node.onResize();
    assert.equal(h.widget.container.style.transform, undefined);
    assert.equal(h.widget.container.style.zoom, .5);
    assert.equal(h.widget.container.style.width, `${LAYOUT_WIDTH}px`);
    assert.equal(h.widget.container.style.height, `${(950 - 90) / .5}px`);
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

test("compact dock preserves stored prompts and profiles while fixing cache defaults", async () => {
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
    assert.equal(JSON.parse(h.data.value).settings.generation.cache_device, "default");
    assert.equal(JSON.parse(h.data.value).settings.generation.cache_dtype, "default");
    assert.equal(JSON.parse(h.data.value).settings.generation.prompt_enhance, false);
});
