import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_character_cloner.js", import.meta.url), "utf8");
const generator = readFileSync(new URL("../web/vnccs_character_generator.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
function block(text, start, end) {
    const offset = text.indexOf(start);
    const finish = text.indexOf(end, offset);
    assert.ok(offset >= 0 && finish > offset, `Missing source block: ${start}`);
    return text.slice(offset, finish);
}

class Element {
    children = [];
    attrs = {};
    classes = new Set();
    classList = { toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name) };
    appendChild(child) { this.children.push(child); }
    setAttribute(name, value) { this.attrs[name] = value; }
}
const controlCenter = (id, kind) => ({
    id, type: "VNCCS_ControlCenter", outputs: [],
    widgets: [{ name: "node_state", value: JSON.stringify({ active_kind: kind }) }],
});

function setup({ kind = "qi2", background = "Blue", previous = "Blue" } = {}) {
    const state = { character: "Clone", source_images: [], source_images_character: "",
        character_info: { background_color: background }, previous_background_color: previous };
    const dataWidget = { name: "widget_data", value: JSON.stringify(state) };
    const node = { id: 42, outputs: [], widgets: [dataWidget] };
    const graph = { _nodes: [node], links: {}, getNodeById(id) { return this._nodes.find(n => n.id === id); }, setDirtyCanvas() {} };
    node.graph = graph;
    if (kind) graph._nodes.push(controlCenter(1, kind));
    const events = [], microtasks = [], listeners = new Map(), timers = new Map();
    const flush = () => { while (microtasks.length) microtasks.shift()(); };
    let removed = 0, serialized = 0, configured = 0;
    node.onRemoved = function () { assert.equal(this, node); removed++; };
    node.onSerialize = function () { assert.equal(this, node); serialized++; };
    node.onConfigure = function () { assert.equal(this, node); configured++; };
    const context = createWidgetContext({
        state, node, dataWidget, els: {}, colAttr: new Element(),
        app: { graph, canvas: { emitEvent(e) { events.push({ type: e.subType, value: dataWidget.value }); } } },
        document: { createElement: () => new Element() },
        window: {
            dispatchEvent() {}, addEventListener: (name, fn) => listeners.set(name, fn),
            removeEventListener: (name, fn) => { if (listeners.get(name) === fn) listeners.delete(name); },
        },
        CustomEvent: class { constructor(name, options) { Object.assign(this, options); } },
        setInterval: fn => { timers.set(1, fn); return 1; }, clearInterval: id => timers.delete(id),
        queueMicrotask: fn => microtasks.push(fn),
        setTimeout() {}, console: { log() {}, error(...args) { throw args.at(-1); } },
        setHelpText() {}, helpFor() {}, syncPoseStudioGender() {}, syncPoseStudioAge() {},
        renderThumbs() {}, beginCharacterRequest() {}, beginCaptionRequest() {}, syncDOMWidgetWidth() {},
    });
    vm.runInContext(
        block(common, "export function registerCleanup", "// Each loader").replace("export ", "") +
        `this.Generator = class { ${block(generator, "    controlCenterWidgetNode() {", "    rememberModelResolution(")} };` +
        "let restoredInfoCharacter = null;" +
        block(source, "const saveState =", "const normalizeAgeValue =") +
        block(source, "const getConnectedModelKind =", "const createGraphicToggle =") +
        block(source, "colAttr.appendChild(createSegmentedField(\"Background\"", "colAttr.appendChild(createSegmentedField(\"Gender\"") +
        block(source, "const updateUIFromState =", "// --- Helpers (Hoisted)") +
        block(source, "const loadState =", "const FIELD_HELP =") +
        block(source, "const syncBackgroundControl =", "// Initialize") +
        "this.load = loadState; this.sync = syncBackgroundControl; this.kind = getConnectedModelKind;", context,
    );
    return { context, state, dataWidget, node, graph, events, listeners, timers, flush,
        segmented: context.colAttr.children[0].children[0],
        callbacks: () => ({ removed, serialized, configured }) };
}

test("Cloner exposes three background options; only QI2 permits Alpha", () => {
    for (const kind of ["anima", "illustrious", "", "qi2"]) {
        const { state, segmented, dataWidget, events } = setup({ kind });
        const buttons = segmented.children;
        assert.deepEqual(buttons.map(b => b.textContent), ["Green", "Blue", "Alpha"]);
        assert.ok(segmented.classes.has("is-three"));
        assert.equal(segmented.attrs["aria-label"], "Background");
        assert.equal(buttons[2].disabled, kind !== "qi2");
        buttons[2].onclick();
        assert.equal(state.character_info.background_color, kind === "qi2" ? "Transparent" : "Blue");
        if (kind === "qi2") {
            assert.deepEqual(events.map(e => e.type), ["before-change", "after-change"]);
            assert.equal(JSON.parse(events[0].value).character_info.background_color, "Blue");
            assert.equal(JSON.parse(events[1].value).character_info.background_color, "Transparent");
            assert.equal(JSON.parse(dataWidget.value).character_info.background_color, "Transparent");
            assert.equal(buttons[2].attrs["aria-pressed"], "true");
        } else {
            assert.match(buttons[2].title, /requires Qwen Image 2.1/);
            assert.equal(events.length, 0);
        }
    }
});

test("model changes restore the last solid color without replacing controls", () => {
    const { graph, state, segmented, listeners, dataWidget, flush } = setup();
    const alpha = segmented.children[2];
    alpha.onclick();
    graph._nodes[1].widgets[0].value = '{"active_kind":"anima"}';
    listeners.get("vnccs-control-center-model-changed")();
    flush();
    assert.equal(state.character_info.background_color, "Blue");
    assert.equal(JSON.parse(dataWidget.value).character_info.background_color, "Blue");
    assert.equal(alpha.disabled, true);
    graph._nodes[1].widgets[0].value = '{"active_kind":"QI2"}';
    listeners.get("vnccs-control-center-model-changed")();
    flush();
    segmented.children[0].onclick();
    alpha.onclick();
    graph._nodes[1].widgets[0].value = '{"active_kind":"illustrious"}';
    listeners.get("vnccs-control-center-model-changed")();
    flush();
    assert.equal(state.character_info.background_color, "Green");
    assert.equal(segmented.children[2], alpha);
});

test("workflow restoration keeps saved Alpha while Control Center is still loading", () => {
    const { context, node, graph, state, dataWidget, segmented } = setup({ kind: "" });
    dataWidget.value = JSON.stringify({ character: "Clone", character_info: { background_color: "alpha" }, previous_background_color: "Blue" });
    node.onConfigure();
    assert.equal(state.character_info.background_color, "Transparent");
    graph._nodes.push(controlCenter(1, "qi2"));
    context.sync();
    assert.equal(state.character_info.background_color, "Transparent");
    assert.equal(segmented.children[2].disabled, false);
    const result = { widgets_values: ["stale"] };
    node.onSerialize(result);
    const restored = setup({ background: "Green" });
    restored.dataWidget.value = result.widgets_values[0];
    restored.node.onConfigure();
    assert.equal(restored.state.character_info.background_color, "Transparent");
    assert.equal(restored.state.previous_background_color, "Blue");
    restored.segmented.children[1].onclick();
    restored.context.sync();
    assert.equal(restored.state.character_info.background_color, "Blue");
});

test("polling and serialization reject legacy Alpha without a compatible model", () => {
    for (const kind of ["anima", ""]) {
        const { state, node, dataWidget, timers, callbacks } = setup({ kind, background: "Transparent" });
        timers.get(1)();
        assert.equal(state.character_info.background_color, "Blue");
        state.character_info.background_color = "Alpha";
        const result = { widgets_values: ["stale"] };
        node.onSerialize(result);
        assert.equal(JSON.parse(result.widgets_values[0]).character_info.background_color, "Blue");
        assert.equal(result.widgets_values[0], dataWidget.value);
        assert.equal(callbacks().serialized, 1);
    }
});

test("connected Clone Generator selects its own Control Center through reroutes", () => {
    const { context, node, graph, segmented, dataWidget, listeners, flush, state } = setup({ kind: "anima" });
    dataWidget.value = JSON.stringify({ character_info: { background_color: "Alpha" }, previous_background_color: "Blue" });
    node.onConfigure();
    listeners.get("vnccs-control-center-model-changed")();
    const center = controlCenter(2, "qi2");
    const reroute = { id: 3, outputs: [{ links: [11] }] };
    const clone = { id: 4, type: "VNCCS_CharacterCloneGenerator", inputs: [{ name: "pipe", link: 12 }], outputs: [] };
    clone._vnccsCharacterGeneratorWidget = new context.Generator();
    clone._vnccsCharacterGeneratorWidget.node = clone;
    node.outputs = [{ links: [10] }];
    graph.links = { 10: { target_id: 3 }, 11: { target_id: 4 }, 12: { origin_id: 2, target_id: 4 } };
    graph._nodes.push(center, reroute, clone);
    flush();
    assert.equal(state.character_info.background_color, "Transparent");
    assert.equal(segmented.children[2].disabled, false);
    center._cc_widget = { _selectedKind: () => "anima", state: { active_kind: "qi2" } };
    context.sync();
    assert.equal(segmented.children[2].disabled, true);
    graph.links[11].target_id = node.id; // A reroute cycle must terminate.
    context.sync();
    assert.equal(segmented.children[2].disabled, true); // Two unrelated centers are ambiguous.
});

test("removing Cloner cleans up polling and model listener, preserving lifecycle callbacks", () => {
    const { node, listeners, timers, callbacks, flush, state } = setup({ background: "Transparent" });
    node.onConfigure();
    listeners.get("vnccs-control-center-model-changed")();
    node.onRemoved();
    state.character_info.background_color = "Alpha";
    flush();
    assert.equal(state.character_info.background_color, "Alpha");
    assert.equal(listeners.size, 0);
    assert.equal(timers.size, 0);
    assert.deepEqual(callbacks(), { removed: 1, serialized: 0, configured: 1 });
});
