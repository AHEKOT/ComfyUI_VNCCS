import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_character_generator.js", import.meta.url), "utf8");

function setup({ kind = "QIE2511", saved = {}, clone = false, clothes = false } = {}) {
    const listeners = new Map();
    const timers = new Map();
    const cleanups = [];
    const ccState = { value: JSON.stringify({ active_kind: kind }) };
    const cc = { id: 1, type: "VNCCS_ControlCenter", widgets: [{ name: "node_state", ...ccState }] };
    const serialized = { name: "widget_data", value: JSON.stringify(saved) };
    const node = { id: 2, inputs: [{ link: 0 }], widgets: [serialized] };
    const graph = { links: { 0: { origin_id: 1 } }, getNodeById: id => graph._nodes.find(n => n.id === id),
        _nodes: [cc, node], setDirtyCanvas() {} };
    node.graph = graph;
    let browserState = null;
    const app = { graph, registerExtension(extension) { this.extension = extension; } };
    const context = vm.createContext({
        app,
        window: { addEventListener: (name, fn) => listeners.set(name, fn), removeEventListener: name => listeners.delete(name) },
        setInterval: fn => { timers.set(1, fn); return 1; },
        clearInterval: id => timers.delete(id),
        registerCleanup: (_, fn) => cleanups.push(fn),
        localStorage: { getItem: () => browserState },
    });
    vm.runInContext(source.replace(/^import .*;\n/gm, "") + "\nthis.Widget = CharacterGeneratorWidget; this.readData = readData;", context);
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, { node, data: context.readData(node), isClone: clone, isClothes: clothes,
        stages: [], renders: 0, renderSettings() { this.renders++; }, syncCharacterSourceData() {}, saveBrowserState() {} });
    widget.bindModelResolutionSync();
    return { widget, graph, serialized, cleanups, listeners, timers, app,
        switchTo(nextKind, sourceId = 1) {
            cc.widgets[0].value = JSON.stringify({ active_kind: nextKind });
            listeners.get("vnccs-control-center-model-changed")({ detail: { node_id: sourceId } });
        },
        reload() { widget.data = context.readData(node); },
        cache(data) { browserState = JSON.stringify({ version: 1, data }); },
    };
}

for (const mode of [{}, { clone: true }, { clothes: true }]) {
    test(`family changes update visible and serialized resolution (${JSON.stringify(mode)})`, () => {
        const { widget, switchTo, serialized } = setup(mode);
        const section = mode.clone ? "common" : "pose_generation";
        for (const [kind, size] of [["QIE2511", 1024], ["MiniMaxH3", 1536], ["Klein9b", 1024], ["MiniMaxH3", 1536], ["QIE2511", 1024]]) {
            switchTo(kind);
            assert.equal(widget.data[section].target_size, size);
            assert.equal(JSON.parse(serialized.value)[section].target_size, size);
            if (mode.clone) {
                assert.equal(widget.data.pose_generation.target_size, size);
                assert.equal(widget.data.remove_clothes.target_size, size);
            }
        }
        assert.equal(widget.renders, 5);
    });
}

test("manual choice survives updates, polling, and workflow reload until a family switch", () => {
    const { widget, switchTo, timers, reload } = setup();
    switchTo("MiniMaxH3");
    widget.set("pose_generation", "target_size", 1024);
    switchTo("MiniMaxH3");
    timers.get(1)();
    reload();
    assert.equal(widget.syncModelResolution(), false);
    assert.equal(widget.data.pose_generation.target_size, 1024);
    switchTo("QIE2511");
    switchTo("MiniMaxH3");
    assert.equal(widget.data.pose_generation.target_size, 1536);
});

test("legacy defaults adapt on load while a saved custom size is preserved", () => {
    const defaults = setup({ kind: "MiniMaxH3" });
    defaults.timers.get(1)();
    assert.equal(defaults.widget.data.pose_generation.target_size, 1536);
    const custom = setup({ kind: "MiniMaxH3", saved: { pose_generation: { target_size: 2048 } } });
    custom.timers.get(1)();
    assert.equal(custom.widget.data.pose_generation.target_size, 2048);
});

test("events are scoped to the upstream widget and reconnection is detected", () => {
    const { widget, switchTo, graph, timers } = setup();
    const other = { id: 3, type: "VNCCS_ControlCenter", widgets: [{ name: "node_state", value: '{"active_kind":"MiniMaxH3"}' }] };
    graph._nodes.unshift(other);
    switchTo("QIE2511");
    switchTo("MiniMaxH3", 3);
    assert.equal(widget.data.pose_generation.target_size, 1024);
    graph.links[0].origin_id = 3;
    timers.get(1)();
    assert.equal(widget.data.pose_generation.target_size, 1536);
});

test("reroutes work and disconnected or cyclic graphs do not select an unrelated widget", () => {
    const { widget, graph, timers } = setup({ kind: "MiniMaxH3" });
    graph._nodes.push({ id: 4, type: "Reroute", inputs: [{ link: 1 }] });
    graph.links[0].origin_id = 4;
    graph.links[1] = { origin_id: 1 };
    timers.get(1)();
    assert.equal(widget.data.pose_generation.target_size, 1536);
    graph.links[1].origin_id = 4;
    assert.equal(widget.syncModelResolution(), false);
    widget.node.inputs[0].link = null;
    assert.equal(widget.syncModelResolution(), false);
});

test("browser session state cannot replace workflow resolution or family", () => {
    const { widget, switchTo, cache } = setup();
    switchTo("MiniMaxH3");
    widget.set("pose_generation", "target_size", 2048);
    cache({ pose_generation: { target_size: 512 }, ui: { resolution_model_kind: "qie2511" } });
    widget.restoreBrowserState();
    assert.equal(widget.data.pose_generation.target_size, 2048);
    assert.equal(widget.data.ui.resolution_model_kind, "minimaxh3");
});

test("removing the widget cleans up model events and polling", () => {
    const { cleanups, listeners, timers } = setup();
    cleanups.forEach(fn => fn());
    assert.equal(listeners.size, 0);
    assert.equal(timers.size, 0);
});

test("serialization synchronizes resolution even before the next UI poll", async () => {
    const { widget, app, graph } = setup();
    class Node {
        onSerialize(result) { result.originalHookCalled = true; }
    }
    await app.extension.beforeRegisterNodeDef(Node, { name: "VNCCS_CharacterGenerator" });
    graph._nodes[0].widgets[0].value = '{"active_kind":"MiniMaxH3"}';
    const node = Object.assign(new Node(), widget.node, { _vnccsCharacterGeneratorWidget: widget });
    const serialized = { widgets_values: ["{}"] };
    node.onSerialize(serialized);
    assert.equal(serialized.originalHookCalled, true);
    assert.equal(JSON.parse(serialized.widgets_values[0]).pose_generation.target_size, 1536);
});
