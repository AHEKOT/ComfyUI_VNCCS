import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_character_generator.js", import.meta.url), "utf8");

function setup({ kind = "QI2", saved = {}, clone = false, clothes = false, emotions = false } = {}) {
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
    Object.assign(widget, { node, data: context.readData(node), isClone: clone, isClothes: clothes, isEmotions: emotions,
        stages: [], renders: 0, renderSettings() { this.renders++; }, renderPreview() {}, renderChain() {}, syncCharacterSourceData() {}, saveBrowserState() {} });
    widget.bindModelResolutionSync();
    return { widget, graph, serialized, cleanups, listeners, timers, app,
        switchTo(nextKind, sourceId = 1, model = "") {
            cc.widgets[0].value = JSON.stringify({ active_kind: nextKind, selected_model: model });
            listeners.get("vnccs-control-center-model-changed")({ detail: { node_id: sourceId } });
        },
        reload() { widget.data = context.readData(node); },
        cache(data) { browserState = JSON.stringify({ version: 1, data }); },
    };
}

function setupEmotionStudio(mode = "qi2") {
    const timers = new Map();
    const settings = { name: "generation_settings", value: JSON.stringify({ generation_mode: mode }) };
    const studio = { id: 10, type: "EmotionGeneratorV2", widgets: [settings] };
    const serialized = { name: "widget_data", value: "{}" };
    const node = { id: 11, inputs: [{ name: "pipe", link: 10 }], widgets: [serialized] };
    const graph = {
        links: { 10: { origin_id: 10 } },
        _nodes: [studio, node],
        getNodeById: id => graph._nodes.find(item => item.id === id),
        setDirtyCanvas() {},
    };
    node.graph = graph;
    const app = { graph, registerExtension(extension) { this.extension = extension; } };
    const context = vm.createContext({
        app,
        window: { addEventListener() {}, removeEventListener() {} },
        setInterval: fn => { timers.set(1, fn); return 1; },
        clearInterval: id => timers.delete(id),
        registerCleanup() {},
        localStorage: { getItem: () => null },
    });
    vm.runInContext(source.replace(/^import .*;\n/gm, "") + "\nthis.Widget = CharacterGeneratorWidget; this.readData = readData;", context);
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        node,
        data: context.readData(node),
        isClone: false,
        isClothes: false,
        isEmotions: true,
        qi2EmotionDefaultsPending: true,
        stages: [],
        renders: 0,
        renderSettings() { this.renders++; },
        renderPreview() { this.renders++; },
        renderChain() { this.renders++; },
        saveBrowserState() {},
    });
    widget.bindModelResolutionSync();
    return { widget, studio, settings, serialized, timers };
}

for (const mode of [{}, { clone: true }, { clothes: true }, { emotions: true }]) {
    test(`retired GAN settings migrate to OFF and stay retired on save (${JSON.stringify(mode)})`, () => {
        const { widget, serialized, reload } = setup({ ...mode, kind: "Klein9b", saved: {
            upscaler: { mode: "gan", gan_model: "old.pth", resolution: 3072 },
        } });
        assert.equal(widget.data.upscaler.mode, "off");
        assert.equal(widget.data.upscaler.gan_model, undefined);
        assert.equal(widget.data.upscaler.resolution, 3072);
        const fields = widget.generatorSettingsGroups().flatMap(group => group.fields);
        assert.ok(fields.every(field => field.key !== "gan_model"));
        if (!mode.emotions) {
            const modes = fields.find(field => field.section === "upscaler" && field.key === "mode");
            assert.equal(JSON.stringify(modes.options), JSON.stringify(["seedvr", "off"]));
        }
        widget.set("upscaler", "mode", "off");
        assert.equal(JSON.parse(serialized.value).upscaler.mode, "off");
        assert.equal(JSON.parse(serialized.value).upscaler.gan_model, undefined);
        widget.set("upscaler", "mode", "seedvr");
        reload();
        assert.equal(widget.data.upscaler.mode, "seedvr");
    });
}

for (const mode of [{}, { clone: true }, { clothes: true }, { emotions: true }]) {
    test(`QI2 selects Native BG Remove for every generator (${JSON.stringify(mode)})`, () => {
        const { widget, timers, switchTo, serialized } = setup(mode);
        timers.get(1)();
        assert.equal(widget.data.bg_remove.preset, "Native");
        assert.equal(JSON.parse(serialized.value).bg_remove.preset, "Native");

        switchTo("Klein9b");
        assert.equal(widget.data.bg_remove.preset, "balanced");
        switchTo("QI2");
        assert.equal(widget.data.bg_remove.preset, "Native");
    });
}

test("manual BG Remove choice survives repeated QI2 updates until the family changes", () => {
    const { widget, timers, switchTo } = setup();
    timers.get(1)();
    widget.set("bg_remove", "preset", "strong");
    switchTo("QI2");
    timers.get(1)();
    assert.equal(widget.data.bg_remove.preset, "strong");
    switchTo("Klein9b");
    switchTo("QI2");
    assert.equal(widget.data.bg_remove.preset, "Native");
});

test("Native is exposed as a BG Remove mode", () => {
    assert.match(source, /const BG_REMOVE_MODES = \["Native", "disabled"/);
});

function connectCreator(harness, mode = "anima") {
    const settings = { name: "widget_data", value: JSON.stringify({ gen_settings: { generation_mode: mode } }) };
    const creator = { id: 5, type: "CharacterCreatorV2", widgets: [settings] };
    harness.graph._nodes.push(creator);
    harness.widget.node.inputs.push({ name: "character", link: 5 });
    harness.graph.links[5] = { origin_id: 5 };
    return {
        creator, settings,
        switchTo(mode) {
            settings.value = JSON.stringify({ gen_settings: { generation_mode: mode } });
            harness.timers.get(1)();
        },
    };
}

test("QI2 Control Center allows Native with every Creator profile", () => {
    const harness = setup({ kind: "QI2", saved: { bg_remove: { preset: "strong" } } });
    const creator = connectCreator(harness);
    harness.timers.get(1)();
    assert.equal(harness.widget.data.bg_remove.preset, "Native");
    assert.equal(JSON.parse(harness.serialized.value).bg_remove.preset, "Native");
    assert.equal(harness.widget.bgRemoveModes().includes("Native"), true);
    assert.equal(harness.listeners.has("vnccs-character-creator-model-changed"), false);
    for (const mode of ["qi2", "illustrious", "anima"]) {
        creator.switchTo(mode);
        assert.equal(harness.widget.data.bg_remove.preset, "Native");
        assert.equal(harness.widget.data.ui.bg_remove_model_kind, "qi2");
    }
    harness.widget.set("bg_remove", "preset", "strong");
    creator.switchTo("qi2");
    creator.switchTo("anima");
    assert.equal(harness.widget.data.bg_remove.preset, "strong");
    harness.widget.set("bg_remove", "preset", "Native");
    assert.equal(harness.widget.data.bg_remove.preset, "Native");
});

test("Control Center switches restore BG Remove without following Creator changes", () => {
    const harness = setup({ kind: "QI2", saved: { bg_remove: { preset: "light" } } });
    const creator = connectCreator(harness, "anima");
    harness.timers.get(1)();
    harness.switchTo("Klein9b");
    assert.equal(harness.widget.data.bg_remove.preset, "light");
    assert.equal(harness.widget.bgRemoveModes().includes("Native"), false);
    creator.switchTo("qi2");
    assert.equal(harness.widget.data.bg_remove.preset, "light");
    harness.switchTo("QI2");
    assert.equal(harness.widget.data.bg_remove.preset, "Native");
    creator.switchTo("anima");
    assert.equal(harness.widget.data.bg_remove.preset, "Native");
});

test("Creator alone cannot select the Generator model or Native mode", () => {
    const harness = setup();
    const creator = connectCreator(harness, "qi2");
    harness.widget.node.inputs[0].link = null;
    harness.timers.get(1)();
    assert.equal(harness.widget.data.bg_remove.preset, "balanced");
    assert.equal(harness.widget.bgRemoveModes().includes("Native"), false);
    creator.switchTo("anima");
    creator.switchTo("qi2");
    assert.equal(harness.widget.data.bg_remove.preset, "balanced");
});

test("non-QI2 workflows repair Native even with an unchanged saved model marker", () => {
    const harness = setup({ kind: "Anima", saved: {
        bg_remove: { preset: "Native" },
        ui: { bg_remove_model_kind: "anima", bg_remove_previous_preset: "light" },
    } });
    harness.timers.get(1)();
    assert.equal(harness.widget.data.bg_remove.preset, "light");
    harness.widget.set("bg_remove", "preset", "Native");
    assert.equal(harness.widget.data.bg_remove.preset, "light");
    const options = harness.widget.generatorSettingsGroups().flatMap(group => group.fields)
        .find(field => field.section === "bg_remove" && field.key === "preset").options;
    assert.equal(options.includes("Native"), false);
});

test("invalid saved restoration presets fall back to balanced", () => {
    const harness = setup({ kind: "Anima", saved: {
        bg_remove: { preset: "Native" }, ui: { bg_remove_previous_preset: "Native" },
    } });
    harness.timers.get(1)();
    assert.equal(harness.widget.data.bg_remove.preset, "balanced");
});

test("serialization preserves Native with a non-QI2 Creator and QI2 Control Center", async () => {
    const harness = setup();
    const { widget, app } = harness;
    const creator = connectCreator(harness, "qi2");
    harness.timers.get(1)();
    class Node {}
    await app.extension.beforeRegisterNodeDef(Node, { name: "VNCCS_CharacterGenerator" });
    creator.settings.value = '{"gen_settings":{"generation_mode":"anima"}}';
    const node = Object.assign(new Node(), widget.node, { _vnccsCharacterGeneratorWidget: widget });
    const serialized = { widgets_values: ["{}"] };
    node.onSerialize(serialized);
    assert.equal(JSON.parse(serialized.widgets_values[0]).bg_remove.preset, "Native");
});

test("Native BG Remove is detected for conditional SAM recovery controls", () => {
    const { widget } = setup({ kind: "QI2" });
    widget.data.bg_remove.preset = "Native";
    assert.equal(widget.isNativeBgRemove(), true);
    const nativeGroups = widget.generatorSettingsGroups();
    assert.equal(nativeGroups.some(group => group.title.includes("SAM3")), false);
    assert.equal(nativeGroups.flatMap(group => group.fields).some(field => field.key === "use_sam3_details_recovery"), false);
    widget.data.bg_remove.preset = "balanced";
    assert.equal(widget.isNativeBgRemove(), false);
    const chromaGroups = widget.generatorSettingsGroups();
    assert.equal(chromaGroups.some(group => group.title.includes("SAM3")), true);
    assert.equal(chromaGroups.flatMap(group => group.fields).some(field => field.key === "use_sam3_details_recovery"), true);
});

test("SAM analysis and SAM3 recovery are disabled by default", () => {
    const { widget } = setup({ kind: "Klein9b" });
    assert.equal(widget.data.emotion_generation.use_sam, false);
    assert.equal(widget.data.bg_remove.use_sam3_details_recovery, false);
});

test("connected QI2 Emotion Studio selects Native BG Remove and restores the prior mode", () => {
    const { widget, settings, serialized, timers } = setupEmotionStudio("qi2");
    timers.get(1)();
    assert.equal(widget.data.bg_remove.preset, "Native");
    assert.equal(JSON.parse(serialized.value).bg_remove.preset, "Native");
    assert.equal(widget.shouldShowEmotionDenoiseControl(), false);
    assert.equal(widget.data.emotion_generation.target_size, 2048);
    assert.equal(widget.data.emotion_generation.bbox_threshold, 0.3);
    assert.equal(widget.data.emotion_generation.bbox_dilation, 50);
    assert.equal(widget.data.emotion_generation.feather, 50);
    assert.equal(widget.data.emotion_generation.drop_size, 10);
    const qi2Groups = widget.generatorSettingsGroups();
    assert.equal(qi2Groups.some(group => group.title.includes("VNCCS BBox Extractor")), true);
    assert.equal(qi2Groups.some(group => group.title === "FaceDetailer"), false);
    const bboxFields = qi2Groups.find(group => group.title.includes("VNCCS BBox Extractor")).fields;
    assert.deepEqual(
        Array.from(bboxFields, field => field.key),
        ["target_size", "bbox_threshold", "bbox_dilation", "feather", "drop_size"],
    );
    const promptGroup = qi2Groups.find(group => group.title.includes("Emotion Prompt"));
    assert.equal(promptGroup.fields[0].key, "qi2_prompt_template");
    assert.match(widget.data.emotion_generation.qi2_prompt_template, /\{emotion\}/);

    settings.value = JSON.stringify({ generation_mode: "anima" });
    timers.get(1)();
    assert.equal(widget.data.bg_remove.preset, "balanced");
    assert.equal(widget.shouldShowEmotionDenoiseControl(), true);
});

test("saved QI2 bbox values are preserved after workflow configuration", () => {
    const { widget, timers } = setupEmotionStudio("qi2");
    widget.qi2EmotionDefaultsPending = false;
    Object.assign(widget.data.emotion_generation, {
        bbox_threshold: 0.42,
        bbox_dilation: 17,
        feather: 9,
        drop_size: 23,
    });
    timers.get(1)();

    assert.equal(widget.data.emotion_generation.bbox_threshold, 0.42);
    assert.equal(widget.data.emotion_generation.bbox_dilation, 17);
    assert.equal(widget.data.emotion_generation.feather, 9);
    assert.equal(widget.data.emotion_generation.drop_size, 23);
});

test("late Emotion Studio restore rebuilds emotion tabs without a click", () => {
    const { widget, studio, timers } = setupEmotionStudio("qi2");
    assert.equal(JSON.stringify(widget.currentStages()), JSON.stringify([["emotion_0001_bg_remove", "Emotion"]]));

    studio.widgets.push(
        { name: "character", value: "Qi2_test" },
        { name: "costumes_data", value: JSON.stringify(["Naked", "Simple"]) },
        { name: "emotions_data", value: JSON.stringify(["angry", "happy"]) },
    );
    const rendersBeforeRestore = widget.renders;
    timers.get(1)();

    assert.equal(JSON.stringify(widget.stages), JSON.stringify([
        ["emotion_0001_bg_remove", "Naked / angry"],
        ["emotion_0002_bg_remove", "Naked / happy"],
        ["emotion_0003_bg_remove", "Simple / angry"],
        ["emotion_0004_bg_remove", "Simple / happy"],
    ]));
    assert.equal(widget.selectedPreview, "emotion_0001_bg_remove");
    assert.ok(widget.renders >= rendersBeforeRestore + 3);
});

test("Emotion Generator exposes only final result stages", () => {
    const { widget } = setup({ emotions: true });
    assert.equal(JSON.stringify(widget.currentStages()), JSON.stringify([["emotion_0001_bg_remove", "Emotion"]]));

    widget.data.emotion_pairs = [
        { costume: "Naked", emotion: "angry" },
        { costume: "Simple", emotion: "happy" },
    ];
    assert.equal(JSON.stringify(widget.currentStages()), JSON.stringify([
        ["emotion_0001_bg_remove", "Naked / angry"],
        ["emotion_0002_bg_remove", "Simple / happy"],
    ]));
    assert.equal(widget.defaultPreviewStage(), "emotion_0001_bg_remove");
});

for (const mode of [{}, { clone: true }, { clothes: true }]) {
    test(`family changes update visible and serialized resolution (${JSON.stringify(mode)})`, () => {
        const { widget, switchTo, serialized } = setup(mode);
        const section = mode.clone ? "common" : "pose_generation";
        for (const [kind, size] of [["QI2", 1024], ["MiniMaxH3", 1536], ["Klein9b", 1024], ["MiniMaxH3", 1536], ["QI2", 1024]]) {
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

test("manual choice survives polling, workflow reload, and switching back to a family", () => {
    const { widget, switchTo, timers, reload } = setup();
    switchTo("MiniMaxH3");
    widget.set("pose_generation", "target_size", 1024);
    switchTo("MiniMaxH3");
    timers.get(1)();
    reload();
    assert.equal(widget.syncModelResolution(), false);
    assert.equal(widget.data.pose_generation.target_size, 1024);
    switchTo("QI2");
    switchTo("MiniMaxH3");
    assert.equal(widget.data.pose_generation.target_size, 1024);
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
    switchTo("QI2");
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

for (const mode of [{}, { clone: true }, { clothes: true }]) {
    test(`each model retains its resolution across changes and reload (${JSON.stringify(mode)})`, () => {
        const { widget, switchTo, serialized, reload, timers } = setup(mode);
        const section = mode.clone ? "common" : "pose_generation";
        switchTo("QI2", 1, "Model A");
        widget.set(section, "target_size", 2560);
        switchTo("QI2", 1, "Model B");
        assert.equal(widget.data[section].target_size, 1024);
        widget.set(section, "target_size", 3072);
        switchTo("MiniMaxH3", 1, "Model C");
        widget.set(section, "target_size", 2048);
        for (const [kind, model, size] of [["QI2", "Model A", 2560], ["QI2", "Model B", 3072], ["MiniMaxH3", "Model C", 2048]]) {
            switchTo(kind, 1, model);
            reload();
            timers.get(1)();
            assert.equal(widget.data[section].target_size, size);
            assert.equal(JSON.parse(serialized.value)[section].target_size, size);
            if (mode.clone) {
                assert.equal(widget.data.pose_generation.target_size, size);
                assert.equal(widget.data.remove_clothes.target_size, size);
            }
        }
    });
}

test("browser backup restores per-model values when the page starts from defaults", () => {
    const original = setup();
    original.switchTo("QI2", 1, "Model A");
    original.widget.set("pose_generation", "target_size", 2560);
    original.switchTo("MiniMaxH3", 1, "Model B");
    original.widget.set("pose_generation", "target_size", 3072);
    const fresh = setup();
    fresh.cache(JSON.parse(original.serialized.value));
    fresh.widget.restoreBrowserState();
    fresh.switchTo("QI2", 1, "Model A");
    assert.equal(fresh.widget.data.pose_generation.target_size, 2560);
    fresh.switchTo("MiniMaxH3", 1, "Model B");
    assert.equal(fresh.widget.data.pose_generation.target_size, 3072);
});

test("explicit workflow model preferences win over an older browser backup", () => {
    const harness = setup();
    harness.switchTo("QI2", 1, "Model A");
    harness.widget.set("pose_generation", "target_size", 1536);
    const older = JSON.parse(harness.serialized.value);
    harness.widget.set("pose_generation", "target_size", 3072);
    harness.cache(older);
    harness.reload();
    harness.widget.restoreBrowserState();
    harness.switchTo("QI2", 1, "Model B");
    harness.switchTo("QI2", 1, "Model A");
    assert.equal(harness.widget.data.pose_generation.target_size, 3072);
});

test("settings dialog edits are remembered by the same model profile", () => {
    const { widget, switchTo } = setup();
    switchTo("QI2", 1, "Model A");
    // The dialog applies a complete draft, then synchronizes model settings.
    widget.data.pose_generation.target_size = 2048;
    widget.syncModelResolution();
    switchTo("QI2", 1, "Model B");
    switchTo("QI2", 1, "Model A");
    assert.equal(widget.data.pose_generation.target_size, 2048);
});

test("refresh recovers the latest slider edit even when workflow autosave is older", () => {
    const live = setup();
    live.switchTo("QI2", 1, "Model A");
    live.widget.set("pose_generation", "target_size", 1536);
    const autosave = JSON.parse(live.serialized.value);
    live.widget.set("pose_generation", "target_size", 2560);
    const refreshed = setup({ saved: autosave });
    refreshed.cache(JSON.parse(live.serialized.value));
    refreshed.widget.restoreBrowserState();
    refreshed.switchTo("QI2", 1, "Model A");
    assert.equal(refreshed.widget.data.pose_generation.target_size, 2560);
    assert.equal(JSON.parse(refreshed.serialized.value).pose_generation.target_size, 2560);
});
