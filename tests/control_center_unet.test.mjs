import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_control_center.js", import.meta.url), "utf8");
const defaultName = "Qwen-Image-Edit-2511-int8-convrot";
const models = [
    { name: "Qwen-Image-Edit-2511-GGUF-Q5", type: "gguf", kind: "QIE2511" },
    { name: "Other native Qwen", type: "unet", kind: "QIE2511" },
    { name: defaultName, type: "unet", kind: "QIE2511" },
    { name: "Flux Klein", type: "unet", kind: "Klein9b" },
];

function setup(state = {}, config = { models }) {
    const events = [];
    const context = vm.createContext({
        window: { dispatchEvent: event => events.push(event) },
        CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    });
    const constants = source.slice(source.indexOf("const INITIAL_NODE_W"), source.indexOf("function _injectVNCCSControlCenterStyles"));
    vm.runInContext(constants + source.slice(source.indexOf("class VNCCSControlCenterWidget"))
        + "\nthis.Widget = VNCCSControlCenterWidget;", context);
    const widget = Object.create(context.Widget.prototype);
    const serialized = { value: JSON.stringify(state) };
    Object.assign(widget, {
        state, config, node: { setDirtyCanvas() {} },
        _getStateWidget: () => serialized,
        _getRepoId: () => "",
        _syncOutputSlots() {},
        _syncCustomModelInput() {},
        _dispatchLoraOptions() {},
    });
    return { widget, serialized, events };
}

test("QIE defaults to native UNet and int8-convrot regardless of catalog ordering", () => {
    const { widget } = setup();
    assert.deepEqual(Array.from(widget._getModelTypeTabs()), ["unet", "custom"]);
    assert.equal(widget._getSelectedType(), "unet");
    assert.equal(widget._getSelectedModelEntry().name, defaultName);
    assert.equal(setup({}, null).widget._getSelectedType(), "unet");
});

for (const state of [
    { selected_type: "gguf", selected_model: models[0].name },
    { active_kind: "QIE2511", selected_types_by_kind: { QIE2511: "gguf" }, selected_models: { "QIE2511:gguf": models[0].name } },
]) {
    test("restoring legacy GGUF state persists a native selection", () => {
        const { widget, serialized } = setup(state);
        widget.restoreState();
        assert.equal(widget._getSelectedType(), "unet");
        assert.equal(widget._getSelectedModelEntry().name, defaultName);
        const saved = JSON.parse(serialized.value);
        assert.equal(saved.selected_type, "unet");
        assert.equal(saved.selected_model, defaultName);
        assert.equal(saved.selected_models["QIE2511:unet"], defaultName);
        assert.equal(saved.selected_models.gguf, undefined);
    });
}

test("explicit native selection and custom mode survive restoration", () => {
    const selected = { selected_type: "unet", selected_model: "Other native Qwen" };
    const { widget } = setup(selected);
    widget.restoreState();
    assert.equal(widget._getSelectedModelEntry().name, "Other native Qwen");
    const custom = setup({ selected_type: "custom", selected_model: models[0].name }).widget;
    custom.restoreState();
    assert.equal(custom._getSelectedType(), "custom");
    assert.equal(custom._getCustomContextModelEntry().name, defaultName);
});

test("migrating a dormant QIE selection preserves active Klein settings", () => {
    const { widget } = setup({
        active_kind: "Klein9b", selected_type: "unet", selected_model: "Flux Klein",
        selected_types_by_kind: { QIE2511: "gguf", Klein9b: "unet" },
        selected_models: { "Klein9b:unet": "Flux Klein" },
    });
    widget.restoreState();
    assert.equal(widget.state.selected_types_by_kind.QIE2511, "unet");
    assert.equal(widget.state.selected_model, "Flux Klein");
    assert.equal(widget._getSelectedModelEntry().name, "Flux Klein");
});

test("serializing a new Control Center writes native defaults", () => {
    const { widget, serialized, events } = setup();
    widget._saveState();
    const saved = JSON.parse(serialized.value);
    assert.equal(saved.selected_type, "unet");
    assert.equal(saved.selected_model, defaultName);
    assert.equal(events.at(-1).type, "vnccs-control-center-model-changed");
});
