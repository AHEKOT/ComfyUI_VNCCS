import { createWidgetContext } from './widget_context.mjs';
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_control_center.js", import.meta.url), "utf8");
const defaultName = "Qwen Image 2.1 INT8 ConvRot";
const models = [
    { name: "Other QI2", type: "unet", kind: "QI2" },
    { name: defaultName, type: "unet", kind: "QI2" },
    { name: "Flux Klein", type: "unet", kind: "Klein9b" },
];
const turbo = { name: "Qwen Image 2.1 Viggle Turbo", type: "TurboLora", kind: "QI2" };

function setup(state = {}, config = { models, lora: [turbo] }) {
    const events = [];
    const context = createWidgetContext({
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
        _renderAll() {},
        _scheduleDependencyRefresh() {},
    });
    return { widget, serialized, events };
}

test("QI2 defaults to native UNet, 25 steps, and CFG 3", () => {
    const { widget } = setup();
    assert.deepEqual(Array.from(widget._getModelTypeTabs()), ["unet", "custom"]);
    assert.equal(widget._getSelectedType(), "unet");
    assert.equal(widget._getSelectedModelEntry().name, defaultName);
    assert.equal(widget._currentModelParams().steps, 25);
    assert.equal(widget._currentModelParams().cfg, 3);
});

test("Viggle Turbo switches QI2 to 6 steps and CFG 1, then restores normal settings", () => {
    const { widget } = setup();
    widget._selectTurboLora(turbo.name, true);
    assert.equal(widget._currentModelParams().steps, 6);
    assert.equal(widget._currentModelParams().cfg, 1);
    assert.equal(widget.state.loras.find(item => item.name === turbo.name).auto_apply, true);
    widget._selectTurboLora(turbo.name, false);
    assert.equal(widget._currentModelParams().steps, 25);
    assert.equal(widget._currentModelParams().cfg, 3);
});

test("legacy QIE2511 selection is marked unsupported until explicitly changed", () => {
    const { widget, serialized } = setup({
        selected_type: "gguf", selected_model: "Qwen-Image-Edit-2511-GGUF-Q5",
    });
    widget.restoreState();
    widget._saveState();
    const saved = JSON.parse(serialized.value);
    assert.equal(saved.active_kind, "QI2");
    assert.equal(saved.unsupported_model_kind, "QIE2511");
    assert.equal(widget._getSelectedModelEntry().name, defaultName);
});

test("native selection and custom mode survive restoration", () => {
    const selected = { active_kind: "QI2", selected_type: "unet", selected_model: "Other QI2" };
    const { widget } = setup(selected);
    widget.restoreState();
    assert.equal(widget._getSelectedModelEntry().name, "Other QI2");
    const custom = setup({ active_kind: "QI2", selected_types_by_kind: { QI2: "custom" } }).widget;
    custom.restoreState();
    assert.equal(custom._getSelectedType(), "custom");
    assert.equal(custom._getCustomContextModelEntry().name, defaultName);
});

test("new Control Center serializes QI2 defaults", () => {
    const { widget, serialized, events } = setup();
    widget._saveState();
    const saved = JSON.parse(serialized.value);
    assert.equal(saved.active_kind, "QI2");
    assert.equal(saved.selected_type, "unet");
    assert.equal(saved.selected_model, defaultName);
    assert.equal(saved.model_params.steps, 25);
    assert.equal(saved.model_params.cfg, 3);
    assert.equal(events.at(-1).type, "vnccs-control-center-model-changed");
});

test("packaged workflows select QI2 without legacy model state", () => {
    let count = 0;
    function inspect(value) {
        if (!value || typeof value !== "object") return;
        if (value.type === "VNCCS_ControlCenter") {
            const saved = JSON.parse(value.widgets_values[1]);
            assert.equal(saved.active_kind, "QI2");
            assert.equal(saved.selected_model, defaultName);
            const turboEnabled = saved.loras.some(lora => lora.name === turbo.name && lora.auto_apply);
            assert.equal(saved.model_params.steps, turboEnabled ? 6 : 25);
            assert.equal(saved.model_params.cfg, turboEnabled ? 1 : 3);
            assert.deepEqual(saved.model_params_by_kind.QI2, saved.model_params);
            if (turboEnabled) {
                assert.deepEqual(saved.model_params.turbo_previous_settings, { steps: 25, cfg: 3 });
            }
            count++;
        }
        Object.values(value).forEach(inspect);
    }
    const directory = new URL("../workflows/", import.meta.url);
    for (const name of readdirSync(directory).filter(name => name.endsWith(".json"))) {
        inspect(JSON.parse(readFileSync(new URL(name, directory), "utf8")));
    }
    assert.equal(count, 3);
});

test("QI2 cache controls are inline above Turbo LoRA and persist node parameters", () => {
    const { widget, serialized } = setup();
    assert.deepEqual({ ...widget._qi2CacheSettings() }, { device: "gpu", dtype: "int8" });
    widget._qi2CacheSave({ device: "cpu", dtype: "int4" });
    assert.deepEqual(JSON.parse(serialized.value).qi2_cache, { device: "cpu", dtype: "int4" });
    assert.equal(setup({ active_kind: "Klein9b" }).widget._buildQI2CacheInlineSection(), null);

    assert.match(source, /_buildQI2CacheInlineSection\(\) \{\s*if \(this\._activeKind\(\) !== "QI2"\) return null/);
    assert.match(source, /label\.textContent = "Qwen Image 2\.1 Cache"/);
    assert.match(source, /\["auto", "gpu", "cpu", "off"\]/);
    assert.match(source, /\["default", "int8", "int4"\]/);
    assert.ok(source.indexOf("const cacheInline = this._buildQI2CacheInlineSection()") <
        source.indexOf("const turboInline = this._buildModelTurboInlineSection()"));
});
