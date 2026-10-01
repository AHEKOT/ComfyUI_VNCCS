import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
function block(start, end) {
    const offset = source.indexOf(start);
    assert.ok(offset >= 0);
    const endOffset = source.indexOf(end, offset);
    assert.ok(endOffset > offset);
    return source.slice(offset, endOffset);
}

class Element {
    constructor() {
        this.children = [];
        this.attrs = {};
        this.classList = { add() {}, toggle() {} };
    }
    appendChild(child) { this.children.push(child); }
    setAttribute(name, value) { this.attrs[name] = value; }
}

function setup({ mode = "anima", background = "Blue", previous = "Blue", marker = mode } = {}) {
    const state = { character_info: { background_color: background },
        gen_settings: { generation_mode: mode, previous_background_color: previous, background_model_kind: marker } };
    const serialized = { name: "widget_data", value: JSON.stringify(state) };
    const context = vm.createContext({
        state, els: {}, node: { id: 42, widgets: [serialized] },
        document: { createElement: () => new Element() },
        setHelpText() {}, helpFor() {}, saveCurrentGenerationModeValues() {},
        localStorage: { setItem() {} },
    });
    vm.runInContext(
        block("const saveState =", "const clearPreviewHandlers") +
        block("const syncBackgroundForGenerationMode =", "const clearCharacterSelection") +
        block("const createSegmentedField =", "const createStyleField") +
        `this.sync = syncBackgroundForGenerationMode; this.save = saveState;
         this.field = createSegmentedField("Background", "background_color", [
             {label: "Green", value: "Green"}, {label: "Blue", value: "Blue"},
             {label: "Alpha", value: "Transparent"}
         ]);`, context,
    );
    vm.runInContext(block("const origSerialize = node.onSerialize;", "// 3. State & Widget Setup"), context);
    return { context, state, serialized, buttons: context.field.children[0].children };
}

for (const mode of ["anima", "illustrious"]) {
    test(`Alpha is disabled and cannot enter ${mode} state`, () => {
        const { context, state, buttons } = setup({ mode });
        context.sync();
        assert.equal(buttons[2].disabled, true);
        assert.match(buttons[2].title, /requires Qwen Image 2.1/);
        buttons[2].onclick();
        assert.equal(state.character_info.background_color, "Blue");
    });
}

test("QI2 enables Alpha and restores the last solid background", () => {
    const { context, state, buttons, serialized } = setup();
    state.gen_settings.generation_mode = "qi2";
    context.save();
    assert.equal(state.character_info.background_color, "Transparent");
    assert.equal(buttons[2].disabled, false);
    assert.equal(buttons[2].attrs["aria-pressed"], "true");
    assert.equal(JSON.parse(serialized.value).character_info.background_color, "Transparent");
    context.save();
    buttons[0].onclick();
    assert.equal(state.character_info.background_color, "Green");
    buttons[2].onclick();
    state.gen_settings.generation_mode = "anima";
    context.save();
    assert.equal(state.character_info.background_color, "Green");
    assert.equal(buttons[2].disabled, true);
});

for (const background of ["Alpha", "Transparent", "transparent"]) {
    test(`restoration repairs legacy ${background} with a matching Anima marker`, () => {
        const { context, state, serialized } = setup({ background });
        context.save();
        assert.equal(state.character_info.background_color, "Blue");
        assert.equal(JSON.parse(serialized.value).character_info.background_color, "Blue");
    });
}

test("repeated QI2 refresh preserves an explicitly selected solid background", () => {
    const { context, state, buttons } = setup({ mode: "qi2", background: "Transparent" });
    buttons[1].onclick();
    context.sync();
    assert.equal(state.character_info.background_color, "Blue");
    buttons[2].onclick();
    assert.equal(state.character_info.background_color, "Transparent");
});

test("serialization normalizes incompatible alpha in the actual workflow widget values", () => {
    const { context, state } = setup();
    state.character_info.background_color = "Alpha";
    const result = { widgets_values: [JSON.stringify(state)] };
    context.node.onSerialize(result);
    assert.equal(JSON.parse(result.widgets_values[0]).character_info.background_color, "Blue");
});
