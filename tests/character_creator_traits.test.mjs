import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { presetGroups, presetSelection } from "../web/character_presets.mjs";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const catalog = JSON.parse(readFileSync(new URL("../character_template/character_presets_v2.json", import.meta.url), "utf8"));
const block = (start, end) => {
    const offset = source.indexOf(start);
    const limit = source.indexOf(end, offset);
    assert.ok(offset >= 0 && limit > offset);
    return source.slice(offset, limit);
};

class Element {
    constructor(tagName) {
        this.tagName = tagName;
        this.children = [];
        this.attrs = {};
        this.style = {};
        this.classList = { toggle() {} };
        this.value = "";
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attrs[name] = value; }
    focus(options) { this.focusOptions = options; }
    blur() { this.onblur?.(); }
    dispatchEvent(event) { this[`on${event.type}`]?.({ target: this }); }
}

function setup(info = {}) {
    const state = { character: "Test", character_info: info, gen_settings: {} };
    const widget = { name: "widget_data", value: "" };
    let modal;
    const context = createWidgetContext({
        state, els: {}, node: { widgets: [widget] }, Event,
        document: { createElement: tag => new Element(tag) },
        setHelpText() {}, helpFor: () => "", syncBackgroundForGenerationMode() {}, saveCurrentGenerationModeValues() {},
        TAG_DATA: catalog, presetGroups, presetSelection,
        showModal(title, builder, actions) {
            modal = { title, content: builder({ style: {} }), actions };
        },
    });
    vm.runInContext(
        block("const saveState =", "const clearPreviewHandlers") +
        "const debouncedSave = () => saveState();" +
        block("const createTraitField =", "const createSegmentedField =") +
        block("const syncCharacterFields =", "const syncBackgroundForGenerationMode =") +
        block("const applyCharacterWizardData =", "const showCharacterWizardError =") +
        block("const openTagConstructor =", "btnDel.onclick =") +
        "this.makeField = createField; this.sync = syncCharacterFields; this.wizard = applyCharacterWizardData;", context,
    );
    return { context, state, widget, modal: () => modal };
}

test("only the seven character traits use label, editable tags and a plus button", () => {
    const info = { race: "human", hair: "black hair, long hair", additional_details: "<img src=x onerror=alert(1)>" };
    const { context, state } = setup(info);
    for (const [label, key] of [
        ["Race", "race"], ["Skin", "skin_color"], ["Body", "body"], ["Face", "face"],
        ["Hair", "hair"], ["Eyes", "eyes"], ["Details", "additional_details"],
    ]) {
        const row = context.makeField(label, key);
        assert.equal(row.className, "vnccs-creator-trait-row");
        assert.equal(row.children[0].textContent, label);
        const [values, input] = row.children[1].children;
        assert.equal(input, context.els[key]);
        assert.equal(input.hidden, true);
        assert.equal(values.tagName, "button");
        assert.match(values.attrs["aria-label"], /^Edit /);
        assert.deepEqual(values.children.map(chip => chip.textContent), (info[key] || "Add tags").split(", "));
        assert.equal(row.children[2].textContent, "+");
        assert.equal(row.children[2].attrs["aria-label"], `Choose ${label.toLowerCase()} presets`);
    }
    assert.equal(context.makeField("Steps", "steps", "number", [], { steps: 20 }).className, "vnccs-creator-field");
    assert.equal(context.makeField("Custom", "custom").className, "vnccs-creator-field");
    assert.equal(state.character_info.hair, "black hair, long hair");
    assert.equal(state.character_info.additional_details, "<img src=x onerror=alert(1)>");
});

test("manual tag editing persists raw text and requests focus without scrolling", () => {
    const { context, state, widget } = setup({ hair: "black long hair" });
    const row = context.makeField("Hair", "hair");
    const [values, input] = row.children[1].children;
    values.onclick();
    assert.equal(values.hidden, true);
    assert.equal(input.hidden, false);
    assert.equal(input.focusOptions.preventScroll, true);
    input.value = "  silver hair, My Custom Hair, , blue ribbons  ";
    input.dispatchEvent(new Event("input"));
    assert.equal(state.character_info.hair, input.value);
    assert.equal(JSON.parse(widget.value).character_info.hair, input.value);
    assert.deepEqual(values.children.map(chip => chip.textContent), ["silver hair", "My Custom Hair", "blue ribbons"]);
    input.onkeydown({ key: "Enter", preventDefault() {} });
    assert.equal(input.hidden, true);
    assert.equal(values.hidden, false);
    assert.equal(values.focusOptions.preventScroll, true);
    assert.equal(context.els.hair, input);
    assert.equal(row.children[1].children[0], values);
});

test("restored fields, wizard results and preset Apply update the same tag rows", async () => {
    const { context, state, widget, modal } = setup({ hair: "black long hair", eyes: "blue eyes" });
    const row = context.makeField("Hair", "hair");
    const [values, input] = row.children[1].children;
    const eyeRow = context.makeField("Eyes", "eyes");
    state.character_info.hair = "white hair, waist-length hair";
    context.sync();
    assert.deepEqual(values.children.map(chip => chip.textContent), ["white hair", "waist-length hair"]);
    context.wizard({ hair: "silver hair, My Custom Hair", eyes: "green eyes" });
    assert.deepEqual(values.children.map(chip => chip.textContent), ["silver hair", "My Custom Hair"]);
    assert.equal(eyeRow.children[1].children[0].children[0].textContent, "green eyes");
    await row.children[2].onclick();
    assert.equal(modal().title, "Choose Presets: hair");
    const chips = modal().content.children[1].children;
    chips.find(chip => chip.innerText === "Black").onclick();
    modal().actions.find(action => action.text === "APPLY").action();
    assert.equal(input.value, "silver hair, My Custom Hair, black hair");
    assert.equal(JSON.parse(widget.value).character_info.hair, input.value);
    assert.deepEqual(values.children.map(chip => chip.textContent), ["silver hair", "My Custom Hair", "black hair"]);
});
