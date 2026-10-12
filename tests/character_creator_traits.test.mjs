import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { presetGroups, presetSelection } from "../web/character_presets.mjs";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const guards = common.slice(common.indexOf('export function registerCleanup'), common.indexOf('// ── DOM Widget Width Sync')).replaceAll('export ', '');
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
        this.classList = { toggle() {}, add() {} };
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
            modal = { title, content: builder({ style: {} }), actions, overlay: { isConnected: true } };
        },
    });
    vm.runInContext(
        guards + 'const beginCharacterWizardRequest = createRequestGuard(node); this.invalidate = beginCharacterWizardRequest;' +
        block("const saveState =", "const clearPreviewHandlers") +
        "const debouncedSave = () => saveState();" +
        block("const createTraitField =", "const createSegmentedField =") +
        block("const createSegmentedField =", "const createStyleField =") +
        block("const syncCharacterFields =", "const syncBackgroundForGenerationMode =") +
        block("const applyCharacterWizardData =", "const showCharacterWizardError =") +
        block("const openTagConstructor =", "btnDel.onclick =") +
        "this.makeField = createField; this.makeSegmented = createSegmentedField; this.sync = syncCharacterFields; this.wizard = applyCharacterWizardData;" +
        "this.makeBreastSection = () => {" + block("const breastSection =", "traitList.appendChild(breastSection);") + "return breastSection; };", context,
    );
    return { context, state, widget, modal: () => modal };
}

test("character traits use label, editable tags and a plus button", () => {
    const info = { race: "human", hair: "black hair, long hair", additional_details: "<img src=x onerror=alert(1)>" };
    const { context, state } = setup(info);
    for (const [label, key] of [
        ["Race", "race"], ["Skin", "skin_color"], ["Body", "body"], ["Face", "face"],
        ["Hair", "hair"], ["Eyes", "eyes"], ["Breast size", "breast_size"], ["Details", "additional_details"],
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

test("QI2 prompt enhancement uses the complete Turbo card style and persists both states", () => {
    const { context, state, widget } = setup();
    vm.runInContext(block("const promptEnhancementRow =", "qi2Models.appendChild(promptEnhancementRow);")
        + "this.enhancementRow = promptEnhancementRow;", context);
    assert.equal(context.enhancementRow.className, "vnccs-creator-model-card is-installed");
    const top = context.enhancementRow.children[0];
    assert.equal(top.className, "vnccs-creator-model-card-top");
    const [badge, label, status, toggle] = top.children;
    assert.equal(badge.className, "vnccs-creator-model-card-badge ok");
    assert.equal(label.className, "vnccs-creator-model-card-name");
    assert.equal(label.textContent, "Prompt Enhancement");
    assert.equal(toggle.className, "vnccs-creator-toggle vnccs-creator-model-card-toggle");
    const [input, track, thumb] = toggle.children;
    assert.equal(track.className, "vnccs-creator-toggle-track");
    assert.equal(thumb.className, "vnccs-creator-toggle-thumb");
    assert.equal(input.checked, true);
    for (const enabled of [false, true]) {
        input.checked = enabled;
        input.onchange({ target: input });
        assert.equal(state.gen_settings.qi2_prompt_enhancement, enabled);
        assert.equal(JSON.parse(widget.value).gen_settings.qi2_prompt_enhancement, enabled);
        assert.equal(status.textContent, enabled ? "Enabled" : "Disabled");
    }
    context.enhancementRow.onclick();
    assert.equal(state.gen_settings.qi2_prompt_enhancement, false);
    assert.equal(status.textContent, "Disabled");
    input.setValue(true);
    assert.equal(input.checked, true);
    assert.equal(status.textContent, "Enabled");
});

test("breast controls follow gender changes and retain the saved selection", () => {
    const { context, state, widget } = setup({ sex: "female", body: "slim waist", breast_size: "small breasts" });
    const section = context.makeBreastSection();
    assert.equal(section.tagName, "fieldset");
    assert.equal(section.disabled, false);
    const gender = context.makeSegmented("Gender", "sex", [
        { label: "Male", value: "male" }, { label: "Female", value: "female" },
    ]);
    gender.children[0].children[0].onclick();
    assert.equal(section.disabled, true);
    assert.equal(JSON.parse(widget.value).character_info.breast_size, "small breasts");
    gender.children[0].children[1].onclick();
    assert.equal(section.disabled, false);
    assert.equal(state.character_info.breast_size, "small breasts");
    assert.equal(context.els.breast_size.value, "small breasts");
    context.els.breast_size.value = "medium breasts";
    context.els.breast_size.dispatchEvent(new Event("input"));
    assert.equal(JSON.parse(widget.value).character_info.breast_size, "medium breasts");
});

test("legacy body values and wizard output restore into the separate breast field", () => {
    const { context, state, widget } = setup({ sex: "female", body: "medium_breasts, custom build" });
    const section = context.makeBreastSection();
    context.makeField("Body", "body");
    context.sync();
    assert.equal(state.character_info.body, "custom build");
    assert.equal(context.els.breast_size.value, "medium_breasts");
    context.wizard({ sex: "male", body: "broad shoulders" });
    assert.equal(section.disabled, true);
    assert.equal(state.character_info.breast_size, "");
    context.wizard({ sex: "female", body: "small breasts, athletic build" });
    assert.equal(section.disabled, false);
    assert.equal(state.character_info.body, "athletic build");
    assert.equal(JSON.parse(widget.value).character_info.breast_size, "small breasts");
});

test("Creator separates breast presets from build presets and rejects stale Apply after gender changes", async () => {
    const { context, state, modal } = setup({ sex: "female", breast_size: "medium breasts", body: "" });
    // A cached helper predates the new fields; Creator must read both categories directly.
    context.presetGroups = (data, field) => {
        assert.ok(!["body", "build", "breast_size"].includes(field));
        return presetGroups(data, field);
    };
    const breastRow = context.makeBreastSection().children[0];
    await breastRow.children[2].onclick();
    assert.equal(modal().title, "Choose Presets: breast size");
    const chips = modal().content.children[1].children.filter(child => child.tagName === "button");
    assert.deepEqual(chips.map(chip => chip.innerText), catalog.tags.breast_size.map(item => item.label || item.tag));
    state.character_info.sex = "male";
    modal().actions.find(action => action.text === "APPLY").action(modal().overlay);
    assert.equal(state.character_info.breast_size, "medium breasts");
    const oldModal = modal();
    await breastRow.children[2].onclick();
    assert.equal(modal(), oldModal);
    const bodyRow = context.makeField("Body", "body");
    await bodyRow.children[2].onclick();
    const bodyChips = modal().content.children[1].children.filter(child => child.tagName === "button");
    assert.ok(bodyChips.length > 0);
    assert.deepEqual(bodyChips.map(chip => chip.innerText), catalog.tags.body_type.map(item => item.label || item.tag));
    assert.ok(bodyChips.every(chip => !/breast|flat chest/i.test(chip.innerText)));
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
    modal().actions.find(action => action.text === "APPLY").action(modal().overlay);
    assert.equal(input.value, "silver hair, My Custom Hair, black hair");
    assert.equal(JSON.parse(widget.value).character_info.hair, input.value);
    assert.deepEqual(values.children.map(chip => chip.textContent), ["silver hair", "My Custom Hair", "black hair"]);
});

for (const obsolete of ['switch', 'configure', 'removal', 'close']) {
    test(`Creator preset Apply preserves authored fields after ${obsolete}`, async () => {
        const { context, state, widget, modal } = setup({ hair: 'Alice hair' });
        const row = context.makeField('Hair', 'hair');
        await row.children[2].onclick();
        if (obsolete === 'switch') state.character = 'Bob';
        if (obsolete === 'configure') context.invalidate();
        if (obsolete === 'removal') context.node.onRemoved();
        if (obsolete === 'close') modal().overlay.isConnected = false;
        state.character_info.hair = 'new authored hair';
        context.sync();
        const original = JSON.stringify(state);
        modal().actions.find(action => action.text === 'APPLY').action(modal().overlay);
        assert.equal(JSON.stringify(state), original);
        assert.equal(widget.value, '');
    });
}
