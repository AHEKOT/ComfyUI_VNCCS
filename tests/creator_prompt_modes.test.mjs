import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const block = (start, end) => {
    const offset = source.indexOf(start), limit = source.indexOf(end, offset);
    assert.ok(offset >= 0 && limit > offset);
    return source.slice(offset, limit);
};

function setup(saved, mode = "illustrious") {
    const state = saved ? structuredClone(saved) : {
        character: "Alice", character_info: {}, prompt_modes: {}, prompt_defaults_version: 4,
        gen_settings: { generation_mode: mode },
    };
    const widget = { name: "widget_data", value: "{}" };
    const node = { widgets: [widget] };
    const style = { setValue(value) { this.value = value; }, close() {} };
    const ctx = createWidgetContext({
        state, node, els: { style, aesthetics: {}, negative_prompt: {} },
        DEFAULT_CHARACTER_STYLE: "ghibli_miyazaki",
        saveCurrentGenerationModeValues() {}, syncBackgroundForGenerationMode() {},
        applyGenerationProfile() {}, syncGenerationControls() {}, refreshGenerationModeUI() {},
    });
    vm.runInContext(
        block("const PROMPT_DEFAULTS_VERSION", "const CC_REPO_ID") +
        block("const saveState =", "const clearPreviewHandlers") +
        block("const getDefaultCharacterInfo =", "const syncCharacterFields") +
        block("const splitPromptTokens =", "const refreshGenerationModeUI") +
        block("const setGenerationMode =", "const setAnimaTurboMode") +
        block("const origSerialize = node.onSerialize;", "// 3. State & Widget Setup") +
        `this.initialize = () => { Object.assign(state.character_info, getDefaultCharacterInfo()); migratePromptModes(); };
         this.restore = migratePromptModes; this.switchMode = setGenerationMode;
         this.savePrompt = saveCurrentPromptModeValues; this.save = saveState;`, ctx,
    );
    if (saved) ctx.restore(); else ctx.initialize();
    return { ctx, state, widget, node, style };
}

test("tag modes default to None and switch to their pre-3.2 tags; QI2 keeps its defaults", () => {
    const { ctx, state, style } = setup();
    assert.equal(state.character_info.style, "none");
    assert.equal(state.character_info.aesthetics, "masterpiece, best quality");
    ctx.switchMode("anima");
    assert.equal(style.value, "none");
    assert.equal(state.character_info.aesthetics, "masterpiece, best quality, score_7, anime");
    assert.match(state.character_info.negative_prompt, /score_1, score_2, score_3/);
    ctx.switchMode("qi2");
    assert.equal(style.value, "ghibli_miyazaki");
    assert.equal(state.character_info.aesthetics, "");
    assert.equal(state.character_info.negative_prompt, "bad quality, worst quality, low quality, blurry, jpeg artifacts");
    for (const mode of ["anima", "qi2"]) assert.equal(setup(undefined, mode).state.character_info.style,
        mode === "qi2" ? "ghibli_miyazaki" : "none");
});

test("each model preserves authored style and tags through switching, serialization and immediate reload", () => {
    const { ctx, state, node } = setup();
    Object.assign(state.character_info, { style: "custom", custom_style: "Charcoal", style_prompt: "Charcoal prompt",
        style_label: "Charcoal", style_description: "Soft graphite", style_reference: "User drawing", aesthetics: "quality, custom tag" });
    ctx.savePrompt(); ctx.save();
    const authored = structuredClone(state.character_info);
    ctx.switchMode("anima");
    assert.equal(state.character_info.style, "none");
    assert.equal(state.character_info.custom_style, "");
    ctx.switchMode("qi2");
    state.character_info.style = "frieren_style";
    state.character_info.style_prompt = "Exact QI2 style";
    ctx.savePrompt(); ctx.save();
    ctx.switchMode("illustrious");
    for (const key of ["style", "custom_style", "style_prompt", "style_label", "style_description", "style_reference", "aesthetics"])
        assert.equal(state.character_info[key], authored[key]);
    const workflow = { widgets_values: ["stale"] };
    node.onSerialize(workflow);
    const restored = setup(JSON.parse(workflow.widgets_values[0]));
    assert.equal(restored.state.character_info.custom_style, "Charcoal");
    restored.ctx.switchMode("qi2");
    assert.equal(restored.state.character_info.style, "frieren_style");
    assert.equal(restored.state.character_info.style_prompt, "Exact QI2 style");
    restored.ctx.switchMode("anima");
    assert.equal(restored.state.character_info.style, "none");
});

test("version 3 migration preserves authored choices and QI2 while restoring Anima's anime tag", () => {
    const saved = {
        character: "Alice", gen_settings: { generation_mode: "qi2" }, prompt_defaults_version: 3,
        character_info: { style: "user_ink", style_prompt: "Saved ink", style_label: "Ink" },
        prompt_modes: {
            illustrious: { aesthetics: "my quality", negative_prompt: "my negative" },
            anima: { aesthetics: "masterpiece, best quality, score_7, custom tag", negative_prompt: "saved anima negative" },
            qi2: { aesthetics: "saved qi2 notes", negative_prompt: "saved qi2 negative" },
        },
    };
    const { ctx, state } = setup(saved);
    assert.equal(state.character_info.style, "user_ink");
    assert.equal(state.character_info.style_prompt, "Saved ink");
    assert.equal(state.character_info.aesthetics, "saved qi2 notes");
    assert.equal(state.character_info.negative_prompt, "saved qi2 negative");
    ctx.switchMode("anima");
    assert.equal(state.character_info.aesthetics, "masterpiece, best quality, score_7, anime, custom tag");
    assert.equal(state.character_info.negative_prompt, "saved anima negative");
    assert.equal(state.character_info.style, "none");
    ctx.switchMode("illustrious");
    assert.equal(state.character_info.aesthetics, "my quality");
    assert.equal(state.character_info.negative_prompt, "my negative");
});
