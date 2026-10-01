import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import test from "node:test";

const source = fs.readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const block = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
class Element {
    constructor(tag) {
        this.tagName = tag;
        this.children = [];
        this.style = {};
        this.attrs = {};
        this.classes = new Set();
        this.classList = { toggle: (key, enabled) => enabled ? this.classes.add(key) : this.classes.delete(key) };
    }
    append(...items) { this.children.push(...items); }
    appendChild(item) { this.children.push(item); return item; }
    setAttribute(key, value) { this.attrs[key] = value; }
    contains(target) { return walk(this).includes(target); }
    matches(selector) { return selector === ":popover-open" && !!this.open; }
    showPopover() { this.open = true; }
    hidePopover() { this.open = false; this.ontoggle?.({ newState: "closed" }); }
    get offsetWidth() { return 320; }
    getBoundingClientRect() {
        return this.tagName === "button"
            ? { left: 800, top: 550, bottom: 570, width: 20, height: 20 }
            : { left: 0, top: 0, width: 320, height: 200 };
    }
    set innerHTML(value) { if (!value) this.children = []; }
}
function walk(root) { return [root, ...root.children.flatMap(walk)]; }
function setup(saved) {
    const state = saved || { character: "Test", character_info: {}, preview_valid: true,
        gen_settings: { generation_mode: "qi2", mode_settings: {} } };
    const widget = { name: "widget_data", value: "" };
    const downloads = [];
    const ctx = vm.createContext({
        window: { innerWidth: 1024, innerHeight: 768 },
        els: {},
        state, node: { id: 42, widgets: [widget] }, localStorage: { setItem() {} },
        document: { createElement: tag => new Element(tag) },
        ccConfig: { lora: [
            { name: "Turbo", kind: "QI2", type: "TurboLora", local_path: "models/loras/turbo.safetensors", status: "installed" },
            { name: "VNCCS Overhaul QI2", kind: "QI2", type: "Helper", local_path: "models/loras/QI2.1/VNCCS/VNCCS_QI2_AnimeOverhaulV1.safetensors", status: "installed" },
            { name: "Other Helper", kind: "QI2", type: "Helper", local_path: "models/loras/other.safetensors" },
        ] },
        ccKind: entry => entry.kind?.toLowerCase(), ccType: entry => entry.type?.toLowerCase(),
        ccRelPath: entry => entry.local_path.replace(/^models\/loras\//, ""),
        ccResolveStatus: entry => entry.status || "missing",
        cardStatusLabel: status => status,
        ccDownloadEntry: (cat, entry) => downloads.push([cat, entry.name]),
        localAssetRelSet: names => new Set(names || []), localAssets: { loras: [] },
        setCcTurboMode() {}, setCcAgeLora() {},
    });
    vm.runInContext(
        block("const QI2_OVERHAUL_LORA_NAME", "// --- STYLES") +
        block("const ANIMA_TURBO_LORA_NAME", "const MODE_PROMPT_DEFAULTS") +
        block("const cloneSettingsValue", "const syncGenerationControls") +
        block("const migrateGenerationModeSettings", "const clearPreviewHandlers") +
        block("const syncBackgroundForGenerationMode", "const clearCharacterSelection") +
        block("const buildAssetCard", "const selectCcAsset") +
        block("const renderModeLoraCards", "const renderCardSection") +
        `this.render = renderModeLoraCards; this.makeCard = buildOverhaulCard;
         this.migrate = migrateGenerationModeSettings; this.applyProfile = applyGenerationProfile;
         this.saveProfile = saveCurrentGenerationModeValues; this.normalize = normalizeOverhaulStrength;`, ctx);
    ctx.migrate();
    return { ctx, state, widget, downloads };
}

test("QI2 card follows Turbo and uses the exact title with five slider positions", () => {
    const { ctx } = setup();
    const root = new Element("div");
    ctx.render(root, "qi2");
    assert.equal(root.children[0].children[0].textContent, "Turbo LoRA");
    const card = root.children[1];
    assert.equal(walk(card).find(el => el.className === "vnccs-model-card-name").textContent, "Qwen Image2.1 Character Overhaul");
    const slider = walk(card).find(el => el.type === "range");
    assert.deepEqual([slider.min, slider.max, slider.step, slider.value], ["0", "1", "0.25", "0.5"]);
    assert.equal(walk(card).some(el => el.type === "checkbox"), false);
    assert.equal(walk(card).some(el => el.tagName === "output" || el.textContent === "Strength"), false);
    const control = walk(card).find(el => el.className === "vnccs-overhaul-control");
    assert.equal(control.children.length, 2);
    assert.equal(control.children[0], slider);
    assert.deepEqual(control.children[1].children.map(el => el.textContent), ["0", "0.25", "0.5", "0.75", "1"]);
    assert.equal(walk(root).some(el => el.textContent === "Other Helper"), false);
    for (const mode of ["anima", "illustrious"]) {
        ctx.render(root, mode);
        assert.equal(walk(root).some(el => el.type === "range"), false);
    }
});

test("slider saves every step, invalidates preview and preserves focus through local changes", () => {
    const { ctx, state, widget } = setup();
    const root = new Element("div");
    ctx.render(root, "qi2");
    const card = root.children[1];
    const slider = walk(card).find(el => el.type === "range");
    for (const strength of [0, .25, .5, .75, 1, 0]) {
        state.preview_valid = true;
        slider.value = String(strength);
        slider.oninput();
        const serialized = JSON.parse(widget.value);
        assert.equal(serialized.gen_settings.mode_settings.qi2.qi2_overhaul_strength, strength);
        assert.equal(serialized.preview_valid, false);
        assert.equal(walk(card).find(el => el.type === "range"), slider);
    }
    assert.equal(slider.attrs["aria-valuetext"], "0 — Off");
    assert.equal(card.classes.has("is-selected"), false);
    state.gen_settings.generation_mode = "anima";
    ctx.applyProfile("anima");
    ctx.saveProfile();
    state.gen_settings.generation_mode = "qi2";
    ctx.applyProfile("qi2");
    assert.equal(state.gen_settings.qi2_overhaul_strength, 0);
    const restored = setup(JSON.parse(widget.value));
    assert.equal(restored.state.gen_settings.qi2_overhaul_strength, 0);
    assert.equal(restored.state.gen_settings.mode_settings.anima.qi2_overhaul_strength, undefined);
});

test("missing card retains catalog identity for Download and can be set to zero", () => {
    const { ctx, downloads, state } = setup();
    ctx.ccConfig.lora[1].status = "missing";
    const root = new Element("div");
    ctx.render(root, "qi2");
    const card = root.children[1];
    const download = walk(card).find(el => el.className === "vnccs-model-card-download");
    download.onclick({ stopPropagation() {} });
    assert.deepEqual(downloads, [["lora", "VNCCS Overhaul QI2"]]);
    const slider = walk(card).find(el => el.type === "range");
    slider.value = "0";
    slider.oninput();
    assert.equal(state.gen_settings.qi2_overhaul_strength, 0);
});

test("legacy defaults, invalid values and Windows LoRA paths normalize safely", () => {
    const { ctx, state } = setup();
    assert.equal(state.gen_settings.qi2_overhaul_strength, .5);
    for (const [value, expected] of [[undefined, .5], [null, .5], ["", .5], ["bad", .5], [NaN, .5], [Infinity, .5], [-1, 0], [2, 1], [.37, .25], [.38, .5]]) {
        assert.equal(ctx.normalize(value), expected);
    }
    const restored = setup({ character_info: {}, gen_settings: { generation_mode: "qi2", mode_settings: {
        qi2: { qi2_overhaul_strength: .5, lora_stack: [
            { name: "QI2.1\\VNCCS\\VNCCS_QI2_AnimeOverhaulV1.safetensors", strength: .75 },
            { name: "other.safetensors", strength: .25 },
        ] },
    } } });
    const stack = restored.state.gen_settings.lora_stack.filter(item => item.name);
    assert.equal(stack.length, 1);
    assert.equal(stack[0].name, "other.safetensors");
    assert.equal(restored.state.gen_settings.qi2_overhaul_strength, .5);
});


test("Overhaul help opens on hover, focus and click without changing strength", () => {
    const { ctx, state, widget } = setup();
    const root = new Element("div");
    ctx.render(root, "qi2");
    const card = root.children[1];
    const heading = walk(card).find(el => el.className === "vnccs-overhaul-heading");
    const help = heading.children[1];
    const [button, tooltip] = help.children;
    const before = JSON.stringify(state);
    assert.equal(heading.children[0].className, "vnccs-model-card-name");
    assert.equal(button.type, "button");
    assert.equal(button.attrs["aria-describedby"], tooltip.id);
    assert.equal(tooltip.attrs.role, "tooltip");
    for (const phrase of ["recommended strength of 0.5", "try 0.75 or 1", "lower the strength to 0.25 or 0", "visual style"])
        assert.ok(tooltip.textContent.includes(phrase));
    help.onpointerenter();
    assert.equal(tooltip.open, true);
    assert.equal(button.attrs["aria-expanded"], "true");
    assert.ok(parseFloat(tooltip.style.left) + 320 <= 1012);
    assert.ok(parseFloat(tooltip.style.top) + 200 < 550);
    help.onpointerleave();
    assert.equal(tooltip.open, false);
    button.onfocus();
    assert.equal(tooltip.open, true);
    let stopped = 0;
    const click = { stopPropagation() { stopped++; } };
    button.onclick(click);
    help.onpointerleave();
    assert.equal(tooltip.open, true);
    button.onclick(click);
    assert.equal(tooltip.open, false);
    assert.equal(stopped, 2);
    button.onclick(click);
    help.onkeydown({ key: "Escape", stopPropagation() {} });
    assert.equal(tooltip.open, false);
    assert.equal(button.attrs["aria-expanded"], "false");
    button.onclick(click);
    tooltip.hidePopover(); // Native light dismissal by an outside click.
    assert.equal(button.attrs["aria-expanded"], "false");
    button.onfocus();
    help.onfocusout({ relatedTarget: null });
    assert.equal(tooltip.open, false);
    assert.equal(JSON.stringify(state), before);
    assert.equal(widget.value, "");
});
