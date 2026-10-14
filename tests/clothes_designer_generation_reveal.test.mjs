import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_clothes_designer.js", import.meta.url), "utf8");
const code = source.slice(
    source.indexOf("                let previewGenerationRevision = 0;"),
    source.indexOf("                spritePreviewNavigator = createSpritePreviewNavigator({"),
);

const classes = () => {
    const values = new Set();
    return {
        add: (...names) => names.forEach(name => values.add(name)),
        remove: (...names) => names.forEach(name => values.delete(name)),
        toggle: (name, force) => force ? values.add(name) : values.delete(name),
        contains: name => values.has(name),
    };
};

test("Clothes Designer keeps the source preview and wipes the generated outfit behind a synchronized scan", async () => {
    const animations = [], finish = [];
    const animatedImage = () => ({
        style: {}, src: "", alt: "", decode: async () => {},
        removeAttribute(name) { if (name === "src") this.src = ""; },
        animate(frames, options) {
            let resolve;
            const finished = new Promise(done => { resolve = done; });
            const animation = { frames, options, finished, cancel: resolve };
            animations.push(animation);
            finish.push(resolve);
            return animation;
        },
    });
    const layer = { hidden: true, classList: classes(), _animations: [] };
    const frame = {
        setAttribute(name, value) { this[name] = value; },
        removeAttribute(name) { delete this[name]; },
    };
    const preview = animatedImage();
    preview.src = "source-sprite.png";
    preview.style.display = "block";
    preview.getAttribute = name => name === "src" ? preview.src : "";
    const previous = { ...animatedImage(), src: "" };
    const next = animatedImage();
    const scan = animatedImage();
    const placeholder = { style: {} };
    const spritePrevBtn = { disabled: false }, spriteNextBtn = { disabled: false };
    const els = {
        generationLayer: layer, generationBase: previous, generationNew: next, generationScan: scan,
        previewFrame: frame, previewImg: preview, placeholder, _ui: {},
    };
    const context = vm.createContext({
        els, spritePrevBtn, spriteNextBtn, node: {},
        registerCleanup() {},
        window: { matchMedia: () => ({ matches: false }), requestAnimationFrame: callback => callback() },
    });
    vm.runInContext(`${code}\nglobalThis.start = startClothesPreviewGeneration; globalThis.reveal = revealClothesPreview;`, context);

    const revision = context.start();
    assert.equal(previous.src, "source-sprite.png");
    assert.equal(layer.hidden, false);
    assert.equal(frame["aria-busy"], "true");
    assert.equal(spritePrevBtn.disabled, true);

    const reveal = context.reveal("outfit-sprite.png", "Alice", revision);
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(Array.from(animations[0].frames, frame => frame.clipPath), ["inset(0 0 100% 0)", "inset(0 0 0% 0)"]);
    assert.deepEqual(Array.from(animations[1].frames, frame => frame.clipPath), ["inset(0% 0 0 0)", "inset(100% 0 0 0)"]);
    assert.deepEqual(Array.from(animations[2].frames, frame => frame.top), ["0%", "100%"]);
    for (const animation of animations.slice(1)) assert.deepEqual(animation.options, animations[0].options);
    finish.splice(0).forEach(resolve => resolve());

    assert.equal(await reveal, true);
    assert.equal(preview.src, "outfit-sprite.png");
    assert.equal(layer.hidden, true);
    assert.equal(frame["aria-busy"], undefined);
    assert.equal(spritePrevBtn.disabled, false);
});

test("Clothes Designer fills the preview by height without resizing the node", () => {
    assert.match(source, /\.vnccs-clothes-top-row\s*\{[^}]*grid-template-columns:\s*400px\s+minmax\(0,\s*1fr\)/s);
    assert.match(source, /\.vnccs-clothes-preview-container\s*\{[^}]*flex:\s*1/s);
    assert.match(source, /\.vnccs-clothes-preview-img\s*\{[^}]*left:\s*50%[^}]*width:\s*auto[^}]*max-width:\s*none[^}]*height:\s*100%[^}]*translateX\(-50%\)/s);
    assert.match(source, /\.vnccs-clothes-generation-img\s*\{[^}]*left:\s*50%[^}]*width:\s*auto[^}]*max-width:\s*none[^}]*height:\s*100%[^}]*translateX\(-50%\)/s);
    assert.doesNotMatch(source, /vnccs-clothes-preview-(?:stage|stack)/);
    assert.doesNotMatch(source, /setSize\(\[1020,\s*940\]\)/);
    const clothesWorkflow = JSON.parse(readFileSync(new URL("../workflows/VNCCS_3.2_Step2_CharacterClothes.json", import.meta.url), "utf8"));
    assert.deepEqual(clothesWorkflow.nodes.find(node => node.type === "ClothesDesigner").size, [710, 920]);
});

test("Clothes Designer keeps the mode tabs and right panel inside their bounds", () => {
    assert.match(source, /\.cd-tab-bar\s*\{[^}]*margin:\s*0\s+0\s+12px/s);
    assert.doesNotMatch(source, /\.cd-tab-bar\s*\{[^}]*margin:\s*0\s+-16px/s);
    assert.match(source, /\.vnccs-clothes-container \.cd-tab-bar\.vnccs-common-ui-segmented::before\s*\{[^}]*border-radius:\s*0/s);
});

test("Clothes Designer replaces only the right panel content when no costume exists", () => {
    const start = source.indexOf("                const syncCostumeEditControls = () => {");
    const end = source.indexOf("\n\n                const onValidationError", start);
    const state = { character: "Alice", costume: "Naked" };
    const panelClasses = classes();
    const contentAttributes = {}, emptyAttributes = {};
    const control = () => ({ disabled: false });
    const els = {
        btnNew: control(), btnGen: control(), btnDel: control(), wizardBtn: control(),
        costSel: { disabled: false, options: [{}] },
        top: control(), bottom: control(), head: control(), face: control(), shoes: control(),
        designPanel: { classList: panelClasses },
        designContent: { inert: false, setAttribute(name, value) { contentAttributes[name] = value; } },
        costumeEmptyState: { setAttribute(name, value) { emptyAttributes[name] = value; } },
        alignCostumeEmptyState() {},
    };
    const node = { _vnccsClothesSelectionPending: false };
    const context = vm.createContext({
        state, els, node,
        hasSelectedEditableCostume: () => !["", "naked", "original"].includes(state.costume.toLowerCase()),
    });
    vm.runInContext(`${source.slice(start, end)}\nglobalThis.sync = syncCostumeEditControls;`, context);

    context.sync();
    assert.equal(panelClasses.contains("is-costume-empty"), true);
    assert.equal(els.designContent.inert, true);
    assert.equal(contentAttributes["aria-hidden"], "true");
    assert.equal(emptyAttributes["aria-hidden"], "false");

    state.costume = "Evening Dress";
    context.sync();
    assert.equal(panelClasses.contains("is-costume-empty"), false);
    assert.equal(els.designContent.inert, false);
    assert.equal(contentAttributes["aria-hidden"], "false");
    assert.equal(emptyAttributes["aria-hidden"], "true");
});
