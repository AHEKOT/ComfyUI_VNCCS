import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const code = source.slice(source.indexOf("                let previewGenerationRevision"), source.indexOf("                const doGenerate = async"));

const classes = () => {
    const values = new Set();
    return {
        add: (...names) => names.forEach(name => values.add(name)),
        remove: (...names) => names.forEach(name => values.delete(name)),
        toggle: (name, enabled) => enabled ? values.add(name) : values.delete(name),
        contains: name => values.has(name),
    };
};

test("Creator generation uses the gender mannequin, then reveals the new preview behind the synchronized scan", async () => {
    const animations = [], completions = [];
    const listeners = new Map();
    const api = {
        addEventListener(name, callback) { listeners.set(name, callback); },
        removeEventListener(name, callback) { if (listeners.get(name) === callback) listeners.delete(name); },
    };
    const animated = () => ({
        style: {}, src: "", removeAttribute(name) { if (name === "src") this.src = ""; }, decode: async () => {},
        animate(frames, options) {
            let resolve;
            const finished = new Promise(done => { resolve = done; });
            const animation = { frames, options, finished, cancel() { resolve(); } };
            animations.push(animation); completions.push(resolve);
            return animation;
        },
    });
    const layer = { hidden: true, classList: classes(), _animations: [] };
    const frame = { classList: classes(), setAttribute(name, value) { this[name] = value; }, removeAttribute(name) { delete this[name]; } };
    let finishPreviewDecode;
    const preview = {
        style: { display: "none" }, src: "",
        getAttribute(name) { return name === "src" ? this.src : ""; },
        decode() { return new Promise(resolve => { finishPreviewDecode = resolve; }); },
    };
    const placeholder = { style: {} };
    const els = {
        generationLayer: layer, generationBase: animated(), generationNew: animated(), generationScan: animated(),
        previewFrame: frame, previewImg: preview, placeholder,
        _ui: { presentPreview(image, empty, src) { image.src = src; image.style.display = "block"; empty.style.display = "none"; } },
    };
    const node = { id: 7 };
    const context = vm.createContext({
        api, els, state: { character: "Kai", character_info: { sex: "male" } }, node,
        registerCleanup() {}, CHARACTER_MANNEQUIN_URL: "female.png", CHARACTER_MANNEQUIN_MALE_URL: "male.png",
        beginPreviewRequest: () => () => true, mediaURL: value => value,
        clearPreviewHandlers() {}, hideSpriteNav() {}, setPreviewLoading() {}, saveState() {},
        window: { matchMedia: () => ({ matches: false }), requestAnimationFrame: callback => callback() },
    });
    vm.runInContext(`${code}\nglobalThis.start = startPreviewGeneration; globalThis.reveal = revealGeneratedPreview;`, context);

    const first = context.start();
    assert.equal(els.generationBase.src, "male.png");
    assert.equal(frame.classList.contains("is-male-character"), true);
    assert.equal(layer.classList.contains("uses-mannequin"), true);
    context.els.cancelPreviewGeneration();

    preview.src = "old-preview.png";
    preview.style.display = "block";
    const second = context.start();
    assert.equal(els.generationBase.src, "old-preview.png");
    assert.equal(layer.classList.contains("uses-mannequin"), false);
    const reveal = context.reveal("new-preview.png", "Kai", second);
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(Array.from(animations[0].frames, frame => frame.clipPath), ["inset(0 0 100% 0)", "inset(0 0 0% 0)"]);
    assert.deepEqual(Array.from(animations[1].frames, frame => frame.clipPath), ["inset(0% 0 0 0)", "inset(100% 0 0 0)"]);
    assert.deepEqual(Array.from(animations[2].frames, frame => frame.transform), ["translateY(-100%)", "translateY(-50%)", "translateY(0)"]);
    for (const animation of animations.slice(1)) {
        for (const key of ["duration", "easing", "fill"]) assert.equal(animations[0].options[key], animation.options[key]);
    }
    completions.splice(0).forEach(resolve => resolve());
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(layer.hidden, false);
    assert.equal(preview.src, "new-preview.png");
    finishPreviewDecode();
    assert.equal(await reveal, true);
    assert.equal(preview.src, "new-preview.png");
    assert.equal(layer.hidden, true);
    assert.equal(frame["aria-busy"], undefined);
    assert.notEqual(first, second);

    assert.equal(listeners.has("executing"), false, "A cache-only workflow must not start a fake generation animation");
    listeners.get("vnccs.preview.generation_started")({ detail: { node_id: 7, character: "Kai" } });
    assert.equal(layer.hidden, false, "A normal workflow run must start the same preview animation");
    const animationCount = animations.length;
    const workflowReveal = node._vnccsRevealWorkflowPreview("Kai");
    assert.equal(await workflowReveal, true);
    assert.equal(preview.src.includes("/vnccs/get_cached_preview?character=Kai"), true);
    assert.equal(layer.hidden, true);
    assert.equal(animations.length, animationCount, "Workflow handoff must not overlap pose generation with a reveal animation");

    const previousPreview = preview.src;
    listeners.get("vnccs.preview.generation_started")({ detail: { node_id: 7, character: "Kai" } });
    listeners.get("execution_interrupted")({ detail: { node_id: 99 } });
    assert.equal(layer.hidden, true, "Interrupted workflow hides the generation animation");
    assert.equal(preview.src, previousPreview, "Interrupted workflow keeps the previous preview");
    assert.equal(preview.style.display, "block");
    assert.equal(frame["aria-busy"], undefined);
    assert.doesNotMatch(source, /\.vnccs-creator-generation-layer\.is-interrupted/);
});
