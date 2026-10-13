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
    const frame = { setAttribute(name, value) { this[name] = value; }, removeAttribute(name) { delete this[name]; } };
    let finishPreviewDecode;
    const preview = {
        style: { display: "none" }, src: "",
        getAttribute(name) { return name === "src" ? this.src : ""; },
        decode() { return new Promise(resolve => { finishPreviewDecode = resolve; }); },
    };
    const placeholder = { style: {} };
    const els = {
        generationLayer: layer, generationBase: animated(), generationNew: animated(), generationScan: animated(),
        previewFrame: frame, previewImg: preview, placeholder, _ui: {},
    };
    const context = vm.createContext({
        els, state: { character: "Kai", character_info: { sex: "male" } }, node: {},
        registerCleanup() {}, CHARACTER_MANNEQUIN_URL: "female.png", CHARACTER_MANNEQUIN_MALE_URL: "male.png",
        window: { matchMedia: () => ({ matches: false }), requestAnimationFrame: callback => callback() },
    });
    vm.runInContext(`${code}\nglobalThis.start = startPreviewGeneration; globalThis.reveal = revealGeneratedPreview;`, context);

    const first = context.start();
    assert.equal(els.generationBase.src, "male.png");
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
    assert.deepEqual(Array.from(animations[2].frames, frame => frame.top), ["0%", "50%", "100%"]);
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
});
