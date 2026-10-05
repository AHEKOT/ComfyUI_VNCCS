import assert from "node:assert/strict";
import test from "node:test";
import { createStylePicker } from "../web/character_styles.mjs";

class Element {
    constructor(tag, document) {
        this.tagName = tag; this.document = document; this.children = [];
        this.style = { setProperty(key, value) { this[key] = value; } }; this.attrs = {}; this.value = ""; this.disabled = false; this.hidden = false; this.inert = false;
    }
    append(...elements) { for (const el of elements) { el.parent = this; this.children.push(el); } }
    replaceChildren(...elements) { this.children = []; this.append(...elements); }
    remove() { this.parent.children = this.parent.children.filter(el => el !== this); }
    setAttribute(key, value) { this.attrs[key] = value; }
    focus() { this.document.activeElement = this; }
    closest() { for (let el = this; el; el = el.parent) if (el.hidden) return el; return null; }
    querySelectorAll(selector = "") {
        if (selector.startsWith(".")) return walk(this).slice(1).filter(el => el.className === selector.slice(1));
        return walk(this).slice(1).filter(el => ["button", "input", "select", "textarea"].includes(el.tagName) && !el.disabled);
    }
}
function walk(root) { return [root, ...root.children.flatMap(walk)]; }
function find(root, className) { return walk(root).find(el => el.className === className); }
const catalog = () => ({ default_style: "legacy", groups: [{ label: "Anime", styles: [
    { id: "legacy", label: "Legacy", description: "Fine contours", reference: "Studio", prompt: "Legacy prompt" },
    { id: "clio_anime", label: "Anime", description: "Cel shading", reference: "Anime tradition", prompt: "Anime prompt" },
] }] });
const response = data => ({ ok: true, json: async () => data });
function setup(info = { style: "legacy" }, fetchApi = async () => response(catalog()), options = {}) {
    const document = { activeElement: null, createElement(tag) { return new Element(tag, this); } };
    globalThis.document = document;
    const host = new Element("div", document);
    const background = new Element("div", document);
    host.append(background);
    const snapshots = []; let teardown;
    const picker = createStylePicker({ host, catalog: catalog(), getInfo: () => info,
        save: () => snapshots.push(JSON.parse(JSON.stringify(info))), fetchApi,
        cleanup: callback => { teardown = callback; }, ...options });
    background.append(picker.root);
    return { picker, host, background, info, snapshots, document, teardown: () => teardown() };
}

test("summary opens the entire workspace, search and category select a serialized style", async () => {
    const ctx = setup();
    const trigger = find(ctx.picker.root, "vnccs-style-summary");
    assert.equal(find(trigger, "vnccs-style-name").textContent, "Legacy");
    assert.equal(find(trigger, "vnccs-style-reference").textContent, "Reference: Studio");
    await trigger.onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    assert.equal(overlay.parent, ctx.host);
    assert.equal(ctx.background.inert, true);
    assert.equal(overlay.attrs.role, "dialog");
    const [search, filter] = walk(overlay).filter(el => ["input", "select"].includes(el.tagName));
    search.value = "cel"; search.oninput();
    assert.equal(find(overlay, "vnccs-style-grid").children.length, 1);
    filter.value = "Missing"; filter.onchange();
    assert.equal(find(overlay, "vnccs-style-status").textContent, "No matching styles");
    filter.value = "Anime"; filter.onchange();
    find(overlay, "vnccs-style-card").onclick();
    assert.equal(ctx.info.style, "clio_anime");
    assert.equal(ctx.snapshots.at(-1).style_prompt, "Anime prompt");
    assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
    assert.equal(ctx.background.inert, false);
    assert.equal(ctx.document.activeElement, trigger);
    assert.equal(walk(ctx.host).some(el => el.tagName === "img"), false);
});

test("custom and unavailable workflows keep their text and style identity", async () => {
    const missing = setup({ style: "user_missing", style_label: "Saved", style_prompt: "Saved prompt" });
    assert.equal(missing.info.style, "user_missing");
    assert.equal(missing.info.style_prompt, "Saved prompt");
    assert.equal(find(missing.picker.root, "vnccs-style-name").textContent, "Saved");
    const ctx = setup({ style: "custom", custom_style: "Ink" });
    assert.equal(ctx.picker.customInput.style.display, "block");
    ctx.picker.customInput.value = "Graphite"; ctx.picker.customInput.oninput();
    assert.equal(ctx.snapshots.at(-1).custom_style, "Graphite");
    const restored = setup(ctx.snapshots.at(-1));
    assert.equal(restored.picker.customInput.value, "Graphite");
});

test("keyboard focus stays in the gallery; Escape and removal discard pending requests", async () => {
    let resolve;
    const ctx = setup(undefined, () => new Promise(done => { resolve = done; }));
    const trigger = find(ctx.picker.root, "vnccs-style-summary");
    const pending = trigger.onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    const controls = overlay.querySelectorAll();
    controls.at(-1).focus();
    let prevented = 0;
    overlay.onkeydown({ key: "Tab", preventDefault() { prevented++; } });
    assert.equal(ctx.document.activeElement, controls[0]);
    overlay.onkeydown({ key: "Tab", shiftKey: true, preventDefault() { prevented++; } });
    assert.equal(ctx.document.activeElement, controls.at(-1));
    overlay.onkeydown({ key: "Escape", stopPropagation() {}, preventDefault() { prevented++; } });
    resolve(response(catalog())); await pending;
    assert.equal(prevented, 3);
    assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
    ctx.teardown();
    await trigger.onclick();
    assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
});

test("user styles save on the server, show inline errors, and edits keep their ID", async () => {
    const saved = { id: "user_" + "a".repeat(32), label: "Mine", description: "Sketch", reference: "Me", prompt: "Ink", user: true };
    const calls = []; let fail = true;
    const data = catalog(); data.groups.push({ label: "My styles", styles: [saved] });
    const ctx = setup({ style: saved.id }, async (url, options) => {
        if (!options) return response(data);
        calls.push(options);
        return fail ? { ok: false, json: async () => ({ error: "Disk full" }) } : response({ style: { ...saved, ...JSON.parse(options.body) } });
    });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    find(overlay, "vnccs-style-edit").onclick();
    const editor = find(overlay, "vnccs-style-editor");
    const inputs = walk(editor).filter(el => ["input", "textarea"].includes(el.tagName));
    inputs[3].value = "Graphite";
    await editor.onsubmit({ preventDefault() {} });
    assert.match(find(overlay, "vnccs-style-status").textContent, /Disk full/);
    assert.equal(ctx.snapshots.length, 0);
    fail = false;
    await editor.onsubmit({ preventDefault() {} });
    assert.equal(JSON.parse(calls.at(-1).body).id, saved.id);
    assert.equal(calls.at(-1).headers["X-VNCCS-CSRF"], "1");
    assert.equal(ctx.snapshots.at(-1).style_prompt, "Graphite");
    assert.equal(ctx.info.style, saved.id);
});

test("New style creates fields without an existing ID", async () => {
    let payload;
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        payload = JSON.parse(options.body);
        return response({ style: { ...payload, id: "user_" + "b".repeat(32), user: true } });
    });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    walk(overlay).find(el => el.textContent === "New style").onclick();
    const editor = find(overlay, "vnccs-style-editor");
    const inputs = walk(editor).filter(el => ["input", "textarea"].includes(el.tagName));
    inputs[0].value = "<b>Mine</b>"; inputs[3].value = "Pencil";
    await editor.onsubmit({ preventDefault() {} });
    assert.equal(payload.id, undefined);
    assert.equal(ctx.snapshots.at(-1).style_prompt, "Pencil");
    assert.equal(find(ctx.picker.root, "vnccs-style-name").textContent, "<b>Mine</b>");
});

const flush = () => new Promise(resolve => setImmediate(resolve));
const previewResponse = id => response({ style_id: id, image: `/vnccs/character_styles/preview?style=${id}&v=1`, width: 1248, height: 1248, saved: true });

test("previews use one immutable settings snapshot and appear before the next render completes", async () => {
    const payload = { node_id: "42", character_info: { hair: "black hair", eyes: "blue eyes" },
        gen_settings: { target_size: 1536, seed: 123, steps: 25, lora_stack: [{ name: "Mine", strength: .5 }] } };
    const requests = [], waiting = []; let handler;
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        requests.push(JSON.parse(options.body));
        return new Promise(resolve => waiting.push(resolve));
    }, { getPreviewPayload: () => payload, imageURL: value => `/proxy${value}`, listenPreview: value => { handler = value; } });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    const button = walk(overlay).find(el => el.textContent === "Generate all previews");
    const pending = button.onclick();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].style_id, "legacy");
    payload.character_info.hair = "red hair";
    payload.gen_settings.target_size = 4096;
    handler({ detail: { node_id: "other", request_id: requests[0].request_id, status: "queued" } });
    assert.match(find(overlay, "vnccs-style-status").textContent, /^Rendering/);
    handler({ detail: { node_id: "42", request_id: requests[0].request_id, status: "queued" } });
    assert.match(find(overlay, "vnccs-style-status").textContent, /^Queued/);
    waiting.shift()(previewResponse("legacy")); await flush();
    assert.equal(requests.length, 2);
    assert.equal(requests[1].character_info.hair, "black hair");
    assert.equal(requests[1].gen_settings.target_size, 1536);
    assert.equal(requests[1].gen_settings.seed, 123);
    assert.deepEqual(requests[1].gen_settings.lora_stack, [{ name: "Mine", strength: .5 }]);
    assert.equal(walk(find(overlay, "vnccs-style-grid")).filter(el => el.tagName === "img").length, 1);
    assert.match(find(ctx.picker.root, "vnccs-style-preview-image").src, /^\/proxy\/vnccs\//);
    assert.equal(ctx.info.style, "legacy");
    waiting.shift()(previewResponse("clio_anime")); await pending;
    assert.equal(walk(find(overlay, "vnccs-style-grid")).filter(el => el.tagName === "img").length, 2);
    assert.match(find(overlay, "vnccs-style-status").textContent, /Saved all 2/);
});

test("Stop and node removal finish the current image without submitting the next style", async () => {
    for (const remove of [false, true]) {
        let finish; const requests = [];
        const ctx = setup(undefined, async (url, options) => {
            if (!options) return response(catalog());
            requests.push(JSON.parse(options.body));
            return new Promise(resolve => { finish = resolve; });
        }, { getPreviewPayload: () => ({ character_info: {}, gen_settings: {} }) });
        await find(ctx.picker.root, "vnccs-style-summary").onclick();
        const overlay = find(ctx.host, "vnccs-style-gallery");
        const button = walk(overlay).find(el => el.textContent === "Generate all previews");
        const pending = button.onclick();
        if (remove) ctx.teardown(); else await button.onclick();
        finish(previewResponse("legacy")); await pending;
        assert.equal(requests.length, 1);
        if (!remove) assert.match(find(overlay, "vnccs-style-status").textContent, /^Stopped/);
        else assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
    }
});

test("render failure keeps completed thumbnails and restores the generation button", async () => {
    let calls = 0;
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        calls++;
        return calls === 1 ? previewResponse("legacy") : { ok: false, text: async () => "Sampler failed" };
    }, { getPreviewPayload: () => ({ character_info: {}, gen_settings: {} }) });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    const button = walk(overlay).find(el => el.textContent === "Generate all previews");
    await button.onclick();
    assert.equal(calls, 2);
    assert.match(find(overlay, "vnccs-style-status").textContent, /Sampler failed.*Completed previews are saved/);
    assert.equal(button.disabled, false);
    assert.equal(walk(find(overlay, "vnccs-style-grid")).filter(el => el.tagName === "img").length, 1);
});

test("existing thumbnails restore in summary and gallery without regenerating", async () => {
    const data = catalog(); data.groups[0].styles[0].image = "/vnccs/character_styles/preview?style=legacy&v=old";
    const ctx = setup(undefined, async () => response(data));
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    assert.match(find(ctx.picker.root, "vnccs-style-preview-image").src, /v=old/);
    assert.equal(walk(find(ctx.host, "vnccs-style-grid")).filter(el => el.tagName === "img").length, 1);
});

test("a fresh picker restores saved previews from the server after page refresh", async () => {
    const stored = catalog();
    stored.preview_directory = "/ComfyUI/output/VNCCS/style_previews";
    let generated = 0;
    const fetchApi = async (url, options) => {
        if (!options) return response(structuredClone(stored));
        const id = JSON.parse(options.body).style_id;
        const result = await previewResponse(id).json();
        stored.groups[0].styles.find(style => style.id === id).image = result.image;
        generated++;
        return response(result);
    };
    const first = setup(undefined, fetchApi, { getPreviewPayload: () => ({ character_info: {}, gen_settings: {} }) });
    await find(first.picker.root, "vnccs-style-summary").onclick();
    await walk(first.host).find(el => el.textContent === "Generate all previews").onclick();
    first.teardown();
    const refreshed = setup(undefined, fetchApi);
    await find(refreshed.picker.root, "vnccs-style-summary").onclick();
    assert.equal(generated, 2);
    assert.equal(walk(find(refreshed.host, "vnccs-style-grid")).filter(el => el.tagName === "img").length, 2);
    assert.match(find(refreshed.picker.root, "vnccs-style-preview-image").src, /style=legacy/);
    assert.equal(find(refreshed.host, "vnccs-style-preview-location").textContent,
        "Preview folder: /ComfyUI/output/VNCCS/style_previews");
});

test("generation stops without a saved-file acknowledgement and shows no transient thumbnail", async () => {
    let submitted = 0;
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        submitted++;
        const result = await previewResponse("legacy").json();
        delete result.saved;
        return response(result);
    }, { getPreviewPayload: () => ({ character_info: {}, gen_settings: {} }) });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    await walk(ctx.host).find(el => el.textContent === "Generate all previews").onclick();
    assert.equal(submitted, 1);
    assert.match(find(ctx.host, "vnccs-style-status").textContent, /Server did not confirm saving.*to disk/);
    assert.equal(walk(ctx.host).some(el => el.tagName === "img"), false);
});

test("card size defaults to 130%, updates the grid without rebuilding cards and survives reopening", async () => {
    const ctx = setup();
    const trigger = find(ctx.picker.root, "vnccs-style-summary");
    await trigger.onclick();
    let overlay = find(ctx.host, "vnccs-style-gallery");
    const grid = find(overlay, "vnccs-style-grid");
    const first = grid.children[0];
    const slider = find(overlay, "vnccs-style-size-slider");
    assert.equal(slider.value, "130");
    assert.equal(grid.style["--vnccs-style-card-size"], "182px");
    assert.equal(slider.attrs["aria-label"], "Style card size");
    for (const [scale, pixels] of [[80, "112px"], [200, "280px"], [250, "350px"]]) {
        slider.value = String(scale); slider.oninput();
        assert.equal(grid.style["--vnccs-style-card-size"], pixels);
        assert.equal(find(overlay, "vnccs-style-size-value").textContent, `${scale}%`);
        assert.equal(grid.children[0], first);
    }
    overlay.onkeydown({ key: "Escape", stopPropagation() {}, preventDefault() {} });
    await trigger.onclick();
    overlay = find(ctx.host, "vnccs-style-gallery");
    assert.equal(find(overlay, "vnccs-style-size-slider").value, "250");
    assert.equal(find(overlay, "vnccs-style-grid").style["--vnccs-style-card-size"], "350px");
    assert.equal(ctx.snapshots.length, 0);
});

test("transparent previews hide the placeholder text and image failure restores it", async () => {
    const data = catalog(); data.groups[0].styles[0].image = "/vnccs/character_styles/preview?style=legacy&v=alpha";
    const ctx = setup(undefined, async () => response(data));
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const image = find(ctx.picker.root, "vnccs-style-preview-image");
    const placeholder = image.parent;
    assert.equal(placeholder.children[0].hidden, true);
    image.onerror();
    assert.equal(placeholder.children[0].hidden, false);
    assert.equal(placeholder.children.length, 1);
});
