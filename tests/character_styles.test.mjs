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
    get isConnected() { return !!this.parent?.children.includes(this) && (!this.parent.parent || this.parent.isConnected); }
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
    { id: "anime_style", label: "Anime", description: "Cel shading", reference: "Anime tradition", prompt: "Anime prompt" },
] }] });
const response = data => ({ ok: true, json: async () => data });
function setup(info = { style: "legacy" }, fetchApi = async () => response(catalog()), options = {}) {
    const document = { activeElement: null, createElement(tag) { return new Element(tag, this); } };
    globalThis.document = document;
    const host = new Element("div", document);
    const background = new Element("div", document);
    host.append(background);
    const snapshots = []; let teardown;
    const showModal = (container, title, build, buttons) => {
        const overlay = new Element("div", document);
        overlay.className = "vnccs-common-modal-overlay";
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-label", title);
        const content = build();
        overlay.append(content);
        for (const config of buttons) {
            const button = new Element("button", document);
            button.textContent = config.text;
            button.onclick = async () => {
                if (button.disabled) return;
                button.disabled = true;
                try {
                    if (!config.action || !await config.action(overlay, button)) overlay.remove();
                } finally { button.disabled = false; }
            };
            overlay.append(button);
        }
        container.append(overlay);
        return { overlay, content };
    };
    const picker = createStylePicker({ host, catalog: catalog(), getInfo: () => info,
        save: () => snapshots.push(JSON.parse(JSON.stringify(info))), fetchApi, showModal,
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
    assert.equal(ctx.info.style, "anime_style");
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
    assert.equal(ctx.picker.customInput.style.display, "none");
    assert.equal(ctx.picker.customInput.hidden, true);
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
const previewResponse = id => response({ style_id: id, image: `/vnccs/character_styles/preview?style=${id}&v=1`, width: 1024, height: 1024, saved: true });

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
    assert.equal(walk(overlay).some(el => el.textContent === "Generate all previews"), false);
    const pending = ctx.picker.generatePreviews();
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
    waiting.shift()(previewResponse("anime_style")); await pending;
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
        const pending = ctx.picker.generatePreviews();
        if (remove) ctx.teardown(); else await ctx.picker.generatePreviews();
        finish(previewResponse("legacy")); await pending;
        assert.equal(requests.length, 1);
        if (!remove) assert.match(find(overlay, "vnccs-style-status").textContent, /^Stopped/);
        else assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
    }
});

test("render failure keeps completed thumbnails and releases the renderer", async () => {
    let calls = 0;
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        calls++;
        return calls === 1 ? previewResponse("legacy") : { ok: false, text: async () => "Sampler failed" };
    }, { getPreviewPayload: () => ({ character_info: {}, gen_settings: {} }) });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    assert.equal(await ctx.picker.generatePreviews(), false);
    assert.equal(calls, 2);
    assert.match(find(overlay, "vnccs-style-status").textContent, /Sampler failed.*Completed previews are saved/);
    assert.equal(walk(overlay).find(el => el.textContent === "New style").disabled, false);
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
    stored.preview_directory = "/node/character_template/style_previews";
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
    await first.picker.generatePreviews();
    first.teardown();
    const refreshed = setup(undefined, fetchApi);
    await find(refreshed.picker.root, "vnccs-style-summary").onclick();
    assert.equal(generated, 2);
    assert.equal(walk(find(refreshed.host, "vnccs-style-grid")).filter(el => el.tagName === "img").length, 2);
    assert.match(find(refreshed.picker.root, "vnccs-style-preview-image").src, /style=legacy/);
    assert.equal(find(refreshed.host, "vnccs-style-preview-location").textContent,
        "Preview folder: /node/character_template/style_previews");
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
    await ctx.picker.generatePreviews();
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

test("Custom style opens all fields and generates only its saved preview with seed 0", async () => {
    const settings = { seed: 456, seed_mode: "randomize", target_size: 2048,
        generation_mode: "qi2", mode_settings: { qi2: { seed: 789 } }, lora_stack: [{ name: "Mine", strength: .5 }] };
    const info = { style: "custom", custom_style: "Ink", hair: "black hair", eyes: "blue eyes", framing: "full_body" };
    const id = "user_" + "c".repeat(32);
    const stored = catalog(), calls = [];
    let saved;
    const fetchApi = async (url, options) => {
        if (!options) return response(structuredClone(stored));
        const payload = JSON.parse(options.body); calls.push({ url, payload });
        if (url === "/vnccs/character_styles") {
            saved = { ...payload, id, user: true, image: saved?.image || "" };
            stored.groups = stored.groups.filter(group => group.label !== "My styles");
            stored.groups.push({ label: "My styles", styles: [saved] });
            return response({ style: structuredClone(saved) });
        }
        saved.image = (await previewResponse(id).json()).image;
        return previewResponse(id);
    };
    const ctx = setup(info, fetchApi, { getPreviewPayload: () => ({ node_id: "42", character_info: info, gen_settings: settings }) });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const overlay = find(ctx.host, "vnccs-style-gallery");
    const grid = find(overlay, "vnccs-style-grid");
    assert.equal(walk(overlay).some(el => el.textContent === "Generate all previews"), false);
    find(grid, "vnccs-style-card").onclick();
    const editor = find(overlay, "vnccs-style-editor");
    assert.equal(editor.hidden, false); assert.equal(grid.hidden, true);
    const fields = walk(editor).filter(el => ["input", "textarea"].includes(el.tagName));
    assert.equal(fields.length, 4);
    assert.deepEqual(walk(editor).filter(el => el.tagName === "label").map(el => el.textContent),
        ["Name", "Short description", "Reference", "Style prompt"]);
    assert.equal(fields[3].value, "Ink");
    fields[0].value = "My ink"; fields[1].value = "Dry brush"; fields[2].value = "My drawing";
    const generate = walk(editor).find(el => el.textContent === "Generate preview");
    await generate.onclick();
    assert.deepEqual(calls.map(call => call.url), ["/vnccs/character_styles", "/vnccs/character_styles/preview"]);
    const payload = calls[1].payload;
    assert.equal(payload.style_id, id);
    assert.equal(payload.node_id, "42");
    assert.deepEqual(payload.gen_settings, { ...settings, seed: 0, seed_mode: "fixed", mode_settings: {} });
    assert.equal(payload.character_info.hair, "black hair");
    assert.equal(payload.character_info.eyes, "blue eyes");
    assert.equal(payload.character_info.framing, "full_body"); // Backend substitutes Portrait on a copy.
    assert.equal(settings.seed, 456); assert.equal(settings.seed_mode, "randomize");
    assert.equal(settings.mode_settings.qi2.seed, 789);
    assert.equal(info.framing, "full_body");
    assert.equal(ctx.info.style, id);
    assert.equal(ctx.snapshots.at(-1).style_prompt, "Ink");
    assert.match(find(editor, "vnccs-style-preview-image").src, new RegExp(`style=${id}`));
    assert.match(find(ctx.picker.root, "vnccs-style-preview-image").src, new RegExp(`style=${id}`));
    assert.match(find(overlay, "vnccs-style-status").textContent, /Saved all 1/);
    assert.equal(editor.hidden, false); assert.equal(generate.disabled, false);
    await generate.onclick();
    assert.equal(calls[2].payload.id, id); assert.equal(calls[3].payload.style_id, id);
    await editor.onsubmit({ preventDefault() {} });
    assert.equal(calls[4].payload.id, id);
    assert.equal(calls.filter(call => call.url.endsWith("/preview")).length, 2);
    assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
    ctx.teardown();
    const restored = setup(structuredClone(info), fetchApi);
    await find(restored.picker.root, "vnccs-style-summary").onclick();
    assert.match(find(restored.picker.root, "vnccs-style-preview-image").src, new RegExp(`style=${id}`));
    assert.equal(stored.groups.find(group => group.label === "My styles").styles.length, 1);
});

test("custom preview waits for a successful save and allows retry after a render error", async () => {
    const id = "user_" + "d".repeat(32);
    const calls = []; let saveFails = true, renderFails = true;
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        const payload = JSON.parse(options.body); calls.push({ url, payload });
        if (url.endsWith("/preview")) return renderFails ? { ok: false, text: async () => "Sampler failed" } : previewResponse(id);
        return saveFails ? { ok: false, json: async () => ({ error: "Disk full" }) }
            : response({ style: { ...payload, id, user: true } });
    }, { getPreviewPayload: () => ({ character_info: {}, gen_settings: { seed: 22 } }) });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    walk(ctx.host).find(el => el.textContent === "New style").onclick();
    const editor = find(ctx.host, "vnccs-style-editor");
    const fields = walk(editor).filter(el => ["input", "textarea"].includes(el.tagName));
    fields[0].value = "New ink"; fields[3].value = "Ink";
    const generate = walk(editor).find(el => el.textContent === "Generate preview");
    await generate.onclick();
    assert.equal(calls.length, 1);
    assert.match(find(ctx.host, "vnccs-style-status").textContent, /Disk full/);
    assert.equal(generate.disabled, false); assert.ok(fields.every(field => !field.disabled));
    saveFails = false;
    await generate.onclick();
    assert.match(find(ctx.host, "vnccs-style-status").textContent, /Sampler failed/);
    assert.equal(generate.disabled, false); assert.ok(fields.every(field => !field.disabled));
    renderFails = false;
    await generate.onclick();
    assert.equal(calls[3].payload.id, id);
    assert.match(find(editor, "vnccs-style-preview-image").src, new RegExp(`style=${id}`));
});

test("closing a custom editor discards a pending save and never starts its render", async () => {
    let finish; const calls = [];
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(catalog());
        calls.push(url);
        return new Promise(resolve => { finish = resolve; });
    }, { getPreviewPayload: () => ({ character_info: {}, gen_settings: {} }) });
    const trigger = find(ctx.picker.root, "vnccs-style-summary");
    await trigger.onclick();
    walk(ctx.host).find(el => el.textContent === "New style").onclick();
    const editor = find(ctx.host, "vnccs-style-editor");
    const fields = walk(editor).filter(el => ["input", "textarea"].includes(el.tagName));
    fields[0].value = "Pending"; fields[3].value = "Ink";
    const pending = walk(editor).find(el => el.textContent === "Generate preview").onclick();
    // A duplicate click cannot submit a second write.
    await walk(editor).find(el => el.textContent === "Generate preview").onclick();
    find(ctx.host, "vnccs-style-gallery").onkeydown({ key: "Escape", stopPropagation() {}, preventDefault() {} });
    await trigger.onclick();
    finish(response({ style: { id: "user_" + "e".repeat(32), label: "Pending", prompt: "Ink", user: true } }));
    await pending;
    assert.deepEqual(calls, ["/vnccs/character_styles"]);
    assert.equal(ctx.info.style, "legacy");
    assert.equal(ctx.snapshots.length, 0);
    assert.equal(find(ctx.host, "vnccs-style-editor").hidden, true);
});

test("legacy aliases select the surviving style and packaged thumbnail", async () => {
    const data = catalog();
    data.aliases = { clio_anime_style: "anime_style", clio_toon_shader: "anime_style" };
    data.groups[0].styles[1].image = "/vnccs/character_styles/preview?style=anime_style&v=packaged";
    const ctx = setup({ style: "clio_toon_shader" }, async () => response(data));
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    assert.equal(ctx.info.style, "anime_style");
    assert.equal(find(ctx.picker.root, "vnccs-style-name").textContent, "Anime");
    assert.match(find(ctx.picker.root, "vnccs-style-preview-image").src, /v=packaged/);
});

const userStyle = () => ({ id: "user_" + "a".repeat(32), label: "<b>My ink</b>", description: "Dry brush",
    prompt: "Ink", reference: "Me", image: "/vnccs/character_styles/preview?style=user_" + "a".repeat(32), user: true });
const userCatalog = () => { const data = catalog(); data.groups.push({ label: "My styles", styles: [userStyle()] }); return data; };

test("only user cards have a delete cross; cancelling confirmation makes no request", async () => {
    const calls = [];
    const ctx = setup(undefined, async (url, options) => {
        if (!options) return response(userCatalog());
        calls.push(url); return response({});
    });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const grid = find(ctx.host, "vnccs-style-grid");
    assert.equal(walk(grid).filter(el => el.className === "vnccs-style-delete").length, 1);
    const cross = find(grid, "vnccs-style-delete");
    assert.equal(cross.textContent, "×");
    assert.equal(cross.attrs["aria-label"], "Delete <b>My ink</b>");
    let stopped = false;
    cross.onclick({ stopPropagation() { stopped = true; } });
    assert.equal(stopped, true);
    const dialog = find(ctx.host, "vnccs-common-modal-overlay");
    assert.equal(dialog.attrs.role, "dialog");
    assert.equal(dialog.attrs["aria-label"], "Delete style");
    assert.match(dialog.children[0].children[0].textContent, /<b>My ink<\/b>/);
    assert.match(dialog.children[0].children[0].textContent, /style and preview.*permanently/);
    assert.equal(calls.length, 0);
    await walk(dialog).find(el => el.textContent === "Cancel").onclick();
    assert.equal(dialog.isConnected, false);
    assert.equal(find(grid, "vnccs-style-delete"), cross);
    assert.equal(ctx.info.style, "legacy");
    assert.equal(ctx.snapshots.length, 0);
    assert.equal(calls.length, 0);
});

test("confirmed deletion removes the card and resets only a deleted current selection", async () => {
    for (const selected of [false, true]) {
        const style = userStyle(), calls = [];
        const ctx = setup({ style: selected ? style.id : "anime_style", style_prompt: selected ? "Ink" : "Anime prompt" }, async (url, options) => {
            if (!options) return response(userCatalog());
            calls.push({ url, options }); return response({ deleted: true, style_id: style.id });
        });
        await find(ctx.picker.root, "vnccs-style-summary").onclick();
        find(ctx.host, "vnccs-style-delete").onclick();
        const dialog = find(ctx.host, "vnccs-common-modal-overlay");
        await walk(dialog).find(el => el.textContent === "Delete").onclick();
        assert.equal(calls.length, 1);
        assert.equal(calls[0].url, `/vnccs/character_styles/delete?style=${style.id}`);
        assert.equal(calls[0].options.method, "POST");
        assert.equal(calls[0].options.headers["X-VNCCS-CSRF"], "1");
        assert.equal(find(ctx.host, "vnccs-style-delete"), undefined);
        assert.equal(find(ctx.host, "vnccs-style-grid").children.length, 3);
        assert.equal(walk(ctx.host).some(el => el.tagName === "option" && el.value === "My styles"), false);
        assert.equal(dialog.isConnected, false);
        assert.match(find(ctx.host, "vnccs-style-status").textContent, /^Deleted:/);
        assert.equal(ctx.info.style, selected ? "legacy" : "anime_style");
        assert.equal(ctx.info.style_prompt, selected ? "Legacy prompt" : "Anime prompt");
        assert.equal(ctx.snapshots.length, selected ? 1 : 0);
        if (selected) assert.equal(find(ctx.picker.root, "vnccs-style-preview-image"), undefined);
    }
});

test("failed deletion leaves the record visible and allows retry in the same modal", async () => {
    let fail = true;
    const style = userStyle();
    const ctx = setup({ style: style.id }, async (url, options) => {
        if (!options) return response(userCatalog());
        return fail ? { ok: false, json: async () => ({ error: "Read only" }) } : response({ deleted: true, style_id: style.id });
    });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    find(ctx.host, "vnccs-style-delete").onclick();
    const dialog = find(ctx.host, "vnccs-common-modal-overlay");
    const button = walk(dialog).find(el => el.textContent === "Delete");
    await button.onclick();
    assert.match(find(dialog, "vnccs-style-status").textContent, /Read only/);
    assert.equal(dialog.isConnected, true);
    assert.equal(button.disabled, false);
    assert.equal(find(ctx.host, "vnccs-style-delete").disabled, false);
    assert.equal(ctx.info.style, style.id);
    assert.equal(ctx.snapshots.length, 0);
    fail = false;
    await button.onclick();
    assert.equal(dialog.isConnected, false);
    assert.equal(ctx.info.style, "legacy");
});

test("duplicate confirmations send one request and removal ignores its late response", async () => {
    let finish; const calls = [];
    const style = userStyle();
    const ctx = setup({ style: style.id }, async (url, options) => {
        if (!options) return response(userCatalog());
        calls.push(url); return new Promise(resolve => { finish = resolve; });
    });
    await find(ctx.picker.root, "vnccs-style-summary").onclick();
    const cross = find(ctx.host, "vnccs-style-delete");
    cross.onclick(); cross.onclick();
    assert.equal(walk(ctx.host).filter(el => el.className === "vnccs-common-modal-overlay").length, 1);
    const dialog = find(ctx.host, "vnccs-common-modal-overlay");
    const button = walk(dialog).find(el => el.textContent === "Delete");
    const pending = button.onclick();
    await button.onclick();
    assert.equal(cross.disabled, true);
    assert.equal(calls.length, 1);
    ctx.teardown();
    assert.equal(dialog.isConnected, false);
    finish(response({ deleted: true, style_id: style.id }));
    await pending;
    assert.equal(ctx.snapshots.length, 0);
    assert.equal(find(ctx.host, "vnccs-style-gallery"), undefined);
});

test("a deletion finishing after reopening cannot be undone by an older catalog refresh", async () => {
    let finishDelete, finishRefresh; let reads = 0;
    const style = userStyle();
    const ctx = setup({ style: style.id }, async (url, options) => {
        if (options) return new Promise(resolve => { finishDelete = resolve; });
        if (++reads === 1) return response(userCatalog());
        return new Promise(resolve => { finishRefresh = resolve; });
    });
    const trigger = find(ctx.picker.root, "vnccs-style-summary");
    await trigger.onclick();
    find(ctx.host, "vnccs-style-delete").onclick();
    const pending = walk(find(ctx.host, "vnccs-common-modal-overlay")).find(el => el.textContent === "Delete").onclick();
    find(ctx.host, "vnccs-style-gallery").onkeydown({ key: "Escape", preventDefault() {}, stopPropagation() {} });
    const refresh = trigger.onclick();
    finishDelete(response({ deleted: true, style_id: style.id })); await pending;
    finishRefresh(response(userCatalog())); await refresh;
    assert.equal(find(ctx.host, "vnccs-style-delete"), undefined);
    assert.equal(ctx.info.style, "legacy");
    assert.equal(walk(ctx.host).find(el => el.textContent === "New style").disabled, false);
});
