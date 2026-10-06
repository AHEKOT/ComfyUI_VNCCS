import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { presetSelection } from "../web/character_presets.mjs";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_character_cloner.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const legacy = JSON.parse(readFileSync(new URL("../character_template/character_tags.json", import.meta.url), "utf8"));
const modern = JSON.parse(readFileSync(new URL("../character_template/character_presets_v2.json", import.meta.url), "utf8"));
const block = (text, start, end) => {
    const offset = text.indexOf(start), limit = text.indexOf(end, offset);
    assert.ok(offset >= 0 && limit > offset);
    return text.slice(offset, limit);
};
const deferred = () => {
    let resolve;
    return { promise: new Promise(done => { resolve = done; }), resolve };
};

class Element {
    constructor(tagName) {
        this.tagName = tagName;
        this.children = [];
        this.attrs = {};
        this.style = {};
        this.value = "";
        this.classList = { toggle() {}, add() {}, remove() {} };
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attrs[name] = value; }
    focus(options) { this.focusOptions = options; }
    blur() { this.onblur?.(); }
    dispatchEvent(event) { this[`on${event.type}`]?.({ target: this }); }
}

function setup(upload) {
    const state = { character: "Alice", character_info: { hair: "Black Hair, My Custom Hair" }, source_images: ["old.png"], selected_idx: 0 };
    const widget = { value: "" }, modals = [], requests = [], events = [];
    const container = new Element("div"), node = { id: 42 };
    const fileInput = new Element("input"), uploadBtn = new Element("button"), autoGenBtn = new Element("button");
    const context = createWidgetContext({
        state, node, dataWidget: widget, container, els: {}, fileInput, uploadBtn, autoGenBtn,
        presetSelection, Event, TAG_DATA: null,
        document: { createElement: tag => new Element(tag) },
        window: { dispatchEvent: event => events.push(event) }, app: { graph: { setDirtyCanvas() {} } },
        CustomEvent: class { constructor(type, data) { this.type = type; this.detail = data.detail; } },
        FormData: class { append() {} }, normalizeUploadFile: file => file,
        helpFor: () => "", setHelpText() {}, syncPoseStudioAge() {}, renderThumbs() {},
        imgList: { querySelectorAll: () => [] }, ensureQwenVLReady: async () => true,
        createLoadingOverlay: () => ({ remove() {} }), console: { log() {} },
        showCommonModal(parent, title, builder, buttons) {
            const overlay = { removed: false, remove() { this.removed = true; } };
            const modal = { style: {} }, content = builder(modal);
            modals.push({ parent, title, content, buttons, overlay, modal });
            return { overlay, modal, content };
        },
        api: { fetchApi: async (path, options) => {
            requests.push({ path, options });
            if (path === "/upload/image") return upload ? upload() : { ok: true, json: async () => ({ name: `upload-${requests.length}.png` }) };
            if (path === "/vnccs/get_tags") return { ok: true, json: async () => structuredClone(legacy) };
            if (path === "/vnccs/get_tags?catalog=creator_v2") return { ok: true, json: async () => structuredClone(modern) };
            assert.equal(path, "/vnccs/cloner_auto_generate");
            return { ok: true, json: async () => ({ hair: "silver hair", face: "freckles", eyes: "blue eyes" }) };
        } },
    });
    vm.runInContext(
        block(common, "export function registerCleanup", "// ── DOM Widget Width Sync").replaceAll("export ", "") +
        "const beginCaptionRequest = createRequestGuard(node); const beginUploadRequest = createRequestGuard(node);" +
        block(source, "const saveState =", "const parsePoseStudioValues =") +
        block(source, "const createTraitField =", "const createSegmentedField =") +
        block(source, "const updateUIFromState =", "// --- Helpers (Hoisted)") +
        block(source, "const showModal =", "const showSourceImageRequiredModal =") +
        block(source, "const openTagConstructor =", "const beginCharacterRequest =") +
        block(source, "fileInput.onchange =", "// Click anywhere on overlay triggers upload") +
        block(source, "autoGenBtn.onclick =", "// Helper: Progress Polling") +
        "this.makeField = createField; this.sync = updateUIFromState; this.prompt = showCharacterDescriptionPrompt;", context,
    );
    return { context, state, widget, modals, requests, events, container, node, fileInput, uploadBtn };
}

test("Cloner uses its own trait rows and preserves free-form edits and restoration", () => {
    const { context, state, widget, events } = setup();
    const row = context.makeField("Hair", "hair");
    const [values, input] = row.children[1].children;
    assert.equal(row.className, "vnccs-cloner-trait-row");
    assert.equal(row.children[2].textContent, "+");
    assert.equal(row.children[2].attrs["aria-label"], "Choose hair presets");
    assert.deepEqual(values.children.map(chip => chip.textContent), ["Black Hair", "My Custom Hair"]);
    values.onclick();
    assert.equal(input.hidden, false);
    assert.equal(input.focusOptions.preventScroll, true);
    input.value = "  White Hair, My Custom Hair  ";
    input.dispatchEvent(new Event("input"));
    assert.equal(JSON.parse(widget.value).character_info.hair, input.value);
    assert.equal(events.at(-1).type, "vnccs-character-cloner-updated");
    input.blur();
    assert.equal(values.hidden, false);
    state.character_info.hair = "silver hair, ribbons";
    context.sync();
    assert.deepEqual(values.children.map(chip => chip.textContent), ["silver hair", "ribbons"]);
    assert.equal(context.makeField("Aesthetics", "aesthetics").className, "vnccs-cloner-field");
});

test("presets preserve custom text and the Skin plus button opens existing skin choices", async () => {
    const { context, state, modals, requests } = setup();
    const hair = context.makeField("Hair", "hair");
    await hair.children[2].onclick();
    modals.at(-1).buttons.find(button => button.text === "APPLY").action();
    assert.equal(state.character_info.hair, "Black Hair, My Custom Hair");
    const skin = context.makeField("Skin", "skin_color");
    await skin.children[2].onclick();
    const modal = modals.at(-1);
    const chip = modal.content.children.find(chip => chip.innerText === "Golden Tan");
    assert.ok(chip);
    chip.onclick();
    modal.buttons.find(button => button.text === "APPLY").action();
    assert.equal(state.character_info.skin_color, "golden tan skin");
    assert.equal(skin.children[1].children[0].children[0].textContent, "golden tan skin");
    assert.equal(requests.filter(request => request.path.includes("catalog=creator_v2")).length, 1);
});

test("a single upload replaces the reference and offers the wizard for that image", async () => {
    const { context, fileInput, uploadBtn, state, widget, modals, requests, container } = setup();
    const face = context.makeField("Face", "face"), eyes = context.makeField("Eyes", "eyes");
    await fileInput.onchange({ target: { files: [{ name: "one.png" }] } });
    assert.equal(state.source_images.length, 1);
    assert.equal(state.selected_idx, 0);
    assert.equal(state.source_images[0].name, "upload-1.png");
    assert.equal(state.source_images_character, "Alice");
    assert.deepEqual(JSON.parse(widget.value).source_images, [{ name: "upload-1.png", type: "input", subfolder: "" }]);
    assert.equal(requests.filter(request => request.path === "/upload/image").length, 1);
    assert.equal(modals.length, 1);
    const popup = modals[0];
    assert.equal(popup.parent, container);
    assert.match(source, /\/\* ── Container ── \*\/\s*\.vnccs-cloner-container \{\s*position: relative;/);
    assert.equal(popup.modal.style.maxHeight, "calc(100% - 32px)");
    assert.equal(popup.modal.style.overflowY, "auto");
    assert.equal(popup.title, "Describe Your Character");
    assert.match(popup.content.children[0].textContent, /manual entry.*attribute fields/);
    const warning = popup.content.children[1];
    assert.equal(warning.children[0].tagName, "strong");
    assert.match(warning.children[0].textContent, /Face and eye descriptions.*consistent emotions/);
    assert.match(warning.children[1].textContent, /Missing or inaccurate.*inconsistent.*generating emotions later/);
    assert.equal(uploadBtn.disabled, false);
    assert.equal(fileInput.value, "");
    await popup.buttons.find(button => button.text === "Analyze Tags").action(popup.overlay);
    assert.equal(popup.overlay.removed, true);
    const analysis = requests.find(request => request.path === "/vnccs/cloner_auto_generate");
    assert.equal(JSON.stringify(JSON.parse(analysis.options.body).image_name), JSON.stringify(state.source_images[0]));
    assert.equal(JSON.parse(widget.value).character_info.face, "freckles");
    assert.equal(face.children[1].children[0].children[0].textContent, "freckles");
    assert.equal(eyes.children[1].children[0].children[0].textContent, "blue eyes");
});

test("multiple selected files are rejected before uploading and leave the reference intact", async () => {
    const { fileInput, state, widget, requests, modals } = setup();
    assert.match(source, /fileInput\.multiple = false;/);
    fileInput.value = "selected files";
    await fileInput.onchange({ target: { files: [{ name: "one.png" }, { name: "two.png" }] } });
    assert.equal(requests.length, 0);
    assert.deepEqual(state.source_images, ["old.png"]);
    assert.equal(widget.value, "");
    assert.equal(fileInput.value, "");
    assert.deepEqual(modals.map(modal => modal.title), ["One Image Only"]);
    assert.match(modals[0].content.textContent, /only one reference image/);
});

test("manual choice opens face editing and leaves values intact without starting analysis", async () => {
    const { context, state, modals, requests } = setup();
    context.makeField("Face", "face");
    context.prompt();
    const popup = modals[0], original = JSON.stringify(state.character_info);
    popup.buttons.find(button => button.text === "Enter Manually").action(popup.overlay);
    assert.equal(popup.overlay.removed, true);
    assert.equal(context.els.face.hidden, false);
    assert.equal(context.els.face.focusOptions.preventScroll, true);
    assert.equal(JSON.stringify(state.character_info), original);
    assert.equal(requests.length, 0);
});

test("failed or cancelled uploads never offer analysis", async () => {
    const h = setup(() => ({ ok: false, json: async () => ({ error: "Upload failed" }) }));
    await h.fileInput.onchange({ target: { files: [] } });
    assert.equal(h.modals.length, 0);
    await h.fileInput.onchange({ target: { files: [{ name: "bad.png" }] } });
    assert.deepEqual(h.modals.map(modal => modal.title), ["Upload Error"]);
    assert.deepEqual(h.state.source_images, ["old.png"]);
    assert.equal(h.uploadBtn.disabled, false);
});

for (const stale of ["character", "reference", "removed"]) {
    test(`a late upload does not update or prompt a ${stale} context`, async () => {
        const request = deferred(), h = setup(() => request.promise);
        const work = h.fileInput.onchange({ target: { files: [{ name: "late.png" }] } });
        if (stale === "character") h.state.character = "Bob";
        if (stale === "reference") h.state.source_images = ["other.png"];
        if (stale === "removed") h.node.onRemoved();
        request.resolve({ ok: true, json: async () => ({ name: "late.png" }) });
        await work;
        assert.equal(h.state.source_images.length, 1);
        assert.equal(h.modals.length, 0);
    });
    test(`popup choices cannot analyze or edit a ${stale} context`, async () => {
        const h = setup();
        h.context.makeField("Face", "face");
        h.context.prompt();
        if (stale === "character") h.state.character = "Bob";
        if (stale === "reference") h.state.source_images = ["other.png"];
        if (stale === "removed") h.node.onRemoved();
        const popup = h.modals[0];
        await popup.buttons.find(button => button.text === "Analyze Tags").action(popup.overlay);
        popup.buttons.find(button => button.text === "Enter Manually").action(popup.overlay);
        assert.equal(h.requests.length, 0);
        assert.equal(h.context.els.face.hidden, true);
    });
}
