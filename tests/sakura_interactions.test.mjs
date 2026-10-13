import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { createWidgetContext } from "./widget_context.mjs";

const creator = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const generator = readFileSync(new URL("../web/vnccs_character_generator.js", import.meta.url), "utf8");
const section = (start, end) => creator.slice(creator.indexOf(start), creator.indexOf(end, creator.indexOf(start)));
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };

function setup() {
    const document = { animations: [], frames: [], observers: [], motion: true };
    class Element {
        constructor(tag) {
            this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.style = { setProperty(key, value) { this[key] = value; } }; this.className = "";
            this.scrollTop = this.scrollLeft = 0; this.clientWidth = 500; this.clientHeight = 700;
            this.classList = {
                contains: name => this.className.split(" ").includes(name),
                add: (...names) => { for (const name of names) if (!this.classList.contains(name)) this.className += ` ${name}`; },
                toggle: (name, enabled) => { this.className = this.className.split(" ").filter(value => value !== name).join(" "); if (enabled) this.classList.add(name); },
            };
        }
        get parentElement() { return this.parentNode; }
        get isConnected() { return this === document.body || !!this.parentNode?.isConnected; }
        get options() { return this.children.filter(child => child.tagName === "OPTION"); }
        get value() { return this.tagName === "SELECT" ? this.options.find(option => option._value === this._value)?.value || "" : this._value || ""; }
        set value(value) { this._value = String(value); }
        get textContent() { return (this._text || "") + this.children.map(child => child.textContent).join(""); }
        set textContent(value) { for (const child of [...this.children]) child.remove(); this._text = String(value); }
        get innerText() { return this.textContent; }
        set innerText(value) { this.textContent = value; }
        set innerHTML(value) { for (const child of [...this.children]) child.remove(); this._html = value; }
        get firstChild() { return this.children[0]; }
        appendChild(child) { child.remove(); this.children.push(child); child.parentNode = this; return child; }
        replaceChild(next, previous) { const index = this.children.indexOf(previous); next.remove(); this.children[index] = next; next.parentNode = this; previous.parentNode = null; }
        append(...children) { children.forEach(child => typeof child === "string" ? this._text = (this._text || "") + child : this.appendChild(child)); }
        add(option) { this.appendChild(option); if (this._value === undefined) this.value = option.value; }
        after(child) { child.remove(); const parent = this.parentNode; parent.children.splice(parent.children.indexOf(this) + 1, 0, child); child.parentNode = parent; }
        remove(index) {
            if (typeof index === "number") { this.options[index]?.remove(); return; }
            if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(child => child !== this);
            this.parentNode = null;
        }
        contains(element) { return element === this || this.children.some(child => child.contains(element)); }
        setAttribute(name, value) { this[name] = String(value); }
        getAttribute(name) { return this[name] ?? null; }
        removeAttribute(name) { delete this[name]; }
        addEventListener(type, callback) { this[`listener_${type}`] = callback; }
        removeEventListener(type, callback) { if (this[`listener_${type}`] === callback) delete this[`listener_${type}`]; }
        dispatchEvent(event) { event.target = this; this[`on${event.type}`]?.(event); this[`listener_${event.type}`]?.(event); }
        focus(options) { document.activeElement = this; this.focusOptions = options; }
        matches(selector) {
            if (selector.includes(",")) return selector.split(",").some(part => this.matches(part.trim()));
            if (selector === "[inert]") return !!this.inert;
            if (selector === "[id]") return !!this.id;
            const data = selector.match(/^\[data-([\w-]+)\]$/);
            if (data) return data[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()) in this.dataset;
            if (selector.startsWith(".")) return selector.slice(1).split(".").every(name => this.classList.contains(name));
            return selector === "*" || this.tagName === selector.toUpperCase();
        }
        closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector) || null; }
        querySelectorAll(selector) {
            const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll("*")]);
            return descendants.filter(element => selector.split(",").some(part => element.matches(part.trim())));
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        getBoundingClientRect() { const rect = this.rect || { left: 40, top: 100, width: 140, height: 28 }; return { ...rect, right: rect.left + rect.width, bottom: rect.top + rect.height }; }
        getClientRects() { for (let element = this; element; element = element.parentNode) if (element.hidden || element.style.display === "none") return []; return this.isConnected ? [this.getBoundingClientRect()] : []; }
        cloneNode(deep) {
            const clone = new Element(this.tagName); clone.className = this.className; clone._text = this._text; clone._value = this._value;
            clone.style = { ...this.style }; clone.dataset = { ...this.dataset }; clone.src = this.src;
            if (deep) for (const child of this.children) clone.appendChild(child.cloneNode(true));
            return clone;
        }
        animate(keyframes, options) {
            const animation = { effect: { target: this, updateTiming(timing) { Object.assign(options, timing); } }, keyframes, options, finish() { this.onfinish?.(); }, cancel() { this.oncancel?.(); } };
            document.animations.push(animation); return animation;
        }
    }
    document.createElement = tag => new Element(tag);
    document.body = new Element("body"); document.activeElement = document.body;
    const context = createWidgetContext({ document,
        window: { matchMedia: () => ({ matches: !document.motion }), getComputedStyle: element => ({ opacity: element.style.opacity || "1" }) },
        requestAnimationFrame: callback => document.frames.push(callback),
        MutationObserver: class { constructor(callback) { this.callback = callback; document.observers.push(this); } observe() {} disconnect() { this.disconnected = true; } },
        Event: class { constructor(type, options) { this.type = type; Object.assign(this, options); } },
    });
    vm.runInContext(common.slice(common.indexOf("export function registerCleanup"), common.indexOf("// ── DOM Widget Width Sync")).replaceAll("export ", ""), context);
    const root = new Element("div"); root.className = "vnccs-creator-container vnccs-common-ui-host";
    root.rect = { left: 0, top: 0, width: 500, height: 700 }; document.body.appendChild(root);
    const node = { id: 7 };
    const ui = context.createSakuraInteractions(root, node);
    const frame = new Element("div"); frame.className = "vnccs-creator-preview-container vnccs-common-ui-preview-host"; root.appendChild(frame);
    const image = new Element("img"); image.style.display = "none"; frame.appendChild(image);
    const placeholder = new Element("div"); frame.appendChild(placeholder);
    for (const name of ["title", "hint"]) { const part = new Element("div"); part.className = `vnccs-creator-placeholder-${name}`; placeholder.appendChild(part); }
    const aura = new Element("div"); aura.className = "vnccs-creator-preview-aura"; frame.appendChild(aura);
    const flush = () => { for (const callback of document.frames.splice(0)) callback(); };
    return { document, context, root, node, ui, frame, image, placeholder, aura, Element, flush };
}

for (const [name, flags] of [["Character", {}], ["Clone", { isClone: true }], ["Clothes", { isClothes: true }], ["Emotions", { isEmotions: true }]]) {
    test(`${name} Generator uses Sakura controls and preserves its view during progress refreshes`, async () => {
        const h = setup(); h.root.className = "vnccs-pipe-root vnccs-common-ui-host";
        h.context.app = { graph: { _nodes: [], setDirtyCanvas() {} }, registerExtension() {} };
        h.context.setHelpText = () => {};
        vm.runInContext(generator.replace(/^import .*;\n/gm, "") + '\nthis.Generator = CharacterGeneratorWidget; this.defaults = DEFAULT_DATA;', h.context);
        const widget = Object.assign(Object.create(h.context.Generator.prototype), {
            ...flags, node: { id: 7, widgets: [{ name: "widget_data", value: "{}" }] }, root: h.root, _ui: h.ui,
            data: JSON.parse(JSON.stringify(h.context.defaults)), title: `${name} Generator`, nodeDefs: {}, fieldDrafts: new Map(),
            stages: [["pose_generation", "Poses"], ["upscaler", "Upscaler"]], selectedPreview: "pose_generation",
            stageState: { pose_generation: { status: "running", current: 1, total: 10, images: ["sprite.png"] }, upscaler: { status: "waiting" } },
            syncCharacterSourceData() {}, syncStagesFromData() {}, schedulePreviewGridLayout() {}, saveBrowserState() {},
            connectedEmotionStudioMode: () => "qi2", seedvrDownloads: {}, seedvrAssets: { models: [] },
        });
        for (const [field, key] of [["settingsEl", "settings"], ["previewEl", "preview"], ["chainEl", "stages"]]) {
            widget[field] = new h.Element("div"); widget[field].dataset.sakuraScroll = key; h.root.appendChild(widget[field]);
        }
        widget.settingsEl.scrollTop = 173; widget.settingsEl.scrollLeft = 8;
        const options = widget.modeTabs("upscaler", "mode", [["seedvr", "SeedVR"], ["off", "OFF"]]);
        assert.equal(options.style["--segment-index"], "0");
        widget.renderSettings = () => {};
        options.children[1].onclick();
        const off = widget.modeTabs("upscaler", "mode", [["seedvr", "SeedVR"], ["off", "OFF"]]);
        assert.equal(off.style["--segment-index"], "1");
        assert.equal(off.children[0].getAttribute("aria-pressed"), "false");
        assert.equal(off.children[1].getAttribute("aria-pressed"), "true");
        assert.equal(off.children[1].classList.contains("is-active"), true);
        assert.equal(JSON.parse(widget.node.widgets[0].value).upscaler.mode, "off");
        off.children[0].onclick();
        const on = widget.modeTabs("upscaler", "mode", [["seedvr", "SeedVR"], ["off", "OFF"]]);
        assert.equal(on.style["--segment-index"], "0");
        assert.equal(on.children[0].getAttribute("aria-pressed"), "true");
        assert.equal(on.children[1].getAttribute("aria-pressed"), "false");
        assert.equal(JSON.parse(widget.node.widgets[0].value).upscaler.mode, "seedvr");
        assert.doesNotMatch(generator, /\.vnccs-pipe-mode-tab\.is-selected\s*\{/);
        const field = widget.field("upscaler", "color_correction", "Color correction", "select", ["lab", "adain"]);
        widget.settingsEl.appendChild(field); h.ui.syncSelects();
        const trigger = field.querySelector(".vnccs-common-ui-select-trigger");
        trigger.onclick(); h.ui._dropdown.menu.children[1].onclick();
        assert.equal(widget.data.upscaler.color_correction, "adain");
        assert.equal(JSON.parse(widget.node.widgets[0].value).upscaler.color_correction, "adain");
        const empty = widget.field("upscaler", "model", "Model", "select", []); widget.settingsEl.appendChild(empty); h.ui.syncSelects();
        assert.equal(empty.querySelector(".vnccs-common-ui-select-trigger").disabled, true);
        widget.renderPreview(); widget.renderChain(); h.flush();
        await widget.previewGrid.children[0]._pendingImage.onload();
        const reveals = () => h.document.animations.filter(animation => animation.options.duration === 420).length;
        assert.equal(reveals(), 1);
        widget.renderPreview(); h.flush(); assert.equal(reveals(), 1, "The same images do not replay their reveal on each progress event");
        widget.stageState.pose_generation.current = 6; widget.renderChain(); h.flush();
        assert.ok(h.document.animations.some(animation => animation.options.duration === 360 && animation.keyframes[0].transform === "scaleX(0.1)" && animation.keyframes[1].transform === "scaleX(0.6)"));
        assert.equal(widget.settingsEl.scrollTop, 173); assert.equal(widget.settingsEl.scrollLeft, 8);
        h.document.motion = false; const count = h.document.animations.length;
        widget.stageState.pose_generation.images = ["new-sprite.png"]; widget.renderPreview(); h.flush();
        assert.equal(h.document.animations.length, count);
        h.ui.dispose(); assert.equal(h.document.observers[0].disconnected, true);
    });
}

test("generator gallery reserves aligned slots and reveals decoded cards in order without rebuilding on progress", async () => {
    const h = setup();
    h.context.app = { registerExtension() {} };
    vm.runInContext(generator.replace(/^import .*;\n/gm, "") + '\nthis.Generator = CharacterGeneratorWidget;', h.context);
    const preview = new h.Element("div"); h.root.appendChild(preview);
    const widget = Object.assign(Object.create(h.context.Generator.prototype), {
        previewEl: preview, _ui: h.ui, syncStagesFromData() {},
        stages: [["pose_generation", "Poses"], ["upscaler", "Upscaler"]], selectedPreview: "pose_generation",
        stageState: { pose_generation: { status: "running", current: 0, total: 4, images: ["a", "b", "c", "d"] }, upscaler: { status: "waiting" } },
        schedulePreviewGridLayout() {},
    });
    widget.renderPreview();
    const grid = widget.previewGrid, tiles = [...grid.children], head = preview.children[0];
    assert.equal(tiles.length, 4);
    assert.equal(tiles[0].classList.contains("is-processing"), true);
    assert.equal(tiles[1].classList.contains("is-waiting"), true);
    await tiles[3]._pendingImage.onload(); await tiles[1]._pendingImage.onload();
    assert.equal(h.document.animations.filter(a => a.options.duration === 420).length, 0, "Out-of-order downloads cannot overtake the first card");
    await tiles[0]._pendingImage.onload(); await tiles[2]._pendingImage.onload();
    const reveal = h.document.animations.filter(a => a.options.duration === 420);
    assert.deepEqual(reveal.map(a => a.effect.target.parentNode), tiles);
    assert.ok(reveal.every((a, index) => index === 0 || a.options.delay > reveal[index - 1].options.delay));
    assert.equal(reveal[0].options.fill, "backwards", "Delayed cards stay hidden until their turn");
    widget.stageState.pose_generation.current = 1; widget.renderPreview();
    assert.equal(widget.previewGrid, grid); assert.deepEqual([...grid.children], tiles); assert.equal(preview.children[0], head);
    assert.equal(tiles[1].classList.contains("is-processing"), true);
    assert.equal(tiles[0].classList.contains("is-waiting"), false);
    widget.stageState.pose_generation.processing_index = 0; widget.renderPreview();
    assert.equal(tiles[0].classList.contains("is-processing"), true, "QI2 phases remain on the same image even when the phase count advances");
    widget.selectedPreview = "upscaler";
    widget.stageState.upscaler = { status: "running", current: 0, total: 4 };
    widget.renderPreview();
    assert.deepEqual([...grid.children], tiles); assert.deepEqual(Array.from(widget.currentImages()), ["a", "b", "c", "d"]);
    widget.stageState.upscaler.processing_indices = [2, 3]; widget.renderPreview();
    assert.equal(tiles[2].classList.contains("is-processing"), true);
    assert.equal(tiles[3].classList.contains("is-processing"), true);
    assert.equal(tiles[0].classList.contains("is-waiting"), false, "A later emotion batch does not mark finished cards as waiting");
    widget.regenerateState = { activeStage: "upscaler", imageIndex: 2, targetStages: ["upscaler"] };
    widget.stageState.upscaler = { status: "running", current: 0, total: 1, images: ["replacement"] };
    widget.renderPreview();
    assert.equal(tiles[2].classList.contains("is-processing"), true);
    assert.equal(tiles[0]._pendingImage, null);
    const old = tiles[2]._pendingImage;
    const staleLoad = old.onload;
    widget.stageState.upscaler.images = ["latest"]; widget.renderPreview(); await staleLoad();
    assert.equal(tiles[2].dataset.src, "c", "A stale decode never replaces the visible image");
    let viewerUpdates = 0;
    widget.viewer = { open: true };
    widget.syncViewerImage = () => { viewerUpdates++; };
    await tiles[2]._pendingImage.onload();
    assert.equal(tiles[2].dataset.src, "latest");
    assert.equal(viewerUpdates, 1, "An open viewer follows a successful replacement after decoding");
    assert.deepEqual(Array.from(widget.currentImages()), ["a", "b", "latest", "d"]);
    widget.stageState.upscaler.images = ["broken"]; widget.renderPreview(); tiles[2]._pendingImage.onerror();
    assert.equal(tiles[2].dataset.src, "latest", "A failed replacement keeps the last successful image");
    grid.clientWidth = 480; grid.clientHeight = 360; widget.layoutPreviewGrid(grid);
    assert.match(grid.style.gridTemplateColumns, /minmax\(0, 1fr\)/);
    assert.ok(tiles.every(tile => !tile.style.width && !tile.style.height), "Mixed image dimensions cannot offset individual frames");
});

test("Creator native selectors keep their state while shared dropdowns handle choices and dynamic inventory", () => {
    const h = setup();
    const select = new h.Element("select"); select.className = "vnccs-creator-select"; select.ariaLabel = "Character";
    const add = name => { const option = new h.Element("option"); option.value = option.textContent = name; select.add(option); };
    add("Alice"); add("Bob"); select.value = "Alice"; h.root.appendChild(select);
    const saved = [];
    select.onchange = event => saved.push(event.target.value);
    h.ui.syncSelects();
    const trigger = h.root.querySelector(".vnccs-common-ui-select-trigger");
    assert.equal(trigger.ariaLabel, "Character"); assert.equal(trigger.value, "Alice");
    assert.equal(select.hidden, true); assert.equal(select.tabIndex, -1); assert.deepEqual(saved, []);
    trigger.onclick(); h.ui._dropdown.menu.scrollTop = 83; h.ui._dropdown.menu.scrollLeft = 5;
    h.ui._dropdown.menu.children[1].onclick();
    assert.equal(select.value, "Bob"); assert.deepEqual(saved, ["Bob"]);
    trigger.onclick(); assert.equal(h.ui._dropdown.menu.scrollTop, 83); assert.equal(h.ui._dropdown.menu.scrollLeft, 5);
    h.ui._closeDropdown(); add("New character"); h.document.observers[0].callback();
    trigger.onclick(); assert.equal(h.ui._dropdown.menu.children.length, 3);
    h.ui.dispose(); assert.equal(h.document.observers[0].disconnected, true);
});

test("shared selectors follow disabled and visibility changes and retain inventory refresh callbacks", () => {
    const h = setup(), select = new h.Element("select"), option = new h.Element("option");
    select.className = "em-select"; select.ariaLabel = "Character";
    option.value = option.textContent = "Alice"; select.add(option); h.root.appendChild(select);
    let focused = 0, refreshed = 0;
    select.onfocus = () => { focused++; };
    select.onpointerdown = () => { refreshed++; };
    h.ui.syncSelects();
    const trigger = h.root.querySelector(".vnccs-common-ui-select-trigger");
    trigger.onfocus({}); trigger.onpointerdown({});
    assert.equal(focused, 1); assert.equal(refreshed, 1);
    select.disabled = true; select.style.cssText = "display: none;";
    h.document.observers[0].callback();
    assert.equal(trigger.disabled, true); assert.equal(trigger.style.cssText, "display: none;");
    select.disabled = false; select.style.cssText = "";
    h.ui.syncSelects(); h.ui.syncSelects();
    assert.equal(trigger.disabled, false); assert.equal(trigger.style.cssText, "");
    assert.equal(h.root.querySelectorAll(".vnccs-common-ui-select-trigger").length, 1);
});

for (const [name, flags] of [["Character", {}], ["Clone", { isClone: true }], ["Clothes", { isClothes: true }]]) {
    test(`${name} Generator keeps its animated upscaler panels through rapid mode changes and model selection`, () => {
        const h = setup(); h.root.className = "vnccs-pipe-root vnccs-common-ui-host";
        h.context.app = { graph: { _nodes: [], setDirtyCanvas() {} }, registerExtension() {} };
        h.context.setHelpText = () => {};
        vm.runInContext(generator.replace(/^import .*;\n/gm, "") + '\nthis.Generator = CharacterGeneratorWidget; this.defaults = DEFAULT_DATA;', h.context);
        const first = h.context.defaults.upscaler.model, second = "seedvr2_7b_fp16.safetensors";
        const widget = Object.assign(Object.create(h.context.Generator.prototype), {
            ...flags, node: { id: 7, widgets: [{ name: "widget_data", value: "{}" }] }, root: h.root, _ui: h.ui,
            data: JSON.parse(JSON.stringify(h.context.defaults)), title: name, nodeDefs: {}, fieldDrafts: new Map(),
            syncCharacterSourceData() {}, syncStagesFromData() {}, saveBrowserState() {}, seedvrDownloads: {},
            seedvrAssets: { models: [first, second].map(name => ({ name, local_path: `models/diffusion_models/${name}`, status: "installed" })) },
        });
        widget.data.upscaler.mode = "off";
        widget.settingsEl = new h.Element("div"); h.root.appendChild(widget.settingsEl);
        widget.renderSettings();
        const panel = widget.upscalerPanel, pickerPanel = widget.seedvrPickerPanel;
        const tabs = widget.settingsEl.querySelector(".vnccs-pipe-mode-tabs");
        assert.equal(panel.inert, true); assert.equal(panel.classList.contains("is-open"), false);
        for (const index of [0, 1, 0, 1, 0]) {
            tabs.children[index].onclick();
            assert.equal(widget.upscalerPanel, panel, "The collapsing element must not be removed or replaced");
            assert.equal(widget.seedvrPickerPanel, pickerPanel);
            assert.equal(panel.inert, index === 1); assert.equal(panel.ariaHidden, String(index === 1));
            assert.equal(panel.classList.contains("is-open"), index === 0);
            assert.equal(JSON.parse(widget.node.widgets[0].value).upscaler.mode, index === 0 ? "seedvr" : "off");
        }
        assert.ok(panel.querySelector(".vnccs-common-ui-select-trigger"), "Opening a restored OFF panel must enhance its previously inert selects");
        const picker = pickerPanel.parentElement;
        picker.children[0].onclick(); assert.equal(pickerPanel.inert, false);
        const selected = pickerPanel.querySelectorAll(".vnccs-seedvr-card")[1];
        selected.onclick();
        assert.equal(widget.seedvrPickerPanel, pickerPanel); assert.equal(pickerPanel.inert, true);
        assert.equal(widget.data.upscaler.model, second); assert.equal(JSON.parse(widget.node.widgets[0].value).upscaler.model, second);
        assert.equal(picker.children[0].dataset.seedvrModel, second);
        picker.children[0].onclick(); assert.equal(pickerPanel.inert, false);
        picker.children[0].onclick(); assert.equal(pickerPanel.inert, true);
        h.ui.dispose();
        const css = generator.match(/const CSS = `([\s\S]*?)`;/)[1];
        assert.match(css, /\.vnccs-pipe-disclosure\s*\{[^}]*grid-template-rows: 0fr;[^}]*transition: grid-template-rows 320ms/);
        assert.match(css, /\.vnccs-pipe-disclosure\.is-open\s*\{[^}]*grid-template-rows: 1fr/);
        assert.match(css, /prefers-reduced-motion: reduce/);
    });
}

test("Generator viewer keeps focus inside its opening and replacement overlays", () => {
    const h = setup(); h.root.className = "vnccs-pipe-root vnccs-common-ui-host";
    h.context.app = { registerExtension() {} };
    h.context.ResizeObserver = class { observe() {} disconnect() {} };
    h.context.cancelAnimationFrame = () => {};
    h.context.window.addEventListener = h.context.window.removeEventListener = () => {};
    h.document.addEventListener = h.document.removeEventListener = () => {};
    vm.runInContext(generator.replace(/^import .*;\n/gm, "") + '\nthis.Generator = CharacterGeneratorWidget;', h.context);
    const widget = Object.assign(Object.create(h.context.Generator.prototype), {
        root: h.root, _ui: h.ui, stages: [["poses", "Poses"], ["upscaler", "Upscaler"]], selectedPreview: "poses",
        viewer: { index: 0 }, currentImages: () => ["sprite.png"], currentViewerFocus: () => null, fitViewer() {},
    });
    const behind = new h.Element("input"); behind.dataset.focusKey = "settings-field";
    h.root.appendChild(behind); behind.focus();
    widget.renderViewer(); h.flush();
    assert.equal(h.document.activeElement, widget.viewer.overlay, "Opening must not restore focus behind the viewer");
    widget.viewer.stageTabs.children[1].focus(); widget.selectedPreview = "upscaler";
    widget.renderViewer(); h.flush();
    assert.equal(h.document.activeElement, widget.viewer.stageTabs.children[1], "Replacement must ignore the inert exit clone");
    widget.closeViewer(); h.ui.dispose();
});

test("replacement fields keep their draft selection and focus without moving their scroll region", () => {
    const h = setup(), region = new h.Element("div"), input = new h.Element("textarea");
    region.dataset.sakuraScroll = "design"; region.scrollTop = 143; region.scrollLeft = 6;
    input.dataset.focusKey = "costume-top"; input.value = "white shirt";
    input.selectionStart = 2; input.selectionEnd = 8; input.selectionDirection = "backward";
    region.appendChild(input); h.root.appendChild(region); input.focus();
    const view = h.ui.captureView(); input.remove();
    const replacement = new h.Element("textarea"); replacement.dataset.focusKey = "costume-top";
    replacement.value = input.value; replacement.selectionStart = 0;
    replacement.setSelectionRange = (start, end, direction) => Object.assign(replacement, { selectionStart: start, selectionEnd: end, selectionDirection: direction });
    region.appendChild(replacement); region.scrollTop = region.scrollLeft = 0;
    h.ui.restoreView(view); h.flush();
    assert.equal(h.document.activeElement, replacement); assert.equal(replacement.focusOptions.preventScroll, true);
    assert.equal(replacement.value, "white shirt"); assert.equal(replacement.selectionStart, 2);
    assert.equal(replacement.selectionEnd, 8); assert.equal(replacement.selectionDirection, "backward");
    assert.equal(region.scrollTop, 143); assert.equal(region.scrollLeft, 6);
});

test("shorter filtered content retains the previous position for navigation back", () => {
    const h = setup(), region = new h.Element("div");
    region.dataset.sakuraScroll = "emotions"; h.root.appendChild(region);
    let limit = 400, top = 230;
    Object.defineProperty(region, "scrollTop", { get: () => top, set: value => { top = Math.min(limit, value); } });
    const before = h.ui.captureView(); limit = 20;
    h.ui.restoreView(before); h.flush(); assert.equal(region.scrollTop, 20);
    const filtered = h.ui.captureView(); limit = 400;
    h.ui.restoreView(filtered); h.flush(); assert.equal(region.scrollTop, 230);
    region.scrollTop = 175;
    const moved = h.ui.captureView();
    h.ui.restoreView(moved); h.flush(); assert.equal(region.scrollTop, 175);
});

test("animated sprite navigation cannot reveal a decoded preview from a previous character", async () => {
    const h = setup(), loaders = [], alice = deferred(), bob = deferred();
    Object.assign(h.context, { node: h.node, image: h.image, placeholder: h.placeholder, ui: h.ui,
        Image: class { constructor() { loaders.push(this); } },
    });
    h.context.api.fetchApi = async () => ({ ok: true, json: async () => ({ count: 0 }) });
    const begin = common.indexOf("export function createSpritePreviewNavigator(");
    const end = common.indexOf("// ── DOM Widget Canvas Navigation", begin);
    vm.runInContext(common.slice(begin, end).replaceAll("export ", "")
        + '\nthis.navigator = createSpritePreviewNavigator({ node, image, placeholder, interactions: ui });', h.context);
    h.image.decode = () => alice.promise;
    await h.context.navigator.load("Alice", { fallbackUrl: "alice.png" }); loaders[0].onload();
    assert.equal(h.image.style.opacity, "0");
    h.image.decode = () => bob.promise;
    await h.context.navigator.load("Bob", { fallbackUrl: "bob.png" }); loaders[1].onload();
    bob.resolve(); await bob.promise;
    const count = h.document.animations.length;
    alice.resolve(); await alice.promise;
    assert.equal(h.image.src, "bob.png"); assert.equal(h.document.animations.length, count);
    assert.equal(h.placeholder.style.display, "none");
    const carol = deferred(); h.image.decode = () => carol.promise;
    h.ui.presentPreview(h.image, h.placeholder, "carol.png", "Carol");
    assert.ok(h.frame.querySelectorAll(".vnccs-common-ui-motion-ghost").length);
    h.ui.emptyPreview(h.image, h.placeholder, "Carol", "No preview");
    carol.resolve(); await carol.promise;
    assert.equal(h.frame.querySelectorAll(".vnccs-common-ui-motion-ghost").length, 0);
    assert.equal(h.image.style.display, "none");
});

for (const [file, root, regions] of [
    ["vnccs_character_cloner.js", "vnccs-cloner", ["source", "source-images", "attributes"]],
    ["vnccs_clothes_designer.js", "vnccs-clothes", ["character", "design"]],
    ["vnccs_emotion_v2.js", "ems", ["costumes", "selected-emotions", "emotions", "generation", "model-picker-qi2", "model-picker-anima", "model-picker-illustrious"]],
]) {
    test(`${root} reuses Sakura motion, selectors and scroll restoration without changing control state`, async () => {
        const h = setup();
        const source = readFileSync(new URL(`../web/${file}`, import.meta.url), "utf8");
        assert.match(source, /createSakuraInteractions\(container, node\)/);
        assert.match(source, /interactions: (?:els\._ui|ui)/);
        assert.match(source, /registerCleanup\(node, \(\) => (?:els\._ui|ui)\.dispose\(\)\)/);
        h.root.className = `${root}-container vnccs-common-ui-host`;
        h.frame.className = "vnccs-common-ui-preview-host";
        h.aura.className = "vnccs-common-ui-preview-aura";
        h.placeholder.textContent = "No preview";
        const group = new h.Element("div");
        h.root.appendChild(group);
        for (const name of ["A", "B", "C"]) { const button = new h.Element("button"); button.textContent = name; group.appendChild(button); }
        group.children[0].classList.add("active"); h.ui.syncSelection(group);
        assert.equal(group.style["--segment-index"], "0");
        group.children[0].classList.toggle("active", false); group.children[2].classList.add("is-active"); h.ui.syncSelection(group);
        assert.equal(group.style["--segment-index"], "2"); assert.equal(group.style["--segments"], "3");
        assert.equal(group.children[2].getAttribute("aria-pressed"), "true");
        for (const key of regions) {
            const region = new h.Element("div"); region.dataset.sakuraScroll = key;
            region.scrollTop = 181; region.scrollLeft = 9; h.root.appendChild(region);
        }
        const view = h.ui.captureView();
        for (const region of h.root.querySelectorAll("[data-sakura-scroll]")) {
            const replacement = new h.Element("div"); replacement.dataset.sakuraScroll = region.dataset.sakuraScroll;
            region.remove(); h.root.appendChild(replacement);
        }
        h.ui.restoreView(view); h.flush();
        for (const region of h.root.querySelectorAll("[data-sakura-scroll]")) {
            assert.equal(region.scrollTop, 181); assert.equal(region.scrollLeft, 9);
        }
        h.image.decode = () => Promise.resolve();
        h.ui.emptyPreview(h.image, h.placeholder, "Alice", "No preview");
        h.ui.presentPreview(h.image, h.placeholder, "alice.png", "Alice"); await Promise.resolve();
        assert.equal(h.placeholder.style.display, "none"); assert.equal(h.image.alt, "Preview of Alice");
        assert.ok(h.document.animations.some(animation => animation.effect.target === h.aura && animation.options.duration === 700));
        const count = h.document.animations.length;
        h.ui.presentPreview(h.image, h.placeholder, "alice.png", "Alice"); await Promise.resolve();
        assert.equal(h.document.animations.length, count, "Refreshing the same portrait does not replay its entrance");
        h.document.motion = false;
        h.ui.presentPreview(h.image, h.placeholder, "bob.png", "Bob"); await Promise.resolve();
        assert.equal(h.image.src, "bob.png"); assert.equal(h.document.animations.length, count);
        h.ui.dispose(); assert.equal(h.document.observers[0].disconnected, true);
    });
}

test("enhancing an unselected model never chooses or saves a default", () => {
    const h = setup(), select = new h.Element("select"), option = new h.Element("option");
    option.value = option.textContent = "First model"; select.add(option); select.value = ""; h.root.appendChild(select);
    let changes = 0; select.onchange = () => { changes++; };
    h.ui.syncSelects();
    const trigger = h.root.querySelector(".vnccs-common-ui-select-trigger");
    assert.equal(select.value, ""); assert.equal(trigger.value, ""); assert.equal(changes, 0);
    trigger.onclick(); h.ui._dropdown.menu.children[1].onclick();
    assert.equal(select.value, "First model"); assert.equal(changes, 1);
});

test("first portrait waits for decoding and gets the longer reveal and glow", async () => {
    const h = setup(), ready = deferred(); h.image.decode = () => ready.promise;
    h.ui.prepareCharacter(h.image, h.placeholder, "Alice", "new");
    assert.equal(h.placeholder.querySelector(".vnccs-creator-placeholder-title").textContent, "Alice");
    assert.equal(h.placeholder.querySelector(".vnccs-creator-placeholder-hint").textContent, "Ready for the first preview");
    h.ui.presentPreview(h.image, h.placeholder, "alice.png", "Alice");
    assert.equal(h.image.style.opacity, "0"); assert.equal(h.placeholder.style.display, "flex");
    ready.resolve(); await ready.promise;
    assert.equal(h.image.style.opacity, ""); assert.equal(h.placeholder.style.display, "none");
    assert.equal(h.image.alt, "Preview of Alice");
    assert.ok(h.document.animations.some(animation => animation.effect.target === h.image && animation.options.duration === 420));
    assert.ok(h.document.animations.filter(animation => animation.effect.target === h.image).every(animation => animation.keyframes.every(frame => !("transform" in frame))));
    assert.ok(h.document.animations.some(animation => animation.effect.target === h.aura && animation.options.duration === 700));
});

test("switching characters invalidates an old image decode and repeats the full reveal and glow", async () => {
    const h = setup(), alice = deferred(), bob = deferred();
    h.image.decode = () => alice.promise; h.ui.presentPreview(h.image, h.placeholder, "alice.png", "Alice");
    h.ui.prepareCharacter(h.image, h.placeholder, "Bob");
    h.image.decode = () => bob.promise; h.ui.presentPreview(h.image, h.placeholder, "bob.png", "Bob");
    bob.resolve(); await bob.promise;
    const count = h.document.animations.length; alice.resolve(); await alice.promise;
    assert.equal(h.document.animations.length, count); assert.equal(h.image.src, "bob.png");
    h.ui.prepareCharacter(h.image, h.placeholder, "Carol");
    h.image.decode = () => Promise.resolve(); h.ui.presentPreview(h.image, h.placeholder, "carol.png", "Carol");
    await Promise.resolve();
    const reveal = h.document.animations.filter(animation => animation.effect.target === h.image).at(-1);
    assert.equal(reveal.options.duration, 420);
    assert.equal(reveal.keyframes[0].filter, "blur(5px)");
    assert.equal(h.document.animations.filter(animation => animation.effect.target === h.aura).at(-1).options.duration, 700);
    assert.ok(h.frame.querySelectorAll(".vnccs-common-ui-motion-ghost").length);
    for (const animation of [...h.document.animations]) animation.finish();
    assert.equal(h.frame.querySelectorAll(".vnccs-common-ui-motion-ghost").length, 0);
});

test("Turbo commits exact generation values before the visual swap finishes", () => {
    const h = setup(), steps = new h.Element("input"), cfg = new h.Element("input");
    steps.value = "25"; cfg.value = "3"; h.root.append(steps, cfg);
    const state = { gen_settings: { generation_mode: "qi2", steps: 25, cfg: 3, turbo_enabled: false } };
    let saved;
    Object.assign(h.context, { state, isQi2TurboModel: () => false,
        saveState() { saved = JSON.stringify(state); },
        syncGenerationControls() { const before = h.ui.captureValues([steps, cfg]); steps.value = state.gen_settings.steps; cfg.value = state.gen_settings.cfg; h.ui.animateValues(before); },
    });
    vm.runInContext(section("const setAnimaTurboMode =", "const FIELD_HELP =") + "\nthis.toggleTurbo = setAnimaTurboMode;", h.context);
    h.context.toggleTurbo(true);
    assert.equal(steps.value, "6"); assert.equal(cfg.value, "1"); assert.equal(JSON.parse(saved).gen_settings.steps, 6);
    assert.ok(h.document.animations.some(animation => animation.effect.target === steps && animation.options.duration === 240));
    assert.ok(h.document.animations.filter(animation => animation.effect.target === steps).every(animation => animation.keyframes.every(frame => !("transform" in frame))));
    h.context.toggleTurbo(false); assert.equal(steps.value, "25"); assert.equal(cfg.value, "3");
});

test("the real Turbo card toggles once and continues its thumb animation through another refresh", () => {
    const h = setup();
    const state = { gen_settings: { generation_mode: "qi2", steps: 25, cfg: 3, turbo_enabled: false } };
    let renders = 0, card, saved;
    const entry = { name: "Turbo", path: "turbo.safetensors" };
    Object.assign(h.context, { state, ccResolveStatus: () => "installed", ccRelPath: item => item.path, cardStatusLabel: () => "Installed",
        isQi2TurboModel: () => false, saveState() { saved = JSON.stringify(state); }, renderControlCenterCards() { assert.fail("Turbo must not rebuild twice"); },
        syncGenerationControls() {
            const view = h.ui.captureView(); card?.remove();
            card = h.context.makeCard({ entry, cat: "lora", selectedValue: state.gen_settings.dmd_lora_name,
                toggled: !!state.gen_settings.turbo_enabled, onToggle: enabled => h.context.toggleTurbo(enabled, entry.path) });
            h.root.appendChild(card); h.ui.restoreView(view); renders++;
        },
    });
    vm.runInContext(section("const buildAssetCard =", "const selectCcAsset =")
        + section("const setAnimaTurboMode =", "const FIELD_HELP =")
        + section("const setCcTurboMode =", "const setCcAgeLora =")
        + "\nthis.makeCard = buildAssetCard; this.toggleTurbo = setCcTurboMode;", h.context);
    h.context.syncGenerationControls();
    const checkbox = card.querySelector("input"), toggle = card.querySelector("label");
    let stopped = 0; toggle.onclick({ stopPropagation() { stopped++; } }); assert.equal(stopped, 1); assert.equal(renders, 1);
    checkbox.checked = true; checkbox.onchange({ target: checkbox, stopPropagation() {} });
    assert.equal(renders, 2); assert.equal(JSON.parse(saved).gen_settings.steps, 6);
    const thumb = card.querySelector(".vnccs-creator-toggle-thumb");
    const animation = h.ui._motionByElement.get(thumb);
    assert.equal(animation.options.duration, 280); assert.equal(animation.keyframes[1].transform, "translateX(16px)");
    animation.currentTime = 100;
    h.context.window.getComputedStyle = element => ({ opacity: "1", transform: element === thumb ? "matrix(1, 0, 0, 1, 6, 0)" : "none" });
    h.context.syncGenerationControls();
    const continued = h.ui._motionByElement.get(card.querySelector(".vnccs-creator-toggle-thumb"));
    assert.equal(continued.options.duration, 180); assert.equal(continued.keyframes[0].transform, "matrix(1, 0, 0, 1, 6, 0)");
    const next = card.querySelector("input"); next.checked = false; next.onchange({ target: next, stopPropagation() {} });
    assert.equal(renders, 4); assert.equal(JSON.parse(saved).gen_settings.steps, 25); assert.equal(state.gen_settings.cfg, 3);
    assert.equal(h.ui._motionByElement.get(card.querySelector(".vnccs-creator-toggle-thumb")).keyframes[1].transform, "none");
});

test("dropdown hover moves one highlight while its rows and popup remain still", () => {
    const h = setup(), select = new h.Element("select");
    for (const value of ["Alice", "Bob", "Carol"]) { const option = new h.Element("option"); option.value = option.textContent = value; select.add(option); }
    h.root.appendChild(select); h.ui.syncSelects(); h.root.querySelector(".vnccs-common-ui-select-trigger").onclick();
    const menu = h.ui._dropdown.menu;
    menu.children.forEach((row, index) => { row.offsetTop = 4 + index * 28; row.offsetHeight = 28; });
    const count = h.document.animations.length;
    for (const index of [1, 1, 0, 2, 1, 2]) {
        menu.children[index].onpointerenter();
        assert.equal(menu.style["--highlight-y"], `${4 + index * 28}px`);
        assert.equal(menu.style["--highlight-height"], "28px");
    }
    assert.equal(h.document.animations.length, count);
    assert.ok(h.document.animations.filter(animation => animation.effect.target === menu).every(animation => animation.keyframes.every(frame => !("transform" in frame))));
});

test("random seed activation changes and crossfades the seed; disabling freezes that value", () => {
    const h = setup(), seedMode = new h.Element("button"), dice = new h.Element("svg"); seedMode.appendChild(dice); h.root.appendChild(seedMode);
    const seed = new h.Element("input"); seed.value = 17; h.root.appendChild(seed);
    const state = { gen_settings: { seed_mode: "fixed", seed: 17 } };
    let saved;
    let draws = 0;
    Object.assign(h.context, { seedMode, state, els: { _ui: h.ui, seed }, generateRandomSeed() { draws++; return 8811; },
        syncGenerationControls() { const before = h.ui.captureValues([seed]); seed.value = state.gen_settings.seed; h.ui.animateValues(before); },
        saveState() { saved = JSON.stringify(state); } });
    vm.runInContext(section("seedMode.onclick =", "els.seed_mode ="), h.context);
    seedMode.onclick(); assert.equal(state.gen_settings.seed_mode, "randomize"); assert.equal(JSON.parse(saved).gen_settings.seed, 8811);
    assert.equal(seed.value, "8811"); assert.equal(draws, 1);
    assert.equal(h.document.animations.at(-1).effect.target, seed); assert.equal(h.document.animations.at(-1).options.duration, 240);
    assert.equal(h.document.animations.some(animation => animation.effect.target === dice), false);
    seedMode.onclick(); assert.equal(state.gen_settings.seed_mode, "fixed");
    assert.equal(JSON.parse(saved).gen_settings.seed, 8811); assert.equal(draws, 1);
});

test("every persistent Creator scroll region keeps both axes through catalog replacement", () => {
    const h = setup(), regions = [];
    for (const name of ["characters", "traits", "generation", "model-picker-qi2", "model-picker-anima", "model-picker-illustrious"]) {
        const region = new h.Element("div"); region.dataset.sakuraScroll = name; region.scrollTop = 173; region.scrollLeft = 7; h.root.appendChild(region); regions.push(region);
    }
    const before = h.ui.captureView();
    const oldMenu = regions.pop(), menu = new h.Element("div"); menu.dataset.sakuraScroll = oldMenu.dataset.sakuraScroll; oldMenu.remove(); h.root.appendChild(menu); regions.push(menu);
    regions.forEach(region => { region.scrollTop = 0; region.scrollLeft = 0; });
    h.ui.restoreView(before); h.flush();
    for (const region of regions) { assert.equal(region.scrollTop, 173); assert.equal(region.scrollLeft, 7); }
    h.ui.restoreView(before); regions[0].scrollTop = 230; h.flush(); assert.equal(regions[0].scrollTop, 230);
});

test("Background and gender move one persistent slider and never animate neighboring buttons", () => {
    const h = setup();
    const state = { character_info: { sex: "female", background_color: "Green" }, gen_settings: { generation_mode: "qi2" } };
    const els = { _ui: h.ui };
    Object.assign(h.context, { state, els, helpFor: () => "", setHelpText() {}, syncCharacterFields() {}, saveState() {} });
    vm.runInContext(section("const createSegmentedField =", "const createStyleField =")
        + '\nthis.sexField = createSegmentedField("Gender", "sex", [{ value: "female", label: "Female" }, { value: "male", label: "Male" }]);'
        + '\nthis.backgroundField = createSegmentedField("Background", "background_color", [{ value: "Green", label: "Green" }, { value: "Blue", label: "Blue" }, { value: "Transparent", label: "Alpha" }]);', h.context);
    h.root.append(h.context.sexField, h.context.backgroundField);
    for (const field of [h.context.sexField, h.context.backgroundField]) {
        const buttons = field.querySelectorAll("button");
        buttons[1].rect = { left: 200, top: 100, width: 140, height: 28 };
        buttons[1].onclick();
    }
    assert.equal(state.character_info.sex, "male"); assert.equal(state.character_info.background_color, "Blue");
    for (const field of [h.context.sexField, h.context.backgroundField]) {
        const slider = field.querySelector(".vnccs-creator-segmented-field");
        assert.equal(slider.style["--selected"], "1");
        const buttons = field.querySelectorAll("button");
        for (const index of [0, 1, 0, 1]) buttons[index].onclick();
        assert.equal(slider.style["--selected"], "1");
        assert.equal(slider.children.length, buttons.length);
    }
    assert.equal(h.document.animations.length, 0);
    assert.equal(h.root.querySelectorAll(".vnccs-common-ui-motion-ghost").length, 0);
});

test("model picker selection returns focus to its current header without scrolling the column", () => {
    const h = setup(), group = new h.Element("div"), option = new h.Element("div");
    group.dataset.sakuraFocusGroup = "qi2"; group.dataset.sakuraScroll = "generation"; group.scrollTop = 200; group.scrollLeft = 4;
    option.dataset.focusKey = "old-model-option"; group.appendChild(option); h.root.appendChild(group); option.focus();
    const before = h.ui.captureView(); option.remove();
    const head = new h.Element("div"); head.dataset.sakuraFocusDefault = "true"; group.appendChild(head);
    h.ui.restoreView(before); h.flush();
    assert.equal(h.document.activeElement, head); assert.equal(head.focusOptions.preventScroll, true);
    assert.equal(group.scrollTop, 200); assert.equal(group.scrollLeft, 4);
});

for (const success of [false, true]) {
    test(`character deletion animates only after its disk acknowledgement (success=${success})`, async () => {
        const h = setup(), pending = deferred();
        h.image.src = "alice.png"; h.image.style.display = "block";
        const selector = new h.Element("select");
        for (const name of ["Alice", "Bob"]) { const option = new h.Element("option"); option.value = option.textContent = name; selector.add(option); }
        const state = { character: "Alice", character_info: { name: "Alice", hair: "authored hair" } };
        let actions, saves = 0;
        Object.assign(h.context, { node: h.node, state, els: { _ui: h.ui, previewImg: h.image, placeholder: h.placeholder, charSelect: selector },
            showModal(_title, content, buttons) { content(); actions = buttons; }, showAlertModal() {},
            clearCharacterSelection() { state.character = ""; h.ui.emptyPreview(h.image, h.placeholder, "", "Create a character"); },
            async loadChar(name) { state.character = name; h.ui.prepareCharacter(h.image, h.placeholder, name); },
            saveState() { saves++; }, api: { ...h.context.api, fetchApi: () => pending.promise },
        });
        vm.runInContext('const beginCharacterRequest = createRequestGuard(node);\n' + section("const doDelete =", "btnDel.onclick =") + "\nthis.deleteCharacter = doDelete;", h.context);
        h.context.deleteCharacter();
        const operation = actions[1].action({ isConnected: true }, new h.Element("button"));
        assert.equal(h.document.animations.length, 0); assert.equal(state.character, "Alice");
        pending.resolve({ ok: success, json: async () => ({ error: "Read-only storage" }) }); await operation;
        if (success) {
            assert.equal(state.character, "Bob"); assert.equal(saves, 1); assert.equal(selector.options.length, 1);
            assert.ok(h.document.animations.some(animation => animation.options.duration === 160));
            assert.equal(h.placeholder.querySelector(".vnccs-creator-placeholder-title").textContent, "Bob");
        } else {
            assert.equal(state.character_info.hair, "authored hair"); assert.equal(state.character, "Alice");
            assert.equal(selector.options.length, 2); assert.equal(h.image.style.display, "block"); assert.equal(h.document.animations.length, 0);
        }
    });
}

test("removal and reduced motion finish UI operations without leaving animation ghosts", async () => {
    const h = setup(), ready = deferred(); h.image.decode = () => ready.promise;
    h.document.motion = false; h.ui.prepareCharacter(h.image, h.placeholder, "Alice", "new"); h.ui.presentPreview(h.image, h.placeholder, "alice.png", "Alice");
    ready.resolve(); await ready.promise; assert.equal(h.document.animations.length, 0);
    h.document.motion = true; h.ui.prepareCharacter(h.image, h.placeholder, "Bob");
    const later = deferred(); h.image.decode = () => later.promise; h.ui.presentPreview(h.image, h.placeholder, "bob.png", "Bob");
    h.ui.dispose(); later.resolve(); await later.promise;
    assert.equal(h.root.querySelectorAll(".vnccs-common-ui-motion-ghost").length, 0);
    assert.equal(h.ui._motionAnimations.size, 0);
    assert.equal(h.document.observers[0].disconnected, true);
    const count = h.document.animations.length;
    const select = new h.Element("select"); h.root.appendChild(select);
    h.document.observers[0].callback();
    h.ui.prepareCharacter(h.image, h.placeholder, "Carol", "new");
    h.ui.presentPreview(h.image, h.placeholder, "carol.png", "Carol");
    const previous = h.ui.captureValues([h.image]); h.image.value = "changed"; h.ui.animateValues(previous);
    assert.equal(select.hidden, undefined);
    assert.equal(h.image.src, "bob.png");
    assert.equal(h.document.animations.length, count);
});

test('a new smaller generation removes old gallery results and pending decodes', async () => {
    const h = setup();
    h.context.app = { registerExtension() {} };
    vm.runInContext(generator.replace(/^import .*;\n/gm, '') + '\nthis.Generator = CharacterGeneratorWidget;', h.context);
    const preview = new h.Element('div'); h.root.appendChild(preview);
    const widget = Object.assign(Object.create(h.context.Generator.prototype), {
        previewEl: preview, syncStagesFromData() {}, schedulePreviewGridLayout() {},
        stages: [['pose_generation', 'Poses']], selectedPreview: 'pose_generation',
        stageState: { pose_generation: {status: 'done', total: 3, images: ['old-1', 'old-2', 'old-3']} },
    });
    widget.renderPreview();
    const tiles = [...widget.previewGrid.children];
    for (const tile of tiles) await tile._pendingImage.onload();
    widget.stageState.pose_generation.images = ['old-1', 'pending-2', 'pending-3'];
    widget.renderPreview();
    const stale = tiles[2]._pendingImage.onload;
    widget.resetStagesFrom('pose_generation');
    widget.stageState.pose_generation = {status: 'running', total: 1};
    widget.renderPreview();
    assert.equal(widget.previewGrid.children.length, 1);
    assert.deepEqual(Array.from(widget.currentImages()), ['']);
    await stale();
    widget.stageState.pose_generation = {status: 'done', total: 1, images: ['new-1']};
    widget.renderPreview();
    await widget.previewGrid.children[0]._pendingImage.onload();
    assert.deepEqual(Array.from(widget.currentImages()), ['new-1']);
    assert.ok(tiles.slice(1).every(tile => !tile.isConnected));
    widget.stages.push(['upscaler', 'Upscaler']);
    widget.stageState.pose_generation = {status: 'done', total: 3, images: ['source-1', 'source-2', 'source-3']};
    widget.stageState.upscaler = {status: 'done', total: 1, images: ['final-1']};
    widget.selectedPreview = 'upscaler';
    widget.renderPreview();
    await widget.previewGrid.children[0]._pendingImage.onload();
    assert.deepEqual(Array.from(widget.currentImages()), ['final-1'], 'Finished results cannot borrow extra images from an earlier stage');
});
