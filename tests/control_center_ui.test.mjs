import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_control_center.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const constants = source.slice(source.indexOf("const INITIAL_NODE_W"), source.indexOf("function _injectVNCCSControlCenterStyles"));
const css = source.match(/s.textContent = `([\s\S]*?)`;/)[1].replace("${SAKURA_INTERACTION_CSS}", common.match(/export const SAKURA_INTERACTION_CSS = `([\s\S]*?)`;/)[1]);

function setup(state = {}, { loaded = true, repoId = "" } = {}) {
    const frames = [];
    const pending = [];
    const document = {};
    class Element {
        constructor(tag) {
            this.tagName = tag.toUpperCase();
            this.children = [];
            this.dataset = {};
            this.style = { setProperty() {} };
            this.className = "";
            this.scrollTop = this.scrollLeft = 0;
            this.classList = {
                contains: name => this.className.split(" ").includes(name),
                add: name => { if (!this.classList.contains(name)) this.className += ` ${name}`; },
                toggle: (name, enabled) => {
                    this.className = this.className.split(" ").filter(item => item !== name).join(" ");
                    if (enabled) this.classList.add(name);
                },
            };
        }
        get isConnected() { return this === document.body || !!this.parentNode?.isConnected; }
        set innerHTML(value) {
            assert.equal(value, "");
            for (const child of [...this.children]) child.remove();
            this.scrollTop = this.scrollLeft = 0;
            this._text = "";
        }
        get textContent() { return (this._text || "") + this.children.map(child => child.textContent || "").join(""); }
        set textContent(value) {
            for (const child of [...this.children]) child.remove();
            this._text = String(value);
        }
        get firstChild() { return this.children[0]; }
        appendChild(child) {
            if (child.parentNode) child.parentNode.children = child.parentNode.children.filter(item => item !== child);
            child.parentNode = this;
            this.children.push(child);
        }
        append(...children) { children.forEach(child => this.appendChild(child)); }
        insertBefore(child, before) {
            child.parentNode = this;
            const index = this.children.indexOf(before);
            this.children.splice(index < 0 ? 0 : index, 0, child);
        }
        replaceChild(next, old) {
            this.children.splice(this.children.indexOf(old), 1, next);
            next.parentNode = this;
            if (old.contains(document.activeElement)) document.activeElement = document.body;
            old.parentNode = null;
            // Simulate a replacement that loses the scroll container's incidental DOM state.
            this.scrollTop = this.scrollLeft = 0;
        }
        remove() {
            if (this.contains(document.activeElement)) document.activeElement = document.body;
            if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(child => child !== this);
            this.parentNode = null;
        }
        contains(element) { return element === this || this.children.some(child => child.contains(element)); }
        setAttribute(name, value) { this[name] = value; }
        removeAttribute(name) { delete this[name]; }
        addEventListener(name, callback) { this[`on${name}`] = callback; }
        removeEventListener(name, callback) { if (this[`on${name}`] === callback) delete this[`on${name}`]; }
        closest(selector) {
            if (selector.startsWith(".") && this.classList.contains(selector.slice(1))) return this;
            return this.parentNode?.closest(selector) || null;
        }
        getBoundingClientRect() {
            const rect = document.measure?.(this) || this.rect || { left: 0, top: 0, width: 0, height: 0 };
            return { ...rect, right: rect.left + rect.width, bottom: rect.top + rect.height };
        }
        cloneNode(deep) {
            const clone = new Element(this.tagName);
            for (const name of ["className", "value", "type", "id"]) clone[name] = this[name];
            clone._text = this._text;
            clone.dataset = { ...this.dataset };
            Object.assign(clone.style, this.style);
            if (deep) for (const child of this.children) clone.appendChild(child.cloneNode(true));
            return clone;
        }
        get animate() {
            if (!document.motionEnabled) return undefined;
            return (keyframes, options) => {
                const animation = { element: this, effect: { target: this }, keyframes, options, cancelled: false,
                    finish() { this.onfinish?.(); },
                    cancel() { this.cancelled = true; this.oncancel?.(); },
                };
                document.animations.push(animation);
                return animation;
            };
        }
        getClientRects() { return this.isConnected && !this.hidden ? [{}] : []; }
        focus(options) { document.activeElement = this; this.focusOptions = options; }
        querySelectorAll(selector) {
            const matches = element => selector.split(",").some(part => {
                const text = part.trim();
                const cls = text.match(/^\.([\w-]+)/)?.[1];
                if (cls && !element.classList.contains(cls)) return false;
                const data = text.match(/\[data-([\w-]+)(?:="([^"]*)")?\]/);
                if (data) {
                    const key = data[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
                    return key in element.dataset && (data[2] === undefined || element.dataset[key] === data[2]);
                }
                if (text === "[tabindex]") return element.tabIndex !== undefined;
                if (text === "[id]") return element.id !== undefined;
                if (text.startsWith("#")) return element.id === text.slice(1);
                if (text === "a[href]") return element.tagName === "A" && Boolean(element.href);
                return !!cls || element.tagName === text.toUpperCase();
            });
            const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll("*")]);
            return selector === "*" ? descendants : descendants.filter(matches);
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    }
    document.createElement = tag => new Element(tag);
    document.body = new Element("body");
    document.activeElement = document.body;
    document.animations = [];
    const context = createWidgetContext({
        document, setHelpText() {},
        api: { fetchApi: (route, options) => new Promise((resolve, reject) => pending.push({ route, options, resolve, reject })) },
        enableMiddleMouseCanvasPan() {}, attachHelpTooltips() {}, syncDOMWidgetWidthSoon() {},
        requestAnimationFrame: callback => frames.push(callback),
        setTimeout() {}, clearTimeout() {},
        window: { dispatchEvent() {} },
        CustomEvent: class {},
    });
    vm.runInContext(common.slice(common.indexOf("export function registerCleanup"), common.indexOf("// ── DOM Widget Width Sync"))
        .replaceAll("export ", "") + constants + source.slice(source.indexOf("class VNCCSControlCenterWidget"))
        + "\nthis.Widget = VNCCSControlCenterWidget;", context);
    const widget = Object.create(context.Widget.prototype);
    const serialized = { name: "node_state", value: JSON.stringify(state) };
    Object.assign(widget, {
        node: { id: 1, widgets: [serialized, { name: "repo_id", value: repoId }], setDirtyCanvas() {}, addDOMWidget() {} },
        state: { active_kind: "QI2", ...state }, dlStatus: {},
        _samplers: ["euler", "res_multistep"], _schedulers: ["simple", "normal"],
        _syncCustomModelInput() {}, _dispatchLoraOptions() {}, _syncCnetSlots() {},
        config: {
            models: [
                { name: "Qwen Image 2.1 INT8 ConvRot", type: "unet", kind: "QI2", status: "installed" },
                { name: "QI2 Alternative", type: "unet", kind: "QI2", status: "installed" },
            ],
            lora: [
                { name: "Style", custom: true, kind: "QI2", status: "installed", description: "Long local style description" },
                { name: "Turbo", type: "TurboLora", kind: "QI2", status: "installed" },
            ],
            clip: [], vae: [], controlnet: [], other: [],
        },
    });
    if (!loaded) widget.config = null;
    widget._beginConfigRequest = context.createRequestGuard(widget.node);
    widget._beginDownloadStatusRequest = context.createRequestGuard(widget.node);
    widget._buildUI();
    document.body.appendChild(widget.container);
    widget._renderAll();
    const flush = () => { for (const frame of frames.splice(0)) frame(); };
    flush();
    const focused = key => widget.scrollArea.querySelectorAll("[data-focus-key]").find(element => element.dataset.focusKey === key);
    return { widget, document, serialized, flush, focused, pending, context };
}

function respond(request, data) {
    request.resolve({ ok: true, json: async () => data });
}

function key(key, extra = {}) {
    return { key, preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...extra };
}

test("unsupported H3 shows an accessible update modal once per selection", () => {
    const { widget, document, serialized, flush } = setup();
    widget._h3Runtime = { version: "0.38.0", h3_still_supported: false,
        h3_update_message: "Update ComfyUI to include PR #16881, then restart the ComfyUI server." };
    widget.scrollArea.scrollTop = 145;
    widget._setActiveKind("MiniMaxH3");
    flush();
    const overlay = widget._h3UpdateOverlay;
    const panel = overlay.firstChild;
    assert.equal(panel.role, "dialog");
    assert.equal(panel.ariaModal, "true");
    assert.match(panel.textContent, /Update ComfyUI.*PR #16881.*restart.*0\.38\.0/);
    assert.equal(JSON.parse(serialized.value).active_kind, "MiniMaxH3");
    assert.equal(widget.scrollArea.scrollTop, 145);
    const link = panel.querySelector("a[href]");
    const close = panel.querySelector("button");
    assert.equal(link.href, "https://github.com/Comfy-Org/ComfyUI/pull/16881");
    assert.equal(link.rel, "noopener noreferrer");
    assert.equal(document.activeElement, link);
    close.focus();
    overlay.onkeydown(key("Tab"));
    assert.equal(document.activeElement, link);
    overlay.onkeydown(key("Tab", { shiftKey: true }));
    assert.equal(document.activeElement, close);
    overlay.onkeydown(key("Escape"));
    assert.equal(overlay.isConnected, false);
    assert.equal(widget.scrollArea.scrollTop, 145);
    widget._checkH3Compatibility();
    assert.equal(widget._h3UpdateOverlay, overlay);
    widget._setActiveKind("QI2");
    widget._setActiveKind("MiniMaxH3");
    assert.notEqual(widget._h3UpdateOverlay, overlay);
});

test("restored H3 warns without changing saved workflow data", () => {
    const { widget, serialized, flush } = setup({ active_kind: "MiniMaxH3", selected_type: "custom" });
    const saved = serialized.value;
    widget._h3Runtime = { version: null, h3_still_supported: false, h3_update_message: "Update ComfyUI and restart." };
    widget.restoreState();
    flush();
    assert.equal(widget._h3UpdateOverlay.isConnected, true);
    assert.match(widget._h3UpdateOverlay.textContent, /version: unknown/);
    assert.equal(serialized.value, saved);
    widget._removed = true;
    widget._h3WarningShown = false;
    widget._checkH3Compatibility();
    assert.equal(widget._overlays.length, 1);
});

test("supported H3 closes an obsolete warning and other families never show it", () => {
    const { widget } = setup();
    widget._h3Runtime = { h3_still_supported: false, h3_update_message: "Update ComfyUI." };
    widget._checkH3Compatibility();
    assert.equal(widget._h3UpdateOverlay, undefined);
    widget._setActiveKind("MiniMaxH3");
    const overlay = widget._h3UpdateOverlay;
    widget._h3Runtime = { h3_still_supported: true };
    widget._checkH3Compatibility();
    assert.equal(overlay.isConnected, false);
    widget._setActiveKind("QI2");
    widget._setActiveKind("MiniMaxH3");
    assert.equal(widget._overlays.length, 0);
});

function motionLayout({ widget, document }) {
    widget.container.clientWidth = 300;
    widget.container.clientHeight = 700;
    document.motionEnabled = true;
    document.measure = element => {
        let left = 100, top = 50, width = 600, height = 1400;
        if (element !== widget.container) {
            left = 120; top = 220; width = 560; height = 80;
            if (element === widget.scrollArea) height = 1100;
            if (element.dataset.blockKey) {
                top += widget.scrollArea.children.indexOf(element) * 140;
                height = element.classList.contains("vnccs-cc-block--collapsed") ? 40 : 130;
            }
            if (element.dataset.downloadKey) { left = 120; top = 480; width = 300; height = 80; }
            if (element.dataset.focusKey === "Steps") { left = 440; top = 490; width = 100; height = 56; }
            if (element.dataset.focusKey === "CFG") { left = 440; top = 590; width = 100; height = 56; }
            if (element.classList.contains("vnccs-cc-family-tab")) {
                left = 120 + ["QI2", "Klein9b", "MiniMaxH3"].indexOf(element.dataset.kind) * 180;
                width = 170; top = 210; height = 60;
            }
            if (element.classList.contains("vnccs-cc-model-tab")) {
                left = element.dataset.focusKey.endsWith("custom") ? 420 : 120;
                width = 260; top = 300; height = 60;
            }
        }
        return { left, top, width, height };
    };
}

test("animation copies remain out of layout after inheriting card styles", () => {
    const fixture = setup();
    const { widget } = fixture;
    motionLayout(fixture);
    const positionOf = element => {
        let position, specificity = -1;
        for (const [, selectors, body] of css.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
            const declared = body.match(/(?:^|;)\s*position:\s*(\w+)/)?.[1];
            if (!declared) continue;
            for (const selector of selectors.split(",").map(item => item.trim())) {
                const classes = selector.match(/^\.([\w-]+)(?:\s*>\s*\.([\w-]+))?$/);
                if (!classes) continue;
                const matches = classes[2]
                    ? element.parentNode?.classList.contains(classes[1]) && element.classList.contains(classes[2])
                    : element.classList.contains(classes[1]);
                const rank = classes[2] ? 2 : 1;
                if (matches && rank >= specificity) { position = declared; specificity = rank; }
            }
        }
        return position;
    };
    const footer = widget.downloadAllButton.parentNode;
    assert.doesNotMatch(css.match(/\.vnccs-cc-btn--download-all:hover\s*\{([^}]*)\}/)[1], /transform:/);
    assert.match(css, /:not\(\.vnccs-cc-btn--download-all\):active:not\(:disabled\)/);
    const snapshot = widget._captureMotion();
    for (const [key, previous] of snapshot) {
        if (!key.startsWith("card:")) continue;
        const ghost = widget._motionGhost(previous);
        assert.equal(ghost.parentNode, widget.container);
        assert.equal(positionOf(ghost), "absolute", `${key} must not become a flex item below the footer`);
        ghost.remove();
    }
    for (const action of [() => widget._setSelectedType("custom"), () => widget._setSelectedType("unet"),
        () => widget._setActiveKind("Klein9b"), () => widget._setActiveKind("QI2")]) {
        action();
        assert.equal(widget.downloadAllButton.parentNode, footer);
        assert.equal(footer.parentNode, widget.container);
        assert.equal(fixture.document.animations.some(animation => animation.element === footer
            || animation.element === widget.downloadAllButton), false);
        for (const ghost of widget.container.querySelectorAll(".vnccs-cc-motion-ghost")) assert.equal(positionOf(ghost), "absolute");
    }
});

test("every Control Center dropdown is custom and keyboard selection commits only on Enter", () => {
    const fixture = setup();
    const { widget, document, focused, serialized } = fixture;
    motionLayout(fixture);
    assert.doesNotMatch(source, /createElement\("select"\)/);
    const sampler = focused("vnccs-cc-sampler");
    assert.equal(sampler.role, "combobox");
    assert.equal(sampler.ariaLabel, "Sampler");
    const original = serialized.value;
    sampler.onkeydown(key("Enter"));
    assert.equal(sampler.ariaExpanded, "true");
    assert.equal(document.activeElement, sampler);
    assert.equal(widget._dropdown.menu.role, "listbox");
    sampler.onkeydown(key("End"));
    assert.equal(serialized.value, original, "Navigating choices does not change the workflow");
    sampler.onkeydown(key("Enter"));
    assert.equal(widget._dropdown, null);
    assert.equal(JSON.parse(serialized.value).model_params.sampler, "res_multistep");
    assert.equal(focused("vnccs-cc-sampler").value, "res_multistep");
    assert.equal(document.activeElement, focused("vnccs-cc-sampler"));
});

test("dropdown typeahead, Escape, Tab, outside click, and pointer selection preserve focus and both scroll axes", () => {
    const fixture = setup();
    const { widget, document, focused, serialized } = fixture;
    motionLayout(fixture);
    let sampler = focused("vnccs-cc-sampler");
    const original = serialized.value;
    sampler.onkeydown(key("r"));
    assert.equal(sampler["aria-activedescendant"], widget._dropdown.menu.children[1].id);
    widget._dropdown.menu.scrollTop = 83;
    widget._dropdown.menu.scrollLeft = 5;
    sampler.onkeydown(key("Escape"));
    assert.equal(serialized.value, original);
    assert.equal(document.activeElement, sampler);
    sampler.onclick();
    assert.equal(widget._dropdown.menu.scrollTop, 83);
    assert.equal(widget._dropdown.menu.scrollLeft, 5);
    const tab = key("Tab");
    sampler.onkeydown(tab);
    assert.equal(tab.prevented, undefined, "Tab advances to the next control");
    assert.equal(widget._dropdown, null);
    sampler.onclick();
    widget._dropdown.outside({ target: focused("Steps") });
    assert.equal(widget._dropdown, null);
    sampler.onclick();
    widget._dropdown.menu.children[1].onclick();
    sampler = focused("vnccs-cc-sampler");
    assert.equal(sampler.value, "res_multistep");
    assert.equal(JSON.parse(serialized.value).model_params.sampler, "res_multistep");
    assert.equal(sampler.ariaExpanded, "false");
});

test("the popup fits inside a zoomed node and opens above a control near its footer", () => {
    const fixture = setup();
    const { widget, document, focused } = fixture;
    motionLayout(fixture);
    const measure = document.measure;
    const trigger = focused("vnccs-cc-sampler");
    document.measure = element => element === trigger
        ? { left: 620, top: 1320, width: 60, height: 56 } : measure(element);
    trigger.onclick();
    const menu = widget._dropdown.menu;
    assert.equal(menu.style.width, "200px");
    assert.equal(menu.style.left, "92px");
    assert.ok(Number.parseFloat(menu.style.top) < (1320 - 50) / 2);
    assert.ok(Number.parseFloat(menu.style.left) + Number.parseFloat(menu.style.width) <= 300 - 8);
    const end = key("End");
    menu.clientHeight = 30;
    menu.children[1].offsetTop = 40;
    menu.children[1].offsetHeight = 28;
    trigger.onkeydown(end);
    assert.equal(menu.scrollTop, 38, "Keyboard navigation deliberately reveals the active option");
});

test("Turbo changes exact generation values immediately and animates their display without retaining an old draft", () => {
    const fixture = setup({ model_params: { steps: 25, cfg: 3 } });
    const { widget, document, focused, serialized, flush } = fixture;
    motionLayout(fixture);
    widget.scrollArea.scrollTop = 175;
    widget.scrollArea.scrollLeft = 7;
    focused("Steps").value = "31";
    focused("Steps").focus();
    widget._selectTurboLora("Turbo", true);
    flush();
    assert.equal(Number(focused("Steps").value), 6);
    assert.equal(Number(focused("CFG").value), 1);
    assert.equal(JSON.parse(serialized.value).model_params.steps, 6);
    assert.equal(JSON.parse(serialized.value).model_params.cfg, 1);
    assert.ok(document.animations.some(animation => animation.element === focused("Steps") && animation.keyframes[0].opacity === 0));
    assert.equal(widget.scrollArea.scrollTop, 175);
    assert.equal(widget.scrollArea.scrollLeft, 7);
    widget._selectTurboLora("Turbo", false);
    assert.equal(Number(focused("Steps").value), 25);
    assert.equal(Number(focused("CFG").value), 3);
    assert.equal(JSON.parse(serialized.value).model_params.steps, 25);
});

test("family and model mode switches animate their indicators and settle repeated transitions on removal", () => {
    const fixture = setup();
    const { widget, document, focused, flush } = fixture;
    motionLayout(fixture);
    widget.scrollArea.scrollTop = 210;
    widget.scrollArea.scrollLeft = 3;
    widget._setSelectedType("custom");
    assert.ok(document.animations.some(animation => animation.element.classList.contains("vnccs-common-ui-tab-glide")));
    widget._setSelectedType("unet");
    widget._setActiveKind("Klein9b");
    flush();
    assert.equal(document.activeElement, focused("family-Klein9b"));
    assert.equal(widget.scrollArea.scrollTop, 210);
    assert.equal(widget.scrollArea.scrollLeft, 3);
    widget._removed = true;
    widget._stopMotion();
    assert.equal(widget._motionAnimations.size, 0);
    assert.equal(widget.container.querySelectorAll(".vnccs-cc-motion-ghost").length, 0);
});

test("new custom LoRA cards enter with animation and activation keeps the current viewport", () => {
    const fixture = setup();
    const { widget, document, focused, flush, serialized } = fixture;
    motionLayout(fixture);
    widget.scrollArea.scrollTop = 230;
    widget.scrollArea.scrollLeft = 4;
    widget.config.lora.push({ name: "New style", custom: true, kind: "QI2", status: "installed" });
    widget._renderAll();
    const card = focused("cc_lora_New style");
    assert.ok(document.animations.some(animation => animation.element === card && animation.keyframes[0].opacity === 0));
    card.focus();
    card.onclick();
    flush();
    assert.equal(JSON.parse(serialized.value).loras.find(lora => lora.name === "New style").auto_apply, true);
    assert.equal(document.activeElement, focused("cc_lora_New style"));
    assert.equal(widget.scrollArea.scrollTop, 230);
    assert.equal(widget.scrollArea.scrollLeft, 4);
});

test("reduced motion skips visual effects while dropdowns, progress, and disclosures still work", () => {
    const fixture = setup();
    const { widget, document, context, focused } = fixture;
    motionLayout(fixture);
    context.window.matchMedia = () => ({ matches: true });
    widget._selectTurboLora("Turbo", true);
    const sampler = focused("vnccs-cc-sampler");
    sampler.onclick();
    sampler.onkeydown(key("Escape"));
    focused("block-models").onclick();
    assert.equal(widget.scrollArea.querySelector(".vnccs-cc-disclosure").inert, true);
    assert.equal(document.animations.length, 0);
    assert.equal(widget.container.querySelectorAll(".vnccs-common-ui-dropdown").length, 0);
    assert.match(css, /transition: transform 360ms linear/);
    assert.doesNotMatch(css, /transition: width/);
    assert.doesNotMatch(css.match(/@media \(prefers-reduced-motion: reduce\)[\s\S]*$/)[0], /\.vnccs-cc-root \* \{[^}]*transform:/);
});

test("initial config load reserves the panel before the deferred request starts", async () => {
    const { widget, pending, serialized, document } = setup({}, { loaded: false, repoId: "owner/models" });
    const original = serialized.value;
    assert.ok(widget.container.classList.contains("vnccs-cc-root--config-skeleton"));
    assert.equal(widget.container.querySelectorAll(".vnccs-cc-loading-block").length, 3);
    assert.equal(widget.scrollArea.inert, true);
    assert.equal(widget.scrollArea["aria-hidden"], "true");
    assert.equal(widget.downloadAllButton.disabled, true);
    const request = widget.fetchConfig("owner/models");
    assert.equal(widget.scrollArea["aria-busy"], "true");
    respond(pending[0], { name: "Catalog", models: [], lora: [] });
    await request;
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-skeleton"), false);
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-loading"), false);
    assert.equal(widget.scrollArea.inert, false);
    assert.equal(widget.scrollArea["aria-busy"], "false");
    assert.equal(widget.downloadAllButton.disabled, false);
    assert.ok(widget.scrollArea.querySelector(".vnccs-cc-family-tabs"));
    assert.equal(serialized.value, original);
    assert.equal(document.activeElement, document.body);
});

test("refresh keeps controls, scroll, focus, and numeric drafts visible during the request", async () => {
    const { widget, focused, pending, flush } = setup();
    const config = widget.config;
    const input = focused("Steps");
    input.value = "31";
    input.focus();
    widget.scrollArea.scrollTop = 184;
    widget.scrollArea.scrollLeft = 7;
    const request = widget.fetchConfig("owner/models", true);
    assert.ok(widget.container.classList.contains("vnccs-cc-root--config-loading"));
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-skeleton"), false);
    assert.equal(widget.scrollArea.inert, false);
    assert.equal(focused("Steps"), input);
    assert.equal(widget.scrollArea.scrollTop, 184);
    assert.equal(widget.scrollArea.scrollLeft, 7);
    respond(pending[0], config);
    await request;
    flush();
    assert.equal(focused("Steps").value, "31");
    assert.equal(widget.scrollArea.scrollTop, 184);
    assert.equal(widget.scrollArea.scrollLeft, 7);
});

test("a cached catalog replaces the skeleton while its background refresh continues", async () => {
    const { widget, pending, context } = setup({}, { loaded: false });
    context.serverRegistry("VNCCS_CC_REGISTRY")["owner/models"] = { name: "Cached", models: [], lora: [] };
    const request = widget.fetchConfig("owner/models");
    assert.equal(widget.statusText.textContent, "Cached");
    assert.ok(widget.container.classList.contains("vnccs-cc-root--config-loading"));
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-skeleton"), false);
    assert.ok(widget.scrollArea.querySelector(".vnccs-cc-family-tabs"));
    respond(pending[0], { name: "Fresh", models: [], lora: [] });
    await request;
    assert.equal(widget.statusText.textContent, "Fresh");
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-loading"), false);
});

test("a stale request cannot stop the newer skeleton and a failure exposes its error", async () => {
    const { widget, pending } = setup({}, { loaded: false });
    const old = widget.fetchConfig("owner/old");
    const current = widget.fetchConfig("owner/current");
    respond(pending[0], { name: "Old", models: [], lora: [] });
    await old;
    assert.equal(widget.config, null);
    assert.ok(widget.container.classList.contains("vnccs-cc-root--config-skeleton"));
    assert.equal(widget.scrollArea["aria-busy"], "true");
    pending[1].reject(new Error("Catalog unavailable"));
    await current;
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-skeleton"), false);
    assert.equal(widget.scrollArea["aria-busy"], "false");
    assert.equal(widget.scrollArea.inert, false);
    assert.match(widget.scrollArea.querySelector(".vnccs-cc-error").textContent, /Catalog unavailable/);
    assert.equal(widget.downloadAllButton.disabled, true);
});

test("an empty repository and a damaged workflow do not leave a loading overlay", () => {
    const { widget, serialized } = setup({}, { loaded: false });
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-skeleton"), false);
    widget._setConfigLoading(true);
    serialized.value = "{broken";
    widget.showMessage = () => {};
    widget.restoreState();
    assert.equal(widget.container.classList.contains("vnccs-cc-root--config-skeleton"), false);
    assert.equal(widget.scrollArea.inert, false);
    assert.equal(serialized.value, "{broken");
});

test("loading motion uses a fixed content frame, crossfade, and a reduced-motion override", () => {
    assert.match(css, /\.vnccs-cc-content\s*\{[^}]*flex: 1;[^}]*min-height: 0;/);
    assert.match(css, /\.vnccs-cc-loading\s*\{[^}]*position: absolute;[^}]*inset: 0;/);
    assert.match(css, /transition: opacity 280ms ease-out, transform 280ms ease-out/);
    assert.match(css, /transition: opacity 180ms ease-out, visibility 0s linear 180ms/);
    assert.match(css, /@media \(prefers-reduced-motion: reduce\)\s*\{[\s\S]*animation: none !important/);
});

test("Control Center keeps its columns shrinkable and peers at the same control height", () => {
    assert.match(css, /\.vnccs-cc-twocol\s*\{[^}]*display: flex/);
    assert.match(css, /width: min\(24%, 130px\)/);
    assert.match(css, /\.vnccs-cc-twocol-right2\s*\{[^}]*width: min\(30%, 130px\)/);
    for (const name of ["input", "select", "params--right .vnccs-cc-select"]) {
        assert.match(css, new RegExp(`\\.vnccs-cc-${name}\\s*\\{[^}]*height: var\\(--vnccs-cc-control-height\\)`));
    }
    assert.match(css, /\.vnccs-cc-scroll > \* \{ flex-shrink: 0; \}/);
    assert.match(css, /prefers-reduced-motion: reduce/);
});

test("MODEL and CLIP + VAE retain their compact heading, status, padding, and footer", () => {
    const rule = selector => css.match(new RegExp(`(?:^|\\n)${selector.replaceAll(".", "\\.")}\\s*\\{([^}]*)\\}`))[1];
    const card = rule(".vnccs-cc-model-card");
    const top = rule(".vnccs-cc-model-card-top");
    const name = rule(".vnccs-cc-model-card-name");
    const status = rule(".vnccs-cc-model-card-status");
    const footer = rule(".vnccs-cc-model-card-footer");
    assert.match(card, /padding: 10px 12px 8px/);
    assert.doesNotMatch(card, /overflow-wrap/);
    assert.doesNotMatch(top, /flex-wrap/);
    assert.match(name, /white-space: nowrap/);
    assert.match(name, /text-overflow: ellipsis/);
    assert.match(status, /white-space: nowrap/);
    assert.doesNotMatch(status, /flex-basis|overflow-wrap/);
    assert.doesNotMatch(footer, /flex-wrap/);
    assert.doesNotMatch(rule(".vnccs-cc-model-card-select"), /flex: 1 1 100%/);
    assert.doesNotMatch(css, /\.vnccs-cc-root \*::after \{ box-sizing/);
});

for (const [name, category, render] of [
    ["MODEL", "models", (widget, entry) => widget._renderModelCard("unet", [entry])],
    ["CLIP", "clip", (widget, entry) => widget._renderClipVaeCard({ ...entry, _cat: "clip" })],
    ["VAE", "vae", (widget, entry) => widget._renderClipVaeCard({ ...entry, _cat: "vae" })],
    ["Turbo LoRA", "lora", (widget, entry) => widget._renderTurboInlineStrip(entry)],
    ["compact LoRA", "lora", (widget, entry) => widget._renderLoraCard(entry, { compact: true })],
    ["full LoRA", "lora", (widget, entry) => widget._renderLoraCard(entry)],
    ["ControlNet", "controlnet", (widget, entry) => widget._renderCnetEntry("controlnet", entry)],
    ["Other", "other", (widget, entry) => widget._renderCnetEntry("other", entry)],
]) {
    test(`${name} replaces a missing status with one download button in the heading`, () => {
        const { widget, document } = setup();
        const entry = { name: "Asset with a long name", kind: "QI2", type: "unet", description: "Description" };
        if (category !== "lora") widget.state.loras = [{ name: entry.name, auto_apply: true }];
        const key = `cc_${category}_${entry.name}`;
        const downloads = [];
        widget._downloadEntry = (cat, asset) => downloads.push([cat, asset.name]);
        const input = document.createElement("input");
        widget.scrollArea.appendChild(input);
        input.focus();
        widget.scrollArea.scrollTop = 120;
        widget.scrollArea.scrollLeft = 8;

        for (const status of ["missing", "outdated", "error"]) {
            entry.status = status;
            widget.dlStatus = {};
            const card = render(widget, entry);
            widget.scrollArea.appendChild(card);
            const heading = card.querySelector(".vnccs-cc-model-card-top, .vnccs-cc-lora-card-top") || card;
            const label = card.querySelector(".vnccs-cc-model-card-status, .vnccs-cc-lora-card-status, .vnccs-cc-turbo-strip-status");
            const button = card.querySelector(".vnccs-cc-download-btn");
            const action = status === "outdated" ? "Update" : "Download";
            assert.equal(card.querySelectorAll("button").length, 1);
            assert.ok(heading.contains(button));
            assert.equal(label.hidden, true);
            assert.equal(button.type, "button");
            assert.equal(button.textContent, action);
            assert.equal(button.ariaLabel, `${action} ${entry.name}`);
            let stopped = false;
            button.onclick({ stopPropagation() { stopped = true; } });
            assert.equal(stopped, true);
            assert.deepEqual(downloads.at(-1), [category, entry.name]);

            widget._downloadRefreshPending = true;
            for (const nextStatus of ["queued", "downloading", "success", "error", "auth_required"]) {
                widget.dlStatus[key] = { status: nextStatus, progress: 37, message: "Transfer details" };
                widget._updateDownloadIndicators();
                const retry = nextStatus === "error";
                assert.equal(card.querySelector(".vnccs-cc-download-btn"), button);
                assert.equal(button.hidden, !retry);
                assert.equal(button.disabled, !retry);
                assert.equal(label.hidden, retry);
                assert.equal(label.ariaLabel, nextStatus === "success" ? "Checking installation…"
                    : nextStatus === "auth_required" ? "⚠ Key Required"
                    : nextStatus === "queued" ? "⏳ Queued" : "Transfer details");
                assert.equal(document.activeElement, input);
                assert.equal(widget.scrollArea.scrollTop, 120);
                assert.equal(widget.scrollArea.scrollLeft, 8);
            }
            card.remove();
        }
        for (const status of ["installed", "queued", "downloading", "auth_required"]) {
            entry.status = status;
            widget.dlStatus = {};
            const card = render(widget, entry);
            assert.equal(card.querySelector(".vnccs-cc-download-btn").hidden, true);
            assert.equal(card.querySelector(".vnccs-cc-model-card-status, .vnccs-cc-lora-card-status, .vnccs-cc-turbo-strip-status").hidden, false);
            if (status === "installed" && category !== "lora") assert.equal(card._downloadView.label.textContent, "Installed");
        }
    });

    test(`${name} retains its card and status slot through animated download and retry transitions`, async () => {
        const { widget, document, pending, context } = setup({}, { repoId: "owner/models" });
        const entry = { name: `${name} download`, kind: "QI2", type: "unet", status: "missing" };
        widget.config[category] ??= [];
        widget.config[category].push(entry);
        const card = render(widget, entry);
        widget.scrollArea.appendChild(card);
        const { slot, label, button, badge } = card._downloadView;
        const key = `cc_${category}_${entry.name}`;
        widget._renderAll = () => assert.fail("Download transitions must not rebuild the interface");
        widget.scrollArea.scrollTop = 170;
        widget.scrollArea.scrollLeft = 9;
        button.focus();
        document.motionEnabled = true;
        const request = widget._downloadEntry(category, entry);
        await widget._downloadEntry(category, entry);
        assert.equal(pending.length, 1, "Repeated clicks cannot submit a second download");
        assert.equal(document.activeElement, label);
        assert.equal(label.textContent, "Queued");
        pending[0].resolve({ ok: true, text: async () => '{"status":"queued"}' });
        await request;

        for (const status of ["downloading", "error", "queued", "downloading", "success", "installed"]) {
            widget._downloadRefreshPending = status === "success";
            if (status === "installed") entry.status = status;
            widget.dlStatus[key] = { status, progress: 37, message: "Long transfer details ".repeat(50) };
            widget._updateDownloadIndicators();
            assert.equal(card._downloadView.slot, slot);
            assert.equal(slot.children.length, 2);
            assert.equal(card._downloadView.label, label);
            assert.equal(card._downloadView.button, button);
            assert.equal(card._downloadView.badge, badge);
            assert.equal(widget.scrollArea.scrollTop, 170);
            assert.equal(widget.scrollArea.scrollLeft, 9);
            assert.equal(label.role, "status");
            const count = document.animations.length;
            widget.dlStatus[key].progress = 38;
            widget._updateDownloadIndicators();
            assert.equal(document.animations.length, count, "Percentage ticks must not restart status animations");
        }
        for (const animation of document.animations.filter(item => item.element === slot)) {
            assert.ok(animation.keyframes.every(frame => frame.transform === undefined), "Status transitions only fade within their reserved space");
        }
        context.window.matchMedia = () => ({ matches: true });
        const count = document.animations.length;
        widget.dlStatus[key] = { status: "queued" };
        widget._updateDownloadIndicators();
        assert.equal(document.animations.length, count);
        widget._stopMotion();
        widget.node.onRemoved();
        assert.equal(widget._motionAnimations.size, 0);
    });
}

test("download verification preserves cards, open controls and drafts through a failed catalog refresh", async () => {
    const { widget, document, pending, serialized, focused } = setup({}, { repoId: "owner/models" });
    widget._downloadStatusRepoId = "owner/models";
    const entry = widget.config.models[0];
    entry.status = "missing";
    widget._renderAll();
    const card = widget.scrollArea.querySelector(`[data-download-key="cc_models_${entry.name}"]`);
    const view = card._downloadView;
    const steps = focused("Steps");
    steps.value = "31";
    steps.focus();
    widget.scrollArea.scrollTop = 180;
    const original = serialized.value;
    widget.dlStatus[`cc_models_${entry.name}`] = { status: "success" };
    widget._downloadRefreshPending = true;
    widget._updateDownloadIndicators();
    assert.equal(view.label.textContent, "Checking");
    widget._renderAll = () => assert.fail("Catalog verification must preserve existing cards");
    const failed = widget.fetchConfig("owner/models", true, { preserveCards: true });
    pending[0].resolve({ ok: false, json: async () => ({ error: "Catalog unavailable" }) });
    await failed;
    assert.equal(widget._downloadRefreshPending, true);
    assert.equal(view.label.textContent, "Checking");
    assert.equal(document.activeElement, steps);
    const retry = widget.fetchConfig("owner/models", true, { preserveCards: true });
    assert.match(pending[1].route, /force_refresh=true/);
    respond(pending[1], { ...widget.config, models: widget.config.models.map(model => ({ ...model, status: "installed" })) });
    await retry;
    assert.equal(widget._downloadRefreshPending, false);
    assert.equal(view.label.textContent, "Installed");
    assert.equal(card._downloadView, view);
    assert.equal(view.button.hidden, true);
    assert.equal(steps.value, "31");
    assert.equal(document.activeElement, steps);
    assert.equal(widget.scrollArea.scrollTop, 180);
    assert.equal(serialized.value, original);
});

test("muted labels meet normal-text contrast on the existing dark surfaces", () => {
    const luminance = color => {
        const channels = color.match(/\w\w/g).map(value => parseInt(value, 16) / 255)
            .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
        return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    };
    const muted = css.match(/--vnccs-cc-muted: #([\da-f]+)/)[1];
    for (const surface of ["0a0a0f", "12121a", "1a1a26"]) {
        assert.ok((luminance(muted) + 0.05) / (luminance(surface) + 0.05) >= 4.5);
    }
});

test("sampling and cache controls have enclosing labels and retain focus after edits", () => {
    const { widget, document, flush, focused } = setup();
    for (const key of ["Steps", "CFG", "vnccs-cc-sampler", "vnccs-cc-scheduler", "vnccs-cc-qi2-cache-device", "vnccs-cc-qi2-cache-dtype"]) {
        assert.equal(focused(key).parentNode.tagName, "LABEL");
    }
    const input = focused("CFG");
    input.focus();
    widget.scrollArea.scrollTop = 184;
    widget.scrollArea.scrollLeft = 7;
    widget._modelParamsSave({ cfg: 2 });
    flush();
    assert.equal(document.activeElement, focused("CFG"));
    assert.notEqual(document.activeElement, input);
    assert.equal(document.activeElement.focusOptions.preventScroll, true);
    assert.equal(widget.scrollArea.scrollTop, 184);
    assert.equal(widget.scrollArea.scrollLeft, 7);
});

test("partial LoRA selection and full refresh preserve both scroll axes and the focused card", () => {
    const { widget, document, flush, focused } = setup();
    const card = focused("cc_lora_Style");
    card.focus();
    widget.scrollArea.scrollTop = 240;
    widget.scrollArea.scrollLeft = 9;
    let prevented = false;
    card.onkeydown({ key: " ", target: card, preventDefault() { prevented = true; }, stopPropagation() {} });
    flush();
    assert.equal(prevented, true);
    assert.equal(widget.state.loras.find(lora => lora.name === "Style").auto_apply, true);
    assert.equal(document.activeElement, focused("cc_lora_Style"));
    assert.equal(document.activeElement.ariaPressed, "true");
    assert.equal(widget.scrollArea.scrollTop, 240);
    assert.equal(widget.scrollArea.scrollLeft, 9);
    assert.equal(focused("strength-Style").ariaLabel, "Style strength");
    widget._renderAll();
    flush();
    assert.equal(widget.scrollArea.scrollTop, 240);
    assert.equal(widget.scrollArea.scrollLeft, 9);
    assert.equal(document.activeElement, focused("cc_lora_Style"));
});

test("nested LoRA controls do not toggle their card through a bubbled keyboard event", () => {
    const { widget, focused } = setup({ loras: [{ name: "Style", auto_apply: true, strength: 1 }] });
    const before = JSON.stringify(widget.state);
    focused("cc_lora_Style").onkeydown({ key: " ", target: focused("strength-Style"),
        preventDefault() { assert.fail("The slider must retain its native keyboard action"); } });
    assert.equal(JSON.stringify(widget.state), before);
});

test("a scheduled render retains an uncommitted numeric draft without serializing it", () => {
    const { widget, document, serialized, flush, focused } = setup();
    const input = focused("CFG");
    input.value = "";
    input.focus();
    const original = serialized.value;
    widget._renderAll();
    flush();
    assert.equal(document.activeElement, focused("CFG"));
    assert.equal(focused("CFG").value, "");
    assert.equal(serialized.value, original);
});

test("deferred restoration respects a deliberate scroll made after the render", () => {
    const { widget, flush } = setup();
    widget.scrollArea.scrollTop = 130;
    widget._renderAll();
    widget.scrollArea.scrollTop = 260;
    flush();
    assert.equal(widget.scrollArea.scrollTop, 260);
});

test("collapsing MODEL hides all of its controls and keeps keyboard focus on its header", () => {
    const { widget, document, flush, focused } = setup();
    const header = focused("block-models");
    assert.equal(header.tagName, "BUTTON");
    assert.equal(header.ariaExpanded, "true");
    header.focus();
    const steps = focused("Steps");
    header.onclick();
    flush();
    assert.equal(focused("block-models").ariaExpanded, "false");
    assert.equal(document.activeElement, focused("block-models"));
    const block = widget.scrollArea.querySelector('.vnccs-cc-block[data-block-key="models"]');
    const disclosure = block.querySelector(".vnccs-cc-disclosure");
    assert.equal(disclosure.inert, true);
    assert.equal(disclosure.ariaHidden, "true");
    assert.ok(block.classList.contains("vnccs-cc-block--collapsed"));
    assert.equal(focused("Steps"), steps, "Collapsed controls retain their values and DOM identity");
    header.onclick();
    flush();
    assert.equal(disclosure.inert, false);
    assert.equal(disclosure.ariaHidden, "false");
    assert.equal(focused("Steps"), steps);
});

test("switching between base QI2 and built-in Turbo updates Turbo LoRA visibility", () => {
    const { widget } = setup();
    const turboModel = { name: "QI2 Turbo", kind: "QI2", type: "unet", status: "installed",
        local_path: "models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors" };
    widget.config.models.push(turboModel);
    widget._setSelectedModelName("unet", turboModel.name);
    widget._renderAll();
    assert.equal(widget.scrollArea.querySelector(".vnccs-cc-turbo-strip"), null);
    widget._setSelectedModelName("unet", "Qwen Image 2.1 INT8 ConvRot");
    widget._renderAll();
    assert.ok(widget.scrollArea.querySelector(".vnccs-cc-turbo-strip"));
});

test("LoRA cards keep two columns and fixed text, tool and detail rows", () => {
    const { widget } = setup();
    const card = widget._renderLoraCard({ name: "A".repeat(200), kind: "QI2", custom: true,
        status: "missing", hf_path: "style.safetensors", description: "Long description" }, { compact: true });
    assert.equal(card.querySelectorAll("button").length, 2);
    const remove = card.querySelector(".vnccs-cc-lora-remove-btn");
    assert.equal(remove.parentNode, card.querySelector(".vnccs-cc-lora-card-tools"));
    assert.equal(remove.ariaLabel, `Remove ${"A".repeat(200)} from list`);
    assert.match(css, /\.vnccs-cc-lora-grid\s*\{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
    assert.match(css, /\.vnccs-cc-lora-card\s*\{[^}]*display: grid;[^}]*grid-template-rows: 28px 14px 28px;[^}]*height: 96px/);
    assert.match(css, /\.vnccs-cc-lora-card--compact\s*\{[^}]*grid-template-rows: 28px 14px;[^}]*height: 60px/);
    assert.match(css, /\.vnccs-cc-lora-card-detail-row\s*\{[^}]*grid-template-columns: minmax\(0, 1fr\) 40px/);
    for (const name of ["name", "desc"]) {
        assert.match(css, new RegExp(`\\.vnccs-cc-lora-card-${name}\\s*\\{[^}]*height: 14px;[^}]*white-space: nowrap;[^}]*overflow: hidden;[^}]*text-overflow: ellipsis;`));
    }
    assert.doesNotMatch(css.match(/\.vnccs-cc-lora-card:hover[^}]+\}/)[0], /transform:/);
    assert.doesNotMatch(css.match(/\.vnccs-cc-lora-remove-btn\s*\{([^}]*)\}/)[1], /position: absolute/);
});

test("compact LoRA statuses and download buttons retain the same reserved slot", () => {
    const { widget } = setup();
    const entry = { name: "VNCCS Overhaul QI2", kind: "QI2", status: "missing",
        description: "Overhaul for Character Creator", version: "2.0" };
    for (const dls of [
        { status: "queued" },
        { status: "downloading", progress: 98, message: "Downloading: 98% · 149.0/152.1 MB" },
        { status: "downloading", message: "Preparing " + "a very long download message ".repeat(30) },
        { status: "error", message: "Transfer interrupted " + "retry details ".repeat(30) },
        { status: "auth_required" },
        { status: "installed" },
    ]) {
        entry.status = dls.status;
        widget.dlStatus = { [`cc_lora_${entry.name}`]: dls };
        const card = widget._renderLoraCard(entry, { compact: true, mode: "pipe" });
        const status = card.querySelector(".vnccs-cc-lora-card-status");
        assert.equal(status.parentNode.parentNode, card.querySelector(".vnccs-cc-lora-card-tools"));
        assert.equal(card.querySelector(".vnccs-cc-lora-card-meta").children.length, 1, "Only the name participates in the text column's height");
        assert.equal(card.querySelector(".vnccs-cc-lora-card-detail-row").textContent, "Overhaul for Character Creatorv2.0");
        assert.equal(card.querySelector(".vnccs-cc-row-progress"), null, "Compact warnings use the existing status slot");
        if (dls.status === "downloading") {
            assert.equal(status.textContent, Number.isFinite(dls.progress) ? "98%" : "Loading");
            assert.equal(status.ariaLabel, dls.message);
            assert.equal(status.title, dls.message);
            assert.equal(status.tabIndex, 0);
        }
        if (dls.status === "error") {
            assert.equal(status.hidden, true);
            assert.ok(card.querySelector(".vnccs-cc-download-btn"));
        }
    }
    assert.match(css, /\.vnccs-cc-lora-card-top\s*\{[^}]*flex-wrap: nowrap/);
    assert.match(css, /\.vnccs-cc-lora-card-tools\s*\{[^}]*flex: 0 0 auto;[^}]*flex-wrap: nowrap/);
    assert.match(css, /\.vnccs-cc-download-slot\s*\{[^}]*flex: 0 0 80px;[^}]*height: var\(--vnccs-cc-control-height\)/);
    assert.match(css, /\.vnccs-cc-download-btn\s*\{[^}]*flex-shrink: 0;[^}]*white-space: nowrap/);
    assert.match(css, /\.vnccs-cc-lora-card--compact \.vnccs-cc-lora-card-status\s*\{[^}]*overflow: hidden;[^}]*white-space: nowrap/);
    assert.match(css, /\.vnccs-cc-download-btn\[hidden\]\s*\{[^}]*display: none/);
});

test("absent LoRA description and version retain their static row and cells", () => {
    const { widget } = setup();
    const card = widget._renderLoraCard({ name: "Style", status: "installed" }, { compact: true });
    const row = card.querySelector(".vnccs-cc-lora-card-detail-row");
    assert.ok(row);
    assert.equal(row.children.length, 2);
    assert.equal(row.children[0].textContent, "");
    assert.equal(row.children[1].textContent, "");
    assert.match(css, /\.vnccs-cc-lora-card-version\s*\{[^}]*height: 14px/);
});

test("live LoRA polling keeps its compact status in the same slot and retains the full message", () => {
    const { widget, document } = setup();
    const entry = { name: "Overhaul", kind: "QI2", status: "missing", description: "Creator style" };
    const key = `cc_lora_${entry.name}`;
    const card = widget._renderLoraCard(entry, { compact: true, mode: "pipe" });
    widget.scrollArea.appendChild(card);
    const status = card.querySelector(".vnccs-cc-lora-card-status"), slot = status.parentNode;
    const download = card.querySelector(".vnccs-cc-download-btn");
    assert.equal(status.hidden, true);
    assert.notEqual(download.disabled, true);
    const input = document.createElement("input");
    widget.scrollArea.appendChild(input);
    input.focus();
    widget.scrollArea.scrollTop = 120;
    widget.scrollArea.scrollLeft = 8;
    document.motionEnabled = true;
    widget._downloadRefreshPending = true;
    for (const [dls, expected] of [
        [{ status: "queued" }, "Queued"],
        [{ status: "downloading", progress: 98, message: "Downloading: 98% · 149.0/152.1 MB" }, "98%"],
        [{ status: "downloading", progress: 150, message: "Verifying transferred bytes" }, "100%"],
        [{ status: "downloading", progress: -2, message: "Starting download" }, "0%"],
        [{ status: "downloading", message: "Unknown size " + "download details ".repeat(100) }, "Loading"],
        [{ status: "success" }, "Checking"],
        [{ status: "error", message: "Transfer interrupted" }, "Transfer interrupted"],
        [{ status: "queued" }, "Queued"],
    ]) {
        widget.dlStatus[key] = dls;
        widget._updateDownloadIndicators();
        assert.equal(status.parentNode, slot);
        assert.equal(status.textContent, expected);
        assert.equal(status.ariaLabel, dls.message || (dls.status === "queued" ? "⏳ Queued" : "Checking installation…"));
        const retry = dls.status === "error";
        assert.equal(status.hidden, retry, "Polling must reveal the queued status that initially shared the download button's slot");
        assert.equal(status.tabIndex, retry ? -1 : 0);
        assert.equal(download.hidden, !retry);
        assert.equal(download.disabled, !retry, "Queued or active downloads must not allow a second submission");
        assert.equal(download.parentNode, slot, "The button and status retain one fixed slot");
        assert.equal(document.activeElement, input);
        assert.equal(widget.scrollArea.scrollTop, 120);
        assert.equal(widget.scrollArea.scrollLeft, 8);
        assert.equal(card.querySelector(".vnccs-cc-lora-card-meta").children.length, 1);
    }
    const fullCard = widget._renderLoraCard(entry);
    widget.dlStatus[key] = { status: "downloading", progress: 98, message: "Downloading: 98% · 149.0/152.1 MB" };
    assert.equal(widget._statusLabel("downloading", widget.dlStatus[key]), "Downloading: 98% · 149.0/152.1 MB");
    assert.equal(fullCard.classList.contains("vnccs-cc-lora-card--compact"), false);
});

test("LoRA status colors match fresh cards after live queue and error transitions", () => {
    const { widget } = setup();
    const colorOf = element => {
        let color;
        for (const [, selector, declarations] of css.matchAll(/(?:^|\n)(\.[\w-]+)\s*\{([^}]*)\}/g)) {
            if (!element.classList.contains(selector.slice(1))) continue;
            const declared = declarations.match(/(?:^|;)\s*color:\s*([^;]+);/);
            if (declared) color = declared[1];
        }
        return color;
    };
    for (const compact of [false, true]) {
        for (const initial of ["missing", "queued", "error", "installed"]) {
            const entry = { name: `LoRA ${initial}`, kind: "QI2", status: initial };
            const key = `cc_lora_${entry.name}`;
            widget.dlStatus = {};
            const card = widget._renderLoraCard(entry, { compact, mode: "pipe" });
            widget.scrollArea.appendChild(card);
            const label = card.querySelector(".vnccs-cc-lora-card-status");
            for (const [status, expected] of [
                ["queued", "#b8a9e8"], ["downloading", "#b8a9e8"],
                ["error", "#ff4757"], ["queued", "#b8a9e8"], ["installed", "#00d68f"],
            ]) {
                if (status === "installed") entry.status = status;
                widget.dlStatus[key] = { status, progress: 37 };
                widget._updateDownloadIndicators();
                const fresh = widget._renderLoraCard({ ...entry, status }, { compact, mode: "pipe" });
                assert.equal(colorOf(label), expected, `${initial} to ${status} must update the existing label color`);
                assert.equal(colorOf(label), colorOf(fresh.querySelector(".vnccs-cc-lora-card-status")));
                assert.equal(label.classList.contains("vnccs-cc-model-card-status--missing"), false);
            }
            card.remove();
        }
    }
});

test("nested dialogs contain Tab, close with Escape, and return focus without scrolling or saving drafts", () => {
    const { widget, document, serialized, flush, focused } = setup();
    const opener = focused("CFG");
    opener.focus();
    widget.scrollArea.scrollTop = 200;
    const original = serialized.value;
    widget._showSettings();
    flush();
    const settings = widget._overlays[0];
    const panel = settings.children[0];
    assert.equal(panel.role, "dialog");
    assert.equal(panel.ariaModal, "true");
    const buttons = panel.querySelectorAll("button");
    const first = document.activeElement;
    assert.equal(first.className, "vnccs-cc-settings-summary");
    assert.equal(first.ariaExpanded, "true");
    panel.querySelector(".vnccs-cc-settings-select").value = "fp8_e4m3fn";
    buttons.at(-1).focus();
    settings.onkeydown({ key: "Tab", preventDefault() {}, stopPropagation() {} });
    assert.equal(document.activeElement, first);
    settings.onkeydown({ key: "Tab", shiftKey: true, preventDefault() {}, stopPropagation() {} });
    assert.equal(document.activeElement, buttons.at(-1));
    widget.showMessage("A long nested error", true);
    flush();
    const message = widget._overlays.at(-1);
    message.onkeydown({ key: "Escape", preventDefault() {}, stopPropagation() {} });
    assert.equal(document.activeElement, buttons.at(-1));
    assert.equal(panel.querySelector(".vnccs-cc-settings-select").value, "fp8_e4m3fn");
    settings.onkeydown({ key: "Escape", preventDefault() {}, stopPropagation() {} });
    assert.equal(document.activeElement, opener);
    assert.equal(widget.scrollArea.scrollTop, 200);
    assert.equal(serialized.value, original);
    assert.equal(widget._overlays.length, 0);
});
