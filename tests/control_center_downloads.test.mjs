import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_control_center.js", import.meta.url), "utf8");

function setup() {
    const input = {};
    const label = { textContent: "⏳ Queued" };
    const badge = {};
    const row = {
        dataset: { downloadKey: "cc_lora_model" },
        classList: { contains: () => false },
        bg: null,
        querySelector(selector) {
            if (selector === ".vnccs-cc-row-bg") return this.bg;
            if (selector === ".vnccs-cc-badge") return badge;
            return label;
        },
        insertBefore(element) {
            this.bg = element;
            element.remove = () => { this.bg = null; };
        },
    };
    let poll;
    let statuses = {};
    const document = {
        activeElement: input,
        createElement: () => ({ style: {}, dataset: {} }),
    };
    const context = vm.createContext({
        document,
        api: { fetchApi: async () => ({ ok: true, json: async () => statuses }) },
        setInterval: callback => { poll = callback; return 1; },
        clearInterval: () => {},
    });
    vm.runInContext(source.slice(source.indexOf("class VNCCSControlCenterWidget"))
        + "\nthis.Widget = VNCCSControlCenterWidget;", context);
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        container: { contains: element => element === input },
        scrollArea: { scrollTop: 120, scrollLeft: 8, querySelectorAll: () => [row] },
        dlStatus: { cc_lora_model: { status: "queued" } },
        _downloadRefreshPending: false,
        renders: 0,
        refreshes: 0,
        _renderAll() { this.renders++; },
        async fetchConfig() { this.refreshes++; },
        _getRepoId: () => "public/models",
    });
    widget._startPolling();
    return {
        widget, row, label, badge, document, input,
        async poll(status) {
            statuses = { cc_lora_model: status };
            await poll();
        },
    };
}

test("polling updates real progress while preserving the focused input and scroll", async () => {
    const state = setup();
    await state.poll({ status: "downloading", progress: 37, message: "Downloading: 37%" });
    assert.equal(state.label.textContent, "Downloading: 37%");
    assert.equal(state.row.bg.style.width, "37%");
    assert.equal(state.badge.textContent, "⬇");
    assert.equal(state.widget.renders, 0);
    assert.equal(state.document.activeElement, state.input);
    assert.equal(state.widget.scrollArea.scrollTop, 120);
    assert.equal(state.widget.scrollArea.scrollLeft, 8);
});

test("completion while editing keeps its pending refresh until the input loses focus", async () => {
    const state = setup();
    await state.poll({ status: "success" });
    await state.poll({ status: "success" });
    assert.equal(state.widget.refreshes, 0);
    assert.equal(state.widget._downloadRefreshPending, true);
    state.document.activeElement = null;
    await state.poll({ status: "success" });
    assert.equal(state.widget.refreshes, 1);
    assert.equal(state.widget._downloadRefreshPending, false);
    await state.poll({ status: "success" });
    assert.equal(state.widget.refreshes, 1);

    state.label.textContent = "Pipe";
    state.document.activeElement = state.input;
    await state.poll({ status: "success" });
    assert.equal(state.label.textContent, "Pipe");
});

test("unknown progress remains indeterminate, and measured zero remains zero", () => {
    const { widget, row } = setup();
    widget._applyProgressLayer(row, "cc_lora_model", { status: "downloading" });
    assert.match(row.bg.style.background, /repeating-linear-gradient/);
    widget._applyProgressLayer(row, "cc_lora_model", { status: "downloading", progress: 0 });
    assert.equal(row.bg.style.width, "0%");
    widget._applyProgressLayer(row, "cc_lora_model", { status: "downloading", progress: 150 });
    assert.equal(row.bg.style.width, "100%");
    widget._applyProgressLayer(row, "cc_lora_model", { status: "success" });
    assert.equal(row.bg, null);
});

test("download failures become visible without replacing the focused form", async () => {
    const state = setup();
    await state.poll({ status: "error", message: "Interrupted transfer" });
    assert.equal(state.label.textContent, "Interrupted transfer");
    assert.equal(state.badge.textContent, "✕");
    assert.equal(state.widget.renders, 0);
    assert.equal(state.document.activeElement, state.input);
});
