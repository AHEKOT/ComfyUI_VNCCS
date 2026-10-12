import { createWidgetContext } from './widget_context.mjs';
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_control_center.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");

function setup(fetchApi) {
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
    const context = createWidgetContext({
        document,
        api: { fetchApi: fetchApi || (async () => ({ ok: true, json: async () => statuses })) },
        setInterval: callback => { poll = callback; return 1; },
        clearInterval: () => {},
    });
    vm.runInContext(common.slice(common.indexOf("export function registerCleanup"), common.indexOf("// ── DOM Widget Width Sync"))
        .replaceAll("export ", "") + source.slice(source.indexOf("class VNCCSControlCenterWidget"))
        + "\nthis.Widget = VNCCSControlCenterWidget;", context);
    const node = {};
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        node,
        _beginDownloadStatusRequest: context.createRequestGuard(node),
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
        widget, node, row, label, badge, document, input, context,
        pollOnce: () => poll(),
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
    assert.equal(state.row.bg.style.transform, "scaleX(0.37)");
    assert.equal(state.badge.textContent, "⬇");
    assert.equal(state.widget.renders, 0);
    assert.equal(state.document.activeElement, state.input);
    assert.equal(state.widget.scrollArea.scrollTop, 120);
    assert.equal(state.widget.scrollArea.scrollLeft, 8);
});

test("idle download progress updates retain the layer without rebuilding the cards", async () => {
    const state = setup();
    state.document.activeElement = null;
    await state.poll({ status: "downloading", progress: 10 });
    const layer = state.row.bg;
    await state.poll({ status: "downloading", progress: 45 });
    assert.equal(state.row.bg, layer);
    assert.equal(layer.style.transform, "scaleX(0.45)");
    assert.equal(state.widget.renders, 0);
    await state.poll({ status: "error", message: "Transfer interrupted" });
    assert.equal(state.widget.renders, 1, "An error rebuilds the card to expose its retry button");
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
    const layer = row.bg;
    assert.match(row.bg.style.background, /repeating-linear-gradient/);
    widget._applyProgressLayer(row, "cc_lora_model", { status: "downloading", progress: 0 });
    assert.equal(row.bg, layer, "Progress updates retain the same layer so CSS can interpolate");
    assert.equal(row.bg.style.transform, "scaleX(0)");
    widget._applyProgressLayer(row, "cc_lora_model", { status: "downloading", progress: 150 });
    assert.equal(row.bg.style.transform, "scaleX(1)");
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

test("a slow status poll prevents overlapping requests and still publishes its response", async () => {
    const pending = [];
    const state = setup(() => new Promise(resolve => pending.push(resolve)));
    const old = state.pollOnce();
    await state.pollOnce();
    await state.pollOnce();
    assert.equal(pending.length, 1);
    pending[0]({ ok: true, json: async () => ({ cc_lora_model: { status: "success" } }) });
    await old;
    assert.equal(state.widget.dlStatus.cc_lora_model.status, "success");
    assert.equal(state.widget._downloadRefreshPending, true);
    assert.equal(state.widget._downloadStatusPolling, false);
    const next = state.pollOnce();
    assert.equal(pending.length, 2);
    pending[1]({ ok: true, json: async () => ({ cc_lora_model: { status: "error", message: "Retry failed" } }) });
    await next;
    assert.equal(state.label.textContent, "Retry failed");
});

test("removal invalidates an in-flight status poll before DOM updates or refresh", async () => {
    let resolve;
    const state = setup(() => new Promise(complete => { resolve = complete; }));
    state.document.activeElement = null;
    const request = state.pollOnce();
    state.node.onRemoved();
    resolve({ ok: true, json: async () => ({ cc_lora_model: { status: "success" } }) });
    await request;
    assert.equal(state.widget.dlStatus.cc_lora_model.status, "queued");
    assert.equal(state.widget._downloadRefreshPending, false);
    assert.equal(state.widget.refreshes, 0);
    assert.equal(state.widget.renders, 0);
    assert.equal(state.widget._downloadStatusPolling, false);
});

for (const previousStatus of [{}, { cc_lora_model: { status: "success" } }]) {
    test(`a status snapshot started before a new download cannot erase its queued state (${JSON.stringify(previousStatus)})`, async () => {
        const pending = [];
        const state = setup((route, options) => new Promise(resolve => pending.push({ route, options, resolve })));
        state.widget.state = {};
        state.widget._activeKind = () => "QI2";
        state.widget.dlStatus = {};
        state.context.CustomEvent = class { constructor(type, options) { this.type = type; this.detail = options.detail; } };
        state.context.window.dispatchEvent = () => {};

        const previousPoll = state.pollOnce();
        const download = state.widget._downloadEntry("lora", { name: "model" });
        assert.equal(state.widget.dlStatus.cc_lora_model.status, "queued");
        assert.equal(pending[0].route, "/vnccs/manager/status?repo_id=public%2Fmodels");
        assert.equal(pending[1].route, "/vnccs/control_center/download");
        pending[0].resolve({ ok: true, json: async () => previousStatus });
        await previousPoll;
        assert.equal(state.widget.dlStatus.cc_lora_model.status, "queued");
        assert.equal(state.widget._downloadStatusPolling, false);

        pending[1].resolve({ ok: true, text: async () => '{"status":"queued"}' });
        await download;
        const currentPoll = state.pollOnce();
        pending[2].resolve({ ok: true, json: async () => ({ cc_lora_model: { status: "downloading", progress: 25 } }) });
        await currentPoll;
        assert.equal(state.widget.dlStatus.cc_lora_model.status, "downloading");
        assert.equal(state.widget.dlStatus.cc_lora_model.progress, 25);
    });
}
