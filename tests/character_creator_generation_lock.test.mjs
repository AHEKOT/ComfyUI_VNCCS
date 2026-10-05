import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
function between(text, start, end) {
    const offset = text.indexOf(start);
    const limit = text.indexOf(end, offset);
    assert.ok(offset >= 0 && limit > offset);
    return text.slice(offset, limit);
}
const guards = between(common, "export function registerCleanup", "// ── Widget Data Sync").replaceAll("export ", "");
const response = (data, ok = true) => ({ ok, status: ok ? 200 : 500, json: async () => data, text: async () => "sampler failed" });

function setup() {
    const listeners = new Map(), requests = [], messages = [];
    const api = {
        addEventListener(name, callback) {
            if (!listeners.has(name)) listeners.set(name, new Set());
            listeners.get(name).add(callback);
        },
        removeEventListener(name, callback) { listeners.get(name)?.delete(callback); },
        fetchApi(route) {
            let resolve;
            const promise = new Promise(done => { resolve = done; });
            requests.push({ route, resolve });
            return promise;
        },
    };
    const node = { _randomizeSeedIfNeeded() {} };
    const context = createWidgetContext({
        api, node, els: { btnGen: {} }, container: {},
        state: { character: "Alice", character_info: {}, gen_settings: {
            generation_mode: "qi2", diffusion_model_name: "model", clip_name: "clip", vae_name: "vae", lora_stack: [],
        } },
        console: { warn() {} }, isSelectedCcAssetInstalled: () => true,
        saveCurrentGenerationModeValues() {}, saveState() {}, refreshPreviewImage() {},
        createLoadingOverlay: () => ({ remove() {} }),
        showAlertModal() { assert.fail("Disabled generation must not show dialogs"); },
        showMessage: (_container, message) => messages.push(message),
    });
    vm.runInContext(`${guards}
        const beginPreviewRequest = createRequestGuard(node);
        ${between(source, "let previewRunning = false;", "registerCleanup(node, () => stopCcPolling());")}
        ${between(source, "const doGenerate = async", "// 7. Graph Restore Hook / Main Entry Point")}
        globalThis.generate = doGenerate; globalThis.refreshBusy = refreshWorkflowBusy;`, context);
    const emit = (name, detail) => {
        for (const callback of listeners.get(name) || []) callback({ detail });
    };
    return { context, node, requests, listeners, messages, button: context.els.btnGen,
        emit, status: count => emit("status", { exec_info: { queue_remaining: count } }) };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

for (const field of ["queue_running", "queue_pending"]) {
    test(`preview remains disabled when restored ${field} is busy`, async () => {
        const h = setup();
        assert.equal(h.button.disabled, true);
        h.requests[0].resolve(response({ queue_running: [], queue_pending: [], [field]: [{}] }));
        await tick();
        await h.context.generate();
        assert.equal(h.button.disabled, true);
        assert.equal(h.requests.length, 1, "Disabled click must not submit a preview");
        h.status(0);
        assert.equal(h.button.disabled, false);
        h.emit("execution_start", {});
        assert.equal(h.button.disabled, true);
        h.status(0);
        assert.equal(h.button.innerText, "GENERATE PREVIEW");
    });
}

test("late idle queue snapshot cannot override a workflow start event", async () => {
    const h = setup();
    h.emit("execution_start", {});
    h.requests[0].resolve(response({ queue_running: [], queue_pending: [] }));
    await tick();
    assert.equal(h.button.disabled, true);
});

for (const ok of [true, false]) {
    test(`preview ${ok ? "completion" : "failure"} respects workflow busy and pending preview state`, async () => {
        const h = setup();
        h.status(0);
        const work = h.context.generate();
        assert.equal(h.requests[1].route, "/vnccs/preview_generate");
        h.status(0);
        assert.equal(h.button.disabled, true, "Idle workflow must not unlock an active preview");
        assert.equal(h.button.innerText, "GENERATING...");
        h.status(1);
        h.requests[1].resolve(response({}, ok));
        await work;
        assert.equal(h.button.disabled, true, "Preview result must not unlock a running workflow");
        h.status(0);
        assert.equal(h.button.disabled, false);
        assert.equal(h.messages.length, ok ? 0 : 1);
    });
}

test("reconnect refreshes queue and removal discards pending snapshots and listeners", async () => {
    const h = setup();
    h.status(0);
    h.emit("reconnecting");
    assert.equal(h.button.disabled, true);
    h.emit("reconnected");
    h.requests[1].resolve(response({ queue_running: [], queue_pending: [] }));
    await tick();
    assert.equal(h.button.disabled, false);
    const pending = h.context.refreshBusy();
    h.node.onRemoved();
    h.requests[2].resolve(response({ queue_running: [{}], queue_pending: [] }));
    await pending;
    assert.equal(h.button.disabled, false);
    assert.ok([...h.listeners.values()].every(callbacks => callbacks.size === 0));
});

test("unknown or failed queue state keeps preview disabled until authoritative status arrives", async () => {
    const h = setup();
    h.requests[0].resolve(response({}));
    await tick();
    assert.equal(h.button.disabled, true);
    h.status(0);
    const pending = h.context.refreshBusy();
    h.requests[1].resolve(response({}, false));
    await pending;
    assert.equal(h.button.disabled, true);
    h.status(0);
    assert.equal(h.button.disabled, false);
});
