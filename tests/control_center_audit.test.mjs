import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { createWidgetContext } from "./widget_context.mjs";

const source = readFileSync(new URL("../web/vnccs_control_center.js", import.meta.url), "utf8");
const common = readFileSync(new URL("../web/vnccs_common.js", import.meta.url), "utf8");
const constants = source.slice(source.indexOf("const INITIAL_NODE_W"), source.indexOf("function _injectVNCCSControlCenterStyles"));
const catalog = { name: "Catalog", models: [
    { name: "QI2", kind: "QI2", type: "unet" },
    { name: "Klein A", kind: "Klein9b", type: "unet" },
    { name: "Klein B", kind: "Klein9b", type: "unet" },
], lora: [] };

function setup(state = {}) {
    const pending = [];
    const timers = [];
    const context = createWidgetContext({
        api: { fetchApi: (route, options) => new Promise((resolve, reject) => pending.push({ route, options, resolve, reject })) },
        window: { addEventListener() {}, removeEventListener() {}, dispatchEvent() {} },
        document: { createElement: () => ({}) },
        CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
        setTimeout: callback => { timers.push(callback); return timers.length; },
        clearTimeout() {},
        _injectVNCCSControlCenterStyles() {},
    });
    vm.runInContext(common.slice(common.indexOf("export function registerCleanup"), common.indexOf("// ── DOM Widget Width Sync"))
        .replaceAll("export ", "") + constants + source.slice(source.indexOf("class VNCCSControlCenterWidget"))
        + "\nthis.Widget = VNCCSControlCenterWidget;", context);
    const serialized = { name: "node_state", value: typeof state === "string" ? state : JSON.stringify(state) };
    const node = { id: 1, widgets: [serialized], setDirtyCanvas() {} };
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        node, state, config: catalog, statusText: {}, scrollArea: { appendChild() {}, querySelector: () => null },
        _beginConfigRequest: context.createRequestGuard(node),
        _beginDownloadStatusRequest: context.createRequestGuard(node),
        _getRepoId: () => "", _syncOutputSlots() {}, _syncCnetSlots() {},
        _syncCustomModelInput() { this.inputSyncs = (this.inputSyncs || 0) + 1; }, _dispatchLoraOptions() {}, _renderAll() {},
        showMessage(message) { this.message = message; },
    });
    return { widget, node, serialized, pending, context, timers };
}

function respond(request, data) {
    request.resolve({ ok: true, json: async () => data });
}

for (const selectedType of ["unet", "custom"]) {
    test(`legacy per-type model selection survives restoration and unrelated edits (${selectedType})`, () => {
        const original = JSON.stringify({ selected_type: selectedType, selected_models: { unet: " qi2 alternative " } });
        const { widget, serialized } = setup(original);
        widget.config = { ...catalog, models: [
            { name: "QI2", kind: "QI2", type: "unet" },
            { name: "QI2 Alternative", kind: "QI2", type: "unet" },
        ] };
        widget.restoreState();
        assert.equal(serialized.value, original, "Restoration must not rewrite legacy workflow data");
        assert.equal(widget._getSelectedModelEntry().name, "QI2 Alternative");
        widget._modelParamsSave({ cfg: 2 });
        assert.equal(widget._getSelectedModelEntry().name, "QI2 Alternative");
        const saved = JSON.parse(serialized.value);
        assert.equal(saved.selected_models.unet, " qi2 alternative ");
        if (selectedType === "unet") assert.equal(saved.selected_model, "QI2 Alternative");
    });
}

for (const kind of ["Klein9b", "MiniMaxH3"]) {
    test(`legacy custom model context keeps its active family (${kind})`, () => {
        const { widget, serialized } = setup({ active_kind: kind, selected_type: "custom",
            selected_models: { unet: "Alternative" } });
        widget.config = { ...catalog, models: [
            { name: "First", kind, type: "unet" }, { name: "Alternative", kind, type: "unet" },
        ] };
        const original = serialized.value;
        widget.restoreState();
        assert.equal(widget._getSelectedModelEntry().name, "Alternative");
        assert.equal(serialized.value, original);
        widget._modelParamsSave({ cfg: 2 });
        assert.equal(widget._getSelectedModelEntry().name, "Alternative");
    });
}

for (const selectedType of ["unet", "custom"]) {
    for (const [selectedModel, familyName, legacyName, expected] of [
        ["QI2", "", "QI2 Alternative", selectedType === "custom" ? "QI2 Alternative" : "QI2"],
        ["Removed", "", "QI2 Alternative", "QI2 Alternative"],
        ["QI2 Alternative", "QI2", "QI2 Alternative", "QI2"],
        ["QI2 Alternative", "Removed", "QI2 Alternative", "QI2"],
        ["QI2 Alternative", "", "", "QI2 Alternative"],
    ]) {
        test(`legacy model priority matches execution (${selectedType}, global=${selectedModel}, family=${familyName}, legacy=${legacyName})`, () => {
            const { widget, serialized } = setup({ active_kind: "QI2", selected_type: selectedType,
                selected_model: selectedModel, selected_models: {
                    ...(familyName ? { "QI2:unet": familyName } : {}), ...(legacyName ? { unet: legacyName } : {}),
                } });
            widget.config = { ...catalog, models: [
                { name: "QI2", kind: "QI2", type: "unet" },
                { name: "QI2 Alternative", kind: "QI2", type: "unet" },
            ] };
            const original = serialized.value;
            widget.restoreState();
            assert.equal(widget._getSelectedModelEntry().name, expected);
            assert.equal(serialized.value, original);
        });
    }
}

test("a slower previous repository cannot replace the current catalog", async () => {
    const { widget, pending } = setup();
    const old = widget.fetchConfig("owner/old");
    const current = widget.fetchConfig("owner/current");
    respond(pending[1], { ...catalog, name: "Current" });
    await current;
    respond(pending[0], { ...catalog, name: "Old" });
    await old;
    assert.equal(widget.config.name, "Current");
    assert.equal(widget.statusText.textContent, "Current");
});

test("a superseded force refresh does not delete or overwrite its newer shared promise", async () => {
    const { widget, pending, context } = setup();
    const old = widget.fetchConfig("owner/models", true);
    const current = widget.fetchConfig("owner/models", true);
    const registry = context.serverRegistry("VNCCS_CC_FETCH_PROMISES");
    const latest = registry["owner/models"];
    respond(pending[0], { ...catalog, name: "Old" });
    await old;
    assert.equal(registry["owner/models"], latest);
    respond(pending[1], { ...catalog, name: "Current" });
    await current;
    assert.equal(widget.config.name, "Current");
    assert.equal(context.serverRegistry("VNCCS_CC_REGISTRY")["owner/models"].name, "Current");
});

test("removal invalidates pending catalog responses", async () => {
    const { widget, node, pending } = setup();
    const request = widget.fetchConfig("owner/models");
    node.onRemoved();
    respond(pending[0], { ...catalog, name: "Removed" });
    await request;
    assert.equal(widget.config, catalog);
    assert.equal(widget.statusText.textContent, "Loading…");
});

test("another catalog's same-name download does not replace a rejected target's status", async () => {
    const { widget, context, pending } = setup();
    widget._getRepoId = () => "second/catalog";
    widget.dlStatus = { cc_lora_Style: { status: "error", message: "Wait, then retry" } };
    widget._updateDownloadIndicators = () => {};
    widget._isUserInteracting = () => true;
    let poll;
    context.setInterval = callback => { poll = callback; return 1; };
    context.clearInterval = () => {};
    widget._startPolling();
    const request = poll();
    const read = pending.at(-1);
    respond(read, read.route.includes("repo_id=second%2Fcatalog") ? {}
        : { cc_lora_Style: { status: "downloading", progress: 20 } });
    await request;
    assert.equal(widget._resolveStatus(widget.dlStatus.cc_lora_Style?.status, "missing"), "missing",
        "The rejected target remains downloadable rather than borrowing another catalog's progress");
    assert.equal(read.route, "/vnccs/manager/status?repo_id=second%2Fcatalog");
});

test("a catalog change clears the previous repository's progress before fetching", async () => {
    const { widget, pending } = setup();
    widget._downloadStatusRepoId = "first/catalog";
    widget.dlStatus = { cc_lora_Style: { status: "downloading" } };
    widget._downloadRefreshPending = true;
    const request = widget.fetchConfig("second/catalog");
    assert.deepEqual(Object.keys(widget.dlStatus), []);
    assert.equal(widget._downloadRefreshPending, false);
    respond(pending.at(-1), catalog);
    await request;
    widget.dlStatus = { cc_lora_Style: { status: "downloading" } };
    const refresh = widget.fetchConfig("second/catalog", true);
    assert.equal(widget.dlStatus.cc_lora_Style.status, "downloading", "Same-repository refresh preserves progress");
    respond(pending.at(-1), catalog);
    await refresh;
});

test("a repository change discards its pending download status snapshot", async () => {
    const { widget, context, pending } = setup();
    let repo = "first/catalog", updates = 0;
    widget._getRepoId = () => repo;
    widget.dlStatus = {};
    widget._updateDownloadIndicators = () => { updates++; };
    widget._isUserInteracting = () => true;
    let poll;
    context.setInterval = callback => { poll = callback; return 1; };
    context.clearInterval = () => {};
    widget._startPolling();
    const request = poll();
    repo = "second/catalog";
    respond(pending.at(-1), { cc_lora_Style: { status: "downloading" } });
    await request;
    assert.equal(updates, 0);
    assert.deepEqual(Object.keys(widget.dlStatus), []);
});

test("all nodes sharing a failed request show the catalog error", async () => {
    const first = setup();
    const other = Object.create(first.context.Widget.prototype);
    Object.assign(other, first.widget, { statusText: {}, node: { setDirtyCanvas() {} } });
    other._beginConfigRequest = first.context.createRequestGuard(other.node);
    const requests = [first.widget.fetchConfig("owner/models"), other.fetchConfig("owner/models")];
    assert.equal(first.pending.length, 1);
    first.pending[0].reject(new Error("Catalog unavailable"));
    await Promise.all(requests);
    assert.equal(first.widget.statusText.textContent, "Error");
    assert.equal(other.statusText.textContent, "Error");
});

test("both nodes sharing a catalog synchronize custom model sockets after loading", async () => {
    const first = setup();
    const other = Object.create(first.context.Widget.prototype);
    Object.assign(other, first.widget, { statusText: {}, node: { setDirtyCanvas() {} } });
    other._beginConfigRequest = first.context.createRequestGuard(other.node);
    const requests = [first.widget.fetchConfig("owner/models"), other.fetchConfig("owner/models")];
    assert.equal(first.pending.length, 1);
    respond(first.pending[0], catalog);
    await Promise.all(requests);
    assert.equal(first.widget.inputSyncs, 1);
    assert.equal(other.inputSyncs, 1);
});

for (const order of ["old first", "refresh first"]) {
    for (const failure of [false, true]) {
        test(`another node's force refresh settles every waiter (${order}, failure=${failure})`, async () => {
            const first = setup();
            const other = Object.create(first.context.Widget.prototype);
            Object.assign(other, first.widget, { statusText: {}, node: { setDirtyCanvas() {} } });
            other._beginConfigRequest = first.context.createRequestGuard(other.node);
            const requests = [first.widget.fetchConfig("owner/models"), other.fetchConfig("owner/models", true)];
            const settleOld = () => respond(first.pending[0], { ...catalog, name: "Old" });
            const settleRefresh = () => failure
                ? first.pending[1].reject(new Error("Refresh unavailable"))
                : respond(first.pending[1], { ...catalog, name: "Refreshed" });
            if (order === "old first") settleOld(); else settleRefresh();
            await new Promise(resolve => setImmediate(resolve));
            if (order === "old first") settleRefresh(); else settleOld();
            await Promise.all(requests);
            for (const widget of [first.widget, other]) {
                assert.equal(widget.statusText.textContent, failure ? "Error" : "Refreshed");
                if (!failure) assert.equal(widget.config.name, "Refreshed");
            }
        });
    }
}

test("invalid restored state stays byte-for-byte recoverable through edits", () => {
    for (const raw of ['{"selected_model":', "null", "[]"]) {
        const { widget, serialized } = setup(raw);
        widget.restoreState();
        assert.match(widget.message, /Original workflow data is preserved/);
        widget._saveState();
        assert.equal(serialized.value, raw);
    }
});

test("invalid nested state stays byte-for-byte recoverable through edits and reconfiguration", () => {
    for (const state of [
        { selected_models: [] }, { selected_types_by_kind: { QI2: [] } },
        { model_params: ["authored"] }, { model_params_by_kind: { QI2: ["authored"] } },
        { type_settings: { unet: "authored" } }, { loras: [null] },
        { qi2_cache: [] }, { collapsed: [] },
        { model_params: { steps: [] } }, { model_params: { cfg: {} } },
        { model_params_by_kind: { QI2: { steps: "broken" } } },
        { model_params: { sampler: [] } }, { model_params: { scheduler: {} } },
        { model_params: { cfg: "nan" } },
        { model_params: { steps: "2.5" } }, { model_params: { steps: "1e2" } },
        { model_params: { steps: "0x10" } }, { model_params: { steps: "1__0" } },
        { model_params: { cfg: "0x10" } }, { model_params: { cfg: "1__0" } },
        { loras: [{ name: "Style", strength: "0x10" }] },
        { qi2_cache: { device: [] } }, { qi2_cache: { dtype: {} } },
        { loras: [{ name: "Style", auto_apply: true, strength: [] }] },
    ]) {
        const raw = JSON.stringify(state);
        const { widget, serialized } = setup(raw);
        widget.restoreState();
        assert.match(widget.message, /Original workflow data is preserved/);
        assert.doesNotThrow(() => widget._saveState());
        assert.equal(serialized.value, raw);
        serialized.value = "{}";
        widget.restoreState();
        assert.equal(widget._stateRestoreError, null);
        widget._saveState();
        assert.equal(JSON.parse(serialized.value).active_kind, "QI2");
    }
});

test("numeric strings and null parameter fallbacks remain valid on restoration", () => {
    const state = { model_params: { steps: "25", cfg: "3.5", sampler: null },
        model_params_by_kind: { QI2: { steps: null, cfg: null, scheduler: null } },
        qi2_cache: { device: null, dtype: null }, loras: [{ strength: "-0.5" }] };
    const { widget } = setup(state);
    widget.restoreState();
    assert.equal(widget._stateRestoreError, null);
    assert.equal(widget.state.model_params.cfg, "3.5");
    assert.equal(widget.state.model_params_by_kind.QI2.steps, null);
    assert.equal(widget.state.loras[0].strength, "-0.5");
});

test("backend-compatible decimal strings remain valid and unchanged on restoration", () => {
    const state = { model_params: { steps: " +2_5 ", cfg: "3_0.5e-1" },
        model_params_by_kind: { QI2: { steps: true, cfg: ".5" } },
        loras: [{ strength: "-1_0.5e-1" }] };
    const { widget, serialized } = setup(state);
    const original = serialized.value;
    widget.restoreState();
    assert.equal(widget._stateRestoreError, null);
    assert.equal(serialized.value, original);
    assert.equal(widget.state.model_params.steps, state.model_params.steps);
    assert.equal(widget.state.loras[0].strength, state.loras[0].strength);
});

test("invalid numeric edits preserve the last valid serialized parameter", () => {
    const { widget, context, serialized } = setup({ active_kind: "QI2", model_params: { steps: 41 } });
    context.document.createElement = () => ({ checkValidity() { return this.valid; } });
    const input = widget._numInput(41, 1, 200, 1, value => widget._modelParamsSave({ steps: value }));
    input.valid = true;
    input.value = "57";
    input.valueAsNumber = 57;
    input.onchange();
    assert.equal(JSON.parse(serialized.value).model_params.steps, 57);
    const saved = serialized.value;
    for (const [value, valid] of [["", true], ["-1", false], ["201", false], ["2.5", false]]) {
        input.value = value;
        input.valueAsNumber = value ? Number(value) : NaN;
        input.valid = valid;
        input.onchange();
        assert.equal(serialized.value, saved);
        assert.equal(Number(input.value), 57);
    }
});

function installNumericDom(context) {
    const elements = [];
    context.setHelpText = () => {};
    context.document.createElement = tag => {
        let value = "";
        const element = { tag, children: [], dataset: {}, style: {},
            classList: { add() {}, toggle() {} },
            appendChild(child) { this.children.push(child); },
            append(...children) { this.children.push(...children); },
            querySelector: () => null,
            checkValidity() { return this.value !== ""; },
            get value() { return tag === "select" ? (this.children.find(child => child.selected) || this.children[0])?.value ?? "" : value; },
            get valueAsNumber() { return value === "" ? NaN : Number(value); },
            set value(next) {
                const text = String(next);
                // Native numeric inputs sanitize Python-compatible strings and booleans.
                const valid = /^-?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text)
                    && Number.isFinite(Number(text));
                value = this.type === "number" && !valid ? "" : this.type === "range"
                    ? String(valid ? Math.min(this.max, Math.max(this.min, Number(text))) : (Number(this.min) + Number(this.max)) / 2)
                    : text;
            },
        };
        elements.push(element);
        return element;
    };
    return elements;
}

test("restored sampling controls display backend values without rewriting authored state", () => {
    for (const [kind, steps, cfg, expectedSteps, expectedCfg] of [
        ["QI2", " +2_5 ", "3_0.0e-1", 25, 3],
        ["QI2", true, true, 1, 1],
        ["QI2", null, null, 25, 3],
        ["QI2", false, false, 25, 0],
        ["QI2", "", 0, 25, 0],
        ["QI2", 2.9, " +.5 ", 2, 0.5],
        ["Klein9b", null, null, 4, 1],
        ["MiniMaxH3", null, null, 8, 1],
    ]) {
        const { widget, context, serialized } = setup({ active_kind: kind, model_params: { steps, cfg } });
        const original = serialized.value;
        widget.restoreState();
        const elements = installNumericDom(context);
        widget._renderModelParams();
        const [stepInput, cfgInput] = elements.filter(element => element.type === "number");
        assert.equal(stepInput.valueAsNumber, expectedSteps, `${kind} steps=${JSON.stringify(steps)}`);
        assert.equal(cfgInput.valueAsNumber, expectedCfg, `${kind} cfg=${JSON.stringify(cfg)}`);
        assert.equal(serialized.value, original);
        stepInput.value = "";
        stepInput.onchange();
        assert.equal(stepInput.valueAsNumber, expectedSteps);
        assert.equal(serialized.value, original);
        cfgInput.value = "2";
        cfgInput.onchange();
        assert.equal(JSON.parse(serialized.value).model_params.steps, steps,
            "Editing CFG must retain the authored steps representation");
    }
});

test("restored LoRA strength displays the backend value and only changes after an edit", () => {
    for (const [strength, expected] of [[" +1_0e-1 ", 1], [true, 1], [false, 0], [" -.5 ", -0.5], [4, 4], ["-4", -4]]) {
        const entry = { name: "Style", kind: "QI2", custom: true, status: "installed" };
        const { widget, context, serialized } = setup({ loras: [{ name: entry.name, auto_apply: true, strength }] });
        widget.dlStatus = {};
        const original = serialized.value;
        widget.restoreState();
        const elements = installNumericDom(context);
        widget._renderLoraCard(entry, { active: true });
        const slider = elements.find(element => element.type === "range");
        assert.equal(Number(slider.value), expected);
        assert.equal(elements.find(element => element.className === "vnccs-cc-lora-val").textContent, expected.toFixed(2));
        assert.equal(serialized.value, original);
        slider.value = "0.5";
        slider.oninput();
        assert.equal(JSON.parse(serialized.value).loras[0].strength, 0.5);
    }
});

test("restored empty sampler settings show the effective family defaults", () => {
    for (const kind of ["QI2", "Klein9b", "MiniMaxH3"]) {
        for (const empty of [null, ""]) {
            const { widget, context, serialized } = setup({ active_kind: kind,
                model_params: { sampler: empty, scheduler: empty } });
            const original = serialized.value;
            widget.restoreState();
            widget._samplers = ["euler", "res_multistep"];
            widget._schedulers = ["normal", "simple"];
            const elements = installNumericDom(context);
            widget._renderModelSamplerParams();
            assert.equal(elements.find(element => element.id === "vnccs-cc-sampler").value,
                kind === "MiniMaxH3" ? "res_multistep" : "euler");
            assert.equal(elements.find(element => element.id === "vnccs-cc-scheduler").value, "simple");
            assert.equal(serialized.value, original);
            widget._modelParamsSave({ cfg: 2 });
            assert.equal(JSON.parse(serialized.value).model_params.sampler, empty);
            assert.equal(JSON.parse(serialized.value).model_params.scheduler, empty);
        }
    }
});

test("dependency installation stops when the node is removed while policy is loading", async () => {
    for (const removed of [false, true]) {
        const { widget, pending } = setup();
        const queued = [];
        widget._queueDependencyInstall = async item => { queued.push(item.key); return "legacy"; };
        const installing = widget._installDependency({ key: "package", manager_id: "registered" }, { textContent: "Install" });
        assert.equal(pending[0].route, "/vnccs/manager/install_policy");
        widget._removed = removed;
        respond(pending[0], { install_allowed: true });
        assert.equal(await installing, !removed);
        assert.deepEqual(queued, removed ? [] : ["package"]);
    }
});

test("removing a node stops new dependency installs after an already queued package", async () => {
    const { widget, pending } = setup();
    const queued = [];
    let finishFirst;
    widget._queueDependencyInstall = async item => {
        queued.push(item.key);
        if (item.key === "first") await new Promise(resolve => { finishFirst = resolve; });
        return "legacy";
    };
    const items = ["first", "second"].map(key => ({ key, manager_id: key }));
    const installing = widget._installAllDependencies(items, { textContent: "Install all" },
        new Map(items.map(item => [item.key, { textContent: "Install" }])));
    respond(pending[0], { install_allowed: true });
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(queued, ["first"]);
    widget._removed = true;
    finishFirst();
    await installing;
    assert.deepEqual(queued, ["first"]);
});

test("removed dependency installs do not create a task through the Manager fallback", async () => {
    for (const removed of [false, true]) {
        const { widget, pending } = setup();
        widget._dependencyInstallTasks = new Map();
        widget._getManagerInstallPolicy = async () => ({ install_allowed: true });
        const installing = widget._installDependency({ key: "package", manager_id: "registered" }, { textContent: "Install" });
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(pending[0].route, "/manager/queue/install");
        widget._removed = removed;
        pending[0].resolve({ ok: false, status: 404 });
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(pending.length, removed ? 1 : 2);
        if (!removed) {
            assert.equal(pending[1].route, "/v2/manager/queue/task");
            pending[1].resolve({ ok: true });
            await new Promise(resolve => setImmediate(resolve));
            assert.equal(pending[2].route, "/v2/manager/queue/start");
            pending[2].resolve({ ok: true });
        }
        assert.equal(await installing, !removed);
        if (removed) assert.equal(widget._dependencyInstallTasks.size, 0);
    }
});

test("legacy model selections and parameters survive restoration and later edits", () => {
    const { widget, serialized } = setup({
        active_kind: "Klein9b", selected_type: "unet", selected_model: "Klein B",
        model_params: { steps: 41, cfg: 2.5, sampler: "heun" },
    });
    widget.restoreState();
    assert.equal(widget._getSelectedModelEntry().name, "Klein B");
    widget._modelParamsSave({ cfg: 3.5 });
    const saved = JSON.parse(serialized.value);
    assert.equal(saved.selected_models["Klein9b:unet"], "Klein B");
    assert.deepEqual(saved.model_params, { steps: 41, cfg: 3.5, sampler: "heun", scheduler: "simple" });
    widget.state.active_kind = "QI2";
    assert.equal(widget._currentModelParams().steps, 25);
});

for (const savedName of ["klein b", "  KLEIN B  "]) {
    for (const layout of ["legacy", "family"]) {
        test(`model identity matches backend through rendering and edits (${layout}, ${JSON.stringify(savedName)})`, () => {
            const { widget, serialized, context } = setup({
                active_kind: "Klein9b", selected_type: "unet",
                ...(layout === "legacy" ? { selected_model: savedName }
                    : { selected_models: { "Klein9b:unet": savedName } }),
            });
            const original = serialized.value;
            widget.restoreState();
            assert.equal(serialized.value, original);
            assert.equal(widget._getSelectedModelEntry().name, "Klein B");
            delete widget._renderAll;
            context.requestAnimationFrame = () => {};
            widget._renderFamilyTabs = () => ({});
            for (const name of ["_renderTwoColBlockSingle", "_renderClipVaeBlock", "_renderLoraBlock",
                "_renderCustomLoraBlock", "_renderBlock"]) widget[name] = () => {};
            widget._buildQI2CacheInlineSection = widget._buildModelTurboInlineSection = () => null;
            widget._renderAll();
            assert.equal(serialized.value, original);
            assert.equal(widget._getSelectedModelEntry().name, "Klein B");
            widget._modelParamsSave({ cfg: 3.5 });
            const saved = JSON.parse(serialized.value);
            assert.equal(saved.selected_model, "Klein B");
            assert.equal(saved.selected_models["Klein9b:unet"], "Klein B");
        });
    }
}

test("H3 restoration keeps backend sampling defaults and authored step counts", () => {
    for (const steps of [undefined, 20]) {
        const state = {
            active_kind: "MiniMaxH3", selected_types_by_kind: { MiniMaxH3: "custom" },
            ...(steps === undefined ? {} : { model_params: { steps } }),
        };
        const { widget, serialized } = setup(state);
        const original = serialized.value;
        widget.restoreState();
        assert.equal(serialized.value, original);
        assert.equal(widget._currentModelParams().steps, steps ?? 8);
        assert.equal(widget._currentModelParams().sampler, "res_multistep");
        widget._modelParamsSave({ cfg: 2 });
        const saved = JSON.parse(serialized.value);
        assert.equal(saved.model_params.steps, steps ?? 8);
        assert.equal(saved.model_params_by_kind.MiniMaxH3.steps, steps ?? 8);
        assert.equal(saved.model_params.cfg, 2);
    }
});

test("switching to H3 serializes the same defaults shown after restoration", () => {
    const { widget, serialized } = setup({ active_kind: "QI2" });
    widget.config = { ...catalog, models: [...catalog.models,
        { name: "H3", kind: "minimaxh3", type: "unet" },
    ] };
    widget._setActiveKind("MiniMaxH3");
    const saved = JSON.parse(serialized.value);
    assert.equal(saved.selected_models["MiniMaxH3:unet"], "H3");
    assert.equal(saved.model_params.steps, 8);
    assert.equal(saved.model_params.sampler, "res_multistep");
});

for (const value of ["{}", ""]) {
    test(`empty reconfiguration clears the previous workflow state (${JSON.stringify(value)})`, () => {
        const { widget, serialized } = setup({ active_kind: "Klein9b", model_params: { steps: 41 },
            loras: [{ name: "Old style", auto_apply: true }] });
        widget.restoreState();
        serialized.value = value;
        widget.restoreState();
        assert.equal(serialized.value, value, "Restoration must preserve the incoming workflow bytes");
        assert.equal(widget._activeKind(), "QI2");
        assert.equal(widget._currentModelParams().steps, 25);
        widget._modelParamsSave({ cfg: 2 });
        const saved = JSON.parse(serialized.value);
        assert.equal(saved.active_kind, "QI2");
        assert.equal(saved.model_params.steps, 25);
        assert.equal(saved.loras, undefined);
    });
}

test("constructor preserves restored bytes until configuration arrives", () => {
    const { context, node, serialized, timers } = setup();
    Object.assign(context.Widget.prototype, {
        _buildUI() { this.container = { isConnected: true }; },
        _startPolling() {}, _fetchSamplerLists() {}, _fetchModuleStatus() {},
        _syncOutputSlots() {}, _syncCustomModelInput() {}, _getRepoId: () => "",
    });
    const widget = new context.Widget(node);
    assert.equal(serialized.value, "{}");
    const restored = JSON.stringify({ active_kind: "Klein9b", selected_model: "Klein B", selected_type: "unet" });
    serialized.value = restored;
    timers[0]();
    assert.equal(widget.state.selected_model, "Klein B");
    assert.equal(serialized.value, restored);
});

function changeCustomLoraContext(fixture, change) {
    const { widget, serialized } = fixture;
    if (change === "reconfiguration") {
        serialized.value = JSON.stringify({ active_kind: "Klein9b", selected_type: "unet",
            selected_model: "Klein B", loras: [{ name: "Style", auto_apply: true, strength: 0.4 }] });
        widget.restoreState();
    } else if (change === "repository") {
        widget._getRepoId = () => "owner/current";
    } else if (change === "family") {
        widget._scheduleRender = () => {};
        widget._setActiveKind("Klein9b");
    } else if (change === "invalid reconfiguration") {
        serialized.value = '{"loras":';
        widget.restoreState();
    } else {
        widget._removed = true;
    }
}

for (const change of ["reconfiguration", "repository", "family", "removal", "invalid reconfiguration"]) {
    test(`late custom LoRA deletion preserves the newer context (${change})`, async () => {
        const fixture = setup({ active_kind: "QI2", loras: [{ name: "Style", auto_apply: true }] });
        const { widget, serialized, pending, context } = fixture;
        const refreshes = [];
        let confirm;
        widget._getRepoId = () => "owner/original";
        widget.showConfirm = (_, callback) => { confirm = callback; };
        widget.fetchConfig = async repo => { refreshes.push(repo); };
        widget._refreshLoraBlock = () => assert.fail("A stale deletion must not refresh the current UI");
        context.serverRegistry("VNCCS_CC_REGISTRY")["owner/original"] = catalog;
        await widget._removeCustomLora({ name: "Style", local_path: "models/loras/style.safetensors" });
        const deletion = confirm();
        assert.equal(pending.length, 1);
        changeCustomLoraContext(fixture, change);
        const saved = serialized.value;
        const alreadyRefreshed = [...refreshes];
        respond(pending[0], { status: "ok" });
        await deletion;
        assert.equal(serialized.value, saved);
        assert.equal(widget.state.loras.length, 1);
        assert.deepEqual(refreshes, alreadyRefreshed);
        assert.equal(context.serverRegistry("VNCCS_CC_REGISTRY")["owner/original"], undefined);
    });

    test(`late custom LoRA file list does not open a dialog in a newer context (${change})`, async () => {
        const fixture = setup({ active_kind: "QI2" });
        const { widget, pending } = fixture;
        widget._getRepoId = () => "owner/original";
        widget.fetchConfig = async () => {};
        widget.container = { appendChild() { assert.fail("A stale file list must not open a dialog"); } };
        const dialog = widget._showCustomLoraDialog();
        changeCustomLoraContext(fixture, change);
        respond(pending[0], { items: [{ path: "style.safetensors", already_added: false }] });
        await dialog;
        if (change !== "invalid reconfiguration") assert.equal(widget.message, undefined);
    });
}

test("custom LoRA deletion in the current context still persists and refreshes", async () => {
    const { widget, serialized, pending } = setup({ loras: [{ name: "Style" }, { name: "Other" }] });
    let confirm, refreshed = 0;
    widget._getRepoId = () => "owner/models";
    widget.showConfirm = (_, callback) => { confirm = callback; };
    widget.fetchConfig = async repo => { assert.equal(repo, "owner/models"); };
    widget._refreshLoraBlock = () => { refreshed++; };
    await widget._removeCustomLora({ name: "Style", local_path: "models/loras/style.safetensors" });
    const deletion = confirm();
    respond(pending[0], { status: "ok" });
    await deletion;
    assert.deepEqual(JSON.parse(serialized.value).loras, [{ name: "Other" }]);
    assert.equal(refreshed, 1);
});

for (const change of [null, "reconfiguration"]) {
    test(`custom LoRA addition respects the current workflow (${change || "unchanged"})`, async () => {
        const fixture = setup({ active_kind: "QI2" });
        const { widget, context, pending, serialized } = fixture;
        const elements = [], refreshes = [];
        context.document.createElement = tag => {
            const element = { children: [], style: {}, classList: { add() {} },
                appendChild(child) { this.children.push(child); if (tag === "select") this.value ||= child.value; },
                append(...children) { children.forEach(child => this.appendChild(child)); },
                remove() { this.removed = true; },
            };
            elements.push(element);
            return element;
        };
        widget.container = { appendChild() {} };
        widget._getRepoId = () => "owner/models";
        widget.fetchConfig = async repo => { refreshes.push(repo); };
        let refreshedBlocks = 0;
        widget._refreshLoraBlock = () => { refreshedBlocks++; };
        const dialog = widget._showCustomLoraDialog();
        respond(pending[0], { items: [{ path: "style.safetensors", already_added: false }] });
        await dialog;
        const add = elements.find(element => element.textContent === "Add");
        assert.ok(add, "A current file list must open the add dialog");
        const adding = add.onclick();
        assert.equal(pending[1].route, "/vnccs/control_center/custom_lora");
        assert.deepEqual(JSON.parse(pending[1].options.body), {
            repo_id: "owner/models", path: "style.safetensors", kind: "QI2",
        });
        if (change) changeCustomLoraContext(fixture, change);
        const saved = serialized.value;
        const previousRefreshes = refreshes.length;
        respond(pending[1], { status: "ok" });
        await adding;
        assert.equal(serialized.value, saved);
        assert.equal(refreshes.length, previousRefreshes + (change ? 0 : 1));
        assert.equal(refreshedBlocks, change ? 0 : 1);
    });
}

for (const change of [null, "reconfiguration", "repository", "family", "removal", "invalid reconfiguration"]) {
    test(`settings dialog preserves the current workflow (${change || "unchanged"})`, async () => {
        const fixture = setup({ active_kind: "QI2", type_settings: { unet: { weight_dtype: "default" } } });
        const { widget, context, serialized } = fixture;
        const elements = [];
        context.setHelpText = () => {};
        context.document.createElement = () => {
            const element = { children: [], classList: { add() {} },
                appendChild(child) { this.children.push(child); },
                append(...children) { children.forEach(child => this.appendChild(child)); },
                querySelector(selector) { return elements.find(item => `#${item.id}` === selector); },
                remove() { this.removed = true; },
            };
            elements.push(element);
            return element;
        };
        widget.container = { appendChild() {} };
        widget._getRepoId = () => "owner/original";
        widget.fetchConfig = async () => {};
        widget._showSettings();
        elements.find(element => element.id === "vnccs-cc-unet-dtype").value = "fp8_e4m3fn";
        if (change) changeCustomLoraContext(fixture, change);
        const saved = serialized.value;
        const currentState = JSON.stringify(widget.state);
        await elements.find(element => element.textContent === "Save").onclick();
        if (change) {
            assert.equal(serialized.value, saved);
            assert.equal(JSON.stringify(widget.state), currentState);
        }
        else assert.equal(JSON.parse(serialized.value).type_settings.unet.weight_dtype, "fp8_e4m3fn");
    });
}

function setupDownloads() {
    const fixture = setup({ active_kind: "QI2", selected_type: "unet", selected_model: "QI2" });
    Object.assign(fixture.widget, {
        config: { ...catalog, models: catalog.models.map(entry => ({ ...entry, status: "missing" })),
            clip: [{ name: "QI2 encoder", kind: "QI2", status: "missing" }] },
        dlStatus: {}, _getRepoId: () => "owner/original", fetchConfig: async () => {},
        _renderAll() { this.renders = (this.renders || 0) + 1; },
    });
    return fixture;
}

function respondDownload(request, ok = true) {
    request.resolve({ ok, text: async () => JSON.stringify(ok ? { status: "queued" } : { error: "Download rejected" }) });
}

for (const change of ["reconfiguration", "repository", "family", "removal", "invalid reconfiguration"]) {
    test(`stale bulk download confirmation cannot enqueue models (${change})`, async () => {
        const fixture = setupDownloads();
        const { widget, pending } = fixture;
        let confirm;
        widget.showConfirm = (_, callback) => { confirm = callback; };
        await widget._downloadAllMissing();
        changeCustomLoraContext(fixture, change);
        const queueing = confirm();
        assert.equal(pending.length, 0, "An old confirmation must not enqueue downloads for the current workflow");
        await queueing;
    });

    test(`bulk downloads stop queuing after the context changes (${change})`, async () => {
        const fixture = setupDownloads();
        const { widget, pending, timers, serialized } = fixture;
        let confirm;
        widget.showConfirm = (_, callback) => { confirm = callback; };
        await widget._downloadAllMissing();
        const queueing = confirm();
        assert.equal(pending.length, 1);
        assert.equal(JSON.parse(pending[0].options.body).repo_id, "owner/original");
        changeCustomLoraContext(fixture, change);
        const saved = serialized.value;
        const renders = widget.renders;
        if (change === "removal") pending[0].reject(new Error("Connection closed"));
        else respondDownload(pending[0], false);
        await new Promise(resolve => setImmediate(resolve));
        if (timers.length) timers.at(-1)();
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(pending.length, 1, "Only the already-submitted request may finish");
        assert.equal(widget.renders, renders, "An old response must not redraw the current workflow");
        assert.equal(serialized.value, saved);
        assert.notEqual(widget.dlStatus.cc_models_QI2.status, "error");
        await queueing;
    });
}

test("bulk downloads in the unchanged context enqueue every selected asset", async () => {
    const { widget, pending, timers } = setupDownloads();
    let confirm;
    widget.showConfirm = (_, callback) => { confirm = callback; };
    await widget._downloadAllMissing();
    const queueing = confirm();
    for (let index = 0; index < 2; index++) {
        assert.equal(pending.length, index + 1);
        assert.equal(JSON.parse(pending[index].options.body).repo_id, "owner/original");
        respondDownload(pending[index]);
        await new Promise(resolve => setImmediate(resolve));
        timers.at(-1)();
        await new Promise(resolve => setImmediate(resolve));
    }
    await queueing;
    assert.deepEqual(pending.map(request => JSON.parse(request.options.body).name), ["QI2", "QI2 encoder"]);
});

test("missing custom LoRA files are not queued as remote downloads", async () => {
    const { widget, context, pending } = setupDownloads();
    widget.config.models.forEach(entry => { entry.status = "installed"; });
    widget.config.clip.forEach(entry => { entry.status = "installed"; });
    const entry = { name: "Local style", custom: true, kind: "QI2", status: "missing",
        local_path: "models/loras/local-style.safetensors" };
    widget.config.lora = [entry];
    widget.showConfirm = () => assert.fail("Local files without a remote source must not be offered for download");
    await widget._downloadAllMissing();
    assert.equal(pending.length, 0);
    const elements = [];
    context.document.createElement = () => {
        const element = { children: [], dataset: {}, classList: { add() {} },
            appendChild(child) { this.children.push(child); },
            append(...children) { this.children.push(...children); },
            querySelector: () => null,
        };
        elements.push(element);
        return element;
    };
    for (const compact of [false, true]) widget._renderLoraCard(entry, { compact });
    assert.equal(elements.some(element => /Download|Update/.test(element.textContent || "")), false,
        "Custom cards must keep removal available without offering an impossible remote download");
    assert.equal(elements.filter(element => element.textContent === "×").length, 2);
});

for (const failure of ["HTTP", "network"]) {
    test(`download errors in the unchanged context remain visible (${failure})`, async () => {
        const { widget, pending } = setupDownloads();
        const downloading = widget._downloadEntry("models", widget.config.models[0]);
        if (failure === "network") pending[0].reject(new Error("Connection closed"));
        else respondDownload(pending[0], false);
        await downloading;
        assert.equal(widget.dlStatus.cc_models_QI2.status, "error");
        assert.equal(widget.message, failure === "network" ? "Connection closed" : "Download rejected");
        assert.equal(widget.renders, 2);
    });
}
