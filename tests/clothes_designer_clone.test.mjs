import { createWidgetContext } from './widget_context.mjs';
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_clothes_designer.js", import.meta.url), "utf8");
function block(start, end) {
    const offset = source.indexOf(start);
    const endOffset = source.indexOf(end, offset);
    assert.ok(offset >= 0 && endOffset > offset);
    return source.slice(offset, endOffset);
}

function setup(selectedType = "unet") {
    const state = { character: "Alice", costume: "Dress", activeTab: "clone", clone_image: null,
        gen_settings: { seed_mode: "fixed" } };
    const dataWidget = { name: "widget_data", value: "{}" };
    const messages = [];
    const apiPreviews = [];
    const queued = [];
    const uploads = [];
    const node = { widgets: [dataWidget], _randomizeSeedIfNeeded() {},
        onSerialize(result) { result.originalCallbackCalled = true; } };
    const context = createWidgetContext({
        state, dataWidget, node,
        beginPreviewRequest: () => () => true, beginClothesUploadRequest: () => () => true,
        beginClothesWizardRequest: () => () => true, beginDeleteRequest: () => () => true,
        spritePreviewNavigator: null,
        localStorage: { setItem() {} },
        fileInp: {}, btn: {}, btnGen: { disabled: false },
        els: { previewImg: { style: {} }, placeholder: { style: {} } },
        container: { appendChild() {} },
        document: { createElement: () => ({ dataset: {}, remove() {} }) },
        FormData: class { values = []; append(...args) { this.values.push(args); } },
        normalizeUploadFile: file => file,
        renderClonePanel() {},
        showInfo: (title, message) => messages.push({ title, message }),
        hasSelectedEditableCostume: () => true,
        showCreateCostumeRequired() { throw new Error("Unexpected costume validation"); },
        setClothesCoreLora() {},
        startClothesPreviewGeneration: () => 1,
        revealClothesPreview: async () => true,
        stopClothesPreviewGeneration() {},
        saveCostumeToBackend: async () => {},
        getConnectedControlCenterState: () => ({ repo_id: "VNCCS", node_state: "{}", selected_type: selectedType }),
        queueConnectedPreview: async () => { queued.push(JSON.parse(dataWidget.value)); return { cached: false }; },
        api: { fetchApi: async (url, options) => {
            if (url === "/upload/image") {
                uploads.push(options);
                return { ok: true, json: async () => ({ name: "donor.png", subfolder: "clothes" }) };
            }
            assert.equal(url, "/vnccs/control_center/clothes_preview");
            apiPreviews.push(JSON.parse(options.body));
            return { ok: true, json: async () => ({}) };
        } },
    });
    vm.runInContext(
        block("const saveState = () =>", "node._randomizeSeedIfNeeded =") +
        block("const onSerialize = node.onSerialize;", "const saveCostumeToBackend =") +
        block("fileInp.onchange = async", "overlay.onclick =") +
        block("btnGen.onclick = async", "els.btnGen = btnGen;"), context,
    );
    return { context, state, dataWidget, messages, apiPreviews, queued, uploads,
        upload: () => context.fileInp.onchange({ target: { files: [{ name: "donor.png" }] } }),
    };
}

test("clone preview without a donor stops before API or queue execution", async () => {
    const harness = setup();
    await harness.context.btnGen.onclick();
    assert.equal(harness.apiPreviews.length, 0);
    assert.equal(harness.queued.length, 0);
    assert.equal(harness.messages[0].title, "Reference Required");
});

for (const selectedType of ["unet", "custom"]) {
    test(`uploaded donor metadata reaches the ${selectedType} preview path and workflow`, async () => {
        const harness = setup(selectedType);
        await harness.upload();
        assert.ok(!harness.uploads[0].body.values.some(([name]) => name === 'overwrite'), 'References used by older workflows must not be overwritten');
        const expected = { name: "donor.png", type: "input", subfolder: "clothes" };
        assert.deepEqual(JSON.parse(harness.dataWidget.value).clone_image, expected);
        await harness.context.btnGen.onclick();
        assert.equal(harness.queued.length, 0, "Preview must never submit the workflow");
        const sent = harness.apiPreviews[0].clothes_state;
        assert.deepEqual(sent.clone_image, expected);
        assert.equal(sent.activeTab, "clone");
        assert.equal(harness.messages.length, 0);

        const serialized = { widgets_values: ["{}"] };
        harness.context.node.onSerialize(serialized);
        assert.equal(serialized.originalCallbackCalled, true);
        assert.deepEqual(JSON.parse(serialized.widgets_values[0]).clone_image, expected);
        assert.equal(JSON.parse(serialized.widgets_values[0]).activeTab, "clone");
    });
}

test("background buttons update restored QI2 state, preview payload and workflow", async () => {
    const harness = setup();
    await harness.upload();
    Object.assign(harness.context, {
        defaultState: { costume_info: {}, character_info: {}, gen_settings: { seed_mode: "fixed" } },
        getConnectedModelKind: () => "qi2",
        syncResolutionControl() {}, renderClothesCoreLoraCard() {},
        refreshTabs() {}, syncCostumeEditControls() {},
        beginSelectionRequest() {}, beginCharacterInfoRequest() {}, beginCostumeInfoRequest() {}, loadCharList() {},
        setHelpText() {}, helpFor() {},
        document: { createElement: () => ({
            children: [], attrs: {}, classList: { add() {}, toggle() {} },
            appendChild(child) { this.children.push(child); },
            append(...children) { this.children.push(...children); },
            setAttribute(name, value) { this.attrs[name] = value; },
            remove() {},
        }) },
    });
    vm.runInContext(
        block("const createSegmentedField =", "const syncResolutionControl =") +
        block("const syncBackgroundForModel =", "const renderClothesCoreLoraCard =") +
        block("node._vnccsRestoreClothesState =", "registerCleanup(node, () => { delete node._vnccsRestoreClothesState;") +
        `this.field = createSegmentedField("Background", "background_color", [
            { label: "Green", value: "Green" }, { label: "Blue", value: "Blue" },
            { label: "Alpha", value: "Transparent" }
        ]); this.sync = syncGenerationControls;`, harness.context,
    );

    for (const [index, background] of [[1, "Blue"], [0, "Green"], [2, "Transparent"]]) {
        harness.dataWidget.value = JSON.stringify({ ...harness.state, gen_settings: {
            seed_mode: "fixed", background_color: "Transparent", background_model_kind: "qi2",
        } });
        harness.context.node._vnccsRestoreClothesState();
        const buttons = harness.context.field.children[1].children;
        buttons[index].onclick();
        assert.equal(buttons[index].attrs["aria-pressed"], "true");
        assert.equal(harness.state.gen_settings.background_color, background);
        assert.equal(JSON.parse(harness.dataWidget.value).gen_settings.background_color, background);
        harness.context.sync();
        assert.equal(harness.state.gen_settings.background_color, background);

        await harness.context.btnGen.onclick();
        assert.equal(harness.apiPreviews.at(-1).clothes_state.gen_settings.background_color, background);
        const serialized = { widgets_values: ["{}"] };
        harness.context.node.onSerialize(serialized);
        assert.equal(JSON.parse(serialized.widgets_values[0]).gen_settings.background_color, background);
    }
});

test("QI2 Prompt Enhancement toggle persists and is hidden outside Generate Clothes", () => {
    const harness = setup();
    harness.state.activeTab = "generate";
    let connectedKind = "qi2";
    const makeElement = tagName => {
        const classes = new Set();
        return {
            tagName, children: [], attrs: {}, hidden: false, disabled: false, checked: false,
            dataset: {}, style: {},
            classList: {
                toggle(name, force) {
                    if (force === undefined ? !classes.has(name) : force) classes.add(name);
                    else classes.delete(name);
                },
                contains: name => classes.has(name),
            },
            append(...children) { this.children.push(...children); },
            appendChild(child) { this.children.push(child); },
            setAttribute(name, value) { this.attrs[name] = value; },
            dispatchEvent(event) { this[`on${event.type}`]?.({ target: this }); },
        };
    };
    Object.assign(harness.context, {
        Event: class { constructor(type) { this.type = type; } },
        getConnectedModelKind: () => connectedKind,
        setHelpText() {}, helpFor() {},
        document: { createElement: makeElement },
    });
    vm.runInContext(
        block("const syncPromptEnhancementControl =", "const syncGenerationControls =")
        + block("const createPromptEnhancementControl =", "const createGenerationControls =")
        + "this.promptEnhancementControl = createPromptEnhancementControl(); this.syncPromptEnhancement = syncPromptEnhancementControl;",
        harness.context,
    );

    const row = harness.context.promptEnhancementControl;
    const [badge, label, status, toggle] = row.children;
    const [input, track, thumb] = toggle.children;
    assert.equal(row.hidden, false);
    assert.equal(badge.className, "vnccs-clothes-prompt-badge");
    assert.equal(label.textContent, "Prompt Enhancement");
    assert.equal(input.attrs["aria-label"], "Prompt Enhancement");
    assert.equal(track.className, "vnccs-clothes-prompt-toggle-track");
    assert.equal(thumb.className, "vnccs-clothes-prompt-toggle-thumb");
    assert.equal(input.checked, true);
    assert.equal(status.textContent, "Enabled");

    row.onclick();
    assert.equal(harness.state.gen_settings.qi2_prompt_enhancement, false);
    assert.equal(JSON.parse(harness.dataWidget.value).gen_settings.qi2_prompt_enhancement, false);
    assert.equal(status.textContent, "Disabled");

    input.setDisabled(true);
    row.onclick();
    assert.equal(harness.state.gen_settings.qi2_prompt_enhancement, false);
    input.setDisabled(false);
    harness.state.activeTab = "clone";
    harness.context.syncPromptEnhancement();
    assert.equal(row.hidden, true);
    harness.state.activeTab = "generate";
    connectedKind = "minimaxh3";
    harness.context.syncPromptEnhancement();
    assert.equal(row.hidden, true);
});

test("Prompt Enhancement is rendered directly below Clothes Wizard only in the Generate panel", () => {
    const attributes = block("const renderAttributes =", "// Helper to render Clone Panel");
    const layout = block("const controlsWrap =", "// Tab Header");
    assert.match(attributes, /container\.appendChild\(wizardBtn\);\s*container\.appendChild\(promptEnhancementControl\);/);
    assert.match(layout, /const promptEnhancementControl = createPromptEnhancementControl\(\);/);
    assert.doesNotMatch(layout, /designContent\.appendChild\(createPromptEnhancementControl\(\)\)/);
});

for (const stale of ['character', 'costume', 'tab', 'reference', 'removed', 'superseded', 'corrupt', 'current']) {
    test(`Clothes donor upload respects ${stale} ownership`, async () => {
        const h = setup();
        let finish;
        h.context.api.fetchApi = () => new Promise(resolve => { finish = resolve; });
        const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
        const start = common.indexOf('export function registerCleanup');
        const end = common.indexOf('// ── DOM Widget Width Sync', start);
        vm.runInContext(`${common.slice(start, end).replaceAll('export ', '')}
            beginClothesUploadRequest = createRequestGuard(node);`, h.context);
        const pending = h.upload();
        if (stale === 'character') h.state.character = 'Bob';
        if (stale === 'costume') h.state.costume = 'Coat';
        if (stale === 'tab') h.state.activeTab = 'generate';
        if (stale === 'reference') h.state.clone_image = { name: 'newer-donor.png' };
        if (stale === 'removed') h.context.node.onRemoved();
        if (stale === 'superseded') h.context.beginClothesUploadRequest();
        if (stale === 'corrupt') h.context.node._vnccsStateRestoreError = 'Cannot restore saved clothes state.';
        finish({ ok: true, json: async () => ({ name: 'previous-donor.png' }) });
        await pending;
        assert.equal(h.state.clone_image?.name, stale === 'current' ? 'previous-donor.png' : stale === 'reference' ? 'newer-donor.png' : undefined);
        assert.equal(h.dataWidget.value, stale === 'current' ? JSON.stringify(h.state) : '{}');
        assert.deepEqual(h.messages, []);
    });
}

test('corrupt Clothes state blocks direct upload and preview without modifying the workflow', async () => {
    const h = setup();
    h.context.node._vnccsStateRestoreError = 'Cannot restore saved clothes state.';
    h.dataWidget.value = '{broken workflow state';
    await h.upload();
    await h.context.btnGen.onclick();
    assert.equal(h.uploads.length, 0);
    assert.equal(h.apiPreviews.length, 0);
    assert.equal(h.dataWidget.value, '{broken workflow state');
});

test('an earlier clothes preview cannot enable generation while selection metadata is loading', async () => {
    const h = setup();
    await h.upload();
    let finish;
    h.context.api.fetchApi = () => new Promise(resolve => { finish = resolve; });
    const generation = h.context.btnGen.onclick();
    await new Promise(resolve => setImmediate(resolve));
    h.context.node._vnccsClothesSelectionPending = true;
    finish({ ok: true, json: async () => ({}) });
    await generation;
    assert.equal(h.context.btnGen.disabled, true);
});

for (const selection of ['character', 'costume', 'tab']) {
    test(`Clothes upload stays obsolete after ${selection} changes away and back`, async () => {
        const h = setup();
        let finish;
        h.context.api.fetchApi = () => new Promise(resolve => { finish = resolve; });
        Object.assign(h.context, { charSel: {}, costSel: {},
            beginClothesWizardRequest() {}, beginSelectionRequest: () => () => true,
            loadCharacterInfo: async character => { h.state.character = character; return true; },
            loadCostumes: async () => true,
            loadCostumeInfo: async (preserveDraft, costume) => { h.state.costume = costume; return true; },
            syncCostumeEditControls() {}, updatePreviewImage() {}, refreshTabs() {},
            document: { createElement: () => ({ style: {}, dataset: {} }) },
        });
        h.context.els.charSelect = {};
        h.context.els.costSel = {};
        const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
        const start = common.indexOf('export function registerCleanup');
        const end = common.indexOf('// ── DOM Widget Width Sync', start);
        vm.runInContext(`${common.slice(start, end).replaceAll('export ', '')}
            beginClothesUploadRequest = createRequestGuard(node);
            ${block('charSel.onchange =', 'charRow.appendChild(charSel)')}
            ${block('costSel.onchange =', 'els.costSel = costSel;')}
            ${block('const createTab =', 'const contentArea =')}
            globalThis.tab = createTab;`, h.context);
        const pending = h.upload();
        if (selection === 'tab') {
            h.context.tab('generate', 'Generate').onclick();
            h.context.tab('clone', 'Clone').onclick();
        } else {
            const control = h.context[selection === 'character' ? 'charSel' : 'costSel'];
            await control.onchange({ target: { value: selection === 'character' ? 'Bob' : 'Coat' } });
            await control.onchange({ target: { value: selection === 'character' ? 'Alice' : 'Dress' } });
        }
        finish({ ok: true, json: async () => ({ name: 'obsolete-donor.png' }) });
        await pending;
        assert.equal(h.state.clone_image, null);
        assert.equal(JSON.parse(h.dataWidget.value).clone_image, null);
        assert.deepEqual(h.messages, []);
    });
}
