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
    const node = { widgets: [dataWidget], _randomizeSeedIfNeeded() {},
        onSerialize(result) { result.originalCallbackCalled = true; } };
    const context = createWidgetContext({
        state, dataWidget, node,
        beginPreviewRequest: () => () => true,
        spritePreviewNavigator: null,
        localStorage: { setItem() {} },
        fileInp: {}, btn: {}, btnGen: { disabled: false },
        els: { previewImg: { style: {} }, placeholder: { style: {} } },
        container: { appendChild() {} },
        document: { createElement: () => ({ remove() {} }) },
        FormData: class { append() {} },
        normalizeUploadFile: file => file,
        renderClonePanel() {},
        showInfo: (title, message) => messages.push({ title, message }),
        hasSelectedEditableCostume: () => true,
        showCreateCostumeRequired() { throw new Error("Unexpected costume validation"); },
        setClothesCoreLora() {},
        saveCostumeToBackend: async () => {},
        getConnectedControlCenterState: () => ({ repo_id: "VNCCS", node_state: "{}", selected_type: selectedType }),
        queueConnectedPreview: async () => { queued.push(JSON.parse(dataWidget.value)); return { cached: false }; },
        api: { fetchApi: async (url, options) => {
            if (url === "/upload/image") {
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
    return { context, state, dataWidget, messages, apiPreviews, queued,
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
