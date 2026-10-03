import { createWidgetContext } from './widget_context.mjs';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../web/vnccs_character_generator.js', import.meta.url), 'utf8');
const models = ['seedvr2_3b_fp8_e4m3fn.safetensors', 'seedvr2_7b_fp16.safetensors'].map(name => ({
    name, local_path: `models/diffusion_models/${name}`, status: 'installed',
}));

async function setup(type) {
    const events = [];
    let persisted;
    let changeCount = 0;
    const app = {
        graph: { setDirtyCanvas() {} },
        registerExtension(extension) { this.extension = extension; },
        canvas: {
            emitEvent({ subType }) {
                events.push(subType);
                if (subType === 'before-change') changeCount++;
                if (subType === 'after-change' && --changeCount === 0) capture();
            },
        },
    };
    const context = createWidgetContext({
        app,
        document: { createElement: () => ({ children: [], appendChild(child) { this.children.push(child); } }) },
        syncDOMWidgetWidthSoon() {},
    });
    vm.runInContext(source.replace(/^import .*;\n/gm, '')
        + '\nthis.Widget = CharacterGeneratorWidget; this.readData = readData; this.writeData = writeData;', context);
    class Node {}
    await app.extension.beforeRegisterNodeDef(Node, { name: type });
    const state = { name: 'widget_data', value: '{}' };
    const node = Object.assign(new Node(), { widgets: [state], type, id: 17 });
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        node, data: context.readData(node), stages: [], seedvrDownloads: {},
        seedvrAssets: { models }, seedvrModelPickerOpen: true,
        syncCharacterSourceData() {}, syncModelResolution() {}, rememberModelResolution() {},
        syncStagesFromData() {}, renderSettings() {}, renderPreview() {}, renderChain() {},
        saveBrowserState() {},
    });
    node._vnccsCharacterGeneratorWidget = widget;
    function capture() {
        persisted = { widgets_values: [state.value] };
        node.onSerialize(persisted);
    }
    capture();
    return { widget, node, state, events, context, capture,
        savedModel: () => JSON.parse(persisted.widgets_values[0]).upscaler.model,
        reload() {
            state.value = persisted.widgets_values[0];
            node.onConfigure();
        },
    };
}

for (const type of ['VNCCS_CharacterGenerator', 'VNCCS_CharacterCloneGenerator',
    'VNCCS_ClothesGenerator', 'VNCCS_EmotionsGenerator']) {
    test(`${type}: picking another SeedVR model persists before any subsequent mouseup`, async () => {
        const { widget, state, events, capture, reload, savedModel } = await setup(type);
        // ComfyUI captures on mouseup, which precedes the model card's click handler.
        capture();
        widget.buildSeedvrCard(models[1]).onclick();
        assert.equal(JSON.parse(state.value).upscaler.model, models[1].name);
        assert.equal(savedModel(), models[1].name);
        assert.equal(widget.seedvrModelPickerOpen, false);
        assert.deepEqual(events, ['before-change', 'after-change']);
        reload();
        assert.equal(widget.data.upscaler.model, models[1].name);
        const head = widget.seedvrModelCards().children[0].children[0];
        assert.match(head.innerHTML, new RegExp(models[1].name));
        assert.match(head.className, /is-selected/);
        assert.deepEqual(events, ['before-change', 'after-change']);
    });
}

test('settings writes notify persistence without recording serialization as a new change', async () => {
    const { widget, node, state, context, events, reload, savedModel } = await setup('VNCCS_CharacterGenerator');
    const callbacks = [];
    state.callback = value => callbacks.push(value);
    widget.data.upscaler.model = models[1].name;
    context.writeData(node, widget.data, { trackChange: true });
    assert.equal(savedModel(), models[1].name);
    assert.equal(callbacks.length, 1);
    assert.deepEqual(events, ['before-change', 'after-change']);
    reload();
    assert.equal(widget.data.upscaler.model, models[1].name);
    assert.deepEqual(events, ['before-change', 'after-change']);
});

test('a throwing widget callback still closes the change transaction', async () => {
    const { widget, node, state, context, events } = await setup('VNCCS_CharacterGenerator');
    state.callback = () => { throw new Error('Callback failed'); };
    assert.throws(() => context.writeData(node, widget.data, { trackChange: true }), /Callback failed/);
    assert.deepEqual(events, ['before-change', 'after-change']);
});
