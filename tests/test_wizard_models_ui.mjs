import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

for (const name of ['vnccs_clothes_designer', 'vnccs_character_creator_v2', 'vnccs_character_cloner']) {
    const source = readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8');
    const start = source.indexOf('const ensureQwenVLReady =');
    const end = source.indexOf('\n                };', start) + '\n                };'.length;
    const helper = source.slice(start, end);
    for (const choice of ['local', 'cancel', 'escape', 'download']) {
        test(`${name}: ${choice} model preparation`, async () => {
            const calls = [];
            const context = vm.createContext({
                document: { createElement: () => ({}) },
                setTimeout: callback => callback(),
                showModal: (title, build, buttons) => {
                    const modal = { querySelector: () => ({ style: {} }), addEventListener: (_name, callback) => {
                        if (choice === 'escape') queueMicrotask(() => callback({ key: 'Escape' }));
                    } };
                    build(modal);
                    if (title === 'Qwen3.5 Model Required' && choice !== 'escape') {
                        queueMicrotask(() => buttons[choice === 'download' ? 1 : 0].action());
                    }
                    return { modal, overlay: { remove() {} } };
                },
                api: { fetchApi: async (url, options = {}) => {
                    calls.push([url, options.method || 'GET']);
                    return { ok: true, json: async () => url.includes('model_status')
                        ? { ready: choice === 'local', model_name: 'Qwen3.5-4B-Q8_0.gguf' }
                        : { status: 'completed', progress: 100 } };
                } },
            });
            const run = new vm.Script(`${helper}\nensureQwenVLReady`).runInContext(context);
            assert.equal(await run(), choice === 'local' || choice === 'download');
            assert.equal(calls.filter(([, method]) => method === 'POST').length, choice === 'download' ? 1 : 0);
            assert.ok(calls[0][0].includes('qwen_vl_model_status'));
            assert.equal(calls[0][0].includes('vision=false'), !name.endsWith('cloner'));
        });
    }
}

const designer = readFileSync(new URL('../web/vnccs_clothes_designer.js', import.meta.url), 'utf8');
const syncStart = designer.indexOf('const syncResolutionControl =');
const syncEnd = designer.indexOf('const syncGenerationControls =', syncStart);

test('resolution follows the model in Auto and preserves restored manual sizes', () => {
    const select = { options: [{ value: '', textContent: '' }], value: '', add(option) { this.options.push(option); } };
    const state = { gen_settings: { target_size: null } };
    let kind = 'QIE2511';
    const context = vm.createContext({
        state, els: { target_size: select }, getConnectedModelKind: () => kind,
        Option: function (text, value) { this.textContent = text; this.value = value; },
    });
    const sync = new vm.Script(`${designer.slice(syncStart, syncEnd)}\nsyncResolutionControl`).runInContext(context);
    sync();
    assert.equal(select.options[0].textContent, 'Auto (1024)');
    kind = 'MiniMaxH3';
    sync();
    assert.equal(select.options[0].textContent, 'Auto (1536)');
    assert.equal(state.gen_settings.target_size, null);
    state.gen_settings.target_size = 1408;
    sync();
    assert.equal(select.value, '1408');
    kind = 'Klein9b';
    sync();
    assert.equal(select.value, '1408');
    assert.equal(select.options[0].textContent, 'Auto (1024)');
});

test('workflow restoration merges generation defaults and preserves manual resolution', () => {
    const state = { gen_settings: { target_size: null, seed: 0 } };
    const dataWidget = { value: JSON.stringify({ gen_settings: { target_size: 2048 } }) };
    const node = {};
    let synchronized = 0;
    const context = vm.createContext({
        node, state, dataWidget,
        defaultState: { gen_settings: { target_size: null, seed: 0 }, character_info: {}, costume_info: {} },
        syncGenerationControls: () => synchronized++,
    });
    const start = designer.indexOf('node._vnccsRestoreClothesState =');
    const end = designer.indexOf('registerCleanup(node, () => { delete node._vnccsRestoreClothesState;', start);
    new vm.Script(designer.slice(start, end)).runInContext(context);
    node._vnccsRestoreClothesState();
    assert.equal(state.gen_settings.target_size, 2048);
    assert.equal(state.gen_settings.seed, 0);
    dataWidget.value = JSON.stringify({ gen_settings: { seed: 99 } });
    node._vnccsRestoreClothesState();
    assert.equal(state.gen_settings.target_size, null);
    assert.equal(state.gen_settings.seed, 99);
    assert.equal(synchronized, 2);
});

test('workflow serialization saves resolution and preserves the original callback', () => {
    let called = 0;
    const node = { onSerialize() { assert.equal(this, node); called++; } };
    const dataWidget = {};
    const state = { gen_settings: { target_size: 1536 } };
    const context = vm.createContext({ node, dataWidget, state });
    const start = designer.indexOf('const onSerialize = node.onSerialize;');
    const end = designer.indexOf('const saveCostumeToBackend =', start);
    new vm.Script(designer.slice(start, end)).runInContext(context);
    node.onSerialize({});
    assert.equal(JSON.parse(dataWidget.value).gen_settings.target_size, 1536);
    assert.equal(called, 1);
});
