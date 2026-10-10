import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = readFileSync(new URL('../web/vnccs_clothes_designer.js', import.meta.url), 'utf8');
const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
const between = (text, start, end) => {
    const first = text.indexOf(start), last = text.indexOf(end, first);
    assert.ok(first >= 0 && last > first);
    return text.slice(first, last);
};
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};
const response = data => ({ ok: true, json: async () => data });
const guards = between(common, 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '');

function setup({ contexts, metadata, save, disk = { top: 'disk cotton', negative_prompt: 'blur', extension: { id: 7 } } } = {}) {
    const dataWidget = { name: 'widget_data', value: '{}' };
    const node = { widgets: [dataWidget] }, requests = [], messages = [];
    const selection = () => ({ options: [], add(option) { this.options.push(option); },
        set innerHTML(value) { this.options = []; } });
    const controls = { charSelect: selection(), costSel: selection(),
        ...Object.fromEntries(['btnGen', 'btnDel', 'wizardBtn', 'top', 'bottom', 'head', 'face', 'shoes', 'placeholder', 'previewImg']
            .map(key => [key, { value: '', style: {} }])) };
    const nodeType = { prototype: { onConfigure(info) { this.widgets[0].value = info.widgets_values[0]; } } };
    const context = createWidgetContext({
        node, nodeType, dataWidget, controls,
        setTimeout() {},
        storage: { getItem() { return null; }, setItem() {} },
        document: { createElement: () => ({ style: {}, appendChild() {} }) },
        helpFor: () => '', setHelpText() {},
        setClothesCoreLora() {}, syncGenerationControls() {}, syncCostumeEditControls() {},
        syncDOMWidgetWidth() {},
        spritePreviewNavigator: { invalidate() {}, hideNav() {} },
        updatePreviewImage() {}, loadCharacterInfo: async () => true,
        showInfo: (title, message) => messages.push({ title, message }),
        Option: class { constructor(text, value) { this.text = text; this.value = value; } },
        api: { fetchApi: async (route, options) => {
            requests.push({ route, options });
            if (route === '/vnccs/context_lists') return contexts ? contexts.promise : response({ characters: ['Alice'] });
            if (route.startsWith('/vnccs/list_costumes?')) return response(['Naked', 'Dress', 'Coat']);
            if (route.startsWith('/vnccs/get_costume?')) return metadata ? metadata.promise : response(disk);
            if (route === '/vnccs/save_costume') {
                disk = JSON.parse(options.body).info;
                return save ? save.promise : response({ status: 'ok' });
            }
            throw new Error(route);
        } },
    });
    const initial = between(source, '// Initial Load\n', 'container.appendChild(topRow);')
        .replace('(async () => {', 'globalThis.initialLoad = (async () => {');
    vm.runInContext(`${guards}
        ${between(source, '// State\n', '// Modal Helper')}
        Object.assign(els, controls);
        ${between(source, 'const hasSelectedEditableCostume =', 'const showCreateCostumeRequired =')}
        ${between(source, 'const beginCostumesRequest =', 'const updatePreviewImage =')}
        ${between(source, 'node._vnccsRestoreClothesState =', '// Initial Load')}
        ${between(source, 'const onConfigure = nodeType.prototype.onConfigure;', '\n        }\n    }\n});')}
        ${initial}
        globalThis.h = { state, els, loadCostumeInfo, saveCostumeToBackend };
    `, context);
    const configure = state => nodeType.prototype.onConfigure.call(node, { widgets_values: [JSON.stringify(state)] });
    const serialize = () => {
        const workflow = { widgets_values: ['old snapshot'] };
        node.onSerialize(workflow);
        return JSON.parse(workflow.widgets_values[0]);
    };
    return { ...context.h, context, node, dataWidget, requests, messages, configure, serialize };
}

for (const pendingAt of ['context', 'metadata']) {
    test(`constructor/configure preserves workflow draft with ${pendingAt} response pending`, async () => {
        const contexts = deferred(), metadata = pendingAt === 'metadata' ? deferred() : null;
        const h = setup({ contexts, metadata });
        if (metadata) {
            h.configure({ character: 'Alice', costume: 'Dress' });
            contexts.resolve(response({ characters: ['Alice'] }));
            await new Promise(resolve => setImmediate(resolve));
            assert.ok(h.requests.some(({ route }) => route.startsWith('/vnccs/get_costume?')));
        }
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'workflow silk', extension: { id: 9 } } });
        assert.equal(h.els.top.value, 'workflow silk');
        assert.equal(h.serialize().costume_info.top, 'workflow silk');
        (metadata || contexts).resolve(response(metadata ? { top: 'disk cotton', negative_prompt: 'blur' } : { characters: ['Alice'] }));
        await h.context.initialLoad;
        const serialized = h.serialize();
        assert.equal(serialized.costume_info.top, 'workflow silk');
        assert.equal(serialized.costume_info.negative_prompt, 'blur');
        assert.equal(serialized.costume_info.extension.id, 9);
        assert.equal(JSON.parse(h.dataWidget.value).costume_info.top, 'workflow silk');
        assert.deepEqual(h.messages, []);
    });
}

test('reload before costume save acknowledgement retains authored draft and unknown metadata', async () => {
    const pendingSave = deferred(), h = setup({ save: pendingSave });
    h.configure({ character: 'Alice', costume: 'Dress' });
    await h.context.initialLoad;
    h.state.costume_info.top = 'typed silk';
    const save = h.saveCostumeToBackend();
    const workflow = h.serialize();
    const posted = JSON.parse(h.requests.at(-1).options.body).info;
    assert.equal(posted.negative_prompt, 'blur');
    assert.deepEqual(posted.extension, { id: 7 });
    const reloaded = setup();
    reloaded.configure(workflow);
    await reloaded.context.initialLoad;
    assert.equal(reloaded.serialize().costume_info.top, 'typed silk');
    assert.equal(reloaded.serialize().costume_info.negative_prompt, 'blur');
    assert.equal(reloaded.serialize().costume_info.extension.id, 7);
    pendingSave.resolve(response({ status: 'ok' }));
    await save;
});

test('typing before initial server response preserves the draft in the hidden workflow widget', async () => {
    const contexts = deferred(), h = setup({ contexts });
    h.configure({ character: 'Alice', costume: 'Dress' });
    vm.runInContext(`${between(source, 'const createField =', 'const createSegmentedField =')}\ncreateField('top', '', false);`, h.context);
    h.els.top.oninput({ target: { value: 'typed before response' } });
    assert.equal(JSON.parse(h.dataWidget.value).costume_info.top, 'typed before response');
    contexts.resolve(response({ characters: ['Alice'] }));
    await h.context.initialLoad;
    assert.equal(h.serialize().costume_info.top, 'typed before response');
    assert.equal(h.els.top.value, 'typed before response');
});

test('explicit costume selection loads disk metadata after restoration', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'draft silk' } });
    await h.context.initialLoad;
    h.state.costume = 'Coat';
    assert.equal(await h.loadCostumeInfo(), true);
    assert.equal(h.serialize().costume_info.top, 'disk cotton');
    assert.equal(h.serialize().costume_info.negative_prompt, 'blur');
});

test('unknown costume metadata is preserved without changing unrelated controls', async () => {
    const h = setup({ disk: { top: 'silk', costSel: 'extension data' } });
    h.configure({ character: 'Alice', costume: 'Dress' });
    await h.context.initialLoad;
    assert.equal(h.serialize().costume_info.costSel, 'extension data');
    assert.equal(h.els.costSel.value, 'Dress');
});

test('failed initial metadata read keeps the restored draft and reports the error', async () => {
    const metadata = deferred(), h = setup({ metadata });
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'draft silk' } });
    metadata.resolve({ ok: false, json: async () => ({ error: 'unavailable' }) });
    await h.context.initialLoad;
    assert.equal(h.serialize().costume_info.top, 'draft silk');
    assert.equal(h.serialize().costume, 'Dress');
    assert.equal(h.messages.length, 1);
});
