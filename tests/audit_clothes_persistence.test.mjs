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

function setup({ contexts, metadata, save, nativeCharacterSelect = false, disk = { top: 'disk cotton', negative_prompt: 'blur', extension: { id: 7 } } } = {}) {
    const dataWidget = { name: 'widget_data', value: '{}' };
    const node = { widgets: [dataWidget] }, requests = [], messages = [], renderedTabs = [];
    const selection = () => ({ options: [], add(option) { this.options.push(option); },
        set innerHTML(value) { this.options = []; } });
    const controls = { charSelect: selection(), costSel: selection(),
        ...Object.fromEntries(['btnGen', 'btnDel', 'wizardBtn', 'top', 'bottom', 'head', 'face', 'shoes', 'placeholder', 'previewImg']
            .map(key => [key, { value: '', style: {} }])) };
    controls._ui = {
        presentPreview(image, placeholder, src) { image.src = src; image.style.display = 'block'; placeholder.style.display = 'none'; },
        emptyPreview(image, placeholder) { image.style.display = 'none'; placeholder.style.display = 'flex'; },
    };
    if (nativeCharacterSelect) {
        let selected = '';
        controls.charSelect = {
            options: [],
            get value() { return selected; },
            set value(value) { selected = this.options.some(option => option.value === value) ? value : ''; },
            add(option) { this.options.push(option); if (this.options.length === 1) selected = option.value; },
            set innerHTML(value) { this.options = []; selected = ''; },
        };
    }
    const nodeType = { prototype: { onConfigure(info) { this.widgets[0].value = info.widgets_values[0]; } } };
    const context = createWidgetContext({
        node, nodeType, dataWidget, controls,
        setTimeout() {},
        storage: { getItem() { return null; }, setItem() {} },
        document: { createElement: () => ({ style: {}, appendChild() {} }) },
        helpFor: () => '', setHelpText() {},
        setClothesCoreLora() {}, syncGenerationControls() {}, syncCostumeEditControls() {},
        refreshTabs() { renderedTabs.push(context.h.state.activeTab); },
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
        globalThis.h = { state, els, loadCostumes, loadCostumeInfo, saveCostumeToBackend };
    `, context);
    const configure = state => nodeType.prototype.onConfigure.call(node, { widgets_values: [JSON.stringify(state)] });
    const serialize = () => {
        const workflow = { widgets_values: ['old snapshot'] };
        node.onSerialize(workflow);
        return JSON.parse(workflow.widgets_values[0]);
    };
    return { ...context.h, context, node, dataWidget, requests, messages, renderedTabs, configure, serialize };
}

function wireSelection(h) {
    Object.assign(h.context, { charSel: h.els.charSelect, costSel: h.els.costSel, applyPoseStudioValues() {} });
    vm.runInContext(`${between(source, 'const syncCostumeEditControls =', 'const onValidationError =')}
        ${between(source, 'const beginCharacterInfoRequest =', '// UI Builders')}
        ${between(source, 'charSel.onchange =', 'charRow.appendChild(charSel)')}
        ${between(source, 'costSel.onchange =', 'els.costSel = costSel;')}`, h.context);
}

for (const firstFails of [false, true]) {
test(`Clothes field saves keep the newest draft after an earlier ${firstFails ? 'failed' : 'late'} request`, async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'old cotton' } });
    await h.context.initialLoad;
    const pending = [deferred(), deferred()], bodies = [];
    let persisted;
    h.context.api.fetchApi = (route, options) => {
        assert.equal(route, '/vnccs/save_costume');
        const body = JSON.parse(options.body), index = bodies.length;
        bodies.push(body);
        return pending[index].promise.then(() => {
            if (index === 0 && firstFails) return { ok: false, status: 500, json: async () => ({ error: 'Earlier save failed' }) };
            persisted = body;
            return response({ status: 'ok' });
        });
    };
    const first = h.saveCostumeToBackend();
    const firstResult = firstFails ? assert.rejects(first, /Earlier save failed/) : first;
    h.state.costume_info.top = 'new silk';
    const second = h.saveCostumeToBackend();
    h.state.character = 'Bob';
    h.state.costume = 'Coat';
    h.state.costume_info = { top: 'Bob wool' };
    pending[1].resolve();
    await new Promise(resolve => setImmediate(resolve));
    pending[0].resolve();
    await Promise.all([firstResult, second]);
    assert.equal(persisted.character, 'Alice', 'Queued saves must retain their original owner');
    assert.equal(persisted.costume, 'Dress');
    assert.equal(persisted.info.top, 'new silk', 'A late older save must not overwrite the newest draft');
    assert.deepEqual(bodies.map(body => body.info.top), ['old cotton', 'new silk']);
});
}

for (const characters of [[], ['Alice']]) {
    test(`Clothes inventory ${JSON.stringify(characters)} keeps the restored missing character visible`, async () => {
        const contexts = deferred(), h = setup({ contexts, nativeCharacterSelect: true });
        h.configure({ character: 'Missing Bob', costume: 'Dress', costume_info: { top: 'authored silk' } });
        contexts.resolve(response({ characters }));
        await h.context.initialLoad;
        assert.equal(h.els.charSelect.value, 'Missing Bob');
        assert.equal(h.serialize().character, 'Missing Bob');
        assert.equal(h.serialize().costume_info.top, 'authored silk');
        assert.ok(h.els.charSelect.options.some(option => option.value === 'Missing Bob'));
    });
}

test('Clothes configure keeps an unavailable character visible after inventory has loaded', async () => {
    const h = setup({ nativeCharacterSelect: true });
    await h.context.initialLoad;
    h.configure({ character: 'Missing Bob', costume: 'Dress', costume_info: { top: 'restored wool' } });
    assert.equal(h.els.charSelect.value, 'Missing Bob');
    assert.equal(h.serialize().character, 'Missing Bob');
    assert.equal(h.serialize().costume_info.top, 'restored wool');
});

for (const phase of ['confirmation', 'response', 'error_body']) {
    test(`Clothes restoration discards an obsolete wizard at ${phase}`, async () => {
        const h = setup();
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
        await h.context.initialLoad;
        const pending = deferred();
        let buttons;
        Object.assign(h.context, {
            document: { createElement: () => ({ value: 'old wizard', append() {}, focus() {} }) },
            showModal(title, builder, actions) { builder(); buttons = actions; },
            ensureQwenVLReady: async () => true,
            showWizardModelError: error => h.messages.push(error),
        });
        h.context.api.fetchApi = async route => {
            h.requests.push({ route });
            if (route === '/vnccs/clothes_wizard') return phase === 'error_body'
                ? { ok: false, json: () => pending.promise } : pending.promise;
            if (route === '/vnccs/save_costume') return response({ status: 'ok' });
            throw new Error(route);
        };
        vm.runInContext(`${between(source, 'const openClothesWizard =', 'const getConnectedControlCenterState =')}
            globalThis.openWizard = openClothesWizard;`, h.context);
        h.context.openWizard();
        const overlay = { isConnected: true };
        const operation = phase === 'confirmation' ? null : buttons[1].action(overlay, {});
        if (operation) await new Promise(resolve => setImmediate(resolve));
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'new workflow wool' } });
        const original = h.dataWidget.value, count = h.requests.length;
        pending.resolve(phase === 'error_body' ? { message: 'Old wizard error' } : response({ top: 'old wizard cotton' }));
        await (operation || buttons[1].action(overlay, {}));
        assert.equal(h.state.costume_info.top, 'new workflow wool');
        assert.equal(h.dataWidget.value, original);
        assert.equal(h.requests.length, count, 'An old wizard must neither generate nor save after restoration');
        assert.deepEqual(h.messages, []);
    });
}

for (const phase of ['confirmation', 'response']) {
    test(`Clothes restoration discards an obsolete deletion at ${phase}`, async () => {
        const h = setup();
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
        await h.context.initialLoad;
        const pending = deferred();
        let buttons;
        Object.assign(h.context, {
            charSel: h.els.charSelect, costSel: h.els.costSel, btnDelCostume: h.els.btnDel, btnNewCostume: {},
            showModal(title, builder, actions) { builder(); buttons = actions; },
        });
        h.context.api.fetchApi = async route => { h.requests.push({ route }); return pending.promise; };
        vm.runInContext(between(source, 'btnDelCostume.onclick =', 'actionRow.appendChild(btnDelCostume)'), h.context);
        h.context.btnDelCostume.onclick();
        const operation = phase === 'confirmation' ? null : buttons[1].action();
        if (operation) await new Promise(resolve => setImmediate(resolve));
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'new workflow wool' } });
        const original = h.dataWidget.value, count = h.requests.length;
        pending.resolve(response({ status: 'ok' }));
        await (operation || buttons[1].action());
        assert.equal(h.dataWidget.value, original);
        assert.equal(h.state.costume_info.top, 'new workflow wool');
        assert.equal(h.requests.length, count);
    });
}

test('Clothes restoration discards a preview check started for the previous workflow', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
    await h.context.initialLoad;
    const pending = deferred(), previews = [];
    h.context.api.fetchApi = () => pending.promise;
    h.context.newNavigator = { invalidate() {}, showFallback: (...args) => previews.push(args) };
    vm.runInContext(`spritePreviewNavigator = newNavigator;
        ${between(source, 'const updatePreviewImage =', 'const onPreviewUpdated =')}
        globalThis.refreshPreview = updatePreviewImage;`, h.context);
    const operation = h.context.refreshPreview(true);
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'new workflow wool' } });
    pending.resolve(response({}));
    await operation;
    assert.deepEqual(previews, []);
});

for (const change of ['restoration', 'edit', 'current']) {
    test(`Clothes generated preview respects the current draft after ${change}`, async () => {
        const h = setup();
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
        await h.context.initialLoad;
        const pending = deferred(), loading = { remove() {} };
        Object.assign(h.context, {
            btnGen: h.els.btnGen, container: { appendChild() {} },
            document: { createElement: () => loading }, getConnectedControlCenterState: () => ({}),
        });
        h.context.node._randomizeSeedIfNeeded = () => {};
        h.context.api.fetchApi = async route => {
            if (route === '/vnccs/save_costume') return response({ status: 'ok' });
            assert.equal(route, '/vnccs/control_center/clothes_preview');
            return pending.promise;
        };
        vm.runInContext(between(source, 'btnGen.onclick = async', 'els.btnGen = btnGen;'), h.context);
        const operation = h.context.btnGen.onclick();
        await new Promise(resolve => setImmediate(resolve));
        if (change === 'restoration') h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'new workflow wool' } });
        if (change === 'edit') h.state.costume_info.top = 'new edited wool';
        pending.resolve(response({ image: 'old-silk-preview' }));
        await operation;
        assert.equal(h.els.previewImg.src, change === 'current' ? 'data:image/png;base64,old-silk-preview' : undefined);
        if (change !== 'current') assert.match(h.state.costume_info.top, /new .*wool/);
    });
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
    await new Promise(resolve => setImmediate(resolve));
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

test('Clothes metadata started by New cannot overwrite a replacement workflow for the same costume', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'old silk' } });
    await h.context.initialLoad;
    const metadata = deferred();
    h.context.api.fetchApi = () => metadata.promise;
    const operation = h.loadCostumeInfo();
    assert.equal(h.node._vnccsClothesSelectionPending, undefined, 'New reloads metadata without a selector transaction');
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'new workflow wool' } });
    const restored = h.serialize();
    metadata.resolve(response({ top: 'stale disk cotton' }));
    assert.equal(await operation, false);
    assert.deepEqual(h.serialize(), restored);
    assert.equal(h.els.top.value, 'new workflow wool');
    assert.deepEqual(h.messages, []);
});

test('Clothes inventory started by New cannot load metadata over a replacement workflow', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'old silk' } });
    await h.context.initialLoad;
    const inventory = deferred(), laterRequests = [];
    h.context.api.fetchApi = route => {
        laterRequests.push(route);
        return route.startsWith('/vnccs/list_costumes?') ? inventory.promise : response({ top: 'stale disk cotton' });
    };
    const operation = h.loadCostumes();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'new workflow wool' } });
    const restored = h.serialize();
    inventory.resolve(response(['Naked', 'Dress', 'Coat']));
    assert.equal(await operation, false);
    assert.deepEqual(h.serialize(), restored);
    assert.equal(h.els.top.value, 'new workflow wool');
    assert.equal(laterRequests.length, 1, 'An obsolete inventory must not begin a metadata request');
    assert.deepEqual(h.messages, []);
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

test('configure renders the restored Clone Clothes tab', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', activeTab: 'clone', clone_image: { name: 'coat.png' } });
    await h.context.initialLoad;
    assert.deepEqual(h.renderedTabs, ['clone']);
    assert.equal(h.serialize().activeTab, 'clone');
});

test('reconfigure clears clone and preview state absent from the new workflow', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', activeTab: 'clone', clone_image: { name: 'alice-coat.png' },
        selected_preview_sprite: { character: 'Alice', costume: 'Dress', index: 2 } });
    await h.context.initialLoad;
    h.configure({ character: 'Bob', costume: 'Coat', costume_info: { top: 'Bob top' } });
    const state = h.serialize();
    assert.equal(state.character, 'Bob');
    assert.equal(state.activeTab, 'generate');
    assert.equal(state.clone_image, null);
    assert.equal(state.selected_preview_sprite, null);
    assert.equal(state.costume_info.top, 'Bob top');
});

test('character selection retains committed costume ownership and blocks edits until metadata is ready', async () => {
    const h = setup();
    h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' },
        character_info: { name: 'Alice', age: 21, sex: 'female', extra: 'only Alice' } });
    await h.context.initialLoad;
    const pending = deferred(), saved = [];
    h.context.api.fetchApi = async (route, options) => {
        if (route.startsWith('/vnccs/character_info?')) return pending.promise;
        if (route.startsWith('/vnccs/list_costumes?')) return response(['Dress']);
        if (route.startsWith('/vnccs/get_costume?')) return response({ top: 'Bob cotton' });
        if (route === '/vnccs/save_costume') { saved.push(JSON.parse(options.body)); return response({ status: 'ok' }); }
        throw new Error(route);
    };
    wireSelection(h);
    vm.runInContext(`${between(source, 'const createField =', 'const createSegmentedField =')} createField('top', '', false);`, h.context);
    let newCostumeButtons, opened = 0;
    h.context.btnNewCostume = {};
    h.context.showModal = (_title, _build, buttons) => { opened++; newCostumeButtons = buttons; };
    vm.runInContext(between(source, 'btnNewCostume.onclick =', 'els.btnNew = btnNewCostume;'), h.context);
    h.context.btnNewCostume.onclick();
    const change = h.context.charSel.onchange({ target: { value: 'Bob' } });
    assert.equal(h.state.character, 'Alice');
    assert.equal(h.serialize().costume_info.top, 'Alice silk');
    assert.equal(h.els.top.disabled, true);
    h.context.btnNewCostume.onclick();
    assert.equal(await newCostumeButtons[1].action(), true);
    assert.equal(opened, 1);
    h.els.top.oninput({ target: { value: 'Wrong owner edit' } });
    assert.equal(h.serialize().costume_info.top, 'Alice silk');
    await assert.rejects(h.saveCostumeToBackend(), /loading/i);
    assert.deepEqual(saved, []);
    pending.resolve(response({ name: 'Bob', age: 32, sex: 'male' }));
    await change;
    assert.equal(h.state.character, 'Bob');
    assert.equal(h.state.character_info.name, 'Bob');
    assert.equal(h.state.character_info.extra, undefined);
    assert.equal(h.state.costume_info.top, 'Bob cotton');
    assert.equal(h.els.top.disabled, false);
    await h.saveCostumeToBackend();
    assert.deepEqual(saved, [{ character: 'Bob', costume: 'Dress', info: { top: 'Bob cotton', bottom: '', head: '', face: '', shoes: '' } }]);
});

for (const failure of ['http', 'json', 'network', 'schema']) {
    test(`failed clothes character selection preserves authored state and reports ${failure} errors`, async () => {
        const h = setup();
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
        await h.context.initialLoad;
        wireSelection(h);
        const original = h.serialize();
        h.context.api.fetchApi = async () => {
            if (failure === 'network') throw new Error('Offline');
            if (failure === 'json') return { ok: true, json: async () => { throw new Error('Invalid response'); } };
            return { ok: failure !== 'http', status: 502, json: async () => failure === 'schema' ? [] : {} };
        };
        await h.context.charSel.onchange({ target: { value: 'Bob' } });
        assert.deepEqual(h.serialize(), original);
        assert.equal(h.els.charSelect.value, 'Alice');
        assert.equal(h.els.top.disabled, false);
        assert.equal(h.node._vnccsClothesSelectionPending, false);
        assert.equal(h.messages.at(-1).title, 'Character Load Failed');
    });
}

for (const replacement of ['new selection', 'configuration']) {
    test(`obsolete clothes character response cannot unblock or overwrite ${replacement}`, async () => {
        const h = setup();
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
        await h.context.initialLoad;
        wireSelection(h);
        const bob = deferred(), carol = deferred();
        h.context.api.fetchApi = async route => {
            if (route.includes('character_info?character=Bob')) return bob.promise;
            if (route.includes('character_info?character=Carol')) return carol.promise;
            if (route.includes('list_costumes?')) return response(['Dress']);
            if (route.includes('get_costume?')) return response({ top: 'Carol cotton' });
            throw new Error(route);
        };
        const older = h.context.charSel.onchange({ target: { value: 'Bob' } });
        if (replacement === 'configuration') {
            h.configure({ character: 'Carol', costume: 'Dress', costume_info: { top: 'Carol draft' } });
            bob.resolve(response({ name: 'Bob', age: 30 }));
            await older;
            assert.equal(h.state.costume_info.top, 'Carol draft');
        } else {
            const newer = h.context.charSel.onchange({ target: { value: 'Carol' } });
            bob.resolve(response({ name: 'Bob', age: 30 }));
            await older;
            assert.equal(h.state.character, 'Alice');
            assert.equal(h.els.top.disabled, true);
            assert.equal(h.node._vnccsClothesSelectionPending, true);
            carol.resolve(response({ name: 'Carol', age: 40 }));
            await newer;
            assert.equal(h.state.costume_info.top, 'Carol cotton');
        }
        assert.equal(h.state.character, 'Carol');
        assert.equal(h.els.top.disabled, false);
        assert.equal(h.node._vnccsClothesSelectionPending, false);
    });
}

for (const success of [true, false]) {
    test(`costume selection keeps committed metadata until ${success ? 'success' : 'failure'}`, async () => {
        const h = setup();
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'Alice silk' } });
        await h.context.initialLoad;
        wireSelection(h);
        const pending = deferred();
        h.context.api.fetchApi = async () => pending.promise;
        const original = h.serialize();
        const change = h.context.costSel.onchange({ target: { value: 'Coat' } });
        assert.deepEqual(h.serialize(), original);
        assert.equal(h.els.top.disabled, true);
        await assert.rejects(h.saveCostumeToBackend(), /loading/i);
        pending.resolve({ ok: success, json: async () => ({ top: 'Coat cotton' }) });
        await change;
        assert.equal(h.state.costume, success ? 'Coat' : 'Dress');
        assert.equal(h.state.costume_info.top, success ? 'Coat cotton' : 'Alice silk');
        assert.equal(h.els.costSel.value, h.state.costume);
        assert.equal(h.els.top.disabled, false);
        assert.equal(h.node._vnccsClothesSelectionPending, false);
    });
}

for (const raw of ['{"character":"Alice"', 'null', '[]', '{"costume_info":[]}']) {
    test(`corrupt clothes state remains recoverable during initialization: ${raw}`, async () => {
        const contexts = deferred(), h = setup({ contexts });
        h.dataWidget.value = raw;
        h.node._vnccsRestoreClothesState();
        const workflow = { widgets_values: ['older snapshot'] };
        h.node.onSerialize(workflow);
        assert.equal(workflow.widgets_values[0], raw);
        assert.equal(h.dataWidget.value, raw);
        await assert.rejects(h.saveCostumeToBackend(), /Cannot restore/);
        contexts.resolve(response({ characters: ['Alice'] }));
        await h.context.initialLoad;
        assert.equal(h.dataWidget.value, raw);
        assert.equal(h.requests.length, 1, 'Failed restoration must not load or save replacement metadata');
        assert.equal(h.messages[0].title, 'State Restore Failed');
        h.configure({ character: 'Alice', costume: 'Dress', costume_info: { top: 'recovered silk' } });
        assert.equal(h.node._vnccsStateRestoreError, undefined);
        assert.equal(h.serialize().costume_info.top, 'recovered silk');
    });
}
