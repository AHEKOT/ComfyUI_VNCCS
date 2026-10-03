import { createWidgetContext } from './widget_context.mjs';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../web/vnccs_clothes_designer.js', import.meta.url), 'utf8');
const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
const between = (text, start, end) => {
    const offset = text.indexOf(start);
    const finish = text.indexOf(end, offset);
    assert.ok(offset >= 0 && finish > offset);
    return text.slice(offset, finish);
};
const guards = between(common, 'export function registerCleanup', '// ── Widget Data Sync').replaceAll('export ', '');
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};

function setup({ costumes = ['Naked', 'Dress', 'Casual'], deletion, costumeInfo, wizard } = {}) {
    const state = { character: 'Alice', costume: 'Dress', costume_info: { top: 'silk' },
        selected_preview_sprite: { filename: 'old.png' } };
    const dataWidget = { name: 'widget_data', value: '{}' };
    const node = { widgets: [dataWidget] };
    const control = () => ({ disabled: false, style: {} });
    const costSel = { ...control(), options: [], add(option) { this.options.push(option); },
        set innerHTML(value) { this.options = []; } };
    const els = { costSel, btnDel: control(), btnGen: control(), wizardBtn: control(),
        previewImg: control(), placeholder: control(), top: { value: 'silk' },
        bottom: { value: '' }, head: { value: '' }, face: { value: '' }, shoes: { value: '' } };
    const requests = [], modals = [], messages = [], previews = [];
    const context = createWidgetContext({
        state, dataWidget, node, els,
        defaultState: { costume_info: { top: '', bottom: '', head: '', face: '', shoes: '' } },
        charSel: control(), costSel, btnDelCostume: els.btnDel, btnNewCostume: control(),
        storage: { setItem() {} },
        document: { createElement: tag => ({ tagName: tag, children: [], value: '',
            append(...children) { this.children.push(...children); }, focus() {} }) },
        setTimeout() {}, ensureQwenVLReady: async () => true,
        showWizardModelError(error) { messages.push({ title: 'Wizard Error', message: error }); },
        Option: class { constructor(text, value) { this.text = text; this.value = value; } },
        showModal(title, content, buttons) { modals.push({ title, content: content(), buttons }); },
        showInfo(title, message) { messages.push({ title, message }); },
        spritePreviewNavigator: { invalidate() {}, hideNav() {} },
        updatePreviewImage() { previews.push(state.costume); },
        api: { fetchApi: async (route, options) => {
            requests.push({ route, options });
            if (route === '/vnccs/delete_costume') {
                if (deletion) return await deletion();
                costumes = costumes.filter(name => name !== JSON.parse(options.body).costume);
                return { ok: true, json: async () => ({ status: 'ok' }) };
            }
            if (route === '/vnccs/clothes_wizard') return await wizard();
            if (route === '/vnccs/save_costume') return { ok: true, json: async () => ({ status: 'ok' }) };
            if (route.startsWith('/vnccs/list_costumes')) return { ok: true, json: async () => costumes };
            if (route.startsWith('/vnccs/get_costume')) {
                if (costumeInfo) return await costumeInfo();
                return { ok: true, json: async () => state.costume ? { top: 'shirt' } : {} };
            }
            throw new Error(`Unexpected request: ${route}`);
        } },
    });
    vm.runInContext(`${guards}
        const beginDeleteRequest = createRequestGuard(node);
        const beginClothesWizardRequest = createRequestGuard(node);
        const beginSelectionRequest = createRequestGuard(node);
        const beginPreviewRequest = createRequestGuard(node);
        ${between(source, 'const saveState = () =>', 'node._randomizeSeedIfNeeded =')}
        ${between(source, 'const pendingCostumeSaves =', '// Modal Helper')}
        ${between(source, 'const hasSelectedEditableCostume = () =>', 'const onValidationError =')}
        ${between(source, 'const beginCostumesRequest', 'const updatePreviewImage')}
        ${between(source, 'btnDelCostume.onclick = () =>', 'actionRow.appendChild(btnDelCostume)')}
        ${between(source, 'const openClothesWizard =', 'const getConnectedControlCenterState =')}
        ${between(source, 'costSel.onchange =', 'els.costSel = costSel;')}
        globalThis.openWizard = openClothesWizard;
        globalThis.pendingSaves = pendingCostumeSaves;
    `, context);
    const open = () => context.btnDelCostume.onclick();
    const confirm = () => modals.at(-1).buttons.find(button => button.text === 'DELETE').action();
    return { context, state, dataWidget, node, els, requests, modals, messages, previews,
        pendingCostumeSaves: context.pendingSaves, open, confirm };
}

for (const remaining of [true, false]) {
    test(`delete refreshes and serializes the ${remaining ? 'next costume' : 'empty selection'}`, async () => {
        const h = setup({ costumes: remaining ? ['Naked', 'Dress', 'Casual'] : ['Naked', 'Original', 'Dress'] });
        h.open();
        assert.match(h.modals[0].content.innerText, /Dress.*Alice/);
        assert.equal(await h.confirm(), false);
        const sent = h.requests[0];
        assert.equal(sent.route, '/vnccs/delete_costume');
        assert.equal(sent.options.method, 'POST');
        assert.equal(new Headers(sent.options.headers).get('X-VNCCS-CSRF'), '1');
        assert.deepEqual(JSON.parse(sent.options.body), { character: 'Alice', costume: 'Dress' });
        assert.equal(h.state.costume, remaining ? 'Casual' : '');
        assert.equal(h.els.btnDel.disabled, !remaining);
        assert.equal(h.els.top.disabled, !remaining);
        assert.equal(h.state.selected_preview_sprite, null);
        const saved = JSON.parse(h.dataWidget.value);
        assert.equal(saved.costume, h.state.costume);
        assert.equal(saved.costume_info.top, remaining ? 'shirt' : '');
        assert.deepEqual(h.previews, [h.state.costume]);
    });
}

test('API failure keeps the selected costume and all its serialized data available for retry', async () => {
    const h = setup({ deletion: async () => ({ ok: false, status: 500, json: async () => ({ error: 'Delete denied' }) }) });
    const original = JSON.stringify(h.state);
    h.open();
    await assert.rejects(h.confirm(), /Delete denied/);
    assert.equal(JSON.stringify(h.state), original);
    assert.equal(h.els.btnDel.disabled, false);
    assert.equal(h.els.btnGen.disabled, false);
    assert.equal(h.requests.length, 1);
    assert.equal(h.previews.length, 0);
});

for (const costume of ['', 'Naked', 'Original']) {
    test(`delete refuses the protected or empty selection '${costume}'`, () => {
        const h = setup();
        h.state.costume = costume;
        h.open();
        assert.equal(h.modals.length, 0);
        assert.equal(h.requests.length, 0);
        assert.equal(h.messages.length, 1);
    });
}

test('confirmation cannot delete an earlier selection after changing the character', async () => {
    const h = setup();
    h.open();
    h.state.character = 'Bob';
    assert.equal(await h.confirm(), false);
    assert.equal(h.requests.length, 0);
});

for (const change of ['selection', 'removal']) {
    test(`a deletion response does not change widget state after ${change}`, async () => {
        const task = deferred();
        const h = setup({ deletion: () => task.promise });
        h.open();
        const deleting = h.confirm();
        await new Promise(resolve => setImmediate(resolve));
        if (change === 'selection') h.state.character = 'Bob';
        else h.node.onRemoved();
        const expected = JSON.stringify(h.state);
        task.resolve({ ok: true, json: async () => ({ status: 'ok' }) });
        assert.equal(await deleting, false);
        assert.equal(JSON.stringify(h.state), expected);
        assert.equal(h.requests.length, 1);
        assert.equal(h.previews.length, 0);
    });
}

test('deletion waits for earlier field saves and blocks conflicting controls', async () => {
    const save = deferred();
    const h = setup();
    h.pendingCostumeSaves.add(save.promise);
    h.open();
    const deleting = h.confirm();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(h.requests.length, 0);
    assert.equal(h.els.btnGen.disabled, true);
    assert.equal(h.els.top.disabled, true);
    save.resolve();
    await deleting;
    assert.equal(h.requests[0].route, '/vnccs/delete_costume');
    assert.equal(h.els.btnGen.disabled, false);
});

test('an active preview blocks deletion before opening a confirmation', () => {
    const h = setup();
    h.els.btnGen.disabled = true;
    h.open();
    assert.equal(h.modals.length, 0);
    assert.equal(h.messages.length, 1);
    assert.equal(h.requests.length, 0);
});

test('metadata failure after DELETE leaves a safe serialized selection and allows an explicit retry', async () => {
    let attempts = 0;
    const h = setup({ costumeInfo: async () => ++attempts === 1
        ? { ok: false, json: async () => ({ error: 'Disk unavailable' }) }
        : { ok: true, json: async () => ({ top: 'jacket' }) } });
    h.open();
    await h.confirm();
    assert.equal(h.state.costume, '');
    assert.equal(JSON.parse(h.dataWidget.value).costume, '');
    assert.equal(h.els.btnGen.disabled, true);
    assert.equal(h.els.top.disabled, true);
    assert.equal(h.els.costSel.disabled, false);
    assert.equal(h.requests.filter(request => request.route === '/vnccs/save_costume').length, 0);
    await h.context.costSel.onchange({ target: { value: 'Casual' } });
    assert.equal(h.state.costume, 'Casual');
    assert.equal(JSON.parse(h.dataWidget.value).costume_info.top, 'jacket');
    assert.equal(h.els.btnGen.disabled, false);
    assert.equal(h.els.top.disabled, false);
});

test('committed deletion with a cleanup warning refreshes state and reports the remaining files', async () => {
    const h = setup({ costumes: ['Naked', 'Casual'], deletion: async () => ({ ok: true,
        json: async () => ({ status: 'ok', warning: 'Cleanup pending at .vnccs-delete-example' }) }) });
    h.open();
    assert.equal(await h.confirm(), false);
    assert.equal(h.state.costume, 'Casual');
    assert.equal(JSON.parse(h.dataWidget.value).costume, 'Casual');
    assert.match(h.messages[0].message, /Cleanup pending/);
});

for (const change of ['delete', 'selection', 'close', 'removal']) {
    test(`late Wizard results do not save costume fields after ${change}`, async () => {
        const pending = deferred();
        const h = setup({ wizard: () => pending.promise });
        h.context.openWizard();
        const modal = h.modals[0];
        modal.content.children[1].value = 'A silk dress';
        const overlay = { isConnected: true };
        const filling = modal.buttons.find(button => button.text === 'FILL FIELDS').action(overlay, {});
        await new Promise(resolve => setImmediate(resolve));
        if (change === 'delete') { h.open(); await h.confirm(); }
        if (change === 'selection') await h.context.costSel.onchange({ target: { value: 'Casual' } });
        if (change === 'close') overlay.isConnected = false;
        if (change === 'removal') h.node.onRemoved();
        const expected = JSON.stringify(h.state);
        pending.resolve({ ok: true, json: async () => ({ top: 'silk dress' }) });
        assert.equal(await filling, false);
        assert.equal(JSON.stringify(h.state), expected);
        assert.equal(h.requests.filter(request => request.route === '/vnccs/save_costume').length, 0);
    });
}
