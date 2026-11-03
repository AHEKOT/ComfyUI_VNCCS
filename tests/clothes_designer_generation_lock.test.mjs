import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = readFileSync(new URL('../web/vnccs_clothes_designer.js', import.meta.url), 'utf8');
const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
function between(text, start, end) {
    const offset = text.indexOf(start), limit = text.indexOf(end, offset);
    assert.ok(offset >= 0 && limit > offset);
    return text.slice(offset, limit);
}
const response = (data, ok = true) => ({ ok, status: ok ? 200 : 500, json: async () => data });
const idleQueue = { queue_running: [], queue_pending: [] };
const tick = () => new Promise(resolve => setImmediate(resolve));
const control = () => ({
    disabled: false, style: {}, classList: { toggle() {}, add() {}, remove() {} },
    setAttribute(name, value) { this[name] = value; },
    getAttribute(name) { return this[name] || ''; },
    removeAttribute(name) { delete this[name]; },
});

function setup() {
    const listeners = new Map(), requests = [], refreshes = [];
    const node = { id: 753, _vnccsClothesWorkflowBusy: true };
    const state = { character: 'Alina', costume: 'Shelter', character_info: {}, activeTab: 'generate' };
    const els = Object.fromEntries(['btnGen', 'btnNew', 'btnDel', 'charSelect', 'wizardBtn', 'top', 'bottom', 'head', 'face', 'shoes',
        'generationLayer', 'generationBase', 'generationNew', 'generationScan', 'previewFrame', 'previewImg', 'placeholder']
        .map(name => [name, control()]));
    els.costSel = { ...control(), options: [{}] };
    els.tabBar = { children: [control(), control()] };
    const context = createWidgetContext({
        node, state, els, btnGen: els.btnGen, spritePrevBtn: control(), spriteNextBtn: control(),
        spritePreviewNavigator: null, previewLoading: control(), URLSearchParams,
        window: { addEventListener() {}, removeEventListener() {}, requestAnimationFrame: callback => callback() },
        console: { warn() {} },
        hasSelectedEditableCostume: () => !!state.costume && !['Naked', 'Original'].includes(state.costume),
        api: {
            addEventListener(name, callback) {
                if (!listeners.has(name)) listeners.set(name, new Set());
                listeners.get(name).add(callback);
            },
            removeEventListener(name, callback) { listeners.get(name)?.delete(callback); },
            fetchApi(route) {
                return new Promise(resolve => requests.push({ route, resolve }));
            },
        },
    });
    context.refreshPreviewImage = () => refreshes.push(state.costume);
    vm.runInContext(`
        ${between(common, 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '')}
        const beginPreviewRequest = createRequestGuard(node);
        const beginClothesUploadRequest = createRequestGuard(node);
        ${between(source, 'const syncCostumeEditControls =', 'const onValidationError =')}
        ${between(source, 'let previewGenerationRevision = 0;', 'spritePreviewNavigator = createSpritePreviewNavigator(')}
        ${between(source, 'const beginWorkflowStatusRequest =', 'enableMiddleMouseCanvasPan(container);')}
        ${between(source, 'btnGen.onclick = async', 'els.btnGen = btnGen;')}
        globalThis.sync = syncCostumeEditControls;
        globalThis.refreshBusy = refreshWorkflowBusy;
        globalThis.startPreview = startClothesPreviewGeneration;
        globalThis.stopPreview = stopClothesPreviewGeneration;
    `, context);
    const emit = (name, detail) => {
        for (const callback of listeners.get(name) || []) callback({ detail });
    };
    return { context, node, state, els, requests, refreshes, listeners, emit,
        status: count => emit('status', { exec_info: { queue_remaining: count } }) };
}

function assertLocked(h, locked) {
    for (const name of ['btnGen', 'btnNew', 'btnDel', 'charSelect', 'costSel', 'wizardBtn', 'top', 'bottom', 'head', 'face', 'shoes']) {
        assert.equal(h.els[name].disabled, locked, name);
    }
    for (const tab of h.els.tabBar.children) assert.equal(tab.disabled, locked);
}

for (const field of ['queue_running', 'queue_pending']) {
    test(`Clothes Designer opens locked during ${field}`, async () => {
        const h = setup();
        assertLocked(h, true);
        assert.equal(h.requests[0].route, '/queue');
        h.requests[0].resolve(response({ ...idleQueue, [field]: [[1, 'prompt']] }));
        await tick();
        assertLocked(h, true);
        await h.els.btnGen.onclick();
        assert.equal(h.requests.length, 1, 'Run must prevent a separate preview request');
        h.status(0);
        assertLocked(h, false);
    });
}

test('Clothes preview completion does not unlock controls while later workflow nodes run', async () => {
    const h = setup();
    h.status(0);
    assertLocked(h, false);
    h.emit('execution_start', { prompt_id: 'workflow' });
    assertLocked(h, true);
    h.emit('executing', '753');
    assert.equal(h.node._vnccsClothesPreviewRunning, true);
    await h.node._vnccsRevealWorkflowPreview('Alina');
    assert.equal(h.node._vnccsClothesPreviewRunning, false);
    h.emit('executing', '751');
    h.context.sync();
    assertLocked(h, true);
    assert.equal(h.els.btnGen.innerText, 'WORKFLOW BUSY...');
    h.status(0);
    assertLocked(h, false);
    assert.equal(h.els.btnGen.innerText, 'GENERATE PREVIEW / SAVE');
});

test('idle queue cannot release a standalone preview or pending costume selection', () => {
    const h = setup();
    h.status(0);
    const revision = h.context.startPreview();
    h.status(0);
    assertLocked(h, true);
    h.node._vnccsClothesSelectionPending = true;
    h.context.stopPreview(revision);
    assertLocked(h, true);
    h.node._vnccsClothesSelectionPending = false;
    h.context.sync();
    assertLocked(h, false);
});

test('late idle queue snapshots cannot override a newer workflow start', async () => {
    const h = setup();
    h.emit('execution_start', {});
    h.requests[0].resolve(response(idleQueue));
    await tick();
    assertLocked(h, true);
});

for (const event of ['execution_error', 'execution_interrupted']) {
    test(`${event} waits for queue status before unlocking Clothes Designer`, () => {
        const h = setup();
        h.emit('executing', '753');
        h.emit(event, { node_id: '753' });
        assert.equal(h.node._vnccsClothesPreviewRunning, false);
        assertLocked(h, true);
        h.status(1);
        assertLocked(h, true);
        h.status(0);
        assertLocked(h, false);
    });
}

for (const result of [response({}, false), response({})]) {
    test(`unavailable or invalid queue state keeps Clothes Designer locked (${result.status})`, async () => {
        const h = setup();
        h.requests[0].resolve(result);
        await tick();
        assertLocked(h, true);
        h.status(0);
        assertLocked(h, false);
    });
}

test('reconnect restores queue state; removal discards late replies and node listeners', async () => {
    const h = setup();
    h.status(0);
    h.emit('reconnecting');
    assertLocked(h, true);
    h.emit('reconnected');
    h.requests[1].resolve(response(idleQueue));
    await tick();
    assertLocked(h, false);
    assert.deepEqual(h.refreshes, ['Shelter']);
    const pending = h.context.refreshBusy();
    h.node.onRemoved();
    h.requests[2].resolve(response({ ...idleQueue, queue_running: [[1, 'prompt']] }));
    await pending;
    assertLocked(h, false);
    assert.deepEqual([...h.listeners].filter(([, callbacks]) => callbacks.size)
        .map(([name, callbacks]) => [name, callbacks.size]), [['vnccs.wizard.stage', 1]]);
});

test('finishing a workflow preserves protected base costume restrictions', () => {
    const h = setup();
    h.state.costume = 'Naked';
    h.status(0);
    assert.equal(h.els.btnNew.disabled, false);
    assert.equal(h.els.charSelect.disabled, false);
    assert.equal(h.els.btnGen.disabled, true);
    assert.equal(h.els.btnDel.disabled, true);
});

test('Run blocks both NEW and a create dialog opened before the workflow started', async () => {
    const h = setup();
    let buttons;
    Object.assign(h.context, {
        btnNewCostume: h.els.btnNew,
        document: { createElement: control },
        showModal(_title, build, actions) { build(); buttons = actions; },
    });
    vm.runInContext(between(source, 'btnNewCostume.onclick =', 'els.btnNew = btnNewCostume;'), h.context);
    h.els.btnNew.onclick();
    assert.equal(buttons, undefined);
    h.status(0);
    h.els.btnNew.onclick();
    assert.ok(buttons);
    h.emit('execution_start', {});
    assert.equal(await buttons[1].action(), true);
    assert.equal(h.requests.length, 1, 'An open dialog must not create a costume during Run');
});
