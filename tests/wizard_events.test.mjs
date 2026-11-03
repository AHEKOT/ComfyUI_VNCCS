import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
const guards = common.slice(common.indexOf('export function registerCleanup'), common.indexOf('// ── DOM Widget Width Sync'))
    .replaceAll('export ', '');

function harness() {
    const listeners = new Map(), calls = [];
    const context = createWidgetContext({
        console,
        api: {
            addEventListener(name, handler) { listeners.set(name, handler); },
            fetchApi(route, options) {
                return new Promise((resolve, reject) => {
                    calls.push({ route, payload: JSON.parse(options.body), resolve, reject });
                });
            },
        },
    });
    vm.runInContext(guards, context);
    const emit = (payload, status, extra = {}) => listeners.get('vnccs.wizard.stage')({ detail: { ...payload, status, ...extra } });
    assert.ok(listeners.has('vnccs.wizard.stage'), 'Register the event before any local wizard starts');
    return { context, calls, emit };
}

for (const route of ['character_wizard', 'clothes_wizard', 'cloner_auto_generate']) {
    test(`${route}: scoped stages reach the UI only while the HTTP request owns it`, async () => {
        const { context, calls, emit } = harness();
        const node = { id: 753 }, labels = [];
        const current = context.createRequestGuard(node)();
        const job = context.requestWizard(node, `/vnccs/${route}`, { description: 'Costume' }, current, label => labels.push(label));
        const request = calls[0];
        assert.equal(request.route, `/vnccs/${route}`);
        assert.equal(request.payload.node_id, 753);
        assert.ok(request.payload.request_id);
        assert.equal(request.payload.description, 'Costume');
        emit(request.payload, 'queued', { node_id: 'another-node' });
        emit(request.payload, 'queued', { request_id: 'another-request' });
        emit(request.payload, 'unknown');
        emit(request.payload, 'toString');
        assert.deepEqual(labels, []);
        for (const status of ['queued', 'running', 'done']) emit(request.payload, status, { node_id: '753' });
        assert.deepEqual(labels, ['WAITING...', 'THINKING...', 'APPLYING...']);
        const response = { ok: true };
        request.resolve(response);
        assert.equal(await job, response);
        emit(request.payload, 'error');
        assert.equal(labels.length, 3, 'Late completion must not change the next UI state');
    });
}

test('wizard stages cannot cross requests, selections, nodes, or removed widgets', async () => {
    const { context, calls, emit } = harness();
    const node = { id: 753 }, other = { id: 754 }, labels = [];
    const begin = context.createRequestGuard(node);
    const first = context.requestWizard(node, '/vnccs/clothes_wizard', {}, begin(), label => labels.push(['old', label]));
    const second = context.requestWizard(node, '/vnccs/clothes_wizard', {}, begin(), label => labels.push(['new', label]));
    const third = context.requestWizard(other, '/vnccs/clothes_wizard', {}, () => true, label => labels.push(['other', label]));
    assert.equal(new Set(calls.map(call => call.payload.request_id)).size, 3);
    emit(calls[0].payload, 'running');
    emit(calls[1].payload, 'queued');
    emit(calls[2].payload, 'running');
    assert.deepEqual(labels, [['new', 'WAITING...'], ['other', 'THINKING...']]);
    node.onRemoved();
    other.onRemoved();
    for (const call of calls) {
        emit(call.payload, 'error');
        call.resolve({ ok: true });
    }
    await Promise.all([first, second, third]);
    assert.equal(labels.length, 2);
});

test('wizard transport failure releases its stage handler and permits a retry', async () => {
    const { context, calls, emit } = harness();
    const node = { id: 753 }, labels = [];
    const job = context.requestWizard(node, '/vnccs/clothes_wizard', {}, () => true, label => labels.push(label));
    emit(calls[0].payload, 'error');
    assert.deepEqual(labels, ['FAILED']);
    calls[0].reject(new Error('Connection lost'));
    await assert.rejects(job, /Connection lost/);
    emit(calls[0].payload, 'done');
    assert.deepEqual(labels, ['FAILED']);
    const retry = context.requestWizard(node, '/vnccs/clothes_wizard', {}, () => true, label => labels.push(label));
    emit(calls[1].payload, 'queued');
    calls[1].resolve({ ok: true });
    await retry;
    assert.deepEqual(labels, ['FAILED', 'WAITING...']);
});
