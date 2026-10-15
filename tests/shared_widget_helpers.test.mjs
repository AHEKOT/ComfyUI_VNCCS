import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
const guards = common.slice(common.indexOf('export function registerCleanup'), common.indexOf('// ── DOM Widget Width Sync')).replaceAll('export ', '');
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};
const response = data => ({ ok: true, json: async () => data });

test('shared input helpers retain resolution presets, age bounds and upload contents', async () => {
    const context = createWidgetContext({ File });
    for (const [input, expected] of [[1.3, 1344], [1.5, 1536], [1.7, 1741], [0, 1024], [9, 4096], [Infinity, 4096], ['invalid', 1024]]) {
        assert.equal(context.resolutionScaleValue(input), expected);
    }
    for (const input of [Infinity, -Infinity, 'Infinity', 'invalid']) {
        assert.equal(context.finiteResolutionScaleValue(input), 1024);
    }
    assert.equal(context.resolutionScaleMegapixels(1408), 1.375);
    assert.equal(context.resolutionScaleText(1408), '1.4 MP');
    for (const [input, expected] of [['invalid', 18], [null, 18], [-1, 1], [120, 100], ['18.5', 18.5]]) {
        assert.equal(context.normalizeAgeValue(input), expected);
    }
    const clean = new File(['image'], 'portrait.png');
    assert.equal(context.normalizeUploadFile(clean), clean);
    const original = new File(['image'], 'portrait final.png', { type: 'image/png', lastModified: 123 });
    const normalized = context.normalizeUploadFile(original, 'clone');
    assert.equal(normalized.name, 'clone_portrait_final.png');
    assert.equal(normalized.type, original.type);
    assert.equal(normalized.lastModified, original.lastModified);
    assert.equal(await normalized.text(), await original.text());
    assert.match(context.normalizeUploadFile(new File(['image'], '???'), 'clone').name, /^clone_clone_\d+\.png$/);
});

test('Control Center clients retain scoped caches, polling updates, errors and removal cleanup', async () => {
    const calls = [], timers = new Map(), cache = new Map();
    let config = { complete: true, version: 1 }, status = {}, failed = false, renders = 0, removed = 0;
    const node = { onRemoved() { removed++; } };
    const context = createWidgetContext({
        node,
        localStorage: { getItem: key => cache.get(key), setItem: (key, value) => cache.set(key, value) },
        setInterval(callback, delay) { assert.equal(delay, 2000); timers.set(1, callback); return 1; },
        clearInterval: id => timers.delete(id),
        api: { apiURL: route => `/server-a${route}`, fetchApi: async route => {
            calls.push(route);
            return { ok: !failed, json: async () => failed ? { error: 'Unavailable' } : route.includes('/check?') ? config : status };
        } },
    });
    const state = { config: null, downloadStatus: {} };
    const client = context.createControlCenterClient('public/catalog', value => value?.complete, state, () => renders++);
    assert.equal(await client.fetchConfig(), config);
    assert.equal(JSON.parse([...cache.values()][0]).version, 1);
    const otherState = { config: null, downloadStatus: {} };
    const other = context.createControlCenterClient('public/catalog', value => value?.complete, otherState, () => {});
    assert.equal(await other.fetchConfig(), config);
    assert.equal(calls.length, 1, 'The registry avoids a second network request');
    client.startPolling(); client.startPolling();
    assert.equal(timers.size, 1);
    status = { model: { status: 'downloading' } };
    await timers.get(1)();
    assert.equal(state.downloadStatus, status);
    assert.equal(calls.at(-1), '/vnccs/manager/status?repo_id=public%2Fcatalog');
    assert.equal(otherState.downloadStatus.model, undefined);
    assert.equal(timers.size, 1);
    config = { complete: true, version: 2 }; status = { model: { status: 'completed' } };
    await timers.get(1)();
    assert.equal(timers.size, 0);
    assert.equal(state.config.version, 2);
    assert.match(calls.at(-1), /force_refresh=true/);
    failed = true;
    await assert.rejects(client.fetchConfig(true), /Unavailable/);
    assert.equal(state.config.version, 2);
    assert.ok(renders >= 3);
    client.startPolling();
    context.client = client;
    const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
    const cleanup = common.slice(common.indexOf('export function registerCleanup'), common.indexOf('// Each loader'));
    vm.runInContext(`${cleanup.replace('export ', '')}; registerCleanup(node, client.stopPolling);`, context);
    node.onRemoved();
    assert.equal(timers.size, 0);
    assert.equal(removed, 1);
    failed = false;
    context.api.apiURL = route => `/server-b${route}`;
    const isolated = context.createControlCenterClient('public/catalog', value => value?.complete, { config: null, downloadStatus: {} }, () => {});
    await isolated.fetchConfig();
    assert.equal(cache.size, 2, 'Another server has its own cache');
});

for (const owners of ['one client', 'two clients']) {
    test(`Control Center ${owners} keep the latest forced catalog when responses reverse`, async () => {
        const requests = [], cache = new Map(), node = {};
        const context = createWidgetContext({ node,
            localStorage: { getItem: key => cache.get(key), setItem: (key, value) => cache.set(key, value) },
            api: { fetchApi: () => { const pending = deferred(); requests.push(pending); return pending.promise; } },
        });
        vm.runInContext(guards, context);
        const state = { config: null }, otherState = { config: null };
        const first = context.createControlCenterClient('public/catalog', () => true, state, () => {}, node);
        const second = owners === 'one client' ? first
            : context.createControlCenterClient('public/catalog', () => true, otherState, () => {}, {});
        const older = first.fetchConfig(true), newer = second.fetchConfig(true);
        requests[1].resolve(response({ version: 2 }));
        await newer;
        requests[0].resolve(response({ version: 1 }));
        await older;
        assert.equal(state.config.version, 2);
        if (owners === 'two clients') assert.equal(otherState.config.version, 2);
        assert.equal(JSON.parse([...cache.values()][0]).version, 2);
        const observer = context.createControlCenterClient('public/catalog', () => true, { config: null }, () => {});
        assert.equal((await observer.fetchConfig()).version, 2);
    });
}

test('a completed Control Center status cannot be reverted by an overlapping older poll', async () => {
    const requests = [], timers = new Map(), node = {};
    const context = createWidgetContext({ node,
        setInterval(callback) { timers.set(1, callback); return 1; }, clearInterval: id => timers.delete(id),
        api: { fetchApi: route => {
            if (route.includes('/check?')) return Promise.resolve(response({ complete: true }));
            const pending = deferred(); requests.push(pending); return pending.promise;
        } },
    });
    vm.runInContext(guards, context);
    const state = { config: null, downloadStatus: {} };
    const client = context.createControlCenterClient('public/catalog', () => true, state, () => {}, node);
    client.startPolling();
    const callback = timers.get(1), first = callback(), second = callback();
    const count = requests.length;
    requests[0].resolve(response({ model: { status: 'completed' } }));
    await first;
    if (requests[1]) requests[1].resolve(response({ model: { status: 'downloading' } }));
    await second;
    assert.equal(count, 1, 'Only one status read may be in flight');
    assert.equal(state.downloadStatus.model.status, 'completed');
    assert.equal(timers.size, 0);
});

for (const operation of ['config', 'status']) {
    test(`removed Control Center client discards its pending ${operation} and cannot resume polling`, async () => {
        const pending = deferred(), timers = new Map(), cache = new Map(), node = {};
        let rendered = 0;
        const context = createWidgetContext({ node,
            localStorage: { getItem: key => cache.get(key), setItem: (key, value) => cache.set(key, value) },
            setInterval(callback) { timers.set(1, callback); return 1; }, clearInterval: id => timers.delete(id),
            api: { fetchApi: () => pending.promise },
        });
        vm.runInContext(guards, context);
        const state = { config: null, downloadStatus: {} }, original = JSON.stringify(state);
        const client = context.createControlCenterClient('public/catalog', () => true, state, () => rendered++, node);
        context.client = client;
        vm.runInContext('registerCleanup(node, client.stopPolling);', context);
        client.startPolling();
        const work = operation === 'config' ? client.fetchConfig(true) : timers.get(1)();
        node.onRemoved();
        client.startPolling();
        pending.resolve(response(operation === 'config' ? { version: 1 } : { model: { status: 'downloading' } }));
        await work;
        assert.equal(timers.size, 0);
        assert.equal(JSON.stringify(state), original);
        assert.equal(rendered, 0);
        assert.equal(cache.size, 0);
    });
}

for (const ok of [true, false]) {
    test(`Creator download ${ok ? 'success' : 'failure'} cannot change or poll after removal`, async () => {
        const source = readFileSync(new URL('../web/vnccs_character_creator_v2.js', import.meta.url), 'utf8');
        const start = source.indexOf('const { fetchConfig: fetchCcConfig'), end = source.indexOf('const els =', start);
        const pending = deferred(), timers = new Map(), node = {}, ccState = { downloadStatus: {} };
        let rendered = 0;
        const context = createWidgetContext({ node, ccState, CC_REPO_ID: 'public/catalog',
            ccHasRequiredFamilies: () => true, ccStatusKey: () => 'model', renderControlCenterCards: () => rendered++,
            setInterval(callback) { timers.set(1, callback); return 1; }, clearInterval: id => timers.delete(id),
            api: { fetchApi: () => pending.promise },
        });
        vm.runInContext(`${guards}
            ${source.slice(start, end)}
            registerCleanup(node, () => stopCcPolling()); globalThis.download = ccDownloadEntry;`, context);
        const work = context.download('models', { name: 'Model' });
        node.onRemoved();
        const original = JSON.stringify(ccState);
        pending.resolve({ ok, json: async () => ok ? { status: 'started' } : { error: 'Old download failed' } });
        await work;
        assert.equal(timers.size, 0);
        assert.equal(rendered, 1);
        assert.equal(JSON.stringify(ccState), original);
    });
}

test('a previous empty status cannot stop polling after a new Creator download starts', async () => {
    const source = readFileSync(new URL('../web/vnccs_character_creator_v2.js', import.meta.url), 'utf8');
    const start = source.indexOf('const { fetchConfig: fetchCcConfig'), end = source.indexOf('const els =', start);
    const oldStatus = deferred(), timers = new Map(), node = {}, ccState = { downloadStatus: {} };
    let nextStatus = { model: { status: 'queued' } }, reads = 0;
    const context = createWidgetContext({ node, ccState, CC_REPO_ID: 'public/catalog',
        ccHasRequiredFamilies: () => true, ccStatusKey: () => 'model', renderControlCenterCards() {},
        setInterval(callback) { timers.set(1, callback); return 1; }, clearInterval: id => timers.delete(id),
        api: { fetchApi: async route => {
            if (route === '/vnccs/manager/status?repo_id=public%2Fcatalog') return ++reads === 1 ? oldStatus.promise : response(nextStatus);
            return response(route.includes('/check?') ? { complete: true } : { status: 'started' });
        } },
    });
    vm.runInContext(`${guards}
        ${source.slice(start, end)}
        globalThis.download = ccDownloadEntry; globalThis.start = startCcPolling;`, context);
    context.start();
    const older = timers.get(1)();
    await context.download('models', { name: 'New model' });
    oldStatus.resolve(response({}));
    await older;
    assert.equal(timers.size, 1, 'The newly queued download must keep its poller');
    assert.equal(ccState.downloadStatus.model.status, 'queued');
    await timers.get(1)();
    assert.equal(reads, 2);
    nextStatus = { model: { status: 'completed' } };
    await timers.get(1)();
    assert.equal(timers.size, 0);
    assert.equal(ccState.downloadStatus.model.status, 'completed');
});

for (const initiator of ['creator client', 'Control Center widget']) {
    for (const scenario of ['newer refresh', 'removed initiator']) {
        test(`${initiator} and Control Center share catalog waiters through ${scenario}`, async () => {
            const pending = [], clientNode = {}, widgetNode = {};
            const context = createWidgetContext({
                window: { dispatchEvent() {} },
                CustomEvent: class { constructor(type, data) { this.type = type; this.detail = data.detail; } },
                api: { fetchApi: () => { const request = deferred(); pending.push(request); return request.promise; } },
            });
            const control = readFileSync(new URL('../web/vnccs_control_center.js', import.meta.url), 'utf8');
            vm.runInContext(guards + control.slice(control.indexOf('class VNCCSControlCenterWidget'))
                + '\nthis.Widget = VNCCSControlCenterWidget;', context);
            const state = { config: null, downloadStatus: {} };
            const client = context.createControlCenterClient('public/catalog', () => true, state, () => {}, clientNode);
            const widget = Object.create(context.Widget.prototype);
            Object.assign(widget, { node: widgetNode, state: {}, config: null, statusText: {},
                _beginConfigRequest: context.createRequestGuard(widgetNode),
                _syncCustomModelInput() {}, _renderAll() {}, _dispatchLoraOptions() {}, _syncCnetSlots() {},
                _setConfigLoading() {},
            });
            const first = initiator === 'creator client' ? client : widget;
            const second = initiator === 'creator client' ? widget : client;
            const request = (owner, force) => owner === client ? client.fetchConfig(force) : widget.fetchConfig('public/catalog', force);
            const older = request(first, false), newer = request(second, scenario === 'newer refresh');
            if (scenario === 'removed initiator') {
                (initiator === 'creator client' ? clientNode : widgetNode).onRemoved();
                assert.equal(pending.length, 1, 'A normal waiter must share the existing HTTP read');
                pending[0].resolve(response({ name: 'Shared catalog', version: 2 }));
            } else {
                assert.equal(pending.length, 2);
                pending[1].resolve(response({ name: 'New catalog', version: 2 }));
                await newer;
                pending[0].resolve(response({ name: 'Old catalog', version: 1 }));
            }
            await Promise.all([older, newer]);
            const result = second === client ? state.config : widget.config;
            assert.equal(result.version, 2);
            if (scenario === 'newer refresh') assert.equal((first === client ? state.config : widget.config).version, 2);
            else assert.equal((first === client ? state.config : widget.config), null);
        });
    }
}

test('a failed status read releases the poll guard for a later successful read', async () => {
    const timers = new Map(), node = {};
    let failed = true, reads = 0;
    const context = createWidgetContext({
        setInterval(callback) { timers.set(1, callback); return 1; }, clearInterval: id => timers.delete(id),
        api: { fetchApi: async () => { reads++; if (failed) throw new Error('Offline'); return response({ model: { status: 'downloading' } }); } },
    });
    vm.runInContext(guards, context);
    const state = { config: null, downloadStatus: {} };
    const client = context.createControlCenterClient('public/catalog', () => true, state, () => {}, node);
    client.startPolling();
    await timers.get(1)();
    assert.deepEqual(state.downloadStatus, {});
    failed = false;
    await timers.get(1)();
    assert.equal(reads, 2);
    assert.equal(state.downloadStatus.model.status, 'downloading');
    node.onRemoved();
    assert.equal(timers.size, 0);
});
