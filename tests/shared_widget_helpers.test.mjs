import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

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
