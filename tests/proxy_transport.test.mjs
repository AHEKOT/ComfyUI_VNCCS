import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { createWidgetContext } from './widget_context.mjs';

const read = name => readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8')
    .replace(/new URL\(([^,]+), import\.meta\.url\)\.href/g, '$1');
const run = (context, code) => vm.runInContext(code, context);
const response = (status, data) => ({ status, ok: status >= 200 && status < 300, json: async () => data });

function transport(options = {}) {
    const calls = [], values = new Map();
    const api = {
        user: 'alice',
        base: '/comfy/api',
        apiURL(route) { return this.base + route.replace(/^\/api(?=\/)/, ''); },
        fetchApi(route, init) { calls.push({ route, init, owner: this }); return Promise.resolve(response(200, { ok: true })); },
    };
    const localStorage = { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
    const context = createWidgetContext({ api, localStorage, ...options });
    return { context, calls, values, api, localStorage };
}

test('transport preserves the Comfy API receiver, custom headers and multipart uploads', async () => {
    const h = transport();
    await run(h.context, 'vnccsApi.fetchApi("/vnccs/save_costume", { method: "POST", headers: { "X-Test": "yes" }, body: "{}" })');
    const { init, route, owner } = h.calls[0];
    assert.equal(owner, h.api);
    assert.equal(route, '/vnccs/save_costume');
    assert.equal(init.cache, 'no-store');
    const headers = new Headers(init.headers);
    assert.equal(headers.get('X-Test'), 'yes');
    assert.equal(headers.get('X-VNCCS-CSRF'), '1');
    assert.equal(headers.get('Content-Type'), 'application/json');
    await run(h.context, 'vnccsApi.fetchApi("/upload/image", { method: "POST", body: { multipart: true } })');
    assert.equal(new Headers(h.calls[1].init.headers).has('Content-Type'), false);
});

test('VNCCS worker requests disable ComfyUI response deadlines', async () => {
    const h = transport();
    for (const route of [
        '/vnccs/preview_generate', '/vnccs/cloner_auto_generate',
        '/vnccs/control_center/clothes_preview', '/vnccs/character_styles/preview',
        '/vnccs/character_wizard', '/vnccs/clothes_wizard',
        '/vnccs/character_generator/regenerate',
    ]) {
        h.context.route = route;
        await run(h.context, 'vnccsApi.fetchApi(route, { method: "POST", body: "{}" })');
        assert.equal(h.calls.at(-1).init.timeoutMs, null, route);
    }
});

test('slow preview requests remain pending past ComfyUI default deadline and return their result', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const h = transport();
    let complete;
    let outcome = null;
    let requests = 0;
    h.api.fetchApi = (_route, { timeoutMs = 60_000 }) => new Promise((resolve, reject) => {
        requests++;
        complete = resolve;
        if (timeoutMs !== null) setTimeout(() => reject(new DOMException('Fetch timeout', 'TimeoutError')), timeoutMs);
    });
    const work = run(h.context, 'vnccsApi.fetchApi("/vnccs/preview_generate", { method: "POST", body: "{}" })')
        .then(result => { outcome = result; }, error => { outcome = error; });
    t.mock.timers.tick(120_000);
    await Promise.resolve();
    assert.equal(outcome, null, 'Loading models and generation must not be cut off by an HTTP response timer');
    const result = response(200, { image: 'finished-preview' });
    complete(result);
    await work;
    assert.equal(outcome, result);
    assert.equal(requests, 1, 'A long generation must not submit duplicate jobs');
});

test('transport retains metadata deadlines, explicit timeouts and caller cancellation', async () => {
    const h = transport();
    await run(h.context, 'vnccsApi.fetchApi("/vnccs/context_lists")');
    assert.equal(h.calls.at(-1).init.timeoutMs, undefined);
    await run(h.context, 'vnccsApi.fetchApi("/upload/image", { method: "POST" })');
    assert.equal(h.calls.at(-1).init.timeoutMs, undefined);
    const controller = new AbortController();
    h.context.signal = controller.signal;
    await run(h.context, 'vnccsApi.fetchApi("/vnccs/preview_generate", { method: "POST", timeoutMs: 250, signal })');
    assert.equal(h.calls.at(-1).init.timeoutMs, 250);
    assert.equal(h.calls.at(-1).init.signal, controller.signal);
    controller.abort();
    assert.equal(h.calls.at(-1).init.signal.aborted, true);
});

test('media URLs preserve the proxy prefix without changing data and already resolved URLs', () => {
    const { context } = transport();
    for (const [input, expected] of [
        ['/view?filename=a.png', '/comfy/api/view?filename=a.png'],
        ['/vnccs/get_preview?character=A', '/comfy/api/vnccs/get_preview?character=A'],
        ['/comfy/api/view?filename=a.png', '/comfy/api/view?filename=a.png'],
        ['data:image/png;base64,AAAA', 'data:image/png;base64,AAAA'],
        ['blob:preview', 'blob:preview'],
    ]) {
        context.input = input;
        assert.equal(run(context, 'mediaURL(input)'), expected);
    }
});

test('checked writes reject proxy errors, invalid JSON and application errors', async () => {
    const { context, api } = transport();
    for (const status of [403, 413, 502]) {
        api.fetchApi = async () => response(status, { error: `failure ${status}` });
        await assert.rejects(run(context, 'checkedJSON("/vnccs/create", { method: "POST", body: "{}" })'), new RegExp(`failure ${status}`));
    }
    api.fetchApi = async () => response(200, { error: 'Disk full' });
    await assert.rejects(run(context, 'checkedJSON("/vnccs/save_costume")'), /Disk full/);
    api.fetchApi = async () => ({ ok: true, status: 200, json: async () => { throw new Error('HTML login page'); } });
    await assert.rejects(run(context, 'checkedJSON("/vnccs/create")'), /Invalid server response/);
});

test('preferences and in-memory catalogs are isolated by backend path and Comfy user', () => {
    const { context, api } = transport();
    run(context, 'storage.setItem("prefs", "A"); serverRegistry("models").repo = "A"');
    api.base = '/second/api';
    assert.equal(run(context, 'storage.getItem("prefs")'), null);
    assert.equal(run(context, 'serverRegistry("models").repo'), undefined);
    api.base = '/comfy/api'; api.user = 'bob';
    assert.equal(run(context, 'storage.getItem("prefs")'), null);
    api.user = 'alice';
    assert.equal(run(context, 'storage.getItem("prefs")'), 'A');
    assert.equal(run(context, 'serverRegistry("models").repo'), 'A');
});

test('storage denial and quota exhaustion do not escape optional cache operations', () => {
    const { context } = transport({ localStorage: new Proxy({}, { get() { throw new Error('Storage denied'); } }) });
    assert.equal(run(context, 'storage.getItem("key")'), null);
    assert.equal(run(context, 'storage.setItem("key", "value")'), false);
    assert.doesNotThrow(() => run(context, 'storage.removeItem("key")'));
});

function generator() {
    const h = transport();
    h.context.app = { registerExtension(extension) { h.extension = extension; } };
    run(h.context, read('vnccs_character_generator').replace(/^import .*;\n/gm, '') + '\nthis.Widget = CharacterGeneratorWidget;');
    const node = { id: 5, type: 'VNCCS_CharacterGenerator', graph: { extra: {} } };
    const widget = Object.create(h.context.Widget.prototype);
    Object.assign(widget, {
        node, data: { pose_generation: { target_size: 1536 }, ui: {} }, stages: [['pose_generation'], ['bg_remove']],
        stageState: { pose_generation: { status: 'running' }, bg_remove: { status: 'waiting' } },
        renderPreview() {}, renderChain() {}, updateRegenerateProgress() {}, saveBrowserState() {},
        syncCharacterSourceData() {}, syncStagesFromData() {}, syncModelResolution() {},
        finishRegenerate() { this.finished = true; },
    });
    h.context.widget = widget;
    return { ...h, widget, node };
}

test('generator UI cache restores view preferences without importing cached results or settings', () => {
    const h = generator();
    const firstKey = h.widget.storageKey();
    const saved = { version: 2, settings: JSON.stringify(h.widget.data), selectedPreview: 'bg_remove', stageState: {
        pose_generation: { status: 'waiting', message: 'Stopped', current: 3, total: 4, images: ['/view?old-1', '/view?old-2'] },
        bg_remove: { status: 'done', images: ['/view?a'] },
    } };
    h.widget.stageState.pose_generation = { status: 'waiting', images: null };
    h.context.saved = saved;
    run(h.context, 'storage.setItem(widget.storageKey(), JSON.stringify(saved))');
    h.widget.restoreBrowserState();
    assert.equal(h.widget.selectedPreview, 'bg_remove');
    assert.equal(h.widget.stageState.bg_remove.status, 'waiting');
    assert.equal(h.widget.stageState.pose_generation.images, null);
    assert.equal(h.widget.stageState.pose_generation.total, undefined);
    const before = JSON.stringify(h.widget.data);
    saved.settings = '{}'; saved.data = { pose_generation: { target_size: 8192 } };
    run(h.context, 'storage.setItem(widget.storageKey(), JSON.stringify(saved))');
    h.widget.restoreBrowserState();
    assert.equal(JSON.stringify(h.widget.data), before);
    h.node.graph = { extra: {} };
    assert.notEqual(h.widget.storageKey(), firstKey);
    h.widget.stageState.bg_remove = { status: 'waiting' };
    h.widget.restoreBrowserState();
    assert.equal(h.widget.stageState.bg_remove.status, 'waiting');
});

test('fresh server has no cards even when the browser retains an interrupted four-card batch', async () => {
    const h = generator();
    h.widget.stageState.pose_generation = { status: 'waiting', images: null };
    h.context.saved = { version: 2, settings: JSON.stringify(h.widget.data), stageState: {
        pose_generation: { status: 'waiting', images: ['/view?old-1', '/view?old-2'], current: 3, total: 4, message: 'Stopped' },
    } };
    run(h.context, 'storage.setItem(widget.storageKey(), JSON.stringify(saved))');
    h.widget.restoreBrowserState();
    h.api.fetchApi = async () => response(200, { snapshot: null });
    await h.widget.refreshProgress();
    assert.equal(h.widget.stageState.pose_generation.status, 'waiting');
    assert.equal(h.widget.stageState.pose_generation.images, null);
    assert.equal(h.widget.stageState.pose_generation.total, undefined);
});

test('browser UI cache stores no transient stage results or image payloads', () => {
    const h = generator();
    h.widget.serializableViewerState = () => ({ open: false });
    h.widget.stageState.pose_generation = { status: 'done', images: ['data:image/png;base64,AAAA'], total: 4 };
    h.context.Widget.prototype.saveBrowserState.call(h.widget);
    const saved = JSON.parse(run(h.context, 'storage.getItem(widget.storageKey())'));
    assert.equal(saved.settings, JSON.stringify(h.widget.data));
    assert.equal(saved.stageState, undefined);
    assert.deepEqual(saved.viewer, { open: false });
});

test('current server snapshot restores results and the cached viewer after reload', async () => {
    const h = generator();
    h.widget.stageState.pose_generation = { status: 'waiting', images: null };
    h.widget.selectedPreview = 'pose_generation';
    h.widget.userSelectedPreview = false;
    const viewer = { open: true, stage: 'pose_generation', index: 1, centerNormX: 0.4, centerNormY: 0.6 };
    h.widget.restoredViewer = viewer;
    h.widget.openViewer = (index, restored) => { h.widget.viewer = { open: true, index, restored }; };
    const scope = h.widget.progressScope();
    h.api.fetchApi = async () => response(200, { snapshot: {
        scope, node_id: '5', epoch: 'current-server', revision: 1, run_id: 'current-run', stages: {
            pose_generation: { status: 'done', images: ['/view?new-1', '/view?new-2'], total: 2, preview_aspect: 2 / 3 },
        },
    } });
    await h.widget.refreshProgress();
    assert.equal(h.widget.stageState.pose_generation.preview_aspect, 2 / 3);
    assert.deepEqual(Array.from(h.widget.stageState.pose_generation.images), ['/view?new-1', '/view?new-2']);
    assert.equal(h.widget.viewer?.index, 1);
    assert.equal(h.widget.viewer?.restored, viewer);
});

test('character selection changing during a progress request rejects its late snapshot', async () => {
    const h = generator();
    h.widget.data.character_name = 'Alice';
    let resolve;
    h.api.fetchApi = () => new Promise(done => { resolve = done; });
    const pending = h.widget.refreshProgress();
    h.widget.syncSourceCharacter('Bob');
    resolve(response(200, { snapshot: { scope: h.widget.progressScope(), node_id: '5', epoch: 'server', revision: 1,
        stages: { pose_generation: { status: 'done', images: ['Alice-preview'] } } } }));
    await pending;
    assert.equal(h.widget.data.character_name, 'Bob');
    assert.equal(h.widget.stageState.pose_generation.images, null);
});

test('progress snapshots restore missed stages but cannot replace a newer event', async () => {
    const h = generator();
    const scope = h.widget.progressScope();
    let resolve;
    h.api.fetchApi = () => new Promise(done => { resolve = done; });
    const work = h.widget.refreshProgress();
    resolve(response(200, { snapshot: { scope, node_id: '5', revision: 10, stages: { pose_generation: { status: 'done', images: ['/view?x'] } } } }));
    await work;
    assert.equal(h.widget.stageState.pose_generation.status, 'done');
    const stale = h.widget.refreshProgress();
    h.widget._progressRevision = 12;
    h.widget.stageState.pose_generation = { status: 'running' };
    resolve(response(200, { snapshot: { scope, node_id: '5', revision: 11, stages: {} } }));
    await stale;
    assert.equal(h.widget.stageState.pose_generation.status, 'running');
    const disposed = h.widget.refreshProgress(); h.widget._disposed = true;
    resolve(response(200, { snapshot: null })); await disposed;
    assert.equal(h.widget.stageState.pose_generation.status, 'running');
});

test('missing server progress releases stale running UI without starting another job', async () => {
    const h = generator();
    h.api.fetchApi = async () => response(200, { snapshot: null });
    await h.widget.refreshProgress();
    assert.equal(h.widget.stageState.pose_generation.status, 'error');
    assert.equal(h.widget.finished, undefined);
});

test('active workflow sources use the shared transport and checked create/save operations', () => {
    for (const name of ['vnccs_character_creator_v2', 'vnccs_character_cloner', 'vnccs_clothes_designer', 'vnccs_emotion_v2', 'vnccs_character_generator', 'vnccs_control_center']) {
        const text = read(name);
        assert.match(text, /vnccsApi as api/);
        assert.doesNotMatch(text, /(?<![\w.])fetch\(/);
        assert.doesNotMatch(text, /localStorage\./);
    }
    for (const name of ['vnccs_character_creator_v2', 'vnccs_character_cloner']) assert.match(read(name), /checkedJSON\("\/vnccs\/create", \{ method: "POST"/);
    assert.match(read('vnccs_clothes_designer'), /checkedJSON\("\/vnccs\/save_costume"/);
    assert.doesNotMatch(read('vnccs_character_creator_v2'), /applyStoredPrefs\(true\)/);
});

test('workflow cache identity also works outside secure browser contexts', () => {
    const { context } = transport({ crypto: undefined });
    context.node = { id: 1, type: 'generator', graph: { extra: {} } };
    const first = run(context, 'workflowScope(node)');
    assert.equal(run(context, 'workflowScope(node)'), first);
    context.node.graph = { extra: {} };
    assert.notEqual(run(context, 'workflowScope(node)'), first);
});

test('independent tabs isolate saved workflows while reconnect and reload retain their scope', () => {
    const values = new Map();
    const sessionStorage = { getItem: key => values.get(key) || null,
        setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
    const create = (type, store = sessionStorage) => {
        const h = transport({ sessionStorage: store,
            performance: { getEntriesByType: () => [{ type }] } });
        h.context.node = { id: 42, type: 'VNCCS_CharacterGenerator',
            graph: { extra: { vnccs_workflow_id: 'saved-workflow' } } };
        return h;
    };
    const first = create('navigate');
    const firstScope = run(first.context, 'workflowScope(node)');
    const copiedValues = new Map(values);
    const copiedStorage = { getItem: key => copiedValues.get(key) || null,
        setItem: (key, value) => copiedValues.set(key, value), removeItem: key => copiedValues.delete(key) };
    const duplicate = create('navigate', copiedStorage);
    assert.notEqual(run(duplicate.context, 'workflowScope(node)'), firstScope);
    assert.equal(run(first.context, 'workflowScope(node)'), firstScope);
    assert.equal(run(create('reload').context, 'workflowScope(node)'), firstScope);
    assert.equal(first.context.node.graph.extra.vnccs_workflow_id, 'saved-workflow');
});

test('a restarted server can restore snapshots with a lower revision from a new epoch', async () => {
    const h = generator();
    const scope = h.widget.progressScope();
    h.widget._progressScope = scope; h.widget._progressRevision = 100; h.widget._progressEpoch = 'old-server';
    h.api.fetchApi = async () => response(200, { snapshot: { scope, node_id: '5', revision: 2, epoch: 'new-server', stages: { pose_generation: { status: 'done' } } } });
    await h.widget.refreshProgress();
    assert.equal(h.widget.stageState.pose_generation.status, 'done');
    assert.equal(h.widget._progressEpoch, 'new-server');
});

test('a queued regeneration is not cancelled while the preview worker is still busy', async () => {
    const h = generator(); h.widget._regenerateRequestPending = true;
    h.api.fetchApi = async () => response(200, { snapshot: null });
    await h.widget.refreshProgress();
    assert.equal(h.widget.stageState.pose_generation.status, 'running');
    assert.equal(h.widget.finished, undefined);
});

test('connection listeners serialize refreshes and are removed with the widget', async () => {
    const events = new Map(), windowEvents = new Map(), documentEvents = new Map(), cleanup = [];
    const target = handlers => ({ addEventListener: (key, fn) => handlers.set(key, fn), removeEventListener: key => handlers.delete(key) });
    let finish, calls = 0;
    const { context } = transport({ api: target(events), window: target(windowEvents), document: { ...target(documentEvents), visibilityState: 'visible' } });
    context.refresh = () => { calls++; return new Promise(done => { finish = done; }); };
    context.cleanup = (_, fn) => cleanup.push(fn);
    run(context, 'watchConnection({}, refresh, cleanup)');
    events.get('reconnected')(); windowEvents.get('focus')();
    assert.equal(calls, 1);
    finish(); await new Promise(done => setImmediate(done));
    documentEvents.get('visibilitychange')(); assert.equal(calls, 2);
    const late = windowEvents.get('online'); cleanup[0](); late();
    assert.equal(calls, 2);
    assert.equal(events.size + windowEvents.size + documentEvents.size, 0);
    finish();
});

test('media resolution is idempotent when the mount overlaps API route names', () => {
    for (const base of ['', '/comfy', '/api', '/vnccs']) {
        const { context } = transport({ api: {
            apiURL: route => base + (route.startsWith('/api/') ? route : '/api' + route),
        } });
        for (const route of ['/vnccs/get_preview?character=A', '/view?filename=a.png', '/api/vnccs/get_preview?character=A']) {
            context.route = route;
            const once = run(context, 'mediaURL(route)');
            context.once = once;
            assert.equal(run(context, 'mediaURL(once)'), once);
            assert.equal(once, base + (route.startsWith('/api/') ? route : '/api' + route));
        }
    }
});

test('focus refresh preserves the same selected preview and ignores failed or stale images', () => {
    const images = [];
    const { context } = transport({
        URL, window: { location: { href: 'https://comfy.example/comfy/' } },
        Image: class { constructor() { images.push(this); } },
    });
    for (const source of ['/comfy/api/vnccs/get_character_pose_preview?character=A&index=3', '/comfy/api/vnccs/get_preview?character=A&costume=Red&index=2']) {
        const img = { src: source, isConnected: true, getAttribute() { return this.src; } };
        context.img = img;
        run(context, 'refreshPreviewImage(img)');
        const loader = images.at(-1);
        const url = new URL(loader.src);
        assert.equal(url.pathname, new URL(source, 'https://comfy.example').pathname);
        assert.equal(url.searchParams.get('index'), new URL(source, 'https://comfy.example').searchParams.get('index'));
        // A failed refresh has no fallback and leaves both source and selection intact.
        loader.onerror?.();
        assert.equal(img.src, source);
        loader.onload();
        assert.equal(img.src, loader.src);
        run(context, 'refreshPreviewImage(img)');
        img.src = '/new-selection'; images.at(-1).onload();
        assert.equal(img.src, '/new-selection');
        img.src = source; run(context, 'refreshPreviewImage(img)');
        img.isConnected = false; images.at(-1).onload();
        assert.equal(img.src, source);
    }
    for (const name of ['vnccs_character_creator_v2', 'vnccs_clothes_designer']) {
        assert.match(read(name), /watchConnection\(node, [\s\S]*?refreshPreviewImage\(els.previewImg\)/);
    }
});

test('an old completed snapshot cannot finish a new queued regeneration', async () => {
    const h = generator();
    const scope = h.widget.progressScope();
    h.widget.data.ui.progress_request_id = 'new-request';
    h.widget._regenerateRequestPending = 'new-request';
    h.api.fetchApi = async () => response(200, { snapshot: {
        scope, node_id: '5', revision: 99, request_id: 'old-request',
        stages: { pose_generation: { status: 'done' }, bg_remove: { status: 'done' } },
    } });
    await h.widget.refreshProgress();
    assert.equal(h.widget.stageState.pose_generation.status, 'running');
    assert.equal(h.widget.finished, undefined);
});

test('a refresh begun before a new request cannot apply to that request', async () => {
    const h = generator();
    let resolve;
    h.api.fetchApi = () => new Promise(done => { resolve = done; });
    const refresh = h.widget.refreshProgress();
    h.widget.data.ui.progress_request_id = 'new-request';
    resolve(response(200, { snapshot: null }));
    await refresh;
    assert.equal(h.widget.finished, undefined);
    assert.equal(h.widget.stageState.pose_generation.status, 'running');
});

test('superseded regenerate responses cannot clear the newer request state', async () => {
    const h = generator(), pending = [];
    h.node.widgets = [{ name: 'widget_data', value: '{}' }];
    Object.assign(h.widget, { syncCharacterSourceData() {}, syncModelResolution() {}, syncStagesFromData() {} });
    h.api.fetchApi = () => new Promise(done => pending.push(done));
    const first = h.widget.regenerateFrom('pose_generation');
    const second = h.widget.regenerateFrom('bg_remove');
    const currentId = h.widget.data.ui.progress_request_id;
    pending[0](response(200, { ok: true })); await first;
    assert.equal(h.widget.finished, undefined);
    assert.equal(h.widget._regenerateRequestPending, currentId);
    assert.equal(h.widget.data.regenerate_from, 'bg_remove');
    pending[1](response(200, { ok: true })); await second;
    assert.equal(h.widget.finished, true);
    assert.equal(h.widget._regenerateRequestPending, false);
    assert.equal(h.widget.data.regenerate_from, undefined);
});


async function queueHarness() {
    const h = generator();
    h.context.registerCleanup = () => {};
    h.context.window = { addEventListener() {}, removeEventListener() {} };
    h.widget.bindEvents();
    h.widget.validateNativeSeedvr = () => true;
    h.node._vnccsCharacterGeneratorWidget = h.widget;
    h.context.app.graph = { _nodes: [h.node] };
    h.submission = { run: async () => true };
    h.context.app.queuePrompt = async () => h.submission.run();
    const source = read('vnccs_character_generator');
    const start = source.indexOf('this.node._vnccsCharacterGeneratorSyncBeforeQueue = () => {');
    const end = source.indexOf('registerCleanup(this.node, () => delete this.node._vnccsCharacterGeneratorSyncBeforeQueue)', start);
    run(h.context, '(function(){' + source.slice(start, end) + '}).call(widget)');
    await h.extension.setup();
    h.stage = (requestId, revision, status, runId) => h.widget.onStage({ detail: {
        node_id: '5', scope: h.widget.progressScope(), request_id: requestId,
        epoch: 'server', revision, stage: 'pose_generation', status, run_id: runId,
    } });
    return h;
}

test('rejected queue validation preserves active identity across every generator node', async () => {
    const h = await queueHarness();
    await h.context.app.queuePrompt();
    const first = h.widget.data.ui.progress_request_id;
    h.stage(first, 1, 'running');
    h.widget.validateNativeSeedvr = () => false;
    await h.context.app.queuePrompt();
    assert.equal(h.widget.data.ui.progress_request_id, first);
    h.widget.validateNativeSeedvr = () => true;
    h.context.app.graph._nodes.push({ _vnccsCharacterGeneratorSyncBeforeQueue: () => false });
    await h.context.app.queuePrompt();
    assert.equal(h.widget.data.ui.progress_request_id, first);
    h.stage(first, 2, 'done');
    assert.equal(h.widget.stageState.pose_generation.status, 'done');
});

test('multiple queued runs and a failed submission keep following the active execution', async () => {
    const h = await queueHarness();
    await h.context.app.queuePrompt();
    const first = 'server-run-one';
    h.stage(undefined, 1, 'running', first);
    const serialized = JSON.stringify(h.widget.data);
    await h.context.app.queuePrompt();
    const second = 'server-run-two';
    assert.equal(JSON.stringify(h.widget.data), serialized);
    assert.equal(h.widget.data.ui.progress_request_id, undefined);
    assert.equal(h.widget._activeProgressRunId, first);
    h.stage(undefined, 2, 'done', first);
    assert.equal(h.widget.stageState.pose_generation.status, 'done');
    h.stage(undefined, 3, 'running', second);
    assert.equal(h.widget._activeProgressRunId, second);
    // A transport/submission failure after preparing a later request cannot
    // change the active server execution that events and snapshots describe.
    h.submission.run = async () => { throw new Error('Proxy unavailable'); };
    await assert.rejects(h.context.app.queuePrompt(), /Proxy unavailable/);
    h.stage(undefined, 4, 'done', second);
    assert.equal(h.widget.stageState.pose_generation.status, 'done');
    h.api.fetchApi = async () => response(200, { snapshot: {
        scope: h.widget.progressScope(), node_id: '5', run_id: second,
        epoch: 'server', revision: 5, stages: { pose_generation: { status: 'done', images: ['last'] } },
    } });
    await h.widget.refreshProgress();
    assert.deepEqual(h.widget.stageState.pose_generation.images, ['last']);
});

test('snapshot recovery follows the latest stage while preserving explicit preview selection', async () => {
    for (const userSelectedPreview of [false, true]) {
        const h = generator();
        h.widget.selectedPreview = 'pose_generation';
        h.widget.userSelectedPreview = userSelectedPreview;
        h.api.fetchApi = async () => response(200, { snapshot: {
            scope: h.widget.progressScope(), node_id: '5', revision: 1, epoch: 'server',
            stages: { pose_generation: { status: 'done', images: ['first'] }, bg_remove: { status: 'done', images: ['final'] } },
        } });
        await h.widget.refreshProgress();
        assert.equal(h.widget.selectedPreview, userSelectedPreview ? 'pose_generation' : 'bg_remove');
    }
});

test('Clothes snapshot follows the source upscaler without exposing source poses as results', async () => {
    const h = generator();
    Object.assign(h.widget, {
        isClothes: true,
        stages: [['source_upscaler'], ['pose_generation'], ['upscaler']],
        stageState: {
            source_upscaler: { status: 'waiting' },
            pose_generation: { status: 'waiting' },
            upscaler: { status: 'waiting' },
        },
        selectedPreview: 'pose_generation',
        userSelectedPreview: false,
    });
    h.api.fetchApi = async () => response(200, { snapshot: {
        scope: h.widget.progressScope(), node_id: '5', revision: 1, epoch: 'server',
        stages: {
            source_upscaler: { status: 'running', images: ['source'] },
            pose_generation: { status: 'waiting', current: 0, total: 2 },
        },
    } });
    await h.widget.refreshProgress();
    assert.equal(h.widget.selectedPreview, 'source_upscaler');
    assert.equal(h.widget.stageState.pose_generation.images, undefined);
});

test('unchanged or absent idle snapshots do not rebuild interactive preview DOM', async () => {
    const h = generator();
    let preview = 0, chain = 0, viewer = 0;
    Object.assign(h.widget, {
        viewer: { open: true }, renderPreview() { preview++; }, renderChain() { chain++; },
        syncViewerImage() { viewer++; }, stageState: { pose_generation: { status: 'waiting' } },
    });
    h.api.fetchApi = async () => response(200, { snapshot: null });
    await h.widget.refreshProgress(); await h.widget.refreshProgress();
    assert.deepEqual([preview, chain, viewer], [0, 0, 0]);
    h.api.fetchApi = async () => response(200, { snapshot: {
        scope: h.widget.progressScope(), node_id: '5', revision: 1, epoch: 'server',
        stages: { pose_generation: { status: 'done', images: ['first'] } },
    } });
    await h.widget.refreshProgress(); await h.widget.refreshProgress();
    assert.deepEqual([preview, chain, viewer], [1, 1, 1]);
});

test('queued normal runs resume after regenerate success or failure, including recovered snapshots', async () => {
    for (const fail of [false, true]) {
        const h = await queueHarness();
        h.widget.finishRegenerate = h.context.Widget.prototype.finishRegenerate;
        await h.context.app.queuePrompt();
        const queued = h.widget.data.ui.progress_request_id;
        h.api.fetchApi = async () => response(fail ? 500 : 200, fail ? { error: 'Failure' } : { ok: true });
        if (fail) await assert.rejects(h.widget.regenerateFrom('pose_generation'), /Regenerate failed/);
        else await h.widget.regenerateFrom('pose_generation');
        assert.equal(h.widget.acceptsProgressRequest(queued), true);
        h.stage(queued, 10, 'running');
        assert.equal(h.widget.stageState.pose_generation.status, 'running');
        h.api.fetchApi = async () => response(200, { snapshot: {
            scope: h.widget.progressScope(), node_id: '5', request_id: queued, revision: 11, epoch: 'server',
            stages: { pose_generation: { status: 'done', images: ['queued-result'] } },
        } });
        await h.widget.refreshProgress();
        assert.equal(h.widget.stageState.pose_generation.status, 'done');
        assert.deepEqual(h.widget.stageState.pose_generation.images, ['queued-result']);
        // Serialized metadata from a completed Regenerate is not an active lock.
        const reopened = generator();
        reopened.widget.data = JSON.parse(JSON.stringify(h.widget.data));
        assert.equal(reopened.widget.acceptsProgressRequest(queued), true);
    }
});

test('live and recovered terminal failures remain visible before and after stage completion', async () => {
    for (const completed of [false, true]) {
        const h = await queueHarness();
        h.widget.finishRegenerate = h.context.Widget.prototype.finishRegenerate;
        h.widget.stageState = { pose_generation: { status: completed ? 'done' : 'waiting', images: completed ? ['preview'] : null } };
        h.widget.onStage({ detail: { node_id: '5', stage: 'error', message: 'Cannot write output' } });
        assert.equal(h.widget.stageState.pose_generation.status, 'error');
        assert.equal(h.widget.stageState.pose_generation.message, 'Cannot write output');
        h.widget.stageState = {};
        h.api.fetchApi = async () => response(200, { snapshot: {
            scope: h.widget.progressScope(), node_id: '5', revision: 1, epoch: 'server',
            stages: completed ? { pose_generation: { status: 'done', images: ['preview'] } } : {},
            error: { message: 'Cannot write output', stage: completed ? 'pose_generation' : null },
        } });
        await h.widget.refreshProgress();
        assert.equal(h.widget.stageState.pose_generation.status, 'error');
        assert.equal(h.widget.stageState.pose_generation.message, 'Cannot write output');
        if (completed) assert.deepEqual(h.widget.stageState.pose_generation.images, ['preview']);
    }
});

test('a full snapshot repairs missed stages even after a live event with the same revision', async () => {
    const h = await queueHarness();
    h.stage(undefined, 10, 'done', 'server-run');
    h.widget.stageState.pose_generation.images = ['last'];
    h.api.fetchApi = async () => response(200, { snapshot: {
        scope: h.widget.progressScope(), node_id: '5', epoch: 'server', revision: 10,
        run_id: 'server-run', stages: {
            pose_generation: { status: 'done', images: ['last'] },
            bg_remove: { status: 'done', images: ['missed-event-image'] },
        },
    } });
    let renders = 0;
    h.widget.renderPreview = () => renders++;
    await h.widget.refreshProgress();
    assert.deepEqual(h.widget.stageState.bg_remove.images, ['missed-event-image']);
    await h.widget.refreshProgress();
    assert.equal(renders, 1);
});

test('in-flight snapshots cannot undo a newer server epoch or a new live execution', async () => {
    for (const oldSnapshot of [null, { epoch: 'old-server', revision: 99, stages: {} }]) {
        const h = await queueHarness();
        h.widget._progressEpoch = 'old-server';
        h.widget._progressRevision = 98;
        let resolve;
        h.api.fetchApi = () => new Promise(done => { resolve = done; });
        const refresh = h.widget.refreshProgress();
        h.stage(undefined, 1, 'running', 'new-run');
        resolve(response(200, { snapshot: oldSnapshot && {
            ...oldSnapshot, scope: h.widget.progressScope(), node_id: '5',
        } }));
        await refresh;
        assert.equal(h.widget._progressEpoch, 'server');
        assert.equal(h.widget._progressRevision, 1);
        assert.equal(h.widget.stageState.pose_generation.status, 'running');
    }
    const h = await queueHarness();
    h.stage(undefined, 1, 'done', 'first');
    let resolve;
    h.api.fetchApi = () => new Promise(done => { resolve = done; });
    const refresh = h.widget.refreshProgress();
    h.stage(undefined, 2, 'running', 'second');
    resolve(response(200, { snapshot: null }));
    await refresh;
    assert.equal(h.widget.stageState.pose_generation.status, 'running');
});

test('pending dependency installations stay with their backend and user and ignore legacy queues', async () => {
    const values = new Map(), installs = [];
    const h = transport({
        sessionStorage: { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) },
        app: { registerExtension() {} }, document: { createElement: () => ({}) },
    });
    run(h.context, read('vnccs_control_center').replace(/^import .*;\n/gm, '') + '\nthis.CC = VNCCSControlCenterWidget;');
    const widget = Object.create(h.context.CC.prototype);
    widget._installDependency = async item => { installs.push([h.api.base, h.api.user, item.key]); return true; };
    widget.showMessage = () => {};
    const items = [{ key: 'package', manager_id: 'public-package' }];
    widget._storePendingDependencyInstalls(items);
    h.api.base = '/second/api';
    assert.equal(await widget._resumePendingDependencyInstalls(items), false);
    h.api.base = '/comfy/api'; h.api.user = 'bob';
    assert.equal(await widget._resumePendingDependencyInstalls(items), false);
    h.api.user = 'alice';
    assert.equal(await widget._resumePendingDependencyInstalls(items), true);
    assert.deepEqual(installs, [['/comfy/api', 'alice', 'package']]);
    assert.equal(await widget._resumePendingDependencyInstalls(items), false);
    values.set('vnccs-control-center-pending-dependency-installs', JSON.stringify(['package']));
    assert.equal(await widget._resumePendingDependencyInstalls(items), false);
});
