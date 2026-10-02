import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = name => readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8');
const between = (text, start, end) => text.slice(text.indexOf(start), text.indexOf(end, text.indexOf(start)));
const guardCode = between(source('vnccs_common'), 'export function registerCleanup', '// ── Widget Data Sync').replaceAll('export ', '');
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};
const response = data => ({ ok: true, json: async () => data });

test('request guards reject superseded and removed-node responses, preserving cleanup', () => {
    let originalCleanup = 0;
    const node = { onRemoved() { originalCleanup++; } };
    const context = vm.createContext({ node, console });
    vm.runInContext(`${guardCode}; globalThis.begin = createRequestGuard(node)`, context);
    const first = context.begin();
    const second = context.begin();
    assert.equal(first(), false);
    assert.equal(second(), true);
    node.onRemoved();
    assert.equal(second(), false);
    assert.equal(context.begin()(), false);
    assert.equal(originalCleanup, 1);
});

for (const widget of ['clothes', 'cloner', 'creator']) {
    test(`${widget} ignores reversed metadata responses and responses after removal`, async () => {
        const requests = [];
        const previews = [];
        const state = { character: 'Alice', costume: 'Dress', character_info: {}, gen_settings: {} };
        const node = {};
        const context = vm.createContext({
            state, node, console, els: {},
            api: { fetchApi() { const task = deferred(); requests.push(task); return task.promise; } },
            showInfo() { assert.fail('Unexpected current-request error'); },
            updateUIFromState() {},
            spritePreviewNavigator: { load: async name => previews.push(name) },
            getDefaultCharacterInfo: () => ({}), syncBackgroundForGenerationMode() {},
            MODE_PROMPT_DEFAULTS: { illustrious: {}, anima: {}, qi2: {} }, PROMPT_DEFAULTS_VERSION: 1,
            applyPromptModeToFields() {}, syncCharacterFields() {}, hideSpriteNav() {},
            tryCachePreview: name => previews.push(name), showSpritePreview: name => previews.push(name),
        });
        let code;
        if (widget === 'clothes') {
            code = between(source('vnccs_clothes_designer'), 'const beginCostumeInfoRequest', 'const updatePreviewImage');
            code += '\nglobalThis.load = loadCostumeInfo;';
        } else {
            const name = widget === 'cloner' ? 'vnccs_character_cloner' : 'vnccs_character_creator_v2';
            code = between(source(name), 'const beginCharacterRequest', widget === 'cloner' ? 'const loadCharList' : 'const doGenerate');
            code += '\nglobalThis.load = loadChar;';
        }
        vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\n${code}`, context);
        const older = context.load('Alice');
        state.character = 'Bob';
        const newer = context.load('Bob');
        requests[1].resolve(response(widget === 'cloner' ? { character_info: { hair: 'new' } } : { hair: 'new', top: 'new' }));
        if (widget === 'creator') {
            // The current character starts a separate metadata request for its preview.
            await new Promise(resolve => setImmediate(resolve));
            requests[2].resolve(response({ count: 0 }));
        }
        await newer;
        requests[0].resolve(response(widget === 'cloner' ? { character_info: { hair: 'old' } } : { hair: 'old', top: 'old' }));
        await older;
        assert.equal(widget === 'clothes' ? state.costume_info.top : state.character_info.hair, 'new');
        if (widget !== 'clothes') assert.deepEqual(previews, ['Bob']);
        const last = context.load('Bob');
        node.onRemoved();
        requests.at(-1).resolve(response(widget === 'cloner' ? { character_info: { hair: 'removed' } } : { hair: 'removed', top: 'removed' }));
        await last;
        assert.equal(widget === 'clothes' ? state.costume_info.top : state.character_info.hair, 'new');
    });
}

function migrationHarness() {
    const events = [], errors = [], timers = new Map(), requests = [];
    const state = { runId: 'job', running: true, selected: new Set(['Alice']), scan: { characters: [{ legacy_name: 'Alice' }] } };
    const node = {};
    const context = vm.createContext({
        state, node, AbortController,
        console: { error: (...args) => errors.push(args), warn() {} },
        setTimeout: fn => { const key = Symbol(); timers.set(key, fn); return key; },
        clearTimeout: key => timers.delete(key),
        api: { fetchApi: () => { const task = deferred(); requests.push(task); return task.promise; } },
        fill: { style: {} }, progressText: {},
        scanBtn: {}, selectedBtn: {}, allBtn: {}, repairBtn: {}, log: {},
        window: { dispatchEvent: event => events.push(event.type) },
        CustomEvent: class { constructor(type) { this.type = type; } },
    });
    const text = source('vnccs_migration_assistant');
    vm.runInContext(`${guardCode}\n${between(text, 'let removed = false', 'const root = el')}\n${between(text, 'const renderStatus =', 'const scan =')}\n${between(text, 'const poll = async', 'const start = async')}\n${between(text, 'const start = async', 'scanBtn.onclick =')}\nglobalThis.poll = poll; globalThis.setBusy = setBusy; globalThis.start = start; globalThis.repair = repairSprites;`, context);
    return { context, state, node, requests, events, errors, timers };
}

test('migration status failures unlock retry without starting a duplicate job', async () => {
    const h = migrationHarness();
    let work = h.context.poll();
    h.requests[0].resolve(response({ status: 'running' }));
    await work;
    assert.equal(h.timers.size, 1);
    work = h.context.poll();
    h.requests[1].resolve({ ok: false, status: 503, json: async () => ({ error: 'Service unavailable' }) });
    await work;
    assert.equal(h.state.running, false);
    assert.equal(h.context.scanBtn.textContent, 'Retry Status');
    assert.equal(h.context.allBtn.disabled, true);
    assert.equal(h.errors.length, 1);
    assert.equal(h.timers.size, 0);
    // This is also called by a checkbox change after the status connection fails.
    h.state.selected.delete('Alice'); h.context.setBusy(h.state.running);
    h.state.selected.add('Alice'); h.context.setBusy(h.state.running);
    assert.equal(h.context.scanBtn.disabled, false);
    assert.equal(h.context.selectedBtn.disabled, true);
    assert.equal(h.context.allBtn.disabled, true);
    assert.equal(h.context.repairBtn.disabled, true);
    await h.context.start(true); await h.context.repair();
    assert.equal(h.requests.length, 2, 'A disconnected existing job must prevent new mutations');
    work = h.context.poll();
    h.requests[2].resolve(response({ status: 'done' }));
    await work;
    assert.equal(h.state.runId, '');
    assert.deepEqual(h.events, ['vnccs.characters.updated', 'vnccs.migration.complete']);
});

test('migration handles missing jobs and cleans up pending polling on removal', async () => {
    const h = migrationHarness();
    let work = h.context.poll();
    h.requests[0].resolve({ ok: false, status: 404, json: async () => ({ error: 'Unknown job' }) });
    await work;
    assert.equal(h.state.runId, '');
    assert.equal(h.context.scanBtn.textContent, 'Scan');
    h.state.runId = 'next';
    work = h.context.poll();
    h.node.onRemoved();
    h.requests[1].resolve(response({ status: 'done' }));
    await work;
    assert.deepEqual(h.events, []);
    assert.equal(h.timers.size, 0);
});

test('terminal backend job errors release migration controls and report their cause', async () => {
    const h = migrationHarness();
    const work = h.context.poll();
    h.requests[0].resolve(response({ status: 'error', error: 'Disk full', log: ['Converting sprites'] }));
    await work;
    assert.equal(h.state.runId, '');
    assert.equal(h.state.running, false);
    assert.equal(h.context.scanBtn.textContent, 'Scan');
    assert.equal(h.context.selectedBtn.disabled, false);
    assert.equal(h.context.allBtn.disabled, false);
    assert.equal(h.context.repairBtn.disabled, false);
    assert.match(h.context.log.textContent, /Disk full/);
    assert.doesNotMatch(h.context.log.textContent, /may still be running/);
    assert.equal(h.errors.length, 1);
    assert.equal(h.timers.size, 0);
    assert.deepEqual(h.events, []);
});

for (const oldResult of ['success', 'missing', 'error', 'removed']) {
    test(`clothes preview ignores stale ${oldResult} after a selection change`, async () => {
        const requests = [], navigated = [], saved = [];
        const state = { character: 'Alice', costume: 'Dress' }, node = {};
        const context = vm.createContext({
            state, node, console,
            hasSelectedEditableCostume: () => true,
            saveState: () => saved.push(state.selected_preview_sprite),
            fetch: () => { const task = deferred(); requests.push(task); return task.promise; },
            els: { previewImg: { style: {} }, placeholder: { style: {} } },
            spritePreviewNavigator: {
                invalidate() {}, hideNav() { assert.fail('Stale missing response changed navigation'); },
                load: async (character, options) => navigated.push({ character, ...options }),
            },
        });
        vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\n${between(source('vnccs_clothes_designer'), 'const updatePreviewImage =', 'const onPreviewUpdated =')}\nglobalThis.update = updatePreviewImage;`, context);
        const old = context.update();
        state.character = 'Bob'; state.costume = 'Suit';
        const current = context.update();
        requests[1].resolve({ ok: true }); await current;
        const savedCount = saved.length;
        if (oldResult === 'removed') node.onRemoved();
        requests[0].resolve(oldResult === 'error' ? Promise.reject(new Error('Old request failed')) : { ok: oldResult !== 'missing' });
        await old;
        assert.equal(navigated.length, 1);
        assert.equal(navigated[0].character, 'Bob');
        assert.equal(navigated[0].costume, 'Suit');
        assert.match(navigated[0].fallbackUrl, /character=Bob&costume=Suit/);
        assert.equal(saved.length, savedCount);
    });
}

function creatorImageHarness() {
    const images = [], saved = [], requests = [];
    const state = { character: 'Alice', character_info: {}, sprite_preview_count: 1,
        gen_settings: { ckpt_name: 'model', lora_stack: [] } };
    const node = { _randomizeSeedIfNeeded() {} };
    const els = { previewImg: { style: {}, removeAttribute(name) { delete this[name]; } }, placeholder: { style: {} }, btnGen: {} };
    const context = vm.createContext({
        state, node, els, console, container: {},
        Image: class { constructor() { images.push(this); } },
        clearPreviewHandlers() {}, setPreviewLoading() {}, updateSpriteNav() {},
        hideSpriteNav: () => { state.sprite_preview_count = 0; },
        saveState: valid => { state.preview_valid = valid; saved.push({ character: state.character, valid, source: state.preview_source }); },
        restoredWidgetInfoCharacter: null, getDefaultCharacterInfo: () => ({}), syncCharacterFields() {},
        showAlertModal() { assert.fail('Unexpected model validation error'); },
        isSelectedCcAssetInstalled: () => true,
        saveCurrentGenerationModeValues() {}, createLoadingOverlay: () => ({ remove() {} }),
        showMessage() { assert.fail('Stale generation error surfaced'); },
        api: { fetchApi: () => { const task = deferred(); requests.push(task); return task.promise; } },
    });
    const text = source('vnccs_character_creator_v2');
    vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\n${between(text, 'const clearCharacterSelection =', 'const applyStoredPrefs =')}\n${between(text, 'const beginCharacterRequest =', 'const doGenerate = async')}\n${between(text, 'const doGenerate = async', '// 7. Graph Restore Hook')}\nglobalThis.preview = { showSpritePreview, tryCachePreview, clearCharacterSelection, doGenerate, loadChar };`, context);
    return { context, state, node, els, images, saved, requests, ...context.preview };
}

test('creator rejects an old pose callback after current cache preview succeeds', () => {
    const h = creatorImageHarness();
    h.showSpritePreview('Alice', 0);
    h.state.character = 'Bob';
    h.tryCachePreview('Bob');
    h.images[1].onload();
    h.images[0].onload();
    h.images[0].onerror();
    assert.match(h.els.previewImg.src, /character=Bob/);
    assert.equal(h.state.preview_source, 'gen');
    assert.equal(h.saved.length, 1);
});

test('creator rejects old cache callbacks after pose loading, clearing and removal', () => {
    const h = creatorImageHarness();
    h.tryCachePreview('Alice');
    h.state.sprite_preview_count = 1;
    h.showSpritePreview('Alice', 0);
    h.images[1].onload();
    h.images[0].onload(); h.images[0].onerror();
    assert.equal(h.state.preview_source, 'pose');
    assert.equal(h.saved.length, 1);
    h.tryCachePreview('Alice');
    h.clearCharacterSelection();
    h.images[2].onload();
    assert.equal(h.state.preview_valid, false);
    assert.equal(h.els.previewImg.src, undefined);
    h.state.character = 'Bob';
    h.tryCachePreview('Bob');
    h.node.onRemoved();
    h.images[3].onload(); h.images[3].onerror();
    assert.equal(h.saved.length, 1);
});

test('creator generation invalidates older images and ignores responses for a previous character', async () => {
    const h = creatorImageHarness();
    h.showSpritePreview('Alice', 0);
    const generation = h.doGenerate();
    h.images[0].onload();
    assert.equal(h.els.previewImg.src, undefined);
    h.state.character = 'Bob';
    h.tryCachePreview('Bob');
    h.images[1].onload();
    h.requests[0].resolve(response({ image: 'old-generated-image' }));
    await generation;
    assert.match(h.els.previewImg.src, /character=Bob/);
    assert.equal(h.saved.filter(item => item.valid).length, 1);
});

for (const result of ['poses', 'cache', 'error']) {
    test(`creator ignores old ${result} metadata after a newer generation completes`, async () => {
        const h = creatorImageHarness();
        const metadata = h.loadChar('Alice', true);
        const generated = h.doGenerate();
        h.requests[1].resolve(response({ image: 'fresh-generated' }));
        await generated;
        h.requests[0].resolve(result === 'error' ? Promise.reject(new Error('Old request failed')) : response({ count: result === 'poses' ? 1 : 0 }));
        await metadata;
        assert.equal(h.els.previewImg.src, 'data:image/png;base64,fresh-generated');
        assert.equal(h.state.preview_source, 'gen');
        assert.equal(h.images.length, 0, 'Old metadata must not initiate a new image request');
    });
}

for (const operation of ['character', 'costume', 'initialization']) {
    test(`clothes ${operation} metadata cannot supersede a newer preview operation`, async () => {
        const metadata = deferred();
        const state = { character: 'Alice', costume: 'Dress' }, node = {};
        let previewUpdates = 0, costumeLoads = 0;
        const context = vm.createContext({
            state, node, console, charSel: {}, costSel: {},
            els: { charSelect: { innerHTML: '', add() {} } }, Option: class {},
            spritePreviewNavigator: { invalidate() {} },
            api: { fetchApi: async () => response({ characters: ['Alice'] }) },
            loadCharacterInfo: () => metadata.promise,
            loadCostumeInfo: () => metadata.promise,
            loadCostumes: async () => { costumeLoads++; state.costumes = ['Dress', 'Casual']; return true; },
            syncCostumeEditControls() {}, saveState() {}, setClothesCoreLora() {}, syncGenerationControls() {},
            updatePreviewImage: () => { previewUpdates++; },
        });
        const text = source('vnccs_clothes_designer');
        let code;
        if (operation === 'character') code = between(text, 'charSel.onchange =', 'charRow.appendChild(charSel)');
        else if (operation === 'costume') code = between(text, 'costSel.onchange =', 'els.costSel = costSel;');
        else code = between(text, '// Initial Load', 'container.appendChild(topRow)').replace('(async () => {', 'globalThis.initialization = (async () => {');
        vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\nconst beginSelectionRequest = createRequestGuard(node);\nglobalThis.supersede = beginPreviewRequest;\n${code}`, context);
        const pending = operation === 'initialization' ? context.initialization : context[operation === 'character' ? 'charSel' : 'costSel'].onchange({ target: { value: operation === 'character' ? 'Alice' : 'Dress' } });
        await new Promise(resolve => setImmediate(resolve));
        context.supersede(); // A newer generated/cache preview now owns the display.
        state.selected_preview_sprite = { character: 'Alice', costume: 'Dress', index: 3 };
        metadata.resolve(true);
        await pending;
        assert.equal(previewUpdates, 0);
        assert.equal(state.selected_preview_sprite.index, 3);
        if (operation !== 'costume') {
            assert.equal(costumeLoads, 1, 'New preview must not cancel essential costume loading');
            assert.deepEqual(state.costumes, ['Dress', 'Casual']);
        }
    });
}

test('new clothes cache preview does not cancel delayed context and selector initialization', async () => {
    const contextRequest = deferred(), options = [], shown = [];
    const state = { character: 'Alice', costume: 'Dress', selected_preview_sprite: { index: 3 } }, node = {};
    let metadataLoads = 0;
    const context = vm.createContext({
        state, node, console, Option: class { constructor(label, value) { this.value = value; } },
        els: { charSelect: { innerHTML: '', add: value => options.push(value.value) }, previewImg: { style: {} }, placeholder: { style: {} } },
        api: { fetchApi: () => contextRequest.promise }, fetch: async () => ({ ok: true }),
        spritePreviewNavigator: { invalidate() {}, showFallback: url => shown.push(url) },
        hasSelectedEditableCostume: () => true, setClothesCoreLora() {}, syncGenerationControls() {}, saveState() {},
        loadCharacterInfo: async () => { metadataLoads++; return true; },
        loadCostumes: async () => { state.costumes = ['Dress', 'Casual']; return true; },
    });
    const text = source('vnccs_clothes_designer');
    const initialCode = between(text, '// Initial Load', 'container.appendChild(topRow)').replace('(async () => {', 'globalThis.initialization = (async () => {');
    vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\nconst beginSelectionRequest = createRequestGuard(node);\n${between(text, 'const updatePreviewImage =', 'const onPreviewUpdated =')}\nglobalThis.update = updatePreviewImage;\n${initialCode}`, context);
    await context.update(true);
    contextRequest.resolve(response({ characters: ['Alice', 'Bob'] }));
    await context.initialization;
    assert.deepEqual(options, ['Alice', 'Bob']);
    assert.equal(context.els.charSelect.value, 'Alice');
    assert.equal(metadataLoads, 1);
    assert.deepEqual(state.costumes, ['Dress', 'Casual']);
    assert.equal(shown.length, 1);
    assert.match(shown[0], /force_cache=true/);
    assert.equal(state.selected_preview_sprite.index, 3);
});

test('delayed clothes initialization cannot take ownership from a newer character selection', async () => {
    const contextRequest = deferred(), metadataRequest = deferred(), shown = [], options = [];
    const state = { character: 'Alice', costume: 'AliceDress' }, node = {};
    let metadataLoads = 0;
    const context = vm.createContext({
        state, node, console, charSel: {}, Option: class { constructor(label, value) { this.value = value; } },
        els: { charSelect: { innerHTML: '', add: option => options.push(option.value) } },
        api: { fetchApi: () => contextRequest.promise }, spritePreviewNavigator: { invalidate() {} },
        loadCharacterInfo: () => { metadataLoads++; return metadataRequest.promise; },
        loadCostumes: async () => { state.costume = 'BobSuit'; return true; },
        setClothesCoreLora() {}, syncGenerationControls() {}, saveState() {},
        updatePreviewImage: () => shown.push([state.character, state.costume]),
    });
    const text = source('vnccs_clothes_designer');
    const initialCode = between(text, '// Initial Load', 'container.appendChild(topRow)').replace('(async () => {', 'globalThis.initialization = (async () => {');
    vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\nconst beginSelectionRequest = createRequestGuard(node);\n${between(text, 'charSel.onchange =', 'charRow.appendChild(charSel)')}\n${initialCode}`, context);
    const selection = context.charSel.onchange({ target: { value: 'Bob' } });
    contextRequest.resolve(response({ characters: ['Alice', 'Bob'] }));
    await context.initialization;
    assert.equal(metadataLoads, 1, 'Obsolete init must not start a second metadata request');
    metadataRequest.resolve(true); await selection;
    assert.deepEqual(options, ['Alice', 'Bob']);
    assert.equal(context.els.charSelect.value, 'Bob');
    assert.deepEqual(shown, [['Bob', 'BobSuit']]);
});

test('shared preview navigator ignores stale metadata errors and images after selection changes or removal', async () => {
    const requests = [], images = [], loaded = [], missing = [];
    const node = {}, selection = { character: 'Alice', costume: 'Dress' };
    const context = vm.createContext({
        node, selection, console, URLSearchParams,
        Image: class { constructor() { images.push(this); } },
        api: { fetchApi: () => { const task = deferred(); requests.push(task); return task.promise; } },
        onLoaded: (...args) => loaded.push(args), onMissing: () => missing.push(true),
    });
    const navigatorCode = between(source('vnccs_common'), 'export function createSpritePreviewNavigator', '// ── DOM Widget Canvas Navigation').replace('export ', '');
    vm.runInContext(`${guardCode}\n${navigatorCode}\nglobalThis.navigator = createSpritePreviewNavigator({ node, isSelectionCurrent: value => value.character === selection.character && value.costume === selection.costume, onLoaded, onMissing });`, context);
    const nav = context.navigator;
    const old = nav.load('Alice', { costume: 'Dress' });
    selection.character = 'Bob'; selection.costume = 'Suit';
    const current = nav.load('Bob', { costume: 'Suit' });
    requests[1].resolve(response({ count: 1 })); await current;
    requests[0].resolve(Promise.reject(new Error('Old metadata request failed'))); await old;
    images[0].onload();
    assert.equal(loaded.length, 1);
    assert.equal(loaded[0][1].character, 'Bob');
    nav.show(0);
    selection.character = 'Carol';
    images[1].onload(); images[1].onerror();
    assert.equal(loaded.length, 1); assert.equal(missing.length, 0);
    selection.character = 'Bob';
    nav.show(0); node.onRemoved();
    images[2].onload(); images[2].onerror();
    assert.equal(loaded.length, 1); assert.equal(missing.length, 0);
});

for (const previousSelection of [false, true]) {
    test(`clothes cache refresh supersedes pending preview with ${previousSelection ? 'previous' : 'fresh'} navigator ownership`, async () => {
        const requests = [], images = [];
        const state = { character: 'Bob', costume: 'Suit' }, node = {};
        const els = { previewImg: { style: {} }, placeholder: { style: {} } };
        const context = vm.createContext({
            state, node, els, console, URLSearchParams,
            Image: class { constructor() { images.push(this); } },
            api: { fetchApi: async () => response({ count: 1 }) },
            fetch: () => { const task = deferred(); requests.push(task); return task.promise; },
            saveState() {}, hasSelectedEditableCostume: () => true,
        });
        const navigatorCode = between(source('vnccs_common'), 'export function createSpritePreviewNavigator', '// ── DOM Widget Canvas Navigation').replace('export ', '');
        vm.runInContext(`${guardCode}\n${navigatorCode}\nconst beginPreviewRequest = createRequestGuard(node);
            const spritePreviewNavigator = createSpritePreviewNavigator({ node, image: els.previewImg, placeholder: els.placeholder,
                isSelectionCurrent: value => value.character === state.character && value.costume === state.costume,
                onLoaded: url => { if (!url.includes('force_cache=true')) state.selected_preview_sprite = null; }
            });
            ${between(source('vnccs_clothes_designer'), 'const updatePreviewImage =', 'const onPreviewUpdated =')}
            globalThis.update = updatePreviewImage; globalThis.navigator = spritePreviewNavigator;`, context);
        if (previousSelection) {
            state.character = 'Alice'; state.costume = 'Dress';
            await context.navigator.load('Alice', { costume: 'Dress' });
            images[0].onload();
            state.character = 'Bob'; state.costume = 'Suit';
        }
        const normal = context.update();
        const selected = { character: 'Bob', costume: 'Suit', index: 2 };
        state.selected_preview_sprite = selected;
        const cached = context.update(true);
        requests[1].resolve({ ok: true }); await cached;
        const freshImage = images.at(-1);
        assert.match(freshImage.src, /character=Bob&costume=Suit.*force_cache=true/);
        freshImage.onload();
        requests[0].resolve({ ok: true }); await normal;
        if (previousSelection) images[0].onload();
        assert.match(els.previewImg.src, /character=Bob&costume=Suit.*force_cache=true/);
        assert.equal(state.selected_preview_sprite, selected, 'Displaying cache must retain its source reference');
        assert.equal(context.navigator.state.character, 'Bob');
        assert.equal(context.navigator.state.costume, 'Suit');
        assert.equal(images.length, previousSelection ? 2 : 1);
    });
}
