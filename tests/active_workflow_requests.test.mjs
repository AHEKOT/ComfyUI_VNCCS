import { createWidgetContext } from './widget_context.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = name => readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8');
const between = (text, start, end) => text.slice(text.indexOf(start), text.indexOf(end, text.indexOf(start)));
const guardCode = between(source('vnccs_common'), 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '');
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};
const response = data => ({ ok: true, json: async () => data });

for (const kind of ['creator', 'cloner']) {
    for (const obsolete of ['current', 'before_confirmation', 'selection', 'removal', 'close']) {
        test(`${kind} character delete preserves selection ownership through ${obsolete}`, async () => {
            const text = source(kind === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner');
            const state = { character: 'Alice', character_info: { name: 'Alice', hair: 'Alice hair' } };
            const node = {}, pending = deferred(), requests = [];
            const select = { options: [{ value: 'Alice' }, { value: 'Bob' }], remove(index) { this.options.splice(index, 1); } };
            let buttons;
            const overlay = { isConnected: true };
            const replace = name => { state.character = name; state.character_info = { name }; };
            const context = createWidgetContext({
                node, state, els: { charSelect: select }, console,
                document: { createElement: () => ({ style: {}, append() {}, appendChild() {} }) },
                showModal(title, content, actions) { content(); buttons = actions; },
                showAlertModal() { assert.fail('Stale deletion must not show an error'); },
                clearCharacterSelection: () => replace(''), loadChar: async name => replace(name),
                setCharacter: async name => replace(name), saveState() {},
                api: { fetchApi(route) { requests.push(route); return pending.promise; } },
            });
            const deleteCode = between(text, 'const doDelete =', kind === 'creator' ? 'const openTagConstructor =' : '// --- LAYOUT ---');
            vm.runInContext(`${guardCode}\nconst beginCharacterRequest = createRequestGuard(node);\n${deleteCode}\nglobalThis.openDelete = doDelete; globalThis.selectCharacter = beginCharacterRequest;`, context);
            context.openDelete();
            if (obsolete === 'before_confirmation') {
                context.selectCharacter();
                state.character = 'Bob';
                state.character_info = { name: 'Bob', hair: 'authored Bob draft' };
                await buttons[1].action(overlay, {});
                assert.deepEqual(requests, []);
                assert.equal(state.character_info.hair, 'authored Bob draft');
                return;
            }
            const operation = buttons[1].action(overlay, {});
            assert.deepEqual(requests, ['/vnccs/delete']);
            if (obsolete === 'selection') context.selectCharacter();
            else if (obsolete === 'removal') node.onRemoved();
            else if (obsolete === 'close') overlay.isConnected = false;
            if (obsolete !== 'current') {
                state.character = 'Bob';
                state.character_info = { name: 'Bob', hair: 'authored Bob draft' };
            }
            pending.resolve({ ok: true });
            await operation;
            assert.equal(state.character, 'Bob');
            assert.equal(state.character_info.hair, obsolete === 'current' ? undefined : 'authored Bob draft');
            assert.equal(select.options.length, obsolete === 'current' ? 1 : 2);
        });
    }
}

test('request guards reject superseded and removed-node responses, preserving cleanup', () => {
    let originalCleanup = 0;
    const node = { onRemoved() { originalCleanup++; } };
    const context = createWidgetContext({ node, console });
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
        const state = { character: 'Alice', costume: 'Dress', character_info: {}, gen_settings: {}, source_images: [] };
        const node = {};
        const context = createWidgetContext({
            state, node, console, els: {},
            defaultCharacterInfo: {}, restoredInfoCharacter: null, restoredWidgetInfoCharacter: null,
            saveState() {}, showModal() { assert.fail('Unexpected load error'); },
            showAlertModal() { assert.fail('Unexpected load error'); },
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
            code = between(source('vnccs_clothes_designer'), 'const hasSelectedEditableCostume =', 'const onValidationError =');
            code += between(source('vnccs_clothes_designer'), 'const beginCostumeInfoRequest', 'const updatePreviewImage');
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

for (const oldResult of ['success', 'missing', 'error', 'removed']) {
    test(`clothes preview ignores stale ${oldResult} after a selection change`, async () => {
        const requests = [], navigated = [], saved = [];
        const state = { character: 'Alice', costume: 'Dress' }, node = {};
        const context = createWidgetContext({
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
    const context = createWidgetContext({
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
        api: { fetchApi: (route, options) => { const task = deferred(); task.route = route; task.options = options; requests.push(task); return task.promise; } },
    });
    const text = source('vnccs_character_creator_v2');
    const previewButton = between(text, 'let previewRunning = false;', 'const beginWorkflowStatusRequest =');
    vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\n${previewButton}\nworkflowBusy = false;\n${between(text, 'const clearCharacterSelection =', 'const applyStoredPrefs =')}\n${between(text, 'const beginCharacterRequest =', 'const doGenerate = async')}\n${between(text, 'const doGenerate = async', '// 7. Graph Restore Hook')}\nglobalThis.preview = { showSpritePreview, tryCachePreview, clearCharacterSelection, doGenerate, loadChar };`, context);
    return { context, state, node, els, images, saved, requests, ...context.preview };
}

for (const previewSource of ['gen', 'pose']) {
    test(`Creator keeps the restored ${previewSource} preview when loading saved character state`, async () => {
        const h = creatorImageHarness();
        Object.assign(h.state, { preview_valid: true, preview_source: previewSource, sprite_preview_index: 3 });
        const operation = h.loadChar('Alice', true);
        if (h.requests.length) h.requests[0].resolve(response({ count: 4 }));
        await operation;
        assert.equal(h.images.length, 1);
        if (previewSource === 'gen') {
            assert.equal(h.requests.length, 0, 'A generated preview must load its cache instead of a random pose');
            assert.match(h.images[0].src, /get_cached_preview/);
        } else {
            assert.match(h.images[0].src, /get_character_pose_preview.*index=3/);
        }
        h.images[0].onload();
        assert.equal(h.state.preview_valid, true);
        assert.equal(h.state.preview_source, previewSource);
        if (previewSource === 'pose') assert.equal(h.state.sprite_preview_index, 3);
    });
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

test('Creator preview submits a temporary resolved-seed marker without saving it in widget state', async () => {
    const h = creatorImageHarness();
    Object.assign(h.state.gen_settings, { seed: 23, seed_mode: 'randomize' });
    const generation = h.doGenerate();
    const sent = JSON.parse(h.requests[0].options.body);
    assert.equal(sent._vnccs_seed_resolved, true);
    assert.equal(sent.gen_settings.seed, 23);
    assert.equal(sent.gen_settings.seed_mode, 'randomize');
    assert.equal(h.state._vnccs_seed_resolved, undefined);
    h.requests[0].resolve(response({ image: 'generated-image' }));
    await generation;
    assert.equal(h.state.gen_settings.seed_mode, 'randomize');
    assert.equal(h.state._vnccs_seed_resolved, undefined);
});

for (const field of ['hair', 'seed']) {
    test(`creator ignores pending preview success after ${field} changes on the same character`, async () => {
        const h = creatorImageHarness();
        const generation = h.doGenerate();
        if (field === 'hair') h.state.character_info.hair = 'new authored hair';
        else h.state.gen_settings.seed = 42;
        h.requests[0].resolve(response({ image: 'old-generated-image' }));
        await generation;
        assert.equal(h.els.previewImg.src, undefined);
        assert.equal(h.saved.filter(item => item.valid).length, 0);
        assert.equal(h.state.preview_valid, undefined);
        assert.equal(h.els.btnGen.disabled, false);
    });
}

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
        const context = createWidgetContext({
            state, node, console, charSel: {}, costSel: {},
            els: { charSelect: { innerHTML: '', add() {} }, costSel: {} }, Option: class {},
            spritePreviewNavigator: { invalidate() {} },
            api: { fetchApi: async () => response({ characters: ['Alice'] }) },
            loadCharacterInfo: () => metadata.promise,
            loadCostumeInfo: () => metadata.promise,
            loadCostumes: async () => { costumeLoads++; state.costumes = ['Dress', 'Casual']; return true; },
            syncCostumeEditControls() {}, saveState() {}, setClothesCoreLora() {}, syncGenerationControls() {}, beginClothesUploadRequest() {},
            updatePreviewImage: () => { previewUpdates++; },
        });
        const text = source('vnccs_clothes_designer');
        let code;
        if (operation === 'character') code = between(text, 'charSel.onchange =', 'charRow.appendChild(charSel)');
        else if (operation === 'costume') code = between(text, 'costSel.onchange =', 'els.costSel = costSel;');
        else code = between(text, '// Initial Load', 'container.appendChild(topRow)').replace('(async () => {', 'globalThis.initialization = (async () => {');
        vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\nconst beginSelectionRequest = createRequestGuard(node);\nconst beginClothesWizardRequest = createRequestGuard(node);\nglobalThis.supersede = beginPreviewRequest;\n${code}`, context);
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
    const context = createWidgetContext({
        state, node, console, Option: class { constructor(label, value) { this.value = value; } },
        els: { charSelect: { innerHTML: '', add: value => options.push(value.value) }, previewImg: { style: {} }, placeholder: { style: {} } },
        api: { fetchApi: route => route.startsWith('/vnccs/get_preview?') ? Promise.resolve({ ok: true }) : contextRequest.promise },
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
    const context = createWidgetContext({
        state, node, console, charSel: {}, Option: class { constructor(label, value) { this.value = value; } },
        beginClothesWizardRequest: () => () => true, beginClothesUploadRequest() {},
        els: { charSelect: { innerHTML: '', add: option => options.push(option.value) } },
        api: { fetchApi: () => contextRequest.promise }, spritePreviewNavigator: { invalidate() {} },
        loadCharacterInfo: async character => {
            metadataLoads++;
            const ready = await metadataRequest.promise;
            if (ready) state.character = character;
            return ready;
        },
        loadCostumes: async () => { state.costume = 'BobSuit'; return true; },
        setClothesCoreLora() {}, syncGenerationControls() {}, syncCostumeEditControls() {}, saveState() {},
        updatePreviewImage: () => shown.push([state.character, state.costume]),
    });
    const text = source('vnccs_clothes_designer');
    const initialCode = between(text, '// Initial Load', 'container.appendChild(topRow)').replace('(async () => {', 'globalThis.initialization = (async () => {');
    vm.runInContext(`${guardCode}\nconst beginPreviewRequest = createRequestGuard(node);\nconst beginSelectionRequest = createRequestGuard(node);\nconst beginClothesWizardRequest = createRequestGuard(node);\n${between(text, 'charSel.onchange =', 'charRow.appendChild(charSel)')}\n${initialCode}`, context);
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
    const context = createWidgetContext({
        node, selection, console, URLSearchParams,
        Image: class { constructor() { images.push(this); } },
        api: { fetchApi: (route, options) => { const task = deferred(); task.route = route; task.options = options; requests.push(task); return task.promise; } },
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
        const context = createWidgetContext({
            state, node, els, console, URLSearchParams,
            Image: class { constructor() { images.push(this); } },
            api: { fetchApi: route => {
                if (!route.startsWith('/vnccs/get_preview?')) return Promise.resolve(response({ count: 1 }));
                const task = deferred(); requests.push(task); return task.promise;
            } },
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

for (const widget of ['creator', 'cloner']) {
    for (const failure of ['http', 'json', 'network', 'schema']) {
        test(`${widget} keeps the previous character and reports ${failure} metadata failure`, async () => {
            const state = { character: 'Alice', character_info: { name: 'Alice', hair: 'Alice hair' }, source_images: ['alice.png'] };
            const original = JSON.stringify(state), saved = [], errors = [];
            const context = createWidgetContext({
                state, node: {}, els: { charSelect: { value: 'Bob' } },
                document: { createElement: () => ({}) },
                saveState: () => saved.push(JSON.stringify(state)),
                showModal: title => errors.push(title), showAlertModal: title => errors.push(title),
                api: { fetchApi: async () => {
                    if (failure === 'network') throw new Error('Offline');
                    if (failure === 'json') return { ok: true, status: 200, json: async () => { throw new Error('HTML'); } };
                    return { ok: failure !== 'http', status: failure === 'http' ? 502 : 200, json: async () => failure === 'http' ? { error: 'Gateway unavailable' } : [] };
                } },
            });
            const name = widget === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner';
            const load = between(source(name), 'const beginCharacterRequest', widget === 'creator' ? 'const doGenerate' : 'const loadCharList');
            const select = widget === 'creator'
                ? between(source(name), 'const charSel = document.createElement("select");', 'els.charSelect = charSel;') + '\nthis.select = () => charSel.onchange({ target: { value: "Bob" } });'
                : between(source(name), 'const setCharacter =', 'const updateUIFromState =') + '\nthis.select = () => setCharacter("Bob", { clearSources: true });';
            vm.runInContext(guardCode + '\nconst beginPreviewRequest = createRequestGuard(node);\n' + load + '\n' + select, context);
            await context.select();
            assert.equal(JSON.stringify(state), original);
            assert.deepEqual(saved, []);
            assert.deepEqual(errors, ['Character Load Failed']);
            assert.equal(context.els.charSelect.value, 'Alice');
        });
    }
    test(`${widget} keeps workflow ownership until successful metadata arrival`, async () => {
        const pending = deferred(), saved = [];
        const state = { character: 'Alice', character_info: { name: 'Alice', hair: 'old', extra: 'only Alice' }, gen_settings: {}, source_images: [] };
        const context = createWidgetContext({
            state, node: {}, els: { charSelect: { value: 'Bob' } },
            defaultCharacterInfo: {}, restoredInfoCharacter: null, restoredWidgetInfoCharacter: null,
            updateUIFromState() {}, saveState: () => saved.push(JSON.parse(JSON.stringify(state))),
            spritePreviewNavigator: { invalidate() {}, load: async () => {} },
            getDefaultCharacterInfo: () => ({}), syncBackgroundForGenerationMode() {},
            MODE_PROMPT_DEFAULTS: { illustrious: {}, anima: {}, qi2: {} }, PROMPT_DEFAULTS_VERSION: 1,
            applyPromptModeToFields() {}, syncCharacterFields() {}, hideSpriteNav() {}, tryCachePreview() {},
            showModal() { assert.fail('Unexpected error'); }, showAlertModal() { assert.fail('Unexpected error'); },
            api: { fetchApi: route => route.includes('preview_meta') ? Promise.resolve(response({ count: 0 })) : pending.promise },
        });
        const name = widget === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner';
        vm.runInContext(guardCode + '\nconst beginPreviewRequest = createRequestGuard(node);\n'
            + between(source(name), 'const beginCharacterRequest', widget === 'creator' ? 'const doGenerate' : 'const loadCharList')
            + '\nthis.load = loadChar;', context);
        const work = context.load('Bob');
        assert.equal(state.character, 'Alice');
        assert.equal(saved.length, 0);
        pending.resolve(response(widget === 'cloner' ? { character_info: { hair: 'Bob hair' } } : { hair: 'Bob hair' }));
        await work;
        assert.equal(state.character, 'Bob');
        assert.equal(state.character_info.name, 'Bob');
        assert.equal(state.character_info.hair, 'Bob hair');
        assert.equal(state.character_info.extra, undefined);
        assert.equal(saved[0].character_info.name, 'Bob');
    });
}
