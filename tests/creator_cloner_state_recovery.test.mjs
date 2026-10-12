import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = name => readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8');
const between = (text, start, end) => {
    const offset = text.indexOf(start), limit = text.indexOf(end, offset);
    assert.ok(offset >= 0 && limit > offset, start);
    return text.slice(offset, limit);
};
const guards = between(source('vnccs_common'), 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '');

for (const status of [400, 500]) {
    test(`Cloner reports failed character deletion (${status}) without changing authored state`, async () => {
        const state = { character: 'Alice', character_info: { name: 'Alice', hair: 'authored hair' },
            source_images: [{ name: 'authored-reference.png' }] };
        const original = JSON.stringify(state), errors = [], requests = [];
        let actions;
        const context = createWidgetContext({
            node: {}, state, console,
            document: { createElement: () => ({ append() {}, appendChild() {} }) },
            showModal(title, content, buttons) {
                const element = content();
                if (title === 'Delete Character') actions = buttons;
                else errors.push(element.innerText);
            },
            setCharacter() { assert.fail('Failed deletion must not change the selected character'); },
            els: { charSelect: { options: [{ value: 'Alice' }], remove() { assert.fail('Failed deletion must retain its option'); } } },
            api: { fetchApi(route) {
                requests.push(route);
                return Promise.resolve({ ok: false, status, text: async () => 'Cannot delete character: storage is read-only' });
            } },
        });
        vm.runInContext(`${guards}
            const beginCharacterRequest = createRequestGuard(node);
            ${between(source('vnccs_character_cloner'), 'const doDelete =', '// --- LAYOUT ---')}
            globalThis.openDelete = doDelete;`, context);
        context.openDelete();
        await actions[1].action({ isConnected: true }, {});
        assert.deepEqual(requests, ['/vnccs/delete']);
        assert.equal(JSON.stringify(state), original);
        assert.equal(errors.length, 1, 'A failed deletion must show its reason');
        assert.match(errors[0], /storage is read-only/);
    });
}

for (const phase of ['restored_reference', 'pending_metadata', 'pending_image', 'no_reference']) {
    test(`Cloner source preview remains authoritative during ${phase}`, async () => {
        const text = source('vnccs_character_cloner'), loaders = [];
        const element = () => ({ style: {}, children: [], classList: { toggle() {}, add() {}, remove() {} },
            appendChild(child) { this.children.push(child); }, addEventListener() {} });
        const image = element(), loading = element(), node = {};
        let loadingVisible = false, resolveMetadata, requests = 0;
        loading.classList.toggle = (name, value) => { if (name === 'is-visible') loadingVisible = value; };
        const metadata = new Promise(resolve => { resolveMetadata = resolve; });
        const reference = { name: 'authored-reference.png', type: 'input' };
        const state = { character: 'Alice', character_info: {}, selected_idx: 0,
            source_images: phase === 'restored_reference' ? [reference] : [], selected_preview_sprite: null };
        const context = createWidgetContext({
            node, state, console, URLSearchParams,
            Image: class { constructor() { loaders.push(this); } },
            document: { createElement: element }, imgList: element(), previewImg: image,
            previewLoading: loading, previewPlaceholder: element(), previewContainer: element(),
            uploadOverlay: element(), uploadBtn: element(), spriteNav: element(),
            spritePrevBtn: element(), spriteNextBtn: element(), spriteCount: element(),
            els: { _ui: { captureView() { return {}; }, restoreView() {}, syncSelects() {}, prepareCharacter() {},
                presentPreview(image, placeholder, url) { image.src = url; image.style.display = 'block'; placeholder.style.display = 'none'; },
                emptyPreview(image, placeholder) { image.style.display = 'none'; placeholder.style.display = 'flex'; },
            } }, defaultCharacterInfo: {}, restoredInfoCharacter: null,
            saveState() {}, updateUIFromState() { context.render(); }, clearSourceImages() {},
            showModal() { assert.fail('No error modal expected'); }, mediaURL: value => value,
            api: { fetchApi() { requests++; return metadata; } },
        });
        vm.runInContext(`${guards}
            ${between(source('vnccs_common'), 'export function createSpritePreviewNavigator', '// ── DOM Widget Canvas Navigation').replaceAll('export ', '')}
            let spritePreviewNavigator, renderThumbs;
            const beginCharacterRequest = createRequestGuard(node), beginCaptionRequest = createRequestGuard(node);
            ${between(text, 'spritePreviewNavigator = createSpritePreviewNavigator(', 'colSrc.appendChild(fileInput);')}
            ${between(text, 'renderThumbs = () => {\n                    const view', 'const syncBackgroundControl =')}
            ${between(text, 'const loadChar = async', 'const loadCharList =')}
            globalThis.load = loadChar; globalThis.render = renderThumbs;`, context);
        const operation = context.load('Alice', true);
        if (phase === 'restored_reference') {
            resolveMetadata({ ok: true, json: async () => ({ count: 2 }) });
            await operation;
            assert.equal(requests, 0, 'An uploaded reference must not start a competing disk preview');
        } else {
            if (phase === 'pending_metadata') {
                state.source_images.push(reference);
                context.render();
            }
            resolveMetadata({ ok: true, json: async () => ({ count: 2 }) });
            await operation;
            if (phase === 'pending_image') {
                state.source_images.push(reference);
                context.render();
            }
            loaders[0]?.onload?.();
        }
        if (phase === 'no_reference') {
            assert.match(image.src, /get_character_pose_preview/);
            assert.equal(state.selected_preview_sprite.character, 'Alice');
        } else {
            assert.match(image.src, /filename=authored-reference.png/);
            assert.equal(state.selected_preview_sprite, null);
        }
        assert.equal(loadingVisible, false, 'A completed source upload must clear obsolete preview loading');
    });
}

for (const obsolete of ['current', 'configure', 'replacement', 'removal', 'close']) {
    test(`Cloner source image removal preserves the current reference after ${obsolete}`, () => {
        const image = { name: 'alice.png' };
        const images = [image], state = { character: 'Alice', source_images: images, selected_idx: 0 }, node = {};
        const overlay = { isConnected: true };
        let actions, saved = 0;
        const context = createWidgetContext({
            node, state, images, imgObj: image, idx: 0, console,
            document: { createElement: () => ({}) },
            showModal(title, content, buttons) { content(); actions = buttons; },
            saveState() { saved++; }, renderThumbs() {},
        });
        const guards = between(source('vnccs_common'), 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '');
        const removal = between(source('vnccs_character_cloner'), '// Delete X', 'wrap.appendChild(img);');
        vm.runInContext(`${guards}
            const beginCaptionRequest = createRequestGuard(node);
            ${removal}
            globalThis.openRemoval = delBtn.onclick; globalThis.configure = beginCaptionRequest;`, context);
        context.openRemoval({ stopPropagation() {} });
        if (obsolete === 'configure') context.configure();
        if (obsolete === 'removal') node.onRemoved();
        if (obsolete === 'close') overlay.isConnected = false;
        if (obsolete === 'replacement') images[0] = { name: 'new-alice-reference.png' };
        if (obsolete === 'configure' || obsolete === 'removal') {
            state.character = 'Bob';
            state.source_images = [{ name: 'bob.png' }];
        }
        actions[1].action(overlay);
        assert.equal(state.source_images.length, obsolete === 'current' ? 0 : 1);
        assert.equal(saved, obsolete === 'current' ? 1 : 0);
        if (obsolete !== 'current') assert.equal(state.source_images[0].name,
            obsolete === 'replacement' ? 'new-alice-reference.png' : obsolete === 'close' ? 'alice.png' : 'bob.png');
    });
}

function setup(kind) {
    const text = source(kind === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner');
    const state = { character: 'Alice', character_info: { name: 'Alice', hair: 'authored hair' },
        gen_settings: { generation_mode: 'illustrious' }, source_images: [] };
    const dataWidget = { name: 'widget_data', value: '{}' }, node = { widgets: [dataWidget] };
    const alerts = [], pendingTimers = new Map();
    let configured = 0, initialized = 0, serialized = 0, galleryClosed = 0;
    node.onConfigure = function () { assert.equal(this, node); configured++; };
    node.onSerialize = function () { assert.equal(this, node); serialized++; };
    const context = createWidgetContext({
        node, state, dataWidget, els: { style: { close(focus) { assert.equal(focus, false); galleryClosed++; } } }, app: { graph: { setDirtyCanvas() {} } },
        defaultGenerationSettings: { generation_mode: 'illustrious', steps: 20 },
        defaultCharacterInfo: { hair: '', eyes: '' },
        beginCharacterRequest() {}, beginCharacterWizardRequest() {}, beginCaptionRequest() {}, beginPreviewRequest() {},
        ensureLoraStack() {}, syncBackgroundForGenerationMode() {}, saveCurrentGenerationModeValues() {}, syncBackgroundControl() {},
        getDefaultCharacterInfo: () => ({ hair: '', eyes: '' }), updateUIFromState() {}, applyStoredPrefs() {},
        loadCharList: async () => {},
        showAlertModal: (...args) => alerts.push(args), showModal: (...args) => alerts.push(args),
        document: { createElement: () => ({}) }, window: { dispatchEvent() {} }, CustomEvent: class {},
        console: { log() {}, warn() {}, error() {} }, syncDOMWidgetWidth() {},
        setTimeout(fn) { const id = pendingTimers.size + 1; pendingTimers.set(id, fn); return id; },
        clearTimeout(id) { pendingTimers.delete(id); }, registerCleanup() {},
    });
    const load = kind === 'creator'
        ? between(text, 'const loadState =', 'const syncSliderValue =')
        : between(text, 'const loadState =', 'const FIELD_HELP =');
    const save = between(text, 'const saveState =', kind === 'creator' ? 'const clearPreviewHandlers =' : 'const parsePoseStudioValues =');
    const serialize = kind === 'creator'
        ? between(text, 'const origSerialize = node.onSerialize;', '// 3. State & Widget Setup')
        : between(text, 'const origSerialize = node.onSerialize;', '// Initialize');
    vm.runInContext(`let restoredWidgetInfoCharacter = null, restoredInfoCharacter = null;
        ${save}
        ${load}
        ${serialize}
        globalThis.load = loadState; globalThis.save = saveState;`, context);
    if (kind === 'creator') {
        context.init = () => { initialized++; context.load(); };
        vm.runInContext(between(text, 'let initialized = false;', '\n            };\n\n            const onResize'), context);
    }
    return { state, node, dataWidget, alerts, context, pendingTimers,
        counts: () => ({ configured, initialized, serialized, galleryClosed }), load: context.load, save: context.save };
}

test('Creator restores legacy breast tags and serializes the separate field for immediate reload', () => {
    const h = setup('creator');
    h.dataWidget.value = JSON.stringify({ character: 'Alice', character_info: {
        name: 'Alice', sex: 'female', body: 'small_breasts, custom build',
    } });
    const original = h.dataWidget.value;
    h.node.onConfigure();
    assert.equal(h.dataWidget.value, original, 'Configuration preserves original saved bytes');
    h.context.els.breastSection = {};
    h.context.els.breast_size = { setValue(value) { this.value = value; } };
    h.context.els.body = { setValue(value) { this.value = value; } };
    vm.runInContext(between(source('vnccs_character_creator_v2'), 'const syncCharacterFields =', 'const syncBackgroundForGenerationMode =')
        + 'globalThis.syncTraits = syncCharacterFields;', h.context);
    h.context.syncTraits();
    assert.equal(h.context.els.breast_size.value, 'small_breasts');
    assert.equal(h.context.els.body.value, 'custom build');
    h.state.character_info.breast_size = 'medium breasts';
    h.save();
    const payload = { widgets_values: ['stale bytes'] };
    h.node.onSerialize(payload);
    const restored = setup('creator');
    restored.dataWidget.value = payload.widgets_values[0];
    restored.node.onConfigure();
    assert.equal(restored.state.character_info.breast_size, 'medium breasts');
    assert.equal(restored.state.character_info.body, 'custom build');
});

for (const kind of ['creator', 'cloner']) {
    for (const [value, enabled] of [
        ['false', false], ['0', false], ['off', false], [false, false],
        ['true', true], ['1', true], ['yes', true], [true, true],
    ]) {
        test(`${kind} restores NSFW ${JSON.stringify(value)} without changing it before an edit`, () => {
            const h = setup(kind);
            h.context.document.createElement = () => ({
                children: [], appendChild(child) { this.children.push(child); },
                setAttribute(name, value) { this[name] = value; },
                classList: { toggle() {} },
            });
            h.context.setHelpText = () => {};
            h.context.helpFor = () => '';
            h.context.syncPoseStudioAge = () => {};
            h.context.renderThumbs = () => {};
            const widgetSource = source(kind === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner');
            vm.runInContext(`${between(widgetSource, 'const createGraphicToggle =', 'const createSlider =')}
                ${kind === 'creator'
                    ? between(widgetSource, 'const syncCharacterFields =', 'const syncBackgroundForGenerationMode =')
                    : between(widgetSource, 'const updateUIFromState =', '// --- Helpers (Hoisted) ---')}
                globalThis.syncGraphic = ${kind === 'creator' ? 'syncCharacterFields' : 'updateUIFromState'};
                globalThis.makeToggle = createGraphicToggle;`, h.context);
            const toggle = h.context.makeToggle('NSFW Mode', 'nsfw');
            assert.equal(h.dataWidget.value, '{}', 'Construction must not persist defaults');
            h.dataWidget.value = JSON.stringify({ character: 'Alice', character_info: { name: 'Alice', nsfw: value } });
            assert.equal(h.load(), true);
            h.context.syncGraphic();
            assert.equal(h.context.els.nsfw.checked, enabled);
            assert.equal(toggle.children[0]['aria-pressed'], String(enabled));
            const workflow = { widgets_values: ['old snapshot'] };
            h.node.onSerialize(workflow);
            assert.equal(JSON.parse(workflow.widgets_values[0]).character_info.nsfw, value);
            toggle.children[0].onclick();
            h.node.onSerialize(workflow);
            assert.equal(JSON.parse(workflow.widgets_values[0]).character_info.nsfw, !enabled);
        });
    }
}

for (const kind of ['creator', 'cloner']) {
    for (const raw of ['{"character":"Alice"', '[]', 'null', '{"character_info":[]}']) {
        test(`${kind} preserves malformed workflow bytes and recovers on valid restoration: ${raw}`, () => {
            const h = setup(kind);
            h.dataWidget.value = raw;
            assert.equal(h.load(), false);
            assert.equal(h.alerts[0][0], 'State Restore Failed');
            h.state.character_info.hair = 'pending edit';
            h.save();
            const workflow = { widgets_values: ['old snapshot'], unrelated: 'authored graph' };
            h.node.onSerialize(workflow);
            assert.equal(workflow.widgets_values[0], raw);
            assert.equal(h.dataWidget.value, raw);
            assert.equal(workflow.unrelated, 'authored graph');
            h.dataWidget.value = JSON.stringify({ character: 'Bob', character_info: { name: 'Bob', hair: 'recovered hair' }, gen_settings: {} });
            assert.equal(h.load(), true);
            assert.equal(h.node._vnccsStateRestoreError, undefined);
            h.save();
            h.node.onSerialize(workflow);
            assert.equal(JSON.parse(workflow.widgets_values[0]).character_info.hair, 'recovered hair');
            assert.equal(h.counts().serialized, 2);
        });
    }
}

test('Creator repeats configure, preserves the original callback, and cancels constructor initialization', () => {
    const h = setup('creator');
    const fallback = [...h.pendingTimers.values()][0];
    h.dataWidget.value = JSON.stringify({ character: 'Bob', character_info: { name: 'Bob', hair: 'Bob hair' } });
    h.node.onConfigure();
    assert.equal(h.state.character_info.hair, 'Bob hair');
    h.dataWidget.value = JSON.stringify({ character: 'Carol', character_info: { name: 'Carol', hair: 'Carol hair' } });
    h.node.onConfigure();
    fallback();
    assert.equal(h.state.character, 'Carol');
    assert.equal(h.state.character_info.hair, 'Carol hair');
    assert.equal(h.counts().configured, 2);
    assert.equal(h.counts().initialized, 2);
    assert.equal(h.counts().galleryClosed, 2);
});

test('Creator restores the selected preview source and pose index without inheriting previous selection', () => {
    const h = setup('creator');
    h.dataWidget.value = JSON.stringify({ character: 'Alice', character_info: { name: 'Alice' },
        preview_valid: true, preview_source: 'pose', sprite_preview_index: 3 });
    assert.equal(h.load(), true);
    assert.equal(h.state.preview_valid, true);
    assert.equal(h.state.preview_source, 'pose');
    assert.equal(h.state.sprite_preview_index, 3);
    h.dataWidget.value = JSON.stringify({ character: 'Bob', character_info: { name: 'Bob' } });
    assert.equal(h.load(), true);
    assert.equal(h.state.preview_valid, false);
    assert.equal(h.state.preview_source, 'gen');
    assert.equal(h.state.sprite_preview_index, 0);
});

test('Cloner repeated configuration never copies traits from the previous character', () => {
    const h = setup('cloner');
    h.state.character_info.eyes = 'Alice green eyes';
    h.dataWidget.value = JSON.stringify({ character: 'Bob', character_info: { name: 'Bob', hair: 'Bob hair' } });
    h.node.onConfigure();
    assert.equal(h.state.character_info.eyes, '');
    assert.equal(h.state.character_info.hair, 'Bob hair');
});

test('Cloner repeated configuration clears references absent from the restored workflow', () => {
    const h = setup('cloner');
    h.dataWidget.value = JSON.stringify({ character: 'Alice', source_images: [{ name: 'alice.png' }],
        character_info: { name: 'Alice', hair: 'Alice hair' }, selected_preview_sprite: { character: 'Alice', index: 2 }, selected_idx: 0 });
    h.node.onConfigure();
    h.dataWidget.value = JSON.stringify({ character: 'Bob', character_info: { name: 'Bob', hair: 'Bob hair' } });
    h.node.onConfigure();
    assert.deepEqual(Array.from(h.state.source_images), []);
    assert.equal(h.state.source_images_character, '');
    assert.equal(h.state.selected_preview_sprite, null);
    assert.equal(h.state.selected_idx, 0);
    assert.equal(JSON.parse(h.dataWidget.value).character, 'Bob');
});

test('Cloner preserves workflow bytes when reference ownership differs from the selected character', () => {
    const h = setup('cloner');
    const raw = JSON.stringify({ character: 'Alice', character_info: { name: 'Alice' },
        source_images: [{ name: 'bob.png' }], source_images_character: 'Bob' });
    h.dataWidget.value = raw;
    assert.equal(h.load(), false);
    assert.deepEqual(Array.from(h.state.source_images), []);
    h.save();
    h.node.onSerialize({ widgets_values: [] });
    assert.equal(h.dataWidget.value, raw);
    assert.match(h.node._vnccsStateRestoreError, /Cannot restore/);
});

for (const kind of ['creator', 'cloner']) {
    for (const available of [[], ['Bob']]) {
        test(`${kind} inventory ${JSON.stringify(available)} preserves the restored missing character and draft`, async () => {
            const h = setup(kind);
            const selected = { value: '', options: [], add(option) { this.options.push(option); },
                set innerHTML(value) { this.options = []; } };
            Object.assign(h.context.els, { charSelect: selected, ckptSelect: { options: [], add() {} }, loraStackSelects: [] });
            Object.assign(h.context, {
                Option: class { constructor(text, value) { this.text = text; this.value = value; } },
                fetchInventory: async route => ({ ok: true, json: async () => {
                    if (route === '/vnccs/context_lists') return { characters: available };
                    if (route.includes('preview_meta')) return { count: 0 };
                    return kind === 'cloner' ? { character_info: { name: 'Bob', hair: 'disk Bob hair' } }
                        : { name: 'Bob', hair: 'disk Bob hair' };
                } }),
                migrateGenerationModeSettings() {}, migratePromptModes() {}, renderControlCenterCards() {},
                syncGenerationControls() {}, refreshGenerationModeUI() {}, applyGenerationProfile() {},
                saveCurrentGenerationModeValues() {}, getModeProfile: () => ({}),
                fetchCcConfig: async () => ({}), applyPromptModeToFields() {}, syncCharacterFields() {},
                MODE_PROMPT_DEFAULTS: { illustrious: {}, anima: {}, qi2: {} }, PROMPT_DEFAULTS_VERSION: 1,
                STYLE_PROMPT_KEYS: JSON.parse(source('vnccs_character_creator_v2').match(/const STYLE_PROMPT_KEYS = (\[[^\n]+\]);/)[1]),
                ANIMA_CLIP_NAME: 'clip', ANIMA_VAE_NAME: 'vae', QI2_MODEL_NAME: 'model', QI2_CLIP_NAME: 'clip', QI2_VAE_NAME: 'vae',
                isCreatorOverhaulLora: () => false, clearSourceImages() { h.state.source_images = []; },
                spritePreviewNavigator: { invalidate() {}, load: async () => {}, hideNav() {} },
                hideSpriteNav() {}, tryCachePreview() {}, clearCharacterSelection() { h.state.character = ''; },
                ensureOption(element, value) { if (element && value && !element.options.some(option => option.value === value)) element.add({ value }); },
            });
            h.context.api.fetchApi = h.context.fetchInventory;
            const text = source(kind === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner');
            const loaders = kind === 'creator'
                ? between(text, 'const beginInitRequest =', 'const doGenerate =')
                : between(text, 'const setCharacter =', 'const updateUIFromState =') +
                    between(text, 'const beginCharacterRequest =', 'const doCreate =');
            vm.runInContext(`${guards}
                const beginPreviewRequest = createRequestGuard(node);
                ${loaders}
                globalThis.loadInventory = ${kind === 'creator' ? 'init' : 'loadCharList'};`, h.context);
            h.dataWidget.value = JSON.stringify({ character: 'Alice', character_info: { name: 'Alice', hair: 'authored Alice hair' },
                source_images: [{ name: 'alice.png' }], source_images_character: 'Alice', gen_settings: {} });
            h.node.onConfigure();
            await h.context.loadInventory();
            const workflow = { widgets_values: ['old snapshot'] };
            h.node.onSerialize(workflow);
            const saved = JSON.parse(workflow.widgets_values[0]);
            assert.equal(saved.character, 'Alice');
            assert.equal(saved.character_info.hair, 'authored Alice hair');
            if (kind === 'cloner') assert.equal(saved.source_images[0].name, 'alice.png');
            assert.equal(selected.value, 'Alice');
            assert.ok(selected.options.some(option => option.value === 'Alice'));
        });
    }
}

for (const kind of ['creator', 'cloner']) {
    for (const order of ['inventory_first', 'selection_first']) {
    test(`${kind} initial inventory preserves pending character metadata: ${order}`, async () => {
        const text = source(kind === 'creator' ? 'vnccs_character_creator_v2' : 'vnccs_character_cloner');
        const state = { character: 'Alice', character_info: { name: 'Alice', hair: 'Alice hair' },
            gen_settings: { generation_mode: 'illustrious' }, source_images: [] };
        let resolveInventory, resolveCharacter;
        const inventory = new Promise(resolve => { resolveInventory = resolve; });
        const character = new Promise(resolve => { resolveCharacter = resolve; });
        const requests = [], selector = { value: 'Bob', options: [{ value: 'Bob' }], add(option) { this.options.push(option); },
            set innerHTML(value) { this.options = []; } };
        let restores = 0;
        const response = value => ({ ok: true, json: async () => value });
        const context = createWidgetContext({
            node: {}, state, console, els: { charSelect: selector, ckptSelect: { options: [], add() {} }, loraStackSelects: [] },
            Option: class { constructor(text, value) { this.text = text; this.value = value; } },
            defaultCharacterInfo: {}, restoredInfoCharacter: null, restoredWidgetInfoCharacter: null,
            getDefaultCharacterInfo: () => ({}), loadState() { restores++; return true; }, saveState() {}, updateUIFromState() {},
            syncBackgroundForGenerationMode() {}, syncCharacterFields() {}, applyPromptModeToFields() {},
            migrateGenerationModeSettings() {}, migratePromptModes() {}, renderControlCenterCards() {},
            syncGenerationControls() {}, refreshGenerationModeUI() {}, applyGenerationProfile() {},
            saveCurrentGenerationModeValues() {}, getModeProfile: () => ({}), fetchCcConfig: async () => ({}),
            MODE_PROMPT_DEFAULTS: { illustrious: {}, anima: {}, qi2: {} }, PROMPT_DEFAULTS_VERSION: 1,
            STYLE_PROMPT_KEYS: JSON.parse(source('vnccs_character_creator_v2').match(/const STYLE_PROMPT_KEYS = (\[[^\n]+\]);/)[1]),
            ANIMA_CLIP_NAME: 'clip', ANIMA_VAE_NAME: 'vae', QI2_MODEL_NAME: 'model', QI2_CLIP_NAME: 'clip', QI2_VAE_NAME: 'vae',
            isCreatorOverhaulLora: () => false, hideSpriteNav() {}, tryCachePreview() {}, clearCharacterSelection() {},
            spritePreviewNavigator: { invalidate() {}, load: async () => {}, hideNav() {} }, mediaURL: value => value,
            showModal() { assert.fail('Unexpected metadata error'); }, showAlertModal() { assert.fail('Unexpected metadata error'); },
            ensureOption(element, value) { if (element && value && !element.options.some(option => option.value === value)) element.add({ value }); },
            api: { fetchApi(route) {
                requests.push(route);
                if (route === '/vnccs/context_lists') return inventory;
                if (route.includes('preview_meta')) return Promise.resolve(response({ count: 0 }));
                const name = route.includes('Bob') ? 'Bob' : 'Alice';
                const data = { name, hair: `${name} hair` };
                return name === 'Bob' ? character : Promise.resolve(response(kind === 'cloner' ? { character_info: data } : data));
            } },
        });
        const loaders = kind === 'creator'
            ? between(text, 'const beginInitRequest =', 'const doGenerate =')
            : between(text, 'const setCharacter =', 'const updateUIFromState =') +
                between(text, 'const beginCharacterRequest =', 'const doCreate =');
        vm.runInContext(`${guards}
            const beginPreviewRequest = createRequestGuard(node);
            ${loaders}
            globalThis.loadInventory = ${kind === 'creator' ? 'init' : 'loadCharList'};
            globalThis.loadCharacter = loadChar;`, context);
        let initial, selection;
        if (order === 'inventory_first') {
            initial = context.loadInventory();
            selection = context.loadCharacter('Bob');
        } else {
            selection = context.loadCharacter('Bob');
            initial = context.loadInventory();
        }
        assert.equal(state.character, 'Alice', 'The old character remains selected while Bob metadata is pending');
        resolveInventory(response({ characters: ['Alice'], checkpoints: ['checkpoint.safetensors'] }));
        await initial;
        assert.equal(selector.value, 'Bob', 'Inventory must retain the pending selection');
        assert.ok(selector.options.some(option => option.value === 'Bob'), 'The newly created character option remains available');
        assert.ok(selector.options.some(option => option.value === 'Alice'), 'Available characters still populate the selector');
        if (kind === 'creator') {
            assert.deepEqual(Array.from(context.localAssets.checkpoints), ['checkpoint.safetensors'], 'Model inventory still loads');
            if (order === 'selection_first') assert.equal(restores, 0, 'Late bootstrap must not restore over a newer user request');
        }
        resolveCharacter(response(kind === 'cloner' ? { character_info: { name: 'Bob', hair: 'Bob hair' } } : { name: 'Bob', hair: 'Bob hair' }));
        await selection;
        assert.equal(state.character, 'Bob', 'Inventory must not invalidate the newer character request');
        assert.equal(state.character_info.hair, 'Bob hair');
        assert.ok(!requests.some(route => /(?:name|character)=Alice/.test(route)), 'Initial inventory must not reload the previous character');
    });
    }
}

test('Cloner configure refreshes inventory and loads the restored preview without replacing the draft', async () => {
    const h = setup('cloner'), requests = [], previews = [];
    Object.assign(h.context.els, { charSelect: { value: '', options: [], add(option) { this.options.push(option); },
        set innerHTML(value) { this.options = []; } } });
    Object.assign(h.context, {
        Option: class { constructor(text, value) { this.text = text; this.value = value; } },
        clearSourceImages() { h.state.source_images = []; }, mediaURL: value => value,
        spritePreviewNavigator: { invalidate() {}, load: async name => { previews.push(name); }, hideNav() {} },
    });
    h.context.api.fetchApi = async route => {
        requests.push(route);
        assert.equal(route, '/vnccs/context_lists', 'A restored metadata draft must not be replaced by disk metadata');
        return { ok: true, json: async () => ({ characters: ['Alice'] }) };
    };
    const text = source('vnccs_character_cloner');
    vm.runInContext(`${guards}
        ${between(text, 'const setCharacter =', 'const updateUIFromState =')}
        ${between(text, 'const beginCharacterRequest =', 'const doCreate =')}`, h.context);
    h.dataWidget.value = JSON.stringify({ character: 'Alice', character_info: { name: 'Alice', hair: 'authored Alice hair' } });
    h.node.onConfigure();
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(requests, ['/vnccs/context_lists']);
    assert.deepEqual(previews, ['Alice']);
    assert.equal(h.state.character_info.hair, 'authored Alice hair');
    assert.equal(h.context.els.charSelect.value, 'Alice');
    assert.equal(h.counts().configured, 1, 'The original configuration callback remains active');
});

test('Cloner empty workflow state clears the previous character and reference', () => {
    const h = setup('cloner');
    h.dataWidget.value = JSON.stringify({ character: 'Alice', source_images: [{ name: 'alice.png' }],
        character_info: { name: 'Alice', hair: 'Alice hair' }, selected_preview_sprite: { character: 'Alice', index: 2 } });
    h.node.onConfigure();
    h.dataWidget.value = '{}';
    h.node.onConfigure();
    assert.equal(h.state.character, '');
    assert.deepEqual(Array.from(h.state.source_images), []);
    assert.equal(h.state.source_images_character, '');
    assert.equal(h.state.character_info.hair, '');
    assert.equal(h.state.character_info.name, undefined);
    assert.equal(h.state.selected_preview_sprite, null);
    assert.equal(h.dataWidget.value, '{}');
    const workflow = { widgets_values: ['old snapshot'] };
    h.node.onSerialize(workflow);
    assert.equal(JSON.parse(workflow.widgets_values[0]).character, '');
});

test('Cloner restores visible extra and negative prompts before editing and serialization', () => {
    const h = setup('cloner');
    h.context.document = { createElement: () => ({ children: [], appendChild(child) { this.children.push(child); } }) };
    h.context.syncPoseStudioAge = () => {};
    const text = source('vnccs_character_cloner');
    vm.runInContext(`let renderThumbs = null;
        ${between(text, 'const createTA =', 'botRow.appendChild(createTA')}
        ${between(text, 'const updateUIFromState =', '// --- Helpers (Hoisted)')}
        globalThis.extraPrompt = createTA('Extra / LoRA Prompt', 'lora_prompt').children[0];
        globalThis.negativePrompt = createTA('Negative Prompt', 'negative_prompt').children[0];`, h.context);
    for (const character of ['Alice', 'Bob']) {
        h.dataWidget.value = JSON.stringify({ character, character_info: { name: character,
            lora_prompt: `${character} extra tags`, negative_prompt: `${character} excluded tags` } });
        h.node.onConfigure();
        assert.equal(h.context.extraPrompt.value, `${character} extra tags`);
        assert.equal(h.context.negativePrompt.value, `${character} excluded tags`);
    }
    h.context.extraPrompt.oninput({ target: { value: 'Bob edited tags' } });
    const workflow = { widgets_values: [] };
    h.node.onSerialize(workflow);
    assert.equal(JSON.parse(workflow.widgets_values[0]).character_info.lora_prompt, 'Bob edited tags');
    assert.equal(JSON.parse(workflow.widgets_values[0]).character_info.negative_prompt, 'Bob excluded tags');
});

for (const [file, type] of [['vnccs_character_creator_v2', 'CharacterCreatorV2'],
    ['vnccs_character_cloner', 'CharacterCloner'], ['vnccs_clothes_designer', 'ClothesDesigner']]) {
    test(`${type} blocks execution after failed restoration while allowing inactive nodes`, async () => {
        const node = { type, _vnccsStateRestoreError: 'Saved state cannot be restored.' };
        let queued = 0;
        const context = createWidgetContext({ app: { graph: { _nodes: [node] }, queuePrompt() { queued++; assert.equal(this, context.app); } } });
        const text = source(file);
        const setupMethod = between(text, 'async setup()', 'async beforeRegisterNodeDef').trim().replace(/,$/, '');
        vm.runInContext(`globalThis.extension = { ${setupMethod} };`, context);
        await context.extension.setup();
        const queueWrapper = context.app.queuePrompt;
        await context.extension.setup();
        assert.equal(context.app.queuePrompt, queueWrapper);
        await assert.rejects(context.app.queuePrompt(), /Saved state/);
        assert.equal(queued, 0);
        for (const mode of [2, 4]) {
            node.mode = mode;
            await context.app.queuePrompt();
        }
        assert.equal(queued, 2);
        delete node._vnccsStateRestoreError;
        node.mode = 0;
        await context.app.queuePrompt();
        assert.equal(queued, 3);
        if (type === 'ClothesDesigner') {
            node._vnccsClothesSelectionPending = true;
            await assert.rejects(context.app.queuePrompt(), /metadata is loading/);
            assert.equal(queued, 3);
            for (const mode of [2, 4]) {
                node.mode = mode;
                await context.app.queuePrompt();
            }
            assert.equal(queued, 5);
        }
    });
}

test('Creator ignores context responses from a previous configuration', async () => {
    const tasks = [], node = {}, state = { character: 'Alice' }, els = {};
    const context = createWidgetContext({ node, state, els, console: { error() {} }, loadState: () => true,
        api: { fetchApi: () => new Promise(resolve => tasks.push(resolve)) }, localAssets: 'untouched' });
    const guards = between(source('vnccs_common'), 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '');
    const initialization = between(source('vnccs_character_creator_v2'), 'const beginInitRequest =', 'const beginCharacterRequest =');
    vm.runInContext(`${guards}\nconst beginCharacterRequest = createRequestGuard(node);
        let currentInitializationSelection = beginCharacterRequest();
        ${initialization}\nglobalThis.init = init;`, context);
    const first = context.init();
    state.character = 'Bob';
    const second = context.init();
    tasks[1]({ ok: false, json: async () => ({ characters: [] }) });
    await second;
    tasks[0]({ ok: true, json: async () => ({ characters: ['Alice'] }) });
    await first;
    assert.equal(state.character, 'Bob');
    assert.equal(context.localAssets, 'untouched');
    assert.deepEqual(els, {});
});

test('corrupt Creator state blocks direct character and style previews before API or disk work', async () => {
    const node = { _vnccsStateRestoreError: 'Cannot restore saved character state.' };
    let picker;
    const context = createWidgetContext({ node, container: {}, characterStyleCatalog: {}, els: {},
        state: { character_info: {} }, createStylePicker: options => { picker = options; return { root: {}, customInput: {} }; },
        setHelpText() {}, helpFor() {}, mediaURL: value => value,
        saveState() { assert.fail('Corrupt character state must not save'); },
        api: { fetchApi() { assert.fail('Corrupt character state must not start API work'); } },
    });
    const text = source('vnccs_character_creator_v2');
    vm.runInContext(`${between(text, 'const createStyleField =', 'const createGraphicToggle =')}
        ${between(text, 'const doGenerate = async', '// 7. Graph Restore Hook')}
        createStyleField(); globalThis.generate = doGenerate;`, context);
    await context.generate();
    assert.throws(() => picker.fetchApi('/vnccs/character_styles', { method: 'POST' }), /Cannot restore/);
    assert.throws(() => picker.getPreviewPayload(), /Cannot restore/);
});

test('corrupt Cloner state blocks upload and caption analysis before API work', async () => {
    const node = { _vnccsStateRestoreError: 'Cannot restore saved cloner state.' };
    const context = createWidgetContext({ node, fileInput: {}, autoGenBtn: {},
        api: { fetchApi() { assert.fail('Corrupt cloner state must not start API work'); } } });
    const text = source('vnccs_character_cloner');
    vm.runInContext(`${between(text, 'fileInput.onchange =', '// Click anywhere on overlay triggers upload')}
        ${between(text, 'autoGenBtn.onclick =', '// Helper: Progress Polling')}`, context);
    await context.fileInput.onchange({ target: { files: [{ name: 'image.png' }] } });
    await context.autoGenBtn.onclick();
});
