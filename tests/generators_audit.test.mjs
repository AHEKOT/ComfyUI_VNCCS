import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { createWidgetContext } from './widget_context.mjs';

const generatorSource = readFileSync(new URL('../web/vnccs_character_generator.js', import.meta.url), 'utf8');
const emotionSource = readFileSync(new URL('../web/vnccs_emotion_v2.js', import.meta.url), 'utf8');
const commonSource = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
// State tests isolate motion; sakura_interactions.test.mjs exercises the real UI owner.
const motionUI = {
    captureView() { return {}; }, restoreView() {}, syncSelects() {}, syncSelection() {}, prepareCharacter() {},
    presentPreview(image, placeholder, src) { image.src = src; image.style.display = 'block'; placeholder.style.display = 'none'; },
    emptyPreview(image, placeholder) { image.style.display = 'none'; placeholder.style.display = 'flex'; },
};

for (const mode of ['illustrious', 'anima', 'qi2']) {
    test(`Emotion Studio ${mode} never offers another family's fallback Turbo LoRA`, () => {
        class Element {
            constructor(tag) { this.tag = tag; this.children = []; }
            append(...items) { this.children.push(...items); }
            appendChild(item) { this.children.push(item); return item; }
        }
        const animaLora = 'Anima/anima-turbo-lora-v0.1.safetensors';
        const qi2Lora = 'QI2/Viggle/turbo.safetensors';
        const container = new Element('container');
        const state = { gen: { generation_mode: mode, steps: 20, cfg: 8, turbo_enabled: false } };
        const writes = [];
        const context = createWidgetContext({ ui: motionUI, state,
            document: { createElement: tag => new Element(tag), createTextNode: text => ({ text }) },
            ccState: { config: null }, localAssets: { loras: [animaLora, qi2Lora] },
            ANIMA_TURBO_LORA_NAME: animaLora, QI2_TURBO_LORA_NAME: qi2Lora,
            ccKind: entry => String(entry.kind || '').toLowerCase(),
            ccType: entry => String(entry.type || '').toLowerCase(),
            ccRelPath: entry => entry.local_path.split('/').slice(2).join('/'),
            ccResolveStatus: entry => entry.status, cardStatusLabel: status => status,
            isQi2TurboModel: () => false, syncGenerationControls() {}, renderControlCenterCards() {},
            saveGenerationSettings() { writes.push({ ...state.gen }); },
            setAnimaTurboMode(enabled, name) { state.gen.turbo_enabled = enabled; state.gen.dmd_lora_name = name; },
            ccDownloadEntry() { assert.fail('Installed fallback must never download'); },
        });
        const toggleStart = emotionSource.indexOf('                const setCcTurboMode =');
        const toggleEnd = emotionSource.indexOf('                const buildAssetCard =', toggleStart);
        const renderStart = emotionSource.indexOf('                const renderModeLoraCards =');
        const renderEnd = emotionSource.indexOf('                const renderControlCenterCards =', renderStart);
        vm.runInContext(emotionSource.slice(toggleStart, toggleEnd)
            + emotionSource.slice(renderStart, renderEnd)
            + '\nthis.renderLoras = renderModeLoraCards;', context);
        context.renderLoras(container, mode);
        const fallbackCards = container.children.filter(child => child.className === 'ems-lora-card');
        assert.equal(fallbackCards.length, mode === 'illustrious' ? 0 : 1);
        if (fallbackCards[0]) {
            fallbackCards[0].onclick();
            assert.equal(state.gen.dmd_lora_name, mode === 'qi2' ? qi2Lora : animaLora);
        }
        if (mode === 'illustrious') {
            assert.deepEqual(writes, []);
            assert.deepEqual(state.gen, { generation_mode: mode, steps: 20, cfg: 8, turbo_enabled: false });
            container.children = [];
            context.ccState.config = { lora: [{ name: 'SDXL Turbo', kind: 'SDXL', type: 'TurboLora',
                local_path: 'models/loras/DMD2/sdxl-turbo.safetensors', status: 'installed' }] };
            context.renderLoras(container, mode);
            const card = container.children.find(child => child.className === 'ems-lora-card');
            assert.ok(card);
            card.onclick();
            assert.equal(state.gen.dmd_lora_name, 'DMD2/sdxl-turbo.safetensors');
            assert.deepEqual([state.gen.steps, state.gen.cfg], [4, 1]);
        }
    });
}

for (const mode of ['anima', 'qi2']) {
    for (const authored of [{ clip_name: 'custom-clip.safetensors', vae_name: 'custom-vae.safetensors' },
        { clip_name: 'custom-clip.safetensors', vae_name: '' }, { clip_name: '', vae_name: '' }]) {
        test(`Emotion Studio ${mode} catalog rendering preserves explicit auxiliary assets ${JSON.stringify(authored)}`, () => {
            const settings = { name: 'generation_settings', value: '{}' };
            const state = { gen: { generation_mode: mode, diffusion_model_name: 'selected-model.safetensors',
                ...authored, mode_settings: {} }, selectedPoseIndices: null };
            const catalog = { clip: { local_path: `models/text_encoders/${mode}-default-clip.safetensors` },
                vae: { local_path: `models/vae/${mode}-default-vae.safetensors` } };
            const context = createWidgetContext({ ui: motionUI, state, generationSettingsWidget: settings,
                modelWidget: null, styleWidget: null, initialSharedSeed: 721,
                GENERATION_DEFAULTS: { mode_settings: { anima: {}, qi2: {}, illustrious: {} } },
                ANIMA_CLIP_NAME: 'anima-clip', ANIMA_VAE_NAME: 'anima-vae',
                QI2_CLIP_NAME: 'qi2-clip', QI2_VAE_NAME: 'qi2-vae',
                ccState: { config: { models: [] } }, localAssets: {},
                generationEls: { animaModelCards: { style: {} }, illustriousModelCards: { style: {} },
                    qi2ModelCards: { style: {} } },
                ccFirstEntry: category => catalog[category],
                ccRelPath: entry => entry.local_path.split('/').slice(2).join('/'),
                ccEntries: () => [], mergeCcAndLocalEntries: () => [], renderModelPicker() {},
                downloadAnimaBundle() {}, downloadQi2Bundle() {},
                commitWidget(widget, value) { if (widget) widget.value = value; },
                syncSelectedPoseIndicesToGenerationSettings() {},
                window: { dispatchEvent() {} }, CustomEvent: class {},
            });
            const section = (start, end) => emotionSource.slice(emotionSource.indexOf(start), emotionSource.indexOf(end, emotionSource.indexOf(start)));
            vm.runInContext(section('                const ensureAnimaDefaultAux =', '                const selectCcAsset =')
                + section('                const renderControlCenterCards =', '                function populateSelect(')
                + section('                function saveGenerationSettings()', '                function setGenerationMode(')
                + '\nthis.renderCards = renderControlCenterCards;', context);
            context.renderCards();
            context.saveGenerationSettings();
            const stored = JSON.parse(settings.value);
            assert.equal(stored.clip_name, authored.clip_name || `${mode}-default-clip.safetensors`);
            assert.equal(stored.vae_name, authored.vae_name || `${mode}-default-vae.safetensors`);
            assert.equal(stored.mode_settings[mode].clip_name, stored.clip_name);
            assert.equal(stored.mode_settings[mode].vae_name, stored.vae_name);
            catalog.clip.local_path = `models/text_encoders/${mode}-refreshed-clip.safetensors`;
            catalog.vae.local_path = `models/vae/${mode}-refreshed-vae.safetensors`;
            context.renderCards();
            context.saveGenerationSettings();
            assert.equal(JSON.parse(settings.value).clip_name, stored.clip_name);
            assert.equal(JSON.parse(settings.value).vae_name, stored.vae_name);
        });
    }
}

for (const removedAt of ['current', 'before-load', 'response', 'json', 'config-error']) {
    test(`Emotion Studio asset initialization ignores removal at ${removedAt}`, async () => {
        const deferred = () => {
            let resolve, reject;
            const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
            return { promise, resolve, reject };
        };
        const response = deferred(), json = deferred(), config = deferred();
        const updates = [], requests = [], state = { gen: {} };
        const context = createWidgetContext({ ui: motionUI, state, disposed: false, localAssets: {},
            ANIMA_CLIP_NAME: 'anima-clip', ANIMA_VAE_NAME: 'anima-vae',
            generationEls: { sampler: {}, scheduler: {}, loraSelects: [{}] },
            api: { fetchApi: route => { requests.push(route); return response.promise; } },
            populateSelect() { updates.push('options'); },
            syncGenerationControls() { updates.push('controls'); },
            saveGenerationSettings() { updates.push('save-and-broadcast'); },
            fetchCcConfig() { requests.push('catalog'); return config.promise; },
            renderControlCenterCards() { updates.push('cards'); },
            console: { warn() {} },
        });
        const start = emotionSource.indexOf('                async function loadGenerationAssets()');
        const end = emotionSource.indexOf('                function syncGenerationControls()', start);
        vm.runInContext(emotionSource.slice(start, end), context);
        if (removedAt === 'before-load') context.disposed = true;
        const operation = context.loadGenerationAssets();
        if (removedAt === 'response') context.disposed = true;
        response.resolve({ ok: true, json: () => json.promise });
        await new Promise(setImmediate);
        if (removedAt === 'json') context.disposed = true;
        json.resolve({ checkpoints: ['portrait.safetensors'], diffusion_models: ['anima.safetensors'],
            text_encoders: ['anima-clip'], vae_models: ['anima-vae'], loras: ['portrait-lora'] });
        await new Promise(setImmediate);
        const beforeConfig = [...updates];
        if (removedAt === 'config-error') {
            context.disposed = true;
            config.reject(new Error('Catalog unavailable'));
        } else config.resolve({});
        await operation;
        if (removedAt === 'current' || removedAt === 'config-error') {
            assert.equal(state.gen.ckpt_name, 'portrait.safetensors');
            assert.equal(updates.includes('save-and-broadcast'), true);
            assert.deepEqual(updates, beforeConfig);
            assert.deepEqual(requests, ['/vnccs/context_lists', 'catalog']);
        } else {
            assert.deepEqual(state.gen, {});
            assert.deepEqual(updates, []);
            assert.deepEqual(requests, removedAt === 'before-load' ? [] : ['/vnccs/context_lists']);
        }
    });
}

for (const model of ['Illustrious', 'Anima', 'QI2']) {
    test(`Emotion Studio preserves legacy ${model} model when settings omit generation_mode`, () => {
        const mode = model.toLowerCase();
        const modelWidget = { name: 'generation_model', value: model };
        const generationSettingsWidget = { name: 'generation_settings', value: '{"seed":721}' };
        const state = { character: 'Alice', gen: {}, selectedCostumes: new Set(), selectedEmotions: new Set() };
        const modes = [];
        const context = createWidgetContext({ ui: motionUI, state, modelWidget, generationSettingsWidget,
            charWidget: null, styleWidget: null, costumesDataWidget: null, emotionsDataWidget: null,
            poseSelectionSummary: null, poseSelectionGrid: null,
            disposed: false, restoreErrors: new Map(), beginSelectionRequest() {}, markNodeDirty() {},
            syncGenerationControls() { modes.push(state.gen.generation_mode); }, renderPoseSelection() {},
            renderCostumes() {}, window: { dispatchEvent() {} }, CustomEvent: class {},
        });
        const section = (start, end) => emotionSource.slice(emotionSource.indexOf(start), emotionSource.indexOf(end, emotionSource.indexOf(start)));
        vm.runInContext(section('                const ANIMA_TURBO_LORA_NAME =', '                // State')
            + section('                function commitWidget(', '                function setGenerationMode(')
            + section('                function restoreStateFromWidgets()', '                node._vnccsEmotionRestoreFromWidgets'), context);
        context.restoreStateFromWidgets();
        assert.equal(modes.at(-1), mode);
        assert.equal(state.gen.seed, 721);
        assert.equal(generationSettingsWidget.value, '{"seed":721}');
        context.persistAllState();
        assert.equal(modelWidget.value, model);
        const stored = JSON.parse(generationSettingsWidget.value);
        assert.equal(stored.generation_mode, mode);
        assert.equal(stored.seed, 721);
        assert.deepEqual([stored.steps, stored.cfg], mode === 'illustrious' ? [20, 8] : mode === 'qi2' ? [25, 3] : [30, 4]);
        generationSettingsWidget.value = JSON.stringify({ generation_mode: mode, seed: 721, seed_mode: 'randomize',
            steps: 99, cfg: 9, selected_pose_indices: [2, 7], mode_settings: { [mode]: {
                steps: 18, cfg: 0, seed: 123, seed_mode: 'fixed', selected_pose_indices: [1], selected_preview_sprite: { index: 9 } } } });
        context.restoreStateFromWidgets();
        context.persistAllState();
        assert.equal(modelWidget.value, model);
        assert.deepEqual([state.gen.steps, state.gen.cfg, state.gen.seed, state.gen.seed_mode], [18, 0, 721, 'randomize']);
        assert.deepEqual([...state.selectedPoseIndices], [2, 7]);
        assert.equal(state.gen.selected_preview_sprite, undefined);
        generationSettingsWidget.value = JSON.stringify({ generation_mode: mode, mode_settings: { [mode]: {
            steps: 18, cfg: 0, seed: 321, seed_mode: 'fixed', selected_pose_indices: [1] } } });
        context.restoreStateFromWidgets();
        context.persistAllState();
        assert.deepEqual([state.gen.steps, state.gen.cfg, state.gen.seed, state.gen.seed_mode], [18, 0, 321, 'fixed']);
        assert.equal(state.selectedPoseIndices, null);
        assert.equal(JSON.parse(generationSettingsWidget.value).selected_pose_indices, undefined);
        generationSettingsWidget.value = '{"generation_mode":"anima","steps":17,"cfg":0}';
        context.restoreStateFromWidgets();
        context.persistAllState();
        assert.equal(modelWidget.value, 'Anima');
        assert.deepEqual([state.gen.steps, state.gen.cfg], [17, 0]);
    });
}

for (const indices of [[1.5], ['1garbage'], [true], [0], [-1], '1', {}, [null], [1, '2', '1_0'], null, []]) {
    test(`Emotion Studio preserves the exact saved pose selection ${JSON.stringify(indices)}`, () => {
        const valid = indices === null || (Array.isArray(indices) && indices.every(value =>
            typeof value === 'number' && Number.isInteger(value) && value > 0
            || typeof value === 'string' && /^[+-]?\d(?:_?\d)*$/.test(value.trim()) && Number(value.replaceAll('_', '')) > 0));
        const modelWidget = { name: 'generation_model', value: 'Anima' };
        const raw = JSON.stringify({ selected_pose_indices: indices, seed: 721 });
        const generationSettingsWidget = { name: 'generation_settings', value: raw };
        const state = { character: 'Alice', gen: {}, selectedCostumes: new Set(['Coat']), selectedEmotions: new Set(['smile']) };
        const node = {}, restoreErrors = new Map();
        const context = createWidgetContext({ ui: motionUI, node, state, modelWidget, generationSettingsWidget,
            charWidget: null, styleWidget: null, costumesDataWidget: null, emotionsDataWidget: null,
            disposed: false, restoreErrors, beginSelectionRequest() {}, markNodeDirty() {},
            syncGenerationControls() {}, renderPoseSelection() {}, renderCostumes() {}, showModalText() {},
            window: { dispatchEvent() {} }, CustomEvent: class {},
        });
        const section = (start, end) => emotionSource.slice(emotionSource.indexOf(start), emotionSource.indexOf(end, emotionSource.indexOf(start)));
        vm.runInContext(section('                const ANIMA_TURBO_LORA_NAME =', '                // State')
            + section('                function commitWidget(', '                function isPoseSelected(')
            + section('                function persistAllState()', '                function setGenerationMode(')
            + section('                node._validateBeforeQueue', '                const showConfirm =')
            + section('                function restoreStateFromWidgets()', '                node._vnccsEmotionRestoreFromWidgets'), context);
        context.restoreStateFromWidgets();
        context.persistAllState();
        assert.equal(restoreErrors.has('generation_settings'), !valid);
        if (valid) {
            assert.deepEqual(state.selectedPoseIndices === null ? null : [...state.selectedPoseIndices],
                indices === null ? null : indices.map(value => Number(String(value).replaceAll('_', ''))));
            assert.equal(node._validateBeforeQueue(), indices === null || indices.length > 0);
        } else {
            assert.equal(generationSettingsWidget.value, raw);
            assert.equal(node._validateBeforeQueue(), false);
        }
    });
}

function emotionModalContext() {
    class Element {
        constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.isConnected = true; }
        append(...items) { this.children.push(...items); }
        appendChild(item) { this.children.push(item); return item; }
        replaceChildren(...items) { this.children = items; }
        remove() { this.isConnected = false; }
        focus() {}
    }
    let resolve;
    const pending = new Promise(done => { resolve = done; });
    const state = { character: 'Alice', emotions: [], selectedEmotions: new Set(), gen: {} };
    const container = new Element('container'), node = {}, writes = [], requests = [], readers = [];
    const context = createWidgetContext({ ui: motionUI,
        state, node, container, disposed: false,
        document: { createElement: tag => new Element(tag) },
        api: { fetchApi: (_, request) => { requests.push(JSON.parse(request.body)); return pending; } },
        FileReader: class {
            constructor() { readers.push(this); }
            readAsDataURL(file) { this.file = file; }
        },
        renderEmotions() {}, updateEmotionsData() { writes.push([...state.selectedEmotions]); },
        charWidget: null, modelWidget: null, generationSettingsWidget: null, styleWidget: null,
        costumesDataWidget: null, emotionsDataWidget: null,
        syncGenerationControls() {}, renderPoseSelection() {}, renderCostumes() {}, saveGenerationSettings() {},
    });
    const cleanupStart = commonSource.indexOf('export function registerCleanup');
    const cleanupEnd = commonSource.indexOf('// ── DOM Widget Width Sync', cleanupStart);
    const modalStart = emotionSource.indexOf('                function createCustomEmotionField(');
    const modalEnd = emotionSource.indexOf('                node._validateBeforeQueue', modalStart);
    const restoreStart = emotionSource.indexOf('                function restoreStateFromWidgets()');
    const restoreEnd = emotionSource.indexOf('                node._vnccsEmotionRestoreFromWidgets', restoreStart);
    const resetStart = emotionSource.indexOf('                function resetSelectedEmotionsForCharacterChange()');
    const resetEnd = emotionSource.indexOf('                function createEmotionCard(', resetStart);
    vm.runInContext(commonSource.slice(cleanupStart, cleanupEnd).replaceAll('export ', '')
        + '\nconst beginSelectionRequest = createRequestGuard(node);\n'
        + emotionSource.slice(modalStart, modalEnd) + emotionSource.slice(restoreStart, restoreEnd)
        + emotionSource.slice(resetStart, resetEnd), context);
    context.showCustomEmotionModal();
    const backdrop = container.children[0];
    const find = (element, condition) => condition(element) ? element : element.children.map(child => find(child, condition)).find(Boolean);
    const save = find(backdrop, element => element.innerText === 'Save');
    const name = find(backdrop, element => element.placeholder === 'Emotion name');
    name.value = 'Happy';
    return { context, node, state, backdrop, save, writes, requests, readers, resolve, find };
}

for (const stale of ['current', 'character', 'away-and-back', 'configure', 'removed', 'closed']) {
    test(`Emotion custom save preserves selection ownership after ${stale}`, async () => {
        const harness = emotionModalContext();
        const operation = harness.save.onclick();
        if (stale === 'character') harness.state.character = 'Bob';
        else if (stale === 'away-and-back') {
            harness.state.character = 'Bob';
            harness.context.resetSelectedEmotionsForCharacterChange();
            harness.state.character = 'Alice';
            harness.context.resetSelectedEmotionsForCharacterChange();
        }
        else if (stale === 'configure') harness.context.restoreStateFromWidgets();
        else if (stale === 'removed') harness.node.onRemoved();
        else if (stale === 'closed') harness.backdrop.remove();
        if (stale !== 'current') harness.state.selectedEmotions.add('authored-draft');
        const previousWrites = harness.writes.length;
        harness.resolve({ ok: true, json: async () => ({ emotion: { safe_name: 'happy', description: 'happy' } }) });
        await operation;
        assert.deepEqual([...harness.state.selectedEmotions], stale === 'current' ? ['happy'] : ['authored-draft']);
        assert.equal(harness.writes.length - previousWrites, stale === 'current' ? 1 : 0);
    });
}

for (const stale of ['current', 'character', 'away-and-back', 'configure', 'removed']) {
    test(`Emotion Select ALL confirmation preserves current selection after ${stale}`, () => {
        const harness = emotionModalContext();
        const btnAll = {};
        let confirm;
        Object.assign(harness.context, {
            btnAll, getFilteredEmotions: () => [{ safe_name: 'happy' }],
            selectedPoseCount: () => 1,
            showConfirm: (_, __, callback) => { confirm = callback; },
        });
        harness.state.selectedCostumes = new Set(['Dress']);
        const start = emotionSource.indexOf('                btnAll.onclick =');
        const end = emotionSource.indexOf('                // Functions', start);
        vm.runInContext(emotionSource.slice(start, end), harness.context);
        btnAll.onclick();
        if (stale === 'character') harness.state.character = 'Bob';
        else if (stale === 'away-and-back') {
            harness.state.character = 'Bob';
            harness.context.resetSelectedEmotionsForCharacterChange();
            harness.state.character = 'Alice';
            harness.context.resetSelectedEmotionsForCharacterChange();
        }
        else if (stale === 'configure') harness.context.restoreStateFromWidgets();
        else if (stale === 'removed') harness.node.onRemoved();
        if (stale !== 'current') harness.state.selectedEmotions.add('authored-draft');
        const previousWrites = harness.writes.length;
        confirm();
        assert.deepEqual([...harness.state.selectedEmotions], stale === 'current' ? ['happy'] : ['authored-draft']);
        assert.equal(harness.writes.length - previousWrites, stale === 'current' ? 1 : 0);
    });
}

test('Emotion custom image keeps the last file when FileReader callbacks arrive backwards', async () => {
    const harness = emotionModalContext();
    const fileInput = harness.find(harness.backdrop, element => element.type === 'file');
    const first = {}, second = {};
    fileInput.files = [first]; fileInput.onchange();
    fileInput.files = [second]; fileInput.onchange();
    harness.readers[1].result = 'data:image/png;base64,SECOND'; harness.readers[1].onload();
    harness.readers[0].result = 'data:image/png;base64,FIRST'; harness.readers[0].onload();
    const operation = harness.save.onclick();
    assert.equal(harness.requests[0].image_data, 'data:image/png;base64,SECOND');
    harness.resolve({ ok: true, json: async () => ({ emotion: { safe_name: 'happy' } }) });
    await operation;
});

for (const removed of [false, true]) {
    test(`Emotion catalog bootstrap keeps a saved custom emotion after ${removed ? 'removal' : 'a late inventory response'}`, async () => {
        const harness = emotionModalContext();
        let resolveCatalog;
        const catalog = new Promise(done => { resolveCatalog = done; });
        const timers = [], renders = [];
        const originalFetch = harness.context.api.fetchApi;
        Object.assign(harness.context, {
            receivedSerializedConfig: true, characterFetchToken: 0, characterListRefreshToken: 0,
            setTimeout(callback) { timers.push(callback); },
            renderEmotions() { renders.push('emotions'); }, renderCostumes() { renders.push('costumes'); },
            updateButtonState() { renders.push('buttons'); },
            refreshCharacterList() {}, fetchCharacterData() {},
        });
        harness.context.api.fetchApi = (route, request) => route === '/vnccs/get_emotions'
            ? catalog : originalFetch(route, request);
        const start = emotionSource.indexOf('                // Initial Load');
        const end = emotionSource.indexOf('                // Hook callback', start);
        vm.runInContext(emotionSource.slice(start, end), harness.context);
        const operation = harness.save.onclick();
        harness.resolve({ ok: true, json: async () => ({ emotion: { safe_name: 'happy', description: 'saved expression' } }) });
        await operation;
        assert.deepEqual(harness.state.emotions.map(emotion => emotion.safe_name), ['happy']);
        if (removed) {
            vm.runInContext('registerCleanup(node, () => { disposed = true; });', harness.context);
            harness.node.onRemoved();
        }
        const previousRenders = renders.length;
        resolveCatalog({ ok: true, json: async () => ({ Basic: [{ safe_name: 'smile', description: 'catalog expression' }] }) });
        await new Promise(setImmediate);
        assert.deepEqual([...harness.state.selectedEmotions], ['happy']);
        assert.deepEqual([...harness.state.emotions].map(emotion => emotion.safe_name), removed ? ['happy'] : ['smile', 'happy']);
        assert.equal(renders.length > previousRenders, !removed);
    });
}

test('removed Emotion Studio ignores initialization timers before restoring state or sending character requests', async () => {
    const node = {}, state = { character: 'Alice', gen: {}, selectedCostumes: new Set(), selectedEmotions: new Set() };
    const requests = [], restores = [];
    const context = createWidgetContext({ ui: motionUI, node, state, disposed: false, characterFetchToken: 0, characterListRefreshToken: 0,
        charWidget: null, modelWidget: null, generationSettingsWidget: null, styleWidget: null,
        costumesDataWidget: null, emotionsDataWidget: null,
        syncGenerationControls() { restores.push('generation'); }, renderPoseSelection() { restores.push('poses'); },
        api: { fetchApi(route) { requests.push(route); return Promise.resolve({ ok: true, json: async () => [] }); } },
        setCharacterOptions() { restores.push('characters'); return ''; },
    });
    const cleanupStart = commonSource.indexOf('export function registerCleanup');
    const cleanupEnd = commonSource.indexOf('// ── DOM Widget Width Sync', cleanupStart);
    const section = (start, end) => emotionSource.slice(emotionSource.indexOf(start), emotionSource.indexOf(end, emotionSource.indexOf(start)));
    vm.runInContext(commonSource.slice(cleanupStart, cleanupEnd).replaceAll('export ', '')
        + '\nconst beginSelectionRequest = createRequestGuard(node); registerCleanup(node, () => { disposed = true; characterFetchToken++; characterListRefreshToken++; });\n'
        + section('                const refreshCharacterList = async', '                if (charWidget) {')
        + section('                function restoreStateFromWidgets()', '                node._vnccsEmotionRestoreFromWidgets')
        + '\nthis.refreshList = refreshCharacterList;', context);
    node.onRemoved();
    context.restoreStateFromWidgets();
    await context.refreshList({ fetchData: true });
    assert.deepEqual(requests, []);
    assert.deepEqual(restores, []);
});

test('Emotion download caller keeps new jobs polling and removes its shared client with the node', async () => {
    let resolveOldStatus;
    const oldStatus = new Promise(done => { resolveOldStatus = done; });
    const timers = new Map(), node = {}, ccState = { downloadStatus: {} };
    let reads = 0;
    const response = data => ({ ok: true, json: async () => data });
    const context = createWidgetContext({ ui: motionUI, node, ccState, disposed: false, CC_REPO_ID: 'public/catalog',
        ccHasRequiredFamilies: () => true, ccStatusKey: () => 'model', renderControlCenterCards() {},
        setInterval(callback) { timers.set(1, callback); return 1; }, clearInterval: id => timers.delete(id),
        api: { fetchApi: async route => route === '/vnccs/manager/status?repo_id=public%2Fcatalog'
            ? ++reads === 1 ? oldStatus : response({ model: { status: 'queued' } })
            : response({ status: 'queued' }) },
    });
    const cleanupStart = commonSource.indexOf('export function registerCleanup');
    const cleanupEnd = commonSource.indexOf('// ── DOM Widget Width Sync', cleanupStart);
    const callerStart = emotionSource.indexOf('                const { fetchConfig: fetchCcConfig');
    const callerEnd = emotionSource.indexOf('                const ensureAnimaDefaultAux', callerStart);
    vm.runInContext(commonSource.slice(cleanupStart, cleanupEnd).replaceAll('export ', '')
        + emotionSource.slice(callerStart, callerEnd)
        + '\nthis.download = ccDownloadEntry; this.start = startCcPolling;', context);
    context.start();
    const pending = timers.get(1)();
    await context.download('models', { name: 'New emotion model' });
    resolveOldStatus(response({}));
    await pending;
    assert.equal(timers.size, 1);
    assert.equal(ccState.downloadStatus.model.status, 'queued');
    await timers.get(1)();
    assert.equal(reads, 2);
    node.onRemoved();
    assert.equal(timers.size, 0);
    context.start();
    assert.equal(timers.size, 0);
});

function generatorContext(graph = { _nodes: [], setDirtyCanvas() {} }) {
    const context = createWidgetContext({ ui: motionUI, app: { graph, registerExtension(extension) { this.extension = extension; } } });
    vm.runInContext(generatorSource.replace(/^import .*;\n/gm, '')
        .replace(/new URL\(([^,]+), import\.meta\.url\)\.href/g, '$1')
        + '\nthis.Widget = CharacterGeneratorWidget; this.readData = readData; this.writeData = writeData;', context);
    return context;
}

for (const [type, flags, savedPreview, legacyPreview] of [
    ['VNCCS_CharacterGenerator', {}, 'bg_remove', 'pose_generation'],
    ['VNCCS_CharacterCloneGenerator', { isClone: true }, 'original_upscaler', 'original_pose_generation'],
    ['VNCCS_ClothesGenerator', { isClothes: true }, 'upscaler', 'pose_generation'],
    ['VNCCS_EmotionsGenerator', { isEmotions: true }, 'emotion_0001_bg_remove', 'emotion_0001_bg_remove'],
]) {
    for (const restore of ['workflow', 'matching-cache', 'legacy']) {
        test(`${type} configure restores ${restore} preview ownership`, async () => {
            const cached = new Map();
            const context = generatorContext();
            Object.assign(context, {
                localStorage: { getItem: key => cached.get(key), setItem: (key, value) => cached.set(key, value) },
                syncDOMWidgetWidthSoon() {},
            });
            class Node {}
            await context.app.extension.beforeRegisterNodeDef(Node, { name: type });
            const node = new Node();
            node.id = 17;
            node.graph = { extra: {} };
            node.widgets = [{ name: 'widget_data', value: '{}' }];
            let disposedViewer = 0;
            const widget = Object.assign(Object.create(context.Widget.prototype), {
                node, ...flags, data: context.readData(node), stageState: {},
                syncCharacterSourceData() {}, syncModelResolution() {},
                renderSettings() {}, renderPreview() {}, renderChain() {}, renderViewer() {},
                viewer: { open: true, overlay: { remove() {} }, dispose() { disposedViewer++; } },
                viewerFocus: { centerNormX: 0.2 }, restoredViewer: { open: true, index: 2 },
            });
            node._vnccsCharacterGeneratorWidget = widget;
            widget.stages = widget.currentStages();
            widget.selectedPreview = widget.defaultPreviewStage();
            widget.userSelectedPreview = false;
            for (const [key] of widget.stages) widget.stageState[key] = { status: 'done', images: ['Alice'], message: 'Alice finished' };
            const saved = { character_name: 'Bob' };
            if (flags.isEmotions) saved.emotion_pairs = [{ costume: 'Hat', emotion: 'smile' }];
            if (restore !== 'legacy') saved.ui = { selected_preview: savedPreview, user_selected_preview: true };
            node.widgets[0].value = JSON.stringify(saved);
            if (restore === 'matching-cache') {
                cached.set(vm.runInContext('scopedKey', context)(widget.storageKey()), JSON.stringify({ version: 2,
                    settings: JSON.stringify(context.readData(node)), selectedPreview: savedPreview,
                    userSelectedPreview: true, stageState: { [savedPreview]: { status: 'done', images: ['Bob'], message: 'Bob finished' } },
                    viewer: { open: false },
                }));
            }
            node.onConfigure();
            assert.equal(widget.selectedPreview, restore === 'legacy' ? legacyPreview : savedPreview);
            assert.equal(widget.userSelectedPreview, restore !== 'legacy');
            assert.equal(widget.viewer, null);
            assert.equal(widget.viewerFocus, null);
            assert.equal(disposedViewer, 1);
            if (restore === 'matching-cache') {
                assert.deepEqual(Array.from(widget.currentImages()), ['Bob']);
            } else {
                assert.equal(widget.restoredViewer?.open ?? false, false);
                assert.ok(Object.values(widget.stageState).every(stage => stage.status === 'waiting' && stage.images === null));
            }
            widget.persistUI();
            const serialized = { widgets_values: ['old data'] };
            node.onSerialize(serialized);
            const outgoing = JSON.parse(serialized.widgets_values[0]);
            assert.equal(outgoing.character_name, 'Bob');
            assert.equal(outgoing.ui.selected_preview, widget.selectedPreview);
            assert.equal(outgoing.ui.user_selected_preview, widget.userSelectedPreview);
        });
    }
}

test('generator restores legacy SAM switches without enabling false string values', async () => {
    const context = generatorContext();
    class Element {
        constructor() { this.children = []; this.dataset = {}; }
        append(...items) { this.children.push(...items); }
        addEventListener() {}
    }
    Object.assign(context, { syncDOMWidgetWidthSoon() {},
        document: { createElement: () => new Element() }, setHelpText() {},
    });
    class Node {}
    await context.app.extension.beforeRegisterNodeDef(Node, { name: 'VNCCS_EmotionsGenerator' });
    const node = new Node();
    node.widgets = [{ name: 'widget_data', value: '{}' }];
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        node, isEmotions: true, stages: [], stageState: {},
        syncCharacterSourceData() {}, syncStagesFromData() {}, syncModelResolution() {},
        restoreBrowserState() {}, renderSettings() {}, renderPreview() {}, renderChain() {},
        protectNativeControl() {},
    });
    node._vnccsCharacterGeneratorWidget = widget;
    for (const [value, enabled] of [['false', false], ['0', false], ['off', false],
        [false, false], ['true', true], ['1', true], [true, true]]) {
        node.widgets[0].value = JSON.stringify({ emotion_generation: { use_sam_model: value } });
        node.onConfigure();
        assert.equal(widget.data.emotion_generation.use_sam, enabled);
        const control = widget.field('emotion_generation', 'use_sam', 'Use SAM', 'checkbox');
        assert.equal(control.children[0].checked, enabled);
        const serialized = { widgets_values: ['stale'] };
        node.onSerialize(serialized);
        const saved = JSON.parse(serialized.widgets_values[0]);
        assert.equal(saved.emotion_generation.use_sam, enabled);
        assert.equal(saved.emotion_generation.use_sam_model, value);

        node.widgets[0].value = JSON.stringify({ emotion_generation: { use_sam: value } });
        node.onConfigure();
        assert.equal(widget.field('emotion_generation', 'use_sam', 'Use SAM', 'checkbox').children[0].checked, enabled);
        node.onSerialize(serialized);
        assert.equal(JSON.parse(serialized.widgets_values[0]).emotion_generation.use_sam, value);
    }
});

async function generatorRegenerateLifecycle() {
    const context = generatorContext();
    const timers = new Map();
    let nextTimer = 0, writes = 0, renders = 0;
    const responses = [], listeners = new Map();
    Object.assign(context, {
        setInterval(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
        clearInterval(id) { timers.delete(id); },
        syncDOMWidgetWidthSoon() {}, watchConnection() {},
        console: { warn() {} },
    });
    const cleanupStart = commonSource.indexOf('export function registerCleanup');
    vm.runInContext(commonSource.slice(cleanupStart, commonSource.indexOf('// Each loader owns a guard.', cleanupStart))
        .replace('export ', ''), context);
    class Node {}
    await context.app.extension.beforeRegisterNodeDef(Node, { name: 'VNCCS_CharacterGenerator' });
    const node = new Node();
    node.id = 17;
    node.type = 'VNCCS_CharacterGenerator';
    node.graph = { extra: {} };
    node.widgets = [{ name: 'widget_data', value: '{}', callback() { writes++; } }];
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        node, data: { character_name: 'Alice', ui: { selected_preview: 'bg_remove' } },
        stages: [['pose_generation'], ['bg_remove']],
        selectedPreview: 'bg_remove', userSelectedPreview: true,
        stageState: { pose_generation: { status: 'done', images: ['Alice'] }, bg_remove: { status: 'done' } },
        syncCharacterSourceData() {}, syncModelResolution() {}, syncStagesFromData() {}, restoreBrowserState() {},
        renderSettings() {}, renderPreview() { renders++; }, renderChain() { renders++; }, saveBrowserState() {},
    });
    node._vnccsCharacterGeneratorWidget = widget;
    context.api.addEventListener = (name, callback) => listeners.set(name, callback);
    context.api.removeEventListener = name => listeners.delete(name);
    widget.bindEvents();
    context.api.fetchApi = () => new Promise(done => { responses.push(done); });
    const pending = widget.regenerateFrom('pose_generation');
    return { context, node, widget, timers, pending,
        complete(ok, index = 0) { responses[index]({ ok, status: ok ? 200 : 500, text: async () => '{"error":"Old request failed"}' }); },
        emit(requestId, revision, status = 'running') {
            listeners.get('vnccs.character_generator.stage')?.({ detail: {
                node_id: String(node.id), scope: widget.progressScope(), request_id: requestId,
                epoch: 'server', revision, stage: 'pose_generation', status, images: [requestId || 'normal-run'],
            } });
        },
        counts: () => [writes, renders],
    };
}

for (const lifecycle of ['configure', 'removed']) {
    for (const ok of [false, true]) {
        test(`Regenerate ignores a late ${ok ? 'success' : 'failure'} after node ${lifecycle}`, async () => {
            const harness = await generatorRegenerateLifecycle();
            if (lifecycle === 'configure') {
                harness.node.widgets[0].value = '{"character_name":"Bob","ui":{"selected_preview":"bg_remove"}}';
                harness.node.onConfigure();
                harness.widget.stageState.pose_generation = { status: 'done', images: ['Bob'] };
            } else harness.node.onRemoved();
            const saved = harness.node.widgets[0].value;
            const state = JSON.stringify(harness.widget.data);
            const stages = JSON.stringify(harness.widget.stageState);
            const counts = harness.counts();
            assert.equal(harness.widget.regenerateState, null);
            assert.equal(harness.widget.regenerateTimer, null);
            assert.equal(harness.timers.size, lifecycle === 'configure' ? 1 : 0);
            harness.complete(ok);
            await harness.pending;
            assert.equal(harness.node.widgets[0].value, saved);
            assert.equal(JSON.stringify(harness.widget.data), state);
            assert.equal(JSON.stringify(harness.widget.stageState), stages);
            assert.deepEqual(harness.counts(), counts);
        });
    }
}

test('configured generator rejects old same-scope Regenerate events and snapshots while accepting its next run', async () => {
    const harness = await generatorRegenerateLifecycle();
    const oldId = harness.widget.data.ui.progress_request_id;
    harness.node.widgets[0].value = '{"character_name":"Bob","ui":{"selected_preview":"bg_remove"}}';
    harness.node.onConfigure();
    harness.widget.stageState.pose_generation = { status: 'done', images: ['Bob'] };
    const saved = harness.node.widgets[0].value;
    const stages = JSON.stringify(harness.widget.stageState);
    harness.emit(oldId, 1);
    assert.equal(JSON.stringify(harness.widget.stageState), stages);
    assert.equal(harness.node.widgets[0].value, saved);
    harness.context.api.fetchApi = async () => ({ ok: true, status: 200, json: async () => ({ snapshot: {
        scope: harness.widget.progressScope(), node_id: String(harness.node.id), request_id: oldId,
        epoch: 'server', revision: 2, stages: { pose_generation: { status: 'done', images: ['Alice'] } },
    } }) });
    await harness.widget.refreshProgress();
    assert.equal(JSON.stringify(harness.widget.stageState), stages);
    assert.equal(harness.node.widgets[0].value, saved);
    harness.widget.prepareQueuedRun();
    harness.emit(undefined, 3);
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), ['normal-run']);
    harness.emit(oldId, 4, 'done');
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), ['normal-run']);
    harness.complete(false);
    await harness.pending;
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), ['normal-run']);
    harness.emit(undefined, 5, 'done');
    assert.equal(harness.widget.stageState.pose_generation.status, 'done');
    assert.equal(harness.widget._configuredRegenerateRequestId, null);
    harness.node.onRemoved();
});

test('configure retains its restored request ID but rejects that settled operation\'s late progress', async () => {
    const harness = await generatorRegenerateLifecycle();
    const oldId = harness.widget.data.ui.progress_request_id;
    harness.complete(true);
    await harness.pending;
    harness.node.widgets[0].value = JSON.stringify({ character_name: 'Bob', ui: { progress_request_id: oldId } });
    harness.node.onConfigure();
    harness.widget.stageState.pose_generation = { status: 'done', images: ['Bob'] };
    const saved = harness.node.widgets[0].value;
    harness.emit(oldId, 6, 'done');
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), ['Bob']);
    assert.equal(harness.node.widgets[0].value, saved);
    assert.equal(JSON.parse(saved).ui.progress_request_id, oldId);
    harness.context.api.fetchApi = async () => ({ ok: true, status: 200, json: async () => ({ snapshot: {
        scope: harness.widget.progressScope(), node_id: String(harness.node.id), request_id: oldId,
        epoch: 'server', revision: 7, stages: { pose_generation: { status: 'done', images: ['Alice'] } },
    } }) });
    await harness.widget.refreshProgress();
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), ['Bob']);
    assert.equal(harness.node.widgets[0].value, saved);
    harness.node.onRemoved();
});

test('configure invalidates all overlapping Regenerate requests and leaves a fresh operation intact', async () => {
    const harness = await generatorRegenerateLifecycle();
    const firstId = harness.widget.data.ui.progress_request_id;
    const second = harness.widget.regenerateFrom('bg_remove');
    const secondId = harness.widget.data.ui.progress_request_id;
    harness.node.widgets[0].value = '{"character_name":"Bob","ui":{}}';
    harness.node.onConfigure();
    harness.widget.stageState.pose_generation = { status: 'done', images: ['Bob'] };
    harness.emit(firstId, 1);
    harness.emit(secondId, 2);
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), ['Bob']);
    const fresh = harness.widget.regenerateFrom('pose_generation');
    const freshId = harness.widget.data.ui.progress_request_id;
    harness.complete(false);
    harness.complete(true, 1);
    await harness.pending;
    await second;
    assert.equal(harness.widget._regenerateRequestPending, freshId);
    harness.emit(freshId, 3);
    assert.deepEqual(Array.from(harness.widget.stageState.pose_generation.images), [freshId]);
    harness.complete(true, 2);
    await fresh;
    assert.equal(harness.widget.regenerateState, null);
    harness.node.onRemoved();
    assert.equal(harness.timers.size, 0);
});

function generatorSettingsContext() {
    class Element {
        constructor() { this.children = []; this.dataset = {}; }
        append(...items) { this.children.push(...items); }
        appendChild(item) { this.children.push(item); return item; }
        replaceChildren(...items) { this.children = items; }
        addEventListener() {}
        remove() { this.removed = true; }
    }
    const context = generatorContext();
    context.document.createElement = () => new Element();
    context.requestAnimationFrame = () => 0;
    const node = { widgets: [{ name: 'widget_data', value: '{}' }] };
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        node, title: 'Generator', root: new Element(),
        data: { character_name: 'Alice', pose_sampler: { cfg: 2 }, ui: { progress_request_id: 'old' } },
        generatorSettingsGroups: () => [{ title: 'Sampler', fields: [{ section: 'pose_sampler', key: 'cfg' }] }],
        createGeneratorSettingsField(field, draft) { draft[field.section][field.key] = 5; return new Element(); },
        protectNativeControl() {}, syncModelResolution() {}, saveBrowserState() {}, renderSettings() {},
    });
    widget.openGeneratorSettings();
    const find = element => element.textContent === 'Apply' ? element : element.children.map(find).find(Boolean);
    return { node, widget, apply: find(widget.modalEl) };
}

for (const stale of ['configure', 'closed', 'reopened', 'removed']) {
    test(`Generator Settings ignores an obsolete Apply callback after ${stale}`, () => {
        const { node, widget, apply } = generatorSettingsContext();
        if (stale === 'configure') widget.data = { character_name: 'Bob', pose_sampler: { cfg: 9 } };
        else if (stale === 'closed') widget.closeModal();
        else if (stale === 'reopened') widget.openGeneratorSettings();
        else widget._disposed = true;
        const before = JSON.stringify(widget.data), saved = node.widgets[0].value;
        apply.onclick();
        assert.equal(JSON.stringify(widget.data), before);
        assert.equal(node.widgets[0].value, saved);
    });
}

test('Generator Settings applies edited controls without reverting live progress or character state', () => {
    const { node, widget, apply } = generatorSettingsContext();
    widget.data.ui.progress_request_id = 'current-run';
    widget.data.ui.selected_preview = 'bg_remove';
    widget.data.character_name = 'Bob';
    apply.onclick();
    const saved = JSON.parse(node.widgets[0].value);
    assert.equal(saved.pose_sampler.cfg, 5);
    assert.equal(saved.character_name, 'Bob');
    assert.equal(saved.ui.progress_request_id, 'current-run');
    assert.equal(saved.ui.selected_preview, 'bg_remove');
});

for (const [surface, section, key, original, raw, step, expected] of [
    ['settings', 'pose_sampler', 'steps', 30, '2.5', 1, 30],
    ['settings', 'pose_sampler', 'seed', 721, '2.5', 1, 721],
    ['settings', 'pose_sampler', 'steps', 30, '1e2', 1, 100],
    ['settings', 'pose_sampler', 'cfg', 4, '0', 0.01, 0],
    ['settings', 'bg_remove', 'softness', 0.15, '0.251', 0.01, 0.251],
    ['settings', 'upscaler', 'resolution', 128, '130.5', 2, 128],
    ['inline', 'upscaler', 'resolution', 128, '130.5', 2, 128],
    ['inline', 'upscaler', 'resolution', 128, '130', 2, 130],
    ['face-detailer', 'emotion_generation', 'bbox_dilation', 50, '2.5', 1, 50],
    ['face-detailer', 'emotion_generation', 'task_batch_size', 0, '2.5', 1, 0],
    ['face-detailer', 'emotion_generation', 'drop_size', 10, '2.5', 1, 10],
    ['face-detailer', 'emotion_generation', 'bbox_dilation', 50, '37', 1, 37],
    ['face-detailer', 'emotion_generation', 'bbox_threshold', 0.5, '0.251', 0.01, 0.251],
]) {
    test(`generator ${surface} ${key}=${raw} retains the exact executable value in its workflow`, () => {
        class Element {
            constructor() { this.children = []; this.dataset = {}; this.classList = { add() {} }; }
            append(...items) { this.children.push(...items); }
            appendChild(item) { this.children.push(item); return item; }
            replaceChildren(...items) { this.children = items; }
            addEventListener() {}
            remove() {}
        }
        const context = generatorContext();
        context.document.createElement = () => new Element();
        context.requestAnimationFrame = () => 0;
        context.setHelpText = () => {};
        const node = { widgets: [{ name: 'widget_data', value: JSON.stringify({ [section]: { [key]: original } }) }] };
        const field = { section, key, label: key, type: 'number', min: 0, max: 16384, step };
        const widget = Object.assign(Object.create(context.Widget.prototype), {
            node, data: context.readData(node), root: new Element(), fieldDrafts: new Map(),
            generatorSettingsGroups: () => [{ title: 'Processing', fields: [field] }],
            syncCharacterSourceData() {}, syncModelResolution() {}, rememberModelResolution() {},
            protectNativeControl() {}, saveBrowserState() {}, renderSettings() {},
        });
        const find = (element, condition) => condition(element) ? element : element.children.map(child => find(child, condition)).find(Boolean);
        let input;
        if (surface === 'settings') {
            widget.openGeneratorSettings();
            input = find(widget.modalEl, element => element.type === 'number');
        } else if (surface === 'face-detailer') {
            input = find(widget.faceDetailerNumberField(key, key, field), element => element.type === 'number');
        } else {
            input = find(widget.field(section, key, key, 'number', field), element => element.type === 'number');
        }
        input.value = raw;
        input.oninput();
        if (surface === 'settings') find(widget.modalEl, element => element.textContent === 'Apply').onclick();
        else input.onblur();
        assert.equal(widget.data[section][key], expected);
        assert.equal(JSON.parse(node.widgets[0].value)[section][key], expected);
    });
}

for (const [key, text, number, valid] of [
    ['steps', '', NaN, false], ['cfg', '', NaN, false],
    ['steps', '2.5', 2.5, false], ['steps', '-1', -1, false],
    ['cfg', '21', 21, false], ['steps', '1e1', 10, true], ['cfg', '0', 0, true],
    ['seed', '', NaN, false], ['seed', '2.5', 2.5, false], ['seed', '1e3', 1000, true], ['seed', '-1', -1, true],
]) {
    test(`Emotion numeric controls preserve authored settings for ${key}=${text}`, () => {
        const writes = [], state = { gen: { steps: 30, cfg: 4, seed: 721 } };
        const original = state.gen[key];
        const context = createWidgetContext({ ui: motionUI, state,
            document: { createElement: () => ({}) },
            updateGenerationValue(name, value) { state.gen[name] = value; writes.push(JSON.stringify(state.gen)); },
            syncGenerationControls() {},
        });
        const start = emotionSource.indexOf('                function createNumberInput(');
        const end = emotionSource.indexOf('                function createSelectInput(', start);
        vm.runInContext(emotionSource.slice(start, end), context);
        const input = context.createNumberInput(key, key === 'seed' ? undefined : key === 'steps' ? 1 : 0,
            key === 'seed' ? undefined : key === 'steps' ? 100 : 20, key === 'cfg' ? 0.1 : 1);
        input.value = text; input.valueAsNumber = number; input.checkValidity = () => valid;
        input.onchange();
        assert.equal(state.gen[key], valid ? number : original);
        assert.equal(writes.length, valid ? 1 : 0);
        if (valid) assert.equal(JSON.parse(writes[0])[key], number);
    });
}

for (const [text, number, valid] of [['', NaN, false], ['3', 3, false], ['0', 0, true], ['1.5', 1.5, true]]) {
    test(`Emotion LoRA strength keeps valid state for ${text}`, () => {
        const state = { gen: { lora_stack: [{ name: 'authored.safetensors', strength: 0.75 }] } }, writes = [];
        const context = createWidgetContext({ ui: motionUI, state, i: 0,
            document: { createElement: () => ({}) }, syncGenerationControls() {},
            saveGenerationSettings() { writes.push(JSON.stringify(state.gen)); },
            updateGenerationValue() { assert.fail('LoRA must update its own row'); },
        });
        const start = emotionSource.indexOf('                function createNumberInput(');
        const end = emotionSource.indexOf('                function createSelectInput(', start);
        const strengthStart = emotionSource.indexOf('                    const strength = createNumberInput(');
        const strengthEnd = emotionSource.indexOf('                    setHelpText(strength', strengthStart);
        vm.runInContext(emotionSource.slice(start, end) + emotionSource.slice(strengthStart, strengthEnd)
            + '\nthis.strength = strength;', context);
        context.strength.value = text; context.strength.valueAsNumber = number; context.strength.checkValidity = () => valid;
        context.strength.onchange();
        assert.equal(state.gen.lora_stack[0].strength, valid ? number : 0.75);
        assert.equal(state.gen.lora_stack[0].name, 'authored.safetensors');
        assert.equal(writes.length, valid ? 1 : 0);
    });
}

for (const clone of [false, true]) {
    test(`generator follows its own character branch (${clone ? 'clone' : 'creator'})`, () => {
        const type = clone ? 'CharacterCloner' : 'CharacterCreatorV2';
        const character = (id, name, nsfw, sex) => ({ id, type, widgets: [{ name: 'widget_data', value: JSON.stringify({
            character: name, character_info: { nsfw, sex },
        }) }] });
        const alice = character(1, 'Alice', true, 'female'), bob = character(2, 'Bob', false, 'male');
        const relay = { id: 3, inputs: [{ link: 2 }, { link: 3 }] };
        const node = { id: 4, inputs: [{ link: 1 }] };
        const graph = { _nodes: [alice, bob, relay, node], links: {
            1: { origin_id: 3 }, 2: { origin_id: 2 }, 3: { origin_id: 4 },
        }, getNodeById(id) { return this._nodes.find(item => item.id === id); } };
        node.graph = graph;
        const context = generatorContext({ _nodes: [alice] });
        const widget = Object.assign(Object.create(context.Widget.prototype), { node, data: {}, isClone: clone });
        assert.equal(widget.syncCharacterSourceData(), true);
        assert.equal(widget.data.character_name, 'Bob');
        assert.equal(widget.data.gender, 'male');
        assert.match(widget.mannequinURL(), /Mannequin_Male\.png$/);
        if (clone) assert.equal(widget.data.nsfw_enabled, false);
        node.inputs = [];
        assert.equal(widget.syncCharacterSourceData(), undefined);
        assert.equal(widget.data.character_name, 'Bob');
    });
}

test('clothes generator follows its own Designer and synchronizes character and costume before queueing', () => {
    const designer = (id, character, costume) => ({ id, type: 'ClothesDesigner', widgets: [{
        name: 'widget_data', value: JSON.stringify({ character, costume }),
    }] });
    const alice = designer(1, 'Alice', 'Coat'), bob = designer(2, 'Bob', 'Hat');
    const relay = { id: 3, inputs: [{ link: 2 }, { link: 3 }] };
    const node = { id: 4, inputs: [{ link: 1 }] };
    const graph = { _nodes: [alice, bob, relay, node], links: {
        1: { origin_id: 3 }, 2: { origin_id: 2 }, 3: { origin_id: 4 },
    }, getNodeById(id) { return this._nodes.find(item => item.id === id); } };
    node.graph = graph;
    const context = generatorContext({ _nodes: [alice] });
    const widget = Object.assign(Object.create(context.Widget.prototype), { node, data: {}, isClothes: true });
    assert.equal(widget.syncCharacterSourceData(), true);
    assert.equal(widget.data.character_name, 'Bob');
    assert.equal(widget.data.costume, 'Hat');
    bob.widgets[0].value = JSON.stringify({ character: 'Alice', costume: 'Dress' });
    assert.equal(widget.syncCharacterSourceData(), true);
    assert.equal(widget.data.character_name, 'Alice');
    assert.equal(widget.data.costume, 'Dress');
    assert.equal(widget.syncCharacterSourceData(), false);
});

for (const value of ['{broken', 'null', '[]', '{"common":null}', '{"bg_remove":null}',
    '{"vae_decode":[]}', '{"emotion_generation":"broken"}', '{"ui":1}']) {
    test(`damaged generator state remains recoverable: ${value}`, async () => {
        const context = generatorContext();
        const node = { widgets: [{ name: 'widget_data', value }], type: 'VNCCS_CharacterGenerator' };
        const data = context.readData(node);
        assert.match(node._vnccsGeneratorStateError, /saved data is preserved/);
        context.writeData(node, data);
        assert.equal(node.widgets[0].value, value);
        node.widgets[0].value = '{"pose_generation":{"target_size":2048}}';
        const recovered = context.readData(node);
        assert.equal(node._vnccsGeneratorStateError, undefined);
        context.writeData(node, recovered);
        assert.equal(JSON.parse(node.widgets[0].value).pose_generation.target_size, 2048);
    });
}

test('generator construction leaves serialized input untouched', () => {
    class Element {
        constructor() { this.style = { setProperty() {} }; this.classList = { toggle() {}, add() {} }; this.dataset = {}; }
        append() {}
        appendChild() {}
        setAttribute() {}
    }
    const context = generatorContext();
    Object.assign(context, {
        document: { getElementById: () => null, createElement: () => new Element(), head: new Element() },
        enableMiddleMouseCanvasPan() {}, attachHelpTooltips() {}, syncDOMWidgetWidthSoon() {},
        registerCleanup() {}, createSakuraInteractions: () => ({ dispose() {} }), ResizeObserver: class { observe() {} }, setTimeout: () => 0,
    });
    for (const name of ['protectNativeControl', 'renderSettings', 'renderPreview', 'renderChain',
        'bindEvents', 'bindModelResolutionSync', 'loadNodeDefs', 'loadSeedvrAssets']) context.Widget.prototype[name] = () => {};
    for (const original of ['{"pose_generation":{"target_size":2048}}', '{"bg_remove":null}', '{"emotion_generation":"broken"}']) {
        const node = { widgets: [{ name: 'widget_data', value: original }], addDOMWidget() {} };
        const widget = new context.Widget(node);
        assert.equal(node.widgets[0].value, original);
        assert.equal(node.widgets[0].options.dynamicPrompts, false);
        if (!original.includes('target_size')) {
            assert.match(node._vnccsGeneratorStateError, /settings must be an object/);
            assert.equal(typeof widget.data.bg_remove, 'object');
            assert.equal(typeof widget.data.emotion_generation, 'object');
        }
    }
});

test('Emotion Studio restores configured state before immediate serialization or queueing', async () => {
    const context = createWidgetContext({ ui: motionUI,
        app: { registerExtension(extension) { this.extension = extension; } },
        SAKURA_THEME_CSS: '', syncDOMWidgetWidth() {}, setTimeout: () => 0,
        document: { createElement: () => ({}), head: { appendChild() {} } },
    });
    vm.runInContext(emotionSource.replace(/^import .*;\n/gm, ''), context);
    class Node {}
    await context.app.extension.beforeRegisterNodeDef(Node, { name: 'EmotionGeneratorV2' }, context.app);
    const node = new Node();
    const calls = [];
    node._vnccsEmotionApplySerializedInfo = info => calls.push(['apply', info.widgets_values[0]]);
    node._vnccsEmotionRestoreFromWidgets = () => calls.push(['restore']);
    node.onConfigure({ widgets_values: ['QI2'] });
    assert.deepEqual(calls, [['apply', 'QI2'], ['restore']]);
});

test('Emotion Studio writes latest hidden values into the outgoing workflow', () => {
    const context = createWidgetContext();
    const widgets = ['generation_model', 'generation_settings', 'prompt_style', 'character', 'costumes_data', 'emotions_data']
        .map(name => ({ name, value: 'old' }));
    const names = ['modelWidget', 'generationSettingsWidget', 'styleWidget', 'charWidget', 'costumesDataWidget', 'emotionsDataWidget'];
    names.forEach((name, index) => { context[name] = widgets[index]; });
    Object.assign(context, { node: { widgets }, persistAllState() { widgets.forEach(item => { item.value = `new:${item.name}`; }); } });
    const start = emotionSource.indexOf('                const origSerialize = node.onSerialize;');
    const end = emotionSource.indexOf('\n                // Fix layout', start);
    vm.runInContext(emotionSource.slice(start, end), context);
    const serialized = { widgets_values: widgets.map(item => item.value) };
    context.node.onSerialize(serialized);
    assert.deepEqual(serialized.widgets_values, widgets.map(item => item.value));
});

test('muted and bypassed Emotion Studios do not block the queue or randomize seeds', async () => {
    let submitted = 0, seeds = 0;
    const context = createWidgetContext({ ui: motionUI, app: {
        graph: { _nodes: [2, 4].map(mode => ({ type: 'EmotionGeneratorV2', mode,
            _randomizeSeedIfNeeded() { seeds++; }, _validateBeforeQueue: () => false })) },
        queuePrompt: async () => { submitted++; }, registerExtension(extension) { this.extension = extension; },
    }, SAKURA_THEME_CSS: '', document: { createElement: () => ({}), head: { appendChild() {} } } });
    vm.runInContext(emotionSource.replace(/^import .*;\n/gm, ''), context);
    await context.app.extension.setup();
    await context.app.queuePrompt();
    assert.equal(submitted, 1);
    assert.equal(seeds, 0);
});

test('Emotion queue transports the visible random seed without persisting its execution marker', async () => {
    const node = { id: 2, type: 'EmotionGeneratorV2', _validateBeforeQueue: () => true };
    const state = { gen: { generation_mode: 'anima', seed: 0, seed_mode: 'randomize', mode_settings: {} } };
    const generationSettingsWidget = { name: 'generation_settings', value: '{}' };
    const generationEls = { seed: {} };
    const app = {
        graph: { _nodes: [node] }, registerExtension(extension) { this.extension = extension; },
        async graphToPrompt() {
            return { workflow: { widgets_values: [generationSettingsWidget.value] }, output: {
                2: { class_type: 'EmotionGeneratorV2', inputs: { generation_settings: generationSettingsWidget.value } },
            } };
        }, async queuePrompt() { return this.graphToPrompt(); },
    };
    let nextSeed = 721;
    const context = createWidgetContext({ ui: motionUI,
        app, node, state, generationSettingsWidget, generationEls,
        modelWidget: {}, styleWidget: null, disposed: false, restoreErrors: new Map(),
        initialSharedSeed: 0, GENERATION_DEFAULTS: { mode_settings: { illustrious: {}, anima: {}, qi2: {} } },
        syncSelectedPoseIndicesToGenerationSettings() {}, markNodeDirty() {},
        generateRandomSeed: () => nextSeed++,
        window: { dispatchEvent() {} }, CustomEvent: class {},
        SAKURA_THEME_CSS: '', document: { createElement: () => ({}), head: { appendChild() {} } },
    });
    const section = (start, end) => emotionSource.slice(emotionSource.indexOf(start), emotionSource.indexOf(end, emotionSource.indexOf(start)));
    vm.runInContext(emotionSource.replace(/^import .*;\n/gm, ''), context);
    vm.runInContext(section('                function commitWidget(', '                function selectedPoseIndexList()')
        + section('                function saveGenerationSettings()', '                function setGenerationMode(')
        + section('                node._randomizeSeedIfNeeded =', '                const ccNormalize ='), context);
    await app.extension.setup();
    for (const expected of [721, 722]) {
        const result = await app.queuePrompt();
        const execution = JSON.parse(result.output[2].inputs.generation_settings);
        const workflow = JSON.parse(result.workflow.widgets_values[0]);
        assert.equal(Number(generationEls.seed.value), expected);
        assert.equal(execution.seed, expected);
        assert.equal(execution._vnccs_seed_resolved, true);
        assert.equal(workflow.seed, expected);
        assert.equal(workflow.seed_mode, 'randomize');
        assert.equal(workflow._vnccs_seed_resolved, undefined);
        assert.equal(JSON.parse(generationSettingsWidget.value)._vnccs_seed_resolved, undefined);
        for (const profile of Object.values(workflow.mode_settings)) assert.equal(profile.seed, expected);
    }
});

for (const value of ['{broken', 'null', '[]', '{"mode_settings":[]}', '{"mode_settings":null}',
    '{"mode_settings":{"qi2":null}}', '{"qi2_cache":[]}', '{"lora_stack":{}}',
    '{"lora_stack":[null]}', '{"mode_settings":{"anima":{"lora_stack":"damaged"}}}', '{"turbo_previous_settings":0}']) {
    test(`Emotion Studio preserves damaged settings on every write: ${value}`, () => {
        const settings = { name: 'generation_settings', value };
        const costumes = { name: 'costumes_data', value: '["Dress"]' };
        const context = createWidgetContext({ ui: motionUI, generationSettingsWidget: settings, modelWidget: null, GENERATION_DEFAULTS: { mode_settings: {} },
            restoreErrors: new Map(), disposed: false, markNodeDirty() {} });
        const start = emotionSource.indexOf('                function parseGenerationSettings()');
        const end = emotionSource.indexOf('                // State', start);
        const writeStart = emotionSource.indexOf('                function commitWidget(');
        const writeEnd = emotionSource.indexOf('                function selectedPoseIndexList()', writeStart);
        vm.runInContext(emotionSource.slice(start, end) + emotionSource.slice(writeStart, writeEnd), context);
        context.parseGenerationSettings();
        context.commitWidget(settings, '{}');
        context.commitWidget(costumes, '[]');
        assert.equal(settings.value, value);
        assert.equal(costumes.value, '["Dress"]');
        settings.value = '{"seed":721}';
        assert.equal(context.parseGenerationSettings().seed, 721);
        context.commitWidget(settings, '{"seed":722}');
        assert.equal(settings.value, '{"seed":722}');
    });
}

test('Emotion Studio retains corrupt selections until a valid configure restores them', () => {
    const selected = { name: 'costumes_data', value: '{broken' };
    const context = createWidgetContext({ ui: motionUI, restoreErrors: new Map() });
    const start = emotionSource.indexOf('                function parseSelection(');
    const end = emotionSource.indexOf('                // State', start);
    vm.runInContext(emotionSource.slice(start, end), context);
    context.parseSelection(selected);
    assert.equal(context.restoreErrors.has('costumes_data'), true);
    assert.equal(selected.value, '{broken');
    selected.value = '["Dress"]';
    assert.deepEqual([...context.parseSelection(selected)], ['Dress']);
    assert.equal(context.restoreErrors.size, 0);
});

test('removed generator cannot install download polling after its POST completes', async () => {
    let resolve, timers = 0;
    const pending = new Promise(done => { resolve = done; });
    const context = generatorContext();
    context.api.fetchApi = () => pending;
    context.setInterval = () => { timers++; return 1; };
    const widget = Object.assign(Object.create(context.Widget.prototype), { seedvrDownloads: {}, renderSettings() {} });
    const work = widget.downloadSeedvrAsset('models', { name: 'seedvr2_3b_fp16.safetensors' });
    widget._disposed = true;
    resolve({ ok: true, json: async () => ({ status: 'queued' }) });
    await work;
    assert.equal(timers, 0);
});

for (const loader of ['loadNodeDefs', 'loadSeedvrAttentionInfo', 'loadSeedvrAssets']) {
    test(`removed generator ignores late ${loader} results and performs no follow-up requests`, async () => {
        const context = generatorContext();
        const cleanupStart = commonSource.indexOf('export function registerCleanup');
        vm.runInContext(commonSource.slice(cleanupStart, commonSource.indexOf('// Each loader owns a guard.', cleanupStart))
            .replace('export ', ''), context);
        let resolve, requests = 0, renders = 0, writes = 0, modals = 0;
        const pending = new Promise(done => { resolve = done; });
        context.api.fetchApi = () => { requests++; return pending; };
        context.api.addEventListener = context.api.removeEventListener = () => {};
        context.watchConnection = () => {};
        context.setInterval = () => 0;
        context.clearInterval = () => {};
        const raw = '{"upscaler":{"mode":"seedvr","attention_mode":"sdpa"}}';
        const node = { id: 17, widgets: [{ name: 'widget_data', value: raw, callback() { writes++; } }] };
        const widget = Object.assign(Object.create(context.Widget.prototype), {
            node, data: JSON.parse(raw), nodeDefs: {}, seedvrAssets: null,
            seedvrAttention: { current: null, available: ['sdpa'] },
            renderSettings() { renders++; }, validateNativeSeedvr() { modals++; },
        });
        widget.bindEvents();
        const work = widget[loader]();
        node.onRemoved();
        const count = requests;
        resolve({ ok: true, json: async () => ({ current: 'flash_attn_3', available: ['sdpa', 'flash_attn_3'] }) });
        await work;
        assert.equal(requests, count);
        assert.equal(node.widgets[0].value, raw);
        assert.equal(widget.data.upscaler.attention_mode, 'sdpa');
        assert.equal(widget.seedvrAttention.current, null);
        assert.equal(widget.seedvrAssets, null);
        assert.equal(Object.keys(widget.nodeDefs).length, 0);
        assert.deepEqual([renders, writes, modals], [0, 0, 0]);
    });
}

test('download transport failure becomes a visible error', async () => {
    const context = generatorContext();
    context.api.fetchApi = async () => { throw new Error('Offline'); };
    const widget = Object.assign(Object.create(context.Widget.prototype), { seedvrDownloads: {}, renderSettings() {} });
    await widget.downloadSeedvrAsset('models', { name: 'model' });
    assert.equal(widget.seedvrDownloads['models:model'].status, 'error');
    assert.equal(widget.seedvrDownloads['models:model'].message, 'Offline');
});

test('Emotion Studio resize preserves the existing lifecycle hook and receiver', () => {
    const calls = [];
    const node = { onResize(size) { calls.push({ node: this, size }); } };
    const context = createWidgetContext({ ui: motionUI, node, syncDOMWidgetWidth() {}, applySize() {}, requestAnimationFrame: () => 0 });
    const start = emotionSource.indexOf('                // Apply size on explicit resize too');
    const end = emotionSource.indexOf('                // Helper', start);
    vm.runInContext(emotionSource.slice(start, end), context);
    const size = [920, 650];
    node.onResize(size);
    assert.equal(calls[0].node, node);
    assert.equal(calls[0].size, size);
});

test('damaged generator settings block Regenerate before requests and preserve serialized bytes', async () => {
    let requests = 0;
    const context = generatorContext();
    context.api.fetchApi = async () => { requests++; throw new Error('No request may be sent'); };
    const original = '{broken';
    const node = { widgets: [{ name: 'widget_data', value: original }] };
    const data = context.readData(node);
    const messages = [];
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        node, data, stages: [['pose_generation', 'Pose']], showModal(...args) { messages.push(args); },
    });
    await widget.regenerateFrom('pose_generation', 0);
    assert.equal(requests, 0);
    assert.equal(node.widgets[0].value, original);
    assert.equal(data.regenerate_from, undefined);
    assert.match(messages[0][1], /saved data is preserved/);
});

test('emotion generator follows its pipe branch through reroutes and ignores cyclic links', () => {
    const studio = (id, character, mode) => ({ id, type: 'EmotionGeneratorV2', widgets: [
        { name: 'character', value: character }, { name: 'generation_model', value: mode },
        { name: 'costumes_data', value: '["Coat","Original"]' }, { name: 'emotions_data', value: '["smile"]' },
    ] });
    const alice = studio(1, 'Alice', 'Anima'), bob = studio(2, 'Bob', 'QI2');
    const relay = { id: 3, inputs: [{ link: 2 }, { link: 3 }] };
    const node = { id: 4, inputs: [{ name: 'pipe', link: 1 }] };
    node.graph = { _nodes: [alice, bob, relay, node], links: {
        1: { origin_id: 3 }, 2: { origin_id: 4 }, 3: { origin_id: 2 },
    }, getNodeById(id) { return this._nodes.find(item => item.id === id); } };
    const context = generatorContext({ _nodes: [alice] });
    const widget = Object.assign(Object.create(context.Widget.prototype), { node, data: {}, isEmotions: true });
    assert.equal(widget.connectedEmotionStudioNode(), bob);
    assert.equal(widget.connectedEmotionStudioMode(), 'qi2');
    assert.equal(widget.syncEmotionStudioSourceData(), true);
    assert.equal(widget.data.character_name, 'Bob');
    assert.deepEqual(Array.from(widget.currentStages(), item => item[0]), ['emotion_0001_bg_remove', 'emotion_0002_bg_remove']);
    relay.inputs = [{ link: 2 }];
    assert.equal(widget.connectedEmotionStudioNode(), null);
});

for (const fail of [false, true]) {
    test(`removed Emotion Studio cannot restart model polling after download ${fail ? 'failure' : 'success'}`, async () => {
        let finish, renders = 0, polls = 0, requests = 0;
        const response = new Promise((resolve, reject) => { finish = fail ? reject : resolve; });
        const context = createWidgetContext({ ui: motionUI, disposed: false,
            api: { fetchApi() { requests++; return response; } },
            ccStatusKey: (category, entry) => `${category}:${entry.name}`, CC_REPO_ID: 'catalog',
            ccState: { downloadStatus: {} }, renderControlCenterCards() { renders++; }, startCcPolling() { polls++; },
        });
        const start = emotionSource.indexOf('                const ccDownloadEntry = async');
        const end = emotionSource.indexOf('                const ensureAnimaDefaultAux', start);
        vm.runInContext(emotionSource.slice(start, end) + '\nthis.downloadEntry = ccDownloadEntry;', context);
        const work = context.downloadEntry('models', { name: 'Anima' });
        context.disposed = true;
        finish(fail ? new Error('Offline') : { ok: true, json: async () => ({ status: 'queued' }) });
        await work;
        await context.downloadEntry('models', { name: 'Anima' });
        assert.equal(requests, 1);
        assert.equal(polls, 0);
        assert.equal(renders, 1);
        assert.equal(context.ccState.downloadStatus['models:Anima'].status, 'queued');
    });
}

for (const available of [[], ['Bob']]) {
    test(`Emotion Studio inventory preserves unavailable saved character (${available.length} available)`, () => {
        const character = { name: 'character', value: 'Alice', options: {} };
        const select = { value: 'Alice', appendChild() {} };
        const node = { widgets: [character] };
        const context = createWidgetContext({ ui: motionUI, state: { character: 'Alice' }, charWidget: character,
            charSelect: select, node, spritePreviewNavigator: { hideNav() {} },
            Option: class { constructor(label, value) { this.text = label; this.value = value; } },
            commitWidget(widget, value) { widget.value = value; }, persistAllState() { character.value = context.state.character; },
            modelWidget: null, generationSettingsWidget: null, styleWidget: null, costumesDataWidget: null, emotionsDataWidget: null,
        });
        const start = emotionSource.indexOf('                const setCharacterOptions =');
        const end = emotionSource.indexOf('                let characterListRefreshToken', start);
        const serializeStart = emotionSource.indexOf('                const origSerialize = node.onSerialize;');
        const serializeEnd = emotionSource.indexOf('                // Fix layout', serializeStart);
        vm.runInContext(emotionSource.slice(start, end) + '\nthis.setOptions = setCharacterOptions;\n'
            + emotionSource.slice(serializeStart, serializeEnd), context);
        assert.equal(context.setOptions(available), 'Alice');
        assert.equal(character.value, 'Alice');
        assert.equal(select.value, 'Alice');
        const saved = { widgets_values: ['old'] };
        node.onSerialize(saved);
        assert.equal(saved.widgets_values[0], 'Alice');
        assert.equal(character.options.values[0], 'Alice');
    });
}

for (const available of [[], ['Naked', 'Shelter'], ['Original']]) {
    test(`Emotion Studio excludes absent Original and preserves saved user costumes (${available.join(', ') || 'empty'})`, async () => {
        class Element {
            constructor() { this.children = []; this.dataset = {}; }
            set innerHTML(value) { this.children = []; }
            appendChild(child) { this.children.push(child); }
        }
        const costumes = { name: 'costumes_data', value: '["Original","Coat"]' };
        const costumesList = new Element();
        const context = createWidgetContext({ ui: motionUI, state: { character: 'Alice', costumes: [], selectedCostumes: new Set(['Original', 'Coat']) },
            disposed: false, characterFetchToken: 0, costumesDataWidget: costumes, restoreErrors: new Map(),
            spritePreviewNavigator: null, costumesList, markNodeDirty() {},
            document: { createElement: () => new Element() },
            api: { fetchApi: async () => ({ ok: true, json: async () => available }) },
        });
        const parseStart = emotionSource.indexOf('                function parseSelection(');
        const parseEnd = emotionSource.indexOf('                // State', parseStart);
        const commitStart = emotionSource.indexOf('                function commitWidget(');
        const commitEnd = emotionSource.indexOf('                function selectedPoseIndexList()', commitStart);
        const updateStart = emotionSource.indexOf('                function updateCostumesData()');
        const updateEnd = emotionSource.indexOf('                function updateEmotionsData()', updateStart);
        const fetchStart = emotionSource.indexOf('                async function fetchCharacterData(');
        const fetchEnd = emotionSource.indexOf('                // Initial Load', fetchStart);
        const renderStart = emotionSource.indexOf('                function renderCostumes()');
        const renderEnd = emotionSource.indexOf('                function renderEmotions()', renderStart);
        vm.runInContext(emotionSource.slice(parseStart, parseEnd) + emotionSource.slice(commitStart, commitEnd)
            + emotionSource.slice(updateStart, updateEnd) + emotionSource.slice(renderStart, renderEnd)
            + emotionSource.slice(fetchStart, fetchEnd), context);
        const labels = () => costumesList.children.map(label => label.children[1].innerText);
        context.renderCostumes();
        assert.deepEqual(labels(), ['Coat (missing sprites)']);
        assert.equal(costumes.value, '["Original","Coat"]', 'Rendering before inventory must preserve saved data');
        await context.fetchCharacterData('Alice');
        const expected = available.includes('Original') ? ['Original', 'Coat'] : ['Coat'];
        assert.deepEqual([...context.state.selectedCostumes], expected);
        assert.deepEqual(labels(), [...available, 'Coat (missing sprites)']);
        assert.equal(costumes.value, JSON.stringify(expected));
        context.updateCostumesData();
        assert.equal(costumes.value, JSON.stringify(expected));
        if (available.includes('Original')) {
            const checkbox = costumesList.children[0].children[0].children[0];
            assert.equal(checkbox.checked, true);
            checkbox.checked = false;
            checkbox.onchange();
            assert.equal(costumes.value, '["Coat"]');
        }
        costumes.value = '["Original","Coat"]';
        context.api.fetchApi = async () => ({ ok: false, status: 503 });
        await context.fetchCharacterData('Alice');
        assert.equal(costumes.value, '["Original","Coat"]', 'Failed inventory must preserve saved data');
    });
}

for (const removed of [false, true]) {
    test(`Emotion Studio preview preserves selected poses and ignores removal (${removed})`, async () => {
        const cleanups = [], loaders = [], settings = { name: 'generation_settings', value: '{}' };
        const state = { character: 'Alice', gen: { generation_mode: 'anima', seed: 1 },
            selectedPoseIndices: new Set([2, 7]), poseCount: 0 };
        const context = createWidgetContext({ ui: motionUI, state, node: {}, disposed: false, URLSearchParams,
            registerCleanup(node, cleanup) { cleanups.push(cleanup); },
            Image: class { constructor() { loaders.push(this); } },
            spritePreviewNavigator: null, charImg: { style: {} }, charPlaceholder: { style: {} },
            charPreviewLoading: null, spriteNav: null, spritePrevBtn: null, spriteNextBtn: null, spriteCount: null,
            poseSelectionSummary: null, poseSelectionGrid: null,
            generationSettingsWidget: settings, modelWidget: null, styleWidget: null,
            initialSharedSeed: 1, GENERATION_DEFAULTS: { mode_settings: { anima: {}, illustrious: {}, qi2: {} } },
            commitWidget(widget, value) { if (widget) widget.value = value; },
            window: { dispatchEvent() {} }, CustomEvent: class {},
            api: { fetchApi: async () => ({ ok: true, json: async () => ({ count: 2 }) }) },
        });
        const commonStart = commonSource.indexOf('export function createSpritePreviewNavigator(');
        const commonEnd = commonSource.indexOf('// ── DOM Widget Canvas Navigation', commonStart);
        const selectionStart = emotionSource.indexOf('                function selectedPoseIndexList()');
        const selectionEnd = emotionSource.indexOf('                function persistAllState()', selectionStart);
        const saveStart = emotionSource.indexOf('                function saveGenerationSettings()');
        const saveEnd = emotionSource.indexOf('                function setGenerationMode(', saveStart);
        const previewStart = emotionSource.indexOf('                spritePreviewNavigator = createSpritePreviewNavigator(');
        const previewEnd = emotionSource.indexOf('                leftCol.appendChild(charSection);', previewStart);
        vm.runInContext(commonSource.slice(commonStart, commonEnd).replace('export ', '')
            + emotionSource.slice(selectionStart, selectionEnd) + emotionSource.slice(saveStart, saveEnd)
            + emotionSource.slice(previewStart, previewEnd), context);
        await context.spritePreviewNavigator.load('Alice', { random: false });
        assert.equal(loaders.length, 1);
        if (removed) { context.disposed = true; cleanups.forEach(cleanup => cleanup()); }
        loaders[0].onload();
        assert.deepEqual([...state.selectedPoseIndices], [2, 7]);
        if (removed) {
            assert.equal(settings.value, '{}');
            assert.equal(state.poseCount, 0);
        } else {
            assert.deepEqual(JSON.parse(settings.value).selected_pose_indices, [2, 7]);
            assert.equal(context.selectedPoseCount(), 1);
        }
    });
}
