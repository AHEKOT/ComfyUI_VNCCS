import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = readFileSync(new URL('../web/vnccs_character_creator_v2.js', import.meta.url), 'utf8');
const section = (start, end) => {
    const first = source.indexOf(start), last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first, start);
    return source.slice(first, last);
};

function setup(mode, authored, selectedModel = 'authored-model.safetensors') {
    const widget = { name: 'widget_data', value: '{}' };
    const state = { character: 'Alice', character_info: { name: 'Alice' }, gen_settings: {
        generation_mode: mode, diffusion_model_name: selectedModel, ...authored, mode_settings: {
            anima: { clip_name: 'saved-inactive-anima-clip', vae_name: 'saved-inactive-anima-vae' },
            qi2: { clip_name: 'saved-inactive-qi2-clip', vae_name: 'saved-inactive-qi2-vae' },
        },
    } };
    const catalog = Object.fromEntries(['Anima', 'QI2'].map(family => [family, {
        models: { local_path: `models/diffusion_models/${family}-default-model.safetensors`, kind: family, type: 'unet' },
        clip: { local_path: `models/text_encoders/${family}-default-clip.safetensors` },
        vae: { local_path: `models/vae/${family}-default-vae.safetensors` },
    }]));
    const context = createWidgetContext({ state, node: { widgets: [widget] },
        els: { animaModelCards: { style: {} }, qi2ModelCards: { style: {} } },
        ANIMA_CLIP_NAME: 'anima-fallback-clip', ANIMA_VAE_NAME: 'anima-fallback-vae',
        QI2_CLIP_NAME: 'qi2-fallback-clip', QI2_VAE_NAME: 'qi2-fallback-vae',
        ccState: { config: { models: [] } }, localAssets: {},
        ccEntries: (category, family) => [catalog[family][category]],
        ccFirstEntry: (category, family) => catalog[family][category],
        ccRelPath: entry => entry.local_path.split('/').slice(2).join('/'),
        ccType: entry => entry.type, ccKind: entry => entry.kind.toLowerCase(),
        mergeCcAndLocalEntries: entries => entries, renderModelPicker() {},
        modelPickerOpen: {}, downloadAnimaBundle() {}, downloadQi2Bundle() {}, closeOverhaulHelp: null,
        isQi2TurboModel: () => false, syncGenerationControls() {},
        syncBackgroundForGenerationMode() {}, ensureLoraStack() {},
        getModeProfile: active => state.gen_settings.mode_settings[active] ||= {},
        cloneSettingsValue: value => value,
        MODE_SETTING_KEYS: { anima: ['clip_name', 'vae_name', 'diffusion_model_name'], qi2: ['clip_name', 'vae_name', 'diffusion_model_name'] },
    });
    vm.runInContext(section('const saveCurrentGenerationModeValues =', 'const applyGenerationProfile =')
        + section('const saveState =', 'const clearPreviewHandlers =')
        + section('const ensureAnimaDefaultAux =', 'const downloadAnimaBundle =')
        + section('const renderControlCenterCards =', 'const isSelectedCcAssetInstalled =')
        + '\nthis.renderCards = renderControlCenterCards; this.save = saveState;', context);
    context.selectCcAsset = (key, value) => { state.gen_settings[key] = value; context.save(); };
    return { context, catalog, state, stored: () => JSON.parse(widget.value) };
}

for (const mode of ['anima', 'qi2']) {
    for (const authored of [{ clip_name: 'custom-clip.safetensors', vae_name: 'custom-vae.safetensors' },
        { clip_name: 'custom-clip.safetensors', vae_name: '' }, { clip_name: '', vae_name: '' }]) {
        test(`Creator ${mode} catalog rendering preserves auxiliary assets ${JSON.stringify(authored)}`, () => {
            const h = setup(mode, authored);
            const inactive = mode === 'anima' ? 'qi2' : 'anima';
            const inactiveProfile = structuredClone(h.state.gen_settings.mode_settings[inactive]);
            h.context.renderCards(); h.context.save();
            const stored = h.stored().gen_settings;
            const family = mode === 'anima' ? 'Anima' : 'QI2';
            assert.equal(stored.clip_name, authored.clip_name || `${family}-default-clip.safetensors`);
            assert.equal(stored.vae_name, authored.vae_name || `${family}-default-vae.safetensors`);
            assert.equal(stored.mode_settings[mode].clip_name, stored.clip_name);
            assert.equal(stored.mode_settings[mode].vae_name, stored.vae_name);
            assert.deepEqual(stored.mode_settings[inactive], inactiveProfile);
            h.catalog[family].clip.local_path = 'models/text_encoders/refreshed-clip.safetensors';
            h.catalog[family].vae.local_path = 'models/vae/refreshed-vae.safetensors';
            h.context.renderCards(); h.context.save();
            assert.equal(h.stored().gen_settings.clip_name, stored.clip_name);
            assert.equal(h.stored().gen_settings.vae_name, stored.vae_name);
        });
    }
    test(`Creator ${mode} fills an empty model from its active family`, () => {
        const h = setup(mode, { clip_name: '', vae_name: '' }, '');
        h.context.renderCards(); h.context.save();
        assert.equal(h.stored().gen_settings.diffusion_model_name,
            `${mode === 'anima' ? 'Anima' : 'QI2'}-default-model.safetensors`);
    });
}

test('Creator Illustrious rendering keeps auxiliary state until its own family is selected', () => {
    const h = setup('illustrious', { clip_name: 'retained-clip', vae_name: 'retained-vae' });
    h.context.renderCards(); h.context.save();
    assert.equal(h.stored().gen_settings.clip_name, 'retained-clip');
    assert.equal(h.stored().gen_settings.vae_name, 'retained-vae');
});
