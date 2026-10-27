import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

for (const filename of ['vnccs_character_creator_v2.js', 'vnccs_emotion_v2.js']) {
    const source = readFileSync(new URL('../web/' + filename, import.meta.url), 'utf8');
    const section = (a, b) => source.slice(source.indexOf(a), source.indexOf(b, source.indexOf(a)));
    const field = filename.includes('creator') ? 'gen_settings' : 'gen';
    function setup(mode) {
        const config = {};
        for (const [category, folder] of [['models', 'diffusion_models'], ['clip', 'text_encoders'], ['vae', 'vae']]) {
            config[category] = [{ name: category, local_path: `models/${folder}/${category}.safetensors`, kind: mode, status: 'missing' }];
        }
        const profile = { generation_mode: mode, clip_name: 'clip.safetensors', vae_name: 'vae.safetensors', mode_settings: {} };
        const calls = [], ccState = { config, downloadStatus: {} };
        const context = createWidgetContext({ state: { [field]: profile }, ccState, localAssets: {}, els: {}, disposed: false,
            CC_REPO_ID: 'public/models', isCcMounted: () => !context.disposed,
            renderControlCenterCards() {}, startCcPolling() {}, ensureOption() {},
            api: { fetchApi: async (_route, options) => { calls.push(JSON.parse(options.body)); return { ok: true, json: async () => ({ status: 'queued' }) }; } },
        });
        vm.runInContext(section('                const ccNormalize =', '                const ccHasRequiredFamilies =')
            + section('                const ccDownloadEntry =', field === 'gen' ? '                const ensureAnimaDefaultAux =' : '                const els =')
            + section('                const ensureAnimaDefaultAux =', '                const setCcTurboMode =')
            + '\nthis.status = ccResolveStatus; this.bundle = ' + (mode === 'anima' ? 'downloadAnimaBundle' : 'downloadQi2Bundle') + ';', context);
        return { context, config, ccState, profile, calls, model: config.models[0] };
    }
    for (const mode of ['anima', 'qi2']) {
        test(`${filename} ${mode} stays Missing until all parts are installed and retries only missing parts`, async () => {
            const h = setup(mode); h.model.status = 'installed'; h.config.vae[0].status = 'installed';
            assert.equal(h.context.status(h.model, 'models'), 'missing');
            await h.context.bundle('models', h.model);
            assert.deepEqual(h.calls.map(call => call.category), ['clip']);
            assert.equal(h.calls[0].skip_existing, true);
            for (const status of ['queued', 'downloading', 'error']) {
                h.ccState.downloadStatus.cc_clip_clip = { status, progress: 50 };
                assert.equal(h.context.status(h.model, 'models'), 'missing');
            }
            await h.context.bundle('models', h.model);
            assert.equal(h.calls.length, 2);
            h.config.clip[0].status = 'installed';
            h.ccState.downloadStatus.cc_clip_clip = { status: 'success' };
            assert.equal(h.context.status(h.model, 'models'), 'installed');
            await h.context.bundle('models', h.model);
            assert.equal(h.calls.length, 2);
        });
        test(`${filename} ${mode} accepts complete existing older weights without forcing an overwrite`, async () => {
            const h = setup(mode);
            Object.values(h.config).forEach(entries => entries[0].status = 'outdated');
            assert.equal(h.context.status(h.model, 'models'), 'installed');
            await h.context.bundle('models', h.model); assert.equal(h.calls.length, 0);
        });
        test(`${filename} ${mode} downloads all three existing catalog records`, async () => {
            const h = setup(mode); await h.context.bundle('models', h.model);
            assert.deepEqual(h.calls.map(call => call.category), ['models', 'clip', 'vae']);
            assert.ok(h.calls.every(call => call.skip_existing));
        });
        test(`${filename} ${mode} preserves a selected local encoder instead of downloading the default`, async () => {
            const h = setup(mode); h.profile.clip_name = 'custom/encoder.safetensors';
            h.context.localAssets.text_encoders = ['custom/encoder.safetensors'];
            await h.context.bundle('models', h.model);
            assert.deepEqual(h.calls.map(call => call.category), ['models', 'vae']);
            assert.equal(h.profile.clip_name, 'custom/encoder.safetensors');
        });
        test(`${filename} ${mode} keeps component choices captured before a slow main-file request`, async () => {
            const h = setup(mode); let release;
            h.context.api.fetchApi = (_route, options) => {
                h.calls.push(JSON.parse(options.body));
                return h.calls.length === 1 ? new Promise(resolve => { release = resolve; }) : Promise.resolve({ ok: true, json: async () => ({}) });
            };
            const work = h.context.bundle('models', h.model);
            h.profile.clip_name = 'a-new-selection'; h.profile.vae_name = 'another-vae';
            release({ ok: true, json: async () => ({}) }); await work;
            assert.deepEqual(h.calls.map(call => call.name), ['models', 'clip', 'vae']);
            assert.equal(h.profile.clip_name, 'a-new-selection');
        });
    }
}

test('download dialog shows missing parts and progress, retries errors, and closing leaves the download running', async () => {
    class Element {
        constructor() { this.style = {}; this.children = []; this.isConnected = true; }
        append(...children) { this.children.push(...children); }
        removeAttribute(name) { delete this[name]; }
        remove() { this.isConnected = false; }
    }
    const timers = new Map(), node = {}, parts = [{ name: 'main', status: 'installed' }, { name: 'encoder', status: 'missing' }];
    let downloads = 0;
    const context = createWidgetContext({ document: { createElement: () => new Element() },
        registerCleanup(_node, fn) { node.cleanup = fn; },
        setInterval(fn) { timers.set(1, fn); return 1; }, clearInterval(id) { timers.delete(id); },
    });
    const show = (_title, content, actions) => ({ overlay: new Element(), content: content(),
        buttons: actions.map(action => ({ disabled: false, async click() { if (!this.disabled && action.action) return action.action(); } })) });
    const dialog = context.showModelDownloadDialog(node, show, () => parts, async () => {
        downloads++; parts[1] = { name: 'encoder', status: 'downloading', progress: 40 };
    }, false);
    assert.match(dialog.content.children[1].textContent, /encoder: missing/);
    await dialog.buttons[0].click(); assert.equal(dialog.buttons[0].disabled, true);
    assert.equal(dialog.content.children[2].value, 70);
    parts[1] = { name: 'encoder', status: 'error', message: 'Interrupted transfer' };
    timers.get(1)(); assert.match(dialog.content.children[1].textContent, /Interrupted transfer/);
    await dialog.buttons[0].click(); assert.equal(downloads, 2);
    dialog.overlay.remove(); timers.get(1)(); assert.equal(timers.size, 0);
    assert.equal(parts[1].status, 'downloading'); node.cleanup();
});
