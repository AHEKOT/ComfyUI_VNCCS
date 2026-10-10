import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const catalog = JSON.parse(readFileSync(new URL('../control_center.json', import.meta.url), 'utf8'));
const qi2Models = catalog.models.filter(entry => entry.kind === 'QI2');
const turboName = 'qwen_image_2.1_turbo_int8_convrot.safetensors';

for (const [filename, stateKey] of [['vnccs_character_creator_v2', 'gen_settings'], ['vnccs_emotion_v2', 'gen']]) {
    const source = readFileSync(new URL(`../web/${filename}.js`, import.meta.url), 'utf8');
    const modelCheck = source.match(/const isQi2TurboModel = .*;/)?.[0] || '';
    test(`${filename}: QI2 catalog cards keep Turbo second`, () => {
        const merge = source.match(/const mergeCcAndLocalEntries = [\s\S]*?\n                };/)?.[0];
        assert.ok(merge);
        const context = vm.createContext({ ccRelPath: entry => entry.local_path.split('/').slice(2).join('/') });
        vm.runInContext(`const localAssetRelSet = items => new Set(items); ${merge}\nthis.merge = mergeCcAndLocalEntries;`, context);
        const cards = context.merge(qi2Models, ['local.safetensors'], 'diffusion_models', 'unet', 'QI2');
        assert.equal(cards[1].local_path.split('/').pop(), turboName);
        assert.ok(source.includes('entries: qi2Models,'));
        assert.ok(source.includes('onSelect: rel => selectQi2Model(rel)'));
    });

    test(`${filename}: selecting built-in Turbo persists base pipeline with 8/1`, () => {
        const selection = source.match(/const selectQi2Model = [\s\S]*?\n                };/)?.[0];
        assert.ok(selection);
        const profile = { diffusion_model_name: 'base.safetensors', steps: 6, cfg: 1, turbo_enabled: true };
        let saved;
        let synced = 0;
        const context = vm.createContext({
            state: { [stateKey]: profile }, QI2_TURBO_MODEL_NAME: turboName, modelPickerOpen: { qi2: true },
            ensureQi2DefaultAux() {}, syncGenerationControls() { synced++; },
            selectCcAsset(key, rel) { profile[key] = rel; saved = JSON.parse(JSON.stringify(profile)); },
        });
        vm.runInContext(`${modelCheck}\n${selection}\nthis.select = selectQi2Model;`, context);
        context.select(`QI2\\${turboName}`);
        assert.equal(saved.diffusion_model_name, `QI2\\${turboName}`);
        assert.equal(saved.steps, 8);
        assert.equal(saved.cfg, 1);
        assert.equal(saved.turbo_enabled, false);
        assert.equal(saved.turbo_previous_settings, null);
        assert.equal(context.modelPickerOpen.qi2, false);
        assert.equal(synced, 1);
        profile.steps = 10;
        context.select(`QI2\\${turboName}`);
        assert.equal(saved.steps, 10);
        context.select('qwen_image_2.1_int8_convrot.safetensors');
        assert.equal(saved.steps, 25);
        assert.equal(saved.cfg, 3);
    });

    test(`${filename}: built-in Turbo blocks Viggle callbacks while base QI2 still supports them`, () => {
        const selection = source.match(/const selectQi2Model = [\s\S]*?\n                };/)[0];
        const toggle = source.match(/const setCcTurboMode = [\s\S]*?\n                };/)[0];
        const animaToggle = source.match(/(?:const setAnimaTurboMode = |function setAnimaTurboMode)[\s\S]*?\n                };?/)[0];
        const profile = { generation_mode: 'qi2', diffusion_model_name: 'base.safetensors', steps: 25, cfg: 3, turbo_enabled: false };
        let saved;
        const context = vm.createContext({
            state: { [stateKey]: profile }, QI2_TURBO_MODEL_NAME: turboName,
            QI2_TURBO_LORA_NAME: 'viggle.safetensors', ANIMA_TURBO_LORA_NAME: 'anima.safetensors', modelPickerOpen: { qi2: false },
            ensureQi2DefaultAux() {}, syncGenerationControls() {}, renderControlCenterCards() {},
            saveState() { saved = JSON.parse(JSON.stringify(profile)); },
            saveGenerationSettings() { saved = JSON.parse(JSON.stringify(profile)); },
            selectCcAsset(key, rel) { profile[key] = rel; saved = JSON.parse(JSON.stringify(profile)); },
        });
        vm.runInContext(`${modelCheck}\n${selection}\n${animaToggle}\n${toggle}\nthis.select = selectQi2Model; this.toggle = setCcTurboMode; this.directToggle = setAnimaTurboMode;`, context);
        context.select(`QI2\\${turboName}`);
        for (const enabled of [true, false]) {
            context.toggle(enabled, 'viggle.safetensors');
            context.directToggle(enabled);
            assert.equal(profile.turbo_enabled, false);
            assert.equal(profile.steps, 8);
            assert.equal(profile.cfg, 1);
            assert.equal(saved.turbo_enabled, false);
        }
        profile.steps = 10;
        context.toggle(true, 'viggle.safetensors');
        assert.equal(profile.steps, 10);
        context.select('qwen_image_2.1_int8_convrot.safetensors');
        context.toggle(true, 'viggle.safetensors');
        assert.equal(saved.turbo_enabled, true);
        assert.equal(saved.steps, 6);
        assert.equal(saved.cfg, 1);
        context.toggle(false, 'viggle.safetensors');
        assert.equal(saved.turbo_enabled, false);
        assert.equal(saved.steps, 25);
        assert.equal(saved.cfg, 3);
    });

    test(`${filename}: built-in Turbo renders no Viggle controls and clears old cards`, () => {
        const render = source.match(/const renderModeLoraCards = [\s\S]*?\n                };/)[0];
        class Element {
            constructor() { this.children = []; this.style = {}; }
            set innerHTML(value) { this.children = []; }
            appendChild(child) { this.children.push(child); return child; }
            append(...children) { this.children.push(...children); }
        }
        const container = new Element();
        container.appendChild('old Viggle card');
        const context = vm.createContext({
            state: { [stateKey]: { diffusion_model_name: `QI2/${turboName}` } },
            QI2_TURBO_MODEL_NAME: turboName, QI2_OVERHAUL_ENTRY: { local_path: 'models/loras/overhaul.safetensors' },
            QI2_TURBO_LORA_NAME: 'viggle.safetensors',
            ccState: { config: { lora: [{ name: 'Viggle', kind: 'QI2', type: 'TurboLora', local_path: 'models/loras/viggle.safetensors' }] } },
            ccKind: entry => entry.kind?.toLowerCase(), ccType: entry => entry.type?.toLowerCase(),
            ccRelPath: entry => entry.local_path, ccResolveStatus: () => 'installed', cardStatusLabel: () => 'Installed',
            localAssets: { loras: [] }, localAssetRelSet: items => new Set(items),
            document: { createElement: () => new Element(), createTextNode: text => ({ textContent: text }) },
            buildAssetCard: () => ({ type: 'Viggle card' }), buildOverhaulCard: () => ({ type: 'Overhaul card' }),
        });
        vm.runInContext(`${modelCheck}\n${render}\nthis.render = renderModeLoraCards;`, context);
        context.render(container, 'qi2');
        assert.deepEqual(container.children.map(child => child.type), stateKey === 'gen_settings' ? ['Overhaul card'] : []);
    });
}
