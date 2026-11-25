import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

test('resolved seed transport preserves workflow preferences, original snapshots, wrappers, and inactive nodes', async () => {
    const random = JSON.stringify({ gen_settings: { seed: 23, seed_mode: 'randomize' } });
    const emotion = JSON.stringify({ seed: 71, seed_mode: 'randomize' });
    const entry = (class_type, name, value) => ({ class_type, inputs: { [name]: value, pipe: ['upstream', 0] } });
    const original = { workflow: { widgets_values: [random, emotion] }, unrelated: 'preserved', output: {
        '1': entry('CharacterCreatorV2', 'widget_data', random),
        '2': entry('EmotionGeneratorV2', 'generation_settings', emotion),
        '3': entry('CharacterCreatorV2', 'widget_data', random),
        '4': entry('CharacterCreatorV2', 'widget_data', random),
        '5': entry('CharacterCreatorV2', 'widget_data', 'broken JSON'),
        '6': entry('CharacterCreatorV2', 'widget_data', JSON.stringify({ gen_settings: { seed: 0, seed_mode: 'fixed' } })),
        '7': entry('OtherNode', 'widget_data', random),
        '8': entry('CharacterCreatorV2', 'widget_data', 'null'),
    } };
    let calls = 0;
    const app = { graph: { _nodes: [{ id: 1 }, { id: 2 }, { id: 3, mode: 2 }, { id: 4, mode: 4 }] },
        async graphToPrompt(...args) { calls++; assert.equal(this, app); assert.deepEqual(args, ['queue']); return original; } };
    const { installResolvedSeedTransport: install } = createWidgetContext({ app });
    install(app, ['CharacterCreatorV2']);
    const wrapper = app.graphToPrompt;
    install(app, ['EmotionGeneratorV2'], 'generation_settings');
    install(app, ['CharacterCreatorV2']);
    assert.equal(app.graphToPrompt, wrapper);
    const result = await app.graphToPrompt('queue');
    assert.equal(calls, 1);
    assert.equal(result.workflow, original.workflow);
    assert.equal(result.unrelated, 'preserved');
    for (const [id, name] of [['1', 'widget_data'], ['2', 'generation_settings']]) {
        const transported = JSON.parse(result.output[id].inputs[name]);
        assert.equal(transported._vnccs_seed_resolved, true);
        assert.equal((transported.gen_settings || transported).seed_mode, 'randomize');
        assert.equal(result.output[id].inputs.pipe, original.output[id].inputs.pipe);
        assert.equal(JSON.parse(original.output[id].inputs[name])._vnccs_seed_resolved, undefined);
    }
    for (const id of ['3', '4', '5', '6', '7', '8']) assert.equal(result.output[id], original.output[id]);
    assert.equal(original.workflow.widgets_values[0], random);
    assert.equal(original.workflow.widgets_values[1], emotion);
});

test('Creator queue sends the displayed random seed while workflow and widget retain random mode without a marker', async () => {
    const source = readFileSync(new URL('../web/vnccs_character_creator_v2.js', import.meta.url), 'utf8');
    const between = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
    const state = { character: 'Alice', character_info: { name: 'Alice' },
        gen_settings: { seed: 0, seed_mode: 'randomize', mode_settings: { illustrious: { seed: 0, seed_mode: 'randomize' } } } };
    const widget = { name: 'widget_data', value: JSON.stringify(state) };
    const node = { id: 1, type: 'CharacterCreatorV2', widgets: [widget] };
    const submissions = [], els = { seed: {} };
    const app = { graph: { _nodes: [node] },
        async graphToPrompt() { return { workflow: { widgets_values: [widget.value] },
            output: { '1': { class_type: node.type, inputs: { widget_data: widget.value } } } }; },
        async queuePrompt() { assert.equal(this, app); submissions.push(await this.graphToPrompt()); } };
    let seed = 20;
    const context = createWidgetContext({ app, node, state, els,
        generateRandomSeed: () => ++seed,
        syncBackgroundForGenerationMode() {},
        saveCurrentGenerationModeValues() { state.gen_settings.mode_settings.illustrious.seed = state.gen_settings.seed; } });
    vm.runInContext(`${between('const saveState =', 'const clearPreviewHandlers =')}
        ${between('node._randomizeSeedIfNeeded =', '// Override onSerialize')}
        globalThis.extension = { ${between('async setup()', 'async beforeRegisterNodeDef').trim().replace(/,$/, '')} };`, context);
    await context.extension.setup();
    await app.queuePrompt();
    await app.queuePrompt();
    assert.equal(els.seed.value, 22);
    for (const [index, result] of submissions.entries()) {
        const queued = JSON.parse(result.output['1'].inputs.widget_data);
        const saved = JSON.parse(result.workflow.widgets_values[0]);
        assert.equal(queued.gen_settings.seed, 21 + index);
        assert.equal(queued._vnccs_seed_resolved, true);
        assert.equal(saved.gen_settings.seed_mode, 'randomize');
        assert.equal(saved.gen_settings.mode_settings.illustrious.seed, 21 + index);
        assert.equal(saved._vnccs_seed_resolved, undefined);
    }
    assert.equal(JSON.parse(widget.value)._vnccs_seed_resolved, undefined);
});


test('Background seed transport keeps random preference and marks only queued inputs resolved', async () => {
    const raw = JSON.stringify({editor: {generation: {seed_mode: 'randomize', seed: 23}}});
    const original = {workflow: {widgets_values: [raw]}, output: {
        '7': {class_type: 'VNCCS_BackgroundCreator', inputs: {background_data: raw}},
    }};
    const app = {graph: {_nodes: [{id: 7}]}, async graphToPrompt() { return original; }};
    const { installResolvedSeedTransport: install } = createWidgetContext({app});
    install(app, ['VNCCS_BackgroundCreator'], 'background_data');
    const result = await app.graphToPrompt();
    assert.equal(JSON.parse(result.output['7'].inputs.background_data)._vnccs_seed_resolved, true);
    assert.equal(JSON.parse(result.output['7'].inputs.background_data).editor.generation.seed, 23);
    assert.equal(JSON.parse(result.workflow.widgets_values[0])._vnccs_seed_resolved, undefined);
    assert.equal(original.output['7'].inputs.background_data, raw);
});
