import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = readFileSync(new URL('../web/vnccs_character_generator.js', import.meta.url), 'utf8')
    .replace(/^import .*;\n/gm, '')
    .replace(/new URL\(([^,]+), import\.meta\.url\)\.href/g, '$1');

function gallery({ images = [], previewAspect, count = 4 } = {}) {
    const context = createWidgetContext({ app: { registerExtension() {} } });
    vm.runInContext(source + '\nthis.Widget = CharacterGeneratorWidget;', context);
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        selectedPreview: 'pose_generation', stageState: { pose_generation: { preview_aspect: previewAspect } },
    });
    const grid = {
        isConnected: true, style: {}, clientWidth: 2600, clientHeight: 700,
        children: Array.from({ length: count }, (_, index) => ({
            style: {}, dataset: {}, offsetLeft: 0, offsetTop: 0,
            _previewImage: images[index], _previewSource: index < 2 ? '/view?pending' : null,
            _previewAspect: index < 2 ? undefined : 2 / 3,
            classList: { contains: name => name === 'is-placeholder' },
        })),
    };
    return { widget, grid };
}

for (const scenario of [
    { name: 'pending previews and untouched mannequins' },
    { name: 'decoded previews and mannequins', images: [{ naturalWidth: 1024, naturalHeight: 1024 }] },
    { name: 'mixed preview dimensions', images: [{ naturalWidth: 1024, naturalHeight: 1024 }, { naturalWidth: 640, naturalHeight: 1536 }] },
    { name: 'backend batch aspect', previewAspect: 640 / 1536, images: [{ naturalWidth: 1024, naturalHeight: 1024 }] },
]) {
    test(`gallery keeps equal card dimensions for ${scenario.name} through resize`, () => {
        const { widget, grid } = gallery(scenario);
        for (const [width, height] of [[2600, 700], [900, 700], [600, 1200]]) {
            grid.clientWidth = width;
            grid.clientHeight = height;
            widget.layoutPreviewGrid(grid);
            const first = grid.children[0].style;
            for (const tile of grid.children) {
                assert.equal(tile.style.width, first.width);
                assert.equal(tile.style.height, first.height);
                assert.ok(parseFloat(tile.style.width) > 0 && parseFloat(tile.style.width) <= width);
                assert.ok(parseFloat(tile.style.height) > 0 && parseFloat(tile.style.height) <= height);
            }
            const expectedAspect = scenario.previewAspect || (scenario.images?.length ? 1 : 2 / 3);
            assert.ok(Math.abs(parseFloat(first.width) / parseFloat(first.height) - expectedAspect) < 1e-9);
        }
    });
}

function backgroundWidget() {
    const context = createWidgetContext({ app: { registerExtension() {} } });
    vm.runInContext(source + '\nthis.Widget = CharacterGeneratorWidget; this.defaults = DEFAULT_DATA;', context);
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        isBackground: true, data: JSON.parse(JSON.stringify(context.defaults)),
    });
    return widget;
}

test('background generator reuses generator controls with exactly three stages and no character operations', () => {
    const widget = backgroundWidget();
    assert.deepEqual(Array.from(widget.currentStages(), pair => Array.from(pair)), [
        ['source_upscaler', 'Source Upscaler'], ['background_generation', 'Background Generation'], ['upscaler', 'Variant Upscaler'],
    ]);
    assert.equal(widget.defaultPreviewStage(), 'source_upscaler');
    assert.equal(widget.mannequinURL(), '');
    const groups = widget.generatorSettingsGroups();
    assert.equal(groups[0].title, 'Background Generation');
    assert.deepEqual(Array.from(groups[0].fields, field => field.key), ['target_size']);
    assert.ok(groups.some(group => group.title === 'Native SeedVR2'));
    assert.ok(groups.every(group => !/Pose Generation|Clothes|Chroma|SAM|Detailer/.test(group.title)));
    assert.ok(groups.every(group => group.fields.every(field => field.section !== 'bg_remove')));
});

test('background source upscale still requires SeedVR when optional final upscale is off', () => {
    const widget = backgroundWidget(); widget.data.upscaler.mode = 'off'; widget.nativeSeedvrAvailable = false;
    assert.equal(widget.validateNativeSeedvr(), false);
    widget.isBackground = false;
    assert.equal(widget.validateNativeSeedvr(), true);
});

test('background gallery uses landscape frames before the first image arrives', () => {
    const { widget, grid } = gallery(); widget.isBackground = true;
    grid.children.forEach(tile => { tile._previewImage = null; tile._previewSource = null; });
    widget.layoutPreviewGrid(grid);
    assert.ok(Math.abs(parseFloat(grid.children[0].style.width) / parseFloat(grid.children[0].style.height) - 16 / 9) < 1e-9);
});

test('background generator reads variant labels through reroutes without owning Creator state', () => {
    const widget = backgroundWidget();
    const state = { background_id: 'a'.repeat(32), selected_image_id: 'b'.repeat(32), editor: { variants: [{ name: 'Rainy dawn' }] } };
    const sourceNode = { id: 1, type: 'VNCCS_BackgroundCreator', widgets: [{ name: 'background_data', value: JSON.stringify(state) }] };
    const reroute = { id: 2, inputs: [{ link: 10 }] };
    const graph = { links: { 10: { origin_id: 1 }, 11: { origin_id: 2 } }, getNodeById: id => [sourceNode, reroute].find(node => node.id === id) };
    widget.node = { graph, inputs: [{ link: 11 }] };
    assert.equal(widget.syncBackgroundSourceData(), true);
    assert.deepEqual(Array.from(widget.data.background_variants), ['Rainy dawn']);
    assert.equal(widget.data.background_id, state.background_id);
    assert.equal(widget.data.background_source_image_id, state.selected_image_id);
    assert.equal(sourceNode.widgets[0].value, JSON.stringify(state));
    assert.equal(widget.syncBackgroundSourceData(), false);
    state.editor.variants = []; sourceNode.widgets[0].value = JSON.stringify(state);
    assert.equal(widget.syncBackgroundSourceData(), true);
    assert.deepEqual(Array.from(widget.data.background_variants), []);
    assert.equal(widget.syncBackgroundSourceData(), false);
    state.selected_image_id = 'c'.repeat(32); sourceNode.widgets[0].value = JSON.stringify(state);
    assert.equal(widget.syncBackgroundSourceData(), true);
    assert.equal(widget.data.background_source_image_id, state.selected_image_id);
});
