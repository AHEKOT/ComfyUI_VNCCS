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
