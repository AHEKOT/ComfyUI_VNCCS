import assert from "node:assert/strict";
import test from "node:test";
import { fitPreviewGrid, packPreviewRows } from "../web/preview_grid_layout.mjs";

test("mixed-width previews fill earlier rows with later cards without changing their sizes", () => {
    assert.deepEqual(packPreviewRows([360, 360, 160, 160, 160], 700), [[0, 2, 3], [1, 4]]);
    assert.deepEqual(packPreviewRows([], 700), []);
    for (const available of [260, 700, 1100]) {
        const widths = [360, 360, 160, 160, 160, 800, 90, 130, 250, 390];
        const rows = packPreviewRows(widths, available);
        assert.deepEqual(rows.flat().sort((a, b) => a - b), widths.map((_, i) => i));
        rows.forEach((row, i) => {
            const used = row.reduce((sum, index) => sum + Math.min(widths[index], available), 0) + (row.length - 1) * 8;
            assert.ok(used <= available);
            assert.ok(rows.slice(i + 1).flat().every(index => used + 8 + Math.min(widths[index], available) > available),
                "No card from a later row fits in a remaining gap");
        });
    }
});

test("compact gallery tracks fit every card count and center incomplete rows", () => {
    const counts = [...Array.from({ length: 32 }, (_, i) => i + 1), 64, 128, 600];
    for (const count of counts) for (const aspect of [640 / 1536, 1, 1536 / 640]) {
        for (const [width, height] of [[600, 900], [900, 700], [1400, 600]]) for (const footerHeight of [0, 32]) {
            const grid = fitPreviewGrid(count, width, height, aspect, footerHeight);
            assert.ok(grid.trackW > 0 && grid.trackH > 0);
            assert.equal(grid.sizes.length, count);
            assert.ok(grid.cols * grid.trackW + (grid.cols - 1) * 8 <= width + 1e-8);
            assert.ok(grid.rows * grid.trackH + (grid.rows - 1) * 8 <= height + 1e-8);
            for (const card of grid.sizes) {
                assert.equal(card.width, grid.trackW);
                assert.equal(card.height, grid.trackH);
                assert.ok(Math.abs(card.width / (card.height - footerHeight) - aspect) < 1e-8);
            }
            const fullWidth = grid.cols * grid.trackW + (grid.cols - 1) * 8;
            const lastWidth = grid.lastRowCount * grid.trackW + (grid.lastRowCount - 1) * 8;
            assert.ok(Math.abs(grid.lastRowOffset + lastWidth / 2 - fullWidth / 2) < 1e-8);
        }
    }
});
