import assert from "node:assert/strict";
import test from "node:test";
import { fitPreviewGrid } from "../web/preview_grid_layout.mjs";

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
