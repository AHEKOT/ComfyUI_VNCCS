import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

export async function verifyManagerLiveFraming(page, output) {
    const reports = [];
    const settled = () => page.waitForFunction(() => {
        const w = poseNode.studioWidget;
        return !w._morphSolveInFlight && !w._pendingMorphSolve
            && w._managerPreviewRefreshGeneration === w._managerPreviewRefreshCompletedGeneration;
    });
    await page.evaluate(() => {
        window.readCardBounds = () => [...poseNode.studioWidget.managerGrid.querySelectorAll("canvas.vnccs-pm-live")].map(source => {
            const copy = document.createElement("canvas"); copy.width = source.width; copy.height = source.height;
            const ctx = copy.getContext("2d"); ctx.drawImage(source, 0, 0);
            const pixels = ctx.getImageData(0, 0, copy.width, copy.height).data;
            let left = copy.width, top = copy.height, right = -1, bottom = -1;
            for (let y = 0; y < copy.height; y++) for (let x = 0; x < copy.width; x++) {
                const i = (y * copy.width + x) * 4;
                if (Math.max(Math.abs(pixels[i] - pixels[0]), Math.abs(pixels[i + 1] - pixels[1]), Math.abs(pixels[i + 2] - pixels[2])) < 35) continue;
                left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);
            }
            if (right < left) throw Error("The test must measure a visible mannequin");
            const rect = source.getBoundingClientRect(), scale = Math.min(rect.width / source.width, rect.height / source.height);
            return [left, top, right, bottom].map(value => value * scale);
        });
    });
    for (const count of [10, 12]) {
        await page.evaluate(count => {
            const w = poseNode.studioWidget;
            w.loadPoseSetAsset({ type: "pose_set", poses: Array.from({ length: count }, () => structuredClone(w.poses[1])) });
            const input = w.managerControls.age.input;
            input.value = "80"; input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
        }, count);
        await settled();
        const slider = await page.evaluate(() => {
            const w = poseNode.studioWidget, capture = w.viewer.capture, update = w.updatePoseManagerPreviewImage;
            let rendered = null;
            window.liveFitSamples = [];
            w.viewer.capture = function (...args) {
                const result = capture.apply(this, args);
                rendered = { age: w.meshParams.age, zoom: args[2], held: w._poseGestureActive };
                return result;
            };
            w.updatePoseManagerPreviewImage = function (index) {
                if (rendered) { liveFitSamples.push({ ...rendered, index }); rendered = null; }
                return update.apply(this, arguments);
            };
            window.restoreFitProbe = () => { w.viewer.capture = capture; w.updatePoseManagerPreviewImage = update; };
            const r = w.managerControls.age.input.getBoundingClientRect();
            return { x: r.x, y: r.y + r.height / 2, width: r.width };
        });
        await page.mouse.move(slider.x + slider.width * .95, slider.y);
        await page.mouse.down();
        await page.mouse.move(slider.x + slider.width * .03, slider.y, { steps: 40 });
        const during = await page.evaluate(count => Array.from({ length: count }, (_, index) => {
            const samples = liveFitSamples.filter(sample => sample.held && sample.index === index);
            return { index, ages: [...new Set(samples.map(sample => sample.age))], zooms: [...new Set(samples.map(sample => sample.zoom))] };
        }), count);
        assert.ok(during.every(card => card.ages.length > 1 && card.zooms.length > 1),
            `Every card must refit while the mouse is moving: ${JSON.stringify(during)}`);
        await settled();
        const held = await page.evaluate(() => ({ active: poseNode.studioWidget._poseGestureActive, bounds: readCardBounds() }));
        assert.equal(held.active, true);
        assert.equal(held.bounds.length, count);
        await page.screenshot({ path: join(output, `age-${count}-held.png`) });
        await page.mouse.up();
        await settled();
        const released = await page.evaluate(() => { restoreFitProbe(); return readCardBounds(); });
        held.bounds.forEach((bounds, index) => bounds.forEach((value, axis) =>
            assert.ok(Math.abs(value - released[index][axis]) <= 1, `Pose ${index + 1}: release must not change framing`)));
        await page.screenshot({ path: join(output, `age-${count}-released.png`) });
        reports.push({ count, during, held, released });
    }
    await writeFile(join(output, "live-framing-results.json"), JSON.stringify(reports, null, 2));
}
