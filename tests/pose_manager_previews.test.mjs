import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../web/pose_manager_previews.mjs", import.meta.url), "utf8").replace("export function", "function");
function setup(count) {
    const frames = new Map(), captures = [], versions = Array(count).fill(0);
    let nextFrame = 0;
    const requestAnimationFrame = callback => { frames.set(++nextFrame, callback); return nextFrame; };
    const cancelAnimationFrame = id => frames.delete(id);
    const w = {
        poses: Array.from({ length: count }, () => ({})), poseCaptures: [], managerPoseMetrics: [],
        exportParams: { view_width: 640, view_height: 1536 }, interfaceMode: "manager", _poseGestureActive: true,
        viewer: { animate() {} }, layoutPoseManager() {}, forgetPoseManagerImageMetrics() {},
        scheduleAllManagerPreviewRefresh() {
            const generation = this._managerPreviewRefreshGeneration = (this._managerPreviewRefreshGeneration || 0) + 1;
            this._managerPreviewRefreshNextIndex = 0;
            cancelAnimationFrame(this._managerPreviewRefreshFrame);
            this._managerPreviewRefreshFrame = requestAnimationFrame(() => this.refreshAllManagerPreviews(generation));
            return generation;
        },
        refreshAllManagerPreviews(generation = this._managerPreviewRefreshGeneration) {
            if (generation !== this._managerPreviewRefreshGeneration) return;
            const start = this._managerPreviewRefreshNextIndex || 0, end = Math.min(start + 2, count);
            for (let i = start; i < end; i++) {
                captures.push(i); versions[i] = generation; this.setPoseCapture(i, "preview");
            }
            this._managerPreviewRefreshNextIndex = end;
            if (end === count) {
                this._managerPreviewRefreshNextIndex = 0;
                this._managerPreviewRefreshCompletedGeneration = generation;
            } else this._managerPreviewRefreshFrame = requestAnimationFrame(() => this.refreshAllManagerPreviews(generation));
        },
    };
    vm.runInNewContext(`${source}\nmountLivePreviews(w);`, { w, requestAnimationFrame, cancelAnimationFrame });
    return { w, captures, versions, frame() {
        const [id, callback] = frames.entries().next().value;
        frames.delete(id); callback();
    } };
}

test("continuous input reaches every card without declaring a partial generation complete", () => {
    const { w, captures, versions, frame } = setup(12);
    for (let i = 0; i < 6; i++) { w.scheduleAllManagerPreviewRefresh(); frame(); }
    assert.deepEqual(captures, Array.from({ length: 12 }, (_, i) => i));
    assert.notEqual(w._managerPreviewRefreshCompletedGeneration, w._managerPreviewRefreshGeneration);
    for (let i = 0; i < 6; i++) frame();
    assert.ok(versions.every(version => version === w._managerPreviewRefreshGeneration));
    assert.equal(w._managerPreviewRefreshCompletedGeneration, w._managerPreviewRefreshGeneration);
});

test("ordinary edits restart a complete pass and a single card completes immediately", () => {
    const { w, captures, frame } = setup(12);
    w.scheduleAllManagerPreviewRefresh(); frame();
    w._poseGestureActive = false;
    w.scheduleAllManagerPreviewRefresh(); frame();
    assert.deepEqual(captures, [0, 1, 0, 1]);
    const single = setup(1);
    single.w.scheduleAllManagerPreviewRefresh(); single.frame();
    assert.equal(single.w._managerPreviewRefreshCompletedGeneration, single.w._managerPreviewRefreshGeneration);
});
