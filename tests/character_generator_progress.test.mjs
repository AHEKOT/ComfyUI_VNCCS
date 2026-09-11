import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_character_generator.js", import.meta.url), "utf8");

function setup(stage = "pose_generation") {
    const listeners = new Map();
    const cleanups = [];
    const context = vm.createContext({
        api: {
            addEventListener: (name, callback) => listeners.set(name, callback),
            removeEventListener: (name, callback) => {
                assert.equal(listeners.get(name), callback);
                listeners.delete(name);
            },
        },
        registerCleanup: (_, callback) => cleanups.push(callback),
    });
    vm.runInContext(source.slice(source.indexOf("class CharacterGeneratorWidget"), source.indexOf("app.registerExtension"))
        + "\nthis.Widget = CharacterGeneratorWidget;", context);
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        node: { id: 17 },
        stageState: { [stage]: { status: "waiting", images: null }, upscaler: { status: "waiting" } },
        stages: [[stage, "Pose Generation"], ["upscaler", "Upscaler"]],
        data: { ui: {} },
        renders: 0,
        persistUI() {},
        updateRegenerateProgress() {},
        renderPreview() { this.renders++; },
        renderChain() {},
        saveBrowserState() {},
        finishRegenerate() {},
    });
    widget.bindEvents();
    return {
        widget,
        cleanups,
        listeners,
        emit: detail => listeners.get("vnccs.character_generator.stage")({ detail: {
            node_id: "17", stage, status: "running", ...detail,
        } }),
    };
}

for (const stage of ["pose_generation", "original_pose_generation", "naked_pose_generation"]) {
    test(`${stage}: phase updates display counts and preserve preview selection`, () => {
        const { widget, emit } = setup(stage);
        emit({ message: "Encoding poses", current: 0, total: 12 });
        assert.equal(widget.formatStageStatus(stage), "Encoding poses (0/12)");
        assert.equal(widget.selectedPreview, stage);
        const previews = ["/preview/pose.png"];
        widget.stageState[stage].images = previews;
        widget.selectedPreview = "upscaler";
        widget.userSelectedPreview = true;
        widget.data.ui.user_selected_preview = true;

        for (const message of ["Encoding poses", "Sampling poses", "Decoding poses"]) {
            emit({ message, current: 5, total: 12 });
            assert.equal(widget.formatStageStatus(stage), `${message} (5/12)`);
            assert.equal(widget.selectedPreview, "upscaler");
            assert.equal(widget.userSelectedPreview, true);
            assert.equal(widget.data.ui.user_selected_preview, true);
            assert.equal(widget.stageState[stage].images, previews);
        }
        emit({ status: "done", message: "Generated 12 pose images", current: 12, total: 12 });
        assert.equal(widget.stageState[stage].status, "done");
        assert.equal(widget.selectedPreview, "upscaler");
    });
}

test("events from other nodes do not alter this widget; removal cleans up the listener", () => {
    const { widget, emit, cleanups, listeners } = setup();
    emit({ node_id: "another-node", message: "Sampling poses", current: 10, total: 12 });
    assert.equal(widget.stageState.pose_generation.status, "waiting");
    assert.equal(widget.renders, 0);
    cleanups.forEach(callback => callback());
    assert.equal(listeners.size, 0);
});

test("starting a new generation resets downstream stages and follows its preview", () => {
    const { widget, emit } = setup();
    widget.stageState.pose_generation.status = "done";
    widget.stageState.upscaler.status = "done";
    widget.selectedPreview = "upscaler";
    widget.userSelectedPreview = true;
    emit({ message: "Preparing pose generation", current: 0, total: 1 });
    assert.equal(widget.stageState.upscaler.status, "waiting");
    assert.equal(widget.selectedPreview, "pose_generation");
    assert.equal(widget.userSelectedPreview, false);
});
