import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../web/vnccs_pose_manager.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "");
test("Manager imports bypass older cached modules throughout its dependency chain", () => {
    const entry = readFileSync(new URL("../web/vnccs_pose_manager.js", import.meta.url), "utf8");
    const version = entry.match(/pose_manager_ui\.mjs\?v=([^"\s]+)/)[1];
    assert.match(version, /^pose-manager-\d+$/);
    for (const [file, dependencies] of [["vnccs_pose_manager.js", ["pose_manager_ui.mjs"]],
        ["pose_manager_ui.mjs", ["pose_manager_previews.mjs", "pose_manager_library.mjs", "pose_set_library.mjs", "preview_grid_layout.mjs"]]]) {
        const imports = [...readFileSync(new URL(`../web/${file}`, import.meta.url), "utf8").matchAll(/from "([^"]+)"/g)]
            .map(match => new URL(match[1], "https://fixture.test/web/"));
        for (const name of dependencies) {
            const url = imports.find(url => url.pathname.endsWith(`/${name}`));
            assert.equal(url?.searchParams.get("v"), version, `${file} must use the widget version for ${name}`);
        }
    }
});
function extension(extensions, mountManager, disabled = []) {
    const app = { extensions, registerExtension(value) { this.manager = value; },
        extensionManager: { setting: { get: () => disabled } } };
    vm.runInNewContext(source, { app, mountManager, document: { createElement: () => ({ setAttribute() {} }) } });
    return app.manager;
}

test("Manager reuses Studio hooks without registering or modifying the original node", async () => {
    const calls = [], widget = { closeSetLibrary() { calls.push("close"); },
        _closeLibraryModal() { calls.push("close pose library"); } };
    const studio = { name: "VNCCS.PoseStudio", beforeRegisterNodeDef(type, data) {
        calls.push(data.name);
        type.prototype.onNodeCreated = function () { this.studioWidget = widget; return "created"; };
        type.prototype.onConfigure = function (data) { calls.push(data); };
    } };
    const manager = extension([studio], actual => assert.equal(actual, widget));
    class Manager {}
    const definition = { name: "VNCCS_PoseManager" };
    await manager.beforeRegisterNodeDef(class Original {}, { name: "VNCCS_PoseStudio" });
    await manager.beforeRegisterNodeDef(Manager, definition);
    const node = new Manager();
    assert.equal(node.onNodeCreated(), "created");
    node.onConfigure("saved data");
    assert.equal(definition.name, "VNCCS_PoseManager");
    assert.deepEqual(calls, ["VNCCS_PoseStudio", "close", "close pose library", "saved data"]);
});

test("Missing or disabled Studio preserves hidden authored data and explains the dependency", async () => {
    for (const available of [[], [{ name: "VNCCS.PoseStudio", beforeRegisterNodeDef: assert.fail }]]) {
        class Manager { addDOMWidget(name, type, notice, options) { this.notice = notice; assert.equal(options.serialize, false); } }
        const manager = extension(available, assert.fail, ["VNCCS.PoseStudio"]);
        await manager.beforeRegisterNodeDef(Manager, { name: "VNCCS_PoseManager" });
        const node = new Manager();
        node.widgets = [{ name: "pose_data", value: "authored bytes" }];
        node.onNodeCreated();
        assert.equal(node.widgets[0].value, "authored bytes");
        assert.equal(node.widgets[0].hidden, true);
        assert.match(node.notice.textContent, /requires VNCCS-Utils/);
    }
});

test("Manager accepts the existing character workflows' twelve-pose state and socket layout", () => {
    for (const name of ["Step1_CharacterCreator", "Step1_CharacterCloner", "Step2_CharacterClothes"]) {
        const graph = JSON.parse(readFileSync(new URL(`../workflows/VNCCS_3.2_${name}.json`, import.meta.url)));
        const node = graph.nodes.find(node => ["VNCCS_PoseStudio", "VNCCS_PoseManager"].includes(node.type));
        assert.ok(node);
        assert.equal(node.inputs[0].name, "pose_image");
        assert.deepEqual(node.outputs.map(output => output.name), ["images", "lighting_prompt"]);
        assert.equal(JSON.parse(node.widgets_values[0]).poses.length, 12);
        assert.equal(graph.links.filter(link => link[1] === node.id).length, 2);
    }
});
