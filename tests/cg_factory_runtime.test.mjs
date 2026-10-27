import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { runtimeSource } from "./helpers/cg_frontend.mjs";
const loadFactoryRuntime = vm.runInNewContext(runtimeSource.replaceAll("import.meta.url", '"https://localhost/extensions/VNCCS/vnccs_cg_creator.js"') + "\nloadFactoryRuntime", { URL });

test("missing, incompatible and external Factory modules fail before loading", async () => {
    for (const descriptor of [undefined, {}, { version: 2 },
        { version: 1, module_url: "https://other.example/viewer.mjs" },
        { version: 1, module_url: "/extensions/Utils/../../viewer.mjs" }]) {
        await assert.rejects(loadFactoryRuntime(descriptor), /3D Factory viewer is unavailable/);
    }
    await assert.rejects(loadFactoryRuntime({ error: "Utils is unavailable" }), /Utils is unavailable/);
});

test("existing Factory modules load without a CG provider and preserve the reverse-proxy prefix", async () => {
    const requested = [], styles = [];
    const viewer = { Factory3DViewer: class {}, FactoryCommandHistory: class {} };
    const load = vm.runInNewContext(runtimeSource
        .replaceAll("import.meta.url", '"https://comfy.example/proxy/extensions/VNCCS/vnccs_cg_creator.js"')
        .replaceAll("import(", "loadModule(") + "\nloadFactoryRuntime", {
        URL, document: { getElementById: () => false, createElement: () => ({}), head: { appendChild: link => styles.push(link) } },
        loadModule: async url => { requested.push(url); return viewer; },
    });
    const loaded = await load({ version: 1, module_url: "/extensions/Renamed%20Utils/factory3d/viewer.js" });
    assert.equal(loaded.Factory3DViewer, viewer.Factory3DViewer);
    assert.equal(loaded.FactoryCommandHistory, viewer.FactoryCommandHistory);
    assert.deepEqual(requested, ["https://comfy.example/proxy/extensions/Renamed%20Utils/factory3d/viewer.js", "https://comfy.example/proxy/extensions/Renamed%20Utils/factory3d/editor_commands.mjs"]);
    assert.equal(styles[0].href, "https://comfy.example/proxy/extensions/Renamed%20Utils/factory3d/styles.css");
});
