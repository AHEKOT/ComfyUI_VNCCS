import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { runtimeSource } from "./helpers/cg_frontend.mjs";
const loadFactoryRuntime = vm.runInNewContext(runtimeSource.replaceAll("import.meta.url", '"https://localhost/extensions/VNCCS/vnccs_cg_creator.js"') + "\nloadFactoryRuntime", { URL });

test("missing, incompatible and external Factory modules fail before loading", async () => {
    for (const descriptor of [undefined, {}, { version: 2 },
        { version: 1, module_url: "https://other.example/viewer.mjs" },
        { version: 1, module_url: "/extensions/Utils/../../viewer.mjs" }]) {
        await assert.rejects(loadFactoryRuntime(descriptor), /Update ComfyUI_VNCCS_Utils/);
    }
    await assert.rejects(loadFactoryRuntime({ error: "Utils is unavailable" }), /Utils is unavailable/);
});

test("Utils viewer discovery preserves the ComfyUI reverse-proxy prefix", async () => {
    let requested;
    const viewer = { Factory3DViewer: class {}, FactoryCommandHistory: class {} };
    const load = vm.runInNewContext(runtimeSource
        .replaceAll("import.meta.url", '"https://comfy.example/proxy/extensions/VNCCS/vnccs_cg_creator.js"')
        .replace("await import(", "await loadModule(") + "\nloadFactoryRuntime", {
        URL, document: { getElementById: () => true },
        loadModule: async url => { requested = url; return viewer; },
    });
    assert.equal(await load({ version: 1, module_url: "/extensions/Renamed%20Utils/factory3d/cg_provider.mjs" }), viewer);
    assert.equal(requested, "https://comfy.example/proxy/extensions/Renamed%20Utils/factory3d/cg_provider.mjs");
});
