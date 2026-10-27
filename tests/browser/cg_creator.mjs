// Adapted from VNCCS tests/browser/background_creator.mjs: real widget, local fixtures, After captures.
import assert from "node:assert/strict";
import vm from "node:vm";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { settingsDefaults, BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH } from "../helpers/cg_frontend.mjs";
import { base, backgroundId, imageId, modelId, setId, sourceVariant, variantModelId, baseModel, turboModel, turboLora, manualLora, controlCenter, camera, document as initial, scene, roomSplat } from "./fixtures/cg_creator.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const utilsWeb = resolve(process.argv[4]);
const project = fileURLToPath(new URL("../../", import.meta.url));
const output = resolve(process.argv[3] || join(project, "tests/browser/evidence/cg-creator"));
await mkdir(output, { recursive: true });
for (const name of ["after", "after-compact", "after-resize-landscape", "after-resize-portrait", "after-resolution-slider", "after-settings", "after-model-picker", "after-model-settings", "after-model-generating", "after-preview", "after-cg-preview", "after-save-view", "after-generating", "after-completed", "after-empty", "after-read-only", "after-failure"])
    await rm(join(output, name + ".png"), { force: true });
for (const name of ["after.geometry.json", "report.json"]) await rm(join(output, name), { force: true });
const room = await readFile(new URL("./fixtures/room.svg", import.meta.url));
const splat = roomSplat(), assets = new Map();
let document = { ...structuredClone(initial), cg_settings: settingsDefaults() }, job = null, readOnly = false, catalogAvailable = true;
const scenes = new Map([[modelId, structuredClone(scene)]]);
const config = () => ({ factory: { version: 1, module_url: "/extensions/Utils/factory3d/viewer.js" }, settings: settingsDefaults(), model: "Qwen Image 2.1 INT8 ConvRot", installed: true,
    assets: [{ category: "diffusion_models", name: baseModel, installed: true }], diffusion_models: [baseModel, turboModel], model_presets: controlCenter.models,
    loras: [manualLora, turboLora], turbo_lora: { ...controlCenter.lora[0], installed: true, filename: turboLora },
    samplers: ["euler", "dpmpp_2m"], schedulers: ["simple", "normal"] });
const mutations = [], errors = [], unexpected = [], warnings = [], scenarios = [];
const publicDocument = () => ({ ...document, cgs: document.cgs.filter(cg => !cg.archived), ...(readOnly ? { _recovery: { source: "Evening room.json.backup" } } : {}) });

const backgroundSource = await readFile(join(project, "web/vnccs_background_creator.js"), "utf8");
const backgroundRootRule = backgroundSource.match(/\.vnccs-background-creator \{ display:flex[^}]+\}/)[0];
const backgroundColumns = backgroundSource.match(/\.vnccs-background-creator \.bg-background \{ grid-template-columns:([^;]+);/)[1]
    .replace(/\$\{([^}]+)\}/g, (_match, expression) => String(vm.runInNewContext(expression, { GENERATION_PANEL_WIDTH: (1280 / .67 - 24) * .35 })));
const html = `<!doctype html><html lang="en"><meta charset="utf-8"><title>VNCCS CG Creator browser test</title>
<style>body{margin:0;padding:20px;background:#202024}#host{margin:0 auto;position:relative}</style><div id="host"></div><script type="module">
import '/web/vnccs_cg_creator.js';
import { app } from '/scripts/app.js';
const graphCanvas=document.createElement('canvas');graphCanvas.hidden=true;document.body.appendChild(graphCanvas);app.canvasEl=graphCanvas;
window.graphInputs=[];
window.sourceAnimations=()=>[...(window.widget?.customSelects?._motionAnimations||[])].filter(animation=>['running','paused'].includes(animation.playState)&&animation.effect.target?.matches('[data-model-source],[data-source-incoming]'));
for(const type of ['pointerdown','mousedown','pointermove','mousemove','pointerup','mouseup','wheel'])graphCanvas.addEventListener(type,event=>graphInputs.push({type:event.type,clientX:event.clientX,clientY:event.clientY,buttons:event.buttons,deltaY:event.deltaY,forwarded:!!event._vnccsForwardedCanvasInput}));
window.setup=async({width=${LAYOUT_WIDTH},height=950,empty=false,raw=null}={})=>{
 window.widget?.node.onRemoved();const host=document.querySelector('#host');host.replaceChildren();
 const saved=await(await fetch('/fixture')).json();
 class Node{
  constructor(){this.id=7;this.type='VNCCS_CGCreator';this.size=[width,height];this.graph=app.graph;this.properties={};this.widgets=[{name:'cg_data',value:'unconfigured'}];}
  setSize(size){this.size=[...size];host.style.width=size[0]+'px';}
  addDOMWidget(name,type,element,options){const widget={name,type,element,options,triggerDraw(){}};this.widgets.push(widget);return widget;}
 }
 app.extensions.find(extension=>extension.name==='VNCCS.CGCreator').beforeRegisterNodeDef(Node,{name:'VNCCS_CGCreator'});
 const node=new Node();app.graph._nodes=[node];node.onNodeCreated();window.widget=node.cgCreator;
 if(node.widgets[0].value!=='unconfigured')throw new Error('Construction overwrote workflow data');
 node.widgets[0].value=raw||JSON.stringify({schema_version:1,background_id:empty?'':saved.background_id,
  selected_image_id:empty?'':saved.selected_image_id,model_id:empty?'':saved.models[0].model_id,selected_cg_id:empty?'':saved.cgs[0]?.cg_id||'',
  model_settings:saved.model_settings,settings:saved.cg_settings});
 node.onConfigure();clearTimeout(node._cgRestore);node.setSize([width,height]);host.append(widget.frame);node.onResize();await widget.restore();
 if(widget.restoring)throw new Error(widget.status.textContent);await document.fonts.ready;window.scrollTo(0,0);
};
window.ready=true;
</script></html>`;
const mime = { ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".png": "image/png", ".svg": "image/svg+xml", ".woff2": "font/woff2" };
const server = createServer(async (request, response) => {
    try {
        const url = new URL(request.url, "http://localhost");
        let content, type = "application/json";
        if (url.pathname === "/") { content = html; type = "text/html"; }
        else if (url.pathname === "/scripts/app.js") { content = "export const app={extensions:[],graph:{_nodes:[],setDirtyCanvas(){}},registerExtension(e){this.extensions.push(e)}};"; type = "text/javascript"; }
        else if (url.pathname === "/scripts/api.js") { content = "export const api=Object.assign(new EventTarget(),{apiURL:path=>path,fetchApi:(path,options)=>fetch(path,options),queuePrompt(){throw new Error('The browser fixture must not queue a workflow')}});"; type = "text/javascript"; }
        else if (url.pathname.startsWith("/extensions/Utils/")) {
            const path = resolve(utilsWeb, decodeURIComponent(url.pathname.slice("/extensions/Utils/".length)));
            assert.ok(path.startsWith(utilsWeb + sep));
            content = await readFile(path); type = mime[extname(path)] || "application/octet-stream";
        } else if (url.pathname.startsWith("/web/")) {
            const path = resolve(project, "." + decodeURIComponent(url.pathname));
            assert.ok(path.startsWith(join(project, "web") + sep));
            content = await readFile(path); type = mime[extname(path)] || "application/octet-stream";
        } else if (url.pathname === "/fixture") content = JSON.stringify(publicDocument());
        else if (url.pathname === "/vnccs/control_center/check") {
            if (!catalogAvailable) { response.writeHead(404, { "Content-Type": type }).end(JSON.stringify({ error: "Control Center is not installed" })); return; }
            content = JSON.stringify(controlCenter);
        } else if (url.pathname === "/vnccs/manager/status") content = "{}";
        else if (url.pathname === "/vnccs/control_center/download" && request.method === "POST") {
            let raw = ""; for await (const part of request) raw += part;
            const payload = JSON.parse(raw); assert.equal(payload.repo_id, "MIUProject/VNCCS_v3.0"); mutations.push({ download: payload }); content = "{}";
        } else if (url.pathname === `${base}/config`) content = JSON.stringify(config());
        else if (url.pathname === `${base}/backgrounds`) content = JSON.stringify({ backgrounds: [publicDocument()] });
        else if (url.pathname.startsWith(`${base}/backgrounds/${backgroundId}/models/`) && scenes.has(url.pathname.split("/").at(-1))) content = JSON.stringify(scenes.get(url.pathname.split("/").at(-1)));
        else if (url.pathname.endsWith(`/objects/${scene.objects[0].object_id}/splat`)) { content = splat; type = "application/octet-stream"; }
        else if (document.images.some(image => image.url.split("?")[0] === url.pathname) || document.generated_sets.some(group => group.images.some(image => image.url.split("?")[0] === url.pathname)) || document.cgs.some(cg => cg.url === url.pathname)) {
            const asset = assets.get(url.pathname);
            content = asset || (url.pathname.endsWith(`/sets/${setId}/1`) ? Buffer.from(room.toString().replaceAll("#f4b38e", "#c7dbf0"))
                : url.pathname.endsWith(`/sets/${setId}/2`) ? Buffer.from(room.toString().replaceAll("#f4b38e", "#807eaa")) : room);
            type = asset ? "image/png" : "image/svg+xml";
        } else if (url.pathname.startsWith(`${base}/backgrounds/${backgroundId}/cg/`) && request.method === "DELETE") {
            let raw = ""; for await (const part of request) raw += part;
            const payload = JSON.parse(raw), cgId = url.pathname.split("/").at(-1);
            assert.equal(payload.revision, document.revision);
            const cg = document.cgs.find(item => item.cg_id === cgId); assert.ok(cg && !cg.archived);
            document = { ...document, revision: document.revision + 1, cgs: document.cgs.map(item => item.cg_id === cgId ? { ...item, archived: true } : item) };
            mutations.push({ delete_cg: cgId }); content = JSON.stringify(publicDocument());
        } else if (url.pathname === `${base}/backgrounds/${backgroundId}/jobs` && request.method === "POST") {
            const parts = []; for await (const part of request) parts.push(part);
            const bytes = Buffer.concat(parts); let values, image;
            if (request.headers["content-type"]?.startsWith("multipart/form-data")) {
                const form = await new Request(url, { method: "POST", headers: request.headers, body: bytes }).formData();
                values = JSON.parse(form.get("data")); const file = form.get("image"); assert.ok(file.size > 0); image = Buffer.from(await file.arrayBuffer());
            } else values = JSON.parse(bytes.toString());
            assert.ok(["model", "cg"].includes(values.action)); assert.equal(values.image_id, imageId);
            assert.deepEqual(values.source_variant, sourceVariant); assert.equal(values.revision, document.revision);
            if (values.action === "cg") assert.equal(values.model_id, variantModelId);
            job = { job_id: "8".repeat(32), status: "running", progress: 42, message: `${values.action === "cg" ? "Refining CG" : "Creating Gaussian model"} · fixture progress`, values, image };
            mutations.push({ ...values }); content = JSON.stringify({ job_id: job.job_id });
        } else if (url.pathname === "/__fixture__/complete") {
            assert.ok(job);
            if (job.values.action === "model") {
                const metadata = { model_id: variantModelId, source_image_id: imageId, source_variant: job.values.source_variant, settings: job.values.model_settings };
                scenes.set(variantModelId, { ...structuredClone(scene), model_id: variantModelId, source_variant: job.values.source_variant,
                    objects: scene.objects.map(object => ({ ...object, urls: { splat: `${base}/backgrounds/${backgroundId}/models/${variantModelId}/objects/${object.object_id}/splat` } })) });
                document = { ...document, revision: document.revision + 1, models: [...document.models, metadata] };
            } else {
                const cgId = "9".repeat(32), imageURL = `${base}/backgrounds/${backgroundId}/cg/${cgId}/image`; assets.set(imageURL, job.image);
                document = { ...document, revision: document.revision + 1, cgs: [...document.cgs, { cg_id: cgId, name: job.values.name, camera: job.values.camera,
                    model_id: variantModelId, source_image_id: imageId, source_variant: job.values.source_variant,
                    width: job.image.readUInt32BE(16), height: job.image.readUInt32BE(20), url: imageURL }] };
            }
            job.status = "completed"; job.result = { document: publicDocument() }; content = "{}";
        } else if (url.pathname === `${base}/jobs/${job?.job_id}`) content = JSON.stringify({ job_id: job.job_id, status: job.status, progress: job.progress, message: job.message, result: job.result });
        else if (url.pathname === `${base}/backgrounds/${backgroundId}`) {
            if (request.method === "PATCH") {
                let raw = ""; for await (const chunk of request) raw += chunk;
                const payload = JSON.parse(raw); assert.equal(payload.revision, document.revision);
                mutations.push(payload); document = { ...document, revision: document.revision + 1, cg_settings: payload.cg_settings, model_settings: payload.model_settings };
            }
            content = JSON.stringify(publicDocument());
        } else if (url.pathname === "/favicon.ico") { response.writeHead(204).end(); return; }
        else { unexpected.push(url.pathname); response.writeHead(404).end(); return; }
        response.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(content);
    } catch (error) { errors.push(String(error)); response.writeHead(500).end(JSON.stringify({ error: String(error) })); }
});
await new Promise((done, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", done); });
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-swiftshader"], ...(process.argv[2] ? { executablePath: process.argv[2] } : {}) })
    .catch(error => { server.close(); throw error; });
const page = await browser.newPage({ viewport: { width: 6000, height: 3500 }, deviceScaleFactor: 1 });
page.on("pageerror", error => errors.push({ message: String(error), stack: error.stack, scenario: scenarios.at(-1)?.name }));
page.on("console", message => { if (["warning", "error"].includes(message.type())) warnings.push(message.text()); });
await page.route("**/*", route => {
    const url = new URL(route.request().url());
    if (url.origin === origin || ["blob:", "data:"].includes(url.protocol)) return route.continue();
    unexpected.push(url.href); return route.abort();
});
const settle = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
const capture = async name => {
    await page.evaluate(() => document.fonts.ready); await settle();
    const host = page.locator("#host"), dialog = page.getByRole("dialog");
    if (await dialog.count()) {
        const a = await host.boundingBox(), b = await dialog.boundingBox(), x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
        await page.screenshot({ path: join(output, name + ".png"), animations: name === "after-generating" ? "allow" : "disabled", clip: { x, y, width: Math.max(a.x + a.width, b.x + b.width) - x, height: Math.max(a.y + a.height, b.y + b.height) - y } });
    } else await host.screenshot({ path: join(output, name + ".png"), animations: name === "after-generating" ? "allow" : "disabled" });
};
const geometry = () => page.evaluate(() => {
    const result = {};
    for (const [key, selector] of Object.entries({ mount: ".vnccs-cg-widget", widget: ".vnccs-cg-creator", workspace: ".cg-workspace", source: ".bg-model-source", sourcePreview: ".bg-preview", variants: ".bg-variation-cards", viewport: ".bg-viewport-panel", frame: ".bg-viewport", captureFrame: ".vnccs-i3s__camera-frame", gallery: ".bg-cg-panel", dock: ".bg-generation-dock", modelSettings: ".bg-qi2-model-panel", samplingSettings: ".bg-qi2-sampling-panel", toolbar: ".bg-camera-toolbar" })) {
        const element = document.querySelector(selector), rect = element.getBoundingClientRect();
        result[key] = { x: rect.x, y: rect.y, width: rect.width, height: rect.height, overflow: element.scrollWidth - element.clientWidth, verticalOverflow: element.scrollHeight - element.clientHeight };
    }
    return result;
});
const assertBounds = (actual, expected, message) => {
    for (const key of ["x", "y", "width", "height"]) assert.ok(Math.abs(actual[key] - expected[key]) <= 1, `${message}: ${key}`);
};
const verifyQwenModelSwitch = async (width = LAYOUT_WIDTH) => {
    await page.evaluate(width => setup({ width }), width); await settle();
    const height = await page.locator(".vnccs-cg-creator").evaluate(element => element.clientHeight), samples = [];
    for (const label of ["Qwen Image 2.1 Turbo INT8 ConvRot", "Qwen Image 2.1 INT8 ConvRot"]) {
        await page.locator(".vnccs-common-creator-model-picker .is-picker-head").click();
        await page.locator(".vnccs-common-creator-model-picker-menu .vnccs-common-creator-model-card").filter({ hasText: label }).click();
        const ghosts = await page.locator(".vnccs-common-ui-motion-ghost").evaluateAll(elements => elements.map(element => {
            for (const animation of element.getAnimations()) { animation.pause(); animation.currentTime = 80; }
            const original = document.querySelector('[data-qi2-setting="' + element.dataset.qi2Setting + '"]:not([inert])');
            const bounds = node => { const rect = node.getBoundingClientRect(); return { x: rect.x, y: rect.y, width: rect.width, height: rect.height }; };
            return { value: element.value, position: getComputedStyle(element).position, ghost: bounds(element), original: bounds(original) };
        }));
        samples.push({ label, ghosts });
        await writeFile(join(output, "model-switch-motion.json"), JSON.stringify(samples, null, 2));
        assert.ok(ghosts.length > 0, "Model profile changes exercise the numeric value transition");
        for (const ghost of ghosts) {
            assert.equal(ghost.position, "absolute", "Numeric animation copies cannot enter the widget layout flow");
            for (const key of ["x", "y", "width", "height"]) assert.ok(Math.abs(ghost.ghost[key] - ghost.original[key]) <= 2, "Numeric copy stays on its original field: " + key);
        }
        assert.equal(await page.locator(".vnccs-cg-creator").evaluate(element => element.clientHeight), height, "Switching model never adds a footer row");
        await page.locator("[data-qi2]").screenshot({ path: join(output, "after-model-switch.png"), animations: "allow" });
        await page.locator(".vnccs-common-ui-motion-ghost").evaluateAll(elements => elements.forEach(element => element.getAnimations().forEach(animation => animation.finish())));
        await page.waitForFunction(() => !document.querySelector(".vnccs-common-ui-motion-ghost"));
    }
    scenarios.push({ name: "qwen-model-switch-no-phantom-footer-" + width, samples });
};
try {
    await page.goto(origin, { waitUntil: "domcontentloaded" }); await page.waitForFunction(() => window.ready);
    await verifyQwenModelSwitch();
    await verifyQwenModelSwitch(2102);

    const referencePanelWidth = nodeWidth => page.evaluate(({ rootRule, columns, nodeWidth }) => {
        const style = document.createElement("style");
        style.textContent = rootRule + ".vnccs-background-creator .bg-step{display:grid;gap:12px;grid-template-columns:" + columns + "}";
        const reference = document.createElement("div"); reference.className = "vnccs-background-creator";
        reference.style.cssText = "position:absolute;visibility:hidden;pointer-events:none;box-sizing:border-box;zoom:.67;width:" + nodeWidth / .67 + "px";
        const grid = document.createElement("div"); grid.className = "bg-step";
        for (let index = 0; index < 3; index++) { const panel = document.createElement("div"); panel.className = "vnccs-common-creator-col"; panel.style.minWidth = "0"; grid.append(panel); }
        reference.append(grid); document.head.append(style); document.body.append(reference);
        const width = grid.firstElementChild.getBoundingClientRect().width;
        reference.remove(); style.remove(); return width;
    }, { rootRule: backgroundRootRule, columns: backgroundColumns, nodeWidth });
    const referenceWidths = [];
    for (const width of [LAYOUT_WIDTH, 1481, 2102, 4800]) {
        await page.evaluate(width => setup({ width, height: 950 }), width); await settle();
        assert.equal(await page.locator("[data-step],[data-wizard],[data-upload],.bg-header,.bg-status,footer").count(), 0);
        assert.equal(await page.locator(".bg-job-feedback,progress,[data-stop],[data-model-feedback],[data-cg-feedback]").count(), 0, "Status panel, progress bar and Stop are deleted");
        assert.equal(await page.locator("[data-cgs] .bg-cg-card").count(), 2);
        assert.equal(await page.locator("[data-delete-cg]").count(), 2);
        assert.equal(await page.locator("[data-delete-cg]").first().isVisible(), true);
        assert.equal(await page.locator("[data-image]").count(), 0);
        assert.equal(await page.locator("[data-model-dimensions]").count(), 0);
        assert.equal(await page.locator(".vnccs-i3s__camera-frame>span").isVisible(), false, "The viewport has no resolution badge");
        assert.doesNotMatch(await page.locator("[data-cgs]").innerText(), /\d+\s*×\s*\d+/, "CG cards have no resolution labels");
        assert.equal(await page.locator("[data-cg-id] > span").count(), 0, "CG cards reserve no row for FOV");
        assert.equal(await page.locator("[data-source-choice]").count(), 4, "Every source retains its card, including the selected original");
        assert.equal(await page.locator('.vnccs-common-creator-lora-stack,[data-qi2-setting*="lora_stack"],[data-qi2-setting="generation.turbo"]').count(), 0);
        assert.equal(await page.locator('[data-prompt],.vnccs-common-creator-qi2-cache,[data-qi2-setting="generation.cache_device"],[data-qi2-setting="generation.cache_dtype"]').count(), 0);
        assert.equal(await page.getByLabel("Prompt Enhancement", { exact: true }).count(), 0);
        assert.equal(await page.locator("[data-save-view]").isEnabled(), true);
        const bounds = await geometry(), scale = bounds.widget.width / LAYOUT_WIDTH;
        assert.ok(Math.abs(bounds.mount.width - width) <= 1, "The ComfyUI mount follows the node width");
        assert.ok(bounds.widget.x >= bounds.mount.x - 1 && bounds.widget.x + bounds.widget.width <= bounds.mount.x + bounds.mount.width + 1,
            "The full widget stays inside the node at every restored width");
        const dockWrapper = await page.locator(".bg-qi2-contents").evaluate(element => ({ display: getComputedStyle(element).display,
            border: getComputedStyle(element).borderTopWidth, decoration: getComputedStyle(element, "::before").content }));
        assert.deepEqual(dockWrapper, { display: "contents", border: "0px", decoration: "none" }, "The settings wrapper adds no frame or separator");
        const topInset = await page.locator(".vnccs-cg-creator").evaluate(element => { const style = getComputedStyle(element); return parseFloat(style.paddingTop) + parseFloat(style.borderTopWidth); });
        assert.ok(Math.abs(bounds.workspace.y - bounds.widget.y - topInset * scale) < 1, "Workspace starts at the normal scaled inset with no reserved title row");
        const bottomInset = await page.locator(".vnccs-cg-creator").evaluate(element => { const style = getComputedStyle(element); return parseFloat(style.paddingBottom) + parseFloat(style.borderBottomWidth); });
        assert.ok(Math.abs(bounds.widget.y + bounds.widget.height - bounds.workspace.y - bounds.workspace.height - bottomInset * scale) < 1, "Workspace ends at the normal scaled inset with no reserved footer space");
        assert.ok(bounds.source.x < bounds.viewport.x && bounds.viewport.x < bounds.gallery.x);
        for (const [name, box] of Object.entries(bounds)) assert.ok(box.overflow <= Math.ceil(1 / scale), `${name} has horizontal overflow at ${width}px`);
        assert.ok(Math.abs(bounds.viewport.width / scale - 2247) <= 2, "The middle panel keeps its fixed logical width");
        assert.ok(Math.abs(bounds.frame.width / bounds.frame.height - 16 / 9) < .002, "The viewport window is always landscape 16:9");
        assert.ok(bounds.toolbar.y + bounds.toolbar.height <= bounds.frame.y, "Camera toolbar is above the rendered viewport and never covers it");
        assert.ok(Math.abs(bounds.frame.x - bounds.toolbar.x) < 1 && Math.abs(bounds.frame.width - bounds.toolbar.width) < 1, "Model preview fills the middle panel without side gutters");
        const expectedSource = await referencePanelWidth(2102);
        assert.ok(Math.abs(bounds.source.width / scale - expectedSource) <= 8, "The source keeps the Background Creator reference width inside the scaled composition");
        referenceWidths.push({ requestedNode: width, node: bounds.widget.width, creator: expectedSource, cg: bounds.source.width, viewport: bounds.viewport.width });
        assert.ok(bounds.viewport.width > bounds.source.width && bounds.source.width > bounds.gallery.width, "The wide middle composition and side-panel proportions remain ordered");
        assert.ok(Math.abs(bounds.gallery.width / scale - 960) <= 2, "The gallery is twice its original logical width");
        const thumbnailWidths = await page.locator("[data-source-choice] .bg-variation-preview").evaluateAll(elements => elements.map(element => element.getBoundingClientRect().width));
        const expectedThumbnailWidth = (bounds.variants.width - (6 + 15) * scale) / 2 - 4 * scale;
        assert.ok(thumbnailWidths.every(value => Math.abs(value - expectedThumbnailWidth) <= 1), JSON.stringify({ thumbnailWidths, expectedThumbnailWidth }));
        if (width === LAYOUT_WIDTH) {
            assert.ok(bounds.variants.verticalOverflow <= 1, "Source cards use two rows without vertical collection scrolling");
        }
        assert.ok(Math.abs(bounds.modelSettings.y - bounds.samplingSettings.y) <= 1, "Both complete settings panels retain their original shared row");
        assert.ok(bounds.modelSettings.width / scale > 1000 && bounds.samplingSettings.width / scale > 1000, "Both settings panels keep their logical width");
        assert.ok(bounds.captureFrame.width <= bounds.frame.width && bounds.captureFrame.height <= bounds.frame.height, "The selected capture frame fits the full-size model window");
        assert.ok(Math.abs(bounds.captureFrame.width / bounds.captureFrame.height - 16 / 9) < .01, "Capture aspect is independent of the expanded viewport stage");
        assert.ok(bounds.dock.y >= bounds.viewport.y + bounds.viewport.height, "Generation settings are below the viewport");
        assert.ok(bounds.dock.y + bounds.dock.height <= bounds.gallery.y + bounds.gallery.height + 1, "The complete generation form fits above the workspace bottom");
        for (const name of ["source", "viewport", "dock", "modelSettings", "samplingSettings"]) assert.ok(bounds[name].verticalOverflow <= 1, `${name} hides controls below its visible area at ${width}px`);
        const visibleControls = await page.locator("[data-qi2] input:not([hidden]):not([type=checkbox]),[data-qi2] button[role=combobox],.bg-camera-toolbar input,.bg-camera-toolbar button,[data-model-settings],[data-action=model],[data-source-choice]").evaluateAll(elements => elements.filter(element => !element.closest("[inert]")).map(element => {
            const r = element.getBoundingClientRect(), owner = element.closest(".bg-panel").getBoundingClientRect();
            return { label: element.getAttribute("aria-label") || element.dataset.qi2Setting || element.textContent, visible: r.width > 0 && r.height > 0 && r.left >= owner.left && r.right <= owner.right + 1 && r.top >= owner.top && r.bottom <= owner.bottom + 1 };
        }));
        assert.ok(visibleControls.every(control => control.visible), JSON.stringify(visibleControls.filter(control => !control.visible)));
        const rendered = await page.evaluate(async () => {
            const blob = await widget.viewer.capturePreview({ width: 640, height: 352, cameraState: widget.viewer.getCameraState() });
            const bitmap = await createImageBitmap(blob), canvas = new OffscreenCanvas(bitmap.width, bitmap.height), context = canvas.getContext("2d");
            context.drawImage(bitmap, 0, 0); const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
            let colored = 0; for (let i = 0; i < data.length; i += 4) if (data[i] > 30 && data[i + 1] > 20) colored++;
            return { splats: [...widget.viewer.objects.values()][0].splat.numSplats, coverage: colored / (bitmap.width * bitmap.height) };
        });
        assert.ok(rendered.splats > 1000 && rendered.coverage > 0.15, "The production WebGL viewport renders the Gaussian fixture");
        await capture(width === LAYOUT_WIDTH ? "after" : width === 1481 ? "after-compact" : width === 2102 ? "after-reference-workflow" : "after-wide");
        if (width === LAYOUT_WIDTH) await writeFile(join(output, "after.geometry.json"), JSON.stringify(bounds, null, 2));
        const resolutionY = (await page.locator('[data-qi2-setting="resolution_mp"]').boundingBox()).y;
        await page.locator(".vnccs-common-creator-model-picker .is-picker-head").click(); await settle();
        const modelMenu = await page.locator(".vnccs-common-creator-model-picker-menu").boundingBox();
        assert.equal(await page.locator(".vnccs-common-creator-model-picker-menu").evaluate(element => getComputedStyle(element).backgroundColor), "rgb(23, 18, 29)", "Floating presets use the original opaque Sakura dropdown surface");
        assert.ok(modelMenu.y >= bounds.widget.y && modelMenu.y + modelMenu.height <= bounds.widget.y + bounds.widget.height, "Both model presets remain inside the visible widget");
        const presetBounds = await page.locator(".vnccs-common-creator-model-picker-menu .vnccs-common-creator-model-card").evaluateAll(elements => elements.map(element => {
            const r = element.getBoundingClientRect(), menu = element.closest(".vnccs-common-creator-model-picker-menu").getBoundingClientRect();
            return { name: element.textContent, visible: r.top >= menu.top && r.bottom <= menu.bottom };
        }));
        assert.ok(presetBounds.every(card => card.visible), JSON.stringify(presetBounds));
        assert.equal((await page.locator('[data-qi2-setting="resolution_mp"]').boundingBox()).y, resolutionY, "Opening the model list never pushes generation controls below the dock");
        await page.locator(".vnccs-common-creator-model-picker .is-picker-head").click();
        scenarios.push({ name: `workspace-${width}`, geometry: bounds, rendered });
    }
    scenarios.push({ name: "background-creator-source-width-reference", widths: referenceWidths });

    await page.evaluate(() => setup()); await settle();
    await page.evaluate(() => {
        const group = widget.document.generated_sets[0], image = group.images[1];
        for (let index = 0; index < 4; index++) {
            group.variants.push({ id: "extra_" + index, name: "Extra variant " + (index + 1) });
            group.images.push({ ...image });
        }
        widget.render();
    });
    await page.waitForFunction(() => [...document.querySelectorAll("[data-source-choice] img")].every(image => image.complete && image.naturalWidth));
    const sourceGrid = await page.locator("[data-source-choice]").evaluateAll(elements => elements.map(element => {
        const rect = element.getBoundingClientRect(); return { key: element.dataset.sourceChoice, x: rect.x, y: rect.y, width: rect.width, height: rect.height };
    }));
    assert.equal(sourceGrid.length, 8);
    assert.equal(sourceGrid[0].y, sourceGrid[1].y); assert.equal(sourceGrid[2].y, sourceGrid[3].y);
    assert.ok(sourceGrid[2].y > sourceGrid[0].y); assert.ok(sourceGrid[4].x > sourceGrid[1].x + sourceGrid[1].width);
    const strip = page.locator("[data-source-choices]");
    const scrollShape = await strip.evaluate(element => ({ width: element.clientWidth, x: element.scrollWidth - element.clientWidth, y: element.scrollHeight - element.clientHeight }));
    assert.ok(scrollShape.x >= scrollShape.width); assert.ok(scrollShape.y <= 1);
    const arrows = await page.locator("[data-scroll-source]").evaluateAll(elements => elements.map(button => {
        const rect = button.getBoundingClientRect(), list = document.querySelector("[data-source-choices]").getBoundingClientRect(), svg = button.querySelector("svg"), icon = svg.getBoundingClientRect();
        const overlapsPreview = [list, document.querySelector("[data-model-source]").getBoundingClientRect()].some(preview =>
            rect.left < preview.right && rect.right > preview.left && rect.top < preview.bottom && rect.bottom > preview.top);
        const collectionGap = button.dataset.scrollSource === "-1" ? list.left - rect.right : rect.left - list.right;
        return { width: rect.width, height: rect.height, collectionGap, centerOffset: rect.y + rect.height / 2 - list.y - list.height / 2, overlapsPreview,
            iconWidth: icon.width, iconHeight: icon.height, stroke: getComputedStyle(svg).stroke, color: getComputedStyle(button).color };
    }));
    const sourceScale = await page.evaluate(() => widget.layoutScale);
    for (const arrow of arrows) {
        assert.ok(Math.abs(arrow.width - 36 * sourceScale) < 1); assert.ok(Math.abs(arrow.height - 44 * sourceScale) < 1); assert.ok(arrow.collectionGap >= 8 * sourceScale - 1);
        assert.ok(Math.abs(arrow.centerOffset) <= 1);
        assert.equal(arrow.overlapsPreview, false, "Side navigation remains outside the entire card collection");
        assert.ok(Math.abs(arrow.iconWidth - 24 * sourceScale) < 1); assert.ok(Math.abs(arrow.iconHeight - 24 * sourceScale) < 1); assert.equal(arrow.stroke, arrow.color);
    }
    assert.equal(await page.locator('[data-scroll-source="-1"]').isDisabled(), true, JSON.stringify(await strip.evaluate(element => ({ left: element.scrollLeft, width: element.clientWidth, total: element.scrollWidth }))));
    assert.equal(await page.locator('[data-scroll-source="1"]').isDisabled(), false);
    await page.mouse.move(0, 0);
    await page.locator(".bg-model-source").screenshot({ path: join(output, "after-source-panel.png"), animations: "disabled" });
    await page.locator(".bg-variation-strip").screenshot({ path: join(output, "after-source-navigation.png"), animations: "disabled" });
    await page.locator('[data-scroll-source="1"]').click();
    await page.waitForFunction(() => document.querySelector('[data-scroll-source="1"]').disabled);
    assert.equal(await page.locator('[data-scroll-source="-1"]').isDisabled(), false);
    await page.locator('[data-scroll-source="-1"]').click();
    await page.waitForFunction(() => document.querySelector("[data-source-choices]").scrollLeft <= 1);
    assert.equal(await page.locator('[data-scroll-source="-1"]').isDisabled(), true);
    scenarios.push({ name: "two-by-two-source-grid-horizontal-overflow", cards: sourceGrid, overflow: scrollShape, arrows });

    const originalOrder = sourceGrid.map(card => card.key), clickedIndex = 2;
    const originalImages = await page.locator("[data-source-choice] img").evaluateAll(images => images.map(image => image.src));
    await page.locator("[data-source-choice]").nth(clickedIndex).click();
    await page.waitForFunction(() => sourceAnimations().length === 1);
    const fades = await page.evaluate(() => sourceAnimations().map(animation => {
        animation.pause(); animation.currentTime = 120;
        return { frames: animation.effect.getKeyframes(), duration: animation.effect.getTiming().duration, main: animation.effect.target.hasAttribute("data-source-incoming") };
    }));
    for (const fade of fades) {
        assert.equal(fade.duration, 240);
        assert.equal(fade.main, true, "Only the main preview fades; variant cards remain unchanged");
        assert.equal(fade.frames[0].opacity, "0"); assert.equal(fade.frames.at(-1).opacity, "1");
        assert.ok(fade.frames.every(frame => !("transform" in frame)), "Image changes animate opacity only");
    }
    await page.locator(".bg-model-source").screenshot({ path: join(output, "after-source-swap.png"), animations: "allow" });
    await page.evaluate(() => sourceAnimations().forEach(animation => animation.finish()));
    await page.waitForFunction(() => !widget.sourceLoad && !sourceAnimations().length && !widget.busy);
    const selectedGrid = await page.locator("[data-source-choice]").evaluateAll(elements => elements.map(element => {
        const rect = element.getBoundingClientRect(); return { key: element.dataset.sourceChoice, x: rect.x, y: rect.y, width: rect.width, height: rect.height };
    }));
    assert.deepEqual(selectedGrid, sourceGrid, "Every card keeps its source and exact position after selection");
    assert.deepEqual(await page.locator("[data-source-choice] img").evaluateAll(images => images.map(image => image.src)), originalImages);
    assert.match(await page.locator("[data-model-source]").getAttribute("src"), /\/sets\/[^/]+\/2/);
    assert.equal(await page.evaluate(() => document.activeElement.dataset.sourceChoice), originalOrder[clickedIndex], "Keyboard focus stays on the selected variant");
    assert.equal(await page.locator('[data-source-choice][aria-pressed="true"]').getAttribute("data-source-choice"), originalOrder[clickedIndex]);
    assert.equal(await page.locator(".bg-source-motion").count(), 0);
    await page.locator(".bg-model-source").screenshot({ path: join(output, "after-source-selected.png"), animations: "disabled" });
    await page.locator("[data-source-choice]").nth(clickedIndex).click();
    assert.equal(await page.evaluate(() => sourceAnimations().length), 0, "Clicking the selected card does not restart the fade or reload the model");
    scenarios.push({ name: "fixed-source-card-order-and-main-preview-fade", before: sourceGrid, after: selectedGrid, fades });

    const collectionBounds = await strip.boundingBox();
    await page.mouse.move(collectionBounds.x + 200, collectionBounds.y + 60); await page.mouse.wheel(420, 0);
    await page.waitForFunction(() => document.querySelector("[data-source-choices]").scrollLeft > 100);
    await page.locator("[data-source-choice]").nth(6).scrollIntoViewIfNeeded(); await settle();
    const scrolledBefore = await strip.evaluate(element => element.scrollLeft);
    await page.locator("[data-source-choice]").nth(6).click();
    await page.waitForFunction(() => !widget.sourceLoad && !sourceAnimations().length && !widget.busy); await settle();
    assert.ok(Math.abs(await strip.evaluate(element => element.scrollLeft) - scrolledBefore) <= 1, "Selecting an off-origin card preserves horizontal scroll");
    assert.deepEqual(await page.locator("[data-source-choice]").evaluateAll(elements => elements.map(element => element.dataset.sourceChoice)), originalOrder);
    await page.locator('[data-scroll-source="-1"]').click();
    await page.waitForFunction(() => document.querySelector("[data-source-choices]").scrollLeft < 50);
    scenarios.push({ name: "horizontal-source-scroll-retained-across-selection" });

    await page.locator("[data-source-choice]").first().click();
    await page.waitForFunction(() => sourceAnimations().length > 0);
    await page.locator("[data-source-choice]").nth(1).click();
    await page.waitForFunction(() => !widget.sourceLoad && !sourceAnimations().length && !widget.busy);
    assert.equal(await page.locator(".bg-source-motion").count(), 0);
    assert.equal(await page.locator('[data-model-source]').evaluate(image => getComputedStyle(image).visibility), "visible");
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.locator("[data-source-choice]").first().click();
    await page.waitForFunction(() => !widget.sourceLoad && !widget.busy);
    assert.equal(await page.evaluate(() => sourceAnimations().length), 0);
    assert.equal(await page.locator(".bg-source-motion").count(), 0);
    await page.emulateMedia({ reducedMotion: "no-preference" });
    scenarios.push({ name: "source-swap-interruption-and-reduced-motion" });

    await page.evaluate(async () => { await setup(); const host = document.querySelector("#host"); host.style.transform = "scale(.7)"; host.style.transformOrigin = "top left"; });
    await settle(); await page.waitForFunction(() => document.querySelector("[data-model-source]").complete);
    const scaledMain = await page.locator("[data-model-source]").boundingBox(), scaledCard = await page.locator("[data-source-choice] img").first().boundingBox();
    await page.locator("[data-source-choice]").nth(1).click();
    await page.waitForFunction(() => sourceAnimations().length === 1);
    const motionRects = await page.evaluate(() => {
        const animations = sourceAnimations(), images = [...document.querySelectorAll("[data-model-source], [data-source-choice] img")];
        const bounds = () => images.map(image => { const rect = image.getBoundingClientRect(); return { x: rect.x, y: rect.y, width: rect.width, height: rect.height }; });
        for (const animation of animations) { animation.pause(); animation.currentTime = 0; }
        const from = bounds();
        for (const animation of animations) animation.currentTime = 239;
        const to = bounds();
        for (const animation of animations) animation.finish();
        return { from, to };
    });
    for (const key of ["x", "y", "width", "height"]) {
        assert.ok(Math.abs(motionRects.from[0][key] - scaledMain[key]) < 2, "Main preview remains at its actual scaled bounds: " + key);
        assert.ok(Math.abs(motionRects.from[1][key] - scaledCard[key]) < 2, "Original card remains in its slot: " + key);
        assert.equal(motionRects.to[0][key], motionRects.from[0][key], "Main preview never moves or scales: " + key);
        for (let index = 1; index < motionRects.from.length; index++)
            assert.equal(motionRects.to[index][key], motionRects.from[index][key], "Every variant card keeps its bounds: " + key);
    }
    await page.waitForFunction(() => !widget.sourceLoad && !sourceAnimations().length);
    await page.evaluate(() => { document.querySelector("#host").style.transform = ""; });
    scenarios.push({ name: "source-fade-stationary-at-graph-scale", rectangles: motionRects });

    await page.locator("[data-source-choice]").first().click(); await page.waitForFunction(() => sourceAnimations().length > 0);
    await page.evaluate(() => { widget.node.onConfigure(); });
    await page.waitForFunction(() => !widget.restoring);
    assert.equal(await page.evaluate(() => sourceAnimations().length), 0);
    assert.equal(await page.locator(".bg-source-motion").count(), 0);
    await page.waitForFunction(() => document.querySelector("[data-model-source]").complete);
    await page.locator("[data-source-choice]").nth(1).click(); await page.waitForFunction(() => sourceAnimations().length > 0);
    const disposedSwap = await page.evaluate(() => {
        const current = widget; current.node.onRemoved();
        return { disposed: current.disposed, active: sourceAnimations().length > 0, layers: current.container.querySelectorAll(".bg-source-motion").length };
    });
    assert.deepEqual(disposedSwap, { disposed: true, active: false, layers: 0 });
    await page.waitForFunction(() => !widget.customSelects._motionAnimations.size);
    scenarios.push({ name: "source-swap-configure-and-removal-cleanup" });
    await page.evaluate(() => setup());
    const resizeCases = [];

    for (const aspect of ["16:9", "9:16"]) {
        await page.evaluate(aspect => { const select = document.querySelector("[data-aspect]"); select.value = aspect; select.dispatchEvent(new Event("change", { bubbles: true })); }, aspect);
        const savedCamera = await page.evaluate(() => widget.viewer.getCameraState());
        await page.mouse.move(2580, 1280); await page.mouse.down();
        for (const [width, height] of [[2744, 950], [3000, 1100], [3400, 850]]) {
            await page.evaluate(size => { widget.node.setSize(size); widget.node.onResize(); }, [width, height]); await settle();
            const fitted = await page.evaluate(() => {
                const canvas = widget.viewer.canvas.getBoundingClientRect(), toolbar = document.querySelector(".bg-camera-toolbar").getBoundingClientRect();
                const panel = document.querySelector(".bg-viewport-panel"), dock = document.querySelector(".bg-generation-dock").getBoundingClientRect(), container = widget.container.getBoundingClientRect();
                const capture = document.querySelector(".vnccs-i3s__camera-frame").getBoundingClientRect();
                return { scale: widget.layoutScale, width: canvas.width, height: canvas.height, aspect: widget.state.frame.width / widget.state.frame.height, toolbar: { x: toolbar.x, width: toolbar.width },
                    capture: { width: capture.width, height: capture.height }, x: canvas.x, nodeHeight: widget.node.size[1], viewportOverflow: panel.scrollHeight - panel.clientHeight, dockBottom: dock.bottom, containerBottom: container.bottom, camera: widget.viewer.getCameraState() };
            });
            assert.ok(Math.abs(fitted.capture.width - fitted.capture.height * fitted.aspect) < 2, "Capture frame retains the selected aspect ratio while the pointer remains held");
            assert.ok(fitted.capture.width <= fitted.width && fitted.capture.height <= fitted.height, "Capture fits the viewport at every resize and aspect");
            assert.ok(Math.abs(fitted.x - fitted.toolbar.x - fitted.scale) < 1 && Math.abs(fitted.width - fitted.toolbar.width + 2 * fitted.scale) < 1, "Preview fills the full middle-panel width while resizing");
            assert.ok(fitted.dockBottom <= fitted.containerBottom, "Node height follows the frame so generation controls remain inside it");
            assert.ok(fitted.viewportOverflow <= 1, "Node height follows the entire frame without clipping it");
            assert.deepEqual(fitted.camera, savedCamera, "Resizing never changes the authored camera");
            resizeCases.push({ aspect, width, height, fitted });
        }
        await page.mouse.up();
        await capture(aspect === "16:9" ? "after-resize-landscape" : "after-resize-portrait");
    }
    scenarios.push({ name: "aspect-preserving-live-resize", cases: resizeCases });
    await page.evaluate(() => setup()); await settle();
    const beforeHeightResize = await geometry(), beforeNodeSize = await page.evaluate(() => [...widget.node.size]);
    await page.evaluate(() => { widget.node.setSize([widget.node.size[0], (widget.node.size[1] - 90) * 1.25 + 90]); widget.node.onResize(); }); await settle();
    const afterHeightResize = await geometry(), afterNodeSize = await page.evaluate(() => [...widget.node.size]);
    assert.equal(afterNodeSize[0], beforeNodeSize[0], "Height dragging does not rewrite the node width");
    assert.ok(afterHeightResize.widget.height > beforeHeightResize.widget.height, "The Background Creator height contract fills the taller node");
    for (const key of ["source", "viewport", "gallery", "dock", "modelSettings", "samplingSettings", "toolbar"])
        assert.ok(Math.abs(afterHeightResize[key].width - beforeHeightResize[key].width) < 2, `Height resizing preserves ${key}.width`);
    await capture("after-proportional-height-resize");
    scenarios.push({ name: "background-contract-height-resize", beforeNodeSize, afterNodeSize, before: beforeHeightResize, after: afterHeightResize });
    await page.evaluate(height => { widget.node.setSize([widget.node.size[0], height]); widget.node.onResize(); }, (beforeNodeSize[1] - 90) * .6 + 90); await settle();
    const shrunk = await geometry();
    assert.ok(Math.abs(shrunk.widget.width - beforeHeightResize.widget.width) < 2, "Height-only dragging preserves the node width");
    assert.ok(Math.abs(shrunk.frame.width / shrunk.frame.height - 16 / 9) < .002);
    await page.evaluate(() => setup()); await settle();
    const navigationHeading = await page.locator(".bg-model-source > .vnccs-common-creator-section-title").first().boundingBox();
    await page.evaluate(() => { graphInputs.length = 0; });
    await page.mouse.move(navigationHeading.x + 40, navigationHeading.y + navigationHeading.height / 2);
    await page.mouse.wheel(0, -120); await settle();
    assert.deepEqual(await page.evaluate(() => graphInputs.map(event => [event.type, event.deltaY, event.forwarded])), [["wheel", -120, true]], "Interface wheel input reaches the ComfyUI canvas");
    await page.evaluate(() => { graphInputs.length = 0; });
    await page.mouse.down({ button: "middle" });
    await page.mouse.move(navigationHeading.x + 75, navigationHeading.y + navigationHeading.height / 2 + 15); await settle();
    assert.ok(await page.evaluate(() => graphInputs.some(event => event.type === "pointermove" && event.buttons === 4)), "Canvas pan updates before pointer release");
    await page.mouse.move(5, 5); await page.mouse.up({ button: "middle" });
    const forwardedPan = await page.evaluate(() => structuredClone(graphInputs));
    assert.deepEqual(forwardedPan.slice(0, 2).map(event => event.type), ["pointerdown", "mousedown"]);
    assert.deepEqual(forwardedPan.slice(-2).map(event => event.type), ["pointerup", "mouseup"]);
    assert.ok(forwardedPan.every(event => event.forwarded));
    assert.equal(forwardedPan.at(-1).buttons, 0);
    await page.evaluate(() => { graphInputs.length = 0; document.querySelector(".bg-cg-list").style.maxHeight = "150px"; });
    const scrollList = await page.locator(".bg-cg-list").boundingBox();
    await page.mouse.move(scrollList.x + scrollList.width / 2, scrollList.y + 50); await page.mouse.wheel(0, 100);
    await page.waitForFunction(() => document.querySelector(".bg-cg-list").scrollTop > 0);
    assert.equal(await page.evaluate(() => graphInputs.length), 0, "Scrollable galleries keep wheel input");
    await page.evaluate(() => { const list = document.querySelector(".bg-cg-list"); list.style.maxHeight = ""; list.scrollTop = 0; });
    const modelCanvas = await page.locator("[data-viewer] canvas").boundingBox(), cameraBeforePan = await page.evaluate(() => widget.viewer.getCameraState());
    await page.mouse.move(modelCanvas.x + modelCanvas.width / 2, modelCanvas.y + modelCanvas.height / 2);
    await page.mouse.down({ button: "middle" }); await page.mouse.move(modelCanvas.x + modelCanvas.width / 2 + 30, modelCanvas.y + modelCanvas.height / 2 + 15); await settle();
    assert.notDeepEqual(await page.evaluate(() => widget.viewer.getCameraState().target), cameraBeforePan.target, "Middle drag in the viewport still moves the camera");
    await page.mouse.up({ button: "middle" });
    const cameraBeforeWheel = await page.evaluate(() => widget.viewer.getCameraState());
    await page.mouse.wheel(0, -80); await settle();
    assert.notDeepEqual(await page.evaluate(() => widget.viewer.getCameraState()), cameraBeforeWheel, "Wheel in the viewport still moves the camera");
    assert.equal(await page.evaluate(() => graphInputs.length), 0, "Viewport navigation never reaches the graph canvas");
    await page.mouse.move(navigationHeading.x + 40, navigationHeading.y + navigationHeading.height / 2); await page.mouse.down({ button: "middle" });
    const cleanup = await page.evaluate(() => { const owner = widget; owner.node.onRemoved(); return { disposed: owner.disposed, released: !owner.container._vnccsMiddleMouseCanvasPan, events: graphInputs.length }; });
    assert.ok(cleanup.disposed && cleanup.released);
    assert.deepEqual(await page.evaluate(() => graphInputs.slice(-2).map(event => event.type)), ["pointerup", "mouseup"], "Removal releases an active graph pan");
    await page.mouse.move(5, 5); await page.mouse.up({ button: "middle" });
    assert.equal(await page.evaluate(() => graphInputs.length), cleanup.events, "Removal cleans up navigation listeners");
    scenarios.push({ name: "copied-vnccs-canvas-navigation", forwardedPan });
    await page.evaluate(() => setup()); await settle();
    const resolutionSlider = page.locator('[data-qi2-setting="resolution_mp"]');
    await resolutionSlider.scrollIntoViewIfNeeded();
    const resolutionBounds = await resolutionSlider.boundingBox(), layout = await geometry();
    const nodeSize = await page.evaluate(() => [...widget.node.size]), resolutionSamples = [];
    await page.mouse.move(resolutionBounds.x + 8, resolutionBounds.y + resolutionBounds.height / 2); await page.mouse.down();
    for (let step = 0; step <= 10; step++) {
        await page.mouse.move(resolutionBounds.x + 8 + (resolutionBounds.width - 16) * step / 10, resolutionBounds.y + resolutionBounds.height / 2); await settle();
        const sample = await page.evaluate(() => ({ mp: widget.state.settings.resolution_mp, frame: structuredClone(widget.state.frame),
            capture: widget.viewer.getCaptureSettings(), saved: JSON.parse(widget.node.widgets[0].value).settings.resolution_mp, nodeSize: [...widget.node.size] }));
        assert.equal(sample.saved, sample.mp, "Resolution persists before the pointer is released");
        assert.equal(sample.capture.width, sample.frame.width); assert.equal(sample.capture.height, sample.frame.height);
        assert.deepEqual(sample.nodeSize, nodeSize, "Resolution changes never resize the node");
        const current = await geometry();
        for (const key of ["widget", "source", "viewport", "frame", "gallery", "dock", "toolbar"])
            assert.deepEqual(current[key], layout[key], `${key} stays fixed during resolution dragging`);
        resolutionSamples.push(sample);
    }
    assert.equal(resolutionSamples[0].mp, 1); assert.equal(resolutionSamples.at(-1).mp, 4);
    assert.notDeepEqual(resolutionSamples[0].frame, resolutionSamples.at(-1).frame, "Capture dimensions update while the pointer remains held");
    await page.mouse.move(resolutionBounds.x + 8 + (resolutionBounds.width - 16) * 0.6, resolutionBounds.y + resolutionBounds.height / 2); await settle();
    await capture("after-resolution-slider"); await page.mouse.up();
    scenarios.push({ name: "resolution-slider-keeps-layout-stable", samples: resolutionSamples });
    await page.evaluate(() => setup());
    const slider = page.locator("[data-fov]"), sliderBox = await slider.boundingBox();
    await page.mouse.move(sliderBox.x + sliderBox.width * 0.5, sliderBox.y + sliderBox.height / 2); await page.mouse.down();
    await page.mouse.move(sliderBox.x + sliderBox.width * 0.7, sliderBox.y + sliderBox.height / 2); await settle();
    const live = await page.evaluate(() => ({ slider: Number(widget.fov.value), exact: Number(widget.fovNumber.value), camera: widget.viewer.getCameraState().fov, saved: JSON.parse(widget.node.widgets[0].value).camera.fov }));
    assert.ok(live.camera > 70); assert.equal(live.slider, live.camera); assert.equal(live.exact, live.camera); assert.equal(live.saved, live.camera);
    await page.mouse.up(); assert.equal(await page.evaluate(() => widget.history.undoStack.length), 1);
    await page.locator("[data-recenter]").click();
    const viewport = await page.locator("[data-viewer] canvas").boundingBox(), before = await page.evaluate(() => widget.viewer.getCameraState());
    await page.mouse.move(viewport.x + viewport.width / 2, viewport.y + viewport.height / 2); await page.mouse.down();
    await page.mouse.move(viewport.x + viewport.width / 2 + 30, viewport.y + viewport.height / 2 + 12, { steps: 4 }); await settle();
    assert.notDeepEqual(await page.evaluate(() => widget.viewer.getCameraState().target), before.target, "Camera motion is visible before pointer release");
    const synchronized = await page.evaluate(() => ({ saved: JSON.parse(widget.node.widgets[0].value).camera, current: widget.viewer.getCameraState() }));
    for (const key of ["position", "target", "up"]) synchronized.saved[key].forEach((value, i) => assert.ok(Math.abs(value - synchronized.current[key][i]) < 1e-9, `${key} stays synchronized during camera movement`));
    assert.equal(synchronized.saved.fov, synchronized.current.fov);
    await page.mouse.up(); scenarios.push({ name: "continuous-camera", live });
    const modelFrameBeforePreview = await page.locator(".bg-viewport").boundingBox(), modelCanvasBeforePreview = await page.locator("[data-viewer] canvas").boundingBox();
    await page.locator(`[data-cg-id="${initial.cgs[1].cg_id}"]`).click();
    await page.waitForFunction(() => !widget.busy);
    assert.equal(await page.evaluate(() => widget.viewer.getCameraState().fov), 48);
    const cgPreview = page.locator('.bg-viewport .bg-image-preview'), cgImage = cgPreview.locator("[data-large-preview]");
    await page.waitForFunction(() => document.querySelector('.bg-viewport [data-large-preview]')?.naturalWidth > 0);
    await (await cgPreview.elementHandle()).waitForElementState("stable");
    assert.equal(await cgImage.getAttribute("src"), initial.cgs[1].url, "CG preview loads the original saved image");
    const cgBounds = await cgPreview.boundingBox(), frameBounds = await page.locator(".bg-viewport").boundingBox();
    assertBounds(frameBounds, modelFrameBeforePreview, "Opening a CG preview preserves the model window position and size");
    assertBounds(await cgPreview.locator("[data-preview-viewport]").boundingBox(), modelCanvasBeforePreview,
        "The image navigation surface exactly replaces the previous model canvas");
    assert.equal(await page.locator(".bg-camera-toolbar").isVisible(), false);
    assert.equal(await cgPreview.locator(".vnccs-common-modal-title").count(), 0);
    assert.equal(await cgPreview.evaluate(element => element.getAnimations().length), 0, "Image preview never moves or scales into place");
    assert.ok(Math.abs(cgBounds.x - frameBounds.x - 1) < 1 && Math.abs(cgBounds.width - frameBounds.width + 2) < 1 && Math.abs(cgBounds.height - frameBounds.height + 2) < 1, "CG preview fills the model window without covering other panels");
    await capture("after-cg-preview");
    const previewCamera = await page.evaluate(() => widget.viewer.getCameraState());
    const cgNavigation = cgPreview.locator("[data-preview-viewport]"), cgNavigationBox = await cgNavigation.boundingBox();
    await page.mouse.move(cgNavigationBox.x + cgNavigationBox.width / 2, cgNavigationBox.y + cgNavigationBox.height / 2);
    await page.mouse.wheel(0, -100); await settle();
    const cgTransform = await cgImage.evaluate(image => image.style.transform);
    await page.mouse.down(); await page.mouse.move(cgNavigationBox.x + cgNavigationBox.width / 2 + 30, cgNavigationBox.y + cgNavigationBox.height / 2 + 15); await settle();
    assert.notEqual(await cgImage.evaluate(image => image.style.transform), cgTransform, "CG image pans continuously before pointer release");
    assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), previewCamera, "Image navigation never moves the model camera");
    await page.mouse.up();
    await page.keyboard.press("Escape"); await cgPreview.waitFor({ state: "detached" }); assert.equal(await cgPreview.count(), 0);
    assertBounds(await page.locator(".bg-viewport").boundingBox(), modelFrameBeforePreview, "Closing a preview preserves the model window");
    assertBounds(await page.locator("[data-viewer] canvas").boundingBox(), modelCanvasBeforePreview, "Closing a preview preserves the model canvas");
    assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), previewCamera);
    await page.locator(`[data-cg-id="${initial.cgs[1].cg_id}"]`).click(); await cgPreview.waitFor();
    await cgPreview.locator("[data-close-preview]").click(); await cgPreview.waitFor({ state: "detached" }); assert.equal(await cgPreview.count(), 0);
    scenarios.push({ name: "full-size-cg-preview-in-model-window", geometry: cgBounds });
    const saved = await page.evaluate(() => { const value = { widgets_values: [widget.node.widgets[0].serializeValue()] }; widget.node.onSerialize(value); return value.widgets_values[0]; });
    await page.evaluate(raw => setup({ raw }), saved);
    assert.equal(await page.evaluate(() => widget.state.selected_cg_id), initial.cgs[1].cg_id);
    assert.equal(await page.evaluate(() => widget.viewer.getCameraState().fov), 48); scenarios.push({ name: "saved-cg-workflow-restore" });
    const chooseSelect = async (select, label) => {
        const trigger = select.locator("xpath=following-sibling::button[1]");
        await trigger.scrollIntoViewIfNeeded(); await settle(); await trigger.click(); const list = page.locator('[role="listbox"]:not([inert])'); await list.waitFor();
        await list.getByRole("option", { name: label, exact: true }).click();
        await page.waitForFunction(() => !widget.busy); await settle();
    };
    assert.equal(await page.locator('[data-qi2-setting*="lora_stack"]').count(), 0);
    const picker = page.locator(".vnccs-common-creator-model-picker");
    await picker.locator(".is-picker-head").click();
    await capture("after-model-picker");
    await picker.locator(".vnccs-common-creator-model-picker-menu .vnccs-common-creator-model-card").filter({ hasText: "Qwen Image 2.1 Turbo INT8 ConvRot" }).click();
    assert.equal(await page.evaluate(() => widget.state.settings.generation.steps), 8);
    assert.equal(await page.evaluate(() => widget.state.settings.generation.diffusion_model_name), turboModel);
    await picker.locator(".is-picker-head").click();
    await picker.locator(".vnccs-common-creator-model-picker-menu .vnccs-common-creator-model-card").filter({ hasText: "Qwen Image 2.1 INT8 ConvRot" }).click();
    assert.equal(await page.evaluate(() => widget.state.settings.generation.steps), 25);
    await page.locator('[data-qi2-setting="generation.steps"]:not([inert])').fill("27");
    await page.locator('[data-qi2-setting="generation.cfg"]:not([inert])').fill("2.5");
    assert.equal(await page.evaluate(() => widget.state.settings.generation.steps), 27, "Numeric settings update before blur");
    await page.locator('[data-qi2-setting="generation.steps"]:not([inert])').hover();
    await page.getByRole("tooltip").waitFor();
    assert.match(await page.getByRole("tooltip").textContent(), /Number of sampling steps/);
    await chooseSelect(page.locator('[data-qi2-setting="generation.sampler"]:not([inert])'), "dpmpp_2m");
    await chooseSelect(page.locator('[data-qi2-setting="generation.scheduler"]:not([inert])'), "normal");
    const seedMode = page.locator('[data-qi2-setting="generation.seed_mode"]:not([inert])');
    await seedMode.click(); assert.equal(await page.evaluate(() => widget.state.settings.generation.seed_mode), "randomize");
    await seedMode.click(); assert.equal(await page.evaluate(() => widget.state.settings.generation.seed_mode), "fixed");
    await page.locator('[data-qi2-setting="generation.seed"]:not([inert])').fill("12345");
    assert.equal(await page.evaluate(() => widget.state.settings.generation.prompt_enhance), false);
    const resolution = page.locator('[data-qi2-setting="resolution_mp"]:not([inert])'); await resolution.scrollIntoViewIfNeeded(); const resolutionBox = await resolution.boundingBox();
    await page.mouse.click(resolutionBox.x + resolutionBox.width * 0.2, resolutionBox.y + resolutionBox.height / 2); await settle();
    assert.ok(await page.evaluate(() => widget.state.settings.resolution_mp < 2));
    await capture("after-settings"); scenarios.push({ name: "original-qi2-controls", settings: await page.evaluate(() => structuredClone(widget.state.settings)) });
    await page.locator("[data-model-settings]").click(); await page.getByRole("dialog").waitFor();
    const dialogStyle = await page.getByRole("dialog").evaluate(modal => {
        const style = getComputedStyle(modal), owner = getComputedStyle(modal.closest(".vnccs-cg-creator"));
        return { background: style.backgroundColor, radius: style.borderRadius, font: style.fontFamily,
            expectedRadius: owner.getPropertyValue("--radius-md").trim(), expectedFont: owner.fontFamily };
    });
    assert.equal(dialogStyle.background, "rgba(26, 26, 38, 0.96)", "Use the shared Creator dialog surface");
    assert.equal(dialogStyle.radius, dialogStyle.expectedRadius);
    assert.equal(dialogStyle.font, dialogStyle.expectedFont);
    await chooseSelect(page.getByRole("dialog").locator('select[name="num_gaussians"]'), "131072");
    await page.getByRole("dialog").locator('input[name="steps"]').fill("12");
    await page.getByRole("dialog").locator('input[name="guidance_scale"]').fill("2.5");
    await capture("after-model-settings"); await page.getByRole("dialog").getByRole("button", { name: "Save", exact: true }).click();
    assert.equal(await page.evaluate(() => widget.state.model_settings.steps), 12);
    await page.getByRole("button", { name: `Select ${sourceVariant.name} background variant`, exact: true }).click(); await page.waitForFunction(() => !widget.busy);
    assert.equal(await page.getByRole("button", { name: `Select ${sourceVariant.name} background variant`, exact: true }).getAttribute("aria-pressed"), "true", "Chosen variant remains selected in its fixed card");
    assert.deepEqual(await page.evaluate(() => widget.state.selected_variant), sourceVariant);
    await page.waitForFunction(expected => document.querySelector('[data-model-source]')?.getAttribute('src')?.includes(expected), `/sets/${setId}/1`);
    assert.match(await page.locator("[data-model-source]").getAttribute("src"), new RegExp(`/sets/${setId}/1`));
    assert.equal(await page.locator("[data-save-view]").isDisabled(), true);
    const backgroundFrameBeforePreview = await page.locator(".bg-viewport").boundingBox();
    await page.locator("[data-enlarge]").click(); await page.locator('.bg-image-preview').waitFor();
    await page.waitForFunction(() => document.querySelector("[data-large-preview]")?.naturalWidth > 0);
    assert.match(await page.locator("[data-large-preview]").getAttribute("src"), new RegExp(`/sets/${setId}/1`));
    const backgroundPreview = page.locator('.bg-viewport .bg-image-preview');
    await (await backgroundPreview.elementHandle()).waitForElementState("stable");
    const backgroundBounds = await backgroundPreview.boundingBox(), modelBounds = await page.locator(".bg-viewport").boundingBox();
    assertBounds(modelBounds, backgroundFrameBeforePreview, "Opening a background preview preserves the model window");
    assertBounds(await backgroundPreview.locator("[data-preview-viewport]").boundingBox(), backgroundBounds, "Background preview has no modal gutters");
    assert.ok(Math.abs(backgroundBounds.x - modelBounds.x - 1) < 1 && Math.abs(backgroundBounds.width - modelBounds.width + 2) < 1 && Math.abs(backgroundBounds.height - modelBounds.height + 2) < 1, "Background preview fills the model window");
    await capture("after-preview");
    const preview = page.locator("[data-preview-viewport]"), previewBox = await preview.boundingBox();
    await page.mouse.move(previewBox.x + previewBox.width / 2, previewBox.y + previewBox.height / 2); await page.mouse.wheel(0, -100); await settle();
    const transform = await page.locator("[data-large-preview]").evaluate(image => image.style.transform);
    await page.mouse.down(); await page.mouse.move(previewBox.x + previewBox.width / 2 + 25, previewBox.y + previewBox.height / 2 + 12); await settle();
    assert.notEqual(await page.locator("[data-large-preview]").evaluate(image => image.style.transform), transform, "Source preview pans before release");
    await page.mouse.up(); await page.locator("[data-close-preview]").click();
    await page.locator('[data-action="model"]').click(); await page.waitForFunction(() => Boolean(widget.jobId));
    assert.equal(job.values.action, "model"); assert.deepEqual(job.values.source_variant, sourceVariant); assert.equal(job.values.model_settings.steps, 12);
    assert.match(await page.locator("[data-model-status]").textContent(), /Creating model/);
    assert.equal(await page.locator(".bg-job-feedback,progress,[data-stop]").count(), 0);
    assert.equal(await page.locator(".bg-status,footer").count(), 0); await capture("after-model-generating");
    await page.request.post(origin + "/__fixture__/complete"); await page.waitForFunction(() => !widget.busy);
    assert.equal(await page.evaluate(() => widget.state.model_id), variantModelId);
    await chooseSelect(page.locator("[data-aspect]"), "9:16");
    assert.ok(await page.evaluate(() => widget.viewer.getCaptureSettings().height > widget.viewer.getCaptureSettings().width), "Aspect selection immediately previews a portrait frame");
    await chooseSelect(page.locator("[data-aspect]"), "4:3");
    await page.locator("[data-save-view]").click();
    const nameDialog = page.getByRole("dialog"), sceneName = nameDialog.getByLabel("Scene name (required)");
    await nameDialog.getByRole("button", { name: "Generate", exact: true }).click();
    assert.equal(await nameDialog.isVisible(), true); assert.equal(job.values.action, "model", "An empty scene name never starts generation");
    await sceneName.fill("Morning camera");
    await nameDialog.getByRole("button", { name: "Generate", exact: true }).click();
    await page.waitForFunction(() => Boolean(widget.jobId));
    await page.waitForFunction(() => document.querySelector("[data-pending-cg] .vnccs-common-creator-generation-scan")?.getAnimations().some(animation => animation.playState === "running" && animation.currentTime >= 600));
    await capture("after-generating");
    await page.emulateMedia({ reducedMotion: "reduce" });
    assert.equal(await page.locator(".vnccs-common-creator-generation-scan").evaluate(element => getComputedStyle(element).display), "none");
    assert.equal(await page.locator(".vnccs-common-creator-generation-base").evaluate(element => getComputedStyle(element).animationName), "none");
    assert.equal(await page.locator("[data-pending-cg]").isVisible(), true);
    await page.emulateMedia({ reducedMotion: "no-preference" });
    scenarios.push({ name: "reduced-motion-keeps-pending-result-visible" });
    assert.equal(await page.getByRole("dialog").count(), 0, "The scene name dialog closes before capture starts");
    assert.equal(await page.locator("[data-fov]").isDisabled(), true);
    assert.equal(await page.locator(".bg-job-feedback,progress,[data-stop]").count(), 0);
    assert.equal(await page.locator(".bg-status,footer").count(), 0);
    const pendingCard = page.locator("[data-pending-cg]");
    assert.equal(await pendingCard.isEnabled(), true);
    assert.equal(await page.locator("[data-cg-count]").textContent(), "1");
    assert.equal(await page.locator("[data-cg-empty]").isVisible(), false);
    const queuedCamera = await page.evaluate(() => widget.viewer.getCameraState());
    const queuedOutput = await page.evaluate(() => JSON.parse(widget.node.widgets[0].value));
    assert.ok(!JSON.stringify(queuedOutput).includes("blob:"), "Pending card is never serialized");
    await pendingCard.click();
    assert.equal(await page.locator(".bg-camera-toolbar").isVisible(), false);
    await page.waitForFunction(() => document.querySelector('.bg-viewport [data-large-preview]')?.naturalWidth > 0);
    assert.ok((await page.locator('.bg-viewport [data-large-preview]').getAttribute("src")).startsWith("blob:"));
    await page.locator('.bg-viewport [data-close-preview]').click();
    await page.locator('.bg-viewport .bg-image-preview').waitFor({ state: "detached" });
    assert.equal(await page.locator(".bg-camera-toolbar").isVisible(), true);
    assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), queuedCamera, "Pending preview does not change the generation camera");
    assert.equal(await page.locator(".vnccs-common-creator-generation-scan").count(), 1);
    scenarios.push({ name: "animated-transient-card-and-preview-during-generation" });
    assert.equal(job.values.action, "cg"); assert.deepEqual(job.values.source_variant, sourceVariant); assert.equal(job.values.settings.generation.sampler, "dpmpp_2m");
    assert.equal(job.values.name, "Morning camera");
    assert.equal(job.values.settings.generation.cache_device, "default"); assert.equal(job.values.settings.generation.cache_dtype, "default");
    assert.equal(job.values.settings.prompt, "Change the camera angle from <image2> to <image1>.");
    assert.equal(job.values.settings.generation.prompt_enhance, false);
    await page.request.post(origin + "/__fixture__/complete"); await page.waitForFunction(() => !widget.busy);
    assert.equal(await page.locator("[data-pending-cg]").count(), 0);
    assert.equal(await page.locator("[data-cgs] .bg-cg-card").count(), 1);
    assert.equal(await page.evaluate(() => JSON.parse(widget.node.widgets[0].value).selected_cg_id), "9".repeat(32));
    await capture("after-completed"); scenarios.push({ name: "named-variant-model-and-cg", model: mutations.find(item => item.action === "model"), cg: job.values });
    await page.route(`**${base}/backgrounds/${backgroundId}/models/${variantModelId}`, route => route.fulfill({ status: 404, json: { error: "Fixture model unavailable" } }), { times: 1 });
    await page.locator("[data-cgs] .bg-cg-card").click();
    assert.equal(await page.locator(".bg-camera-toolbar").isVisible(), false);
    await page.waitForFunction(() => document.querySelector('.bg-viewport [data-large-preview]')?.naturalWidth > 0 && !widget.busy);
    assert.equal(await page.locator('.bg-viewport [data-large-preview]').getAttribute("src"), document.cgs.at(-1).url);
    await page.locator('.bg-viewport [data-close-preview]').click();
    await page.locator('.bg-viewport .bg-image-preview').waitFor({ state: "detached" });
    scenarios.push({ name: "saved-cg-preview-survives-missing-model" });
    const beforeDelete = await page.evaluate(() => widget.viewer.getCameraState());
    const deleteCount = mutations.filter(item => item.delete_cg).length;
    await page.locator("[data-delete-cg]").click();
    assert.equal(mutations.filter(item => item.delete_cg).length, deleteCount);
    await page.getByRole("dialog", { name: "Delete CG", exact: true }).getByRole("button", { name: "Delete", exact: true }).click();
    await page.waitForFunction(() => !widget.busy);
    assert.equal(await page.locator("[data-cgs] .bg-cg-card").count(), 0);
    assert.equal(await page.locator("[data-cg-count]").textContent(), "0");
    assert.equal(await page.evaluate(() => JSON.parse(widget.node.widgets[0].value).selected_cg_id), "");
    assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), beforeDelete);
    scenarios.push({ name: "delete-cg-without-changing-camera-or-background" });
    const stepsField = page.locator('[data-qi2-setting="generation.steps"]');
    const previousSteps = await page.evaluate(() => widget.state.settings.generation.steps);
    await stepsField.fill("");
    assert.equal(await stepsField.inputValue(), "");
    assert.equal(await page.evaluate(() => widget.state.settings.generation.steps), previousSteps);
    await stepsField.blur();
    assert.equal(await stepsField.inputValue(), String(previousSteps));
    const cfgField = page.locator('[data-qi2-setting="generation.cfg"]');
    await cfgField.fill("0");
    assert.equal(await page.evaluate(() => JSON.parse(widget.node.widgets[0].value).settings.generation.cfg), 0);
    await cfgField.blur();
    scenarios.push({ name: "numeric-drafts-and-zero-cfg" });
    catalogAvailable = false;
    const missingCatalog = page.waitForResponse(response => response.url().includes("/vnccs/control_center/check"));
    await page.evaluate(async () => { localStorage.clear(); await setup(); }); await missingCatalog; await settle();
    assert.ok(await page.locator(".vnccs-common-creator-model-picker .is-picker-head").count());
    assert.equal(await page.locator(".vnccs-common-creator-lora-stack").count(), 0);
    assert.doesNotMatch(await page.locator(".bg-hint[role=status]").textContent(), /failed to load Control Center/i);
    await page.locator(".vnccs-common-creator-model-picker .is-picker-head").click();
    assert.equal(await page.locator(".vnccs-common-creator-model-picker-menu .vnccs-common-creator-model-card").count(), 2);
    await page.locator(".vnccs-common-creator-model-picker-menu .vnccs-common-creator-model-card").filter({ hasText: "Qwen Image 2.1 Turbo INT8 ConvRot" }).click();
    assert.equal(await page.evaluate(() => widget.state.settings.generation.steps), 8);
    assert.equal(await page.evaluate(() => widget.state.settings.generation.cfg), 1);
    assert.equal(await page.evaluate(() => widget.state.settings.generation.diffusion_model_name), turboModel);
    scenarios.push({ name: "installed-model-catalog-unavailable-fallback" }); catalogAvailable = true;
    await page.evaluate(() => setup({ empty: true })); await capture("after-empty");
    assert.equal(await page.locator('[data-action="model"]').isDisabled(), true); assert.equal(await page.locator("[data-save-view]").isDisabled(), true);
    readOnly = true; await page.evaluate(() => setup()); await capture("after-read-only");
    assert.equal(await page.locator('[data-action="model"]').isDisabled(), true); assert.equal(await page.locator("[data-save-view]").isDisabled(), true);
    assert.match(await page.locator(".bg-hint[role=status]").textContent(), /read-only/);
    assert.equal(await page.locator('.bg-hint[role="status"]').isVisible(), true); scenarios.push({ name: "empty-and-recovered" });
    assert.deepEqual(unexpected, []); assert.deepEqual(errors, []);
    await writeFile(join(output, "report.json"), JSON.stringify({ fixture: "Deterministic room; production widget and Gaussian renderer; inference mocked", scenarios, mutations: mutations.length, errors, unexpected, warnings }, null, 2));
    console.log(JSON.stringify({ output, scenarios: scenarios.length, errors, unexpected, warnings }));
} catch (error) {
    await page.screenshot({ path: join(output, "after-failure.png"), fullPage: true });
    await writeFile(join(output, "report.json"), JSON.stringify({ scenarios, geometry: await geometry().catch(() => null), errors, unexpected, warnings, failure: String(error) }, null, 2));
    throw error;
} finally { await browser.close(); server.close(); }
