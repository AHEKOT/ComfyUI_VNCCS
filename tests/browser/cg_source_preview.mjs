// Production CG widget/CSS; local API and external 3D-provider fixtures; isolated test Chromium.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { settingsDefaults, LAYOUT_WIDTH } from "../helpers/cg_frontend.mjs";
import { base, backgroundId, imageId, modelId, setId, baseModel, turboModel, document as saved, scene, controlCenter } from "./fixtures/cg_creator.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const root = resolve(fileURLToPath(new URL("../../", import.meta.url))), output = resolve(process.argv[3]);
await mkdir(output, { recursive: true });
const room = await readFile(new URL("./fixtures/room.svg", import.meta.url));
const document = { ...structuredClone(saved), cg_settings: settingsDefaults() };
const nextBackgroundId = "4".repeat(32), nextModelId = "5".repeat(32);
const nextDocument = JSON.parse(JSON.stringify(document).replaceAll(backgroundId, nextBackgroundId).replaceAll(modelId, nextModelId));
nextDocument.name = "Morning garden room"; nextDocument.cg_settings.prompt = "Morning light through the garden windows";
const nextScene = JSON.parse(JSON.stringify(scene).replaceAll(backgroundId, nextBackgroundId).replaceAll(modelId, nextModelId));
const localModel = "User models/other_family_custom.safetensors";
const config = { factory: { version: 1, module_url: "/extensions/Fixture/factory3d/cg_provider.mjs" }, installed: true,
    settings: settingsDefaults(), diffusion_models: [baseModel, turboModel, localModel], model_presets: controlCenter.models,
    samplers: ["euler", "dpmpp_2m"], schedulers: ["simple", "normal"], loras: [] };
const provider = `export {FactoryCommandHistory} from '/tests/fixtures/factory_history.mjs';
export const stylesURL='/fixture.css';
export class Factory3DViewer {
 constructor(host){this.controls={};this.canvas=document.createElement('canvas');this.canvas.width=1600;this.canvas.height=900;this.canvas.style.cssText='width:100%;height:100%;display:block';host.append(this.canvas);}
 async setScene(scene){this.canvas.getContext('2d').clearRect(0,0,1600,900);window.loadingFixtureModel=scene.background_id;await window.fixtureModelGate;
  const image=new Image();image.src='/fixture-model.svg?background='+scene.background_id;await image.decode();this.canvas.getContext('2d').drawImage(image,0,0,1600,900);
  this.camera=structuredClone(scene.camera);return {failures:[]};}
 async capturePreview(){return new Promise(resolve=>this.canvas.toBlob(resolve));}
 setCameraState(value){this.camera=structuredClone(value);} getCameraState(){return structuredClone(this.camera);}
 setLighting(){} setCaptureSettings(){} select(){} resize(){} dispose(){this.canvas.remove();}
}`;
const html = `<!doctype html><meta charset="utf-8"><style>
body{margin:0;padding:20px;background:#202024}#node{position:relative;box-sizing:border-box;padding:12px;border:1px solid #806074;border-radius:14px;background:#17121d}
header{height:64px;box-sizing:border-box;padding:12px;color:#ffe3eb;font:18px sans-serif}
</style><div id="node"><header>VNCCS CG Creator</header></div><script type="module">
import '/web/vnccs_cg_creator.js';import {app} from '/scripts/app.js';
window.setup=async(width=${LAYOUT_WIDTH},scale=1,raw=null,height=950,extra=null)=>{
 window.widget?.node.onRemoved();const host=document.querySelector('#node');host.querySelectorAll('.vnccs-cg-widget').forEach(element=>element.remove());host.style.transform='scale('+scale+')';host.style.transformOrigin='top left';
 if(extra)app.graph.extra=extra;
 const saved=await(await fetch('/fixture')).json();
 class Node{
  constructor(){this.id=7;this.type='VNCCS_CGCreator';this.size=[width,950];this.properties={};this.graph=app.graph;this.widgets=[{name:'cg_data',value:'unconfigured'}];}
  setSize(size){this.size=[...size];host.style.width=size[0]+'px';host.style.height=size[1]+'px';this.widgets.find(widget=>widget.element)?.triggerDraw();}
  addDOMWidget(name,type,element,options){const widget={name,type,element,options,_node:this,triggerDraw:()=>{element.style.width=(this.size[0]-26)+'px';element.style.height=(this.size[1]-90)+'px';}};this.widgets.push(widget);return widget;}
 }
 app.extensions.find(extension=>extension.name==='VNCCS.CGCreator').beforeRegisterNodeDef(Node,{name:'VNCCS_CGCreator'});
 const node=new Node();node.onNodeCreated();window.widget=node.cgCreator;host.append(widget.frame);
 node.widgets[0].value=raw||JSON.stringify({schema_version:1,background_id:saved.background_id,selected_image_id:saved.selected_image_id,model_id:saved.models[0].model_id,selected_cg_id:'',settings:saved.cg_settings});
 node.onConfigure();clearTimeout(node._cgRestore);node.setSize([width,height]);node.onResize();await widget.restore();if(widget.restoring)throw Error(widget.status.textContent);await document.fonts.ready;
};window.ready=true;
</script>`;
const mime = { ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml" };
const server = createServer(async (request, response) => {
    try {
        const path = new URL(request.url, "http://localhost").pathname;
        let body, type = "application/json";
        if (path === "/") { body = html; type = "text/html"; }
        else if (path === "/scripts/app.js") { body = "export const app={extensions:[],graph:{extra:{},setDirtyCanvas(){}},registerExtension(value){this.extensions.push(value)}}"; type = "text/javascript"; }
        else if (path === "/scripts/api.js") { body = "export const api=new EventTarget();api.apiURL=value=>value;api.fetchApi=(path,options)=>fetch(path,options)"; type = "text/javascript"; }
        else if (path === config.factory.module_url) { body = provider; type = "text/javascript"; }
        else if (path === "/fixture.css") { body = ""; type = "text/css"; }
        else if (path.startsWith("/web/") || path === "/tests/fixtures/factory_history.mjs") {
            const file = resolve(root, "." + decodeURIComponent(path)); assert.ok(file.startsWith(root + sep));
            body = await readFile(file); type = mime[extname(file)] || "application/octet-stream";
        } else if (path === "/fixture" || path === `${base}/backgrounds/${backgroundId}`) body = JSON.stringify(document);
        else if (path === `${base}/backgrounds/${nextBackgroundId}`) body = JSON.stringify(nextDocument);
        else if (path === `${base}/backgrounds`) body = JSON.stringify({ backgrounds: [document, nextDocument] });
        else if (path === `${base}/config`) body = JSON.stringify(config);
        else if (path === "/vnccs/control_center/check") body = JSON.stringify(controlCenter);
        else if (path === "/vnccs/manager/status") body = "{}";
        else if (path === `${base}/backgrounds/${backgroundId}/models/${modelId}`) body = JSON.stringify(scene);
        else if (path === `${base}/backgrounds/${nextBackgroundId}/models/${nextModelId}`) body = JSON.stringify(nextScene);
        else if (path.startsWith(`${base}/backgrounds/${backgroundId}/`) || path.startsWith(`${base}/backgrounds/${nextBackgroundId}/`) || path === "/fixture-model.svg") {
            body = path.includes(nextBackgroundId) || request.url.includes(`background=${nextBackgroundId}`) ? room.toString().replaceAll("#f4b38e", "#a2d4af").replaceAll("#865563", "#526975")
                : path.endsWith(`/sets/${setId}/1`) ? room.toString().replaceAll("#f4b38e", "#c7dbf0")
                : path.endsWith(`/sets/${setId}/2`) ? room.toString().replaceAll("#f4b38e", "#807eaa") : room;
            type = "image/svg+xml";
        }
        else if (path === "/favicon.ico") { response.writeHead(204).end(); return; }
        else throw Error(`Unexpected fixture request: ${path}`);
        response.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(body);
    } catch (error) { response.writeHead(500).end(String(error)); }
});
await new Promise((done, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", done); });
let browser, page;
const errors = [], samples = [];
try {
    browser = await chromium.launch({ headless: true, executablePath: process.argv[2] });
    page = await browser.newPage({ viewport: { width: 5600, height: 3500 }, deviceScaleFactor: 1 });
    page.on("pageerror", error => errors.push(String(error)));
    const origin = `http://127.0.0.1:${server.address().port}`;
    await page.route("**/*", route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    const settle = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
    const geometry = () => page.evaluate(() => Object.fromEntries(["#node", ".vnccs-cg-widget", ".bg-model-source", ".bg-preview", ".bg-variation-cards", ".bg-viewport-panel", ".bg-viewport", ".bg-generation-dock", ".bg-cg-panel"].map(selector => {
        const element = document.querySelector(selector), rect = element.getBoundingClientRect();
        return [selector, Object.fromEntries(["x", "y", "width", "height"].map(key => [key, rect[key]]))];
    })));
    const equalGeometry = (actual, expected) => {
        for (const selector of Object.keys(expected)) for (const key of ["x", "y", "width", "height"])
            assert.ok(Math.abs(actual[selector][key] - expected[selector][key]) <= 1, `${selector}.${key} changed`);
    };
    const capture = async name => { await settle(); await page.locator("#node").screenshot({ path: join(output, name + ".png"), animations: "allow" }); };
    const ready = async (width = LAYOUT_WIDTH, scale = 1) => {
        await page.evaluate(([width, scale]) => setup(width, scale), [width, scale]);
        await page.waitForFunction(() => { const image = document.querySelector("[data-model-source]"); return image && !image.hidden && image.complete; }); await settle();
    };
    const switchBackground = id => page.evaluate(id => {
        const select = widget.container.querySelector('[data-library]'); select.value = id;
        window.backgroundNavigation = widget.change({ target: select });
    }, id);
    const backgroundSettled = () => page.evaluate(() => window.backgroundNavigation);
    const workflow = () => page.evaluate(() => {
        const info = { widgets_values: ['stale'] }; widget.node.onSerialize(info);
        return { raw: info.widgets_values[0], size: [...widget.node.size], extra: structuredClone(widget.node.graph.extra) };
    });
    const reloadWorkflow = async () => {
        const saved = await workflow();
        await page.reload(); await page.waitForFunction(() => window.ready);
        await page.evaluate(saved => setup(saved.size[0], 1, saved.raw, saved.size[1], saved.extra), saved);
        await page.waitForFunction(() => document.querySelector('[data-model-source]')?.naturalWidth > 0); await settle();
        return saved;
    };
    await page.goto(origin); await page.waitForFunction(() => window.ready); await ready();
    await page.evaluate(() => {
        const controls = document.createElement('nav'); controls.id = 'workflow-controls';
        controls.innerHTML = '<button data-workflow-tab>Workflow tab</button><button data-workflow-save>Save workflow</button><input aria-label="Workflow name" value="Original workflow"><button data-workflow-after>After preview</button>';
        document.querySelector('#node').before(controls);
        controls.querySelector('[data-workflow-save]').onclick = () => {
            const info = { widgets_values: ['stale'] }; widget.node.onSerialize(info); window.savedWorkflow = JSON.stringify(info);
        };
        controls.querySelector('[data-workflow-tab]').oncontextmenu = event => {
            event.preventDefault();
            const menu = document.createElement('div'); menu.setAttribute('role', 'menu');
            const rename = document.createElement('button'); rename.setAttribute('role', 'menuitem'); rename.textContent = 'Rename';
            rename.onblur = () => menu.remove();
            rename.onclick = () => { controls.querySelector('input').focus(); menu.remove(); };
            menu.append(rename); controls.append(menu); rename.focus();
        };
    });
    for (const [width, scale, previewSelector, label] of [[1481, 1, '[data-cg-id]', 'compact'], [LAYOUT_WIDTH, 1, '[data-cg-id]', 'default'], [5200, 1, '[data-cg-id]', 'wide'], [LAYOUT_WIDTH, .7, '[data-cg-id]', 'scaled'], [LAYOUT_WIDTH, 1, '[data-enlarge]', 'source']]) {
        await ready(width, scale);
        await page.locator(previewSelector).first().click();
        await page.waitForFunction(() => document.querySelector('[data-large-preview]')?.naturalWidth > 0 && !widget.busy);
        const before = await geometry(); await capture(`preview-external-controls-${label}`);
        await page.locator('[data-workflow-save]').click();
        assert.ok(await page.evaluate(() => window.savedWorkflow), 'The workflow saves with its preview open');
        const name = page.getByRole('textbox', { name: 'Workflow name' });
        await name.focus();
        assert.equal(await name.evaluate(element => element === document.activeElement), true, 'Inline preview must not steal focus from workflow controls');
        await name.fill('Renamed workflow'); assert.equal(await name.inputValue(), 'Renamed workflow');
        await page.locator('[data-workflow-tab]').click({ button: 'right' });
        const rename = page.getByRole('menuitem', { name: 'Rename', exact: true });
        await rename.waitFor({ state: 'visible' }); await rename.click();
        assert.equal(await name.evaluate(element => element === document.activeElement), true);
        assert.equal(await page.locator('[data-large-preview]').count(), 1, 'External controls leave the preview open');
        equalGeometry(await geometry(), before);
        await page.locator('[data-close-preview]').focus(); await page.keyboard.press('Tab');
        assert.equal(await page.locator('[data-close-preview]').evaluate(element => element === document.activeElement), false, 'Tab can leave the inline preview');
        await page.locator('[data-preview-viewport]').focus(); await page.keyboard.press('Escape');
        assert.equal(await page.locator('[data-large-preview]').count(), 0);
        samples.push({ name: `inline-preview-external-controls-${label}`, width, scale, before, after: await geometry() });
    }
    await page.evaluate(() => document.querySelector('#workflow-controls').remove());
    samples.push({ name: 'inline-previews-allow-workflow-save-rename-and-tab-context-menu' });
    const deleteId = saved.cgs[0].cg_id, deleteRoute = `**${base}/backgrounds/${backgroundId}/cg/${deleteId}`;
    let deletions = 0, releaseDelete, receivedDelete;
    await page.route(deleteRoute, async route => {
        assert.equal(route.request().method(), 'DELETE'); deletions++; receivedDelete();
        await new Promise(done => { releaseDelete = done; });
        await route.fulfill({ json: { ...document, revision: document.revision + 1, cgs: document.cgs.filter(cg => cg.cg_id !== deleteId) } });
    });
    for (const [width, scale, name] of [[1481, 1, 'compact'], [LAYOUT_WIDTH, 1, 'default'], [5200, 1, 'wide'], [LAYOUT_WIDTH, .7, 'scaled']]) {
        await ready(width, scale);
        const cgName = name === 'compact' ? '<Scene> "Evening room" with a long descriptive title '.repeat(2).slice(0, 96) : saved.cgs[0].name;
        await page.evaluate(name => { widget.document.cgs[0].name = name; widget.render(); }, cgName);
        const before = await geometry(), raw = (await workflow()).raw;
        const camera = await page.evaluate(() => widget.viewer.getCameraState()), count = deletions;
        await capture(`delete-${name}-before`);
        const remove = page.locator(`[data-delete-cg="${deleteId}"]`), dialog = page.getByRole('dialog', { name: 'Delete CG', exact: true });
        for (const dismiss of ['Cancel', 'Escape']) {
            await remove.click(); await settle();
            assert.equal(deletions, count, 'Opening confirmation never submits deletion');
            assert.equal(await dialog.locator('.bg-dialog-fields').textContent(), `Delete "${cgName}" from the CG library?`);
            assert.equal(await dialog.getByRole('button', { name: 'Cancel', exact: true }).evaluate(element => element === document.activeElement), true);
            assert.equal(await dialog.getByRole('button', { name: 'Delete', exact: true }).evaluate(element => element.classList.contains('vnccs-common-modal-btn-danger')), true);
            equalGeometry(await geometry(), before);
            const bounds = await dialog.boundingBox(), nodeBounds = before['#node'];
            assert.ok(bounds.x >= nodeBounds.x && bounds.y >= nodeBounds.y && bounds.x + bounds.width <= nodeBounds.x + nodeBounds.width && bounds.y + bounds.height <= nodeBounds.y + nodeBounds.height);
            assert.equal(await dialog.evaluate(element => element.scrollWidth > element.clientWidth || element.scrollHeight > element.clientHeight), false);
            const buttons = await dialog.getByRole('button').evaluateAll(elements => elements.map(element => { const rect = element.getBoundingClientRect(); return { y: rect.y, height: rect.height }; }));
            assert.ok(Math.abs(buttons[0].y - buttons[1].y) <= 1 && Math.abs(buttons[0].height - buttons[1].height) <= 1);
            if (dismiss === 'Cancel') { await capture(`delete-${name}-confirmation`); await dialog.getByRole('button', { name: 'Cancel', exact: true }).click(); }
            else await page.keyboard.press('Escape');
            await dialog.waitFor({ state: 'detached' }); await page.waitForFunction(() => !widget.busy);
            assert.equal(deletions, count); assert.equal((await workflow()).raw, raw);
            assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), camera);
            equalGeometry(await geometry(), before);
        }
        await capture(`delete-${name}-cancelled`);
        const received = new Promise(done => { receivedDelete = done; });
        await remove.click(); await dialog.getByRole('button', { name: 'Delete', exact: true }).click(); await received;
        assert.equal(await page.locator('[data-delete-cg]').count(), saved.cgs.length, 'Cards stay until the server acknowledges deletion');
        releaseDelete(); await page.waitForFunction(() => !widget.busy);
        assert.equal(deletions, count + 1); assert.equal(await page.locator('[data-delete-cg]').count(), saved.cgs.length - 1);
        assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), camera);
        equalGeometry(await geometry(), before); await capture(`delete-${name}-confirmed`);
        samples.push({ name: `delete-confirmation-${name}`, width, scale, before, after: await geometry(), deletions: deletions - count });
    }
    await page.unroute(deleteRoute); await ready();
    assert.equal(await page.locator('[data-refresh]').count(), 0, 'The library uses shared automatic refresh without an extra button');
    const automaticGeometry = await geometry(), automaticWorkflow = await workflow();
    await capture('automatic-library-before');
    const automaticRoute = `**${base}/backgrounds`, addedId = '8'.repeat(32);
    let releaseCatalog, catalogRequested;
    const catalogGate = new Promise(done => { releaseCatalog = done; }), catalogReceived = new Promise(done => { catalogRequested = done; });
    await page.route(automaticRoute, async route => {
        catalogRequested(); await catalogGate;
        await route.fulfill({ json: { backgrounds: [document, nextDocument, { ...document, background_id: addedId, name: 'Automatically discovered background' }] } });
    });
    await page.evaluate(() => window.dispatchEvent(new Event('focus'))); await catalogReceived;
    equalGeometry(await geometry(), automaticGeometry); await capture('automatic-library-loading');
    releaseCatalog();
    await page.waitForFunction(id => !widget.busy && [...widget.container.querySelector('[data-library]').options].some(option => option.value === id), addedId);
    assert.equal((await workflow()).raw, automaticWorkflow.raw, 'Automatic library updates retain the selected scene, model and camera');
    equalGeometry(await geometry(), automaticGeometry); await capture('automatic-library-after');
    samples.push({ name: 'automatic-library-without-refresh-button', before: automaticGeometry, after: await geometry() });
    await page.unroute(automaticRoute);
    const modelDocument = structuredClone(document), modelDocumentRoute = `**/backgrounds/${backgroundId}`;
    await page.route(modelDocumentRoute, async route => {
        if (route.request().method() === 'PATCH') {
            const values = route.request().postDataJSON();
            Object.assign(modelDocument, { cg_settings: values.cg_settings, model_settings: values.model_settings, revision: modelDocument.revision + 1 });
        }
        await route.fulfill({ json: modelDocument });
    });
    const picker = page.locator('.vnccs-common-creator-model-picker');
    const selectModel = async name => {
        await picker.locator('.is-picker-head').click();
        await picker.locator('.vnccs-common-creator-model-picker-menu [role="button"]').filter({ hasText: name }).click();
    };
    for (const [filename, label] of [[turboModel, 'Qwen Image 2.1 Turbo INT8 ConvRot'], [localModel, localModel]]) {
        await selectModel(label);
        assert.equal(JSON.parse((await workflow()).raw).settings.generation.diffusion_model_name, filename, 'Model choice is serialized immediately');
        await reloadWorkflow();
        assert.equal(await page.evaluate(() => widget.state.settings.generation.diffusion_model_name), filename);
        assert.match(await picker.locator('.is-picker-head').textContent(), new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
        assert.equal(await page.evaluate(() => widget.unsaved), true, 'A restored model choice still needs its library acknowledgement');
        await switchBackground(nextBackgroundId); await backgroundSettled();
        assert.equal(modelDocument.cg_settings.generation.diffusion_model_name, filename);
        await switchBackground(backgroundId); await backgroundSettled();
        assert.equal(await page.evaluate(() => widget.state.settings.generation.diffusion_model_name), filename, 'Leaving and reopening the background retains its active model');
    }
    await page.evaluate(() => widget.persist());
    assert.equal(modelDocument.cg_settings.generation.diffusion_model_name, localModel, 'Disk acknowledgement retains the active local filename');
    const modelGeometry = await geometry(); await capture('model-selection-before');
    const selectedWorkflow = await workflow(), missingModel = 'Unavailable models/saved_custom.safetensors';
    const missingState = JSON.parse(selectedWorkflow.raw); missingState.settings.generation.diffusion_model_name = missingModel;
    await page.evaluate(raw => setup(undefined, 1, raw), JSON.stringify(missingState));
    assert.ok((await picker.locator('.is-picker-head').textContent()).includes(missingModel), 'A missing saved model keeps its identity instead of displaying the first catalog model');
    assert.match(await picker.locator('.is-picker-head .vnccs-common-creator-model-card-status').textContent(), /Missing/);
    assert.equal(await picker.locator('.is-picker-head button').count(), 0, 'An unknown local model has no registry download action');
    assert.equal(JSON.parse((await workflow()).raw).settings.generation.diffusion_model_name, missingModel);
    equalGeometry(await geometry(), modelGeometry); await capture('model-selection-missing');
    await reloadWorkflow();
    assert.ok((await picker.locator('.is-picker-head').textContent()).includes(missingModel));
    const configRoute = `**${base}/config`, catalogRoute = '**/vnccs/control_center/check?*';
    await page.route(configRoute, route => route.fulfill({ json: { ...config, diffusion_models: [], model_presets: [] } }));
    await page.route(catalogRoute, route => route.fulfill({ status: 503, json: { error: 'Catalog unavailable in fixture' } }));
    await page.evaluate(() => localStorage.clear()); await reloadWorkflow();
    assert.ok((await picker.locator('.is-picker-head').textContent()).includes(missingModel), 'Unavailable catalogs cannot replace the saved model');
    assert.equal(JSON.parse((await workflow()).raw).settings.generation.diffusion_model_name, missingModel);
    equalGeometry(await geometry(), modelGeometry); await capture('model-selection-catalog-unavailable');
    await page.unroute(configRoute); await page.unroute(catalogRoute); await reloadWorkflow();
    await picker.locator('.is-picker-head').click(); await capture('model-selection-picker');
    await picker.locator('.is-picker-head').click();
    await selectModel('Qwen Image 2.1 Turbo INT8 ConvRot');
    await reloadWorkflow();
    assert.equal(await page.evaluate(() => widget.state.settings.generation.diffusion_model_name), turboModel);
    equalGeometry(await geometry(), modelGeometry); await capture('model-selection-after');
    await page.waitForFunction(() => !document.querySelector('.vnccs-common-ui-motion-ghost'));
    await page.locator('[data-qi2-setting="generation.steps"]').fill('19');
    await page.locator('[data-qi2-setting="generation.cfg"]').fill('2.5');
    await page.locator('[data-qi2-setting="generation.seed"]').fill('918273');
    for (const [key, value] of [['sampler', 'dpmpp_2m'], ['scheduler', 'normal']]) {
        const select = page.locator(`[data-qi2-setting="generation.${key}"]`);
        await select.locator('xpath=following-sibling::button[1]').click();
        await page.locator('[role="listbox"]:not([inert])').getByRole('option', { name: value, exact: true }).click();
    }
    const authoredModel = JSON.parse((await workflow()).raw).settings.generation;
    await reloadWorkflow();
    assert.deepEqual(await page.evaluate(() => widget.state.settings.generation), authoredModel, 'Model and all authored sampling settings reload together');
    await page.evaluate(() => widget.persist());
    assert.deepEqual(modelDocument.cg_settings.generation, authoredModel);
    const legacyModelState = JSON.parse((await workflow()).raw);
    delete legacyModelState.settings.generation.diffusion_model_name;
    legacyModelState.settings.generation.steps = 23;
    for (let index = 0; index < 2; index++) {
        await page.evaluate(raw => setup(undefined, 1, raw), JSON.stringify(legacyModelState));
        assert.equal(await page.evaluate(() => widget.state.settings.generation.diffusion_model_name), turboModel, 'A legacy workflow inherits the saved background model');
        assert.equal(await page.evaluate(() => widget.state.settings.generation.steps), 23, 'Explicit legacy sampling edits take priority over library defaults');
    }
    const submissions = [], jobId = '9'.repeat(32), modelJobs = `**/backgrounds/${backgroundId}/jobs`, modelStatus = `**/jobs/${jobId}`;
    await page.route(modelJobs, async route => {
        submissions.push(JSON.parse(/name="data"\r\n\r\n([^\r]+)/.exec(route.request().postData())[1]));
        await route.fulfill({ json: { job_id: jobId } });
    });
    await page.route(modelStatus, route => route.fulfill({ json: { status: 'completed', result: { document: modelDocument } } }));
    const submittedGeneration = await page.evaluate(() => structuredClone(widget.state.settings.generation));
    await page.locator('[data-save-view]').click();
    await page.locator('[role="dialog"] input[name="name"]').fill('Selected model contract');
    await page.locator('[role="dialog"] button').filter({ hasText: 'Generate' }).click();
    await page.waitForFunction(() => !widget.busy && !widget.state.job);
    assert.equal(submissions.length, 1); assert.deepEqual(submissions[0].settings.generation, submittedGeneration, 'Generation submits the restored model and sampling values exactly');
    await page.unroute(modelJobs); await page.unroute(modelStatus);
    equalGeometry(await geometry(), modelGeometry); await capture('model-selection-generated');
    samples.push({ name: 'active-model-reload', installed: [turboModel, localModel], missing: missingModel, before: modelGeometry, after: await geometry() });
    await page.evaluate(() => localStorage.clear()); await page.unroute(modelDocumentRoute); await ready();
    const before = await geometry(), original = await page.locator("[data-model-source]").getAttribute("src");
    await capture("before");
    let release, requested;
    const pending = new Promise(done => { release = done; }), received = new Promise(done => { requested = done; });
    await page.route(`**/sets/${setId}/1*`, async route => { requested(); await pending; await route.fulfill({ contentType: "image/svg+xml", body: room.toString().replaceAll("#f4b38e", "#c7dbf0") }); });
    await page.locator(`[data-source-choice="${imageId}:${setId}:1"]`).click(); await received; await settle();
    await capture("loading");
    assert.equal(await page.locator("[data-model-source]").isVisible(), true, "The displayed image stays visible until the replacement is ready");
    assert.equal(await page.locator("[data-model-source]").getAttribute("src"), original);
    assert.equal(await page.locator("[data-empty]").isVisible(), false, "Variant loading never replaces a visible image with a loading screen");
    equalGeometry(await geometry(), before);
    release();
    await page.waitForFunction(() => [...(widget.customSelects._motionAnimations || [])].some(animation => animation.effect.target.hasAttribute("data-source-incoming")));
    const fade = await page.evaluate(() => {
        const animation = [...widget.customSelects._motionAnimations].find(animation => animation.effect.target.hasAttribute("data-source-incoming"));
        animation.pause(); animation.currentTime = 120;
        return { frames: animation.effect.getKeyframes(), duration: animation.effect.getTiming().duration };
    });
    assert.equal(fade.duration, 240); assert.equal(fade.frames[0].opacity, "0"); assert.equal(fade.frames.at(-1).opacity, "1");
    assert.ok(fade.frames.every(frame => !("transform" in frame)), "Crossfade never moves or scales the image");
    assert.equal(await page.locator("[data-model-source]").isVisible(), true);
    assert.equal(await page.locator("[data-source-incoming]").isVisible(), true);
    assert.equal(await page.locator("[data-enlarge]").isVisible(), false);
    equalGeometry(await geometry(), before); await capture("crossfade");
    await page.evaluate(() => [...widget.customSelects._motionAnimations].forEach(animation => animation.finish()));
    await page.waitForFunction(() => !document.querySelector("[data-source-incoming]"));
    assert.match(await page.locator("[data-model-source]").getAttribute("src"), /\/1\?v=1$/);
    assert.equal(await page.locator("[data-enlarge]").isVisible(), true);
    equalGeometry(await geometry(), before); await capture("after"); samples.push({ width: LAYOUT_WIDTH, before, after: await geometry(), fade });
    await page.route(`**/sets/${setId}/2*`, route => route.abort());
    await page.locator(`[data-source-choice="${imageId}:${setId}:2"]`).click();
    await page.waitForFunction(() => !document.querySelector("[data-source-incoming]"));
    assert.match(await page.locator("[data-model-source]").getAttribute("src"), /\/1\?v=1$/);
    assert.equal(await page.locator("[data-model-source]").isVisible(), true, "A failed replacement preserves the previous frame");
    assert.equal(await page.locator("[data-empty]").isVisible(), false);
    assert.match(await page.locator(".bg-hint").textContent(), /Background image unavailable/);
    equalGeometry(await geometry(), before); await capture("error");
    await page.unroute(`**/sets/${setId}/2*`);
    assert.equal(await page.locator(".bg-camera-toolbar").evaluate(element => element.inert && getComputedStyle(element).opacity === "0"), true);
    let releaseLate, lateRequested;
    const late = new Promise(done => { releaseLate = done; }), lateReceived = new Promise(done => { lateRequested = done; });
    await page.route(`**/sets/${setId}/3*`, async route => { lateRequested(); await late; await route.fulfill({ contentType: "image/svg+xml", body: room }); });
    await page.locator(`[data-source-choice="${imageId}:${setId}:3"]`).click(); await lateReceived;
    await page.locator(`[data-source-choice="${imageId}"]`).click();
    await page.waitForFunction(() => document.querySelector("[data-model-source]")?.src.includes('/images/') && !document.querySelector("[data-source-incoming]"));
    releaseLate(); await page.waitForLoadState("networkidle");
    assert.equal(await page.locator("[data-model-source]").getAttribute("src"), original, "An obsolete delayed response cannot replace the latest selection");
    assert.equal(await page.locator(".bg-hint").evaluate(element => element.classList.contains("error")), false);
    equalGeometry(await geometry(), before); await capture("latest-selection");
    await page.unroute(`**/sets/${setId}/3*`);
    await page.emulateMedia({ reducedMotion: "reduce" });
    for (const [width, scale, name] of [[1481, 1, "compact"], [LAYOUT_WIDTH, .7, "graph-scale"], [5200, 1, "wide"]]) {
        await ready(width, scale); const bounds = await geometry();
        await page.locator(`[data-source-choice="${imageId}:${setId}:1"]`).click();
        await page.waitForFunction(() => document.querySelector("[data-model-source]")?.src.includes('/1?v=1') && !document.querySelector("[data-source-incoming]"));
        equalGeometry(await geometry(), bounds);
        const node = bounds["#node"], mount = bounds[".vnccs-cg-widget"];
        assert.ok(mount.x >= node.x && mount.x + mount.width <= node.x + node.width + 1);
        const clipping = await page.locator(".bg-model-source").evaluate(panel => {
            const bounds = panel.getBoundingClientRect();
            return [...panel.children].filter(child => child.getClientRects().length).some(child => child.getBoundingClientRect().bottom > bounds.bottom + 1);
        });
        assert.equal(clipping, false, "Every source control fits inside its panel");
        await switchBackground(nextBackgroundId); await backgroundSettled();
        equalGeometry(await geometry(), bounds);
        assert.equal(await page.evaluate(() => widget.state.background_id), nextBackgroundId);
        assert.equal(await page.locator("[data-background-snapshot]").count(), 0);
        await capture(name); samples.push({ width, scale, bounds });
    }
    await page.emulateMedia({ reducedMotion: "no-preference" }); await ready();
    const view = await page.locator(".bg-viewport").boundingBox();
    await page.locator("[data-cg-id]").first().click();
    await page.waitForFunction(() => document.querySelector("[data-large-preview]")?.naturalWidth > 0);
    const preview = await page.locator("[data-preview-viewport]").boundingBox(), current = await page.locator(".bg-viewport").boundingBox();
    for (const key of ["x", "y", "width", "height"]) assert.ok(Math.abs(current[key] - view[key]) <= 1);
    assert.ok(Math.abs(preview.width - view.width) <= 2); await capture("cg-preview");
    await page.locator("[data-close-preview]").click();

    const backgroundBefore = await geometry(); await capture("background-before");
    await page.evaluate(() => { window.fixtureModelGate = new Promise(resolve => { window.releaseFixtureModel = resolve; }); });
    let releaseAssets, assetsRequested;
    const assetGate = new Promise(done => { releaseAssets = done; }), assetRequest = new Promise(done => { assetsRequested = done; });
    const heldAsset = `**/backgrounds/${nextBackgroundId}/sets/${setId}/3*`;
    await page.route(heldAsset, async route => { assetsRequested(); await assetGate; await route.continue(); });
    await switchBackground(nextBackgroundId);
    await page.waitForFunction(id => window.loadingFixtureModel === id, nextBackgroundId);
    assert.equal(await page.locator("[data-background-snapshot]").count(), 1);
    assert.equal(await page.evaluate(() => widget.busy), true);
    assert.equal(await page.locator("[data-background-snapshot] canvas").evaluate(canvas => canvas.getContext('2d').getImageData(800, 450, 1, 1).data[3]), 255, "The old model frame covers the cleared loading canvas");
    equalGeometry(await geometry(), backgroundBefore); await capture("background-model-loading");
    await page.evaluate(() => { window.releaseFixtureModel(); window.fixtureModelGate = undefined; }); await assetRequest;
    assert.equal(await page.locator("[data-background-snapshot]").isVisible(), true);
    assert.equal(await page.evaluate(() => widget.busy), true);
    equalGeometry(await geometry(), backgroundBefore); await capture("background-assets-loading");
    releaseAssets();
    await page.waitForFunction(() => [...(widget.customSelects._motionAnimations || [])].some(animation => animation.effect.target.hasAttribute('data-background-snapshot')));
    const backgroundFade = await page.evaluate(() => {
        const animation = [...widget.customSelects._motionAnimations].find(animation => animation.effect.target.hasAttribute('data-background-snapshot'));
        animation.pause(); animation.currentTime = 120;
        return { frames: animation.effect.getKeyframes(), duration: animation.effect.getTiming().duration };
    });
    assert.equal(backgroundFade.duration, 240); assert.equal(backgroundFade.frames[0].opacity, '1'); assert.equal(backgroundFade.frames.at(-1).opacity, '0');
    assert.ok(backgroundFade.frames.every(frame => !('transform' in frame)));
    const loadedAssets = await page.evaluate(() => [...widget.container.querySelectorAll('img')].filter(image => image.src).every(image => image.complete && image.naturalWidth > 0));
    assert.equal(loadedAssets, true, "Every new preview, variant and gallery image is ready before reveal");
    equalGeometry(await geometry(), backgroundBefore); await capture("background-crossfade");
    await page.evaluate(() => [...widget.customSelects._motionAnimations].forEach(animation => animation.finish())); await backgroundSettled();
    assert.equal(await page.locator("[data-background-snapshot]").count(), 0); assert.equal(await page.evaluate(() => widget.busy), false);
    assert.equal(await page.evaluate(() => widget.state.background_id), nextBackgroundId);
    equalGeometry(await geometry(), backgroundBefore); await capture("background-after");
    samples.push({ backgroundBefore, backgroundAfter: await geometry(), backgroundFade }); await page.unroute(heldAsset);

    await page.emulateMedia({ reducedMotion: "reduce" }); await switchBackground(backgroundId); await backgroundSettled();
    const failedAsset = `**/backgrounds/${nextBackgroundId}/sets/${setId}/2*`;
    await page.route(failedAsset, route => route.abort()); await switchBackground(nextBackgroundId); await backgroundSettled();
    assert.equal(await page.evaluate(() => widget.state.background_id), backgroundId);
    assert.equal(await page.locator("[data-background-snapshot]").count(), 0); assert.equal(await page.evaluate(() => widget.busy), false);
    assert.match(await page.locator(".bg-hint").textContent(), /Background assets could not be loaded/);
    equalGeometry(await geometry(), backgroundBefore); await capture("background-error"); await page.unroute(failedAsset);

    let releaseObsolete, obsoleteRequested;
    const obsoleteGate = new Promise(done => { releaseObsolete = done; }), obsoleteRequest = new Promise(done => { obsoleteRequested = done; });
    await page.route(heldAsset, async route => { obsoleteRequested(); await obsoleteGate; await route.continue(); });
    await switchBackground(nextBackgroundId); await obsoleteRequest;
    await ready(); releaseObsolete(); await page.waitForLoadState('networkidle');
    assert.equal(await page.locator("[data-background-snapshot]").count(), 0);
    assert.equal(await page.evaluate(() => widget.state.background_id), backgroundId);
    equalGeometry(await geometry(), backgroundBefore); await capture("background-reconfigured"); await page.unroute(heldAsset);

    const longDocument = structuredClone(document);
    longDocument.cgs = Array.from({ length: 16 }, (_, index) => ({ ...longDocument.cgs[index % 2], cg_id: (index + 16).toString(16).padStart(32, '0'), name: `Saved view ${index + 1}` }));
    const group = longDocument.generated_sets[0];
    for (let index = 4; index < 10; index++) {
        group.variants.push({ id: `extra_${index}`, name: `Extra ${index}` });
        group.images.push({ ...group.images[1], url: `${base}/backgrounds/${backgroundId}/sets/${setId}/${index}?v=1` });
    }
    const documentRoute = `**/backgrounds/${backgroundId}`;
    await page.route(documentRoute, async route => {
        if (route.request().method() === 'PATCH') {
            const values = route.request().postDataJSON(); longDocument.cg_settings = values.cg_settings; longDocument.model_settings = values.model_settings;
        }
        await route.fulfill({ json: longDocument });
    });
    await ready();
    await page.locator('[data-qi2-setting="generation.steps"]').fill('31');
    await page.locator('[data-qi2-setting="generation.cfg"]').fill('2.5');
    await page.locator('[data-model-settings]').click();
    await page.locator('[role="dialog"] input[name="steps"]').fill('33');
    await page.locator('[role="dialog"] button').filter({ hasText: 'Save' }).click();
    await page.locator('[data-cg-id]').first().click();
    await page.waitForFunction(() => document.querySelector('[data-large-preview]')?.naturalWidth > 0 && !widget.busy);
    const previewNavigation = page.locator('[data-preview-viewport]'), box = await previewNavigation.boundingBox();
    await page.mouse.move(box.x + box.width * .4, box.y + box.height * .45); await page.mouse.wheel(0, -200);
    await page.mouse.down(); await page.mouse.move(box.x + box.width * .45, box.y + box.height * .5); await page.mouse.up();
    await page.evaluate(() => { widget.container.querySelector('[data-source-choices]').scrollLeft = 1000; widget.container.querySelector('[data-cgs]').scrollTop = 700; }); await settle();
    const previewGeometry = await geometry(), previewTransform = await page.locator('[data-large-preview]').evaluate(image => image.style.transform);
    const beforeReload = await workflow();
    assert.ok(JSON.parse(beforeReload.raw).scroll.sources > 0); assert.ok(JSON.parse(beforeReload.raw).scroll.cgs > 0);
    assert.ok(JSON.parse(beforeReload.raw).preview.zoom > 1); await capture('reload-preview-before');
    await reloadWorkflow(); await page.waitForFunction(() => document.querySelector('[data-large-preview]')?.naturalWidth > 0); await settle();
    equalGeometry(await geometry(), previewGeometry);
    const restoredTransform = await page.locator('[data-large-preview]').evaluate(image => image.style.transform);
    const matrix = value => value.slice(7, -1).split(',').map(Number);
    matrix(restoredTransform).forEach((value, index) => assert.ok(Math.abs(value - matrix(previewTransform)[index]) < 1e-6));
    const restored = JSON.parse((await workflow()).raw), authored = JSON.parse(beforeReload.raw);
    for (const key of ['background_id', 'selected_image_id', 'selected_variant', 'model_id', 'selected_cg_id', 'camera', 'frame', 'settings', 'model_settings', 'scroll', 'preview']) assert.deepEqual(restored[key], authored[key], `${key} survives a full page reload`);
    assert.equal(await page.locator('.bg-camera-toolbar').evaluate(element => getComputedStyle(element).opacity), '0');
    await capture('reload-preview-after'); samples.push({ name: 'page-reload-preview-and-settings', before: previewGeometry, after: await geometry(), state: restored });
    await page.locator('[data-close-preview]').click();
    await page.locator('[data-fov-number]').fill('74.5'); await page.locator('[data-fov-number]').dispatchEvent('change');
    await reloadWorkflow();
    assert.equal(await page.locator('[data-large-preview]').count(), 0); assert.equal(await page.evaluate(() => widget.viewer.getCameraState().fov), 74.5);
    assert.equal(await page.evaluate(() => widget.canEditCamera()), true); await capture('reload-model-after');
    await page.locator(`[data-source-choice="${imageId}:${setId}:1"]`).click();
    await page.waitForFunction(() => !widget.busy && !document.querySelector('[data-source-incoming]'));
    await page.locator('[data-enlarge]').click(); await page.waitForFunction(() => document.querySelector('[data-large-preview]')?.naturalWidth > 0);
    const variantState = JSON.parse((await workflow()).raw); await reloadWorkflow();
    await page.waitForFunction(() => document.querySelector('[data-large-preview]')?.naturalWidth > 0);
    assert.deepEqual(await page.evaluate(() => widget.state.selected_variant), variantState.selected_variant);
    assert.equal(await page.evaluate(() => widget.state.preview.kind), 'source'); assert.equal(await page.locator('[data-viewer]').isVisible(), false);
    await capture('reload-variant-source-after'); await page.unroute(documentRoute);

    for (const [action, acknowledged] of [['model', true], ['cg', false]]) {
        await ready();
        const jobId = '6'.repeat(32), finishedModelId = action === 'model' ? '7'.repeat(32) : modelId;
        let terminal = false, submissions = 0, lookups = 0, releaseAcknowledgement;
        const acknowledgement = new Promise(done => { releaseAcknowledgement = done; });
        const completed = structuredClone(document);
        if (action === 'model') completed.models.push({ ...completed.models[0], model_id: finishedModelId });
        else completed.cgs.push({ ...completed.cgs[0], cg_id: '8'.repeat(32), name: 'Reloaded generation' });
        const postRoute = `**/backgrounds/${backgroundId}/jobs`, lookupRoute = `**/backgrounds/${backgroundId}/jobs/*`, statusRoute = `**/jobs/${jobId}`;
        const modelRoute = `**/backgrounds/${backgroundId}/models/${finishedModelId}`;
        await page.route(modelRoute, route => route.fulfill({ json: { ...scene, model_id: finishedModelId } }));
        await page.route(postRoute, async route => {
            submissions++;
            if (!acknowledged) { await acknowledgement; await route.abort().catch(error => assert.match(error.message, /closed|Invalid InterceptionId|already|canceled/)); }
            else await route.fulfill({ json: { job_id: jobId } });
        });
        await page.route(lookupRoute, route => { lookups++; return route.fulfill({ json: { job_id: jobId } }); });
        await page.route(statusRoute, route => route.fulfill({ json: terminal ? { status: 'completed', result: { document: completed } } : { status: 'running' } }));
        if (action === 'model') await page.locator('[data-action="model"]').click();
        else {
            await page.locator('[data-save-view]').click(); await page.locator('[role="dialog"] input').fill('Reloaded generation');
            await page.locator('[role="dialog"]').getByRole('button', { name: 'Generate', exact: true }).click();
        }
        await page.waitForFunction(acknowledged => acknowledged ? Boolean(widget.state.job?.id) : Boolean(widget.state.job?.request_id), acknowledged);
        if (action === 'cg') { await page.locator('[data-pending-cg]').click(); await page.waitForFunction(() => document.querySelector('[data-large-preview]')?.naturalWidth > 0); }
        const pendingGeometry = await geometry(); await capture(`reload-${action}-pending-before`);
        await reloadWorkflow(); releaseAcknowledgement();
        await page.waitForFunction(id => widget.jobId === id && widget.busy, jobId);
        if (action === 'cg') await page.waitForFunction(() => widget.pendingCG?.url && document.querySelector('[data-large-preview]')?.naturalWidth > 0);
        assert.equal(submissions, 1); assert.equal(lookups, acknowledged ? 0 : 1);
        assert.equal(await page.locator('[data-action="model"]').isDisabled(), true);
        equalGeometry(await geometry(), pendingGeometry); await capture(`reload-${action}-pending-after`);
        terminal = true; await page.waitForFunction(() => !widget.busy && !widget.state.job);
        assert.equal(await page.evaluate(() => widget.state.model_id), finishedModelId);
        if (action === 'cg') {
            assert.equal(await page.evaluate(() => widget.state.selected_cg_id), '8'.repeat(32));
            assert.equal(await page.evaluate(() => widget.previewSource.cg_id), '8'.repeat(32));
        }
        await capture(`reload-${action}-completed`); samples.push({ name: `page-reload-${action}-job`, acknowledged, submissions, lookups });
        for (const route of [postRoute, lookupRoute, statusRoute, modelRoute]) await page.unroute(route);
    }
    assert.deepEqual(errors, []);
    await writeFile(join(output, "report.json"), JSON.stringify({ fixture: "Production CG UI/CSS; local API and external 3D-provider DOM fixture, no GPU rendering", samples, errors }, null, 2));
    console.log(JSON.stringify({ output, samples: samples.length, errors }));
} catch (error) {
    await page?.screenshot({ path: join(output, "failure.png"), fullPage: true });
    await writeFile(join(output, "failure.json"), JSON.stringify({ failure: String(error), errors }, null, 2));
    throw error;
} finally {
    await browser?.close(); server.close();
}
