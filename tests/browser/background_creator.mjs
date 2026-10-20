// Visual regression: Playwright on NODE_PATH; args: Chromium executable, output folder, optional saved scene folder.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(import.meta.url)("playwright");
const project = fileURLToPath(new URL("../../", import.meta.url));
const output = resolve(process.argv[3] || join(tmpdir(), "vnccs-background-visual"));
await mkdir(output, { recursive: true });
const base = "/vnccs/background-creator";
const backgroundId = "a".repeat(32), imageId = "b".repeat(32);
const assets = new Map();
let document = { schema_version: 2, background_id: backgroundId, name: "Evening room", revision: 1,
    selected_image_id: imageId, editor: { description: "A cozy bedroom with warm evening light.", attributes: { location: "Tokyo", architecture: "wooden interior", lighting: "warm window light", objects: "bed, desk, bookshelves" } },
    images: [{ image_id: imageId, width: 1600, height: 896 }] };
if (process.argv[4]) {
    const folder = resolve(process.argv[4]);
    document = JSON.parse(await readFile(join(folder, folder.split(sep).at(-1) + "_config.json"), "utf8"));
    for (const image of document.images) assets.set(`${base}/backgrounds/${document.background_id}/images/${image.image_id}`, { bytes: await readFile(join(folder, image.file)), type: "image/png" });

} else {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="896"><defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#506488"/><stop offset="1" stop-color="#d79b85"/></linearGradient></defs><rect width="1600" height="896" fill="#9e7a65"/><path d="M0 0h1600L1280 150H320Z" fill="#c3a58b"/><path d="m0 896 320-230h960l320 230" fill="#624e4b"/><rect x="530" y="140" width="540" height="400" fill="url(#sky)" stroke="#483d43" stroke-width="30"/><path d="M800 150v380M540 360h520" stroke="#483d43" stroke-width="18"/><rect x="150" y="560" width="650" height="170" rx="30" fill="#b77a80"/><rect x="1050" y="560" width="390" height="50" fill="#5f4340"/><path d="M1080 610v180m330-180v180" stroke="#5f4340" stroke-width="30"/></svg>';
    assets.set(`${base}/backgrounds/${backgroundId}/images/${imageId}`, { bytes: svg, type: "image/svg+xml" });

}
document = structuredClone(document);
for (const image of document.images) image.url = `${base}/backgrounds/${document.background_id}/images/${image.image_id}`;
const catalog = JSON.parse(await readFile(join(project, "control_center.json"), "utf8"));
for (const category of ["models", "clip", "vae", "lora"]) for (const entry of catalog[category] || []) entry.status = "installed";
const config = { installed: true, assets: [], diffusion_models: ["qwen_image_2.1_int8_convrot.safetensors", "user/Scenery.safetensors", "moss_image_sft_int8_convrot.safetensors"],
    model_types: { moss: { installed: true, assets: [{ category: "diffusion_models", installed: true }, { category: "text_encoders", installed: true }, { category: "vae", installed: true }] } },
    loras: ["scenery/Painted_Backgrounds.safetensors", "scenery/Warm_Light.safetensors"], samplers: ["euler", "er_sde", "dpmpp_2m"], schedulers: ["simple", "normal"] };
const creatorSource = await readFile(join(project, "web/vnccs_character_creator_v2.js"), "utf8");
const { STYLE_PICKER_CSS } = await import("../../web/character_styles.mjs?v=style-mix-9");
const styleCatalog = JSON.parse(await readFile(join(project,"background_template/background_styles.json"),"utf8"));
config.styles = styleCatalog.groups.flatMap(group => group.styles);
const expectedStyleCount = config.styles.length;
const coverAssets = new Map(), coverRequests = [];
const commonSource = await readFile(join(project, "web/vnccs_common.js"), "utf8");
const creatorTheme = commonSource.match(/export const SAKURA_THEME_CSS = `([\s\S]*?)`;/)[1];
const referenceCreatorCSS = `.vnccs-creator-container { ${creatorTheme} }` + creatorSource.slice(creatorSource.indexOf(".vnccs-creator-generation-layer {"), creatorSource.indexOf(".vnccs-creator-preview-loading {"))
    + creatorSource.split("\n").filter(line => line.startsWith("@keyframes vnccs-creator-generation-")).join("\n")
    + STYLE_PICKER_CSS
    + creatorSource.slice(creatorSource.indexOf(".vnccs-creator-input, .vnccs-creator-textarea {"), creatorSource.indexOf("/* Slider */"))
    + creatorSource.slice(creatorSource.indexOf(".vnccs-creator-container .vnccs-common-modal {"), creatorSource.indexOf(".vnccs-creator-qwenvl-download-status {"))
    + creatorSource.slice(creatorSource.indexOf("/* Main Host */"), creatorSource.indexOf("/* Columns */"))
    + creatorSource.slice(creatorSource.indexOf(".vnccs-creator-resolution-field {"), creatorSource.indexOf(".vnccs-creator-overhaul-control {"))
    + creatorSource.slice(creatorSource.indexOf(".vnccs-creator-trait-row {"), creatorSource.indexOf(".vnccs-creator-tag-grid {"));
const appModule = `export const app={extensions:[],graph:{_nodes:[],extra:{},setDirtyCanvas(){}},canvas:{ds:{scale:1,offset:[0,0]}},registerExtension(extension){this.extensions.push(extension)}};`;
const apiModule = `export const api=new EventTarget();api.apiURL=route=>route;api.fetchApi=(route,options)=>fetch(route,options);`;
const html = `<!doctype html><meta charset="utf-8"><title>Background Creator visual regression</title>
<link rel="stylesheet" href="/reference-creator.css"><style>body{margin:0;background:#202024;padding:20px}#host{margin:0 auto;position:relative}</style><div id="host"></div><script type="module">
import '/web/vnccs_background_creator.js';
import '/web/vnccs_character_generator.js';
import { api } from '/scripts/api.js';
import { app } from '/scripts/app.js';
import { editorDefaults, composePrompt } from '/web/vnccs_background_creator.js';
window.composedPrompt=()=>composePrompt(widget.state.editor);
window.setup=async({width=1600,height=1000,empty=false}={})=>{
 window.widget?.node.onRemoved?.();const host=document.querySelector('#host');host.replaceChildren();host.style.height='';
 const saved=await (await fetch('/fixture')).json();
 const state={schema_version:2,background_id:empty?'':saved.background_id,background_name:empty?'':saved.name,selected_image_id:empty?'':saved.selected_image_id,...(empty?{}:{revision:saved.revision}),editor:editorDefaults(empty?{}:saved.editor)};
 class Node{
  constructor(){this.id=7;this.type='VNCCS_BackgroundCreator';this.size=[width,height];this.graph=app.graph;this.properties={};this.widgets=[{name:'seed',value:0},{name:'background_data',value:'unconfigured'}];}
  setSize(size){this.size=[...size];host.style.width=size[0]+'px';this.widgets.find(widget=>widget.element)?.triggerDraw();}
  addDOMWidget(name,type,element,options){const widget={name,type,element,options,_node:this,triggerDraw:()=>{element.style.width=(this.size[0]/0.67)+'px'}};this.widgets.push(widget);return widget;}
 }
 app.extensions.find(extension=>extension.name==='VNCCS.BackgroundCreator').beforeRegisterNodeDef(Node,{name:'VNCCS_BackgroundCreator'});
 const node=new Node();app.graph._nodes=[node];node.onNodeCreated();window.widget=node.backgroundCreator;
 if(node.widgets[1].value!=='unconfigured')throw new Error('Construction overwrote workflow state');
 node.widgets[1].value=JSON.stringify(state);node.onConfigure();clearTimeout(node._backgroundRestore);node.setSize([width,height]);host.append(widget.container);node.onResize();await widget.restore();
 if(widget.restoring)throw new Error(widget.status.textContent);await document.fonts.ready;
};
window.setupGenerator=async({width=1400,height=900}={})=>{
 window.widget?.node.onRemoved?.();const host=document.querySelector('#host');host.replaceChildren();
 host.style.width=width+'px';host.style.height=height+'px';
 class Node{
  constructor(){this.id=23;this.type='VNCCS_BackgroundGenerator';this.size=[width,height];this.graph=app.graph;this.properties={};this.inputs=[];this.widgets=[{name:'widget_data',value:'{}'}];}
  setSize(size){this.size=[...size];this.widgets.find(widget=>widget.element)?.triggerDraw();}
  addDOMWidget(name,type,element,options){const widget={name,type,element,options,_node:this,triggerDraw:()=>{element.style.width=this.size[0]+'px';element.style.height=this.size[1]+'px';}};this.widgets.push(widget);return widget;}
 }
 await app.extensions.find(extension=>extension.name==='VNCCS.CharacterGenerator').beforeRegisterNodeDef(Node,{name:'VNCCS_BackgroundGenerator'});
 const node=new Node();app.graph._nodes=[node];node.onNodeCreated();window.widget=node._vnccsCharacterGeneratorWidget;
 if(node.widgets[0].value!=='{}')throw new Error('Generator construction overwrote saved settings');
 node.widgets[0].value=JSON.stringify({background_id:'fixture',background_variants:['Morning','Evening','Night'],upscaler:{mode:'off'}});
 node.onConfigure();node.setSize([width,height]);host.append(widget.root);node.onResize();await document.fonts.ready;
};
window.generatorEvent=(stage,detail)=>api.dispatchEvent(new CustomEvent('vnccs.character_generator.stage',{detail:{node_id:'23',stage,...detail}}));
window.backgroundSetEvent=detail=>api.dispatchEvent(new CustomEvent('vnccs.background.set',{detail}));
window.connectionEvent=name=>api.dispatchEvent(new CustomEvent(name,{detail:{}}));
window.ready=true;
</script>`;
const mutations = [];
let previewJob = null;
const server = createServer(async (request, response) => {
    try {
        const url = new URL(request.url, "http://localhost");
        let content, type = "application/json";
        if (url.pathname === "/") { content = html; type = "text/html"; }
        else if (url.pathname === "/reference-creator.css") { content = referenceCreatorCSS; type = "text/css"; }
        else if (url.pathname === "/scripts/app.js") { content = appModule; type = "text/javascript"; }
        else if (url.pathname === "/scripts/api.js") { content = apiModule; type = "text/javascript"; }
        else if (url.pathname.startsWith("/web/")) {
            const path = resolve(project, "." + decodeURIComponent(url.pathname));
            assert.ok(path.startsWith(join(project, "web") + sep));
            content = await readFile(path); type = path.endsWith(".png") ? "image/png" : "text/javascript";
        } else if (url.pathname === "/fixture") content = JSON.stringify(document);
        else if (url.pathname === "/__fixture__/complete-background-set") {
            let raw='';for await(const chunk of request)raw+=chunk;
            const cards=JSON.parse(raw), set_id='2'.repeat(32), source_image_id=document.selected_image_id;
            const images=[{name:'Original',url:document.images.find(image=>image.image_id===source_image_id).url}];
            for(const [index,card] of cards.entries()){
                const path=`${base}/backgrounds/${document.background_id}/sets/${set_id}/${index+1}`;
                const color=['#a0bddc','#bd795a','#283550'][index%3];
                const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900"><rect width="1600" height="900" fill="${color}"/><path d="M0 0h1600l-240 160H240Z" fill="#6e7688"/><path d="M0 900 240 650h1120l240 250" fill="#393746"/><rect x="640" y="180" width="330" height="380" fill="${color}" stroke="#323343" stroke-width="24"/><path d="M805 180v380M640 370h330" stroke="#323343" stroke-width="16"/><rect x="120" y="580" width="650" height="160" rx="20" fill="#806775"/><text x="800" y="820" text-anchor="middle" font-family="sans-serif" font-size="64" fill="white">Fixture ${index+1}</text></svg>`;
                assets.set(path,{bytes:svg,type:'image/svg+xml'});images.push({name:card.name,url:path});
            }
            document.generated_sets??=[];document.generated_sets.push({set_id,version:"",source_image_id,variants:cards,images});document.revision++;
            content=JSON.stringify({background_id:document.background_id,source_image_id,set_id,revision:document.revision});
        }
        else if (assets.has(url.pathname)) { ({ bytes: content, type } = assets.get(url.pathname)); }
        else if (url.pathname === base + "/styles") content=JSON.stringify(styleCatalog);
        else if (url.pathname === base + "/styles/preview") {
            if (request.method === 'POST') {
                let raw='';for await(const chunk of request)raw+=chunk;
                const payload=JSON.parse(raw);coverRequests.push(payload);
                assert.equal(payload.character_info,undefined);
                const style=styleCatalog.groups.flatMap(group=>group.styles).find(style=>style.id===payload.style_id);
                assert.ok(style);
                style.image=`${base}/styles/preview?style=${style.id}&v=fixture`;
                const hue=(coverRequests.length*37)%360;
                coverAssets.set(style.id,`<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024"><rect width="1024" height="1024" fill="hsl(${hue} 25% 35%)"/><path d="M0 0h1024L790 180H230Z" fill="hsl(${hue} 20% 65%)"/><path d="M0 1024 230 720h560l234 304" fill="hsl(${hue} 20% 20%)"/><rect x="390" y="230" width="310" height="350" fill="hsl(${(hue+60)%360} 35% 70%)"/><path d="M540 230v350M390 405h310" stroke="#34303b" stroke-width="18"/><rect x="70" y="640" width="400" height="180" rx="30" fill="hsl(${hue} 30% 50%)"/></svg>`);
                content=JSON.stringify({saved:true,style_id:style.id,image:style.image,width:1024,height:1024});
            } else {
                content=coverAssets.get(url.searchParams.get('style'));type='image/svg+xml';
                if(!content){response.writeHead(404);response.end();return;}
            }
        }
        else if (url.pathname === base + "/config") content = JSON.stringify(config);
        else if (url.pathname === base + "/backgrounds") {
            if(request.method==='POST'){
                let raw='';for await(const chunk of request)raw+=chunk;
                const payload=JSON.parse(raw);mutations.push(payload);
                document={schema_version:2,background_id:'8'.repeat(32),name:payload.name,revision:1,editor:payload.editor,selected_image_id:'',images:[]};
                content=JSON.stringify(document);
            }else content = JSON.stringify({ backgrounds: [document] });
        }
        else if (url.pathname === `${base}/backgrounds/${document.background_id}/jobs`) {
            let raw = ""; for await (const chunk of request) raw += chunk;
            const payload = JSON.parse(raw); assert.equal(payload.action, "background");
            previewJob = { job_id: "f".repeat(32), status: "running", progress: 12, message: "Rendering background: 3/25", editor: payload.editor };
            content = JSON.stringify({ job_id: previewJob.job_id });
        } else if (url.pathname === `${base}/jobs/${previewJob?.job_id}`) content = JSON.stringify(previewJob);
        else if (url.pathname === "/__fixture__/finish-preview") {
            const source = document.images.find(image => image.image_id === document.selected_image_id) || document.images[0];
            const image_id = source.preview_revision ? source.image_id : "9".repeat(32);
            const path = `${base}/backgrounds/${document.background_id}/images/${image_id}`;
            assets.set(path, assets.get(new URL(source.url, 'http://localhost').pathname));
            const entry = { ...source, image_id, preview_revision: document.revision + 1, url: `${path}?v=${document.revision + 1}` };
            if(source.preview_revision)Object.assign(source,entry);else document.images.push(entry);
            document.selected_image_id = image_id;
            document.editor = previewJob.editor; document.revision++;
            previewJob = { ...previewJob, status: "completed", progress: 100, result: { document } };
            content = "{}";
        }
        else if (url.pathname === `${base}/backgrounds/${document.background_id}`) {
            if (request.method === "PATCH") {
                let raw = ""; for await (const chunk of request) raw += chunk;
                const data = JSON.parse(raw); mutations.push(data);
                document.editor = data.editor; document.selected_image_id = data.selected_image_id; document.revision++;
            }
            content = request.method === "DELETE"
                ? JSON.stringify({ ok: true, name: document.name, background_id: document.background_id, deleted: true })
                : JSON.stringify(document);
        }
        else if (url.pathname === "/queue") content = JSON.stringify({queue_running:[],queue_pending:[]});
        else if (url.pathname === "/vnccs/control_center/check") content = JSON.stringify(catalog);
        else if (url.pathname === "/vnccs/manager/status") content = "{}";
        else if (url.pathname === "/vnccs/character_generator/progress") content = JSON.stringify({snapshot:null});
        else if (url.pathname === "/vnccs/character_generator/seedvr_models") content = JSON.stringify({models:[{name:"seedvr2_3b_fp8_e4m3fn.safetensors",local_path:"models/diffusion_models/seedvr2_3b_fp8_e4m3fn.safetensors",status:"installed",description:"Native SeedVR source and variant upscaler"}],vae:[]});
        else if (url.pathname === "/vnccs/character_generator/seedvr_attention") content = JSON.stringify({current:"sdpa",available:["sdpa"]});
        else if (url.pathname.startsWith("/object_info")) content = JSON.stringify(Object.fromEntries(["SeedVR2Preprocess","SeedVR2Conditioning","SeedVR2PostProcessing"].map(name=>[name,{input:{required:{color_correction_method:[["lab","wavelet","adain","none"]]}}}])));
        else if (url.pathname === "/favicon.ico") { response.writeHead(204).end(); return; }
        else { response.writeHead(404).end(JSON.stringify({ error: "Unexpected visual fixture route: " + url.pathname })); return; }
        response.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(content);
    } catch (error) { response.writeHead(500).end(JSON.stringify({ error: String(error) })); }
});
await new Promise((done, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", done); });
const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-swiftshader"], ...(process.argv[2] ? { executablePath: process.argv[2] } : {}) })
    .catch(error => { server.close(); throw error; });
const page = await browser.newPage({ viewport: { width: 1800, height: 1120 }, deviceScaleFactor: 2 });
const retiredRequests = [], errors = [], warnings = [], report = { scenarios: [], sceneLoaded: Boolean(process.argv[4]) };
page.on("request", request => { if (/vendor\/spark|background-creator\/backgrounds\/[^/]+\/models/.test(request.url())) retiredRequests.push(request.url()); });
page.on("pageerror", error => errors.push(String(error)));
page.on("console", message => { if (["warning", "error"].includes(message.type())) warnings.push(message.text()); });
const settle = async () => { await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))); };
const capture = async (name, animations = "disabled") => {
    await page.waitForFunction(() => [...document.fonts].some(font => font.family === 'Sora' && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready); await settle(); await page.locator("#host").screenshot({ path: join(output, name + ".png"), animations }); };
const setVariantField=async(key,value,id='morning')=>{
    const input=page.locator(`[data-variation="${id}"] [data-variation-field="${key}"]`);
    await input.locator('..').locator('.vnccs-common-creator-trait-values').click();
    await input.fill(value);await input.press('Tab');
};
const variantCardLayout=async()=>page.locator('.bg-variation-item').evaluateAll(cards=>cards.map(card=>{
    const preview=card.querySelector('.bg-variation-preview').getBoundingClientRect(),settings=card.querySelector('.bg-variation-settings'),bounds=settings.getBoundingClientRect();
    const rows=[...settings.querySelectorAll('.vnccs-common-creator-trait-row')].map(row=>{
        const rowBounds=row.getBoundingClientRect(),label=row.querySelector('.vnccs-common-creator-trait-label').getBoundingClientRect();
        const editor=row.querySelector('.vnccs-common-creator-trait-editor').getBoundingClientRect(),add=row.querySelector('.vnccs-common-creator-trait-add').getBoundingClientRect();
        return {top:rowBounds.top,bottom:rowBounds.bottom,height:rowBounds.height,separated:label.right<=editor.left+1&&editor.right<=add.left+1};
    });
    return {id:card.dataset.variation,width:card.getBoundingClientRect().width,preview:{top:preview.top,bottom:preview.bottom},settings:{top:bounds.top,bottom:bounds.bottom},belowPreview:bounds.top>=preview.bottom-1,rows,overflow:card.scrollWidth>card.clientWidth+1||settings.scrollWidth>settings.clientWidth+1};
}));
const previewNavigation=async name=>{
    const dialog=page.getByRole('dialog'), viewport=dialog.locator('[data-preview-viewport]'), image=dialog.locator('[data-large-preview]');
    await image.waitFor({state:'visible'});await image.evaluate(image=>image.decode());
    await page.getByRole('dialog').evaluate(element=>Promise.allSettled(element.getAnimations({subtree:true}).map(animation=>animation.finished)));
    const read=()=>image.evaluate(image=>{
        const bounds=image.getBoundingClientRect(), viewport=image.parentElement;
        return {x:bounds.x,y:bounds.y,width:bounds.width,height:bounds.height,scale:new DOMMatrix(image.style.transform).a,
            fit:Math.min(viewport.clientWidth/image.naturalWidth,viewport.clientHeight/image.naturalHeight)};
    });
    const state=await page.evaluate(()=>({raw:widget.node.widgets[1].value,size:[...widget.node.size],source:widget.state.selected_image_id}));
    const bounds=await viewport.boundingBox(), anchor={x:bounds.x+bounds.width*.65,y:bounds.y+bounds.height*.4}, before=await read();
    await page.mouse.move(anchor.x,anchor.y);await page.mouse.wheel(0,-240);
    await page.waitForFunction(scale=>new DOMMatrix(document.querySelector('.vnccs-common-modal-overlay:not([inert]) [data-large-preview]').style.transform).a>scale,before.scale);
    const enlarged=await read();
    (report.previewChecks??=[]).push({name,anchor,before,enlarged});
    assert.ok(enlarged.width>before.width,'Mouse wheel zooms the image');
    assert.ok(Math.abs((anchor.x-before.x)/before.width-(anchor.x-enlarged.x)/enlarged.width)<.002,'Zoom retains the image point under the horizontal cursor position');
    assert.ok(Math.abs((anchor.y-before.y)/before.height-(anchor.y-enlarged.y)/enlarged.height)<.002,'Zoom retains the image point under the vertical cursor position');
    for(const button of ['left']){
        const start=await read();await page.mouse.move(anchor.x,anchor.y);await page.mouse.down({button});
        await page.mouse.move(anchor.x+60,anchor.y+35,{steps:5});await page.mouse.up({button});
        const end=await read();
        assert.ok(Math.abs(end.x-start.x-60)<1 && Math.abs(end.y-start.y-35)<1,`${button} drag pans in rendered pixels despite node zoom`);
        assert.equal(await viewport.evaluate(element=>element.classList.contains('is-dragging')),false);
    }
    const beforeMiddle=await read();
    await page.mouse.move(anchor.x,anchor.y);await page.mouse.down({button:'middle'});
    await page.mouse.move(anchor.x+35,anchor.y+20);await page.mouse.up({button:'middle'});
    assert.deepEqual(await read(),beforeMiddle,'MMB is reserved for graph PAN and must not drag the image');
    await capture(name);
    await page.mouse.move(anchor.x,anchor.y);await page.mouse.down();
    await viewport.dispatchEvent('pointercancel',{pointerId:1});const cancelled=await read();
    await page.mouse.move(anchor.x+30,anchor.y+20);await page.mouse.up();
    assert.deepEqual(await read(),cancelled,'Cancelled drags cannot continue moving the image');
    for(const [delta,ratio] of [[-100,8],[100,1]]){
        await viewport.evaluate((element,delta)=>{
            const bounds=element.getBoundingClientRect();
            for(let index=0;index<80;index++)element.dispatchEvent(new WheelEvent('wheel',{deltaY:delta,clientX:bounds.x+bounds.width/2,clientY:bounds.y+bounds.height/2,cancelable:true,bubbles:true}));
        },delta);
        const zoom=await read();assert.ok(Math.abs(zoom.scale/zoom.fit-ratio)<.0001,'Zoom stays between the fit scale and eight times the fit scale');
    }
    assert.deepEqual(await page.evaluate(()=>({raw:widget.node.widgets[1].value,size:[...widget.node.size],source:widget.state.selected_image_id})),state,'Zoom and pan do not change saved settings, the source or node dimensions');
    return {wheel:true,leftPan:true,middleGraphPan:true,cursorAnchor:true,cancel:true,limits:[1,8]};
};
const geometry = async () => page.evaluate(() => {
    const rect = element => { const r = element.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; };
    const root = widget.container;
    return { root: rect(root), panels: [...root.querySelectorAll('[data-panel="background"]>.bg-panel')].filter(panel=>panel.getClientRects().length).map(rect), preview: rect(root.querySelector('.bg-preview')),
        rows: [...root.querySelectorAll('.vnccs-common-creator-gen-param-grid,.vnccs-common-creator-lora-row,.bg-actions')].map(row => [...row.children].filter(element => !element.hidden).map(rect)),
        overflow: [...root.querySelectorAll('.bg-panel,.bg-bottom')].filter(element => element.getClientRects().length && element.scrollWidth > element.clientWidth + 1).map(element => element.className),
        labels: [...root.querySelectorAll('.bg-attributes .bg-field')].map(field => {
            const label = (field.querySelector('.vnccs-creator-label') || field.children[0]).getBoundingClientRect();
            const visible = field.querySelector('.vnccs-style-summary') || [...field.children].slice(1).find(element => !element.hidden && element.getBoundingClientRect().height > 0);
            if (!visible) throw new Error('Scene field has no visible control');
            const control = visible.getBoundingClientRect();
            return { text: field.querySelector('.vnccs-creator-label')?.textContent || field.children[0].textContent, gap: control.top - label.bottom };
        }), fonts: [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family) };
});
try {
    await page.goto(`http://127.0.0.1:${server.address().port}`); await page.waitForFunction(() => window.ready);
    let metadataRequested;
    const heldConfig=new Promise(resolve=>metadataRequested=resolve);
    await page.route('**/vnccs/background-creator/config',route=>metadataRequested(route));
    await page.evaluate(()=>{
        window.pendingWorkflowRestore=setup({width:1600,height:1000});
        window.restoreSelections=[];
        const sample=()=>{
            if(window.widget?.state)restoreSelections.push(widget.container.querySelector('[data-library]').value);
            if(!window.widget?.state || widget.restoring)requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
    });
    const pendingConfig=await heldConfig;
    await page.waitForFunction(id=>window.widget?.state?.background_id===id,document.background_id);
    assert.equal(await page.locator('[data-library]').first().inputValue(),document.background_id,'The selected background is restored before the config request completes');
    assert.equal(await page.locator('[data-library]').first().locator('option:checked').textContent(),document.name);
    await page.waitForFunction(()=>document.querySelector('[data-preview]').complete && document.querySelector('[data-preview]').naturalWidth);
    assert.equal(await page.locator('[data-empty]').isVisible(),false,'A saved preview never shows the empty-state placeholder');
    await capture('workflow-restoring');
    await pendingConfig.continue();await page.unroute('**/vnccs/background-creator/config');
    await page.evaluate(()=>pendingWorkflowRestore);
    assert.ok(await page.evaluate(()=>restoreSelections.length>0 && restoreSelections.every(id=>id===widget.state.background_id)),'No frame flashes an unselected background');
    report.workflowRestore={selectionBeforeMetadata:true,previewBeforeMetadata:true,noEmptySelectionFrames:true};
    await page.route('**/vnccs/background-creator/config',route=>route.abort('failed'));
    const interrupted=await page.evaluate(async()=>{
        try{await setup({width:1600,height:1000});}catch(error){return {error:error.message,raw:widget.node.widgets[1].value};}
    });
    assert.match(interrupted.error,/Failed to fetch/);
    const recoveryOK=page.getByRole('button',{name:'OK',exact:true});
    assert.equal(await recoveryOK.isEnabled(),true,'The network error can be dismissed while restoration is blocked');
    await capture('connection-error');await recoveryOK.click();
    await page.waitForFunction(()=>!document.querySelector('[data-background-error]:not([inert])'));
    assert.equal(await page.evaluate(()=>widget.node.widgets[1].value),interrupted.raw,'Acknowledging an error preserves the original workflow bytes');
    assert.equal(await page.locator('[data-action="background"]').isDisabled(),true);
    await page.unroute('**/vnccs/background-creator/config');
    await page.evaluate(()=>connectionEvent('reconnected'));
    await page.waitForFunction(()=>!widget.restoring);
    assert.equal(await page.locator('[data-action="background"]').isEnabled(),true);
    assert.equal(await page.evaluate(()=>widget.state.editor.scene_description),JSON.parse(interrupted.raw).editor.scene_description);
    await capture('connection-recovered');
    report.connectionRecovery={dismissedOffline:true,restoredAfterReconnect:true,workflowPreserved:true};
    report.expectedDisconnectWarnings=warnings.filter(message=>message.includes('net::ERR_FAILED'));
    for(let index=warnings.length-1;index>=0;index--)if(warnings[index].includes('net::ERR_FAILED'))warnings.splice(index,1);
    for (const [width, height, name] of [[1600, 1000, "background"], [1280, 800, "background-default"], [900, 1000, "background-narrow"], [680, 1000, "background-compact"]]) {
        await page.evaluate(options => setup(options), { width, height });
        await page.waitForFunction(() => [...document.querySelectorAll('[data-preview]')].every(image => image.complete && image.naturalWidth));
        await capture(name); const measured = await geometry(); report.scenarios.push({ name, width, requestedHeight:height, height:await page.evaluate(()=>widget.node.size[1]), ...measured });
        if(width>=1280){
            const fit=await page.evaluate(()=>[...widget.container.querySelectorAll('.bg-panel')].map(panel=>{
                const p=panel.getBoundingClientRect();return {name:panel.className,client:panel.clientHeight,scroll:panel.scrollHeight,top:panel.scrollTop,
                    outside:[...panel.children].filter(e=>!e.hidden&&e.getClientRects().length).filter(e=>{const r=e.getBoundingClientRect();return r.top<p.top-1||r.bottom>p.bottom+1||r.left<p.left-1||r.right>p.right+1;}).map(e=>e.className)};
            }));
            assert.ok(fit.every(p=>p.scroll<=p.client+1&&p.top===0&&!p.outside.length),JSON.stringify(fit));
            report.scenarios.at(-1).fit=fit;
        }
        if(width===1600){
            const range=page.locator('[data-setting="generation.resolution_mp"]'), original=await page.evaluate(()=>widget.state.editor.generation.resolution_mp);
            assert.deepEqual(await range.evaluate(element=>({min:element.min,max:element.max,step:element.step})),{min:'1',max:'4',step:'0.1'});
            const resolutionStyle=await range.evaluate(control=>{
                const field=control.closest('.vnccs-common-creator-resolution-field'), reference=field.cloneNode(true);
                for(const element of [reference,...reference.querySelectorAll('*')])element.className=element.className.replaceAll('vnccs-common-creator-','vnccs-creator-');
                Object.assign(reference.style,{position:'absolute',left:'-100000px',width:field.clientWidth+'px'});widget.container.append(reference);
                const values=field=>{
                    const read=(selector,properties)=>{const style=getComputedStyle(selector?field.querySelector(selector):field);return Object.fromEntries(properties.map(key=>[key,style[key]]));};
                    return {field:read('', ['display','marginTop','marginBottom']),range:read('input',['accentColor','cursor','height','padding','borderWidth','borderRadius','backgroundColor']),
                        head:read('[class$="-resolution-head"]',['display','alignItems','justifyContent','gap']),value:read('[data-resolution]',['color','fontFamily','fontSize','fontWeight'])};
                };
                const result={actual:values(field),reference:values(reference)};reference.remove();return result;
            });
            assert.deepEqual(resolutionStyle.actual,resolutionStyle.reference,'The resolution control uses Character Creator styling');
            const values=[];
            for(const [key,value] of [['Home',1],['ArrowLeft',1],['ArrowRight',1.1],['ArrowRight',1.2],['ArrowRight',1.3],['End',4],['ArrowRight',4]]){
                await range.focus();await range.press(key);
                assert.equal(await range.inputValue(),String(value));
                assert.equal(await page.locator('[data-resolution]').textContent(),`${value.toFixed(1)} MP`);
                assert.equal(await page.evaluate(()=>JSON.parse(widget.node.widgets[1].value).editor.generation.resolution_mp),value);
                values.push(value);
                if(value===1.3)await range.locator('..').screenshot({path:join(output,'resolution-step-1-3.png')});
            }
            await range.evaluate((element,value)=>{element.value=String(value);element.dispatchEvent(new Event('input',{bubbles:true}));element.blur();},original);
            await page.evaluate(()=>widget.persist());
            report.resolutionControl={min:1,max:4,step:.1,keyboardValues:values,styleMatchesCharacterCreator:true};
            const description=page.locator('[data-setting="scene_description"]');
            const surface=()=>description.evaluate(element=>{
                const style=getComputedStyle(element);
                return {background:style.backgroundColor,shadow:style.boxShadow,border:style.borderColor};
            });
            await description.evaluate(element=>element.blur());const idle=await surface();
            await description.click();
            await description.evaluate(element=>Promise.allSettled(element.getAnimations().map(animation=>animation.finished)));
            const focused=await surface();report.sceneDescriptionFocus={idle,focused};
            await page.locator('.bg-attributes').screenshot({path:join(output,'scene-description-focused.png'),animations:'disabled'});
            assert.equal(focused.background,idle.background,'Focusing the scene description must not tint its writing surface');
            assert.equal(focused.shadow,idle.shadow,'Focusing the scene description must not add a glow');
            assert.notEqual(focused.border,idle.border,'The field retains a visible focus border');
            await description.evaluate(element=>element.blur());
            await description.evaluate(element=>Promise.allSettled(element.getAnimations().map(animation=>animation.finished)));
        }
        assert.equal(await page.locator('header,nav,[data-step],[data-viewer],[data-panel="model"]').count(),0,'The header and model stage are removed');
        assert.equal(measured.panels.length,3);
        assert.ok(measured.panels[0].y-measured.root.y<12,'No space is reserved for the removed header');
        for(const key of ['prompt','aesthetics'])assert.equal(await page.locator('[data-setting="'+key+'"]').count(),0);
        assert.equal(await page.locator('.bg-attributes [data-setting="negative"]').count(),1,'The current Negative Prompt stays in the scene column');
        assert.equal(await page.locator('.bg-bottom').count(),0);
        assert.equal(await page.locator('.bg-meta,[data-dimensions]').count(),0,'The background preview has no image dimensions label');
        const stripRows=await page.locator('.bg-variation-item').evaluateAll(cards=>cards.map(card=>card.getBoundingClientRect().top));
        assert.ok(stripRows.every(top=>Math.abs(top-stripRows[0])<1),'Variant cards always use one row');
        const variantPlaceholders=await page.locator('[data-variation-image]').evaluateAll(images=>images.map(image=>({src:image.src,hidden:image.hidden,placeholder:image.classList.contains('is-placeholder')})));
        assert.ok(variantPlaceholders.every(image=>!image.hidden&&image.placeholder&&/VNCCS_Background_Mannequin_Interior\.png$/.test(image.src)),'Empty variants use the matching synthetic interior preview');
        assert.equal(await page.locator('[data-scroll-variation]').count(),2);
        const variantContrast=await page.evaluate(()=>{
            const card=document.querySelector('[data-view-variation]'), label=card.closest('.bg-variation-item').querySelector(':scope>span'), style=getComputedStyle(card);
            const channels=color=>color.startsWith('color(srgb')?color.match(/[\d.]+/g).map(Number).map(value=>value*255):color.match(/[\d.]+/g).map(Number);
            const luminance=color=>channels(color).slice(0,3).map(value=>value/255).map(value=>value<=.04045?value/12.92:((value+.055)/1.055)**2.4).reduce((sum,value,index)=>sum+value*[.2126,.7152,.0722][index],0);
            const contrast=(first,second)=>{const values=[luminance(first),luminance(second)].sort((a,b)=>b-a);return (values[0]+.05)/(values[1]+.05);};
            return {background:style.backgroundColor,borderWidth:parseFloat(style.borderTopWidth)*card.getBoundingClientRect().width/card.offsetWidth,borderContrast:contrast(style.borderColor,style.backgroundColor),
                labelContrast:contrast(getComputedStyle(label).color,style.backgroundColor),labelSize:parseFloat(getComputedStyle(label).fontSize)};
        });
        report.scenarios.at(-1).variantContrast=variantContrast;
        assert.notEqual(variantContrast.background,'rgba(0, 0, 0, 0)','Unselected cards have a visible filled surface');
        assert.ok(variantContrast.borderWidth>=.99 && variantContrast.borderContrast>=3,'Unselected card outlines retain a full rendered pixel and 3:1 contrast without hovering');
        assert.ok(variantContrast.labelSize>=12 && variantContrast.labelContrast>=4.5,'Variant names remain readable');
        if(width===1600){
            await page.locator('#host').evaluate(host=>host.style.zoom='.5');
            await capture('variant-cards-zoom-50');
            await page.locator('#host').evaluate(host=>host.style.zoom='');
        }
        if(width>=1280){
            const reference=await page.evaluate(()=>{
                const host=document.createElement('div');host.className='vnccs-creator-container';
                Object.assign(host.style,{position:'fixed',left:'-100000px',width:(1280/.67)+'px',height:'800px'});
                const row=document.createElement('div');row.className='vnccs-creator-top-row';
                for(let i=0;i<3;i++)row.appendChild(document.createElement('div'));host.append(row);document.body.append(host);
                const generation=row.lastElementChild.getBoundingClientRect().width;host.remove();
                const old=document.createElement('div');Object.assign(old.style,{position:'fixed',left:'-100000px',display:'grid',zoom:'.67',gap:'12px',width:widget.container.querySelector('.bg-background').clientWidth+'px',gridTemplateColumns:'minmax(360px,1fr) minmax(240px,calc((100% - 24px)*.21875)) '+(generation/.67)+'px'});
                for(let i=0;i<3;i++)old.appendChild(document.createElement('div'));document.body.append(old);
                const columns=[...old.children].map(child=>child.getBoundingClientRect().width);old.remove();return {generation,columns};
            });
            assert.ok(Math.abs(measured.panels[2].width-reference.columns[2]*.8)<1,'The generation panel is exactly 20 percent narrower');
            assert.ok(Math.abs(measured.panels[1].width-reference.columns[1]-reference.columns[2]*.2)<1,'All released width goes to the middle panel');
            assert.ok(Math.abs(measured.panels[0].width-reference.columns[0])<1,'The left panel retains its previous width');
            report.scenarios.at(-1).reference=reference;
        }
        assert.equal(await page.locator("[data-connected]").count(), 0);
        assert.equal(await page.locator("[data-saved]").count(), 0);
        assert.equal(await page.locator('[data-variants],[data-variant-count],[data-variant]').count(),0,'Previews have no history navigation');
        assert.deepEqual(measured.overflow, [], "No panel may overflow horizontally");
        assert.ok(Math.abs(measured.root.width - width) < 1, "The real rendered widget follows node width");
        assert.ok(Math.abs(measured.preview.width / measured.preview.height - 16 / 9) < .005,'The preview frame has the standard 16:9 ratio');
        const fixedFrameNodeSize=await page.evaluate(()=>[...widget.node.size]);
        const fixedFrame=await page.evaluate(async()=>{
            const frame=widget.container.querySelector('.bg-preview'), image=frame.querySelector('[data-preview]'), original=image.src;
            const dimensions=[];
            for(const [width,height] of [[1600,900],[900,1600],[1000,1000],[2100,900]]){
                image.src='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="'+width+'" height="'+height+'"><rect width="100%" height="100%" fill="#485064"/></svg>');
                await image.decode();await new Promise(resolve=>requestAnimationFrame(resolve));
                const bounds=frame.getBoundingClientRect();dimensions.push({source:[image.naturalWidth,image.naturalHeight],width:bounds.width,height:bounds.height,nodeSize:[...widget.node.size]});
            }
            image.src=original;await image.decode();return dimensions;
        });
        for(const frame of fixedFrame){
            assert.ok(Math.abs(frame.width-measured.preview.width)<1 && Math.abs(frame.height-measured.preview.height)<1,'Loaded image dimensions must not resize the frame');
            assert.deepEqual(frame.nodeSize,fixedFrameNodeSize,'Image format must not resize the node');
        }
        report.scenarios.at(-1).fixedFrame=fixedFrame;
        for (const row of measured.rows) for (let index = 0; index + 1 < row.length; index += 2) {
            if (row.length === 3) break;
            assert.ok(Math.abs(row[index].y - row[index + 1].y) <= 1, "Peer control top edges align");
            assert.ok(Math.abs(row[index].height - row[index + 1].height) <= 1, "Peer controls have identical height");
        }
        for (const label of measured.labels) assert.ok(label.gap >= 2, `Scene field label overlaps its control: ${label.text}`);
        assert.ok(measured.fonts.includes('Sora'), 'The actual Creator font must be loaded');
        if (width >= 1280) { assert.ok(measured.preview.width > 400); assert.ok(measured.preview.height > 240); }
        assert.equal(await page.locator('[data-variation-editor]').count(),0,'The separate variant editor is removed');
        assert.equal(await page.locator('.bg-variation-settings').count(),3,'Every card always contains its settings');
        const cardLayout=await variantCardLayout();
        assert.ok(cardLayout.every(card=>card.belowPreview&&!card.overflow),'Every settings block stays below and inside its own card');
        assert.ok(cardLayout.every(card=>card.rows.length===3&&card.rows.every((field,index,fields)=>field.separated&&(!index||field.top>=fields[index-1].bottom-1))),'Each card keeps Time, Weather and Additional Details in separate non-overlapping columns');
        for(const key of ['width'])assert.ok(Math.max(...cardLayout.map(card=>card[key]))-Math.min(...cardLayout.map(card=>card[key]))<=1,'Variant cards keep equal widths');
        for(const group of ['preview','settings'])for(const key of ['top','bottom'])assert.ok(Math.max(...cardLayout.map(card=>card[group][key]))-Math.min(...cardLayout.map(card=>card[group][key]))<=1,`Variant ${group} ${key} edges stay aligned`);
        const navAlignment=await page.locator('.bg-variation-strip').evaluate(strip=>{
            const preview=strip.querySelector('.bg-variation-preview').getBoundingClientRect(),target=(preview.top+preview.bottom)/2;
            return [...strip.querySelectorAll('[data-scroll-variation]')].map(button=>{const bounds=button.getBoundingClientRect();return Math.abs((bounds.top+bounds.bottom)/2-target);});
        });
        assert.ok(navAlignment.every(offset=>offset<=1),'Variant arrows stay vertically centered on the preview row');
        await capture('variant-inline-'+width);
        if(width===1280)await page.locator('.bg-variation-strip').screenshot({path:join(output,'variant-cards-with-fields.png')});
        report.scenarios.at(-1).variantCards={nodeSize:await page.evaluate(()=>[...widget.node.size]),layout:cardLayout};
        if(width>=1280)assert.equal(await page.locator('.bg-panel').evaluateAll(panels=>panels.some(p=>p.scrollHeight>p.clientHeight+1||p.scrollTop!==0)),false,'Always-visible variant fields do not create panel scrolling');
    }
    await page.evaluate(() => setup({ width: 1600, height: 1000 }));
    const mainBeforeSet=await page.locator('[data-preview]').getAttribute('src');
    await page.evaluate(async()=>{
        const result=await fetch('/__fixture__/complete-background-set',{method:'POST',body:JSON.stringify(widget.state.editor.variants)});
        backgroundSetEvent(await result.json());
    });
    await page.waitForFunction(()=>[...document.querySelectorAll('[data-variation-image]')].length===3 && [...document.querySelectorAll('[data-variation-image]')].every(image=>!image.hidden && image.complete && image.naturalWidth));
    assert.equal(await page.locator('[data-preview]').getAttribute('src'),mainBeforeSet,'A completed set does not replace the working source preview');
    await capture('variant-results');
    const previewURLs=await page.locator('[data-variation-image]').evaluateAll(images=>images.map(image=>image.getAttribute('src')));
    assert.equal(new Set(previewURLs).size,3,'Each card receives its own generated image');
    await page.reload();await page.waitForFunction(()=>window.ready);await page.evaluate(()=>setup({width:1600,height:1000}));
    await page.waitForFunction(()=>[...document.querySelectorAll('[data-variation-image]')].every(image=>!image.hidden && image.complete && image.naturalWidth));
    assert.deepEqual(await page.locator('[data-variation-image]').evaluateAll(images=>images.map(image=>image.getAttribute('src'))),previewURLs,'Saved card previews return after reopening the workflow');
    report.variantResults={completedEvent:true,reload:true,mainPreviewPreserved:true,fixtureImages:true};
    const cards=await page.locator('[data-view-variation]').evaluateAll(cards=>cards.map(card=>card.dataset.viewVariation));
    for(const [index,id] of cards.entries()){
        await page.locator(`[data-view-variation="${id}"]`).click();
        assert.equal(await page.getByRole('dialog').locator('[data-large-preview]').getAttribute('src'),previewURLs[index],'Each generated card opens its own image');
        await page.evaluate(()=>widget.render());
        assert.equal(await page.getByRole('dialog').locator('[data-large-preview]').getAttribute('src'),previewURLs[index],'A render must not replace a modal variant with the base background');
        if(!index)report.variantPreviewNavigation=await previewNavigation('variant-preview-zoomed');
        await page.getByRole('dialog').locator('[data-close-preview]').click();
        assert.equal(await page.locator(`[data-variation="${id}"] .bg-variation-settings`).isVisible(),true,'Card settings remain visible after viewing a generated result');
    }
    await page.locator('[data-model-toggle]').click(); await capture('model-picker');
    await page.locator('[data-model-choice="user/Scenery.safetensors"]').click();
    assert.equal(await page.evaluate(() => JSON.parse(widget.node.widgets[1].value).editor.generation.diffusion_model_name), "user/Scenery.safetensors");
    await page.locator('[data-model-toggle]').click();
    await page.locator('[data-model-choice="qwen_image_2.1_turbo_int8_convrot.safetensors"]').click();
    assert.equal(await page.locator('[data-turbo-lora]').isVisible(), false, 'Built-in Turbo must not show the separate Turbo LoRA');
    assert.equal(await page.locator('[data-turbo-lora]').locator('..').isVisible(), true, 'The ordinary LoRA Stack remains available');
    await capture('turbo-model-without-turbo-lora');
    await page.locator('[data-model-toggle]').click();
    await page.locator('[data-model-choice="qwen_image_2.1_int8_convrot.safetensors"]').click();
    await page.evaluate(()=>widget.persist());
    await page.locator('[data-enlarge]').click(); await capture('background-enlarged');
    report.previewNavigation=await previewNavigation('background-preview-zoomed');
    await page.locator('#host').evaluate(host=>host.style.zoom='.75');
    report.scaledPreviewNavigation=await previewNavigation('background-preview-node-zoom-75');
    await page.locator('#host').evaluate(host=>host.style.zoom='');
    assert.equal(await page.getByRole('dialog').locator('[data-variants],[data-variant-count],[data-variant]').count(),0);
    await page.locator('[data-close-preview]').click();
    await page.evaluate(()=>{widget.styleInfo.custom_style='Anime, digital art, cel shaded';widget.stylePicker.setValue('custom',true);});
    await page.locator('[data-setting="scene_description"]').fill('A large, circular stone table sits in the center of a spacious, futuristic room with a high, arched ceiling featuring blue sky panels. The floor is made of water or a reflective surface, creating a mirror-like effect. On either side of the table are rectangular planters filled with vibrant purple flowers. In the background, there are several statues of figures in various poses, standing on pedestals. The room is illuminated by soft, ambient lighting from the ceiling and recessed lights, giving it a serene and mysterious atmosphere.');
    await page.locator('[data-setting="scene_description"]').dispatchEvent('input');
    assert.equal(await page.locator('[data-trait="time"]').count(), 0);
    const formatted = await page.evaluate(()=>composedPrompt());
    assert.equal(formatted, 'Style: Anime, digital art, cel shaded\nTime: Day\nInterior/Exterior: Interior\nScene description: A large, circular stone table sits in the center of a spacious, futuristic room with a high, arched ceiling featuring blue sky panels. The floor is made of water or a reflective surface, creating a mirror-like effect. On either side of the table are rectangular planters filled with vibrant purple flowers. In the background, there are several statues of figures in various poses, standing on pedestals. The room is illuminated by soft, ambient lighting from the ceiling and recessed lights, giving it a serene and mysterious atmosphere.');
    await capture('scene-fields');
    assert.deepEqual(await page.locator('[data-variation-cards] .bg-variation-item>span').allTextContents(), ['Morning','Evening','Night']);
    assert.equal(await page.locator('[data-variation-editor]').count(),0);
    assert.equal(await page.locator('[data-variation="morning"] [data-variation-field="time"]').inputValue(),'Morning');
    assert.equal(await page.locator('[data-variation="morning"] [data-variation-field="weather"]').inputValue(),'');
    assert.equal(await page.locator('.bg-variation-settings select').count(),0);
    await page.locator('[data-variation="morning"]').getByRole('button',{name:'Choose weather presets',exact:true}).click();
    await page.getByRole('dialog').getByRole('button',{name:'Light rain',exact:true}).click();
    await capture('variant-presets');
    await page.getByRole('dialog').getByRole('button',{name:'Apply',exact:true}).click();
    await setVariantField('additional_details','Keep the curtains open.');
    await capture('variant-parameters');
    assert.ok((await variantCardLayout()).every(card=>!card.overflow),'Editing fields keeps every card within its width');
    const variantType=await page.evaluate(()=>{
        const panel=widget.container.querySelector('.bg-variation-settings'),row=panel.querySelector('.vnccs-common-creator-trait-row');
        const rounded=element=>Math.round(parseFloat(element));
        return {panelHeight:rounded(getComputedStyle(panel).height),rowHeight:rounded(getComputedStyle(row).height),labelSize:rounded(getComputedStyle(row.querySelector('.vnccs-common-creator-trait-label')).fontSize),addSize:rounded(getComputedStyle(row.querySelector('.vnccs-common-creator-trait-add')).width)};
    });
    assert.deepEqual(variantType,{panelHeight:216,rowHeight:54,labelSize:11,addSize:34},'Variant typography and controls use the compact aligned card scale');
    await page.locator('[data-add-variation]').click();
    const addedId=await page.evaluate(()=>widget.state.editor.variants.at(-1).id);
    await page.locator(`[data-variation="${addedId}"] [data-variation-name]`).fill('Rainy dawn');
    await setVariantField('time','Dawn',addedId);await setVariantField('weather','Fog',addedId);
    assert.equal(await page.locator('[data-variation]').count(),4);
    const strip=page.locator('[data-variation-cards]');
    await strip.evaluate(list=>list.scrollTo({left:0,behavior:'instant'}));
    await page.waitForFunction(()=>!document.querySelector('[data-scroll-variation="1"]').disabled);
    await page.getByRole('button',{name:'Scroll variants right',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('[data-variation-cards]').scrollLeft>20);
    await page.getByRole('button',{name:'Scroll variants left',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('[data-variation-cards]').scrollLeft<5);
    assert.equal(await page.getByRole('dialog').count(),0);
    await capture('variant-cards');
    const deleteStyle = await page.evaluate(() => {
        const reference = document.createElement('div'); reference.className='vnccs-creator-container';
        const button = document.createElement('button'); button.className='vnccs-style-delete'; reference.append(button);
        widget.container.append(reference);
        const properties=['width','height','padding','borderRadius','backgroundColor','color','fontSize','lineHeight','top','right'];
        const values = element => Object.fromEntries(properties.map(key=>[key,getComputedStyle(element)[key]]));
        const result={actual:values(widget.container.querySelector('[data-delete-variation]')),reference:values(button)};
        reference.remove();return result;
    });
    assert.deepEqual(deleteStyle.actual,deleteStyle.reference,'Variant delete controls match Character Creator style cards');
    await page.evaluate(() => widget.persist());
    assert.equal(await page.locator('text=Unsaved changes').count(), 0);
    assert.equal(await page.evaluate(() => Object.keys(localStorage).filter(key => key.startsWith('VNCCS.BackgroundCreator:')).length), 0);
    await page.locator('[data-model-type="moss"]').click();
    assert.equal(await page.locator('[data-model-type="moss"]').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('[data-setting="generation.cfg"]').inputValue(), '2.5');
    assert.equal(await page.locator('[data-setting="generation.sampler"]').inputValue(), 'er_sde');
    assert.equal(await page.locator('[data-qwen-only]').evaluateAll(sections => sections.every(section => section.hidden)), true);
    assert.match(await page.locator('[data-model-toggle]').textContent(), /MOSS-Image/);
    await capture('moss');
    await page.locator('[data-setting="generation.steps"]').fill('19');
    await page.evaluate(() => widget.persist());
    const mossSaved = await page.evaluate(async () => (await (await fetch('/vnccs/background-creator/backgrounds/' + widget.state.background_id)).json()).editor);
    assert.equal(mossSaved.generation.model_type, 'moss'); assert.equal(mossSaved.generation.steps, 19);
    assert.equal(mossSaved.generation_profiles.qi2.diffusion_model_name, 'qwen_image_2.1_int8_convrot.safetensors');
    await page.locator('[data-model-type="qi2"]').click();
    assert.equal(await page.locator('[data-setting="generation.cfg"]').inputValue(), '1');
    assert.equal(await page.locator('[data-setting="generation.sampler"]').inputValue(), 'euler');
    assert.equal(await page.locator('[data-qwen-only]').evaluateAll(sections => sections.every(section => !section.hidden)), true);
    await page.evaluate(async () => {
        await widget.persist();
        const output = { background_preview: [{ background_id: widget.state.background_id, image_id: widget.state.selected_image_id,
            request_image_id: widget.state.selected_image_id, source: 'cache', editor: structuredClone(widget.state.editor) }] };
        widget.api.dispatchEvent(new CustomEvent('executing', { detail: '7' }));
        if (!widget.busy) throw new Error('RUN did not lock the background controls');
        widget.node.onExecuted(output); await widget.workflowCompletion;
        if (widget.busy) throw new Error('RUN left controls locked');
    });
    assert.equal(await page.locator('[data-panel="model"]').count(), 0);
    assert.match(await page.locator('[role="status"]').textContent(), /reused from cache/);
    await capture('run-background');
    const prompt = page.locator('[data-setting="scene_description"]'); await prompt.fill('A quiet bedroom. Keep my exact composition.'); await prompt.dispatchEvent('input');
    assert.equal(await page.locator('[data-trait="time"]').count(), 0);
    assert.match(await prompt.inputValue(), /Keep my exact composition/);
    const fits = await page.evaluate(() => [...widget.container.querySelectorAll('.bg-panel')].map(panel=>({name:panel.className,height:panel.clientHeight,content:panel.scrollHeight,top:panel.scrollTop})));
    assert.ok(fits.every(panel=>panel.content<=panel.height+1 && panel.top===0), 'Standard panels fit without content scrolling');


    await page.evaluate(() => {
        window.jobDone=false;window.jobError=null;
        window.previewJobRun=widget.runJob('background').then(()=>{window.jobDone=true},error=>{window.jobError=String(error);window.jobDone=true});
    });
    await page.waitForFunction(()=>Boolean(widget.jobId));
    assert.equal(await page.locator('[data-action="background"]').isDisabled(),true);
    assert.equal(await page.locator('[data-action="background"]').textContent(),'Generating...');
    assert.equal(await page.locator('footer,progress,[data-stop]').count(),0);
    assert.equal(await page.locator('[data-generation-layer]').isVisible(),true);
    assert.equal(await page.locator('.bg-preview').getAttribute('aria-busy'),'true');
    const styleMatch=await page.evaluate(()=>{
        const frame=widget.container.querySelector('.bg-preview');
        const reference=document.createElement('div');reference.className='vnccs-creator-generation-layer';
        reference.style.transform='translateX(-10000px)';
        reference.innerHTML='<span class="vnccs-creator-generation-scan"></span>';frame.appendChild(reference);
        const scan=frame.querySelector('.vnccs-common-creator-generation-scan'), expected=reference.firstChild;
        const actualCSS=getComputedStyle(scan), expectedCSS=getComputedStyle(expected);
        const properties=['animationDuration','animationTimingFunction','animationIterationCount','mixBlendMode','willChange','pointerEvents'];
        const pairs=Object.fromEntries(properties.map(key=>[key,[actualCSS[key],expectedCSS[key]]]));
        const actualBeam=getComputedStyle(scan,'::after'), expectedBeam=getComputedStyle(expected,'::after');
        for(const key of ['backgroundImage','boxShadow','height'])pairs['beam.'+key]=[actualBeam[key],expectedBeam[key]];
        reference.remove();
        for(const animation of scan.getAnimations()){animation.pause();animation.currentTime=750;}
        return pairs;
    });
    for(const [key,[actual,expected]] of Object.entries(styleMatch))assert.equal(actual,expected,'Creator style mismatch: '+key);
    report.generationStyles=styleMatch;
    await capture('working','allow');
    await page.evaluate(()=>fetch('/__fixture__/finish-preview',{method:'POST'}));
    await page.waitForFunction(()=>widget.previewGeneration?.layer.classList.contains('is-revealing'));
    const reveal=await page.evaluate(()=>{
        const animations=widget.previewGeneration.animations;
        animations.forEach(animation=>{animation.pause();animation.currentTime=675});
        return animations.map(animation=>({duration:animation.effect.getTiming().duration,easing:animation.effect.getTiming().easing,frames:animation.effect.getKeyframes().map(frame=>frame.clipPath||frame.transform)}));
    });
    assert.equal(reveal.length,3);assert.ok(reveal.every(animation=>animation.duration===1350&&animation.easing==='cubic-bezier(0.4, 0, 0.2, 1)'));
    report.generationReveal=reveal;
    await capture('generation-reveal','allow');
    await page.evaluate(()=>widget.previewGeneration.animations.forEach(animation=>animation.finish()));
    await page.waitForFunction(()=>window.jobDone);
    assert.equal(await page.evaluate(()=>window.jobError),null);
    assert.equal(await page.locator('[data-generation-layer]').isVisible(),false);
    assert.equal(await page.locator('.bg-preview').getAttribute('aria-busy'),null);
    assert.equal(await page.locator('[data-action="background"]').textContent(),'Generate background');
    await capture('generation-complete');
    const firstPreview=await page.evaluate(()=>({id:widget.state.selected_image_id,count:widget.document.images.length,url:widget.container.querySelector('[data-preview]').src}));
    await page.emulateMedia({reducedMotion:'reduce'});
    await page.evaluate(()=>{
        window.jobDone=false;window.jobError=null;
        window.previewJobRun=widget.runJob('background').then(()=>{window.jobDone=true},error=>{window.jobError=String(error);window.jobDone=true});
    });
    await page.waitForFunction(()=>Boolean(widget.jobId));
    await page.evaluate(()=>fetch('/__fixture__/finish-preview',{method:'POST'}));
    await page.waitForFunction(()=>window.jobDone);
    assert.equal(await page.evaluate(()=>window.jobError),null);
    const replacement=await page.evaluate(()=>({id:widget.state.selected_image_id,count:widget.document.images.length,url:widget.container.querySelector('[data-preview]').src}));
    assert.equal(replacement.id,firstPreview.id);assert.equal(replacement.count,firstPreview.count);
    assert.notEqual(replacement.url,firstPreview.url,'The current image refreshes even when its ID is reused');
    assert.equal(await page.locator('[data-variants],[data-variant-count],[data-variant]').count(),0);
    await page.locator('[data-enlarge]').click();
    assert.equal(await page.getByRole('dialog').locator('[data-large-preview]').getAttribute('src'),replacement.url.replace(/^https?:\/\/[^/]+/,''));
    assert.equal(await page.getByRole('dialog').locator('[data-variants],[data-variant-count],[data-variant]').count(),0);
    await page.locator('[data-close-preview]').click();
    await capture('single-preview-after-regeneration');
    await page.reload();await page.waitForFunction(()=>window.ready);await page.evaluate(()=>setup());
    assert.equal(await page.evaluate(()=>widget.state.selected_image_id),replacement.id);
    assert.equal(await page.locator('[data-variants],[data-variant-count],[data-variant]').count(),0);
    report.singlePreview={replaces:true,noHistoryNavigation:true,enlarged:true,reload:true};
    await page.emulateMedia({reducedMotion:null});
    await page.evaluate(() => setup({ width: 1280, height: 800, empty: true })); await capture('empty');
    const emptyFrame=await geometry();assert.ok(Math.abs(emptyFrame.preview.width/emptyFrame.preview.height-16/9)<.005,'The empty frame uses the same fixed ratio');
    assert.equal(await page.locator('[data-enlarge]').isVisible(), false); assert.equal(await page.locator('[data-action="model"]').count(), 0);
    await page.evaluate(()=>setup());
    const imagesBeforeDeletion=await page.evaluate(()=>widget.document.images.length);
    await page.getByRole('button',{name:'Delete Rainy dawn variant',exact:true}).click();
    await page.getByRole('button',{name:'Delete Morning variant',exact:true}).click();
    assert.equal(await page.getByRole('dialog').count(),0,'Delete must not open the edit dialog');
    await capture('variant-deletion');await page.evaluate(()=>widget.persist());
    await page.reload();await page.waitForFunction(()=>window.ready);await page.evaluate(()=>setup());
    assert.deepEqual(await page.locator('[data-variation-cards] .bg-variation-item>span').allTextContents(),['Evening','Night']);
    for(const name of ['Evening','Night'])await page.getByRole('button',{name:'Delete '+name+' variant',exact:true}).click();
    assert.equal(await page.locator('[data-variation]').count(),0);
    assert.equal(await page.locator('[data-add-variation]').evaluate(button=>document.activeElement===button),true,'Focus moves to Add variant after the final deletion');
    await page.evaluate(()=>widget.persist());await capture('variants-empty');
    await page.reload();await page.waitForFunction(()=>window.ready);await page.evaluate(()=>setup());
    assert.equal(await page.locator('[data-variation]').count(),0,'Deleted defaults stay deleted after JSON reload');
    assert.equal(await page.evaluate(()=>widget.document.images.length),imagesBeforeDeletion);
    await page.locator('[data-add-variation]').click();
    await page.locator('[data-variation-name]').fill('Rainy dawn');
    await page.evaluate(()=>widget.persist());assert.equal(await page.locator('[data-variation]').count(),1);
    report.variantDeletion={defaults:true,custom:true,lastCard:true,reload:true,sourceImagesPreserved:true};

    for(const width of [1180,1600]) {
        await page.evaluate(width=>setupGenerator({width,height:850}),width);
        await page.waitForTimeout(200);
        assert.equal(await page.locator('.vnccs-pipe-stage').count(),3);
        assert.equal(await page.locator('.vnccs-pipe-settings').getByText('BG Remove',{exact:true}).count(),0);
        await page.evaluate(()=>{
            widget.data.background_variants=[];widget.selectedPreview='background_generation';
            widget.stageState={background_generation:{status:'waiting'}};widget.renderPreview();
        });
        assert.equal(await page.locator('.vnccs-pipe-img-label').count(),1,'An empty variant list only reserves the original frame');
        assert.deepEqual(await page.locator('.vnccs-pipe-img-label').allTextContents(),['Original']);
        await page.evaluate(()=>{widget.data.background_variants=['Morning','Evening','Night'];});
        await page.evaluate(source=>{
            const labels=['Original','Morning','Evening','Night'];
            for(const stage of ['source_upscaler','background_generation','upscaler']) {
                generatorEvent(stage,{status:'running',current:0,total:stage==='source_upscaler'?1:4,labels:stage==='source_upscaler'?['Original']:labels,preview_aspect:16/9});
                generatorEvent(stage,{status:'done',current:stage==='source_upscaler'?1:4,total:stage==='source_upscaler'?1:4,images:stage==='source_upscaler'?[source]:Array(4).fill(source),labels:stage==='source_upscaler'?['Original']:labels,preview_aspect:16/9});
            }
        }, document.images[0].url);
        await page.waitForFunction(()=>widget.previewGrid.children.length===4 && [...widget.previewGrid.children].every(tile=>tile.dataset.src));
        await page.waitForTimeout(800);
        assert.deepEqual(await page.locator('.vnccs-pipe-img-label').allTextContents(),['Original','Morning','Evening','Night']);
        assert.equal(await page.locator('.vnccs-pipe-img-placeholder').count(),0);
        await capture('generator-'+width);
        assert.equal(await page.locator('.vnccs-pipe-root').evaluate(root=>root.scrollWidth>root.clientWidth+1),false);
        const before=await page.evaluate(()=>JSON.parse(widget.node.widgets[0].value));
        await page.locator('.vnccs-pipe-settings-open').click();await capture('generator-settings-'+width);
        assert.equal(await page.locator('.vnccs-pipe-settings-groups').getByText('BG Remove',{exact:true}).count(),0);
        await page.locator('.vnccs-pipe-modal-actions').getByRole('button',{name:'Cancel',exact:true}).click();
        assert.deepEqual(await page.evaluate(()=>JSON.parse(widget.node.widgets[0].value)),before);
    }
    await page.evaluate(()=>setup({width:1600,height:1000}));
    await page.evaluate(()=>{
        widget.state.editor.variants.push({id:'day',name:'Day',time:'Day',weather:'',additional_details:''});
        widget.state.editor.variants[0].weather='Light rain';widget.markEdited();widget.render();
    });
    await page.locator('[data-new]').click();
    const newDialog=page.getByRole('dialog',{name:'New Background',exact:true});
    assert.equal(await newDialog.locator('label').count(),0,'Creation matches Character Creator and does not add a redundant field label');
    const nameInput=newDialog.getByPlaceholder('Name...', {exact:true});
    assert.equal(await nameInput.inputValue(),'');
    await capture('new-background-dialog');
    await nameInput.fill('New defaults');
    await newDialog.getByRole('button',{name:'Create',exact:true}).click();
    await page.waitForFunction(()=>widget.document.name==='New defaults');
    assert.equal(await page.locator('.vnccs-style-name').textContent(),'None');
    assert.deepEqual(await page.evaluate(()=>({style:widget.state.editor.style,styleId:widget.state.editor.style_id})),{style:'',styleId:undefined});
    assert.deepEqual(await page.locator('[data-setting="scene_description"],[data-setting="front"],[data-setting="back"],[data-setting="left"],[data-setting="right"],[data-setting="negative"]').evaluateAll(fields=>fields.map(field=>field.value)),['','','','','',''],'New clears every authored scene field in the visible form');
    assert.deepEqual(await page.evaluate(()=>widget.state.editor.attributes),{scene_type:'Interior',time:'Day'});
    assert.deepEqual(await page.locator('.bg-variation-item>span').allTextContents(),['Morning','Evening','Night']);
    assert.equal(await page.evaluate(()=>widget.state.editor.variants.every(card=>card.weather==='')),true,'New does not copy the previous variant parameters');
    const neutralPlaceholders=await page.locator('[data-setting="front"],[data-setting="back"],[data-setting="left"],[data-setting="right"]').evaluateAll(fields=>Object.fromEntries(fields.map(field=>[field.dataset.setting,field.placeholder])));
    assert.deepEqual(neutralPlaceholders,{
        front:'Describe the nearest surfaces, objects, or framing elements.',
        back:'Describe the distant environment, horizon, architecture, or backdrop.',
        left:'Describe elements and spatial relationships on the left side.',
        right:'Describe elements and spatial relationships on the right side.',
    });
    assert.doesNotMatch(Object.values(neutralPlaceholders).join(' '),/garage|toolbox|tires|paint cans|workbench/i);
    await capture('new-background-defaults');
    await page.locator('.bg-attributes').screenshot({path:join(output,'neutral-background-field-placeholders.png'),animations:'disabled'});
    report.newBackground={legacyDayCopied:false,freshVariants:true,defaultStyle:'none'};
    await page.locator('[data-delete]').click();
    const deleteDialog=page.getByRole('dialog',{name:'Delete Background',exact:true});
    assert.equal(await deleteDialog.locator('input,textarea,select').count(),0,'Deletion confirmation has no editable fields');
    assert.match(await deleteDialog.textContent(),/PERMANENTLY DELETE/);
    assert.match(await deleteDialog.textContent(),/This action cannot be undone/);
    await capture('delete-background-confirmation');
    await deleteDialog.getByRole('button',{name:'CONFIRM DELETE',exact:true}).click();
    await page.waitForFunction(()=>widget.document?.name==='Evening room');
    assert.equal(await page.locator('[data-library]').inputValue(),backgroundId,'Deleting selects the next saved background like Character Creator');
    report.deleteBackground={editableConfirmation:false,permanent:true,nextSelected:true};
    assert.ok(mutations.length > 0, 'Edits still save into the selected background JSON');
    assert.deepEqual(retiredRequests, []); report.retiredModelRequests=retiredRequests;
    await page.evaluate(()=>setup({width:1280,height:800}));
    report.wizardFit=await page.evaluate(async()=>{
        const {animateWizardRows}=await import('/web/vnccs_common.js');
        const field=widget.container.querySelector('.bg-description'),target=field.querySelector('textarea');
        window.wizardFitRun=animateWizardRows([field],()=>{target.value='A detailed room';});
        const layer=field.querySelector('.vnccs-common-wizard-reveal'),a=target.getBoundingClientRect(),b=layer.getBoundingClientRect();
        return {dx:Math.abs(a.x-b.x),dy:Math.abs(a.y-b.y),dw:Math.abs(a.width-b.width),dh:Math.abs(a.height-b.height)};
    });
    assert.ok(Object.values(report.wizardFit).every(delta=>delta<=1),JSON.stringify(report.wizardFit));
    await capture('wizard-fit');await page.evaluate(()=>wizardFitRun);
    await page.locator('.vnccs-style-summary').click();
    await page.waitForFunction(()=>document.querySelector('.vnccs-style-status')?.textContent.includes('styles'));
    await page.locator('.vnccs-style-size-slider').evaluate(input=>{input.value='250';input.dispatchEvent(new Event('input',{bubbles:true}));});
    await settle();await capture('style-library');
    const libraryFit=await page.locator('.vnccs-style-content').evaluate(content=>({height:content.clientHeight,scroll:content.scrollHeight,width:content.clientWidth,scrollWidth:content.scrollWidth,overflow:getComputedStyle(content).overflowY}));
    assert.ok(libraryFit.scroll>libraryFit.height+1&&libraryFit.scrollWidth<=libraryFit.width+1&&libraryFit.overflow==='auto',JSON.stringify(libraryFit));
    assert.equal(await page.locator('.vnccs-style-gallery').getByRole('button',{name:/^(Previous|Next)$/}).count(),0,'The existing style gallery scrolls without invented page navigation');
    await page.locator('.vnccs-style-content').hover();await page.mouse.wheel(0,450);
    await page.waitForFunction(()=>document.querySelector('.vnccs-style-content').scrollTop>0);
    await page.waitForTimeout(250);
    const libraryScroll=await page.locator('.vnccs-style-content').evaluate(content=>content.scrollTop);
    await page.evaluate(()=>widget.render());await settle();
    assert.ok(Math.abs(await page.locator('.vnccs-style-content').evaluate(content=>content.scrollTop)-libraryScroll)<=1,'Refreshing background controls retains the gallery scroll position');
    await capture('style-library-scrolled');
    report.styleLibraryFit={...libraryFit,scrollAfterWheel:libraryScroll};
    const libraryControls=await page.locator('.vnccs-style-gallery .vnccs-creator-btn').evaluateAll(buttons=>buttons.filter(button=>!button.disabled&&!button.hidden).map(button=>({font:getComputedStyle(button).fontFamily,color:getComputedStyle(button).color})));
    assert.ok(libraryControls.length && libraryControls.every(control=>control.font.includes('Sora')&&control.color==='rgb(255, 255, 255)'),JSON.stringify(libraryControls));
    assert.ok(await page.locator('.vnccs-style-size-slider').evaluate(input=>input.offsetWidth<=180),'Library card-size slider retains its shared 180px width');
    const categoryStyle=await page.locator('.vnccs-style-gallery .vnccs-common-ui-select-trigger').evaluate(button=>({background:getComputedStyle(button).backgroundColor,color:getComputedStyle(button).color,font:getComputedStyle(button).fontFamily}));
    assert.ok(categoryStyle.background!=='rgb(239, 239, 239)'&&categoryStyle.background!=='rgb(255, 255, 255)'&&categoryStyle.color!=='rgb(0, 0, 0)'&&categoryStyle.font.includes('Sora'),JSON.stringify(categoryStyle));
    report.styleLibraryCategory=categoryStyle;
    await page.getByRole('button',{name:'New style',exact:true}).click();
    report.styleEditorFit=await page.locator('.vnccs-style-editor').evaluate(editor=>{
        const content=editor.parentElement,preview=editor.querySelector('.vnccs-style-placeholder');
        const fields=[...editor.querySelectorAll(':scope>.vnccs-creator-field')],actions=editor.querySelector(':scope>.vnccs-style-toolbar');
        return {width:editor.clientWidth,contentWidth:content.clientWidth,scrollWidth:editor.scrollWidth,height:editor.clientHeight,contentHeight:content.clientHeight,
            labels:[...editor.querySelectorAll('.vnccs-creator-label')].map(label=>label.textContent),
            previewBeforeFields:fields.every(field=>preview.offsetLeft+preview.offsetWidth<field.offsetLeft),
            fieldsAligned:fields.every(field=>field.offsetLeft===fields[0].offsetLeft&&field.offsetWidth===fields[0].offsetWidth),
            actionsAligned:actions.offsetLeft===fields[0].offsetLeft};
    });
    assert.ok(report.styleEditorFit.width>=Math.min(1000,report.styleEditorFit.contentWidth-6),JSON.stringify(report.styleEditorFit));
    assert.ok(report.styleEditorFit.scrollWidth<=report.styleEditorFit.width+1&&report.styleEditorFit.height<=report.styleEditorFit.contentHeight+1,JSON.stringify(report.styleEditorFit));
    assert.deepEqual(report.styleEditorFit.labels,['Name','Short description','Reference','Style prompt']);
    assert.equal(report.styleEditorFit.previewBeforeFields&&report.styleEditorFit.fieldsAligned&&report.styleEditorFit.actionsAligned,true,JSON.stringify(report.styleEditorFit));
    await capture('style-editor');
    await page.getByRole('button',{name:'Back to library',exact:true}).click();
    const beforeCovers=await page.evaluate(()=>widget.node.widgets.find(item=>item.name==='background_data').value);
    await page.evaluate(()=>{widget.styleInfo.custom_style='';});
    await page.getByRole('button',{name:'Generate all previews',exact:true}).click();
    await page.waitForFunction(expected=>document.querySelector('.vnccs-style-status')?.textContent===`Saved all ${expected} style previews.`,expectedStyleCount);
    assert.equal(coverRequests.length,expectedStyleCount);
    assert.equal(new Set(coverRequests.map(payload=>payload.style_id)).size,expectedStyleCount);
    assert.ok(coverRequests.every(payload=>payload.node_id==='7'&&payload.editor.generation&&payload.editor.scene_description!==undefined));
    assert.equal(await page.evaluate(()=>widget.node.widgets.find(item=>item.name==='background_data').value),beforeCovers,'Generating covers leaves the authored background unchanged');
    assert.equal(await page.locator('.vnccs-style-preview-image').count(),expectedStyleCount);
    await page.locator('.vnccs-style-content').evaluate(content=>{content.scrollTop=0;});
    await settle();await capture('background-style-covers-generated');
    report.backgroundStyleCovers={styles:expectedStyleCount,requests:coverRequests.length,source:'mock inference; real gallery and HTTP requests'};
    await page.getByRole('button',{name:'Close',exact:true}).click();
    const beforeModelCatalog=await page.evaluate(()=>[...widget.config.diffusion_models]);
    await page.evaluate(()=>{widget.config.diffusion_models.push(...Array.from({length:30},(_,i)=>`scenery/Model_${i}.safetensors`));widget.modelPickerOpen=true;widget.renderModelPicker();});
    const modelLibrary=await page.locator('.vnccs-common-creator-model-picker-menu').evaluate(menu=>({height:menu.clientHeight,scroll:menu.scrollHeight,choices:menu.querySelectorAll('[data-model-choice]').length,pages:menu.querySelectorAll('[data-model-page]').length}));
    assert.ok(modelLibrary.scroll>modelLibrary.height&&modelLibrary.choices>=30&&modelLibrary.pages===0,JSON.stringify(modelLibrary));
    report.modelLibrary=modelLibrary;await capture('model-library');
    await page.evaluate(models=>{widget.config.diffusion_models=models;widget.modelPickerOpen=false;widget.renderModelPicker();},beforeModelCatalog);
    report.backgroundMannequins=[];
    for(const sceneType of ['Interior','Exterior']){
        await page.evaluate(type=>{
            setup({width:1280,height:800,empty:true}).then(()=>{
                widget.state.editor.attributes.scene_type=type;widget.render();widget.busy=true;widget.startPreviewGeneration();widget.updateControls();window.mannequinReady=true;
            });
        },sceneType);
        await page.waitForFunction(()=>window.mannequinReady&&widget.previewGeneration?.base.complete);
        const mannequin=await page.evaluate(()=>({src:widget.previewGeneration.base.src,uses:widget.previewGeneration.layer.classList.contains('uses-mannequin'),empty:widget.container.querySelector('[data-empty]').hidden}));
        assert.match(mannequin.src,new RegExp(`VNCCS_Background_Mannequin_${sceneType}\\.png$`));
        assert.equal(mannequin.uses,true);assert.equal(mannequin.empty,true);
        report.backgroundMannequins.push({sceneType,...mannequin});await capture('background-mannequin-'+sceneType.toLowerCase());
        await page.evaluate(()=>{window.mannequinReady=false;widget.stopPreviewGeneration();});
    }
    report.standardStates=[];
    for(const state of ['empty','populated','missing-model','busy']){
        await page.evaluate(empty=>setup({width:1280,height:800,empty}),state==='empty');
        if(state==='missing-model')await page.evaluate(()=>{
            widget.config.installed=false;widget.config.diffusion_models=[];
            widget.config.assets=[{category:'diffusion_models',name:'qwen_image_2.1_int8_convrot.safetensors',installed:false},{category:'text_encoders',name:'qwen_encoder_missing.safetensors',installed:false},{category:'vae',name:'qwen_vae_missing.safetensors',installed:false}];
            widget.ccState.config={models:[],clip:[],vae:[]};widget.render();
        });
        if(state==='busy')await page.evaluate(()=>{widget.busy=true;widget.startPreviewGeneration();widget.updateControls();});
        const fit=await page.locator('.bg-panel').evaluateAll(panels=>panels.map(panel=>({name:panel.className,client:panel.clientHeight,scroll:panel.scrollHeight,top:panel.scrollTop})));
        assert.ok(fit.every(panel=>panel.scroll<=panel.client+1&&panel.top===0),state+': '+JSON.stringify(fit));
        report.standardStates.push({state,fit});await capture('standard-'+state);
    }
    await page.evaluate(()=>setup({width:1280,height:840}));
    const wizardJob='9'.repeat(32);
    const wizardRequests=[];
    await page.route('**/backgrounds/*/jobs',async route=>{
        const payload=route.request().postDataJSON();wizardRequests.push(payload);
        assert.equal(payload.action,'wizard');
        await route.fulfill({json:{job_id:wizardJob}});
    });
    await page.route(`**/jobs/${wizardJob}`,route=>route.fulfill({json:{status:'failed',error:'Describe the background for the wizard'}}));
    await page.evaluate(()=>{for(const key of ['scene_description','front','back','left','right'])widget.state.editor[key]='';widget.markEdited();widget.render();});
    await page.locator('[data-wizard]').click();
    const wizard=page.getByRole('dialog',{name:'Background Wizard',exact:true});
    await wizard.waitFor();await settle();
    assert.equal(wizardRequests.length,0,'Opening the wizard must not start inference');
    assert.equal(await wizard.locator('.bg-wizard-text').textContent(),'Describe the background in a broad way. The model will expand it into the scene fields while preserving the selected style and scene type.');
    assert.equal(await wizard.locator('textarea').getAttribute('placeholder'),'e.g. a quiet interior, open landscape, city street, or fantasy environment');
    await wizard.getByRole('button',{name:'FILL FIELDS',exact:true}).click();
    assert.equal(wizardRequests.length,0,'An empty wizard brief must not start inference');
    assert.equal(await wizard.locator('textarea').evaluate(input=>input===document.activeElement),true,'Empty submission keeps focus in the wizard input');
    report.wizardDialogComparison=await page.evaluate(async()=>{
        const {showModal}=await import('/web/vnccs_common.js');
        const current=widget.wizardDialog;
        const owner=document.createElement('div');owner.className='vnccs-creator-container';
        Object.assign(owner.style,{position:'fixed',left:'-100000px',top:'0',width:widget.container.clientWidth+'px',height:widget.container.clientHeight+'px'});
        document.body.append(owner);
        const reference=showModal(owner,'Character Wizzard',()=>{
            const wrap=document.createElement('div');wrap.className='vnccs-creator-character-wizard-modal';
            const text=document.createElement('div');text.className='vnccs-creator-character-wizard-modal-text';text.textContent='Reference';
            const input=document.createElement('textarea');input.className='vnccs-creator-textarea';wrap.append(text,input);return wrap;
        },[{text:'Cancel'},{text:'FILL FIELDS',class:'vnccs-creator-btn-primary'}]);
        const props=['fontFamily','fontSize','fontWeight','lineHeight','color','backgroundColor','backgroundImage','borderTopWidth','borderTopStyle','borderTopColor','borderRadius','padding','gap','overflow','width','minHeight','resize'];
        const read=(element,keys=props)=>Object.fromEntries(keys.map(key=>[key,getComputedStyle(element)[key]]));
        const currentTextarea=current.content.querySelector('textarea'),referenceTextarea=reference.content.querySelector('textarea');
        const textareaProps=props.filter(key=>!['backgroundColor','borderTopColor'].includes(key));
        currentTextarea.focus();const actualTextarea=read(currentTextarea,textareaProps);referenceTextarea.focus();const expectedTextarea=read(referenceTextarea,textareaProps);
        const pairs={modal:[current.modal,reference.modal],body:[current.content,reference.content],text:[current.content.querySelector('.bg-wizard-text'),reference.content.querySelector('.vnccs-creator-character-wizard-modal-text')],cancel:[current.buttons[0],reference.buttons[0]],confirm:[current.buttons[1],reference.buttons[1]],actions:[current.actions,reference.actions]};
        const result=Object.fromEntries(Object.entries(pairs).map(([key,[actual,expected]])=>[key,{actual:read(actual),expected:read(expected)}]));
        result.textarea={actual:actualTextarea,expected:expectedTextarea};
        const modal=current.modal.getBoundingClientRect(),host=widget.container.getBoundingClientRect();
        result.fit={left:modal.left>=host.left,right:modal.right<=host.right,top:modal.top>=host.top,bottom:modal.bottom<=host.bottom};
        reference.overlay.remove();owner.remove();return result;
    });
    for(const [part,comparison] of Object.entries(report.wizardDialogComparison)){
        if(part==='fit')assert.ok(Object.values(comparison).every(Boolean),JSON.stringify(comparison));
        else assert.deepEqual(comparison.actual,comparison.expected,part+' must match the actual Character Creator wizard styles');
    }
    await capture('background-wizard-input');
    await wizard.locator('textarea').fill('A quiet coastal town beneath overcast skies');
    await wizard.getByRole('button',{name:'FILL FIELDS',exact:true}).click();
    assert.equal(wizardRequests.length,1);
    assert.equal(wizardRequests[0].description,'A quiet coastal town beneath overcast skies');
    assert.equal(wizardRequests[0].editor.scene_description,'','The wizard brief must not overwrite authored fields before success');
    const wizardError=page.getByRole('dialog',{name:'Background Wizard Error',exact:true});
    await wizardError.waitFor();await settle();
    assert.equal(await page.locator('.vnccs-common-message-box').count(),0);
    assert.equal(await wizardError.locator('.bg-error-text').textContent(),'Describe the background for the wizard');
    assert.equal(await wizard.locator('textarea').inputValue(),'A quiet coastal town beneath overcast skies','A failed request keeps the wizard draft');
    report.errorDialogComparison=await page.evaluate(async()=>{
        const {showModal}=await import('/web/vnccs_common.js');
        const current=widget.errorDialog;
        const owner=document.createElement('div');owner.className='vnccs-creator-container';
        Object.assign(owner.style,{position:'fixed',left:'-100000px',top:'0',width:widget.container.clientWidth+'px',height:widget.container.clientHeight+'px'});
        document.body.append(owner);
        const reference=showModal(owner,'Character Wizzard Error',()=>{
            const wrap=document.createElement('div');wrap.className='vnccs-creator-character-wizard-modal';
            const text=document.createElement('div');text.className='vnccs-creator-character-wizard-modal-text';
            text.textContent='Describe the background for the wizard';wrap.append(text);return wrap;
        },[{text:'OK',class:'primary'}]);
        const props=['fontFamily','fontSize','fontWeight','lineHeight','color','backgroundColor','backgroundImage','borderTopWidth','borderTopStyle','borderTopColor','borderRadius','padding','gap','overflow','width'];
        const read=(element,keys=props)=>Object.fromEntries(keys.map(key=>[key,getComputedStyle(element)[key]]));
        const pairs={
            modal:[current.modal,reference.modal],
            title:[current.title,reference.title],
            body:[current.content,reference.content.firstElementChild],
            button:[current.buttons[0],reference.buttons[0]],
            actions:[current.actions,reference.actions],
        };
        const result=Object.fromEntries(Object.entries(pairs).map(([key,[actual,expected]])=>[key,{actual:read(actual),expected:read(expected)}]));
        const modal=current.modal.getBoundingClientRect(), host=widget.container.getBoundingClientRect();
        result.fit={left:modal.left>=host.left,right:modal.right<=host.right,top:modal.top>=host.top,bottom:modal.bottom<=host.bottom};
        reference.overlay.remove();owner.remove();return result;
    });
    for(const [part,comparison] of Object.entries(report.errorDialogComparison)){
        if(part==='fit')assert.ok(Object.values(comparison).every(Boolean),JSON.stringify(comparison));
        else assert.deepEqual(comparison.actual,comparison.expected,part+' must match the actual Character Creator error dialog styles');
    }
    await capture('background-wizard-error');
    await wizardError.getByRole('button',{name:'OK',exact:true}).focus();
    await page.keyboard.press('Tab');
    assert.equal(await wizardError.getByRole('button',{name:'OK',exact:true}).evaluate(button=>button===document.activeElement),true,'Keyboard focus stays in the error dialog');
    await page.keyboard.press('Escape');
    await page.waitForFunction(()=>!document.querySelector('[data-background-error]'));
    assert.equal(await wizard.locator('textarea').evaluate(input=>input===document.activeElement),true,'Dismissing the error restores focus to the retained wizard draft');
    await wizard.getByRole('button',{name:'Cancel',exact:true}).click();
    await page.waitForFunction(()=>!document.querySelector('[data-background-wizard]'));
    assert.equal(await page.locator('[data-wizard]').evaluate(button=>button===document.activeElement),true,'Closing the wizard restores focus to its launcher');
    await page.unroute('**/backgrounds/*/jobs');await page.unroute(`**/jobs/${wizardJob}`);
    await page.evaluate(()=>{widget.message('Retrying...');widget.message('Describe the background for the wizard',true,'Background Wizard Error');widget.restoring=true;widget.updateControls();});
    assert.equal(await page.getByRole('dialog',{name:'Background Wizard Error'}).getByRole('button',{name:'OK',exact:true}).isEnabled(),true);
    await page.getByRole('dialog',{name:'Background Wizard Error'}).getByRole('button',{name:'OK',exact:true}).click();
    await page.waitForFunction(()=>!document.querySelector('[data-background-error]'));
    report.wizardInteraction={opensBeforeInference:true,emptyStaysOpen:true,separateBrief:true,draftRetainedOnFailure:true};
    report.errorDialogInteraction={focusTrap:true,escape:true,nestedOpenerFocus:true,launcherFocus:true,offlineDismissal:true};
    assert.deepEqual(errors, []); report.errors = errors; report.warnings = warnings; report.fixtureOnlyMutations = mutations.length;
    await writeFile(join(output, 'report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ output, scenarios: report.scenarios.length, retiredModelRequests: retiredRequests.length, errors, warnings }));
} catch (error) {
    await page.screenshot({ path: join(output, 'failure.png'), fullPage: true });
    await writeFile(join(output, 'report.json'), JSON.stringify({ ...report, errors, warnings, failure: String(error) }, null, 2));
    throw error;
} finally { await browser.close(); server.close(); }
