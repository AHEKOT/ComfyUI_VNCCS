// Manual browser regression: provide Playwright through NODE_PATH; optional Chromium executable is argv[2].
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

const { chromium } = createRequire(import.meta.url)("playwright");
const project = new URL("../../", import.meta.url);
const output = join(tmpdir(), "vnccs-generator-gallery");
await mkdir(output, { recursive: true });
const pageHTML = `<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#181820;padding:18px}#host{width:100%;height:calc(100vh - 36px);transform-origin:top left}</style><div id="host"></div><script type="module">
import { CharacterGeneratorWidget, DEFAULT_DATA, CSS } from '/web/vnccs_character_generator.js';
import { createSakuraInteractions } from '/web/sakura_interactions.mjs';
import { api } from '/scripts/api.js';
const style = document.createElement('style'); style.textContent = CSS; document.head.append(style);
const animate = Element.prototype.animate;
window.reveals = [];
Element.prototype.animate = function(keyframes, options) {
  const result = animate.apply(this, arguments);
  if (this.classList.contains('vnccs-pipe-img-content') && options.duration === 420) {
    reveals.push({ index: [...this.parentNode.parentNode.children].indexOf(this.parentNode), animation: result, image: this });
  }
  return result;
};
window.setup = flags => {
  if (window.widget) { widget.node.onRemoved?.(); widget._ui.dispose(); }
  const host = document.querySelector('#host'); host.replaceChildren();
  const node = { id: 17, widgets: [{ name:'widget_data',value:'{}' }], properties: {} };
  const w = Object.assign(Object.create(CharacterGeneratorWidget.prototype), {
    isClone:false, isClothes:false, isEmotions:false, ...flags, node, data: structuredClone(DEFAULT_DATA), title:'Gallery test', fieldDrafts:new Map(), nodeDefs:{},
    seedvrDownloads:{}, seedvrAssets:{models:[]},
    syncCharacterSourceData() {}, connectedEmotionStudioMode:()=> 'qi2', saveBrowserState() {}, refreshProgress() {},
  });
  w.data.emotion_pairs = [{costume:'Naked',emotion:'Happy'}, {costume:'Naked',emotion:'Sad'}];
  w.stages = w.currentStages(); w.selectedPreview = w.stages[0][0];
  w.stageState = Object.fromEntries(w.stages.map(([key]) => [key,{status:'waiting',images:null}]));
  w.root = document.createElement('div'); w.root.className='vnccs-pipe-root vnccs-common-ui-host'; host.append(w.root);
  w.settingsEl=document.createElement('div'); w.settingsEl.className='vnccs-pipe-settings'; w.settingsEl.dataset.sakuraScroll='settings';
  const main=document.createElement('div'); main.className='vnccs-pipe-main';
  w.previewEl=document.createElement('div'); w.previewEl.className='vnccs-pipe-preview';
  w.chainEl=document.createElement('div'); w.chainEl.className='vnccs-pipe-chain'; w.chainEl.dataset.sakuraScroll='stages';
  main.append(w.previewEl,w.chainEl); w.root.append(w.settingsEl,main);
  w._ui=createSakuraInteractions(w.root,node); w.updateModeClasses(); w.renderSettings(); w.renderPreview(); w.renderChain(); w.bindEvents();
  w.observer=new ResizeObserver(()=>w.schedulePreviewGridLayout(w.previewGrid)); w.observer.observe(w.previewEl);
  const removed=node.onRemoved; node.onRemoved=()=>{w.observer.disconnect();removed?.();};
  window.widget=w; window.reveals=[];
  window.imageRun=crypto.randomUUID();
};
window.emit=(stage, detail)=>api.dispatchEvent(new CustomEvent('vnccs.character_generator.stage',{detail:{node_id:'17',stage,status:'running',...detail}}));
window.geometry=()=>[...widget.previewGrid.children].map(tile=>{const r=tile.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};});
window.cards=()=>[...widget.previewGrid.children].map(tile=>({src:tile.dataset.src||'',processing:tile.classList.contains('is-processing'),waiting:tile.classList.contains('is-waiting')}));
window.ready=true;
</script>`;

const server = createServer(async (req, res) => {
    try {
        const url = new URL(req.url, "http://localhost");
        let content, type = "text/javascript";
        if (url.pathname === "/") { content = pageHTML; type = "text/html"; }
        else if (url.pathname === "/scripts/app.js") content = "export const app={graph:{_nodes:[],setDirtyCanvas(){}},registerExtension(){}};";
        else if (url.pathname === "/scripts/api.js") content = "export const api=new EventTarget();api.apiURL=route=>route;api.fetchApi=(route,options)=>fetch(route,options);";
        else if (url.pathname.startsWith("/web/")) {
            assert.match(url.pathname, /^\/web\/[\w.-]+$/);
            content = await readFile(new URL(url.pathname.slice(1), project), "utf8");
            if (url.pathname.endsWith("vnccs_character_generator.js")) content += "\nexport { CharacterGeneratorWidget, DEFAULT_DATA, CSS };";
        } else if (url.pathname.startsWith("/image/")) {
            await new Promise(resolve => setTimeout(resolve, Number(url.searchParams.get("delay")) || 0));
            if (url.searchParams.has("broken")) { res.writeHead(404).end(); return; }
            const index = Number(url.pathname.split("/").pop()) || 0;
            const width = index % 3 === 0 ? 420 : 240, height = index % 3 === 1 ? 460 : 360;
            const color = url.searchParams.get("v") === "new" ? "#b8a9e8" : "#ff8fa3";
            content = '<svg xmlns="http://www.w3.org/2000/svg" width="'+width+'" height="'+height+'" viewBox="0 0 240 360"><rect width="240" height="360" fill="#101923"/><circle cx="120" cy="69" r="34" fill="'+color+'"/><path d="M85 107 Q120 95 155 107 L175 235 L147 237 L141 338 L120 338 L117 245 L108 338 L85 338 L91 237 L66 235Z" fill="'+color+'"/><text x="16" y="25" fill="#e8e8f0" font-family="sans-serif">'+(index+1)+'</text></svg>';
            type = "image/svg+xml";
        } else { content = JSON.stringify({ snapshot: null }); type = "application/json"; }
        res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(content);
    } catch (error) { res.writeHead(500).end(String(error)); }
});
await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
const browser = await chromium.launch({ headless: true, ...(process.argv[2] ? { executablePath: process.argv[2] } : {}) });
const page = await browser.newPage({ viewport: { width: 1100, height: 720 } });
const errors = [];
page.on("pageerror", error => errors.push(String(error)));
const stable = (before, after) => {
    assert.equal(after.length, before.length);
    before.forEach((rect, index) => Object.keys(rect).forEach(key => assert.ok(Math.abs(rect[key] - after[index][key]) <= 1, `Card ${index} ${key} moved: ${rect[key]} -> ${after[index][key]}`)));
};
let scenarios = 0;
try {
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => window.ready);
    for (const [mode, flags] of [["character", {}], ["clone", { isClone: true }], ["clothes", { isClothes: true }], ["emotions", { isEmotions: true }]]) {
        for (const viewport of [{ width: 1100, height: 720 }, { width: 730, height: 560 }]) {
            await page.setViewportSize(viewport);
            await page.evaluate(flags => setup(flags), flags);
            const stages = await page.evaluate(() => widget.stages.map(([key]) => key));
            await page.evaluate(stage => emit(stage, { current: 0, total: 6 }), stages[0]);
            await page.waitForFunction(() => widget.previewGrid.children.length === 6 && widget.previewGrid.style.gridTemplateColumns);
            const before = await page.evaluate(() => { window.savedTiles=[...widget.previewGrid.children]; return geometry(); });
            assert.ok(before.every(r => r.width > 30 && r.height > 30));
            const sortedRows = new Map();
            for (const r of before) { const row = Math.round(r.y); if (!sortedRows.has(row)) sortedRows.set(row, []); sortedRows.get(row).push(r); }
            for (const row of sortedRows.values()) assert.ok(row.every(r => Math.abs(r.height-row[0].height) <= 1 && Math.abs(r.width-row[0].width) <= 1));
            await page.evaluate(stage => emit(stage, { current: 0, total: 6, images: Array.from({length:6}, (_,i)=>'/image/'+i+'?run='+imageRun+'&delay='+(i===0?250:10)) }), stages[0]);
            await page.waitForTimeout(100);
            assert.equal(await page.evaluate(() => reveals.length), 0, "Fast later downloads stay behind the first card");
            await page.waitForFunction(() => reveals.length === 6);
            const timings = await page.evaluate(() => reveals.map(r=>({index:r.index,start:r.animation.startTime+r.animation.effect.getTiming().delay,fill:r.animation.effect.getTiming().fill})));
            assert.deepEqual(timings.map(r=>r.index), [0,1,2,3,4,5]);
            for (let i=1;i<timings.length;i++) assert.ok(timings[i].start-timings[i-1].start >= 85);
            assert.ok(timings.every(r=>r.fill === "backwards"));
            stable(before, await page.evaluate(() => geometry()));
            await page.waitForTimeout(1000);
            await page.evaluate(stage => emit(stage, { status:'done',current:6,total:6 }), stages[0]);
            await page.evaluate(stage => emit(stage, { current:0,total:6 }), stages[1]);
            stable(before, await page.evaluate(() => geometry()));
            assert.equal(await page.evaluate(() => [...widget.previewGrid.children].every((tile,i)=>tile===savedTiles[i] && tile.dataset.src)), true);
            await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(0).click();
            stable(before,await page.evaluate(()=>geometry()));
            await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(1).click();
            stable(before,await page.evaluate(()=>geometry()));
            for (let current=0;current<6;current++) {
                await page.evaluate(({stage,current})=>emit(stage,{current,total:6}), {stage:stages[1],current});
                const cards = await page.evaluate(() => cards());
                assert.deepEqual(cards.map((c,i)=>c.processing?i:null).filter(i=>i!==null), [current]);
                assert.deepEqual(cards.map((c,i)=>c.waiting?i:null).filter(i=>i!==null), Array.from({length:5-current},(_,i)=>i+current+1));
                stable(before, await page.evaluate(() => geometry()));
            }
            await page.evaluate(stage => emit(stage, { current:2,total:6,processing_index:1 }), stages[1]);
            assert.deepEqual((await page.evaluate(() => cards())).map((c,i)=>c.processing?i:null).filter(i=>i!==null), [1]);
            await page.evaluate(stage => emit(stage, { current:0,total:2,processing_indices:[2,3] }), stages[1]);
            assert.deepEqual((await page.evaluate(() => cards())).map((c,i)=>c.processing?i:null).filter(i=>i!==null), [2,3]);
            await page.evaluate(stage => emit(stage, { current:1,total:6,images:Array.from({length:6},(_,i)=>'/image/'+i+'?run='+imageRun+'&v=new&delay='+ (i===0?220:5)) }), stages[1]);
            await page.waitForFunction(() => [...widget.previewGrid.children].every(tile=>tile.dataset.src?.includes('v=new')));
            for (let frame=0;frame<8;frame++) { await page.waitForTimeout(40); stable(before, await page.evaluate(() => geometry())); }
            const overlay = await page.evaluate(() => {
                const tile=widget.previewGrid.children[1], ring=getComputedStyle(tile,'::after');
                return {mask:ring.maskComposite,inset:ring.inset,text:tile.textContent,fit:getComputedStyle(tile._previewImage).objectFit,animation:ring.animationName};
            });
            assert.ok(overlay.mask.split(',').every(value=>value.trim()==="exclude")); assert.equal(overlay.fit,"contain"); assert.equal(overlay.text,"");
            assert.equal(overlay.animation,"vnccs-pipe-card-process");
            if (viewport.width===1100 && mode==='character') await page.screenshot({path:join(output,'processing.png')});
            await page.evaluate(() => {widget.previewGrid.children[3].focus();widget.previewTabs.scrollLeft=31;window.tabScroll=widget.previewTabs.scrollLeft;});
            await page.evaluate(stage=>emit(stage,{current:3,total:6}),stages[1]);
            assert.equal(await page.evaluate(()=>document.activeElement===savedTiles[3] && widget.previewTabs.scrollLeft===tabScroll),true);
            await page.evaluate(()=>widget.openViewer(3));
            await page.waitForFunction(()=>widget.viewer?.img?.classList.contains('is-ready'));
            await page.evaluate(stage=>{widget.regenerateState={activeStage:stage,imageIndex:3,targetStages:[stage]};emit(stage,{current:0,total:1,images:['/image/3?v=target']});},stages[1]);
            assert.deepEqual((await page.evaluate(() => cards())).map((c,i)=>c.processing?i:null).filter(i=>i!==null), [3]);
            await page.waitForFunction(()=>widget.previewGrid.children[3].dataset.src.includes('v=target'));
            assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every((tile,i)=>i===3||tile.dataset.src.includes('v=new'))),true,"Regenerating one image preserves its processed siblings");
            await page.waitForFunction(()=>widget.viewer.img.src.includes('v=target')&&widget.viewer.img.classList.contains('is-ready'));
            await page.keyboard.press('Escape');
            assert.equal(await page.evaluate(()=>widget.viewer===null),true);
            stable(before,await page.evaluate(()=>geometry()));
            if (!flags.isEmotions) {
                const toggles=page.locator('.vnccs-pipe-mode-tab');
                await toggles.filter({hasText:'OFF'}).click();
                for (let frame=0;frame<7;frame++) {await page.waitForTimeout(50);stable(before,await page.evaluate(()=>geometry()));}
                await toggles.filter({hasText:'SeedVR'}).click();await page.waitForTimeout(360);stable(before,await page.evaluate(()=>geometry()));
            }
            await page.evaluate(()=>{widget.regenerateState=null;widget.stageState[widget.selectedPreview].status='done';widget.renderPreview();});
            await page.waitForTimeout(600);
            assert.equal(await page.evaluate(()=>widget.previewGrid.querySelectorAll('.is-processing,.is-waiting').length),0);
            assert.ok(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>tile.querySelectorAll('img').length<=1)));
            if (viewport.width===1100 && mode==='character') await page.screenshot({path:join(output,'complete.png')});
            await page.evaluate(()=>{document.querySelector('#host').style.transform='scale(0.65)';});
            const zoomed=await page.evaluate(()=>geometry());
            await page.evaluate(stage=>emit(stage,{current:0,total:6}),stages[1]);
            await page.waitForTimeout(180);stable(zoomed,await page.evaluate(()=>geometry()));
            await page.evaluate(()=>{document.querySelector('#host').style.transform='none';});
            await page.evaluate(()=>{
                window.reflowAnimations=[];
                const play=widget._ui._playMotion;
                widget._ui._playMotion=(element,keyframes,duration,finish)=>{
                    const animation=play(element,keyframes,duration,finish);
                    if (element.classList.contains('vnccs-pipe-img')&&duration===320) reflowAnimations.push(animation);
                    return animation;
                };
                widget.stageState[widget.selectedPreview].total=8; widget.renderPreview();
            });
            await page.waitForFunction(()=>reflowAnimations.length>0);
            assert.equal(await page.evaluate(()=>savedTiles.every((tile,i)=>tile===widget.previewGrid.children[i])),true);
            await page.waitForTimeout(400);
            assert.equal(await page.evaluate(()=>widget.previewGrid.children.length),8);
            scenarios++;
        }
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.evaluate(()=>{setup({});emit(widget.stages[0][0],{current:0,total:2,images:['/image/0','/image/1']});});
    await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>tile.dataset.src));
    assert.equal(await page.evaluate(()=>reveals.length),0);
    assert.equal(await page.evaluate(()=>getComputedStyle(widget.previewGrid.children[0],'::after').animationName),'none');
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({scenarios, reducedMotion:true, errors, screenshots:output}));
} catch (error) {
    await page.screenshot({path:join(output,'failure.png')});
    console.error(await page.evaluate(()=>({mode:widget.root.className,geometry:geometry(),gridHeight:widget.previewGrid.clientHeight,emptyDisplay:getComputedStyle(widget.previewEmpty).display,headHeight:widget.previewEl.firstChild.clientHeight,previewHeight:widget.previewEl.clientHeight})));
    throw error;
} finally { await browser.close(); server.close(); }
