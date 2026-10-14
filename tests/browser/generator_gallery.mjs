// Manual regression: provide Playwright through NODE_PATH; optional Chromium executable is argv[2].
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

const { chromium } = createRequire(import.meta.url)("playwright");
const project = new URL("../../", import.meta.url);
const output = join(tmpdir(), "vnccs-generator-gallery");
await mkdir(output, { recursive: true });
const fixtures = [];
for (const name of ["Step1_CharacterCreator", "Step1_CharacterCloner", "Step2_CharacterClothes", "Step3_CharacterEmotions"]) {
    const path = `workflows/VNCCS_3.2_${name}.json`;
    const workflow = JSON.parse(await readFile(new URL(path, project), "utf8"));
    const node = workflow.nodes.find(node => /^VNCCS_.*Generator$/.test(node.type));
    assert.ok(node && Array.isArray(node.size) && workflow.extra.ds.scale > 0);
    fixtures.push({ path, node, scale: workflow.extra.ds.scale });
}
const layoutScenarios = [
    ...[3, 6, 7, 12].map(count => ({ count, width: 1024, height: 1024, label: "square" })),
    ...[3, 6, 7, 12].map(count => ({ count, width: 1536, height: 640, label: "wide" })),
    ...[3, 6, 7, 12].map(count => ({ count, width: 640, height: 1536, label: "640x1536" })),
];
const appModule = `export const app={extensions:[],graph:{_nodes:[],links:{},extra:{},getNodeById(id){return this._nodes.find(node=>node.id===id)},setDirtyCanvas(){}},canvas:{ds:{scale:1,offset:[0,0]}},registerExtension(extension){this.extensions.push(extension)}};`;
const apiModule = `export const api=new EventTarget();api.apiURL=route=>route;api.fetchApi=(route,options)=>{
 if(route.startsWith('/vnccs/character_generator/progress')){
   const w=window.widget;
   const snapshot=w&&!w._disposed?{scope:w.progressScope(),node_id:String(w.node.id),epoch:w._progressEpoch||'browser',revision:w._progressRevision||0,request_id:w.data.ui?.progress_request_id,run_id:w._activeProgressRunId,stages:structuredClone(w.stageState)}:null;
   return Promise.resolve(new Response(JSON.stringify({snapshot}),{headers:{'Content-Type':'application/json'}}));
 }
 return fetch(route,options);
};`;
const pageHTML = `<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#181820;padding:18px}#host{transform-origin:top left}</style><div id="host"></div><script type="module">
import '/web/vnccs_character_generator.js';
import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';
const animate = Element.prototype.animate;
window.reveals = [];
window.cardEntrances = [];
window.characterWipes = [];
window.stageWipes = [];
Element.prototype.animate = function(keyframes, options) {
  const animation = animate.apply(this, arguments);
  if (options.duration===380 && this.classList.contains('vnccs-pipe-img')) cardEntrances.push({tile:this,animation});
  if (options.duration===1100 && (this.classList.contains('vnccs-pipe-img-content')||this.classList.contains('vnccs-pipe-img-placeholder')||this.classList.contains('vnccs-pipe-img-placeholder-scan'))) characterWipes.push({element:this,animation});
  if (options.duration===1350 && (this.classList.contains('vnccs-pipe-img-content')||this.classList.contains('vnccs-pipe-img-placeholder-scan'))) stageWipes.push({element:this,animation});
  if ((options.duration===420||options.duration===1100) && (this.classList.contains('vnccs-pipe-img')||this.classList.contains('vnccs-pipe-img-content'))) {
    const tile=this.classList.contains('vnccs-pipe-img')?this:this.parentNode;
    reveals.push({index:[...tile.parentNode.children].indexOf(tile),animation});
  }
  return animation;
};
window.setup = async fixture => {
  window.widget?.node.onRemoved?.();
  localStorage.clear();
  const host=document.querySelector('#host');host.replaceChildren();
  host.style.width=fixture.node.size[0]+'px';host.style.height=fixture.node.size[1]+'px';host.style.transform='scale('+fixture.zoom+')';
  app.canvas.ds.scale=fixture.zoom;app.graph._nodes=[];app.graph.extra={};
  class Node {
    constructor(){this.id=fixture.node.id;this.type=fixture.node.type;this.inputs=fixture.node.inputs;this.widgets=[{name:'widget_data',value:'{}'}];this.properties={};this.graph=app.graph;this.size=[0,0];}
    setSize(size){this.size=[...size];this.widgets.find(widget=>widget.element)?.triggerDraw();}
    addDOMWidget(name,type,element,options){const widget={name,type,element,options,_node:this,triggerDraw:()=>{element.style.width=this.size[0]+'px';}};this.widgets.push(widget);return widget;}
    onNodeCreated(){this.created=true;}
    onConfigure(){this.configured=true;}
  }
  await app.extensions.find(extension=>extension.name==='VNCCS.CharacterGenerator').beforeRegisterNodeDef(Node,{name:fixture.node.type});
  const node=new Node();app.graph._nodes.push(node);node.onNodeCreated();
  const w=node._vnccsCharacterGeneratorWidget;window.widget=w;
  if(w.root.isConnected)throw new Error('Construction must precede DOM mounting');
  if(node.widgets[0].value!=='{}')throw new Error('Construction overwrote saved settings before configure');
  node.setSize(fixture.node.size);
  node.widgets[0].value=fixture.node.widgets_values.find(value=>typeof value==='string'&&value.startsWith('{'));
  node.onConfigure();
  for(let i=0;i<4;i++)w.renderPreview();
  window.detachedCounts=['.vnccs-pipe-grid','.vnccs-pipe-preview-head','.vnccs-pipe-empty'].map(selector=>w.previewEl.querySelectorAll(selector).length);
  window.savedGrid=w.previewGrid;window.savedHead=w.previewEl.firstChild;
  host.append(w.root);node.onResize();w.renderPreview();
  window.reveals=[];window.cardEntrances=[];window.characterWipes=[];window.stageWipes=[];window.imageRun=crypto.randomUUID();
};
window.emit=(stage,detail)=>api.dispatchEvent(new CustomEvent('vnccs.character_generator.stage',{detail:{node_id:String(widget.node.id),stage,status:'running',...detail}}));
window.geometry=()=>[...widget.previewGrid.children].map(tile=>{const r=tile.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};});
window.cards=()=>[...widget.previewGrid.children].map(tile=>({src:tile.dataset.src||'',processing:tile.classList.contains('is-processing'),waiting:tile.classList.contains('is-waiting')}));
window.urls=(version,count=6,width=640,height=1536)=>Array.from({length:count},(_,i)=>'/view?filename=pose_generation_'+String(i+1).padStart(2,'0')+'.png&subfolder=VNCCS/preview&type=output&run='+imageRun+'&v='+version+'&item='+i+'&width='+width+'&height='+height+'&delay='+(i===0?250:5));
window.ready=true;
</script>`;

const server = createServer(async (req, res) => {
    try {
        const url = new URL(req.url, "http://localhost");
        let content, type = "text/javascript";
        if (url.pathname === "/") { content = pageHTML; type = "text/html"; }
        else if (url.pathname === "/scripts/app.js") content = appModule;
        else if (url.pathname === "/scripts/api.js") content = apiModule;
        else if (url.pathname.startsWith("/web/")) {
            assert.match(url.pathname, /^\/web\/(?:assets\/)?[\w.-]+$/);
            const binary = url.pathname.endsWith(".png");
            content = await readFile(new URL(url.pathname.slice(1), project), binary ? undefined : "utf8");
            if (binary) type = "image/png";
        } else if (url.pathname === "/view") {
            await new Promise(resolve => setTimeout(resolve, Number(url.searchParams.get("delay")) || 0));
            if (url.searchParams.has("broken")) { res.writeHead(404).end(); return; }
            const index = Number(url.searchParams.get("item")) || 0;
            const width = Number(url.searchParams.get("width"));
            const height = Number(url.searchParams.get("height"));
            const color=url.searchParams.get('v')==='new'?'#b8a9e8':'#ff8fa3';
            const x=width/2, bodyWidth=Math.min(width*0.4,height*0.2);
            content='<svg xmlns="http://www.w3.org/2000/svg" width="'+width+'" height="'+height+'" viewBox="0 0 '+width+' '+height+'"><rect width="'+width+'" height="'+height+'" fill="#101923"/><circle cx="'+x+'" cy="'+height*0.2+'" r="'+bodyWidth*0.24+'" fill="'+color+'"/><path d="M'+(x-bodyWidth/2)+' '+height*0.31+' L'+(x+bodyWidth/2)+' '+height*0.31+' L'+(x+bodyWidth*0.6)+' '+height*0.65+' L'+(x+bodyWidth*0.28)+' '+height*0.65+' L'+(x+bodyWidth*0.22)+' '+height*0.94+' L'+(x+bodyWidth*0.05)+' '+height*0.94+' L'+x+' '+height*0.7+' L'+(x-bodyWidth*0.05)+' '+height*0.94+' L'+(x-bodyWidth*0.22)+' '+height*0.94+' L'+(x-bodyWidth*0.28)+' '+height*0.65+' L'+(x-bodyWidth*0.6)+' '+height*0.65+'Z" fill="'+color+'"/><text x="20" y="45" fill="#e8e8f0" font-size="32" font-family="sans-serif">'+(index+1)+'</text></svg>';
            type="image/svg+xml";
        } else {
            type="application/json";
            if(url.pathname.startsWith('/object_info')) {
                const name=decodeURIComponent(url.pathname.split('/').pop());
                content=JSON.stringify({[name]:{input:{required:{color_correction_method:[['lab','adain','none']]}}}});
            } else if(url.pathname.endsWith('/seedvr_models')) {
                content=JSON.stringify({models:['seedvr2_3b_fp8_e4m3fn.safetensors','seedvr2_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors'].map(name=>({name,local_path:'models/diffusion_models/'+name,status:'installed',description:'SeedVR test inventory'})),vae:[]});
            } else content=JSON.stringify({});
        }
        res.writeHead(200,{"Content-Type":type,"Cache-Control":"no-store"}).end(content);
    } catch(error){res.writeHead(500).end(String(error));}
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
const browser=await chromium.launch({headless:true,...(process.argv[2]?{executablePath:process.argv[2]}:{})}).catch(error=>{server.close();throw error;});
const page=await browser.newPage({deviceScaleFactor:2});
const errors=[];
page.on('pageerror',error=>errors.push(String(error)));
const stable=(before,after)=>{
    assert.equal(after.length,before.length);
    before.forEach((rect,index)=>Object.keys(rect).forEach(key=>assert.ok(Math.abs(rect[key]-after[index][key])<=1,`Card ${index} ${key} moved: ${rect[key]} -> ${after[index][key]}`)));
};
async function measure(zoom){
    const geometry=await page.evaluate(()=>geometry());
    for(const [index,rect] of geometry.entries()){
        assert.ok(rect.width>0&&rect.height>0);
        const cell=await page.evaluate(()=>({width:parseFloat(getComputedStyle(widget.previewGrid).gridTemplateColumns),height:parseFloat(getComputedStyle(widget.previewGrid).gridTemplateRows)}));
        assert.ok(rect.width<=cell.width*zoom+1&&rect.height<=cell.height*zoom+1);
        assert.ok(Math.abs(rect.width-cell.width*zoom)<=1||Math.abs(rect.height-cell.height*zoom)<=1,'Each image must use the available cell space along one axis');
        const aspect=await page.evaluate(index=>{const tile=widget.previewGrid.children[index],image=tile._previewImage;return image?image.naturalWidth/image.naturalHeight:tile._previewAspect;},index);
        assert.ok(Math.abs(rect.width-rect.height*aspect)<=1,'Card frame must preserve the image aspect ratio');
    }
    return geometry;
}
async function assertPackedRows(zoom){
    const {cards,grid}=await page.evaluate(()=>{const element=widget.previewGrid,rect=element.getBoundingClientRect();return {cards:geometry(),grid:{...rect.toJSON(),clientLeft:element.clientLeft,clientWidth:element.clientWidth}};});
    const rows=[];
    for(const card of cards.sort((a,b)=>a.y-b.y||a.x-b.x)){
        const row=rows.find(row=>Math.abs(row[0].y-card.y)<=1);
        if(row)row.push(card);else rows.push([card]);
    }
    for(const row of rows){
        row.sort((a,b)=>a.x-b.x);
        const left=Math.min(...row.map(card=>card.x));
        const right=Math.max(...row.map(card=>card.x+card.width));
        const contentCenter=grid.left+(grid.clientLeft+grid.clientWidth/2)*zoom;
        assert.ok(Math.abs((left+right)/2-contentCenter)<=1,'Every gallery row must be visually centered');
        row.slice(1).forEach((card,index)=>assert.ok(Math.abs(card.x-(row[index].x+row[index].width)-8*zoom)<=1,'Cards must use the fixed 8px gallery gap'));
    }
}
const report=[];
const targetedOnly=process.env.VNCCS_BROWSER_SCENARIO==='clone-naked';
try {
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(()=>window.ready);
    if(!targetedOnly)for(const fixture of fixtures){
        for(const zoom of [fixture.scale,1]){
            await page.setViewportSize({width:Math.ceil(fixture.node.size[0]*zoom)+36,height:Math.ceil(fixture.node.size[1]*zoom)+36});
            await page.evaluate(fixture=>setup(fixture),{...fixture,zoom});
            assert.deepEqual(await page.evaluate(()=>detachedCounts),[1,1,1]);
            assert.equal(await page.evaluate(()=>widget.previewGrid===savedGrid&&widget.previewEl.firstChild===savedHead),true);
            await page.waitForFunction(()=>widget.seedvrAssets&&widget.nativeSeedvrAvailable);
            const tabAlignment=await page.evaluate(()=>{const head=widget.previewEl.firstChild.getBoundingClientRect(),tabs=widget.previewTabs.getBoundingClientRect();return Math.abs(head.right-tabs.right);});
            assert.ok(tabAlignment<=1,'Stage tabs must stay in the gallery top-right corner');
            const actual=await page.evaluate(()=>({size:widget.node.size,width:widget.root.clientWidth,height:widget.root.clientHeight}));
            assert.deepEqual(actual.size,fixture.node.size);assert.equal(actual.width,fixture.node.size[0]);assert.equal(actual.height,fixture.node.size[1]);
            await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(0).click();
            const stage=await page.evaluate(()=>widget.stages[0][0]);
            const poseStage=stage.endsWith('pose_generation');
            await page.evaluate(stage=>emit(stage,{current:0,total:6}),stage);
            await page.waitForFunction(()=>widget.previewGrid.children.length===6&&widget.previewGrid.style.gridTemplateColumns);
            const scanEdge=await page.evaluate(()=>{
                const scan=widget.previewGrid.children[0].querySelector('.vnccs-pipe-img-placeholder-scan');
                const band=getComputedStyle(scan,'::after');
                return {bottom:parseFloat(band.bottom),height:parseFloat(band.height)};
            });
            assert.ok(Math.abs(scanEdge.bottom+scanEdge.height/2)<=0.5,'The bright scan line must finish on the card bottom edge');
            await page.waitForFunction(()=>cardEntrances.length===6&&cardEntrances.every(entry=>entry.animation.startTime!==null));
            const entranceTimings=await page.evaluate(()=>cardEntrances.map(entry=>({
                index:[...widget.previewGrid.children].indexOf(entry.tile),
                delay:entry.animation.effect.getTiming().delay,
                fill:entry.animation.effect.getTiming().fill,
            })));
            assert.deepEqual(entranceTimings.map(entry=>entry.index),[0,1,2,3,4,5]);
            assert.deepEqual(entranceTimings.map(entry=>entry.delay),[0,70,140,210,280,350]);
            assert.ok(entranceTimings.every(entry=>entry.fill==='backwards'));
            if(zoom===fixture.scale){
                await page.waitForTimeout(120);
                await page.screenshot({path:join(output,fixture.node.type+'-entrance.png')});
            }
            const reserved=await page.evaluate(()=>({
                visible:[...widget.previewGrid.children].every(tile=>getComputedStyle(tile).visibility!=='hidden'),
                processing:cards().flatMap((card,index)=>card.processing?[index]:[]),
                waiting:cards().flatMap((card,index)=>card.waiting?[index]:[]),
            }));
            assert.equal(reserved.visible,poseStage,'Pose slots show mannequins while stages without a visual source stay hidden');
            if(poseStage){assert.deepEqual(reserved.processing,[0]);assert.deepEqual(reserved.waiting,[1,2,3,4,5]);}
            await page.evaluate(stage=>emit(stage,{current:0,total:6,images:urls('original')}),stage);
            await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>reveals.length),0);
            await page.waitForFunction(()=>reveals.length===6);
            await page.waitForFunction(()=>reveals.every(reveal=>reveal.animation.startTime!==null));
            const timings=await page.evaluate(()=>reveals.map(reveal=>({index:reveal.index,start:reveal.animation.startTime+reveal.animation.effect.getTiming().delay,fill:reveal.animation.effect.getTiming().fill})));
            assert.deepEqual(timings.map(reveal=>reveal.index),[0,1,2,3,4,5]);
            timings.slice(1).forEach((timing,index)=>assert.ok(timing.start-timings[index].start>=85));
            assert.ok(timings.every(timing=>timing.fill==='backwards'));
            await page.waitForTimeout(1000);
            const before=await measure(zoom);
            const previousSources=await page.evaluate(()=>cards().map(card=>card.src));
            await page.evaluate(stage=>emit(stage,{status:'running',current:0,total:6,images:[]}),stage);
            assert.deepEqual(await page.evaluate(()=>cards().map(card=>card.src)),previousSources,'A repeated run must keep the displayed gallery');
            assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>getComputedStyle(tile).visibility==='visible')),true,'A repeated run must keep card frames visible');
            assert.deepEqual((await page.evaluate(()=>cards())).flatMap((card,index)=>card.processing?[index]:[]),[0]);
            assert.deepEqual((await page.evaluate(()=>cards())).flatMap((card,index)=>card.waiting?[index]:[]),[1,2,3,4,5]);
            stable(before,await measure(zoom));
            if(poseStage){
                await page.evaluate(stage=>emit(stage,{status:'done',current:6,total:6,images:urls('original'),run_id:'previous-character'}),stage);
                await page.evaluate(()=>{widget.data.gender='male';});
                await page.evaluate(stage=>emit(stage,{status:'running',current:0,total:6,images:[],run_id:'new-character',preview_aspect:640/1536}),stage);
                assert.deepEqual(await page.evaluate(()=>({
                    themed:widget.previewEl.classList.contains('is-male-character'),
                    accent:getComputedStyle(widget.previewEl).getPropertyValue('--vnccs-preview-accent').trim(),
                    rgb:getComputedStyle(widget.previewEl).getPropertyValue('--vnccs-preview-accent-rgb').trim(),
                })),{themed:true,accent:'#63c7ff',rgb:'99, 199, 255'});
                assert.equal(await page.evaluate(()=>cards().every(card=>!card.src)),true,'A new character run must remove the previous character');
                assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>getComputedStyle(tile).visibility==='visible')),true,'A new character run must retain the card frames');
                await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>tile.querySelector('.vnccs-pipe-img-placeholder')?.complete));
                assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>tile.classList.contains('is-placeholder')&&tile.querySelector('.vnccs-pipe-img-placeholder')?.complete)),true,'New character slots must show the mannequin');
                assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>tile.querySelector('.vnccs-pipe-img-placeholder')?.src.includes('VNCCS_Character_Mannequin_Male.png'))),true,'Male character data must select the male mannequin');
                assert.deepEqual((await page.evaluate(()=>cards())).flatMap((card,index)=>card.processing?[index]:[]),[0]);
                assert.deepEqual((await page.evaluate(()=>cards())).flatMap((card,index)=>card.waiting?[index]:[]),[1,2,3,4,5]);
                stable(before,await measure(zoom));
                await page.evaluate(stage=>emit(stage,{status:'running',current:1,total:6,images:[urls('incremental',1,644,1536)[0]],replace_images:true,preview_start:0,processing_index:1,run_id:'new-character',preview_aspect:640/1536}),stage);
                await page.waitForFunction(()=>widget.previewGrid.children[0].dataset.src?.includes('v=incremental'));
                if(fixture.node.type==='VNCCS_CharacterGenerator'){
                    await page.waitForFunction(()=>characterWipes.length>=3);
                    const wipeLayers=await page.evaluate(()=>{
                        const layers=characterWipes.slice(-3);
                        for(const {animation} of layers){
                            animation.pause();
                            animation.currentTime=animation.effect.getTiming().delay+275;
                        }
                        return layers.map(({element,animation})=>({
                            className:element.className,
                            clips:animation.effect.getKeyframes().map(frame=>frame.clipPath).filter(Boolean),
                        }));
                    });
                    assert.match(wipeLayers[0].className,/vnccs-pipe-img-content/);
                    assert.equal(wipeLayers[0].clips.length,2);
                    assert.match(wipeLayers[0].clips[0],/100%/);
                    assert.match(wipeLayers[0].clips[1],/0%/);
                    assert.match(wipeLayers[1].className,/vnccs-pipe-img-placeholder/);
                    assert.equal(wipeLayers[1].clips.length,2);
                    assert.match(wipeLayers[1].clips[0],/0%/);
                    assert.match(wipeLayers[1].clips[1],/100%/);
                    if(zoom===fixture.scale){
                        await page.screenshot({path:join(output,fixture.node.type+'-character-wipe.png')});
                        await page.evaluate(()=>characterWipes.slice(-3).forEach(({animation})=>{animation.currentTime=animation.effect.getTiming().delay+1090;}));
                        await page.screenshot({path:join(output,fixture.node.type+'-character-wipe-bottom.png')});
                    }
                    await page.evaluate(()=>characterWipes.slice(-3).forEach(({animation})=>animation.play()));
                    await page.waitForFunction(()=>!widget.previewGrid.children[0].classList.contains('is-character-revealing'));
                    assert.deepEqual(await page.evaluate(()=>{
                        const style=getComputedStyle(widget.previewGrid.children[0].querySelector('.vnccs-pipe-img-placeholder'));
                        return {display:style.display,visibility:style.visibility,opacity:style.opacity};
                    }),{display:'none',visibility:'hidden',opacity:'0'});
                }
                assert.deepEqual(await page.evaluate(()=>cards().map(card=>Boolean(card.src))),[true,false,false,false,false,false]);
                assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].slice(1).every(tile=>tile.classList.contains('is-placeholder'))),true,'Unfinished sprites must retain mannequin previews');
                await page.evaluate(()=>{widget._previewPlaceholderStage=null;widget.renderPreview();});
                assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].slice(1).every(tile=>tile.classList.contains('is-placeholder'))),true,'Reloaded progress must reconstruct unfinished mannequin previews from stage state');
                stable(before,await page.evaluate(()=>geometry()));
            }
            await page.evaluate(stage=>emit(stage,{status:'running',current:1,total:6,images:urls('original'),run_id:'new-character'}),stage);
            await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>tile.dataset.src));
            await page.waitForTimeout(50);
            if(fixture.node.type==='VNCCS_CharacterGenerator'){
                await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>!tile.classList.contains('is-character-revealing')));
                assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>{
                    const style=getComputedStyle(tile.querySelector('.vnccs-pipe-img-placeholder'));
                    return style.display==='none'&&style.visibility==='hidden'&&style.opacity==='0';
                })),true,'Every completed character card must remove its mannequin layer from rendering');
            }
            stable(before,await measure(zoom));
            const cell=await page.evaluate(()=>({cols:getComputedStyle(widget.previewGrid).gridTemplateColumns.split(' ').length,width:parseFloat(getComputedStyle(widget.previewGrid).gridTemplateColumns),height:parseFloat(getComputedStyle(widget.previewGrid).gridTemplateRows)}));
            const cols=cell.cols;
            before.forEach((rect,index)=>{
                const column=index%cols,row=Math.floor(index/cols);
                assert.ok(Math.abs(rect.x+rect.width/2-(before[0].x+before[0].width/2)-column*(cell.width+8)*zoom)<=1);
                assert.ok(Math.abs(rect.y+rect.height/2-(before[0].y+before[0].height/2)-row*(cell.height+8)*zoom)<=1,'Mixed image formats remain centered in aligned rows');
            });
            await page.evaluate(stage=>{window.savedTiles=[...widget.previewGrid.children];emit(stage,{status:'done',current:6,total:6,message:'SeedVR upscaled batch of 6 image(s)'});},stage);
            const stageLayout=await page.evaluate(()=>{const chain=widget.chainEl.getBoundingClientRect();return {chain:{top:chain.top,bottom:chain.bottom},stages:[...widget.chainEl.children].map(stage=>{const rect=stage.getBoundingClientRect();return {top:rect.top,bottom:rect.bottom,width:rect.width};})};});
            assert.ok(stageLayout.stages.every(stage=>stage.top>=stageLayout.chain.top-1&&stage.bottom<=stageLayout.chain.bottom+1),'Completed stages must stay inside the footer');
            assert.ok(Math.max(...stageLayout.stages.map(stage=>stage.width))-Math.min(...stageLayout.stages.map(stage=>stage.width))<=1,'Completed stages must keep the same width');
            if(fixture.node.type==='VNCCS_EmotionsGenerator'){
                await page.evaluate(()=>{widget.data.emotion_pairs.push({...widget.data.emotion_pairs[0],emotion:'Browser next stage'});widget.renderPreview();widget.renderChain();});
            }
            const next=await page.evaluate(()=>widget.stages[1][0]);
            const entranceCount=await page.evaluate(()=>cardEntrances.length);
            await page.evaluate(stage=>emit(stage,{current:0,total:6}),next);
            await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(1).click();
            stable(before,await measure(zoom));
            await page.waitForFunction(()=>[...widget.previewGrid.children].every((tile,index)=>tile===savedTiles[index]&&tile.dataset.src));
            assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every((tile,index)=>tile===savedTiles[index]&&tile.dataset.src)),true);
            assert.equal(await page.evaluate(()=>cardEntrances.length),entranceCount,'Stage changes must not replay card entrances');
            for(let current=0;current<6;current++){
                await page.evaluate(({stage,current})=>emit(stage,{current,total:6}),{stage:next,current});
                const cards=await page.evaluate(()=>cards());
                assert.deepEqual(cards.flatMap((card,index)=>card.processing?[index]:[]),[current]);
                assert.deepEqual(cards.flatMap((card,index)=>card.waiting?[index]:[]),Array.from({length:5-current},(_,index)=>current+index+1));
                stable(before,await measure(zoom));
            }
            await page.evaluate(stage=>emit(stage,{current:2,total:6,processing_index:1}),next);
            assert.deepEqual((await page.evaluate(()=>cards())).flatMap((card,index)=>card.processing?[index]:[]),[1]);
            await page.evaluate(stage=>emit(stage,{current:0,total:2,processing_indices:[2,3]}),next);
            assert.deepEqual((await page.evaluate(()=>cards())).flatMap((card,index)=>card.processing?[index]:[]),[2,3]);
            if(zoom===fixture.scale)await page.screenshot({path:join(output,fixture.node.type+'-processing.png')});
            await page.evaluate(stage=>emit(stage,{current:1,total:6,images:urls('new')}),next);
            await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>tile.dataset.src?.includes('v=new')));
            for(let frame=0;frame<8;frame++){await page.waitForTimeout(40);stable(before,await measure(zoom));}
            const ring=await page.evaluate(()=>({style:getComputedStyle(widget.previewGrid.children[1],'::after').borderTopStyle,animation:getComputedStyle(widget.previewGrid.children[1],'::after').animationName,text:widget.previewGrid.children[1].textContent,fit:getComputedStyle(widget.previewGrid.children[1]._previewImage).objectFit}));
            assert.equal(ring.style,'solid');assert.equal(ring.animation,'vnccs-pipe-card-process');assert.equal(ring.text,'');assert.equal(ring.fit,'contain');
            await page.evaluate(()=>widget.openViewer(3));await page.waitForFunction(()=>widget.viewer?.img?.classList.contains('is-ready'));
            await page.evaluate(stage=>{widget.regenerateState={activeStage:stage,imageIndex:3,targetStages:[stage]};emit(stage,{current:0,total:1,images:[urls('target',4)[3]]});},next);
            await page.waitForFunction(()=>widget.viewer.img.src.includes('v=target')&&widget.viewer.img.classList.contains('is-ready'));
            assert.equal(await page.evaluate(()=>[...widget.previewGrid.children].every((tile,index)=>index===3||tile.dataset.src.includes('v=new'))),true);
            stable(before,await measure(zoom));await page.keyboard.press('Escape');
            if(fixture.node.type!=='VNCCS_EmotionsGenerator'){
                const toggles=page.locator('.vnccs-pipe-mode-tab');await toggles.filter({hasText:'OFF'}).click();
                for(let frame=0;frame<5;frame++){await page.waitForTimeout(70);stable(before,await measure(zoom));}
                await toggles.filter({hasText:'SeedVR'}).click();await page.waitForTimeout(360);stable(before,await measure(zoom));
            }
            await page.evaluate(()=>{widget.regenerateState=null;widget.stageState[widget.selectedPreview].status='done';widget.renderPreview();widget.renderChain();});
            await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>tile.querySelectorAll('.vnccs-pipe-img-content').length===1));
            stable(before,await measure(zoom));
            assert.equal(await page.evaluate(()=>widget.previewGrid.querySelectorAll('.is-processing,.is-waiting').length),0);
            assert.ok(await page.evaluate(()=>[...widget.previewGrid.children].every(tile=>tile.querySelectorAll('.vnccs-pipe-img-content').length===1&&tile.querySelectorAll('.vnccs-pipe-img-placeholder').length===1)));
            const label=fixture.node.type+'-'+(zoom===1?'100pct':'workflow');
            await page.screenshot({path:join(output,label+'.png')});
            report.push({workflow:fixture.path,type:fixture.node.type,size:fixture.node.size,zoom,cards:before.map(rect=>({width:rect.width/zoom,height:rect.height/zoom})),screenshot:label+'.png'});
            // A single result must fit the full gallery while preserving its own aspect ratio.
            await page.evaluate(()=>{widget.stageState[widget.selectedPreview]={status:'done',total:1,images:urls('single',1)};widget.renderPreview();});
            await page.waitForFunction(()=>widget.previewGrid.children.length===1&&widget.previewGrid.children[0].dataset.src.includes('v=single'));
            await page.waitForTimeout(500);await measure(zoom);
            assert.equal(await page.evaluate(()=>widget.previewEl.querySelectorAll('.vnccs-pipe-grid').length),1);
            const expansion=await page.evaluate(()=>{window.expansionFirst=widget.previewGrid.children[0];return {entrances:cardEntrances.length,next:widget.stages[1][0]};});
            await page.evaluate(stage=>emit(stage,{status:'running',current:0,total:6,images:[]}),expansion.next);
            await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(1).click();
            await page.waitForFunction(count=>widget.previewGrid.children.length===6&&cardEntrances.length===count+5,expansion.entrances);
            const added=await page.evaluate(({entrances})=>({
                firstReused:widget.previewGrid.children[0]===expansionFirst,
                indices:cardEntrances.slice(entrances).map(entry=>[...widget.previewGrid.children].indexOf(entry.tile)),
            }),expansion);
            assert.equal(added.firstReused,true);
            assert.deepEqual(added.indices,[1,2,3,4,5]);
        }
    }
    const layoutReport=[];
    if(!targetedOnly)for(const fixture of fixtures){
        const zoom=fixture.scale;
        await page.setViewportSize({width:Math.ceil(fixture.node.size[0]*zoom)+36,height:Math.ceil(fixture.node.size[1]*zoom)+36});
        for(const scenario of layoutScenarios){
            await page.evaluate(fixture=>setup(fixture),{...fixture,zoom});
            await page.waitForFunction(()=>widget.seedvrAssets&&widget.nativeSeedvrAvailable);
            await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(0).click();
            await page.evaluate(({count})=>emit(widget.stages[0][0],{current:0,total:count}),scenario);
            const initial=await page.evaluate(()=>({
                poseStage:widget.stages[0][0].endsWith('pose_generation'),
                visible:[...widget.previewGrid.children].every(tile=>getComputedStyle(tile).visibility!=='hidden'),
                waiting:[...widget.previewGrid.children].slice(1).map(tile=>({
                    filter:getComputedStyle(tile.querySelector('.vnccs-pipe-img-placeholder')).filter,
                    animation:getComputedStyle(tile.querySelector('.vnccs-pipe-img-placeholder')).animationName,
                })),
            }));
            assert.equal(initial.visible,initial.poseStage);
            if(initial.poseStage){
                assert.ok(initial.waiting.every(card=>card.filter.includes('grayscale(1)')),'Waiting cards must be grayscale');
                assert.ok(initial.waiting.every(card=>card.animation==='none'),'Waiting cards must stay static');
            }
            await page.evaluate(({count,width,height})=>emit(widget.stages[0][0],{status:'done',current:count,total:count,images:urls('layout',count,width,height)}),scenario);
            await page.waitForFunction(count=>widget.previewGrid.children.length===count&&[...widget.previewGrid.children].every(tile=>tile.dataset.src),scenario.count);
            await page.waitForTimeout(460+(scenario.count-1)*100);
            await measure(zoom);
            await assertPackedRows(zoom);
            const screenshot=fixture.node.type+'-'+String(scenario.count).padStart(2,'0')+'-'+scenario.label+'.png';
            await page.screenshot({path:join(output,screenshot)});
            layoutReport.push({type:fixture.node.type,zoom,...scenario,screenshot});
        }
    }
    const cloneFixture=fixtures.find(fixture=>fixture.node.type==='VNCCS_CharacterCloneGenerator');
    const cloneZoom=cloneFixture.scale;
    await page.setViewportSize({width:Math.ceil(cloneFixture.node.size[0]*cloneZoom)+36,height:Math.ceil(cloneFixture.node.size[1]*cloneZoom)+36});
    await page.evaluate(fixture=>setup(fixture),{...cloneFixture,zoom:cloneZoom});
    await page.evaluate(()=>{widget.data.nsfw_enabled=true;widget.syncStagesFromData();widget.renderPreview();widget.renderChain();});
    await page.evaluate(()=>emit('original_pose_generation',{status:'done',current:3,total:3,images:urls('clone-original',3,640,1536)}));
    await page.waitForFunction(()=>widget.previewGrid.children.length===3&&[...widget.previewGrid.children].every(tile=>tile.dataset.src?.includes('v=clone-original')));
    await page.evaluate(()=>emit('remove_clothes',{status:'running',current:0,total:1,images:urls('clone-cleaned',1,640,1536)}));
    await page.waitForFunction(()=>widget.selectedPreview==='remove_clothes'&&widget.stageState.remove_clothes.images?.[0]?.includes('v=clone-cleaned'));
    await page.evaluate(()=>emit('naked_pose_generation',{status:'running',current:0,total:3,preview_aspect:640/1536}));
    await page.waitForFunction(()=>widget.previewGrid.children.length===3&&[...widget.previewGrid.children].every(tile=>tile.dataset.src?.includes('v=clone-original')));
    await page.waitForTimeout(600);
    const nakedBase=await page.evaluate(()=>({
        cards:cards(),
        sources:[...widget.previewGrid.children].map(tile=>tile.dataset.src),
        geometry:geometry(),
    }));
    assert.ok(nakedBase.sources.every(source=>source.includes('v=clone-original')),'Naked Pose must fill every slot with Original Pose sprites');
    assert.deepEqual(nakedBase.cards.flatMap((card,index)=>card.processing?[index]:[]),[0]);
    assert.deepEqual(nakedBase.cards.flatMap((card,index)=>card.waiting?[index]:[]),[1,2]);
    await page.evaluate(()=>emit('naked_pose_generation',{status:'running',current:1,total:3,images:[urls('clone-naked',1,640,1536)[0]],preview_start:0,replace_images:true,processing_index:1}));
    await page.waitForFunction(()=>stageWipes.length===3);
    stable(nakedBase.geometry,await page.evaluate(()=>geometry()));
    await page.screenshot({path:join(output,'VNCCS_CharacterCloneGenerator-naked-transition.png')});
    if(!targetedOnly){
        const fixture=fixtures[0],zoom=fixture.scale;
        await page.emulateMedia({reducedMotion:'reduce'});
        await page.evaluate(fixture=>setup(fixture),{...fixture,zoom});await page.locator('.vnccs-pipe-preview-head .vnccs-pipe-tab').nth(0).click();
        await page.evaluate(()=>emit(widget.stages[0][0],{current:0,total:6,images:urls('reduced')}));
        await page.waitForFunction(()=>[...widget.previewGrid.children].every(tile=>tile.dataset.src));await page.waitForTimeout(50);
        assert.equal(await page.evaluate(()=>reveals.length),0);await measure(zoom);
        assert.equal(await page.evaluate(()=>getComputedStyle(widget.previewGrid.children[0],'::after').animationName),'none');
        const removed=await page.evaluate(()=>{const ui=widget._ui;widget.node.onRemoved();return {disposed:widget._disposed,motions:ui._motionAnimations?.size||0};});
        assert.equal(removed.disposed,true);assert.equal(removed.motions,0);
    }
    assert.deepEqual(errors,[]);
    await writeFile(join(output,'report.json'),JSON.stringify({report,layoutReport,deviceScaleFactor:2,reducedMotion:true,errors},null,2));
    console.log(JSON.stringify({scenarios:report.length,workflowSizes:fixtures.map(f=>({type:f.node.type,size:f.node.size,scale:f.scale})),errors,output}));
} catch(error){
    await page.screenshot({path:join(output,'failure.png')});console.error(await page.evaluate(()=>({type:widget?.node.type,geometry:geometry(),grids:widget?.previewEl.querySelectorAll('.vnccs-pipe-grid').length})));throw error;
} finally{await browser.close();server.close();}
