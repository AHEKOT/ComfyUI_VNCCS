// Production widgets in a ComfyUI-sized fixture; optional Chromium executable is argv[2].
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = createRequire(import.meta.url)('playwright');
const root = new URL('../../', import.meta.url), output = join(tmpdir(), 'vnccs-model-bundles');
await mkdir(output, { recursive: true });
const catalog = JSON.parse(await readFile(new URL('control_center.json', root), 'utf8'));
let baseline = false;
const details = process.argv.includes('--details');
const html = `<!doctype html><style>body{margin:0;background:#181820;padding:20px}#host{border:1px solid #69616d;background:#222;transform-origin:top left;position:relative;box-sizing:border-box}#label{height:30px;padding-left:10px;color:white;font:14px sans-serif}#mount{position:absolute;left:10px;top:40px}</style><div id="host"><div id="label"></div><div id="mount"></div></div><script type="module">
import '/web/vnccs_character_creator_v2.js';import '/web/vnccs_emotion_v2.js';import '/web/vnccs_character_generator.js';import{app}from'/scripts/app.js';
const catalog=${JSON.stringify(catalog)};window.fileState='missing';window.installs=0;window.longNames=()=>catalog.clip.forEach(entry=>entry.name+=' '+('long-component-name-'.repeat(12)));
const reply=data=>Promise.resolve(new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}}));
const enriched=()=>Object.fromEntries(Object.entries(catalog).map(([category,value])=>[category,Array.isArray(value)?value.map(e=>({...e,status:fileState==='installed'||fileState==='partial'&&category==='models'?'installed':'missing'})):value]));
const seed=(category,name)=>({name,local_path:'models/'+(category==='models'?'diffusion_models':'vae')+'/'+name,description:'Native SeedVR2 model.',status:fileState==='installed'?'installed':'missing'});
window.fixtureApi=(route,options={})=>{
 if(route==='/queue')return reply({queue_running:[],queue_pending:[]});
 if(route.includes('/control_center/check'))return reply(enriched());
 if(route.includes('/control_center/download')||route.includes('/seedvr_download')&&!route.includes('status')){window.installs++;window.fileState='downloading';return reply({status:'queued'});}
 if(route.includes('/manager/status'))return reply(Object.fromEntries(Object.entries(catalog).flatMap(([category,entries])=>Array.isArray(entries)?entries.map(e=>['cc_'+category+'_'+e.name,...['downloading','error'].includes(fileState)?[{status:fileState,progress:46,message:fileState==='error'?'Connection interrupted':'Downloading 46%'}]:[{}]]):[])));
 if(route.includes('/seedvr_download_status'))return reply(Object.fromEntries([['models','seedvr2_3b_fp8_e4m3fn.safetensors'],['vae','ema_vae_fp16.safetensors']].map(([category,name])=>[category+':'+name,['downloading','error'].includes(fileState)?{status:fileState,progress:46,message:fileState==='error'?'Connection interrupted':'Downloading 46%'}:{}])));
 if(route.includes('/seedvr_models'))return reply({models:[seed('models','seedvr2_3b_fp8_e4m3fn.safetensors')],vae:[seed('vae','ema_vae_fp16.safetensors')]});
 if(route.includes('/context_lists'))return reply({characters:['Alice'],checkpoints:[],diffusion_models:[],text_encoders:[],vae_models:[],loras:[],samplers:['euler','er_sde'],schedulers:['normal','simple']});
 if(route.includes('/list_characters'))return reply({characters:['Alice']});
 if(route.includes('/character_styles'))return reply({default_style:'custom',groups:[]});
 if(route.includes('/get_tags'))return reply({});if(route.includes('/get_emotions'))return reply({emotions:[]});if(route.includes('/get_character_costumes'))return reply({costumes:['Original'],pose_count:1});
 if(route.startsWith('/object_info'))return reply({UNETLoader:{input:{required:{unet_name:[[]]}}},VAELoader:{input:{required:{vae_name:[[]]}}},SeedVR2Preprocess:{},SeedVR2Conditioning:{},SeedVR2PostProcessing:{input:{required:{color_correction_method:[['lab','adain','none']]}}}});return reply({});
};
window.mountNode=async(type,width,height,scale)=>{
 class Node{constructor(){this.type=type;this.id=17;this.properties={};this.inputs=[];this.size=[0,0];this.graph=app.graph;this.widgets=type==='EmotionGeneratorV2'?[['character',''],['generation_model','Anima'],['generation_settings','{}'],['costumes_data','[]'],['emotions_data','[]'],['prompt_style','Anima']].map(([name,value])=>({name,value,options:{}})):[{name:'widget_data',value:'{}',options:{}}];}
 setSize(size){this.size=[...size];this.widgets.find(w=>w.element)?.triggerDraw();}setDirtyCanvas(){}
 addDOMWidget(name,type,element,options){const widget={name,type,element,options,_node:this,triggerDraw:()=>{const zoom=Number(getComputedStyle(element).zoom)||1;element.style.width=(this.size[0]-20)/zoom+'px';element.style.height=(this.size[1]-60)/zoom+'px';}};this.widgets.push(widget);return widget;}}
 const extension=app.extensions.find(e=>e.name===(type==='CharacterCreatorV2'?'VNCCS.CharacterCreatorV2':type==='EmotionGeneratorV2'?'VNCCS.EmotionGeneratorV2':'VNCCS.CharacterGenerator'));
 await extension.beforeRegisterNodeDef(Node,{name:type},app);const node=new Node();app.graph._nodes.push(node);node.onNodeCreated();window.node=node;node.setSize([width,height]);node.onConfigure?.();node.onResize?.();
 const widget=node.widgets.find(w=>w.element);document.querySelector('#mount').append(widget.element);widget.triggerDraw();node.onDrawBackground?.({});
 const host=document.querySelector('#host');host.style.width=width+'px';host.style.height=height+'px';host.style.transform='scale('+scale+')';document.querySelector('#mount').style.width=width-20+'px';document.querySelector('#mount').style.height=height-60+'px';document.querySelector('#label').textContent=type;app.canvas.ds.scale=scale;
};
window.rects=()=>[...document.querySelectorAll('.vnccs-creator-top-row,.vnccs-creator-col,.vnccs-creator-bottom-row,.ems-left-col,.ems-right-col,.ems-generation-panel,.vnccs-pipe-settings,.vnccs-pipe-preview,.vnccs-pipe-chain')].map(e=>({class:e.className,rect:[e.getBoundingClientRect().x,e.getBoundingClientRect().y,e.getBoundingClientRect().width,e.getBoundingClientRect().height]}));window.ready=true;
</script>`;
const server=createServer(async(req,res)=>{try{const path=new URL(req.url,'http://localhost').pathname;let body,type='text/javascript';
 if(path==='/'){body=html;type='text/html';}
 else if(path==='/scripts/app.js')body=`export const app={extensions:[],graph:{_nodes:[],links:{},extra:{},setDirtyCanvas(){},getNodeById(id){return this._nodes.find(n=>n.id===id)}},canvas:{ds:{scale:1,offset:[0,0]}},queuePrompt:async()=>{},registerExtension(e){this.extensions.push(e)}};`;
 else if(path==='/scripts/api.js')body=`export const api=new EventTarget();api.apiURL=r=>r;api.fetchApi=(r,o)=>window.fixtureApi(r,o);`;
 else if(path.startsWith('/web/')){assert.match(path,/^\/web\/(?:assets\/)?[\w.-]+$/);const binary=/\.(png|webp)$/.test(path);type=binary?'image/png':'text/javascript';body=baseline&&!binary?execFileSync('git',['show','HEAD:'+path.slice(1)],{cwd:root,encoding:'utf8'}):await readFile(new URL(path.slice(1),root),binary?undefined:'utf8');}
 else if(path.startsWith('/vnccs/')){body='{}';type='application/json';}else{res.writeHead(404).end();return;}
 res.writeHead(200,{'Content-Type':type,'Cache-Control':'no-store'}).end(body);
}catch(error){res.writeHead(500).end(String(error));}});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve)});
const browser=await chromium.launch({headless:true,...(process.argv[2]?{executablePath:process.argv[2]}:{})}).catch(error=>{server.close();throw error});
const results=[],errors=[];
const stable=(a,b,label)=>{assert.equal(a.length,b.length,label);a.forEach((entry,i)=>entry.rect.forEach((v,j)=>assert.ok(Math.abs(v-b[i].rect[j])<=1,label+' '+entry.class+' '+j+': '+v+' / '+b[i].rect[j])))};
try{
 const types=details ? [['CharacterCreatorV2',1280,800,'Qwen Image 2.1'],['EmotionGeneratorV2',920,650,'Anima'],['CharacterCreatorV2',1280,800,'Illustrious'],['EmotionGeneratorV2',920,650,'Illustrious']] : [['CharacterCreatorV2',1280,800,'Anima'],['EmotionGeneratorV2',920,650,'Qwen Image 2.1'],['VNCCS_CharacterGenerator',1180,760],['VNCCS_CharacterCloneGenerator',1180,760],['VNCCS_ClothesGenerator',1180,760]];
 for(const[type,defaultWidth,height,mode]of types)for(const delta of details?[0]:[-80,0,160])for(const scale of details?[1]:[0.75,1]){
  const width=defaultWidth+delta,label=type+'-'+width+'-'+scale+(details?'-'+mode.replaceAll(' ','_'):'');let original;
  for(const before of [true,false]){baseline=before;const page=await browser.newPage({viewport:{width:1700,height:1150},deviceScaleFactor:1});page.on('pageerror',error=>errors.push({type,before,error:String(error)}));
   await page.route(/^https?:\/\/(?!127\.0\.0\.1)/,route=>route.abort());await page.goto('http://127.0.0.1:'+server.address().port);await page.waitForFunction(()=>window.ready);
   if(!before){await page.evaluate(value=>window.fileState=value,mode==='Illustrious'?'missing':'partial');if(details)await page.evaluate(()=>longNames());}
   await page.evaluate(args=>mountNode(...args),[type,width,height,scale]);await page.waitForTimeout(500);
   assert.deepEqual(await page.evaluate(()=>node.size),[width,height],'Restored node size is preserved');
   if(mode)await page.getByRole('button',{name:new RegExp('^'+mode+'$','i')}).click();await page.waitForTimeout(250);
   const frame=await page.evaluate(()=>rects());
   if(before){original=frame;await page.locator('#host').screenshot({path:join(output,label+'-before.png')});}
   else{
    stable(original,frame,'Partial bundle keeps node composition');await page.locator('#host').screenshot({path:join(output,label+'-partial.png')});
    if(type.startsWith('VNCCS_'))await page.locator('.vnccs-seedvr-card.is-picker-head').click();
    const selector=type==='CharacterCreatorV2'?'.vnccs-creator-model-card-download:visible':type==='EmotionGeneratorV2'?'.ems-model-card-download:visible':'.vnccs-seedvr-download:visible';
    await page.locator(selector).first().click();await page.waitForFunction(()=>window.installs>0);
    const modal=page.getByRole('dialog');await modal.waitFor({state:'visible'});await page.waitForTimeout(2300);
    assert.equal(await modal.getByRole('button',{name:'Install Missing Files',exact:true}).isDisabled(),true);
    stable(frame,await page.evaluate(()=>rects()),'Download does not move panels');
    assert.equal(await page.locator(selector).first().isDisabled(),false,'The card keeps its Download action while incomplete');
    const bounds=await modal.boundingBox(),host=await page.locator('#host').boundingBox();assert.ok(bounds.x>=host.x-1&&bounds.y>=host.y-1&&bounds.x+bounds.width<=host.x+host.width+1&&bounds.y+bounds.height<=host.y+host.height+1,'Dialog fits node');
    await page.locator('#host').screenshot({path:join(output,label+'-downloading.png')});
    if(delta===0&&scale===1){await page.evaluate(()=>window.fileState='error');await page.waitForTimeout(2300);await page.waitForFunction(()=>document.querySelector('[role=dialog]').textContent.includes('Connection interrupted'));
     await page.locator('#host').screenshot({path:join(output,label+'-error.png')});await modal.getByRole('button',{name:'Install Missing Files',exact:true}).click();await page.waitForFunction(()=>window.fileState==='downloading');}
    await modal.getByRole('button',{name:'Close',exact:true}).click();await modal.waitFor({state:'detached'});
    await page.evaluate(()=>window.fileState='installed');await page.waitForTimeout(2300);stable(frame,await page.evaluate(()=>rects()),'Complete bundle keeps node composition');
    await page.locator('#host').screenshot({path:join(output,label+'-installed.png')});results.push({type,width,scale,frame,modal:bounds});
   }
   await page.evaluate(()=>node.onRemoved?.());await page.close();
  }
 }
 assert.deepEqual(errors,[]);console.log(JSON.stringify({scenarios:results.length,output}));
}finally{await writeFile(join(output,details?'details.json':'geometry.json'),JSON.stringify({results,errors},null,2));await browser.close();server.close();}
