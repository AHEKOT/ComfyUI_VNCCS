// Run with Playwright on NODE_PATH and optionally pass a Chromium executable.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const { chromium } = createRequire(import.meta.url)('playwright');
const root = new URL('../../', import.meta.url);
const output = join(tmpdir(), 'vnccs-clothes-clone-layout');
await mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true, ...(process.argv[2] ? {executablePath:process.argv[2]} : {})});
const page = await browser.newPage({viewport:{width:1500,height:1100}});
const errors = [];
page.on('pageerror', err => errors.push(err.message));
const html = `<!doctype html><style>
body{margin:16px;background:#292734}#host{position:relative;transform-origin:top left;background:#17151e;color:white;outline:1px solid #666;overflow:hidden}#label{height:30px;padding-left:10px;font:14px sans-serif}#mount{position:absolute;top:40px;left:10px}
</style><div id="host"><div id="label">Clothes Designer</div><div id="mount"></div></div>
<script type="module">
import '/web/vnccs_clothes_designer.js';
import {app} from '/scripts/app.js';
class Node {
  constructor() {this.id=7;this.type='ClothesDesigner';this.graph=app.graph;this.inputs=[];this.properties={};this.widgets=[{name:'widget_data',value:'{}'}];}
  setSize(size) {this.size=[...size];this.widgets.find(w=>w.element)?.triggerDraw();}
  addDOMWidget(name,type,element,options) {
    document.querySelector('#mount').append(element);
    const w={name,type,element,options,_node:this,triggerDraw:()=>{
      const zoom=Number(getComputedStyle(element).zoom)||1;
      element.style.width=(this.size[0]-20)/zoom+'px';element.style.height=(this.size[1]-60)/zoom+'px';
    }};this.widgets.push(w);return w;
  }
}
await app.extension.beforeRegisterNodeDef(Node,{name:'ClothesDesigner'});
window.node=new Node();app.graph._nodes=[node];node.onNodeCreated();
window.sizeNode=(width,height,scale)=>{
  Object.assign(host.style,{width:width+'px',height:height+'px',transform:'scale('+scale+')'});
  Object.assign(mount.style,{width:(width-20)+'px',height:(height-60)+'px'});
  app.canvas.ds.scale=scale;node.setSize([width,height]);node.onResize(node.size);
};
window.restore=(saved)=>{node.widgets[0].value=JSON.stringify(saved);node.onConfigure({});};
window.ready=true;
</script>`;
const apiData = {
    '/queue': {queue_running:[],queue_pending:[]}, '/vnccs/context_lists': {characters:['Fixture']},
    '/vnccs/character_info': {name:'Fixture'}, '/vnccs/list_costumes': ['Coat'], '/vnccs/get_costume': {},
    '/vnccs/get_character_pose_preview_meta': {count:2},
};
const portrait = [1024,3072], landscape = [1536,512];
let sourceSize = portrait;
await page.route('**/*', async route => {
    const url = new URL(route.request().url()), path = url.pathname;
    if (url.hostname !== '127.0.0.1') return route.abort();
    if (path === '/') return route.fulfill({contentType:'text/html',body:html});
    if (path === '/scripts/app.js') return route.fulfill({contentType:'text/javascript',body:
        'export const app={graph:{_nodes:[],links:{},setDirtyCanvas(){}},canvas:{ds:{}},registerExtension(e){this.extension=e;}};'});
    if (path === '/scripts/api.js') return route.fulfill({contentType:'text/javascript',body:
        'export const api=new EventTarget();api.apiURL=p=>p;api.fetchApi=(p,o)=>fetch(p,o);'});
    if (/^\/web\/[\w.-]+$/.test(path)) return route.fulfill({contentType:'text/javascript',body:await readFile(new URL('.'+path,root))});
    if (['/view','/vnccs/get_preview','/vnccs/get_character_pose_preview'].includes(path)) {
        if (url.searchParams.get('filename')==='missing.png') return route.fulfill({status:404});
        const [w,h] = path === '/view' ? (url.searchParams.get('filename')==='wide.png' ? landscape : portrait) : sourceSize;
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 100 100" preserveAspectRatio="none"><rect width="100" height="100" fill="#295060"/><rect x="2" y="2" width="96" height="96" stroke="#ffc080" fill="none"/><circle cx="50" cy="50" r="12" fill="#cfe8e0"/></svg>`;
        return route.fulfill({contentType:'image/svg+xml',body:svg});
    }
    return route.fulfill({json:apiData[path] ?? {}});
});
const saved = name => ({character:'Fixture',costume:'Coat',activeTab:'generate',clone_image:name ? {name,type:'input'} : null});
const clone = page.getByRole('button',{name:'CLONE CLOTHES',exact:true});
const generate = page.getByRole('button',{name:'GENERATE CLOTHES',exact:true});
const frame = '.vnccs-clothes-preview-container';
async function geometry(expectedImages) {
    const snapshot = await page.waitForFunction(({frame,expectedImages}) => {
        const rect = el => {
            const {left,top,right,bottom,width,height}=el.getBoundingClientRect();
            return {left,top,right,bottom,width,height};
        };
        const at = s=>rect(document.querySelector(s));
        const nav = at('.cd-sprite-nav');
        const images = [...document.querySelectorAll(frame)].map(box=>box.querySelector(':scope > img'))
            .filter(img=>img.naturalWidth && getComputedStyle(img).display !== 'none');
        if (!nav.width || window.node._vnccsClothesSelectionPending || images.length !== expectedImages) return false;
        return {root:at('.vnccs-clothes-container'), grid:at('.vnccs-clothes-top-row'),
            cols:[...document.querySelectorAll('.vnccs-clothes-col')].map(rect),
            previews:[...document.querySelectorAll(frame)].map(rect),
            images:images.map(rect), contained:images.every(img=>getComputedStyle(img).objectFit==='contain'),
            action:at(frame+' div > button'), nav};
    },{frame,expectedImages});
    return snapshot.jsonValue();
}
function inside(inner,outer,label) {
    assert.ok(inner.width>0 && inner.height>0 && inner.top>=outer.top-1 && inner.bottom<=outer.bottom+1
        && inner.left>=outer.left-1 && inner.right<=outer.right+1,label+' '+JSON.stringify({inner,outer}));
}
try {
    await page.goto('http://127.0.0.1/');
    await page.waitForFunction(()=>window.ready);
    for (const [width,height,zoom,name] of [[710,920,1,'portrait.png'],[1280,800,0.75,'wide.png'],[895,670,1,null],[710,920,1,'missing.png']]) {
        sourceSize = name==='wide.png' ? landscape : portrait;
        await page.evaluate(args=>sizeNode(...args),[width,height,zoom]);
        await page.evaluate(restoreState=>restore(restoreState),saved(name));
        await generate.waitFor();
        await clone.click();
        const upload = page.getByRole('button',{name:name?'REPLACE IMAGE':'+ UPLOAD IMAGE',exact:true});
        await upload.waitFor();
        const label = width+'-'+(name?.replace('.png','')||'empty');
        for (const mode of ['switch','resize','switch back','restore']) {
            if (mode==='resize') await page.evaluate(args=>sizeNode(...args),[width-60,height-80,zoom]);
            if (mode==='switch back') {await generate.click();await clone.click();}
            if (mode==='restore') await page.evaluate(()=>restore(JSON.parse(node.widgets[0].value)));
            await upload.waitFor();
            const g = await geometry(name && name!=='missing.png' ? 2 : 1);
            assert.ok(g.contained, 'Preview images preserve their aspect ratios');
            if (mode==='switch') {
                await page.waitForFunction(() => {
                    const img = document.querySelector('.vnccs-clothes-col .vnccs-clothes-preview-img');
                    return !document.querySelector('.vnccs-common-ui-motion-ghost') && getComputedStyle(img).opacity==='1'
                        && !img.getAnimations().some(animation=>animation.playState==='running');
                });
                await page.locator('#host').screenshot({path:join(output,label+'.png')});
            }
            inside(g.grid,g.root,'Grid overflow after '+mode);
            for (const col of g.cols) inside(col,g.grid,'Panel overflow after '+mode);
            g.previews.forEach((preview,i)=>inside(preview,g.cols[i],'Preview overflow after '+mode));
            g.images.forEach((image,i)=>inside(image,g.previews[i],'Image cropped after '+mode));
            inside(g.action,g.previews[1],'Upload action overflow');inside(g.nav,g.cols[0],'Sprite navigation overflow');
            assert.deepEqual(await page.evaluate(()=>node.size),mode==='switch'?[width,height]:[width-60,height-80]);
        }
    }
    assert.deepEqual(errors,[]);
    console.log('Clothes Designer clone layout passed:',output);
} finally {await browser.close();}
