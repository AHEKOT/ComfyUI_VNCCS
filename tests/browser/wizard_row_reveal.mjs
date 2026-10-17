// Manual visual regression: provide Playwright through NODE_PATH; optional Chromium executable is argv[2].
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

const { chromium } = createRequire(import.meta.url)("playwright");
const project = new URL("../../", import.meta.url);
const output = join(tmpdir(), "vnccs-wizard-row-reveal");
await mkdir(output, { recursive: true });

const appModule = "export const app={graph:{_nodes:[]},registerExtension(){}};";
const apiModule = "export const api={apiURL:route=>route,fetchApi:()=>Promise.resolve(new Response('{}'))};";
const pageHTML = `<!doctype html><meta charset="utf-8">
<style>
body{margin:0;background:#0a0a0f;color:#e8e8f0;font:600 16px Sora,Arial,sans-serif;padding:36px}
#panel{width:720px;padding:24px;background:rgba(20,16,30,.88);border:1px solid rgba(255,143,163,.25);border-radius:20px}
.field{display:grid;grid-template-columns:110px 1fr;align-items:center;min-height:66px;box-sizing:border-box;padding:10px 12px;margin:0 0 8px;border-bottom:1px solid rgba(255,143,163,.18)}
.label{font-size:11px;color:#9898a8;letter-spacing:.08em}
.control{box-sizing:border-box;width:100%;min-height:44px;padding:10px 12px;background:#211e2a;border:1px solid rgba(255,255,255,.1);border-radius:10px;color:#f0eaf4;font:600 14px Sora,Arial,sans-serif}
.tags{display:flex;align-items:center;gap:6px}.tag{padding:4px 8px;border-radius:7px;background:rgba(184,169,232,.14);border:1px solid rgba(255,182,200,.22)}
textarea.control{height:54px;resize:none;overflow:hidden}
</style><div id="panel"></div><script type="module">
import { animateWizardRows, SAKURA_INTERACTION_CSS } from '/web/vnccs_common.js';
document.head.append(Object.assign(document.createElement('style'),{textContent:SAKURA_INTERACTION_CSS}));
const entries=[
  {name:'RACE',kind:'tags',old:['human'],next:['elf','high elf']},
  {name:'SKIN',kind:'textarea',old:'pale skin',next:'porcelain skin, warm blush'},
  {name:'BODY',kind:'tags',old:['slim'],next:['athletic','tall']},
  {name:'HAIR',kind:'textarea',old:'short brown hair',next:'long silver hair, side braid'},
  {name:'EYES',kind:'textarea',old:'brown eyes',next:'bright blue eyes'},
];
const panel=document.querySelector('#panel');
function render(control,entry,next=false){
  const value=next?entry.next:entry.old;
  if(entry.kind==='textarea'){control.value=value;control.textContent=value;control.style.height=next&&entry.name==='HAIR'?'72px':'54px';return;}
  if(entry.kind==='tags'){control.replaceChildren(...value.map(text=>Object.assign(document.createElement('span'),{className:'tag',textContent:text})));return;}
}
window.rows=entries.map(entry=>{
  const row=document.createElement('div');row.className='field';
  const label=Object.assign(document.createElement('div'),{className:'label',textContent:entry.name});
  const control=document.createElement(entry.kind==='textarea'?'textarea':'div');
  control.className='control '+entry.kind;control.dataset.vnccsWizardRevealTarget='';
  render(control,entry,false);row.append(label,control);panel.append(row);return row;
});
const nativeAnimate=Element.prototype.animate;window.animationStarts=[];
Element.prototype.animate=function(frames,options){if(this.classList.contains('vnccs-common-wizard-reveal-scan'))animationStarts.push({index:rows.indexOf(this.closest('.field')),time:performance.now(),options});return nativeAnimate.apply(this,arguments)};
window.run=(male=false)=>animateWizardRows(rows,()=>rows.forEach((row,index)=>render(row.querySelector('.control'),entries[index],true)),{duration:240,gap:30,delay:90,accent:male?'#63c7ff':'#ff8fa3'});
window.reset=()=>rows.forEach((row,index)=>render(row.querySelector('.control'),entries[index],false));
window.ready=true;
</script>`;

const server = createServer(async (request, response) => {
    try {
        const url = new URL(request.url, "http://localhost");
        let content;
        let type = "text/javascript";
        if (url.pathname === "/") { content = pageHTML; type = "text/html"; }
        else if (url.pathname === "/scripts/app.js") content = appModule;
        else if (url.pathname === "/scripts/api.js") content = apiModule;
        else if (url.pathname.startsWith("/web/")) {
            assert.match(url.pathname, /^\/web\/[\w.-]+$/);
            content = await readFile(new URL(url.pathname.slice(1), project), "utf8");
        } else { response.writeHead(404).end(); return; }
        response.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(content);
    } catch (error) { response.writeHead(500).end(String(error)); }
});
await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });

const browser = await chromium.launch({ headless: true, ...(process.argv[2] ? { executablePath: process.argv[2] } : {}) })
    .catch(error => { server.close(); throw error; });
const page = await browser.newPage({ viewport: { width: 800, height: 540 }, deviceScaleFactor: 2 });
const errors = [];
page.on("pageerror", error => errors.push(String(error)));
try {
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => window.ready);
    await page.evaluate(() => { run(false); });
    assert.equal(await page.locator(".vnccs-common-wizard-reveal").count(), 5, "Old values must cover every live field before the first paint");
    const coveredValues = await page.locator(".vnccs-common-wizard-reveal-snapshot").allTextContents();
    assert.deepEqual(coveredValues.map(value => value.trim()), ["human", "pale skin", "slim", "short brown hair", "brown eyes"],
        "Every waiting row must retain its old visible value until its own wipe begins");
    const alignment = await page.locator(".field").evaluateAll(fields => fields.map(field => {
        const target = field.querySelector(".control").getBoundingClientRect();
        const layer = field.querySelector(".vnccs-common-wizard-reveal").getBoundingClientRect();
        return { dx: Math.abs(target.x-layer.x), dy: Math.abs(target.y-layer.y), dw: Math.abs(target.width-layer.width), dh: Math.abs(target.height-layer.height) };
    }));
    alignment.forEach(delta => Object.values(delta).forEach(value => assert.ok(value < 0.6, "Reveal mask must match the real control")));
    await page.waitForSelector(".vnccs-common-wizard-reveal.is-active");
    await page.locator(".vnccs-common-wizard-reveal.is-active").evaluate(layer => layer.querySelectorAll(".vnccs-common-wizard-reveal-old,.vnccs-common-wizard-reveal-scan")
        .forEach(element => element.getAnimations().forEach(animation => { animation.pause(); animation.currentTime = 120; })));
    const femaleScan = await page.locator(".vnccs-common-wizard-reveal.is-active .vnccs-common-wizard-reveal-scan").evaluate(element => {
        const style = getComputedStyle(element);
        const scan = element.getBoundingClientRect();
        const layer = element.parentElement.getBoundingClientRect();
        const oldStyle = getComputedStyle(element.previousElementSibling);
        const inset = oldStyle.clipPath.match(/inset\([^)]*\s([\d.]+)%\)$/)?.[1];
        return { width: scan.width, height: scan.height, background: style.backgroundImage,
            boundaryDelta: inset == null ? 999 : Math.abs((scan.x + scan.width / 2) - (layer.x + layer.width * Number(inset) / 100)) };
    });
    assert.ok(femaleScan.height > femaleScan.width * 8, "The scanner must be the full-height edge of a horizontal wipe");
    assert.ok(femaleScan.boundaryDelta < 3, "The light edge must stay on the old/new content boundary");
    assert.match(femaleScan.background, /255, 143, 163|ff8fa3/i);
    await page.screenshot({ path: join(output, "female-mid-reveal.png") });
    await page.locator(".vnccs-common-wizard-reveal.is-active").evaluate(layer => layer.querySelectorAll(".vnccs-common-wizard-reveal-old,.vnccs-common-wizard-reveal-scan")
        .forEach(element => element.getAnimations().forEach(animation => animation.play())));
    await page.waitForFunction(() => animationStarts.length === 5);
    const starts = await page.evaluate(() => animationStarts.map(item => item.time));
    starts.slice(1).forEach((time, index) => assert.ok(time - starts[index] >= 260, "Rows must reveal sequentially"));
    await page.waitForFunction(() => !document.querySelector(".vnccs-common-wizard-reveal"));

    await page.evaluate(() => { animationStarts.length = 0; reset(); run(true); });
    await page.waitForSelector(".vnccs-common-wizard-reveal.is-active");
    await page.locator(".vnccs-common-wizard-reveal.is-active").evaluate(layer => layer.querySelectorAll(".vnccs-common-wizard-reveal-old,.vnccs-common-wizard-reveal-scan")
        .forEach(element => element.getAnimations().forEach(animation => { animation.pause(); animation.currentTime = 120; })));
    const maleBackground = await page.locator(".vnccs-common-wizard-reveal.is-active .vnccs-common-wizard-reveal-scan").evaluate(element => getComputedStyle(element).backgroundImage);
    assert.match(maleBackground, /99, 199, 255|63c7ff/i);
    await page.screenshot({ path: join(output, "male-mid-reveal.png") });
    await page.locator(".vnccs-common-wizard-reveal.is-active").evaluate(layer => layer.querySelectorAll(".vnccs-common-wizard-reveal-old,.vnccs-common-wizard-reveal-scan")
        .forEach(element => element.getAnimations().forEach(animation => animation.play())));
    await page.waitForFunction(() => animationStarts.length === 5);
    await page.waitForFunction(() => !document.querySelector(".vnccs-common-wizard-reveal"));
    assert.deepEqual(errors, []);
    await writeFile(join(output, "report.json"), JSON.stringify({ starts, femaleScan, maleBackground, errors }, null, 2));
    console.log(JSON.stringify({ output, rows: 5, sequential: true, themes: ["female", "male"], errors }));
} finally {
    await browser.close();
    server.close();
}
