// Isolated Pose Set Library: production content/CSS with the external Studio shell as a fixture.
// NODE_PATH supplies Playwright; arguments are Chromium executable and screenshot directory.
import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";

const { chromium } = createRequire(import.meta.url)("playwright");
const output = process.argv[3];
assert.ok(output);
await mkdir(output, { recursive: true });
const ui = await readFile(new URL("../../web/pose_manager_ui.mjs", import.meta.url), "utf8");
const common = await readFile(new URL("../../web/vnccs_common.js", import.meta.url), "utf8");
const css = ui.match(/const CSS = `([\s\S]*?)`;/)[1]
    .replace("${SAKURA_THEME_CSS}", common.match(/export const SAKURA_THEME_CSS = `([\s\S]*?)`;/)[1])
    .replace("${POSE_LIBRARY_CSS}", "");
const html = `<!doctype html><meta charset="utf-8"><style>
body{margin:20px;background:#202024;font:14px Arial;color:#eee}#node{position:relative;border:1px solid #666;transform-origin:top left}
.title{height:30px;background:#303039;padding:8px;box-sizing:border-box}.vnccs-pm-shell{height:calc(100% - 30px);position:relative}
.vnccs-ps-modal-overlay{position:absolute;inset:0;display:flex;padding:12px;box-sizing:border-box}
.vnccs-ps-library-modal{display:flex;flex-direction:column;min-width:0;width:100%;border:1px solid #444;border-radius:12px}
.vnccs-ps-library-modal-header,.vnccs-ps-library-toolbar{display:flex;align-items:center;gap:12px;padding:12px}
.vnccs-ps-library-header-actions{display:flex;gap:8px;margin-left:auto}.vnccs-ps-library-toolbar input{flex:1;min-width:0}
.vnccs-ps-library-workspace{display:flex;flex:1;min-height:0}.vnccs-ps-library-modal-grid{flex:1;min-width:0}
.vnccs-ps-library-settings{display:none;flex:1;min-width:0}.settings-mode .vnccs-ps-library-settings{display:block}
.vnccs-set-library .settings-mode .vnccs-ps-library-modal-grid{display:none}button{cursor:pointer;color:inherit;background:#25252e;border:1px solid #555;border-radius:6px;padding:8px}
${css}</style><div id="node"><div class="title">VNCCS Pose Manager</div><div class="vnccs-pm-shell"><div id="manager"><button id="opener">Library</button></div></div></div>
<script type="module">
import { openSetLibrary } from '/web/pose_set_library.mjs';
class Shell {
 loadLibraryThumbnailSize(){return 320;}
 showLibraryModal(){
  const overlay=document.createElement('div');overlay.className='vnccs-ps-modal-overlay';
  overlay.innerHTML='<section class="vnccs-ps-library-modal"><header class="vnccs-ps-library-modal-header"><strong class="vnccs-ps-library-modal-title">Library</strong><div class="vnccs-ps-library-header-actions"><button class="vnccs-ps-library-save-current">Save current set</button><button class="vnccs-ps-library-menu-btn">Settings</button><button class="vnccs-ps-modal-close">Close</button></div></header><div class="vnccs-ps-library-toolbar"><input aria-label="Search sets"></div><div class="vnccs-ps-library-workspace"><div class="vnccs-ps-library-modal-grid"></div><div class="vnccs-ps-library-settings"></div></div></section>';
  this.container.append(overlay);this.libraryModal=overlay.firstElementChild;
  this.librarySearchInput=overlay.querySelector('input');this.libraryWorkspace=overlay.querySelector('.vnccs-ps-library-workspace');
  this.libraryGrid=overlay.querySelector('.vnccs-ps-library-modal-grid');this.librarySettingsEl=overlay.querySelector('.vnccs-ps-library-settings');
  this.librarySearchInput.oninput=()=>this.renderLibrary();this._closeLibraryModal=()=>overlay.remove();
  overlay.querySelector('.vnccs-ps-library-save-current').onclick=()=>this.showSaveToLibraryModal();
 }
}
const w=window.widget=new Shell();w.container=document.querySelector('.vnccs-pm-shell');w.managerPanel=document.querySelector('#manager');
w.poses=[{bones:{head:[1,2,3]}}];w.getPosePrompt=()=> 'Authored pose';w.awaitCurrentManagerPreviews=async()=>{};
w.loadPoseSetAsset=asset=>{window.loaded=asset;};w.setInterfaceMode=()=>{};
const preview=document.createElement('canvas');preview.width=120;preview.height=180;preview.getContext('2d').fillRect(40,20,40,140);
window.openLibrary=()=>openSetLibrary(w,document.querySelector('#opener'),()=>preview);
window.resize=([width,height,scale])=>{const node=document.querySelector('#node');node.style.width=width+'px';node.style.height=height+'px';node.style.transform='scale('+scale+')';};
window.ready=true;
</script>`;
const saved = { source: "local", id: "a".repeat(32), name: "Healthy set", description: "Authored poses", count: 1 };
const bundled = JSON.parse(await readFile(new URL("../../pose_set_defaults/17ee87dd13254ade89cddfddc4a65779.json", import.meta.url), "utf8"));
const female = JSON.parse(await readFile(new URL("../../pose_set_defaults/9e9d22f219d04b8794cc5831f521f7e7.json", import.meta.url), "utf8"));
const bundledSets = [bundled, female];
const bundledEntries = bundledSets.map(set => ({ source: "bundled", id: set.id, name: set.name, description: set.description, count: set.asset.poses.length }));
const bundledEntry = { source: "bundled", id: bundled.id, name: bundled.name, description: bundled.description, count: bundled.asset.poses.length };
const userCopy = structuredClone(bundled);
userCopy.asset.poses[0].prompt = "User-authored pose";
const errors = [{ path: `local/${"b".repeat(32)}.json`, message: "Invalid pose set. <img src=x onerror=alert(1)>" }];
let catalog = { sets: [bundledEntry, saved], repositories: [], errors: [] }, saveMode = "normal";
const saves = [], pageErrors = [];
const browser = await chromium.launch({ executablePath: process.argv[2], headless: true });
try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
    page.on("pageerror", error => { pageErrors.push(error.message); console.error(error); });
    await page.route("http://pose.fixture/**", async route => {
        const path = new URL(route.request().url()).pathname;
        if (path === "/") return route.fulfill({ contentType: "text/html", body: html });
        if (path === "/scripts/api.js") return route.fulfill({ contentType: "text/javascript", body: "export const api=Object.assign(new EventTarget(),{apiURL:x=>x,fetchApi:(path,options)=>fetch(path,options)});" });
        if (path === "/scripts/app.js") return route.fulfill({ contentType: "text/javascript", body: "export const app={};" });
        if (path.startsWith("/web/")) return route.fulfill({ contentType: "text/javascript", body: await readFile(new URL(`../../${path.slice(1)}`, import.meta.url), "utf8") });
        if (path === "/vnccs/pose-sets") return route.fulfill({ json: catalog });
        if (path === "/vnccs/pose-sets/preview-sources") return route.fulfill({ json: [] });
        for (const set of bundledSets) {
            const bundledPath = `/vnccs/pose-sets/bundled/${set.id}`;
            if (path === bundledPath) return route.fulfill({ json: set });
            if (path.startsWith(`${bundledPath}/preview/`)) return route.fulfill({ contentType: "image/webp", body: Buffer.from(set.previews[Number(path.split("/").at(-1))], "base64") });
        }
        if (path === `/vnccs/pose-sets/local/${userCopy.id}`) return route.fulfill({ json: userCopy });
        if (path === "/vnccs/pose-sets/save") {
            saves.push(route.request().postDataJSON());
            if (saveMode === "failure") return route.fulfill({ status: 500, json: { error: "Disk unavailable" } });
            if (saveMode === "refresh-error") return route.fulfill({ json: { ok: true, errors: [{ path: "PoseSetLibrary", message: "Changes saved, but the library could not be refreshed: Permission denied" }] } });
            catalog = { ...catalog, sets: [...catalog.sets, { ...saved, id: "c".repeat(32), name: saves.at(-1).name }] };
            return route.fulfill({ json: { ok: true, ...catalog } });
        }
        if (path.includes("/preview/")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="180"><rect x="40" y="20" width="40" height="140" fill="pink"/></svg>' });
        return route.fulfill({ status: 404, json: { error: "Unexpected request" } });
    });
    await page.goto("http://pose.fixture/");
    assert.deepEqual(pageErrors, []);
    await page.waitForFunction(() => window.ready);
    const status = page.locator(".vnccs-set-status");
    const open = async () => {
        await page.evaluate(() => openLibrary());
        await page.locator(".vnccs-set-load").first().waitFor();
        await page.waitForFunction(() => !document.querySelector('.vnccs-ps-library-modal').inert);
        await page.waitForFunction(() => [...document.querySelectorAll('.vnccs-set-preview img')]
            .every(image => image.complete && image.naturalWidth > 0));
    };
    for (const [name, size] of [["narrow", [900, 800, .75]], ["default", [1210, 940, .85]], ["wide", [1400, 1000, 1]]]) {
        await page.evaluate(size => resize(size), size);
        catalog.errors = [];
        await open();
        assert.equal(await status.textContent(), "");
        assert.equal(await page.getByRole("button", { name: `Delete set ${bundled.name}` }).count(), 0);
        assert.equal(await page.getByRole("button", { name: `Load set ${bundled.name}` }).locator("img").count(), 12);
        await page.screenshot({ path: join(output, `${name}-healthy.png`) });
        await page.evaluate(() => widget.closeSetLibrary());
        catalog.errors = errors;
        await open();
        assert.match(await status.textContent(), /read error.*Files were preserved/);
        assert.equal(await status.locator("img").count(), 0);
        assert.ok(await page.evaluate(() => {
            const root = document.querySelector('#node').getBoundingClientRect();
            return [...document.querySelectorAll('.vnccs-ps-library-modal button,.vnccs-set-status,.vnccs-set-card')].every(element => {
                const box = element.getBoundingClientRect();
                return box.x >= root.x - 1 && box.right <= root.right + 1 && box.y >= root.y - 1 && box.bottom <= root.bottom + 1
                    && element.scrollWidth <= element.clientWidth + 1;
            });
        }), `${name}: warning and controls fit the complete node`);
        await page.screenshot({ path: join(output, `${name}-warning.png`) });
        await page.getByRole("button", { name: "Save current set", exact: true }).click();
        await page.getByLabel("Name", { exact: true }).fill("Preserved draft");
        assert.ok(await page.evaluate(() => {
            const root = document.querySelector('#node').getBoundingClientRect();
            return [...document.querySelectorAll('.vnccs-set-form,.vnccs-set-status,.vnccs-set-form input,.vnccs-set-form textarea,.vnccs-set-form select,.vnccs-set-form button')].every(element => {
                const box = element.getBoundingClientRect();
                return box.x >= root.x && box.right <= root.right && box.y >= root.y && box.bottom <= root.bottom
                    && element.scrollHeight <= element.clientHeight + 1 && element.scrollWidth <= element.clientWidth + 1;
            });
        }), `${name}: complete save form and warning fit without scrolling`);
        await page.screenshot({ path: join(output, `${name}-warning-form.png`) });
        await page.evaluate(() => widget.closeSetLibrary());
    }
    await open();
    const save = async name => {
        await page.getByRole("button", { name: "Save current set", exact: true }).click();
        await page.getByLabel("Name", { exact: true }).fill(name);
        await page.getByRole("button", { name: "Save set", exact: true }).click();
        await page.waitForFunction(() => !document.querySelector('.vnccs-ps-library-modal').inert);
    };
    await save("New set");
    await page.getByRole("button", { name: "Load set New set" }).waitFor();
    assert.equal(saves.length, 1);
    assert.equal(saves[0].asset.poses[0].prompt, "Authored pose");
    assert.match(await status.textContent(), /Files were preserved/);
    saveMode = "refresh-error";
    await save("Saved despite refresh error");
    assert.equal(saves.length, 2);
    assert.match(await status.textContent(), /Changes saved/);
    assert.equal(await page.getByRole("button", { name: "Load set New set" }).count(), 1);
    await page.screenshot({ path: join(output, "save-refresh-error.png") });
    saveMode = "failure";
    await save("Retain this draft");
    assert.match(await status.textContent(), /Disk unavailable/);
    assert.equal(await page.getByLabel("Name", { exact: true }).inputValue(), "Retain this draft");
    await page.screenshot({ path: join(output, "save-failed-draft.png") });
    await page.getByRole("button", { name: "Close library" }).click();
    assert.equal(await page.evaluate(() => widget.managerPanel.inert), false);
    await open();
    await page.getByRole("button", { name: `Load set ${bundled.name}`, exact: true }).click();
    await page.waitForFunction(() => window.loaded?.poses?.length === 12);
    assert.deepEqual(await page.evaluate(() => loaded), bundled.asset);
    assert.equal(await page.evaluate(() => widget.managerPanel.inert), false);
    catalog = { sets: bundledEntries, repositories: [], errors: [] };
    for (const [name, size] of [["narrow", [900, 800, .75]], ["default", [1210, 940, .85]], ["wide", [1400, 1000, 1]]]) {
        await page.evaluate(size => resize(size), size);
        await open();
        assert.equal(await page.locator(".vnccs-set-card").count(), 2);
        assert.equal(await page.locator(".vnccs-set-delete").count(), 0);
        assert.equal(await page.locator(".vnccs-set-preview img").count(), 24);
        assert.ok(await page.locator(".vnccs-ps-library-modal-grid").evaluate(grid => {
            const outer = grid.getBoundingClientRect();
            return [...grid.children].every(card => {
                const box = card.getBoundingClientRect();
                return box.x >= outer.x && box.right <= outer.right && box.y >= outer.y && box.bottom <= outer.bottom;
            }) && grid.scrollWidth <= grid.clientWidth + 1 && grid.scrollHeight <= grid.clientHeight + 1;
        }), `${name}: both system sets fit the existing gallery`);
        await page.screenshot({ path: join(output, `${name}-system-sets.png`) });
        await page.evaluate(() => widget.closeSetLibrary());
    }
    catalog = { ...catalog, sets: [...bundledEntries, { ...bundledEntry, source: "local" }] };
    await open();
    assert.equal(await page.getByRole("button", { name: `Load set ${bundled.name}`, exact: true }).count(), 2);
    assert.equal(await page.getByRole("button", { name: `Delete set ${bundled.name}`, exact: true }).count(), 1);
    const userCard = page.locator(".vnccs-set-card").filter({ has: page.getByRole("button", { name: `Delete set ${bundled.name}`, exact: true }) });
    await page.evaluate(() => { window.loaded = null; });
    await userCard.locator(".vnccs-set-load").click();
    await page.waitForFunction(() => window.loaded?.poses?.[0]?.prompt === "User-authored pose");
    assert.deepEqual(await page.evaluate(() => loaded), userCopy.asset);
    catalog = { ...catalog, sets: bundledEntries };
    await open();
    await page.evaluate(() => { window.loaded = null; });
    await page.getByRole("button", { name: `Load set ${female.name}`, exact: true }).click();
    await page.waitForFunction(() => window.loaded?.poses?.length === 12);
    assert.deepEqual(await page.evaluate(() => loaded), female.asset);
    assert.deepEqual(pageErrors, []);
    console.log("Pose library browser checks passed: male/female system sets with 24 original previews, user/system isolation with identical IDs, loading each source, three sizes/scales, corrupt catalog warning, successful save, refresh failure, failed-write draft and cleanup.");
} finally {
    await browser.close();
}
