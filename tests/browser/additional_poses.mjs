import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const { chromium } = createRequire(import.meta.url)("playwright");
const project = new URL("../../", import.meta.url);
const output = join(tmpdir(), "vnccs-additional-poses-browser");
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.argv[2] ? { executablePath: process.argv[2] } : {}) });
const page = await browser.newPage({ viewport: { width: 1600, height: 950 }, reducedMotion: "reduce" });
const errors = [];
page.on("pageerror", error => errors.push(error.message));

const html = `<!doctype html><meta charset="utf-8"><style>
body{display:flex;gap:16px;margin:16px;background:#292734;color:#e8e8f0;font:14px sans-serif}
.node{position:relative;flex:none;background:#17151e;outline:1px solid #666;transform-origin:top left}
.node header{height:38px;padding:10px;box-sizing:border-box}
.mount{position:absolute;inset:44px 10px 12px}
#selector .slots{padding:0 10px;text-align:right;color:#a6a4b0;font:12px/16px sans-serif}
#selector .mount{top:92px}
</style><section class="node" id="selector"><header>VNCCS Additional Poses</header>
<div class="slots">character<br>sheets_path<br>background</div><div class="mount"></div></section>
<section class="node" id="generator"><header>VNCCS Additional Poses Generator</header><div class="mount"></div></section>
<script type="module">
import '/web/vnccs_additional_poses.js';
import '/web/vnccs_character_generator.js';
import '/web/vnccs_pose_manager.js';
import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';
await app.extensions.find(extension => extension.name === 'VNCCS.CharacterGenerator').setup();
window.queueGenerator = async () => { await app.queuePrompt(); return app.submissions; };
const poseSync = {
    studios:new Set(), sourceSignatures:new Map(), findSourceNode(){ return null; },
    registerStudio(studio){ this.studios.add(studio); },
    applyToStudio(studio,values){
        if (Number.isFinite(values.age)) studio.meshParams.age = values.age;
        if (Number.isFinite(values.gender)) studio.meshParams.gender = values.gender;
        return true;
    },
};
window.enablePoseSync = () => { window.__vnccsPoseStudioCharacterCreatorSync = poseSync; };
app.registerExtension({name:'VNCCS.PoseStudio',beforeRegisterNodeDef(type,data){
    if (data.name !== 'VNCCS_PoseStudio') return;
    type.prototype.onNodeCreated = function () {
        this.studioWidget = {meshParams:{age:12,gender:0}};
        poseSync.registerStudio(this.studioWidget);
    };
}});
async function mount(id, host, type, inputs = []) {
    class Node {
        constructor() {
            Object.assign(this, {id, type, inputs, graph: app.graph, properties: {}, widgets: [{name:'widget_data', value:'{}'}]});
        }
        setSize(size) {
            const widget = this.widgets.find(widget => widget.element);
            const minimum = widget?.options.getMinHeight?.() ?? 0;
            const mount = host.querySelector('.mount');
            const insets = mount.offsetTop + parseInt(getComputedStyle(mount).bottom);
            this.size = [size[0], Math.max(size[1], minimum + insets)];
            [host.style.width, host.style.height] = this.size.map(value => value + 'px');
            widget?.triggerDraw();
        }
        addDOMWidget(name, type, element, options) {
            const mount = host.querySelector('.mount');
            mount.append(element);
            const widget = {name, element, options, _node:this, triggerDraw: () => {
                // LiteGraph gives a flexible DOM widget the remaining height. A legacy
                // computeSize override can allocate a fixed height, leaving unused space.
                element.style.width = mount.clientWidth + 'px';
                element.style.height = (widget.computeSize?.(mount.clientWidth)?.[1] ?? mount.clientHeight) + 'px';
            }};
            this.widgets.push(widget);
            return widget;
        }
    }
    for (const extension of app.extensions) await extension.beforeRegisterNodeDef?.(Node, {name:type});
    const node = new Node();
    app.graph._nodes.push(node);
    node.onNodeCreated();
    if (node.widgets[0].value !== '{}') throw new Error('Construction overwrote stored data');
    return node;
}
window.source = await mount(1, selector, 'VNCCS_AdditionalPoses');
app.graph.links[1] = {origin_id:1};
window.target = await mount(2, generator, 'VNCCS_AdditionalPosesGenerator', [{name:'sheets_path',link:1}]);
target.setSize([1000,650]);
window.addPoseManager = id => mount(id, document.createElement('section'), 'VNCCS_PoseManager');
window.poseManager = await addPoseManager(3);
window.resizeSource = (width,height,scale) => {
    selector.style.transform = 'scale(' + scale + ')';
    selector.style.marginRight = (width * (scale - 1)) + 'px';
    source.setSize([width,height]);
    source.onResize();
};
window.selectState = () => JSON.parse(source.widgets[0].value);
window.restore = raw => {source.widgets[0].value = raw;source.onConfigure();};
window.widget = target._vnccsCharacterGeneratorWidget;
window.emit = (stage, status, extra = {}) => api.dispatchEvent(new CustomEvent('vnccs.character_generator.stage', {detail:{
    node_id:'2',character_name:'Ada',scope:widget.progressScope(),epoch:'fixture',revision:++window.revision,stage,status,...extra,
}}));
window.revision = 0;
</script>`;
const app = `export const app = {extensions:[],submissions:0,async queuePrompt(){this.submissions++},graph:{_nodes:[],extra:{},links:{},getNodeById(id){return this._nodes.find(node=>node.id===id)},setDirtyCanvas(){}},canvas:{ds:{scale:1,offset:[0,0]}},registerExtension(e){this.extensions.push(e)}};`;
const api = `export const api = new EventTarget(); api.apiURL = url => url; api.fetchApi = (url, options) => fetch(url, options);`;
const sprite = name => ({ name, url: `/view?filename=${name}` });
const happy = [sprite("happy_1.png"), sprite("happy_2.png")];
const seedvrModel = "seedvr2_3b_fp8_e4m3fn.safetensors";
const seedvrVae = "ema_vae_fp16.safetensors";
const availableFiles = { UNETLoader: [], VAELoader: [seedvrVae] };
await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.hostname !== "127.0.0.1") return route.abort();
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: html });
    if (url.pathname === "/scripts/app.js") return route.fulfill({ contentType: "text/javascript", body: app });
    if (url.pathname === "/scripts/api.js") return route.fulfill({ contentType: "text/javascript", body: api });
    if (url.pathname === "/web/pose_manager_ui.mjs") return route.fulfill({ contentType: "text/javascript", body: "export function mountManager() {}" });
    if (/^\/web\/(?:assets\/)?[\w.-]+$/.test(url.pathname)) {
        return route.fulfill({ body: await readFile(new URL(`.${url.pathname}`, project)),
            contentType: url.pathname.endsWith(".png") ? "image/png" : "text/javascript" });
    }
    if (url.pathname === "/vnccs/additional_poses/catalog") {
        if (!url.searchParams.has("character")) return route.fulfill({ json: { characters: ["Ada", "Bea", "Retired"] } });
        if (url.searchParams.get("character") === "Retired") return route.fulfill({ status: 404, json: { error: "Character configuration is missing." } });
        if (!url.searchParams.has("costume")) return route.fulfill({ json: { costumes: ["Coat", "Empty", "A long costume name for a narrow node"] } });
        const emotions = url.searchParams.get("costume") === "Coat"
            ? [{ name: "Neutral", sprites: [sprite("neutral.png")] }, { name: "Happy", sprites: happy }]
            : [];
        return route.fulfill({ json: { emotions } });
    }
    if (url.pathname === "/vnccs/character_info") {
        const character = url.searchParams.get("character");
        const profiles = { Ada: { age: 40, sex: "Male" }, Bea: { age: 67, gender: "female" } };
        return route.fulfill({ json: profiles[character] || {} });
    }
    if (url.pathname === "/view") {
        const long = url.searchParams.get("filename") === "happy_2.png";
        return route.fulfill({ contentType: "image/svg+xml", body: `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="240" viewBox="0 0 160 240"><circle cx="80" cy="62" r="32" fill="#edb0a0"/><path d="M42 116Q80 80 118 116L140 220H20Z" fill="${long ? "#ad95db" : "#6caab3"}"/></svg>` });
    }
    if (url.pathname.startsWith("/object_info/")) {
        const name = decodeURIComponent(url.pathname.split("/").at(-1));
        const required = name === "UNETLoader" ? { unet_name: [availableFiles[name]] }
            : name === "VAELoader" ? { vae_name: [availableFiles[name]] } : { color_correction_method: [["lab"]] };
        return route.fulfill({ json: { [name]: { input: { required } } } });
    }
    return route.fulfill({ json: {} });
});

const choose = async (field, value) => {
    await page.getByRole("combobox", { name: field, exact: true }).click();
    await page.getByRole("option", { name: value, exact: true }).click();
};
const preview = page.locator("#selector .vnccs-additional-poses-preview > img");
const scope = () => page.evaluate(() => { widget.syncCharacterSourceData(); return widget.progressScope(); });
async function shown(filename) {
    await page.waitForFunction(name => {
        const image = document.querySelector("#selector img");
        return image.naturalWidth && getComputedStyle(image).display !== "none" && image.src.includes(name);
    }, filename);
}
async function mannequins(filename) {
    await page.waitForFunction(name => {
        const cards = [...document.querySelectorAll("#generator .vnccs-pipe-img.is-placeholder")];
        return cards.length === 2 && cards[0].classList.contains("is-processing") && cards[1].classList.contains("is-waiting")
            && cards.every(card => {
                const image = card.querySelector(".vnccs-pipe-img-placeholder");
                return image?.naturalWidth && image.src.endsWith(`/${name}`);
            });
    }, filename);
}
async function layout(name) {
    const mount = await page.locator("#selector .mount").boundingBox();
    const root = await page.locator("#selector .vnccs-additional-poses").boundingBox();
    for (const metric of ["x", "y", "width", "height"]) {
        assert.ok(Math.abs(mount[metric] - root[metric]) <= 1, `${name} widget does not fill its allocation on ${metric}`);
    }
    for (const selector of [".vnccs-additional-poses-fields", ".vnccs-additional-poses-preview", ".vnccs-common-creator-sprite-nav"]) {
        const box = await page.locator(`#selector ${selector}`).boundingBox();
        assert.ok(box.width > 0 && box.height > 0 && box.x >= root.x - 1 && box.y >= root.y - 1
            && box.x + box.width <= root.x + root.width + 1 && box.y + box.height <= root.y + root.height + 1, `${name} ${selector} exceeds the node`);
    }
    await page.locator("#selector").screenshot({ path: join(output, `${name}.png`) });
}

try {
    await page.goto("http://127.0.0.1/");
    await shown("neutral.png");
    assert.deepEqual(await page.evaluate(() => selectState()), { character: "Ada", costume: "Coat", emotion: "Neutral", sprite: "neutral.png" });
    await page.evaluate(() => enablePoseSync());
    await page.waitForFunction(() => poseManager.studioWidget.meshParams.age === 40 && poseManager.studioWidget.meshParams.gender === 1);
    await choose("Character", "Bea");
    await page.waitForFunction(() => poseManager.studioWidget.meshParams.age === 67 && poseManager.studioWidget.meshParams.gender === 0);
    assert.deepEqual(await page.evaluate(async () => {
        const lateManager = await addPoseManager(4);
        return [lateManager.studioWidget.meshParams.age, lateManager.studioWidget.meshParams.gender];
    }), [67, 0]);
    await shown("neutral.png");
    await scope();
    await page.evaluate(() => emit("pose_generation", "running", {
        character_name: "Bea", run_id: "female", current: 0, total: 2,
    }));
    await mannequins("VNCCS_Character_Mannequin.png");
    await choose("Character", "Ada");
    await page.waitForFunction(() => poseManager.studioWidget.meshParams.age === 40 && poseManager.studioWidget.meshParams.gender === 1);
    await shown("neutral.png");
    for (const [field, choices] of [
        ["Character", ["Ada", "Bea", "Retired"]], ["Costume", ["Coat", "Empty", "A long costume name for a narrow node"]],
        ["Emotion", ["Neutral", "Happy"]],
    ]) {
        const select = page.getByRole("combobox", { name: field, exact: true });
        await select.click();
        assert.deepEqual(await page.locator("#selector").getByRole("option").allTextContents(), choices);
        await select.press("Escape");
    }
    await choose("Character", "Retired");
    await page.getByText("Character configuration is missing.").waitFor();
    await choose("Character", "Ada");
    await choose("Costume", "Empty");
    await page.getByText("No sprites in this costume.").waitFor();
    assert.equal(await page.getByRole("combobox", { name: "Emotion" }).isDisabled(), true);
    await choose("Costume", "Coat");
    await shown("neutral.png");
    await choose("Emotion", "Happy");
    await shown("happy_1.png");
    const before = await preview.boundingBox();
    const oldScope = await scope();
    await page.getByRole("button", { name: "Next sprite" }).click();
    await shown("happy_2.png");
    assert.deepEqual(await preview.boundingBox(), before);
    const newScope = await scope();
    assert.notEqual(newScope, oldScope);
    const queue = () => page.evaluate(() => queueGenerator());
    const modal = page.locator("#generator .vnccs-pipe-modal");
    assert.equal(await queue(), 0);
    assert.match(await modal.innerText(), /SeedVR2 Model Required/iu);
    assert.ok((await modal.innerText()).includes(seedvrModel));
    await page.locator("#generator").screenshot({ path: join(output, "generator-missing-seedvr.png") });
    await modal.getByRole("button", { name: "OK" }).click();

    availableFiles.UNETLoader = [seedvrModel];
    availableFiles.VAELoader = [];
    assert.equal(await queue(), 0);
    assert.ok((await modal.innerText()).includes(seedvrVae));
    await modal.getByRole("button", { name: "OK" }).click();

    await page.evaluate(() => widget.set("upscaler", "model", ""));
    assert.equal(await queue(), 0);
    assert.match(await modal.innerText(), /Select a SeedVR2 diffusion model/);
    await modal.getByRole("button", { name: "OK" }).click();
    await page.evaluate(model => {
        widget.set("upscaler", "model", model);
        widget.set("upscaler", "mode", "off");
    }, seedvrModel);
    assert.equal(await queue(), 1);

    availableFiles.VAELoader = [seedvrVae];
    await page.evaluate(() => widget.set("upscaler", "mode", "seedvr"));
    assert.equal(await queue(), 2);
    assert.equal(await page.evaluate(() => JSON.parse(target.widgets[0].value).ui.progress_scope), newScope);
    await page.evaluate(oldScope => emit("pose_generation", "running", { scope: oldScope, message: "Old selection" }), oldScope);
    assert.notEqual(await page.evaluate(() => widget.stageState.pose_generation.message), "Old selection");
    await page.evaluate(() => emit("pose_generation", "running", { message: "Generating new pose", run_id: "male", current: 0, total: 2 }));
    assert.equal(await page.evaluate(() => widget.stageState.pose_generation.status), "running");
    await mannequins("VNCCS_Character_Mannequin_Male.png");
    assert.equal(await page.locator("#generator .vnccs-pipe-preview.is-male-character").count(), 1);
    await page.locator("#generator").screenshot({ path: join(output, "generator-male-placeholder.png") });
    happy.push(sprite("sprite_pose_0003.png"));
    await page.evaluate(() => emit("bg_remove", "done", { images: ["/view?filename=sprite_pose_0003.png"], current: 1, total: 1 }));
    await page.getByText("2/3", { exact: true }).waitFor();
    assert.equal(await page.getByRole("button", { name: "Regenerate", exact: true }).count(), 0);

    const saved = await page.evaluate(() => {
        const serialized = { widgets_values: ["outdated"] };
        source.onSerialize(serialized);
        return serialized.widgets_values[0];
    });
    assert.equal(JSON.parse(saved).sprite, "happy_2.png");
    await page.evaluate(raw => { resizeSource(280,500,.8);restore(raw); }, saved);
    await shown("happy_2.png");
    await layout("narrow-restored");
    await page.evaluate(() => resizeSource(360,620,1));
    await layout("default");
    await page.evaluate(() => resizeSource(430,580,1.15));
    await layout("wide-zoomed");
    await page.evaluate(raw => { resizeSource(750,940,.75);restore(raw); }, saved);
    await shown("happy_2.png");
    await layout("tall-restored");
    await page.evaluate(() => resizeSource(360,520,1));
    await page.locator('#generator [data-focus-key="preview-bg_remove"]').click();
    await page.locator("#generator .vnccs-pipe-img-content").first().waitFor({ state: "visible" });
    await page.locator("#generator").screenshot({ path: join(output, "generator.png") });

    await page.evaluate(raw => restore(JSON.stringify({ ...JSON.parse(raw), sprite: "missing.png" })), saved);
    await page.getByText("Sprite 'missing.png' is missing.").waitFor();
    await layout("missing-sprite");
    assert.equal(await page.evaluate(() => selectState().sprite), "missing.png");
    await page.getByRole("button", { name: "Previous sprite" }).click();
    await shown("happy_1.png");
    await page.evaluate(() => { source.onRemoved();target.onRemoved(); });
    assert.deepEqual(errors, []);
    console.log(`Additional Poses layouts and selection checks passed. Screenshots: ${output}`);
} finally {
    await browser.close();
}
