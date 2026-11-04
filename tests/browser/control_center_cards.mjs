// Run with Playwright on NODE_PATH; optionally pass a Chromium executable as argv[2].
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const { chromium } = createRequire(import.meta.url)('playwright');
const source = await readFile(new URL('../../web/vnccs_control_center.js', import.meta.url), 'utf8');
const common = await readFile(new URL('../../web/vnccs_common.js', import.meta.url), 'utf8');
const interactions = await readFile(new URL('../../web/sakura_interactions.mjs', import.meta.url), 'utf8');
const css = source.match(/s.textContent = `([\s\S]*?)`;/)[1]
    .replace('${SAKURA_INTERACTION_CSS}', common.match(/export const SAKURA_INTERACTION_CSS = `([\s\S]*?)`;/)[1]);
const browser = await chromium.launch({ headless: true, ...(process.argv[2] ? { executablePath: process.argv[2] } : {}) });
try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 3 });
    await page.setContent('<div class="vnccs-cc-root" id="root"><div class="vnccs-cc-lora-grid vnccs-cc-lora-grid--compact" id="grid"></div></div>');
    await page.addStyleTag({ content: css + '\nbody{margin:0;background:#0a0a0f}#root{height:auto;padding:6px}' });
    await page.addScriptTag({ content: `const app = { registerExtension() {} }; const SAKURA_INTERACTION_CSS = "";
        const setHelpText = (element, text) => { if (text) element.dataset.vnccsHelp = text; };
        ${source.replace(/^import .*;\n/gm, '')}
        window.widget = Object.assign(Object.create(VNCCSControlCenterWidget.prototype), {
            state: { loras: [] }, config: { lora: [] }, dlStatus: {}, _playMotion() {},
        });
        window.mount = ({ width, long = false, empty = false, custom = false, compact = true }) => {
            document.querySelector('#root').style.width = width + 'px';
            const grid = document.querySelector('#grid'); grid.replaceChildren();
            const names = ['VNCCS Pose Studio QI2', 'VNCCS Overhaul QI2', 'VNCCS Clothes Core QI2'];
            const descriptions = ['Pose Studio LoRA for Qwen Image 2.1.', 'Overhaul for Character Creator', 'QI2 LoRA for clothing consistency.'];
            widget.config.lora = names.map((name, index) => ({ name: long ? name.repeat(20) : name,
                description: empty ? '' : long ? descriptions[index].repeat(20) : descriptions[index], version: empty ? '' : long ? '123.456.789.123' : ['2.0', '2.1', '2.6'][index],
                kind: 'QI2', status: 'installed', custom, hf_path: 'model.safetensors' }));
            widget.state.loras = widget.config.lora.map(entry => ({ name: entry.name, auto_apply: !compact, strength: 1 }));
            widget.config.lora.forEach(entry => grid.appendChild(widget._renderLoraCard(entry, { compact, active: !compact, mode: custom ? 'custom' : 'pipe' })));
        };
        window.geometry = () => [...document.querySelector('#grid').children].map(card => {
            const frame = card.getBoundingClientRect();
            const result = { frame: [frame.x, frame.y, frame.width, frame.height] };
            for (const name of ['top', 'name', 'detail-row', 'desc', 'version', 'tools']) {
                const rect = card.querySelector('.vnccs-cc-lora-card-' + name).getBoundingClientRect();
                result[name] = [rect.x - frame.x, rect.y - frame.y, rect.width, rect.height];
            }
            for (const name of ['.vnccs-cc-download-slot', '.vnccs-cc-lora-remove-btn', '.vnccs-cc-lora-strength-row']) {
                const element = card.querySelector(name);
                if (!element) continue;
                const rect = element.getBoundingClientRect();
                result[name] = [rect.x - frame.x, rect.y - frame.y, rect.width, rect.height];
            }
            return result;
        });` });
    let scenarios = 0;
    for (const custom of [false, true]) for (const compact of [false, true]) {
        for (const width of [420, 550, 660, 1000]) {
            await page.evaluate(options => window.mount(options), { width, custom, compact });
            const baseline = await page.evaluate(() => geometry());
            const expectedHeight = compact ? 60 : 96;
            assert.ok(baseline.every(card => Math.abs(card.frame[3] - expectedHeight) <= 1), 'Card height is fixed');
            assert.equal(baseline[0].frame[1], baseline[1].frame[1], 'The first two cards share a row');
            assert.equal(baseline[0].frame[0], baseline[2].frame[0], 'The third card starts the next row');
            await page.evaluate(options => window.mount(options), { width, custom, compact, empty: true });
            assert.deepEqual(await page.evaluate(() => geometry()), baseline, 'Absent optional text retains its reserved rectangle');
            for (const long of [false, true]) {
                await page.evaluate(options => window.mount(options), { width, custom, compact, long });
                assert.deepEqual(await page.evaluate(() => geometry()), baseline, 'Text length cannot change any card or field rectangle');
                for (const status of ['missing', 'queued', 'downloading', 'success', 'error', 'auth_required', 'installed']) {
                    await page.evaluate(status => {
                        widget.config.lora.forEach(entry => { widget.dlStatus['cc_lora_' + entry.name] = { status, progress: 98, message: 'Transfer details '.repeat(100) }; });
                        document.querySelectorAll('.vnccs-cc-lora-card').forEach(card => widget._updateDownloadCard(card, false));
                    }, status);
                    assert.deepEqual(await page.evaluate(() => geometry()), baseline, 'Download status cannot change geometry');
                    scenarios++;
                }
                assert.equal(await page.evaluate(() => [...document.querySelectorAll('.vnccs-cc-lora-card-name,.vnccs-cc-lora-card-desc')]
                    .every(element => getComputedStyle(element).whiteSpace === 'nowrap' && getComputedStyle(element).textOverflow === 'ellipsis')), true);
            }
        }
    }
    await page.evaluate(options => window.mount(options), { width: 550, compact: true });
    const still = await page.evaluate(() => geometry());
    await page.locator('.vnccs-cc-lora-card').first().hover();
    await page.waitForTimeout(250);
    assert.deepEqual(await page.evaluate(() => geometry()), still, 'Hover does not move or resize cards');
    await page.mouse.move(1190, 890);
    await page.waitForTimeout(250);
    await page.locator('#root').screenshot({ path: join(tmpdir(), 'vnccs-control-center-cards.png') });
    console.log('Verified fixed card and field rectangles across ' + scenarios + ' text, status, variant and width scenarios.');
    await writeFile(join(tmpdir(), 'vnccs-control-center-cards.json'), JSON.stringify(await page.evaluate(() => geometry()), null, 2));

    await page.addScriptTag({ content: interactions.replaceAll('export ', '') });
    await page.addScriptTag({ content: `
        const enableMiddleMouseCanvasPan = () => {}, attachHelpTooltips = () => {}, syncDOMWidgetWidthSoon = () => {};
        window.mountNode = ({ width, height, scale, status }) => {
            mount({ width, compact: true });
            const lora = widget.config.lora.map((entry, index) => ({ ...entry, status: index ? 'installed' : status }));
            document.body.innerHTML = '<div id="host"><div id="title">VNCCS Control Center</div><div id="mount"></div></div>';
            const host = document.querySelector('#host'), mountPoint = document.querySelector('#mount');
            Object.assign(host.style, { width: width + 'px', height: height + 'px', transform: 'scale(' + scale + ')',
                transformOrigin: 'top left', position: 'relative', border: '1px solid #69616d', boxSizing: 'border-box' });
            Object.assign(document.querySelector('#title').style, { height: '30px', color: 'white', font: '14px sans-serif' });
            Object.assign(mountPoint.style, { position: 'absolute', left: '10px', top: '40px', width: width - 20 + 'px', height: height - 60 + 'px' });
            Object.assign(widget, {
                node: { id: 1, widgets: [], setDirtyCanvas() {}, addDOMWidget(name, type, element) { mountPoint.appendChild(element); } },
                state: { active_kind: 'QI2', loras: [] }, dlStatus: {},
                _samplers: ['euler'], _schedulers: ['simple'],
                config: { models: [{ name: 'Qwen Image 2.1 INT8 ConvRot', kind: 'QI2', type: 'unet', status: 'installed' }],
                    clip: [{ name: 'Qwen CLIP', kind: 'QI2', status: 'installed' }],
                    vae: [{ name: 'Qwen VAE', kind: 'QI2', status: 'installed' }], lora, controlnet: [], other: [] },
            });
            widget._buildUI(); widget._renderAll();
            window.downloadCard = widget.scrollArea.querySelector('.vnccs-cc-lora-card');
            window.downloadBadge = downloadCard.querySelector('.vnccs-cc-badge');
        };
        window.nodeGeometry = () => [...document.querySelectorAll('#host, #mount, .vnccs-cc-root, .vnccs-cc-root > *, .vnccs-cc-block, [data-download-key], .vnccs-cc-download-slot, .vnccs-cc-input')].map(element => {
                const rect = element.getBoundingClientRect(); return [rect.x, rect.y, rect.width, rect.height];
            });
    ` });
    for (const [width, height, scale] of [[300, 700, 1], [550, 920, 1], [900, 1000, .75]]) {
        for (const initial of ['missing', 'outdated']) {
            // Restore the gallery mount consumed by mountNode for each fresh node fixture.
            await page.evaluate(() => { document.body.innerHTML = '<div id="root"><div id="grid"></div></div>'; });
            await page.evaluate(options => mountNode(options), { width, height, scale, status: initial });
            await page.waitForTimeout(250);
            const before = await page.evaluate(() => nodeGeometry());
            await page.locator('#host').screenshot({ path: join(tmpdir(), `vnccs-control-center-${width}-${initial}.png`) });
            const geometryFits = await page.evaluate(() => {
                const host = document.querySelector('#mount').getBoundingClientRect();
                return [...widget.container.children].every(element => {
                    const rect = element.getBoundingClientRect();
                    return rect.left >= host.left - 1 && rect.right <= host.right + 1
                        && rect.top >= host.top - 1 && rect.bottom <= host.bottom + 1;
                }) && widget.scrollArea.scrollWidth <= widget.scrollArea.clientWidth + 1;
            });
            assert.ok(geometryFits, 'Full node panels fit the host; catalog scrolls only vertically');
            for (const status of ['queued', 'downloading', 'error', 'queued', 'success', 'installed']) {
                await page.evaluate(status => {
                    const entry = widget.config.lora[0];
                    widget._downloadRefreshPending = status === 'success';
                    if (status === 'installed') widget.config.lora[0] = { ...entry, status };
                    widget.dlStatus['cc_lora_' + entry.name] = { status, progress: 37 };
                    widget._updateDownloadIndicators();
                }, status);
                assert.equal(await page.evaluate(() => downloadBadge.textContent),
                    { queued: '…', downloading: '⬇', error: '✕', success: '●', installed: '●' }[status]);
                assert.deepEqual(await page.evaluate(() => nodeGeometry()), before, 'Status transitions preserve the whole node layout');
            }
            await page.waitForFunction(() => getComputedStyle(downloadBadge).color === 'rgb(0, 214, 143)');
            assert.deepEqual(await page.evaluate(() => nodeGeometry()), before, 'Settled status keeps the node layout');
            await page.locator('#host').screenshot({ path: join(tmpdir(), `vnccs-control-center-${width}-${initial}-installed.png`) });
            assert.deepEqual(await page.evaluate(() => ({
                sameCard: downloadCard === widget.scrollArea.querySelector('.vnccs-cc-lora-card'),
                sameBadge: downloadBadge === downloadCard.querySelector('.vnccs-cc-badge'),
                text: downloadBadge.textContent, color: getComputedStyle(downloadBadge).color,
                label: downloadCard._downloadView.label.textContent,
            })), { sameCard: true, sameBadge: true, text: '●', color: 'rgb(0, 214, 143)', label: 'Pipe' });
        }
    }
    console.log('Verified full-node download completion without reload at three restored sizes and two graph scales.');
} finally {
    await browser.close();
}
