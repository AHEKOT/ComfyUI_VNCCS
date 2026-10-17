// Run with Playwright on NODE_PATH; optionally pass a Chromium executable as argv[2].
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const { chromium } = createRequire(import.meta.url)('playwright');
const source = await readFile(new URL('../../web/vnccs_control_center.js', import.meta.url), 'utf8');
const common = await readFile(new URL('../../web/vnccs_common.js', import.meta.url), 'utf8');
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
} finally {
    await browser.close();
}
