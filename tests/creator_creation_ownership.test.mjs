import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = name => readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8');
const between = (text, start, end) => {
    const first = text.indexOf(start), last = text.indexOf(end, first);
    assert.ok(first >= 0 && last > first, start);
    return text.slice(first, last);
};
const guards = between(source('vnccs_common'), 'export function registerCleanup', '// ── DOM Widget Width Sync').replaceAll('export ', '');
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};

for (const kind of ['creator', 'cloner', 'clothes']) {
    for (const obsolete of ['current', 'before_confirmation', 'configure', 'configure_same_character', 'removal', 'close']) {
        test(`${kind} creation preserves widget ownership through ${obsolete}`, async () => {
            const state = { character: 'Alice', costume: 'Dress', character_info: { hair: 'Alice hair' }, costume_info: { top: 'Alice silk' } };
            const node = {}, requests = [], pending = deferred(), overlay = { isConnected: true, querySelector: () => ({ value: 'New' }) };
            const selector = { options: [{ value: 'Alice' }], add(option) { this.options.push(option); } };
            let actions, refreshes = 0;
            const loadCharacter = async name => { state.character = name; state.character_info = { name }; return true; };
            const context = createWidgetContext({
                node, state, btnNewCostume: {}, costSel: {}, els: { charSelect: selector },
                document: { createElement: () => ({ value: 'New', focus() {} }) },
                Option: class { constructor(text, value) { this.text = text; this.value = value; } },
                showModal(title, content, buttons) { content(); actions = buttons; return { content: {} }; },
                showAlertModal() { assert.fail('Stale creation must not report an error'); },
                loadChar: loadCharacter, setCharacter: loadCharacter,
                loadCostumes: async () => { refreshes++; return true; }, loadCostumeInfo: async () => true,
                syncCostumeEditControls() {}, updatePreviewImage() {}, saveState() {},
                api: { fetchApi(route, options) { requests.push({ route, body: JSON.parse(options.body) }); return pending.promise; } },
            });
            const text = source(kind === 'creator' ? 'vnccs_character_creator_v2' : kind === 'cloner' ? 'vnccs_character_cloner' : 'vnccs_clothes_designer');
            const creation = kind === 'clothes'
                ? between(text, 'btnNewCostume.onclick =', 'els.btnNew = btnNewCostume;')
                : between(text, 'const doCreate =', 'const doDelete =');
            vm.runInContext(`${guards}
                const beginCharacterRequest = createRequestGuard(node);
                const beginClothesUploadRequest = createRequestGuard(node);
                ${creation}
                globalThis.open = ${kind === 'clothes' ? 'btnNewCostume.onclick' : 'doCreate'};
                globalThis.configure = ${kind === 'clothes' ? 'beginClothesUploadRequest' : 'beginCharacterRequest'};`, context);
            context.open();
            if (obsolete === 'before_confirmation') {
                context.configure();
                state.character = 'Bob';
                const operation = actions[1].action(overlay, {});
                pending.resolve({ ok: true, json: async () => ({ status: 'ok' }) });
                await operation;
                assert.equal(requests.length, 0, 'An obsolete form must not write to disk');
                return;
            }
            const operation = actions[1].action(overlay, {});
            assert.equal(requests.length, 1);
            if (kind === 'clothes') {
                assert.equal(requests[0].body.character, 'Alice');
                assert.equal(requests[0].body.create_only, true);
            }
            if (obsolete.startsWith('configure')) context.configure();
            if (obsolete === 'removal') node.onRemoved();
            if (obsolete === 'close') overlay.isConnected = false;
            if (obsolete !== 'current') {
                state.character = obsolete === 'configure_same_character' ? 'Alice' : 'Bob';
                state.character_info = { hair: 'authored Bob hair' };
                state.costume = 'Coat';
                state.costume_info = { top: 'authored Bob wool' };
            }
            const original = JSON.stringify(state);
            pending.resolve({ ok: true, json: async () => ({ status: 'ok' }) });
            await operation;
            if (obsolete === 'current') {
                assert.equal(kind === 'clothes' ? state.costume : state.character, 'New');
                assert.equal(kind === 'clothes' ? refreshes : selector.options.length, kind === 'clothes' ? 1 : 2);
            } else {
                assert.equal(JSON.stringify(state), original, 'Late creation response must preserve authored state');
                assert.equal(refreshes, 0);
                assert.equal(selector.options.length, 1);
            }
        });
    }
}
