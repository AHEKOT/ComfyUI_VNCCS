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

for (const [kind, operation] of [['cloner', 'syncPoseStudioAge'], ['cloner', 'syncPoseStudioGender'], ['clothes', 'applyPoseStudioValues']]) {
    for (const lifecycle of ['mounted', 'removed', 'corrupt']) {
        test(`${kind} delayed ${operation} respects the ${lifecycle} node`, () => {
            const node = { type: kind === 'cloner' ? 'CharacterCloner' : 'ClothesDesigner', widgets: [] };
            const state = { character_info: { age: 30, sex: 'male' } }, timers = [], updates = [];
            const context = createWidgetContext({
                node, state, app: { graph: { _nodes: [node] } },
                setTimeout(callback) { timers.push(callback); },
                window: {},
            });
            const text = source(kind === 'cloner' ? 'vnccs_character_cloner' : 'vnccs_clothes_designer');
            vm.runInContext(guards + '\n' + between(text, 'const parsePoseStudioValues =',
                kind === 'cloner' ? 'const loadState =' : 'const beginCharacterInfoRequest =') + `\nglobalThis.sync = ${operation};`, context);
            context.sync();
            assert.equal(timers.length, 1, 'Unavailable integration schedules one retry');
            if (lifecycle === 'removed') node.onRemoved?.();
            if (lifecycle === 'corrupt') node._vnccsStateRestoreError = 'Saved state is corrupt';
            context.window.__vnccsPoseStudioCharacterCreatorSync = {
                _vnccsClonerPatched: true, _vnccsClothesDesignerPatched: true,
                sourceSignatures: new Map(), hookClonerNode() {}, hookClothesDesignerNode() {},
                studios: [{ applyExternalCharacterCreatorValues(values) { updates.push(values); return true; } }],
            };
            timers.shift()();
            assert.equal(updates.length, lifecycle === 'mounted' ? 1 : 0);
            if (lifecycle === 'mounted') {
                assert.equal(updates[0].age, 30);
                assert.equal(updates[0].gender, 1);
            }
            assert.equal(timers.length, 0, 'The retry does not outlive the node');
        });
    }
    test(`${kind} removed ${operation} stops retrying when the integration remains unavailable`, () => {
        const node = {}, timers = [];
        const context = createWidgetContext({ node, state: { character_info: { age: 30, sex: 'male' } },
            window: {}, setTimeout(callback) { timers.push(callback); } });
        const text = source(kind === 'cloner' ? 'vnccs_character_cloner' : 'vnccs_clothes_designer');
        vm.runInContext(guards + '\n' + between(text, 'const parsePoseStudioValues =',
            kind === 'cloner' ? 'const loadState =' : 'const beginCharacterInfoRequest =') + `\nglobalThis.sync = ${operation};`, context);
        context.sync();
        node.onRemoved?.();
        timers.shift()();
        assert.equal(timers.length, 0);
    });
}
