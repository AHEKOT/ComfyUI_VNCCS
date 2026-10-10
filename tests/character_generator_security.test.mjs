import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = readFileSync(new URL('../web/vnccs_character_generator.js', import.meta.url), 'utf8');

class Element {
    constructor() { this.children = []; }
    appendChild(child) { this.children.push(child); return child; }
    set innerHTML(html) {
        assert.doesNotMatch(html, /<(?:img|svg)\b|\bon(?:error|load)\s*=/i,
            'Untrusted character markup must never reach innerHTML');
        this.html = html;
        this.children = [];
    }
    querySelector(selector) {
        assert.match(this.html, new RegExp(`class="[^"]*\\b${selector.slice(1)}\\b`));
        return this.character ||= new Element();
    }
}

async function setup() {
    const app = {
        graph: { links: {}, setDirtyCanvas() {} },
        registerExtension(extension) { this.extension = extension; },
    };
    const context = createWidgetContext({
        app, document: { createElement: () => new Element() },
        syncDOMWidgetWidthSoon() {},
    });
    vm.runInContext(source.replace(/^import .*;\n/gm, '') + '\nthis.Widget = CharacterGeneratorWidget;', context);
    class Node {}
    await app.extension.beforeRegisterNodeDef(Node, { name: 'VNCCS_EmotionsGenerator' });
    const state = { name: 'widget_data', value: '{}' };
    const node = Object.assign(new Node(), { id: 17, widgets: [state], inputs: [] });
    const widget = Object.assign(Object.create(context.Widget.prototype), {
        node, isEmotions: true, title: 'VNCCS Emotions Generator',
        settingsEl: new Element(), stages: [], stageState: {},
        syncStagesFromData() {}, syncModelResolution() {},
        storageKey() { return null; },
        renderPreview() {}, renderChain() {},
        block() { return new Element(); }, field() { return new Element(); },
        faceDenoiseSlider() { return new Element(); },
        faceDetailerNumberField() { return new Element(); },
        bgRemoveFields() { return []; },
    });
    node._vnccsCharacterGeneratorWidget = widget;
    return { app, node, widget, state,
        character() { return widget.settingsEl.children[1].querySelector('.vnccs-pipe-character'); },
    };
}

for (const [label, name] of [
    ['image event handler', '<img src="data:image/png,invalid" onerror="document.documentElement.dataset.mbe2e=\'xss\'">'],
    ['SVG event handler', '<svg onload="document.documentElement.dataset.mbe2e=\'xss\'"></svg>'],
    ['markup and entities', '<b>Alice</b> &amp; < > " \''],
    ['plain Unicode name', 'Alice 雪'],
]) {
    test(`emotion workflow renders ${label} as literal text and preserves it on save/reload`, async () => {
        const h = await setup();
        const pairs = [{ costume: 'Original', emotion: 'happy' }];
        h.state.value = JSON.stringify({ character_name: name, emotion_pairs: pairs });
        h.node.onConfigure();
        assert.equal(h.character().textContent, name);
        assert.deepEqual(h.character().children, []);
        assert.match(h.widget.settingsEl.children[1].html, /1 costume \/ emotion pair\(s\)/);
        const workflow = { widgets_values: ['{}'] };
        h.node.onSerialize(workflow);
        const saved = JSON.parse(workflow.widgets_values[0]);
        assert.equal(saved.character_name, name);
        assert.deepEqual(saved.emotion_pairs, pairs);
        h.state.value = workflow.widgets_values[0];
        h.node.onConfigure();
        assert.equal(h.character().textContent, name);
    });
}

test('missing and empty character names keep the Emotion Studio placeholder', async () => {
    const h = await setup();
    for (const data of [{}, { character_name: '' }]) {
        h.state.value = JSON.stringify(data);
        h.node.onConfigure();
        assert.equal(h.character().textContent, 'Select in Emotion Studio');
    }
});

test('connected Emotion Studio names also render as literal text', async () => {
    const h = await setup();
    const name = '<img src="data:image/png,invalid" onerror="document.documentElement.dataset.mbe2e=\'xss\'">';
    const studio = { type: 'EmotionGeneratorV2', widgets: [{ name: 'character', value: name }] };
    h.node.inputs = [{ name: 'pipe', link: 42 }];
    h.app.graph.links[42] = { origin_id: 5 };
    h.app.graph.getNodeById = () => studio;
    h.node.onConfigure();
    assert.equal(h.character().textContent, name);
    assert.deepEqual(h.character().children, []);
    assert.equal(JSON.parse(h.state.value).character_name, name);
});
