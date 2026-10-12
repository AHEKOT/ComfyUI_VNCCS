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

class Element {
    constructor() { this.children = []; this.style = {}; this.value = ''; this.classList = { add() {}, toggle() {} }; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); }
    checkValidity() { return this.valid !== false; }
}

function setup(kind, field) {
    const text = source(kind === 'creator' ? 'vnccs_character_creator_v2' : kind === 'cloner' ? 'vnccs_character_cloner' : 'vnccs_clothes_designer');
    const widget = { name: 'widget_data', value: '{}' }, node = { widgets: [widget] };
    const state = { character: 'Alice', character_info: { name: 'Alice', age: 30 }, gen_settings: { seed: 17, steps: 20, cfg: 5,
        lora_stack: Array.from({ length: 5 }, () => ({ name: 'adapter', strength: 1 })) } };
    const context = createWidgetContext({ state, node, dataWidget: widget, els: {},
        document: { createElement: () => new Element() },
        app: { graph: { setDirtyCanvas() {} } }, syncBackgroundControl() {},
        helpFor: () => '', setHelpText() {}, saveCurrentGenerationModeValues() {}, syncBackgroundForGenerationMode() {},
        window: { dispatchEvent() {} }, CustomEvent: class {}, syncPoseStudioAge() {},
    });
    const save = between(text, 'const saveState =', kind === 'creator' ? 'const clearPreviewHandlers =' : kind === 'cloner' ? 'const parsePoseStudioValues =' : 'node._randomizeSeedIfNeeded =');
    const serialize = kind === 'clothes'
        ? between(text, 'const onSerialize = node.onSerialize;', 'const pendingCostumeSaves =')
        : between(text, 'const origSerialize = node.onSerialize;', kind === 'creator' ? '// 3. State & Widget Setup' : '// Initialize');
    let control;
    if (field === 'seed') {
        control = between(text, 'const seedInp =', 'const seedMode =') + '\nglobalThis.input = seedInp;';
    } else if (field === 'age') {
        control = between(text, 'const createSlider =', kind === 'creator' ? 'const createCompactNumberField =' : '// Define renderThumbs placeholder')
            + '\ncreateSlider("Age", "age", 1, 100, 1, state.character_info); globalThis.input = els.age.num;';
    } else if (field === 'lora') {
        context.stackContainer = new Element();
        control = 'els.loraStackSelects = [];\n' + between(text, 'for (let i = 0; i < 5; i++) {', 'loraSection.appendChild(stackContainer);')
            + '\nglobalThis.input = els.loraStackSelects[0].rng; input.value = state.gen_settings.lora_stack[0].strength;';
    } else {
        control = between(text, 'const createCompactNumberField =', 'const createCompactSelectField =')
            + `\ncreateCompactNumberField("${field}", "${field}", 1, ${field === 'steps' ? 100 : 20}, ${field === 'steps' ? 1 : 0.1}); globalThis.input = els.${field}.num;`;
    }
    vm.runInContext(save + serialize + control, context);
    return { context, state, input: context.input, serialize() {
        const workflow = { widgets_values: ['old snapshot'] };
        node.onSerialize(workflow);
        return JSON.parse(workflow.widgets_values[0]);
    } };
}

for (const kind of ['creator', 'clothes']) {
    for (const [raw, valid, expected] of [['1e3', true, 1000], ['', true, 17], ['3.5', false, 17], ['Infinity', false, 17]]) {
        test(`${kind} seed '${raw}' keeps the native number in the serialized workflow`, () => {
            const h = setup(kind, 'seed');
            h.input.value = raw; h.input.valid = valid;
            h.input.onchange({ target: h.input });
            assert.equal(h.state.gen_settings.seed, expected);
            assert.equal(h.serialize().gen_settings.seed, expected);
            assert.equal(Number(h.input.value), expected);
        });
    }
}

for (const kind of ['creator', 'cloner']) {
    test(`${kind} an empty age input preserves the last authored age`, () => {
        const h = setup(kind, 'age');
        h.input.value = '';
        h.input.onchange({ target: h.input });
        assert.equal(h.state.character_info.age, 30);
        assert.equal(h.serialize().character_info.age, 30);
        assert.equal(Number(h.input.value), 30);
    });
}

for (const [field, raw, valid, expected] of [['steps', '3.5', false, 20], ['cfg', '3.25', false, 5], ['steps', '1e2', true, 100], ['cfg', '3.2', true, 3.2],
    ['steps', '101', false, 20], ['cfg', '21', false, 5]]) {
    test(`Creator ${field} '${raw}' respects the native step and numeric value`, () => {
        const h = setup('creator', field);
        assert.equal(h.input.step, field === 'steps' ? 1 : 0.1);
        h.input.value = raw; h.input.valid = valid;
        h.input.onchange({ target: h.input });
        assert.equal(h.serialize().gen_settings[field], expected);
        assert.equal(Number(h.input.value), expected);
    });
}

for (const [raw, valid, expected] of [['', true, 1], ['0.07', false, 1], ['0', true, 0], ['1e1', true, 10]]) {
    test(`Creator LoRA strength '${raw}' preserves authored settings through serialization`, () => {
        const h = setup('creator', 'lora');
        assert.equal(h.input.step, '0.05');
        h.input.value = raw; h.input.valid = valid;
        h.input.onchange({ target: h.input });
        assert.equal(h.serialize().gen_settings.lora_stack[0].strength, expected);
        assert.equal(Number(h.input.value), expected);
    });
}

for (const field of ['age', 'stack']) {
test(`Creator control refresh keeps saved ${field} LoRAs visible when inventory lacks their options`, () => {
    const h = setup('creator', 'lora');
    const selection = () => {
        let selected = '';
        return { options: [{ value: '' }],
            get value() { return selected; },
            set value(value) { selected = this.options.some(option => option.value === value) ? value : ''; },
            add(option) { this.options.push(option); }, closest() { return null; } };
    };
    h.state.gen_settings.age_lora_name = 'missing/age.safetensors';
    h.state.gen_settings.lora_stack.forEach((item, i) => { item.name = `missing/adapter${i}.safetensors`; });
    h.context.els.ageSelect = selection();
    h.context.els.loraStackSelects.forEach(ref => { ref.sel = selection(); });
    Object.assign(h.context, { Option: class { constructor(text, value) { this.text = text; this.value = value; } },
        syncSliderValue() {}, ensureLoraStack() {}, renderControlCenterCards() {} });
    const text = source('vnccs_character_creator_v2');
    vm.runInContext(between(text, 'const ensureOption =', 'const splitPromptTokens =') + '\n'
        + between(text, 'const syncGenerationControls =', 'const migrateGenerationModeSettings =') + '\nsyncGenerationControls();', h.context);
    if (field === 'age') {
        assert.equal(h.context.els.ageSelect.value, h.state.gen_settings.age_lora_name);
        assert.equal(h.serialize().gen_settings.age_lora_name, h.context.els.ageSelect.value);
    } else h.context.els.loraStackSelects.forEach((ref, i) => {
        assert.equal(ref.sel.value, h.state.gen_settings.lora_stack[i].name);
        assert.equal(h.serialize().gen_settings.lora_stack[i].name, ref.sel.value);
    });
});
}
