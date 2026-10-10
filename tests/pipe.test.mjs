import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function setup(options = ['none']) {
    const listeners = new Map();
    const hooks = [];
    let extension;
    const source = readFileSync(new URL('../web/vnccs_pipe.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '');
    vm.runInNewContext(source, {
        app: { registerExtension: value => { extension = value; } },
        window: {
            addEventListener: (name, callback) => listeners.set(name, callback),
            removeEventListener: name => listeners.delete(name),
        },
    });
    const node = {
        comfyClass: 'VNCCS_Pipe',
        widgets: [
            { name: 'lora_name', value: 'none', options: { values: options } },
            { name: 'lora_options_json', value: '["none"]' },
        ],
        setDirtyCanvas() {},
        onConfigure() { hooks.push('configure'); },
        onSerialize() { hooks.push('serialize'); },
        onRemoved() { hooks.push('removed'); },
    };
    extension.nodeCreated(node);
    return { node, hooks, listeners, update: options => listeners.get('vnccs-lora-options-updated')({ detail: { options } }) };
}

test('new Pipe preserves installed LoRA options and saves them for reload', () => {
    const { node } = setup(['none', 'installed.safetensors']);
    assert.deepEqual(node.widgets[0].options.values, ['none', 'installed.safetensors']);
    assert.equal(node.widgets[1].value, '["none"]');
    const output = { widgets_values: ['none', '["none"]'] };
    node.onSerialize(output);
    assert.deepEqual(JSON.parse(output.widgets_values[1]), ['none', 'installed.safetensors']);
    const reloaded = setup().node;
    reloaded.widgets[1].value = output.widgets_values[1];
    reloaded.onConfigure({});
    assert.deepEqual(Array.from(reloaded.widgets[0].options.values), ['none', 'installed.safetensors']);
});

test('Pipe restores workflow LoRA options after widget values without overwriting construction state', () => {
    const { node, hooks, update } = setup();
    update(['new.safetensors']);
    assert.equal(node.widgets[1].value, '["none"]');
    node.widgets[0].value = 'saved.safetensors';
    node.widgets[1].value = '["none","saved.safetensors"]';
    node.onConfigure({});
    update(['new.safetensors']);
    assert.equal(node.widgets[0].value, 'saved.safetensors');
    assert.ok(node.widgets[0].options.values.includes('saved.safetensors'));
    const output = { widgets_values: ['saved.safetensors', 'stale'] };
    node.onSerialize(output);
    assert.deepEqual(JSON.parse(output.widgets_values[1]), ['none', 'new.safetensors', 'saved.safetensors']);
    assert.deepEqual(hooks, ['configure', 'serialize']);
});

test('Pipe preserves the selected LoRA on corrupt cache and removes its listener', () => {
    const { node, hooks, listeners } = setup();
    node.widgets[0].value = 'saved.safetensors';
    node.widgets[1].value = 'broken JSON';
    node.onConfigure({});
    assert.equal(node.widgets[0].value, 'saved.safetensors');
    assert.equal(node.widgets[1].value, 'broken JSON');
    node.onRemoved();
    assert.equal(listeners.size, 0);
    assert.deepEqual(hooks, ['configure', 'removed']);
});
