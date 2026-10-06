import { createWidgetContext } from './widget_context.mjs';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../web/vnccs_control_center.js', import.meta.url), 'utf8');

function setup(version = '3.2.0', versions = [{ version: '3.2.3' }], info = {}, utils = { error: 'not_found' }) {
    const requests = [];
    const context = createWidgetContext({
        AbortSignal,
        api: { async fetchApi(route) {
            requests.push(route);
            if (route === '/vnccs/module_status') return { ok: true, json: async () => ({
                main: { version, ...info }, utils, dependencies: {},
            }) };
            if (versions instanceof Error) throw versions;
            return { ok: versions !== null, json: async () => versions };
        } },
    });
    vm.runInContext(source.slice(source.indexOf('class VNCCSControlCenterWidget'))
        + '\nthis.Widget = VNCCSControlCenterWidget;', context);
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        _pillMain: {}, _pillUtils: {},
        _beginModuleStatusRequest: () => () => true,
        _resumePendingDependencyInstalls: async () => false,
        _showMissingDependenciesModal() {},
        _showUpdateBanner(lines) { this.banner = Array.from(lines); },
    });
    return { widget, requests };
}

test('an older installed module shows the newer Registry release and update banner', async () => {
    const { widget, requests } = setup('3.2.0', [
        { version: '3.2.3' }, { version: '3.2.10' }, { version: '3.2.0' }, { version: 'nightly' },
    ]);
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--update');
    assert.match(widget._pillMain.textContent, /3\.2\.0.*3\.2\.10/);
    assert.match(widget._pillMain.title, /3\.2\.10/);
    assert.equal(widget.banner.length, 1);
    assert.match(widget.banner[0], /3\.2\.0.*3\.2\.10/);
    assert.deepEqual(requests, ['/vnccs/module_status', '/customnode/versions/vnccs']);
});

test('a removed widget ignores late status responses', async () => {
    const { widget } = setup();
    widget._beginModuleStatusRequest = () => () => false;
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, undefined);
    assert.equal(widget.banner, undefined);
});

test('green requires a successful comparison with published releases', async () => {
    for (const version of ['3.2.3', '3.2.4']) {
        const { widget } = setup(version);
        await widget._fetchModuleStatus();
        assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--ok');
        assert.deepEqual(widget.banner, []);
    }
    for (const versions of [null, [], {}, [{ version: 'nightly' }], new Error('offline')]) {
        const { widget } = setup('3.2.0', versions);
        await widget._fetchModuleStatus();
        assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--warning');
        assert.match(widget._pillMain.textContent, /update check unavailable/);
        assert.deepEqual(widget.banner, []);
    }
});

test('Utils is compared with its own Manager package ID', async () => {
    const { widget, requests } = setup('3.2.3', [{ version: '3.2.3' }], {}, { version: '3.1.0' });
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--ok');
    assert.equal(widget._pillUtils.className, 'vnccs-cc-pill vnccs-cc-pill--update');
    assert.match(widget.banner[0], /Utils v3\.1\.0.*3\.2\.3/);
    assert.deepEqual(requests, ['/vnccs/module_status', '/customnode/versions/vnccs', '/customnode/versions/vnccs-utils']);
});

test('missing and duplicate modules retain their existing status without a release lookup', async () => {
    for (const [info, state] of [[{ error: 'not_found' }, 'error'],
        [{ duplicate: true, duplicate_folders: ['vnccs', 'ComfyUI_VNCCS'] }, 'dup']]) {
        const { widget, requests } = setup('3.2.0', null, info);
        await widget._fetchModuleStatus();
        assert.equal(widget._pillMain.className, `vnccs-cc-pill vnccs-cc-pill--${state}`);
        assert.deepEqual(requests, ['/vnccs/module_status']);
    }
});
