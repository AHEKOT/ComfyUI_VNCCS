import { createWidgetContext } from './widget_context.mjs';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../web/vnccs_control_center.js', import.meta.url), 'utf8');

function setup(version = '3.2.0', versions = [{ version: '3.2.3' }], info = {}, utils = { error: 'not_found' }, runtime, dependencies = {}) {
    const requests = [];
    const context = createWidgetContext({
        AbortSignal,
        api: { async fetchApi(route) {
            requests.push(route);
            if (route === '/vnccs/module_status') return { ok: true, json: async () => ({
                main: { version, ...info }, utils, dependencies, comfyui: runtime,
            }) };
            if (versions instanceof Error) throw versions;
            return { ok: versions !== null, json: async () => versions };
        } },
    });
    const constants = source.slice(source.indexOf('const INITIAL_NODE_W'), source.indexOf('function _injectVNCCSControlCenterStyles'));
    vm.runInContext(constants + source.slice(source.indexOf('class VNCCSControlCenterWidget'))
        + '\nthis.Widget = VNCCSControlCenterWidget;', context);
    const widget = Object.create(context.Widget.prototype);
    Object.assign(widget, {
        _pillMain: {}, _pillUtils: {},
        _dependencyInstallTasks: new Map(),
        state: { active_kind: 'QI2' },
        _beginModuleStatusRequest: () => () => true,
        _resumePendingDependencyInstalls: async () => false,
        _showMissingDependenciesModal() {},
        statusStrip: { appendChild() { assert.fail("Module updates belong in the existing pill row"); } },
    });
    return { widget, requests, api: context.api };
}

test('runtime response warns for selected H3 and ignores stale or unrelated selections', async () => {
    for (const [supported, selected, stale, expected] of [
        [false, 'MiniMaxH3', false, 1], [true, 'MiniMaxH3', false, 0],
        [false, 'QI2', false, 0], [false, 'MiniMaxH3', true, 0],
    ]) {
        const { widget } = setup('3.2.3', [], {}, {}, { h3_still_supported: supported });
        widget.state.active_kind = selected;
        let warnings = 0;
        widget._showH3UpdateModal = () => warnings++;
        if (stale) widget._beginModuleStatusRequest = () => () => false;
        await widget._fetchModuleStatus();
        assert.equal(warnings, expected);
    }
    const { widget } = setup('3.2.3', [], {}, {}, { h3_still_supported: false });
    widget.state.active_kind = 'MiniMaxH3';
    widget._showH3UpdateModal = () => assert.fail('A late response must use the current family');
    const pending = widget._fetchModuleStatus();
    widget.state.active_kind = 'QI2';
    await pending;
});

test('an older installed module shows the newer Registry release only in its pill', async () => {
    const { widget, requests } = setup('3.2.0', [
        { version: '3.2.3' }, { version: '3.2.10' }, { version: '3.2.0' }, { version: 'nightly' },
    ]);
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--update');
    assert.equal(widget._pillMain.textContent, 'VNCCS v3.2.0 is outdated');
    assert.match(widget._pillMain.title, /3\.2\.10/);
    assert.deepEqual(requests, ['/vnccs/module_status', '/vnccs/module_versions/vnccs']);
});

test('VNCCS 3.3.1 shows newer downloadable versions without requiring Registry status', async () => {
    const versions = [{ version: '3.2.4' }, { version: '3.3.2' }];
    for (const [list, version] of [
        [versions, '3.3.2'],
        [[...versions, { version: '3.3.3' }], '3.3.3'],
    ]) {
        const { widget } = setup('3.3.1', list);
        await widget._fetchModuleStatus();
        assert.equal(widget._pillMain.textContent, 'VNCCS v3.3.1 is outdated');
        assert.equal(widget._pillMain.title, 'Update to v' + version);
    }
});

test('a removed widget ignores late status responses', async () => {
    const { widget } = setup();
    widget._beginModuleStatusRequest = () => () => false;
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, undefined);
});

test('green requires a successful comparison with published releases', async () => {
    const { widget } = setup('3.2.3');
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--ok');
    const ahead = setup('3.3.1', [{ version: '3.2.4' }]).widget;
    await ahead._fetchModuleStatus();
    assert.equal(ahead._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--warning');
    assert.match(ahead._pillMain.title, /3\.3\.1.*3\.2\.4/);
    for (const versions of [null, [], {}, [{ version: 'nightly' }], new Error('offline')]) {
        const { widget } = setup('3.2.0', versions);
        await widget._fetchModuleStatus();
        assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--warning');
        assert.match(widget._pillMain.textContent, /update check unavailable/);
    }
});

test('Utils is compared with its own Manager package ID', async () => {
    const { widget, requests } = setup('3.2.3', [{ version: '3.2.3' }], {}, { version: '3.1.0' });
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.className, 'vnccs-cc-pill vnccs-cc-pill--ok');
    assert.equal(widget._pillUtils.className, 'vnccs-cc-pill vnccs-cc-pill--update');
    assert.equal(widget._pillUtils.textContent, 'Utils v3.1.0 is outdated');
    assert.deepEqual(requests, ['/vnccs/module_status', '/vnccs/module_versions/vnccs', '/vnccs/module_versions/vnccs-utils']);
});

test('dependency warnings leave the current module pill green', async () => {
    const { widget } = setup('3.3.1', [{ version: '3.3.1' }], {}, undefined, undefined, {
        pack: { status: 'warning' }, other: { status: 'unsupported' },
    });
    widget._ensureDependencyPill = () => ({});
    await widget._fetchModuleStatus();
    assert.equal(widget._pillMain.textContent, 'VNCCS v3.3.1 ●');
});

for (const [mode, id] of [['legacy', 'vnccs'], ['modern', 'vnccs-utils']]) {
    test(`outdated ${id} queues the displayed version once through ${mode} Manager`, async () => {
        const { widget, api } = setup('3.3.1', [{ version: '3.3.3' }], {}, { version: '0.7.1' });
        const fetch = api.fetchApi;
        const bodies = [];
        const starts = [];
        let releasePolicy;
        const policy = new Promise(resolve => { releasePolicy = resolve; });
        api.fetchApi = async (route, options) => {
            if (route === '/vnccs/manager/install_policy') return policy;
            if (route === '/manager/queue/install' && mode === 'modern') return { ok: false, status: 404 };
            if (route === '/manager/queue/install' || route === '/v2/manager/queue/task') {
                bodies.push(JSON.parse(options.body));
                return { ok: true };
            }
            if (route.endsWith('/queue/start')) { starts.push(route); return { ok: true }; }
            return fetch(route, options);
        };
        await widget._fetchModuleStatus();
        const button = id === 'vnccs' ? widget._pillMain : widget._pillUtils;
        assert.equal(button.disabled, false);
        const clicked = button.onclick();
        assert.equal(button.disabled, true);
        assert.equal(await button.onclick(), false);
        await widget._fetchModuleStatus();
        assert.equal(button.disabled, true);
        releasePolicy({ ok: true, json: async () => ({ install_allowed: true }) });
        assert.equal(await clicked, true);
        assert.equal(bodies.length, 1);
        const params = mode === 'modern' ? bodies[0].params : bodies[0];
        if (mode === 'modern') assert.equal(bodies[0].kind, 'install');
        assert.equal(params.id, id);
        assert.equal(params.version, '3.3.3');
        assert.equal(params.selected_version, '3.3.3');
        assert.deepEqual(starts, [mode === 'modern' ? '/v2/manager/queue/start' : '/manager/queue/start']);
        assert.equal(button.textContent, 'Updating…');
        if (mode === 'modern') widget._handleManagerTaskCompleted({ detail: { ui_id: bodies[0].ui_id, result: 'success' } });
        else widget._handleManagerQueueStatus({ detail: { status: 'done', nodepack_result: { [params.ui_id]: 'success' } } });
        await widget._fetchModuleStatus();
        assert.equal(button.textContent, 'Restart required');
        assert.equal(button.disabled, true);
    });
}

test('a rejected update can be retried and a current module cannot queue an update', async () => {
    const versions = [{ version: '3.3.3' }];
    const { widget } = setup('3.3.1', versions);
    await widget._fetchModuleStatus();
    const button = widget._pillMain;
    widget._ensureManagerInstallPolicy = async () => false;
    widget._queueDependencyInstall = () => assert.fail('Policy must be checked first');
    assert.equal(await button.onclick(), false);
    assert.equal(button.disabled, false);
    widget._ensureManagerInstallPolicy = async () => true;
    widget._queueDependencyInstall = async () => { throw Error('Rejected by Manager'); };
    widget.showMessage = message => assert.match(message, /Rejected by Manager/);
    assert.equal(await button.onclick(), false);
    assert.equal(button.disabled, false);
    assert.equal(button.textContent, 'VNCCS v3.3.1 is outdated');
    versions[0].version = '3.3.1';
    await widget._fetchModuleStatus();
    assert.equal(button.disabled, true);
    assert.equal(button.onclick, null);
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
