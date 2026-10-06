"""Proxy and reconnect regressions run without model dependencies or PyTorch."""

import asyncio
from types import SimpleNamespace

import pytest

from nodes import progress_state as progress
from nodes.http_state import prevent_runtime_cache, install_cache_policy
from utils import validate_privileged_request


def request(**headers):
    return SimpleNamespace(headers={key.replace('_', '-'): value for key, value in headers.items()})


def test_rewritten_host_accepts_browser_same_origin_metadata_and_marker():
    validate_privileged_request(request(Host='127.0.0.1:8188', Origin='https://comfy.example',
        Referer='https://comfy.example/comfy/', Sec_Fetch_Site='same-origin', X_VNCCS_CSRF='1'))


@pytest.mark.parametrize('origin,referer,site,marker', [
    ('http://127.0.0.1:8188', 'https://comfy.example/comfy/', '', '1'),
    ('http://127.0.0.1:8188', 'https://comfy.example/comfy/', '', None),
    ('http://127.0.0.1:8188', 'https://comfy.example/comfy/', 'same-origin', '1'),
    ('https://comfy.example', 'http://127.0.0.1:8188/', 'same-origin', '1'),
])
def test_verified_origin_accepts_rewritten_referrer_and_older_clients(origin, referer, site, marker):
    headers = dict(Host='127.0.0.1:8188', Origin=origin, Sec_Fetch_Site=site, X_VNCCS_CSRF=marker)
    validate_privileged_request(request(**headers, Referer=referer))
    validate_privileged_request(request(**headers))
    with pytest.raises(ValueError, match='cross-site privileged request rejected'):
        validate_privileged_request(request(**{**headers, 'Sec_Fetch_Site': 'cross-site'}))


def test_referer_is_checked_when_origin_is_absent():
    validate_privileged_request(request(Host='localhost:8188', Referer='http://localhost:8188/comfy/'))
    with pytest.raises(ValueError, match='cross-origin privileged request rejected'):
        validate_privileged_request(request(Host='localhost:8188', Referer='https://attacker.example/', X_VNCCS_CSRF='1'))


@pytest.mark.parametrize('origin', ['https://attacker.example', 'null', '', 'file:///tmp/comfy.html'])
def test_matching_referer_cannot_rescue_an_untrusted_or_invalid_origin(origin):
    with pytest.raises(ValueError):
        validate_privileged_request(request(Host='localhost:8188', Origin=origin,
            Referer='http://localhost:8188/', X_VNCCS_CSRF='1'))


@pytest.mark.parametrize('site', ['', 'same-site', 'cross-site'])
def test_proxy_does_not_trust_forwarded_host_or_unverified_metadata(site):
    with pytest.raises(ValueError):
        validate_privileged_request(request(Host='127.0.0.1:8188', Origin='https://attacker.example',
            X_Forwarded_Host='attacker.example', Sec_Fetch_Site=site, X_VNCCS_CSRF='1'))


@pytest.mark.parametrize('extra', [
    {}, {'X-VNCCS-CSRF': 'wrong'},
    {'X-VNCCS-CSRF': '1', 'Origin': 'null', 'Referer': 'http://localhost:8188/'},
])
def test_proxy_origin_bypass_requires_valid_origin_metadata_and_marker(extra):
    headers = {'Host': 'localhost:8188', 'Origin': 'https://comfy.example', 'Sec-Fetch-Site': 'same-origin', **extra}
    with pytest.raises(ValueError):
        validate_privileged_request(SimpleNamespace(headers=headers))


def test_mutable_routes_disable_browser_and_proxy_cache_without_touching_core_routes():
    for path in ['/vnccs/create', '/api/vnccs/config']:
        response = SimpleNamespace(headers={'Cache-Control': 'public, max-age=3600'})
        asyncio.run(prevent_runtime_cache(SimpleNamespace(path=path), response))
        assert response.headers['Cache-Control'] == 'no-store, private'
    response = SimpleNamespace(headers={})
    asyncio.run(prevent_runtime_cache(SimpleNamespace(path='/assets/app.js'), response))
    assert response.headers == {}
    server = SimpleNamespace(app=SimpleNamespace(on_response_prepare=[]))
    install_cache_policy(server); install_cache_policy(server)
    assert len(server.app.on_response_prepare) == 1


@pytest.fixture(autouse=True)
def clear_progress():
    progress._states.clear()
    yield
    progress._states.clear()


def event(stage='pose_generation', status='running', **extra):
    return {'node_id': '5', 'stage': stage, 'status': status, **extra}


def test_progress_reconstructs_append_replace_and_error_events():
    scope = progress.begin_progress('workflow:generator:5', 5)
    first = progress.record_progress(scope, event(images=['one', 'two']))
    progress.record_progress(scope, event(images=['three'], append_images=True))
    last = progress.record_progress(scope, event(images=['replacement'], replace_images=True, preview_start=1))
    snapshot = progress.progress_snapshot(scope)
    assert snapshot['revision'] == last > first
    assert snapshot['stages']['pose_generation']['images'] == ['one', 'replacement', 'three']
    snapshot['stages'].clear()
    assert progress.progress_snapshot(scope)['stages']
    progress.record_progress(scope, event(stage='error', status='error', message='Interrupted'))
    assert progress.progress_snapshot(scope)['stages']['pose_generation']['status'] == 'error'


def test_progress_isolated_between_workflows_and_resets_for_a_new_run():
    a = progress.begin_progress('a:5', 5)
    b = progress.begin_progress('b:5', 5)
    progress.record_progress(a, event(status='done'))
    assert progress.progress_snapshot(b)['stages'] == {}
    assert progress.record_progress(a, {**event(), 'node_id': 'different'}) is None
    progress.begin_progress(a, 5)
    assert progress.progress_snapshot(a)['stages'] == {}
    assert progress.begin_progress('../bad', 5) is None


def test_progress_limits_memory_and_expires_old_snapshots(monkeypatch):
    monkeypatch.setattr(progress, 'MAX_STATES', 2)
    monkeypatch.setattr(progress, 'MAX_IMAGE_CHARS', 10)
    scope = progress.begin_progress('a', 5)
    progress.record_progress(scope, event(images=['12345678']))
    progress.record_progress(scope, event(stage='bg_remove', images=['abcdefghi']))
    snapshot = progress.progress_snapshot(scope)
    assert sum(len(image) for stage in snapshot['stages'].values() for image in stage['images']) <= 10
    progress.begin_progress('b', 5); progress.begin_progress('c', 5)
    assert progress.progress_snapshot('a') is None
    monkeypatch.setattr(progress.time, 'monotonic', lambda: float('inf'))
    assert progress.progress_snapshot('c') is None


def test_superseded_run_cannot_overwrite_the_current_snapshot():
    scope = progress.begin_progress('shared', 5, 'old')
    progress.begin_progress(scope, 5, 'new')
    assert progress.record_progress(scope, event(status='done'), 'old') is None
    assert progress.progress_snapshot(scope)['stages'] == {}
    assert progress.record_progress(scope, event(status='running'), 'new')


def test_partial_regeneration_keeps_unchanged_previews_and_resets_downstream_status():
    scope = progress.begin_progress('partial', 5, 'first')
    progress.record_progress(scope, event(status='done', images=['pose-a', 'pose-b']), 'first')
    progress.record_progress(scope, event(stage='bg_remove', status='done', images=['bg-a', 'bg-b']), 'first')
    progress.begin_progress(scope, 5, 'second', 'bg_remove', True)
    progress.record_progress(scope, event(stage='bg_remove', images=['new-bg'], replace_images=True, preview_start=1), 'second')
    snapshot = progress.progress_snapshot(scope)
    assert snapshot['stages']['pose_generation']['status'] == 'done'
    assert snapshot['stages']['bg_remove']['images'] == ['bg-a', 'new-bg']
    assert snapshot['stages']['bg_remove']['status'] == 'running'


def test_regenerate_contexts_separate_workflows_with_identical_node_ids(monkeypatch):
    from nodes import generator_context as context
    from collections import OrderedDict
    monkeypatch.setattr(context, '_LIVE_GENERATOR_CONTEXTS', OrderedDict())
    key, remember, contexts = context._generator_context_key, context._remember_generator_context, context._LIVE_GENERATOR_CONTEXTS
    remember(5, 'base', 'cache-a', 'pipe-a', 'workflow-a')
    remember(5, 'base', 'cache-b', 'pipe-b', 'workflow-b')
    assert contexts[key(5, 'workflow-a')]['pipe'] == 'pipe-a'
    assert contexts[key(5, 'workflow-b')]['pipe'] == 'pipe-b'
    assert key(5, 'missing') not in contexts


def test_create_route_accepts_checked_post_and_rejects_invalid_or_cross_site_requests(monkeypatch, tmp_path):
    import importlib.util
    import inspect
    import json
    import os
    import sys
    import traceback
    import types
    from pathlib import Path
    import server
    from aiohttp import web
    import utils

    handlers = {}
    def register(method):
        def route(path):
            def decorate(callback):
                handlers[method, path] = callback
                return callback
            return decorate
        return route
    monkeypatch.setattr(server.PromptServer, 'instance', SimpleNamespace(
        routes=SimpleNamespace(get=register('GET'), post=register('POST'))))
    monkeypatch.setattr(web, 'json_response', lambda data, status=200, **kwargs: SimpleNamespace(data=data, status=status), raising=False)
    package_name = '_vnccs_proxy_routes'
    monkeypatch.setattr(utils, 'base_output_dir', lambda: str(tmp_path))
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(package_name, root / '__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, package_name + '.utils', utils)
    nodes_package = types.ModuleType(package_name + '.nodes')
    nodes_package.__path__ = [str(root / 'nodes')]
    monkeypatch.setitem(sys.modules, package_name + '.nodes', nodes_package)
    original_find_spec = importlib.util.find_spec
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *args: None if name == 'torch' else original_find_spec(name, *args))
    spec.loader.exec_module(package)

    async def invoke(data, headers):
        async def body():
            return data
        req = SimpleNamespace(method='POST', headers=headers, json=body)
        return await handlers['POST', '/vnccs/create'](req)
    headers = {'Host': 'localhost:8188', 'Origin': 'http://localhost:8188', 'Referer': 'https://comfy.example/comfy/',
               'Sec-Fetch-Site': 'same-origin', 'X-VNCCS-CSRF': '1'}
    result = asyncio.run(invoke({'name': 'Alice', 'catalog': 'creator_v2'}, headers))
    assert result.status == 200 and result.data['name'] == 'Alice'
    config = Path(utils.config_path('Alice'))
    saved = json.loads(config.read_text())
    assert saved['character_info']['hair'] == 'black hair, waist-length hair'
    assert saved['character_info']['eyes'] == 'blue eyes'
    saved['character_info']['hair'] = 'My Custom Hair'
    config.write_text(json.dumps(saved))
    assert asyncio.run(invoke({'name': 'Alice'}, headers)).data['existing'] is True
    assert json.loads(config.read_text()) == saved
    config.write_text('broken JSON')
    assert asyncio.run(invoke({'name': 'Alice'}, headers)).status == 500
    assert config.read_text() == 'broken JSON'
    with monkeypatch.context() as patch:
        patch.setattr(utils, 'save_config', lambda *args: '')
        failed = asyncio.run(invoke({'name': 'CannotSave'}, headers))
        assert failed.status == 500
    assert asyncio.run(invoke([], headers)).status == 400
    assert asyncio.run(invoke({'name': '../bad'}, headers)).status == 400
    assert asyncio.run(invoke({'name': 'Bob'}, {**headers, 'Sec-Fetch-Site': 'cross-site'})).status == 403
    assert handlers['GET', '/vnccs/create'] is handlers['POST', '/vnccs/create']
    async def legacy_get(headers):
        req = SimpleNamespace(method='GET', headers=headers, rel_url=SimpleNamespace(query={'name': 'Legacy'}))
        return await handlers['GET', '/vnccs/create'](req)
    trusted = {'Host': 'comfy.example', 'Referer': 'https://comfy.example/comfy/', 'Sec-Fetch-Site': 'same-origin'}
    assert asyncio.run(legacy_get(trusted)).status == 200
    assert json.loads(Path(utils.config_path('Legacy')).read_text())['character_info']['hair'] == 'black long hair'
    assert asyncio.run(invoke({'name': 'Cloner'}, headers)).status == 200
    assert json.loads(Path(utils.config_path('Cloner')).read_text())['character_info']['hair'] == 'black long hair'
    hostile = {**trusted, 'Origin': 'https://evil.example', 'Referer': 'https://evil.example/', 'Sec-Fetch-Site': 'same-site'}
    assert asyncio.run(legacy_get(hostile)).status == 403
    assert asyncio.run(legacy_get({**hostile, 'Sec-Fetch-Site': 'cross-site'})).status == 403


def test_disk_inputs_and_previews_are_isolated_for_identical_nodes(tmp_path):
    from pathlib import Path
    from nodes.generator_context import scoped_cache_dir
    base = tmp_path / 'Alice' / 'cache' / 'poses' / '5'
    a = Path(scoped_cache_dir(str(base), 'workflow-a:generator:5'))
    b = Path(scoped_cache_dir(str(base), 'workflow-b:generator:5'))
    for directory, value in [(a, b'input-a'), (b, b'input-b')]:
        directory.mkdir(parents=True)
        (directory / 'inputs.pt').write_bytes(value)
        (directory / 'preview.png').write_bytes(value)
    # A remains the source of regeneration after B writes the same logical files.
    for name in ['inputs.pt', 'preview.png']:
        assert (a / name).read_bytes() == b'input-a'
        assert (b / name).read_bytes() == b'input-b'
    assert scoped_cache_dir(str(base)) == str(base)
    assert Path(scoped_cache_dir(str(base), '../../outside')).is_relative_to(base)


def test_live_context_eviction_releases_pipes_and_preserves_running_callers(monkeypatch):
    from collections import OrderedDict
    import gc
    import weakref
    from nodes import generator_context as context
    monkeypatch.setattr(context, '_LIVE_GENERATOR_CONTEXTS', OrderedDict())
    monkeypatch.setattr(context, 'MAX_CONTEXTS', 2)
    now = [0]
    monkeypatch.setattr(context.time, 'monotonic', lambda: now[0])
    class Pipe:
        pass
    pipe = Pipe()
    reference = weakref.ref(pipe)
    context._remember_generator_context(5, 'base', 'a', pipe, 'a')
    active = context._get_generator_context(5, 'a')
    del pipe
    context._remember_generator_context(5, 'base', 'b', Pipe(), 'b')
    context._remember_generator_context(5, 'base', 'c', Pipe(), 'c')
    assert context._get_generator_context(5, 'a') is None
    assert reference() is active['pipe']
    del active
    gc.collect()
    assert reference() is None
    now[0] = context.TTL_SECONDS + 1
    assert context._get_generator_context(5, 'c') is None
    assert not context._LIVE_GENERATOR_CONTEXTS


def test_progress_keeps_client_request_identity_separate_from_server_run():
    scope = progress.begin_progress('identity:5', 5, 'server-run', request_id='client-request')
    snapshot = progress.progress_snapshot(scope)
    assert snapshot['request_id'] == 'client-request'
    assert snapshot['run_id'] == 'server-run'


@pytest.mark.parametrize('host,site,expected', [
    ('127.0.0.1:8188', 'same-origin', 403),
    ('localhost:8188', 'same-origin', 403),
    ('comfy.example', 'same-origin', 200),
    ('comfy.example', 'cross-site', 403),
])
def test_proxy_contract_includes_outer_comfy_origin_guard(host, site, expected):
    # Source-mock of the Host/Origin and fetch-metadata branches in ComfyUI's
    # create_origin_only_middleware. This guard runs BEFORE extension handlers.
    from urllib.parse import urlsplit
    import ipaddress
    req = request(Host=host, Origin='https://comfy.example', Sec_Fetch_Site=site, X_VNCCS_CSRF='1')
    async def guarded_handler(req):
        if req.headers['Sec-Fetch-Site'] == 'cross-site':
            return 403
        host_url = urlsplit('//' + req.headers['Host'].lower())
        origin = urlsplit(req.headers['Origin'].lower())
        try:
            loopback = ipaddress.ip_address(host_url.hostname).is_loopback
        except ValueError:
            loopback = host_url.hostname == 'localhost'
        host_domain = host_url.netloc if origin.port is not None else host_url.hostname
        origin_domain = origin.netloc if host_url.port is not None else origin.hostname
        if loopback and host_domain != origin_domain:
            return 403
        validate_privileged_request(req)
        return 200
    assert asyncio.run(guarded_handler(req)) == expected


@pytest.mark.parametrize('completed', [False, True])
def test_terminal_error_survives_without_running_stages(completed):
    scope = progress.begin_progress('failed:5', 5, 'failed-run')
    if completed:
        progress.record_progress(scope, event(status='done', images=['saved-preview']), 'failed-run')
    progress.record_progress(scope, event(stage='error', status='error', message='Cannot write output'), 'failed-run')
    snapshot = progress.progress_snapshot(scope)
    assert snapshot['error']['message'] == 'Cannot write output'
    assert snapshot['error']['stage'] == ('pose_generation' if completed else None)
    progress.begin_progress(scope, 5, 'retry')
    assert progress.progress_snapshot(scope)['error'] is None


@pytest.fixture
def generator_module(monkeypatch):
    """Load real generator logic with model imports replaced at the boundary."""
    import importlib.util
    import json
    import sys
    import types
    from pathlib import Path
    from contextlib import contextmanager
    import utils
    import server

    root = Path(__file__).parents[1]
    package = '_vnccs_recovery_test'
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod
    module(package, __path__=[str(root)])
    module(package + '.nodes', __path__=[str(root / 'nodes')])
    monkeypatch.setitem(sys.modules, package + '.utils', utils)
    tensor = module('torch', is_tensor=lambda value: False)
    tensor.nn = module('torch.nn')
    tensor.nn.functional = module('torch.nn.functional')
    prefix = package + '.nodes.'
    @contextmanager
    def inference_stage():
        yield
    from threading import RLock
    module(prefix + 'runtime_cleanup', inference_stage=inference_stage, inference_lock=RLock())
    module(prefix + 'vnccs_control_center', _apply_lora_standard=lambda *a: None,
           _find_model_on_disk=lambda *a: None, _rel_within_folder=lambda *a: None,
           _entry_kind=lambda entry: '')
    module(prefix + 'qi2_viggle', apply_viggle_turbo_lora=lambda *a: None, viggle_turbo_sigmas=lambda *a: None)
    module(prefix + 'image_processing', ChromaKeyProcessor=object, fill_alpha_with_color=object)
    spec = importlib.util.spec_from_file_location(prefix + 'character_generator', root / 'nodes/character_generator.py')
    generator = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, generator)
    spec.loader.exec_module(generator)
    return generator


@pytest.mark.parametrize('tasks,error', [
    ([], 'No emotion tasks'),
    ([{'emotion_prompt': 'Happy', 'source_path': 'missing.png'}], 'No readable source pose'),
])
def test_emotion_preparation_failure_records_terminal_progress_without_tensors(monkeypatch, generator_module, tasks, error):
    # Execute the actual process method with unavailable tensor/model operations
    # stubbed at the import boundary. These cases fail before inference begins.
    import json
    import server
    generator = generator_module
    monkeypatch.setattr(generator, '_character_cache_dir_from_sheets_path', lambda *args: None)
    monkeypatch.setattr(generator, '_load_run_inputs', lambda *args, **kwargs: {})
    monkeypatch.setattr(generator.VNCCS_EmotionsGenerator, '_source_sprite_hw', lambda *args: None)
    emitted = []
    monkeypatch.setattr(server.PromptServer.instance, 'send_sync', lambda name, payload: emitted.append(payload), raising=False)
    widget = json.dumps({'ui': {'progress_scope': 'emotion-prep:5', 'progress_request_id': 'prepare'}})
    with pytest.raises(RuntimeError, match=error):
        generator.VNCCS_EmotionsGenerator().process(None, SimpleNamespace(model_kind='qi2'), tasks, widget_data=widget, unique_id=['5'])
    snapshot = generator.progress_snapshot('emotion-prep:5')
    assert error in snapshot['error']['message']
    assert snapshot['stages'] == {}
    assert emitted[-1]['node_id'] == '5'
    assert emitted[-1]['stage'] == 'error'
    assert error in emitted[-1]['message']


@pytest.mark.parametrize('contents', ['broken JSON', '[]', '{"costumes": []}'])
def test_unreadable_configuration_is_not_replaced_by_a_costume_save(monkeypatch, tmp_path, contents):
    import utils
    from pathlib import Path
    monkeypatch.setattr(utils, 'base_output_dir', lambda: str(tmp_path))
    path = Path(utils.config_path('Alice'))
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    with pytest.raises(OSError, match='Cannot read configuration'):
        utils.save_costume_info('Alice', 'Dress', {'top': 'silk'})
    assert path.read_text() == contents
    assert utils.load_config('Missing', strict=True) is None


def test_read_permission_failure_is_not_treated_as_missing_configuration(monkeypatch, tmp_path):
    import utils
    from pathlib import Path
    monkeypatch.setattr(utils, 'base_output_dir', lambda: str(tmp_path))
    path = Path(utils.save_config('Alice', {'costumes': {'Dress': {'top': 'silk'}}}))
    original = path.read_bytes()
    actual_open = open
    def denied(name, *args, **kwargs):
        if str(name) == str(path):
            raise PermissionError('Denied')
        return actual_open(name, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(utils, 'open', denied, raising=False)
        with pytest.raises(OSError, match='Cannot read configuration'):
            utils.save_costume_info('Alice', 'New', {})
    assert path.read_bytes() == original


@pytest.mark.parametrize('failure', ['serialize', 'write', 'replace'])
def test_failed_config_save_preserves_original_file(monkeypatch, tmp_path, failure):
    import utils
    from pathlib import Path
    monkeypatch.setattr(utils, 'base_output_dir', lambda: str(tmp_path))
    path = Path(utils.save_config('Alice', {'costumes': {'Dress': {'top': 'silk'}}}))
    original = path.read_bytes()
    data = {'costumes': {'New': {}}}
    if failure == 'serialize':
        data['unsupported'] = object()
    elif failure == 'write':
        def failed_dump(data, handle, **kwargs):
            handle.write('{"partial":')
            raise OSError('Disk full')
        monkeypatch.setattr(utils.json, 'dump', failed_dump)
    else:
        def failed_replace(*args):
            raise PermissionError('Replace denied')
        monkeypatch.setattr(utils.os, 'replace', failed_replace)
    assert utils.save_config('Alice', data) == ''
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.fixture
def tensor_cache(generator_module, monkeypatch, tmp_path):
    import json
    from pathlib import Path
    generator = generator_module
    class Tensor:
        shape = (1, 2, 2, 3)
        def __init__(self, value):
            self.value = value
        def detach(self):
            return self
        def cpu(self):
            return self
    monkeypatch.setattr(generator, '_safe_cache_dir', lambda value: str(value) if value else '')
    monkeypatch.setattr(generator.torch, 'is_tensor', lambda value: isinstance(value, Tensor))
    monkeypatch.setattr(generator.torch, 'save', lambda value, path: Path(path).write_text(json.dumps(value.value)), raising=False)
    monkeypatch.setattr(generator, '_safe_torch_load_tensor', lambda path: Tensor(json.loads(Path(path).read_text())))
    return generator, Tensor, str(tmp_path)


@pytest.mark.parametrize('failure', ['tensor', 'metadata'])
def test_failed_input_cache_transaction_never_mixes_old_tensor_with_new_metadata(tensor_cache, monkeypatch, failure):
    from pathlib import Path
    generator, Tensor, cache = tensor_cache
    assert generator._save_run_inputs(cache, poses=Tensor('old-pose'), prompt='old-prompt')
    original = (Path(cache) / '_stage_cache' / 'inputs.json').read_bytes()
    if failure == 'tensor':
        def failed_save(value, path):
            Path(path).write_text('partial')
            raise OSError('Disk full')
        monkeypatch.setattr(generator.torch, 'save', failed_save)
    else:
        def failed_dump(value, handle, **kwargs):
            handle.write('partial')
            raise OSError('Disk full')
        monkeypatch.setattr(generator.json, 'dump', failed_dump)
    with pytest.raises(OSError, match='Disk full'):
        generator._save_run_inputs(cache, poses=Tensor('new-pose'), prompt='new-prompt')
    restored = generator._load_run_inputs(cache)
    assert restored['poses'].value == 'old-pose'
    assert restored['prompt'] == 'old-prompt'
    assert (Path(cache) / '_stage_cache' / 'inputs.json').read_bytes() == original
    assert len(list((Path(cache) / '_stage_cache').iterdir())) == 2


def test_required_tensor_loss_invalidates_the_entire_input_bundle(tensor_cache):
    from pathlib import Path
    generator, Tensor, cache = tensor_cache
    generator._save_run_inputs(cache, poses=Tensor('pose'), prompt='prompt')
    next((Path(cache) / '_stage_cache').glob('*.pt')).unlink()
    assert generator._load_run_inputs(cache) == {}


@pytest.mark.parametrize('node_type', ['VNCCS_CharacterGenerator', 'VNCCS_CharacterCloneGenerator', 'VNCCS_ClothesGenerator'])
def test_cache_failure_drops_live_regeneration_context_and_records_error(tensor_cache, monkeypatch, node_type):
    import json
    generator, Tensor, cache = tensor_cache
    node = getattr(generator, node_type)()
    monkeypatch.setattr(generator, '_character_cache_dir_from_sheets_path', lambda *args: cache)
    monkeypatch.setattr(generator, '_rotate_preview_cache', lambda *args: None)
    monkeypatch.setattr(node, '_list_to_batch', lambda value: value)
    def failed_save(*args):
        raise OSError('Disk full')
    monkeypatch.setattr(generator.torch, 'save', failed_save)
    scope = 'cache-failure:5'
    with pytest.raises(OSError, match='Disk full'):
        node.process(Tensor('pose'), Tensor('character'), object(), 'prompt', unique_id=['5'],
                     widget_data=[json.dumps({'ui': {'progress_scope': scope}})])
    assert generator._get_generator_context('5', scope) is None
    assert 'Disk full' in generator.progress_snapshot(scope)['error']['message']
    monkeypatch.setattr(generator.web, 'json_response', lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    result = generator._regenerate_response({'unique_id': '5', 'stage': 'upscaler', 'widget_data': {'ui': {'progress_scope': scope}}})
    assert result.status == 409


def test_generator_execution_lock_serializes_same_scope_and_releases_on_error():
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from nodes import generator_context as context
    started = threading.Event()
    entered = threading.Event()
    class Node:
        @context.serialized_generator
        def process(self, widget_data, unique_id):
            entered.set()
            with context.generator_execution_lock('5', 'same:5'):
                raise OSError('Released')
    def run():
        started.set()
        return Node().process([json.dumps({'ui': {'progress_scope': 'same:5'}})], ['5'])
    with ThreadPoolExecutor(max_workers=1) as pool:
        with context.generator_execution_lock('5', 'same:5'):
            future = pool.submit(run)
            assert started.wait(2)
            assert not entered.wait(.05)
            with context.generator_execution_lock('5', 'different:5'):
                pass
        with pytest.raises(OSError, match='Released'):
            future.result(timeout=2)
    assert entered.is_set()
    assert not context._execution_locks


@pytest.fixture
def creative_modules(generator_module, monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    import utils
    prefix = generator_module.__package__ + '.'
    from nodes import qwen_vl
    monkeypatch.setitem(sys.modules, prefix + 'qwen_vl', qwen_vl)
    helpers = qwen_vl
    for name, value in {'_ensure_qwen_vl_assets': lambda *a: None, '_find_qwen_vl_model': lambda *a: None,
                        'QWEN_VL_MODEL_FILENAME': 'unused-model'}.items():
        monkeypatch.setattr(helpers, name, value, raising=False)
    monkeypatch.setattr(utils, 'base_output_dir', lambda: str(tmp_path))
    modules = {}
    for name in ('character_creator_v2', 'clothes_designer'):
        spec = importlib.util.spec_from_file_location(prefix + name, Path(__file__).parents[1] / 'nodes' / (name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        modules[name] = mod
    from aiohttp import web
    monkeypatch.setattr(web, 'json_response', lambda data, status=200, **kw: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(web, 'Response', lambda status=200, text='': SimpleNamespace(status=status, text=text), raising=False)
    return SimpleNamespace(creator=modules['character_creator_v2'], clothes=modules['clothes_designer'], utils=utils)


def test_creator_and_clothes_routes_report_failed_configuration_saves(creative_modules, monkeypatch):
    import json
    m = creative_modules
    monkeypatch.setattr(m.creator, 'save_config', lambda *args: '')
    with pytest.raises(OSError, match='Could not save character configuration'):
        m.creator.CharacterCreatorV2().process(json.dumps({'character': 'Alice'}))
    monkeypatch.setattr(m.clothes, 'save_costume_info', lambda *args: False)
    async def payload():
        return {'character': 'Alice', 'costume': 'Dress', 'info': {'top': 'silk'}}
    response = asyncio.run(m.clothes.vnccs_save_costume(SimpleNamespace(json=payload, headers={'X-VNCCS-CSRF': '1'})))
    assert response.status == 500
    assert 'Could not save costume' in response.data['error']


@pytest.mark.parametrize('payload,status', [
    ({'character': 'Alice', 'costume': 'Dress'}, 200),
    ({'character': 'Alice', 'costume': 'Missing'}, 404),
    ({'character': 'Alice', 'costume': 'Naked'}, 400),
    ({'character': 'Alice', 'costume': 'Original'}, 400),
    ({'character': '../Alice', 'costume': 'Dress'}, 400),
    ({'character': 'Alice', 'costume': '../Dress'}, 400),
    ({'character': 'Alice'}, 400), ({'costume': 'Dress'}, 400),
    ({'character': [], 'costume': 'Dress'}, 400), ([], 400),
])
def test_costume_delete_route_validates_requests(creative_modules, payload, status):
    m = creative_modules
    m.utils.save_costume_info('Alice', 'Dress', {'top': 'silk'})
    async def body():
        return payload
    response = asyncio.run(m.clothes.vnccs_delete_costume(SimpleNamespace(
        json=body, headers={'X-VNCCS-CSRF': '1'},
    )))
    assert response.status == status
    assert ('Dress' in m.utils.list_costumes('Alice')) == (status != 200)


@pytest.mark.parametrize('headers', [{}, {'X-VNCCS-CSRF': '1', 'Sec-Fetch-Site': 'cross-site'}])
def test_costume_delete_route_rejects_untrusted_requests_before_reading_body(creative_modules, headers):
    async def body():
        pytest.fail('Rejected request must not read its body')
    response = asyncio.run(creative_modules.clothes.vnccs_delete_costume(SimpleNamespace(json=body, headers=headers)))
    assert response.status == 403


def test_costume_delete_route_reports_storage_failure(creative_modules, monkeypatch):
    def denied(*args):
        raise PermissionError('Delete denied')
    monkeypatch.setattr(creative_modules.clothes, 'delete_costume', denied)
    async def body():
        return {'character': 'Alice', 'costume': 'Dress'}
    response = asyncio.run(creative_modules.clothes.vnccs_delete_costume(SimpleNamespace(
        json=body, headers={'X-VNCCS-CSRF': '1'},
    )))
    assert response.status == 500
    assert response.data['error'] == 'Delete denied'


def test_costume_delete_route_reports_cleanup_warning_as_success(creative_modules, monkeypatch):
    monkeypatch.setattr(creative_modules.clothes, 'delete_costume', lambda *args: 'Cleanup pending')
    async def body():
        return {'character': 'Alice', 'costume': 'Dress'}
    response = asyncio.run(creative_modules.clothes.vnccs_delete_costume(SimpleNamespace(
        json=body, headers={'X-VNCCS-CSRF': '1'},
    )))
    assert response.status == 200
    assert response.data == {'status': 'ok', 'warning': 'Cleanup pending'}


@pytest.mark.parametrize('failure', ['image', 'metadata'])
def test_clothes_cache_cannot_reuse_partial_image_metadata_pair(creative_modules, monkeypatch, tmp_path, failure):
    from PIL import Image
    m = creative_modules
    image_path = tmp_path / 'preview.png'
    info_path = tmp_path / 'preview.json'
    old = Image.new('RGB', (2, 2), 'red')
    m.clothes._save_preview_cache(old, image_path, info_path, {'hash': 'old'})
    original = image_path.read_bytes()
    if failure == 'image':
        def fail_save(self, path, **kwargs):
            from pathlib import Path
            Path(path).write_bytes(b'partial')
            raise OSError('Disk full')
        monkeypatch.setattr(Image.Image, 'save', fail_save)
    else:
        def fail_dump(data, handle, **kwargs):
            handle.write('partial')
            raise OSError('Disk full')
        monkeypatch.setattr(m.clothes.json, 'dump', fail_dump)
    with pytest.raises(OSError, match='Disk full'):
        m.clothes._save_preview_cache(Image.new('RGB', (2, 2), 'blue'), image_path, info_path, {'hash': 'new'})
    assert not info_path.exists()
    if failure == 'image':
        assert image_path.read_bytes() == original
    else:
        with Image.open(image_path) as image:
            assert image.getpixel((0, 0)) == (0, 0, 255)
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('route', ['preview', 'workflow'])
def test_creator_preview_cache_failure_reports_error_and_preserves_old_image(creative_modules, monkeypatch, tmp_path, route):
    import json
    import numpy as np
    from contextlib import nullcontext
    from pathlib import Path
    from PIL import Image
    creator = creative_modules.creator
    path = Path(creator.character_dir('Alice')) / 'cache' / 'preview.png'
    path.parent.mkdir(parents=True)
    Image.new('RGB', (2, 2), 'red').save(path)
    original = path.read_bytes()
    monkeypatch.setattr(creator.torch, 'inference_mode', nullcontext, raising=False)
    monkeypatch.setattr(creator, 'acquire_preview_assets', lambda *a: ('model', 'clip', 'vae'))
    monkeypatch.setattr(creator, 'load_generation_assets', lambda *a: ('key', 'model', 'clip', 'vae'))
    monkeypatch.setattr(creator, 'get_lora_full_path', lambda *a: None)
    monkeypatch.setattr(creator, 'encode_generation_conditioning', lambda *a, **k: ('pos', 'neg', 'prompt'))
    monkeypatch.setattr(creator, 'create_generation_latent', lambda *a: 'latent')
    monkeypatch.setattr(creator, 'sample_generation_latent', lambda **k: 'sampled')
    decoded = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.ones((1, 2, 2, 3))))
    monkeypatch.setattr(creator, 'decode_generation_samples', lambda *a: decoded)
    def fail_save(self, destination, **kwargs):
        Path(destination).write_bytes(b'partial')
        raise OSError('Preview disk full')
    monkeypatch.setattr(Image.Image, 'save', fail_save)
    data = {'character': 'Alice', 'preview_valid': False}
    if route == 'preview':
        result = creator._generate_preview_response(data)
        assert result.status == 500
        assert 'Preview disk full' in result.text
    else:
        with pytest.raises(RuntimeError, match='saving the preview cache.*Preview disk full'):
            creator.CharacterCreatorV2().process(json.dumps(data))
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.fixture
def empty_contexts(monkeypatch):
    from collections import OrderedDict
    from nodes import generator_context as context
    monkeypatch.setattr(context, '_LIVE_GENERATOR_CONTEXTS', OrderedDict())
    monkeypatch.setattr(context, '_execution_locks', {})
    return context


def test_disk_retention_bounds_workflows_without_touching_final_outputs(empty_contexts, tmp_path):
    import sys
    from pathlib import Path
    context = empty_contexts
    snapshots = sys.modules[context.expire_cache_progress.__module__]
    base = tmp_path / 'Alice' / 'cache' / 'poses' / '5'
    final = tmp_path / 'Alice' / 'Sprites' / 'result.png'
    final.parent.mkdir(parents=True)
    final.write_bytes(b'final character image')
    for index in range(20):
        scope = f'workflow-{index}:5'
        snapshots.begin_progress(scope, '5')
        snapshots.record_progress(scope, event(status='done', images=['preview']))
        directory = context.scoped_cache_dir(str(base), scope)
        with context.generator_execution_lock('5', scope):
            context._remember_generator_context('5', 'base', directory, object(), scope)
            (Path(directory) / 'tensor.pt').write_bytes(b'cached tensor')
    assert len(list((base / 'workflows').iterdir())) == context.MAX_DISK_CACHES
    assert len(context._LIVE_GENERATOR_CONTEXTS) == context.MAX_CONTEXTS
    assert context._get_generator_context('5', 'workflow-0:5') is None
    assert context._get_generator_context('5', 'workflow-19:5')
    expired = snapshots.progress_snapshot('workflow-0:5')
    assert expired['stages'] == {}
    assert 'expired' in expired['error']['message']
    assert final.read_bytes() == b'final character image'


def test_disk_expiry_after_restart_preserves_live_contexts_legacy_files_and_symlinks(empty_contexts, tmp_path):
    import os
    from pathlib import Path
    context = empty_contexts
    base = tmp_path / 'Alice' / 'cache' / 'poses' / '5'
    old = Path(context.scoped_cache_dir(str(base), 'old:5'))
    live = Path(context.scoped_cache_dir(str(base), 'live:5'))
    old.mkdir(parents=True)
    os.utime(old, (0, 0))
    legacy = base / 'legacy.pt'
    legacy.write_bytes(b'legacy')
    outside = tmp_path / 'final-output'
    outside.mkdir()
    (outside / 'keep.png').write_bytes(b'keep')
    (old.parent / ('f' * 64)).symlink_to(outside, target_is_directory=True)
    context._remember_generator_context('5', 'base', str(live), object(), 'live:5')
    assert not old.exists()
    os.utime(live, (0, 0))
    current = context.scoped_cache_dir(str(base), 'current:5')
    context._remember_generator_context('5', 'base', current, object(), 'current:5')
    assert live.exists(), 'Usable live Regenerate contexts must retain their disk cache'
    assert legacy.read_bytes() == b'legacy'
    assert (outside / 'keep.png').read_bytes() == b'keep'
    assert (old.parent / ('f' * 64)).is_symlink()


def test_running_executions_protect_disk_cache_until_they_complete(empty_contexts, monkeypatch, tmp_path):
    from pathlib import Path
    context = empty_contexts
    monkeypatch.setattr(context, 'MAX_CONTEXTS', 1)
    monkeypatch.setattr(context, 'MAX_DISK_CACHES', 1)
    base = tmp_path / 'Alice' / 'cache' / 'poses' / '5'
    a = context.scoped_cache_dir(str(base), 'a:5')
    b = context.scoped_cache_dir(str(base), 'b:5')
    with context.generator_execution_lock('5', 'a:5'):
        context._remember_generator_context('5', 'base', a, object(), 'a:5')
        with context.generator_execution_lock('5', 'b:5'):
            context._remember_generator_context('5', 'base', b, object(), 'b:5')
            assert Path(a).is_dir() and Path(b).is_dir()
        assert Path(a).is_dir()
    assert Path(a).is_dir()
    assert not Path(b).exists()
    assert not context._execution_locks


def test_regenerate_reads_retained_inputs_and_rejects_evicted_workflows(tensor_cache, monkeypatch):
    import sys
    from pathlib import Path
    from collections import OrderedDict
    from contextlib import nullcontext
    generator, Tensor, root = tensor_cache
    context = sys.modules[generator.__package__ + '.generator_context']
    monkeypatch.setattr(context, '_LIVE_GENERATOR_CONTEXTS', OrderedDict())
    monkeypatch.setattr(context, '_execution_locks', {})
    monkeypatch.setattr(context, 'MAX_CONTEXTS', 2)
    monkeypatch.setattr(context, 'MAX_DISK_CACHES', 2)
    base = Path(root) / 'Alice' / 'cache' / 'poses' / '5'
    for index in range(3):
        scope = f'workflow-{index}:5'
        cache = context.scoped_cache_dir(str(base), scope)
        with context.generator_execution_lock('5', scope):
            generator._remember_generator_context('5', 'VNCCS_CharacterGenerator', cache, SimpleNamespace(seed=1), scope)
            generator._save_run_inputs(cache, poses=Tensor(index), prompt=f'prompt-{index}')
    seen = []
    def process(self, poses, character, pipe, prompt, **kwargs):
        seen.append((poses.value, prompt))
    monkeypatch.setattr(generator.VNCCS_CharacterGenerator, 'process', process)
    monkeypatch.setattr(generator.torch, 'inference_mode', nullcontext, raising=False)
    monkeypatch.setattr(generator.web, 'json_response', lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    def regenerate(scope):
        return generator._regenerate_response({'unique_id': '5', 'stage': 'pose_generation', 'widget_data': {'ui': {'progress_scope': scope}}})
    assert regenerate('workflow-0:5').status == 409
    assert regenerate('workflow-2:5').status == 200
    assert seen == [(2, 'prompt-2')]


def test_regenerate_lock_cannot_resurrect_an_expired_live_context(empty_contexts, monkeypatch):
    context = empty_contexts
    now = [0]
    monkeypatch.setattr(context.time, 'monotonic', lambda: now[0])
    context._remember_generator_context('5', 'base', None, object(), 'expired:5')
    now[0] = context.TTL_SECONDS + 1
    with context.generator_execution_lock('5', 'expired:5'):
        assert context._get_generator_context('5', 'expired:5') is None
