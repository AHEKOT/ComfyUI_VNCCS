"""Behavioral regressions for the active VNCCS workflows (mocked model calls)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from PIL import Image
from conftest import _preload_node


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Reuse the public module when other test files have already imported it.
    # Loading a second alias would make their monkeypatches target the wrong copy.
    cc = sys.modules.get('nodes.vnccs_control_center')
    if cc is not None:
        monkeypatch.setitem(sys.modules, '_vnccs.nodes.vnccs_control_center', cc)
        monkeypatch.setattr(sys.modules['_vnccs.nodes'], 'vnccs_control_center', cc, raising=False)
    utils = sys.modules['_vnccs.utils']
    monkeypatch.setattr(utils, 'base_output_dir', lambda: str(tmp_path / 'Characters'))
    cloner = _preload_node('character_cloner')
    creator = _preload_node('character_creator_v2')
    clothes = _preload_node('clothes_designer')
    generator = sys.modules['_vnccs.nodes.character_generator']
    monkeypatch.setattr(creator.server.PromptServer.instance, 'send_sync', lambda *args: None, raising=False)
    inputs = tmp_path / 'input'
    inputs.mkdir()
    Image.new('RGB', (16, 32), 'red').save(inputs / 'ref.png')
    monkeypatch.setattr(cloner.folder_paths, 'get_input_directory', lambda: str(inputs), raising=False)
    return SimpleNamespace(utils=utils, cloner=cloner, creator=creator, clothes=clothes, generator=generator, inputs=inputs)


def test_cloner_preserves_costumes_and_unedited_character_metadata(runtime):
    runtime.utils.save_config('Alice', {'character_info': {'hair': 'old', 'custom': 'keep'},
                                     'costumes': {'Dress': {'top': 'silk'}}, 'extra': {'keep': True}})
    runtime.cloner.CharacterCloner().process(json.dumps({
        'character': 'Alice', 'character_info': {'hair': 'new'}, 'source_images': ['ref.png'],
    }))
    saved = runtime.utils.load_config('Alice')
    assert saved['costumes'] == {'Dress': {'top': 'silk'}}
    assert saved['extra'] == {'keep': True}
    assert saved['character_info'] == {'name': 'Alice', 'hair': 'new', 'custom': 'keep'}


def test_cloner_refuses_to_replace_unreadable_config(runtime):
    runtime.utils.save_config('Alice', {})
    path = runtime.utils.config_path('Alice')
    with open(path, 'w') as handle:
        handle.write('broken JSON')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        runtime.cloner.CharacterCloner().process(json.dumps({'character': 'Alice', 'source_images': ['ref.png']}))
    with open(path) as handle:
        assert handle.read() == 'broken JSON'


@pytest.mark.parametrize('source', ['gen', 'pose'])
def test_creator_reuses_saved_preview_without_loading_models(runtime, monkeypatch, source):
    creator = runtime.creator
    cache = Path(runtime.utils.character_dir('Alice')) / 'cache'
    cache.mkdir(parents=True)
    Image.new('RGB', (16, 32), 'red').save(cache / 'preview.png')
    monkeypatch.setattr(creator, 'get_pose_preview', lambda *a, **k: (torch.ones(1, 32, 16, 3), 0, 1))
    def forbidden(*args):
        pytest.fail('A valid saved preview must not load generation models')
    monkeypatch.setattr(creator, 'load_generation_assets', forbidden)
    image, _, _ = creator.CharacterCreatorV2().process(json.dumps({
        'character': 'Alice', 'preview_valid': True, 'preview_source': source,
    }))
    assert image.shape == (1, 32, 16, 3)
    assert image.any()


@pytest.mark.parametrize('failed_stage', ['loading generation models', 'sampling the character image', 'decoding the character image'])
def test_creator_errors_include_stage_and_traceback(runtime, monkeypatch, capsys, failed_stage):
    creator = runtime.creator
    monkeypatch.setattr(creator, 'load_generation_assets', lambda *a: ('key', object(), object(), object()))
    monkeypatch.setattr(creator, 'encode_generation_conditioning', lambda *a, **k: ('pos', 'neg', 'prompt'))
    monkeypatch.setattr(creator, 'create_generation_latent', lambda *a, **k: {})
    monkeypatch.setattr(creator, 'sample_generation_latent', lambda **k: {})
    target = {'loading generation models': 'load_generation_assets',
              'sampling the character image': 'sample_generation_latent',
              'decoding the character image': 'decode_generation_samples'}[failed_stage]
    def fail(*args, **kwargs):
        raise RuntimeError('simulated device error')
    monkeypatch.setattr(creator, target, fail)
    with pytest.raises(RuntimeError, match=failed_stage) as caught:
        creator.CharacterCreatorV2().process(json.dumps({'character': 'Alice', 'preview_valid': False}), unique_id='42')
    assert isinstance(caught.value.__cause__, RuntimeError)
    logged = capsys.readouterr()
    assert 'Alice' in logged.out and 'node 42' in logged.out
    assert 'simulated device error' in logged.out
    assert 'Traceback' in logged.err


def test_clothes_cache_and_comfy_signature_follow_primary_reference(runtime, tmp_path, monkeypatch):
    cd = runtime.clothes
    source = runtime.inputs / 'ref.png'
    monkeypatch.setattr(cd, 'get_latest_sprite_path', lambda *a: str(source))
    monkeypatch.setattr(cd, '_resolve_pipe_clothes_core_lora', lambda pipe: None)
    calls = []
    def call_node(name, **kwargs):
        calls.append(name)
        if name == cd.KLEIN_ENCODER_CLASS:
            return 'pos', 'neg', {}
        if name == 'KSampler':
            return ({},)
        if name.startswith('VAEDecode'):
            return (torch.ones(1, 8, 8, 3),)
        raise AssertionError(name)
    monkeypatch.setattr(cd, '_call_comfy_node', call_node)
    pipe = SimpleNamespace(model=object(), clip=object(), vae=object(),
                           model_entry={'kind': 'Klein9b'}, model_cache_key={'model': 'test'})
    state = json.dumps({'character': 'Alice', 'costume': 'Dress'})
    first_signature = cd.ClothesDesigner.IS_CHANGED(state)
    node = cd.ClothesDesigner()
    node.process(pipe, state)
    node.process(pipe, state)
    assert calls.count('KSampler') == 1
    Image.new('RGB', (16, 32), 'blue').save(source)
    assert cd.ClothesDesigner.IS_CHANGED(state) != first_signature
    node.process(pipe, state)
    assert calls.count('KSampler') == 2


@pytest.mark.parametrize('node_name', ['VNCCS_CharacterGenerator', 'VNCCS_CharacterCloneGenerator', 'VNCCS_ClothesGenerator'])
@pytest.mark.parametrize('regenerate', [False, True])
def test_pose_prompt_lists_survive_caching_and_single_pose_regeneration(runtime, monkeypatch, node_name, regenerate):
    cg = runtime.generator
    node = getattr(cg, node_name)()
    poses = torch.zeros(2, 8, 8, 3)
    character = torch.zeros(1, 8, 8, 3)
    prompts = ['front light', 'back light']
    seen, cached = [], []
    class ReachedEncoder(Exception):
        pass
    def inspect(*args, **kwargs):
        seen.append(args[3])
        raise ReachedEncoder()
    monkeypatch.setattr(cg, '_character_cache_dir_from_sheets_path', lambda *a: None)
    monkeypatch.setattr(cg, '_save_run_inputs', lambda *a, **k: cached.append(k))
    monkeypatch.setattr(cg, '_load_run_inputs', lambda *a, **k: {'poses': poses, 'character': character, 'prompt': prompts})
    monkeypatch.setattr(node, '_find_pose_lora', lambda *a: None)
    monkeypatch.setattr(node, '_run_pose_generation', inspect)
    monkeypatch.setattr(node, '_emit', lambda *a, **k: None)
    monkeypatch.setattr(node, '_extract_pipe', lambda *a: {'seed': 0})
    monkeypatch.setattr(node, '_run_source_upscaler', lambda image, *a, **k: image)
    state = {'regenerate_from': 'pose_generation', 'regenerate_index': 1} if regenerate else {}
    with pytest.raises(ReachedEncoder):
        node.process(poses, character, SimpleNamespace(), prompts, widget_data=json.dumps(state))
    assert cached[0]['prompt'] == prompts
    assert seen == ['back light' if regenerate else prompts]


def test_emotion_regeneration_preserves_supplied_shifted_seed(runtime, monkeypatch):
    cg = runtime.generator
    captured = []
    old = [{'seed': 42}]
    shifted = cg._shift_emotion_data_seeds(old, 100)
    class ReachedTasks(Exception):
        pass
    def inspect(data):
        captured.extend(data)
        raise ReachedTasks()
    monkeypatch.setattr(cg, '_load_run_inputs', lambda *a, **k: {'emotion_data': old})
    monkeypatch.setattr(cg.VNCCS_EmotionsGenerator, '_parse_emotion_data', lambda self, data: inspect(data))
    with pytest.raises(ReachedTasks):
        cg.VNCCS_EmotionsGenerator().process([], SimpleNamespace(), shifted, widget_data='{"regenerate_from":"emotion_generation"}')
    assert captured[0]['seed'] == 142


@pytest.mark.parametrize('kind', ['qi2', 'klein9b', 'minimaxh3'])
def test_each_pose_encoder_receives_its_own_prompt(runtime, monkeypatch, kind):
    cg = runtime.generator
    node = cg.VNCCS_CharacterGenerator()
    values = {'model': object(), 'clip': object(), 'vae': object(), 'audio_vae': object(),
              'model_kind': kind, 'seed': 1, 'steps': 4, 'cfg': 1, 'sampler': 'euler', 'scheduler': 'simple'}
    monkeypatch.setattr(node, '_extract_pipe', lambda *a: values)
    monkeypatch.setattr(node, '_apply_pose_lora_to_model', lambda model, *a: model)
    monkeypatch.setattr(node, '_emit', lambda *a, **k: None)
    monkeypatch.setattr(cg, 'VNCCS_MaskExtractor', lambda: SimpleNamespace(fill_alpha_with_color=lambda image: (image,)))
    seen = []
    class EncodingVerified(Exception):
        pass
    def capture(name, **kwargs):
        if name in {'TextEncodeQwenImage21', 'VNCCS_Flux_Klein_Encoder', 'MiniMaxH3ReferenceToVideo'}:
            seen.append(kwargs['prompt'])
            if len(seen) == 2:
                raise EncodingVerified()
            if name == 'MiniMaxH3ReferenceToVideo':
                return 'positive', 'latent'
            return 'positive', 'negative', 'latent'
        if name == 'ImageScaleToTotalPixels':
            return (kwargs['image'],)
        if name == 'VAEDecode':
            return (torch.ones(1, 32, 32, 3),)
        return ('result',)
    monkeypatch.setattr(cg, '_call_comfy_node', capture)
    with pytest.raises(EncodingVerified):
        node._run_pose_generation(torch.ones(2, 32, 32, 3), torch.ones(1, 32, 32, 3),
                                  SimpleNamespace(), ['front light', 'back light'], {})
    assert 'front light' in seen[0] and 'back light' not in seen[0]
    assert 'back light' in seen[1] and 'front light' not in seen[1]


def test_custom_preview_reuses_available_pipe_and_never_builds_missing_inputs(runtime, monkeypatch):
    cc = sys.modules['_vnccs.nodes.vnccs_control_center']
    state = {'selected_type': 'custom', 'active_kind': 'Klein9b'}
    pipe = cc.VNCCSPipeProxy(object(), object(), object())
    monkeypatch.setattr(cc, '_build_control_center_pipe', lambda *a, **k: pipe)
    cc.VNCCS_ControlCenter().execute('catalog', state, model=object(), clip=object(), vae=object(), unique_id='test-preview')
    def forbidden(*args, **kwargs):
        pytest.fail('Standalone custom preview must not build a workflow or load missing inputs')
    monkeypatch.setattr(cc, '_build_control_center_pipe', forbidden)
    monkeypatch.setattr(cc.web, 'Response', lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(cc.web, 'json_response', lambda data, **kwargs: data, raising=False)
    used = []
    def preview(self, **kwargs):
        used.append(kwargs['pipe'])
        assert torch.is_inference_mode_enabled()
        return (torch.ones(1, 8, 8, 3),)
    monkeypatch.setattr(runtime.clothes.ClothesDesigner, 'process', preview)
    payload = {'repo_id': 'catalog', 'node_state': state, 'control_center_id': 'test-preview'}
    assert 'image' in cc._clothes_preview_response(payload)
    assert used == [pipe]
    payload['node_state'] = {**state, 'seed': 99}
    assert cc._clothes_preview_response(payload)['status'] == 409
    payload['control_center_id'] = 'missing'
    assert cc._clothes_preview_response(payload)['status'] == 409
    assert used == [pipe]
