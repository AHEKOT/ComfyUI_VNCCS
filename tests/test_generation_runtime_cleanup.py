import types
import weakref

import pytest

torch = pytest.importorskip("torch")

from nodes import character_generator as cg
from runtime_cleanup_helpers import dynamic_runtime, install_node_calls


def test_list_mapping_releases_owned_inputs_on_failure(dynamic_runtime, monkeypatch):
    _, events, pending = dynamic_runtime
    values = [object() for _ in range(48)]

    def call(name, **kwargs):
        pending.append(object())
        raise RuntimeError('interrupted')

    install_node_calls(monkeypatch, call)
    with pytest.raises(RuntimeError, match='interrupted'):
        cg.VNCCS_CharacterGenerator()._run_list_mapped(
            'KSampler', {'latent_image': values}, consume_inputs=True,
        )
    assert values == []
    assert not pending
    assert events == ['prefetch', 'cast_buffers', 'watermarks']


@pytest.mark.parametrize('failure', [None, 'VAEEncodeTiled', 'KSampler', 'VAEDecodeTiled'])
def test_seedvr_repeated_images_clean_every_stage(dynamic_runtime, monkeypatch, failure):
    _, events, pending = dynamic_runtime
    calls = []
    image = torch.ones(1, 32, 32, 3)

    def call(name, **kwargs):
        assert not pending
        pending.append(object())
        calls.append(name)
        if name == failure:
            raise RuntimeError('interrupted')
        if name == 'SeedVR2Conditioning':
            return object(), object()
        if name in {'VAEEncodeTiled', 'KSampler'}:
            return ({'samples': torch.ones(1)},)
        return (image,)

    install_node_calls(monkeypatch, call)
    generator = cg.VNCCS_CharacterGenerator()
    if failure:
        with pytest.raises(RuntimeError, match='interrupted'):
            generator._run_seedvr_upscale_one(image, object(), object(), {}, 77)
    else:
        for _ in range(48):
            result = generator._run_seedvr_upscale_one(image, object(), object(), {}, 77)
            assert torch.equal(result, image)
        assert calls.count('KSampler') == 48
    assert not pending
    assert events == ['prefetch', 'cast_buffers', 'watermarks'] * len(calls)


@pytest.mark.parametrize('mode,turbo', [('illustrious', False), ('anima', False), ('qi2', False), ('qi2', True)])
@pytest.mark.parametrize('failure', [None, 'encode', 'sample', 'decode'])
def test_creator_direct_stages_cleanup_including_fallbacks(dynamic_runtime, monkeypatch, mode, turbo, failure):
    from _vnccs.nodes import character_creator_v2 as creator

    _, events, pending = dynamic_runtime
    settings = {'generation_mode': mode}
    latent = {'samples': torch.ones(1)}
    image = torch.ones(1, 8, 8, 3)

    def stage(name, output):
        assert not pending
        pending.append(object())
        if name == failure:
            raise RuntimeError('interrupted')
        return output

    clip = types.SimpleNamespace(
        tokenize=lambda text: text,
        encode_from_tokens=lambda *args, **kwargs: stage('encode', (torch.ones(1), torch.ones(1))),
    )
    vae = types.SimpleNamespace(
        decode=lambda *args, **kwargs: stage('decode', image),
        decode_tiled=lambda *args, **kwargs: stage('decode', image),
    )
    # Empty mappings exercise Creator's direct CLIP/VAE/common_ksampler fallbacks.
    monkeypatch.setattr(creator, 'nodes', types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        common_ksampler=lambda **kwargs: (stage('sample', latent),),
    ))
    def call(name, **kwargs):
        if name in {'KSampler', 'SamplerCustomAdvanced'}:
            return (stage('sample', latent),)
        return (object(),)
    install_node_calls(monkeypatch, call)
    monkeypatch.setattr(cg, 'viggle_turbo_sigmas', lambda latent: torch.ones(7))

    def generate():
        positive = creator.encode_generation_prompt(clip, 'prompt', settings)
        samples = creator.sample_generation_latent(
            object(), positive, [], latent, 1, 6, 1.0, 'euler', 'simple', settings, qi2_turbo=turbo,
        )
        return creator.decode_generation_samples(vae, samples, settings)

    if failure:
        with pytest.raises(RuntimeError, match='interrupted'):
            generate()
    else:
        for _ in range(4):
            assert torch.equal(generate(), image)
    assert not pending
    stage_count = {'encode': 1, 'sample': 2, 'decode': 3, None: 12}[failure]
    assert events == ['prefetch', 'cast_buffers', 'watermarks'] * stage_count


@pytest.mark.parametrize('kind,turbo', [('klein9b', False), ('qi2', False), ('qi2', True)])
def test_other_pose_families_release_resources_and_consumed_tensors(dynamic_runtime, monkeypatch, kind, turbo):
    _, events, pending = dynamic_runtime
    refs, sampled_refs = [], []
    sample_count = 0
    decode_count = 0
    generator = cg.VNCCS_CharacterGenerator()
    count = 48

    def encode(**kwargs):
        assert not pending
        if kind == "qi2":
            assert all(ref() is None for ref in refs + sampled_refs)
        pending.append(object())
        positive = torch.ones(1)
        refs.append(weakref.ref(positive))
        return positive, torch.ones(1), {'samples': torch.ones(1)}

    def sample(**kwargs):
        nonlocal sample_count
        assert not pending
        pending.append(object())
        assert sum(ref() is not None for ref in refs) == (1 if kind == "qi2" else count - sample_count)
        sample_count += 1
        samples = torch.ones(1)
        sampled_refs.append(weakref.ref(samples))
        return ({'samples': samples},)

    def decode(**kwargs):
        assert not pending
        pending.append(object())
        nonlocal decode_count
        assert sum(ref() is not None for ref in refs) == (1 if kind == "qi2" else 0)
        assert sum(ref() is not None for ref in sampled_refs) == (1 if kind == "qi2" else count - decode_count)
        decode_count += 1
        return (torch.ones(1, 8, 8, 3),)

    mappings = {}
    for name, method in {
        'VNCCS_Flux_Klein_Encoder': encode, 'ProbeEncode': encode,
        'KSampler': sample, 'SamplerCustomAdvanced': sample,
        'VAEDecode': decode, 'VAEDecodeTiled': decode,
        'RandomNoise': lambda **kwargs: (object(),),
        'BasicGuider': lambda **kwargs: (object(),),
        'KSamplerSelect': lambda **kwargs: (object(),),
    }.items():
        mappings[name] = type(name, (), {'FUNCTION': 'run', 'run': staticmethod(method)})
    monkeypatch.setattr(cg, 'comfy_nodes', types.SimpleNamespace(NODE_CLASS_MAPPINGS=mappings))
    monkeypatch.setattr(cg, 'VNCCS_MaskExtractor', lambda: types.SimpleNamespace(fill_alpha_with_color=lambda image: (image,)))
    monkeypatch.setattr(cg, 'viggle_turbo_sigmas', lambda latent: torch.ones(7))
    monkeypatch.setattr(generator, '_extract_pipe', lambda pipe: {
        'model': object(), 'clip': object(), 'vae': object(), 'model_kind': kind,
        'seed': 1, 'steps': 6, 'cfg': 1.0, 'denoise': 1.0,
        'sampler': 'euler', 'scheduler': 'simple',
    })
    monkeypatch.setattr(generator, '_apply_pose_lora_to_model', lambda model, *args: model)
    monkeypatch.setattr(generator, '_validate_conditioning_for_model', lambda *args: None)
    monkeypatch.setattr(generator, '_qi2_encode', lambda *args, **kwargs: cg._call_comfy_node('ProbeEncode'))
    monkeypatch.setattr(generator, '_qi2_prepare_model', lambda model, *args: (model, turbo))
    monkeypatch.setattr(generator, '_emit', lambda *args, **kwargs: None)
    result = generator._run_pose_generation(
        torch.ones(count, 32, 32, 3), torch.ones(1, 32, 32, 3),
        object(), 'prompt', {'target_size': 1024}, background='Green',
    )
    assert result.shape == (count, 8, 8, 3)
    assert len(events) >= count * 3 * 3
    assert not pending
    assert all(ref() is None for ref in refs + sampled_refs)
