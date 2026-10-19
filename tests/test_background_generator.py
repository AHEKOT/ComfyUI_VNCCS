"""Exercise the real background orchestration without PyTorch or model weights."""
import ast
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from conftest import _preload_node

backgrounds = _preload_node("background_creator")
from _vnccs.operation_logger import log_event


class Tensor(np.ndarray):
    def detach(self):
        return self

    def cpu(self):
        return self


def tensor(value=0, shape=(1, 8, 16, 3)):
    return np.full(shape, value, dtype=np.float32).view(Tensor)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    tree = ast.parse((Path(__file__).parents[1] / "nodes" / "character_generator.py").read_text())
    original = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "VNCCS_CharacterGenerator")
    methods = {"_unwrap_scalar", "_widget_data", "_regenerate_from", "_regenerate_index", "_stage_index",
               "_should_regenerate_stage", "_sampler_settings", "_encoder_call", "_is_qi2_pipe", "_is_klein_pipe", "_is_h3_pipe"}
    base = copy.deepcopy(original)
    base.body = [item for item in base.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    node_class = copy.deepcopy(next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "VNCCS_BackgroundGenerator"))
    for item in node_class.body:
        if isinstance(item, ast.FunctionDef):
            item.decorator_list = [decorator for decorator in item.decorator_list if isinstance(decorator, ast.Name) and decorator.id == "classmethod"]
    cache, events, upscales, encodes, comfy_calls, inputs = {}, [], [], [], [], {}
    settings = {"upscaler": {"mode": "off"}, "pose_generation": {"target_size": 1024}, "pose_sampler": {}, "vae_decode": {}}
    values = {"model": "selected-model", "clip": "selected-clip", "vae": "selected-vae", "audio_vae": "audio-vae",
              "model_kind": "qi2", "seed": 42, "steps": 25, "cfg": 1, "sampler": "euler", "scheduler": "simple"}
    def call(name, **kwargs):
        comfy_calls.append((name, kwargs))
        if name in {"VNCCS_Flux_Klein_Encoder", "MiniMaxH3ReferenceToVideo"}:
            return "positive", "negative", "latent" if name.startswith("VNCCS") else "unused"
        return (tensor(.4),)
    namespace = {"__name__": "_vnccs.nodes.background_generator_contract", "__package__": "_vnccs.nodes",
        "log_event": log_event, "torch": SimpleNamespace(is_tensor=lambda item: isinstance(item, Tensor)), "json": json,
        "DEFAULT_WIDGET_DATA": {}, "_as_bool": lambda value, default=True: default if value is None else bool(value), "_entry_kind": lambda entry: entry.get("kind", "") if entry else "",
        "_call_comfy_node": call, "_resolution_scale_megapixels": lambda target: target / 1024,
        "_safe_cache_part": lambda value: str(value), "scoped_cache_dir": lambda path, scope: path,
        "_remember_generator_context": lambda *args: None, "_rotate_preview_cache": lambda path: None,
        "_save_run_inputs": lambda path, **kwargs: inputs.update(kwargs),
        "_save_cached_tensor": lambda path, key, value: cache.__setitem__(key, value),
        "_load_cached_tensor": lambda path, key: cache.get(key), "_upscaler_mode": lambda value: value.get("mode", "seedvr"),
        "require_h3_still_support": lambda: None, "H3_FRAME_COUNT": 5}
    module = ast.fix_missing_locations(ast.Module(body=[base, node_class], type_ignores=[]))
    exec(compile(module, "background_generator_contract", "exec"), namespace)
    node = namespace["VNCCS_BackgroundGenerator"]()
    node._settings = lambda raw: settings
    node._begin_progress = lambda *args: setattr(node, "_progress_scope", None)
    node._emit = lambda *args, **kwargs: events.append((args, kwargs))
    node._extract_pipe = lambda pipe: values
    node._image_aspect = lambda image: image.shape[2] / image.shape[1]
    node._save_stage = lambda path, stage, image: cache.__setitem__(stage, image)
    node._tensor_item_to_pil = lambda image: Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8"))
    def upscale(image, config, seed, unique_id=None, cache_dir=None, stage="source_upscaler"):
        upscales.append((stage, image, config, seed))
        return image + .1
    node._run_source_upscaler = upscale
    def encode(values, prompt, images, target_size):
        encodes.append((values, prompt, images, target_size))
        return "positive", "negative", "latent"
    node._qi2_encode = encode
    node._qi2_prepare_model = lambda model, *args: (model, False)
    node._qi2_sample = lambda *args, **kwargs: tensor(.3)
    node._qi2_decode = lambda image, vae: image
    node._vae_decode_settings = lambda value: value
    node._validate_conditioning_for_model = lambda *args: None
    monkeypatch.setattr(backgrounds, "root", lambda: tmp_path / "VNCCS" / "scenes")
    document = backgrounds.create_background("Bedroom", backgrounds.normalize_editor({}))
    document = backgrounds.add_image(document["background_id"], Image.new("RGB", (16, 8)), {"source": "upload"}, document["revision"])
    prompts = backgrounds.variant_output(document, document["editor"], document["selected_image_id"])
    return SimpleNamespace(node=node, prompts=prompts, values=values, settings=settings, cache=cache, events=events,
                           upscales=upscales, encodes=encodes, calls=comfy_calls, document=document, inputs=inputs)


@pytest.mark.parametrize("mode", ["off", "seedvr"])
def test_three_stages_share_original_and_never_upscale_it_twice(pipeline, mode):
    p = pipeline
    p.settings["upscaler"]["mode"] = mode
    source = tensor()
    result, original, generated = p.node.process([object()], [source], p.prompts, ["{}"], ["node-7"])
    assert len(result) == len(generated) == 4
    assert len(p.upscales) == (1 if mode == "off" else 4)
    assert p.upscales[0][0] == "source_upscaler" and p.upscales[0][2]["mode"] == "seedvr"
    assert result[0] is original and generated[0] is original
    assert len(p.encodes) == 3 and all(call[2] == (original, None, None) for call in p.encodes)
    assert all(call[0]["model"] == "selected-model" for call in p.encodes)
    assert [call[1].split(".")[0] for call in p.encodes] == [f"Change time of the day to {name}" for name in ["morning", "evening", "night"]]
    assert all(call[1] is not original for call in p.upscales[1:])
    saved = backgrounds.load_background(p.document["background_id"])
    assert len(saved["generated_sets"][0]["images"]) == 4
    assert saved["generator_settings"]["sampler"]["seed"] == 42
    assert saved["editor"] == p.document["editor"]
    assert p.node.RETURN_NAMES[0] == "backgrounds" and p.node.OUTPUT_IS_LIST == (True, False, True)
    assert set(p.node.INPUT_TYPES()["required"]) == {"pipe", "background", "prompt", "widget_data"}


def test_regenerating_one_variant_uses_caches_and_preserves_the_others(pipeline):
    p = pipeline
    first = p.node.process(object(), tensor(), p.prompts, unique_id="node-7")[0]
    p.encodes.clear(); p.upscales.clear()
    result = p.node.process(object(), tensor(), p.prompts, json.dumps({"regenerate_from": "background_generation", "regenerate_index": 2}), "node-7")[0]
    assert not p.upscales and len(p.encodes) == 1
    assert "to evening." in p.encodes[0][1]
    for index in [0, 1, 3]:
        assert result[index] is first[index]
    assert result[2] is not first[2]
    assert len(backgrounds.load_background(p.document["background_id"])["generated_sets"]) == 2


def test_final_upscale_regeneration_does_not_rerun_source_or_model(pipeline):
    p = pipeline
    p.node.process(object(), tensor(), p.prompts, unique_id="node-7")
    p.upscales.clear(); p.encodes.clear(); p.settings["upscaler"]["mode"] = "seedvr"
    p.node.process(object(), tensor(), p.prompts, json.dumps({"regenerate_from": "upscaler"}), "node-7")
    assert not p.encodes
    assert len(p.upscales) == 3 and all(call[0] == "upscaler" for call in p.upscales)


def test_model_failure_keeps_existing_json_and_published_images(pipeline):
    p = pipeline
    p.node.process(object(), tensor(), p.prompts, unique_id="node-7")
    before = backgrounds.load_background(p.document["background_id"])
    p.node._qi2_sample = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Sampling failed"))
    with pytest.raises(RuntimeError, match="Sampling failed"):
        p.node.process(object(), tensor(), p.prompts, unique_id="node-7")
    assert backgrounds.load_background(p.document["background_id"]) == before
    assert p.events[-1][0] == ("node-7", "error", "error")


def test_stale_regeneration_and_image_batches_fail_before_upscaling(pipeline):
    p = pipeline
    for image, prompts, settings in [(tensor(shape=(2, 8, 16, 3)), p.prompts, {}),
        (tensor(), p.prompts, {"regenerate_from": "background_generation"}),
        (tensor(), p.prompts, {"regenerate_from": "missing"}),
        (tensor(), p.prompts, {"regenerate_from": "upscaler", "regenerate_index": 8})]:
        with pytest.raises(ValueError):
            p.node.process(object(), image, prompts, json.dumps(settings), "node-7")
    assert not p.upscales


@pytest.mark.parametrize("bad", [[], ["not JSON"], ["{}"], ["[]"], ["{}"] * 65, [123]])
def test_prompt_contract_is_a_bounded_comfy_list(pipeline, bad):
    with pytest.raises((ValueError, TypeError)):
        pipeline.node._background_tasks(bad)


def test_empty_prompt_list_upscales_and_publishes_only_original_without_restoring_cards(pipeline):
    p = pipeline
    editor = copy.deepcopy(p.document['editor']); editor['variants'] = []
    document = backgrounds.save_background(p.document['background_id'], editor, p.document['revision'])
    payload = {'background_id': document['background_id'], 'background_source_image_id': document['selected_image_id'], 'background_variants': []}
    result, original, generated = p.node.process([object()], [tensor()], [], [json.dumps(payload)], ['node-7'])
    assert len(result) == len(generated) == 1 and result[0] is generated[0] is original
    assert len(p.upscales) == 1 and not p.encodes
    saved = backgrounds.load_background(document['background_id'])
    assert saved['editor']['variants'] == []
    assert saved['generated_sets'][0]['variants'] == []
    assert [image['name'] for image in saved['generated_sets'][0]['images']] == ['Original']
    assert p.inputs['prompt'] == []
    p.upscales.clear()
    p.node.process(object(), tensor(), [], json.dumps({**payload, 'regenerate_from': 'upscaler'}), 'node-7')
    assert not p.upscales and not p.encodes


def test_empty_prompt_list_rejects_stale_or_missing_source_before_upscaling(pipeline):
    p = pipeline
    payload = {'background_id': p.document['background_id'], 'background_source_image_id': p.document['selected_image_id']}
    with pytest.raises(ValueError, match='variants changed'):
        p.node.process(object(), tensor(), [], json.dumps(payload), 'node-7')
    with pytest.raises(ValueError):
        p.node.process(object(), tensor(), [], json.dumps({**payload, 'background_id': '../bad'}), 'node-7')
    with pytest.raises(ValueError):
        p.node.process(object(), tensor(), [], json.dumps({**payload, 'background_source_image_id': '../bad', 'regenerate_from': 'upscaler'}), 'node-7')
    assert not p.upscales


def test_empty_prompt_list_uses_source_created_after_workflow_serialization(pipeline):
    p = pipeline
    editor = copy.deepcopy(p.document['editor']); editor['variants'] = []
    document = backgrounds.save_background(p.document['background_id'], editor, p.document['revision'])
    payload = {'background_id': document['background_id'], 'background_source_image_id': document['selected_image_id']}
    document = backgrounds.add_image(document['background_id'], Image.new('RGB', (16, 8)), {'source': 'generated'}, document['revision'])
    p.node.process(object(), tensor(), [], json.dumps(payload), 'node-7')
    saved = backgrounds.load_background(document['background_id'])
    assert saved['generated_sets'][0]['source_image_id'] == document['selected_image_id']
    assert p.inputs['widget_payload']['background_source_image_id'] == document['selected_image_id']


def test_klein_receives_only_first_image_and_actual_pipe_assets(pipeline):
    p = pipeline; p.values["model_kind"] = "klein9b"
    p.node.process(object(), tensor(), p.prompts, unique_id="node-7")
    calls = [kwargs for name, kwargs in p.calls if name == "VNCCS_Flux_Klein_Encoder"]
    assert len(calls) == 3
    for call in calls:
        assert call["clip"] == "selected-clip" and call["vae"] == "selected-vae"
        assert call["image1"] is p.cache["source_upscaler"]
        assert call["image2"] is call["image3"] is None
    assert all(kwargs["model"] == "selected-model" for name, kwargs in p.calls if name == "KSampler")


def test_h3_receives_background_as_its_only_reference(pipeline):
    p = pipeline; p.values["model_kind"] = "minimaxh3"
    previous = p.node._background_variant.__globals__["_call_comfy_node"]
    def call(name, **kwargs):
        if name == 'MiniMaxH3ReferenceToVideo':
            p.calls.append((name, kwargs))
            return 'positive', 'latent'
        return previous(name, **kwargs)
    p.node._background_variant.__globals__["_call_comfy_node"] = call
    p.node._resolution_scale_dimensions = lambda *args, **kwargs: (1024, 576)
    p.node._h3_decode = lambda samples, vae: tensor(.5)
    p.node.process(object(), tensor(), p.prompts, unique_id='node-7')
    references = [kwargs for name, kwargs in p.calls if name == 'MiniMaxH3ReferenceToVideo']
    assert len(references) == 3
    assert all(set(kwargs['ref_images']) == {'ref_image_1'} for kwargs in references)
    assert all(kwargs['ref_images']['ref_image_1'] is p.cache['source_upscaler'] for kwargs in references)
    assert all(kwargs['audio_vae'] == 'audio-vae' for kwargs in references)


def test_regeneration_rejects_edited_card_parameters_without_touching_saved_sets(pipeline):
    p = pipeline
    p.node.process(object(), tensor(), p.prompts, unique_id='node-7')
    document = backgrounds.load_background(p.document['background_id'])
    editor = copy.deepcopy(document['editor']); editor['variants'][0]['weather'] = 'Light rain'
    document = backgrounds.save_background(document['background_id'], editor, document['revision'])
    p.encodes.clear(); p.upscales.clear()
    with pytest.raises(ValueError, match='variants changed'):
        p.node.process(object(), tensor(), p.prompts, json.dumps({'regenerate_from': 'background_generation'}), 'node-7')
    assert not p.encodes and not p.upscales
    assert backgrounds.load_background(document['background_id']) == document


def test_each_variant_logs_its_full_actual_prompt_before_model_execution(pipeline, caplog):
    p = pipeline
    caplog.set_level(logging.INFO, logger="VNCCS")
    entries = [json.loads(value) for value in p.prompts]
    entries[0]["weather"] = "Light rain"
    entries[0]["additional_details"] = "Keep the blue curtains open and preserve every existing object. " * 20
    prompts = [json.dumps(entry) for entry in entries]
    run = p.node._background_variant
    seen = []
    def variant(source, pipe, values, prompt, settings, unique_id):
        records = [record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompts"]
        assert len(records) == len(seen) + 1
        record = records[-1]
        assert record.levelno == logging.INFO and record.vnccs["component"] == "BackgroundGenerator"
        assert record.vnccs["positive"] == prompt and record.vnccs["node_id"] == "node-7"
        assert record.vnccs["context"] == f"Variant {len(seen) + 1}/3: {entries[len(seen)]['name']}"
        assert prompt in record.getMessage()
        seen.append(prompt)
        return run(source, pipe, values, prompt, settings, unique_id)
    p.node._background_variant = variant
    p.node.process(object(), tensor(), prompts, unique_id="node-7")
    assert len(seen) == 3 and len(seen[0]) > 240
    assert seen[0].startswith("Change time of the day to morning, weather to light rain.")
