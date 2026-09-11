"""Exercise stage events during work, without model loading or GPU sampling."""

import json
from types import SimpleNamespace

import pytest
import torch

from nodes import character_generator as cg


@pytest.fixture
def generator(monkeypatch):
    node = cg.VNCCS_CharacterGenerator()
    values = {
        "clip": "clip", "vae": "vae", "audio_vae": "audio_vae", "model": "model",
        "seed": 42, "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "model_kind": "qie2511",
    }
    monkeypatch.setattr(node, "_extract_pipe", lambda pipe: values)
    monkeypatch.setattr(node, "_apply_pose_lora_to_model", lambda model, *args: model)
    monkeypatch.setattr(node, "_apply_lora_to_model", lambda model, *args: model)
    monkeypatch.setattr(node, "_validate_conditioning_for_model", lambda *args: None)
    monkeypatch.setattr(cg, "VNCCS_MaskExtractor", lambda: SimpleNamespace(fill_alpha_with_color=lambda image: (image,)))
    events = []

    def capture(name, payload):
        assert name == "vnccs.character_generator.stage"
        events.append(payload)

    monkeypatch.setattr(cg.server.PromptServer.instance, "send_sync", capture, raising=False)
    return node, values, events


@pytest.mark.parametrize("kind", ["qie2511", "klein9b", "minimaxh3"])
@pytest.mark.parametrize("stage,count", [("pose_generation", 12), ("original_pose_generation", 2), ("naked_pose_generation", 1)])
def test_pose_progress_tracks_each_phase_and_item(generator, monkeypatch, kind, stage, count):
    node, values, events = generator
    values["model_kind"] = kind
    calls = []
    phases = {
        "VNCCS_QWEN_Encoder": "Encoding poses",
        "VNCCS_Flux_Klein_Encoder": "Encoding poses",
        "MiniMaxH3ReferenceToVideo": "Encoding poses",
        "KSampler": "Sampling poses",
        "SamplerCustomAdvanced": "Sampling poses",
        "VAEDecodeTiled": "Decoding poses",
    }
    completed = dict.fromkeys(phases.values(), 0)
    image = torch.zeros(1, 8, 8, 3)

    def call(class_name, **kwargs):
        assert "progress_callback" not in kwargs
        phase = phases.get(class_name)
        if phase:
            # The event must arrive before this expensive operation. Counts
            # advance only after its successful return, never optimistically.
            assert events[-1]["message"] == phase
            assert events[-1]["current"] == completed[phase]
            assert events[-1]["total"] == count
            completed[phase] += 1
            calls.append(class_name)
        if class_name in {"VNCCS_QWEN_Encoder", "VNCCS_Flux_Klein_Encoder"}:
            return "positive", "negative", "latent"
        if class_name == "MiniMaxH3ReferenceToVideo":
            return "positive", "latent"
        if class_name == "VAEDecodeTiled":
            return (image,)
        return ("result",)

    monkeypatch.setattr(cg, "_call_comfy_node", call)
    result = node._run_pose_generation(
        image.repeat(count, 1, 1, 1), image, object(), "Pose prompt", {},
        unique_id="node-17", stage=stage, lora_info={"name": "Pose LoRA"},
    )
    expected = [(phase, index) for phase in completed for index in range(count + 1)]
    assert [(event["message"], event["current"]) for event in events] == expected
    for event in events:
        assert event["node_id"] == "node-17"
        assert event["stage"] == stage
        assert event["status"] == "running"
        assert event["lora_info"] == {"name": "Pose LoRA"}
        assert "images" not in event
    assert [phases[name] for name in calls] == [phase for phase in completed for _ in range(count)]
    assert result.shape == (count, 8, 8, 3)


def test_failed_item_does_not_advance_progress_or_run_decode(generator, monkeypatch):
    node, _, events = generator
    calls = []

    def sample(class_name, **kwargs):
        calls.append(kwargs["item"])
        if kwargs["item"] == 2:
            raise RuntimeError("Sampling interrupted")
        return (kwargs["item"],)

    monkeypatch.setattr(cg, "_call_comfy_node", sample)
    with pytest.raises(RuntimeError, match="Sampling interrupted"):
        node._run_list_mapped(
            "KSampler", {"item": [0, 1, 2, 3]},
            progress_callback=node._stage_progress_callback("node-17", "pose_generation", "Sampling poses"),
        )
    assert calls == [0, 1, 2]
    assert [event["current"] for event in events] == [0, 1, 2]
    assert all(event["status"] == "running" for event in events)


def test_remove_clothes_reports_encoding_sampling_and_decoding(generator, monkeypatch):
    node, _, events = generator
    image = torch.zeros(1, 8, 8, 3)

    def encode(*args, **kwargs):
        assert events[-1]["message"] == "Encoding source character"
        return "positive", "negative", "latent"

    def call(class_name, **kwargs):
        expected = "Sampling source character" if class_name == "KSampler" else "Decoding source character"
        assert events[-1]["message"] == expected
        assert events[-1]["current"] == 0
        return (image,)

    monkeypatch.setattr(node, "_encoder_call", encode)
    monkeypatch.setattr(cg, "_call_comfy_node", call)
    assert node._run_remove_clothes(image, object(), {}, unique_id="node-17") is image
    assert [(e["message"], e["current"]) for e in events] == [
        (f"{phase} source character", current)
        for phase in ("Encoding", "Sampling", "Decoding") for current in (0, 1)
    ]
    assert all(e["stage"] == "remove_clothes" and e["node_id"] == "node-17" for e in events)


def test_clothes_generator_forwards_progress_context(monkeypatch):
    node = cg.VNCCS_ClothesGenerator()
    seen = {}
    image = torch.zeros(1, 8, 8, 3)

    def poses(*args, **kwargs):
        seen.update(kwargs)
        return image

    monkeypatch.setattr(node, "_run_pose_generation", poses)
    result = node._run_clothes_pose_generation(
        image, image, object(), "prompt", "Green", {}, unique_id="clothes-node",
    )
    assert result is image
    assert seen["unique_id"] == "clothes-node"
    assert seen["stage"] == "pose_generation"


@pytest.mark.parametrize("mode", ["character", "clothes", "original", "naked"])
@pytest.mark.parametrize("regenerate_index", [None, 2])
def test_entry_points_forward_node_stage_and_actual_pose_count(monkeypatch, mode, regenerate_index):
    node_type = {
        "character": cg.VNCCS_CharacterGenerator,
        "clothes": cg.VNCCS_ClothesGenerator,
        "original": cg.VNCCS_CharacterCloneGenerator,
        "naked": cg.VNCCS_CharacterCloneGenerator,
    }[mode]
    node = node_type()
    stage = f"{mode}_pose_generation" if mode in {"original", "naked"} else "pose_generation"
    poses = torch.zeros(4, 8, 8, 3)
    events = []
    for name in ("_remember_generator_context", "_rotate_preview_cache", "_save_run_inputs"):
        monkeypatch.setattr(cg, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cg, "_character_cache_dir_from_sheets_path", lambda *args: "")
    monkeypatch.setattr(cg, "_load_run_inputs", lambda *args: {})
    monkeypatch.setattr(cg, "_tensor_to_preview_urls", lambda *args, **kwargs: [])
    monkeypatch.setattr(node, "_save_stage", lambda *args: None)
    monkeypatch.setattr(node, "_load_cached_stage", lambda *args: None)
    monkeypatch.setattr(node, "_extract_pipe", lambda *args: {"seed": 42})
    monkeypatch.setattr(node, "_find_pose_lora", lambda *args: {})
    monkeypatch.setattr(cg.server.PromptServer.instance, "send_sync", lambda name, payload: events.append(payload), raising=False)
    if mode == "clothes":
        monkeypatch.setattr(node, "_run_source_upscaler", lambda image, *args, **kwargs: image)

    class ReachedPoseGeneration(Exception):
        pass

    def run(poses, *args, **kwargs):
        assert kwargs["unique_id"] == "17"
        assert kwargs.get("stage", "pose_generation") == stage
        assert events[-1]["stage"] == stage
        assert events[-1]["node_id"] == "17"
        assert events[-1]["total"] == poses.shape[0] == (1 if regenerate_index is not None else 4)
        raise ReachedPoseGeneration

    monkeypatch.setattr(node, "_run_pose_generation", run)
    with pytest.raises(ReachedPoseGeneration):
        if mode in {"original", "naked"}:
            node._run_sprite_branch(
                poses, poses[:1], object(), "prompt", "Green", node._settings("{}"),
                "17", "", mode, {}, regenerate_index=regenerate_index,
            )
        else:
            payload = {} if regenerate_index is None else {"regenerate_from": stage, "regenerate_index": regenerate_index}
            node.process(poses, poses[:1], object(), "prompt", widget_data=json.dumps(payload), unique_id=["17"])
