"""Background Creator contracts: isolated from Utils, PyTorch, models and browser."""

import asyncio
import ast
import copy
import io
import json
import logging
import math
import sys
import threading
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image
from conftest import _preload_node

storage = generation = service = routes_module = sampling = progress = _preload_node("background_creator")
SCENE_COMPOSITION = {
    "front": "A broad path crosses the foreground between low planting beds.",
    "back": "Distant architecture frames the horizon beneath an open sky.",
    "left": "A shaded colonnade runs along the left edge of the scene.",
    "right": "A reflective pool borders the right side of the composition.",
}


class Response:
    def __init__(self, payload=None, status=200, headers=None):
        self.status = status
        self.text = json.dumps(payload)
        self.headers = headers or {}


class FileResponse(Response):
    def __init__(self, path, headers=None):
        super().__init__(headers=headers)
        self.path = path


class HTTPException(Exception):
    pass


class HTTPRequestEntityTooLarge(HTTPException):
    def __init__(self, **kwargs):
        super().__init__("Request is too large")


class RouteTable:
    def __init__(self):
        self.routes = []

    def __iter__(self):
        return iter(self.routes)

    def __getattr__(self, method):
        def route(path):
            def register(handler):
                self.routes.append(types.SimpleNamespace(method=method.upper(), path=path, handler=handler))
                return handler
            return register
        return route


web = types.SimpleNamespace(json_response=Response, FileResponse=FileResponse, RouteTableDef=RouteTable,
                            HTTPException=HTTPException, HTTPRequestEntityTooLarge=HTTPRequestEntityTooLarge)


@pytest.fixture
def library(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "VNCCS" / "Backgrounds")
    return storage.root()


def files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def document_path(document):
    return storage.root() / document["name"] / f"{document['name']}_config.json"


def with_image(name="Sunset room"):
    document = storage.create_background(name, {"prompt": "A quiet room at sunset"})
    return storage.add_image(document["background_id"], Image.new("RGB", (32, 16), "red"),
                             {"seed": 7}, document["revision"])


def test_empty_library_and_named_document_backup(library):
    assert storage.list_backgrounds() == []
    document = storage.create_background("Evening Café", {"prompt": "Sunset"})
    assert len(document["background_id"]) == 32
    assert document["schema_version"] == 2 and document["revision"] == 1
    assert not {"models", "model", "cgs", "model_settings"} & document.keys()
    assert sorted(item.name for item in document_path(document).parent.iterdir()) == ["Evening Café_config.json", "Evening Café_config.json.backup"]
    path = document_path(document)
    assert path.is_file()
    assert json.loads(path.read_bytes()) == document
    assert path.with_suffix(".json.backup").read_bytes() == path.read_bytes()
    assert storage.load_background(document["background_id"]) == document
    with pytest.raises(storage.BackgroundConflictError):
        storage.create_background("evening café")


def test_editor_revisions_selection_and_previous_json_backup(library):
    document = with_image()
    path = document_path(document)
    previous = path.read_bytes()
    state = {"prompt": "Changed", "selected_image_id": document["selected_image_id"],
             "unfinished_prompt": "(bright"}
    updated = storage.save_background(document["background_id"], state, document["revision"])
    assert updated["revision"] == document["revision"] + 1
    assert updated["editor"] == {key: value for key, value in state.items() if key != "selected_image_id"}
    assert updated["selected_image_id"] == state["selected_image_id"]
    assert path.with_suffix(".json.backup").read_bytes() == previous
    state["prompt"] = "Mutated caller object"
    assert storage.load_background(document["background_id"])["editor"]["prompt"] == "Changed"


def test_legacy_image_is_preserved_when_creating_the_working_preview(library):
    document = with_image()
    document["images"][0].pop("preview_revision")
    document_path(document).write_text(json.dumps(document))
    first_id = document["selected_image_id"]
    first_path = storage.image_path(document["background_id"], first_id)
    assert first_path == library / document["name"] / "source.png"
    original = first_path.read_bytes()
    updated = storage.add_image(document["background_id"], Image.new("RGBA", (20, 10), "blue"),
                                {"source": "upload"}, document["revision"])
    assert len(updated["images"]) == 2
    assert updated["selected_image_id"] != first_id
    retained = storage.image_path(document["background_id"], first_id)
    assert retained.read_bytes() == original
    assert retained == library / document["name"] / "Sources" / "V1.png"
    with Image.open(storage.image_path(document["background_id"], updated["selected_image_id"])) as image:
        assert image.mode == "RGB"
        assert image.size == (20, 10)


def test_repeated_preview_generation_replaces_one_image_and_refreshes_its_url(library):
    document = with_image()
    identity = document["selected_image_id"]
    old_url = storage.public_document(document)["images"][0]["url"]
    for color in ("blue", "green", "yellow"):
        document = storage.add_image(document["background_id"], Image.new("RGB", (20, 10), color),
                                     {"prompt": color}, document["revision"])
        assert len(document["images"]) == 1 and document["selected_image_id"] == identity
        with Image.open(storage.image_path(document["background_id"], identity)) as image:
            assert image.size == (20, 10) and image.getpixel((0, 0)) == Image.new("RGB", (1, 1), color).getpixel((0, 0))
        url = storage.public_document(document)["images"][0]["url"]
        assert url != old_url and url.endswith(f"?v={document['images'][0]['preview_revision']}")
        old_url = url
    assert len(list((library / document["name"]).glob("*.png"))) == 1


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["png", "json"])
def test_preview_write_failure_retains_image_metadata_and_cleans_temporary_files(library, monkeypatch, existing, failure):
    document = with_image() if existing else storage.create_background("Failed preview", {})
    before = files(library)
    atomic = storage._atomic
    def write(path, payload):
        if (failure == "png" and path.suffix == ".png") or (failure == "json" and path.name == f"{document['name']}_config.json"):
            raise OSError("Disk full")
        return atomic(path, payload)
    monkeypatch.setattr(storage, "_atomic", write)
    with pytest.raises(OSError, match="Disk full"):
        storage.add_image(document["background_id"], Image.new("RGB", (20, 10), "blue"), {}, document["revision"])
    after = files(library)
    backup_name = f"{document['name']}/{document['name']}_config.json.backup"
    assert after.pop(backup_name) in (before[backup_name], before[f"{document['name']}/{document['name']}_config.json"])
    before.pop(backup_name)
    assert after == before
    assert storage.load_background(document["background_id"]) == document


@pytest.mark.parametrize("operation", ["save", "archive", "image"])
def test_stale_writes_preserve_every_file_byte(library, operation):
    document = with_image()
    background_id, revision = document["background_id"], document["revision"] - 1
    before = files(library)
    image = Image.new("RGB", (8, 8))
    operations = {
        "save": lambda: storage.save_background(background_id, {"prompt": "Stale"}, revision),
        "archive": lambda: storage.archive_background(background_id, revision),
        "image": lambda: storage.add_image(background_id, image, {}, revision),
    }
    with pytest.raises(storage.BackgroundConflictError):
        operations[operation]()
    assert files(library) == before


def test_concurrent_editor_revisions_allow_only_one_writer(library):
    document = storage.create_background("Concurrent")

    def save(prompt):
        try:
            return storage.save_background(document["background_id"], {"prompt": prompt}, document["revision"])
        except storage.BackgroundConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["First", "Second"]))
    assert sum(result is not None for result in results) == 1
    assert storage.load_background(document["background_id"])["revision"] == 2


def test_damaged_primary_recovers_backup_read_only_without_replacing_bytes(library):
    document = storage.create_background("Damaged", {"prompt": "Keep this"})
    path = document_path(document)
    damaged = b'{"unfinished":'
    path.write_bytes(damaged)
    before = files(library)
    recovered = storage.load_background(document["background_id"])
    assert recovered["editor"] == {"prompt": "Keep this"}
    assert recovered["_recovery"]["source"] == f"{document['name']}_config.json.backup"
    assert path.read_bytes() == damaged
    with pytest.raises(storage.BackgroundStorageError, match="read-only"):
        storage.save_background(document["background_id"], {}, document["revision"])
    assert files(library) == before


def test_corrupt_primary_and_backup_never_become_empty_library(library):
    document = storage.create_background("Unreadable")
    path = document_path(document)
    path.write_bytes(b"{broken")
    path.with_suffix(".json.backup").write_bytes(b"[]")
    before = files(library)
    with pytest.raises(storage.BackgroundStorageError, match="original files are preserved"):
        storage.list_backgrounds()
    assert files(library) == before


def test_failed_atomic_replace_preserves_primary(library, monkeypatch):
    document = storage.create_background("Disk failure")
    path = document_path(document)
    original = path.read_bytes()
    replace = storage.os.replace

    def fail_primary(source, target):
        if Path(target) == path:
            raise OSError("Disk unavailable")
        replace(source, target)

    monkeypatch.setattr(storage.os, "replace", fail_primary)
    with pytest.raises(OSError, match="Disk unavailable"):
        storage.save_background(document["background_id"], {"prompt": "Unsaved"}, document["revision"])
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("name", ["../escape", "Bad/name", "Bad\\name", "NUL", "CON.txt", "Bad:", "End.", "\x00bad", "", ".", ".."])
def test_names_cannot_escape_or_break_windows_paths(library, name):
    with pytest.raises(ValueError):
        storage.create_background(name)
    assert files(library) == {}


def test_invalid_ids_nan_unknown_selection_and_limits_do_not_write(library, monkeypatch):
    document = with_image()
    before = files(library)
    with pytest.raises(ValueError, match="ID"):
        storage.load_background("../escape")
    with pytest.raises(ValueError, match="finite JSON"):
        storage.save_background(document["background_id"], {"fov": float("nan")}, document["revision"])
    with pytest.raises(ValueError, match="does not exist"):
        storage.save_background(document["background_id"], {"selected_image_id": "f" * 32}, document["revision"])
    monkeypatch.setattr(storage, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="dimensions"):
        storage.add_image(document["background_id"], Image.new("RGB", (2, 2)), {}, document["revision"])
    assert files(library) == before


def test_image_symlinks_are_rejected(library, tmp_path):
    document = with_image()
    image_id = document["selected_image_id"]
    target = storage.image_path(document["background_id"], image_id)
    outside = tmp_path / "outside.png"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links are not available on this platform")
    with pytest.raises(ValueError, match="symbolic links"):
        storage.image_path(document["background_id"], image_id)

def test_defaults_are_isolated_and_unknown_authored_fields_survive():
    editor = generation.normalize_editor({"generation_mode": "illustrious",
        "generation": {"diffusion_model_name": "other.safetensors", "mode_settings": {"anima": {}}}})
    assert editor == {**generation.DEFAULT_EDITOR, "generation_mode": "illustrious", "generation": {**generation.DEFAULT_EDITOR["generation"], "diffusion_model_name": "other.safetensors", "mode_settings": {"anima": {}}}}
    editor["attributes"]["location"] = "Room"
    editor["generation"]["lora_stack"].append({"name": "test"})
    assert "location" not in generation.DEFAULT_EDITOR["attributes"]
    assert generation.DEFAULT_EDITOR["generation"]["lora_stack"] == []
    assert "composition" not in editor["attributes"]


@pytest.mark.parametrize("data", [[], "{}", {"attributes": []}, {"generation": []},
    {"scene_description": 1}, {"scene_description": "x" * 40001}, {"prompt": 1}, {"prompt": "x" * 40001}, {"description": 1}, {"description": "x" * 8001}, {"style": "bad\x00style"},
    {"custom_style": None}, {"custom_style": "x" * 16001}, {"custom_style": "bad\x00style"},
    {"aspect": "not-an-aspect"}, {"aspect": []}, {"attributes": {"objects": ["desk"]}},
    {"attributes": {"composition": []}}, {"attributes": {"composition": "x" * 2001}},
    {"generation": {"diffusion_model_name": "../secret"}}, {"generation": {"diffusion_model_name": ""}},
    {"generation": {"diffusion_model_name": "/secret"}}, {"generation": {"diffusion_model_name": "C:\\secret"}},
    {"generation": {"steps": True}}, {"generation": {"steps": 1.5}},
    {"generation": {"cfg": float("nan")}}, {"generation": {"cfg": float("inf")}},
    {"generation": {"seed": -1}}, {"generation": {"seed": 2**80}},
    {"generation": {"resolution_mp": 20}}, {"generation": {"prompt_enhance": "yes"}},
    {"generation": {"turbo": 1}}, {"generation": {"cache_device": "cuda"}},
    {"generation": {"cache_dtype": "fp64"}}, {"generation": {"sampler": ""}},
    {"generation": {"lora_stack": {}}}, {"generation": {"lora_stack": [{}] * 9}},
    {"generation": {"lora_stack": [None]}},
    {"generation": {"lora_stack": [{"name": "../secret"}]}},
    {"generation": {"lora_stack": [{"name": "C:\\secret"}]}},
    {"generation": {"lora_stack": [{"name": "\\\\server\\secret"}]}},
    {"generation": {"lora_stack": [{"name": "/secret"}]}},
    {"generation": {"lora_stack": [{"name": "good.safetensors", "strength": float("nan")}]}},
    {"generation": {"lora_stack": [{"name": generation.TURBO_LORA}]}}])
def test_rejects_malformed_settings(data):
    with pytest.raises(ValueError):
        generation.normalize_editor(data)


@pytest.mark.parametrize("key", ["front", "back", "left", "right"])
@pytest.mark.parametrize("value", [None, [], 7, "x" * 40001, "bad\x00field"])
def test_composition_fields_reject_malformed_text(key, value):
    with pytest.raises(ValueError):
        generation.normalize_editor({key: value})


def test_composition_fields_round_trip_disk_and_structured_prompts(library):
    editor = generation.normalize_editor({"scene_description": "A quiet courtyard with layered architecture.", **SCENE_COMPOSITION})
    prompt = generation.compose_prompt(editor)
    assert prompt.splitlines()[4:] == [f"{key.title()}: {value}" for key, value in SCENE_COMPOSITION.items()]
    assert generation.compose_prompt({"prompt": prompt}) == prompt
    document = storage.create_background("Courtyard", editor)
    restored = generation.normalize_editor(storage.load_background(document["background_id"])["editor"])
    assert restored == editor
    restored["front"] = ""
    assert generation.normalize_editor(restored)["front"] == ""
    assert "Front:" not in generation.compose_prompt(restored)
    assert generation.normalize_editor({"prompt": prompt, "front": ""})["front"] == ""
    legacy = generation.normalize_editor({"scene_description": "Keep the original room."})
    assert legacy["scene_description"] == "Keep the original room."
    assert all(legacy[key] == "" for key in SCENE_COMPOSITION)


@pytest.mark.parametrize("aspect", generation.ASPECTS)
@pytest.mark.parametrize("mp", [0.05, 2, 16])
def test_dimensions_follow_aspect_and_bounded_pixel_budget(aspect, mp):
    width, height = generation.dimensions({"aspect": aspect, "generation": {"resolution_mp": mp}})
    a, b = map(int, aspect.split(":"))
    assert width % 8 == height % 8 == 0
    assert width * b == height * a
    assert 64 <= width <= 4096 and 64 <= height <= 4096
    assert width * height <= 16_000_000
    assert width / height == pytest.approx(a / b, rel=0.08)


def test_prompt_uses_environment_attributes_and_visual_style():
    prompt = generation.compose_prompt({"description": "A small bedroom", "attributes": {
        "location": "Tokyo", "objects": "bookshelves and a desk", "lighting": "warm window light",
        "composition": "Wide establishing shot with a central vanishing point"},
        "style": "Painted animation background", "aesthetics": "Cinematic lighting"})
    assert "A small bedroom" in prompt
    assert "Location: Tokyo." in prompt
    assert "Objects and furnishings: bookshelves and a desk." in prompt
    assert "Composition: Wide establishing shot with a central vanishing point." in prompt
    assert prompt.startswith("Style: Painted animation background\nTime: Day\nInterior/Exterior: Interior\nScene description: ")
    assert len(prompt.splitlines()) == 4
    assert "1girl" not in prompt


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    clips = []
    base = mock.Mock(name="cached-model")
    base.clone.side_effect = lambda: mock.Mock(name="request-model")
    vae = object()
    monkeypatch.setattr(generation, "_ASSET_CACHE", None)
    monkeypatch.setattr(generation, "_ACTIVE", None)
    monkeypatch.setattr(generation, "_asset_name", lambda category, name: name)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(sampling, "_ensure_direct_sampling_prompt_context", lambda *_: None)
    monkeypatch.setattr(sampling, "_report_comfy_sampling_progress", lambda *_: nullcontext())
    monkeypatch.setattr(progress, "_set_draw_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(progress, "set_interrupt", mock.Mock())
    decoded_size = [generation.dimensions({})]
    monkeypatch.setattr(generation, "_tensor_to_pil", lambda tensor: Image.new("RGB", decoded_size[0]))

    def encode_qwen(clip, prompt, negative_prompt, vae=None, resolution=1024, images=None):
        return "positive", "negative", {"samples": "unused-encoder-latent"}

    def call(name, **kwargs):
        calls.append((name, kwargs))
        if name == "UNETLoader":
            return (base,)
        if name == "CLIPLoader":
            clip = object()
            clips.append(clip)
            return (clip,)
        if name == "VAELoader":
            return (vae,)
        if name == "LoraLoader":
            return kwargs["model"], kwargs["clip"]
        if name == "QwenImage21Cache":
            return (kwargs["model"],)
        if name == "TextGenerate":
            return (generation.compose_prompt({"scene_description": "Enhanced empty background with warm light", **SCENE_COMPOSITION}),)
        if name == "TextEncodeQwenImage21":
            return encode_qwen(**kwargs)
        if name == "EmptyLatentImage":
            return ({"samples": "empty"},)
        if name == "EmptyFlux2LatentImage":
            return ({"samples": "flux2-empty"},)
        if name == "CLIPTextEncode":
            return ("conditioning:" + kwargs["text"],)
        return (name + "-output",)

    monkeypatch.setattr(generation, "_call_comfy_node", call)
    return types.SimpleNamespace(calls=calls, clips=clips, base=base, decoded_size=decoded_size, call=call)


@pytest.mark.parametrize("aspect", ["16:9", "9:16", "1:1"])
def test_text_to_image_pipeline_uses_creator_assets_cache_and_true_empty_latent(runtime, aspect):
    editor = {"aspect": aspect}
    runtime.decoded_size[0] = generation.dimensions(editor)
    image, metadata = generation.generate_image(editor, "background-t2i", threading.Event())
    names = [name for name, _ in runtime.calls]
    assert names == ["UNETLoader", "CLIPLoader", "VAELoader", "QwenImage21Cache", "TextGenerate",
                     "TextEncodeQwenImage21", "EmptyLatentImage", "KSampler", "VAEDecode"]
    calls = dict(runtime.calls)
    assert calls["UNETLoader"]["unet_name"] == generation.ASSETS["diffusion_models"]
    assert calls["CLIPLoader"]["clip_name"] == generation.ASSETS["text_encoders"]
    assert calls["CLIPLoader"]["type"] == "qwen_image"
    assert calls["VAELoader"]["vae_name"] == generation.ASSETS["vae"]
    assert calls["QwenImage21Cache"]["device"] == "gpu"
    assert calls["QwenImage21Cache"]["dtype"] == "int8"
    assert calls["TextGenerate"]["sampling_mode"] == generation.TEXT_GENERATION["sampling_mode"]
    assert calls["TextGenerate"]["sampling_mode"]["temperature"] == 0.7
    assert calls["TextEncodeQwenImage21"]["images"] == {}
    assert calls["TextEncodeQwenImage21"]["prompt"] == generation.compose_prompt({"scene_description": "Enhanced empty background with warm light", **SCENE_COMPOSITION}) + f" {generation.BACKGROUND_PROMPT_SUFFIX}"
    assert calls["KSampler"]["latent_image"] == {"samples": "empty"}
    assert calls["KSampler"]["steps"] == 25 and calls["KSampler"]["cfg"] == 1
    assert calls["KSampler"]["scheduler"] == "simple" and calls["KSampler"]["denoise"] == 1
    assert set(calls["TextEncodeQwenImage21"]) == {"clip", "vae", "prompt", "negative_prompt", "resolution", "images"}
    assert calls["TextEncodeQwenImage21"]["resolution"] == 1024
    assert calls["EmptyLatentImage"] == {"width": image.width, "height": image.height, "batch_size": 1}
    assert image.size == generation.dimensions(editor)
    assert metadata["editor"] == generation.normalize_editor(editor)
    assert metadata["model"] == "qi2"
    runtime.base.clone.assert_called_once()


def test_fresh_clip_per_request_and_no_automatic_character_loras(runtime):
    editor = {"generation": {"prompt_enhance": False, "lora_stack": [{"name": "scenery.safetensors", "strength": 0.7}]}}
    generation.generate_image(editor, "first", threading.Event())
    generation.generate_image(editor, "second", threading.Event())
    names = [name for name, _ in runtime.calls]
    assert names.count("UNETLoader") == names.count("VAELoader") == 1
    assert names.count("CLIPLoader") == 2 and runtime.clips[0] is not runtime.clips[1]
    assert "TextGenerate" not in names
    assert [call["lora_name"] for name, call in runtime.calls if name == "LoraLoader"] == ["scenery.safetensors"] * 2
    assert runtime.base.clone.call_count == 2


@pytest.mark.parametrize("enhance", [True, False])
def test_moss_t2i_uses_attached_workflow_assets_flux2_latent_and_our_enhancer(runtime, enhance):
    editor = {"style": "Anime, digital art, cel shaded", "scene_description": "badroom",
              "attributes": {"time": "Night", "scene_type": "Interior"},
              "negative": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
              "generation": {"model_type": "moss", "prompt_enhance": enhance, "turbo": True}}
    runtime.decoded_size[0] = generation.dimensions(editor)
    image, metadata = generation.generate_image(editor, "moss-t2i", threading.Event())
    calls, names = dict(runtime.calls), [name for name, _ in runtime.calls]
    assert calls["UNETLoader"]["unet_name"] == "moss_image_sft_int8_convrot.safetensors"
    assert calls["CLIPLoader"] == {"clip_name": "qwen3.5_9b_w4a8_gptq.safetensors", "type": "moss_image", "device": "default"}
    assert calls["VAELoader"]["vae_name"] == "flux2-vae_small.safetensors"
    encodes = [value for name, value in runtime.calls if name == "CLIPTextEncode"]
    expected = generation.compose_prompt({**editor, "scene_description": "Enhanced empty background with warm light", **SCENE_COMPOSITION}) if enhance else generation.compose_prompt(editor)
    expected += f" {generation.BACKGROUND_PROMPT_SUFFIX}"
    assert [item["text"] for item in encodes] == [expected, editor["negative"]]
    assert calls["EmptyFlux2LatentImage"] == {"width": image.width, "height": image.height, "batch_size": 1}
    sampler = calls["KSampler"]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]) == (25, 2.5, "er_sde", "simple", 1)
    assert sampler["latent_image"] == {"samples": "flux2-empty"}
    assert sampler["positive"] == "conditioning:" + expected
    assert sampler["negative"] == "conditioning:" + editor["negative"]
    assert not {"QwenImage21Cache", "TextEncodeQwenImage21", "EmptyLatentImage", "LoraLoader", "SamplerCustomAdvanced"}.intersection(names)
    assert ("TextGenerate" in names) is enhance
    if enhance:
        assert calls["TextGenerate"]["system_prompt"].startswith(generation.SCENE_EXPANSION_SYSTEM)
        assert "Keep Style, Time and Interior/Exterior exactly as supplied" in calls["TextGenerate"]["system_prompt"]
    assert metadata["model"] == "moss" and metadata["editor"]["generation"]["turbo"] is False
    assert metadata["prompt"] == expected


def test_switching_generation_families_reloads_assets_and_keeps_text_encoders_fresh(runtime):
    for model_type in ("moss", "moss", "qi2"):
        runtime.decoded_size[0] = generation.dimensions({"generation": {"model_type": model_type}})
        generation.generate_image({"generation": {"model_type": model_type, "prompt_enhance": False}}, model_type, threading.Event())
    assert [value["unet_name"] for name, value in runtime.calls if name == "UNETLoader"] == [
        generation.MOSS_ASSETS["diffusion_models"], generation.ASSETS["diffusion_models"]]
    assert [value["type"] for name, value in runtime.calls if name == "CLIPLoader"] == ["moss_image", "moss_image", "qwen_image"]
    assert len({id(clip) for clip in runtime.clips}) == 3


def test_replaced_model_weights_reload_the_in_memory_model_cache(runtime, tmp_path, monkeypatch):
    paths = {}
    for category, name in generation.ASSETS.items():
        path = tmp_path / name; path.write_bytes(name.encode()); paths[(category, name)] = str(path)
    monkeypatch.setattr(sys.modules["folder_paths"], "get_full_path", lambda category, name: paths[(category, name)])
    generation.generate_image({}, "first", threading.Event())
    generation.generate_image({}, "cached", threading.Event())
    Path(paths[("diffusion_models", generation.ASSETS["diffusion_models"]) ]).write_bytes(b"replacement model weights")
    generation.generate_image({}, "replaced", threading.Event())
    assert [name for name, _ in runtime.calls].count("UNETLoader") == 2
    assert [name for name, _ in runtime.calls].count("CLIPLoader") == 3


def test_generation_profiles_preserve_qwen_settings_and_default_legacy_scenes_to_qwen():
    old = {"diffusion_model_name": "my/room.safetensors", "steps": 31, "turbo": True,
           "lora_stack": [{"name": "room.safetensors", "strength": 0.6}]}
    editor = generation.normalize_editor({"generation": {"model_type": "moss"}, "generation_profiles": {"qi2": old, "moss": {"steps": 19}}})
    assert editor["generation"]["sampler"] == "er_sde" and editor["generation"]["cfg"] == 2.5
    assert editor["generation_profiles"]["qi2"]["diffusion_model_name"] == old["diffusion_model_name"]
    assert editor["generation_profiles"]["qi2"]["lora_stack"] == old["lora_stack"]
    assert editor["generation_profiles"]["moss"]["steps"] == 19
    assert generation.normalize_editor(editor) == editor
    assert generation.normalize_editor({"generation": old})["generation"]["model_type"] == "qi2"
    for invalid in ({"generation": {"model_type": "unknown"}}, {"generation_profiles": []}, {"generation_profiles": {"unknown": {}}}):
        with pytest.raises(ValueError):
            generation.normalize_editor(invalid)


def test_decoding_must_match_requested_size(runtime):
    runtime.decoded_size[0] = (64, 64)
    with pytest.raises(RuntimeError, match="expected"):
        generation.generate_image({}, "wrong-size", threading.Event())


def test_turbo_uses_main_project_adapter_and_advanced_sampler(runtime, monkeypatch):
    adapter = types.SimpleNamespace(apply_viggle_turbo_lora=lambda model, name, strength: runtime.calls.append(("TurboAdapter", {"lora_name": name})) or model)
    sigmas = types.SimpleNamespace(viggle_turbo_sigmas=lambda latent: runtime.calls.append(("TurboSigmas", {"latent": latent})) or [1, 0])
    monkeypatch.setitem(sys.modules, generation.__package__ + ".character_generator", adapter)
    monkeypatch.setitem(sys.modules, generation.__package__ + ".qi2_viggle", sigmas)
    generation.generate_image({"generation": {"turbo": True, "steps": 6, "cfg": 1}}, "turbo", threading.Event())
    names = [name for name, _ in runtime.calls]
    assert names.index("TurboAdapter") < names.index("QwenImage21Cache")
    assert "KSampler" not in names and "SamplerCustomAdvanced" in names
    assert dict(runtime.calls)["TurboSigmas"]["latent"] == {"samples": "empty"}


@pytest.mark.parametrize("settings", [{"turbo": True}, {"turbo": True, "steps": 6, "cfg": 2}])
def test_turbo_normalizes_schedule_like_character_creator(runtime, settings):
    adapter = types.SimpleNamespace(apply_viggle_turbo_lora=lambda model, name, strength: model)
    monkey = mock.patch.dict(sys.modules, {generation.__package__ + ".character_generator": adapter,
        generation.__package__ + ".qi2_viggle": types.SimpleNamespace(viggle_turbo_sigmas=lambda latent: [1, 0])})
    with monkey:
        _image, metadata = generation.generate_image({"generation": settings}, "turbo-normalized", threading.Event())
    assert metadata["editor"]["generation"]["steps"] == 6
    assert metadata["editor"]["generation"]["cfg"] == 1
    assert "SamplerCustomAdvanced" in [name for name, _ in runtime.calls]


def test_cancelled_pending_job_never_loads_models_or_interrupts_other_jobs(runtime):
    event = threading.Event()
    event.set()
    with pytest.raises(generation.GenerationCancelled):
        generation.generate_image({}, "cancelled", event)
    assert runtime.calls == []
    assert generation.interrupt("someone-else") is False
    progress.set_interrupt.assert_not_called()


def test_new_owned_job_clears_a_previous_workflow_interrupt_before_loading(runtime, monkeypatch):
    def call(name, **kwargs):
        progress.set_interrupt.assert_called_once_with(False)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    generation.generate_image({}, "fresh-request", threading.Event())
    assert generation._ACTIVE is None


def test_interrupt_only_targets_the_active_background_job(runtime):
    event = threading.Event()
    with generation._model_session("owned", event):
        assert generation.interrupt("another-widget") is False
        assert generation.interrupt("owned") is True
        assert event.is_set()
    assert progress.set_interrupt.call_args_list == [mock.call(False), mock.call(True), mock.call(False)]
    assert generation._ACTIVE is None
    assert generation.interrupt("owned") is False


def test_cancel_after_encoding_discards_result_before_sampling(runtime, monkeypatch):
    event = threading.Event()
    def call(name, **kwargs):
        result = runtime.call(name, **kwargs)
        if name == "TextEncodeQwenImage21":
            event.set()
        return result
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    with pytest.raises(generation.GenerationCancelled):
        generation.generate_image({}, "cancel-encode", event)
    assert "KSampler" not in dict(runtime.calls)
    assert generation._ACTIVE is None


def test_wizard_uses_only_fresh_qi2_encoder_and_parses_thinking_and_json(runtime, monkeypatch):
    result = {"attributes": {"location": "Old Kyoto", "objects": "wooden shelves", "composition": "Symmetrical wide shot"},
              "style": "Painted anime"}
    raw = '<think>{"location":"ignore this reasoning"}</think>\n```json\n' + json.dumps(result) + "\n```"
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (raw,)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    editor = generation.wizard("A quiet old Japanese shop", {}, "wizard", threading.Event())
    assert [name for name, _ in runtime.calls] == ["CLIPLoader", "TextGenerate"]
    assert editor["attributes"]["location"] == "Old Kyoto"
    assert editor["attributes"]["objects"] == "wooden shelves"
    assert editor["attributes"]["time"] == "Day"
    assert editor["attributes"]["composition"] == "Symmetrical wide shot"
    assert editor["style"] == "Painted anime"
    assert "A quiet old Japanese shop" in editor["scene_description"]
    assert "English" in dict(runtime.calls)["TextGenerate"]["system_prompt"]


@pytest.mark.parametrize("value", ["No JSON here", "<think>{\"location\":\"secret\"}</think>", "[]", "{bad}"])
def test_wizard_rejects_invalid_json(value):
    with pytest.raises(ValueError, match="valid JSON"):
        generation._json_object(generation._clean_text(value))


def test_configuration_reports_qi2_assets_and_installed_loras_without_loads(monkeypatch):
    installed = {category: ["QI2/" + name] for category, name in generation.ASSETS.items()}
    installed["loras"] = ["backgrounds/example.safetensors"]
    monkeypatch.setattr(generation, "_installed", lambda category: installed.get(category, []))
    config = generation.configuration()
    assert config["installed"] is True
    assert config["model"] == "Qwen Image 2.1 INT8 ConvRot"
    assert config["loras"] == installed["loras"]
    from nodes.sampler_scheduler_picker import fetch_sampler_scheduler_lists
    assert (config["samplers"], config["schedulers"]) == fetch_sampler_scheduler_lists()
    installed["vae"] = []
    assert generation.configuration()["installed"] is False


def test_configuration_reports_moss_assets_independently_of_qwen(monkeypatch):
    installed = {category: ["QI2/" + name, "Moss/" + generation.MOSS_ASSETS[category]] for category, name in generation.ASSETS.items()}
    monkeypatch.setattr(generation, "_installed", lambda category: installed.get(category, []))
    config = generation.configuration()
    assert config["model_types"]["moss"]["installed"] is True
    assert {item["name"] for item in config["model_types"]["moss"]["assets"]} == set(generation.MOSS_ASSETS.values())
    installed["vae"] = ["QI2/" + generation.ASSETS["vae"]]
    config = generation.configuration()
    assert config["installed"] is True and config["model_types"]["moss"]["installed"] is False


def png(size=(128, 64), color="red"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "VNCCS" / "Backgrounds")
    monkeypatch.setattr(service, "_JOBS", {})
    worker = mock.Mock()
    monkeypatch.setattr(service, "_WORKER", worker)
    monkeypatch.setattr(service, "_emit", mock.Mock())
    monkeypatch.setattr(generation, "progress", lambda _job_id: {})
    monkeypatch.setattr(generation, "interrupt", mock.Mock(return_value=False))
    generated = mock.Mock(return_value=(Image.new("RGB", (128, 64), "blue"), {"seed": 7}))
    monkeypatch.setattr(generation, "generate_image", generated)
    wizard = mock.Mock(return_value=generation.normalize_editor({"attributes": {"location": "Tokyo"}}))
    monkeypatch.setattr(generation, "wizard", wizard)
    for category, name in [*generation.ASSETS.items(), *generation.MOSS_ASSETS.items(), ("loras", generation.TURBO_LORA), ("loras", "room.safetensors")]:
        path = tmp_path / "weights" / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    monkeypatch.setattr(generation, "_resolve_model_path", lambda category, name: tmp_path / "weights" / category / name)
    monkeypatch.setattr(generation, "_preview_tensor", lambda path: np.asarray(Image.open(path).convert("RGB"))[None])

    def new_job(kind, scene_id=""):
        return {"job_id": uuid.uuid4().hex, "kind": kind, "scene_id": scene_id,
                "status": "queued", "stage": "queued", "progress": 0, "message": "Queued",
                "result": None, "cancel_event": threading.Event(), "error": ""}

    monkeypatch.setattr(service, "_new_job", mock.Mock(side_effect=new_job))
    document = storage.create_background("Room", generation.normalize_editor({"description": "Quiet bedroom"}))

    def run_pending():
        function, job, callback = worker.submit.call_args.args
        function(job, callback)
        return job

    return types.SimpleNamespace(document=document, worker=worker, generated=generated,
                                 wizard=wizard, run_pending=run_pending, tmp_path=tmp_path)


def job_with_image(workspace):
    document = workspace.document
    return storage.add_image(document["background_id"], Image.new("RGB", (128, 64), "red"),
                             {"source": "upload"}, document["revision"])


def job_payload(document, action, **changes):
    payload = {"action": action, "revision": document["revision"], "editor": copy.deepcopy(document["editor"]), **changes}
    if action == "wizard":
        payload.setdefault("description", "A quiet room")
    return payload


@pytest.mark.parametrize("payload", [None, [], {"action": "unknown"}, {"action": "export"}, {"action": "model"}, {"action": "cg"}])
def test_invalid_job_action_never_queues_or_changes_document(workspace, payload):
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], payload)
    workspace.worker.submit.assert_not_called()
    assert storage.load_background(workspace.document["background_id"]) == workspace.document


@pytest.mark.parametrize("action", ["background", "wizard"])
@pytest.mark.parametrize("revision", [None, True, 0, 999])
def test_jobs_ignore_workflow_revision_and_capture_current_disk_state(workspace, revision, action):
    payload = job_payload(workspace.document, action, revision=revision)
    document = storage.save_background(workspace.document["background_id"], workspace.document["editor"])
    service.start_job(document["background_id"], payload)
    assert workspace.run_pending()["status"] == "completed"
    if action == "background":
        assert storage.load_background(document["background_id"])["revision"] == document["revision"] + 1


def test_background_generation_is_queued_and_uses_captured_settings(workspace):
    document = workspace.document
    payload = job_payload(document, "background")
    payload["editor"]["attributes"]["location"] = "Original scene"
    status = service.start_job(document["background_id"], payload)
    assert status["status"] == "queued" and status["action"] == "background"
    assert "cancel_event" not in status
    workspace.generated.assert_not_called()
    payload["editor"]["attributes"]["location"] = "Next request"
    job = workspace.run_pending()
    assert job["status"] == "completed"
    assert workspace.generated.call_args.args[0]["attributes"]["location"] == "Original scene"
    updated = storage.load_background(document["background_id"])
    assert updated["revision"] == document["revision"] + 1 and len(updated["images"]) == 1
    assert job["result"]["document"]["images"][0]["url"].endswith(f"/images/{updated['selected_image_id']}?v={updated['images'][0]['preview_revision']}")


def test_one_background_cannot_start_conflicting_jobs(workspace):
    document = workspace.document
    service.start_job(document["background_id"], job_payload(document, "background"))
    with pytest.raises(RuntimeError, match="busy"):
        service.start_job(document["background_id"], job_payload(document, "wizard"))
    assert workspace.worker.submit.call_count == 1
    other = storage.create_background("Other room")
    service.start_job(other["background_id"], job_payload(other, "background"))
    assert workspace.worker.submit.call_count == 2


def test_wizard_returns_draft_without_overwriting_authored_document(workspace):
    document = workspace.document
    service.start_job(document["background_id"], job_payload(document, "wizard", description="Misty mountain village"))
    job = workspace.run_pending()
    assert job["status"] == "completed"
    assert job["result"]["editor"]["attributes"]["location"] == "Tokyo"
    assert storage.load_background(document["background_id"]) == document
    workspace.generated.assert_not_called()
    assert workspace.wizard.call_args.args[0] == "Misty mountain village"


@pytest.mark.parametrize("description", [None, "", "   ", [], "x" * 40001])
def test_wizard_rejects_invalid_brief_before_queueing(workspace, description):
    payload = job_payload(workspace.document, "wizard", description=description)
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], payload)
    workspace.worker.submit.assert_not_called()
    assert storage.load_background(workspace.document["background_id"]) == workspace.document


@pytest.mark.parametrize("condition", ["archived", "recovered"])
def test_uneditable_documents_fail_before_inference(workspace, condition):
    document = workspace.document
    if condition == "archived":
        document = storage.archive_background(document["background_id"], document["revision"])
    else:
        path = storage.root() / document["name"] / f"{document['name']}_config.json"
        path.write_bytes(b"{damaged")
    with pytest.raises(ValueError):
        service.start_job(document["background_id"], job_payload(document, "background"))
    workspace.worker.submit.assert_not_called()


def test_cancelled_queued_job_never_calls_models_or_writes(workspace):
    document = workspace.document
    queued = service.start_job(document["background_id"], job_payload(document, "background"))
    service.cancel_job(queued["job_id"])
    job = workspace.run_pending()
    assert job["status"] == "cancelled"
    generation.interrupt.assert_called_once_with(queued["job_id"])
    workspace.generated.assert_not_called()
    assert storage.load_background(document["background_id"]) == document


def test_cancel_after_sampling_prevents_commit(workspace):
    document = workspace.document

    def generate(_editor, _job_id, event, **_kwargs):
        event.set()
        return Image.new("RGB", (128, 64)), {}

    workspace.generated.side_effect = generate
    service.start_job(document["background_id"], job_payload(document, "background"))
    job = workspace.run_pending()
    assert job["status"] == "cancelled"
    assert storage.load_background(document["background_id"]) == document


def test_failed_generation_preserves_existing_background(workspace):
    document = job_with_image(workspace)
    workspace.generated.side_effect = RuntimeError("QI2 decoder unavailable")
    service.start_job(document["background_id"], job_payload(document, "background"))
    job = workspace.run_pending()
    assert job["status"] == "failed" and job["error"] == "QI2 decoder unavailable"
    assert storage.load_background(document["background_id"]) == document


def test_late_generation_cannot_replace_newer_editor_revision(workspace):
    document = workspace.document
    service.start_job(document["background_id"], job_payload(document, "background"))
    newer = storage.save_background(document["background_id"], {"description": "New authored draft"}, document["revision"])
    job = workspace.run_pending()
    assert job["status"] == "failed"
    assert "changed" in job["error"]
    assert storage.load_background(document["background_id"]) == newer


def test_inference_cancellation_is_reported_as_cancelled(workspace):
    document = workspace.document
    workspace.generated.side_effect = generation.GenerationCancelled("Background generation cancelled")
    service.start_job(document["background_id"], job_payload(document, "background"))
    job = workspace.run_pending()
    assert job["status"] == "cancelled"
    assert storage.load_background(document["background_id"]) == document


class Request:
    def __init__(self, payload=None, *, info=None, raw=None, multipart=None, headers=None):
        self.raw = json.dumps({} if payload is None else payload).encode() if raw is None else raw
        self.match_info = info or {}
        self.headers = {"Content-Length": str(len(self.raw)), "X-VNCCS-CSRF": "1", **(headers or {})}
        self.content_length = len(self.raw)
        self.content_type = "multipart/form-data" if multipart is not None else "application/json"
        self.multipart = multipart

    async def read(self):
        return self.raw

    async def post(self):
        return self.multipart


@pytest.fixture
def routes(workspace, monkeypatch):
    monkeypatch.setattr(routes_module, "_REGISTERED", False)
    monkeypatch.setattr(sys.modules["aiohttp"], "web", web)
    table = web.RouteTableDef()
    routes_module.register_routes(table)
    return {(route.method, route.path): route.handler for route in table}


def call(routes, method, suffix, request):
    return asyncio.run(routes[(method, service.API_BASE + suffix)](request))


def test_http_create_load_save_and_archive_ignore_stale_workflow_revision(routes, workspace):
    response = call(routes, "POST", "/backgrounds", Request({"name": "HTTP room", "editor": {"description": "Old scene"}}))
    assert response.status == 200
    document = json.loads(response.text)
    info = {"background_id": document["background_id"]}
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
        Request({"revision": document["revision"], "editor": {"description": "Updated scene"}}, info=info))
    assert response.status == 200
    updated = json.loads(response.text)
    assert updated["editor"]["description"] == "Updated scene"
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
        Request({"revision": document["revision"], "editor": {}}, info=info))
    assert response.status == 200
    updated = json.loads(response.text)
    assert updated["editor"] == generation.normalize_editor({})
    response = call(routes, "GET", "/backgrounds/{background_id}", Request(info=info))
    assert json.loads(response.text)["revision"] == updated["revision"]
    response = call(routes, "DELETE", "/backgrounds/{background_id}", Request({"revision": document["revision"]}, info=info))
    assert response.status == 200 and json.loads(response.text)["archived"] is True
    response = call(routes, "GET", "/backgrounds/{background_id}", Request(info=info))
    assert response.status == 200 and json.loads(response.text)["archived"] is True


@pytest.mark.parametrize("revision", [None, 0, 999])
def test_background_http_actions_ignore_missing_or_obsolete_workflow_revisions(routes, workspace, revision):
    document = job_with_image(workspace)
    info = {"background_id": document["background_id"]}
    metadata = {} if revision is None else {"revision": revision}
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
                    Request({**metadata, "editor": {"scene_description": "Current session"}}, info=info))
    assert response.status == 200
    assert json.loads(response.text)["editor"]["scene_description"] == "Current session"
    png = io.BytesIO()
    Image.new("RGB", (12, 6), "green").save(png, format="PNG")
    multipart = {"data": json.dumps(metadata), "image": types.SimpleNamespace(file=io.BytesIO(png.getvalue()))}
    response = call(routes, "POST", "/backgrounds/{background_id}/upload", Request(info=info, multipart=multipart))
    assert response.status == 200
    uploaded = json.loads(response.text)
    assert uploaded["images"][-1]["settings"]["source"] == "upload"
    with Image.open(storage.image_path(document["background_id"], uploaded["selected_image_id"])) as image:
        assert image.size == (12, 6)
    response = call(routes, "DELETE", "/backgrounds/{background_id}", Request(metadata, info=info))
    assert response.status == 200 and json.loads(response.text)["archived"] is True


def test_http_save_preserves_legacy_selected_image_references(routes, workspace):
    document = job_with_image(workspace)
    document["images"][0].pop("preview_revision")
    document_path(document).write_text(json.dumps(document))
    first_id = document["selected_image_id"]
    document = storage.add_image(document["background_id"], Image.new("RGB", (128, 64), "green"),
                                 {"source": "upload"}, document["revision"])
    assert document["selected_image_id"] != first_id
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
        Request({"revision": document["revision"], "editor": document["editor"], "selected_image_id": first_id},
                info={"background_id": document["background_id"]}))
    assert response.status == 200
    assert json.loads(response.text)["selected_image_id"] == first_id
    assert storage.load_background(document["background_id"])["selected_image_id"] == first_id


@pytest.mark.parametrize("raw", [b"[]", b"null", b"false", b"{invalid"])
def test_http_invalid_json_returns_400_without_creating_documents(routes, workspace, raw):
    before = storage.list_backgrounds()
    response = call(routes, "POST", "/backgrounds", Request(raw=raw))
    assert response.status == 400 and json.loads(response.text)["error"]
    assert storage.list_backgrounds() == before
    workspace.worker.submit.assert_not_called()


def test_http_limits_reject_payload_before_processing(routes):
    request = Request(raw=b"{}", headers={"Content-Length": str(service.MAX_REQUEST_BYTES + 1)})
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        call(routes, "POST", "/backgrounds", request)
    response = call(routes, "POST", "/backgrounds", Request(raw=b" " * (service.MAX_JSON_BYTES + 1)))
    assert response.status == 400


def test_http_upload_persists_current_preview_without_browser_cache_staleness(routes, workspace):
    document = workspace.document
    info = {"background_id": document["background_id"]}
    form = {"data": json.dumps({"revision": document["revision"]}),
            "image": types.SimpleNamespace(file=io.BytesIO(png()))}
    response = call(routes, "POST", "/backgrounds/{background_id}/upload", Request(info=info, multipart=form))
    assert response.status == 200
    updated = json.loads(response.text)
    image = updated["images"][0]
    assert image["url"].endswith(f"/images/{image['image_id']}?v={image['preview_revision']}")
    response = call(routes, "GET", "/backgrounds/{background_id}/images/{image_id}",
                    Request(info={**info, "image_id": image["image_id"]}))
    assert isinstance(response, web.FileResponse)
    assert response.headers["Cache-Control"] == "no-store"
    response = call(routes, "GET", "/backgrounds/{background_id}/images/{image_id}",
                    Request(info={**info, "image_id": "../outside"}))
    assert response.status == 404


def test_http_job_actions_status_cancellation_and_busy_edit_guard(routes, workspace):
    document = workspace.document
    info = {"background_id": document["background_id"]}
    response = call(routes, "POST", "/backgrounds/{background_id}/jobs", Request(job_payload(document, "background"), info=info))
    assert response.status == 200
    queued = json.loads(response.text)
    response = call(routes, "GET", "/jobs/{job_id}", Request(info={"job_id": queued["job_id"]}))
    assert json.loads(response.text)["status"] == "queued"
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
                    Request({"revision": document["revision"], "editor": {}}, info=info))
    assert response.status >= 400
    assert storage.load_background(document["background_id"]) == document
    response = call(routes, "POST", "/jobs/{job_id}/cancel", Request(info={"job_id": queued["job_id"]}))
    assert response.status == 200
    assert service._JOBS[queued["job_id"]]["cancel_event"].is_set()
    job = workspace.run_pending()
    assert job["status"] == "cancelled"
    response = call(routes, "GET", "/jobs/{job_id}", Request(info={"job_id": "missing"}))
    assert response.status == 404


@pytest.mark.parametrize("raw", ["{", "[]", '{"schema_version":true}', '{"schema_version":3}',
    '{"schema_version":1}', '{"schema_version":1,"scene_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}', '{"schema_version":2,"selected_image_id":{}}',
    '{"schema_version":2,"operation":"generate"}', '{"schema_version":2,"operation":"import_connected"}'])
def test_invalid_workflow_never_executes(raw):
    assert storage.VNCCS_BackgroundCreator.VALIDATE_INPUTS(raw) is not True


def test_node_outputs_the_uploaded_background_without_inference(library, monkeypatch):
    node = storage.VNCCS_BackgroundCreator()
    inputs = node.INPUT_TYPES()
    assert list(inputs["required"]) == ["seed", "background_data"]
    assert inputs["required"]["background_data"][1]["dynamicPrompts"] is False
    assert "image" not in inputs.get("optional", {}) and "image" not in inputs["required"]
    assert node.RETURN_TYPES == ("IMAGE", "STRING") and node.RETURN_NAMES == ("image", "Prompt")
    assert node.OUTPUT_IS_LIST == (False, True)
    document = storage.create_background("Output", generation.normalize_editor({}))
    document = storage.add_image(document["background_id"], Image.new("RGB", (160, 80), "blue"), {"source": "upload"}, document["revision"])
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda pixels:
        types.SimpleNamespace(unsqueeze=lambda axis: np.expand_dims(pixels, axis))))
    inference = mock.Mock(side_effect=AssertionError("Uploaded backgrounds must not run inference"))
    monkeypatch.setattr(storage, "generate_image", inference)
    raw = json.dumps({"schema_version": 2, "background_id": document["background_id"]})
    result = node.create_background(background_data=raw)
    tensor, prompts = result["result"]
    assert len(prompts) == 3 and tensor.shape == (1, 80, 160, 3)
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor[0, 0, 0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="Select or create"):
        node.create_background()
    inference.assert_not_called()


def test_preview_uses_real_torch_rgb_batch_when_available(library):
    torch = pytest.importorskip("torch")
    document = storage.create_background("Real tensor")
    document = storage.add_image(document["background_id"], Image.new("RGB", (7, 5), (255, 128, 0)),
                                 {"source": "upload"}, document["revision"])
    result = storage.VNCCS_BackgroundCreator().create_background(background_data=json.dumps({
        "schema_version": 2, "background_id": document["background_id"]}))["result"][0]
    assert isinstance(result, torch.Tensor) and result.dtype == torch.float32
    assert tuple(result.shape) == (1, 5, 7, 3)
    torch.testing.assert_close(result[0, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))


def test_removed_import_operation_cannot_change_saved_backgrounds(library):
    document = storage.create_background("Retained scene")
    before = files(library)
    state = {"schema_version": 2, "background_id": document["background_id"],
             "operation": "import_connected", "import_token": "a" * 32}
    with pytest.raises(ValueError, match="Use Upload image"):
        storage.VNCCS_BackgroundCreator().create_background(background_data=json.dumps(state))
    assert files(library) == before


def background_run_state(document, **changes):
    return {"schema_version": 2, "background_id": document["background_id"], "selected_image_id": document["selected_image_id"],
            "revision": document["revision"], "editor": copy.deepcopy(document["editor"]), **changes}


def run_background(document, **changes):
    return storage.VNCCS_BackgroundCreator().create_background(background_data=json.dumps(background_run_state(document, **changes)), unique_id="42")


@pytest.mark.parametrize("model_type", ["qi2", "moss"])
def test_run_generates_once_then_reuses_preview_without_advancing_stage(workspace, model_type):
    document = workspace.document
    editor = generation.normalize_editor({"scene_description": "A bedroom", "generation": {"model_type": model_type}})
    first = run_background(document, editor=editor)
    assert first["ui"]["background_preview"][0]["source"] == "generated"
    assert first["result"][0].shape == (1, 64, 128, 3)
    saved = storage.load_background(document["background_id"])
    assert saved["editor"]["scene_description"] == "A bedroom"
    assert saved["images"][-1]["settings"]["cache_key"] == generation._background_cache_key(editor)
    image_bytes = storage.image_path(saved["background_id"], saved["selected_image_id"]).read_bytes()
    second = run_background(saved, editor=editor)
    assert second["ui"]["background_preview"][0]["source"] == "cache"
    np.testing.assert_array_equal(first["result"][0], second["result"][0])
    assert workspace.generated.call_count == 1
    assert storage.image_path(saved["background_id"], saved["selected_image_id"]).read_bytes() == image_bytes
    assert len(storage.load_background(saved["background_id"])["images"]) == 1
    assert not storage._JOBS


def test_widget_preview_and_run_share_the_same_cache(workspace):
    document = workspace.document
    service.start_job(document["background_id"], job_payload(document, "background"))
    job = workspace.run_pending()
    assert job["status"] == "completed", job["error"]
    saved = storage.load_background(document["background_id"])
    result = run_background(saved)
    assert result["ui"]["background_preview"][0]["source"] == "cache"
    workspace.generated.assert_called_once()


@pytest.mark.parametrize("change", ["prompt", "front", "back", "left", "right", "seed", "model", "lora", "weights", "enhancer", "suffix"])
def test_changed_generation_inputs_invalidate_run_cache_and_replace_current_preview(workspace, monkeypatch, change):
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    original_image = storage.image_path(document["background_id"], document["selected_image_id"])
    original_id = document["selected_image_id"]
    original_revision = document["images"][0]["preview_revision"]
    editor = copy.deepcopy(document["editor"])
    if change == "prompt": editor["scene_description"] = "A different room"
    if change in SCENE_COMPOSITION: editor[change] = SCENE_COMPOSITION[change]
    if change == "seed": editor["generation"]["seed"] += 1
    if change == "model": editor["generation"] = {"model_type": "moss"}
    if change == "lora": editor["generation"]["lora_stack"] = [{"name": "room.safetensors", "strength": 0.6}]
    if change == "weights": generation._resolve_model_path("diffusion_models", generation.ASSETS["diffusion_models"]).write_bytes(b"updated weights")
    if change == "enhancer": monkeypatch.setattr(generation, "BACKGROUND_ENHANCEMENT_SYSTEM", generation.BACKGROUND_ENHANCEMENT_SYSTEM + "Updated wording.")
    if change == "suffix": monkeypatch.setattr(generation, "BACKGROUND_PROMPT_SUFFIX", generation.BACKGROUND_PROMPT_SUFFIX + " Updated wording.")
    result = run_background(document, editor=editor)
    assert result["ui"]["background_preview"][0]["source"] == "generated"
    assert workspace.generated.call_count == 2
    updated = storage.load_background(document["background_id"])
    assert len(updated["images"]) == 1 and updated["selected_image_id"] == original_id
    assert updated["images"][0]["preview_revision"] > original_revision
    assert storage.image_path(document["background_id"], original_id) == original_image


def test_inactive_profiles_do_not_invalidate_background_cache(workspace):
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    editor = copy.deepcopy(document["editor"])
    editor["generation_profiles"] = {"moss": {"steps": 19}}
    result = run_background(document, editor=editor)
    assert result["ui"]["background_preview"][0]["source"] == "cache"
    workspace.generated.assert_called_once()
    saved = storage.load_background(document["background_id"])
    assert saved["editor"]["generation_profiles"]["moss"]["steps"] == 19
    assert "model_settings" not in saved


def test_uploaded_background_run_never_loads_models_and_saves_user_settings(workspace, monkeypatch):
    document = job_with_image(workspace)
    image_path = storage.image_path(document["background_id"], document["selected_image_id"])
    original = image_path.read_bytes()
    editor = generation.normalize_editor({"scene_description": "Authored room", "generation": {"model_type": "moss", "seed": 91}})
    monkeypatch.setattr(generation, "_background_cache_key", mock.Mock(side_effect=AssertionError("Uploads do not need model assets")))
    result = run_background(document, editor=editor)
    assert result["ui"]["background_preview"][0]["source"] == "upload"
    assert result["result"][0].shape == (1, 64, 128, 3)
    workspace.generated.assert_not_called()
    saved = storage.load_background(document["background_id"])
    assert saved["editor"]["scene_description"] == "Authored room" and saved["editor"]["generation"]["seed"] == 91
    assert saved["images"] == document["images"]
    assert image_path.read_bytes() == original
    signature = storage.VNCCS_BackgroundCreator.IS_CHANGED(background_data=json.dumps(background_run_state(saved)))
    assert signature


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_corrupt_generated_preview_is_regenerated(workspace, corrupt):
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    path = storage.image_path(document["background_id"], document["selected_image_id"])
    path.write_bytes(b"damaged PNG") if corrupt else path.unlink()
    assert run_background(document)["ui"]["background_preview"][0]["source"] == "generated"
    assert workspace.generated.call_count == 2


def test_failed_run_keeps_previous_preview_and_authored_json(workspace):
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    path = storage.image_path(document["background_id"], document["selected_image_id"])
    original = path.read_bytes()
    editor = copy.deepcopy(document["editor"]); editor["scene_description"] = "Keep my new description"
    workspace.generated.side_effect = RuntimeError("Decoder unavailable")
    with pytest.raises(RuntimeError, match="Decoder unavailable"):
        run_background(document, editor=editor)
    saved = storage.load_background(document["background_id"])
    assert saved["editor"]["scene_description"] == "Keep my new description"
    assert saved["images"] == document["images"] and path.read_bytes() == original
    assert not storage._JOBS


def test_newer_metadata_during_inference_cannot_be_overwritten_by_run(workspace):
    document = workspace.document
    def generate(*_args, **_kwargs):
        current = storage.load_background(document["background_id"])
        storage.save_background(document["background_id"], {"scene_description": "Newer authored data"}, current["revision"])
        return Image.new("RGB", (128, 64)), {}
    workspace.generated.side_effect = generate
    with pytest.raises(storage.BackgroundConflictError, match="changed"):
        run_background(document)
    saved = storage.load_background(document["background_id"])
    assert saved["editor"]["scene_description"] == "Newer authored data"
    assert saved["images"] == [] and not storage._JOBS


def test_workflow_run_refuses_busy_documents_before_writing_or_generating(workspace):
    service.start_job(workspace.document["background_id"], job_payload(workspace.document, "wizard"))
    before = files(storage.root())
    with pytest.raises(storage.BackgroundConflictError, match="busy"):
        run_background(workspace.document)
    assert files(storage.root()) == before
    workspace.generated.assert_not_called()


def test_comfy_cache_signature_detects_disk_changes_with_unchanged_workflow(workspace):
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    raw = json.dumps(background_run_state(document))
    before = storage.VNCCS_BackgroundCreator.IS_CHANGED(background_data=raw)
    generation._resolve_model_path("diffusion_models", generation.ASSETS["diffusion_models"]).write_bytes(b"new weights")
    assert storage.VNCCS_BackgroundCreator.IS_CHANGED(background_data=raw) != before


def test_cross_origin_mutations_fail_before_reading_request(routes, workspace):
    request = Request({"name": "Forbidden"}, headers={"Sec-Fetch-Site": "cross-site"})
    response = call(routes, "POST", "/backgrounds", request)
    assert response.status == 403
    assert [item["name"] for item in storage.list_backgrounds()] == [workspace.document["name"]]


def test_model_path_resolution_rejects_ambiguous_and_escaping_names(monkeypatch):
    monkeypatch.setattr(storage, "_installed", lambda _: ["one/weight.safetensors", "two/weight.safetensors"])
    with pytest.raises(FileNotFoundError, match="exactly one"):
        storage._resolve_model_name("vae", "weight.safetensors")
    assert storage._resolve_model_name("vae", "one/weight.safetensors") == "one/weight.safetensors"
    with pytest.raises(ValueError):
        storage._resolve_model_name("vae", "../weight.safetensors")


def test_feature_imports_are_local_and_do_not_load_model_dependencies():
    tree = ast.parse(Path(storage.__file__).read_text())
    imports = [entry.module or "" for entry in ast.walk(tree) if isinstance(entry, ast.ImportFrom)]
    assert not any("factory3d" in name or "unicanvas" in name or "VNCCS_Utils" in name for name in imports)
    top = [name.name for entry in tree.body if isinstance(entry, ast.Import) for name in entry.names]
    assert "torch" not in top and "torchvision" not in top
    assert not any(name.startswith("data.") for name in imports)
    base = Path(storage.__file__).parents[1]
    assert not (base / "data" / "triposplat" / "triposplat.py").exists()
    assert not (base / "web" / "vendor" / "spark" / "spark.module.js").exists()
    for name in ("_reconstruct", "_triposplat_pipeline", "inspect_ply", "ply_to_splat", "commit_model", "commit_cg", "public_model", "_migrate_layout", "_copy_model", "_legacy_scene", "import_legacy", "_camera"):
        assert not hasattr(storage, name)


def test_full_prompt_override_preserves_legacy_description():
    editor = generation.normalize_editor({"description": "Original authored room description",
        "prompt": "My exact finished environment prompt.", "attributes": {"time": "Night"}})
    assert generation.compose_prompt(editor).splitlines()[-1] == "Scene description: My exact finished environment prompt."
    assert editor["description"] == "Original authored room description"
    assert "Original authored room description" in generation.compose_prompt({"description": editor["description"]})


def test_selected_diffusion_model_rekeys_cache_and_preserves_five_lora_slots(runtime):
    settings = {"prompt_enhance": False, "diffusion_model_name": "custom/rooms.safetensors",
        "lora_stack": [{"name": "", "strength": 0.4}, {"name": "room.safetensors", "strength": 0.7},
                       {"name": "", "strength": 1}, {"name": "", "strength": 1}, {"name": "", "strength": 1}]}
    editor = generation.normalize_editor({"generation": settings})
    assert len(editor["generation"]["lora_stack"]) == 5
    assert editor["generation"]["lora_stack"][1]["name"] == "room.safetensors"
    generation.generate_image({"prompt": "An exact room prompt.", "generation": settings}, "custom", threading.Event())
    assert [call["unet_name"] for name, call in runtime.calls if name == "UNETLoader"] == ["custom/rooms.safetensors"]
    assert [call["lora_name"] for name, call in runtime.calls if name == "LoraLoader"] == ["room.safetensors"]
    assert dict(runtime.calls)["TextEncodeQwenImage21"]["prompt"].splitlines()[-1] == f"Scene description: An exact room prompt. {generation.BACKGROUND_PROMPT_SUFFIX}"
    generation.generate_image({"generation": {"prompt_enhance": False}}, "default", threading.Event())
    assert len([name for name, _ in runtime.calls if name == "UNETLoader"]) == 2


def test_turbo_model_disables_second_adapter_and_preserves_explicit_steps(runtime):
    _image, metadata = generation.generate_image({"generation": {
        "diffusion_model_name": "qwen_image_2.1_turbo_int8_convrot.safetensors", "turbo": True, "steps": 10, "cfg": 1.5}},
        "built-in-turbo", threading.Event())
    settings = metadata["editor"]["generation"]
    assert (settings["turbo"], settings["steps"], settings["cfg"]) == (False, 10, 1.5)
    assert "KSampler" in [name for name, _ in runtime.calls]
    assert "SamplerCustomAdvanced" not in [name for name, _ in runtime.calls]


def test_requested_four_line_prompt_and_legacy_scene_migration():
    description = "A large, circular stone table sits in a spacious futuristic room with reflective water and purple flowers."
    editor = generation.normalize_editor({"style": "Anime, digital art, cel shaded", "scene_description": description,
        "attributes": {"time": "Night", "scene_type": "Interior"}})
    expected = f"Style: Anime, digital art, cel shaded\nTime: Day\nInterior/Exterior: Interior\nScene description: {description}"
    assert generation.compose_prompt(editor) == expected
    restored = generation.normalize_editor({"prompt": expected})
    assert generation.compose_prompt(restored) == expected
    old = {"description": "Original authored room", "attributes": {"architecture": "stone arches", "lighting": "blue ceiling panels"}}
    migrated = generation.normalize_editor(old)
    assert migrated["description"] == old["description"]
    assert migrated["attributes"]["lighting"] == "blue ceiling panels"
    assert "stone arches" in migrated["scene_description"]
    assert generation.normalize_editor(migrated) == migrated
    migrated["scene_description"] = ""
    assert generation.compose_prompt(migrated).splitlines()[-1] == "Scene description: "


@pytest.mark.parametrize("model_type", ["qi2", "moss"])
@pytest.mark.parametrize("prefilled", [False, True])
def test_prompt_enhancement_expands_all_scene_fields_and_keeps_scene_settings(runtime, monkeypatch, model_type, prefilled):
    editor = {"style": "Anime, digital art, cel shaded", "scene_description": "A futuristic circular room.",
              "attributes": {"time": "Night", "scene_type": "Interior"},
              "generation": {"model_type": model_type}, **(SCENE_COMPOSITION if prefilled else {})}
    original = copy.deepcopy(editor)
    runtime.decoded_size[0] = generation.dimensions(editor)
    expanded = {key: value + " A secondary structure continues behind it." for key, value in SCENE_COMPOSITION.items()}
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (generation.compose_prompt({"style": "Wrong style", "attributes": {"scene_type": "Exterior"},
                "scene_description": "An enhanced futuristic circular room.", **expanded}),)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    _, metadata = generation.generate_image(editor, "structured-enhancement", threading.Event())
    prompt = next(values["text"] if model_type == "moss" else values["prompt"] for name, values in runtime.calls
                  if name == ("CLIPTextEncode" if model_type == "moss" else "TextEncodeQwenImage21"))
    expected = generation.compose_prompt({**editor, "scene_description": "An enhanced futuristic circular room.", **expanded})
    assert prompt == metadata["prompt"] == expected + f" {generation.BACKGROUND_PROMPT_SUFFIX}"
    request = dict(runtime.calls)["TextGenerate"]
    assert request["prompt"] == generation.compose_prompt(editor)
    assert "Return exactly eight lines" in request["system_prompt"]
    assert "Fill every scene field without inventing unrelated focal objects or decorative motifs" in request["system_prompt"]
    assert "preserving its objects, positions and constraints" in request["system_prompt"]
    assert "Describe only physical scene content, objects and spatial relationships" in request["system_prompt"]
    assert "Never repeat, paraphrase or apply Style inside" in request["system_prompt"]
    assert "light sources and shadows, colors and atmosphere" not in request["system_prompt"]
    assert editor == original and metadata["editor"] == generation.normalize_editor(original)


@pytest.mark.parametrize("description", ["A futuristic room", ""])
def test_wizard_returns_all_scene_fields_and_accepts_composition_only_briefs(runtime, monkeypatch, description):
    fields = {"style": "Anime, digital art, cel shaded", "time": "Night", "scene_type": "Interior",
              "scene_description": "A room with a circular stone table.", **SCENE_COMPOSITION}
    supplied = {"front": SCENE_COMPOSITION["front"]}
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (json.dumps(fields),)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    editor = generation.wizard(description, supplied, "scene-wizard", threading.Event())
    assert editor["scene_description"] == fields["scene_description"]
    assert editor["style"] == fields["style"]
    assert editor["attributes"] == {"time": "Day", "scene_type": "Interior"}
    assert {key: editor[key] for key in SCENE_COMPOSITION} == SCENE_COMPOSITION
    schema = json.loads(dict(runtime.calls)["TextGenerate"]["system_prompt"].split("not a finished answer: ")[1])
    if description:
        assert schema["scene_type"] == "" and schema["scene_description"] == description
        assert all(schema[key] == "" for key in SCENE_COMPOSITION)
    else:
        assert schema["scene_type"] == "Interior" and schema["scene_description"] == ""
        assert schema["front"] == supplied["front"] and all(schema[key] == "" for key in ("back", "left", "right"))
    assert generation.compose_prompt(editor).splitlines()[1] == "Time: Day"


def test_new_wizard_brief_does_not_reuse_a_saved_interior_scene(runtime, monkeypatch):
    brief = "street with sakura trees"
    previous = {"scene_description": "A furnished bedroom.", "attributes": {"scene_type": "Interior"},
                **{key: f"Old bedroom {key}." for key in SCENE_COMPOSITION}}
    result = {"style": "Painted animation", "time": "Day", "scene_type": "Exterior",
              "scene_description": "An outdoor street lined with flowering trees.",
              "front": "The street enters the foreground.", "back": "The road continues into the distance.",
              "left": "Trees border the left side.", "right": "Building fronts line the right side."}

    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (json.dumps(result),)
        return runtime.call(name, **kwargs)

    monkeypatch.setattr(generation, "_call_comfy_node", call)
    editor = generation.wizard(brief, previous, "fresh-scene-wizard", threading.Event())
    request = dict(runtime.calls)["TextGenerate"]
    schema = json.loads(request["system_prompt"].split("not a finished answer: ")[1])
    assert request["prompt"] == brief
    assert schema["scene_description"] == brief and schema["scene_type"] == ""
    assert all(schema[key] == "" for key in SCENE_COMPOSITION)
    assert "furnished bedroom" not in request["system_prompt"].lower()
    assert editor["attributes"]["scene_type"] == "Exterior"
    assert editor["scene_description"] == result["scene_description"]


@pytest.mark.parametrize("use_wizard", [False, True])
def test_sparse_scene_prompts_request_detail_and_preserve_field_contract(runtime, monkeypatch, use_wizard):
    editor = {"style": "Anime, digital art, cel shaded", "scene_description": "badroom",
              "attributes": {"time": "Night", "scene_type": "Interior"}}
    original = copy.deepcopy(editor)
    description = (
        "A bedroom is organized around a low bed placed against the back wall, with a bedside table on each side. "
        "A woven rug crosses the foreground and leaves a clear walking route from the doorway to the bed. "
        "Tall shelves containing books, ceramic bowls and folded blankets occupy the left wall. "
        "A broad window is centered behind the bed, with curtains hanging on both sides. "
        "The right side contains a narrow writing desk, a chair and stacked notebooks. "
        "A wardrobe stands near the entrance without blocking the central path."
    )
    expected = generation.compose_prompt({**editor, "scene_description": description, **SCENE_COMPOSITION})
    fields = {"style": editor["style"], "time": "Night", "scene_type": "Interior", "scene_description": description, **SCENE_COMPOSITION}
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (json.dumps(fields) if use_wizard else expected,)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    if use_wizard:
        result = generation.wizard("badroom", editor, "sparse-wizard", threading.Event())
        assert generation.compose_prompt(result) == expected
    else:
        _, metadata = generation.generate_image(editor, "sparse-enhancement", threading.Event())
        assert dict(runtime.calls)["TextEncodeQwenImage21"]["prompt"] == expected + f" {generation.BACKGROUND_PROMPT_SUFFIX}"
        assert metadata["editor"]["scene_description"] == "badroom"
    request = dict(runtime.calls)["TextGenerate"]
    assert "badroom" in request["prompt"]
    static_prompt = request["system_prompt"].split("The following object supplies")[0]
    assert "obvious typos from context" in static_prompt and "bedroom" not in static_prompt
    assert "80-140 words" in request["system_prompt"]
    assert "foreground/midground/background" in request["system_prompt"]
    assert "Style" in request["system_prompt"] and "Interior/Exterior" in request["system_prompt"]
    assert "Do not invent or elaborate visual style, rendering technique, lighting" in request["system_prompt"]
    assert "Style is the only field that controls aesthetics and rendering" in request["system_prompt"]
    assert "light sources and shadows, colors and atmosphere" not in request["system_prompt"]
    assert "surfaces, depth and lighting" not in request["system_prompt"]
    if use_wizard:
        assert "exactly eight string keys in this order: style, time, scene_type, scene_description, front, back, left, right" in request["system_prompt"]
        assert json.loads(request["system_prompt"].split("not a finished answer: ")[1]) == {
            **fields, "time": "Day", "scene_type": "", "scene_description": "badroom",
            **{key: "" for key in SCENE_COMPOSITION}}
    else:
        assert "Keep Style, Time and Interior/Exterior exactly as supplied" in request["system_prompt"]
        assert request["prompt"] == generation.compose_prompt(editor)
    assert editor == original


def test_variants_default_to_three_empty_cards_and_base_prompt_always_uses_day():
    editor = generation.normalize_editor({"attributes": {"time": "Night"}})
    assert editor["attributes"]["time"] == "Night", "Legacy authored settings remain recoverable"
    assert generation.compose_prompt(editor).splitlines()[1] == "Time: Day"
    assert [card["name"] for card in editor["variants"]] == ["Morning", "Evening", "Night"]
    assert all(card["weather"] == card["additional_details"] == "" for card in editor["variants"])
    assert [card["time"] for card in editor["variants"]] == ["Morning", "Evening", "Night"]
    assert generation.normalize_editor(editor) == editor


@pytest.mark.parametrize("cards", [None, {}, [{}], [{"id": "../bad", "name": "Rain"}],
    [{"id": "rain", "name": "../Rain"}], [{"id": "rain", "name": "Rain", "time": False}],
    [{"id": "day", "name": "Day"}] * 2, [{"id": str(i), "name": "Day"} for i in range(65)]])
def test_variant_parameters_reject_malformed_or_unbounded_data(cards):
    with pytest.raises(ValueError):
        generation.normalize_editor({"variants": cards})


def test_variant_parameters_round_trip_in_background_json_and_prompt_list(library):
    editor = generation.normalize_editor({"scene_description": "My original room"})
    editor["variants"].append({"id": "rain", "name": "Rainy morning", "time": "Morning", "weather": "Light rain",
                               "additional_details": "Keep the curtains open.", "custom": {"authored": True}})
    document = storage.create_background("Variant room", editor)
    document = storage.add_image(document["background_id"], Image.new("RGB", (16, 8)), {"source": "upload"}, document["revision"])
    restored = storage.load_background(document["background_id"])
    prompts = storage.variant_output(restored, restored["editor"], restored["selected_image_id"])
    assert isinstance(prompts, list) and len(prompts) == 4 and all(isinstance(item, str) for item in prompts)
    last = json.loads(prompts[-1])
    assert last["background_id"] == document["background_id"] and last["source_image_id"] == document["selected_image_id"]
    assert last["prompt"].startswith("Change time of the day to morning, weather to light rain. Keep the curtains open.")
    assert last["custom"] == {"authored": True}
    assert restored["editor"] == editor


@pytest.mark.parametrize("remaining", [2, 0])
def test_deleted_variants_stay_deleted_in_json_and_prompt_without_removing_images(library, remaining):
    document = with_image("Deleted variants")
    cards = generation.normalize_editor(document["editor"])["variants"]
    storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
        [Image.new("RGB", (16, 8)) for _ in range(len(cards) + 1)], {}, document["revision"])
    document = storage.load_background(document["background_id"])
    before = files(library)
    editor = copy.deepcopy(document["editor"])
    editor["variants"] = cards[:remaining]
    storage.save_background(document["background_id"], editor, document["revision"])
    restored = storage.load_background(document["background_id"])
    assert restored["editor"] == editor
    assert restored["images"] == document["images"] and restored["generated_sets"] == document["generated_sets"]
    assert [json.loads(item)["id"] for item in storage.variant_output(restored, restored["editor"], restored["selected_image_id"])] == [card["id"] for card in cards[:remaining]]
    after = files(library)
    for path, contents in before.items():
        if path.endswith(".png"):
            assert after[path] == contents


def test_replacing_preview_preserves_sources_of_already_saved_background_sets(library):
    document = with_image("Saved set source")
    identity = document["selected_image_id"]
    original = storage.image_path(document["background_id"], identity).read_bytes()
    cards = generation.normalize_editor(document["editor"])["variants"]
    storage.publish_background_set(document["background_id"], identity, cards,
        [Image.new("RGB", (16, 8)) for _ in range(len(cards) + 1)], {}, document["revision"])
    document = storage.load_background(document["background_id"])
    sets = copy.deepcopy(document["generated_sets"])
    document = storage.add_image(document["background_id"], Image.new("RGB", (16, 8), "blue"), {}, document["revision"])
    current = document["selected_image_id"]
    assert current != identity
    document = storage.add_image(document["background_id"], Image.new("RGB", (16, 8), "green"), {}, document["revision"])
    assert len(document["images"]) == 2 and document["selected_image_id"] == current
    assert document["generated_sets"] == sets
    assert storage.image_path(document["background_id"], identity).read_bytes() == original


def test_background_sets_publish_current_images_version_previous_images_and_same_json(library, monkeypatch):
    document = with_image("Published variants")
    before = files(library)
    cards = generation.normalize_editor({})["variants"]
    images = [Image.new("RGB", (32 + index, 16), (index * 40, 0, 0)) for index in range(len(cards) + 1)]
    settings = {"upscaler": {"mode": "off"}, "sampler": {"seed": 42}}
    paths = storage.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, settings, document["revision"])
    saved = storage.load_background(document["background_id"])
    assert saved["images"] == document["images"] and saved["editor"] == document["editor"]
    assert saved["generator_settings"] == settings
    assert saved["generated_sets"][0]["variants"] == cards
    assert [path.name for path in paths] == ["00_Original.png", "01_Morning.png", "02_Evening.png", "03_Night.png"]
    for path, image in zip(paths, images):
        assert path.is_relative_to(library / document["name"] / "Images")
        with Image.open(path) as loaded:
            assert loaded.size == image.size
    assert (library / document["name"] / "source.png").read_bytes() == before[f"{document['name']}/source.png"]
    before_failure = files(library)
    commit = storage._commit
    monkeypatch.setattr(storage, "_commit", mock.Mock(side_effect=OSError("Disk full")))
    with pytest.raises(OSError, match="Disk full"):
        storage.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, settings, saved["revision"])
    assert files(library) == before_failure
    monkeypatch.setattr(storage, "_commit", commit)
    with pytest.raises(storage.BackgroundConflictError):
        storage.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, settings, document["revision"])
    assert files(library) == before_failure
    more = storage.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, settings, saved["revision"])
    assert more == paths and all(path.is_file() for path in paths)
    updated = storage.load_background(document["background_id"])
    assert [group["version"] for group in updated["generated_sets"]] == ["V1", ""]
    for index, path in enumerate(paths):
        archived = path.parent / "V1" / path.name
        assert archived.read_bytes() == before_failure[path.relative_to(library).as_posix()]
        assert storage.background_set_image_path(document["background_id"], saved["generated_sets"][0]["set_id"], index) == archived


def test_publishing_background_sets_does_not_expire_the_creators_comfy_cache(library, monkeypatch):
    document = storage.create_background("Stable source", generation.normalize_editor({}))
    document = storage.add_image(document["background_id"], Image.new("RGB", (32, 16)), {"source": "upload"}, document["revision"])
    raw = json.dumps({"schema_version": 2, "step": "background", "background_id": document["background_id"],
                      "selected_image_id": document["selected_image_id"], "editor": document["editor"]})
    before = storage.VNCCS_BackgroundCreator.IS_CHANGED(background_data=raw)
    storage.publish_background_set(document["background_id"], document["selected_image_id"], document["editor"]["variants"],
        [Image.new("RGB", (32, 16)) for _ in range(len(document["editor"]["variants"]) + 1)], {"upscaler": {"mode": "off"}}, document["revision"])
    assert storage.VNCCS_BackgroundCreator.IS_CHANGED(background_data=raw) == before
    updated = storage.load_background(document["background_id"])
    editor = copy.deepcopy(updated["editor"]); editor["variants"][0]["weather"] = "Light rain"
    storage.save_background(document["background_id"], editor, updated["revision"])
    assert storage.VNCCS_BackgroundCreator.IS_CHANGED(background_data=raw) != before


@pytest.mark.parametrize("model_type", ["qi2", "moss"])
@pytest.mark.parametrize("enhance", [False, True])
def test_complete_prompts_are_logged_at_info_before_encoding(runtime, monkeypatch, caplog, model_type, enhance):
    caplog.set_level(logging.INFO, logger="VNCCS")
    description = "Preserve every original bedroom furnishing and its layout. " * 16
    enhanced = "A richly detailed bedroom with a blue wardrobe, patterned curtains and a wooden bed. " * 12
    negative = ("No people, text or logos. " * 20).strip()
    editor = {"style": "Anime, cel shaded", "scene_description": description, "negative": negative,
              "generation": {"model_type": model_type, "prompt_enhance": enhance}}
    runtime.decoded_size[0] = generation.dimensions(editor)
    raw_result = "Style: Wrong style\nTime: Night\nInterior/Exterior: Exterior\nScene description: " + enhanced.strip()
    raw_result += "\n" + "\n".join(f"{key.title()}: {value}" for key, value in SCENE_COMPOSITION.items())
    expected = generation.compose_prompt({**editor, "scene_description": enhanced if enhance else description,
        **(SCENE_COMPOSITION if enhance else {})})
    expected += " Wide landscape composition, 16:9. Background only, no people or characters, no text or logos, no watermark, no border, no collage."
    def call(name, **kwargs):
        if name == "TextGenerate":
            logged = [record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "text_generation_input"]
            assert len(logged) == 1
            assert logged[0].vnccs["prompt"] == kwargs["prompt"] == generation.compose_prompt(editor)
            assert generation.BACKGROUND_PROMPT_SUFFIX not in kwargs["prompt"]
            assert logged[0].vnccs["system_prompt"] == kwargs["system_prompt"]
            runtime.calls.append((name, kwargs))
            return ("<think>Private reasoning</think>\n" + raw_result,)
        if name == "TextEncodeQwenImage21" or name == "CLIPTextEncode" and kwargs["text"] == expected:
            records = [record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompts"]
            assert len(records) == 1, "The actual inference prompt must be visible before encoding starts"
            record = records[0]
            assert record.levelno == logging.INFO
            assert record.vnccs["positive"] == expected
            assert record.vnccs["negative"] == negative
            assert record.vnccs["node_id"] == "logged-background"
            assert expected in record.getMessage() and negative in record.getMessage()
            assert "Positive prompt:\n" in record.getMessage() and "Negative prompt:\n" in record.getMessage()
            assert len(record.vnccs["positive"]) > 240 and len(record.vnccs["negative"]) > 240
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    _, metadata = generation.generate_image(editor, "logged-background", threading.Event())
    records = [record for record in caplog.records if hasattr(record, "vnccs")]
    outputs = [record for record in records if record.vnccs["event"] == "text_generation_output"]
    assert len(outputs) == int(enhance)
    if enhance:
        assert outputs[0].vnccs["output"] == raw_result
        assert raw_result in outputs[0].getMessage() and "Private reasoning" not in outputs[0].getMessage()
    assert metadata["prompt"] == expected
    assert metadata["prompt"].count(generation.BACKGROUND_PROMPT_SUFFIX) == 1


def test_wizard_logs_its_system_prompt_brief_and_complete_result(runtime, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="VNCCS")
    result = json.dumps({"style": "Anime", "time": "Day", "scene_type": "Interior",
                         "scene_description": "A spacious bedroom with a central bed and open shelves. " * 20})
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (result,)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    generation.wizard("badroom", {}, "wizard-node", threading.Event())
    records = [record for record in caplog.records if hasattr(record, "vnccs")]
    request, output = [record for record in records if record.vnccs["event"].startswith("text_generation_")]
    assert request.vnccs["context"] == "Background wizard" and request.vnccs["node_id"] == "wizard-node"
    assert request.vnccs["prompt"] == "badroom"
    assert request.vnccs["system_prompt"] == dict(runtime.calls)["TextGenerate"]["system_prompt"]
    assert output.vnccs["output"] == result and result in output.getMessage()


def test_cached_run_logs_the_saved_enhanced_prompt_without_running_enhancer(workspace, monkeypatch, caplog):
    prompt = generation.compose_prompt({"scene_description": "An enhanced bedroom with detailed wooden furniture. " * 20})
    negative = "No text or characters. " * 20
    workspace.generated.return_value = (Image.new("RGB", (128, 64)), {"prompt": prompt, "negative": negative})
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    caplog.clear(); caplog.set_level(logging.INFO, logger="VNCCS")
    result = run_background(document)
    records = [record for record in caplog.records if hasattr(record, "vnccs")]
    actual = next(record for record in records if record.vnccs["event"] == "prompts")
    assert actual.vnccs["context"] == "Cached background"
    assert actual.vnccs["positive"] == prompt and actual.vnccs["negative"] == negative
    assert prompt in actual.getMessage() and negative in actual.getMessage()
    output = next(record.vnccs for record in records if record.vnccs["event"] == "background_output")
    assert output["source"] == "cache" and output["inference_skipped"] is True
    assert result["ui"]["background_preview"][0]["source"] == "cache"
    assert workspace.generated.call_count == 1


def test_uploaded_run_logs_inference_skip_without_claiming_to_encode_a_prompt(workspace, caplog):
    document = storage.add_image(workspace.document["background_id"], Image.new("RGB", (128, 64)),
                                 {"source": "upload"}, workspace.document["revision"])
    caplog.set_level(logging.INFO, logger="VNCCS")
    run_background(document)
    records = [record.vnccs for record in caplog.records if hasattr(record, "vnccs")]
    assert not any(record["event"] in {"prompts", "text_generation_input", "text_generation_output"} for record in records)
    output = next(record for record in records if record["event"] == "background_output")
    assert output["source"] == "upload" and output["inference_skipped"] is True
    workspace.generated.assert_not_called()


def test_explicit_saved_day_variant_remains_authored_data():
    editor = generation.normalize_editor({})
    day = {"id": "day", "name": "Day", "time": "Day", "weather": "Rain", "additional_details": "Keep the lights on."}
    editor["variants"].append(day)
    assert generation.normalize_editor(editor)["variants"][-1] == day


def test_saved_sets_publish_only_after_commit_and_serve_fresh_owned_images(routes, workspace, monkeypatch):
    document = job_with_image(workspace)
    cards = generation.normalize_editor({})["variants"]
    events = []
    def publish(name, detail):
        saved = storage.load_background(detail["background_id"])
        assert saved["revision"] == detail["revision"]
        assert saved["generated_sets"][-1]["set_id"] == detail["set_id"]
        assert all(storage.background_set_image_path(detail["background_id"], detail["set_id"], index).is_file()
                   for index in range(len(cards) + 1))
        events.append((name, detail))
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(send_sync=publish))))
    storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
        [Image.new("RGB", (32, 16), (index * 40, 0, 0)) for index in range(len(cards) + 1)], {}, document["revision"])
    assert len(events) == 1 and events[0][0] == "vnccs.background.set"
    saved = storage.load_background(document["background_id"])
    public = storage.public_document(saved)
    group = public["generated_sets"][-1]
    assert group["source_image_id"] == document["selected_image_id"]
    for index, image in enumerate(group["images"]):
        assert image["url"].endswith(f"/sets/{group['set_id']}/{index}?v={image['revision']}")
        info = {"background_id": document["background_id"], "set_id": group["set_id"], "index": str(index)}
        result = call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info))
        assert result.status == 200 and result.headers["Cache-Control"] == "no-store"
        assert result.path == storage.background_set_image_path(document["background_id"], group["set_id"], index)
    for invalid in ["-1", "99", "not-an-index"]:
        info["index"] = invalid
        assert call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info)).status == 404
    info.update(set_id="f" * 32, index="1")
    assert call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info)).status == 404
    info["background_id"] = storage.create_background("Other set owner", {})["background_id"]
    info["set_id"] = group["set_id"]
    assert call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info)).status == 404
    group = saved["generated_sets"][-1]
    group["images"][1]["file"] = "source.png"
    monkeypatch.setattr(storage, "_locate", lambda _id: (storage.root() / saved["name"], saved, storage._json_bytes(saved)))
    with pytest.raises(ValueError, match="outside its saved set"):
        storage.background_set_image_path(document["background_id"], group["set_id"], 1)
    group["images"][1]["file"] = "Images/01_Morning.png"
    monkeypatch.setattr(storage, "_commit", mock.Mock(side_effect=OSError("Disk full")))
    with pytest.raises(OSError, match="Disk full"):
        storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
            [Image.new("RGB", (32, 16)) for _ in range(len(cards) + 1)], {}, saved["revision"])
    assert len(events) == 1, "Failed or partial sets must never be announced"


def test_untouched_legacy_day_is_removed_without_mutating_or_losing_authored_variants():
    plain = {"id": "day", "name": "Day", "time": "Day", "weather": "", "additional_details": ""}
    data = {"variants": [*generation.normalize_editor({})["variants"], plain]}
    before = copy.deepcopy(data)
    assert [card["name"] for card in generation.normalize_editor(data)["variants"]] == ["Morning", "Evening", "Night"]
    assert data == before
    for authored in [{**plain, "weather": "Rain"}, {**plain, "additional_details": "Lamps on"},
                     {**plain, "id": "custom-day"}, {**plain, "custom": {"retained": True}}]:
        assert generation.normalize_editor({"variants": [authored]})["variants"] == [authored]


def test_routes_have_no_model_cg_or_legacy_endpoints(routes):
    assert routes
    for method, path in routes:
        assert not any(part in path.split("/") for part in ("models", "cg", "legacy"))


@pytest.mark.parametrize("legacy", [False, True])
def test_same_workflow_runs_repeatedly_after_disk_settings_change(workspace, legacy):
    document = job_with_image(workspace)
    state = background_run_state(document)
    if legacy:
        state.pop("revision")
    raw = json.dumps(state)
    original_image = storage.image_path(document["background_id"], document["selected_image_id"]).read_bytes()
    for _ in range(2):
        newer = copy.deepcopy(document["editor"])
        newer["scene_description"] = "Saved settings from a previous session"
        storage.save_background(document["background_id"], newer)
        result = storage.VNCCS_BackgroundCreator().create_background(background_data=raw)
        assert result["ui"]["background_preview"][0]["source"] == "upload"
        saved = storage.load_background(document["background_id"])
        assert generation.normalize_editor(saved["editor"]) == state["editor"]
        assert storage.image_path(document["background_id"], document["selected_image_id"]).read_bytes() == original_image
        assert json.dumps(state) == raw
    workspace.generated.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_matching_workflow_can_randomize_seed_and_restore_missing_preview(workspace, monkeypatch, legacy):
    editor = copy.deepcopy(workspace.document["editor"])
    editor["generation"]["seed_mode"] = "randomize"
    document = storage.save_background(workspace.document["background_id"], editor, workspace.document["revision"])
    run_background(document, _vnccs_seed_resolved=True)
    document = storage.load_background(document["background_id"])
    storage.image_path(document["background_id"], document["selected_image_id"]).unlink()
    state = background_run_state(document)
    if legacy:
        state.pop("revision")
    else:
        state["revision"] -= 1
    monkeypatch.setattr(sys.modules["_vnccs.utils"], "generate_seed", lambda _: 123)
    result = storage.VNCCS_BackgroundCreator().create_background(background_data=json.dumps(state))
    saved = storage.load_background(document["background_id"])
    assert result["ui"]["background_preview"][0]["source"] == "generated"
    assert saved["editor"]["generation"]["seed"] == 123
    assert storage.image_path(saved["background_id"], saved["selected_image_id"]).is_file()
    assert workspace.generated.call_count == 2


def test_background_can_be_saved_without_a_workflow_revision(library):
    document = with_image()
    image = storage.image_path(document["background_id"], document["selected_image_id"])
    before = image.read_bytes()
    saved = storage.save_background(document["background_id"], {"prompt": "Current workflow settings"})
    assert saved["editor"]["prompt"] == "Current workflow settings"
    assert saved["revision"] == document["revision"] + 1
    assert image.read_bytes() == before


@pytest.mark.parametrize("revision", [None, True, 0, -1, 1.5, "2"])
def test_obsolete_workflow_revision_metadata_cannot_block_validation(revision):
    raw = json.dumps({"schema_version": 2, "revision": revision})
    assert storage.VNCCS_BackgroundCreator.VALIDATE_INPUTS(raw) is True


def test_workflow_reports_its_committed_revision(workspace):
    result = run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    assert result["ui"]["background_preview"][0]["revision"] == document["revision"]


def test_catalog_and_healthy_documents_survive_an_unreadable_neighbor(routes, workspace, monkeypatch):
    damaged = storage.create_background("Damaged neighbor", {})
    path = document_path(damaged)
    path.write_bytes(b"{broken")
    path.with_suffix(".json.backup").write_bytes(b"[]")
    healthy = workspace.document
    monkeypatch.setattr(storage, "_directories", lambda: [path.parent, document_path(healthy).parent])
    before = files(storage.root())
    result = call(routes, "GET", "/backgrounds", Request())
    assert result.status == 200
    catalog = json.loads(result.text)
    assert [item["background_id"] for item in catalog["backgrounds"]] == [healthy["background_id"]]
    assert catalog["errors"][0]["name"] == damaged["name"]
    assert "original files are preserved" in catalog["errors"][0]["error"]
    assert storage.load_background(healthy["background_id"]) == healthy
    assert files(storage.root()) == before
    changed = storage.save_background(healthy["background_id"], {"scene_description": "Healthy edit"}, healthy["revision"])
    assert changed["editor"]["scene_description"] == "Healthy edit"
    with pytest.raises(storage.BackgroundStorageError, match="original files are preserved"):
        storage.save_background(damaged["background_id"], {}, damaged["revision"])
    assert path.read_bytes() == b"{broken" and path.with_suffix(".json.backup").read_bytes() == b"[]"


@pytest.mark.parametrize("aspect, expected", [("9:16", (1080, 1920)), ("16:9", (1920, 1080))])
def test_resolution_matches_creator_scale_and_exact_aspect(aspect, expected):
    assert generation.dimensions({"aspect": aspect, "generation": {"resolution_mp": 2}}) == expected
    for model in ("qi2", "moss"):
        width, height = generation.dimensions({"aspect": aspect, "generation": {"model_type": model, "resolution_mp": 4}})
        a, b = map(int, aspect.split(":"))
        assert width * b == height * a
        if model == "moss":
            assert width % 16 == height % 16 == 0


def test_turbo_defaults_and_previous_settings_survive_profiles():
    settings = generation.normalize_editor({"generation": {
        "diffusion_model_name": "QI2/qwen_image_2.1_turbo_int8_convrot.safetensors"}})["generation"]
    assert (settings["steps"], settings["cfg"], settings["turbo"]) == (8, 1, False)
    previous = {"steps": 40, "cfg": 2.5}
    editor = generation.normalize_editor({"generation": {"turbo": True, "steps": 7, "cfg": 2,
        "seed_mode": "randomize", "turbo_previous_settings": previous}})
    assert editor["generation"]["turbo_previous_settings"] == previous
    assert editor["generation"]["seed_mode"] == "randomize"
    assert generation.normalize_editor(editor) == editor
    for invalid in ({"seed_mode": "bad"}, {"turbo_previous_settings": []}):
        with pytest.raises(ValueError):
            generation.normalize_editor({"generation": invalid})


def test_turbo_resolves_current_control_center_catalog_path(monkeypatch):
    from nodes import vnccs_control_center as center
    entry = {"name": "Qwen Image 2.1 Viggle Turbo", "kind": "QI2", "type": "TurboLora",
             "local_path": "models/loras/QI2/Viggle/current.safetensors"}
    monkeypatch.setattr(center, "_get_cc_config", lambda repo: {"lora": [entry]})
    monkeypatch.setattr(center, "_apply_active_installed_paths", lambda value: value)
    monkeypatch.setattr(generation, "_installed", lambda category: ["QI2/Viggle/current.safetensors"] if category == "loras" else [])
    assert generation._turbo_name() == "QI2/Viggle/current.safetensors"
    config = generation.configuration()
    assert config["turbo_lora"]["status"] == "installed"
    monkeypatch.setattr(generation, "_installed", lambda category: [])
    assert generation.configuration()["turbo_lora"]["status"] == "missing"


def test_background_catalog_uses_full_environment_prompts_and_keeps_work_references(monkeypatch):
    from nodes import character_styles
    monkeypatch.setattr(character_styles, "load_user_styles", lambda path=None: [])
    choices = generation.style_choices()
    assert len(choices) == 100
    ghibli = next(item for item in choices if item["id"] == "bg_soft_ghibli_watercolor")
    assert "Reference: Studio Ghibli; My Neighbor Totoro." in ghibli["prompt"]
    assert "supplied environment" in ghibli["prompt"]
    prompt = generation.compose_prompt({"style_id": ghibli["id"], "style": ghibli["prompt"]})
    assert "Reference: Studio Ghibli; My Neighbor Totoro." in prompt
    assert prompt.startswith("Style: Soft Ghibli Watercolor.")
    assert len(generation.normalize_editor({"style": "x" * 3000})["style"]) == 3000


def test_random_seed_resolves_once_for_preview_and_workflow(workspace, monkeypatch):
    from _vnccs import utils
    seeds = iter([101, 102])
    monkeypatch.setattr(utils, "generate_seed", lambda value: next(seeds))
    editor = generation.normalize_editor({"generation": {"seed_mode": "randomize", "seed": 7}})
    assert generation._resolve_editor_seed(editor)["generation"]["seed"] == 101
    assert generation._resolve_editor_seed(editor, True)["generation"]["seed"] == 7
    document = workspace.document
    payload = job_payload(document, "background", editor=editor, _vnccs_seed_resolved=True)
    generation.start_job(document["background_id"], payload)
    workspace.run_pending()
    assert workspace.generated.call_args.args[0]["generation"]["seed"] == 7
    assert editor["generation"]["seed"] == 7
    assert generation._resolve_editor_seed(editor)["generation"]["seed"] == 102


@pytest.mark.parametrize("model_type", ["qi2", "moss"])
@pytest.mark.parametrize("enhance", [False, True])
def test_style_references_reach_prompt_enhancement_and_model_encoding(runtime, monkeypatch, enhance, model_type):
    from nodes import character_styles
    saved = {"id": "user_" + "a" * 32, "label": "Ink wash", "prompt": "Ink wash\nSoft paper grain",
             "description": "Soft ink rendering", "reference": "Alice Example; Bob Example", "image": "", "user": True,
             "revision": "1"}
    monkeypatch.setattr(character_styles, "load_user_styles", lambda path=None: [saved])
    style = next(item for item in generation.style_choices() if item["id"] == saved["id"])
    assert "Reference: Alice Example; Bob Example" in style["prompt"]
    editor = {"style_id": style["id"], "style": style["prompt"],
              "scene_description": "Empty library", "generation": {"prompt_enhance": enhance, "model_type": model_type}}
    runtime.decoded_size[0] = generation.dimensions(editor)
    _image, metadata = generation.generate_image(editor, "style-with-reference", threading.Event())
    if enhance:
        enhancement_prompt = next(values["prompt"] for name, values in runtime.calls if name == "TextGenerate")
        assert "Reference: Alice Example; Bob Example" in enhancement_prompt
    encoded_prompt = next(values["prompt"] if model_type == "qi2" else values["text"]
                          for name, values in runtime.calls
                          if name == ("TextEncodeQwenImage21" if model_type == "qi2" else "CLIPTextEncode"))
    assert "Reference: Alice Example; Bob Example" in encoded_prompt
    assert "Soft paper grain" in encoded_prompt
    assert metadata["prompt"] == encoded_prompt


def test_custom_style_draft_survives_background_json_without_entering_selected_prompt(library):
    editor = generation.normalize_editor({"style_id": "watercolor", "style": "Watercolor washes",
                                          "custom_style": "My authored ink style", "negative": "text, watermark"})
    document = storage.create_background("Saved custom style", editor)
    updated = storage.save_background(document["background_id"], generation.normalize_editor(editor), document["revision"])
    restored = generation.normalize_editor(storage.load_background(document["background_id"])["editor"])
    assert restored["custom_style"] == "My authored ink style"
    assert restored["negative"] == "text, watermark"
    assert "My authored ink style" not in generation.compose_prompt(restored)
    assert "Style: Watercolor washes" in generation.compose_prompt(restored)
    assert updated["revision"] == document["revision"] + 1


def test_corrupt_user_styles_do_not_block_background_configuration_or_replace_authored_files(library, monkeypatch, tmp_path):
    from nodes import character_styles
    path = tmp_path / "background_styles.user.json"
    original = b'{"styles": [broken'
    path.write_bytes(original)
    monkeypatch.setattr(generation, "USER_BACKGROUND_STYLES_PATH", path)
    monkeypatch.setattr(generation, "_turbo_entry", lambda: None)
    document = storage.create_background("Healthy room", generation.normalize_editor({"style": "My saved rendering"}))
    before = files(library)
    config = generation.configuration()
    assert config["styles"] == [] and config["styles_error"]
    assert "model_types" in config and "samplers" in config
    assert storage.load_background(document["background_id"])["editor"]["style"] == "My saved rendering"
    assert files(library) == before and path.read_bytes() == original


@pytest.mark.parametrize("model_type", ["qi2", "moss"])
def test_negative_prompt_reaches_each_encoder_without_entering_enhanced_scene(runtime, model_type):
    editor = {"scene_description": "My empty room", "negative": "text, watermark, blur",
              "generation": {"model_type": model_type}}
    runtime.decoded_size[0] = generation.dimensions(editor)
    _image, metadata = generation.generate_image(editor, "negative-prompt", threading.Event())
    if model_type == "qi2":
        assert next(values for name, values in runtime.calls if name == "TextEncodeQwenImage21")["negative_prompt"] == editor["negative"]
    else:
        assert [values["text"] for name, values in runtime.calls if name == "CLIPTextEncode"][-1] == editor["negative"]
    assert metadata["negative"] == editor["negative"] and editor["negative"] not in metadata["prompt"]


@pytest.mark.parametrize("failure", ["json", "publish"])
@pytest.mark.parametrize("regenerate", [False, True])
def test_background_publication_failure_restores_images_json_and_previous_versions(library, monkeypatch, failure, regenerate):
    document = with_image("Transactional room")
    cards = generation.normalize_editor({})["variants"]
    images = [Image.new("RGB", (16, 8), (index * 40, 0, 0)) for index in range(4)]
    storage.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, {}, document["revision"])
    document = storage.load_background(document["background_id"])
    before = files(library)
    primary = document_path(document)
    replace = storage.os.replace
    def fail(source, target):
        if failure == "json" and Path(target) == primary:
            raise OSError("Publication failed")
        if failure == "publish" and Path(source).name.startswith(".vnccs-sprites-"):
            raise OSError("Publication failed")
        return replace(source, target)
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError, match="Publication failed"):
        storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
            [Image.new("RGB", (16, 8), "blue") for _ in images], {}, document["revision"],
            regenerate_indices=[1] if regenerate else None)
    assert files(library) == before
    assert storage.load_background(document["background_id"]) == document
    assert not list(library.rglob(".vnccs-*"))


def test_background_regeneration_changes_only_selected_file_and_keeps_archives(library):
    document = with_image("Selected variant")
    cards = generation.normalize_editor({})["variants"]
    images = [Image.new("RGB", (16, 8), (index * 40, 0, 0)) for index in range(4)]
    for _ in range(2):
        paths = storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
                                               images, {"seed": 10}, document["revision"])
        document = storage.load_background(document["background_id"])
    before = files(library)
    stamps = [path.stat().st_mtime_ns for path in paths]
    old = storage.public_document(document)["generated_sets"][-1]
    images[2] = Image.new("RGB", (16, 8), "blue")
    result = storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
                                            images, {"seed": 20}, document["revision"], regenerate_indices=[2])
    saved = storage.load_background(document["background_id"])
    current = storage.public_document(saved)["generated_sets"][-1]
    assert result == paths and current["set_id"] == old["set_id"]
    assert [group["version"] for group in saved["generated_sets"]] == ["V1", ""]
    assert not (paths[0].parent / "V2").exists()
    for index, path in enumerate(paths):
        relative = path.relative_to(library).as_posix()
        assert (path.read_bytes() != before[relative]) == (index == 2)
        assert (current["images"][index]["url"] != old["images"][index]["url"]) == (index == 2)
        if index != 2:
            assert path.stat().st_mtime_ns == stamps[index]
            assert current["images"][index]["settings"] == {"seed": 10}
    assert current["images"][2]["settings"] == {"seed": 20}
    for relative, payload in before.items():
        if "/V1/" in relative:
            assert (library / relative).read_bytes() == payload


def test_every_full_background_run_uses_next_readable_version(library):
    document = with_image("Versioned backgrounds")
    cards = generation.normalize_editor({})["variants"]
    images = [Image.new("RGB", (16, 8)) for _ in range(4)]
    for _ in range(3):
        storage.publish_background_set(document["background_id"], document["selected_image_id"], cards, images, {}, document["revision"])
        document = storage.load_background(document["background_id"])
    assert [group["version"] for group in document["generated_sets"]] == ["V1", "V2", ""]
    directory = library / document["name"]
    assert {path.name for path in (directory / "Images").iterdir()} == {"V1", "V2", "00_Original.png", "01_Morning.png", "02_Evening.png", "03_Night.png"}
    assert not (directory / "background").exists()
    assert not (directory / "sets").exists()
    assert document_path(document).name == "Versioned backgrounds_config.json"
    assert storage.image_path(document["background_id"], document["selected_image_id"]).name == "source.png"


def test_backup_references_the_retained_source_and_archived_output_files(library):
    document = with_image("Recoverable outputs")
    cards = generation.normalize_editor({})["variants"]
    images = [Image.new("RGB", (16, 8), "red") for _ in range(4)]
    for _ in range(2):
        storage.publish_background_set(document["background_id"], document["selected_image_id"], cards,
                                       images, {}, document["revision"])
        document = storage.load_background(document["background_id"])
    path = document_path(document)
    backup = json.loads(path.with_suffix(".json.backup").read_bytes())
    assert backup["generated_sets"][-1]["version"] == "V1"
    assert all("/V1/" in entry["file"] for entry in backup["generated_sets"][-1]["images"])
    old_id = document["selected_image_id"]
    document = storage.add_image(document["background_id"], Image.new("RGB", (16, 8), "blue"), {}, document["revision"])
    backup = json.loads(path.with_suffix(".json.backup").read_bytes())
    retained = next(entry for entry in backup["images"] if entry["image_id"] == old_id)
    assert retained["file"] == "Sources/V1.png"
    with Image.open(path.parent / retained["file"]) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)


def test_additional_authored_fields_survive_http_save_and_reopening(routes, library):
    editor = {"scene_description": "A room", "render_notes": {"authored": "Keep me"},
              "attributes": {"custom_note": "Keep this too"},
              "generation": {"custom_sampler_note": "Also keep", "lora_stack": [{"name": "", "strength": 1, "note": "Keep LoRA metadata"}]},
              "generation_profiles": {"moss": {"custom_profile_note": "Keep inactive profile"}}}
    document = storage.create_background("Authored fields", generation.normalize_editor(editor))
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
                    Request({"editor": editor}, info={"background_id": document["background_id"]}))
    assert response.status == 200
    saved = storage.load_background(document["background_id"])["editor"]
    assert saved["render_notes"] == editor["render_notes"]
    assert saved["attributes"]["custom_note"] == "Keep this too"
    assert saved["generation"]["custom_sampler_note"] == "Also keep"
    assert saved["generation"]["lora_stack"][0]["note"] == "Keep LoRA metadata"
    assert saved["generation_profiles"]["moss"]["custom_profile_note"] == "Keep inactive profile"


@pytest.mark.parametrize("failure", [False, True])
def test_background_inference_boundary_cleans_executor_on_all_exits(monkeypatch, failure):
    calls = []
    monkeypatch.setattr(generation, "cleanup_runtime", lambda: calls.append("cleanup"))
    monkeypatch.setattr(generation, "set_interrupt", lambda value: None)
    try:
        with generation._model_session("cleanup-proof", threading.Event()):
            if failure:
                raise ValueError("Inference failed")
    except ValueError:
        assert failure
    assert calls == ["cleanup"]
    assert generation._ACTIVE is None


@pytest.mark.parametrize("scene_type", ["Ocean", "", "interior"])
def test_scene_type_is_an_owned_enum(scene_type):
    with pytest.raises(ValueError, match="Interior or Exterior"):
        generation.normalize_editor({"attributes": {"scene_type": scene_type}})


def test_maximum_variant_details_round_trip_workflow_api_and_disk(routes, library):
    editor = generation.normalize_editor({"scene_description": "A room", "variants": [
        {"id": f"variant_{i}", "name": f"Variant {i}", "time": "Morning", "weather": "", "additional_details": "🌧" * 8000}
        for i in range(64)]})
    document = storage.create_background("Large authored scene", editor)
    state = {"schema_version": 2, "background_id": document["background_id"], "editor": editor}
    raw = json.dumps(state, ensure_ascii=False)
    assert len(raw.encode("utf-8")) > 256 * 1024
    assert generation.parse_state(raw)["editor"] == editor
    response = call(routes, "PATCH", "/backgrounds/{background_id}",
                    Request(raw=json.dumps({"editor": editor}, ensure_ascii=False).encode("utf-8"), info={"background_id": document["background_id"]}))
    assert response.status == 200
    assert storage.load_background(document["background_id"])["editor"]["variants"] == editor["variants"]


def test_cache_keys_include_complete_enhancer_template_but_not_authored_notes(workspace, monkeypatch):
    baseline = generation._background_cache_key({})
    assert generation._background_cache_key({"generation": {"custom_note": "Do not affect inference"}}) == baseline
    monkeypatch.setattr(generation, "BACKGROUND_ENHANCEMENT_SYSTEM", generation.BACKGROUND_ENHANCEMENT_SYSTEM + " New output instruction.")
    assert generation._background_cache_key({}) != baseline


@pytest.fixture
def background_style_storage(monkeypatch, tmp_path):
    from nodes import character_styles
    monkeypatch.setattr(generation, "USER_BACKGROUND_STYLES_PATH", tmp_path / "background" / "background_styles.user.json")
    monkeypatch.setattr(generation, "STYLE_PREVIEWS_DIR", tmp_path / "background" / "style_previews")
    monkeypatch.setattr(character_styles, "USER_CHARACTER_STYLES_PATH", str(tmp_path / "character" / "character_styles.user.json"))
    monkeypatch.setattr(character_styles, "STYLE_PREVIEWS_DIR", str(tmp_path / "character" / "style_previews"))
    return character_styles


def test_background_styles_are_100_unique_traceable_environment_renderings(background_style_storage):
    import re
    raw = json.loads((generation.STYLE_TEMPLATE_DIR / "background_styles.json").read_text())
    assert set(raw) == {"version", "default_style", "groups", "sources", "aliases"}
    assert raw["version"] == 7 and raw["aliases"] == {}
    assert raw["sources"] == [{
        "name": "Neon's Style Explorer",
        "url": "https://github.com/Neon-Sparks/ComfyUI-NeonsStyleExplorer",
        "credit": "100 environment styles adapted from the upstream catalog under the MIT license; see NEON_STYLES_LICENSE.txt. Original source revisions are recorded in that license file.",
        "style_count": 100,
    }]
    catalog = generation.style_catalog()
    styles = [style for group in catalog["groups"] for style in group["styles"]]
    assert len(styles) == len({style["id"] for style in styles}) == len({style["label"].casefold() for style in styles}) == 100
    assert catalog["default_style"] in {style["id"] for style in styles}
    assert len(catalog["groups"]) == 8
    assert (generation.STYLE_TEMPLATE_DIR / "NEON_STYLES_LICENSE.txt").is_file()
    forbidden = r"\b(?:characters?|people|person|portraits?|body|faces?|girls?|boys?|women|men|skin|hair|eyes|costumes?)\b"
    for style in styles:
        assert set(style) == {"id", "label", "prompt", "description", "reference", "image"}
        assert re.fullmatch(r"bg_[a-z0-9_]+", style["id"])
        assert style["label"] and style["reference"] and style["description"] and style["prompt"]
        assert len(style["description"]) <= 240 and len(style["prompt"]) <= 16000
        assert not re.search(forbidden, style["description"] + " " + style["prompt"], re.I), style["id"]
        assert "preserving its layout, objects, perspective, time and weather" in style["prompt"]
        assert not re.search(r"\b(?:at night|at dawn|at sunset|light rain|stormy|full.body|close.up)\b", style["prompt"], re.I)
        assert style["prompt"].startswith(f"Style: {style['label']}.\nReference: {style['reference']}")
        assert "\nVisual description: " in style["prompt"]
        assert f"Reference: {style['reference']}" in generation.compose_prompt({"style": style["prompt"]}).splitlines()[0]
        assert style["image"] == ""
    assert not generation.USER_BACKGROUND_STYLES_PATH.exists()


@pytest.mark.parametrize("model_type", ["qi2", "moss"])
def test_background_cover_uses_current_inference_settings_without_writing_backgrounds(workspace, background_style_storage, model_type):
    before = files(storage.root())
    editor = generation.normalize_editor({"scene_description": "An open riverside plaza", "style": "Old custom text", "aspect": "16:9",
        "generation": {"model_type": model_type, "seed": 777, "seed_mode": "randomize", "resolution_mp": 3.5,
                       "steps": 19, "cfg": 2, "prompt_enhance": True, "lora_stack": [{"name": "room.safetensors", "strength": .7}]}})
    original = copy.deepcopy(editor)
    workspace.generated.return_value = (Image.new("RGB", (1056, 1056), "blue"), {})
    result = generation.generate_style_preview({"style_id": "bg_soft_ghibli_watercolor", "editor": editor})
    used, draw_id, cancelled = workspace.generated.call_args.args
    assert used["aspect"] == "1:1" and used["generation"]["seed"] == 0 and used["generation"]["seed_mode"] == "fixed"
    assert used["generation"]["model_type"] == model_type and used["generation"]["resolution_mp"] == 3.5
    assert used["generation"]["steps"] == 19 and used["generation"]["cfg"] == 2
    assert used["generation"]["prompt_enhance"] is True and used["generation"]["lora_stack"] == editor["generation"]["lora_stack"]
    assert used["scene_description"] == editor["scene_description"] and "Studio Ghibli" in used["style"]
    assert draw_id.startswith("style_") and not cancelled.is_set()
    assert editor == original and files(storage.root()) == before
    assert result["saved"] is True and result["image"].startswith(generation.API_BASE + "/styles/preview?")
    with Image.open(result["path"]) as image:
        assert image.format == "WEBP" and image.size == (1024, 1024) and image.mode == "RGB"
    catalog_style = next(style for style in generation.style_choices() if style["id"] == result["style_id"])
    assert catalog_style["image"] == result["image"]
    assert not Path(background_style_storage.STYLE_PREVIEWS_DIR).exists()


@pytest.mark.parametrize("style_id", ["missing", "../escape", "user_" + "a" * 32, "", None])
def test_background_cover_rejects_unknown_or_unsafe_styles(workspace, background_style_storage, style_id):
    with pytest.raises(ValueError):
        generation.generate_style_preview({"style_id": style_id, "editor": {}})
    workspace.generated.assert_not_called()


def test_background_custom_cover_and_empty_scene_use_environment_fallback(workspace, background_style_storage):
    workspace.generated.return_value = (Image.new("RGB", (32, 32)), {})
    generation.generate_style_preview({"style_id": "custom", "editor": {"custom_style": "Blue ink washes"}})
    editor = workspace.generated.call_args.args[0]
    assert editor["style"] == "Blue ink washes"
    assert editor["scene_description"] == "A spacious environment with clear foreground, middle distance and background, varied surfaces and coherent lighting."
    with pytest.raises(ValueError, match="Enter a custom style"):
        generation.generate_style_preview({"style_id": "custom", "editor": {"custom_style": ""}})


@pytest.mark.parametrize("action", ["edit", "delete"])
def test_background_user_cover_does_not_publish_after_style_changes(workspace, background_style_storage, action):
    helper = background_style_storage
    saved = helper.save_user_style({"label": "My background", "prompt": "Watercolor"}, generation.USER_BACKGROUND_STYLES_PATH)
    def render(*args):
        if action == "edit":
            helper.save_user_style({**saved, "prompt": "Ink"}, generation.USER_BACKGROUND_STYLES_PATH)
        else:
            helper.delete_user_style(saved["id"], generation.USER_BACKGROUND_STYLES_PATH, preview_directory=generation.STYLE_PREVIEWS_DIR)
        return Image.new("RGB", (32, 32)), {}
    workspace.generated.side_effect = render
    with pytest.raises(ValueError, match="changed during rendering|no longer exists"):
        generation.generate_style_preview({"style_id": saved["id"], "editor": {}})
    assert not list(generation.STYLE_PREVIEWS_DIR.glob("*.webp"))


def test_background_style_http_crud_preview_ack_and_character_isolation(routes, workspace, background_style_storage, monkeypatch):
    helper = background_style_storage
    character = helper.save_user_style({"label": "Character", "prompt": "Cel shading"})
    character_cover = helper.save_style_preview(character["id"], Image.new("RGB", (32, 32), "red"))
    character_before = files(Path(helper.USER_CHARACTER_STYLES_PATH).parent)
    result = call(routes, "POST", "/styles", Request({"label": "Scenery", "description": "Ink", "prompt": "Fine ink"}))
    assert result.status == 200
    saved = json.loads(result.text)["style"]
    assert helper.load_user_styles(generation.USER_BACKGROUND_STYLES_PATH)[0]["id"] == saved["id"]
    events = []
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(send_sync=lambda name, data: events.append((name, data))))))
    workspace.generated.return_value = (Image.new("RGB", (32, 32), "blue"), {})
    result = call(routes, "POST", "/styles/preview", Request({"style_id": saved["id"], "node_id": "7", "request_id": "cover-1", "editor": {}}))
    assert result.status == 200, result.text
    cover = json.loads(result.text)
    assert cover["saved"] is True and cover["style_id"] == saved["id"]
    assert [data["status"] for name, data in events] == ["queued", "running", "done"]
    assert all(name == "vnccs.style_preview.stage" and data["node_id"] == "7" and data["request_id"] == "cover-1" for name, data in events)
    request = Request(); request.query = {"style": saved["id"]}
    assert call(routes, "GET", "/styles/preview", request).path == Path(cover["path"])
    assert len(json.loads(call(routes, "GET", "/styles", Request()).text)["groups"][-1]["styles"]) == 1
    result = call(routes, "POST", "/styles/delete", Request({"id": saved["id"]}))
    assert json.loads(result.text) == {"deleted": True, "style_id": saved["id"]}
    assert not Path(cover["path"]).exists()
    assert files(Path(helper.USER_CHARACTER_STYLES_PATH).parent) == character_before
    assert Path(character_cover["path"]).is_file()
    assert call(routes, "GET", "/styles/preview", request).status == 404
    denied = Request({"style_id": "custom"}, headers={"X-VNCCS-CSRF": ""})
    assert call(routes, "POST", "/styles/preview", denied).status == 403
    assert call(routes, "POST", "/styles", Request({"prompt": "x" * 16001, "label": "Too long"})).status == 400
    assert call(routes, "POST", "/styles/preview", Request(raw=b"[]")).status == 400
