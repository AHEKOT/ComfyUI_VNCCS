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
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "VNCCS" / "scenes")
    return storage.root()


def files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def document_path(document):
    return storage.root() / document["name"] / f"{document['name']}.json"


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
    assert sorted(item.name for item in document_path(document).parent.iterdir()) == ["Evening Café.json", "Evening Café.json.backup", "background"]
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
    assert updated["editor"] == state
    assert path.with_suffix(".json.backup").read_bytes() == previous
    state["prompt"] = "Mutated caller object"
    assert storage.load_background(document["background_id"])["editor"]["prompt"] == "Changed"


def test_legacy_image_is_preserved_when_creating_the_working_preview(library):
    document = with_image()
    document["images"][0].pop("preview_revision")
    document_path(document).write_text(json.dumps(document))
    first_id = document["selected_image_id"]
    first_path = storage.image_path(document["background_id"], first_id)
    assert first_path == library / document["name"] / "background" / f"{first_id}.png"
    original = first_path.read_bytes()
    updated = storage.add_image(document["background_id"], Image.new("RGBA", (20, 10), "blue"),
                                {"source": "upload"}, document["revision"])
    assert len(updated["images"]) == 2
    assert updated["selected_image_id"] != first_id
    assert first_path.read_bytes() == original
    assert storage.image_path(document["background_id"], first_id) == first_path
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
    assert len(list((library / document["name"] / "background").glob("*.png"))) == 1


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["png", "json"])
def test_preview_write_failure_retains_image_metadata_and_cleans_temporary_files(library, monkeypatch, existing, failure):
    document = with_image() if existing else storage.create_background("Failed preview", {})
    before = files(library)
    atomic = storage._atomic
    def write(path, payload):
        if (failure == "png" and path.suffix == ".png") or (failure == "json" and path.name == f"{document['name']}.json"):
            raise OSError("Disk full")
        return atomic(path, payload)
    monkeypatch.setattr(storage, "_atomic", write)
    with pytest.raises(OSError, match="Disk full"):
        storage.add_image(document["background_id"], Image.new("RGB", (20, 10), "blue"), {}, document["revision"])
    after = files(library)
    backup_name = f"{document['name']}/{document['name']}.json.backup"
    assert after.pop(backup_name) in (before[backup_name], before[f"{document['name']}/{document['name']}.json"])
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
    assert recovered["_recovery"]["source"] == f"{document['name']}.json.backup"
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

def test_defaults_are_isolated_and_only_qi2_fields_survive():
    editor = generation.normalize_editor({"generation_mode": "illustrious",
        "generation": {"diffusion_model_name": "other.safetensors", "mode_settings": {"anima": {}}}})
    assert editor == {**generation.DEFAULT_EDITOR, "generation": {**generation.DEFAULT_EDITOR["generation"], "diffusion_model_name": "other.safetensors"}}
    editor["attributes"]["location"] = "Room"
    editor["generation"]["lora_stack"].append({"name": "test"})
    assert "location" not in generation.DEFAULT_EDITOR["attributes"]
    assert generation.DEFAULT_EDITOR["generation"]["lora_stack"] == []
    assert "composition" not in editor["attributes"]


@pytest.mark.parametrize("data", [[], "{}", {"attributes": []}, {"generation": []},
    {"scene_description": 1}, {"scene_description": "x" * 40001}, {"prompt": 1}, {"prompt": "x" * 40001}, {"description": 1}, {"description": "x" * 8001}, {"style": "bad\x00style"},
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


@pytest.mark.parametrize("aspect", generation.ASPECTS)
@pytest.mark.parametrize("mp", [0.05, 2, 16])
def test_dimensions_follow_aspect_and_bounded_pixel_budget(aspect, mp):
    width, height = generation.dimensions({"aspect": aspect, "generation": {"resolution_mp": mp}})
    a, b = map(int, aspect.split(":"))
    assert width % 16 == height % 16 == 0
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
            return ("Enhanced empty background with warm light",)
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
    assert calls["TextEncodeQwenImage21"]["images"] == {}
    assert calls["TextEncodeQwenImage21"]["prompt"] == generation.compose_prompt({"scene_description": "Enhanced empty background with warm light"})
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
    image, metadata = generation.generate_image(editor, "moss-t2i", threading.Event())
    calls, names = dict(runtime.calls), [name for name, _ in runtime.calls]
    assert calls["UNETLoader"]["unet_name"] == "moss_image_sft_int8_convrot.safetensors"
    assert calls["CLIPLoader"] == {"clip_name": "qwen3.5_9b_w4a8_gptq.safetensors", "type": "moss_image", "device": "default"}
    assert calls["VAELoader"]["vae_name"] == "flux2-vae_small.safetensors"
    encodes = [value for name, value in runtime.calls if name == "CLIPTextEncode"]
    expected = generation.compose_prompt({**editor, "scene_description": "Enhanced empty background with warm light"}) if enhance else generation.compose_prompt(editor)
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
def test_turbo_rejects_unsupported_schedule_before_loading(runtime, settings):
    with pytest.raises(ValueError, match="six steps and CFG 1"):
        generation.generate_image({"generation": settings}, "turbo-bad", threading.Event())
    assert runtime.calls == []


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
    monkeypatch.setattr(storage, "root", lambda: tmp_path / "VNCCS" / "scenes")
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
    return {"action": action, "revision": document["revision"], "editor": copy.deepcopy(document["editor"]), **changes}


@pytest.mark.parametrize("payload", [None, [], {"action": "unknown"}, {"action": "export"}, {"action": "model"}, {"action": "cg"}])
def test_invalid_job_action_never_queues_or_changes_document(workspace, payload):
    with pytest.raises(ValueError):
        service.start_job(workspace.document["background_id"], payload)
    workspace.worker.submit.assert_not_called()
    assert storage.load_background(workspace.document["background_id"]) == workspace.document


@pytest.mark.parametrize("revision", [None, True, 0, 999])
def test_job_requires_current_integer_revision(workspace, revision):
    payload = job_payload(workspace.document, "background", revision=revision)
    with pytest.raises(storage.BackgroundConflictError):
        service.start_job(workspace.document["background_id"], payload)
    workspace.worker.submit.assert_not_called()


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
    service.start_job(document["background_id"], job_payload(document, "wizard"))
    job = workspace.run_pending()
    assert job["status"] == "completed"
    assert job["result"]["editor"]["attributes"]["location"] == "Tokyo"
    assert storage.load_background(document["background_id"]) == document
    workspace.generated.assert_not_called()


@pytest.mark.parametrize("condition", ["archived", "recovered"])
def test_uneditable_documents_fail_before_inference(workspace, condition):
    document = workspace.document
    if condition == "archived":
        document = storage.archive_background(document["background_id"], document["revision"])
    else:
        path = storage.root() / document["name"] / f"{document['name']}.json"
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


def test_http_create_load_save_conflict_and_archive_preserve_document(routes, workspace):
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
    assert response.status == 409
    response = call(routes, "GET", "/backgrounds/{background_id}", Request(info=info))
    assert json.loads(response.text)["revision"] == updated["revision"]
    response = call(routes, "DELETE", "/backgrounds/{background_id}", Request({"revision": updated["revision"]}, info=info))
    assert response.status == 200 and json.loads(response.text)["archived"] is True
    response = call(routes, "GET", "/backgrounds/{background_id}", Request(info=info))
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
    response = call(routes, "POST", "/backgrounds", Request(raw=b" " * (256 * 1024 + 1)))
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
    monkeypatch.setattr(storage, "_preview_tensor", lambda source: np.asarray(Image.open(source))[None])
    inference = mock.Mock(side_effect=AssertionError("Uploaded backgrounds must not run inference"))
    monkeypatch.setattr(storage, "generate_image", inference)
    raw = json.dumps({"schema_version": 2, "background_id": document["background_id"]})
    result = node.create_background(background_data=raw)
    tensor, prompts = result["result"]
    assert len(prompts) == 3 and tensor.shape == (1, 80, 160, 3)
    with pytest.raises(ValueError, match="Select or create"):
        node.create_background()
    inference.assert_not_called()


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
            "editor": copy.deepcopy(document["editor"]), **changes}


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


@pytest.mark.parametrize("change", ["prompt", "seed", "model", "lora", "weights", "enhancer"])
def test_changed_generation_inputs_invalidate_run_cache_and_replace_current_preview(workspace, monkeypatch, change):
    run_background(workspace.document)
    document = storage.load_background(workspace.document["background_id"])
    original_image = storage.image_path(document["background_id"], document["selected_image_id"])
    original_id = document["selected_image_id"]
    original_revision = document["images"][0]["preview_revision"]
    editor = copy.deepcopy(document["editor"])
    if change == "prompt": editor["scene_description"] = "A different room"
    if change == "seed": editor["generation"]["seed"] += 1
    if change == "model": editor["generation"] = {"model_type": "moss"}
    if change == "lora": editor["generation"]["lora_stack"] = [{"name": "room.safetensors", "strength": 0.6}]
    if change == "weights": generation._resolve_model_path("diffusion_models", generation.ASSETS["diffusion_models"]).write_bytes(b"updated weights")
    if change == "enhancer": monkeypatch.setattr(generation, "SCENE_EXPANSION_SYSTEM", generation.SCENE_EXPANSION_SYSTEM + "Updated wording.")
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
    assert dict(runtime.calls)["TextEncodeQwenImage21"]["prompt"].splitlines()[-1] == "Scene description: An exact room prompt."
    generation.generate_image({"generation": {"prompt_enhance": False}}, "default", threading.Event())
    assert len([name for name, _ in runtime.calls if name == "UNETLoader"]) == 2


def test_turbo_model_cannot_apply_a_second_acceleration_adapter(runtime):
    with pytest.raises(ValueError, match="already includes acceleration"):
        generation.generate_image({"generation": {"diffusion_model_name": "qwen_image_2.1_turbo_int8_convrot.safetensors",
            "turbo": True, "steps": 6}}, "double-turbo", threading.Event())
    assert not runtime.calls


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


def test_prompt_enhancement_only_changes_scene_description(runtime, monkeypatch):
    editor = {"style": "Anime, digital art, cel shaded", "scene_description": "A futuristic circular room.",
              "attributes": {"time": "Night", "scene_type": "Interior"}}
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return ("Style: Wrong style\nTime: Day\nInterior/Exterior: Exterior\nScene description: An enhanced futuristic circular room.",)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    generation.generate_image(editor, "structured-enhancement", threading.Event())
    prompt = dict(runtime.calls)["TextEncodeQwenImage21"]["prompt"]
    assert prompt == "Style: Anime, digital art, cel shaded\nTime: Day\nInterior/Exterior: Interior\nScene description: An enhanced futuristic circular room."


def test_wizard_returns_the_four_scene_fields(runtime, monkeypatch):
    fields = {"style": "Anime, digital art, cel shaded", "time": "Night", "scene_type": "Interior", "scene_description": "A room with a circular stone table."}
    def call(name, **kwargs):
        if name == "TextGenerate":
            runtime.calls.append((name, kwargs))
            return (json.dumps(fields),)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    editor = generation.wizard("A futuristic room", {}, "four-field-wizard", threading.Event())
    assert editor["scene_description"] == fields["scene_description"]
    assert editor["style"] == fields["style"]
    assert editor["attributes"] == {"time": "Day", "scene_type": "Interior"}
    assert generation.compose_prompt(editor).splitlines()[1] == "Time: Day"


@pytest.mark.parametrize("use_wizard", [False, True])
def test_sparse_scene_prompts_request_detail_and_preserve_field_contract(runtime, monkeypatch, use_wizard):
    editor = {"style": "Anime, digital art, cel shaded", "scene_description": "badroom",
              "attributes": {"time": "Night", "scene_type": "Interior"}}
    original = copy.deepcopy(editor)
    description = (
        "A quiet bedroom opens around a low wooden bed with rumpled linen and layered pillows. "
        "A woven rug crosses the foreground, leading toward a small bedside table and a warm reading lamp. "
        "Tall shelves hold books, ceramic bowls and folded blankets along the left wall. "
        "In the background, blue moonlight enters through a broad window framed by loose curtains. "
        "The cool light catches the grain of the floorboards and softens against textured plaster walls, "
        "while the lamp casts gentle amber shadows across the bedding. A narrow writing desk, a potted plant "
        "and neatly stacked notebooks complete the calm, lived-in room."
    )
    expected = generation.compose_prompt({**editor, "scene_description": description})
    fields = {"style": editor["style"], "time": "Night", "scene_type": "Interior", "scene_description": description}
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
        assert dict(runtime.calls)["TextEncodeQwenImage21"]["prompt"] == expected
        assert metadata["editor"]["scene_description"] == "badroom"
    request = dict(runtime.calls)["TextGenerate"]
    assert "badroom" in request["prompt"]
    assert "interpret 'badroom' as 'bedroom'" in request["system_prompt"]
    assert "100-180 words" in request["system_prompt"]
    assert "foreground/midground/background" in request["system_prompt"]
    assert "Style" in request["system_prompt"] and "Interior/Exterior" in request["system_prompt"]
    if use_wizard:
        assert "exactly four string keys in this order: style, time, scene_type, scene_description" in request["system_prompt"]
        assert json.loads(request["system_prompt"].split("not a finished answer: ")[1]) == {**fields, "time": "Day", "scene_description": "badroom"}
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


def test_background_sets_publish_complete_immutable_images_and_same_json(library, monkeypatch):
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
        assert path.is_relative_to(library / document["name"] / "background" / "sets")
        with Image.open(path) as loaded:
            assert loaded.size == image.size
    assert (library / document["name"] / "background" / f"{document['selected_image_id']}.png").read_bytes() == before[f"{document['name']}/background/{document['selected_image_id']}.png"]
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
    assert set(more).isdisjoint(paths) and all(path.is_file() for path in paths)


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
    raw_result = "Style: Wrong style\nTime: Night\nInterior/Exterior: Exterior\nScene description: " + enhanced.strip()
    expected = generation.compose_prompt({**editor, "scene_description": enhanced if enhance else description})
    def call(name, **kwargs):
        if name == "TextGenerate":
            logged = [record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "text_generation_input"]
            assert len(logged) == 1
            assert logged[0].vnccs["prompt"] == kwargs["prompt"] == generation.compose_prompt(editor)
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


def test_saved_sets_publish_only_after_commit_and_serve_immutable_owned_images(routes, workspace, monkeypatch):
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
        assert image["url"].endswith(f"/sets/{group['set_id']}/{index}")
        info = {"background_id": document["background_id"], "set_id": group["set_id"], "index": str(index)}
        result = call(routes, "GET", "/backgrounds/{background_id}/sets/{set_id}/{index}", Request(info=info))
        assert result.status == 200 and "immutable" in result.headers["Cache-Control"]
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
    group["images"][1]["file"] = f"background/{document['selected_image_id']}.png"
    monkeypatch.setattr(storage, "_locate", lambda _id: (storage.root() / saved["name"], saved, b""))
    with pytest.raises(ValueError, match="outside its saved set"):
        storage.background_set_image_path(document["background_id"], group["set_id"], 1)
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
