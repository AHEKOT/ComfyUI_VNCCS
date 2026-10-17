"""Reject malformed Creator state before touching authored files or model caches."""

import json
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

import pytest
import utils
from creator_storage_helpers import load_creator_storage


def mock_creator_generation(monkeypatch, tmp_path):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    monkeypatch.setattr(creator.torch, "inference_mode", nullcontext, raising=False)
    monkeypatch.setattr(creator.torch, "is_tensor", lambda value: False, raising=False)
    monkeypatch.setattr(creator.web, "Response", lambda status=200, **data: SimpleNamespace(status=status, **data), raising=False)
    monkeypatch.setattr(creator.web, "json_response", lambda data: SimpleNamespace(status=200, data=data), raising=False)
    monkeypatch.setattr(creator.server.PromptServer.instance, "send_sync", lambda *args: None, raising=False)
    monkeypatch.setattr(creator, "acquire_preview_assets", lambda settings: (object(), object(), object()))
    monkeypatch.setattr(creator, "load_generation_assets", lambda settings: ("key", object(), object(), object()))
    monkeypatch.setattr(creator, "create_generation_latent", lambda *args: {})
    calls = []

    def encode(*args, **kwargs):
        calls.append(kwargs["character_info"]["hair"])
        return "positive", "negative", "prompt"

    def decode(*args):
        array = np.zeros((1, 3, 2, 3), dtype=np.float32)
        array[..., 0 if calls[-1] == "red hair" else 2] = 1.0
        tensor = SimpleNamespace(numpy=lambda: array)
        tensor.cpu = lambda: tensor
        return tensor

    monkeypatch.setattr(creator, "encode_generation_conditioning", encode)
    monkeypatch.setattr(creator, "sample_generation_latent", lambda **kwargs: {})
    monkeypatch.setattr(creator, "decode_generation_samples", decode)
    monkeypatch.setattr(creator, "tensor2pil", lambda image: Image.fromarray((image.numpy()[0] * 255).astype(np.uint8)))
    monkeypatch.setattr(creator, "pil2tensor", lambda image: np.asarray(image)[None, ...] / 255.0)
    return creator, calls


def test_creator_workflow_cannot_reuse_another_nodes_preview(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
             "gen_settings": {"seed": 7, "lora_stack": [{"name": "None", "strength": 1.0}]},
             "preview_valid": True, "preview_source": "gen"}
    preview = {**state, "gen_settings": {**state["gen_settings"], "lora_stack": []}}
    assert creator._generate_preview_response(preview).status == 200
    image, _, _ = creator.CharacterCreatorV2().process(json.dumps(state))
    assert len(calls) == 1, "Matching normalized inputs must reuse the preview"
    assert image[0, 0, 0, 0] == 1.0
    other = {**state, "character_info": {"name": "Alice", "hair": "blue hair"}}
    assert creator._generate_preview_response(other).status == 200
    monkeypatch.setattr(creator, "get_random_pose_preview", lambda *args: pytest.fail("A mismatched generated preview must regenerate"))
    result, _, _ = creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair", "blue hair", "red hair"]
    assert result.numpy()[0, 0, 0, 0] == 1.0


def test_creator_preview_rejects_foreign_metadata_before_replacing_the_selected_characters_image(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Bob", "character_info": {"name": "Bob", "hair": "red hair"},
             "gen_settings": {"seed": 7}}
    assert creator._generate_preview_response(state).status == 200
    path = Path(utils.character_dir("Bob")) / "cache" / "preview.png"
    original = path.read_bytes()
    monkeypatch.setattr(creator, "acquire_preview_assets", lambda *args: pytest.fail("Foreign metadata must not load models"))
    response = creator._generate_preview_response({**state, "character_info": {"name": "Alice", "hair": "blue hair"}})
    assert response.status == 400
    assert calls == ["red hair"], "Foreign metadata must be rejected before loading or sampling"
    assert path.read_bytes() == original


def test_creator_workflow_generated_cache_retains_input_ownership(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Alice", "character_info": {"hair": "red hair"},
             "gen_settings": {"seed": 7}, "preview_valid": False}
    creator.CharacterCreatorV2().process(json.dumps(state))
    state["preview_valid"] = True
    result, _, _ = creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair"]
    assert result[0, 0, 0, 0] == 1.0
    state["character_info"]["hair"] = "blue hair"
    result, _, _ = creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair", "blue hair"]
    assert result.numpy()[0, 0, 0, 2] == 1.0


def test_replaced_checkpoint_invalidates_the_saved_preview(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"old model")
    monkeypatch.setattr(creator.folder_paths, "get_folder_paths", lambda category: [str(tmp_path)])
    state = {"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
             "gen_settings": {"ckpt_name": checkpoint.name, "seed": 7},
             "preview_valid": True, "preview_source": "gen"}
    assert creator._generate_preview_response(dict(state)).status == 200
    creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair"]
    checkpoint.write_bytes(b"replacement model")
    creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair", "red hair"]


@pytest.mark.parametrize("mode, replaced", [
    ("illustrious", "model"), ("illustrious", "lora"),
    ("anima", "model"), ("anima", "clip"), ("anima", "vae"), ("anima", "lora"),
    ("qi2", "model"), ("qi2", "clip"), ("qi2", "vae"), ("qi2", "lora"),
])
def test_creator_cache_follows_replaced_assets_and_adapters(monkeypatch, tmp_path, mode, replaced):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    categories = {"model": "checkpoints" if mode == "illustrious" else "diffusion_models",
                  "clip": "text_encoders", "vae": "vae", "lora": "loras"}
    paths = {}
    for name, category in categories.items():
        path = tmp_path / category / (name + ".safetensors")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        paths[name] = path
    monkeypatch.setattr(creator.folder_paths, "get_folder_paths", lambda category: [str(tmp_path / category)])
    settings = {"generation_mode": mode, "ckpt_name": paths["model"].name,
                "diffusion_model_name": paths["model"].name, "clip_name": paths["clip"].name,
                "vae_name": paths["vae"].name, "qi2_overhaul_strength": 0, "seed": 7,
                "lora_stack": [{"name": paths["lora"].name, "strength": 1}]}

    class Asset:
        def __init__(self, contents):
            self.contents = contents

        def clone(self):
            return Asset(self.contents)

    loads = []
    def load_assets(settings):
        loads.append(True)
        return None, *(Asset(paths[name].read_bytes()) for name in ("model", "clip", "vae"))

    monkeypatch.setattr(creator, "load_generation_assets", load_assets)
    monkeypatch.setattr(creator, "load_generation_clip", lambda settings: Asset(paths["clip"].read_bytes()))
    monkeypatch.setitem(creator.PREVIEW_CACHE, "asset_key", None)
    monkeypatch.setitem(creator.PREVIEW_CACHE, "asset_obj", None)
    info = {"name": "Alice", "hair": "red hair"}
    state = json.dumps({"character": "Alice", "character_info": info, "gen_settings": settings})
    creator.acquire_preview_assets(settings)
    previous_key = creator._preview_cache_key(info, settings)
    previous_signature = creator.CharacterCreatorV2.IS_CHANGED(state)
    paths[replaced].write_bytes(b"new weights")
    assets = creator.acquire_preview_assets(settings)
    assert len(loads) == (1 if replaced == "lora" else 2)
    if replaced != "lora":
        assert assets[("model", "clip", "vae").index(replaced)].contents == b"new weights"
    assert creator._preview_cache_key(info, settings) != previous_key
    assert creator.CharacterCreatorV2.IS_CHANGED(state) != previous_signature


def test_preview_adapter_cache_releases_unused_weights_and_reloads_changed_file(monkeypatch, tmp_path):
    import gc
    import weakref

    creator, _calls = mock_creator_generation(monkeypatch, tmp_path)
    control = creator.control_center
    control._cached_lora_file.cache_clear()
    paths = [tmp_path / f"adapter-{index}.safetensors" for index in range(8)]
    for path in paths:
        path.write_bytes(b"adapter weights")
    monkeypatch.setattr(creator, "get_lora_full_path", lambda name: str(tmp_path / name))
    references, loaded = [], []

    class Weights:
        pass

    def load(path, **kwargs):
        assert kwargs == {"safe_load": True, "return_metadata": True}
        weights = Weights()
        references.append(weakref.ref(weights))
        loaded.append(path)
        return {"weights": weights}, {}

    monkeypatch.setattr(creator.comfy.utils, "load_torch_file", load, raising=False)
    monkeypatch.setattr(creator.comfy.sd, "load_lora_for_models", lambda model, clip, *args: (model, clip), raising=False)
    def preview(path):
        return creator._generate_preview_response({"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
            "gen_settings": {"ckpt_name": "unused", "dmd_lora_name": "None", "age_lora_name": "None",
                             "lora_stack": [{"name": path.name, "strength": 1}]}})

    try:
        for path in paths:
            assert preview(path).status == 200
        gc.collect()
        assert all(reference() is None for reference in references[:4])
        assert all(reference() is not None for reference in references[4:])
        assert preview(paths[-1]).status == 200
        assert len(loaded) == 8
        paths[-1].write_bytes(b"replacement weights")
        assert preview(paths[-1]).status == 200
        assert len(loaded) == 9
    finally:
        control._cached_lora_file.cache_clear()


def test_creator_ui_seed_reaches_preview_sampler_cache_and_workflow_config_once(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    sampled = []
    monkeypatch.setattr(creator, "sample_generation_latent", lambda **kwargs: sampled.append(kwargs["seed"]) or {})
    monkeypatch.setattr(creator, "generate_seed", lambda *args: pytest.fail("The UI has already resolved its random seed"))
    state = {"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
             "gen_settings": {"seed": 23, "seed_mode": "randomize",
                              "mode_settings": {"illustrious": {"seed": 23, "seed_mode": "randomize"}}},
             "preview_valid": True, "preview_source": "gen", "_vnccs_seed_resolved": True}
    assert creator._generate_preview_response(dict(state)).status == 200
    creator.CharacterCreatorV2().process(json.dumps(state))
    assert sampled == [23] and calls == ["red hair"]
    config = json.loads(Path(utils.config_path("Alice")).read_text())
    assert config["character_info"]["seed"] == 23
    assert "_vnccs_seed_resolved" not in json.dumps(config)
    assert state["gen_settings"]["seed_mode"] == "randomize"


@pytest.mark.parametrize("channel", ["preview", "workflow"])
@pytest.mark.parametrize("mode, expected", [("randomize", [101, 102]), ("fixed", [0, 0])])
def test_creator_headless_seed_policy_and_cache_key_use_actual_sampler_seed(monkeypatch, tmp_path, channel, mode, expected):
    creator, _ = mock_creator_generation(monkeypatch, tmp_path)
    sampled = []
    next_seed = iter([101, 102])
    monkeypatch.setattr(creator, "generate_seed", lambda *args: next(next_seed))
    monkeypatch.setattr(creator, "sample_generation_latent", lambda **kwargs: sampled.append(kwargs["seed"]) or {})
    state = {"character": "Alice", "character_info": {"hair": "red hair"},
             "gen_settings": {"seed": 0, "seed_mode": mode,
                              "mode_settings": {"illustrious": {"seed": 0, "seed_mode": mode}}},
             "preview_valid": False}
    for actual_seed in expected:
        if channel == "preview":
            assert creator._generate_preview_response(dict(state)).status == 200
        else:
            creator.CharacterCreatorV2().process(json.dumps(state))
            assert json.loads(Path(utils.config_path("Alice")).read_text())["character_info"]["seed"] == actual_seed
        settings = creator.normalize_gen_settings(state["gen_settings"])
        settings["seed"] = actual_seed
        with Image.open(Path(utils.character_dir("Alice")) / "cache" / "preview.png") as image:
            assert image.info["vnccs_preview_key"] == creator._preview_cache_key(state["character_info"], settings)
    assert sampled == expected


@pytest.mark.parametrize("mode, resolved, changes", [("fixed", False, False), ("randomize", True, False),
                                                       ("randomize", False, True)])
def test_creator_headless_random_mode_bypasses_comfy_cache_but_resolved_ui_seeds_do_not(monkeypatch, tmp_path, mode, resolved, changes):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    state = {"character": "Alice", "gen_settings": {"seed_mode": "fixed",
        "mode_settings": {"illustrious": {"seed": 23, "seed_mode": mode}}}, "_vnccs_seed_resolved": resolved}
    signature = creator.CharacterCreatorV2.IS_CHANGED(json.dumps(state))
    if changes:
        assert math.isnan(signature)
    else:
        assert signature == creator.CharacterCreatorV2.IS_CHANGED(json.dumps(state))


def test_creator_failed_preview_publish_preserves_image_and_metadata_together(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Alice", "character_info": {"hair": "red hair"},
             "gen_settings": {"seed": 7}, "preview_valid": True}
    assert creator._generate_preview_response(state).status == 200
    path = Path(utils.character_dir("Alice")) / "cache" / "preview.png"
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        def fail_replace(*args):
            raise OSError("Simulated interrupted preview publish")
        patch.setattr(utils.os, "replace", fail_replace)
        assert creator._generate_preview_response({**state, "character_info": {"hair": "blue hair"}}).status == 500
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    image, _, _ = creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair", "blue hair"]
    assert image[0, 0, 0, 0] == 1.0


def test_creator_pose_cache_cannot_replace_another_nodes_generated_reference(monkeypatch, tmp_path):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Alice", "character_info": {"hair": "red hair"},
             "gen_settings": {"seed": 7}, "preview_valid": True, "preview_source": "gen"}
    assert creator._generate_preview_response(state).status == 200
    array = np.zeros((1, 3, 2, 3), dtype=np.float32)
    array[..., 2] = 1.0
    pose = SimpleNamespace(numpy=lambda: array)
    monkeypatch.setattr(creator, "get_pose_preview", lambda *args, **kwargs: (pose, 0, 1))
    creator.CharacterCreatorV2().process(json.dumps({**state, "preview_source": "pose"}))
    with Image.open(Path(utils.character_dir("Alice")) / "cache" / "preview.png") as image:
        assert image.info["vnccs_preview_key"] == "pose"
    result, _, _ = creator.CharacterCreatorV2().process(json.dumps(state))
    assert calls == ["red hair", "red hair"]
    assert result.numpy()[0, 0, 0, 0] == 1.0


@pytest.mark.parametrize("state", ["broken JSON", "[]", "null", "{}", '{"character":"../outside"}',
                                  '{"character":"Alice","character_info":[]}',
                                  '{"character":"Alice","gen_settings":null}'])
def test_invalid_creator_state_preserves_disk_and_preview_assets(monkeypatch, tmp_path, state):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    path = Path(utils.config_path("Alice"))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"character_info": {"name": "Alice", "hair": "authored hair"}}))
    original = path.read_bytes()
    assets = object()
    monkeypatch.setitem(creator.PREVIEW_CACHE, "asset_obj", assets)
    with pytest.raises(ValueError):
        creator.CharacterCreatorV2().process(state)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path.parent]
    assert creator.PREVIEW_CACHE["asset_obj"] is assets


def test_saved_qi2_pose_preview_preserves_transparency(monkeypatch, tmp_path):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(creator, "pil2tensor", lambda image: np.asarray(image)[None, ...])
    directory = Path(utils.character_dir("Alice")) / "Sprites" / "Naked" / "Neutral"
    directory.mkdir(parents=True)
    Image.new("RGBA", (2, 3), (255, 0, 0, 128)).save(directory / "pose.png")
    image, index, count = creator.get_pose_preview("Alice", index=0)
    assert image.shape == (1, 3, 2, 4)
    assert np.all(image[..., 3] == 128)
    assert index == 0 and count == 1


def test_saved_creator_pose_uses_displayed_exif_orientation(monkeypatch, tmp_path):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(creator, "pil2tensor", lambda image: np.asarray(image)[None, ...])
    directory = Path(utils.character_dir("Alice")) / "Sprites" / "Original" / "Neutral"
    directory.mkdir(parents=True)
    stored = Image.new("RGB", (3, 7), "red")
    exif = stored.getexif()
    exif[274] = 6
    stored.save(directory / "pose.jpg", exif=exif)
    image, index, count = creator.get_pose_preview("Alice", index=0)
    assert image.shape == (1, 3, 7, 3)
    assert index == 0 and count == 1


@pytest.mark.parametrize("link_kind", ["file", "neutral_directory"])
def test_creator_preview_skips_unsafe_entries_without_hiding_valid_sprites(monkeypatch, tmp_path, link_kind):
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private image")
    sprites = Path(utils.character_dir("Alice")) / "Sprites"
    valid_root = sprites / "Original" / "Neutral" if link_kind == "neutral_directory" else sprites / "Naked" / "Neutral"
    valid_root.mkdir(parents=True)
    valid = valid_root / "valid.png"
    valid.write_bytes(b"valid image")
    if link_kind == "file":
        (valid_root / "private.png").symlink_to(outside / "private.png")
    else:
        (sprites / "Naked").mkdir()
        (sprites / "Naked" / "Neutral").symlink_to(outside, target_is_directory=True)
    assert creator.list_pose_preview_files("Alice") == [str(valid)]


@pytest.mark.parametrize("link_kind", ["directory", "image"])
def test_creator_cached_preview_never_serves_an_external_symlink(monkeypatch, tmp_path, link_kind):
    import asyncio
    from types import SimpleNamespace
    creator = load_creator_storage(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    monkeypatch.setattr(creator.web, "Response", lambda status=200, **kwargs: SimpleNamespace(status=status), raising=False)
    monkeypatch.setattr(creator.web, "FileResponse", lambda path: pytest.fail("External cache files must not be served"), raising=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private image")
    root = Path(utils.character_dir("Alice"))
    root.mkdir(parents=True)
    cache = root / "cache"
    if link_kind == "directory":
        cache.symlink_to(outside, target_is_directory=True)
    else:
        cache.mkdir()
        (cache / "preview.png").symlink_to(outside / "private.png")
    response = asyncio.run(creator.get_cached_preview(SimpleNamespace(rel_url=SimpleNamespace(query={"character": "Alice"}))))
    assert response.status >= 400
