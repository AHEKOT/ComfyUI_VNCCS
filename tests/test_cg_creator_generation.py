"""Captured-frame QI2 refinement contracts without Torch or model weights."""

import sys
import ast
import inspect
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
import types
from contextlib import nullcontext
from unittest import mock

import pytest
import numpy as np
from PIL import Image

from helpers.backend_package import service_package, stub_imports


load = service_package("vnccs_cg_generation_test")
generation = load("nodes.cg_creator")
runtime_cleanup = load("nodes.runtime_cleanup")



def test_defaults_are_isolated_and_do_not_add_other_generation_features():
    settings = generation.normalize_settings({})
    assert settings == generation.DEFAULT_SETTINGS
    settings["generation"]["lora_stack"].append({"name": "test"})
    assert len(generation.DEFAULT_SETTINGS["generation"]["lora_stack"]) == 5
    assert all(item["name"] == "" for item in generation.DEFAULT_SETTINGS["generation"]["lora_stack"])
    assert generation.normalize_settings({"prompt": "  "})["prompt"] == generation.DEFAULT_SETTINGS["prompt"]
    assert not hasattr(generation, "wizard")


def test_cg_inference_never_imports_another_editor_runtime():
    for module in (generation, runtime_cleanup):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "unicanvas" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all("unicanvas" not in item.name for item in node.names)


def test_sampling_progress_belongs_to_cg_and_restores_the_global_hook(monkeypatch, caplog):
    from concurrent.futures import ThreadPoolExecutor
    original = mock.Mock()
    utils = sys.modules["comfy.utils"]
    monkeypatch.setattr(utils, "PROGRESS_BAR_HOOK", original, raising=False)
    monkeypatch.setattr(generation, "_PROGRESS", {})
    with pytest.raises(RuntimeError, match="sampler failed"):
        with generation._report_sampling_progress("cg-only", 8):
            hook = utils.PROGRESS_BAR_HOOK
            hook(1, 8)
            assert generation.progress("cg-only")["message"] == "Sampling 1/8"
            original.assert_not_called()
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(hook, 2, 8).result()
            original.assert_called_once_with(2, 8)
            raise RuntimeError("sampler failed")
    assert utils.PROGRESS_BAR_HOOK is original
    assert any(record.vnccs.get("component") == "CGCreator" for record in caplog.records)


def test_cg_progress_is_bounded(monkeypatch):
    monkeypatch.setattr(generation, "_PROGRESS", {})
    for index in range(70):
        generation._set_progress(str(index), "complete", 1, "Complete")
    assert len(generation._PROGRESS) == 64
    assert generation.progress("0")["stage"] == "unknown"


def test_empty_lora_slots_and_future_profile_metadata_round_trip_without_mutation():
    data = {"future_profiles": {"new_renderer": {"keep": [1, "two"]}}, "generation": {
        "future_option": {"untouched": True}, "lora_stack": [
            {"name": "", "strength": 0.7}, {"name": "scenery.safetensors", "strength": 0.4}]}}
    settings = generation.normalize_settings(data)
    assert settings["future_profiles"] == data["future_profiles"]
    assert settings["generation"]["future_option"] == data["generation"]["future_option"]
    assert settings["generation"]["lora_stack"][:2] == data["generation"]["lora_stack"]
    assert len(settings["generation"]["lora_stack"]) == 5
    settings["future_profiles"]["new_renderer"]["keep"].append(3)
    assert data["future_profiles"]["new_renderer"]["keep"] == [1, "two"]


@pytest.mark.parametrize("settings", [[], "{}", {"generation": []}, {"prompt": 1},
    {"prompt": "x" * 8001}, {"prompt": "bad\x00prompt"}, {"aspect": "not-an-aspect"}, {"aspect": []},
    {"generation": {"steps": True}}, {"generation": {"steps": 1.5}},
    {"generation": {"cfg": float("nan")}}, {"generation": {"cfg": float("inf")}},
    {"generation": {"seed": -1}}, {"generation": {"seed": 2**80}},
    {"generation": {"diffusion_model_name": "../qwen_image_2.1.safetensors"}},
    {"generation": {"seed_mode": "automatic"}}, {"generation": {"turbo_previous_settings": []}},
    {"generation": {"turbo_previous_settings": {"steps": float("nan")}}},
    {"future_profiles": {"bad": float("nan")}}, {"future_profiles": "x" * (256 * 1024)},
    {"resolution_mp": 20}, {"resolution_mp": float("nan")},
    {"generation": {"prompt_enhance": "yes"}}, {"generation": {"turbo": 1}},
    {"generation": {"cache_device": "cuda"}}, {"generation": {"cache_dtype": "fp64"}},
    {"generation": {"sampler": ""}}, {"generation": {"lora_stack": {}}},
    {"generation": {"lora_stack": [{}] * 9}}, {"generation": {"lora_stack": [None]}},
    {"generation": {"lora_stack": [{"name": "../secret"}]}},
    {"generation": {"lora_stack": [{"name": ".. /secret"}]}},
    {"generation": {"lora_stack": [{"name": "C:\\secret"}]}},
    {"generation": {"lora_stack": [{"name": "\\\\server\\secret"}]}},
    {"generation": {"lora_stack": [{"name": "/secret"}]}},
    {"generation": {"lora_stack": [{"name": "good.safetensors", "strength": float("nan")}]}},
    {"generation": {"lora_stack": [{"name": generation.TURBO_LORA}]}}])
def test_rejects_malformed_settings(settings):
    with pytest.raises(ValueError):
        generation.normalize_settings(settings)


@pytest.fixture
def runtime(monkeypatch):
    calls, clips = [], []
    base = mock.Mock(name="cached-model")
    base.clone.side_effect = lambda: mock.Mock(name="request-model", patches={})
    vae = object()
    monkeypatch.setattr(generation, "_ASSET_CACHE", None)
    monkeypatch.setattr(generation, "_ACTIVE", None)
    monkeypatch.setattr(generation, "_asset_name", lambda category, name: name)
    monkeypatch.setattr(generation, "_get_lora_full_path", lambda name: "/models/loras/" + name)
    monkeypatch.setattr(generation, "_load_lora_file", lambda path: ({"lora": path}, {}))
    monkeypatch.setattr(generation, "resolve_anyangle_lora", lambda _cancel: generation.ANYANGLE_LORA)
    def apply_lora(model, clip, state, strength, clip_strength):
        calls.append(("ApplyLoRA", {"state": state, "strength": strength, "clip_strength": clip_strength}))
        model.patches[state["lora"]] = [(strength, clip_strength)]
        return model, clip
    monkeypatch.setattr(sys.modules["comfy.sd"], "load_lora_for_models", apply_lora, raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(generation, "model_operation_lock", lambda: runtime_cleanup.inference_lock)
    monkeypatch.setattr(generation, "_set_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation, "_set_interrupt", mock.Mock())
    monkeypatch.setattr(generation, "_pil_to_tensor", lambda image: ("image", image.size, image.mode, image.getpixel((0, 0))))
    size, latent_size = [1936, 1088], [1936, 1083]
    monkeypatch.setattr(generation, "_tensor_to_pil", lambda tensor: Image.new("RGB", tuple(size)))

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
        if name == "TextEncodeQwenImage21":
            return "positive", "negative", {"samples": "encoder-reference"}
        if name == "ImageScaleToTotalPixels":
            return ("scaled-camera-frame",)
        if name == "GetImageSize":
            return *latent_size, 1
        if name == "EmptyLatentImage":
            return ({"samples": "requested-frame", "size": (kwargs["width"], kwargs["height"])},)
        return (name + "-output",)

    monkeypatch.setattr(generation, "_call_comfy_node", call)
    return types.SimpleNamespace(calls=calls, clips=clips, base=base, size=size, latent_size=latent_size, call=call,
                                 reference=Image.new("RGB", (1888, 1056), "red"),
                                 background=Image.new("RGB", (1200, 672), "green"))


def test_anyangle_uses_creator_assets_fixed_prompt_ordered_images_and_scaled_frame(runtime, caplog):
    image, metadata = generation.generate_image({}, "cg-frame", threading.Event(), runtime.reference, runtime.background)
    names = [name for name, _ in runtime.calls]
    assert names == ["UNETLoader", "CLIPLoader", "VAELoader", "ApplyLoRA", "QwenImage21Cache",
                     "TextEncodeQwenImage21", "ImageScaleToTotalPixels", "GetImageSize",
                     "EmptyLatentImage", "KSampler", "VAEDecode"]
    calls = dict(runtime.calls)
    assert calls["UNETLoader"]["unet_name"] == generation.ASSETS["diffusion_models"]
    assert calls["CLIPLoader"]["clip_name"] == generation.ASSETS["text_encoders"]
    assert calls["CLIPLoader"]["type"] == "qwen_image"
    assert calls["VAELoader"]["vae_name"] == generation.ASSETS["vae"]
    assert calls["QwenImage21Cache"]["device"] == "auto"
    assert calls["QwenImage21Cache"]["dtype"] == "default"
    assert calls["ApplyLoRA"] == {"state": {"lora": "/models/loras/" + generation.ANYANGLE_LORA},
                                 "strength": 1.0, "clip_strength": 1.0}
    encode = calls["TextEncodeQwenImage21"]
    assert encode["images"] == {"image_1": ("image", (1200, 672), "RGB", (0, 128, 0)),
                                "image_2": ("image", (1888, 1056), "RGB", (255, 0, 0))}
    assert encode["resolution"] == 1024 and encode["negative_prompt"] == ""
    assert encode["prompt"] == metadata["prompt"] == "Change the camera angle from <image2> to <image1>."
    assert image.size == (1936, 1088)
    assert "width" not in encode and "height" not in encode
    assert calls["ImageScaleToTotalPixels"] == {"image": encode["images"]["image_2"],
        "upscale_method": "nearest-exact", "megapixels": 2.0, "resolution_steps": 1}
    assert calls["ImageScaleToTotalPixels"]["image"] is encode["images"]["image_2"]
    assert calls["GetImageSize"] == {"image": "scaled-camera-frame"}
    assert calls["EmptyLatentImage"] == {"width": 1936, "height": 1083, "batch_size": 1}
    assert calls["KSampler"]["latent_image"] == {"samples": "requested-frame", "size": (1936, 1083)}
    assert calls["KSampler"]["steps"] == 25 and calls["KSampler"]["cfg"] == 1
    assert calls["KSampler"]["scheduler"] == "simple" and calls["KSampler"]["denoise"] == 1
    assert metadata["settings"] == generation.DEFAULT_SETTINGS and metadata["model"] == "qi2"
    assert metadata["anyangle_lora"] == {"name": generation.ANYANGLE_LORA, "strength": 1.0,
                                         "revision": generation.ANYANGLE_REVISION}
    runtime.base.clone.assert_called_once()
    events = [record.vnccs for record in caplog.records if hasattr(record, "vnccs")]
    lora = next(event for event in events if event["event"] == "lora_applied")
    assert lora["path"] == "/models/loras/" + generation.ANYANGLE_LORA
    assert lora["strength_model"] == 1 and lora["model_patch_keys"] == 1
    for slot, source, reference in (("image_1", "original background", runtime.background),
                                    ("image_2", "captured camera frame", runtime.reference)):
        event = next(item for item in events if item.get("slot") == slot)
        assert (event["source"], event["width"], event["height"]) == (source, *reference.size)
        assert event["rgb_sha256"] == hashlib.sha256(reference.tobytes()).hexdigest()
    conditioning = next(event for event in events if event["event"] == "conditioning")
    assert conditioning["prompt"] == generation.ANYANGLE_PROMPT and conditioning["negative_prompt"] == ""
    assert conditioning["resolution"] == 1024
    latent = next(event for event in events if event["event"] == "latent")
    assert (latent["width"], latent["height"], latent["resolution_mp"]) == (1936, 1083, 2)
    sampling = next(event for event in events if event["event"] == "sampling")
    assert sampling["settings"]["steps"] == 25 and sampling["settings"]["cfg"] == 1
    assert sampling["settings"]["seed"] == metadata["seed"]


def test_fixed_prompt_never_runs_enhancement_and_preserves_the_legacy_profile(runtime):
    settings = {"aspect": "1:1", "resolution_mp": 0.5, "prompt": "Repair the room; keep the view.",
                "generation": {"prompt_enhance": True}}
    image, metadata = generation.generate_image(settings, "camera-cg", threading.Event(), runtime.reference, runtime.background)
    encode = dict(runtime.calls)["TextEncodeQwenImage21"]
    assert image.size == (metadata["width"], metadata["height"]) == (1936, 1088)
    assert encode["prompt"] == metadata["prompt"] == generation.ANYANGLE_PROMPT
    assert metadata["settings"]["prompt"] == settings["prompt"]
    assert metadata["settings"]["generation"]["prompt_enhance"] is False
    assert "TextGenerate" not in dict(runtime.calls)
    assert dict(runtime.calls)["EmptyLatentImage"] == {"width": 1936, "height": 1083, "batch_size": 1}


def test_fresh_clip_per_request_and_no_automatic_character_loras(runtime):
    settings = {"generation": {"prompt_enhance": False, "lora_stack": [{"name": "scenery.safetensors", "strength": 0.7}]}}
    generation.generate_image(settings, "first", threading.Event(), runtime.reference, runtime.background)
    generation.generate_image(settings, "second", threading.Event(), runtime.reference, runtime.background)
    names = [name for name, _ in runtime.calls]
    assert names.count("UNETLoader") == names.count("VAELoader") == 1
    assert names.count("CLIPLoader") == 2 and runtime.clips[0] is not runtime.clips[1]
    applied = [call for name, call in runtime.calls if name == "ApplyLoRA"]
    assert len(applied) == 4
    assert applied[0] == applied[2] == {"state": {"lora": "/models/loras/" + generation.ANYANGLE_LORA}, "strength": 1, "clip_strength": 1}
    assert applied[1] == applied[3] == {"state": {"lora": "/models/loras/scenery.safetensors"}, "strength": 0.7, "clip_strength": 0.7}
    assert runtime.base.clone.call_count == 2


@pytest.mark.parametrize("encoded", [("positive", "negative"), (None, "negative", None), None])
def test_invalid_conditioning_fails_before_sampling(runtime, monkeypatch, encoded):
    def call(name, **kwargs):
        return encoded if name == "TextEncodeQwenImage21" else runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    with pytest.raises(RuntimeError, match="invalid image conditioning"):
        generation.generate_image({}, "missing-latent", threading.Event(), runtime.reference, runtime.background)
    assert "KSampler" not in dict(runtime.calls)


@pytest.mark.parametrize("reference", [None, "image.png", {}, 1])
def test_reference_is_required_before_loading_models(runtime, reference):
    with pytest.raises(ValueError, match="captured frame"):
        generation.generate_image({}, "no-reference", threading.Event(), reference, runtime.background)
    assert runtime.calls == []


@pytest.mark.parametrize("size", [(100, 100), (16, 16), (4112, 64), (4096, 4096)])
def test_rejects_invalid_capture_geometry_before_loading(runtime, size):
    with pytest.raises(ValueError, match="Captured frame"):
        generation.generate_image({}, "bad-capture", threading.Event(), Image.new("RGB", size), runtime.background)
    assert runtime.calls == []


@pytest.mark.parametrize("megapixels,latent_size,decoded_size", [
    (1.0, (1369, 766), (1376, 768)),
    (2.0, (1936, 1083), (1936, 1088)),
    (4.0, (2738, 1532), (2736, 1536)),
])
def test_resolution_scale_drives_native_latent_dimensions_and_saves_decoded_size(runtime, megapixels, latent_size, decoded_size):
    runtime.latent_size[:], runtime.size[:] = latent_size, decoded_size
    image, metadata = generation.generate_image({"resolution_mp": megapixels}, "scaled-frame", threading.Event(), runtime.reference, runtime.background)
    calls = dict(runtime.calls)
    assert runtime.reference.size == (1888, 1056)
    assert calls["ImageScaleToTotalPixels"] == {"image": calls["TextEncodeQwenImage21"]["images"]["image_2"],
        "upscale_method": "nearest-exact", "megapixels": megapixels, "resolution_steps": 1}
    assert calls["GetImageSize"] == {"image": "scaled-camera-frame"}
    assert calls["EmptyLatentImage"] == {"width": latent_size[0], "height": latent_size[1], "batch_size": 1}
    assert calls["KSampler"]["latent_image"]["size"] == latent_size
    assert image.size == (metadata["width"], metadata["height"]) == decoded_size
    assert metadata["settings"]["resolution_mp"] == megapixels


def test_turbo_uses_existing_adapter_sigmas_and_advanced_sampler(runtime, monkeypatch):

    def sigmas(latent):
        runtime.calls.append(("TurboSigmas", {"latent": latent}))
        return [1, 0.9375, 0.875, 0.75, 0.5, 0.25, 0]

    monkeypatch.setattr(generation, "_viggle_turbo_sigmas", sigmas)
    settings = {"generation": {"turbo": True, "steps": 6, "cfg": 1}}
    generation.generate_image(settings, "turbo", threading.Event(), runtime.reference, runtime.background)
    names = [name for name, _ in runtime.calls]
    assert names.index("LoraLoaderModelOnly") < names.index("QwenImage21Cache")
    assert dict(runtime.calls)["LoraLoaderModelOnly"]["lora_name"] == generation.TURBO_LORA
    assert "KSampler" not in names and "SamplerCustomAdvanced" in names
    assert dict(runtime.calls)["KSamplerSelect"]["sampler_name"] == "euler"
    assert dict(runtime.calls)["BasicGuider"]["conditioning"] == "positive"
    assert dict(runtime.calls)["TurboSigmas"]["latent"] == {"samples": "requested-frame", "size": (1936, 1083)}


@pytest.mark.parametrize("settings", [{"steps": 25, "cfg": 1, "denoise": 1}, {"steps": 6, "cfg": 2, "denoise": 1}])
def test_copied_turbo_sampler_rejects_unsupported_schedule(runtime, settings):
    with pytest.raises(ValueError, match="six steps and CFG 1"):
        generation._sample_generation_latent("model", "positive", "negative", {}, settings, turbo=True)
    assert runtime.calls == []


def test_turbo_controls_normalize_like_the_original_creator():
    previous = {"steps": 31, "cfg": 2.5}
    settings = generation.normalize_settings({"generation": {"turbo": True, "steps": 99, "cfg": 5,
        "turbo_previous_settings": previous}})["generation"]
    assert settings["steps"] == 6 and settings["cfg"] == 1
    assert settings["turbo_previous_settings"] == previous


def test_copied_viggle_schedule_uses_sampling_token_count_without_torch(monkeypatch):
    class Tensor(np.ndarray):
        def new_zeros(self, count):
            return np.zeros(count).view(Tensor)

        def float(self):
            return self.astype(np.float32)

    stub = types.SimpleNamespace(float64=np.float64,
        tensor=lambda values, dtype=None: np.asarray(values, dtype=dtype).view(Tensor),
        cat=lambda values: np.concatenate(values).view(Tensor))
    monkeypatch.setitem(sys.modules, "torch", stub)
    def schedule(height, width, ratio=16):
        return generation._viggle_turbo_sigmas({"samples": types.SimpleNamespace(shape=(1, 64, height, width)),
                                               "downscale_ratio_spacial": ratio})
    small, large = schedule(32, 32), schedule(64, 96)
    assert len(large) == 7 and large[0] == 1 and large[-1] == 0
    assert np.all(np.diff(large) < 0) and large.dtype == np.float32
    assert np.all(large[1:-1] > small[1:-1])
    np.testing.assert_allclose(schedule(32, 48, 32), large)


def test_selected_builtin_turbo_model_uses_standard_eight_step_sampling(runtime):
    settings = {"generation": {"diffusion_model_name": "QI2/" + generation.TURBO_MODEL, "turbo": True}}
    _image, metadata = generation.generate_image(settings, "builtin-turbo", threading.Event(), runtime.reference, runtime.background)
    calls = dict(runtime.calls)
    assert calls["UNETLoader"]["unet_name"] == "QI2/" + generation.TURBO_MODEL
    assert calls["KSampler"]["steps"] == 8 and calls["KSampler"]["cfg"] == 1
    assert "LoraLoaderModelOnly" not in calls and "SamplerCustomAdvanced" not in calls
    assert metadata["settings"]["generation"]["turbo"] is False


@pytest.mark.parametrize("model", ["QI2/" + generation.TURBO_MODEL, "User models/other_family_custom.safetensors"])
def test_restored_model_and_authored_sampling_reach_the_loader_and_sampler(runtime, model):
    settings = {"generation": {"diffusion_model_name": model, "steps": 19, "cfg": 2.5,
        "seed": 918273, "seed_mode": "fixed", "sampler": "dpmpp_2m", "scheduler": "normal"}}
    _image, metadata = generation.generate_image(settings, "restored-model", threading.Event(), runtime.reference, runtime.background)
    calls = dict(runtime.calls)
    assert calls["UNETLoader"]["unet_name"] == model
    for key, sampler_key in [("steps", "steps"), ("cfg", "cfg"), ("seed", "seed"), ("sampler", "sampler_name"), ("scheduler", "scheduler")]:
        assert calls["KSampler"][sampler_key] == settings["generation"][key]
    for key, value in settings["generation"].items():
        assert metadata["settings"]["generation"][key] == value


def test_random_seed_is_resolved_once_and_saved_with_the_actual_frame(runtime, monkeypatch):
    bits = mock.Mock(side_effect=[0, 42])
    monkeypatch.setattr(generation.random, "getrandbits", bits)
    settings = {"generation": {"seed": 7, "seed_mode": "randomize"}}
    _image, metadata = generation.generate_image(settings, "random", threading.Event(), runtime.reference, runtime.background)
    assert dict(runtime.calls)["KSampler"]["seed"] == metadata["seed"] == 42
    assert metadata["settings"]["generation"]["seed"] == 42
    assert settings["generation"]["seed"] == 7 and bits.call_count == 2


def test_changed_model_file_invalidates_cache_but_each_request_always_gets_a_fresh_clip(runtime, monkeypatch, tmp_path):
    paths = {}
    for category, name in generation.ASSETS.items():
        paths[(category, name)] = tmp_path / (category + ".safetensors")
        paths[(category, name)].write_bytes(b"original weights")
    monkeypatch.setattr(sys.modules["folder_paths"], "get_full_path", lambda category, name: str(paths[(category, name)]))
    generation.generate_image({}, "first", threading.Event(), runtime.reference, runtime.background)
    generation.generate_image({}, "unchanged", threading.Event(), runtime.reference, runtime.background)
    assert [name for name, _ in runtime.calls].count("UNETLoader") == 1
    paths[("diffusion_models", generation.ASSETS["diffusion_models"])].write_bytes(b"replacement weights with a changed size")
    generation.generate_image({}, "changed", threading.Event(), runtime.reference, runtime.background)
    names = [name for name, _ in runtime.calls]
    assert names.count("UNETLoader") == names.count("VAELoader") == 2
    assert names.count("CLIPLoader") == 3 and len(set(runtime.clips)) == 3


def test_copied_lora_file_cache_uses_file_signatures_and_skips_disabled_slots(monkeypatch, tmp_path):
    path = tmp_path / "scenery.safetensors"
    path.write_bytes(b"original")
    loader = mock.Mock(side_effect=lambda *_args, **_kwargs: ({"weights": path.read_bytes()}, {"source": "fixture"}))
    monkeypatch.setattr(sys.modules["comfy.utils"], "load_torch_file", loader, raising=False)
    generation._cached_lora_file.cache_clear()
    first = generation._load_lora_file(str(path))
    assert generation._load_lora_file(str(path)) == first
    loader.assert_called_once_with(str(path), safe_load=True, return_metadata=True)
    path.write_bytes(b"changed and longer")
    assert generation._load_lora_file(str(path))[0]["weights"] == b"changed and longer"
    assert loader.call_count == 2
    lookup = mock.Mock(side_effect=AssertionError("Empty or zero-strength slots must not load a LoRA"))
    monkeypatch.setattr(generation, "_get_lora_full_path", lookup)
    assert generation.apply_generation_lora("model", "clip", "", 1) == ("model", "clip")
    assert generation.apply_generation_lora("model", "clip", "scenery.safetensors", 0) == ("model", "clip")
    lookup.assert_not_called()
    generation._cached_lora_file.cache_clear()


def test_missing_selected_lora_fails_instead_of_silently_omitting_it(monkeypatch):
    monkeypatch.setattr(generation, "_get_lora_full_path", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="Selected LoRA is not installed"):
        generation.apply_generation_lora("model", "clip", "missing.safetensors", 1)


def test_asset_resolution_uses_exact_paths_or_one_unique_basename(monkeypatch):
    name = generation.ASSETS["diffusion_models"]
    monkeypatch.setattr(generation, "_installed", lambda _category: ["QI2/" + name])
    assert generation._asset_name("diffusion_models", name) == "QI2/" + name
    monkeypatch.setattr(generation, "_installed", lambda _category: ["QI2/" + name, "other/" + name])
    assert generation._asset_name("diffusion_models", "QI2/" + name) == "QI2/" + name
    with pytest.raises(FileNotFoundError, match="exactly one"):
        generation._asset_name("diffusion_models", name)
    with pytest.raises(ValueError, match="relative"):
        generation._asset_name("diffusion_models", "../" + name)


def test_refinement_remains_independent_of_the_main_vnccs_runtime(runtime, monkeypatch):
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {}, raising=False)
    image, _metadata = generation.generate_image({}, "standalone", threading.Event(), runtime.reference, runtime.background)
    assert image.size == (1936, 1088)


@pytest.mark.parametrize("background", [None, "background.png", {}, 1])
def test_original_background_is_required_before_loading_models(runtime, background):
    with pytest.raises(ValueError, match="original background"):
        generation.generate_image({}, "no-background", threading.Event(), runtime.reference, background)
    assert runtime.calls == []


def test_anyangle_is_always_applied_once_at_full_strength(runtime):
    settings = {"generation": {"lora_stack": [{"name": generation.ANYANGLE_LORA, "strength": 0.2}]}}
    generation.generate_image(settings, "mandatory-adapter", threading.Event(), runtime.reference, runtime.background)
    applied = [values for name, values in runtime.calls if name == "ApplyLoRA"]
    assert len(applied) == 1 and applied[0]["strength"] == applied[0]["clip_strength"] == 1


def test_encoder_latent_never_controls_output_dimensions(runtime, monkeypatch):
    def call(name, **kwargs):
        if name == "TextEncodeQwenImage21":
            return "positive", "negative", None
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    generation.generate_image({}, "requested-frame", threading.Event(), runtime.reference, runtime.background)
    assert dict(runtime.calls)["KSampler"]["latent_image"] == {"samples": "requested-frame", "size": (1936, 1083)}


@pytest.fixture
def anyangle_store(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    cached = tmp_path / "hub-cache.safetensors"
    weights = b"pinned AnyAngle weights"
    cached.write_bytes(weights)
    monkeypatch.setattr(generation, "ANYANGLE_SHA256", hashlib.sha256(weights).hexdigest())
    folder_paths = sys.modules["folder_paths"]
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda category: [str(root)] if category == "loras" else [], raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda category: [path.relative_to(root).as_posix() for path in root.rglob("*.safetensors")] if category == "loras" else [])
    monkeypatch.setattr(generation, "_get_lora_full_path", lambda name: str(root / name) if (root / name).is_file() else None)
    download = mock.Mock(return_value=str(cached))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    return types.SimpleNamespace(root=root, cached=cached, weights=weights, download=download,
                                 target=root / generation.ANYANGLE_LORA)


def test_anyangle_download_is_pinned_anonymous_atomic_and_reused(anyangle_store):
    store = anyangle_store
    assert generation.resolve_anyangle_lora(threading.Event()) == generation.ANYANGLE_LORA
    assert store.target.read_bytes() == store.weights
    store.download.assert_called_once_with(repo_id="lilylilith/QI_2.1_AnyAngle", filename="QI2.1_AnyAngle.safetensors",
                                           revision=generation.ANYANGLE_REVISION, token=False)
    assert len(generation.ANYANGLE_REVISION) == 40
    assert not list(store.root.rglob("*.tmp"))
    assert generation.resolve_anyangle_lora(threading.Event()) == generation.ANYANGLE_LORA
    assert store.download.call_count == 1


def test_anyangle_reuses_an_existing_control_center_lora_folder(anyangle_store):
    path = anyangle_store.root / "QI2.1" / "imported" / generation.ANYANGLE_FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(anyangle_store.weights)
    assert generation.resolve_anyangle_lora(threading.Event()) == path.relative_to(anyangle_store.root).as_posix()
    anyangle_store.download.assert_not_called()


def test_invalid_anyangle_download_never_publishes_partial_weights(anyangle_store):
    anyangle_store.cached.write_bytes(b"interrupted download")
    with pytest.raises(ValueError, match="invalid or incomplete"):
        generation.resolve_anyangle_lora(threading.Event())
    assert not anyangle_store.target.exists() and not list(anyangle_store.root.rglob("*.tmp"))


def test_existing_invalid_anyangle_file_is_preserved(anyangle_store):
    anyangle_store.target.parent.mkdir(parents=True)
    anyangle_store.target.write_bytes(b"damaged original")
    with pytest.raises(ValueError, match="invalid or incomplete"):
        generation.resolve_anyangle_lora(threading.Event())
    assert anyangle_store.target.read_bytes() == b"damaged original"
    anyangle_store.download.assert_not_called()


def test_anyangle_installation_rejects_symlinks_outside_the_model_root(anyangle_store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (anyangle_store.root / "QI2.1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the configured LoRA folder"):
        generation.resolve_anyangle_lora(threading.Event())
    anyangle_store.download.assert_not_called()
    assert not list(outside.iterdir())


def test_cancelled_anyangle_download_does_not_install_weights(anyangle_store):
    event = threading.Event()
    def download(**_kwargs):
        event.set()
        return str(anyangle_store.cached)
    anyangle_store.download.side_effect = download
    with pytest.raises(generation.GenerationCancelled):
        generation.resolve_anyangle_lora(event)
    assert not anyangle_store.target.exists()


def test_concurrent_anyangle_requests_download_only_once(anyangle_store):
    with ThreadPoolExecutor(max_workers=2) as worker:
        results = list(worker.map(lambda _: generation.resolve_anyangle_lora(threading.Event()), range(2)))
    assert results == [generation.ANYANGLE_LORA, generation.ANYANGLE_LORA]
    assert anyangle_store.download.call_count == 1


@pytest.fixture
def allocator(monkeypatch):
    calls = []
    modules = {
        "comfy.memory_management": types.SimpleNamespace(aimdo_enabled=True),
        "comfy.model_management": types.SimpleNamespace(reset_cast_buffers=lambda: calls.append("cast"), unload_all_models=mock.Mock()),
        "comfy.model_prefetch": types.SimpleNamespace(cleanup_prefetch_queues=lambda: calls.append("prefetch")),
        "comfy_aimdo.model_vbar": types.SimpleNamespace(vbars_reset_watermark_limits=lambda: calls.append("watermark")),
    }
    monkeypatch.setitem(sys.modules, "comfy_aimdo", types.ModuleType("comfy_aimdo"))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, attribute = name.rsplit(".", 1)
        monkeypatch.setattr(sys.modules[parent], attribute, module, raising=False)
    return types.SimpleNamespace(calls=calls, modules=modules)


@pytest.mark.parametrize("failed", [False, True])
def test_copied_creator_stage_cleanup_handles_nesting_and_failure(allocator, failed):
    def run():
        with runtime_cleanup.inference_stage():
            with runtime_cleanup.inference_stage():
                assert allocator.calls == []
                if failed:
                    raise RuntimeError("stage failed")
    if failed:
        with pytest.raises(RuntimeError, match="stage failed"):
            run()
    else:
        run()
    assert allocator.calls == ["prefetch", "cast", "watermark"]
    allocator.modules["comfy.model_management"].unload_all_models.assert_not_called()


def test_lora_failure_cleans_dynamic_vram_and_releases_model_ownership(runtime, allocator, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("LoRA load failed")
    monkeypatch.setattr(generation, "apply_generation_lora", fail)
    with pytest.raises(RuntimeError, match="LoRA load failed"):
        generation.generate_image({}, "failed-adapter", threading.Event(), runtime.reference, runtime.background)
    assert allocator.calls == ["prefetch", "cast", "watermark"] * 2
    assert generation._ACTIVE is None
    def take_lock():
        acquired = runtime_cleanup.inference_lock.acquire(timeout=0.5)
        if acquired:
            runtime_cleanup.inference_lock.release()
        return acquired
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(take_lock).result()
    allocator.modules["comfy.model_management"].unload_all_models.assert_not_called()


@pytest.mark.parametrize("blocked", [False, True])
def test_copied_native_caller_unwraps_v3_results_and_cleans_failed_calls(allocator, monkeypatch, blocked):
    class NodeOutput:
        result = ("model", "clip")
        block_execution = "Blocked node" if blocked else None
    class Node:
        FUNCTION = "execute"
        def execute(self, value):
            assert value == "input"
            return NodeOutput()
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {"Test": Node}, raising=False)
    if blocked:
        with pytest.raises(RuntimeError, match="Blocked node"):
            generation._call_comfy_node("Test", value="input")
    else:
        assert generation._call_comfy_node("Test", value="input") == ("model", "clip")
    assert allocator.calls == ["prefetch", "cast", "watermark"]


def test_native_caller_rejects_contract_mismatch_instead_of_dropping_arguments(allocator, monkeypatch):
    class Node:
        FUNCTION = "execute"
        def execute(self, value):
            raise AssertionError("An incompatible node must not execute")
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {"Test": Node}, raising=False)
    with pytest.raises(RuntimeError, match="incompatible interface"):
        generation._call_comfy_node("Test", value="input", unsupported=True)


def test_copied_native_caller_keeps_nested_loader_cleanup_at_the_stage_boundary(allocator, monkeypatch):
    class Node:
        FUNCTION = "execute"
        def execute(self):
            return ("result",)
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {"Test": Node}, raising=False)
    with runtime_cleanup.inference_stage():
        assert generation._call_comfy_node("Test") == ("result",)
        assert allocator.calls == []
    assert allocator.calls == ["prefetch", "cast", "watermark"]


def test_native_v3_image_size_gets_a_private_hidden_holder(allocator, monkeypatch):
    class GetImageSize:
        FUNCTION = "execute"
        hidden = None

        @classmethod
        def PREPARE_CLASS_CLONE(cls, v3_data):
            assert v3_data is None
            class Prepared(cls):
                hidden = types.SimpleNamespace(unique_id=None)
            return Prepared

        @classmethod
        def execute(cls, image):
            assert cls.hidden.unique_id is None
            return image.shape[2], image.shape[1], image.shape[0]

    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {"GetImageSize": GetImageSize}, raising=False)
    assert generation._call_comfy_node("GetImageSize", image=types.SimpleNamespace(shape=(1, 766, 1369, 3))) == (1369, 766, 1)
    assert GetImageSize.hidden is None
    assert allocator.calls == ["prefetch", "cast", "watermark"]


def test_cancelled_pending_job_never_loads_or_interrupts_other_jobs(runtime):
    event = threading.Event()
    event.set()
    with pytest.raises(generation.GenerationCancelled):
        generation.generate_image({}, "cancelled", event, runtime.reference, runtime.background)
    assert runtime.calls == []
    assert generation.interrupt("another-widget") is False
    generation._set_interrupt.assert_not_called()


def test_cancel_while_waiting_does_not_interrupt_the_current_gpu_owner(runtime, monkeypatch):
    event = threading.Event()
    waiting_lock = mock.Mock()
    def acquire(*, timeout):
        event.set()
        return False
    waiting_lock.acquire.side_effect = acquire
    monkeypatch.setattr(generation, "model_operation_lock", lambda: waiting_lock)
    with pytest.raises(generation.GenerationCancelled):
        generation.generate_image({}, "waiting", event, runtime.reference, runtime.background)
    assert runtime.calls == [] and generation._ACTIVE is None
    waiting_lock.release.assert_not_called()
    generation._set_interrupt.assert_not_called()


def test_new_owned_job_clears_previous_workflow_interrupt_before_loading(runtime, monkeypatch):
    def call(name, **kwargs):
        generation._set_interrupt.assert_called_once_with(False)
        return runtime.call(name, **kwargs)
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    generation.generate_image({}, "fresh-request", threading.Event(), runtime.reference, runtime.background)
    assert generation._ACTIVE is None


def test_interrupt_only_targets_the_active_cg_job(runtime):
    event = threading.Event()
    with generation._model_session("owned", event):
        assert generation.interrupt("another-widget") is False
        assert generation.interrupt("owned") is True
        assert event.is_set()
    assert generation._set_interrupt.call_args_list == [mock.call(False), mock.call(True), mock.call(False)]
    assert generation._ACTIVE is None and generation.interrupt("owned") is False


def test_cancel_after_encoding_discards_result_before_sampling(runtime, monkeypatch):
    event = threading.Event()
    def call(name, **kwargs):
        result = runtime.call(name, **kwargs)
        if name == "TextEncodeQwenImage21":
            event.set()
        return result
    monkeypatch.setattr(generation, "_call_comfy_node", call)
    with pytest.raises(generation.GenerationCancelled):
        generation.generate_image({}, "cancel-encode", event, runtime.reference, runtime.background)
    assert "KSampler" not in dict(runtime.calls) and generation._ACTIVE is None


def test_configuration_reports_all_local_models_and_installed_loras_without_loads(monkeypatch):
    import comfy.samplers
    monkeypatch.setattr(comfy.samplers.KSampler, "SAMPLERS", ["euler"])
    monkeypatch.setattr(comfy.samplers.KSampler, "SCHEDULERS", ["simple"])
    installed = {category: ["QI2/" + name] for category, name in generation.ASSETS.items()}
    installed["diffusion_models"].extend(["QI2/" + generation.TURBO_MODEL, "moss_image.safetensors"])
    installed["loras"] = ["scenery/example.safetensors", generation.TURBO_LORA]
    monkeypatch.setattr(generation, "_installed", lambda category: installed.get(category, []))
    config = generation.configuration()
    assert config["installed"] is True and config["model"] == "Qwen Image 2.1 INT8 ConvRot"
    assert config["loras"] == installed["loras"]
    assert config["samplers"] == ["euler"] and config["schedulers"] == ["simple"]
    assert config["settings"] == generation.DEFAULT_SETTINGS and "editor" not in config
    assert config["diffusion_models"] == installed["diffusion_models"]
    assert generation.normalize_settings({"generation": {"diffusion_model_name": "local_alias.safetensors"}})["generation"]["diffusion_model_name"] == "local_alias.safetensors"
    assert config["turbo_lora"]["status"] == "installed"
    assert config["turbo_lora"]["hf_repo"] == "Viggle/Qwen-Image-2.1-viggle-turbo"
    assert config["turbo_lora"]["filename"] == generation.TURBO_LORA
    assert [entry["name"] for entry in config["model_presets"]] == [entry["name"] for entry in generation.MODEL_PRESETS]
    assert all(entry["status"] == "installed" for entry in config["model_presets"])
    installed["vae"] = []
    assert generation.configuration()["installed"] is False
    installed["loras"] = []
    assert generation.configuration()["turbo_lora"]["status"] == "missing"
    installed["diffusion_models"] = []
    presets = generation.configuration()["model_presets"]
    assert len(presets) == 2 and all(entry["status"] == "missing" for entry in presets)
