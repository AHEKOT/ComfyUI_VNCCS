"""Security-focused tests for character generator path handling."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")

from nodes import character_generator as cg


def test_character_root_ignores_external_sheets_path(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    external = tmp_path / "elsewhere" / "Sheets" / "Bad"
    char_root.mkdir(parents=True)
    external.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._character_root_from_sheets_path(str(external), "Alice") == str(char_root)


def test_character_root_accepts_windows_style_sheets_path(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    sheets = char_root / "Sheets" / "Naked" / "neutral"
    sheets.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    windows_style = str(sheets).replace(os.sep, "\\")
    assert cg._character_root_from_sheets_path(windows_style, "Alice") == str(char_root)
    assert cg._costume_name_from_sheets_path(windows_style) == "Naked"


def test_cache_tensor_path_rejects_external_cache(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    outside = tmp_path / "outside" / "cache"
    outside.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._cache_tensor_path(str(outside), "stage") == ""


def test_emotion_output_prefix_must_stay_under_character_sprites(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    safe_prefix = char_root / "Sprites" / "Happy" / "Neutral" / "sprite_"
    unsafe_prefix = tmp_path / "outside" / "sprite_"
    char_root.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._safe_emotion_output_prefix(str(safe_prefix), "Alice") == str(safe_prefix)
    assert cg._safe_emotion_output_prefix(str(unsafe_prefix), "Alice") == ""


def test_bg_remove_disabled_skips_chroma_key(monkeypatch):
    torch = pytest.importorskip("torch")

    class FailingChromaKey:
        def chroma_key(self, *args, **kwargs):
            raise AssertionError("chroma key should not run")

    monkeypatch.setattr(cg, "VNCCSChromaKey", FailingChromaKey)
    images = torch.rand(1, 4, 4, 3)

    result = cg.VNCCS_CharacterGenerator()._run_bg_remove(
        images,
        {"preset": "disabled"},
        background="Green",
    )

    assert torch.equal(result, images)


def test_settings_force_internal_rmbg_off_for_legacy_workflows():
    settings = cg.VNCCS_CharacterGenerator()._settings(
        json.dumps({"bg_remove": {"use_internal_rmbg": True}})
    )

    assert settings["bg_remove"]["use_internal_rmbg"] is False


def test_internal_rmbg_cannot_run_when_directly_requested(monkeypatch):
    torch = pytest.importorskip("torch")

    class FailingRMBG:
        def process_image(self, *args, **kwargs):
            raise AssertionError("internal RMBG should be force-disabled")

    generator = cg.VNCCS_CharacterGenerator()
    monkeypatch.setattr(cg, "VNCCS_RMBG2", FailingRMBG)
    monkeypatch.setattr(
        generator,
        "_run_seedvr_upscale_one",
        lambda image, dit, vae, settings, seed: image,
    )
    image = torch.rand(1, 4, 4, 3)

    result = generator._run_upscale_one(
        image,
        dit=None,
        vae=None,
        background="Green",
        settings={},
        seed=42,
        use_internal_rmbg=True,
    )

    assert torch.equal(result, image)


def test_upscaler_batch_internal_rmbg_cannot_run_when_directly_requested(monkeypatch):
    torch = pytest.importorskip("torch")

    class FailingRMBG:
        def process_image(self, *args, **kwargs):
            raise AssertionError("internal RMBG should be force-disabled")

    generator = cg.VNCCS_CharacterGenerator()
    monkeypatch.setattr(cg, "VNCCS_RMBG2", FailingRMBG)
    monkeypatch.setattr(
        generator,
        "_run_upscaler_models",
        lambda settings, node_id=None: (None, None),
    )
    monkeypatch.setattr(
        generator,
        "_run_seedvr_upscale_batch",
        lambda images, dit, vae, settings, seed, **kwargs: images,
    )
    monkeypatch.setattr(generator, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator, "_log_stage", lambda *args, **kwargs: None)
    images = torch.rand(2, 4, 4, 3)

    result = generator._run_upscaler(
        images,
        "Green",
        {"mode": "seedvr"},
        seed=42,
        use_internal_rmbg=True,
    )

    assert torch.equal(result, images)


def test_clothes_internal_rmbg_cannot_run_when_directly_requested(monkeypatch):
    torch = pytest.importorskip("torch")

    class FailingRMBG:
        def process_image(self, *args, **kwargs):
            raise AssertionError("internal RMBG should be force-disabled")

    generator = cg.VNCCS_ClothesGenerator()
    monkeypatch.setattr(cg, "VNCCS_RMBG2", FailingRMBG)
    monkeypatch.setattr(
        generator,
        "_run_pose_generation",
        lambda poses, character, pipe, prompt, settings, **kwargs: poses,
    )
    poses = torch.rand(1, 4, 4, 3)

    result = generator._run_clothes_pose_generation(
        poses,
        character=None,
        pipe=None,
        prompt="",
        background="Green",
        settings={},
        use_internal_rmbg=True,
    )

    assert torch.equal(result, poses)


def test_bg_remove_uses_sam3_details_recovery_by_default(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class CapturingChromaKey:
        def chroma_key(self, *args, **kwargs):
            seen["use_sam3_recovery_mask"] = args[12]
            return (args[0], None, None)

    monkeypatch.setattr(cg, "VNCCSChromaKey", CapturingChromaKey)
    images = torch.rand(1, 4, 4, 3)

    cg.VNCCS_CharacterGenerator()._run_bg_remove(
        images,
        {"preset": "balanced"},
        background="Green",
    )

    assert seen["use_sam3_recovery_mask"] is True


def test_bg_remove_can_disable_sam3_details_recovery(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class CapturingChromaKey:
        def chroma_key(self, *args, **kwargs):
            seen["use_sam3_recovery_mask"] = args[12]
            return (args[0], None, None)

    monkeypatch.setattr(cg, "VNCCSChromaKey", CapturingChromaKey)
    images = torch.rand(1, 4, 4, 3)

    cg.VNCCS_CharacterGenerator()._run_bg_remove(
        images,
        {"preset": "balanced", "use_sam3_details_recovery": False},
        background="Green",
    )

    assert seen["use_sam3_recovery_mask"] is False


def test_bg_remove_custom_settings_reach_chroma_and_sam3(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class CapturingChromaKey:
        def chroma_key(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return (args[0], None, None)

    monkeypatch.setattr(cg, "VNCCSChromaKey", CapturingChromaKey)
    images = torch.rand(1, 4, 4, 3)
    settings = {
        "preset": "balanced",
        "use_preset_values": False,
        "tolerance": 0.31,
        "softness": 0.22,
        "despill_strength": 0.77,
        "edge_width": 6,
        "matte_cleanup": 0.44,
        "foreground_recover": 0.66,
        "edge_decontaminate": 0.88,
        "edge_choke": 0.12,
        "matte_method": "chroma_soft",
        "screen_mode": "blue",
        "output_mode": "premultiplied_rgba",
        "use_sam3_details_recovery": True,
        "sam3_prompt": "face, hair",
        "sam3_threshold": 0.63,
    }

    cg.VNCCS_CharacterGenerator()._run_bg_remove(images, settings, background="Green")

    args = seen["args"]
    assert args[1:9] == pytest.approx((0.31, 0.22, 0.77, 6, 0.44, 0.66, 0.88, 0.12))
    assert args[9:13] == ("chroma_soft", "blue", "premultiplied_rgba", True)
    assert seen["kwargs"]["sam3_settings"] is settings


def test_generator_internal_node_settings_are_forwarded(monkeypatch):
    torch = pytest.importorskip("torch")
    decoded = torch.rand(1, 8, 8, 3)
    calls = {}

    class FakeMaskExtractor:
        def fill_alpha_with_color(self, image):
            return (image,)

    class TestGenerator(cg.VNCCS_CharacterGenerator):
        def _extract_pipe(self, pipe):
            return {
                "clip": object(),
                "vae": object(),
                "model": object(),
                "seed": 10,
                "steps": 11,
                "cfg": 1.5,
                "sampler": "euler",
                "scheduler": "simple",
            }

        def _run_list_mapped(self, class_name, list_kwargs, **kwargs):
            calls[class_name] = kwargs
            if class_name == "VNCCS_QWEN_Encoder":
                return ([object()], [object()], [{"samples": torch.rand(1, 4, 8, 8)}])
            if class_name == "KSampler":
                return ([{"samples": torch.rand(1, 4, 8, 8)}],)
            if class_name == "VAEDecodeTiled":
                return ([decoded],)
            raise AssertionError(class_name)

        def _apply_pose_lora_to_model(self, model, clip, pipe, lora_info):
            return model

        def _validate_conditioning_for_model(self, *args, **kwargs):
            return None

    monkeypatch.setattr(cg, "VNCCS_MaskExtractor", FakeMaskExtractor)
    generator = TestGenerator()
    generator._run_pose_generation(
        torch.rand(1, 8, 8, 3),
        torch.rand(1, 8, 8, 3),
        object(),
        "prompt",
        {
            "target_size": 1344,
            "upscale_method": "area",
            "crop_method": "pad",
            "vl_size": 512,
            "weight1": 0.75,
            "qwen_2511": False,
        },
        sampler_settings={
            "inherit_pipe": False,
            "seed": 99,
            "steps": 23,
            "cfg": 4.25,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 0.82,
        },
        vae_decode_settings={
            "tile_size": 768,
            "overlap": 96,
            "temporal_size": 32,
            "temporal_overlap": 4,
        },
    )

    assert calls["VNCCS_QWEN_Encoder"]["target_size"] == 1344
    assert calls["VNCCS_QWEN_Encoder"]["upscale_method"] == "area"
    assert calls["VNCCS_QWEN_Encoder"]["crop_method"] == "pad"
    assert calls["VNCCS_QWEN_Encoder"]["vl_size"] == 512
    assert calls["VNCCS_QWEN_Encoder"]["weight1"] == pytest.approx(0.75)
    assert calls["VNCCS_QWEN_Encoder"]["qwen_2511"] is False
    assert calls["KSampler"]["seed"] == 99
    assert calls["KSampler"]["steps"] == 23
    assert calls["KSampler"]["cfg"] == pytest.approx(4.25)
    assert calls["KSampler"]["sampler_name"] == "dpmpp_2m"
    assert calls["KSampler"]["scheduler"] == "karras"
    assert calls["KSampler"]["denoise"] == pytest.approx(0.82)
    assert calls["VAEDecodeTiled"]["tile_size"] == 768
    assert calls["VAEDecodeTiled"]["overlap"] == 96
    assert calls["VAEDecodeTiled"]["temporal_size"] == 32
    assert calls["VAEDecodeTiled"]["temporal_overlap"] == 4


def test_emotion_detailer_uses_pipe_values_and_forwards_local_controls(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand(1, 16, 16, 3)
    mask = torch.ones((1, 16, 16), dtype=torch.float32)
    seen = {}
    bbox_detector = object()
    segm_detector = object()
    sam_model = object()

    class TestGenerator(cg.VNCCS_EmotionsGenerator):
        def _extract_pipe(self, pipe):
            return {
                "clip": object(),
                "vae": object(),
                "model": object(),
                "seed": 1,
                "steps": 12,
                "cfg": 1.0,
                "denoise": 0.42,
                "sampler": "euler",
                "scheduler": "simple",
            }

    def fake_call(class_name, **kwargs):
        if class_name == "CLIPTextEncode":
            return (object(),)
        if class_name == "UltralyticsDetectorProvider":
            if kwargs["model_name"] == "bbox/custom.pt":
                return (bbox_detector, object())
            return (object(), segm_detector)
        if class_name == "SAMLoader":
            seen["sam_loader"] = kwargs
            return (sam_model,)
        if class_name == "FaceDetailer":
            seen["face_detailer"] = kwargs
            return (image, image, None, mask)
        raise AssertionError(class_name)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_call)
    TestGenerator()._run_emotion_generation_one(
        image,
        mask,
        object(),
        "smile",
        "",
        "",
        123,
        detailer_settings={
            "bbox_model": "bbox/custom.pt",
            "segm_model": "segm/custom.pt",
            "sam_model": "sam_custom.pth",
            "sam_device_mode": "CPU",
            "use_sam": True,
            "guide_size": 1024,
            "guide_size_for": False,
            "max_size": 2048,
            "inherit_pipe_sampler": False,
            "steps": 31,
            "cfg": 3.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "face_denoise": 0.67,
            "feather": 9,
            "noise_mask": False,
            "force_inpaint": False,
            "bbox_crop_factor": 5.25,
            "sam_detection_hint": "rect-4",
            "sam_mask_hint_threshold": 0.61,
            "sam_mask_hint_use_negative": "True",
            "drop_size": 15,
            "cycle": 2,
            "inpaint_model": True,
            "noise_mask_feather": 24,
            "tiled_encode": False,
            "tiled_decode": False,
        },
    )

    assert seen["sam_loader"] == {"model_name": "sam_custom.pth", "device_mode": "CPU"}
    detailer = seen["face_detailer"]
    assert detailer["bbox_detector"] is bbox_detector
    assert detailer["segm_detector_opt"] is segm_detector
    assert detailer["sam_model_opt"] is sam_model
    assert detailer["guide_size"] == 1024
    assert detailer["guide_size_for"] is False
    assert detailer["max_size"] == 2048
    assert detailer["steps"] == 12
    assert detailer["cfg"] == pytest.approx(1.0)
    assert detailer["sampler_name"] == "dpmpp_2m"
    assert detailer["scheduler"] == "karras"
    assert detailer["denoise"] == pytest.approx(0.42)
    assert detailer["feather"] == 9
    assert detailer["noise_mask"] is False
    assert detailer["force_inpaint"] is False
    assert detailer["bbox_crop_factor"] == pytest.approx(5.25)
    assert detailer["sam_detection_hint"] == "rect-4"
    assert detailer["sam_mask_hint_threshold"] == pytest.approx(0.61)
    assert detailer["sam_mask_hint_use_negative"] == "True"
    assert detailer["drop_size"] == 15
    assert detailer["cycle"] == 2
    assert detailer["inpaint_model"] is True
    assert detailer["noise_mask_feather"] == 24
    assert detailer["tiled_encode"] is False
    assert detailer["tiled_decode"] is False


def test_emotion_detailer_defaults_match_face_detailer_and_step3_workflow():
    defaults = cg.DEFAULT_WIDGET_DATA["emotion_generation"]
    expected = {
        "guide_size": 1536,
        "guide_size_for": True,
        "max_size": 1536,
        "feather": 5,
        "noise_mask": True,
        "force_inpaint": True,
        "bbox_threshold": 0.5,
        "bbox_dilation": 10,
        "bbox_crop_factor": 3.0,
        "sam_detection_hint": "center-1",
        "sam_dilation": 0,
        "sam_threshold": 0.93,
        "sam_bbox_expansion": 0,
        "sam_mask_hint_threshold": 0.7,
        "sam_mask_hint_use_negative": "False",
        "drop_size": 10,
        "cycle": 1,
        "inpaint_model": False,
    }
    assert {key: defaults[key] for key in expected} == expected
    assert "face_denoise" not in defaults
    assert "steps" not in defaults
    assert "cfg" not in defaults

    workflow_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "workflows",
        "VNCCS_3.0_Step3_CharacterEmotions.json",
    )
    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    node = next(item for item in workflow["nodes"] if item["type"] == "VNCCS_EmotionsGenerator")
    workflow_settings = json.loads(node["widgets_values"][0])["emotion_generation"]
    assert {key: workflow_settings[key] for key in expected} == expected
    assert "face_denoise" not in workflow_settings
    assert "steps" not in workflow_settings
    assert "cfg" not in workflow_settings


def test_regenerate_seed_shift_restores_pipe_seed():
    class Pipe:
        seed_int = 42

    pipe = Pipe()
    restore = cg._temporarily_shift_pipe_seed(pipe, 17)

    assert pipe.seed_int == 59
    restore()
    assert pipe.seed_int == 42


def test_emotions_generator_bg_remove_uses_character_background_color(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    monkeypatch.setattr(cg, "_character_cache_dir_from_sheets_path", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(cg, "_rotate_preview_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(cg, "_save_run_inputs", lambda *args, **kwargs: None)

    node = cg.VNCCS_EmotionsGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_save_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_load_source_sprite_from_path", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(node, "_pad_alpha_sources_to_uniform_canvas", lambda data, items: (items, (4, 4)))
    monkeypatch.setattr(
        node,
        "_run_emotion_generation_one",
        lambda image, *args, **kwargs: (image, image, torch.ones((image.shape[0], 4, 4))),
    )

    def capture_bg_remove(images, source_items, detailer_masks, settings, background="Green", **kwargs):
        seen["background"] = background
        return images

    monkeypatch.setattr(node, "_run_emotion_bg_remove", capture_bg_remove)

    images = torch.rand(1, 4, 4, 3)
    emotion_data = json.dumps([{"emotion_prompt": "angry", "sprite_output_path": "", "background_color": "Green"}])
    widget_data = json.dumps({
        "character_name": "Alice",
        "bg_remove": {"preset": "balanced", "use_sam3_details_recovery": True},
    })

    node.process(images, object(), emotion_data, widget_data=widget_data, unique_id="test-node")

    assert seen["background"] == "Green"


def test_emotions_generator_single_bg_regenerate_slices_cached_raw_batch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    monkeypatch.setattr(cg, "_character_cache_dir_from_sheets_path", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(cg, "_save_run_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(cg, "_load_cached_tensor", lambda *args, **kwargs: torch.zeros(4, 4, 4, 3))

    node = cg.VNCCS_EmotionsGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_save_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(node, "_load_source_sprite_from_path", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(node, "_pad_alpha_sources_to_uniform_canvas", lambda data, items: (items, (4, 4)))
    monkeypatch.setattr(
        node,
        "_run_emotion_generation_one",
        lambda image, *args, **kwargs: (image, image, torch.ones((image.shape[0], 4, 4))),
    )

    cached_raw = torch.stack([
        torch.full((4, 4, 3), float(index) / 10.0)
        for index in range(4)
    ])

    def load_cached_stage(cache_dir, stage, unique_id=None, message=""):
        if stage == "emotion_0001":
            return cached_raw
        return None

    def capture_bg_remove(images, source_items, detailer_masks, settings, background="Green", **kwargs):
        seen["shape"] = tuple(images.shape)
        seen["value"] = float(images[0, 0, 0, 0].item())
        return images

    monkeypatch.setattr(node, "_load_cached_stage", load_cached_stage)
    monkeypatch.setattr(node, "_run_emotion_bg_remove", capture_bg_remove)

    images = torch.rand(4, 4, 4, 3)
    emotion_data = [
        json.dumps({"emotion_prompt": "angry", "sprite_output_path": "same", "background_color": "Green"})
        for _ in range(4)
    ]
    widget_data = json.dumps({
        "character_name": "Alice",
        "regenerate_from": "emotion_0001_bg_remove",
        "regenerate_index": 2,
        "bg_remove": {"preset": "balanced", "use_sam3_details_recovery": True},
    })

    node.process(images, object(), emotion_data, widget_data=widget_data, unique_id="test-node")

    assert seen["shape"] == (1, 4, 4, 3)
    assert seen["value"] == pytest.approx(0.2)


def test_emotion_detailer_input_rebuilds_clean_chroma_plate_from_source_alpha():
    torch = pytest.importorskip("torch")
    node = cg.VNCCS_EmotionsGenerator()
    image = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.6
    image[..., 2] = 0.8
    inverse_alpha = torch.tensor([[[1.0, 0.5, 0.0]]], dtype=torch.float32)

    prepared = node._prepare_emotion_detailer_input(image, inverse_alpha, "Green")

    assert prepared[0, 0, 0].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert prepared[0, 0, 1].tolist() == pytest.approx([0.1, 0.8, 0.4])
    assert prepared[0, 0, 2].tolist() == pytest.approx([0.2, 0.6, 0.8])


def test_emotion_rgba_merge_uses_premultiplied_color_at_soft_alpha_transition():
    torch = pytest.importorskip("torch")
    node = cg.VNCCS_EmotionsGenerator()
    original = torch.tensor([[[[1.0, 0.0, 0.0, 1.0]]]], dtype=torch.float32)
    keyed = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=torch.float32)

    merged = node._merge_emotion_rgba(
        original,
        keyed,
        torch.tensor([[[0.5]]], dtype=torch.float32),
    )

    assert merged[0, 0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.5])


def test_emotion_region_bg_remove_preserves_unchanged_rgba_and_keys_only_detailer_crop(monkeypatch):
    torch = pytest.importorskip("torch")
    node = cg.VNCCS_EmotionsGenerator()
    node.DETAILER_MATTE_EXPAND_RADIUS = 0
    node.DETAILER_MATTE_FEATHER_RADIUS = 0
    node.DETAILER_CHROMA_CONTEXT = 1

    source_rgb = torch.zeros((1, 20, 20, 3), dtype=torch.float32)
    source_rgb[..., 0] = 0.1
    source_rgb[..., 1] = 0.7
    source_rgb[..., 2] = 0.8
    source_rgb[:, 12:16, 12:16, :] = torch.tensor([0.9, 0.6, 0.4])
    source_alpha = torch.zeros((1, 20, 20), dtype=torch.float32)
    source_alpha[:, 12:16, 12:16] = 1.0
    source_mask = 1.0 - source_alpha

    raw = node._prepare_emotion_detailer_input(source_rgb, source_mask, "Green")
    raw[:, 3:7, 3:7, :] = torch.tensor([0.8, 0.2, 0.1])
    detailer_mask = torch.zeros((1, 20, 20), dtype=torch.float32)
    detailer_mask[:, 3:7, 3:7] = 1.0
    seen = {}

    def fake_bg_remove(images, settings, background="Green", **kwargs):
        seen["shape"] = tuple(images.shape)
        is_green = (
            (images[..., 0] < 0.05)
            & (images[..., 1] > 0.95)
            & (images[..., 2] < 0.05)
        )
        alpha = (~is_green).to(dtype=images.dtype)
        return torch.cat([images[..., :3], alpha.unsqueeze(-1)], dim=-1)

    monkeypatch.setattr(node, "_run_bg_remove", fake_bg_remove)

    result = node._run_emotion_bg_remove(
        raw,
        [(source_rgb, source_mask)],
        detailer_mask,
        {"preset": "balanced", "use_sam3_details_recovery": False},
        background="Green",
    )

    assert seen["shape"][1] < raw.shape[1]
    assert seen["shape"][2] < raw.shape[2]
    assert result[0, 13, 13].tolist() == pytest.approx([0.9, 0.6, 0.4, 1.0])
    assert result[0, 10, 10].tolist() == pytest.approx([0.1, 0.7, 0.8, 0.0])
    assert result[0, 4, 4, 3].item() == pytest.approx(1.0)
    assert result[0, 4, 4, :3].tolist() == pytest.approx([0.8, 0.2, 0.1])


def test_emotion_region_bg_remove_skips_chroma_when_detailer_changed_nothing(monkeypatch):
    torch = pytest.importorskip("torch")
    node = cg.VNCCS_EmotionsGenerator()
    source_rgb = torch.rand((1, 8, 8, 3), dtype=torch.float32)
    source_alpha = torch.zeros((1, 8, 8), dtype=torch.float32)
    source_alpha[:, 2:6, 2:6] = 1.0
    source_mask = 1.0 - source_alpha
    raw = node._prepare_emotion_detailer_input(source_rgb, source_mask, "Green")

    def fail_bg_remove(*args, **kwargs):
        raise AssertionError("chroma key must not run without a changed detailer region")

    monkeypatch.setattr(node, "_run_bg_remove", fail_bg_remove)

    result = node._run_emotion_bg_remove(
        raw,
        [(source_rgb, source_mask)],
        torch.zeros((1, 8, 8), dtype=torch.float32),
        {"preset": "balanced"},
        background="Green",
    )

    expected = torch.cat([source_rgb, source_alpha.unsqueeze(-1)], dim=-1)
    assert torch.equal(result, expected)


def test_emotion_region_bg_remove_skips_chroma_for_interior_face_change(monkeypatch):
    torch = pytest.importorskip("torch")
    node = cg.VNCCS_EmotionsGenerator()
    node.DETAILER_MATTE_EXPAND_RADIUS = 0
    node.DETAILER_MATTE_FEATHER_RADIUS = 0
    source_rgb = torch.full((1, 12, 12, 3), 0.4, dtype=torch.float32)
    source_alpha = torch.zeros((1, 12, 12), dtype=torch.float32)
    source_alpha[:, 1:11, 1:11] = 1.0
    source_mask = 1.0 - source_alpha
    raw = node._prepare_emotion_detailer_input(source_rgb, source_mask, "Green")
    raw[:, 5:7, 5:7, :] = torch.tensor([0.9, 0.2, 0.1])
    detailer_mask = torch.zeros((1, 12, 12), dtype=torch.float32)
    detailer_mask[:, 5:7, 5:7] = 1.0

    def fail_bg_remove(*args, **kwargs):
        raise AssertionError("interior-only face changes must reuse the source alpha")

    monkeypatch.setattr(node, "_run_bg_remove", fail_bg_remove)

    result = node._run_emotion_bg_remove(
        raw,
        [(source_rgb, source_mask)],
        detailer_mask,
        {"preset": "balanced"},
        background="Green",
    )

    assert result[0, 5, 5].tolist() == pytest.approx([0.9, 0.2, 0.1, 1.0])
    assert result[0, 0, 0].tolist() == pytest.approx([0.4, 0.4, 0.4, 0.0])


def test_list_to_batch_normalizes_mixed_image_sizes():
    torch = pytest.importorskip("torch")

    small = torch.rand(1, 12, 8, 3)
    large = torch.rand(1, 24, 16, 3)

    result = cg.VNCCS_CharacterGenerator()._list_to_batch([small, large])

    assert result.shape == (2, 12, 8, 3)


def test_emotion_detailer_prompt_orders_emotion_then_face_details():
    generator = cg.VNCCS_EmotionsGenerator()

    result = generator._detailer_positive_prompt(
        "The character is furious.\n\nEmotion Tags: angry, open_mouth",
        "1girl, blue eyes, long black hair, (wear glasses on face:1.0), (wear hood on head:1.0)",
    )

    assert result.index("The character is furious.") < result.index("Character face details:")
    assert "blue eyes" in result
    assert "long black hair" in result
    assert "(wear glasses on face:1.0)" in result
    assert "(wear hood on head:1.0)" in result
    assert "The character is furious." in result
    assert "Emotion Tags: angry, open_mouth" in result
    assert "masterpiece" not in result


def test_pose_generation_decode_preserves_encoder_aspect(monkeypatch):
    torch = pytest.importorskip("torch")

    decoded = torch.rand(1, 1584, 664, 3)

    class FakeMaskExtractor:
        def fill_alpha_with_color(self, image):
            return (image,)

    class TestGenerator(cg.VNCCS_CharacterGenerator):
        def _extract_pipe(self, pipe):
            return {
                "clip": object(),
                "vae": object(),
                "model": object(),
                "seed": 1,
                "steps": 1,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
            }

        def _run_list_mapped(self, class_name, list_kwargs, **kwargs):
            if class_name == "VNCCS_QWEN_Encoder":
                return ([object()], [object()], [{"samples": torch.rand(1, 4, 198, 83)}])
            if class_name == "KSampler":
                return ([{"samples": torch.rand(1, 4, 198, 83)}],)
            if class_name == "VAEDecodeTiled":
                return ([decoded],)
            raise AssertionError(f"Unexpected node call: {class_name}")

        def _apply_pose_lora_to_model(self, model, clip, pipe, lora_info):
            return model

        def _validate_conditioning_for_model(self, pipe_values, positive, negative, stage_label):
            return None

    monkeypatch.setattr(cg, "VNCCS_MaskExtractor", FakeMaskExtractor)

    result = TestGenerator()._run_pose_generation(
        torch.rand(1, 1536, 640, 3),
        torch.rand(1, 1536, 640, 3),
        object(),
        "prompt",
        {"target_size": 1024},
        background="Green",
    )

    assert result.shape == (1, 1584, 664, 3)


def test_seedvr_loader_cleans_vram_and_uses_settings(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class FakeModelManagement:
        def __init__(self):
            self.unloaded = 0
            self.emptied = 0

        def unload_all_models(self):
            self.unloaded += 1

        def soft_empty_cache(self):
            self.emptied += 1

    fake_mm = FakeModelManagement()
    monkeypatch.setattr(cg, "model_management", fake_mm)

    def fake_call(class_name, **kwargs):
        calls.append((class_name, kwargs))
        return (f"{class_name}_out",)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_call)

    settings = cg.VNCCS_CharacterGenerator()._settings("{}")["upscaler"]
    settings.update(
        {
            "model": "custom_dit.gguf",
            "vae": "custom_vae.safetensors",
            "offload_device": "cpu",
            "cache_dit": True,
            "cache_vae": False,
            "resolution": 4096,
            "max_resolution": 3840,
            "color_correction": "adain",
        }
    )

    generator = cg.VNCCS_CharacterGenerator()
    dit, vae = generator._run_upscaler_models(settings)
    generator._run_seedvr_upscale_one(torch.rand(1, 1584, 664, 3), dit, vae, settings, seed=42)

    assert fake_mm.unloaded == 1
    assert fake_mm.emptied == 1
    assert calls[0][0] == "SeedVR2LoadDiTModel"
    assert calls[0][1]["model"] == "custom_dit.gguf"
    assert calls[0][1]["cache_model"] is True
    assert calls[1][0] == "SeedVR2LoadVAEModel"
    assert calls[1][1]["model"] == "custom_vae.safetensors"
    assert calls[2][0] == "SeedVR2VideoUpscaler"
    assert calls[2][1]["resolution"] == 4096
    assert calls[2][1]["max_resolution"] == 3840
    assert calls[2][1]["color_correction"] == "adain"
    assert calls[2][1]["offload_device"] == "cpu"


def test_seedvr_dit_cache_setting_is_not_force_overridden(monkeypatch):
    calls = []

    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_clean_vram_for_seedvr", lambda self: None)
    monkeypatch.setattr(
        cg,
        "_call_comfy_node",
        lambda class_name, **kwargs: calls.append((class_name, kwargs)) or (object(),),
    )
    settings = cg.VNCCS_CharacterGenerator()._settings(
        json.dumps({"upscaler": {"cache_dit": False}})
    )["upscaler"]

    cg.VNCCS_CharacterGenerator()._run_upscaler_models(settings)

    assert calls[0][0] == "SeedVR2LoadDiTModel"
    assert calls[0][1]["cache_model"] is False


def test_upscaler_can_use_local_seed_instead_of_pipe_seed():
    generator = cg.VNCCS_CharacterGenerator()

    assert generator._upscaler_seed(
        {"inherit_pipe_seed": False, "seed": 1234},
        pipe_seed=99,
    ) == 1234
    assert generator._upscaler_seed(
        {"inherit_pipe_seed": True, "seed": 1234},
        pipe_seed=99,
    ) == 99


def test_seedvr_upscaler_runs_whole_batch_once(monkeypatch):
    torch = pytest.importorskip("torch")
    generator = cg.VNCCS_CharacterGenerator()
    calls = []

    monkeypatch.setattr(generator, "_run_upscaler_models", lambda settings: ("dit", "vae"))

    def fake_seedvr(image, dit, vae, settings, seed):
        calls.append(image)
        assert image.shape == (4, 1584, 664, 3)
        return image

    monkeypatch.setattr(generator, "_run_seedvr_upscale_one", fake_seedvr)

    images = torch.rand(4, 1584, 664, 3)
    result = generator._run_upscaler(
        images,
        "Green",
        generator._settings("{}")["upscaler"],
        seed=42,
        use_internal_rmbg=False,
    )

    assert len(calls) == 1
    assert result.shape == images.shape


def test_seedvr_attention_auto_detects_until_manual(monkeypatch):
    generator = cg.VNCCS_CharacterGenerator()
    monkeypatch.setattr(cg, "_detect_seedvr_attention_mode", lambda: "flash_attn_3")

    assert generator._resolve_seedvr_attention_mode({"attention_mode": "sdpa"}) == "flash_attn_3"
    assert generator._resolve_seedvr_attention_mode({"attention_mode": "sdpa", "attention_mode_manual": True}) == "sdpa"
    assert generator._resolve_seedvr_attention_mode({"attention_mode": "flash_attn_2", "attention_mode_manual": True}) == "flash_attn_2"
