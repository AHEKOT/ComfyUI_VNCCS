"""QI2 conditioning and Viggle sampler contracts."""

import pytest

torch = pytest.importorskip("torch")

from nodes import character_generator as cg
from nodes.qi2_viggle import (
    VIGGLE_TURBO_NODES,
    ViggleDetailerSchedule,
    _run_with_viggle_lora,
    viggle_turbo_sigmas,
)


def test_resolution_scale_preserves_fractional_legacy_presets():
    assert cg._resolution_scale_value(1344) == 1344
    assert cg._resolution_scale_value(1536) == 1536
    assert cg._resolution_scale_megapixels(1344) == pytest.approx(1.3125)
    assert cg._resolution_scale_megapixels(1536) == pytest.approx(1.5)
    assert cg._resolution_scale_value(768) == 1024


def test_qi2_system_encoder_scales_references_and_uses_separate_empty_latent(monkeypatch):
    calls = []
    encoder_latent = object()
    sampler_latent = object()
    scaled_pose = torch.zeros(1, 836, 1254, 3)

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "ImageScaleToTotalPixels":
            return (scaled_pose,)
        if name == "TextEncodeQwenImage21":
            return ("positive", "negative", encoder_latent)
        if name == "EmptyLatentImage":
            return (sampler_latent,)
        raise AssertionError(name)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)
    generator = cg.VNCCS_CharacterGenerator()
    pose = torch.zeros(1, 512, 768, 3)
    character = torch.zeros(1, 1024, 1024, 3)
    positive, negative, latent = generator._qi2_encode(
        {"clip": "clip", "vae": "vae"}, "pose prompt", (pose, character), target_size=1024,
    )

    assert (positive, negative, latent) == ("positive", "negative", sampler_latent)
    assert [name for name, _ in calls] == [
        "ImageScaleToTotalPixels", "TextEncodeQwenImage21", "EmptyLatentImage",
    ]
    scale = calls[0][1]
    assert scale == {
        "image": pose,
        "upscale_method": "lanczos",
        "megapixels": 1.0,
        "resolution_steps": 1,
    }
    encode = next(kwargs for name, kwargs in calls if name == "TextEncodeQwenImage21")
    assert encode["images"]["image_1"] is scaled_pose
    assert encode["images"]["image_2"] is character
    assert encode["resolution"] == 1024
    assert encode["negative_prompt"] == ""
    empty = next(kwargs for name, kwargs in calls if name == "EmptyLatentImage")
    assert empty == {"width": 1254, "height": 836, "batch_size": 1}


def test_qi2_2048_setting_means_two_megapixels_not_2048_squared(monkeypatch):
    calls = []
    scaled_pose = torch.zeros(1, 2243, 935, 1)

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "ImageScaleToTotalPixels":
            return (scaled_pose,)
        if name == "TextEncodeQwenImage21":
            return ("positive", "negative", "encoder latent")
        if name == "EmptyLatentImage":
            return ("empty latent",)
        raise AssertionError(name)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)
    pose = torch.zeros(1, 1536, 640, 3)
    character = torch.zeros(1, 1536, 640, 3)

    cg.VNCCS_CharacterGenerator()._qi2_encode(
        {"clip": "clip", "vae": "vae"}, "pose prompt", (pose, character), target_size=2048,
    )

    assert calls[0][0] == "ImageScaleToTotalPixels"
    assert calls[0][1]["megapixels"] == 2.0
    encoder = next(kwargs for name, kwargs in calls if name == "TextEncodeQwenImage21")
    assert encoder["resolution"] == 1024
    empty = next(kwargs for name, kwargs in calls if name == "EmptyLatentImage")
    assert empty == {"width": 935, "height": 2243, "batch_size": 1}


def test_viggle_turbo_uses_unmerged_lora_cache_and_latent_dependent_sigmas(monkeypatch):
    calls = []

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        return (name,)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)
    monkeypatch.setattr(cg, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cg, "apply_viggle_turbo_lora", lambda model, lora_name, strength: calls.append(
        ("apply_viggle_turbo_lora", {"model": model, "lora_name": lora_name, "strength": strength})
    ) or "ViggleTurboLora")
    monkeypatch.setattr(cg, "viggle_turbo_sigmas", lambda latent: calls.append(
        ("viggle_turbo_sigmas", {"latent": latent})
    ) or "ViggleTurboSigmas")
    generator = cg.VNCCS_CharacterGenerator()
    pipe = type("Pipe", (), {
        "lora_entries": [{
            "name": "Qwen Image 2.1 Viggle Turbo", "kind": "QI2", "type": "TurboLora",
            "local_path": "models/loras/QI2/Viggle/turbo.safetensors",
        }],
        "lora_states": [{"name": "Qwen Image 2.1 Viggle Turbo", "auto_apply": True}],
    })()
    model, turbo = generator._qi2_prepare_model(
        "base", pipe, {"qi2_cache": {"device": "cpu", "dtype": "int4"}},
    )
    latent = object()
    result = generator._qi2_sample(
        model, "positive", "negative", latent,
        {"seed": 42, "steps": 6, "cfg": 1, "denoise": 1, "sampler_name": "euler", "scheduler": "simple"},
        turbo=turbo,
    )

    assert result == "SamplerCustomAdvanced"
    assert [name for name, _ in calls] == [
        "apply_viggle_turbo_lora", "QwenImage21Cache", "RandomNoise", "BasicGuider",
        "KSamplerSelect", "viggle_turbo_sigmas", "SamplerCustomAdvanced",
    ]
    assert calls[0][1]["lora_name"] == "QI2/Viggle/turbo.safetensors"
    assert calls[1][1] == {"model": "ViggleTurboLora", "device": "cpu", "dtype": "int4"}
    assert calls[5][1] == {"latent": latent}
    assert calls[6][1]["latent_image"] is latent
    assert calls[6][1]["sigmas"] == "ViggleTurboSigmas"


def test_viggle_sigma_shift_depends_on_sampler_latent_resolution():
    small = viggle_turbo_sigmas({"samples": torch.zeros(1, 4, 64, 64)})
    large = viggle_turbo_sigmas({"samples": torch.zeros(1, 4, 128, 128)})
    assert len(small) == len(VIGGLE_TURBO_NODES) + 1
    assert small[0] == large[0] == 1
    assert small[-1] == large[-1] == 0
    assert torch.all(small[1:-1] < large[1:-1])


def test_viggle_lora_hook_is_unmerged_and_removed_after_execution():
    class DiffusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.block.weight.zero_()

    class Executor:
        def __init__(self, model):
            self.class_obj = model

        def __call__(self, tensor):
            return self.class_obj.block(tensor)

    model = DiffusionModel()
    executor = Executor(model)
    weights = {"block": [torch.eye(2), torch.eye(2)]}
    value = torch.tensor([[2.0, 3.0]])
    assert torch.equal(_run_with_viggle_lora(weights, executor, value), value)
    assert torch.equal(executor(value), torch.zeros_like(value))
    assert not model.block._forward_hooks


def test_qi2_base_sampler_uses_standard_ksampler(monkeypatch):
    calls = []
    monkeypatch.setattr(cg, "_call_comfy_node", lambda name, **kwargs: calls.append((name, kwargs)) or ("latent",))
    result = cg.VNCCS_CharacterGenerator()._qi2_sample(
        "model", "positive", "negative", "empty",
        {"seed": 2, "steps": 25, "cfg": 3, "denoise": 1, "sampler_name": "euler", "scheduler": "simple"},
    )
    assert result == "latent"
    assert [name for name, _ in calls] == ["KSampler"]
    assert calls[0][1]["latent_image"] == "empty"


def test_viggle_detailer_schedule_uses_encoded_crop_latent():
    hook = ViggleDetailerSchedule()
    latent = {"samples": torch.zeros(1, 4, 96, 80)}
    assert hook.post_encode(latent) is latent
    assert torch.equal(hook.scheduler_func(None, "euler", 6), viggle_turbo_sigmas(latent))
    image = torch.zeros(1, 64, 64, 3)
    assert hook.post_paste(image) is image
    assert hook.should_retry_patch(object()) is False


def test_qi2_emotion_crop_alignment_is_square_symmetric_and_reversible():
    generator = cg.VNCCS_EmotionsGenerator()
    crop = torch.rand(1, 119, 122, 3)
    aligned, padding = generator._pad_image_to_square_multiple(crop, multiple=32)
    left, top, right, bottom = padding
    assert aligned.shape == (1, 128, 128, 3)
    assert padding == (3, 4, 3, 5)
    restored = aligned[:, top:128 - bottom, left:128 - right, :]
    assert torch.equal(restored, crop)


@pytest.mark.parametrize("bbox_settings", [{}, {"bbox_dilation": 10, "feather": 5},
                                         {"bbox_dilation": 0, "feather": 0}])
def test_qi2_emotion_uses_bbox_crop_qwen_encoder_and_exact_paste_region(monkeypatch, bbox_settings):
    calls = []
    dilation = bbox_settings.get("bbox_dilation", 50)
    feather = bbox_settings.get("feather", 50)
    image = torch.zeros(1, 512, 384, 4)
    image[..., :3] = 0.2
    image[..., 3] = 0.35
    mask = 1.0 - image[..., 3]

    class Segment:
        crop_region = (100, 50, 200, 150)

    class Detector:
        def detect(self, _image, threshold, actual_dilation, crop_factor, drop_size):
            assert (threshold, actual_dilation, crop_factor, drop_size) == (0.5, dilation, 1.0, 10)
            return ((512, 384), [Segment()])

    detector = Detector()

    class Generator(cg.VNCCS_EmotionsGenerator):
        def _extract_pipe(self, _pipe):
            return {
                "model": "model", "clip": "clip", "vae": "vae", "seed": 1,
                "steps": 25, "cfg": 3.0, "denoise": 1.0,
                "sampler": "euler", "scheduler": "simple",
                "model_kind": "qi2", "qi2_cache": {"device": "cpu", "dtype": "int4"},
            }

        def _qi2_turbo_lora(self, _pipe):
            return ""

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "QwenImage21Cache":
            return (kwargs.get("model", kwargs.get("image")),)
        if name == "TextEncodeQwenImage21":
            return ("positive", "negative", "encoder latent")
        if name == "UltralyticsDetectorProvider":
            return (detector, object())
        if name == "ImageScaleToTotalPixels":
            return (torch.zeros(1, 256, 256, 4),)
        if name == "EmptyLatentImage":
            return ({"samples": torch.zeros(1, 4, 16, 16)},)
        if name == "KSampler":
            return (kwargs["latent_image"],)
        if name == "VAEDecode":
            decoded = torch.ones(1, 256, 256, 4)
            decoded[..., 3] = 0.75
            return (decoded,)
        raise AssertionError(name)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)
    full_image, generated_crop, detailer_mask = Generator()._run_emotion_generation_one(
        image, mask, object(), "warm happy smile", "ignored face tags", "", 42,
        detailer_settings={"face_denoise": 0.25, "target_size": 2048, **bbox_settings},
        bg_remove_settings={"preset": "Native"},
    )

    names = [name for name, _ in calls]
    assert names == [
        "UltralyticsDetectorProvider", "ImageScaleToTotalPixels",
        "TextEncodeQwenImage21", "EmptyLatentImage", "QwenImage21Cache", "KSampler", "VAEDecode",
    ]
    scale = next(kwargs for name, kwargs in calls if name == "ImageScaleToTotalPixels")
    assert scale["megapixels"] == 2.0
    cache = next(kwargs for name, kwargs in calls if name == "QwenImage21Cache")
    assert (cache["device"], cache["dtype"]) == ("cpu", "int4")
    encoder = next(kwargs for name, kwargs in calls if name == "TextEncodeQwenImage21")
    assert encoder["prompt"] == (
        "Upscale face image.\n"
        "Make character's face emotion warm happy smile\n"
        "Change only face. Keep original neck colour, clothes and hairs\n"
        "keep character's clothes\n"
        "Transparent background with alpha channel."
    )
    assert encoder["resolution"] == 1024
    aligned_side = ((100 + 2 * dilation + 31) // 32) * 32
    assert encoder["images"]["image_1"].shape == (1, aligned_side, aligned_side, 4)
    assert torch.allclose(
        encoder["images"]["image_1"][..., 3],
        torch.full((1, aligned_side, aligned_side), 0.35),
    )
    assert scale["image"] is encoder["images"]["image_1"]
    assert scale["resolution_steps"] == 32
    empty = next(kwargs for name, kwargs in calls if name == "EmptyLatentImage")
    assert empty["width"] % 32 == 0
    assert empty["height"] % 32 == 0
    sample = next(kwargs for name, kwargs in calls if name == "KSampler")
    assert sample["seed"] == 42
    assert sample["denoise"] == 1.0
    assert generated_crop.shape == (1, 256, 256, 4)
    assert full_image.shape == image.shape
    assert full_image[0, 100, 150, :].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.75])
    x1, y1, x2, y2 = (100 - dilation, 50 - dilation, 200 + dilation, 150 + dilation)
    blend = min(1.0, 2 / feather) if feather else 1.0
    assert full_image[0, y1 + 2, 150, :].tolist() == pytest.approx(
        [0.2 + 0.8 * blend] * 3 + [0.35 + 0.4 * blend],
    )
    assert torch.allclose(full_image[:, :, :x1, :3], torch.full((1, 512, x1, 3), 0.2))
    assert torch.allclose(full_image[:, :, :x1, 3], torch.full((1, 512, x1), 0.35))
    assert torch.all(detailer_mask[:, y1:y2, x1:x2] == 1)
    assert torch.all(detailer_mask[:, :, :x1] == 0)
    assert "FaceDetailer" not in names
    assert "VNCCS_BBox_Extractor" not in names


def test_qi2_emotion_crop_uses_physical_five_pixel_feather():
    generator = cg.VNCCS_EmotionsGenerator()
    reference = torch.zeros(1, 20, 20, 4)

    mask = generator._crop_feather_mask(20, 20, feather=5, reference=reference)

    assert mask.shape == (1, 20, 20)
    assert mask[0, 0, 10].item() == pytest.approx(0.0)
    assert mask[0, 1, 10].item() == pytest.approx(0.2)
    assert mask[0, 4, 10].item() == pytest.approx(0.8)
    assert mask[0, 5, 10].item() == pytest.approx(1.0)
    assert mask[0, 10, 10].item() == pytest.approx(1.0)


def test_qi2_emotion_prompt_template_keeps_dynamic_emotion_text():
    generator = cg.VNCCS_EmotionsGenerator()

    assert generator._qi2_emotion_prompt(
        "A natural smile.\n\nEmotion Tags: smile, happy",
        {"qi2_prompt_template": "Edit only the face:\n{emotion}\nKeep the hair."},
    ) == (
        "Edit only the face:\n"
        "A natural smile.\n\nEmotion Tags: smile, happy\n"
        "Keep the hair."
    )
    assert generator._qi2_emotion_prompt(
        "angry face",
        {"qi2_prompt_template": "Custom static instruction"},
    ) == "Custom static instruction\nangry face"


def test_emotion_preview_keeps_full_resolution_and_alpha():
    generator = cg.VNCCS_EmotionsGenerator()
    image = torch.rand(1, 1200, 800, 4)

    preview = generator._emotion_preview_tensor(image)

    assert preview.shape == image.shape
    assert preview.device.type == "cpu"
    assert torch.equal(preview, image.cpu())


def test_qi2_decode_uses_standard_vae_decode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cg,
        "_call_comfy_node",
        lambda name, **kwargs: calls.append((name, kwargs)) or ("image",),
    )

    result = cg.VNCCS_CharacterGenerator()._qi2_decode("samples", "vae")

    assert result == "image"
    assert calls == [("VAEDecode", {"samples": "samples", "vae": "vae"})]


def test_qi2_pose_pipeline_matches_reference_encoder_and_decode_nodes(monkeypatch):
    calls = []
    pose = torch.zeros(1, 640, 384, 3)
    character = torch.ones(1, 640, 384, 3)
    decoded = torch.full((1, 640, 384, 3), 0.5)

    class MaskExtractor:
        def fill_alpha_with_color(self, image):
            return (image,)

    class Generator(cg.VNCCS_CharacterGenerator):
        def _extract_pipe(self, _pipe):
            return {
                "model": "model", "clip": "clip", "vae": "vae", "seed": 7,
                "steps": 25, "cfg": 3.0, "denoise": 1.0,
                "sampler": "euler", "scheduler": "simple",
                "model_kind": "qi2", "model_entry": {"kind": "QI2"},
                "qi2_cache": {"device": "gpu", "dtype": "int8"},
            }

        def _apply_pose_lora_to_model(self, model, *_args):
            return model

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        outputs = {
            "ImageScaleToTotalPixels": (kwargs.get("image"),),
            "TextEncodeQwenImage21": ("positive", "negative", "encoder latent"),
            "EmptyLatentImage": ("empty latent",),
            "QwenImage21Cache": ("cached model",),
            "KSampler": ("sampled latent",),
            "VAEDecode": (decoded,),
        }
        return outputs[name]

    pipe = type("Pipe", (), {"lora_entries": [], "lora_states": []})()
    monkeypatch.setattr(cg, "VNCCS_MaskExtractor", MaskExtractor)
    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)

    result = Generator()._run_pose_generation(
        pose, character, pipe, "pose prompt", {"target_size": 1024},
    )

    assert torch.equal(result, decoded)
    assert [name for name, _ in calls] == [
        "QwenImage21Cache",
        "ImageScaleToTotalPixels",
        "TextEncodeQwenImage21",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
    ]
    encoder = calls[2][1]
    assert encoder["resolution"] == 1024
    assert torch.equal(encoder["images"]["image_1"], pose)
    assert torch.equal(encoder["images"]["image_2"], character)
    assert calls[1][1]["megapixels"] == 1.0
    assert calls[-1][1] == {"samples": "sampled latent", "vae": "vae"}
