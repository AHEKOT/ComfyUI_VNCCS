"""QI2 conditioning and Viggle sampler contracts."""

import pytest

torch = pytest.importorskip("torch")

from nodes import character_generator as cg
from nodes.qi2_viggle import (
    VIGGLE_TURBO_NODES,
    viggle_turbo_sigmas,
)


def test_resolution_scale_preserves_fractional_legacy_presets():
    assert cg._resolution_scale_value(1344) == 1344
    assert cg._resolution_scale_value(1536) == 1536
    assert cg._resolution_scale_megapixels(1344) == pytest.approx(1.3125)
    assert cg._resolution_scale_megapixels(1536) == pytest.approx(1.5)
    assert cg._resolution_scale_value(768) == 1024


@pytest.mark.parametrize("shape", [(97, 65), (65, 97), (3840, 1590)])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_qi2_encoder_only_flattens_alpha_and_builds_a_separate_latent(monkeypatch, shape, channels, batched):
    calls = []
    sampling_latent = object()
    source = torch.full((*shape, channels), 0.25)
    donor = torch.full((73, 151, channels), 0.5)
    if channels == 4:
        source[..., 3] = torch.linspace(0, 1, shape[1])
    if batched:
        source, donor = source.unsqueeze(0), donor.unsqueeze(0)
    originals = [image.clone() for image in (source, donor)]

    def call(name, **kwargs):
        calls.append((name, kwargs))
        if name == "ImageScaleToTotalPixels":
            return (torch.empty(1, 53, 71, 3),)
        if name == "TextEncodeQwenImage21":
            return "positive", "negative", object()
        assert name == "EmptyLatentImage"
        return (sampling_latent,)

    monkeypatch.setattr(cg, "_call_comfy_node", call)
    result = cg.VNCCS_CharacterGenerator()._qi2_encode(
        {"clip": "clip", "vae": "vae"}, "outfit prompt", (source, None, donor),
        target_size=2048, negative_prompt="negative",
    )
    assert result == ("positive", "negative", sampling_latent)
    assert [name for name, _ in calls] == ["ImageScaleToTotalPixels", "TextEncodeQwenImage21", "EmptyLatentImage"]
    sizing, encoder, empty = (kwargs for _, kwargs in calls)
    assert (sizing["upscale_method"], sizing["megapixels"], sizing["resolution_steps"]) == ("nearest-exact", 2.0, 1)
    expected_image = encoder["images"]["image_1"]
    torch.testing.assert_close(sizing["image"], expected_image if batched else expected_image.unsqueeze(0))
    assert empty == {"width": 71, "height": 53, "batch_size": 1}
    assert encoder["resolution"] == 1024
    assert encoder["negative_prompt"] == "negative"
    assert set(encoder["images"]) == {"image_1", "image_3"}
    for key, image, original in zip(("image_1", "image_3"), (source, donor), originals):
        reference = encoder["images"][key]
        assert reference.shape == (*image.shape[:-1], 3)
        if channels == 3:
            assert reference is image
        else:
            torch.testing.assert_close(reference, image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4]))
        assert torch.equal(image, original)


@pytest.mark.parametrize("target_size,megapixels", [
    (1024, 1.0), (1229, 1.2), (1280, 1.3), (1344, 1.3), (1536, 1.5), (2048, 2.0), (4096, 4.0),
])
@pytest.mark.parametrize("donor_shape", [(1, 16, 4096, 3), (1, 4096, 16, 3)])
def test_qi2_scale_only_controls_the_canvas(monkeypatch, target_size, megapixels, donor_shape):
    image = torch.tensor([[[[-0.1, 1.1, 0.5]]]], dtype=torch.float64).expand(1, 1024, 512, 3)
    donor = torch.zeros(donor_shape)
    latent = object()
    def call(name, **kwargs):
        if name == "ImageScaleToTotalPixels":
            assert kwargs["image"] is image
            assert kwargs["megapixels"] == megapixels
            assert kwargs["upscale_method"] == "nearest-exact" and kwargs["resolution_steps"] == 1
            return (torch.empty(1, 127, 63, 3),)
        if name == "EmptyLatentImage":
            assert kwargs == {"width": 63, "height": 127, "batch_size": 1}
            return (latent,)
        assert name == "TextEncodeQwenImage21"
        assert kwargs["resolution"] == 1024
        assert kwargs["images"]["image_1"] is image
        assert kwargs["images"]["image_2"] is donor
        return "positive", "negative", object()
    monkeypatch.setattr(cg, "_call_comfy_node", call)
    result = cg.VNCCS_CharacterGenerator()._encoder_call(
        {"model_kind": "qi2", "clip": "clip", "vae": "vae"}, "prompt", image1=image, image2=donor,
        qwen_settings={"target_size": target_size, "crop_method": "center", "upscale_method": "lanczos"},
    )
    assert result[2] is latent


@pytest.mark.parametrize("with_donor", [False, True])
def test_qi2_reference_generation_requires_image1(with_donor):
    images = (None, torch.zeros(1, 32, 64, 3)) if with_donor else ()
    with pytest.raises(ValueError, match="requires image_1"):
        cg.VNCCS_CharacterGenerator()._qi2_encode({"clip": "clip", "vae": "vae"}, "prompt", images)


def test_qi2_rejects_empty_reference_before_scaling():
    with pytest.raises(ValueError, match="non-empty image"):
        cg.VNCCS_CharacterGenerator()._qi2_encode({}, "prompt", (torch.empty(1, 0, 32, 3),))


def test_viggle_turbo_uses_native_lora_cache_and_latent_dependent_sigmas(monkeypatch):
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
    assert VIGGLE_TURBO_NODES == (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
    small = viggle_turbo_sigmas({"samples": torch.zeros(1, 4, 64, 64)})
    large = viggle_turbo_sigmas({"samples": torch.zeros(1, 4, 128, 128)})
    assert len(small) == len(VIGGLE_TURBO_NODES) + 1
    assert small[0] == large[0] == 1
    assert small[-1] == large[-1] == 0
    assert torch.all(small[1:-1] < large[1:-1])


def test_viggle_lora_uses_comfy_model_management(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cg,
        "_call_comfy_node",
        lambda name, **kwargs: calls.append((name, kwargs)) or ("managed-model",),
    )

    assert cg.apply_viggle_turbo_lora("base", "QI2/Viggle/turbo.safetensors", 0.75) == "managed-model"
    assert calls == [("LoraLoaderModelOnly", {
        "model": "base",
        "lora_name": "QI2/Viggle/turbo.safetensors",
        "strength_model": 0.75,
    })]


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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("publish_each", [False, True])
@pytest.mark.parametrize("align_image1", [False, True])
@pytest.mark.parametrize("pose_prompt,pose_settings,mp,encoder_resolution", [
    ("pose prompt", {"target_size": 1024}, 1.0, 1024),
    ("", {"target_size": 2048, "resolution": 1024}, 2.0, 1440),
    ("pose prompt", {"target_size": 3072, "resolution": 0}, 3.0, 1760),
    ("", {"target_size": 4096, "override_resolution": False, "resolution": 1536}, 4.0, 2048),
    ("pose prompt", {"target_size": 1280}, 1.3, 1152),
    ("", {"target_size": 2048, "override_resolution": True, "resolution": 0}, 2.0, 0),
    ("pose prompt", {"target_size": 2048, "override_resolution": True, "resolution": 1536}, 2.0, 1536),
])
def test_qi2_pose_pipeline_matches_reference_encoder_and_decode_nodes(monkeypatch, native, publish_each, align_image1, pose_prompt, pose_settings, mp, encoder_resolution):
    calls = []
    pose = torch.zeros(1, 640, 384, 3)
    character = torch.full((1, 736, 512, 4), 0.4)
    character[..., 3] = 0.25
    decoded = torch.full((1, 640, 384, 4), 0.5)

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
            "ImageScaleToTotalPixels": (pose,),
            "TextEncodeQwenImage21": ("positive", "negative", "encoder latent"),
            "EmptyLatentImage": ("sampling latent",),
            "QwenImage21Cache": ("cached model",),
            "KSampler": ("sampled latent",),
            "VAEDecode": (decoded,),
        }
        return outputs[name]

    pipe = type("Pipe", (), {"lora_entries": [], "lora_states": []})()
    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)

    result = Generator()._run_pose_generation(
        pose, character, pipe, pose_prompt, pose_settings,
        bg_remove_settings={"preset": "Native" if native else "balanced"},
        publish_each=publish_each,
        align_image1=align_image1,
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
    torch.testing.assert_close(dict(calls)["ImageScaleToTotalPixels"]["image"], pose)
    assert dict(calls)["ImageScaleToTotalPixels"]["megapixels"] == mp
    encoder = dict(calls)["TextEncodeQwenImage21"]
    assert encoder["resolution"] == encoder_resolution
    assert torch.equal(encoder["images"]["image_1"], pose)
    expected_character = character[..., :3] * character[..., 3:4] + (1.0 - character[..., 3:4])
    torch.testing.assert_close(encoder["images"]["image_2"], expected_character)
    assert dict(calls)["KSampler"]["latent_image"] == "sampling latent"
    assert dict(calls)["KSampler"]["seed"] == 7
    background_prompt = cg.NATIVE_BACKGROUND_PROMPT if native else "Change background to solid Green color"
    assert encoder["prompt"] == (f"{pose_prompt}, {background_prompt}" if pose_prompt else background_prompt)
    assert calls[-1][1] == {"samples": "sampled latent", "vae": "vae"}


def test_qi2_regeneration_preserves_individual_reference_sizes_and_pixels(monkeypatch, tmp_path):
    cache = str(tmp_path / "cache")
    monkeypatch.setattr(cg, "_safe_cache_dir", lambda path: path)
    monkeypatch.setattr(cg, "_safe_torch_load_tensor", lambda path: torch.load(path, weights_only=True))
    images = [torch.rand(1, 97, 65, 3), torch.rand(1, 72, 151, 4)]
    assert cg._save_run_inputs(cache, poses=images, prompt=["front", "back"])
    restored = cg._load_run_inputs(cache)
    assert restored["prompt"] == ["front", "back"]
    for index, image in enumerate(images):
        result = cg.VNCCS_CharacterGenerator()._slice_batch_item(restored["poses"], index)
        assert torch.equal(result, image)
