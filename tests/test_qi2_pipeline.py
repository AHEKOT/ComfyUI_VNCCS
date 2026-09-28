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


def test_qi2_emotion_detailer_uses_native_prompt_cache_and_system_encoder(monkeypatch):
    calls = []
    image = torch.zeros(1, 512, 384, 3)
    mask = torch.zeros(1, 512, 384)

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
            return (object(), object())
        if name == "SAMLoader":
            return (object(),)
        if name == "FaceDetailer":
            return (image, image, None, mask)
        raise AssertionError(name)

    monkeypatch.setattr(cg, "_call_comfy_node", fake_node)
    Generator()._run_emotion_generation_one(
        image, mask, object(), "warm happy smile", "ignored face tags", "", 42,
        bg_remove_settings={"preset": "Native"},
    )

    assert [name for name, _ in calls[:2]] == ["QwenImage21Cache", "TextEncodeQwenImage21"]
    cache = calls[0][1]
    assert (cache["device"], cache["dtype"]) == ("cpu", "int4")
    encoder = calls[1][1]
    assert encoder["prompt"] == (
        "Make character's face emotion warm happy smile\n"
        "keep character's clothes\n"
        "Transparent background with alpha channel."
    )
    assert encoder["resolution"] == 1024
    assert encoder["images"]["image_1"] is image
    detailer = next(kwargs for name, kwargs in calls if name == "FaceDetailer")
    assert detailer["tiled_encode"] is False
    assert detailer["tiled_decode"] is False


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
        "ImageScaleToTotalPixels",
        "TextEncodeQwenImage21",
        "EmptyLatentImage",
        "QwenImage21Cache",
        "KSampler",
        "VAEDecode",
    ]
    encoder = calls[1][1]
    assert encoder["resolution"] == 1024
    assert torch.equal(encoder["images"]["image_1"], pose)
    assert torch.equal(encoder["images"]["image_2"], character)
    assert calls[0][1]["megapixels"] == 1.0
    assert calls[-1][1] == {"samples": "sampled latent", "vae": "vae"}
