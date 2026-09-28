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


def test_qi2_uses_scaled_pose_and_separate_empty_latent(monkeypatch):
    calls = []
    encoder_latent = object()
    sampler_latent = object()

    def fake_node(name, **kwargs):
        calls.append((name, kwargs))
        if name == "ImageScale":
            return (kwargs["image"],)
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

    expected_width, expected_height = generator._resolution_scale_dimensions(pose, 1024)
    assert (positive, negative, latent) == ("positive", "negative", sampler_latent)
    scales = [kwargs for name, kwargs in calls if name == "ImageScale"]
    assert (scales[0]["width"], scales[0]["height"]) == (expected_width, expected_height)
    encode = next(kwargs for name, kwargs in calls if name == "TextEncodeQwenImage21")
    assert encode["images"]["image_1"] is pose
    assert encode["images"]["image_2"] is character
    assert encode["resolution"] == 0
    assert encode["negative_prompt"] == ""
    empty = next(kwargs for name, kwargs in calls if name == "EmptyLatentImage")
    assert empty == {"width": expected_width, "height": expected_height, "batch_size": 1}


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
        if name in {"QwenImage21Cache", "ImageScale"}:
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

    assert [name for name, _ in calls[:3]] == ["QwenImage21Cache", "ImageScale", "TextEncodeQwenImage21"]
    cache = calls[0][1]
    assert (cache["device"], cache["dtype"]) == ("cpu", "int4")
    encoder = calls[2][1]
    assert encoder["prompt"] == (
        "Make character's face emotion warm happy smile\n"
        "keep character's clothes\n"
        "Transparent background with alpha channel."
    )
    assert encoder["resolution"] == 0
    assert encoder["images"]["image_1"] is image
