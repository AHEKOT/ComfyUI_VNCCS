"""Exercise LoRA routing and stage cleanup without models or GPU sampling."""

from types import SimpleNamespace
import weakref

import pytest

from nodes import vnccs_control_center as cc


@pytest.fixture
def generator(monkeypatch):
    pytest.importorskip("torch")
    from nodes import character_generator as cg

    monkeypatch.setattr(cg, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cg, "VNCCS_MaskExtractor", lambda: SimpleNamespace(fill_alpha_with_color=lambda image: (image,)))
    node = cg.VNCCS_CharacterGenerator()
    node._qwen_settings = lambda *args: {"target_size": 1024}
    node._sampler_settings = lambda *args: {"seed": 1, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}
    node._vae_decode_settings = lambda *args: {}
    node._stage_progress_callback = lambda *args: lambda *counts: None
    node._is_qi2_pipe = lambda values: values["model_kind"] == "qi2"
    node._is_h3_pipe = lambda values: values["model_kind"] == "minimaxh3"
    node._encoder_call = lambda *args, **kwargs: ("positive", "negative", "latent")
    node._resolution_scale_dimensions = lambda *args: (64, 64)
    node._validate_conditioning_for_model = lambda *args: None
    node._h3_first_frame_to_cpu = lambda image: image
    return node, cg


def entries():
    return [{
        "name": f"Wardrobe {kind}", "kind": kind, "type": "Custom", "custom": True,
        "local_path": f"models/loras/{kind}/{kind}_ClothesCore.safetensors",
    } for kind in ("QI2", "Klein9b", "MiniMaxH3")]


@pytest.mark.parametrize("kind", ["QI2", "Klein9b", "MiniMaxH3"])
def test_control_center_defers_clothes_weights_but_preserves_other_loras(monkeypatch, kind):
    loras = entries() + [{"name": "Style", "kind": kind, "custom": True, "local_path": "models/loras/style.safetensors"}]
    states = [{"name": entry["name"], "auto_apply": True} for entry in loras]
    loaded = []
    monkeypatch.setattr(cc, "_find_model_on_disk", lambda path: (path, True))
    monkeypatch.setattr(cc, "_apply_lora_standard", lambda model, clip, path, strength: loaded.append(path) or (model, clip))
    model, clip = object(), object()
    assert cc._apply_loras(model, clip, states, {"lora": loras}, "standard", model_entry={"kind": kind}) == (model, clip)
    assert loaded == ["models/loras/style.safetensors"]


@pytest.mark.parametrize("kind", ["QI2", "Klein9b", "MiniMaxH3"])
@pytest.mark.parametrize("failure", [None, "prepare", "sample", "decode"])
def test_remove_clothes_uses_only_selected_lora_and_releases_it(generator, monkeypatch, kind, failure):
    node, cg = generator
    refs, released, loaded = [], [], []
    physical = {"patches": []}

    class Model:
        def __init__(self, patches):
            self.patches = patches

        def clone(self):
            clone = Model(self.patches.copy())
            clone.parent = self
            refs.append(weakref.ref(clone))
            return clone

        def detach(self):
            physical["patches"] = []
            released.append("detach")

        def cleanup(self):
            released.append("cleanup")

    base = Model(["style"])
    loras = entries()
    pipe = SimpleNamespace(model=base, model_kind=kind.lower(), lora_entries=loras, lora_states=[
        {"name": entry["name"], "auto_apply": True, "strength": index / 10}
        for index, entry in enumerate(loras, 1)
    ])
    info = node._find_clothes_lora(pipe)
    selected = next(entry for entry in loras if entry["kind"] == kind)
    assert info["name"] == selected["name"]
    expected_strength = (loras.index(selected) + 1) / 10
    assert info["strength"] == expected_strength
    node._extract_pipe = lambda pipe: {"model": base, "clip": object(), "vae": object(), "audio_vae": object(), "model_kind": kind.lower()}

    def sample(model):
        assert model is not base
        assert model.patches == ["style", info["rel_path"]]
        physical["patches"] = model.patches.copy()
        if failure == "sample":
            raise RuntimeError("sample failed")
        return "samples"

    def decode():
        if failure == "decode":
            raise RuntimeError("decode failed")
        return "image"

    def call(name, **kwargs):
        if name == "LoraLoaderModelOnly":
            assert kwargs["model"] is base
            loaded.append((kwargs["lora_name"], kwargs["strength_model"]))
            model = base.clone()
            model.patches.append(kwargs["lora_name"])
            return (model,)
        if name == "MiniMaxH3ReferenceToVideo":
            return "positive", "latent"
        if name == "BasicGuider":
            return (SimpleNamespace(model=kwargs["model"]),)
        if name in {"KSampler", "SamplerCustomAdvanced"}:
            return (sample(kwargs.get("model") or kwargs["guider"].model),)
        if name in {"VAEDecode", "VAEDecodeTiled"}:
            return (decode(),)
        return (object(),)

    def prepare(model, *args):
        if failure == "prepare":
            raise RuntimeError("prepare failed")
        return model.clone(), False

    monkeypatch.setattr(cg, "_call_comfy_node", call)
    node._qi2_prepare_model = prepare
    node._qi2_sample = lambda model, *args, **kwargs: sample(model)
    node._qi2_decode = lambda *args: decode()
    if failure in {"sample", "decode"} or failure == "prepare" and kind == "QI2":
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            node._run_remove_clothes("character", pipe, {}, lora_info=info)
    else:
        assert node._run_remove_clothes("character", pipe, {}, lora_info=info) == "image"
    assert loaded == [(info["rel_path"], expected_strength)]
    assert released == ["detach", "cleanup"]
    assert base.patches == ["style"]
    assert physical["patches"] == []
    assert all(ref() is None for ref in refs)


def test_qi2_remove_clothes_cannot_silently_skip_missing_lora(generator, monkeypatch):
    node, cg = generator
    node._extract_pipe = lambda pipe: {"model": object(), "clip": object(), "model_kind": "qi2"}
    monkeypatch.setattr(cg, "_call_comfy_node", lambda *args, **kwargs: pytest.fail("No model may be sampled without the required clothes LoRA"))
    with pytest.raises(RuntimeError, match="Remove Clothes requires LoRA"):
        node._run_remove_clothes("character", object(), {}, lora_info={"exists": False})


def test_direct_lora_loader_fallback_keeps_input_clip_unpatched(generator, monkeypatch):
    node, cg = generator
    model, clip, patched = object(), object(), object()
    calls = []

    def unavailable(*args, **kwargs):
        raise RuntimeError("Native LoRA loader unavailable")

    monkeypatch.setattr(cg, "_call_comfy_node", unavailable)
    monkeypatch.setattr(cg, "_apply_lora_standard", lambda *args: calls.append(args) or (patched, None))
    assert node._apply_lora_to_model(model, clip, object(), {"exists": True, "path": "clothes.safetensors", "strength": 0.7}, "Remove Clothes") is patched
    assert calls == [(model, None, "clothes.safetensors", 0.7)]
