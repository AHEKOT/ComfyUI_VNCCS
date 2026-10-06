"""Release regression coverage for preparation, seed ownership, and preview caching."""

import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from _vnccs.nodes import character_generator as cg
from _vnccs.nodes import clothes_designer as cd


def test_h3_body_preparation_uses_clothes_lora_and_returns_one_frame(monkeypatch):
    node = cg.VNCCS_CharacterGenerator()
    image = torch.zeros(1, 96, 64, 3)
    decoded = torch.rand(5, 96, 64, 3)
    pipe = SimpleNamespace(
        model="model", clip="clip", vae="vae", audio_vae="audio",
        model_entry={"kind": "MiniMaxH3"}, seed_int=17, sample_steps=8,
        cfg=1.0, denoise=1.0, sampler_name="res_multistep", scheduler="simple",
    )
    calls = []
    applied = []
    events = []
    released = []
    clothes_model = SimpleNamespace(
        detach=lambda: released.append("detach"),
        cleanup=lambda: released.append("cleanup"),
    )
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(node, "_apply_lora_to_model", lambda model, clip, pipe, lora, stage:
                        applied.append((lora, stage)) or clothes_model)
    monkeypatch.setattr(cg, 'fill_alpha_with_color', lambda x: (x,))

    def call(name, **kwargs):
        calls.append((name, kwargs))
        if name == "MiniMaxH3ReferenceToVideo":
            return "conditioning", "latent"
        if name == "VAEDecode":
            return (decoded,)
        assert name != "KSampler"
        return (name,)

    monkeypatch.setattr(cg, "_call_comfy_node", call)
    lora = {"name": "VNCCS Clothes Core MiniMaxH3", "exists": True}
    prompt = "Keep the original character.\nUse the requested outfit."
    result = node._run_remove_clothes(image, pipe, {"target_size": 1536, "prompt": prompt}, lora)
    encoding = dict(calls)["MiniMaxH3ReferenceToVideo"]
    assert encoding["prompt"] == prompt
    assert encoding["audio_vae"] == "audio"
    assert encoding["ref_images"]["ref_image_1"] is image
    assert encoding["width"] * encoding["height"] == pytest.approx(1536 * 1024, rel=0.05)
    assert applied == [(lora, "Remove Clothes")]
    assert dict(calls)["BasicScheduler"]["model"] is clothes_model
    assert released == ["detach", "cleanup"]
    assert dict(calls)["RandomNoise"]["noise_seed"] == 17
    assert torch.equal(result, decoded[:1])
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()
    assert [kwargs["current"] for _, kwargs in events] == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("size", [1024, 1536, 4096])
def test_klein_preparation_passes_selected_resolution(monkeypatch, size):
    calls = []
    monkeypatch.setattr(cg, "_call_comfy_node", lambda name, **kwargs: calls.append(kwargs) or (None, None, None))
    cg.VNCCS_CharacterGenerator()._encoder_call(
        {"model_kind": "klein9b", "clip": None, "vae": None}, "prompt",
        image1=torch.zeros(1, 32, 32, 3), qwen_settings={"target_size": size},
    )
    assert calls[0]["megapixels"] == size / 1024


@pytest.fixture
def designer_run(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32)).save(source)
    monkeypatch.setattr(cd, "get_latest_sprite_path", lambda *args: str(source))
    monkeypatch.setattr(cd, "sheets_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(cd, "resolve_comfy_image_path", lambda *args: str(source))
    monkeypatch.setattr(cd.server.PromptServer.instance, "send_sync", lambda *args: None, raising=False)
    node = cd.ClothesDesigner()
    monkeypatch.setattr(node, "get_cache_paths", lambda *args: (str(tmp_path / "preview.png"), str(tmp_path / "preview.json")))
    monkeypatch.setattr(node, "get_reference_sprite", lambda *args: torch.zeros(1, 32, 32, 3))
    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_qi2_encode", lambda *args, **kwargs: (None, None, {}))
    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_qi2_prepare_model", lambda self, model, *args: (model, False))
    seeds = []
    def sample(self, model, positive, negative, latent, sampler, **kwargs):
        seeds.append(sampler["seed"])
        return {"samples": torch.zeros(1, 4, 4, 4)}
    monkeypatch.setattr(cg.VNCCS_CharacterGenerator, "_qi2_sample", sample)
    monkeypatch.setattr(cd, "_call_comfy_node", lambda *args, **kwargs: (torch.zeros(1, 32, 32, 3),))
    pipe = SimpleNamespace(
        model=object(), clip=object(), vae=object(), model_entry={"kind": "QI2"},
        lora_entries=[], seed_int=77, sample_steps=25, cfg=3,
        model_cache_key={"model": {"name": "QI2 A"}, "lora_states": []},
    )
    data = {"character": "Alice", "costume": "Coat", "activeTab": "clone",
            "clone_image": {"filename": "source.png"}, "gen_settings": {"seed": 123}}
    return node, pipe, data, seeds


@pytest.mark.parametrize("seed", [0, 123, 456])
def test_clothes_ui_seed_reaches_sampler(designer_run, seed):
    node, pipe, data, seeds = designer_run
    data["gen_settings"]["seed"] = seed
    node.process(pipe=pipe, widget_data=json.dumps(data))
    assert seeds == [seed]


def test_legacy_clothes_without_seed_inherits_pipe(designer_run):
    node, pipe, data, seeds = designer_run
    data["gen_settings"].pop("seed")
    node.process(pipe=pipe, widget_data=json.dumps(data))
    assert seeds == [77]


@pytest.mark.parametrize("change", ["model", "lora_states", "custom"])
def test_preview_cache_reuses_only_unchanged_model_assets(designer_run, change):
    node, pipe, data, seeds = designer_run
    def run():
        node.process(pipe=pipe, widget_data=json.dumps(data))
    run()
    run()
    assert len(seeds) == 1
    if change == "custom":
        pipe.model_cache_key = None
    else:
        pipe.model_cache_key = copy.deepcopy(pipe.model_cache_key)
        pipe.model_cache_key[change] = {"name": "changed", "strength": 0.5}
    run()
    assert len(seeds) == 2
    if change == "custom":
        run()
        assert len(seeds) == 3


def test_h3_clothes_catalog_entry_is_resolved():
    pipe = SimpleNamespace(model_entry={"kind": "minimaxh3"}, lora_entries=[{
        "name": "VNCCS Clothes Core MiniMaxH3", "kind": "MiniMaxH3",
        "local_path": "models/loras/MiniMax/ClothesCore.safetensors",
    }])
    assert cd._resolve_pipe_clothes_core_lora(pipe) == "MiniMax/ClothesCore.safetensors"
    assert cg.VNCCS_CharacterGenerator()._find_clothes_lora(pipe)["name"] == "VNCCS Clothes Core MiniMaxH3"
