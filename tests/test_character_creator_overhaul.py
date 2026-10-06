"""QI2 Character Overhaul is owned by Creator V2, not the shared pipeline."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from conftest import _preload_node
from nodes import vnccs_control_center as control_center

pytest.importorskip("torch")
creator = _preload_node("character_creator_v2")


@pytest.mark.parametrize("rel_path", [
    "QI2.1/VNCCS/VNCCS_QI2_AnimeOverhaulV1.2.safetensors",
    "QI2.1\\VNCCS\\VNCCS_QI2_AnimeOverhaulV1_2.safetensors",
    "VNCCS_QI2_AnimeOverhaulV1.2.safetensors",
])
@pytest.mark.parametrize("legacy_installed", [False, True])
def test_installed_catalog_overhaul_is_loaded_instead_of_hardcoded_v1(monkeypatch, tmp_path, rel_path, legacy_installed):
    catalog_path = "models/loras/QI2.1/VNCCS/" + rel_path.replace("\\", "/").split("/")[-1]
    monkeypatch.setattr(control_center, "_get_cc_config", lambda repo: {"lora": [{
        "name": "VNCCS Overhaul QI2", "local_path": catalog_path, "version": "1.2",
    }]})
    monkeypatch.setattr(creator.folder_paths, "get_folder_paths", lambda category: [str(tmp_path)])
    installed = tmp_path / rel_path.replace("\\", "/")
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_bytes(b"installed adapter")
    # Even with V1 present, the catalog's current version must win.
    if legacy_installed:
        legacy = tmp_path / creator.QI2_OVERHAUL_LORA_NAME
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy adapter")
    calls = []
    def apply(model, clip, name, *strengths):
        calls.append((creator.get_lora_full_path(name), strengths))
        return "patched", clip
    assert creator.apply_creator_overhaul("model", "clip", {"generation_mode": "qi2"}, apply) == ("patched", "clip")
    assert calls == [(str(installed), (.5, 0.0))]


@pytest.mark.parametrize("value, expected", [(None, .5), ("", .5), ("bad", .5), (float("nan"), .5), (float("inf"), .5), (-1, 0), (2, 1), (0, 0), (.25, .25), (.5, .5), (.75, .75), (1, 1), (.37, .25), (.38, .5)])
def test_strength_is_finite_and_snaps_to_five_positions(value, expected):
    assert creator.normalize_overhaul_strength(value) == expected


def test_old_workflows_default_to_half_and_saved_zero_wins():
    assert creator.normalize_gen_settings({"generation_mode": "qi2"})["qi2_overhaul_strength"] == .5
    settings = creator.normalize_gen_settings({
        "generation_mode": "qi2", "qi2_overhaul_strength": 1,
        "mode_settings": {"qi2": {"qi2_overhaul_strength": 0}},
    })
    assert settings["qi2_overhaul_strength"] == 0


@pytest.mark.parametrize("mode, strength", [("illustrious", 1), ("anima", 1), ("qi2", 0)])
def test_other_families_and_zero_never_resolve_or_apply_overhaul(monkeypatch, mode, strength):
    def unexpected(*args):
        pytest.fail("Disabled Overhaul must not load anything")
    monkeypatch.setattr(creator, "get_lora_full_path", unexpected)
    assert creator.apply_creator_overhaul("model", "clip", {
        "generation_mode": mode, "qi2_overhaul_strength": strength,
    }, unexpected) == ("model", "clip")


@pytest.mark.parametrize("strength", [.25, .5, .75, 1])
def test_overhaul_only_patches_diffusion_weights(monkeypatch, strength):
    monkeypatch.setattr(creator, "get_lora_full_path", lambda name: "/test/overhaul.safetensors")
    calls = []
    def apply(*args):
        calls.append(args)
        return "patched-model", args[1]
    result = creator.apply_creator_overhaul("model", "clip", {
        "generation_mode": "qi2", "qi2_overhaul_strength": strength,
    }, apply)
    assert calls == [("model", "clip", "/test/overhaul.safetensors", strength, 0.0)]
    assert result == ("patched-model", "clip")


def test_missing_enabled_overhaul_has_an_actionable_error(monkeypatch):
    monkeypatch.setattr(creator, "get_lora_full_path", lambda name: None)
    with pytest.raises(ValueError, match="Download.*or set its strength to 0"):
        creator.apply_creator_overhaul("model", "clip", {"generation_mode": "qi2"}, None)


@pytest.mark.parametrize("mode", ["qi2", "anima", "illustrious"])
@pytest.mark.parametrize("version", ["1", "1.2", "1_2"])
def test_manual_slots_cannot_double_apply_or_leak_overhaul(mode, version):
    stack = [{"name": f"QI2.1\\VNCCS\\VNCCS_QI2_AnimeOverhaulV{version}.safetensors", "strength": .5},
             {"name": "other.safetensors", "strength": .75}]
    normalized = creator.normalize_gen_settings({"generation_mode": mode, "lora_stack": stack})
    assert normalized["lora_stack"] == [stack[1]]
    assert len(stack) == 2


@pytest.mark.parametrize("route", ["preview", "workflow"])
@pytest.mark.parametrize("turbo", [False, True])
def test_both_generation_paths_apply_overhaul_before_turbo_and_sampling(monkeypatch, tmp_path, route, turbo):
    calls = []
    monkeypatch.setattr(creator, "character_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(creator, "sheets_dir", lambda name: str(tmp_path / name / "Sheets"))
    monkeypatch.setattr(creator, "faces_dir", lambda name: str(tmp_path / name / "Faces"))
    monkeypatch.setattr(creator, "ensure_character_structure", lambda name: None)
    monkeypatch.setattr(creator, "load_config", lambda name, **kwargs: None)
    monkeypatch.setattr(creator, "save_config", lambda *args: str(tmp_path / "config.json"))
    monkeypatch.setattr(creator, "load_generation_assets", lambda settings: ("key", "model", "clip", "vae"))
    monkeypatch.setattr(creator, "acquire_preview_assets", lambda settings: ("model", "clip", "vae"))
    monkeypatch.setattr(creator, "get_lora_full_path", lambda name: "/test/overhaul.safetensors")
    monkeypatch.setattr(creator.comfy.utils, "load_torch_file", lambda *a, **kw: {"weights": True}, raising=False)
    def load_lora(model, clip, weights, strength_model, strength_clip):
        calls.append(("overhaul", strength_model, strength_clip))
        return "overhauled-model", clip
    monkeypatch.setattr(creator.comfy.sd, "load_lora_for_models", load_lora, raising=False)
    def prepare(model, settings):
        assert model == "overhauled-model"
        calls.append(("turbo", settings["turbo_enabled"]))
        return "prepared-model", settings["turbo_enabled"]
    monkeypatch.setattr(creator, "prepare_qi2_model", prepare)
    monkeypatch.setattr(creator, "encode_generation_conditioning", lambda *a, **kw: ("pos", "neg", "prompt"))
    monkeypatch.setattr(creator, "create_generation_latent", lambda *a: "latent")
    def sample(**kwargs):
        assert kwargs["model"] == "prepared-model"
        assert kwargs["qi2_turbo"] == turbo
        calls.append(("sample",))
        return "samples"
    monkeypatch.setattr(creator, "sample_generation_latent", sample)
    monkeypatch.setattr(creator, "decode_generation_samples", lambda *a: creator.torch.zeros((1, 8, 8, 3)))
    monkeypatch.setattr(creator.web, "json_response", lambda data: data, raising=False)
    monkeypatch.setattr(creator.server.PromptServer.instance, "send_sync", lambda *a: None, raising=False)
    monkeypatch.setitem(creator.PREVIEW_CACHE, "loras", {})
    data = {"character": "OverhaulTest", "character_info": {"sex": "female", "age": 25},
            "gen_settings": {"generation_mode": "qi2", "qi2_overhaul_strength": .75,
                             "turbo_enabled": turbo}, "preview_valid": False}
    if route == "preview":
        async def payload():
            return data
        result = asyncio.run(creator.preview_generate(SimpleNamespace(json=payload, headers={"Host": "localhost", "X-VNCCS-CSRF": "1"})))
        assert result["image"]
    else:
        result = creator.CharacterCreatorV2().process(json.dumps(data))
        assert len(result) == 3
    assert calls == [("overhaul", .75, 0.0), ("turbo", turbo), ("sample",)]
