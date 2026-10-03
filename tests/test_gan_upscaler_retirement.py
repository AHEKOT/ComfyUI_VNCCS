"""GAN retirement preserves old workflows without loading replacement models."""

import asyncio
import json
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nodes import vnccs_control_center as cc


@pytest.fixture
def generator_module():
    pytest.importorskip("torch")
    from nodes import character_generator
    return character_generator


@pytest.mark.parametrize("generator_type", [
    "VNCCS_CharacterGenerator", "VNCCS_CharacterCloneGenerator",
    "VNCCS_ClothesGenerator", "VNCCS_EmotionsGenerator",
])
def test_saved_gan_settings_migrate_to_off(generator_module, generator_type):
    generator = getattr(generator_module, generator_type)()
    for previous, expected in [("gan", "off"), (" GAN ", "off"), ("seedvr", "seedvr"), ("off", "off")]:
        settings = generator._settings(json.dumps({"upscaler": {
            "mode": previous, "gan_model": "old.pth", "resolution": 3072,
        }}))["upscaler"]
        assert settings["mode"] == expected
        assert "gan_model" not in settings
        assert settings["resolution"] == 3072


def test_legacy_gan_execution_preserves_rgba_without_loading_models(monkeypatch, generator_module):
    torch = pytest.importorskip("torch")
    cg = generator_module
    generator = cg.VNCCS_CharacterGenerator()
    def forbidden(*args, **kwargs):
        pytest.fail("Retired GAN mode must not load GAN or SeedVR models")
    monkeypatch.setattr(generator, "_run_upscaler_models", forbidden)
    monkeypatch.setattr(cg, "_call_comfy_node", forbidden)
    monkeypatch.setattr(generator, "_emit", lambda *args, **kwargs: None)
    image = torch.rand(2, 16, 16, 4)
    legacy = {"mode": "gan", "gan_model": "old.pth"}
    assert torch.equal(generator._run_upscaler(image, "Green", legacy, 42), image)
    assert torch.equal(generator._run_source_upscaler(image, legacy, 42), image)


@pytest.fixture
def retired_catalog(tmp_path, monkeypatch):
    retired = {"name": "4x_APISR_GRL_GAN", "type": "Upscaler",
               "local_path": "models/upscale_models/old.pth", "hf_path": "models/upscale_models/old.pth"}
    current = {"name": "SeedVR", "local_path": "models/diffusion_models/seedvr.safetensors"}
    catalog = {"models": [current], "other": [retired]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setattr(cc, "hf_hub_download", lambda **kwargs: str(path))
    monkeypatch.setattr(cc, "_uses_packaged_cc_config", lambda repo: False)
    monkeypatch.setattr(cc, "_load_custom_loras", lambda: [])
    monkeypatch.setattr(cc, "_CC_CONFIG_CACHE", {})
    synced = []
    monkeypatch.setattr(cc, "_sync_packaged_cc_config", lambda repo, data: synced.append(data))
    return catalog, synced


def test_remote_and_cached_catalogs_cannot_restore_gan_models(retired_catalog):
    catalog, synced = retired_catalog
    loaded = cc._get_cc_config("test/catalog", prefer_remote=True)
    assert loaded["other"] == []
    assert loaded["models"] == catalog["models"]
    assert synced[0]["other"] == []
    cc._CC_CONFIG_CACHE["test/catalog"] = {"ts": time.time(), "data": catalog}
    assert cc._get_cc_config("test/catalog")["other"] == []


def test_control_center_hides_and_rejects_retired_downloads(retired_catalog, monkeypatch):
    monkeypatch.setattr(cc.web, "json_response", lambda data, status=200: SimpleNamespace(
        text=json.dumps(data), status=status,
    ), raising=False)
    monkeypatch.setattr(cc, "get_installed_version_info", lambda: {})
    monkeypatch.setattr(cc, "_enrich_config_entries", lambda entries, *args: entries)
    monkeypatch.setattr(cc, "validate_privileged_request", lambda request: None)
    downloads = queue.Queue()
    monkeypatch.setattr(cc, "_DOWNLOAD_QUEUE", downloads)
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"repo_id": "test/catalog"}))
    response = asyncio.run(cc.cc_check(request))
    assert response.status == 200
    assert json.loads(response.text)["other"] == []
    async def payload():
        return {"repo_id": "test/catalog", "category": "other", "name": "4x_APISR_GRL_GAN"}
    request.json = payload
    response = asyncio.run(cc.cc_download(request))
    assert response.status == 404
    assert downloads.empty()


def test_packaged_catalog_no_longer_contains_gan_upscalers():
    catalog = json.loads((Path(__file__).parents[1] / "control_center.json").read_text())
    assert cc._without_gan_upscalers(catalog) == catalog


def test_generator_defaults_no_longer_contain_gan_model(generator_module):
    assert "gan_model" not in generator_module.DEFAULT_WIDGET_DATA["upscaler"]
