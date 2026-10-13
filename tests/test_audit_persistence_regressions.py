"""Audit regressions at the real storage and HTTP boundaries, without models."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import utils
from conftest import _preload_node
from creator_storage_helpers import load_creator_storage
from test_creator_audit_validation import mock_creator_generation
from test_character_cloner import _load_cloner
from test_clothes_audit_paths import clothes_paths, costume_request


@pytest.mark.parametrize("revision", [None, '"obsolete-revision"'])
def test_creator_restored_workflow_saves_its_settings_without_a_profile_version_gate(monkeypatch, tmp_path, revision):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
             "gen_settings": {"seed": 1}}
    if revision is not None:
        state["character_revision"] = revision
    node = creator.CharacterCreatorV2()
    node.process(json.dumps(state))
    config = utils.load_config("Alice", strict=True)
    config["costumes"] = {"Coat": {"top": "cotton"}}
    config["character_info"]["retained_note"] = "authored metadata"
    utils.save_config("Alice", config)
    node.process(json.dumps({**state, "character_info": {"name": "Alice", "hair": "blue hair"}}))
    creator.CharacterCreatorV2().process(json.dumps(state))
    current = utils.load_config("Alice", strict=True)
    assert current["character_info"]["hair"] == "red hair"
    assert current["character_info"]["retained_note"] == "authored metadata"
    assert current["costumes"] == config["costumes"]
    assert calls == ["red hair", "blue hair", "red hair"]


@pytest.mark.parametrize("damage", ['{"character_info":', '{"character_info": []}'])
def test_creator_preserves_unreadable_profiles_before_loading_models(monkeypatch, tmp_path, damage):
    creator, calls = mock_creator_generation(monkeypatch, tmp_path)
    creator.CharacterCreatorV2.create_character("Alice")
    path = Path(utils.config_path("Alice"))
    path.write_text(damage)
    with pytest.raises(OSError, match="Cannot read configuration"):
        creator.CharacterCreatorV2().process(json.dumps({"character": "Alice",
            "character_info": {"name": "Alice", "hair": "red hair"}}))
    assert path.read_text() == damage
    assert calls == []


def test_creator_repeated_runs_save_new_seed_without_requiring_an_acknowledgement(monkeypatch, tmp_path):
    creator, _calls = mock_creator_generation(monkeypatch, tmp_path)
    state = {"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
             "gen_settings": {"seed": 1}, "character_revision": '"obsolete-revision"'}
    node = creator.CharacterCreatorV2()
    node.process(json.dumps(state), unique_id="818")
    node.process(json.dumps({**state, "gen_settings": {"seed": 2}}), unique_id="818")
    info = utils.load_config("Alice", strict=True)["character_info"]
    assert info["seed"] == 2
    assert info["hair"] == "red hair"


@pytest.mark.parametrize("revision", [None, '"obsolete-revision"'])
def test_cloner_restored_workflow_saves_its_authored_settings(monkeypatch, tmp_path, revision):
    cloner = _load_cloner(monkeypatch)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    monkeypatch.setattr(cloner.folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(cloner, "pil2tensor", lambda image: image.copy())
    Image.new("RGB", (2, 3), "red").save(tmp_path / "reference.png")
    state = {"character": "Alice", "character_info": {"name": "Alice", "hair": "red hair"},
             "source_images": ["reference.png"]}
    if revision is not None:
        state["character_revision"] = revision
    node = cloner.CharacterCloner()
    node.process(json.dumps(state))
    config = utils.load_config("Alice", strict=True)
    config["costumes"] = {"Coat": {"top": "cotton"}}
    config["character_info"]["retained_note"] = "authored metadata"
    utils.save_config("Alice", config)
    node.process(json.dumps({**state, "character_info": {"name": "Alice", "hair": "blue hair"}}))
    cloner.CharacterCloner().process(json.dumps(state))
    current = utils.load_config("Alice", strict=True)
    assert current["character_info"]["hair"] == "red hair"
    assert current["character_info"]["retained_note"] == "authored metadata"
    assert current["costumes"] == config["costumes"]


def test_recreated_costume_rejects_its_deleted_editors_revision(clothes_paths, monkeypatch):
    monkeypatch.setattr(clothes_paths.web, "json_response", lambda data, status=200, **kwargs:
                        SimpleNamespace(data=data, status=status, **kwargs), raising=False)
    info = {"top": "cotton"}
    utils.save_costume_info("Alice", "Dress", info)
    old_revision = utils.load_costume_snapshot("Alice", "Dress")[1]
    utils.delete_costume("Alice", "Dress")
    utils.save_costume_info("Alice", "Dress", info)
    assert utils.load_costume_snapshot("Alice", "Dress")[1] != old_revision
    original = Path(utils.config_path("Alice")).read_bytes()
    response = asyncio.run(clothes_paths.vnccs_save_costume(costume_request({
        "character": "Alice", "costume": "Dress", "info": {"top": "old editor"}, "revision": old_revision,
    })))
    assert response.status == 409
    assert Path(utils.config_path("Alice")).read_bytes() == original


def test_preview_read_waits_for_publication_without_blocking_http(clothes_paths, monkeypatch):
    root = utils.character_dir("Alice")
    sprites = Path(root) / "Sprites" / "Naked" / "Neutral"
    sprites.mkdir(parents=True)
    Image.new("RGB", (2, 3), "red").save(sprites / "pose.png")
    monkeypatch.setattr(clothes_paths.web, "Response", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    ready, release = threading.Event(), threading.Event()
    def publish():
        with utils.character_storage_lock(root):
            ready.set()
            release.wait(2)
    async def read():
        task = asyncio.create_task(clothes_paths.vnccs_get_preview(SimpleNamespace(
            rel_url=SimpleNamespace(query={"character": "Alice", "costume": "Naked"}))))
        await asyncio.sleep(.01)
        assert not task.done(), "The HTTP loop must run while the reader waits for publication"
        release.set()
        assert (await task).content_type == "image/png"
    worker = threading.Thread(target=publish)
    worker.start()
    try:
        assert ready.wait(2)
        asyncio.run(read())
    finally:
        release.set()
        worker.join(2)


def test_stale_style_edit_preserves_newer_prompt_and_disk_bytes(tmp_path):
    styles = _preload_node("character_styles")
    path = tmp_path / "styles.json"
    old = styles.save_user_style({"label": "Style", "prompt": "old prompt"}, path)
    newer = styles.save_user_style({**old, "prompt": "new prompt"}, path)
    original = path.read_bytes()
    with pytest.raises(utils.StorageConflictError):
        styles.save_user_style({**old, "label": "Renamed"}, path)
    assert path.read_bytes() == original
    current = styles.save_user_style({**newer, "label": "Renamed"}, path)
    assert current["prompt"] == "new prompt"
    assert "revision" not in json.loads(path.read_text())["styles"][0]


def test_style_edit_invalidates_creator_execution_cache(monkeypatch, tmp_path):
    creator, _calls = mock_creator_generation(monkeypatch, tmp_path)
    styles = _preload_node("character_styles")
    path = tmp_path / "styles.json"
    style = styles.save_user_style({"label": "Style", "prompt": "ink"}, path)
    monkeypatch.setattr(creator, "load_user_styles", lambda: styles.load_user_styles(path))
    creator.CharacterCreatorV2.create_character("Alice")
    state = json.dumps({"character": "Alice", "character_info": {"style": style["id"]}, "gen_settings": {"seed": 1}})
    before = creator.CharacterCreatorV2.IS_CHANGED(state)
    styles.save_user_style({**style, "prompt": "charcoal"}, path)
    assert creator.CharacterCreatorV2.IS_CHANGED(state) != before


@pytest.mark.parametrize("description", ["A young woman with black hair and blue eyes", "A man in black clothes", "A woman with pale hair and olive eyes"])
def test_wizard_does_not_infer_skin_from_hair_eyes_or_clothes(monkeypatch, description):
    creator = load_creator_storage(monkeypatch)
    result = creator.postprocess_character_wizard_result({"sex": "female", "race": "human", "skin_color": "pale skin"}, description)
    assert result["skin_color"] == ""


@pytest.mark.parametrize("strength", [0, 1])
def test_missing_selected_adapter_fails_but_disabled_adapter_is_optional(monkeypatch, tmp_path, strength):
    creator, _calls = mock_creator_generation(monkeypatch, tmp_path)
    monkeypatch.setattr(creator, "get_lora_full_path", lambda name: None)
    if strength:
        with pytest.raises(FileNotFoundError, match="Selected LoRA"):
            creator.apply_generation_lora("model", "clip", "missing.safetensors", strength)
    else:
        assert creator.apply_generation_lora("model", "clip", "missing.safetensors", strength) == ("model", "clip")


def test_catalog_checks_use_cache_and_explicit_refresh_downloads(monkeypatch, tmp_path):
    control = _preload_node("vnccs_control_center")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    downloads = []
    monkeypatch.setattr(control, "hf_hub_download", lambda **kwargs: downloads.append(kwargs) or str(catalog))
    monkeypatch.setattr(control, "_load_custom_loras", lambda: [])
    monkeypatch.setattr(control, "get_installed_version_info", lambda: {})
    monkeypatch.setattr(control, "_CC_CONFIG_CACHE", {})
    monkeypatch.setattr(control.web, "json_response", lambda data, **kwargs: SimpleNamespace(data=data), raising=False)
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"repo_id": "Test/Catalog"}))
    asyncio.run(control.cc_check(request))
    asyncio.run(control.cc_check(request))
    assert len(downloads) == 1 and downloads[0]["force_download"] is False
    request.rel_url.query["force_refresh"] = "true"
    asyncio.run(control.cc_check(request))
    assert len(downloads) == 2 and downloads[-1]["force_download"] is True
