"""Creator V2 owns character initialization, prompts, and safe persistence."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import utils
from creator_storage_helpers import load_creator_storage


@pytest.fixture
def creator(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    return load_creator_storage(monkeypatch)


def test_new_character_uses_creator_v2_prompt_and_storage(creator):
    result = creator.CharacterCreatorV2.create_character("Alice", "creator_v2")
    config = utils.load_config("Alice", strict=True)
    positive, negative = creator.CharacterCreatorV2.construct_prompt(config["character_info"])
    assert result["positive_prompt"] == positive
    assert result["negative_prompt"] == negative
    assert "cowboy_shot" in positive
    assert ":1.0" not in positive
    assert config["character_info"]["hair"] == "black hair, waist-length hair"
    assert config["costumes"] == {}
    assert config["character_path"] == utils.character_dir("Alice")
    assert result["age_lora_strength"] == utils.age_strength(18)
    assert "expressionless" in result["face_details"]
    assert Path(utils.character_dir("Alice")).is_dir()


def test_cloner_creation_keeps_its_existing_default_profile(creator):
    creator.CharacterCreatorV2.create_character("Cloner")
    info = utils.load_config("Cloner", strict=True)["character_info"]
    assert info["hair"] == "black long hair"
    assert info["eyes"] == "blue eyes"


def test_existing_character_is_returned_without_rewriting_authored_data(creator):
    creator.CharacterCreatorV2.create_character("Alice")
    path = Path(utils.config_path("Alice"))
    config = json.loads(path.read_text())
    config["character_info"]["hair"] = "My Custom Hair"
    config["costumes"] = {"Coat": {"prompt": "authored"}}
    path.write_text(json.dumps(config, indent=4))
    original = path.read_bytes()
    result = creator.CharacterCreatorV2.create_character("Alice", "creator_v2")
    assert result["existing"] is True and result["data"] == config
    assert path.read_bytes() == original


def test_concurrent_creation_publishes_one_character(creator):
    barrier = threading.Barrier(2)

    def create():
        barrier.wait()
        return creator.CharacterCreatorV2.create_character("Alice", "creator_v2")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))
    assert sum(result.get("existing", False) for result in results) == 1
    assert utils.load_config("Alice", strict=True)["character_info"]["hair"] == "black hair, waist-length hair"


def test_corrupt_character_is_preserved_and_creation_fails(creator):
    path = Path(utils.config_path("Alice"))
    path.parent.mkdir(parents=True)
    path.write_text("broken JSON")
    with pytest.raises(OSError, match="Cannot read configuration"):
        creator.CharacterCreatorV2.create_character("Alice")
    assert path.read_text() == "broken JSON"


def test_failed_save_is_not_reported_as_success(creator, monkeypatch):
    monkeypatch.setattr(creator, "save_config", lambda *args: "")
    with pytest.raises(OSError, match="Could not save character"):
        creator.CharacterCreatorV2.create_character("CannotSave")


def test_creation_rejects_unsafe_character_names(creator, tmp_path):
    with pytest.raises(ValueError):
        creator.CharacterCreatorV2.create_character("../outside")
    assert not list(tmp_path.iterdir())
