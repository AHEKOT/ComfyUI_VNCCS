"""Costume deletion must preserve other character data and reject unsafe paths."""

from pathlib import Path

import pytest

import utils


@pytest.fixture
def storage(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    config = {
        "character_info": {"name": "Alice", "hair": "red"},
        "costumes": {"Dress": {"top": "silk"}, "Casual": {"top": "shirt"}},
        "extra": {"keep": True},
    }
    utils.save_config("Alice", config)
    root = Path(utils.character_dir("Alice"))
    for group in utils.MAIN_DIRS:
        for costume in ["Dress", "Casual", "Naked", "Original"]:
            image = root / group / costume / "neutral" / "V1" / "sprite.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(costume.encode())
    cache = root / "cache"
    cache.mkdir()
    for name in ["preview_Dress.png", "preview_info_Dress.json", "preview_Casual.png", "preview.png"]:
        (cache / name).write_text(name)
    return root, config


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_deletes_costume_metadata_images_versions_and_its_preview(storage):
    root, config = storage
    before = snapshot(root)
    utils.delete_costume("Alice", "Dress")
    del config["costumes"]["Dress"]
    assert utils.load_config("Alice", strict=True) == config
    assert set(utils.list_costumes("Alice")) == {"Naked", "Casual", "Original"}
    for path, content in before.items():
        if "/Dress/" in path or path in {"cache/preview_Dress.png", "cache/preview_info_Dress.json", "Alice_config.json"}:
            continue
        assert (root / path).read_bytes() == content
    assert all(not (root / group / "Dress").exists() for group in utils.MAIN_DIRS)
    assert not (root / "cache" / "preview_Dress.png").exists()
    assert not (root / "cache" / "preview_info_Dress.json").exists()
    assert not list(root.glob(".vnccs-delete-*"))


@pytest.mark.parametrize("name", ["Naked", "Original", "naked", "ORIGINAL", "../Dress", "Dress/sub", "Dress\\sub", ""])
def test_protected_or_unsafe_costumes_leave_storage_unchanged(storage, name):
    root, _ = storage
    before = snapshot(root)
    with pytest.raises(ValueError):
        utils.delete_costume("Alice", name)
    assert snapshot(root) == before


def test_rejects_missing_costume_and_invalid_character_without_creating_files(storage):
    root, _ = storage
    before = snapshot(root)
    with pytest.raises(FileNotFoundError):
        utils.delete_costume("Alice", "Missing")
    with pytest.raises(ValueError):
        utils.delete_costume("../Alice", "Dress")
    assert snapshot(root) == before


@pytest.mark.parametrize("failure", ["metadata", "rename"])
def test_failed_delete_restores_config_and_every_asset(storage, monkeypatch, failure):
    root, _ = storage
    before = snapshot(root)
    if failure == "metadata":
        monkeypatch.setattr(utils, "save_config", lambda *args: "")
    else:
        replace = utils.os.replace

        def denied(source, target):
            if Path(source) == root / "Faces" / "Dress":
                raise PermissionError("Delete denied")
            return replace(source, target)

        monkeypatch.setattr(utils.os, "replace", denied)
    with pytest.raises(OSError):
        utils.delete_costume("Alice", "Dress")
    assert snapshot(root) == before
    assert not list(root.glob(".vnccs-delete-*"))


def test_corrupt_config_is_not_replaced_or_followed_by_file_deletion(storage):
    root, _ = storage
    (root / "Alice_config.json").write_text("broken JSON")
    before = snapshot(root)
    with pytest.raises(OSError, match="Cannot read configuration"):
        utils.delete_costume("Alice", "Dress")
    assert snapshot(root) == before


def test_cleanup_failure_reports_committed_delete_and_recovery_directory(storage, monkeypatch):
    root, _ = storage

    def denied(path):
        raise PermissionError("Cleanup denied")

    monkeypatch.setattr(utils.shutil, "rmtree", denied)
    warning = utils.delete_costume("Alice", "Dress")
    assert "was deleted" in warning
    assert "Cleanup denied" in warning
    assert "Dress" not in utils.load_config("Alice", strict=True)["costumes"]
    assert all(not (root / group / "Dress").exists() for group in utils.MAIN_DIRS)
    staging, = root.glob(".vnccs-delete-*")
    assert str(staging) in warning
    assert list(staging.iterdir())


@pytest.mark.parametrize("location", ["costume", "group", "cache", "config", "character"])
def test_symlink_alias_cannot_delete_another_set(storage, location):
    root, _ = storage
    if location == "costume":
        alias = root / "Sprites" / "Alias"
        alias.symlink_to(root / "Sprites" / "Naked", target_is_directory=True)
        name = "Alias"
    elif location == "group":
        (root / "Sprites").rename(root / "real_sprites")
        (root / "Sprites").symlink_to(root / "real_sprites", target_is_directory=True)
        name = "Dress"
    elif location == "cache":
        (root / "cache").rename(root / "real_cache")
        (root / "cache").symlink_to(root / "real_cache", target_is_directory=True)
        name = "Dress"
    elif location == "config":
        (root / "Alice_config.json").rename(root / "real_config.json")
        (root / "Alice_config.json").symlink_to(root / "real_config.json")
        name = "Dress"
    else:
        root.rename(root.parent / "real_character")
        root.symlink_to(root.parent / "real_character", target_is_directory=True)
        name = "Dress"
    before = snapshot(root)
    with pytest.raises(ValueError, match="symbolic links"):
        utils.delete_costume("Alice", name)
    assert snapshot(root) == before


def test_disk_only_legacy_costume_is_removed_without_creating_config(storage):
    root, _ = storage
    (root / "Alice_config.json").unlink()
    utils.delete_costume("Alice", "Dress")
    assert not (root / "Alice_config.json").exists()
    assert all(not (root / group / "Dress").exists() for group in utils.MAIN_DIRS)


def test_shared_legacy_preview_is_preserved_for_remaining_costume(storage):
    root, _ = storage
    utils.save_costume_info("Alice", "Red Dress", {})
    utils.save_costume_info("Alice", "Red_Dress", {})
    for name in ["preview_Red_Dress.png", "preview_info_Red_Dress.json"]:
        (root / "cache" / name).write_text("shared")
    utils.delete_costume("Alice", "Red Dress")
    assert "Red Dress" not in utils.list_costumes("Alice")
    assert (root / "cache" / "preview_Red_Dress.png").read_text() == "shared"
    utils.delete_costume("Alice", "Red_Dress")
    assert not (root / "cache" / "preview_Red_Dress.png").exists()
