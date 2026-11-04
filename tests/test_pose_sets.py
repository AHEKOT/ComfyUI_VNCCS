import base64
import io
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import folder_paths
import pytest
from PIL import Image

from conftest import _preload_node

library = _preload_node("pose_sets")


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "ROOT", tmp_path / "PoseSetLibrary")


def record(count=2):
    image = Image.new("RGBA", (48, 96), "pink")
    return {"name": "My poses", "description": "Ordered poses", "asset": {"type": "pose_set",
        "poses": [{"bones": {"head": [0, i * 10, 0]}, "prompt": f"pose {i}", "cameraParams": {"yaw_deg": i}} for i in range(count)]},
        "previews": [library.thumbnail(image)] * count}


def test_save_reload_export_delete_preserves_count_order_and_previews():
    data = record()
    saved = library.save_set(data)
    loaded = library.get_set("local", saved["id"])
    assert loaded["asset"] == data["asset"]
    assert loaded["description"] == data["description"]
    assert len(loaded["previews"]) == 2
    assert library.catalog()["sets"][0]["count"] == 2
    assert library.local_sets() == [loaded]
    assert library.library_path("local", saved["id"]).parent.parent == library.ROOT
    library.delete_set(saved["id"])
    assert library.catalog()["sets"] == []


def test_concurrent_saves_and_failed_write_preserve_records(monkeypatch):
    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = list(pool.map(lambda _: library.save_set(record()), range(8)))
    assert len({item["id"] for item in saved}) == len(library.local_sets()) == 8
    path = library.library_path("local", saved[0]["id"])
    previous = path.read_bytes()
    monkeypatch.setattr(library, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="preserved"):
        library.write_json(path, saved[0])
    assert path.read_bytes() == previous


@pytest.mark.parametrize("change", [
    {"name": ""}, {"asset": {"type": "pose_set", "poses": []}}, {"previews": ["bad"]},
    {"asset": {"type": "pose_set", "poses": [{"x": float("nan")}]}, "previews": ["bad"]},
])
def test_invalid_records_cannot_create_files(change):
    with pytest.raises((ValueError, TypeError)):
        library.save_set({**record(), **change})
    assert not library.ROOT.exists()


def test_identifier_and_symlink_escape_are_rejected(tmp_path):
    for value in ["../outside", "/tmp/file", "C:\\file", "", None]:
        with pytest.raises(ValueError):
            library.library_path("local", value)
    library.ROOT.mkdir()
    (library.ROOT / "local").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        library.library_path("local", "a" * 32)


def test_character_previews_use_exact_indices_and_survive_sprite_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    root = Path(library.character_dir("Alice"))
    costume = root / "Sprites" / "Dress" / "Neutral"
    costume.mkdir(parents=True)
    # CharacterGenerator publishes sprite_pose_0001.png for the first pose.
    for index, color in [(2, "blue"), (1, "red")]:
        path = costume / f"sprite_pose_{index:04d}.png"
        Image.new("RGB", (48, 96), color).save(path)
    assert library.sprite_paths("Alice", "Dress") == {
        index: costume / f"sprite_pose_{index:04d}.png" for index in (1, 2)
    }
    data = {**record(), "preview_source": {"character": "Alice", "costume": "Dress"}}
    saved = library.save_set(data)
    for i, expected in enumerate([(255, 0, 0), (0, 0, 255)]):
        image = Image.open(io.BytesIO(base64.b64decode(saved["previews"][i]))).convert("RGB")
        assert all(abs(a - b) < 5 for a, b in zip(image.getpixel((0, 0)), expected))
    for path in costume.iterdir():
        path.unlink()
    assert library.get_set("local", saved["id"])["previews"] == saved["previews"]
    fallback = root / "Sprites" / "Naked" / "Neutral"
    fallback.mkdir(parents=True)
    Image.new("RGB", (48, 96), "green").save(fallback / "sprite_pose_0001.png")
    with pytest.raises(ValueError, match="missing sprite 1"):
        library.save_set(data)


@pytest.mark.parametrize("count", [1, 12])
def test_generated_sprite_numbering_matches_every_pose(tmp_path, monkeypatch, count):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    directory = Path(library.character_dir("Alice")) / "Sprites" / "Naked" / "Neutral"
    directory.mkdir(parents=True)
    colors = [(index * 17, 100, 180) for index in range(count)]
    for index, color in enumerate(colors, start=1):
        Image.new("RGB", (48, 96), color).save(directory / f"sprite_pose_{index:04d}.png")
    data = {**record(count), "preview_source": {"character": "Alice", "costume": "Naked"}}
    saved = library.save_set(data)
    loaded = library.get_set("local", saved["id"])
    assert loaded["asset"] == data["asset"]
    assert len(loaded["previews"]) == count
    for preview, expected in zip(loaded["previews"], colors):
        image = Image.open(io.BytesIO(base64.b64decode(preview))).convert("RGB")
        assert all(abs(a - b) < 5 for a, b in zip(image.getpixel((0, 0)), expected))
    if count > 1:
        (directory / "sprite_pose_0002.png").unlink()
        with pytest.raises(ValueError, match="missing sprite 2"):
            library.save_set(data)
        assert library.local_sets() == [loaded]


def test_public_repository_download_is_pinned_and_failed_refresh_preserves_cache(tmp_path, monkeypatch):
    manifest = tmp_path / "pose_sets.json"
    item = {**record(), "id": "a" * 32}
    manifest.write_text(json.dumps({"version": 1, "sets": [item]}))
    calls = []
    class Api:
        def __init__(self, token):
            assert token is False
        def model_info(self, name, files_metadata):
            assert name == "owner/poses" and files_metadata
            return types.SimpleNamespace(sha="f" * 40, siblings=[types.SimpleNamespace(rfilename="pose_sets.json", size=manifest.stat().st_size)])
    def download(**kwargs):
        calls.append(kwargs)
        return str(manifest)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api, hf_hub_download=download))
    library.repository_action({"action": "download", "repo_id": "owner/poses"})
    assert calls == [{"repo_id": "owner/poses", "filename": "pose_sets.json", "revision": "f" * 40, "token": False}]
    state = library.catalog()
    repository = state["repositories"][0]
    assert repository["count"] == 1
    assert library.get_set(repository["id"], item["id"])["asset"] == item["asset"]
    manifest.write_text('{"version":1,"sets":[{}]}')
    with pytest.raises(ValueError):
        library.repository_action({"action": "refresh", "repo_id": "owner/poses"})
    assert library.catalog() == state
    library.repository_action({"action": "toggle", "id": repository["id"], "enabled": False})
    assert library.catalog()["sets"] == []
    assert library.catalog()["repositories"][0]["count"] == 1
    library.repository_action({"action": "remove", "id": repository["id"]})
    assert library.catalog()["repositories"] == []
