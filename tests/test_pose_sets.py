import base64
import asyncio
import importlib
import io
import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import folder_paths
import pytest
from PIL import Image

from conftest import _preload_node
from pose_repository_fixtures import git_repository

library = _preload_node("pose_sets")


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "ROOT", tmp_path / "PoseSetLibrary")
    monkeypatch.setattr(library, "BUNDLED", tmp_path / "defaults")
    library._cached_document.cache_clear()


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
    write = Path.write_bytes
    def failed_write(file, data):
        write(file, b"partial write")
        raise OSError("Disk unavailable")
    monkeypatch.setattr(Path, "write_bytes", failed_write)
    with pytest.raises(OSError, match="Disk unavailable"):
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


def test_public_repository_download_is_pinned_and_failed_refresh_preserves_cache(tmp_path, git_repository):
    manifest = tmp_path / "pose_sets.json"
    item = {**record(), "id": "a" * 32}
    manifest.write_text(json.dumps({"version": 1, "sets": [item]}))
    git_repository.files = {"pose_sets.json": manifest.read_bytes()}
    git_repository.revision = "f" * 40
    library.repository_action({"action": "download", "repo_id": "owner/poses"})
    assert len(git_repository.calls) == 1 and git_repository.calls[0]["repo_id"] == "owner/poses"
    state = library.catalog()
    repository = state["repositories"][0]
    assert repository["count"] == 1
    assert repository["revision"] == "f" * 40
    assert library.get_set(repository["id"], item["id"])["asset"] == item["asset"]
    manifest.write_text('{"version":1,"sets":[{}]}')
    git_repository.files["pose_sets.json"] = manifest.read_bytes()
    with pytest.raises(ValueError):
        library.repository_action({"action": "refresh", "repo_id": "owner/poses"})
    assert library.catalog() == state
    library.repository_action({"action": "toggle", "id": repository["id"], "enabled": False})
    assert library.catalog()["sets"] == []
    assert library.catalog()["repositories"][0]["count"] == 1
    library.repository_action({"action": "remove", "id": repository["id"]})
    assert library.catalog()["repositories"] == []


@pytest.mark.parametrize("damaged", [b"{broken", b"[]", b'{"asset": {"type": "pose_set", "poses": null}}'])
def test_corrupt_set_does_not_hide_healthy_sets_or_allow_partial_export(damaged):
    saved = library.save_set(record())
    broken = library.library_path("local", "b" * 32)
    broken.write_bytes(damaged)
    catalog = library.catalog()
    assert [item["id"] for item in catalog["sets"]] == [saved["id"]]
    assert catalog["errors"][0]["path"] == f"local/{broken.name}"
    assert catalog["errors"][0]["message"]
    assert library.get_set("local", saved["id"])["asset"] == saved["asset"]
    with pytest.raises((ValueError, TypeError)):
        library.local_sets()
    assert broken.read_bytes() == damaged


def test_catalog_reports_repository_errors_without_hiding_other_sets():
    saved = library.save_set(record())
    remote = {**record(), "id": "a" * 32}
    repo_id = "c" * 32
    repository = library.library_path("repositories", repo_id)
    library.write_json(repository, {"id": repo_id, "repo_id": "owner/poses", "revision": "f" * 40,
                                    "enabled": True, "sets": [None, {}, remote]})
    before = repository.read_bytes()
    catalog = library.catalog()
    assert {(item["source"], item["id"]) for item in catalog["sets"]} == {
        ("local", saved["id"]), (repo_id, remote["id"])}
    assert [error["path"] for error in catalog["errors"]] == [
        f"repositories/{repo_id}.json, set {index}" for index in (1, 2)]
    assert library.get_set(repo_id, remote["id"]) == remote
    assert repository.read_bytes() == before
    repository.write_bytes(b"{damaged repository")
    catalog = library.catalog()
    assert [item["id"] for item in catalog["sets"]] == [saved["id"]]
    assert catalog["errors"][0]["path"] == f"repositories/{repo_id}.json"
    assert repository.read_bytes() == b"{damaged repository"


@pytest.mark.parametrize("source", ["local", "c" * 32])
def test_preview_cache_reuses_reads_and_detects_replacement_and_deletion(monkeypatch, source):
    saved = library.save_set(record(12))
    path = library.library_path("local", saved["id"])
    if source != "local":
        path = library.library_path("repositories", source)
        library.write_json(path, {"sets": [saved]})
    read = library.read_json
    reads = []
    def counted(file):
        reads.append(file)
        return read(file)
    monkeypatch.setattr(library, "read_json", counted)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: library.get_set(source, saved["id"]), range(24)))
    assert reads == [path]
    assert all(result == saved for result in results)
    results[0]["asset"]["poses"][0]["prompt"] = "Local draft"
    assert library.get_set(source, saved["id"])["asset"] == saved["asset"]
    previous_stat = path.stat()
    changed = {**saved, "name": "My edits"}
    library.write_json(path, changed if source == "local" else {"sets": [changed]})
    os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    assert path.stat().st_size == previous_stat.st_size
    assert library.get_set(source, saved["id"])["name"] == "My edits"
    assert reads == [path, path]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        library.get_set(source, saved["id"])


@pytest.fixture
def save_route(monkeypatch):
    from aiohttp import web
    adapter = importlib.import_module("_vnccs.api.pose_set_library")
    handlers = {}
    def register(url):
        def decorate(handler):
            handlers[url] = handler
            return handler
        return decorate
    monkeypatch.setattr(web, "json_response", lambda data, status=200, **kwargs:
                        types.SimpleNamespace(data=data, status=status), raising=False)
    adapter.register_routes(types.SimpleNamespace(get=register, post=register))
    def save(data, action="save"):
        async def chunks(size):
            yield json.dumps(data).encode()
        request = types.SimpleNamespace(headers={"X-VNCCS-CSRF": "1"}, match_info={"action": action},
                                        content=types.SimpleNamespace(iter_chunked=chunks))
        return asyncio.run(handlers["/vnccs/pose-sets/{action}"](request))
    return save


def test_save_acknowledges_commit_with_an_unreadable_neighbor(save_route):
    broken = library.library_path("local", "b" * 32)
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"{preserve these bytes")
    response = save_route(record())
    assert response.status == 200 and response.data["ok"] is True
    assert len(response.data["sets"]) == 1
    assert response.data["errors"][0]["path"] == f"local/{broken.name}"
    assert len(list(broken.parent.glob("*.json"))) == 2
    assert broken.read_bytes() == b"{preserve these bytes"


def test_save_remains_successful_when_catalog_refresh_fails(save_route, monkeypatch):
    def unavailable():
        raise OSError("Library temporarily unavailable")
    monkeypatch.setattr(library, "catalog", unavailable)
    response = save_route(record())
    assert response.status == 200 and response.data["ok"] is True
    assert "sets" not in response.data
    assert "Changes saved" in response.data["errors"][0]["message"]
    assert len(library.local_sets()) == 1


def test_failed_write_is_not_acknowledged(save_route, monkeypatch):
    def unavailable(*args):
        raise OSError("Disk unavailable")
    monkeypatch.setattr(library, "write_json", unavailable)
    response = save_route(record())
    assert response.status == 500
    assert response.data["error"] == "Disk unavailable"
    assert not library.ROOT.exists()


def test_save_export_and_repository_import_above_previous_32_mib_limit(save_route, git_repository):
    data = record(1)
    data["asset"]["notes"] = "x" * (33 * 1024 * 1024)
    response = save_route(data)
    assert response.status == 200 and response.data["ok"] is True
    exported = {"version": 1, "sets": library.local_sets()}
    raw = json.dumps(exported).encode()
    assert len(raw) > 32 * 1024 * 1024
    git_repository.files = {"pose_sets.json": raw}
    library.repository_action({"action": "download", "repo_id": "owner/large-library"})
    remote = library.catalog()["repositories"][0]
    assert library.get_set(remote["id"], exported["sets"][0]["id"])["asset"] == data["asset"]


@pytest.mark.parametrize("set_id, name", [
    ("17ee87dd13254ade89cddfddc4a65779", "Default Male Poses"),
    ("9e9d22f219d04b8794cc5831f521f7e7", "Default Female Poses"),
])
def test_shipped_sets_load_with_all_previews_without_creating_user_files(monkeypatch, set_id, name):
    bundled = Path(__file__).resolve().parents[1] / "pose_set_defaults"
    monkeypatch.setattr(library, "BUNDLED", bundled)
    file = bundled / f"{set_id}.json"
    original = file.read_bytes()
    saved = json.loads(original)
    catalog = library.catalog()
    assert catalog["errors"] == []
    assert len(catalog["sets"]) == 2
    assert {"source": "bundled", "id": saved["id"], "name": name,
            "description": saved["description"], "count": 12} in catalog["sets"]
    loaded = library.get_set("bundled", saved["id"])
    assert loaded == saved
    assert len(loaded["previews"]) == len(loaded["asset"]["poses"]) == 12
    for preview in loaded["previews"]:
        with Image.open(io.BytesIO(base64.b64decode(preview, validate=True))) as image:
            assert image.format == "WEBP"
            image.verify()
    assert not library.ROOT.exists()
    with pytest.raises(FileNotFoundError):
        library.delete_set(saved["id"])
    assert file.read_bytes() == original


def test_system_update_and_user_delete_are_isolated_even_with_identical_ids():
    system = {**record(), "id": "a" * 32, "name": "System set"}
    user = {**system, "name": "My edited copy"}
    system_file = library.library_path("bundled", system["id"])
    user_file = library.library_path("local", user["id"])
    assert system_file != user_file
    library.write_json(system_file, system)
    library.write_json(user_file, user)
    user_bytes = user_file.read_bytes()
    assert {(item["source"], item["id"], item["name"]) for item in library.catalog()["sets"]} == {
        ("bundled", system["id"], "System set"), ("local", user["id"], "My edited copy")}
    system["asset"]["poses"][0]["prompt"] = "Updated system pose"
    library.write_json(system_file, system)
    assert library.get_set("bundled", system["id"]) == system
    assert library.get_set("local", user["id"])["asset"]["poses"][0]["prompt"] == "pose 0"
    assert user_file.read_bytes() == user_bytes
    with pytest.raises(ValueError, match="Only user"):
        library.delete_set(system["id"], "bundled")
    assert user_file.read_bytes() == user_bytes
    library.delete_set(user["id"], "local")
    assert library.catalog()["sets"][0]["source"] == "bundled"
    assert not user_file.exists()
    assert library.read_json(system_file) == system


def test_saving_user_set_with_system_name_always_creates_separate_user_file():
    system = {**record(), "id": "a" * 32}
    file = library.library_path("bundled", system["id"])
    library.write_json(file, system)
    original = file.read_bytes()
    user = library.save_set({**system, "source": "bundled"})
    assert user["id"] != system["id"]
    assert library.get_set("local", user["id"]) == user
    assert file.read_bytes() == original
    assert library.local_sets() == [user], "User export excludes system sets"


def test_delete_route_cannot_target_user_file_through_system_source(save_route):
    saved = {**record(), "id": "a" * 32}
    for source in ("local", "bundled"):
        library.write_json(library.library_path(source, saved["id"]), saved)
    user_bytes = library.library_path("local", saved["id"]).read_bytes()
    response = save_route({"source": "bundled", "id": saved["id"]}, action="delete")
    assert response.status == 400 and "Only user" in response.data["error"]
    assert library.library_path("local", saved["id"]).read_bytes() == user_bytes
    response = save_route({"source": "local", "id": saved["id"]}, action="delete")
    assert response.status == 200
    assert [item["source"] for item in response.data["sets"]] == ["bundled"]
