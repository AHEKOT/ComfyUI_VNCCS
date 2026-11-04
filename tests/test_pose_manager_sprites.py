import hashlib
import io
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from conftest import _preload_node

sprites = _preload_node("pose_manager_sprites")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.setattr(sprites, "ROOT", tmp_path / "VNCCS" / "PoseSetLibrary" / "sprites")
    studio = tmp_path / "Utils" / "PoseLibrary"
    studio.mkdir(parents=True)
    (studio / "repositories.json").write_text('{"user": "authored settings"}')
    (studio / "local-pose.json").write_text('{"bones": {"head": [1, 2, 3]}}')
    before = {p.name: p.read_bytes() for p in studio.iterdir()}
    output = io.BytesIO()
    Image.new("RGB", (12, 24), "pink").save(output, format="WEBP")
    pose = {"type": "single_pose", "bones": {"head": [1, 2, 3]}, "prompt": "Kept"}
    files = {"poses/Female/Wave.json": json.dumps(pose).encode(), "previews/Female/Wave.webp": output.getvalue()}
    entry = {"name": "Wave", "category": "Female", "tags": ["hello"], "asset_type": "pose"}
    for kind, name in zip(("json", "preview"), files):
        entry[f"{kind}_path"] = name
        entry[f"{kind}_sha256"] = hashlib.sha256(files[name]).hexdigest()
    files["pose_library.json"] = json.dumps({"schema_version": 1, "poses": [entry]}).encode()
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        file = Path(kwargs["cache_dir"]) / kwargs["revision"] / kwargs["filename"]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(files[kwargs["filename"]])
        return str(file)
    info = types.SimpleNamespace(sha="a" * 40, siblings=[types.SimpleNamespace(rfilename=name, size=len(data)) for name, data in files.items()])
    monkeypatch.setattr(sys.modules["huggingface_hub"], "HfApi", lambda **kwargs: types.SimpleNamespace(model_info=lambda *args, **kwargs: info), raising=False)
    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", fetch)
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = type("LocalEntryNotFoundError", (FileNotFoundError,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    yield files, info, calls, pose
    assert {p.name: p.read_bytes() for p in studio.iterdir()} == before, "Manager must never change Studio files"


def test_isolated_download_reload_toggle_and_assets(repository):
    files, info, calls, pose = repository
    assert sprites.catalog()["repository"]["enabled"] is True
    result = sprites.ensure()
    identifier = result["poses"][0]["manager_sprite_id"]
    assert sprites.asset(identifier) == pose
    image, mime = sprites.asset(identifier, preview=True)
    assert image == files["previews/Female/Wave.webp"] and mime == "image/webp"
    assert all(call["repo_id"] == sprites.REPOSITORY and call["token"] is False and call["revision"] == info.sha
               and Path(call["cache_dir"]).is_relative_to(sprites.ROOT) for call in calls)
    assert sprites.toggle(False)["poses"] == []
    assert sprites.ensure()["repository"]["enabled"] is False
    assert sprites.ensure(refresh=True)["repository"]["enabled"] is False
    assert sprites.toggle(True)["poses"] == result["poses"]


def test_failed_refresh_and_bad_checksums_preserve_catalog(repository):
    files, _, _, _ = repository
    result = sprites.ensure()
    before = sprites.path("repositories.json").read_bytes()
    files["pose_library.json"] = b'{"poses": [{"name": "bad"}]}'
    with pytest.raises(ValueError):
        sprites.ensure(refresh=True)
    assert sprites.path("repositories.json").read_bytes() == before
    files["poses/Female/Wave.json"] = b'{"bones": {}}'
    with pytest.raises(ValueError, match="checksum"):
        sprites.asset(result["poses"][0]["id"])
    assert sprites.path("repositories.json").read_bytes() == before


def test_independent_repositories_add_toggle_refresh_remove_and_legacy_cache(repository):
    _, _, calls, pose = repository
    initial = sprites.ensure()
    old = sprites.repositories()[0]
    sprites.write_json(sprites.path("catalog.json"), old)
    sprites.write_json(sprites.path("settings.json"), {"enabled": False})
    sprites.path("repositories.json").unlink()
    assert sprites.ensure()["poses"] == []
    assert sprites.toggle(True)["poses"] == initial["poses"]
    added = sprites.repository_action("add", "https://huggingface.co/other/poses")
    assert [repo["repo_id"] for repo in added["repositories"]] == [sprites.REPOSITORY, "other/poses"]
    assert len(added["poses"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        sprites.repository_action("add", "other/poses")
    one = sprites.repository_action("toggle", sprites.REPOSITORY, False)
    assert [item["repository"] for item in one["poses"]] == ["other/poses"]
    assert sprites.asset(one["poses"][0]["id"], repo_id="other/poses") == pose
    assert calls[-1]["repo_id"] == "other/poses"
    assert len(sprites.repository_action("refresh", "other/poses")["poses"]) == 1
    sprites.repository_action("remove", "other/poses")
    with pytest.raises(FileNotFoundError):
        sprites.asset(one["poses"][0]["id"], repo_id="other/poses")
    sprites.repository_action("remove", sprites.REPOSITORY)
    assert sprites.ensure()["repositories"] == [], "Opening the library must not re-add removed repositories"
    before = sprites.path("repositories.json").read_bytes()
    for value in ["../outside", "https://example.com/owner/repo", "owner/repo?token=x", None]:
        with pytest.raises(ValueError):
            sprites.repository_action("add", value)
    assert sprites.path("repositories.json").read_bytes() == before


@pytest.mark.parametrize("bad_path", ["../escape.json", "/outside.json", "a\\escape.json", "a/../escape.json", "a//escape.json"])
def test_manifest_rejects_paths_before_install(repository, bad_path):
    files, info, calls, _ = repository
    data = json.loads(files["pose_library.json"])
    data["poses"][0]["json_path"] = bad_path
    files["pose_library.json"] = json.dumps(data).encode()
    info.siblings.append(types.SimpleNamespace(rfilename=bad_path, size=10))
    with pytest.raises(ValueError, match="Unsafe"):
        sprites.ensure()
    assert not sprites.path("repositories.json").exists()
    assert [call["filename"] for call in calls] == ["pose_library.json"]


def test_bad_ids_corrupt_state_and_symlinks_do_not_overwrite_data(repository, tmp_path):
    sprites.ensure()
    for identifier in ["../outside", "", "z" * 32]:
        with pytest.raises(ValueError):
            sprites.asset(identifier)
    sprites.path("repositories.json").write_text("{broken")
    with pytest.raises(ValueError):
        sprites.ensure()
    assert sprites.path("repositories.json").read_text() == "{broken"
    outside = tmp_path / "outside"
    outside.mkdir()
    sprites.path("repositories.json").unlink()
    sprites.path("repositories.json").symlink_to(outside / "settings.json")
    with pytest.raises(ValueError):
        sprites.toggle(False)
    assert not (outside / "settings.json").exists()


def test_http_boundary_is_private_bounded_and_never_calls_studio(repository, monkeypatch):
    import asyncio
    import importlib
    from aiohttp import web

    adapter = importlib.import_module("_vnccs.api.pose_manager_sprites")
    handlers = {}
    def register(method):
        def route(url):
            def decorate(handler):
                handlers[method, url] = handler
                return handler
            return decorate
        return route
    monkeypatch.setattr(web, "json_response", lambda data, status=200: types.SimpleNamespace(data=data, status=status), raising=False)
    adapter.register_routes(types.SimpleNamespace(get=register("GET"), post=register("POST")))
    assert all(url.startswith("/vnccs/pose-manager/sprites") for _, url in handlers)
    post = handlers["POST", "/vnccs/pose-manager/sprites"]
    def request(raw, headers):
        async def chunks(size):
            for offset in range(0, len(raw), 13):
                yield raw[offset:offset + 13]
        return types.SimpleNamespace(content=types.SimpleNamespace(iter_chunked=chunks), headers=headers)
    for raw, headers, status in [
        (b'{"action":"ensure"}', {}, 403),
        (b'{"action":"ensure"}', {"X-VNCCS-CSRF": "1", "Sec-Fetch-Site": "cross-site"}, 403),
        (b"x" * 5000, {"X-VNCCS-CSRF": "1"}, 413),
        (b'{"action":"delete"}', {"X-VNCCS-CSRF": "1"}, 400),
        (b'{"action":"toggle","enabled":"false"}', {"X-VNCCS-CSRF": "1"}, 400),
    ]:
        assert asyncio.run(post(request(raw, headers))).status == status
    assert not sprites.path("repositories.json").exists()
    response = asyncio.run(post(request(b'{"action":"ensure"}', {"X-VNCCS-CSRF": "1"})))
    assert response.status == 200 and response.data["poses"]
