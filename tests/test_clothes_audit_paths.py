"""Clothes preview paths stay inside character storage without model dependencies."""

import importlib.util
import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sys
import types
from pathlib import Path

import pytest
from PIL import Image
import utils


@pytest.fixture
def clothes_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    generator = types.ModuleType("_vnccs.nodes.character_generator")
    for name in ("_call_comfy_node", "_resolution_scale_megapixels", "_resolution_scale_value", "VNCCS_CharacterGenerator"):
        setattr(generator, name, lambda *args, **kwargs: pytest.fail("Preview path lookup must not run generation"))
    generator.H3_FRAME_COUNT = 1
    generator.NATIVE_BACKGROUND_PROMPT = "transparent background"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        patch.setitem(sys.modules, generator.__name__, generator)
        spec = importlib.util.spec_from_file_location(
            "_vnccs.nodes._clothes_audit_paths", Path(__file__).parents[1] / "nodes/clothes_designer.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("link_kind", ["file", "neutral_directory", "costume_directory"])
def test_clothes_previews_never_follow_symlinks_outside_character_storage(clothes_paths, tmp_path, link_kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private image")
    sprites = Path(utils.character_dir("Alice")) / "Sprites"
    sprites.mkdir(parents=True)
    if link_kind == "costume_directory":
        (sprites / "Naked").symlink_to(outside, target_is_directory=True)
    elif link_kind == "neutral_directory":
        (sprites / "Naked").mkdir()
        (sprites / "Naked" / "Neutral").symlink_to(outside, target_is_directory=True)
    else:
        neutral = sprites / "Naked" / "Neutral"
        neutral.mkdir(parents=True)
        (neutral / "private.png").symlink_to(outside / "private.png")
    assert clothes_paths.get_latest_sprite_path("Alice", "Naked") is None
    assert clothes_paths.list_preview_sprite_files("Alice", "Naked") == []


def test_normal_and_nested_clothes_previews_keep_existing_selection_order(clothes_paths):
    directory = Path(utils.character_dir("Alice")) / "Sprites" / "Naked" / "Neutral"
    directory.mkdir(parents=True)
    file = directory / "pose.png"
    file.write_bytes(b"image")
    assert clothes_paths.get_latest_sprite_path("Alice", "Naked") == str(file)
    assert clothes_paths.list_preview_sprite_files("Alice", "Naked") == [str(file)]
    nested = directory / "nested"
    nested.mkdir()
    file.rename(nested / file.name)
    assert clothes_paths.get_latest_sprite_path("Alice", "Naked") == str(nested / file.name)


def test_archive_timestamps_cannot_select_an_old_reference(clothes_paths):
    import os
    neutral = Path(utils.character_dir("Alice")) / "Sprites" / "Naked" / "Neutral"
    (neutral / "V1").mkdir(parents=True)
    current, archive = neutral / "pose.png", neutral / "V1" / "pose.png"
    current.write_bytes(b"current")
    archive.write_bytes(b"archive")
    os.utime(current, (1000, 1000))
    os.utime(archive, (2000, 2000))
    assert clothes_paths.get_latest_sprite_path("Alice") == str(current)
    assert clothes_paths.list_preview_sprite_files("Alice") == utils.list_pose_preview_files("Alice") == [str(current)]
    current.unlink()
    assert clothes_paths.get_latest_sprite_path("Alice") is None


@pytest.mark.parametrize("link_kind", ["file", "neutral_directory"])
def test_clothes_preview_skips_unsafe_entries_without_hiding_valid_sprites(clothes_paths, tmp_path, link_kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private image")
    sprites = Path(utils.character_dir("Alice")) / "Sprites"
    valid_root = sprites / "Original" / "Neutral" if link_kind == "neutral_directory" else sprites / "Naked" / "Neutral"
    valid_root.mkdir(parents=True)
    valid = valid_root / "valid.png"
    valid.write_bytes(b"valid image")
    if link_kind == "file":
        (valid_root / "private.png").symlink_to(outside / "private.png")
    else:
        (sprites / "Naked").mkdir()
        (sprites / "Naked" / "Neutral").symlink_to(outside, target_is_directory=True)
    assert clothes_paths.list_preview_sprite_files("Alice") == [str(valid)]


@pytest.mark.parametrize("nested", [False, True])
def test_clothes_latest_preview_skips_unsafe_neutral_directory(clothes_paths, tmp_path, nested):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private image")
    costume = Path(utils.character_dir("Alice")) / "Sprites" / "Naked"
    valid_root = costume / "pose" if nested else costume
    valid_root.mkdir(parents=True)
    valid = valid_root / "valid.png"
    valid.write_bytes(b"valid image")
    (costume / "Neutral").symlink_to(outside, target_is_directory=True)
    assert clothes_paths.get_latest_sprite_path("Alice", "Naked") == str(valid)
    assert clothes_paths.ClothesDesigner.reference_sprite_path("Alice") == str(valid)


def test_clothes_reference_tensor_uses_displayed_exif_orientation(clothes_paths, monkeypatch, tmp_path):
    path = tmp_path / "phone-reference.jpg"
    image = Image.new("RGB", (3, 7), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    monkeypatch.setattr(clothes_paths.torch, "from_numpy",
        lambda array: types.SimpleNamespace(unsqueeze=lambda axis: array[None, ...]), raising=False)
    with Image.open(path) as stored:
        tensor = clothes_paths.ClothesDesigner._pil_image_tensor(stored)
    assert tensor.shape == (1, 3, 7, 3)


@pytest.mark.parametrize("link_kind", ["directory", "image", "metadata"])
def test_clothes_cache_cannot_read_or_write_outside_character_storage(clothes_paths, tmp_path, link_kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "private.json"
    target.write_text("private data")
    root = Path(utils.character_dir("Alice"))
    root.mkdir(parents=True)
    cache = root / "cache"
    if link_kind == "directory":
        cache.symlink_to(outside, target_is_directory=True)
    else:
        cache.mkdir()
        name = "preview_Dress.png" if link_kind == "image" else "preview_info_Dress.json"
        (cache / name).symlink_to(target)
    with pytest.raises(ValueError, match="outside allowed directory"):
        clothes_paths.ClothesDesigner.get_cache_paths("Alice", "Dress")
    assert target.read_text() == "private data"


@pytest.mark.parametrize("cached_costume,metadata,alias,expected", [
    ("Red Dress", "valid", True, "blue"),
    ("Red_Dress", "valid", True, "green"),
    ("Red Dress", "corrupt", False, "green"),
    ("Red Dress", "missing", False, "blue"),
    ("Red Dress", "missing", True, "green"),
])
@pytest.mark.parametrize("force_cache", [False, True])
def test_clothes_preview_cache_is_bound_to_requested_costume(
    clothes_paths, monkeypatch, cached_costume, metadata, alias, expected, force_cache,
):
    base = Path(utils.character_dir("Alice")) / "Sprites" / "Naked" / "Neutral"
    base.mkdir(parents=True)
    Image.new("RGB", (2, 3), "green").save(base / "base.png")
    utils.ensure_costume_structure("Alice", "Red Dress")
    if alias:
        utils.ensure_costume_structure("Alice", "Red_Dress")
    image_path, info_path = clothes_paths.ClothesDesigner.get_cache_paths("Alice", cached_costume)
    clothes_paths._save_preview_cache(Image.new("RGB", (2, 3), "blue"), image_path, info_path, {
        "hash": "preview", "widget_data": {"character": "Alice", "costume": cached_costume},
    })
    if metadata == "corrupt":
        Path(info_path).write_text("{broken metadata")
    elif metadata == "missing":
        Path(info_path).unlink()
    monkeypatch.setattr(clothes_paths.web, "Response", lambda status=200, **kwargs:
        types.SimpleNamespace(status=status, **kwargs), raising=False)
    request = types.SimpleNamespace(rel_url=types.SimpleNamespace(query={
        "character": "Alice", "costume": "Red Dress", "force_cache": str(force_cache).lower(),
    }))
    response = asyncio.run(clothes_paths.vnccs_get_preview(request))
    assert response.status == 200
    with Image.open(io.BytesIO(response.body)) as preview:
        assert preview.getpixel((0, 0)) == Image.new("RGB", (1, 1), expected).getpixel((0, 0))


def test_clothes_cache_route_waits_for_complete_owner_publication(clothes_paths, monkeypatch):
    base = Path(utils.character_dir("Alice")) / "Sprites" / "Naked" / "Neutral"
    base.mkdir(parents=True)
    Image.new("RGB", (2, 3), "green").save(base / "base.png")
    for costume in ("Red Dress", "Red_Dress"):
        utils.ensure_costume_structure("Alice", costume)
    image_path, info_path = clothes_paths.ClothesDesigner.get_cache_paths("Alice", "Red Dress")
    clothes_paths._save_preview_cache(Image.new("RGB", (2, 3), "blue"), image_path, info_path, {
        "widget_data": {"character": "Alice", "costume": "Red Dress"},
    })
    published_image = threading.Event()
    finish_publication = threading.Event()
    reader_waiting = threading.Event()
    atomic_output = clothes_paths.atomic_output_path
    storage_lock = clothes_paths.character_storage_lock

    @contextmanager
    def delayed_metadata(path):
        with atomic_output(path) as temporary:
            if path == info_path:
                published_image.set()
                assert finish_publication.wait(5)
            yield temporary

    @contextmanager
    def observed_lock(path):
        if published_image.is_set():
            reader_waiting.set()
        with storage_lock(path):
            yield

    monkeypatch.setattr(clothes_paths, "atomic_output_path", delayed_metadata)
    monkeypatch.setattr(clothes_paths, "character_storage_lock", observed_lock)
    monkeypatch.setattr(clothes_paths.web, "Response", lambda status=200, **kwargs:
        types.SimpleNamespace(status=status, **kwargs), raising=False)
    request = types.SimpleNamespace(rel_url=types.SimpleNamespace(query={
        "character": "Alice", "costume": "Red Dress", "force_cache": "true",
    }))
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(clothes_paths._save_preview_cache,
            Image.new("RGB", (2, 3), "red"), image_path, info_path,
            {"widget_data": {"character": "Alice", "costume": "Red_Dress"}})
        try:
            assert published_image.wait(5)
            reader = pool.submit(asyncio.run, clothes_paths.vnccs_get_preview(request))
            assert reader_waiting.wait(5)
            assert not reader.done()
        finally:
            finish_publication.set()
        writer.result(timeout=5)
        response = reader.result(timeout=5)
    with Image.open(io.BytesIO(response.body)) as preview:
        assert preview.getpixel((0, 0)) == (0, 128, 0)


def costume_request(data):
    async def body():
        return data
    return types.SimpleNamespace(json=body, headers={"X-VNCCS-CSRF": "1"})


@pytest.mark.parametrize("costume", ["Dress", "dress", "Naked", "Original"])
def test_new_costume_cannot_replace_existing_or_base_metadata(clothes_paths, monkeypatch, costume):
    utils.save_costume_info("Alice", "Dress", {"top": "authored silk", "extension": {"id": 7}})
    root = Path(utils.character_dir("Alice"))
    original = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(clothes_paths.web, "json_response", lambda data, status=200:
        types.SimpleNamespace(data=data, status=status), raising=False)
    response = asyncio.run(clothes_paths.vnccs_save_costume(costume_request({
        "character": "Alice", "costume": costume, "info": {}, "create_only": True,
    })))
    assert response.status == 409
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()} == original


def test_create_only_costume_requests_are_atomic_and_regular_edits_still_work(clothes_paths, monkeypatch):
    monkeypatch.setattr(clothes_paths.web, "json_response", lambda data, status=200:
        types.SimpleNamespace(data=data, status=status), raising=False)
    start = threading.Barrier(2)

    def create(top):
        start.wait(timeout=5)
        response = asyncio.run(clothes_paths.vnccs_save_costume(costume_request({
            "character": "Alice", "costume": "Dress", "info": {"top": top}, "create_only": True,
        })))
        return top, response.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create, top) for top in ("silk", "cotton")]
        results = [future.result(timeout=5) for future in futures]
    assert sorted(status for _, status in results) == [200, 409]
    winner = next(top for top, status in results if status == 200)
    assert utils.load_costume_info("Alice", "Dress") == {"top": winner}
    response = asyncio.run(clothes_paths.vnccs_save_costume(costume_request({
        "character": "Alice", "costume": "Dress", "info": {"top": "edited wool"},
        "revision": utils.load_costume_snapshot("Alice", "Dress")[1],
    })))
    assert response.status == 200
    assert utils.load_costume_info("Alice", "Dress") == {"top": "edited wool"}


def test_stale_costume_edits_preserve_newer_fields_and_cannot_recreate_deleted_costumes(clothes_paths, monkeypatch):
    monkeypatch.setattr(clothes_paths.web, "json_response", lambda data, status=200, **kwargs:
        types.SimpleNamespace(data=data, status=status, **kwargs), raising=False)
    original = {"top": "cotton", "shoes": "boots"}
    utils.save_costume_info("Alice", "Dress", original)
    get_request = types.SimpleNamespace(rel_url=types.SimpleNamespace(query={"character": "Alice", "costume": "Dress"}))
    revision = asyncio.run(clothes_paths.vnccs_get_costume(get_request)).headers["ETag"]
    async def save(info, version=revision):
        return await clothes_paths.vnccs_save_costume(costume_request({"character": "Alice", "costume": "Dress", "info": info, "revision": version}))
    first = asyncio.run(save({**original, "top": "silk"}))
    assert first.status == 200 and first.data["revision"] != revision
    assert asyncio.run(save({**original, "shoes": "sandals"})).status == 409
    assert utils.load_costume_info("Alice", "Dress") == {**original, "top": "silk"}
    assert asyncio.run(save(original, None)).status == 428
    utils.delete_costume("Alice", "Dress")
    assert asyncio.run(save(original, first.data["revision"])).status == 409
    assert "Dress" not in utils.load_config("Alice")["costumes"]
    assert not (Path(utils.character_dir("Alice")) / "Sprites" / "Dress").exists()


def test_costume_save_waits_without_blocking_the_event_loop(clothes_paths, monkeypatch):
    monkeypatch.setattr(clothes_paths.web, "json_response", lambda data, status=200:
        types.SimpleNamespace(data=data, status=status), raising=False)
    utils.save_costume_info("Alice", "Dress", {"top": "cotton"})
    revision = utils.load_costume_snapshot("Alice", "Dress")[1]
    ready, release = threading.Event(), threading.Event()
    def publish():
        with utils.character_storage_lock(utils.character_dir("Alice")):
            ready.set()
            release.wait(2)
    async def run():
        task = asyncio.create_task(clothes_paths.vnccs_save_costume(costume_request({
            "character": "Alice", "costume": "Dress", "info": {"top": "silk"},
            "revision": revision,
        })))
        await asyncio.sleep(.01)
        assert not task.done(), "HTTP processing must continue while publication holds the storage lock"
        release.set()
        assert (await task).status == 200
    thread = threading.Thread(target=publish)
    thread.start()
    try:
        assert ready.wait(2)
        asyncio.run(run())
    finally:
        release.set()
        thread.join(2)
