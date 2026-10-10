"""Audit regressions at storage and HTTP boundaries, without torch."""

import asyncio
import importlib.util
import json
import queue
import sys
import threading
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import utils
from nodes import vnccs_control_center as control
from nodes import generator_context
from nodes.emotion_library import load_emotion_library, save_emotion_library
from nodes.runtime_cleanup import inference_lock


@pytest.fixture
def routes(monkeypatch, tmp_path):
    import server
    from aiohttp import web
    from nodes import preview_runtime
    handlers = {}
    def register(path):
        def decorate(handler):
            handlers[path] = handler
            return handler
        return decorate
    monkeypatch.setattr(server.PromptServer.instance, "routes", SimpleNamespace(get=register, post=register))
    monkeypatch.setattr(web, "json_response", lambda data, status=200: SimpleNamespace(status=status, data=data), raising=False)
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    package_name = "_vnccs_audit_routes"
    nodes = ModuleType(f"{package_name}.nodes")
    nodes.__file__ = __file__
    nodes.NODE_CLASS_MAPPINGS = {"Test": object()}
    nodes.NODE_DISPLAY_NAME_MAPPINGS = {}
    for suffix, module in (("utils", utils), ("nodes", nodes),
                           ("nodes.preview_runtime", preview_runtime), ("nodes.generator_context", generator_context)):
        monkeypatch.setitem(sys.modules, f"{package_name}.{suffix}", module)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(package_name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    return handlers


def request(query=None, data=None, headers=None):
    async def body():
        return data
    return SimpleNamespace(rel_url=SimpleNamespace(query=query or {}), json=body,
                           headers=headers if headers is not None else {"X-VNCCS-CSRF": "1"})


def test_legacy_create_costume_requires_privileged_request(routes):
    handler = routes["/vnccs/create_costume"]
    query = {"character": "Alice", "costume": "Coat"}
    result = asyncio.run(handler(request(query, headers={"Sec-Fetch-Site": "cross-site"})))
    assert result.status == 403
    assert not Path(utils.character_dir("Alice")).exists()
    assert asyncio.run(handler(request(query))).data == {"ok": True, "costume": "Coat"}


@pytest.mark.parametrize("broken", [b"{broken", b"[]", b'{"costumes": null}'])
def test_create_costume_preserves_invalid_config_bytes(routes, broken):
    path = Path(utils.config_path("Alice"))
    path.parent.mkdir(parents=True)
    path.write_bytes(broken)
    result = asyncio.run(routes["/vnccs/create_costume"](request({"character": "Alice", "costume": "Coat"})))
    assert result.status == 500
    assert path.read_bytes() == broken
    assert not (path.parent / "Sprites").exists()


def test_create_costume_read_denial_keeps_existing_profile(routes, monkeypatch):
    import builtins
    utils.save_config("Alice", {"character_info": {"name": "Alice"}, "costumes": {}})
    path = Path(utils.config_path("Alice"))
    original, real_open = path.read_bytes(), builtins.open
    def deny(file, *args, **kwargs):
        if str(file) == str(path):
            raise PermissionError("locked profile")
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", deny)
    result = asyncio.run(routes["/vnccs/create_costume"](request({"character": "Alice", "costume": "Coat"})))
    assert result.status == 500
    assert path.read_bytes() == original


def test_concurrent_costume_creation_preserves_profile_and_both_costumes(routes):
    utils.save_config("Alice", {"character_info": {"name": "Alice", "extension": 42}, "costumes": {}})
    def create(costume):
        return asyncio.run(routes["/vnccs/create_costume"](request({"character": "Alice", "costume": costume})))
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert all(result.status == 200 for result in executor.map(create, ["Coat", "Dress"]))
    config = utils.load_config("Alice", strict=True)
    assert config["character_info"]["extension"] == 42
    assert set(config["costumes"]) == {"Coat", "Dress"}


@pytest.mark.parametrize("model_job", [False, True])
def test_delete_waits_for_publication_and_forgets_regeneration(routes, monkeypatch, model_job):
    utils.save_config("Alice", {"character_info": {"name": "Alice"}, "costumes": {}})
    root = Path(utils.character_dir("Alice"))
    scope = "audit:scope"
    context = {"cache_dir": str(root / "cache"), "updated_at": time.monotonic()}
    other = {"cache_dir": str(root.parent / "Bob" / "cache"), "updated_at": time.monotonic()}
    contexts = OrderedDict([((scope, "node"), context), ("other", other)])
    expired = []
    monkeypatch.setattr(generator_context, "_LIVE_GENERATOR_CONTEXTS", contexts)
    monkeypatch.setattr(generator_context, "expire_cache_progress", expired.append)
    assert generator_context._get_generator_context("node", scope) is context
    ready, release = threading.Event(), threading.Event()
    def publish():
        lock = inference_lock if model_job else utils.character_storage_lock(str(root))
        with lock:
            ready.set()
            assert release.wait(3)
            with utils.staged_image_batch(str(root / "Sprites" / "Coat"), lock_root=str(root)) as stage:
                Path(stage, "sprite.png").write_bytes(b"published")
    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            publication = executor.submit(publish)
            assert ready.wait(2)
            deletion = asyncio.create_task(routes["/vnccs/delete"](request(data={"name": "Alice"})))
            try:
                await asyncio.sleep(0.03)
                assert not deletion.done()
                assert root.exists()
            finally:
                release.set()
            assert (await deletion).data["deleted"] is True
            publication.result(timeout=2)
    asyncio.run(run())
    assert not root.exists()
    assert generator_context._get_generator_context("node", scope) is None
    assert generator_context._get_generator_context("other") is other
    assert expired == [hashlib.sha256(scope.encode()).hexdigest()]


@pytest.mark.parametrize("name", ["../../outside.safetensors", "..\\..\\outside.safetensors", "absolute", "link/outside.safetensors"])
@pytest.mark.parametrize("host_lookup", [False, True])
def test_model_lookup_rejects_every_escape_before_returning(tmp_path, name, host_lookup):
    root = tmp_path / "models" / "loras"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"private")
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    folders = SimpleNamespace(get_folder_paths=lambda category: [str(root)],
                              get_full_path=lambda *args: str(outside) if host_lookup else None)
    value = str(outside) if name == "absolute" else name
    assert utils.get_full_path_agnostic(folders, "loras", value, require_exists=True) is None
    assert utils.get_full_path_agnostic(folders, "loras", value) is None


@pytest.mark.parametrize("slash", ["/", "\\"])
def test_model_lookup_preserves_registered_symlink_roots_and_portable_paths(tmp_path, slash):
    real = tmp_path / "real"
    (real / "pack").mkdir(parents=True)
    model = real / "pack" / "model.safetensors"
    model.write_bytes(b"model")
    root = tmp_path / "registered"
    root.symlink_to(real, target_is_directory=True)
    folders = SimpleNamespace(get_folder_paths=lambda category: [str(root)], get_full_path=lambda *args: None)
    result = utils.get_full_path_agnostic(folders, "loras", f"pack{slash}model.safetensors", require_exists=True)
    assert Path(result).resolve() == model
    assert Path(utils.get_full_path_agnostic(folders, "loras", str(root / "pack" / model.name))).resolve() == model
    assert utils.get_full_path_agnostic(folders, "loras", str(model), require_exists=True) == str(model)


@pytest.mark.parametrize("content", [b"{broken", b"null", b'{"lora": {}}', b'{"lora": [null]}'])
@pytest.mark.parametrize("remove", [False, True])
def test_custom_lora_invalid_restore_never_overwrites_bytes(tmp_path, monkeypatch, content, remove):
    path = tmp_path / "loras.json"
    path.write_bytes(content)
    monkeypatch.setattr(control, "_get_custom_loras_path", lambda **kwargs: str(path))
    with pytest.raises(ValueError):
        if remove:
            control._remove_custom_lora(name="old")
        else:
            control._save_custom_loras([{"name": "new", "local_path": "models/loras/new.safetensors"}])
    assert path.read_bytes() == content


@pytest.mark.parametrize("remove", [False, True])
def test_custom_lora_failed_write_keeps_original(tmp_path, monkeypatch, remove):
    path = tmp_path / "loras.json"
    original = b'{"lora": [{"name":"old", "local_path":"models/loras/old.safetensors", "extension":42}]}'
    path.write_bytes(original)
    monkeypatch.setattr(control, "_get_custom_loras_path", lambda **kwargs: str(path))
    def fail(data, handle, **kwargs):
        handle.write("partial")
        raise OSError("disk full")
    monkeypatch.setattr(control.json, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        if remove:
            control._remove_custom_lora(name="old")
        else:
            control._save_custom_loras([{"name": "new", "local_path": "models/loras/new.safetensors"}])
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_custom_lora_concurrent_additions_keep_both_entries(tmp_path, monkeypatch):
    path = tmp_path / "loras.json"
    monkeypatch.setattr(control, "_get_custom_loras_path", lambda **kwargs: str(path))
    def add(name):
        control._save_custom_loras([{"name": name, "local_path": f"models/loras/{name}.safetensors"}])
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(add, ["one", "two"]))
    assert {entry["name"] for entry in control._load_custom_loras()} == {"one", "two"}


def test_custom_lora_legacy_list_and_read_denial_preserve_data(tmp_path, monkeypatch):
    import builtins
    path = tmp_path / "loras.json"
    entry = {"name": "old", "local_path": "models/loras/old.safetensors", "extension": {"value": 7}}
    path.write_text(json.dumps([entry]))
    monkeypatch.setattr(control, "_get_custom_loras_path", lambda **kwargs: str(path))
    control._save_custom_loras([{"name": "new", "local_path": "models/loras/new.safetensors"}])
    assert control._load_custom_loras()[0] == entry
    original, real_open = path.read_bytes(), builtins.open
    def deny(file, *args, **kwargs):
        if str(file) == str(path):
            raise PermissionError("locked registry")
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", deny)
    with pytest.raises(PermissionError):
        control._remove_custom_lora(name="old")
    assert path.read_bytes() == original


def test_duplicate_model_downloads_share_pending_job_and_can_retry(monkeypatch, tmp_path):
    from aiohttp import web
    pending = queue.Queue()
    statuses = {}
    entry = {"name": "Model", "local_path": "models/diffusion_models/model.safetensors", "hf_path": "model.safetensors"}
    monkeypatch.setattr(control, "_DOWNLOAD_QUEUE", pending)
    monkeypatch.setattr(control, "_DOWNLOAD_STATUS", statuses)
    monkeypatch.setattr(control, "_get_cc_config", lambda *args, **kwargs: {"models": [entry]})
    monkeypatch.setattr(control, "_resolve_model_download_path", lambda path: str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    payload = {"repo_id": "demo/repo", "category": "models", "name": "Model"}
    async def run():
        responses = await asyncio.gather(*(control.cc_download(request(data=payload)) for _ in range(3)))
        assert all(response.status == 200 for response in responses)
        assert pending.qsize() == 1
        statuses["cc_models_Model"] = {"status": "downloading", "message": "50%"}
        assert (await control.cc_download(request(data=payload))).data["message"] == "50%"
        assert pending.qsize() == 1
        statuses["cc_models_Model"] = {"status": "error"}
        await control.cc_download(request(data=payload))
        assert pending.qsize() == 2
    asyncio.run(run())


def test_legacy_custom_emotions_migrate_once_and_survive_default_replacement(tmp_path):
    default, user = tmp_path / "emotions.json", tmp_path / "user" / "emotions.json"
    emotion = {"safe_name": "legacy", "key": "Legacy", "description": "smile", "extension": {"value": 7}}
    default.write_text(json.dumps({"Mood": [], "Custom": [emotion]}))
    original = default.read_bytes()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "legacy.webp").write_bytes(b"original image")
    assert load_emotion_library(str(default), str(user))["Custom"] == [emotion]
    assert default.read_bytes() == original
    assert (user.parent / "images" / "legacy.webp").read_bytes() == b"original image"
    saved = user.read_bytes()
    assert load_emotion_library(str(default), str(user))["Custom"] == [emotion]
    assert user.read_bytes() == saved
    default.write_text('{"Mood": []}')
    assert load_emotion_library(str(default), str(user))["Custom"] == [emotion]


def test_emotion_library_failed_restore_and_publication_keep_bytes(tmp_path, monkeypatch):
    default, user = tmp_path / "emotions.json", tmp_path / "user.json"
    default.write_text('{"Mood": []}')
    user.write_bytes(b"{broken user data")
    with pytest.raises(ValueError):
        load_emotion_library(str(default), str(user))
    assert user.read_bytes() == b"{broken user data"
    user.write_bytes(b'{"Custom": []}')
    replace = utils.os.replace
    def fail(source, target):
        if str(target) == str(user):
            raise OSError("publication failed")
        return replace(source, target)
    monkeypatch.setattr(utils.os, "replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        save_emotion_library(str(user), {"Custom": []})
    assert user.read_bytes() == b'{"Custom": []}'
    assert set(tmp_path.iterdir()) == {default, user}


def test_interrupted_legacy_emotion_migration_is_recoverable_and_retryable(tmp_path, monkeypatch):
    default, user = tmp_path / "emotions.json", tmp_path / "user" / "emotions.json"
    entry = {"safe_name": "legacy", "key": "Legacy", "description": "smile"}
    default.write_text(json.dumps({"Custom": [entry]}))
    original, replace = default.read_bytes(), utils.os.replace
    def fail(source, target):
        if str(target) == str(user):
            raise OSError("publication interrupted")
        return replace(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(utils.os, "replace", fail)
        with pytest.raises(OSError, match="publication interrupted"):
            load_emotion_library(str(default), str(user))
    assert default.read_bytes() == original
    assert not user.exists()
    assert load_emotion_library(str(default), str(user))["Custom"] == [entry]
    assert json.loads(user.read_text())["Custom"] == [entry]
