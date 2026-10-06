"""Disk publication and validation stay testable without the model runtime."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils
from nodes.preview_runtime import run_wizard_job


def test_paths_reject_symlink_escape_and_portable_traversal(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    for path in ("escape/file.png", "../outside/file.png", "..\\outside\\file.png", "C:\\outside\\file.png"):
        with pytest.raises(ValueError):
            utils.safe_join_under(str(root), path)


@pytest.mark.parametrize("group", utils.MAIN_DIRS)
@pytest.mark.parametrize("kind", ["character", "costume"])
@pytest.mark.parametrize("redirect_group", [True, False])
def test_structure_helpers_validate_all_paths_before_creating_directories(tmp_path, monkeypatch, group, kind, redirect_group):
    root, outside = tmp_path / "characters", tmp_path / "outside"
    character = root / "Alice"
    character.mkdir(parents=True)
    outside.mkdir()
    destination = character / group
    if not redirect_group:
        destination.mkdir()
        destination /= "Naked" if kind == "character" else "Coat"
    destination.symlink_to(outside, target_is_directory=True)
    before = set(root.rglob("*"))
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(root))
    with pytest.raises(ValueError, match="outside allowed directory"):
        if kind == "character":
            utils.ensure_character_structure("Alice")
        else:
            utils.ensure_costume_structure("Alice", "Coat")
    assert set(root.rglob("*")) == before
    assert not list(outside.iterdir())


def test_invalid_costume_data_never_writes_config(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    for value in ([], "shirt", {"top": []}, {"negative_prompt": None}):
        with pytest.raises(ValueError):
            utils.save_costume_info("Alice", "Coat", value)
    assert not list(tmp_path.iterdir())
    utils.save_costume_info("Alice", "Coat", {"top": "silk", "extension": {"id": 1}})
    assert utils.load_costume_info("Alice", "Coat")["extension"] == {"id": 1}


def test_failed_image_preparation_keeps_current_batch(tmp_path):
    target = tmp_path / "Neutral"
    target.mkdir()
    (target / "old.png").write_bytes(b"old")
    with pytest.raises(OSError, match="disk full"):
        with utils.staged_image_batch(str(target)) as stage:
            Path(stage, "new.png").write_bytes(b"partial")
            raise OSError("disk full")
    assert list(target.iterdir()) == [target / "old.png"]
    assert (target / "old.png").read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_publication_restores_current_batch(tmp_path, monkeypatch):
    target = tmp_path / "Neutral"
    target.mkdir()
    (target / "old.png").write_bytes(b"old")
    replace = utils.os.replace

    def fail_stage(source, destination):
        if Path(source).name.startswith(".vnccs-sprites-") and Path(destination) == target:
            raise OSError("publication failed")
        return replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", fail_stage)
    with pytest.raises(OSError, match="publication failed"):
        with utils.staged_image_batch(str(target)) as stage:
            Path(stage, "new.png").write_bytes(b"new")
    assert (target / "old.png").read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_successful_publication_versions_current_and_preserves_archives(tmp_path):
    target = tmp_path / "Neutral"
    (target / "V1").mkdir(parents=True)
    (target / "V1" / "first.png").write_bytes(b"first")
    (target / "old.png").write_bytes(b"old")
    with utils.staged_image_batch(str(target)) as stage:
        Path(stage, "new.png").write_bytes(b"new")
    assert (target / "new.png").read_bytes() == b"new"
    assert (target / "V2" / "old.png").read_bytes() == b"old"
    assert (target / "V1" / "first.png").read_bytes() == b"first"
    assert not (target / "old.png").exists()
    assert list(tmp_path.iterdir()) == [target]


def test_regeneration_retains_other_current_images(tmp_path):
    target = tmp_path / "Neutral"
    target.mkdir()
    for name in ("one.png", "two.png"):
        (target / name).write_bytes(b"old")
    with utils.staged_image_batch(str(target), version_existing=False) as stage:
        Path(stage, "one.png").write_bytes(b"new")
    assert (target / "one.png").read_bytes() == b"new"
    assert (target / "two.png").read_bytes() == b"old"


def test_post_commit_cleanup_warns_without_rejecting_published_images(tmp_path, monkeypatch, caplog):
    target = tmp_path / "Neutral"
    target.mkdir()
    (target / "old.png").write_bytes(b"old")
    remove = utils.shutil.rmtree
    def deny_backup(path, **kwargs):
        if ".vnccs-rollback-" in str(path):
            raise PermissionError("Backup is locked")
        return remove(path, **kwargs)
    monkeypatch.setattr(utils.shutil, "rmtree", deny_backup)
    with utils.staged_image_batch(str(target)) as stage:
        Path(stage, "new.png").write_bytes(b"new")
    assert (target / "new.png").read_bytes() == b"new"
    assert (target / "V1" / "old.png").read_bytes() == b"old"
    backups = list(tmp_path.glob(".vnccs-rollback-*"))
    assert len(backups) == 1
    assert str(backups[0]) in caplog.text












def test_wizard_worker_keeps_event_loop_responsive_and_scopes_events(monkeypatch):
    import threading
    import server

    started, release = threading.Event(), threading.Event()
    events = []
    monkeypatch.setattr(server.PromptServer.instance, "send_sync", lambda name, data: events.append((name, data)), raising=False)

    def inference(payload):
        started.set()
        assert release.wait(2)
        return SimpleNamespace(status=200)

    async def run():
        job = asyncio.create_task(run_wizard_job(inference, {"node_id": "17", "request_id": "request"}, "character"))
        try:
            for _ in range(100):
                await asyncio.sleep(0.002)
                if started.is_set():
                    break
            assert started.is_set(), "Inference did not start"
            assert not job.done(), "The event loop must run while inference is pending"
        finally:
            release.set()
        assert (await job).status == 200

    asyncio.run(run())
    assert [data["status"] for _, data in events] == ["queued", "running", "done"]
    assert all(name == "vnccs.wizard.stage" and data["node_id"] == "17" and data["request_id"] == "request" for name, data in events)
