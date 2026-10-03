"""Disk publication and validation stay testable without the model runtime."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils
from nodes import migration_assistant as ma
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


def test_post_commit_cleanup_warns_without_rejecting_published_images(tmp_path, monkeypatch, capsys):
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
    assert str(backups[0]) in capsys.readouterr().out


def test_migration_cleanup_warning_is_successful_sheet_publication(tmp_path, monkeypatch):
    legacy, current = tmp_path / "legacy", tmp_path / "current"
    sheet_dir = legacy / "Alice" / "Sheets" / "Coat" / "neutral"
    sheet_dir.mkdir(parents=True)
    from PIL import Image
    Image.new("RGBA", (2, 2), "blue").save(sheet_dir / "sheet.png")
    target = current / "Alice" / "Sprites" / "Coat" / "neutral"
    target.mkdir(parents=True)
    Image.new("RGBA", (2, 2), "red").save(target / "old.png")
    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(current))
    monkeypatch.setattr(ma, "_crop_sprites", lambda image: [image.copy()])
    remove = utils.shutil.rmtree
    def deny_backup(path, **kwargs):
        if ".vnccs-rollback-" in str(path):
            raise PermissionError("Backup is locked")
        return remove(path, **kwargs)
    monkeypatch.setattr(utils.shutil, "rmtree", deny_backup)
    result = ma._migrate_character({"log": []}, "Alice", "Alice", True)
    assert result["sprites_saved"] == 1
    assert result["failed_sheets"] == 0
    assert (target / "sprite_neutral_0000.png").exists()


def test_later_migration_exception_retains_prior_sheet_failures(monkeypatch):
    run = {"log": []}
    monkeypatch.setitem(ma.RUNS, "mixed-test", run)
    failed_paths = ["Sheets/Coat/neutral/broken.png"]
    def migrate(run, old_name, new_name, force):
        if old_name == "Bob":
            raise OSError("Disk full")
        return {"legacy_name": old_name, "failed_sheets": 1, "failed_sheet_paths": failed_paths}
    monkeypatch.setattr(ma, "_migrate_character", migrate)
    ma._run_migration("mixed-test", [{"legacy_name": name, "new_name": name} for name in ("Alice", "Bob", "Charlie")], False)
    assert run["status"] == "error"
    assert run["failed_sheets"] == 1
    assert run["failed_characters"] == ["Alice", "Bob", "Charlie"]
    assert run["results"][0]["failed_sheet_paths"] == failed_paths


def test_migration_partial_status_reports_failed_sheets(monkeypatch):
    run = {"log": []}
    monkeypatch.setitem(ma.RUNS, "partial-test", run)
    monkeypatch.setattr(ma, "_migrate_character", lambda *args: {
        "legacy_name": "Alice", "sprites_saved": 1, "failed_sheets": 1,
        "failed_sheet_paths": ["Sheets/Coat/neutral/broken.png"],
    })
    ma._run_migration("partial-test", [{"legacy_name": "Alice", "new_name": "Alice"}], False)
    assert run["status"] == "partial"
    assert run["failed_sheets"] == 1
    assert run["failed_characters"] == ["Alice"]
    assert run["results"][0]["failed_sheet_paths"] == ["Sheets/Coat/neutral/broken.png"]


@pytest.mark.parametrize("failure", ["alpha_repair", "target_directory", "target_boundary"])
def test_sheet_preparation_failure_retains_progress_and_retries_only_failed_sheet(tmp_path, monkeypatch, failure):
    from PIL import Image
    legacy, current, outside = (tmp_path / name for name in ("legacy", "current", "outside"))
    outside.mkdir()
    for costume in ("A", "B", "C"):
        source = legacy / "Alice" / "Sheets" / costume / "neutral" / "sheet.png"
        source.parent.mkdir(parents=True)
        Image.new("RGBA", (2, 2), "red").save(source)
    target = current / "Alice" / "Sprites" / "B" / "neutral"
    original_makedirs = ma.os.makedirs
    if failure == "alpha_repair":
        target.mkdir(parents=True)
        (target / "sprite_neutral_0000.png").write_bytes(b"corrupt image")
    elif failure == "target_boundary":
        target.parent.parent.mkdir(parents=True)
        target.parent.symlink_to(outside, target_is_directory=True)
    else:
        def deny_target(path, *args, **kwargs):
            if Path(path) == target:
                raise PermissionError("Target directory is locked")
            return original_makedirs(path, *args, **kwargs)
        monkeypatch.setattr(ma.os, "makedirs", deny_target)
    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(current))
    monkeypatch.setattr(ma, "_crop_sprites", lambda image: [image.copy()])
    run = {"log": []}
    monkeypatch.setitem(ma.RUNS, "sheet-failure", run)
    ma._run_migration("sheet-failure", [{"legacy_name": "Alice", "new_name": "Alice"}], False)
    assert run["status"] == "partial"
    assert run["failed_sheets"] == 1
    assert run["failed_characters"] == ["Alice"]
    result = run["results"][0]
    assert result["sprites_saved"] == 2
    assert result["failed_sheet_paths"] == ["B/neutral/sheet.png"]
    successful = [current / "Alice" / "Sprites" / costume / "neutral" / "sprite_neutral_0000.png" for costume in ("A", "C")]
    for path in successful:
        Image.new("RGBA", (2, 2), "green").save(path)
    original_images = [path.read_bytes() for path in successful]
    monkeypatch.setattr(ma.os, "makedirs", original_makedirs)
    if failure == "target_boundary":
        target.parent.unlink()
    retry = {"log": []}
    monkeypatch.setitem(ma.RUNS, "sheet-retry", retry)
    ma._run_migration("sheet-retry", [{"legacy_name": "Alice", "new_name": "Alice",
                                       "retry_sheets": result["failed_sheet_paths"]}], True)
    assert retry["status"] == "done"
    assert retry["failed_sheets"] == 0
    assert retry["results"][0]["sheet_count"] == 1
    assert retry["results"][0]["sprites_saved"] == 1
    assert [path.read_bytes() for path in successful] == original_images
    assert all(not list(path.parent.glob("V*")) for path in successful)
    assert not list(outside.iterdir())
    with Image.open(target / "sprite_neutral_0000.png") as image:
        assert image.getpixel((0, 0)) == (255, 0, 0, 255)


def test_migration_job_history_and_logs_are_bounded(monkeypatch):
    monkeypatch.setattr(ma, "RUNS", ma.OrderedDict())
    monkeypatch.setattr(ma, "time", SimpleNamespace(time=lambda: 1000))
    for index in range(ma.MAX_RUNS + 20):
        ma.RUNS[str(index)] = {"status": "done", "updated_at": 1000}
    ma.RUNS["active"] = {"status": "running", "updated_at": 0}
    ma._prune_runs()
    assert len(ma.RUNS) == ma.MAX_RUNS
    assert "active" in ma.RUNS
    assert ma._start_job(lambda *args: None, (), 1, "Queued") is None
    run = {"log": []}
    for index in range(ma.MAX_LOG_LINES + 10):
        ma._log(run, str(index))
    assert len(run["log"]) == ma.MAX_LOG_LINES


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
