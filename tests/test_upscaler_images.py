"""Upscale regeneration uses the full-resolution PNGs, without tensor duplicates."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from nodes import character_generator as cg
import utils


@pytest.fixture(params=["character", "clothes", "original", "naked"])
def generated(request, monkeypatch, tmp_path):
    mode = request.param
    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    costume = "Coat" if mode == "clothes" else "Naked"
    sheets = root / "Sheets" / costume / "neutral"
    sheets.mkdir(parents=True)
    monkeypatch.setattr(cg.folder_paths, "get_output_directory", lambda: str(output))
    if mode == "clothes":
        utils.save_costume_info("Alice", costume, {})
    node = {"character": cg.VNCCS_CharacterGenerator,
            "clothes": cg.VNCCS_ClothesGenerator,
            "original": cg.VNCCS_CharacterCloneGenerator,
            "naked": cg.VNCCS_CharacterCloneGenerator}[mode]()
    prefix = f"{mode}_" if mode in {"original", "naked"} else ""
    events, calls = [], []
    change = [0.0]
    monkeypatch.setattr(cg.server.PromptServer.instance, "send_sync", lambda _, payload: events.append(payload), raising=False)
    monkeypatch.setattr(node, "_find_pose_lora", lambda *args: {})
    monkeypatch.setattr(node, "_find_clothes_lora", lambda *args: {})
    monkeypatch.setattr(node, "_extract_pipe", lambda *args: {"seed": 0})
    monkeypatch.setattr(node, "_run_remove_clothes", lambda image, *args, **kwargs: image)
    monkeypatch.setattr(node, "_run_upscaler_models", lambda *args, **kwargs: (None, None))

    def poses(images, *args, **kwargs):
        calls.append((kwargs.get("stage", "pose_generation"), images.shape[0]))
        return images + change[0]

    def upscale(images, *args, **kwargs):
        batch = node._list_to_batch(images)
        calls.append((kwargs["stage"], batch.shape[0]))
        return [batch + change[0]]

    monkeypatch.setattr(node, "_run_pose_generation", poses)
    monkeypatch.setattr(node, "_run_seedvr_upscale_batch", upscale)
    images = torch.stack([torch.full((8, 6, 4), value / 255) for value in (30, 80, 130)])
    images[..., 3] = torch.arange(8).reshape(1, 8, 1) / 7
    state = {"character_name": "Alice", "nsfw_enabled": mode == "naked",
             "bg_remove": {"preset": "Native"}, "upscaler": {"mode": "seedvr"},
             "ui": {"progress_scope": "upscale-audit:png-test"}}
    pipe = SimpleNamespace(seed_int=0)

    def run():
        return node.process(images, images[:1], pipe, "Pose", widget_data=json.dumps(state),
                            sheets_path=str(sheets), unique_id="png-test")

    run()
    stage = prefix + "upscaler"
    cache = Path(cg._get_generator_context("png-test", state["ui"]["progress_scope"])["cache_dir"])
    assert not list(root.rglob("*.pt"))
    assert not (cache / "_stage_cache" / f"{stage}.images").exists()
    return SimpleNamespace(node=node, run=run, state=state, stage=stage, prefix=prefix,
                           cache=cache, calls=calls, events=events, change=change, root=root,
                           mode=mode, images=images, pipe=pipe)


@pytest.mark.parametrize("index", [None, 0, 2])
@pytest.mark.parametrize("stage", ["bg_remove", "upscaler", "pose_generation"])
def test_regenerate_uses_pngs_and_preserves_other_upscaled_images(generated, monkeypatch, stage, index):
    g = generated
    files = sorted((g.cache / g.stage).glob(g.stage + "_*.png"))
    before = [path.read_bytes() for path in files]
    # Existing installations may still have an obsolete tensor. It is never read.
    legacy = g.cache / "_stage_cache" / (g.stage + ".pt")
    legacy.write_bytes(b"obsolete tensor must be ignored")
    g.calls.clear()
    g.events.clear()
    g.change[0] = 0.1
    g.state.update(regenerate_from=g.prefix + stage, regenerate_index=index)
    g.run()
    assert legacy.read_bytes() == b"obsolete tensor must be ignored"

    target_calls = [count for name, count in g.calls if name == g.stage]
    assert target_calls == ([] if stage == "bg_remove" else [3 if index is None else 1])
    pose_calls = [count for name, count in g.calls if name == g.prefix + "pose_generation"]
    assert pose_calls == ([3 if index is None else 1] if stage == "pose_generation" else [])
    assert not any(name == "source_upscaler" for name, _ in g.calls)
    for position, path in enumerate(files):
        changed = stage != "bg_remove" and (index is None or index == position)
        assert (path.read_bytes() != before[position]) == changed
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.size == (6, 8)
    loaded = cg._load_upscaled_images(str(g.cache), g.stage, 3)
    assert loaded.shape == (3, 8, 6, 4)
    if stage == "bg_remove":
        with Image.open(files[2]) as image:
            assert np.array_equal((loaded[2].numpy() * 255).astype(np.uint8), np.array(image))
    done = [e for e in g.events if e["stage"] == g.stage and e["status"] == "done"]
    assert len(done) == 1
    assert done[0]["total"] == 3
    assert all(url.startswith("/view?") for url in done[0]["images"])


@pytest.mark.parametrize("failure", ["missing", "corrupt", "external_link"])
def test_missing_or_invalid_png_never_publishes_a_partial_selected_batch(generated, tmp_path, failure):
    g = generated
    before = {p: p.read_bytes() for p in (g.root / "Sprites").rglob("*.png")}
    target = g.cache / g.stage / (g.stage + "_03.png")
    target.unlink()
    if failure == "corrupt":
        target.write_bytes(b"broken PNG")
    elif failure == "external_link":
        outside = tmp_path / "outside.png"
        Image.new("RGBA", (6, 8), "red").save(outside)
        target.symlink_to(outside)
    g.state.update(regenerate_from=g.prefix + "bg_remove", regenerate_index=0)
    with pytest.raises(ValueError, match="Run the generator normally"):
        g.run()
    assert {p: p.read_bytes() for p in (g.root / "Sprites").rglob("*.png")} == before


@pytest.mark.parametrize("channels", [3, 4])
def test_upscale_png_storage_is_not_limited_to_the_preview_card_count(monkeypatch, tmp_path, channels):
    monkeypatch.setattr(cg, "base_output_dir", lambda: str(tmp_path))
    cache = tmp_path / "cache"
    node = cg.VNCCS_CharacterGenerator()
    images = torch.stack([torch.full((4, 6, channels), i / 255) for i in range(15)])
    node._save_upscaled_stage(str(cache), "upscaler", images)
    restored = cg._load_upscaled_images(str(cache), "upscaler", 15)
    assert torch.equal(restored, images)
    assert len(list(cache.rglob("*.png"))) == 15
    assert not list(cache.rglob("*.pt"))


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_failed_png_write_preserves_existing_image_and_does_not_emit_done(monkeypatch, tmp_path, fail_at):
    monkeypatch.setattr(cg, "base_output_dir", lambda: str(tmp_path))
    cache = tmp_path / "cache"
    node = cg.VNCCS_CharacterGenerator()
    images = torch.ones(3, 4, 6, 4)
    node._save_upscaled_stage(str(cache), "upscaler", images)
    before = {path: path.read_bytes() for path in cache.rglob("*.png")}
    events = []
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: events.append(args))

    save = Image.Image.save
    writes = []

    def fail(image, *args, **kwargs):
        writes.append(args[0])
        if len(writes) == fail_at:
            raise OSError("disk full")
        return save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        node._save_upscaled_stage(str(cache), "upscaler", images * 0)
    assert {path: path.read_bytes() for path in cache.rglob("*.png")} == before
    assert not events
    assert not list(cache.rglob("*.tmp"))


def test_failed_batch_publication_restores_all_upscaled_images(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "base_output_dir", lambda: str(tmp_path))
    cache = tmp_path / "cache"
    target = cache / "upscaler"
    node = cg.VNCCS_CharacterGenerator()
    images = torch.ones(3, 4, 6, 4)
    node._save_upscaled_stage(str(cache), "upscaler", images)
    before = {path: path.read_bytes() for path in target.glob("*.png")}
    replace = utils.os.replace

    def fail(source, destination):
        if Path(source).name.startswith(".vnccs-sprites-") and Path(destination) == target:
            raise PermissionError("Windows file is busy")
        return replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", fail)
    with pytest.raises(PermissionError, match="Windows file is busy"):
        node._save_upscaled_stage(str(cache), "upscaler", images * 0)
    assert {path: path.read_bytes() for path in target.glob("*.png")} == before
    assert torch.equal(cg._load_upscaled_images(str(cache), "upscaler", 3), images)
    assert list(cache.iterdir()) == [target]


@pytest.mark.parametrize("stage", ["bg_remove", "upscaler"])
def test_legacy_png_locations_remain_usable_without_legacy_tensors(generated, stage):
    g = generated
    # Simulate the PNG layout from installations preceding batch publication.
    for path in (g.cache / g.stage).glob("*.png"):
        path.rename(g.cache / path.name)
    (g.cache / g.stage).rmdir()
    before = cg._load_upscaled_images(str(g.cache), g.stage, 3)
    g.calls.clear()
    g.state.update(regenerate_from=g.prefix + stage, regenerate_index=2)
    g.run()
    assert torch.equal(cg._load_upscaled_images(str(g.cache), g.stage, 3), before)
    assert [count for name, count in g.calls if name == g.stage] == ([1] if stage == "upscaler" else [])
    assert (g.cache / g.stage).exists() == (stage == "upscaler")


def test_normal_run_rotates_png_batches_and_recomputes_them(generated):
    g = generated
    before = {path.name: path.read_bytes() for path in (g.cache / g.stage).glob("*.png")}
    g.change[0] = 0.1
    g.calls.clear()
    g.run()
    assert [count for name, count in g.calls if name == g.stage] == [3]
    assert {path.name: path.read_bytes() for path in (g.cache / "V1" / g.stage).glob("*.png")} == before
    assert all(path.read_bytes() != before[path.name] for path in (g.cache / g.stage).glob("*.png"))
    assert not list(g.cache.rglob("*.pt"))


@pytest.mark.parametrize("generated", ["clothes", "naked"], indirect=True)
@pytest.mark.parametrize("index", [None, 2])
def test_source_stage_regeneration_recomputes_only_its_downstream_branch(generated, index):
    g = generated
    originals = {path: path.read_bytes() for path in (g.cache / "original_upscaler").glob("*.png")}
    source = "source_upscaler" if g.mode == "clothes" else "remove_clothes"
    g.calls.clear()
    g.state.update(regenerate_from=source, regenerate_index=index)
    g.run()
    assert [count for name, count in g.calls if name == g.stage] == [3 if index is None else 1]
    assert [count for name, count in g.calls if name == g.prefix + "pose_generation"] == [3 if index is None else 1]
    if g.mode == "clothes":
        assert ("source_upscaler", 1) in g.calls
    else:
        assert not any(name.startswith("original_") for name, _ in g.calls)
        assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize("background", ["Native", "disabled", "balanced"])
def test_upscaler_off_and_background_modes_use_pngs_without_model_work(generated, monkeypatch, background):
    g = generated
    g.state["upscaler"]["mode"] = "off"
    g.state["bg_remove"]["preset"] = background
    seen = []

    def chroma(self, images, *args, **kwargs):
        seen.append(images.clone())
        return (images,)

    monkeypatch.setattr(cg.VNCCSChromaKey, "chroma_key", chroma)
    g.calls.clear()
    g.run()
    before = cg._load_upscaled_images(str(g.cache), g.stage, 3)
    g.calls.clear()
    seen.clear()
    g.state.update(regenerate_from=g.prefix + "bg_remove", regenerate_index=None)
    g.run()
    assert not g.calls
    assert torch.equal(cg._load_upscaled_images(str(g.cache), g.stage, 3), before)
    if background == "balanced":
        assert torch.equal(seen[-1], before)
    else:
        assert not seen


def test_regenerate_api_creates_a_new_node_and_rejects_context_loss(generated, monkeypatch):
    g = generated
    # The HTTP handler creates a fresh node; only inference is replaced here.
    for name in ("_find_pose_lora", "_find_clothes_lora", "_extract_pipe", "_run_remove_clothes",
                 "_run_pose_generation", "_run_upscaler_models", "_run_seedvr_upscale_batch"):
        callback = getattr(g.node, name)
        monkeypatch.setattr(type(g.node), name, lambda self, *args, _call=callback, **kwargs: _call(*args, **kwargs))
    monkeypatch.setattr(cg.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    request = {"unique_id": "png-test", "stage": g.prefix + "bg_remove", "image_index": 2,
               "widget_data": g.state}
    g.calls.clear()
    response = cg._regenerate_response(request)
    assert response.status == 200, response.data
    assert not g.calls
    assert g.pipe.seed_int == 0
    snapshot = cg.progress_snapshot(g.state["ui"]["progress_scope"])
    assert snapshot["stages"][g.stage]["status"] == "done"
    assert all(".png" in url for url in snapshot["stages"][g.stage]["images"])
    before = {path: path.read_bytes() for path in g.cache.rglob("*.png")}
    cg._forget_generator_context("png-test", g.state["ui"]["progress_scope"])
    response = cg._regenerate_response(request)
    assert response.status == 409
    assert "live generator context" in response.data["error"]
    assert {path: path.read_bytes() for path in g.cache.rglob("*.png")} == before


def test_same_node_id_in_another_workflow_does_not_replace_upscale_inputs(generated):
    g = generated
    first_scope = g.state["ui"]["progress_scope"]
    first = cg._load_upscaled_images(str(g.cache), g.stage, 3)
    g.state["ui"]["progress_scope"] = "another-workflow:png-test"
    g.change[0] = 0.1
    g.run()
    other = Path(cg._get_generator_context("png-test", "another-workflow:png-test")["cache_dir"])
    assert other != g.cache
    assert not torch.equal(cg._load_upscaled_images(str(other), g.stage, 3), first)
    g.state["ui"]["progress_scope"] = first_scope
    g.state.update(regenerate_from=g.prefix + "bg_remove")
    g.calls.clear()
    g.run()
    assert not g.calls
    assert torch.equal(cg._load_upscaled_images(str(g.cache), g.stage, 3), first)


@pytest.mark.parametrize("generated", ["naked"], indirect=True)
@pytest.mark.parametrize("stage", ["original_pose_generation", "original_upscaler", "original_bg_remove"])
@pytest.mark.parametrize("index", [None, 0, 2])
def test_clone_original_regeneration_updates_both_branches_without_losing_other_sprites(generated, stage, index):
    g = generated
    before = {path: path.read_bytes() for name in ("original_upscaler", "naked_upscaler")
              for path in (g.cache / name).glob("*.png")}
    g.calls.clear()
    g.change[0] = 0.1
    g.state.update(regenerate_from=stage, regenerate_index=index)
    g.run()
    assert [count for name, count in g.calls if name == "naked_upscaler"] == [3 if index is None else 1]
    assert [count for name, count in g.calls if name == "original_upscaler"] == (
        [] if stage == "original_bg_remove" else [3 if index is None else 1])
    for path, content in before.items():
        selected = index is None or int(path.stem.rsplit("_", 1)[1]) == index + 1
        changed = selected and (path.parent.name == "naked_upscaler" or stage != "original_bg_remove")
        assert (path.read_bytes() != content) == changed
