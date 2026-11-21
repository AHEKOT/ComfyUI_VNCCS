import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

import folder_paths
from nodes import additional_poses as poses
import utils


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    config = Path(utils.config_path("Ada"))
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"character_info": {"background_color": "Blue"}}))
    costume = config.parent / "Sprites" / "Coat"
    for name, color in (
        ("Happy/sprite_Happy_0002.png", (1, 2, 3, 42)),
        ("Happy/sprite_Happy_0007.png", (4, 5, 6, 137)),
        ("V1/old.png", (1, 2, 3, 255)),
        ("old.png", (1, 2, 3, 255)),
    ):
        path = costume / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 12), color).save(path)
    return costume, config


def test_catalog_reads_current_sprites_and_legacy_neutral(library):
    costume, _ = library
    (costume.parents[2] / "Unconfigured").mkdir()
    assert poses.catalog() == {"characters": ["Ada"]}
    assert "Coat" in poses.catalog("Ada")["costumes"]
    assert poses.catalog("Ada", "Coat")["emotions"] == {
        "Neutral": [costume / "old.png"],
        "Happy": [costume / "Happy" / f"sprite_Happy_{index:04}.png" for index in (2, 7)],
    }
    assert poses.destination(utils.sheets_dir("Ada", "Coat", "Happy")) == ("Ada", "Coat", "Happy")
    with pytest.raises(ValueError):
        poses.destination(str(costume / "Happy" / "sheet_neutral"))


def test_selector_uses_saved_background_alpha_and_stable_sprite(library):
    pytest.importorskip("torch")
    costume, config = library
    node = poses.VNCCS_AdditionalPoses()
    state = {"character": "Ada", "costume": "Coat", "emotion": "Happy"}
    image, destination, background = node.process(json.dumps(state))
    assert image.shape == (1, 12, 8, 4)
    assert image[0, 0, 0, 3].item() == pytest.approx(42 / 255)
    assert (destination, background) == (utils.sheets_dir("Ada", "Coat", "Happy"), "Blue")

    state["sprite"] = "sprite_Happy_0007.png"
    raw = json.dumps(state)
    before = node.IS_CHANGED(raw)
    config.write_text(json.dumps({"character_info": {"background_color": "Transparent"}}))
    image, _, background = node.process(raw)
    assert image[0, 0, 0, 3].item() == pytest.approx(137 / 255)
    assert background == "Alpha"
    assert node.IS_CHANGED(raw) != before
    (costume / "Happy" / state["sprite"]).unlink()
    with pytest.raises(FileNotFoundError, match=state["sprite"]):
        node.process(raw)


def test_generator_process_routes_output_and_honors_saved_background(library, monkeypatch):
    torch = pytest.importorskip("torch")
    from nodes.character_generator import VNCCS_AdditionalPosesGenerator

    costume, _ = library
    original = costume.parent / "Original" / "Neutral" / "sprite_pose_0001.png"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original sprite")
    node = VNCCS_AdditionalPosesGenerator()
    image = torch.zeros(2, 12, 8, 4)
    seen = []

    def generate(poses, character, pipe, prompt, settings, **kwargs):
        seen.append((kwargs["background"], kwargs["bg_remove_settings"]["preset"]))
        return image

    monkeypatch.setattr(node, "_extract_pipe", lambda pipe: {"model_kind": "qi2", "seed": 1})
    monkeypatch.setattr(node, "_find_pose_lora", lambda pipe: {})
    monkeypatch.setattr(node, "_run_pose_generation", generate)
    monkeypatch.setattr(node, "_run_upscaler", lambda images, *args, **kwargs: images)
    monkeypatch.setattr(node, "_run_bg_remove", lambda images, *args, **kwargs: images)
    path = utils.sheets_dir("Ada", "Coat", "Happy")
    settings = json.dumps({"character_name": "Other", "bg_remove": {"preset": "Native"}})
    for background in ("Blue", "Alpha"):
        result = node.process([image], [image[:1]], [object()], ["New pose"], [background], [settings], [path])
        assert result[0] is image
    assert seen == [("Blue", "balanced"), ("Alpha", "Native")]
    assert [path.name for path in sorted((costume / "Happy").glob("*.png"))] == [
        "sprite_Happy_0002.png", "sprite_Happy_0007.png",
        "sprite_pose_0008.png", "sprite_pose_0009.png", "sprite_pose_0010.png", "sprite_pose_0011.png",
    ]
    assert original.read_bytes() == b"original sprite"


def test_append_is_atomic_and_allocates_under_the_character_lock(library, monkeypatch):
    torch = pytest.importorskip("torch")
    from nodes.character_generator import VNCCS_AdditionalPosesGenerator

    costume, _ = library
    path = utils.sheets_dir("Ada", "Coat", "Neutral")
    node = VNCCS_AdditionalPosesGenerator()
    batch = torch.zeros(2, 8, 8, 4)
    convert = node._tensor_item_to_pil
    calls = iter([True, False])

    def fail_second(image):
        if not next(calls):
            raise OSError("fixture write failure")
        return convert(image)

    with monkeypatch.context() as patch:
        patch.setattr(node, "_tensor_item_to_pil", fail_second)
        with pytest.raises(OSError, match="fixture write failure"):
            node._save_final_sprites(batch, path)
    assert not list(costume.glob("sprite_pose_*.png"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(node._save_final_sprites, batch, path)
        second = pool.submit(node._save_final_sprites, batch, path)
        assert len(set(first.result() + second.result())) == 4
    assert (costume / "old.png").exists()
    assert (costume / "V1" / "old.png").exists()
    assert (costume / "Happy" / "sprite_Happy_0007.png").exists()
