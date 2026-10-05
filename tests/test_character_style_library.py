"""The style library and persistence contract work without model dependencies."""

import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from conftest import _preload_node


library = _preload_node("character_styles")
ROOT = Path(__file__).parents[1]


def test_packaged_catalog_is_unique_and_every_style_has_a_square_rgb_preview():
    catalog = json.loads((ROOT / "character_template/character_styles.json").read_text())
    styles = [style for group in catalog["groups"] for style in group["styles"]]
    assert len(styles) == len({style["id"] for style in styles}) == 358
    assert len({style["label"].casefold() for style in styles}) == len(styles)
    assert all(style["label"] and style["description"] and style["reference"] and style["prompt"] for style in styles)
    assert all(not style["id"].startswith("clio_") for style in styles)
    assert all("clio" not in group["label"].lower() and "photography" not in group["label"].lower() for group in catalog["groups"])
    assert catalog["aliases"]["clio_toon_shader"] == "cel_shading"
    assert catalog["aliases"]["clio_photography"] == "photorealism"
    directory = ROOT / "character_template/style_previews"
    assert {p.stem for p in directory.glob("*.webp") if not p.stem.startswith("user_") and p.stem != "custom"} == {style["id"] for style in styles}
    for style in styles:
        with Image.open(directory / (style["id"] + ".webp")) as image:
            assert image.format == "WEBP" and image.size == (1024, 1024)
            assert image.mode == "RGB"
    assert catalog["default_style"] == "ghibli_miyazaki"


def test_retired_styles_and_previews_are_removed_with_valid_workflow_fallbacks():
    catalog = json.loads((ROOT / "character_template/character_styles.json").read_text())
    styles = {s["id"] for g in catalog["groups"] for s in g["styles"]}
    retired = {
        "stick_figure_child_drawing", "troll_face", "zelda_wind_waker_style",
        "breath_of_the_wild_style", "isometric_3d_graphic", "blueprint",
        "blacklight_style", "x_ray_style",
    }
    assert styles.isdisjoint(retired)
    assert set(catalog["aliases"].values()) <= styles
    for style_id in retired:
        assert not (ROOT / "character_template/style_previews" / (style_id + ".webp")).exists()
        assert catalog["aliases"][style_id] == catalog["default_style"]
        assert catalog["aliases"]["clio_" + style_id] == catalog["default_style"]


def test_steampunk_is_surface_treatment_and_preserves_the_supplied_costume():
    catalog = json.loads((ROOT / "character_template/character_styles.json").read_text())
    style = next(s for g in catalog["groups"] for s in g["styles"] if s["id"] == "steampunk")
    assert "Victorian engraving" in style["prompt"]
    assert "crosshatching" in style["prompt"]
    assert "existing surfaces" in style["prompt"]
    assert "Do not add or redesign clothing, accessories, mechanical parts or decorative objects" in style["prompt"]
    for term in ("gear-and-cog", "riveted", "ornamental complexity", "mechanical-design"):
        assert term not in style["prompt"] + style["description"] + style["reference"]
    assert (ROOT / "character_template/style_previews/steampunk.webp").is_file()


def test_style_prose_keeps_rendering_and_cannot_set_background_pose_or_camera():
    import re
    catalog = json.loads((ROOT / "character_template/character_styles.json").read_text())
    forbidden = re.compile(r"background|backdrop|environment|\bposes?\b|\bposed\b|posing|framing|camera|foreshorten|\bfov\b|depth.of.field|(?:low|high).angle|head.to.toe|full.body|front.facing|three.quarter|surroundings|\b(?:sky|skies)\b|\bposture\b|orthographic.projection|deep.focus.shot|\bisometric\b|vignette.framed|\bhorizon\b|bedroom|landscape", re.I)
    for group in catalog["groups"]:
        for style in group["styles"]:
            assert not forbidden.search(style["prompt"]), style["id"]
            assert "preserve all specified character details" in style["prompt"]
    styles = {s["id"]: s for g in catalog["groups"] for s in g["styles"]}
    assert "calligraphic brushwork" in styles["chinese_ink_drawing"]["prompt"]
    assert "Lorenzo Ghiberti" in styles["low_relief_sculpture"]["prompt"]
    assert "Square Enix HD-2D" in styles["octopath_traveler_hd_2d_style"]["prompt"]
    assert "watercolor" in styles["beatrix_potter_style"]["prompt"]


def test_save_edit_restart_and_packaged_updates_preserve_user_file(tmp_path):
    path = tmp_path / "character_styles.user.json"
    packaged = ROOT / "character_template/character_styles.json"
    original = packaged.read_bytes()
    assert library.load_user_styles(path) == []
    style = library.save_user_style({"label": "My sketch", "prompt": "fine graphite lines", "reference": "Personal study"}, path)
    assert library.load_user_styles(path) == [style]
    updated = library.save_user_style({**style, "prompt": "charcoal lines"}, path)
    assert library.load_user_styles(path) == [updated]
    assert updated["id"] == style["id"]
    assert packaged.read_bytes() == original
    assert "character_template/character_styles.user.json" in (ROOT / ".gitignore").read_text()


@pytest.mark.parametrize("payload", [[], {}, {"label": "Name", "prompt": ""},
    {"label": "N", "prompt": "P", "id": "../../escape"},
    {"label": "N", "prompt": "P", "id": "ghibli_miyazaki"},
    {"label": "N", "prompt": "P", "description": []},
    {"label": "N", "prompt": "P" * 16001}, {"label": "N\x00", "prompt": "P"}])
def test_invalid_style_never_changes_existing_file(tmp_path, payload):
    path = tmp_path / "character_styles.user.json"
    library.save_user_style({"label": "Kept", "prompt": "Kept"}, path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        library.save_user_style(payload, path)
    assert path.read_bytes() == original


def test_corrupt_library_is_preserved_instead_of_reset(tmp_path):
    path = tmp_path / "character_styles.user.json"
    path.write_text("{broken")
    with pytest.raises(ValueError):
        library.save_user_style({"label": "N", "prompt": "P"}, path)
    assert path.read_text() == "{broken"


def test_concurrent_creators_do_not_lose_styles(tmp_path):
    path = tmp_path / "character_styles.user.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = list(pool.map(lambda i: library.save_user_style({"label": f"Style {i}", "prompt": f"Prompt {i}"}, path), range(12)))
    assert {style["id"] for style in library.load_user_styles(path)} == {style["id"] for style in saved}


@pytest.mark.parametrize("scale, side", [(1024, 1024), (1344, 1168), (1536, 1248), (4096, 2048)])
def test_square_preview_resolution_preserves_creator_pixel_budget(scale, side):
    assert library.square_style_resolution(scale) == (side, side)
    assert side % 16 == 0
    assert abs(side * side - scale * 1024) / (scale * 1024) < .02


def test_webp_is_published_immediately_and_survives_failed_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    image = Image.new("RGB", (96, 96), "pink")
    first = library.save_style_preview("anime_style", image)
    path = Path(library.style_preview_path("anime_style"))
    assert path.exists()
    with Image.open(path) as stored:
        assert stored.format == "WEBP" and stored.size == (96, 96)
    assert first["image"].startswith("/vnccs/character_styles/preview?style=anime_style&v=")
    assert first["width"] == first["height"] == 96
    assert first["saved"] is True and first["path"] == str(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="square"):
        library.save_style_preview("anime_style", Image.new("RGB", (96, 128)))
    assert path.read_bytes() == original
    second = library.save_style_preview("anime_style", Image.new("RGB", (96, 96), "blue"))
    assert second["image"] != first["image"]
    assert list(path.parent.glob("*.png")) == []
    assert list(path.parent.glob("*.tmp")) == []


def test_previews_survive_fresh_backend_load_inside_installed_node(monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    module_path = tmp_path / "installed_node" / "nodes" / "character_styles.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(Path(library.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("_vnccs.nodes.character_styles_restart", module_path)
    first = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(first)
    saved = first.save_style_preview("anime_style", Image.new("RGBA", (32, 32), (255, 0, 0, 100)))
    path = tmp_path / "installed_node" / "character_template" / "style_previews" / "anime_style.webp"
    assert saved["path"] == str(path)
    assert not (tmp_path / "output").exists()
    restarted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restarted)
    assert restarted.style_preview_url("anime_style") == saved["image"]
    with Image.open(restarted.style_preview_path("anime_style")) as stored:
        stored.load()
        assert stored.format == "WEBP" and stored.size == (32, 32)
        assert stored.mode == "RGB"
        assert all(abs(a - b) <= 5 for a, b in zip(stored.getpixel((16, 16)), (119, 15, 25)))


def test_large_custom_preview_is_capped_and_composited_after_resize(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "character_template" / "style_previews"))
    image = Image.new("RGBA", (1456, 1456), (120, 80, 160, 255))
    alpha = Image.new("L", image.size)
    alpha.putdata([x * 255 // 1455 for y in range(1456) for x in range(1456)])
    image.putalpha(alpha)
    source = image.tobytes()
    received = []
    flatten = library.flatten_style_preview
    def capture(resized):
        received.append(resized.size)
        return flatten(resized)
    monkeypatch.setattr(library, "flatten_style_preview", capture)
    result = library.save_style_preview("user_" + "a" * 32, image)
    assert result["width"] == result["height"] == 1024
    with Image.open(result["path"]) as stored:
        assert stored.mode == "RGB"
    assert received == [(1024, 1024)]
    assert image.mode == "RGBA" and image.tobytes() == source


def test_invalid_encoded_file_never_reports_saved_or_replaces_previous_preview(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path))
    image = Image.new("RGB", (32, 32), "pink")
    library.save_style_preview("kept", image)
    path = Path(library.style_preview_path("kept"))
    original = path.read_bytes()
    monkeypatch.setattr(Image.Image, "save", lambda image, path, **kwargs: Path(path).write_bytes(b"broken"))
    with pytest.raises(OSError):
        library.save_style_preview("kept", image)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("style_id", ["../escape", "/escape", "a/b", "a\\b", "..", "a?b", "", None])
def test_style_preview_paths_reject_untrusted_ids(style_id):
    with pytest.raises(ValueError):
        library.style_preview_path(style_id)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_style_webp_flattens_alpha_onto_card_gradient_including_soft_edges(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path))
    if mode == "P":
        image = Image.new("P", (32, 32), 0)
        palette = [0] * 768
        palette[3:6] = [120, 80, 160]
        image.putpalette(palette)
        image.putpixel((31, 31), 1)
        image.info["transparency"] = 0
    else:
        image = Image.new(mode, (32, 32), (128, 255) if mode == "LA" else (120, 80, 160, 255))
        alpha = Image.new("L", image.size)
        alpha.putdata([x * 255 // 31 for y in range(32) for x in range(32)])
        image.putalpha(alpha)
    original = image.tobytes()
    flattened = library.flatten_style_preview(image)
    assert flattened.mode == "RGB"
    assert flattened.getpixel((0, 0)) == (41, 32, 52)
    expected_opaque = (128, 128, 128) if mode == "LA" else (120, 80, 160)
    assert flattened.getpixel((31, 31)) == expected_opaque
    if mode != "P":
        # The middle soft edge mixes the character with the dark card, not black.
        rgba = image.convert("RGBA").getpixel((16, 16))
        expected = tuple(round(color * rgba[3] / 255 + bg * (255 - rgba[3]) / 255)
                         for color, bg in zip(rgba[:3], (32, 25, 41)))
        assert flattened.getpixel((16, 16)) == expected
    library.save_style_preview("alpha_test", image)
    with Image.open(library.style_preview_path("alpha_test")) as stored:
        assert stored.mode == "RGB"
    assert image.tobytes() == original


def test_flattening_keeps_opaque_rgb_pixels_and_matches_card_colors():
    rgb = Image.new("RGB", (32, 32), (120, 80, 160))
    assert library.flatten_style_preview(rgb).tobytes() == rgb.tobytes()
    background = library.flatten_style_preview(Image.new("RGBA", (32, 32), (255, 0, 0, 0)))
    assert background.getpixel((0, 0)) == (41, 32, 52)
    assert background.getpixel((31, 31)) == (23, 19, 31)
    css = (ROOT / "web/vnccs_character_creator_v2.js").read_text()
    assert "background: linear-gradient(145deg, #292034, #17131f)" in css


@pytest.mark.parametrize("with_preview", [False, True])
def test_delete_user_style_removes_only_its_record_and_optional_preview(monkeypatch, tmp_path, with_preview):
    path = tmp_path / "character_styles.user.json"
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    kept = library.save_user_style({"label": "Kept", "prompt": "Graphite"}, path)
    removed = library.save_user_style({"label": "Removed", "prompt": "Ink"}, path)
    library.save_style_preview(kept["id"], Image.new("RGB", (32, 32), "pink"))
    kept_preview = Path(library.style_preview_path(kept["id"]))
    original_preview = kept_preview.read_bytes()
    packaged = (ROOT / "character_template/character_styles.json").read_bytes()
    target = Path(library.style_preview_path(removed["id"]))
    if with_preview:
        library.save_style_preview(removed["id"], Image.new("RGB", (32, 32), "blue"))
    assert library.delete_user_style(removed["id"], path) is True
    assert library.load_user_styles(path) == [kept]
    assert not target.exists()
    assert kept_preview.read_bytes() == original_preview
    assert (ROOT / "character_template/character_styles.json").read_bytes() == packaged
    assert library.delete_user_style(removed["id"], path) is False


@pytest.mark.parametrize("style_id", ["photorealism", "clio_anime_style", "custom", "../escape", "user_bad", None])
def test_delete_rejects_packaged_and_untrusted_style_ids_before_writing(tmp_path, style_id):
    path = tmp_path / "character_styles.user.json"
    library.save_user_style({"label": "Kept", "prompt": "Ink"}, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Only user styles"):
        library.delete_user_style(style_id, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["library_publish", "preview_remove"])
def test_failed_deletion_preserves_style_and_preview(monkeypatch, tmp_path, failure):
    path = tmp_path / "character_styles.user.json"
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    style = library.save_user_style({"label": "Kept", "prompt": "Ink"}, path)
    library.save_style_preview(style["id"], Image.new("RGB", (32, 32), "pink"))
    preview = Path(library.style_preview_path(style["id"]))
    original_library, original_preview = path.read_bytes(), preview.read_bytes()
    operation = "replace" if failure == "library_publish" else "unlink"
    original_operation = getattr(library.os, operation)
    def fail_for_target(*args):
        target = args[1] if operation == "replace" else args[0]
        if Path(target) == (path if operation == "replace" else preview):
            raise PermissionError("Read only")
        return original_operation(*args)
    monkeypatch.setattr(library.os, operation, fail_for_target)
    with pytest.raises(PermissionError):
        library.delete_user_style(style["id"], path)
    assert path.read_bytes() == original_library
    assert preview.read_bytes() == original_preview
    assert list(tmp_path.rglob("*.tmp")) == []


def test_deleted_style_cannot_publish_a_late_render(monkeypatch, tmp_path):
    path = tmp_path / "character_styles.user.json"
    monkeypatch.setattr(library, "USER_CHARACTER_STYLES_PATH", str(path))
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    style = library.save_user_style({"label": "Mine", "prompt": "Ink"})
    image = Image.new("RGB", (32, 32), "pink")
    saved = library.save_user_style_preview(style["id"], image)
    assert Path(saved["path"]).exists()
    assert library.delete_user_style(style["id"]) is True
    with pytest.raises(ValueError, match="no longer exists"):
        library.save_user_style_preview(style["id"], image)
    assert not Path(saved["path"]).exists()
    with pytest.raises(ValueError, match="no longer exists"):
        library.save_user_style({**style, "prompt": "Recreated"})


def test_deletion_preserves_a_corrupt_library_and_linked_preview(monkeypatch, tmp_path):
    path = tmp_path / "character_styles.user.json"
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    style = library.save_user_style({"label": "Mine", "prompt": "Ink"}, path)
    target = Path(library.style_preview_path(style["id"]))
    target.parent.mkdir()
    protected = target.parent / "photorealism.webp"
    protected.write_bytes(b"kept")
    target.symlink_to(protected)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="linked"):
        library.delete_user_style(style["id"], path)
    assert target.is_symlink() and protected.read_bytes() == b"kept"
    assert path.read_bytes() == original
    path.write_text("{broken")
    with pytest.raises(ValueError):
        library.delete_user_style(style["id"], path)
    assert path.read_text() == "{broken"
    assert target.is_symlink()
