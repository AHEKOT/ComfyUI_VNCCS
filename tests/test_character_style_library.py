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


def test_complete_clio_catalog_and_legacy_styles_have_display_metadata():
    catalog = json.loads((ROOT / "character_template/character_styles.json").read_text())
    styles = [style for group in catalog["groups"] for style in group["styles"]]
    imported = [style for style in styles if style["id"].startswith("clio_")]
    assert len(imported) == 414
    assert len(styles) == len({style["id"] for style in styles}) == 454
    assert all(style["label"] and style["description"] and style["reference"] and style["prompt"] for style in styles)
    assert all(style["image"] == "" for style in styles)
    assert {"Anime Style", "Ghibli Style", "Manga Style", "Sailor Moon Style"}.issubset({style["label"] for style in imported})
    assert catalog["default_style"] == "ghibli_miyazaki"


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
    first = library.save_style_preview("clio_anime_style", image)
    path = Path(library.style_preview_path("clio_anime_style"))
    assert path.exists()
    with Image.open(path) as stored:
        assert stored.format == "WEBP" and stored.size == (96, 96)
    assert first["image"].startswith("/vnccs/character_styles/preview?style=clio_anime_style&v=")
    assert first["width"] == first["height"] == 96
    assert first["saved"] is True and first["path"] == str(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="square"):
        library.save_style_preview("clio_anime_style", Image.new("RGB", (96, 128)))
    assert path.read_bytes() == original
    second = library.save_style_preview("clio_anime_style", Image.new("RGB", (96, 96), "blue"))
    assert second["image"] != first["image"]
    assert list(path.parent.glob("*.png")) == []
    assert list(path.parent.glob("*.tmp")) == []


def test_previews_survive_fresh_backend_load_outside_installed_node(monkeypatch, tmp_path):
    import folder_paths
    output = tmp_path / "ComfyUI" / "output"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    spec = importlib.util.spec_from_file_location("_vnccs.nodes.character_styles_restart", library.__file__)
    first = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(first)
    saved = first.save_style_preview("clio_anime_style", Image.new("RGBA", (32, 32), (255, 0, 0, 100)))
    path = output / "VNCCS" / "style_previews" / "clio_anime_style.webp"
    assert saved["path"] == str(path)
    restarted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restarted)
    assert restarted.style_preview_url("clio_anime_style") == saved["image"]
    with Image.open(restarted.style_preview_path("clio_anime_style")) as stored:
        stored.load()
        assert stored.format == "WEBP" and stored.size == (32, 32)
        assert stored.getchannel("A").getextrema() == (100, 100)


def test_legacy_preview_recovery_preserves_source_and_survives_its_removal(monkeypatch, tmp_path):
    legacy = tmp_path / "installed_node" / "character_template" / "style_previews"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "output" / "VNCCS" / "style_previews"))
    monkeypatch.setattr(library, "LEGACY_STYLE_PREVIEWS_DIR", str(legacy))
    source = legacy / "clio_anime_style.webp"
    Image.new("RGBA", (32, 32), (255, 0, 0, 100)).save(source, format="WEBP")
    original = source.read_bytes()
    url = library.style_preview_url("clio_anime_style")
    assert source.read_bytes() == original
    assert Path(library.style_preview_path("clio_anime_style")).is_file()
    source.unlink()
    assert library.style_preview_url("clio_anime_style") == url
    (legacy / "broken.webp").write_bytes(b"not an image")
    assert library.style_preview_url("broken") == ""


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
def test_style_webp_preserves_transparency_including_soft_edges(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path))
    if mode == "P":
        image = Image.new("P", (32, 32), 0)
        image.putpixel((16, 16), 1)
        image.info["transparency"] = 0
    else:
        image = Image.new(mode, (32, 32), (128, 255) if mode == "LA" else (120, 80, 160, 255))
        alpha = Image.new("L", image.size)
        alpha.putdata([x * 255 // 31 for y in range(32) for x in range(32)])
        image.putalpha(alpha)
    expected_alpha = image.convert("RGBA").getchannel("A").tobytes()
    library.save_style_preview("alpha_test", image)
    with Image.open(library.style_preview_path("alpha_test")) as stored:
        assert stored.mode == "RGBA"
        assert stored.getchannel("A").tobytes() == expected_alpha
