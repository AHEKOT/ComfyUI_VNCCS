import os

from PIL import Image, ImageDraw

from nodes import migration_assistant as ma


def test_safe_legacy_name_removes_disallowed_characters():
    assert ma._safe_legacy_name("Alice.2<script>") == "Alice_2_script"


def test_scan_legacy_characters_reports_missing_sprite_targets(tmp_path, monkeypatch):
    legacy_root = tmp_path / "old"
    new_root = tmp_path / "new"
    sheet_dir = legacy_root / "Alice.2" / "Sheets" / "Naked" / "neutral"
    sheet_dir.mkdir(parents=True)
    Image.new("RGB", (256, 256), "green").save(sheet_dir / "sheet_neutral_0001.png")

    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy_root))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(new_root))

    result = ma.scan_legacy_characters()
    assert result["characters"][0]["legacy_name"] == "Alice.2"
    assert result["characters"][0]["new_name"] == "Alice_2"
    assert result["characters"][0]["sheet_count"] == 1
    assert result["characters"][0]["missing_sprite_targets"] == 1


def test_migrate_character_crops_sheet_and_writes_alpha_sprite(tmp_path, monkeypatch):
    legacy_root = tmp_path / "old"
    new_root = tmp_path / "new"
    sheet_dir = legacy_root / "Alice" / "Sheets" / "Naked" / "neutral"
    sheet_dir.mkdir(parents=True)

    sheet = Image.new("RGB", (512, 256), (0, 255, 0))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((40, 30, 180, 230), fill=(220, 40, 80))
    sheet.save(sheet_dir / "sheet_neutral_0001.png")

    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy_root))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(new_root))

    run = {"log": []}
    result = ma._migrate_character(run, "Alice", "Alice", force=False)

    sprite_dir = new_root / "Alice" / "Sprites" / "Naked" / "neutral"
    sprites = list(sprite_dir.glob("sprite_neutral_*.png"))
    assert result["sprites_saved"] >= 1
    assert sprites
    assert Image.open(sprites[0]).mode == "RGBA"
