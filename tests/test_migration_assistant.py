import os
import json

from PIL import Image, ImageDraw

from nodes import migration_assistant as ma


def test_existing_character_config_is_preserved_even_with_force(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    (legacy / "Alice").mkdir(parents=True)
    (current / "Alice").mkdir(parents=True)
    (legacy / "Alice" / "Alice_config.json").write_text(json.dumps({"character_info": {"hair": "old"}}))
    target = current / "Alice" / "Alice_config.json"
    original = json.dumps({"character_info": {"hair": "new"}, "costumes": {"Dress": {"top": "silk"}}})
    target.write_text(original)
    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(current))
    for force in (False, True):
        result = ma._migrate_character({"log": []}, "Alice", "Alice", force)
        assert result["config_copied"] is False
        assert target.read_text() == original


def test_only_latest_legacy_sheet_per_costume_emotion_is_migrated(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    sheets = legacy / "Alice" / "Sheets" / "Naked" / "neutral"
    sheets.mkdir(parents=True)
    old = sheets / "sheet_0001.png"
    latest = sheets / "sheet_0002.png"
    Image.new("RGBA", (16, 32), "red").save(old)
    Image.new("RGBA", (16, 32), "blue").save(latest)
    os.utime(old, (100, 100))
    os.utime(latest, (200, 200))
    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(current))
    monkeypatch.setattr(ma, "_crop_sprites", lambda image: [image.convert("RGBA")])
    assert ma.scan_legacy_characters()["characters"][0]["sheet_count"] == 1
    result = ma._migrate_character({"log": []}, "Alice", "Alice", False)
    assert result["sprites_saved"] == 1
    target = current / "Alice" / "Sprites" / "Naked" / "neutral" / "sprite_neutral_0000.png"
    with Image.open(target) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255, 255)


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


def test_scan_legacy_character_uses_stable_target_when_already_migrated(tmp_path, monkeypatch):
    legacy_root = tmp_path / "old"
    new_root = tmp_path / "new"
    sheet_dir = legacy_root / "Alina_test" / "Sheets" / "Naked" / "neutral"
    sprite_dir = new_root / "Alina_test" / "Sprites" / "Naked" / "neutral"
    duplicate_dir = new_root / "Alina_test 2"
    sheet_dir.mkdir(parents=True)
    sprite_dir.mkdir(parents=True)
    duplicate_dir.mkdir(parents=True)
    Image.new("RGB", (256, 256), "green").save(sheet_dir / "sheet_neutral_0001.png")
    Image.new("RGBA", (128, 128), (255, 0, 0, 255)).save(sprite_dir / "sprite_neutral_0000.png")

    monkeypatch.setattr(ma, "get_legacy_output_dir", lambda: str(legacy_root))
    monkeypatch.setattr(ma, "base_output_dir", lambda: str(new_root))

    result = ma.scan_legacy_characters()
    item = result["characters"][0]
    assert item["new_name"] == "Alina_test"
    assert item["missing_sprite_targets"] == 0
    assert item["status"] == "migrated"


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
