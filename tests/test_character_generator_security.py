"""Security-focused tests for character generator path handling."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")

from nodes import character_generator as cg


def test_character_root_ignores_external_sheets_path(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    external = tmp_path / "elsewhere" / "Sheets" / "Bad"
    char_root.mkdir(parents=True)
    external.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._character_root_from_sheets_path(str(external), "Alice") == str(char_root)


def test_character_root_accepts_windows_style_sheets_path(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    sheets = char_root / "Sheets" / "Naked" / "neutral"
    sheets.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    windows_style = str(sheets).replace(os.sep, "\\")
    assert cg._character_root_from_sheets_path(windows_style, "Alice") == str(char_root)
    assert cg._costume_name_from_sheets_path(windows_style) == "Naked"


def test_cache_tensor_path_rejects_external_cache(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    outside = tmp_path / "outside" / "cache"
    outside.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._cache_tensor_path(str(outside), "stage") == ""


def test_emotion_output_prefix_must_stay_under_character_sprites(tmp_path, monkeypatch):
    base = tmp_path / "output" / "VNCCS" / "Characters"
    char_root = base / "Alice"
    safe_prefix = char_root / "Sprites" / "Happy" / "Neutral" / "sprite_"
    unsafe_prefix = tmp_path / "outside" / "sprite_"
    char_root.mkdir(parents=True)

    monkeypatch.setattr(cg, "base_output_dir", lambda: str(base))

    assert cg._safe_emotion_output_prefix(str(safe_prefix), "Alice") == str(safe_prefix)
    assert cg._safe_emotion_output_prefix(str(unsafe_prefix), "Alice") == ""
