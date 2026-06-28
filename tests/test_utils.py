"""Tests for utils.py — pure functions and filesystem I/O."""

import json
import os
import sys
import types
import tomllib

import pytest

# conftest.py stubs ComfyUI modules before any import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils as U


def _normalize_requirement_name(value):
    name = str(value).split("[", 1)[0]
    for marker in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
        name = name.split(marker, 1)[0]
    return name.strip().lower().replace("_", "-")


def test_pyproject_dependencies_cover_requirements():
    root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(root, "pyproject.toml"), "rb") as handle:
        pyproject = tomllib.load(handle)
    pyproject_deps = {
        _normalize_requirement_name(dep)
        for dep in pyproject["project"]["dependencies"]
    }
    with open(os.path.join(root, "requirements.txt"), "r", encoding="utf-8") as handle:
        requirement_deps = {
            _normalize_requirement_name(line.strip())
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }
    assert requirement_deps <= pyproject_deps


# ── generate_seed ─────────────────────────────────────────────────────────────

class TestGenerateSeed:
    def test_nonzero_returns_as_is(self):
        assert U.generate_seed(42) == 42

    def test_zero_returns_nonzero(self):
        seed = U.generate_seed(0)
        assert seed != 0

    def test_zero_returns_int(self):
        assert isinstance(U.generate_seed(0), int)

    def test_large_value_preserved(self):
        big = 0xDEADBEEFCAFEBABE
        assert U.generate_seed(big) == big


# ── inherit_seed ──────────────────────────────────────────────────────────────

class TestInheritSeed:
    def test_nonzero_input_wins(self):
        assert U.inherit_seed(99, 42) == 99

    def test_zero_input_inherits_upstream(self):
        assert U.inherit_seed(0, 777) == 777

    def test_both_zero_returns_generated(self):
        result = U.inherit_seed(0, 0)
        assert result != 0

    def test_zero_input_none_upstream_generates(self):
        result = U.inherit_seed(0, None)
        assert result != 0


# ── normalize_sex ─────────────────────────────────────────────────────────────

class TestNormalizeSex:
    @pytest.mark.parametrize("raw", ["male", "MALE", "Male", "man", "Man", "boy", "Boy", "m", "M"])
    def test_male_variants(self, raw):
        assert U.normalize_sex(raw) == "male"

    @pytest.mark.parametrize("raw", ["female", "FEMALE", "woman", "girl", "f", ""])
    def test_female_variants(self, raw):
        assert U.normalize_sex(raw) == "female"

    def test_none_returns_female(self):
        assert U.normalize_sex(None) == "female"

    def test_unknown_returns_female(self):
        assert U.normalize_sex("other") == "female"


# ── sex_positive_tokens ───────────────────────────────────────────────────────

class TestSexPositiveTokens:
    def test_female_default(self):
        assert "1girl" in U.sex_positive_tokens("female")

    def test_male_default(self):
        tokens = U.sex_positive_tokens("male")
        assert "1boy" in tokens
        assert "male_focus" in tokens

    def test_male_creator_mode(self):
        tokens = U.sex_positive_tokens("male", mode="creator")
        assert "1boy" in tokens
        assert any("male_focus" in t for t in tokens)


# ── sex_negative_tokens ───────────────────────────────────────────────────────

class TestSexNegativeTokens:
    def test_female_excludes_male_tokens(self):
        tokens = U.sex_negative_tokens("female")
        assert "1boy" in tokens
        assert "penis" in tokens

    def test_male_excludes_female_tokens(self):
        tokens = U.sex_negative_tokens("male")
        assert "1girl" in tokens
        assert "breasts" in tokens

    def test_male_creator_mode_longer(self):
        default = U.sex_negative_tokens("male", mode="default")
        creator = U.sex_negative_tokens("male", mode="creator")
        assert len(creator) > len(default)


# ── apply_sex ─────────────────────────────────────────────────────────────────

class TestApplySex:
    def test_female_adds_1girl(self):
        pos, neg = U.apply_sex("female", "base", "")
        assert "1girl" in pos

    def test_male_adds_1boy(self):
        pos, neg = U.apply_sex("male", "base", "")
        assert "1boy" in pos

    def test_male_neg_uses_quadruple_parens(self):
        _, neg = U.apply_sex("male", "base", "")
        assert "(((" in neg

    def test_female_neg_no_quadruple_parens(self):
        _, neg = U.apply_sex("female", "base", "")
        assert "(((" not in neg

    def test_normalizes_sex_before_applying(self):
        pos, _ = U.apply_sex("MALE", "base", "")
        assert "1boy" in pos


# ── age_strength ──────────────────────────────────────────────────────────────

class TestAgeStrength:
    def test_exact_control_point(self):
        assert U.age_strength(18) == 2.0

    def test_below_minimum_clamps(self):
        assert U.age_strength(-5) == U.AGE_CONTROL_POINTS[0][1]

    def test_above_maximum_clamps(self):
        assert U.age_strength(999) == U.AGE_CONTROL_POINTS[-1][1]

    def test_interpolates_between_points(self):
        # Between age 18 (2.0) and age 30 (2.5): midpoint 24 → 2.25
        result = U.age_strength(24)
        assert 2.0 < result < 2.5

    def test_invalid_type_defaults_to_18(self):
        assert U.age_strength("not_a_number") == U.age_strength(18)

    def test_returns_float(self):
        assert isinstance(U.age_strength(25), float)


# ── age_body_descriptor ───────────────────────────────────────────────────────

class TestAgeBodyDescriptor:
    @pytest.mark.parametrize("age,sex,expected", [
        (2,  "female", "(toddler girl:1.0)"),
        (8,  "female", "(loli:1.0)"),
        (15, "female", "(teenager girl:1.0)"),
        (20, "female", "(young_adult woman:1.0)"),
        (35, "female", "(adult woman:1.0)"),
        (55, "female", "(old woman:1.0)"),
        (2,  "male",   "(toddler boy:1.0)"),
        (8,  "male",   "(shota:1.0)"),
        (14, "male",   "(teenager boy:1.0)"),
        (17, "male",   "(young_adult man:1.0)"),
        (22, "male",   "(young_adult man:1.5)"),
        (40, "male",   "(adult man:1.0)"),
        (56, "male",   "(old man:1.0)"),
    ])
    def test_descriptor(self, age, sex, expected):
        assert U.age_body_descriptor(age, sex) == expected

    def test_very_old_returns_empty(self):
        assert U.age_body_descriptor(90, "female") == ""
        assert U.age_body_descriptor(90, "male") == ""

    def test_invalid_type_returns_empty(self):
        assert U.age_body_descriptor("nope", "female") == ""


# ── append_age ────────────────────────────────────────────────────────────────

class TestAppendAge:
    def test_appends_yo(self):
        result = U.append_age("base", 25, "female")
        assert "25yo" in result

    def test_appends_body_descriptor(self):
        result = U.append_age("base", 5, "female")
        assert "loli" in result

    def test_invalid_age_defaults_to_18(self):
        result = U.append_age("base", "bad", "female")
        assert "18yo" in result


# ── build_face_details ────────────────────────────────────────────────────────

class TestBuildFaceDetails:
    def test_includes_gender_token(self):
        result = U.build_face_details({"sex": "female"})
        assert "1girl" in result

    def test_male_token(self):
        result = U.build_face_details({"sex": "male"})
        assert "1boy" in result

    def test_race_appended(self):
        result = U.build_face_details({"sex": "female", "race": "elf"})
        assert "elf race" in result

    def test_eyes_appended(self):
        result = U.build_face_details({"sex": "female", "eyes": "blue"})
        assert "blue eyes" in result

    def test_hair_appended(self):
        result = U.build_face_details({"sex": "female", "hair": "long black"})
        assert "long black hair" in result

    def test_skin_color_appended(self):
        result = U.build_face_details({"sex": "female", "skin_color": "dark"})
        assert "dark skin" in result

    def test_empty_fields_omitted(self):
        result = U.build_face_details({"sex": "female", "race": "", "eyes": ""})
        assert "race" not in result
        assert "eyes" not in result

    def test_uses_gender_field_as_fallback(self):
        result = U.build_face_details({"gender": "male"})
        assert "1boy" in result


# ── dedupe_tokens ─────────────────────────────────────────────────────────────

class TestDedupeTokens:
    def test_removes_duplicates(self):
        result = U.dedupe_tokens("a,b,a,c")
        assert result.count("a") == 1

    def test_preserves_order(self):
        result = U.dedupe_tokens("c,a,b")
        assert result == "c,a,b"

    def test_strips_whitespace(self):
        result = U.dedupe_tokens("a , b , a")
        assert result == "a,b"

    def test_empty_string_returns_empty(self):
        result = U.dedupe_tokens("")
        assert result == ""

    def test_none_returns_none(self):
        assert U.dedupe_tokens(None) is None

    def test_no_dupes_unchanged(self):
        result = U.dedupe_tokens("x,y,z")
        assert result == "x,y,z"


# ── path helpers ──────────────────────────────────────────────────────────────

class TestPathHelpers:
    def test_character_dir_contains_name(self):
        path = U.character_dir("Alice")
        assert path.endswith("Alice")

    @pytest.mark.parametrize("bad", ["..", "../Alice", "Alice/../../Bob", "Alice\\Bob", "Alice:Bob", "Alice.2", "<img src=x onerror=alert(1)>", "Alice\" onclick=\"x"])
    def test_character_dir_rejects_unsafe_names(self, bad):
        with pytest.raises(ValueError):
            U.character_dir(bad)

    @pytest.mark.parametrize("good", ["Alice", "Alice 2", "Alice_2", "Alice-2"])
    def test_character_dir_allows_safe_display_names(self, good):
        assert os.path.basename(U.character_dir(good)) == good

    @pytest.mark.parametrize("builder", [U.faces_dir, U.sheets_dir, U.sprites_dir])
    def test_character_asset_dirs_reject_path_like_costumes(self, builder):
        with pytest.raises(ValueError):
            builder("Alice", "Bad\\Costume", "neutral")

    def test_safe_join_under_rejects_escape(self, tmp_path):
        with pytest.raises(ValueError):
            U.safe_join_under(str(tmp_path), "..", "outside")

    def test_safe_join_under_accepts_windows_style_relative_parts(self, tmp_path):
        path = U.safe_join_under(str(tmp_path), "Sprites\\Naked\\Neutral")
        assert path == os.path.join(str(tmp_path), "Sprites", "Naked", "Neutral")

    def test_safe_relative_path_normalizes_backslashes(self):
        assert U.safe_relative_path("Sheets\\Naked\\neutral") == "Sheets/Naked/neutral"

    @pytest.mark.parametrize("bad", ["/abs/path", "../x", "a/../b", "~/.ssh", "C:\\Users\\Alice", "\\\\server\\share\\x"])
    def test_safe_relative_path_rejects_unsafe_paths(self, bad):
        with pytest.raises(ValueError):
            U.safe_relative_path(bad)

    def test_config_path_filename(self):
        path = U.config_path("Alice")
        assert os.path.basename(path) == "Alice_config.json"

    def test_privileged_request_requires_header(self):
        request = types.SimpleNamespace(headers={"Host": "127.0.0.1:8188"})
        with pytest.raises(ValueError):
            U.validate_privileged_request(request)

    def test_privileged_request_accepts_same_origin(self):
        request = types.SimpleNamespace(headers={
            "Host": "127.0.0.1:8188",
            "Origin": "http://127.0.0.1:8188",
            "X-VNCCS-CSRF": "1",
        })
        U.validate_privileged_request(request)

    def test_privileged_request_accepts_same_origin_without_header(self):
        request = types.SimpleNamespace(headers={
            "Host": "192.168.1.240:8188",
            "Origin": "http://192.168.1.240:8188",
            "Sec-Fetch-Site": "same-origin",
        })
        U.validate_privileged_request(request)

    def test_privileged_request_rejects_cross_origin(self):
        request = types.SimpleNamespace(headers={
            "Host": "127.0.0.1:8188",
            "Origin": "http://evil.test",
            "X-VNCCS-CSRF": "1",
        })
        with pytest.raises(ValueError):
            U.validate_privileged_request(request)

    def test_privileged_request_rejects_cross_site_even_with_header(self):
        request = types.SimpleNamespace(headers={
            "Host": "127.0.0.1:8188",
            "Origin": "http://127.0.0.1:8188",
            "Sec-Fetch-Site": "cross-site",
            "X-VNCCS-CSRF": "1",
        })
        with pytest.raises(ValueError):
            U.validate_privileged_request(request)

    def test_sheets_dir_structure(self):
        path = U.sheets_dir("Alice")
        assert "Sheets" in path
        assert "Naked" in path
        assert "sheet_neutral" in path

    def test_faces_dir_structure(self):
        path = U.faces_dir("Alice")
        assert "Faces" in path
        assert "face_neutral" in path

    def test_sprites_dir_structure(self):
        path = U.sprites_dir("Alice")
        assert "Sprites" in path
        assert "sprite_neutral" in path

    def test_sheets_dir_custom_costume(self):
        path = U.sheets_dir("Alice", costume="Dress", emotion="happy")
        assert "Dress" in path
        assert "happy" in path


# ── load_config / save_config ─────────────────────────────────────────────────

class TestConfigIO:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))

        data = {"character_info": {"name": "Alice", "sex": "female"}, "config_version": "2.0"}
        U.save_config("Alice", data)
        loaded = U.load_config("Alice")
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        assert U.load_config("NonExistent") is None

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.save_config("Bob", {"x": 1})
        assert os.path.isdir(os.path.join(str(tmp_path), "Bob"))

    def test_load_corrupt_json_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        char_dir = tmp_path / "Bad"
        char_dir.mkdir()
        (char_dir / "Bad_config.json").write_text("NOT JSON")
        assert U.load_config("Bad") is None


# ── load_character_info ───────────────────────────────────────────────────────

class TestLoadCharacterInfo:
    def test_returns_none_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        assert U.load_character_info("Ghost") is None

    def test_unifies_gender_to_sex(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.save_config("A", {"character_info": {"gender": "male"}})
        info = U.load_character_info("A")
        assert info["sex"] == "male"
        assert info["gender"] == "male"

    def test_normalizes_sex_case(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.save_config("B", {"character_info": {"sex": "FEMALE"}})
        info = U.load_character_info("B")
        assert info["sex"] == "female"


# ── list_characters ───────────────────────────────────────────────────────────

class TestListCharacters:
    def test_returns_sorted_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Charlie").mkdir()
        (tmp_path / "Alice").mkdir()
        (tmp_path / "Bob").mkdir()
        chars = U.list_characters()
        assert chars == ["Alice", "Bob", "Charlie"]

    def test_ignores_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        (tmp_path / "Alice").mkdir()
        (tmp_path / "readme.txt").write_text("x")
        assert U.list_characters() == ["Alice"]

    def test_empty_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        assert U.list_characters() == []

    def test_does_not_fallback_to_legacy_when_new_empty(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "new"
        legacy_dir = tmp_path / "legacy"
        new_dir.mkdir()
        legacy_dir.mkdir()
        (legacy_dir / "OldChar").mkdir()

        monkeypatch.setattr(U, "base_output_dir", lambda: str(new_dir))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(legacy_dir))
        assert U.list_characters() == []


def test_deprecated_load_character_sheet_ignores_sheet_files(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")

    base = tmp_path / "characters"
    sheet_dir = base / "Alice" / "Sheets" / "Naked" / "neutral"
    sheet_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "green").save(sheet_dir / "sheet_neutral_0001.png")

    monkeypatch.setattr(U, "base_output_dir", lambda: str(base))

    assert U.load_character_sheet("Alice", "Naked", "neutral") is None


# ── ensure_character_structure ────────────────────────────────────────────────

class TestEnsureCharacterStructure:
    def test_creates_main_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.ensure_character_structure("Alice")
        for d in U.MAIN_DIRS:
            assert os.path.isdir(os.path.join(str(tmp_path), "Alice", d, "Naked"))

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.ensure_character_structure("Alice")
        U.ensure_character_structure("Alice")  # should not raise


# ── save_costume_info / load_costume_info ─────────────────────────────────────

class TestCostumeIO:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.save_config("C", {"character_info": {}, "costumes": {}})
        costume_data = {"top": "red shirt", "bottom": "jeans"}
        U.save_costume_info("C", "Casual", costume_data)
        loaded = U.load_costume_info("C", "Casual")
        assert loaded == costume_data

    def test_load_missing_costume_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        U.save_config("D", {"character_info": {}, "costumes": {}})
        assert U.load_costume_info("D", "NonExistent") == {}

    def test_load_no_config_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path))
        assert U.load_costume_info("NoChar", "Anything") == {}


# ── migrate_legacy_data ───────────────────────────────────────────────────────

class TestMigrateLegacyData:
    def test_no_legacy_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path / "new"))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(tmp_path / "nonexistent"))
        result = U.migrate_legacy_data()
        assert result["migrated"] is False

    def test_moves_characters(self, tmp_path, monkeypatch):
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        old_dir.mkdir()
        char = old_dir / "Alice"
        char.mkdir()
        (char / "Alice_config.json").write_text(json.dumps({"character_info": {}}))

        monkeypatch.setattr(U, "base_output_dir", lambda: str(new_dir))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(old_dir))
        result = U.migrate_legacy_data()
        assert result["migrated"] is True
        assert "Alice" in result["details"]
        assert os.path.isdir(str(new_dir / "Alice"))
        archive = tmp_path / "old_migrated_safe_to_delete"
        assert result["archive_path"] == str(archive)
        assert not old_dir.exists()
        assert (archive / "Alice" / "Alice_config.json").exists()

    def test_skips_when_valid_dst_exists(self, tmp_path, monkeypatch):
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        old_dir.mkdir()
        new_dir.mkdir()

        # src
        src = old_dir / "Alice"
        src.mkdir()
        (src / "Alice_config.json").write_text("{}")

        # dst already has valid config
        dst = new_dir / "Alice"
        dst.mkdir()
        (dst / "Alice_config.json").write_text(json.dumps({"character_info": {"name": "Alice"}}))

        monkeypatch.setattr(U, "base_output_dir", lambda: str(new_dir))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(old_dir))
        result = U.migrate_legacy_data()
        archive = tmp_path / "old_migrated_safe_to_delete"
        assert result["migrated"] is True
        assert not old_dir.exists()
        assert (archive / "Alice" / "Alice_config.json").exists()
        assert (dst / "Alice_config.json").exists()

    def test_replaces_broken_dst(self, tmp_path, monkeypatch):
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        old_dir.mkdir()
        new_dir.mkdir()

        src = old_dir / "Alice"
        src.mkdir()
        (src / "Alice_config.json").write_text(json.dumps({"character_info": {}}))

        # dst exists but broken (no config file)
        dst = new_dir / "Alice"
        dst.mkdir()

        monkeypatch.setattr(U, "base_output_dir", lambda: str(new_dir))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(old_dir))
        result = U.migrate_legacy_data()
        assert result["migrated"] is True
        assert (new_dir / "Alice" / "Alice_config.json").exists()
        assert (tmp_path / "old_migrated_safe_to_delete" / "Alice" / "Alice_config.json").exists()

    def test_uses_unique_archive_name(self, tmp_path, monkeypatch):
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        existing_archive = tmp_path / "old_migrated_safe_to_delete"
        old_dir.mkdir()
        existing_archive.mkdir()
        char = old_dir / "Alice"
        char.mkdir()
        (char / "Alice_config.json").write_text(json.dumps({"character_info": {}}))

        monkeypatch.setattr(U, "base_output_dir", lambda: str(new_dir))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(old_dir))
        result = U.migrate_legacy_data()
        archive = tmp_path / "old_migrated_safe_to_delete_2"
        assert result["migrated"] is True
        assert result["archive_path"] == str(archive)
        assert (archive / "Alice" / "Alice_config.json").exists()

    def test_empty_legacy_dir(self, tmp_path, monkeypatch):
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        monkeypatch.setattr(U, "base_output_dir", lambda: str(tmp_path / "new"))
        monkeypatch.setattr(U, "get_legacy_output_dir", lambda: str(old_dir))
        result = U.migrate_legacy_data()
        assert result["migrated"] is False
