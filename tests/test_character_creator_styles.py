import json
import re
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import _preload_node


pytest.importorskip("torch")

creator = _preload_node("character_creator_v2")
CATALOG_PATH = Path(__file__).parents[1] / "character_template" / "character_styles.json"
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
LEGACY_STYLE_IDS = {
        "shonen_anime", "shojo_anime", "seinen_anime", "josei_anime",
        "anime_1970s", "anime_1980s", "anime_1990s", "anime_2000s", "anime_2010s", "anime_2020s",
        "ghibli_miyazaki", "makoto_shinkai", "kyoto_violet_evergarden", "kyoto_k_on",
        "trigger_imaishi", "yoshiyuki_sadamoto", "clamp", "sailor_moon_90s",
        "akira_toriyama", "rumiko_takahashi", "katsuhiro_otomo", "satoshi_kon",
        "ghost_in_the_shell_1995", "toshihiro_kawamoto", "yoshitoshi_abe", "nobuteru_yuki",
        "disney_renaissance", "don_bluth", "bruce_timm_dcau", "genndy_tartakovsky",
        "cartoon_saloon", "disney_tangled", "pixar_incredibles", "fortiche_arcane",
        "spider_verse", "coraline_selick", "art_nouveau_mucha", "art_deco_lempicka",
        "academic_realism", "photorealism",
    }
STYLES = [style for group in CATALOG["groups"] for style in group["styles"] if style["id"] in LEGACY_STYLE_IDS]


def _base_info(**overrides):
    info = {
        "sex": "female",
        "age": 30,
        "race": "human",
        "background_color": "Green",
    }
    info.update(overrides)
    return info


def test_catalog_contains_the_approved_reference_styles():
    assert {g["label"]: sum(s["id"] in LEGACY_STYLE_IDS for s in g["styles"]) for g in CATALOG["groups"] if any(s["id"] in LEGACY_STYLE_IDS for s in g["styles"])} == {
        "Anime Meta Styles": 10,
        "Anime": 16, "Animation": 10, "Artistic": 2, "Realistic": 2,
    }
    assert {s["id"] for s in STYLES} == LEGACY_STYLE_IDS
    assert len({s["id"] for s in STYLES}) == len(STYLES)
    assert len({s["label"] for s in STYLES}) == len(STYLES)
    assert len({s["prompt"] for s in STYLES}) == len(STYLES)
    assert all(re.fullmatch(r"[a-z0-9_]+", s["id"]) for s in STYLES)


def test_style_catalog_resolves_legacy_aliases():
    assert creator.CHARACTER_STYLE_CATALOG_PATH == str(CATALOG_PATH)
    assert creator.CHARACTER_STYLE_CATALOG == CATALOG
    assert creator.CHARACTER_STYLE_PROMPTS == {s["id"]: s["prompt"] for group in CATALOG["groups"] for s in group["styles"]}
    assert creator.CHARACTER_STYLE_ALIASES == CATALOG["aliases"]
    assert CATALOG["aliases"]["clio_ghibli_style"] == "ghibli_miyazaki"
    assert creator.DEFAULT_CHARACTER_STYLE == CATALOG["default_style"] == "ghibli_miyazaki"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s["id"])
def test_every_style_contains_a_name_reference_and_visual_description(style):
    lines = style["prompt"].splitlines()
    assert len(lines) == 3
    assert lines[0] == f"Style: {style['label']}."
    assert lines[1].startswith("Reference: ") and len(lines[1]) > len("Reference: ")
    assert lines[2].startswith("Visual description: ")
    assert "preserve all specified character details" in lines[2]
    assert "eye colors" in lines[2]
    assert style["prompt"].isascii()


def test_default_style_is_ghibli_miyazaki():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info())
    assert positive.startswith("Style: Hayao Miyazaki / Studio Ghibli.")
    assert "Princess Mononoke; Howl's Moving Castle" in positive


def test_selected_style_template_is_added():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info(style="fortiche_arcane"))
    assert positive.startswith("Style: Fortiche / Arcane.\nReference: Arcane.")


def test_custom_style_replaces_template():
    positive, _negative = creator.CharacterCreatorV2.construct_prompt(_base_info(
        style="custom", custom_style="soft felt puppet illustration",
    ))
    assert positive.startswith("soft felt puppet illustration")
    assert "Hayao Miyazaki" not in positive


def test_user_styles_reload_and_missing_library_uses_workflow_snapshot(monkeypatch):
    style_id = "user_" + "a" * 32
    monkeypatch.setattr(creator, "load_user_styles", lambda: [{"id": style_id, "prompt": "live graphite style"}])
    info = _base_info(style=style_id, style_prompt="saved graphite style")
    assert creator.CharacterCreatorV2.construct_prompt(info)[0].startswith("live graphite style")
    monkeypatch.setattr(creator, "load_user_styles", lambda: [])
    assert creator.CharacterCreatorV2.construct_prompt(info)[0].startswith("saved graphite style")


def test_clio_prompt_resolves_by_stable_id():
    style_id = "clio_anime_style"
    positive, _ = creator.CharacterCreatorV2.construct_prompt(_base_info(style=style_id))
    assert positive.startswith(creator.CHARACTER_STYLE_PROMPTS[CATALOG["aliases"][style_id]])


def test_style_routes_validate_origin_payload_and_expose_live_user_styles(monkeypatch):
    monkeypatch.setattr(creator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    user = {"id": "user_" + "a" * 32, "label": "Mine", "prompt": "Graphite"}
    monkeypatch.setattr(creator, "load_user_styles", lambda: [user])
    result = asyncio.run(creator.get_character_styles(None))
    assert result.data["groups"][-1] == {"label": "My styles", "styles": [{**user, "image": ""}]}
    writes = []
    monkeypatch.setattr(creator, "save_user_style", lambda payload: writes.append(payload) or user)

    class Content:
        async def iter_chunked(self, size):
            for start in range(0, len(self.body), size):
                yield self.body[start:start + size]

    content = Content()
    content.body = json.dumps({"label": "Mine", "prompt": "Graphite"}).encode()
    request = SimpleNamespace(content=content, content_length=None, host="localhost", headers={"Host": "localhost", "X-VNCCS-CSRF": "1"})
    result = asyncio.run(creator.post_character_style(request))
    assert result.status == 200 and result.data["style"] == {**user, "image": ""}
    assert writes == [{"label": "Mine", "prompt": "Graphite"}]
    request.headers = {"Host": "localhost", "Origin": "https://other.example", "X-VNCCS-CSRF": "1"}
    assert asyncio.run(creator.post_character_style(request)).status == 403
    assert len(writes) == 1
    request.headers = {"Host": "localhost", "X-VNCCS-CSRF": "1"}
    content.body = b"x" * 70001
    assert asyncio.run(creator.post_character_style(request)).status == 413
    content.body = b"{broken"
    assert asyncio.run(creator.post_character_style(request)).status == 400
    assert len(writes) == 1


@pytest.mark.parametrize("retired_key", ["classic_anime", "monochrome_manga", "pixel_art"])
def test_retired_style_uses_existing_default_fallback(retired_key):
    assert retired_key not in creator.CHARACTER_STYLE_PROMPTS
    assert creator.CharacterCreatorV2.construct_prompt(_base_info(style=retired_key)) == (
        creator.CharacterCreatorV2.construct_prompt(_base_info())
    )


@pytest.mark.parametrize("style", [s["id"] for s in STYLES])
@pytest.mark.parametrize("mode", ["illustrious", "anima", "qi2"])
def test_style_can_be_separated_without_changing_character_fields(style, mode):
    info = _base_info(
        style=style, framing="Full_body", hair="black hair, long hair", eyes="blue eyes",
        face="freckles", body="broad shoulders", skin_color="dark skin",
        additional_details="silver ear studs", negative_prompt="blurry, missing facial features",
        background_color="Transparent" if mode == "qi2" else "Green",
    )
    body, body_negative = creator.CharacterCreatorV2.construct_prompt(info, mode, include_style=False)
    positive, negative = creator.CharacterCreatorV2.construct_prompt(info, mode)
    assert positive == creator.CHARACTER_STYLE_PROMPTS[style] + ", " + body
    assert negative == body_negative
    for detail in ("blue eyes", "black hair", "freckles", "broad shoulders",
                   "dark skin", "silver ear studs", "wear white bra and panties"):
        assert detail in body


def test_delete_style_route_requires_csrf_and_removes_record_and_preview(monkeypatch, tmp_path):
    library = _preload_node("character_styles")
    path = tmp_path / "character_styles.user.json"
    monkeypatch.setattr(library, "USER_CHARACTER_STYLES_PATH", str(path))
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    monkeypatch.setattr(creator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    style = library.save_user_style({"label": "Mine", "prompt": "Ink"})
    preview = Path(library.style_preview_path(style["id"]))
    preview.parent.mkdir()
    preview.write_bytes(b"preview")
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"style": style["id"]}),
                              host="localhost", headers={"Host": "localhost"})
    assert asyncio.run(creator.delete_character_style(request)).status == 403
    assert library.load_user_styles() == [style] and preview.exists()
    request.headers["X-VNCCS-CSRF"] = "1"
    request.headers["Origin"] = "https://other.example"
    assert asyncio.run(creator.delete_character_style(request)).status == 403
    del request.headers["Origin"]
    for invalid in ["photorealism", "custom", "../escape"]:
        request.rel_url.query["style"] = invalid
        assert asyncio.run(creator.delete_character_style(request)).status == 400
    request.rel_url.query["style"] = style["id"]
    result = asyncio.run(creator.delete_character_style(request))
    assert result.status == 200 and result.data == {"deleted": True, "style_id": style["id"]}
    assert library.load_user_styles() == [] and not preview.exists()
    assert asyncio.run(creator.delete_character_style(request)).status == 404


def test_delete_style_route_reports_disk_failure(monkeypatch):
    monkeypatch.setattr(creator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    def fail(style_id):
        raise PermissionError("Read only")
    monkeypatch.setattr(creator, "delete_user_style", fail)
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"style": "user_" + "a" * 32}),
                              host="localhost", headers={"Host": "localhost", "X-VNCCS-CSRF": "1"})
    result = asyncio.run(creator.delete_character_style(request))
    assert result.status == 500 and "permissions" in result.data["error"]
