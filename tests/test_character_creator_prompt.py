import pytest

from conftest import _preload_node


pytest.importorskip("torch")

character_creator_v2 = _preload_node("character_creator_v2")
CharacterCreatorV2 = character_creator_v2.CharacterCreatorV2


def _base_info(**overrides):
    info = {
        "sex": "female",
        "age": 18,
        "race": "human",
        "aesthetics": "masterpiece, best quality",
    }
    info.update(overrides)
    return info


def test_full_body_framing_replaces_cowboy_shot():
    positive, _ = CharacterCreatorV2.construct_prompt(_base_info(framing="Full_body"))

    assert "standing, full body" in positive
    assert "cowboy_shot" not in positive


def test_generation_prompt_log_summarizes_without_printing_prompts(caplog):
    CharacterCreatorV2.log_generation_prompts(
        "Workflow",
        "masterpiece, Full_body",
        "bad quality",
        framing="Full_body",
    )

    events = [record.vnccs for record in caplog.records if record.name == "VNCCS"]
    assert events == [{"component": "Creator", "event": "prompt_ready", "context": "Workflow",
                       "framing": "Full_body", "positive_chars": 22, "negative_chars": 11}]
    assert "masterpiece, Full_body" not in caplog.text
    assert "bad quality" not in caplog.text


def test_repeated_pose_preview_queries_do_not_write_default_logs(tmp_path, monkeypatch, caplog):
    root = tmp_path / "Alice"
    sprites = root / "Sprites" / "Naked" / "Neutral"
    sprites.mkdir(parents=True)
    (sprites / "pose.png").write_bytes(b"preview")
    monkeypatch.setattr(character_creator_v2, "character_dir", lambda name: str(root))
    for _ in range(20):
        assert character_creator_v2.list_pose_preview_files("Alice") == [str(sprites / "pose.png")]
    assert not [record for record in caplog.records if record.name == "VNCCS"]


@pytest.mark.parametrize("framing", [None, "", "portrait", "cowboy_shot"])
@pytest.mark.parametrize("mode", ["illustrious", "anima"])
def test_missing_or_invalid_framing_keeps_legacy_cowboy_shot(framing, mode):
    info = _base_info()
    if framing is not None:
        info["framing"] = framing

    positive, _ = CharacterCreatorV2.construct_prompt(info, mode)

    assert "cowboy_shot" in positive
    assert "Full_body" not in positive
    assert "head-to-upper-thigh" not in positive
    assert "image edge" not in positive
    assert "fingertips" not in positive
