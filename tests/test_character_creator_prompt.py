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

    assert "Full_body" in positive
    assert "cowboy_shot" not in positive


def test_generation_prompt_log_prints_exact_positive_and_negative(capsys):
    CharacterCreatorV2.log_generation_prompts(
        "Workflow",
        "masterpiece, Full_body",
        "bad quality",
        framing="Full_body",
    )

    output = capsys.readouterr().out
    assert "[VNCCS Character Creator V2] Workflow framing input: 'Full_body'" in output
    assert "[VNCCS Character Creator V2] Workflow positive generation prompt: masterpiece, Full_body" in output
    assert "[VNCCS Character Creator V2] Workflow negative generation prompt: bad quality" in output


@pytest.mark.parametrize("framing", [None, "", "portrait", "cowboy_shot"])
def test_missing_or_invalid_framing_keeps_legacy_cowboy_shot(framing):
    info = _base_info()
    if framing is not None:
        info["framing"] = framing

    positive, _ = CharacterCreatorV2.construct_prompt(info)

    assert "cowboy_shot" in positive
    assert "Full_body" not in positive
