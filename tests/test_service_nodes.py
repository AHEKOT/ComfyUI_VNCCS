import os
import sys

import pytest

pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_service_emotions_generator_registered():
    from nodes.service_nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        VNCCS_Service_Emotions_Generator,
    )

    assert NODE_CLASS_MAPPINGS["VNCCS_Service_Emotions_Generator"] is VNCCS_Service_Emotions_Generator
    assert NODE_DISPLAY_NAME_MAPPINGS["VNCCS_Service_Emotions_Generator"] == "VNCCS Service Emotions Generator"


def test_service_emotions_input_contract():
    from nodes.service_nodes import VNCCS_Service_Emotions_Generator

    inputs = VNCCS_Service_Emotions_Generator.INPUT_TYPES()
    assert set(inputs["required"]) == {"character", "character_prompt", "denoise"}
    assert "optional" not in inputs
    assert inputs["required"]["character_prompt"][1]["multiline"] is True
    assert inputs["required"]["denoise"][1]["default"] == 0.55
    assert VNCCS_Service_Emotions_Generator.RETURN_TYPES == ("STRING", "INT")


def test_flatten_emotions_config_uses_safe_name(monkeypatch):
    import nodes.service_nodes as service_nodes

    monkeypatch.setattr(service_nodes, "load_emotions_data", lambda: {
        "Smile": [
            {
                "safe_name": "light-smile",
                "natural_prompt": "The character smiles softly.",
                "description": "light_smile",
            }
        ]
    })

    result = service_nodes._flatten_emotions_config()

    assert result == [{
        "safe_name": "light-smile",
        "prompt": "The character smiles softly.",
        "description": "light_smile",
    }]


def test_prompt_for_emotion_starts_with_anima_natural_prompt():
    from nodes.service_nodes import VNCCS_Service_Emotions_Generator

    prompt = VNCCS_Service_Emotions_Generator()._prompt_for_emotion({
        "safe_name": "light-smile",
        "prompt": "The character smiles softly.",
        "description": "light_smile",
    })

    assert prompt == "The character smiles softly.\n\nEmotion Tags: light_smile"


def test_prompt_for_emotion_includes_manual_character_prompt_in_wildcard():
    from nodes.service_nodes import VNCCS_Service_Emotions_Generator

    prompt = VNCCS_Service_Emotions_Generator()._prompt_for_emotion(
        {
            "safe_name": "light-smile",
            "prompt": "The character smiles softly.",
            "description": "light_smile",
        },
        "black hair, red eyes, school uniform",
    )

    assert prompt == (
        "The character smiles softly.\n\nEmotion Tags: light_smile Character details: "
        "black hair, red eyes, school uniform"
    )


def test_service_anima_settings_select_diffusion_model():
    import nodes.service_nodes as service_nodes
    from nodes.character_creator_v2 import normalize_gen_settings

    settings = service_nodes._service_anima_generation_settings()
    normalized = normalize_gen_settings(settings)

    assert settings["generation_mode"] == "anima"
    assert settings["diffusion_model_name"] == "anima-base-v1.0.safetensors"
    assert settings["mode_settings"]["anima"]["diffusion_model_name"] == "anima-base-v1.0.safetensors"
    assert normalized["diffusion_model_name"] == "anima-base-v1.0.safetensors"
    assert settings["clip_name"] == "qwen_3_06b_base.safetensors"
    assert settings["vae_name"] == "qwen_image_vae.safetensors"
    assert settings["steps"] == 12
    assert settings["cfg"] == 1
    assert settings["turbo_enabled"] is True


def test_flatten_emotions_config_rejects_duplicate_safe_name(monkeypatch):
    import nodes.service_nodes as service_nodes

    monkeypatch.setattr(service_nodes, "load_emotions_data", lambda: {
        "A": [{"safe_name": "smile"}],
        "B": [{"safe_name": "smile"}],
    })

    with pytest.raises(ValueError, match="Duplicate emotion safe_name"):
        service_nodes._flatten_emotions_config()


def test_safe_emotion_filename_rejects_path_like_names():
    import nodes.service_nodes as service_nodes

    assert service_nodes._safe_emotion_filename("smile-01") == "smile-01.png"
    with pytest.raises(ValueError, match="Invalid emotion safe_name"):
        service_nodes._safe_emotion_filename("../smile")


def test_service_output_dir_is_inside_comfy_output(tmp_path, monkeypatch):
    import nodes.service_nodes as service_nodes

    monkeypatch.setattr(service_nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))

    assert service_nodes._service_emotions_output_dir() == str(tmp_path / "VNCCS" / "ServiceEmotions")


def test_output_image_ui_item_points_to_output_subfolder(tmp_path, monkeypatch):
    import nodes.service_nodes as service_nodes

    monkeypatch.setattr(service_nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    path = tmp_path / "VNCCS" / "ServiceEmotions" / "light-smile.png"

    assert service_nodes._output_image_ui_item(str(path)) == {
        "filename": "light-smile.png",
        "subfolder": "VNCCS/ServiceEmotions",
        "type": "output",
    }


def test_service_generate_saves_named_emotion_files_to_output(tmp_path, monkeypatch):
    import torch
    import nodes.service_nodes as service_nodes

    monkeypatch.setattr(service_nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(service_nodes, "_flatten_emotions_config", lambda: [
        {"safe_name": "light-smile", "prompt": "Smile.", "description": ""},
        {"safe_name": "angry", "prompt": "Angry.", "description": ""},
    ])

    node = service_nodes.VNCCS_Service_Emotions_Generator()
    monkeypatch.setattr(node, "_default_pipe", lambda: {"seed": 10})
    monkeypatch.setattr(node, "_extract_pipe", lambda pipe: pipe)
    monkeypatch.setattr(
        node,
        "_run_emotion_generation_one",
        lambda *args, **kwargs: (torch.zeros((1, 4, 4, 3)), torch.ones((1, 4, 4, 3))),
    )

    result = node.generate(torch.zeros((1, 4, 4, 3)), character_prompt="green hair")

    output_dir = tmp_path / "VNCCS" / "ServiceEmotions"
    assert (output_dir / "light-smile.png").is_file()
    assert (output_dir / "angry.png").is_file()
    assert result["result"][0] == str(output_dir)
    assert result["result"][1] == 2
    assert result["ui"]["images"] == [
        {"filename": "light-smile.png", "subfolder": "VNCCS/ServiceEmotions", "type": "output"},
        {"filename": "angry.png", "subfolder": "VNCCS/ServiceEmotions", "type": "output"},
    ]


def test_service_generate_uses_emotion_detailer_defaults(tmp_path, monkeypatch):
    import torch
    import nodes.service_nodes as service_nodes

    monkeypatch.setattr(service_nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(service_nodes, "_flatten_emotions_config", lambda: [
        {"safe_name": "light-smile", "prompt": "Smile.", "description": ""},
    ])

    calls = []

    def fake_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return torch.zeros((1, 4, 4, 3)), torch.ones((1, 4, 4, 3))

    node = service_nodes.VNCCS_Service_Emotions_Generator()
    monkeypatch.setattr(node, "_default_pipe", lambda: {"seed": 10})
    monkeypatch.setattr(node, "_extract_pipe", lambda pipe: pipe)
    monkeypatch.setattr(node, "_run_emotion_generation_one", fake_run)

    node.generate(torch.zeros((1, 4, 4, 3)), character_prompt="green hair")

    assert len(calls) == 1
    assert "bbox_crop_factor" not in calls[0]["kwargs"]
    assert "use_sam" not in calls[0]["kwargs"]
