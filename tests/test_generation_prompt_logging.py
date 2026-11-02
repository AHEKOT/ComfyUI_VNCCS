"""Prompt logging at native inference boundaries runs without model dependencies."""

from types import SimpleNamespace

import pytest

from test_proxy_runtime import generator_module


@pytest.mark.parametrize("class_name", ["TextEncodeQwenImage21", "MiniMaxH3ReferenceToVideo", "TextGenerate"])
@pytest.mark.parametrize("fail", [False, True])
def test_native_inference_logs_exact_prompts_before_execution(generator_module, monkeypatch, caplog, class_name, fail):
    prompt = "A winter outfit\n" + "embroidered coat " * 350
    negative = "unwanted background " * 300
    system = "Preserve every requested clothing detail.\n" * 100
    parameters = {"prompt": prompt, "clip": object(), "image": "private image bytes"}
    if class_name == "TextEncodeQwenImage21":
        parameters["negative_prompt"] = negative
    elif class_name == "TextGenerate":
        parameters["system_prompt"] = system

    def run(**kwargs):
        assert kwargs == parameters
        record = next(record for record in caplog.records if getattr(record, "vnccs", {}).get("event") == "prompts")
        event = record.vnccs
        assert event["user" if class_name == "TextGenerate" else "positive"] == prompt
        assert prompt in record.getMessage()
        assert "private image bytes" not in record.getMessage()
        assert "clip" not in event and "image" not in event
        if class_name == "TextGenerate":
            assert event["system"] == system and system in record.getMessage()
        elif class_name == "TextEncodeQwenImage21":
            assert event["negative"] == negative and negative in record.getMessage()
        else:
            assert "negative" not in event
        if fail:
            raise RuntimeError("inference failed")
        return ("result",)

    node = type(class_name, (), {"FUNCTION": "run", "run": staticmethod(run)})
    monkeypatch.setattr(generator_module, "comfy_nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={class_name: node}))
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            generator_module._call_comfy_node(class_name, **parameters)
    else:
        assert generator_module._call_comfy_node(class_name, **parameters) == ("result",)
