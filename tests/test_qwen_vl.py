import types

import pytest

from nodes.qwen_vl import get_qwen_vl_chat_handler


def test_selects_qwen35_even_when_legacy_handlers_are_available():
    handler = object()
    llama_cpp = types.SimpleNamespace(llama_chat_format=types.SimpleNamespace(
        Qwen35ChatHandler=handler, Qwen25VLChatHandler=object(), Qwen2VLChatHandler=object(),
    ))
    assert get_qwen_vl_chat_handler(llama_cpp) is handler


@pytest.mark.parametrize("name", ["Qwen25VLChatHandler", "Qwen2VLChatHandler", "Qwen3VLChatHandler", "Llava15ChatHandler"])
def test_rejects_incompatible_handlers(name):
    llama_cpp = types.SimpleNamespace(llama_chat_format=types.SimpleNamespace(**{name: object()}))
    with pytest.raises(RuntimeError, match="Qwen3.5 requires Qwen35ChatHandler"):
        get_qwen_vl_chat_handler(llama_cpp)


def test_text_wizard_disables_thinking_in_model_template(monkeypatch):
    import sys
    from nodes.qwen_vl import configure_qwen_text_chat
    captured = {}
    handler = object()
    class Formatter:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def to_chat_handler(self):
            return handler
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", types.SimpleNamespace(Jinja2ChatFormatter=Formatter))
    llm = types.SimpleNamespace(metadata={"tokenizer.chat_template": "model template"})
    configure_qwen_text_chat(llm)
    assert llm.chat_handler is handler
    assert captured["template"] == "{% set enable_thinking = false %}model template"
    assert captured["eos_token"] == "<|im_end|>"
