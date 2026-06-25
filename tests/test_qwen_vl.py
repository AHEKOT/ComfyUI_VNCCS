import types

import pytest

from nodes.qwen_vl import get_qwen_vl_chat_handler


def test_prefers_qwen25_handler():
    class Qwen25VLChatHandler:
        pass

    llama_cpp = types.SimpleNamespace(
        llama_chat_format=types.SimpleNamespace(Qwen25VLChatHandler=Qwen25VLChatHandler)
    )

    assert get_qwen_vl_chat_handler(llama_cpp) is Qwen25VLChatHandler


def test_falls_back_to_qwen2_handler():
    class Qwen2VLChatHandler:
        pass

    llama_cpp = types.SimpleNamespace(
        llama_chat_format=types.SimpleNamespace(Qwen2VLChatHandler=Qwen2VLChatHandler)
    )

    assert get_qwen_vl_chat_handler(llama_cpp) is Qwen2VLChatHandler


def test_refuses_llava_fallback_for_qwen_vl():
    class Llava15ChatHandler:
        pass

    llama_cpp = types.SimpleNamespace(
        llama_chat_format=types.SimpleNamespace(Llava15ChatHandler=Llava15ChatHandler)
    )

    with pytest.raises(RuntimeError, match="No Qwen VL chat handler"):
        get_qwen_vl_chat_handler(llama_cpp)
