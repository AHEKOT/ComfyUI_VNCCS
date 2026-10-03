"""Lightweight helpers for Qwen VL llama-cpp-python integration."""


QWEN_VL_HANDLER_NAMES = ("Qwen35ChatHandler",)


def get_qwen_vl_chat_handler(llama_cpp):
    chat_format = getattr(llama_cpp, "llama_chat_format", None)
    if chat_format is None:
        try:
            import llama_cpp.llama_chat_format as chat_format
        except Exception:
            chat_format = None

    available = []
    if chat_format is not None:
        available = [name for name in dir(chat_format) if "Handler" in name]
        for name in QWEN_VL_HANDLER_NAMES:
            handler = getattr(chat_format, name, None)
            if handler is not None:
                return handler

    raise RuntimeError(
        "No Qwen3.5 chat handler found in llama-cpp-python. "
        "Qwen3.5 requires Qwen35ChatHandler; "
        "refusing to use Llava15ChatHandler because it can crash with Qwen VL GGUF/mmproj. "
        f"Available handlers: {available}"
    )


def configure_qwen_text_chat(llm):
    """Keep short wizard responses in non-thinking mode using the GGUF template."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise RuntimeError("Qwen3.5 GGUF is missing its chat template.")
    llm.chat_handler = Jinja2ChatFormatter(
        template="{% set enable_thinking = false %}" + template,
        eos_token="<|im_end|>",
        bos_token="",
    ).to_chat_handler()
