"""Lightweight helpers for Qwen VL llama-cpp-python integration."""


QWEN_VL_HANDLER_NAMES = ("Qwen25VLChatHandler", "Qwen2VLChatHandler")


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
        for name in available:
            if "Qwen" in name and "VL" in name:
                return getattr(chat_format, name)

    raise RuntimeError(
        "No Qwen VL chat handler found in llama-cpp-python. "
        "Qwen2.5-VL requires Qwen25VLChatHandler or Qwen2VLChatHandler; "
        "refusing to use Llava15ChatHandler because it can crash with Qwen VL GGUF/mmproj. "
        f"Available handlers: {available}"
    )
