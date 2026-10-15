"""Native MiniMax H3 still decoding and ComfyUI compatibility checks."""

import inspect


H3_UPDATE_MESSAGE = (
    "MiniMax H3 requires ComfyUI's native still-image decoder from PR #16881 "
    "(merged October 10, 2026). Update ComfyUI to a build containing this fix, "
    "then restart the ComfyUI server before generating H3 images."
)


def h3_runtime_status():
    try:
        from comfyui_version import __version__
    except ImportError:
        __version__ = None
    try:
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        # Release numbers alone cannot distinguish git builds or backports.
        supported = (
            "still_frame" in inspect.signature(MiniMaxH3VideoVAE.__init__).parameters
            and "frames" in inspect.signature(MiniMaxH3VideoVAE._adaptive_decode).parameters
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        supported = False
    return {"version": __version__, "h3_still_supported": supported, "h3_update_message": H3_UPDATE_MESSAGE}


def require_h3_still_support():
    status = h3_runtime_status()
    if not status["h3_still_supported"]:
        version = status["version"] or "unknown"
        raise RuntimeError(f"{H3_UPDATE_MESSAGE} Detected ComfyUI version: {version}.")


def h3_still_samples(samples):
    """Keep the first video latent; ComfyUI supplies its native decode context."""
    require_h3_still_support()
    if not isinstance(samples, dict) or "samples" not in samples:
        raise ValueError("MiniMax H3 decoding requires a LATENT with video samples.")
    video = samples["samples"]
    if getattr(video, "is_nested", False):
        video = video.unbind()[0]
    shape = getattr(video, "shape", ())
    if len(shape) != 5 or shape[1] != 24 or shape[2] < 1:
        raise ValueError("MiniMax H3 video samples must have shape [B, 24, T, H, W] with T >= 1.")
    # Own the still storage so the AV clip is not retained by the decode input.
    return {**samples, "samples": video[:, :, :1].clone()}
