"""Flux Klein conditioning encoder assembled from native ComfyUI nodes."""

import inspect

try:
    import nodes as comfy_nodes
except Exception:  # pragma: no cover - only used outside ComfyUI
    comfy_nodes = None


def _unwrap_node_output(value):
    if hasattr(value, "result") and type(value).__name__ == "NodeOutput":
        block_execution = getattr(value, "block_execution", None)
        if block_execution:
            raise RuntimeError(str(block_execution))
        value = value.result
    return value


def _call_comfy_node(class_name, **kwargs):
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) if comfy_nodes else {}
    node_class = mappings.get(class_name)
    if node_class is None:
        raise RuntimeError(f"Required node '{class_name}' is not available. Update ComfyUI and restart it.")

    instance = node_class()
    candidates = []
    function_name = getattr(node_class, "FUNCTION", None)
    if function_name:
        candidates.append(function_name)
    candidates.extend(("encode", "upscale", "zero_out", "append", "generate", "get_size", "execute"))

    for method_name in candidates:
        method = getattr(instance, method_name, None)
        if method is None:
            continue
        signature = inspect.signature(method)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted = kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        result = _unwrap_node_output(method(**accepted))
        return result if isinstance(result, tuple) else (result,)

    raise RuntimeError(f"Node '{class_name}' has no callable FUNCTION")


def _image_dimensions(image):
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        raise RuntimeError("Scaled reference image has no valid B,H,W,C shape.")
    height = int(shape[-3])
    width = int(shape[-2])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Scaled reference image has invalid dimensions: {width}x{height}.")
    return width, height


class VNCCS_Flux_Klein_Encoder:
    UPSCALE_METHODS = ["lanczos", "bicubic", "area", "bilinear", "nearest-exact"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Flux Klein text encoder."}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Positive Flux Klein prompt.",
                    },
                ),
                "vae": ("VAE", {"tooltip": "Flux 2 VAE used for reference images."}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "First optional reference image."}),
                "image2": ("IMAGE", {"tooltip": "Second optional reference image."}),
                "image3": ("IMAGE", {"tooltip": "Third optional reference image."}),
                "upscale_method": (
                    cls.UPSCALE_METHODS,
                    {"default": "lanczos", "tooltip": "Reference image scaling method."},
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": "Target area for each connected reference image.",
                    },
                ),
                "resolution_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Resolution step passed to Scale Image to Total Pixels.",
                    },
                ),
                "empty_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Latent width used only when no reference image is connected.",
                    },
                ),
                "empty_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Latent height used only when no reference image is connected.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "VNCCS/encoding"
    DESCRIPTION = (
        "Builds Flux Klein text and optional reference-image conditioning "
        "with native ComfyUI nodes."
    )

    def encode(
        self,
        clip,
        prompt,
        vae,
        image1=None,
        image2=None,
        image3=None,
        upscale_method="lanczos",
        megapixels=1.0,
        resolution_steps=1,
        empty_width=1024,
        empty_height=1024,
        batch_size=1,
    ):
        positive = _call_comfy_node("CLIPTextEncode", clip=clip, text=prompt)[0]
        negative = _call_comfy_node("ConditioningZeroOut", conditioning=positive)[0]

        conditioned = positive
        first_scaled_image = None
        for image in (image1, image2, image3):
            if image is None:
                continue

            scaled = _call_comfy_node(
                "ImageScaleToTotalPixels",
                image=image,
                upscale_method=upscale_method,
                megapixels=float(megapixels),
                resolution_steps=int(resolution_steps),
            )[0]
            latent = _call_comfy_node("VAEEncode", pixels=scaled, vae=vae)[0]
            conditioned = _call_comfy_node(
                "ReferenceLatent",
                conditioning=conditioned,
                latent=latent,
            )[0]
            if first_scaled_image is None:
                first_scaled_image = scaled

        width = int(empty_width)
        height = int(empty_height)
        if first_scaled_image is not None:
            width, height = _image_dimensions(first_scaled_image)

        latent = _call_comfy_node(
            "EmptyFlux2LatentImage",
            width=width,
            height=height,
            batch_size=int(batch_size),
        )[0]
        return conditioned, negative, latent


NODE_CLASS_MAPPINGS = {
    "VNCCS_Flux_Klein_Encoder": VNCCS_Flux_Klein_Encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_Flux_Klein_Encoder": "VNCCS Flux Klein Encoder",
}
