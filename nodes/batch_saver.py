"""VNCCSBatchSaver — fixes the emotion face save cross-contamination bug.

Root cause
----------
EmotionGeneratorV2 outputs ``face_output_paths`` as LIST type (OUTPUT_IS_LIST=True,
one path per emotion). The QWEN Detailer component node (node 75) receives the
images as a LIST, processes each emotion once, and emits face crops as BATCH type.

ComfyUI's ``merge_result_data`` stores non-list outputs from N iterations as a
Python list of length N. But because the QWEN Detailer is a component/subgraph
node, its face-batch output may be collected differently—leading to a cache entry
of length 1 (one big concatenated tensor of all emotions' faces) while
``filename_prefix`` has length 3 (one per emotion).

``slice_dict`` in ``_async_map_node_over_list`` then **broadcasts** that single
element to every path iteration (i > 0 falls back to ``v[-1]``), so SaveImage
writes the entire multi-emotion face batch to *every* emotion folder.

Fix
---
``INPUT_IS_LIST = True`` tells ComfyUI to pass the raw Python lists directly to
this node (called exactly once) instead of slicing and iterating. We then zip
images with paths ourselves, guaranteeing a correct 1-to-1 mapping regardless of
how ComfyUI collected the per-iteration tensors.
"""

import os
import re

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image as PilImage
except ImportError:
    PilImage = None


def _find_next_counter(directory: str, base_name: str) -> int:
    """Return the next auto-increment counter for files that match
    ``{base_name}{digits}.png`` inside *directory*.

    Scans existing files so it never overwrites previously saved images.
    """
    if not directory or not os.path.isdir(directory):
        return 0
    max_counter = -1
    try:
        for fname in os.listdir(directory):
            if fname.startswith(base_name) and fname.endswith(".png"):
                stem = fname[len(base_name) : -4]  # strip prefix and .png
                m = re.fullmatch(r"\d+", stem)
                if m:
                    n = int(m.group())
                    if n > max_counter:
                        max_counter = n
    except OSError:
        pass
    return max_counter + 1


def _to_4d(tensor):
    """Ensure tensor is [B, H, W, C]. Adds batch dim when [H, W, C]."""
    if tensor is None:
        return None
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def _tensor_to_uint8(frame):
    """Convert a single [H, W, C] float tensor (0-1) to uint8 numpy array."""
    return (frame.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)


class VNCCSBatchSaver:
    """Save face (or sheet) images to per-emotion directories.

    Uses ``INPUT_IS_LIST = True`` to receive the raw per-iteration data from
    ComfyUI's execution engine rather than the iterated/broadcast values that
    cause the emotion face explosion bug.

    Wire up in the workflow:
      - ``images``          ← QWEN Detailer ``faces`` output  (node 75, slot 1)
      - ``filename_prefix`` ← ``face_output_paths`` reroute chain (node 68)
      - ``masks``           ← optional, for RGBA/alpha saves (sheet saver)
    """

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "VNCCS"
    FUNCTION = "save_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output/face_"}),
            },
            "optional": {
                "masks": ("MASK",),
            },
        }

    # ------------------------------------------------------------------
    def save_batch(self, images, filename_prefix, masks=None):
        """
        Parameters
        ----------
        images : list
            One element per emotion; each element is a ``[B, H, W, C]`` or
            ``[H, W, C]`` float tensor (0-1 range) containing the face crops
            for that emotion.  With ``INPUT_IS_LIST=True`` ComfyUI passes the
            raw per-iteration list rather than broadcasting a concatenated batch.
        filename_prefix : list[str]
            One string per emotion, e.g.
            ``"/path/to/Faces/Naked/angry/face_angry_"``.
        masks : list | None
            Optional per-emotion mask tensors for RGBA saves.
        """
        # Normalise everything to plain Python lists
        if not isinstance(images, list):
            images = [images]
        if not isinstance(filename_prefix, list):
            filename_prefix = [filename_prefix]
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        saved_paths = []

        for idx, (img_data, prefix) in enumerate(zip(images, filename_prefix)):
            # ----------------------------------------------------------
            # Normalise prefix to a plain string
            # ----------------------------------------------------------
            if isinstance(prefix, list):
                prefix = prefix[0] if prefix else ""
            prefix = str(prefix)

            # ----------------------------------------------------------
            # Determine output directory and base filename stem
            # ----------------------------------------------------------
            output_dir = os.path.dirname(prefix)
            base_name = os.path.basename(prefix)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # ----------------------------------------------------------
            # Normalise img_data to a 4-D tensor [B, H, W, C]
            # ----------------------------------------------------------
            if img_data is None:
                print(f"[VNCCSBatchSaver] images[{idx}] is None — skipping.")
                continue

            if isinstance(img_data, list):
                # Received a list of tensors (e.g. OUTPUT_IS_LIST upstream)
                if not img_data:
                    print(f"[VNCCSBatchSaver] images[{idx}] is an empty list — skipping.")
                    continue
                tensors = [_to_4d(t) for t in img_data if t is not None]
                if not tensors:
                    continue
                # Concatenate along the batch dimension, handling varying sizes
                try:
                    img_data = torch.cat(tensors, dim=0)
                except RuntimeError:
                    # Spatial dimensions differ — save individually without cat
                    for t_idx, t in enumerate(tensors):
                        counter = _find_next_counter(output_dir, base_name)
                        for frame_i in range(t.shape[0]):
                            self._save_frame(
                                t[frame_i], prefix, output_dir, base_name,
                                counter, masks, idx, frame_i, saved_paths
                            )
                            counter += 1
                    continue

            if not isinstance(img_data, torch.Tensor):
                print(
                    f"[VNCCSBatchSaver] images[{idx}] has unexpected type "
                    f"{type(img_data)} — skipping."
                )
                continue

            img_data = _to_4d(img_data)  # [B, H, W, C]

            # ----------------------------------------------------------
            # Get per-emotion mask (optional)
            # ----------------------------------------------------------
            mask_data = None
            if masks is not None and idx < len(masks):
                mask_data = masks[idx]
                if isinstance(mask_data, list):
                    mask_data = mask_data[0] if mask_data else None

            # ----------------------------------------------------------
            # Find next available counter (avoids overwriting existing files)
            # ----------------------------------------------------------
            counter = _find_next_counter(output_dir, base_name)

            # ----------------------------------------------------------
            # Save every frame in the per-emotion batch
            # ----------------------------------------------------------
            for frame_i in range(img_data.shape[0]):
                frame = img_data[frame_i]  # [H, W, C]
                frame_np = _tensor_to_uint8(frame)

                pil_img = self._make_pil_image(frame_np, mask_data, frame_i)

                out_path = f"{prefix}{counter:05d}.png"
                pil_img.save(out_path)
                print(f"[VNCCSBatchSaver] Saved: {out_path}")
                saved_paths.append(out_path)
                counter += 1

        return {"ui": {}}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_frame(self, frame, prefix, output_dir, base_name,
                    counter, masks, idx, frame_i, saved_paths):
        """Save a single [H, W, C] tensor frame and append path to saved_paths."""
        frame_np = _tensor_to_uint8(frame)
        mask_data = None
        if masks is not None and idx < len(masks):
            mask_data = masks[idx]
            if isinstance(mask_data, list):
                mask_data = mask_data[0] if mask_data else None
        pil_img = self._make_pil_image(frame_np, mask_data, frame_i)
        out_path = f"{prefix}{counter:05d}.png"
        pil_img.save(out_path)
        print(f"[VNCCSBatchSaver] Saved: {out_path}")
        saved_paths.append(out_path)

    def _make_pil_image(self, frame_np, mask_data, frame_i: int):
        """Build a PIL image (RGB or RGBA) from a uint8 numpy frame."""
        if mask_data is not None and torch is not None:
            try:
                m = mask_data
                if isinstance(m, torch.Tensor):
                    # Normalise mask to [H, W]
                    if m.dim() == 3:
                        # [B, H, W] — pick the frame at frame_i (or last)
                        m = m[min(frame_i, m.shape[0] - 1)]
                    elif m.dim() == 2:
                        pass  # already [H, W]
                    else:
                        raise ValueError(f"Unexpected mask shape {m.shape}")

                    alpha_np = _tensor_to_uint8(m)

                    # Resize alpha if spatial dims differ from the image
                    if alpha_np.shape != frame_np.shape[:2]:
                        pil_alpha = PilImage.fromarray(alpha_np).resize(
                            (frame_np.shape[1], frame_np.shape[0]),
                            PilImage.LANCZOS,
                        )
                        alpha_np = np.array(pil_alpha)

                    rgba_np = np.dstack([frame_np[:, :, :3], alpha_np])
                    return PilImage.fromarray(rgba_np, "RGBA")
            except Exception as exc:
                print(f"[VNCCSBatchSaver] Mask processing error: {exc} — saving RGB.")

        # Fall back to RGB (drop alpha channel if present)
        return PilImage.fromarray(frame_np[:, :, :3])


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VNCCSBatchSaver": VNCCSBatchSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCSBatchSaver": "VNCCS Batch Saver",
}
