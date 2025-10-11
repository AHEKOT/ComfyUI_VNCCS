import os
import torch
import numpy as np
import cv2
from PIL import Image


class CharacterSheetCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                # file_paths is expected to be a flat list of strings where each sheet contributes 12 target paths
                "file_paths": ("STRING",),
                "min_size": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "target_height": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 1}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    # This node intentionally produces no outputs; it saves files to disk instead
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "process_and_save"
    CATEGORY = "VNCCS/Util"

    def process_and_save(self, image: torch.Tensor, mask: torch.Tensor, file_paths, min_size: int = 64, target_height: int = 1024, overwrite: bool = True):
        """Crop individual characters from sheets, validate there are exactly 12 crops per sheet,
        resize them to the configured height (preserving aspect ratio), and save them as
        sprite_0.png ... sprite_11.png into the directories provided by `file_paths`.

        file_paths can be provided as a flat list of length batch_size*12, or as 12 entries (reused for single sheet).
        The node does not return any outputs; it will raise ValueError on validation failures.
        """

        # Normalize inputs
        # image: either a tensor [B,H,W,C] or a list of such tensors - we'll accept the common tensor case
        if isinstance(image, list):
            # Convert list of tensors into a batch if needed
            image = torch.cat(image, dim=0)
        if isinstance(mask, list):
            mask = torch.cat(mask, dim=0)

        batch_size = image.shape[0]

        # file_paths may come as a list or a single string; handle both
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not isinstance(file_paths, (list, tuple)):
            raise ValueError("file_paths must be a list of target paths (or a single path).")

        # Accept either flat list (batch_size*12) or per-sheet 12 list reused
        if len(file_paths) == 12 and batch_size >= 1:
            # replicate for each sheet
            file_paths_per_sheet = [file_paths for _ in range(batch_size)]
        elif len(file_paths) == batch_size * 12:
            file_paths_per_sheet = [file_paths[i * 12:(i + 1) * 12] for i in range(batch_size)]
        else:
            raise ValueError(f"file_paths length mismatch: expected 12 or batch_size*12 ({batch_size*12}), got {len(file_paths)}")

        for sheet_idx in range(batch_size):
            img_item = image[sheet_idx].cpu().numpy()
            mask_item = mask[sheet_idx].cpu().numpy()

            # Normalize mask shape to 2D
            if mask_item.ndim == 3 and mask_item.shape[0] == 1:
                current_mask_np = np.squeeze(mask_item, axis=0)
            elif mask_item.ndim == 2:
                current_mask_np = mask_item
            else:
                raise ValueError(f"Mask for sheet {sheet_idx} has unexpected shape {mask_item.shape}")

            mask_uint8 = (current_mask_np * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                raise ValueError(f"No contours found for sheet {sheet_idx}")

            # Build list of bounding boxes with centers for consistent ordering
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0:
                    continue
                if w < min_size or h < min_size:
                    continue
                cx = x + w / 2.0
                cy = y + h / 2.0
                boxes.append((x, y, w, h, cx, cy))

            if len(boxes) != 12:
                raise ValueError(f"After filtering by min_size expected 12 cropped characters for sheet {sheet_idx}, got {len(boxes)}")

            # Sort boxes into stable top-to-bottom, left-to-right order.
            # We'll first sort by center y, then within rows by center x. Because the sheet is typically 2 rows,
            # cluster by y into two rows using a simple threshold.
            img_h = img_item.shape[0]

            # Sort by y then x as a baseline
            boxes.sort(key=lambda b: (b[5], b[4]))

            # To ensure top row first then left-to-right, we split into two rows by y median
            ys = [b[5] for b in boxes]
            median_y = np.median(ys)
            top_row = [b for b in boxes if b[5] <= median_y]
            bottom_row = [b for b in boxes if b[5] > median_y]

            # If clustering didn't yield 6/6, fall back to simple row-slicing: first 6 are top after y-sort
            if len(top_row) != 6 or len(bottom_row) != 6:
                top_row = boxes[:6]
                bottom_row = boxes[6:12]

            # Sort each row left-to-right by center x
            top_row.sort(key=lambda b: b[4])
            bottom_row.sort(key=lambda b: b[4])

            ordered_boxes = top_row + bottom_row

            # Prepare targets directory list for this sheet
            targets = file_paths_per_sheet[sheet_idx]

            if len(targets) != 12:
                raise ValueError(f"Target paths for sheet {sheet_idx} must contain 12 entries, got {len(targets)}")

            # For saving, derive directory from each provided path. We'll ignore any filename suffix and
            # save using consistent names sprite_0.png ... sprite_11.png in the target directory.
            save_dirs = []
            for p in targets:
                # If user passed an empty string or a path that ends with os.sep, use it as directory
                if p.endswith(os.sep) or p == "":
                    save_dirs.append(p or os.getcwd())
                else:
                    save_dirs.append(os.path.dirname(p) or os.getcwd())

            # Ensure all directories exist
            for d in save_dirs:
                os.makedirs(d, exist_ok=True)

            # Crop, resize, and save in order
            for idx, box in enumerate(ordered_boxes):
                x, y, w, h, cx, cy = box
                x2 = int(x + w)
                y2 = int(y + h)
                x1 = int(x)
                y1 = int(y)

                cropped_np = img_item[y1:y2, x1:x2, :]

                # Build RGBA if necessary
                if cropped_np.shape[2] == 4:
                    rgba = cropped_np
                else:
                    alpha = (current_mask_np[y1:y2, x1:x2] * 255).astype(np.uint8)[..., np.newaxis]
                    rgba = np.concatenate([cropped_np[..., :3], alpha], axis=-1)

                # Convert to uint8 if needed
                if rgba.dtype != np.uint8:
                    # Assume floats in 0..1 or larger; clamp and convert
                    rgba = np.clip(rgba, 0.0, 255.0)
                    rgba = rgba.astype(np.uint8)

                # Resize to target height preserving aspect ratio
                h_orig, w_orig = rgba.shape[:2]
                if h_orig == 0 or w_orig == 0:
                    raise ValueError(f"Zero-sized crop encountered for sheet {sheet_idx} index {idx}")

                scale = target_height / float(h_orig)
                new_w = max(1, int(round(w_orig * scale)))
                new_h = target_height

                img_pil = Image.fromarray(rgba)
                img_resized = img_pil.resize((new_w, new_h), resample=Image.LANCZOS)

                # Save into the corresponding directory with stable name
                save_dir = save_dirs[idx]
                save_name = f"sprite_{idx}.png"
                save_path = os.path.join(save_dir, save_name)

                if os.path.exists(save_path) and not overwrite:
                    print(f"[CharacterSheetCropper] Skipping existing file {save_path} (overwrite disabled)")
                else:
                    img_resized.save(save_path)

            print(f"[CharacterSheetCropper] Saved 12 sprites for sheet {sheet_idx} to {set(save_dirs)}")

        # Node returns nothing
        return ()


NODE_CLASS_MAPPINGS = {
    "CharacterSheetCropper": CharacterSheetCropper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterSheetCropper": "VNCCS Character Sheet Cropper"
}
