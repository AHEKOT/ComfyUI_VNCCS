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
                "min_size": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "target_height": ("INT", {"default": 3072, "min": 64, "max": 8192, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images",)
    FUNCTION = "crop_character_sheet"
    CATEGORY = "VNCCS/Util"
    OUTPUT_IS_LIST = (True,)

    def crop_character_sheet(self, image: torch.Tensor, mask: torch.Tensor, min_size: int = 64, target_height: int = 3072):

        batch_size = image.shape[0]
        all_cropped_images = []

        for i in range(batch_size):
            img_item = image[i]
            mask_item = mask[i]

            img_np = img_item.cpu().numpy()
            img_h_orig, img_w_orig = img_np.shape[:2]

            mask_np_raw = mask_item.cpu().numpy()
            current_mask_np = None
            if mask_np_raw.ndim == 3 and mask_np_raw.shape[0] == 1:
                current_mask_np = np.squeeze(mask_np_raw, axis=0)
            elif mask_np_raw.ndim == 2:
                current_mask_np = mask_np_raw
            else:
                print(f"[CharacterSheetCropper] Warning: Mask for item {i} has unexpected shape {mask_np_raw.shape}. Skipping this item.")
                continue

            mask_uint8 = (current_mask_np * 255).astype(np.uint8)

            # Each contour should correspond to a separate character/object
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by grid position (2 rows x 6 cols) for deterministic sprite numbering
            # Assume sheet is divided into 2 rows and 6 columns
            sheet_h, sheet_w = img_h_orig, img_w_orig
            part_h = sheet_h // 2
            part_w = sheet_w // 6
            def contour_key(c):
                x, y, w, h = cv2.boundingRect(c)
                center_y = y + h // 2
                center_x = x + w // 2
                row = min(1, center_y // part_h)  # 0 or 1
                col = min(5, center_x // part_w)  # 0 to 5
                return row * 6 + col
            contours = sorted(contours, key=contour_key)

            if not contours:
                print(f"[CharacterSheetCropper] Info: No contours found in mask for item {i}.")
                continue

            for contour_idx, contour in enumerate(contours):
                # Get the bounding box for the current individual character/object
                char_x, char_y, char_w, char_h = cv2.boundingRect(contour)

                if char_w <= 0 or char_h <= 0:
                    print(f"[CharacterSheetCropper] Warning: Degenerate bounding box (w={char_w}, h={char_h}) for contour {contour_idx} in item {i}. Skipping this character.")
                    continue

                if char_w < min_size or char_h < min_size:
                    print(f"[CharacterSheetCropper] Info: Skipping character in item {i} due to small size (w={char_w}, h={char_h}). Minimum size is {min_size}.")
                    continue

                # Define crop coordinates, ensuring they are within original image bounds
                # cv2.boundingRect provides x, y, w, h relative to the original image
                crop_x_start = char_x
                crop_y_start = char_y
                crop_x_end = char_x + char_w
                crop_y_end = char_y + char_h

                # Slicing handles boundary conditions automatically (e.g., if char_x + char_w > img_w_orig)
                cropped_img_np_slice = img_np[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]
                cropped_mask_np_slice = current_mask_np[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

                if cropped_img_np_slice.shape[0] == 0 or cropped_img_np_slice.shape[1] == 0 or \
                        cropped_mask_np_slice.shape[0] == 0 or cropped_mask_np_slice.shape[1] == 0:
                    print(f"[CharacterSheetCropper] Warning: Zero-sized array after slicing for contour {contour_idx} in item {i}. Skipping this character.")
                    continue

                rgb_part = cropped_img_np_slice[..., :3]
                alpha_part = None

                # Alpha part: use original image's alpha if available, otherwise use the cropped mask
                if img_np.shape[2] == 4:
                    alpha_part = cropped_img_np_slice[..., 3:4]
                else:
                    alpha_part = cropped_mask_np_slice[..., np.newaxis]

                final_cropped_image_np = np.concatenate((rgb_part, alpha_part), axis=-1)

                # Normalize to target height, maintaining aspect ratio
                h, w = final_cropped_image_np.shape[:2]
                if h > 0 and target_height > 0:
                    aspect_ratio = w / h
                    new_w = int(target_height * aspect_ratio)
                    if new_w > 0:
                        pil_img = Image.fromarray((final_cropped_image_np * 255).astype(np.uint8), mode='RGBA')
                        pil_img = pil_img.resize((new_w, target_height), Image.LANCZOS)
                        final_cropped_image_np = np.array(pil_img).astype(np.float32) / 255.0

                img_out_tensor = torch.from_numpy(final_cropped_image_np.astype(np.float32)).unsqueeze(0)
                all_cropped_images.append(img_out_tensor)

        if not all_cropped_images and batch_size > 0:
            print("[CharacterSheetCropper] Warning: No valid character crops were generated for any item in the batch. Returning empty list.")

        return (all_cropped_images,)


NODE_CLASS_MAPPINGS = {
    "CharacterSheetCropper": CharacterSheetCropper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterSheetCropper": "VNCCS Character Sheet Cropper"
}
