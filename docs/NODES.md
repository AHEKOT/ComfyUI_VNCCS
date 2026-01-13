# VNCCS Nodes Reference

## Complete Node List (26 nodes)

### Character Management Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `CharacterCreator` | VNCCS Character Creator | VNCCS | `character_creator.py` | Creates base character with appearance settings, generates prompts and directory structure |
| `CharacterPreview` | VNCCS Character Preview | VNCCS | `character_preview.py` | Simplified version without sheet/face paths, supports batch processing with styles_list |
| `CharacterAssetSelector` | VNCCS Character Selector | VNCCS | `character_selector.py` | Selects existing character and costume, builds prompts from saved config |
| `CharacterAssetSelectorQWEN` | VNCCS Character Selector QWEN | VNCCS | `character_selector.py` | QWEN-specific version with additional qwen_description output |

---

### Emotion Generation Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `EmotionGenerator` | VNCCS Emotion Generator | VNCCS | `emotion_generator.py` | Text-based emotion selection, generates emotion sheets |
| `EmotionGeneratorV2` | VNCCS Emotion Studio | VNCCS | `emotion_generator_v2.py` | Visual UI for emotion/costume selection with grid interface |

---

### Sheet & Image Processing Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `VNCCSSheetManager` | VNCCS Sheet Manager | VNCCS | `sheet_manager.py` | Split sheets into parts (12 images in 2x6 grid) or compose images into sheets |
| `VNCCSSheetExtractor` | VNCCS Sheet Extractor | VNCCS | `sheet_manager.py` | Extract single part from sheet by index (0-11) |
| `VNCCSChromaKey` | VNCCS Chroma Key | VNCCS | `sheet_manager.py` | RGB-based green screen removal with auto-detect background |
| `VNCCS_ColorFix` | VNCCS Color Fix | VNCCS | `sheet_manager.py` | Adjust contrast and saturation, supports alpha channel |
| `VNCCS_Resize` | VNCCS Resize | VNCCS | `sheet_manager.py` | Resize image with chosen method (nearest, bilinear, bicubic, lanczos) |
| `VNCCS_MaskExtractor` | VNCCS Mask Extractor | VNCCS | `sheet_manager.py` | Fill alpha channel with color |
| `VNCCS_RMBG2` | VNCCS RMBG2 | VNCCS | `sheet_manager.py` | Background removal using multiple models (RMBG-2.0, Inspyrenet, BEN, BEN2) |
| `VNCCS_QuadSplitter` | VNCCS Quad Splitter | VNCCS | `sheet_manager.py` | Split/compose 4 quadrants (2x2 grid) |
| `CharacterSheetCropper` | VNCCS Character Sheet Cropper | VNCCS/Util | `sheet_crop.py` | Crop individual characters from sheet using contour detection |

---

### Pose Generation Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `VNCCS_PoseGenerator` | VNCCS Pose Generator | VNCCS/pose | `pose_generator.py` | Generate OpenPose images (512x1536) with interactive 2D/3D editor |

---

### Sprite & Dataset Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `SpriteGenerator` | VNCCS Sprite Generator | VNCCS | `sprite_generator.py` | Load all character sheets and prepare for sprite export |
| `DatasetGenerator` | VNCCS Dataset Generator | VNCCS | `dataset_generator.py` | Create LoRA training dataset with captions |

---

### Encoding & Pipeline Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `VNCCS_QWEN_Encoder` | VNCCS QWEN Encoder | VNCCS | `vnccs_qwen_encoder.py` | QWEN image-to-conditioning encoder for 3 images with weights |
| `VNCCS_Pipe` | VNCCS Pipe | VNCCS | `vnccs_pipe.py` | Aggregation pipe node with model/clip/vae/conditioning/sampler/scheduler |
| `VNCCSSamplerSchedulerPicker` | VNCCS Sampler Scheduler Picker | VNCCS | `sampler_scheduler_picker.py` | Dynamic ComfyUI sampler/scheduler enumeration picker |

---

### Utility Nodes

| Node Class | Display Name | Category | File | Description |
|------------|--------------|----------|------|-------------|
| `VNCCS_Integer` | VNCCS Integer | VNCCS | `common_nodes.py` | Integer passthrough node |
| `VNCCS_Float` | VNCCS Float | VNCCS | `common_nodes.py` | Float passthrough node |
| `VNCCS_String` | VNCCS String | VNCCS | `common_nodes.py` | String passthrough node |
| `VNCCS_MultilineText` | VNCCS Multiline Text | VNCCS | `common_nodes.py` | Multiline text passthrough node |
| `VNCCS_PromptConcat` | VNCCS Prompt Concat | VNCCS | `common_nodes.py` | Concatenate up to 4 strings with separator |

---

## Node Details

### CharacterCreator

**Inputs:**
- `existing_character` (required) - Character name dropdown
- `background_color`, `aesthetics`, `nsfw`, `sex`, `age`, `race`, `eyes`, `hair`, `face`, `body`, `skin_color`, `additional_details`, `seed`, `negative_prompt`, `lora_prompt`, `new_character_name` (optional)

**Outputs:**
- `positive_prompt` (STRING)
- `seed` (INT)
- `negative_prompt` (STRING)
- `age_lora_strength` (FLOAT)
- `sheets_path` (STRING)
- `faces_path` (STRING)
- `face_details` (STRING)

---

### EmotionGeneratorV2

**Inputs:**
- `prompt_style` (required) - "SDXL Style" or "QWEN Style"
- `character` (required) - Character name dropdown
- `costumes_data` (required) - JSON string of selected costumes (hidden, managed by widget)
- `emotions_data` (required) - JSON string of selected emotions (hidden, managed by widget)

**Outputs:**
- `images` (IMAGE list)
- `face_prompts` (STRING list)
- `sheet_save_paths` (STRING list)
- `face_save_paths` (STRING list)
- `sheet_prompt` (STRING)
- `face_details` (STRING)
- `seed` (INT)
- `masks` (MASK list)

---

### VNCCS_PoseGenerator

**Inputs:**
- `pose_data` (required) - JSON string with 12 poses (managed by pose editor widget)
- `line_thickness` (optional, default 3) - OpenPose line thickness
- `safe_zone` (optional, default 100) - Safe zone percentage (0-100)

**Outputs:**
- `openpose_grid` (IMAGE) - 12 OpenPose images in 6x2 grid (3072x3072)

---

### VNCCS_QWEN_Encoder

**Inputs:**
- `clip` (required) - CLIP model
- `prompt` (required) - Text prompt
- `vae`, `image1`, `image2`, `image3` (optional) - VAE and up to 3 images
- `target_size`, `upscale_method`, `crop_method`, `instruction` (optional)
- `image1_name`, `image2_name`, `image3_name` (optional) - Image labels
- `weight1`, `weight2`, `weight3` (optional) - Image weights (0.0-1.0)
- `vl_size`, `latent_image_index`, `qwen_2511` (optional)

**Outputs:**
- `positive` (CONDITIONING)
- `negative` (CONDITIONING)
- `latent` (LATENT)
