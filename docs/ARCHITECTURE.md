# VNCCS Architecture Documentation

## Overview

**VNCCS (Visual Novel Character Creation Suite)** is a comprehensive set of custom nodes for ComfyUI designed for creating visual novel character sprites. The project provides a complete pipeline: from base character creation to generating finished sprites with emotions and costumes.

---

## Project Structure

```
ComfyUI_VNCCS/
├── __init__.py          # Main entry point, registers API endpoints
├── utils.py             # Utilities: config handling, paths, character management
├── nodes/               # Python nodes (15 files)
│   ├── __init__.py      # Node registration aggregator
│   ├── character_creator.py
│   ├── character_preview.py
│   ├── character_selector.py
│   ├── common_nodes.py
│   ├── dataset_generator.py
│   ├── emotion_generator.py
│   ├── emotion_generator_v2.py
│   ├── pose_generator.py
│   ├── sampler_scheduler_picker.py
│   ├── sheet_crop.py
│   ├── sheet_manager.py  # Sheet splitting/composing (SheetManager, SheetExtractor, QuadSplitter)
│   ├── sprite_generator.py
│   ├── vnccs_pipe.py
│   ├── vnccs_qwen_encoder.py
│   └── vnccs_utils.py    # Utility nodes (ChromaKey, ColorFix, Resize, MaskExtractor, RMBG2)
├── web/                 # JavaScript widgets (5 files)
│   ├── vnccs_emotion_v2.js
│   ├── vnccs_emotion_generator.js
│   ├── pose_editor.js
│   ├── pose_editor_3d.js
│   ├── pose_editor_3d_body.js
│   ├── pose_editor_3d_render.js
│   ├── pose_manager.js
│   └── bone_colors.js
├── pose_utils/          # Pose manipulation utilities
├── presets/             # Pose presets (JSON files)
├── emotions-config/     # Emotions configuration JSON
├── character_template/  # Character sheet templates
├── workflows/           # Ready-to-use workflow files
└── images/              # README images
```

---

## API Endpoints

Registered in `__init__.py`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vnccs/config` | GET | Get character configuration by name |
| `/vnccs/create` | GET | Create new character with default settings |
| `/vnccs/create_costume` | GET | Create new costume for character |
| `/vnccs/models/{filename}` | GET | Serve FBX model files for 3D pose editor |
| `/vnccs/pose_presets` | GET | Get list of available pose presets |
| `/vnccs/pose_preset/{filename}` | GET | Get specific pose preset file |
| `/vnccs/get_emotions` | GET | Get emotions configuration (registered in emotion_generator_v2.py) |
| `/vnccs/get_character_costumes` | GET | Get costumes for character |
| `/vnccs/get_character_sheet_preview` | GET | Get character preview image |
| `/vnccs/get_emotion_image` | GET | Get emotion preview image |

---

## Key Utilities (utils.py)

### Path Management
- `base_output_dir()` - Get base output directory (`output/VN_CharacterCreatorSuit`)
- `character_dir(name)` - Get character directory path
- `faces_dir(name, costume, emotion)` - Get faces directory path
- `sheets_dir(name, costume, emotion)` - Get sheets directory path
- `sprites_dir(name, costume, emotion)` - Get sprites directory path

### Character Management
- `list_characters()` - Get list of existing characters
- `load_config(character_name)` - Load character configuration JSON
- `save_config(character_name, data)` - Save character configuration
- `load_character_info(character_name)` - Load character info with sex/gender unification
- `ensure_character_structure(name, emotions, main_dirs)` - Create character directory structure
- `ensure_costume_structure(name, costume, emotions)` - Create costume directory structure

### Prompt Helpers
- `apply_sex(sex, positive_prompt, negative_prompt)` - Apply gender settings to prompts
- `age_strength(age)` - Calculate LoRA strength for age
- `append_age(positive_prompt, age, sex)` - Add age descriptors to prompt
- `build_face_details(char_info)` - Build face_details string from character info
- `dedupe_tokens(line)` - Remove duplicate tokens from prompt string

### Seed Management
- `generate_seed(value)` - Generate seed (if 0, creates new 64-bit non-zero seed)
- `inherit_seed(input_seed, upstream_seed)` - Inherit seed from upstream if input is 0

---

## Data Flow

1. **Character Creation**: `CharacterCreator` → saves config JSON → creates directory structure
2. **Costume Creation**: `CharacterAssetSelector` → updates config with costume data
3. **Emotion Generation**: `EmotionGenerator/V2` → generates emotion sheets → saves to Sheets/{costume}/{emotion}/
4. **Sprite Generation**: `SpriteGenerator` → crops sheets → saves to Sprites/{costume}/{emotion}/
5. **Dataset Creation**: `DatasetGenerator` → creates training dataset with captions
