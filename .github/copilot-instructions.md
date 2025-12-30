# VNCCS Development Guide

## Project Overview
VNCCS (Visual Novel Character Creation Suite) is a ComfyUI custom node suite for creating consistent character sprites for visual novels. It uses Stable Diffusion to generate characters through a multi-stage workflow (base character → costumes → emotions → sprites → dataset).

## Architecture

### ComfyUI Node Pattern
All nodes follow ComfyUI's class-based pattern with specific class methods:
- `INPUT_TYPES`: ClassMethod defining inputs with type constraints `(TYPE_NAME,)` or `(["option1", "option2"],)` 
- `RETURN_TYPES`: Tuple of type strings e.g. `("IMAGE", "STRING", "INT")`
- `RETURN_NAMES`: Tuple matching RETURN_TYPES for UI display
- `FUNCTION`: String name of the processing method
- `CATEGORY`: "VNCCS" for grouping in ComfyUI UI

Example from `character_creator.py`:
```python
RETURN_TYPES = ("STRING", "INT", "STRING", "FLOAT", "STRING", "STRING", "STRING")
FUNCTION = "create_character"
CATEGORY = "VNCCS"
```

### Character Data Management
Character configuration uses a hierarchical JSON structure stored at `ComfyUI/output/VN_CharacterCreatorSuit/{character_name}/{character_name}_config.json`:

```python
{
    "character_info": {...},  # Base traits (age, sex, race, eyes, hair, etc.)
    "costumes": {             # Costume-specific prompts
        "Naked": {...},
        "School Uniform": {...}
    },
    "folder_structure": {
        "main_directories": ["Sprites", "Faces", "Sheets"],
        "emotions": ["neutral", "happy", "sad", ...]
    },
    "config_version": "2.0"
}
```

**Critical**: Always preserve existing config sections when updating. Use `load_config()` → modify → `save_config()` from `utils.py`.

### Directory Structure Convention
Output follows strict hierarchy: `{character}/{'Sprites'|'Faces'|'Sheets'}/{costume}/{emotion}/{type}_{emotion}_####_{seed}.png`

Helper functions in `utils.py`:
- `base_output_dir()`: Resolves ComfyUI output directory
- `character_dir(name)`, `faces_dir(name, costume, emotion)`, `sheets_dir(...)`, `sprites_dir(...)`
- `ensure_character_structure(name, emotions, main_dirs)`: Creates full directory tree

### Web Integration & REST API
`__init__.py` registers aiohttp endpoints on ComfyUI's PromptServer for bidirectional communication with frontend:
- `/vnccs/config?name={char}`: Get character config JSON
- `/vnccs/create?name={char}`: Create/check character existence
- `/vnccs/create_costume`: Add costume to character
- `/vnccs/models/{filename}`: Serve 3D skeleton models
- `/vnccs/pose_presets`: List pose presets
- `/vnccs/pose_preset/{filename}`: Load specific pose

All endpoints use `_vnccs_register_endpoint()` with lazy registration to avoid import errors during ComfyUI analysis.

### Frontend Extensions
JavaScript widgets in `web/` extend ComfyUI nodes:
- `pose_editor.js`: 2D pose editing canvas with skeleton manipulation
- `pose_editor_3d.js`: Three.js-based 3D pose editor using skeleton models from `/vnccs/models/`
- `vnccs_autofill/vnccs_autofill.js`: Auto-populates node fields from character configs
- `vnccs_emotion_generator.js`: Emotion selection UI

Frontend communicates via registered endpoints and ComfyUI's widget system (`app.registerExtension()`).

### Image Processing Patterns
Images always use ComfyUI's tensor format: `torch.Tensor` shaped `[batch, height, width, channels]`
- RGB: `[1, H, W, 3]` with values `0.0-1.0`
- RGBA: `[1, H, W, 4]` for transparency support
- Masks: `[batch, H, W]` single channel

**Critical conversions** in `utils.py::load_sheet_image()`:
```python
# PIL → NumPy → PyTorch
image_np = np.array(img_pil).astype(np.float32) / 255.0  # Normalize to 0-1
img_tensor = torch.from_numpy(image_np)[None,]  # Add batch dimension
```

Use `torch.nn.functional.interpolate()` for resizing (requires `[B, C, H, W]` format - permute before/after).

### Age System
Age affects generation via LoRA strength calculated by linear interpolation over control points in `utils.py::age_strength()`:

```python
AGE_CONTROL_POINTS = [(0, -5.0), (3, -4.0), ..., (80, 5.0)]
```

Also modifies prompts via `append_age()` adding tokens like `"loli"`, `"young woman"`, `"mature female"`, `"old woman"` based on age ranges.

## Development Workflows

### Testing Changes
1. ComfyUI auto-reloads on node changes - refresh browser
2. Check console for errors during node initialization
3. For web/ changes, hard refresh browser (Cmd+Shift+R)
4. Test workflows are in `workflows/` - use latest versions (v5.json files)

### Adding New Nodes
1. Create node class in `nodes/{feature}_node.py`
2. Define `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` dicts
3. Import and merge in `nodes/__init__.py`
4. Follow import pattern with try/except for relative imports:
```python
try:
    from ..utils import function_name
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils import function_name
```

### Working with Pose System
Poses use OpenPose BODY_25 format (17-25 keypoints depending on mode). Joint coordinates are `[x, y]` on 512x1536 canvas.

Default skeleton in `web/pose_editor.js::DEFAULT_SKELETON` defines initial positions. Rendering pipeline:
1. JavaScript editor → JSON joints data
2. `pose_utils/pose_renderer.py::render_schematic()` or `render_openpose()` 
3. Returns RGBA numpy array `[H, W, 4]`
4. Convert to torch tensor for ComfyUI

3D poses use Three.js with skeleton models served from `models/` directory via `/vnccs/models/` endpoint.

### VNCCS Pipe Pattern
The `VNCCS_Pipe` node aggregates common parameters (model, clip, vae, conditioning, seed, steps, cfg, denoise, sampler, scheduler) to reduce connection clutter. Nodes should:
- Accept individual inputs OR pipe input
- Inherit from pipe when individual input is None/default
- Pass pipe through for chaining

Seed logic: `pipe.seed` inherited only if incoming `seed==0`, otherwise override.

### Sampler/Scheduler Handling
Fetch available options dynamically via `sampler_scheduler_picker.py::fetch_sampler_scheduler_lists()` which queries ComfyUI's registered samplers. Default to `DEFAULT_SAMPLERS` and `DEFAULT_SCHEDULERS` if unavailable.

## Code Conventions

### Import Patterns
Always use try/except for relative imports to support both package usage and standalone execution:
```python
try:
    from ..utils import helper_function
except ImportError:
    from utils import helper_function  # Fallback for direct execution
```

### Config Management
Never overwrite entire config - always merge:
```python
config = load_config(character_name) or {"character_info": {}, ...}
config["character_info"]["new_field"] = value
config.setdefault("costumes", {})  # Preserve existing sections
save_config(character_name, config)
```

### Prompt Engineering
Use weighted syntax `(token:weight)` for emphasis. Standard weights:
- Strong emphasis: `(term:1.5)` - `(term:2.0)`
- Normal: `(term:1.0)` or just `term`
- De-emphasis: `(term:0.8)` - `(term:0.5)`

Combine prompts with `dedupe_tokens()` from utils.py to avoid token duplication.

### Error Handling
Nodes should gracefully handle missing dependencies (lazy imports):
```python
try:
    from comfy.node_helpers import conditioning_combine
except ImportError:
    conditioning_combine = None  # Provide fallback
```

Print informative errors prefixed with `[VNCCS]` or `[VNCCS {NodeName}]` for traceability.

## External Dependencies

### Required Models
Project requires specific Stable Diffusion models in ComfyUI directories (see README.md):
- LoRAs: `vn_character_sheet_v4.safetensors`, `dmd2_sdxl_4step_lora_fp16.safetensors`
- ControlNet: `AnytestV4.safetensors`, `IllustriousXL_openpose.safetensors`
- Face detection: `face_yolov8m.pt`, `face_yolov9c.pt`

### Python Dependencies
Core: `torch`, `numpy`, `Pillow`, `opencv-python` (opencv-python-headless acceptable)
Optional: `huggingface_hub` for model downloads

## Common Pitfalls

1. **Tensor shape mismatches**: Always check batch dimension `[B, H, W, C]` vs `[H, W, C]`
2. **Config overwrites**: Use `load_config()` first, never create fresh dicts
3. **Path separators**: Use `os.path.join()`, never hardcode `/` or `\\`
4. **Seed randomization**: Respect `seed==0` convention for "randomize on next generation"
5. **Node enumeration types**: RETURN_TYPES uses type names (strings), INPUT_TYPES uses actual tuples/lists for dropdowns
