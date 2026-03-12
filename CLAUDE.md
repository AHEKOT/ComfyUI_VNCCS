# VNCCS - Visual Novel Character Creation Suite

ComfyUI custom nodes package for creating consistent visual novel character sprites.

## Project Structure

- `__init__.py` - Entry point, registers `/vnccs/*` API endpoints on ComfyUI's PromptServer
- `utils.py` - Shared utilities: path management, config I/O, prompt helpers, seed management
- `nodes/` - Python node modules (~20 files), each exports NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
- `nodes/__init__.py` - Aggregates all node mappings
- `web/` - JavaScript widget files served by ComfyUI (WEB_DIRECTORY = "web")
- `pose_utils/` - Pose rendering and skeleton utilities
- `presets/poses/` - Pose preset JSON files
- `emotions-config/` - Emotion definitions (emotions.json)
- `character_template/` - Character sheet templates
- `background-data/` - WorldMirror/Gaussian splatting components
- `docs/` - ARCHITECTURE.md, NODES.md, WIDGETS.md

## Key Patterns

- Nodes follow ComfyUI convention: class with INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY
- Each node file exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` dicts
- Character data stored as JSON configs in `output/VN_CharacterCreatorSuit/{name}/`
- `utils.py` functions handle all path resolution and config I/O - always use them
- API endpoints registered via `@PromptServer.instance.routes` decorator

## Development Notes

- Python 3.11+, deps: torch, numpy, Pillow, opencv-python, huggingface_hub
- No test framework configured; `tests/` has ad-hoc test scripts
- Branch `cleanup` is current working branch; `main` is the release branch
- Published to ComfyUI Registry (PublisherId: vnccs)
