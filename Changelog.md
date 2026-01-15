# Changelog - Session 2026-01-14

## Bug Fixes

### 🔴 Critical

| File | Issue | Fix |
|:-----|:------|:----|
| `nodes/emotion_generator.py` | `negative_prompt +=` inside loop caused infinite accumulation | Added `base_negative_prompt`, used assignment `=` instead of `+=` inside loop |
| `web/vnccs_emotion_v2.js` | Costume selection reset on page reload | `fetchCharacterData` now checks for saved state in `costumesDataWidget` |

---

## Refactoring

### `web/vnccs_autofill/vnccs_autofill.js`
- **Removed**: Dead code `addCreateButton` (~55 lines)
- **Added**: Helper `updateWidgetValue(widget, value)`
- **Renamed**: `addCreateButtonSafely` → `addCreateButton`
- **Result**: Reduced from 503 to 454 lines

### `nodes/sheet_manager.py`
- **Optimized**: `VNCCSSheetExtractor.extract()` now crops directly instead of splitting entire sheet (12x memory reduction)
- **Simplified**: `VNCCS_QuadSplitter._normalize_image_list()` rewritten with cleaner recursion (33 → 18 lines)
- **Removed**: 5 debug `print()` statements from `VNCCSSheetManager.compose_sheet()`
- **UX**: `VNCCSSheetExtractor.part_index` changed from 0-11 to human-friendly 1-12

### `nodes/vnccs_pipe.py`
- **Removed**: Unused `__init__` method (12 lines)
- **Removed**: Mutation of input `pipe` object (prevents race conditions)
- **Added**: Helper `_inherit(value, pipe, attr_name, zero_is_empty)` for cleaner inheritance logic
- **Result**: Reduced from 168 to 143 lines

---

## Agent Skills

### `.agent/skills/strict_mode/SKILL.md`
Added rules:
- **No Deletion Before Modification** – Prevent file deletion for overwriting
- **No Chat-First Plans** – Plans must be artifacts first
- **No Low-Effort Plans** – Comprehensive detailed plans required
- **Language Settings** – Russian for chat/plans, English for code/docs

### `.agent/skills/code_review/`
- Created new code-review skill

### All Skills
- Added YAML headers
- Created `examples/` directories and files

---

## Analysis (No Changes)

| File | Status | Verdict |
|:-----|:-------|:--------|
| `nodes/pose_generator.py` | ✅ | Clean, no changes needed |
| `web/pose_editor.js` | ⚠️ | 2144 lines, requires refactoring (deferred) |
| `nodes/character_selector.py` | ⚠️ | 90% duplication, plan ready (deferred) |

---

## New Features

### `nodes/background_generator.py` [NEW]
Integrated 3D Background Generation nodes from standalone project:
- **🌐 Load WorldMirror Model** – Load HunyuanWorld-Mirror for 3D reconstruction
- **🏔️ WorldMirror 3D Reconstruction** – Generate point clouds and gaussian splats
- **🔄 360° Panorama to Views** – Extract perspective views from equirectangular panoramas
- **💾 Save PLY File** – Export reconstruction as PLY with rotation options
- **👁️ Background Preview** – Interactive 3D Gaussian Splatting viewer

### `background-data/worldmirror/` [NEW]
WorldMirror ML model files and utilities.

### `web/gaussian_preview/` [NEW]
WebGL viewer files for interactive 3D Gaussian Splatting preview.
- `gaussian_preview.js` – ComfyUI widget integration
- `gsplat-bundle.js` – WebGL 3D Gaussian renderer
- `viewer_gaussian.html` – Standalone viewer HTML