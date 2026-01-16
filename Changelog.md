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
- **[DOCS]** **Strict Mode**: Updated `strict_mode/SKILL.md` with new "Anti-Panic & Error Protocol" (Section 8) to prevent ad-hoc fixes without planning.
- **[FIX]** **VNCCS_BackgroundPreview**: Fixed "Unsupported property type: uchar" error by converting colors to float32 in `save_gs_ply`.
- **[FIX]** **VNCCS_BackgroundPreview**: Fixed missing widget by refactoring `gaussian_preview.js` to correct extension path and node name bindings.
- **[REFACTOR]** **Web**: Rewrote `gaussian_preview.js` to use `GaussianPreviewWidget` class pattern aligned with VNCCS standards.
- **[OPTIMIZE]** **VNCCS_WorldMirror3D**: Added support for CPU offloading via `offload_scheme` (`model_cpu_offload`, `sequential_cpu_offload`) to reduce VRAM usage.
- **[OPTIMIZE]** **VNCCS_LoadWorldMirrorModel**: Changed default load device to `cpu` to prevent VRAM allocation before inference.
- **[FIX]** **VNCCS_WorldMirror3D**: Fixed `target_size` input to respect patch size (14px) divisibility. Logic enforces modulo and sets minimum reachable value (252).
- **[FIX]** **WorldMirror**: Resolved `RuntimeError: Expected all tensors to be on the same device` by enforcing explicit device casting for normalization buffers in `visual_geometry_transformer.py`.
- **[FIX]** **WorldMirror**: Resolved `RuntimeError: Input type (float) and bias type (BFloat16) should be the same` by explicitly casting images to match model dtype.
- **[FIX]** **VNCCS_WorldMirror3D**: Replaced unsafe `worldmirror.device` access with parameter-based check to prevent `AttributeError`.
- **[FIX]** **WorldMirror**: Implemented **Manual Block Offloading** for `sequential_cpu_offload` scheme. This resolves CUDA OOM on cards with <24GB VRAM by loading only one transformer block to GPU at a time.
- **[FIX]** **WorldMirror**: Resolved `RuntimeError: Expected all tensors to be on the same device` in manual offload mode. Conditioning embeddings and special tokens are now explicitly moved to the execution device before concatenation.
- **[FIX]** **WorldMirror**: Updated prediction heads (`CameraHead`, `DPTHead`) to handle CPU-based input tokens, effectively resolving device mismatches during manual offloading.
- **[FIX]** **WorldMirror**: Enforced aggressive chunking (`metrics=2`) for `gs_head` computation. This prevents OOM errors during the final Gaussian Splatting feature generation by processing frames in very small batches.
- **[FIX]** **WorldMirror**: Modified `DenseHead` to accumulate prediction results on CPU instead of GPU. This resolves the final OOM crash when concatenating massive feature tensors (e.g., 6.5GB+) for large context windows.
- **[FIX]** **VNCCS_Equirect360ToViews**: Aligned `output_size` constraints with `WorldMirror` patch requirements. Changed minimum size from 256 to 252 and max to 1022 to strictly follow the step size of 14, preventing resolution mismatch errors.
- `gsplat-bundle.js` – WebGL 3D Gaussian renderer
- `viewer_gaussian.html` – Standalone viewer HTML