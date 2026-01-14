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