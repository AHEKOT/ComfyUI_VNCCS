# Changelog

## [Unreleased]
### Added
- Feature: Added `CharacterCreatorV2` node with standalone GUI and on-demand preview generation (640x1536).
- Feature: `CharacterCreatorV2` now dynamically updates character list after creating a new character.
- Tests: Added unit tests for `CharacterCreatorV2` backend.

### Fixed
- [Fix] CharacterCreatorV2: Removed redundant `pipe` input and output.
- [Fix] CharacterCreatorV2: Added `Steps` and `CFG` inputs to GUI.
- [Fix] CharacterCreatorV2: Fixed aspect ratio aspect-ratio css to `640/1536`.
- [Fix] CharacterCreatorV2: Patched `PromptServer` error (`last_prompt_id`) during on-demand preview.
- [Fix] CharacterCreatorV2: Fixed empty "Character" dropdown list.

### Changed
- Refactored `vnccs_autofill.js` for better error logging.