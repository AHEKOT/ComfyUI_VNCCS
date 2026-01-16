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
- [Fix] CharacterCreatorV2: Fixed population of slider widgets (Age, Steps, CFG) and DMD checkbox during character load/init.
- [Fix] CharacterCreatorV2: Prevented unnecessary preview regeneration when running workflow with a just-loaded character (Smart Cache Fix). Added missing `pil2tensor`/`tensor2pil` helpers to backend.
- [Fix] CharacterCreatorV2: Added fallback logic to use existing character sheet (crop) if preview cache is missing but state is valid.
- [Fix] CharacterCreatorV2: Fixed Stale Cache issue by enforcing `preview_source` sync. Loading a character now invalidates old cache and forces a reload from the sheet.
- [Fix] CharacterCreatorV2: Fixed NSFW checkbox being ignored due to boolean type casting issues.

### Changed
- Refactored `vnccs_autofill.js` for better error logging.