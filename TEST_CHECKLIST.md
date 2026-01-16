# Test Checklist

## Visual Novel Character Creator Suite (VNCCS) Tests

### Character Creator V2
- [ ] **Instantiation**: Add `CharacterCreatorV2`. Verify default size is LARGE (~1400x950).
- [ ] **UI Layout**: Verify 3 Columns.
- [ ] **LoRA Logic**: 
    - Verify "DMD2 LoRA" is a dropdown (not text).
    - Select a LoRA + Strength.
    - Generate Preview -> Should work.
    - Reload Page -> **Settings should persist** (Model, LoRA, etc.).
    - Change Character -> LoRA setting should **NOT** change (independent of char).
- [ ] **Data Loading**: Select character -> Attributes fill.
- [ ] **Persistence**: Queue Prompt -> Check JSON -> Data saved (except LoRA).
- [ ] **Outputs**: Connect `lora_name` / `lora_strength` outputs to a downstream node. Verify values.

### Regression
- [ ] **Frontend**: Check `EmotionGeneratorV2` still loads.
- [ ] **Backend**: Check standard `CharacterCreator` (V1) still works if present.
