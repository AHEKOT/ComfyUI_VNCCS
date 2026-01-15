# Test Checklist - 2026-01-14

## 1. Emotion Generator V1
- [ ] Run Emotion Generator with multiple emotions
- [ ] Check console: ensuring `negative_prompt` does **NOT** grow with each iteration

## 2. Emotion Studio (V2)
- [ ] Open Emotion Studio
- [ ] Uncheck one costume
- [ ] Reload page (Ctrl+Shift+R)
- [ ] The checkbox should **NOT** automatically re-check itself

## 3. Character Creator
- [ ] Open Character Creator
- [ ] Enter a name in `new_character_name`
- [ ] Click "Create New Character"
- [ ] Character is created, fields are autofilled

## 4. Character Selector
- [ ] Select a character
- [ ] Costume list updates correctly
- [ ] Switch costumes
- [ ] Costume fields (face, head, top...) are updated

## 5. Agent Skills (Optional)
- [ ] Verify ComfyUI loads without JS errors

## 6. Sheet Manager
- [ ] Test VNCCSSheetExtractor with any sheet image
- [ ] Verify it returns correct part (1-12 index, where 1=top-left, 12=bottom-right)
- [ ] Test VNCCS_QuadSplitter split mode with a square image
- [ ] Test VNCCS_QuadSplitter compose mode with 4 images

## 7. VNCCS Pipe
- [ ] Create chain: Pipe1 (model) → Pipe2 (clip) → Pipe3 (vae)
- [ ] Verify all values inherited correctly through the chain
- [ ] Override a value in middle pipe, verify downstream receives override

## 8. 3D Background Generator
- [ ] Load WorldMirror model (requires GPU)
- [ ] Test 360° Panorama to Views with equirectangular image
- [ ] Run 3D reconstruction on sample images
- [ ] Save PLY file with rotation options
