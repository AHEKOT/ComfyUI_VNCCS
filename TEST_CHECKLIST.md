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
