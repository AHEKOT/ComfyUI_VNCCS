Ready for review
Select text to add comments on the plan
Plan: VNCCS Widget Improvements — Full Scope
Context
Character Creator V2 widget has seed persistence bugs (always shows 0), broken preview state tracking, and event listener leaks. Other widgets share similar issues: silent error handling, memory leaks, inconsistent styling, missing cache busters. A reference widget (vnccs_pose_studio.js from VNCCS_Utils) demonstrates improved patterns (debounce, pending state queue, error feedback, cleanup on removal) that should be ported across.

Phase 1: Create web/vnccs_common.js — Shared Utility Module
New file: web/vnccs_common.js

Extract reusable patterns used by 4+ widgets:

1.1 debounce(fn, delay)
From pose_studio pattern (lines 6565-6576)
Used to throttle saveState() on text/slider input events
1.2 createLoadingOverlay(container, message)
Dark overlay + spinner + animated dots
Currently duplicated in: character_creator_v2, clothes_designer, sprite_manager
Returns { overlay, remove() } for cleanup
1.3 showModal(container, title, contentFunc, buttons)
Duplicated in: character_creator_v2 (line 613), character_cloner (line 353), clothes_designer (line 194)
Supports Cancel/Confirm/Danger button styles
1.4 showMessage(container, text, isError)
From pose_studio (lines 6197-6226)
Auto-dismissing info/error notifications
Replace silent catch(e) {} blocks across all widgets
1.5 safeFetch(url, options, container)
Wraps api.fetchApi() with try/catch + user-visible error modal on failure
Returns null on error (instead of throwing)
1.6 registerCleanup(node, cleanupFn)
Helper to safely hook into node.onRemoved without clobbering existing handler
Accumulates multiple cleanup functions per node
1.7 syncWidgetData(node, widgetName, data)
Sets widget value + calls widget.callback() + marks canvas dirty
From pose_studio pattern (lines 7248-7252)
Import pattern: import { debounce, showModal, ... } from "./vnccs_common.js";

Phase 2: Fix Character Creator V2 Bugs
Files: web/vnccs_character_creator_v2.js, nodes/character_creator_v2.py

2.1 Seed always shows 0
JS line 1129: After loadState(), if seed === 0 AND seed_mode === "fixed", generate random seed:
if (g.seed === 0 && g.seed_mode !== "randomize") {
    g.seed = Math.floor(Math.random() * 10000000000000);
}
els.seed.value = g.seed;
Python line 585: Replace seed=int(gen_settings.get("seed", 0)) with seed=generate_seed(int(gen_settings.get("seed", 0))) using existing utils.generate_seed()
Python line 499: Same fix for pipe.seed_int
2.2 Preview source/validity state fix
JS line 1171: Change saveState(!!state.character) → saveState(false) (don't assume preview is valid just because character exists)
JS lines 1239-1248: Add els.previewImg.onload handler that calls saveState(true) when image actually loads
JS lines 1225-1231: In tryCache() error handler, explicitly set state.preview_valid = false
2.3 Event listener leak fix
JS line 1063: Store api.addEventListener("vnccs.preview.updated", handler) reference
Add node.onRemoved cleanup via registerCleanup() from vnccs_common
Same pattern for any DOM event listeners
2.4 Replace inline showModal/loadingOverlay with imports from vnccs_common
Replace showModal definition (line 613) with import
Replace loading overlay creation (lines 1263-1269) with createLoadingOverlay()
2.5 Add debounced save
Wrap oninput handlers for text fields and sliders with debounce(saveState, 300)
Keep onchange for selects/checkboxes as immediate
Phase 3: Fix Other Widgets — Memory Leaks & Error Handling
3.1 web/vnccs_panorama_mapper.js
Lines 343, 378: Replace window.onmousemove = ... and window.onmouseup = ... with addEventListener + cleanup via registerCleanup()
Add cache busters for image loads
Add error handling for image load failures
3.2 web/vnccs_sprite_manager.js
Lines 647-649: Track setTimeout IDs, clear on node.onRemoved
Line 937: Track fallback timeout (60s), cancel if cleanup fires first
Line 757: Track scroll event listener, remove on cleanup
3.3 web/vnccs_clothes_designer.js
Lines 160, 176, 643, 651, 855: Replace silent catch(e) {} with console.warn + optional showMessage() for user-facing errors
Line 874: Fix api.addEventListener leak — add cleanup via registerCleanup()
3.4 web/vnccs_emotion_v2.js
Line 429-432: Replace silent catch(e) {} with warning log
Line 555: Add cache buster &t=${Date.now()} for emotion images when reloading after generation
Line 556: Improve error fallback beyond just img.style.display = 'none'
Add cleanup for any event listeners on node removal
3.5 web/vnccs_character_cloner.js
Replace inline showModal with import from vnccs_common
Replace inline loading overlay with import
Add event listener cleanup on node removal
Phase 4: Visual Consistency & UI/UX Polish
4.1 Standardize font sizing
vnccs_emotion_v2.js uses 24px — align with other widgets (14-16px base)
vnccs_clothes_designer.js uses 14px — keep as-is
vnccs_character_creator_v2.js uses zoom:0.67 approach — document as standard
4.2 Consistent color scheme
Standardize on: #1e1e1e (bg), #333/#444 (borders), #3558c7 (primary blue), #d32f2f (danger red)
Fix vnccs_sprite_manager.js using #1a1a1a variant
4.3 Loading indicators for image grids
Add simple spinner/placeholder while emotion grid images load (from pose_studio loading overlay pattern)
Files Modified (Summary)
File	Changes
web/vnccs_common.js	NEW — shared utilities
web/vnccs_character_creator_v2.js	Seed fix, preview state, event cleanup, import common
nodes/character_creator_v2.py	generate_seed() call for seed=0
web/vnccs_panorama_mapper.js	Global listeners → addEventListener + cleanup
web/vnccs_sprite_manager.js	Timeout tracking + cleanup
web/vnccs_clothes_designer.js	Error handling, event cleanup, import common
web/vnccs_emotion_v2.js	Cache busters, error handling, font fix
web/vnccs_character_cloner.js	Import common, event cleanup
Existing Functions to Reuse
utils.generate_seed(value) — utils.py:226-233 — for Python-side seed=0 handling
api.fetchApi() — ComfyUI's built-in fetch wrapper — base for safeFetch()
app.graph.setDirtyCanvas(true, true) — ComfyUI canvas refresh after widget updates
Verification
Seed: Create new CharacterCreatorV2 node → seed field should show non-zero random value, not 0
Seed fixed mode: Set seed to specific value → generate → reload workflow → same seed preserved
Seed randomize mode: Set mode to "randomize" → generate → seed changes each time
Preview: Select character → preview loads from sheet/cache → reload workflow → preview still visible
Preview after generate: Click GENERATE → preview updates → reload → cached preview shown
Delete button: Delete character → should work (endpoint exists, just verify)
Memory: Open/close panorama mapper nodes repeatedly → no console errors or leaked listeners
Error feedback: Disconnect from server → trigger fetch → user sees error modal, not silent failure
Visual: Check all widgets have consistent font sizes and color scheme
Implementation Order
web/vnccs_common.js (foundation for everything else)
web/vnccs_character_creator_v2.js + nodes/character_creator_v2.py (main bug fixes)
web/vnccs_panorama_mapper.js (dangerous global listener bug)
web/vnccs_sprite_manager.js (timeout leaks)
web/vnccs_clothes_designer.js (error handling + event leak)
web/vnccs_emotion_v2.js (cache busters + font)
web/vnccs_character_cloner.js (common imports)
Visual consistency pass across all files