# VNCCS 3.0.2 Changelog

This changelog describes the changes in version `3.0.2` compared with `3.0.1`.
The release focuses on Clothes Designer preview execution, Control Center catalog freshness, and SAM3 batch stability.

## Headline Changes

- Clothes Designer previews now work with the `CUSTOM` Control Center configuration by executing the connected workflow branch, so externally supplied `MODEL`, `CLIP`, and `VAE` inputs are used correctly.
- Clothes Designer is now a valid ComfyUI partial-execution target, fixing the `Prompt has no outputs` error when generating a custom preview.
- SAM3 detail recovery now processes image batches one image at a time, fixing tensor-stack failures when different images produce different numbers of detections.
- The bundled Control Center catalog has been updated to the current model list and is automatically refreshed after a newer catalog is loaded from Hugging Face.

## Clothes Designer

- `Generate Preview` in `CUSTOM` mode now queues only the graph required to execute the connected Clothes Designer node instead of calling the standalone preview endpoint.
- Custom previews now use the model stack and settings supplied through the connected Control Center pipe.
- Preview generation waits for the `vnccs.preview.updated` event before refreshing the displayed image.
- Execution errors, interruptions, and preview timeouts are now reported by the Clothes Designer UI.
- Clothes Designer is now marked as an output node so ComfyUI accepts it as the destination of partial graph execution.

## Control Center and Model Catalog

- Updated the packaged `control_center.json` to match the current remote catalog.
- A successfully downloaded Control Center catalog is now written back to the packaged fallback file, preventing an older bundled catalog from reappearing after model updates or remote access failures.
- Packaged catalog updates use a lock and atomic file replacement so concurrent reads cannot observe a partially written JSON file.
- The packaged file is left untouched when the downloaded catalog has not changed, and synchronization failures no longer prevent Control Center from using the downloaded data.

## Chroma Key and SAM3 Recovery

- SAM3 recovery segmentation now runs separately for every image in a batch.
- This avoids `torch.stack` failures in Easy SAM3 when detection-box counts or shapes differ between batch items.
- The SAM3 model is loaded once, retained between images, and released after the final image in the batch.
- Recovered masks are normalized per image and concatenated back into the original batch order.

## Reliability and Maintenance

- Added regression coverage for atomic packaged-catalog synchronization and remote catalog refresh.
- Added checks that the bundled catalog selects Clothes Core `0.3.7` and no longer exposes the obsolete Emotion Core entry.
- Added frontend contract coverage for custom partial preview execution.
- Added a regression check that Clothes Designer remains a valid output target.
- Added batch recovery coverage that reproduces the variable-detection SAM3 failure.
- Updated test environment model-path stubs for the current Control Center behavior.
- Updated package version metadata from `3.0.1` to `3.0.2`.

# VNCCS 3.0.1 Changelog

This changelog describes the user-visible changes in version `3.0.1` compared with `3.0.0`.
It focuses on workflow behavior and UI changes, not internal refactors.

## Headline Changes

- SeedVR upscaling now exposes a color-correction selector in the generator UI.
- Emotion Studio and Character Creator V2 now use the same seed behavior as Clothes Designer: `seed = 0` can stay fixed, and randomization happens on queue only when random mode is enabled.
- Illustrious turbo mode now works consistently across Emotion Studio, Character Creator V2, and Control Center: enabling turbo sets `steps = 4` and `cfg = 1`, disabling it restores the previous values.
- Emotion Studio no longer uses a separate prompt-style selector in the widget header; prompt style is now derived automatically from the selected generation model.
- Control Center now shows Turbo LoRA as an inline selector in the `MODEL` section instead of a separate `Turbo Model` card.

## Character Generator

- Added a SeedVR color-correction selector with multiple modes such as `lab`, `adain`, `wavelet`, and `none`.
- Added help text so users can more easily switch away from `lab` when a GPU produces unwanted color shifts.

## Emotion Studio

- Removed the old top-left prompt-style selector from the widget.
- Prompt style is now chosen automatically from the selected generation mode: `Anima` mode uses the Anima prompt path, while `Illustrious` mode uses the SDXL-style prompt path.
- The seed field is now shared between Anima and Illustrious profiles so both modes show and use the same seed.
- Random seed mode now generates a new seed only when the workflow is queued, instead of rewriting the seed immediately in the UI.
- The Illustrious turbo toggle now stores the previous `steps/cfg`, switches to `4 / 1` while enabled, and restores the saved values when disabled.
- The LoRA section now stays available for both generation modes and updates its header and card set to match the active mode.

## Character Creator V2

- Seed handling now matches Clothes Designer semantics more closely.
- Default generation seed values are now initialized to `0` in fixed mode instead of being auto-randomized.
- Backend seed resolution now respects `seed_mode`, which keeps preview generation, pipe generation, and sampler execution consistent.
- The Illustrious turbo toggle now behaves like the other updated widgets: it saves previous `steps/cfg`, applies `4 / 1` while active, and restores the earlier values when turned off.

## Control Center

- `CUSTOM` pass-through mode now expects external `MODEL`, `CLIP`, and `VAE` inputs together, instead of showing the normal internal CLIP/VAE asset cards.
- Turbo LoRA has been moved into the `MODEL` block as a flat inline selector under the model and parameter area.
- The old standalone `Turbo Model` section has been removed.
- Turbo is no longer force-enabled just because `steps/cfg` happen to be `4 / 1`.
- Toggling Turbo LoRA on now saves the current `steps/cfg` and applies `4 / 1`; toggling it off restores the saved values.
- Hovering the Turbo LoRA selector no longer makes the toggle visually jump.
- Dependency/setup status can now show warning states more clearly instead of treating every non-OK condition as a hard failure.

# VNCCS 3.0.0 Changelog

This changelog describes the user-visible changes in the current branch compared with `main`.
It focuses on workflow and system behavior, not on internal code changes.

## Headline Changes

- VNCCS has moved from a collection of separate sheet-based workflows to a guided end-to-end character production pipeline.
- The main workflow is now built around Control Center, Character Creator V2, Character Cloner, Clothes Designer, Emotion Studio, Pose Studio, and Migration Assistant.
- Models and required workflow assets can now be downloaded and checked from VNCCS Control Center instead of being installed manually step by step.
- The new pipeline is no longer locked to the old fixed 12-pose character sheet format.
- Characters are now produced and managed as individual sprites, so pose count, sprite count, and sprite dimensions can vary by workflow and by character.
- Individual generated images can be regenerated without restarting the whole workflow.
- Existing VNCCS characters can be moved into the new format through the Migration Assistant.

## New Workflow Structure

- VNCCS 3.0 introduces a smaller and clearer workflow set:
  - Migration Assistant for old projects.
  - Character Creator for new characters.
  - Character Cloner for characters based on an existing image.
  - Character Clothes for outfit sets.
  - Character Emotions for expression sets.
- The old multi-step sheet workflow has been replaced by workflows that are closer to the actual creative process: create or clone a character, choose poses, generate sprites, add clothes, then add emotions.
- The old final sprite-extraction step is no longer a normal part of the flow because sprites are created directly.
- The old LoRA dataset generation workflow is no longer part of the main 3.0 production path.
- Workflow setup is less manual: the new UI widgets expose the choices users actually need instead of requiring them to edit many nodes directly.

## Control Center

- VNCCS Control Center is now the central place for preparing a workflow before generation.
- Users can choose a generation setup that matches their hardware and let Control Center download the required assets.
- Control Center shows whether required assets are already installed or missing.
- Control Center also helps detect incomplete setup, missing helper components, and authentication/token issues before the user starts a long generation.
- Generation settings such as quality, speed, optional style add-ons, and repeatability are now gathered into one shared workflow control area.
- This reduces the need to manually wire or edit many separate model and setting nodes in every workflow.

## Pose Studio

- VNCCS 3.0 workflows now use Pose Studio as the main pose authoring tool.
- Users can choose how many poses they need instead of being restricted to a fixed 12-pose sheet.
- The rest of the workflow now receives exactly the poses the user selected, which lets the pipeline work with any pose count.
- Users can create custom poses, adjust body proportions, age, height, body type, camera framing, and character proportions before generation.
- A reference image can be imported to extract or match a pose, making it easier to reproduce an existing stance.
- Pose Studio is used consistently across character creation, cloning, and clothing generation, so the same pose logic can carry through the whole project.

## Character Creation

- Character Creator V2 replaces the old first-step character setup with a more complete character design panel.
- Users can create a new character, select the generation style, and define the character from structured fields instead of editing raw prompts across the workflow.
- Tag builders are available for common character attributes, reducing prompt setup friction.
- Character Wizard can turn a natural-language character idea into structured character settings.
- The NSFW/base-clothing choice is now part of the character creation workflow instead of being handled as an afterthought.
- Generate Preview lets users test the character look before launching the full multi-pose generation.
- Preview generation can be repeated while editing tags, style, or character details, which makes iteration much faster.
- Existing generated sprites can be previewed from the creator UI, so users can quickly inspect the current character state.

## Character Cloning

- A dedicated Character Cloner workflow has been added for creating a VNCCS character from an existing image.
- Users can start from an image generated elsewhere, a downloaded character image, a screenshot, or their own art.
- The cloner can analyze the source image and help produce captions/tags instead of forcing the user to describe everything manually.
- Cloned characters use the same Pose Studio flow as newly created characters, so they can be converted into the same flexible sprite set.
- Users can optionally generate separate undressed/base sprites for cloned characters, which makes later outfit generation easier.
- Background color selection is now part of the clone workflow, helping avoid cleanup problems when the character has colors similar to the background.

## Clothing Workflow

- Character Clothes is now a dedicated workflow centered on Clothes Designer.
- Users can create as many outfit sets as they need for a character.
- Clothing is described through structured areas such as main clothes, headwear, face accessories, shoes, and extra details.
- Clothes Wizard can turn a simple clothing idea into a detailed outfit description.
- Clothes can now be cloned from a reference image, including an image of another character wearing the outfit.
- Generate Preview lets users check an outfit on the character before running the full pose set.
- Outfit details such as headwear and face accessories are carried forward so they can be respected later during emotion generation.
- The workflow can use an existing character sprite as the visual source for outfit generation, making clothing creation more consistent with the character's actual look.

## Emotion Workflow

- Emotion generation has moved into a dedicated Emotion Studio workflow.
- Users choose the character, the costumes to process, and the emotions to generate from a visual emotion library.
- Multiple costumes can be selected for emotion generation in one workflow.
- Custom emotions can be added when the built-in list does not contain the needed expression.
- Emotion generation works from the character's existing sprites instead of from a fixed sheet.
- Users can test a small subset of costumes and emotions first, then scale up once the settings look right.
- Face Detailer Denoise is exposed as an important creative control: lower values preserve the character more, higher values push the expression harder.
- Emotion prompts now take costume details into account, which helps preserve glasses, masks, hats, and other visible accessories.
- Emotion preview assets have been refreshed and moved to a lighter image format for the new visual selector.

## Sprite-Based Character Format

- Characters are now stored and used as individual transparent sprites rather than one large character sheet.
- Sprites are organized by character, costume, and emotion.
- New sprites can have different dimensions; the system normalizes canvases where needed instead of assuming every image comes from the same sheet layout.
- Generation can continue with whatever pose set the user selected, which is what makes arbitrary pose counts possible.
- Generated sprite outputs are easier to inspect, replace, and reuse outside VNCCS.
- Existing results are preserved through versioning when new output replaces an older set.
- Sprite loading and previewing now use the current sprite set directly, making the new format the default behavior across the workflow.

## Regeneration and Iteration

- VNCCS 3.0 adds regeneration for individual failed images.
- Users can regenerate a single sprite instead of rerunning the full character, clothing, or emotion workflow.
- Users can also restart generation from the part of the process that needs fixing, keeping earlier successful work intact.
- Regeneration is available from the generator UI with progress feedback.
- The workflow automatically updates the affected result after regeneration, so the user can keep iterating from the same screen.
- This is especially important for long runs with many poses, many outfits, or many emotions, where a single bad image used to waste the whole batch.

## Background Removal and Cleanup

- Background cleanup is now integrated into the main generators instead of being a separate manual concern.
- Users can choose cleanup strength presets depending on how aggressive the background removal should be.
- Detail recovery can be enabled when background removal damages important character details.
- This helps preserve eye color, clothing edges, hair details, and accessories that are close to the background color.
- Upscaling can be selected, changed, or disabled from the generator settings.
- The workflow gives stage previews and progress information so users can see where a generation currently is.

## Migration From Older VNCCS Projects

- VNCCS now includes a Migration Assistant workflow for old characters.
- Migration is explicit: old characters are not silently moved or modified during startup.
- Users can scan old VNCCS characters, select which ones to migrate, and run migration from the UI.
- Old character sheets can be converted into the new sprite-based format.
- Migration can also repair sprite canvas mismatches so old assets behave better in the new workflow.
- Users are expected to verify migrated characters before deleting old folders.
