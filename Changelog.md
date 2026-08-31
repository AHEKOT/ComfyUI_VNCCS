# VNCCS 3.1.1 Changelog

This changelog describes the final user-visible and release-level changes in version `3.1.1` compared with `main` (`3.1.0`).
The release focuses on secure model delivery, Comfy Registry compliance, and safer Control Center dependency installation.

## Headline Changes

- Hardened VNCCS for Comfy Registry security requirements and added a mandatory security gate to the release workflow.
- Model downloads are now limited to public Hugging Face repository assets and no longer accept or store access tokens.
- Control Center now respects ComfyUI-Manager installation policy, including remote-server restrictions, restart handling, and automatic installation resume.
- Removed BEN2 background-removal support and its bundled implementation.
- Updated package metadata to version `3.1.1`.

## Secure Model Downloads

- Control Center, Character Cloner, SeedVR2, QwenVL, SAM3, and background-removal model downloads now use the reviewed Hugging Face download path.
- QwenVL, SAM3, SeedVR2, and bundled background-removal assets use fixed repositories and pinned revisions where defined by VNCCS.
- Downloads explicitly disable implicit Hugging Face credential discovery.
- Removed Hugging Face and Civitai token fields, token-saving endpoints, and authentication dialogs from Control Center.
- Obsolete VNCCS token-storage files are removed automatically from both current and legacy locations.
- Direct URL and Civitai downloads are no longer supported. Control Center catalog entries must provide public `hf_repo` and `hf_path` values.
- Authentication-required assets are reported as unavailable instead of asking the user to enter a key.
- Character Cloner now uses the shared validated QwenVL asset downloader instead of maintaining a separate network download implementation.
- Removed the direct `requests` dependency from the package.

## Control Center and ComfyUI-Manager

- Dependency installation now uses ComfyUI-Manager's registered package queue and supports both current and legacy Manager APIs.
- Control Center checks Manager's active `security_level`, `network_mode`, and ComfyUI listener address before requesting an installation.
- VNCCS never lowers ComfyUI-Manager's security level automatically.
- When a non-local ComfyUI server requires `network_mode = personal_cloud`, Control Center explains the security impact and requires explicit confirmation.
- If confirmed, only the Manager `network_mode` setting is changed, the previous configuration is backed up, and the file is replaced atomically.
- Pending dependency installations survive the required restart and resume automatically when Control Center reconnects.
- Restart handling works with both current and legacy Manager endpoints and reloads the page after ComfyUI becomes available again.
- Installation progress and completion are tracked through both current task events and legacy queue-status events.
- Module status now relies on the local ComfyUI backend instead of fetching version metadata directly from GitHub.

## Runtime Hardening

- Removed dynamic code execution from the bundled BiRefNet implementation and replaced it with explicit supported backbone, decoder, and refiner mappings.
- Sampler, scheduler, and SeedVR attention discovery now use direct guarded imports instead of dynamic module loading.
- Removed environment-variable overrides for QwenVL, SAM3, and background-removal download sources and revisions.
- Replaced frontend callback binding patterns with explicit receiver-preserving wrappers without changing queue hooks or widget behavior.
- Removed BEN2 from the available RMBG model list; `RMBG-2.0`, `INSPYRENET`, and `BEN` remain available.

## Release Security Gate

- Added a fail-closed source scanner covering credential handling, raw network clients, external command execution, dynamic execution/imports, unsafe URLs, privilege escalation markers, and removed BEN2 code.
- The security scan runs before the test suite in CI and CI now runs on every push, including `main`.
- Added regression tests that verify every mandatory scanner rule and detect attempts to weaken or bypass the gate.
- Security scanner files, their tests, and the CI workflow now require project-owner review through `CODEOWNERS`.

## Compatibility Notes

- Existing public Hugging Face model downloads continue to work without configuration changes.
- Private, gated, token-authenticated, direct-URL, and Civitai catalog downloads are intentionally unsupported in `3.1.1`.
- Workflows that explicitly selected `BEN2` must switch to `RMBG-2.0`, `INSPYRENET`, or `BEN`.
- Existing Control Center dependency installation remains available, but remote or shared ComfyUI servers may require the new explicit Manager policy confirmation and restart flow.

# VNCCS 3.1.0 Changelog

This changelog describes the changes in version `3.1.0` compared with `3.0.4`.
The release adds FLUX.2 Klein 9B support, moves SeedVR2 upscaling to native ComfyUI nodes, makes Step 3 emotion generation memory-bounded, and substantially improves chroma-key cleanup and Control Center setup.

## Headline Changes

- Added FLUX.2 Klein 9B as a complete generation family with its own model, text encoder, VAE, helper LoRAs, conditioning encoder, and Control Center profile.
- Replaced the external SeedVR2 custom-node path with ComfyUI's native SeedVR2 nodes and added guided model downloads directly to Generator Settings.
- Step 3 now lets users choose which character poses receive emotions and processes large emotion jobs in memory-bounded task batches.
- Chroma Key now removes connected and enclosed screen remnants more reliably, produces cleaner edge colors, and can optionally recover foreground details with SAM3.
- Character Creator now offers three persistent Anima resolution presets up to `1024 × 2456`.
- Control Center can install missing custom-node dependencies through ComfyUI-Manager and guide the user through the required restart.

## FLUX.2 Klein 9B

- Added a `Flux Klein9b` family tab to Control Center alongside `QIE2511`.
- Added catalog entries for the FLUX.2 Klein 9B FP8 diffusion model, Qwen 3 8B text encoder, Flux 2 VAE, VNCCS Pose Studio Klein9b LoRA, and VNCCS Clothes Core Klein9b LoRA.
- Added the `VNCCS Flux Klein Encoder` node, built from native ComfyUI conditioning nodes.
- The Klein encoder accepts text plus up to three optional reference images, scales and encodes each connected reference, validates image dimensions, and creates a correctly sized Flux 2 latent.
- Character Generator and Clothes Designer now select the Klein encoder automatically when the connected pipe uses the Klein family.
- Pose generation and clothing generation now select helper LoRAs that match the active model family instead of reusing an incompatible QIE2511 LoRA.
- Clothes Designer can trace the connected Control Center through intermediate pipe nodes and keeps its Clothes Core LoRA synchronized when the active family changes.
- Control Center stores model type, selected model, sampling parameters, and helper assets separately for each family, so switching families no longer overwrites the other family's setup.
- `CUSTOM` mode now resolves its context model from the active family: GGUF for QIE2511 and UNET for Klein9b.

## Native SeedVR2 Upscaling

- SeedVR2 now runs through the native `SeedVR2Preprocess`, `SeedVR2Conditioning`, and `SeedVR2PostProcessing` nodes included in current ComfyUI releases.
- The old custom SeedVR loader and video-upscaler nodes are no longer required by VNCCS.
- Generator Settings now shows native SeedVR2 model cards with installed, missing, downloading, and error states.
- Added one-click downloads for the official 3B, 7B, and 7B Sharp FP16 or mixed FP8/FP16 models from the pinned SeedVR2 repository revision.
- The required SeedVR2 VAE is downloaded automatically when it is not installed.
- Existing settings that reference legacy `seedvr2_ema_*` or GGUF models automatically migrate to the recommended `seedvr2_3b_fp8_e4m3fn.safetensors` model.
- SeedVR output sizing is now explicit: `target short edge` controls the shorter dimension, while `maximum edge` caps the longer dimension without changing the aspect ratio. A maximum edge of `0` disables the cap.
- Each source image is upscaled independently instead of being interpreted as a video-frame batch.
- Progress now advances per image, and previews can be appended incrementally during long runs.
- Native node availability is checked before queueing. When ComfyUI is too old, the UI identifies the missing nodes and asks the user to update and restart ComfyUI.

## Emotion Studio and Step 3

- Added a `Generate poses` selector to Emotion Studio with individual pose toggles, `SELECT ALL`, and `CLEAR ALL` actions.
- Pose selection is serialized in the workflow and restored when the workflow is reopened.
- The confirmation dialog now reports the selected emotion, costume, and pose counts together with the resulting image total.
- Only selected source poses are loaded and expanded into emotion tasks; clearing every pose blocks the queue with a clear validation message.
- Emotion tasks are now processed in small batches selected from available VRAM, system RAM, source resolution, and task count.
- Added an advanced `task_batch_size` setting. `0` chooses a safe size automatically, while manual values are capped when they exceed the detected memory budget.
- Full-resolution source sprites are loaded lazily per task instead of being decoded and duplicated for every selected emotion in advance.
- Raw results and masks are cached per item, completed output files are reused when possible, and successful cache files are preserved during later batches or regeneration.
- Generator previews are reduced in size and emitted incrementally, while full-resolution sprites continue to be saved to the character directory.
- Full-resolution output tensors are retained only when the corresponding `IMAGE` output is connected. The bundled Step 3 workflow therefore stays within the active task window instead of accumulating the entire run in memory.
- CPU and GPU caches are released between task batches, substantially reducing out-of-memory failures on large costume, pose, and emotion combinations.
- Changing the pose or emotion task list invalidates incompatible stage cache data without discarding unrelated saved character output.

## Chroma Key and SAM3 Detail Recovery

- Retuned the `soft`, `balanced`, `strong`, `aggressive`, and `maximum` cleanup presets for a safer tolerance scale and cleaner silhouettes.
- Added connected-component cleanup for screen-colored fringes and broad residual background patches connected to the image border.
- Enclosed regions that are confidently classified as background can now be removed without erasing similarly colored opaque foreground details.
- Edge decontamination now replaces screen-contaminated RGB with nearby trusted foreground colors, reducing green, blue, or red halos around hair, clothing, and outlines.
- `Despill Strength` now controls all edge color correction consistently; setting it to zero no longer leaves hidden decontamination active.
- Improved matte cleanup, foreground recovery, edge choke, and color-bleed behavior while preserving the original image dimensions and output modes.
- Added an optional `Use SAM3 Recovery Mask` input to `VNCCS Chroma Key`.
- SAM3 recovery now evaluates individual detected objects, keeps only masks with sufficient overlap with the existing foreground, and restores their interior details without manufacturing a new contour.
- SAM3 mask outputs with varying ranks, wrapper axes, object counts, or spatial resolutions are normalized automatically.
- Batch recovery runs one image at a time while keeping the model loaded, avoiding third-party tensor-stack failures when images have different detection counts.
- If SAM3 is missing, incompatible, or fails during recovery, VNCCS falls back to normal chroma keying instead of failing the generation.
- Easy SAM3 recovery is marked unsupported on macOS because its current dependencies require decord and Triton; normal chroma keying remains available there.

## Character Creator

- Added Anima resolution presets to Generator Settings:
  - `Normal`: `640 × 1536`.
  - `High`: `856 × 2048`.
  - `Maximum`: `1024 × 2456`.
- The selected Anima resolution is saved in the mode profile and restored when switching between Anima and Illustrious.
- Missing or invalid legacy resolution values safely fall back to `Normal`.
- The higher presets clearly warn that they require more VRAM and generation time.
- Automatic race, body, and skin-color hint detection now follows the documented English input vocabulary consistently.

## Control Center and Setup

- Missing dependencies can now be installed individually or with `Install all` through ComfyUI-Manager and Comfy Registry.
- Control Center supports both current and legacy ComfyUI-Manager queue APIs and reports Manager rejection or installation errors in the UI.
- Successful installations are tracked until completion, after which Control Center shows a restart-required dialog with `Later` and `Restart server` actions.
- Dependency detection now checks ComfyUI's active node registry and all configured custom-node roots instead of assuming a single installation directory.
- Platform compatibility warnings are displayed separately from missing or partially loaded dependencies.
- Model downloads now respect ComfyUI's configured folder paths for diffusion models, text encoders, VAEs, and LoRAs.
- Downloads are staged beside their final destination, validated, and installed with an atomic replacement to avoid cross-volume moves and partially visible model files.
- Download request and worker errors are surfaced in both the Control Center interface and server log.
- Mutable Control Center data such as user configuration, custom LoRA records, and installed-version records is now stored under the ComfyUI user directory when available. Existing portable-install files remain readable for compatibility.
- Family-specific assets are filtered by the active QIE2511 or Klein9b tab, while global utility assets remain shared.

## Compatibility and Maintenance

- Package metadata has been updated from `3.0.4` to `3.1.0`.
- Existing QIE2511 workflows remain the default and legacy Control Center model selections are migrated into the QIE2511 family state.
- Character Cloner's missing-source-image validation is now consistently shown in English in both the backend and UI.
- Added regression coverage for Klein conditioning, model-family state, native SeedVR sizing and downloads, Step 3 batching and pose filtering, Anima resolutions, dependency installation, and the new chroma-key recovery paths.

# VNCCS 3.0.4 Changelog

This changelog describes the changes in version `3.0.4` compared with `3.0.3`.
This release restores the missing Face Detailer denoise control in VNCCS Emotions Generator.

## Headline Changes

- The `Face Detailer Denoise` slider is visible in the Emotions Generator interface again.
- Denoise is stored as a local emotion-generation setting and is passed directly to FaceDetailer.
- The slider once again shows weak, optimal, and excessive strength zones for the active generation model.
- New Emotions Generator nodes and the bundled Step 3 workflow use a default denoise value of `0.55`.

## Emotions Generator

- Restored the dedicated `Emotion Strength` panel that was accidentally removed in `3.0.3`.
- Changing the slider now updates the serialized `face_denoise` setting instead of relying on the connected pipe value.
- Existing workflows without `face_denoise` remain compatible and receive the default value automatically.

# VNCCS 3.0.3 Changelog

This changelog describes the changes in version `3.0.3` compared with `3.0.2`.
The release focuses on complete generator configuration, safer emotion background cleanup, and more precise SAM3 detail recovery.

## Headline Changes

- Character Creator, Character Cloner, Clothes Generator, and Emotions Generator now provide a dedicated `Generator Settings` modal for their internal processing controls.
- Generator settings can be restored with `Load Defaults` and are written only after the user confirms them with `Apply`.
- Emotion generation now preserves the original sprite outside the FaceDetailer region instead of chroma-keying the complete image.
- SAM3 detail recovery now evaluates individual detected objects and rejects background objects before restoring image details.
- The bundled Step 3 Character Emotions workflow now uses the updated FaceDetailer defaults.

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
