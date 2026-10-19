# VNCCS Background Creator

Status: implementation authorized by the owner on 2026-10-19. This document
records the accepted two-stage design and supersedes the initial three-stage plan.

## Background

The first stage follows the Character Creator V2 layout, adapted to environments:
a background selector, a large expanding image preview and
generation/upload buttons on the left; a background wizard and the three scene fields
in the middle; generation controls on the right. Clicking the
preview opens a larger view of the current image. The generation panel is 20%
narrower than Character Creator's default width. The released width goes entirely
to the middle panel; the left panel retains its width and fixed 16:9 preview frame.
The former bottom prompt panels are absent; existing authored values remain in JSON. CG refinement
instructions belong to the Model stage and Save view dialog. Stage navigation
uses only the header selector.
Workflow selection, scene fields and the saved preview reference render before
server metadata is loaded. The selected name is cached in the workflow state;
legacy workflows retain their selected ID while the display name loads.

The background generator offers Qwen Image 2.1 and Moss in the shared Creator
model-type tabs. Use Character Creator's QI2 defaults for both
background generation and initial CG refinement: Qwen Image 2.1 INT8 ConvRot,
Qwen3-VL INT8 ConvRot, BF16 VAE, Euler/simple, 25 steps, CFG 1, prompt enhancement
and GPU INT8 cache. The optional Viggle Turbo adapter uses its six-step schedule.
The model card picker uses the shared Control Center catalog and installed
diffusion models, with the same VNCCS Models/User Models groups as Character
Creator. Selected model paths persist and reach the loader. The QI2 Turbo model
uses eight steps and CFG 1 without a second acceleration adapter. The LoRA stack
has five slots; restored additional entries are retained. Character-specific
adapters are never added automatically.

Moss follows the supplied text-to-image graph: `moss_image_sft_int8_convrot`,
`qwen3.5_9b_w4a8_gptq` loaded as `moss_image`, `flux2-vae_small`, separate positive
and negative `CLIPTextEncode`, `EmptyFlux2LatentImage`, and KSampler at 25 steps,
CFG 2.5, `er_sde`/`simple`, denoise 1. It uses the existing VNCCS background
enhancement system prompt, not the caption instructions embedded in the sample
workflow. No Qwen cache or Viggle acceleration is applied to Moss. Model-type
selection and both generation profiles live in the background's existing JSON;
old scenes without a type retain QI2. This addition concerns the background
generator only.

The prompt always uses four ordered fields: Style, Time, Interior/Exterior and
Scene description. The middle panel has a comma-separated style input,
scene type controls, and a large scene description editor. Base generation always uses Day; the former time control is absent. Aspect ratio
belongs to the generation panel. Editing a complete structured prompt updates
these fields; free-form text becomes its scene description. Aesthetics are folded
into the description without introducing another prompt heading.

Older descriptions, free-form prompt overrides and detailed attributes remain in
the stored editor. Their effective contents initialize `scene_description` once;
clearing the new field does not resurrect old details. No storage schema or asset
reference changes. The wizard returns the same four fields. Prompt enhancement
rewrites only the description and preserves the selected style and scene type while using Day in the actual encoder request. Both text-generation system prompts expand
sparse scene briefs into detailed English environments (about 100-180 words),
correct obvious typos such as `badroom` to `bedroom`, and describe spatial layout,
materials, furnishings and lighting consistent with the selected fields. Detailed
authored scenes keep their explicit details. The wizard uses the four-key JSON
contract; enhancement returns the four labeled prompt lines. Reference-frame CG
refinement keeps its separate camera and edit constraints.

Users may generate a
new background or upload an image inside the widget. There is no IMAGE input or
connected-image import action. Older workflows lose only that removed socket;
authored editor state and saved assets remain available.

## Model and CG views

The second stage has source/TripoSplat controls on the left, a large Gaussian
viewport in the center, and finished CG cards on the right. The viewport toolbar
contains only FOV, Save view and Recenter. There is no separate third stage.

TripoSplat preserves the complete input background. The original predicted source
camera is retained, adjusted for the input image's aspect and conditioning
padding. The working source view and Recenter use FOV 60 with a proportionally
closer distance: `new_distance = source_distance * tan(source_fov/2) / tan(30°)`.
This retains the image-plane framing while making camera composition practical;
depth perspective changes with the new lens. Previously untouched source poses
are converted once; authored viewpoints and saved CG cameras retain their exact
coordinates. Camera drag gain follows the actual visible FOV and displayed canvas
height, with continuous feedback and model reuse.

Save view captures the chosen camera at the selected aspect ratio, sends the raw
frame to QI2, and stores the finished image together with the raw frame, camera,
model reference and inference settings. The CG card shows the finished image.
Selecting a card restores its camera and dimensions. Multi-object editing, model
exports and other Factory editing features are outside this node's scope.

RUN follows the active stage. In Background it outputs the selected background
without switching to Model. Generated previews are reused only when the recorded
input key matches the effective prompt, active generation settings, model/LoRA
file signatures and enhancement instructions. Generate background records the
same key, so its preview can be reused by RUN. Missing, damaged or stale generated
previews replace the working preview. Uploaded images pass through without loading
models. Every path preserves the background's authored editor and TripoSplat
settings in its existing named JSON. Cache metadata belongs to each image entry;
there is no separate cache file or settings document.

In Model, RUN retains the selected completed CG output. It never propagates an
arbitrary intermediate camera view. Legacy workflows without an explicit stage
retain their completed-CG output when they reference one.

## Storage and preservation

The logical `vnccs/scenes` root resolves through ComfyUI's output directory:

```text
output/VNCCS/scenes/
  NAME/
    NAME.json
    NAME.json.backup
    background/<image_id>.png
    model/<model_id>/scene.json
    model/<model_id>/objects/...
    CG/<cg_name>/<cg_id>/
      raw.png
      image.png
```

`NAME.json` is the scene's canonical document (schema version 2). Its `images`,
`models` and `cgs` arrays retain saved assets. Each model records `source_image_id`;
each CG records both `source_image_id` and `model_id`, plus camera, dimensions and
generation settings. Reconstructing the same background appends a model variant;
previous models and CGs keep their IDs. The working preview is replaced in place
until a saved model or generated set references it. Those source files and legacy
images remain preserved without a preview history selector. Model `scene.json` files are immutable viewer manifests, while CG
metadata belongs in `NAME.json` rather than separate per-CG JSON files.

IDs make completed asset versions immutable. Readable scene and CG names
group their folders. JSON metadata has atomic writes, validated backups and
optimistic revisions. Archiving a selector entry retains every saved asset.
Workflow state pins selected asset IDs and preserves the current camera and editor
settings for ComfyUI restoration. Each background's named JSON is the library
record for its editor, generation and TripoSplat settings. Selecting a background
loads its settings from that record; edits update the same JSON silently, without
a separate save-status label or browser recovery copy. Failed writes retain the
visible edit and report the error. Older backgrounds recover TripoSplat defaults
from their latest model settings without rewriting the original JSON on read.
Corrupt metadata opens a validated backup read-only while preserving the original
bytes.

Earlier scaffold workflows remain readable. Opening them in the widget copies
their Factory model and reference into the new library without changing the
original scene. Existing saved output is still readable before that migration.
The earlier split `backgrounds/NAME`, `models/NAME` and `cg/NAME` library is copied
into `NAME/` on access. The migration discovers historical model variants, keeps
IDs/revisions unchanged, copies assets before publishing the new document, and
retries interrupted copies. Original files remain available for rollback; they
are never removed. Concurrent changes in the old library produce a conflict
instead of overwriting the new document. Rolling back code reads the retained
old library; it does not include later edits made only in the new layout.

## Acceptance and runtime validation

Automated checks cover storage containment and conflicts, backup preservation,
QI2 request construction/cancellation, saved-CG workflow output, upload preservation, predicted
source-camera decoding, widget restoration, serialization, continuous controls,
stale callbacks and cleanup. The final visual and GPU-dependent checks must run
in the owner's actual ComfyUI installation: text generation, image upload,
TripoSplat reconstruction, default camera alignment, reframing, CG refinement,
CG switching, cancellation and workflow reopen.

Future owner-supplied QI2 refinement settings may replace the current Character
Creator defaults. They do not require an additional interface stage.

## Main-package migration

The node is now owned by the main VNCCS package. Node logic, storage, jobs and routes
are in `nodes/background_creator.py`; state, widget, camera history and viewer are in
`web/vnccs_background_creator.js`. No Factory or UniCanvas code is imported. The only
separate runtime assets are the licensed TripoSplat inference core and local Spark/Three
renderer. Complete backgrounds do not need segmentation or torchvision. Existing main
VNCCS inference locks, runtime cleanup, ComfyUI node bridge and Viggle schedule are reused.

The saved node identifier, hidden widget order, HTTP paths, events, IDs, revisions and
output library root are unchanged. Earlier scaffold scenes are read from retained output
files; legacy import is retryable and does not modify those source scenes. Update the
main and Utils packages together to prevent duplicate registration.

## Rendered UI regression

For UI changes, run `tests/browser/background_creator.mjs` with Playwright on
`NODE_PATH`. Arguments are a Chromium executable, screenshot output directory,
and an optional saved scene directory. Without a scene it uses generated test
fixtures. With a scene it reads the source image and latest SPLAT without writing
to that library. API edits are held only in fixture memory.

The test imports the production widget, shared controls, fonts and Spark renderer.
It captures both stages, three widget widths, model selection, enlarged preview,
structured scene fields, Save view, working and empty states. Geometry checks cover paired
control alignment, horizontal overflow, label collisions and scroll retention.
The camera drag check uses the real WebGL viewer. Review the PNGs before reporting
UI completion; these captures verify the widget in Chromium, not ComfyUI's entire
host canvas or inference performance on the owner's GPU.


## Background variants and downstream generation

The editor owns a bounded `variants` list in the same background JSON. Defaults
are Morning, Evening and Night with preset times, empty weather and empty
additional details. Empty cards have their names above them; the inline editor
uses the same editable trait tags and preset buttons as Character Creator. Cards
stay in one horizontal strip with edge arrows. Selecting a card opens its name,
Time, Weather and Additional Details vertically below the card strip inside the
left source column; there is no variant editor
modal. Edits save directly to the selected background JSON. Old scenes gain these defaults on read without
losing their legacy time or other authored fields. Variant-only changes do not
invalidate the base preview generation cache. Day is represented by the original;
the untouched legacy default Day is removed, while authored Day cards remain intact.
New starts fresh variants rather than cloning the preceding background's cards.
Completed immutable sets publish an
update event after the JSON commit. Creator retrieves the saved set and displays
matching source/card/parameter previews, including after reopening the workflow.

The Creator appends a STRING Prompt socket with `OUTPUT_IS_LIST = (False, True)`;
image remains socket zero. Each list entry is JSON carrying the stable card ID,
name, time, weather, additional details, generated editing instruction, background
ID and source image ID. It is never a batched string or image.

Background Generator extends the existing generator backend and widget. Its
three stages are source_upscaler, background_generation, upscaler. It upscales
the original once, uses it as image one for every editing request from the live
Control Center pipe, and optionally upscales the new images. Gallery and IMAGE
list outputs keep Original first. Existing generators retain their own stages.
Native SeedVR controls, sampler inheritance, workflow state, scoped progress,
reconnect snapshots, stage cache and individual regeneration are reused.

Each completed set is written as immutable numbered PNGs below
`background/sets/SET_ID/`. The owning background JSON records variants, image
paths, settings and model metadata under generated_sets, and the latest generator
settings under generator_settings. Revision checks prevent stale publication;
partial writes are discarded without changing previous sets or source assets.
