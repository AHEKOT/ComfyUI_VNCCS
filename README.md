![Header](images/README.png)

---
<table>
<tr>
<td width="50%" align="center">
<strong>Join The Community</strong><br>
Share results, ask questions, and follow VNCCS updates.<br><br>
<a href="https://discord.com/invite/9Dacp4wvQw" target="_blank"><img src="images/VNCCS_Discord_Button.png" alt="Join our Discord"></a>
</td>
<td width="50%" align="center">
<strong>Support VNCCS</strong><br>
VNCCS is developed independently. Support helps keep the project moving.<br><br>
<a href="https://www.buymeacoffee.com/MIUProject" target="_blank"><img src="images/VNCCS_Donate_Button.png" alt="Support VNCCS"></a>
</td>
</tr>
</table>

---

VNCCS is NOT just another workflow for creating consistent characters, it is a complete pipeline for creating sprites for any purpose. It allows you to create unique characters with a consistent appearance across all images, organise them, manage emotions, clothing, poses, and conduct a full cycle of work with characters.

## Description

Many people want to use neural networks to create graphics, but making a unique character that looks the same in every image is much harder than generating a single picture. With VNCCS, it's as simple as pressing a button (just 4 times).

## Installation

For new ComfyUI users: 
Use [VNCCS Easy Install](https://github.com/AHEKOT/VNCCS_Easy-Install). It installs ComfyUI with a preconfigured setup and working VNCCS nodes.

For experienced users:
Find `VNCCS - Visual Novel Character Creation Suite` in Custom Nodes Manager and click install latest version

Manual install:
1. Rename the downloaded node folder to `vnccs` and place it into `ComfyUI/custom_nodes/`. The resulting path must be `ComfyUI/custom_nodes/vnccs/`.
2. Alternatively, in the console: go to `ComfyUI/custom_nodes/` and run `git clone https://github.com/AHEKOT/ComfyUI_VNCCS.git vnccs`
3. `cd vnccs`
4. `pip install -r requirements.txt`
5. `python install.py`

Run both commands with the Python interpreter that starts ComfyUI. ComfyUI-Manager
runs `install.py` as its installation hook.

Post install steps:
Launch ComfyUI and open Comfy Manager
Click "Install missing custom nodes"

## VNCCS 3.2 Workflow

Hi! My name is V-chan, and I am going to show you how to use the new VNCCS!

We got a BIIIIIIG update, and now everything is completely new, so listen carefully!

## Step 1: Character Creator

Open the workflow:

`VNCCS_3.2_Step1_CharacterCreator.json`

Let's start from the very beginning. The first thing you need to do, besides opening the workflow, silly, is figure out the **VNCCS Control Center**.
![Header](images/v3/ReadMe1.png)

Inside it, you will find the models that turn your character into different poses and outfits. Choose the one that fits your computer and press **Download**.

- **Qwen Image 2.1** works from your character image to change poses and clothes. It can even give you a transparent background. Less green-screen trouble, yay!
- **Flux Klein9b** makes your sprites in just 4 steps. Want less waiting while trying poses and outfits? Give it a try!
- **MiniMax H3** is a video model, but here it helps you make character sprites! Try it for poses and clothes too, and see which result you like more.

Choose wisely, but in the end nobody is stopping you from trying them all and deciding later.

And then, at the very bottom of the widget, you can click the big **Download** button and the magic will do everything by itself.

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/controlcenter.webp" width="100%" alt="Downloading model components in VNCCS Control Center">
</td>
<td width="50%" valign="top">
<strong>Let the Control Center do the downloading!</strong><br>
Choose the model components you need and press <strong>Download / Update</strong>. The model cards show download progress and installation status, so you can see when everything is ready for your character adventures.
</td>
</tr>
</table>

Next, go to **VNCCS Character Creator V2**.

![Header](images/v3/ReadMe2.png)
The most important thing here is to create a new character and choose the model for generation.

**Illustrious** may be considered old, but it makes excellent characters and has a huge selection of LoRAs for every style and occasion. Do not worry about quality, it will not disappoint you!

**Anima** is a new and cool model. It can do almost everything, but it will need a bit more resources, and there are not as many LoRAs for it yet.

**Qwen Image 2.1** is here too! It can create your character with a transparent background, and the **Turbo** option cuts down the number of generation steps. More time to try different characters, hehe!

I recommend trying all three and deciding for yourself.

Right now you do not have any characters yet, so press **NEW** and give him or her a name! The name is very important!!! Be creative and unique!

Done? Good job! Now you have two paths:

1. Enter character descriptions or use the pencil buttons to choose curated presets for race, skin tone, body type, face, hair, eyes, and details. Choose sex, age, and generation type. The **NSFW** switch controls whether the base character will have clothes or not :3
2. Press **CHARACTER WIZZARD**, describe the character you want, and after a little magic the system will set all the needed options by itself. Do not forget to check them!

Race presets include natural-language descriptions of their distinctive anatomy, added automatically to generation prompts in Illustrious, Anima, and Qwen Image 2.1. Select multiple species for hybrids, or enter custom traits; the prompt gives explicit character traits priority over preset defaults. Existing character fields and custom text remain supported. Breast-size presets retain their original tags. Creator V2 uses `character_template/character_presets_v2.json`; the legacy catalog remains available to Character Cloner.

Click the **Style** card to open the style library. Built-in styles and their 1024×1024 WebP previews ship with the node in `character_template/character_styles.json` and `character_template/style_previews/`. Style prompts describe artistic rendering; background, pose and framing remain separate Creator settings.

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/Styles_anima.webp" width="100%" alt="Choosing an Anima style and generating a character preview">
</td>
<td width="50%" valign="top">
<strong>Give your character a new drawing style!</strong><br>
With <strong>Anima</strong> selected, open the style library, pick a style card, and press <strong>GENERATE PREVIEW</strong>. Watch the preview appear with the chosen style. Try a few looks before committing to a whole set of sprites!
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/Styles_anima_blend.webp" width="100%" alt="Adding a second Anima style and previewing the blended result">
</td>
<td width="50%" valign="top">
<strong>One style is good. What about two?</strong><br>
Click the <strong>+</strong> card beside the first style and choose another one from the library. Generate a new preview to see how Anima combines them. The animation starts with the first style's preview and ends with the mixed result. Time for a little experimenting, hehe!
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/styles_QI2.webp" width="100%" alt="Searching and filtering Qwen Image 2.1 styles before generating a preview">
</td>
<td width="50%" valign="top">
<strong>Find your look with Qwen Image 2.1!</strong><br>
Browse the style previews, search by name, or narrow the library with the category filter. Choose a style and press <strong>GENERATE PREVIEW</strong> with <strong>Qwen Image 2.1</strong> selected to try it on your character. So many looks to choose from!
</td>
</tr>
</table>

Choose **Custom style** or **New style** to enter a name, short description, reference and style prompt. **Save style** adds it to **My styles**; **Generate preview** saves it and renders only that style using the current character tags and generation settings, a square portrait and seed 0. **Resolution scale** controls the render quality before the preview is resized to 1024×1024. Previews are composited onto the same dark gradient as the style cards and saved as opaque WebP at quality 90 directly inside the node, rather than ComfyUI's output directory. User styles live in `character_template/character_styles.user.json`, with `user_*.webp` previews alongside the built-in images; these user files are excluded from Git and packaged updates.

The **×** button in the upper-right corner of a user style card opens a deletion confirmation. Confirming removes its library entry and preview, if present. Built-in styles cannot be deleted. Deleting the selected style switches the Creator to its default style.

Style edits carry the `revision` returned by `/vnccs/character_styles`. A stale edit returns HTTP 409 and keeps the editor draft open. Keep any draft text before reloading the current style.

If a user style changes during preview generation, the render returns HTTP 409 and preserves the previous preview. Generate the preview again for the updated style.

A new little feature is the **GENERATE PREVIEW** button. It lets you see what the character will look like without restarting the whole generation. So press it already, and if you like everything, move on. If you want to make changes, edit the descriptions and press it again!

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/character_preview.webp" width="100%" alt="Generating a character preview inside VNCCS Character Creator V2">
</td>
<td width="50%" valign="top">
<strong>Meet your character before the big generation!</strong><br>
Press <strong>GENERATE PREVIEW</strong> and watch the character appear in the Creator's preview panel. Check the face, hair, body, and overall look. Something does not feel right? Change the character descriptions and try again until you are happy with the result!
</td>
</tr>
</table>

Creator and Cloner run with the character settings restored from the workflow and save them to the selected character's profile. Restoring a workflow keeps its draft fields; it does not require a profile version. Storage locks and atomic writes protect the save, while unreadable configurations and metadata belonging to another character are rejected.

## VNCCS Pose Studio

The next key node is **VNCCS Pose Studio**.
![Header](images/v3/ReadMe3.png)

It is downloaded from my second project, **VNCCS-Utils**, so do not forget to install that too!

Here, the most important thing is to choose the poses you need and how many of them there should be. You can control the model however you want and make absolutely any poses. Also, using the **Import** button, you can load any picture with a character and get a pose just like the one in the picture!

It is also very important that the body proportions of the model fit your character. Age will be set automatically from the previous step, but nobody will stop you from setting it manually. Also choose height and body type, the result will be much better that way.

In **VNCCS Character Generator**, you do not really need to worry about the settings, but if you want, you can choose the upscaler model or even turn it off.
![Header](images/v3/ReadMe4.png)

In **BG Remove**, you can choose a chroma key preset. **Balanced** is a very good preset, but if it is not enough for you, or if it is too much and starts damaging the character, choose a lighter one.

**SAM3 Details Recovery** makes background removal slower, but it lets you worry less about eye color and clothing elements that are the same color as the background. We will talk about clothes a little later.

Ready? Then press **Run**! Now just wait, and the magic will do everything for you!
![Header](images/v3/ReadMe5.png)

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/Pose_gen.webp" width="100%" alt="Pose guides and a generated character sprite appearing in the generation preview">
</td>
<td width="50%" valign="top">
<strong>From a pose guide to a character sprite!</strong><br>
After you press <strong>Run</strong>, the generation preview shows the pose guides and the finished character sprite as it becomes ready. Check how the character follows the pose and whether the proportions look right. Your character is getting ready to move into the story!
</td>
</tr>
</table>

## Step 1.1: Character Cloner

Open the workflow:

`VNCCS_3.2_Step1_CharacterCloner.json`

This workflow is basically a complete copy of Character Creator, but it is made for cloning existing character images.

Did you generate the character with another model? Download it from the internet? Take a screenshot from your favorite anime? Draw it yourself, with your own hands? Good job!

Try to make sure the picture is good quality and that the character is full body, otherwise the model will invent everything that is not visible in the picture!

![Header](images/v3/ReadMe6.png)

Now load it into **VNCCS Character Cloner**, write tags or press **ANALYZE CAPTIONS**, set up everything you need in **VNCCS Pose Studio**, and do not forget to choose whether you need separate undressed sprites with the **NSFW** button. They are not mandatory, but dressing these characters later will be MUCH easier!

Also try to choose a background color that appears the least in the character. Look at the eyes and hair. If they are green, choose blue. Or the other way around.

Now press **Run** and look at the result!

## Step 2: Character Clothes

Open the workflow:

`VNCCS_3.2_Step2_CharacterClothes.json`

Now we move to the tastiest part! In this workflow, you will make clothes for the character. As many sets as you think you need.

Choose a character in **VNCCS Clothes Designer** and press **New**. Give the outfit set a name and get ready to create!

![Header](images/v3/ReadMe7.png)

You have two options again:

1. Describe all clothing elements in the needed fields. You do not have to follow them exactly, but the **head** and **face** sections will help you later not to lose details during emotion generation, so do not slack off! If the character has glasses, write them in **face**. A hat goes in **head**. Easy!
2. Press **Clothes Wizzard** and simply describe the clothes you want!

You also have an option to clone any clothes from any picture! Open the **CLONE CLOTHES** tab and upload an image of clothes, or a character wearing clothes.

The **GENERATE PREVIEW** button will help you see what your character will look like before starting the big and heavy generation of all poses.

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/new_clothes.webp" width="100%" alt="Creating a costume, filling its fields with Clothes Wizzard, and generating an explorer outfit preview">
</td>
<td width="50%" valign="top">
<strong>A new outfit, from idea to preview!</strong><br>
Choose your character, press <strong>New</strong>, and give the costume a name. Open <strong>Clothes Wizzard</strong>, describe the outfit, and press <strong>FILL FIELDS</strong>. Here, a short explorer-clothes idea becomes descriptions for the top, bottom, shoes, head, and face. Check those fields, then press <strong>GENERATE PREVIEW</strong> to see the character dressed for an adventure. A hat, boots, goggles... ready to go!
</td>
</tr>
</table>

And that is all! Again, do not forget about **VNCCS Pose Studio**, and press **Run**!

When you finish the first set, you can press **New** again and create as many outfits as you want!

Costume saves detect changes made in another editor. A conflicting save preserves your local draft instead of replacing newer metadata or recreating a deleted costume. Reload the costume before retrying; retain any draft text you want to apply to the current version.

API clients must read `/vnccs/get_costume` and send its `ETag` header as the `revision` field when posting an edit to `/vnccs/save_costume`. Successful saves return the next `revision`; stale edits return HTTP 409, and edits without a revision return HTTP 428. Creating a new costume uses `create_only: true`.

Deleting and recreating a costume changes its version even when its name and fields are identical. Clothes Generator rejects the protected `Naked` and `Original` sets and requires an editable costume.

## Step 3: Character Emotions

Open the workflow:

`VNCCS_3.2_Step3_CharacterEmotions.json`

Here we will create emotions for the character! Up to this moment, all sprites had a calm facial expression. This will be our base.

In the **VNCCS Emotion Studio** widget, choose the character you are going to make emotions for. In **Selected costumes**, choose all costumes you want to work with.

![Header](images/v3/ReadMe8.png)

Now, in the huge list, click the emotions you need and add them to the selected ones. Also, if by some miracle you did not find what you need, you can add a **Custom** emotion and describe what you want yourself.

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/emotions.webp" width="100%" alt="Selecting emotion cards and using search to find another expression in VNCCS Emotion Studio">
</td>
<td width="50%" valign="top">
<strong>Pick the expressions your story needs!</strong><br>
Click emotion cards to add them to the selected list above the library. Use the search field to find a particular expression without scrolling through every card. The animation builds a small selection of different moods. A little pout, a happy smile, maybe something more mischievous?
</td>
</tr>
</table>

After that, you again need to decide which model will do the generation.

**Illustrious** is stable, but the variety of emotions depends very strongly on the exact character, style, and model. Not all emotions may come out equally well.

**Anima** makes very cool emotions, but it is still too young, so it can be unstable. It can change character details too much, so try it yourself and decide what you like better.

**Qwen Image 2.1** can make emotions too! It works on the face and puts it back into your sprite, keeping the rest of the image in place. Describe the expression you want and give it a try!

![Header](images/v3/ReadMe9.png)


If you chose **Illustrious** or **Anima**, the most important setting in **VNCCS Emotions Generator** is **Face Detailer Denoise**. It will suggest optimal values by itself, but remember one basic idea: the higher the denoise, the more the original image changes.

More denoise means a brighter emotion, but the character may stop looking like themselves.

Less denoise means a more accurate character, but the emotion may be weaker.

There is no ready-made recipe here. It all depends on the character and the selected model, so be creative and a tiny bit more patient.

For the first try, do not select all costumes and emotions at once. It will take a long time, and if the result does not satisfy you, it will be sad. Better find the optimal settings first, and then go all in!

Press **Run**, and may luck be with you!

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="images/v3.3/emotions_gen.webp" width="100%" alt="Generating the yandere-smile expression across multiple character poses">
</td>
<td width="50%" valign="top">
<strong>Now let your character show some feelings!</strong><br>
Here, <strong>Qwen Image 2.1</strong> applies the selected <strong>yandere-smile</strong> expression to the character's poses. The pose cards update as each sprite is generated, while the progress bar shows the batch status. Check the faces before moving on to a bigger batch!
</td>
</tr>
</table>

At this point, the current VNCCS features end, but not for long! Planned features include animations, 3D environments and CG image creation inside them, character voice generation, and music track generation for your game or project.

You will find all your sprites inside ComfyUI in:`ComfyUI\output\VNCCS\Characters\` folder

Be careful with them, and do not delete them by accident while cleaning your disk!
![Header](images/v3/footer.png)


## Background Creator

**VNCCS Background Creator** generates and manages backgrounds in one workspace,
using the same Sakura controls as the other Creators. The left column contains
the background selector, fixed 16:9 preview, upload action and variant cards;
the middle column contains the wizard and scene fields; the right column contains
generation settings. There is no workspace header or stage selector.

Qwen Image 2.1 and MOSS-Image use the shared Creator model cards and retain separate
generation profiles in the selected background's JSON. Qwen defaults to INT8
ConvRot diffusion and text encoder weights, BF16 VAE, Euler/simple, 25 steps,
CFG 1, prompt enhancement and GPU INT8 cache. The optional Viggle Turbo adapter
uses six steps and CFG 1, resolves its file and installation status through Control
Center, and restores the previous Steps/CFG when disabled. The built-in Qwen Turbo
model defaults to eight steps without the Viggle adapter. The seed dice toggles
fixed/random mode; random mode chooses a new displayed seed before each preview
and queued RUN. Five LoRA slots retain their positions; character adapters are
never added automatically. Resolution uses the shared Creator slider: 1–4 MP in
0.1 MP steps, displayed to one decimal place. Pixel area uses the Creator scale
(MP × 1024²); dimensions retain the selected aspect ratio exactly.

Moss uses MOSS-Image SFT INT8 ConvRot, Qwen3.5 9B W4A8 GPTQ (`moss_image` encoder),
Flux2 small VAE, 25 steps, CFG 2.5, `er_sde`/`simple` and an empty Flux2 latent.
It uses the VNCCS background prompt enhancer. Qwen cache and Viggle controls are
hidden for Moss. Install these files in ComfyUI's registered model folders:

- `diffusion_models/moss_image_sft_int8_convrot.safetensors`
- `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors`
- `vae/flux2-vae_small.safetensors`

The middle panel contains Style, Interior/Exterior and Scene description. Style
uses a simple dropdown from the shared Qwen style library, including My styles,
plus Custom for existing free-text settings. Only the visual rendering description
is inserted into prompts. Catalog Reference fields and Reference lines in user
style prompts are excluded; no style-reference block is appended. The base
background always uses Day. Prompts retain four ordered fields:

```text
Style: Anime, digital art, cel shaded
Time: Day
Interior/Exterior: Interior
Scene description: A complete description of the environment.
```

The wizard and enhancer expand sparse descriptions into detailed environments;
enhancement changes only Scene description. The VNCCS INFO logger prints the
complete enhancer system/input/output and final positive and negative prompts.
Cached RUN outputs log the saved inference prompt; uploads report inference skipped.

The node has no IMAGE input. Generate background and RUN share the same input-based
cache: fresh previews are reused, stale previews are regenerated, and manually
uploaded images pass through unchanged. Prompt, active model, weight signatures,
LoRA, seed and sampling changes invalidate the generated preview. Inactive generation
profiles and variant edits do not. The working preview is replaced without a history
counter. Click the preview to enlarge it; wheel zoom follows the cursor and left or
middle button dragging pans the enlarged image.

Morning, Evening and Night start as empty cards in one horizontal strip with edge
scroll arrows. Selecting a card shows Character Creator's tag and preset fields
for Time, Weather and Additional Details directly beneath the strip inside the left
column. Weather starts empty. Custom cards can be added; every card can be deleted.
An empty list stays empty. Completed Background Generator results populate matching
cards automatically and after reopening; clicking a completed card opens its image
in the enlarged preview. The **Prompt** output is a ComfyUI STRING **LIST**, with
one JSON entry per card containing its name, parameters, editing prompt and source IDs.

**VNCCS Background Generator** takes `pipe`, `background`, and `prompt`. Connect
the Control Center pipe and the Creator's image and Prompt outputs. Its shared
generator widget, gallery, stage chain and Regenerate controls perform these steps:

1. Upscale Original once using the configured native SeedVR settings.
2. Generate every variant from that upscaled Original using the editing model
   selected in the Control Center pipe.
3. Optionally upscale the new variants; Original is never upscaled twice.

The gallery shows Original followed by every named variant. The `backgrounds`
and `variations` outputs are IMAGE lists so native sizes remain intact. Complete
sets use immutable numbered PNGs in `output/VNCCS/scenes/NAME/background/sets/SET_ID/`.
The same `NAME.json` records editor settings, working image, variant parameters,
generator settings and completed sets. There is no separate settings document or
browser draft. Writes use revisions, atomic replacement and a validated backup;
failed saves preserve authored data. Archiving a background retains its files.
