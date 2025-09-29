![Header](images/README.png)
---
# VNCCS - Visual Novel Character Creation Suite

VNCCS is a comprehensive tool for creating character sprites for visual novels. It allows you to create unique characters with a consistent appearance across all images, which was previously a challenging task when using neural networks.

## Description

Many people want to use neural networks to create graphics, but making a unique character that looks the same in every image is much harder than generating a single picture. With VNCCS, it's as simple as pressing a button (just 4 times).

## Character Creation Stages

The character creation process is divided into 5 stages:

1. **Create a base character**
2. **Create clothing sets**
3. **Create emotion sets**
4. **Generate finished sprites**
5. **Create a dataset for LoRA training** (optional)

## Installation

Find `VNCCS - Visual Novel Character Creation Suite` in Custom Nodes Manager or install it manually:

1. Place the downloaded folder into `ComfyUI/custom_nodes/`
2. Launch ComfyUI and open Comfy Manager
3. Click "Install missing custom nodes"
4. Alternatively, in the console: go to `ComfyUI/custom_nodes/` and run `git clone https://github.com/AHEKOT/ComfyUI_VNCCS.git`

## Required Models

VNCCS requires the following models in ComfyUI. Make sure they are installed in the correct directories:

### Stable Diffusion Checkpoints (`models/checkpoints/`)
- Any illustrious based model.
- Should work on any SDXL model, but not tested.

### LoRA Models (`models/loras/`)
- `vn_character_sheet_v4.safetensors`
- `DMD2/dmd2_sdxl_4step_lora_fp16.safetensors`
- `vn_character_sheet.safetensors`
- `IL/mimimeter.safetensors`

### ControlNet Models (`models/controlnet/`)
- `SDXL/AnytestV4.safetensors`
- `SDXL/IllustriousXL_openpose.safetensors`

### Face Detection Models (`models/ultralytics/bbox/` and `models/ultralytics/segm/`)
- `bbox/face_yolov8m.pt`
- `bbox/face_yolov9c.pt`
- `segm/face_yolov8m-seg_60.pt`

### SAM Models (`models/sams/`)
- `sam_vit_b_01ec64.pth`

### Upscale Models (`models/upscale_models/`)
- `4x_APISR_GRL_GAN_generator.pth`
- `2x_APISR_RRDB_GAN_generator.pth`

All models must be placed in the standard ComfyUI directories according to their type.

You can download them from my huggingface: https://huggingface.co/MIUProject/VNCCS/tree/main

## Usage

Always use the latest versions of workflows!

### Step 1: Create a Base Character

Open the workflow `VN_Step1_CharSheetGenerator`.

#### Character Sheet
![Character Sheet Node](images/charsheetnode.png)
Default character sheet lay is in `character_pemplate` folder

---
#### Settings
![Settings Node](images/settings.png)
---
#### VNCCS Character Creator
![Creator Node](images/creatornode.png)
---
#### Service Nodes
![Service Nodes](images/servicenodes.png)
---
#### Workflow Nodes
![Workflow Nodes](images/workflownodes.png)
---
#### **First Pass**
Initial generation. Raw and imprecise, used for visual identification of the character.
- **Match** - how much the new character would look like character from initial charsheet. Too low and too high settings are not recommended and will result to unwanted things.
Safe to stay in about 0.5, but you can test other values!
---
#### **Stabilizer**
Stabilizes the initial generation, bringing it to a common look.
- **Blur** - blur level for complex cases to help small differences disappear.
- **Match** - similarity level (0.85 is safe, range 0.4-1.0)
- LoRA trained on character sheet to help with matching things.
---
#### **Third Pass**
Third generation pass using the stabilized character sheet.
- **RMBG Resolution** - it's very important for great sprites. 1408 - maximum resolution for 8GB VRAM cards! But you can set 1536 if you have better videocard than me.
But if you have troubles - you can lower it to 1024 and edit small mistakes manualy.
Better resolution - better mask!
---
#### **Upscaler**
Improves character sheet quality by adding details.
- **Denoise** - strength in settings (safe values 0.4-0.5). More denoise - more details. More details - more inconsistency with small details on character. Stay in balance!
- **Seam Fix Mode** - to the best results, set to "Half Tile + intersections but this will result in dramatically increased generation time.
---
#### **Face Detailer**
Improves face quality.
- **Value range:** 0.2-0.7 (default 0.65). Higher value - more changes.
---
![FinishFirsStep](images/finishfirststep.png)
---
### Step 2: Create Clothing Sets

Open the workflow `VN_Step2_ClothesChanger`.

It's similar to the first, but has key differences:

#### **VNCCS Character Selector**
![Character Selector Node](images/characterselectornode.png)
---
**Workflow nodes work the same way as in Character Creator.
So, let focus of the nuances of settings:**

- Lower match strength means better and more diverse clothing, but the character may look less like themselves
- Higher strength means the node repeats the previous result more
- Find a balance between diversity and consistency
- For complex clothing, look for similar examples on Danbooru for accurate tags
- Complex multi-layered costumes may give unpredictable results
- If the character is generated without clothes, fix the prompt – the model doesn't understand what to draw.
---

Create as many costumes as you need. All data is saved in `ComfyUI/output/VN_CharacterCreatorSuit/YOUR_CHARACTER_NAME`, parameters are in the JSON config.

After creating costumes, proceed to emotions.

Open the workflow `VN_Step3_EmotionsGenerator`.

---
### Step 3: Create Emotion Sets

This step is similar to the previous ones, but focuses on creating emotions for your character.

#### VNCCS Emotion Generator
![Emotion Node](images/EmotionsNode.png)
- **Denoise** greatly impacts emotion strength but too high values can distort face of character and affect consistency.

After setting the parameters, click "Run".

Emotions will be generated and saved in the `ComfyUI/output/VN_CharacterCreatorSuit/YOUR_CHARACTER_NAME/Sheets/spaceman/EMOTION_NAME` folder for your character.

---
### Step 4: Generate Finished Sprites

After creating the character and their emotions, you can generate finished sprites for use in your visual novel.

Open the workflow `VN_Step4_SpritesGenerator`.

---
#### Sprite Generator
![Sprites Node](images/SpritesNode.png)

- **Select character** - choose the character for which you want to generate sprites.
Click "Run" and wait for the process to finish.

The finished sprites will appear in the folder `ComfyUI/output/VN_CharacterCreatorSuit/YOUR_CHARACTER/Sprites`

---
### Step 5: Create a Dataset for LoRA Training (Optional)

This step is optional and intended for those who want to further train their LoRA model on the created characters.

Open the workflow `VN_Step5_DatasetCreator`.

#### Dataset Creator

- **Select characters** – choose the characters you want to include in the dataset.
- **Game Name** – a prefix that will be added to the character's name in the dataset, e.g. VN_your_character_name.
Click "Run" and wait for the process to finish.
The dataset will appear in the folder `ComfyUI/output/VN_CharacterCreatorSuit/YOUR_CHARACTER/Lora`



## Conclusion

Congratulations! You have successfully created your own character using VNCCS. Now you can use it in your projects or continue experimenting with the settings to create new unique characters.

Don't forget to save all your work and regularly back up your data. Good luck creating your visual novels!

---

**If you found my project useful, you can help me buy a new graphics card! With it, I will be able to add animation and bring static graphics to life!**

Future plans: 
- implement more "modern" models like qwen, flux and nanobanana
- consistent background generation
- animated sprites
- animation of transitions between poses
- automatic translation of the game on RenPy into other languages.
- automatic voice generation for the game on RenPy
---

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20rtx%203090&emoji=☕&slug=MIUProject&button_colour=FFDD00&font_colour=000000&font_family=Comic&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/MIUProject)