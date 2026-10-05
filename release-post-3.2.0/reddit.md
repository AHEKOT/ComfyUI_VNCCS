# VNCCS 3.2.0 Released!

Hi! V-chan here! We got another BIIIG update, and you now have some new toys to play with. New models, transparent sprites, and a Character Creator makeover. We even recruited a video model for sprite duty. Hehe!

If you are new here: **VNCCS is a ComfyUI pipeline for creating characters and turning them into sprite sets with different poses, outfits, and expressions.** For visual novels, games, or whatever your little creative brain is plotting.

Here is the fun stuff in **3.2.0**:

## Qwen Image 2.1 is our new main model now.

**Qwen Image 2.1** can create your base character, change poses, dress them up, and generate expressions. You can use it in Character Creator, the pose and clothing workflows, and Emotion Studio.

Control Center puts the model families, installed assets, and Turbo controls together. Choose your family, check what is missing, and hit **Download / Update**.

![Control Center with Qwen Image 2.1 selected, installed model cards, cache settings, and the active Viggle Turbo switch](screenshots/control-center-qwen.png)

*Qwen Image 2.1 is selected here. Klein9b is still available, and MiniMax H3 has its own tab too.*

## MiniMax H3: a video model doing sprite work? Yep!

**MiniMax H3** is the other new arrival. VNCCS uses it for poses, outfit generation, and clothes cloning, keeping the first frame as your character image.

So yes, you can try a video model in your sprite workflow. No need to turn your visual novel into a movie first, silly.

![MiniMax H3 selected in Control Center, with the FP8 Scaled and INT8 ConvRot model choices visible](screenshots/control-center-h3.png)

*H3 has its own model choices in Control Center, including FP8 Scaled and INT8 ConvRot.*

## Character Creator got a glow-up

The character fields now use **editable tag chips** and little **+ buttons** for presets. Hair, eyes, face, body, skin, species, and details are easier to build and adjust without wrestling a wall of prompt text.

There are **40 visual style presets**, from anime and animation to artistic and realistic looks, plus a custom style field. The new descriptive catalog also includes **61 species presets**, and you can combine species for hybrids. Cannot choose one? Make the character someone else's taxonomy problem. :3

Species presets describe the actual visual traits to the model, and your own character details take priority. You also get **Full body / Cowboy shot** framing choices.
Qwen's Character Overhaul LoRA will help your generations stay in your full control. It add some tags knowlege and stabilize characters by small cost of unique QI2 style loss.

![Character Creator V2 with a character preview, editable trait chips, preset buttons, framing and style selectors, and Qwen generation controls](screenshots/character-creator.png)

*Preview on the left, character design in the middle, generation controls on the right. The Alpha background option and separate Turbo / Character Overhaul controls are visible here too.*

## Transparent sprites, with less background cleanup

With **Qwen Image 2.1**, you can choose **Alpha** in Creator or Clothes Designer and **Native** background mode in the generators. Qwen generates transparency directly, and VNCCS preserves it through clothing edits, emotion editing, and SeedVR upscaling. Green-screen duty can finally take a little vacation. Yay!

The new **Resolution scale** slider runs from **1 to 4 MP**. It controls total image area, while pose and clothing generation keep the source proportions. Your resolution choices are remembered separately for each model family.

![Character Generator controls showing 4.0 MP resolution scale, SeedVR upscaling, and Native background removal](screenshots/native-alpha-settings.png)

*These are the Native background and SeedVR controls. Native Alpha generation is a Qwen Image 2.1 feature; other families still use their compatible background options.*

## More ways to play dress-up

Clothes Designer now supports **Qwen and H3 previews alongside Klein9b**. Describe an outfit or use a clothing reference, check the preview, and then generate your pose set.

Changed the reference or generation settings? The preview cache now checks those changes, so it can regenerate the outfit properly. And unwanted costumes can be deleted from the widget, with confirmation.

![Clothes workflow with an outfit reference in Clone Clothes and the dressed character in the large generator preview](screenshots/clothes-workflow.png)

*The pink outfit comes from the red-haired reference in Clone Clothes. The large generator preview shows it on the orange-haired character.*

## Qwen can give your character feelings too

Emotion Studio now has a **Qwen Image 2.1 profile**. It edits a face crop and blends it back into the original sprite, keeping the rest of the image in place and preserving transparency.

Illustrious and Anima remain available. Qwen has its own face controls, including face resolution and an editable expression prompt template.

![Emotion Studio with the existing visual emotion library, selected costumes and poses, and the new Qwen Image 2.1 generation profile](screenshots/emotion-studio.png)

*The emotion cards help you choose an expression; Qwen's model and Turbo settings are on the right. The cards are selection examples, not generated results for the character on the left.*

A few smaller comforts came along too: **Qwen3.5 now powers the Wizards and image analysis**, downloads show real transfer progress, and generator progress and previews can recover after reconnecting to ComfyUI.

**Updating from an older version?** Use the bundled **3.2 workflows** and a ComfyUI build with native support for your chosen model. Old **QIE2511** setups need to switch to **QI2 with its matching assets**, or a compatible Klein9b setup. In Qwen Creator, download the Character Overhaul LoRA if you use it, or set its strength to **0** to generate without it.

Find VNCCS on [GitHub](https://github.com/AHEKOT/ComfyUI_VNCCS), read the [full changelog](https://github.com/AHEKOT/ComfyUI_VNCCS/blob/main/Changelog.md), or look for **VNCCS - Visual Novel Character Creation Suite** in ComfyUI Manager. Come share your characters and experiments on [Discord](https://discord.com/invite/9Dacp4wvQw)!

Which toy are you trying first: transparent Qwen sprites, H3 outfits, or a suspiciously elaborate hybrid character?
