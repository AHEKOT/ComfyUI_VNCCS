# Qwen Image 2.1 in VNCCS

Select the `QI2` family in VNCCS Control Center. The packaged catalog provides the Qwen Image 2.1 diffusion model, Qwen3-VL text encoder, VAE, Pose Studio LoRA, and Viggle Turbo LoRA. Normal generation defaults to 25 steps and CFG 3. Enabling the Viggle Turbo LoRA switches the preset to 6 steps and CFG 1.

The Control Center's **Qwen Image 2.1 Cache** settings are available only for `QI2`. They pass the chosen device and dtype to ComfyUI's `QwenImage21Cache` node. Generation uses ComfyUI's `TextEncodeQwenImage21` node for conditioning. The sampler receives a separate empty latent sized from the Pose Studio frame after standard aspect-preserving scaling.

## Viggle Turbo setup

1. Download `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` through Control Center, or place it in `ComfyUI/models/loras/QI2/Viggle/`.
2. Enable **Qwen Image 2.1 Viggle Turbo** in the Control Center LoRA list.

The generation pipeline implements Viggle's unmerged adapter and sigma shift directly; no Viggle custom nodes are required. It calculates sigmas from the exact sampler latent, then samples with ComfyUI's `SamplerCustomAdvanced` and an Euler sampler. The raw six-node schedule is `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`.

FaceDetailer performs its own inpaint sampling and does not accept a SIGMAS input. The emotions generator therefore uses the base QI2 model at 25 steps and CFG 3 for that local detailer pass, even when Viggle Turbo is enabled for the main generators.
