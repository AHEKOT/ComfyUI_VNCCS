// Deterministic visual data only; the widget and Gaussian renderer remain production code.
export const backgroundId = "a".repeat(32), imageId = "b".repeat(32), modelId = "c".repeat(32), objectId = "d".repeat(32);
export const base = "/vnccs/cg-creator";
export const setId = "2".repeat(32), variantModelId = "3".repeat(32);
export const sourceVariant = { set_id: setId, index: 1, variant_id: "morning_light", name: "Morning light" };
export const baseModel = "qwen_image_2.1_int8_convrot.safetensors", turboModel = "qwen_image_2.1_turbo_int8_convrot.safetensors";
export const turboLora = "QI2/Viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors";
export const manualLora = "background/style.safetensors";
export const controlCenter = {
    models: [
        { name: "Qwen Image 2.1 INT8 ConvRot", kind: "QI2", type: "unet", hf_repo: "Comfy-Org/Qwen-Image-2.1", hf_path: `diffusion_models/${baseModel}`, local_path: `models/diffusion_models/${baseModel}`, version: "1.0", status: "installed", description: "Qwen Image 2.1 INT8 ConvRot diffusion model." },
        { name: "Qwen Image 2.1 Turbo INT8 ConvRot", kind: "QI2", type: "unet", hf_repo: "Comfy-Org/Qwen-Image-2.1", hf_path: `diffusion_models/${turboModel}`, local_path: `models/diffusion_models/${turboModel}`, version: "1.0", status: "installed", description: "Qwen Image 2.1 Turbo INT8 ConvRot diffusion model. Uses standard sampling with 8 steps and CFG 1." },
    ],
    clip: [{ name: "Qwen 3 VL", kind: "QI2", local_path: "models/text_encoders/qwen3vl_8b_int8_convrot.safetensors", status: "installed" }],
    vae: [{ name: "Qwen Image 2.1 VAE", kind: "QI2", local_path: "models/vae/qwen_image_2.1_vae_bf16.safetensors", status: "installed" }],
    lora: [{ name: "Qwen Image 2.1 Viggle Turbo", kind: "QI2", type: "turbolora", local_path: `models/loras/${turboLora}`, status: "installed" }],
};
export const camera = { position: [0, 1.7, 5.8], target: [0, 1.7, -1.6], up: [0, 1, 0], fov: 60 };
export const sourceCamera = { position: [0, 1.7, 5.8], target: [0, 1.7, -1.6], up: [0, 1, 0], fov: 60 };
export const document = {
    schema_version: 2, background_id: backgroundId, name: "Evening room", revision: 1, archived: false,
    editor: { description: "A quiet bedroom at sunset." }, selected_image_id: imageId,
    images: [{ image_id: imageId, file: "source.png", width: 1600, height: 896, settings: { name: "Original · Sunset" }, preview_revision: 1, url: `${base}/backgrounds/${backgroundId}/images/${imageId}?v=1` }],
    generated_sets: [{ set_id: setId, version: "", source_image_id: imageId, settings: { seed: 0 },
        variants: [{ id: "morning_light", name: "Morning light" }, { id: "rainy_evening", name: "Rainy evening" }, { id: "night", name: "Night" }],
        images: ["Original", "Morning light", "Rainy evening", "Night"].map((name, index) => ({ name, file: `Images/${String(index).padStart(2, "0")}_${name}.png`, width: 1600, height: 896, revision: 1, settings: { seed: 0 }, url: `${base}/backgrounds/${backgroundId}/sets/${setId}/${index}?v=1` })) }],
    models: [{ model_id: modelId, source_image_id: imageId, settings: { seed: 66011999 } }],
    model_settings: { num_gaussians: 262144, steps: 20, guidance_scale: 3, seed: 66011999 },
    cgs: [
        { cg_id: "e".repeat(32), name: "Room · Wide", camera, width: 1600, height: 896 },
        { cg_id: "f".repeat(32), name: "Desk · Evening", camera: { ...camera, position: [0.5, 1.7, 4.8], target: [1.2, 1.5, -1.6], fov: 48 }, width: 1344, height: 1008 },
    ].map(cg => ({ ...cg, model_id: modelId, source_image_id: imageId, url: `${base}/backgrounds/${backgroundId}/cg/${cg.cg_id}/image` })),
};
export const scene = {
    schema_version: 11, scene_id: "1".repeat(32), background_id: backgroundId, model_id: modelId,
    source_image_id: imageId, camera, lighting: { preset: "off", background: "#000000" },
    objects: [{ object_id: objectId, name: "Evening room", locked: true, visible: true, source_camera: sourceCamera,
        transform: { position: [0, 0, 0], rotation: [0, 0, 0], scale: 1 },
        urls: { splat: `${base}/backgrounds/${backgroundId}/models/${modelId}/objects/${objectId}/splat` } }],
};

export function roomSplat() {
    const points = [], step = 0.065;
    const plane = (axis, fixed, a0, a1, b0, b1, color) => {
        for (let a = a0; a <= a1; a += step) for (let b = b0; b <= b1; b += step) {
            const position = axis === "z" ? [a, b, fixed] : axis === "y" ? [a, fixed, b] : [fixed, a, b];
            points.push({ position, color: typeof color === "function" ? color(a, b) : color });
        }
    };
    const box = (x0, y0, z0, x1, y1, z1, color) => {
        plane("z", z1, x0, x1, y0, y1, color); plane("y", y1, x0, x1, z0, z1, color);
        plane("x", x0, y0, y1, z0, z1, color); plane("x", x1, y0, y1, z0, z1, color);
    };
    plane("z", -3, -4, 4, 0, 3.8, (x, y) => {
        if (x > -1.2 && x < 1.2 && y > 1.3 && y < 3.3) {
            if (Math.abs(x) < 0.04 || Math.abs(y - 2.15) < 0.04) return [75, 61, 62];
            return [125 + 30 * (3.3 - y), 131 + 10 * (3.3 - y), 165];
        }
        return [180, 151, 125];
    });
    plane("y", 0, -4, 4, -3, 3, (x, z) => Math.floor((x + 4) * 4) % 2 ? [115, 79, 55] : [128, 90, 61]);
    plane("x", -4, 0, 3.8, -3, 3, [157, 131, 113]); plane("x", 4, 0, 3.8, -3, 3, [157, 131, 113]);
    box(-3.65, 0.2, -2.6, -1.2, 0.7, 0.7, [177, 113, 127]);
    box(-3.5, 0.7, -2.4, -1.35, 0.9, -1.6, [218, 174, 167]);
    box(1.2, 0.85, -2.6, 3.65, 1, -1.4, [128, 86, 63]);
    box(2.65, 0, -2.5, 3.55, 0.84, -1.5, [161, 113, 92]);
    box(2.4, 1.05, -2.9, 3.65, 3.4, -2.65, [96, 72, 58]);
    for (let y = 1.15; y < 3.2; y += 0.55) for (let x = 2.48; x < 3.55; x += 0.13)
        box(x, y, -2.64, x + 0.09, y + 0.38, -2.58, x % 0.3 < 0.15 ? [170, 128, 80] : [97, 124, 110]);
    const bytes = Buffer.alloc(points.length * 32);
    points.forEach(({ position: [x, y, z], color }, i) => {
        const offset = i * 32;
        // Inverse of Factory's canonical TripoSplat rotation.
        [z, -y, x, 0.038, 0.038, 0.038].forEach((value, j) => bytes.writeFloatLE(value, offset + j * 4));
        [...color, 255, 255, 128, 128, 128].forEach((value, j) => { bytes[offset + 24 + j] = Math.round(value); });
    });
    return bytes;
}
