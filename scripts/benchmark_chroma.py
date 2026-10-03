"""Run the local screen-matte corpus without installing ComfyUI.

Example: python scripts/benchmark_chroma.py SOURCE OUTPUT --device mps --repeats 3
Input files are never modified. Timings include device transfers, not PNG I/O.
"""

import argparse
import inspect
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
from chroma_screen_matte import screen_matte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        parser.error("Output must differ from the input directory.")
    if args.repeats < 1:
        parser.error("Repeats must be positive.")
    files = sorted(p for p in source.rglob("*.png") if not p.is_relative_to(output))
    if not files:
        parser.error("No PNG inputs found.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    settings = {
        name: parameter.default for name, parameter in inspect.signature(screen_matte).parameters.items()
        if parameter.default is not inspect.Parameter.empty and name != "processing_device"
    }
    output.mkdir(parents=True, exist_ok=True)
    dark = Image.new("RGB", (1800, ((len(files) + 5) // 6) * 440), "#28282e")
    light = Image.new("RGB", dark.size, "#f0f0f0")
    rows = []
    warmed_shapes = set()
    for index, path in enumerate(files):
        image = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).float() / 255
        if tuple(image.shape) not in warmed_shapes:
            screen_matte(image, processing_device=device, **settings)
            warmed_shapes.add(tuple(image.shape))
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            rgba, alpha, _debug = screen_matte(image, processing_device=device, **settings)
            durations.append(time.perf_counter() - start)
        pixels = (rgba.numpy() * 255).round().clip(0, 255).astype(np.uint8)
        relative = path.relative_to(source)
        destination = output / "rgba" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = Image.fromarray(pixels)
        result.save(destination)
        mask = pixels[..., 3] > 4
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        row = {
            "file": relative.as_posix(), "shape": list(image.shape),
            "seconds": statistics.median(durations), "runs": durations,
            "components": len(areas),
            "pixels_outside_largest_component": int(areas.sum() - areas.max()) if len(areas) else 0,
            "nonzero_rgb_under_zero_alpha": int(np.any(pixels[pixels[..., 3] == 0, :3] != 0, axis=1).sum()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        result.thumbnail((295, 405))
        x, y = (index % 6) * 300, (index // 6) * 440
        for canvas, color in ((dark, "white"), (light, "black")):
            canvas.paste(result, (x + (300 - result.width) // 2, y), result)
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 5, y + 409), relative.as_posix(), fill=color)
            draw.text((x + 5, y + 425), f"{row['seconds']:.3f}s", fill=color)
    dark.save(output / "contact-dark.jpg")
    light.save(output / "contact-light.jpg")
    report = {
        "device": device, "platform": platform.platform(), "torch": torch.__version__,
        "settings": settings,
        "repeats": args.repeats, "timing": "Median after shape warmup; includes host/device transfers; excludes PNG I/O.",
        "quality_note": "Component counts are diagnostics, not a ground-truth alpha quality score.",
        "images": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
