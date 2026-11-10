"""Rebuild the accepted eight confidence overlays from source HDF5 photographs.

Model inference and display-map construction stay at the original 640x480.
Only the photograph and compositing resolution change. The display map is a
processed, structure-aware visualization, not a calibrated probability map.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from WildCamera.benchmark.visualize_confidence_incidence import (
    CONFIDENCE_CMAP,
    denormalize,
    forward_with_confidence_logits,
    load_model,
    make_dataset,
    normalized_confidence,
    visualization_confidence,
)


SELECTION = [
    ("Indoor", "SUN3D", 95, "02"),
    ("Indoor", "SUN3D", 32, "39"),
    ("Indoor", "MVImgNet", 28, "11"),
    ("Indoor", "ARKitScenes", 7, "73"),
    ("Outdoor", "Nuscenes", 0, "08"),
    ("Outdoor", "Objectron", 15, "34"),
    ("Outdoor", "Cityscapes", 46, "50"),
    ("Outdoor", "Cityscapes", 93, "76"),
]


def blend(rgb, display):
    alpha = 0.65 * display[..., None] ** 0.8
    heat = CONFIDENCE_CMAP(display)[..., :3]
    return np.clip(rgb * 0.78 * (1.0 - alpha) + heat * alpha, 0.0, 1.0)


def source_rgb(dataset, index):
    scene, stem = dataset.data_names[index].split()
    path = Path(dataset.data_root) / f"{scene}.hdf5"
    with h5py.File(path, "r") as handle:
        encoded = np.asarray(handle["color"][stem]).tobytes()
    with Image.open(io.BytesIO(encoded)) as image:
        image = image.convert("RGB").copy()
    return image, path, stem


def align_native(rgb, transform, size, crop=None):
    """Compose the dataset resize and grid_sample transform in one native warp.

    T acts on the original 640x480 model-input coordinates. Match the existing
    resample_rgb align_corners=True convention before mapping to source pixels.
    """
    out_w, out_h = size
    src_w, src_h = rgb.size
    x0, y0, width, height = (0, 0, 640, 480) if crop is None else crop
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * width / out_w - 0.5 + x0
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * height / out_h - 0.5 + y0
    if not np.allclose(transform, np.eye(3), atol=1e-7):
        if not np.allclose(transform[[0, 1], [1, 0]], 0.0):
            raise ValueError("Unexpected non-diagonal crop transform")
        x = (transform[0, 0] * (x + 0.5) + transform[0, 2]) * 639 / 640
        y = (transform[1, 1] * (y + 0.5) + transform[1, 2]) * 479 / 480
    x = (x + 0.5) * src_w / 640 - 0.5
    y = (y + 0.5) * src_h / 480 - 0.5
    grid_x, grid_y = np.meshgrid(x, y)
    return cv2.remap(
        np.asarray(rgb), grid_x, grid_y, interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32) / 255.0


def save_rgb(path, pixels):
    Image.fromarray(np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(
        path, dpi=(600, 600)
    )


def comparison(output, records):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 27)
    canvas = Image.new("RGB", (1448, 1128), "white")
    draw = ImageDraw.Draw(canvas)
    choices = [(3, (30, 570, 730, 1070)), (6, (70, 330, 770, 830))]
    for row, (index, box) in enumerate(choices):
        record = records[index]
        with Image.open(record["previous_overlay"]) as im:
            old = im.convert("RGB").resize((1920, 1440), Image.Resampling.LANCZOS)
        with Image.open(record["overlay"]) as im:
            new = im.convert("RGB").copy()
        top = row * 564
        draw.text((12, top + 7), f"{record['dataset']} - previous 640 x 480", fill="black", font=font)
        draw.text((736, top + 7), "Rebuilt from source photograph", fill="black", font=font)
        canvas.paste(old.crop(box), (12, top + 48))
        canvas.paste(new.crop(box), (736, top + 48))
    canvas.save(output / "detail_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    for directory in (output, output / "overlays", output / "source_rgb", output / "maps"):
        directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "model_zoo/multi-modality newcrfs(train置信度)/step_29999.ckpt"
    model = load_model(ckpt, "large07", device, baseline=False)
    datasets = {}
    records = []
    for rank, (group, name, index, old_rank) in enumerate(SELECTION, start=1):
        if name not in datasets:
            datasets[name] = make_dataset(name, SimpleNamespace(data_path=ROOT / "WildCamera/data/MonoCalib"))
        concat = datasets[name]
        sample = concat[index]
        low_rgb = denormalize(sample["rgb"])
        with torch.inference_mode():
            pred, logits = forward_with_confidence_logits(model, sample["rgb"].unsqueeze(0).to(device))
            normalized = normalized_confidence(logits, pred["incidence"])[0, 0].cpu().numpy()
        display = visualization_confidence(normalized, low_rgb)
        old_version = "paper_visualizations_reviewer_final" if name == "MVImgNet" else "paper_visualizations_confidence_100_final_v2"
        previous = ROOT / "output" / old_version / "confidence_typical/overlays" / f"{old_rank}_{name}_{index:04d}_overlay.png"
        with Image.open(previous) as im:
            previous_np = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        old_match_mae = float(np.abs(blend(low_rgb, display) - previous_np).mean())
        # Allow only PNG quantization tolerance: a different visualization
        # recipe must be investigated rather than silently changing heatmaps.
        if old_match_mae > 0.005:
            raise ValueError(f"{name}/{index}: previous overlay differs (MAE={old_match_mae:.6f})")
        native, source_path, stem = source_rgb(concat.datasets[0], index)
        transform = sample["T"].cpu().numpy()
        high_rgb = align_native(native, transform, (1920, 1440))
        high_display = np.clip(cv2.resize(display, (1920, 1440), interpolation=cv2.INTER_CUBIC), 0, 1)
        high_overlay = blend(high_rgb, high_display)
        prefix = f"{rank:02d}_{name}_{index:04d}"
        target = output / "overlays" / f"{prefix}_overlay.png"
        save_rgb(target, high_overlay)
        save_rgb(output / "source_rgb" / f"{prefix}_aligned.png", high_rgb)
        np.savez_compressed(output / "maps" / f"{prefix}.npz", normalized_logits=normalized, display=display, transform=transform)
        # Compare low-frequency registration after native reconstruction. Native
        # detail itself is expected to differ from the pre-resized old image.
        downsampled = cv2.resize(high_rgb, (640, 480), interpolation=cv2.INTER_AREA)
        registration_mae = float(np.abs(cv2.GaussianBlur(downsampled, (0, 0), 2) - cv2.GaussianBlur(low_rgb, (0, 0), 2)).mean())
        if registration_mae > 0.025:
            raise ValueError(f"{name}/{index}: source registration mismatch: {registration_mae}")
        record = dict(rank=rank, group=group, dataset=name, index=index, source=str(source_path),
                      stem=stem, native_size=list(native.size), overlay_size=[1920, 1440],
                      previous_overlay=str(previous), overlay=str(target), transform=transform.tolist(),
                      previous_overlay_mae=old_match_mae, source_registration_mae=registration_mae)
        records.append(record)
        print(f"{rank}/8 {name} {index}: source {native.size}, old map MAE={old_match_mae:.6f}, registration MAE={registration_mae:.6f}", flush=True)
    manifest = dict(checkpoint=str(ckpt), model_input_size=[640, 480],
                    display_note="Existing structure-aware display postprocessing is retained. The 0-1 scale is normalized display intensity, not calibrated probability. Heatmap spatial resolution is not increased by interpolation.",
                    records=records)
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison(output, records)
    print(f"Completed source reconstruction: {output}", flush=True)


if __name__ == "__main__":
    main()
