"""Rebuild the accepted compact incidence figure with native photograph crops."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.rebuild_confidence_from_sources import align_native, source_rgb, save_rgb
from WildCamera.benchmark.visualize_confidence_incidence import denormalize, make_dataset
from tools import compose_incidence_paper_eight as layout


def photo_comparison(output, metas, samples):
    width, height = 900, 675
    sheet = Image.new("RGB", (width * 2 + 36, (height + 53) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 25)
    for row, index in enumerate((4, 7)):
        meta, data = metas[index], samples[index]
        x, y, w, h = meta["crop"]
        old = Image.fromarray(np.round(np.clip(data["rgb"][y:y+h, x:x+w], 0, 1) * 255).astype(np.uint8))
        new = Image.fromarray(np.round(np.clip(data["rgb_crop"], 0, 1) * 255).astype(np.uint8))
        top = row * (height + 53)
        draw.text((12, top + 9), f"{meta['dataset']} - previous 200 x 150 crop", font=font, fill="black")
        draw.text((width + 24, top + 9), "Crop resampled directly from source photograph", font=font, fill="black")
        sheet.paste(old.resize((width, height), Image.Resampling.LANCZOS), (12, top + 45))
        sheet.paste(new.resize((width, height), Image.Resampling.LANCZOS), (width + 24, top + 45))
    sheet.save(output / "native_crop_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "output/incidence_native_2026-11-09")
    parser.add_argument("--pdf_dir", type=Path, default=ROOT / "output/pdf/incidence_native_2026-11-09")
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    pdf_dir = args.pdf_dir if args.pdf_dir.is_absolute() else ROOT / args.pdf_dir
    for directory in (output, output / "assets", output / "individual", pdf_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frozen_path = ROOT / "output/incidence_paper_eight_2026-11-08/manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selection = {m["filename"]: m for m in json.loads((layout.CACHE_ROOT / "selection.json").read_text(encoding="utf-8"))}
    metas, samples, audit = [], [], []
    datasets = {}
    for i, accepted in enumerate(frozen["samples"]):
        desktop_source = Path(accepted["source"])
        cache = Path(accepted["cache"])
        assert layout.sha(cache) == accepted["cache_sha256"], f"Prediction cache changed: {cache}"
        assert layout.sha(desktop_source) == accepted["source_sha256"], f"Selected image changed: {desktop_source}"
        meta = copy.deepcopy(selection[desktop_source.name])
        # Freeze the accepted figure's crop and A/B positions, even if a later
        # selection run has touched the surrounding gallery's metadata.
        meta["crop"] = copy.deepcopy(accepted["crop"])
        meta["markers"] = copy.deepcopy(accepted["markers"])
        with np.load(cache) as archive:
            data = {key: archive[key].copy() for key in ("rgb", "baseline", "ours", "gt", "K")}

        name, index = meta["dataset"], meta["index"]
        if name not in datasets:
            datasets[name] = make_dataset(name, SimpleNamespace(data_path=ROOT / "WildCamera/data/MonoCalib"))
        dataset = datasets[name]
        sample = dataset[index]
        rgb_mae = float(np.abs(denormalize(sample["rgb"]) - data["rgb"]).mean())
        assert rgb_mae < 1e-6, f"Wrong original sample or transform: {name}/{index}, MAE={rgb_mae}"
        assert np.allclose(sample["K"].numpy(), data["K"], atol=1e-5)
        source, hdf5, stem = source_rgb(dataset.datasets[0], index)
        transform = sample["T"].numpy()
        full = align_native(source, transform, (1920, 1440))
        crop = align_native(source, transform, (1200, 900), crop=meta["crop"])
        # Keep the old cached RGB and all three fields intact. Rendering accepts
        # separate high-resolution backgrounds with canonical 640x480 extents.
        data.update(rgb_full=full, rgb_crop=crop)
        x, y, w, h = meta["crop"]
        low_crop = data["rgb"][y:y+h, x:x+w]
        crop_low = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        crop_mae = float(np.abs(cv2.GaussianBlur(crop_low, (0, 0), 2) - cv2.GaussianBlur(low_crop, (0, 0), 2)).mean())
        assert crop_mae < .025, f"Native crop is misregistered: {name}/{index}"
        prefix = f"{chr(97+i)}_{name}_{index:04d}"
        save_rgb(output / "assets" / f"{prefix}_full.png", full)
        save_rgb(output / "assets" / f"{prefix}_crop.png", crop)
        for marker in meta["markers"]:
            assert marker["baseline_deg"] > marker["ours_deg"]
            assert marker["baseline_unit_direction_gap"] > marker["ours_unit_direction_gap"]
        audit.append(dict(
            label=chr(97+i), dataset=name, index=index, group=accepted["group"],
            hdf5=str(hdf5), stem=stem, native_size=list(source.size),
            full_render_size=[1920, 1440], crop_render_size=[1200, 900],
            crop=meta["crop"], markers=meta["markers"], transform=transform.tolist(),
            cache=str(cache), cache_sha256=accepted["cache_sha256"],
            original_image=str(desktop_source), original_image_sha256=accepted["source_sha256"],
            original_sample_mae=rgb_mae, native_crop_registration_mae=crop_mae,
        ))
        metas.append(meta)
        samples.append(data)
        print(f"{chr(97+i)} {name}/{index}: source={source.size}, original RGB MAE={rgb_mae:.8f}, crop registration MAE={crop_mae:.6f}", flush=True)

    photo_comparison(output, metas, samples)
    if args.prepare_only:
        (output / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    result = layout.gallery(metas, samples, compact=True, output_dir=output, pdf_dir=pdf_dir, png_dpi=1200)
    for meta, data in zip(metas, samples):
        layout.individual(meta, data, output_dir=output)
    for accepted in frozen["samples"]:
        assert layout.sha(Path(accepted["cache"])) == accepted["cache_sha256"]
        assert layout.sha(Path(accepted["source"])) == accepted["source_sha256"]
    with Image.open(output / "incidence_eight_compact.png") as image:
        result["png_pixels"] = list(image.size)
    manifest = dict(layout=result, samples=audit, original_manifest=str(frozen_path),
                    predictions_unchanged=True, gt_unchanged=True, roi_and_markers_unchanged=True,
                    field_rendering="Same cached equal-length 2D projected direction arrows; vector in PDF",
                    source_limitations="SUN3D and both NYUv2 sources are 640x480. Interpolation adds no captured detail.")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
