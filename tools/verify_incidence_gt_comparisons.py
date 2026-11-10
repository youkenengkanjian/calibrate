"""Independently verify saved GT geometry, selected rays, and all 100 exports."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/incidence_gt_100_2026-11-08"


def degrees(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a /= np.linalg.norm(a, axis=0, keepdims=True)
    b /= np.linalg.norm(b, axis=0, keepdims=True)
    return np.degrees(np.arccos(np.clip((a*b).sum(axis=0), -1, 1)))


def main():
    records = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    assert len(records) == 100
    assert len({(r["dataset"], r["index"]) for r in records}) == 100
    assert len(list((OUT / "individual").glob("*.png"))) == 100
    assert len(list((OUT / "thumbnails").glob("*.jpg"))) == 100
    assert len(list((OUT / "galleries").glob("*.jpg"))) == 10
    max_gt_diff = 0.0
    min_full_gain = float("inf")
    min_crop_gain = float("inf")
    min_marker_gain = float("inf")
    max_arrow_length_deviation = 0.0
    min_marker_direction_gain = float("inf")
    hashes = set()
    for r in records:
        with np.load(OUT / "cache" / r["cache"]) as data:
            rgb = data["rgb"]
            b, o, g = [data[key] for key in ("baseline", "ours", "gt")]
            assert all(f.shape == (3, 480, 640) for f in (b, o, g))
            assert all(np.isfinite(f).all() for f in (rgb, b, o, g))
            assert all(np.max(np.abs(np.linalg.norm(f, axis=0)-1)) < 1e-5 for f in (b, o, g))
            assert all(np.min(f[2]) > 0 for f in (b, o, g))
            digest = hashlib.sha256(np.round(rgb*255).astype(np.uint8).tobytes()).hexdigest()
            assert digest == r["image_sha256"] and digest not in hashes
            hashes.add(digest)
            yy, xx = np.mgrid[:480, :640]
            pixel = np.stack((xx+0.5, yy+0.5, np.ones_like(xx)))
            gt_check = np.einsum("ij,jhw->ihw", np.linalg.inv(data["K"].astype(np.float64)), pixel)
            gt_check /= np.linalg.norm(gt_check, axis=0, keepdims=True)
            gt_diff = float(np.max(np.abs(gt_check-g)))
            max_gt_diff = max(max_gt_diff, gt_diff)
            assert gt_diff < 2e-6, (r["filename"], gt_diff)
            eb, eo = degrees(b, g), degrees(o, g)
            full_gain = float(np.mean(eb-eo))
            assert full_gain > 0.03
            min_full_gain = min(min_full_gain, full_gain)
            x, y, w, h = r["crop"]
            assert w == 200 and h == 150 and 0 <= x <= 640-w and 0 <= y <= 480-h
            roi_gain = eb[y:y+h, x:x+w]-eo[y:y+h, x:x+w]
            assert roi_gain.mean() > 0.1 and (roi_gain > 0).mean() >= 0.65
            min_crop_gain = min(min_crop_gain, float(roi_gain.mean()))
            assert len(r["markers"]) == 2
            xs, ys = np.meshgrid(np.arange(30, w-29, 35)+x, np.arange(30, h-29, 30)+y)
            # Independent direction-only formula, evaluated without the renderer.
            projected = []
            for field in (b, o, g):
                xy = field[:2].astype(np.float64)
                norm = np.linalg.norm(xy, axis=0, keepdims=True)
                projected.append(np.divide(xy, norm, out=np.zeros_like(xy), where=norm > 1e-8))
            for field in projected:
                rendered_lengths = np.linalg.norm(field[:, ys, xs]*25, axis=0)
                deviation = float(np.max(np.abs(rendered_lengths-25)))
                max_arrow_length_deviation = max(max_arrow_length_deviation, deviation)
                assert deviation < 1e-9
            for marker in r["markers"]:
                mx, my = marker["x"], marker["y"]
                assert ((xs == mx) & (ys == my)).any()
                gain = float(eb[my, mx]-eo[my, mx])
                assert gain > 0.05
                min_marker_gain = min(min_marker_gain, gain)
                pb, po, pg = [f[:, my, mx] for f in projected]
                assert np.linalg.norm(pb-pg) > np.linalg.norm(po-pg)+0.001
                direction_gain = float(np.linalg.norm(pb-pg)-np.linalg.norm(po-pg))
                min_marker_direction_gain = min(min_marker_direction_gain, direction_gain)
                assert abs(np.linalg.norm(pb-pg)-marker["baseline_unit_direction_gap"]) < 1e-8
                assert abs(np.linalg.norm(po-pg)-marker["ours_unit_direction_gap"]) < 1e-8
                guv = pg*25
                center = np.array([mx-x, my-y])+guv*0.5
                radius = np.linalg.norm(guv)*0.5+4
                assert (center-radius >= 0).all() and (center+radius <= [w, h]).all()
            with Image.open(OUT / "individual" / r["filename"]) as im:
                im.verify()
            with Image.open(OUT / "individual" / r["filename"]) as im:
                assert im.size == (3827, 924), im.size
    with (OUT / "candidate_audit.csv").open(encoding="utf-8-sig") as f:
        scanned = Counter(r["dataset"] for r in csv.DictReader(f))
    report = dict(status="passed", count=100, unique_images=len(hashes),
                  scanned_per_dataset=dict(scanned), total_scanned=sum(scanned.values()),
                  max_independent_gt_component_difference=max_gt_diff,
                  min_recomputed_full_gain_deg=min_full_gain,
                  min_recomputed_roi_gain_deg=min_crop_gain,
                  min_recomputed_marker_gain_deg=min_marker_gain,
                  min_marker_unit_direction_gain=min_marker_direction_gain,
                  max_arrow_length_deviation=max_arrow_length_deviation,
                  local_arrow_length=25,
                  checked="100 GT fields, predictions, source hashes, ROI positions, 200 direction-only markers, equal arrow lengths, image files, 100 thumbnails and 10 gallery pages")
    (OUT / "independent_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
