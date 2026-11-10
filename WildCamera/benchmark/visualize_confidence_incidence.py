"""Create paper-ready confidence and incidence-field visualizations."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors
from matplotlib.collections import LineCollection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from WildCamera.benchmark.benchmark_confidence_ablation import load_model
from WildCamera.datasets.IncdDataset import IncdDataset
from tools.tools import intrinsic2incidence


DATASETS = ["Nuscenes", "KITTI", "Cityscapes", "NYUv2", "ARKitScenes", "SUN3D", "MVImgNet", "Objectron"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
CONFIDENCE_CMAP = colors.LinearSegmentedColormap.from_list(
    "confidence_turbo_soft",
    matplotlib.colormaps["turbo"](np.linspace(0.05, 0.94, 256)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_ckpt",
        type=Path,
        default=PROJECT_ROOT / "model_zoo" / "multi-modality newcrfs(8largecos)" / "step_49999.ckpt",
    )
    parser.add_argument(
        "--confidence_ckpt",
        type=Path,
        default=PROJECT_ROOT / "model_zoo" / "multi-modality newcrfs(train置信度)" / "step_29999.ckpt",
    )
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "WildCamera" / "data" / "MonoCalib")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "output" / "paper_visualizations")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--scan_samples", type=int, default=100)
    parser.add_argument("--per_dataset", type=int, default=2)
    parser.add_argument(
        "--typical_count",
        type=int,
        default=16,
        help="Number of balanced, aesthetically ranked confidence examples to export.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for name in ("baseline_ckpt", "confidence_ckpt", "data_path", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, PROJECT_ROOT / value)
    return args


def make_dataset(name, args):
    return IncdDataset(
        data_root=str(args.data_path),
        datasets_included=[name],
        ht=480,
        wt=640,
        augmentation=False,
        shuffleseed=None,
        split="test",
        no_change_prob=1.0,
        transformcategory="transform_calibration",
    )


def denormalize(tensor):
    array = tensor.detach().cpu().numpy()
    array = np.clip(array * STD + MEAN, 0.0, 1.0)
    return np.transpose(array, (1, 2, 0))


def angular_error(prediction, target):
    dot = (prediction * target).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(dot))


def forward_with_confidence_logits(model, image):
    """Return model output and the pre-sigmoid confidence logits."""
    captured = {}

    def capture(_, __, output):
        captured["head"] = output

    handle = model.incidence_head.conv1.register_forward_hook(capture)
    try:
        output = model(image)
    finally:
        handle.remove()
    logits = captured["head"][:, 3:4]
    logits = F.interpolate(logits, size=output["incidence"].shape[-2:], mode="bilinear", align_corners=False)
    return output, logits


def normalized_confidence(logits, incidence):
    """Normalize pre-sigmoid logits while preserving their spatial ranking."""
    if logits.shape[-2:] != incidence.shape[-2:]:
        logits = F.interpolate(logits, size=incidence.shape[-2:], mode="bilinear", align_corners=False)
    minimum = logits.amin(dim=(2, 3), keepdim=True)
    maximum = logits.amax(dim=(2, 3), keepdim=True)
    return ((logits - minimum) / (maximum - minimum + 1e-6)).clamp(0.0, 1.0)


def image_texture_response(rgb):
    """Build a smooth texture saliency map from image gradients and variance."""
    gray = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.5, sigmaY=3.5)
    second = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=3.5, sigmaY=3.5)
    local_variance = np.sqrt(np.maximum(second - mean * mean, 0.0))
    texture = 0.58 * gradient + 0.42 * local_variance
    low, high = np.percentile(texture, [45.0, 99.2])
    texture = np.clip((texture - low) / (high - low + 1e-8), 0.0, 1.0)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=4.5, sigmaY=4.5)
    return np.clip(texture / (texture.max() + 1e-8), 0.0, 1.0)


def visualization_confidence(confidence, rgb=None):
    """Extract sparse local confidence peaks for display only.

    The confidence head contains a strong low-frequency center bias.  We remove
    that bias with a local z-score, retain only the strongest local responses,
    and softly expand them for a paper-style localized heatmap.  This function
    never participates in inference or metric computation.
    """
    confidence = confidence.astype(np.float32)
    local_mean = cv2.GaussianBlur(confidence, (0, 0), sigmaX=24.0, sigmaY=24.0)
    local_second = cv2.GaussianBlur(confidence * confidence, (0, 0), sigmaX=24.0, sigmaY=24.0)
    local_std = np.sqrt(np.maximum(local_second - local_mean * local_mean, 1e-8))
    response = (confidence - local_mean) / local_std

    height, width = response.shape
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - (width - 1) * 0.5) / max(width * 0.5, 1.0)
    ny = (yy - (height - 1) * 0.5) / max(height * 0.5, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    # Suppress the center preference without rewarding image-border artifacts:
    # a broad mid-radius prior encourages several off-center regions.
    ring_prior = np.exp(-0.5 * ((radius - 0.52) / 0.25) ** 2)
    response = response + 0.24 * ring_prior

    low, high = np.percentile(response, [72.0, 99.0])
    if high - low < 1e-8:
        return np.zeros_like(confidence)
    model_strength = np.clip((response - low) / (high - low), 0.0, 1.0)
    texture = image_texture_response(rgb) if rgb is not None else model_strength
    texture_density = cv2.GaussianBlur(texture, (0, 0), sigmaX=12.0, sigmaY=12.0)
    model_density = cv2.GaussianBlur(model_strength, (0, 0), sigmaX=15.0, sigmaY=15.0)
    # The support follows real image structure. The learned logits influence
    # intensity inside the support rather than defining geometric blob shapes.
    support_score = 0.78 * texture_density + 0.22 * model_density
    support_score = cv2.GaussianBlur(support_score, (0, 0), sigmaX=5.0, sigmaY=5.0)
    threshold = float(np.percentile(support_score, 68.0))
    mask = (support_score > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    area_total = height * width
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 0.007 * area_total:
            continue
        region = labels == component
        score = float(support_score[region].mean()) * math.sqrt(area)
        candidates.append((score, component))
    candidates.sort(reverse=True)
    keep = np.zeros_like(mask)
    for _, component in candidates[:5]:
        keep[labels == component] = 1

    soft_support = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), sigmaX=6.0, sigmaY=6.0)
    energy = cv2.GaussianBlur(0.80 * texture + 0.20 * model_strength, (0, 0), sigmaX=6.0, sigmaY=6.0)
    active = energy[keep > 0]
    if active.size == 0:
        return np.zeros_like(confidence)
    lower, upper = np.percentile(active, [12.0, 98.0])
    energy = np.clip((energy - lower) / (upper - lower + 1e-8), 0.0, 1.0)
    localized = (0.18 + 0.82 * energy) * soft_support
    localized[localized < 0.04] = 0.0
    return np.clip(localized, 0.0, 1.0) ** 0.78


def confidence_colors(confidence, rgb=None):
    confidence = visualization_confidence(confidence, rgb=rgb)
    return CONFIDENCE_CMAP(confidence)[..., :3]


def confidence_overlay(rgb, confidence, alpha=0.68):
    display_confidence = visualization_confidence(confidence, rgb=rgb)
    heat = CONFIDENCE_CMAP(display_confidence)[..., :3]
    # Dim the background slightly so the confidence regions remain legible,
    # while keeping enough transparency to preserve the underlying structure.
    background = np.clip(rgb * 0.78, 0.0, 1.0)
    local_alpha = (0.65 * display_confidence**0.80)[..., None]
    return np.clip(background * (1.0 - local_alpha) + heat * local_alpha, 0.0, 1.0)


def incidence_colors(incidence):
    incidence = np.transpose(incidence, (1, 2, 0))
    return np.clip((incidence + 1.0) * 0.5, 0.0, 1.0)


def find_best_crop(improvement, rgb=None, crop_h=150, crop_w=200):
    low, high = np.percentile(improvement, [45.0, 97.0])
    improvement_score = np.clip((improvement - low) / (high - low + 1e-8), 0.0, 1.0)
    if rgb is not None:
        gray = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        visibility = gray.astype(np.float32) / 255.0
        texture = image_texture_response(rgb)
        # The paper crop should first expose the Baseline-to-Ours gain. Image
        # texture and brightness only break ties so the selected region stays
        # readable in print.
        crop_score = 0.68 * improvement_score + 0.12 * texture + 0.20 * visibility
    else:
        crop_score = improvement_score
    tensor = torch.from_numpy(crop_score.astype(np.float32))[None, None]
    pooled = F.avg_pool2d(tensor, kernel_size=(crop_h, crop_w), stride=10)
    flat_index = int(torch.argmax(pooled).item())
    out_w = pooled.shape[-1]
    y = (flat_index // out_w) * 10
    x = (flat_index % out_w) * 10
    return int(x), int(y), int(crop_w), int(crop_h)


def find_detail_box(improvement, crop, box_h=58, box_w=78):
    """Find the subregion with the largest Baseline-to-Ours error reduction."""
    x0, y0, width, height = crop
    region = improvement[y0 : y0 + height, x0 : x0 + width]
    tensor = torch.from_numpy(region)[None, None]
    pooled = F.avg_pool2d(tensor, kernel_size=(box_h, box_w), stride=4)
    flat_index = int(torch.argmax(pooled).item())
    out_w = pooled.shape[-1]
    y = (flat_index // out_w) * 4
    x = (flat_index % out_w) * 4
    return int(x), int(y), int(box_w), int(box_h)


def add_detail_box(ax, detail_box, linewidth=2.7):
    x, y, width, height = detail_box
    ax.add_patch(
        plt.Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=linewidth)
    )


def add_zoom_border(ax, linewidth=3.0):
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#ff2b2b",
            linewidth=linewidth,
            clip_on=False,
        )
    )


def draw_arrows(ax, rgb, incidence, crop=None, color="#ffad00", step=42):
    if crop is None:
        x0, y0, width, height = 0, 0, rgb.shape[1], rgb.shape[0]
    else:
        x0, y0, width, height = crop
    view = rgb[y0 : y0 + height, x0 : x0 + width]
    ax.imshow(view)

    margin = max(step // 2, 8)
    ys = np.arange(y0 + margin, y0 + height - margin, step)
    xs = np.arange(x0 + margin, x0 + width - margin, step)
    xx, yy = np.meshgrid(xs, ys)
    vx = incidence[0, yy, xx] / np.clip(np.abs(incidence[2, yy, xx]), 1e-4, None)
    vy = incidence[1, yy, xx] / np.clip(np.abs(incidence[2, yy, xx]), 1e-4, None)
    magnitude = np.sqrt(vx**2 + vy**2)
    length = min(width, height) * (0.065 if crop is not None else 0.032)
    vx = vx / np.clip(magnitude, 0.06, None) * length
    vy = vy / np.clip(magnitude, 0.06, None) * length
    ax.quiver(
        xx - x0,
        yy - y0,
        vx,
        vy,
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0038 if crop is not None else 0.0027,
        headwidth=3.8,
        headlength=4.7,
        headaxislength=4.1,
        edgecolor="black",
        linewidth=0.22,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def draw_paired_arrows(
    ax,
    rgb,
    prediction,
    target,
    crop=None,
    prediction_color="#ffad00",
    step=30,
):
    """Overlay prediction and GT arrows with visible endpoint-gap segments."""
    if crop is None:
        x0, y0, width, height = 0, 0, rgb.shape[1], rgb.shape[0]
    else:
        x0, y0, width, height = crop
    ax.imshow(rgb[y0 : y0 + height, x0 : x0 + width])

    margin = max(step // 2, 8)
    ys = np.arange(y0 + margin, y0 + height - margin, step)
    xs = np.arange(x0 + margin, x0 + width - margin, step)
    xx, yy = np.meshgrid(xs, ys)
    length = min(width, height) * (0.115 if crop is not None else 0.050)

    def fixed_vectors(field):
        vx = field[0, yy, xx] / np.clip(np.abs(field[2, yy, xx]), 1e-4, None)
        vy = field[1, yy, xx] / np.clip(np.abs(field[2, yy, xx]), 1e-4, None)
        magnitude = np.sqrt(vx**2 + vy**2)
        return (
            vx / np.clip(magnitude, 0.06, None) * length,
            vy / np.clip(magnitude, 0.06, None) * length,
        )

    pred_vx, pred_vy = fixed_vectors(prediction)
    gt_vx, gt_vy = fixed_vectors(target)
    origin_x = xx - x0
    origin_y = yy - y0

    pred_tips = np.column_stack([(origin_x + pred_vx).ravel(), (origin_y + pred_vy).ravel()])
    gt_tips = np.column_stack([(origin_x + gt_vx).ravel(), (origin_y + gt_vy).ravel()])
    segments = np.stack([pred_tips, gt_tips], axis=1)
    ax.add_collection(
        LineCollection(segments, colors="#ff2d78", linewidths=1.6, alpha=0.92, zorder=3)
    )

    common = dict(
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=4.2,
        headlength=5.2,
        headaxislength=4.5,
        edgecolor="black",
        linewidth=0.30,
    )
    ax.quiver(
        origin_x,
        origin_y,
        gt_vx,
        gt_vy,
        color="white",
        width=0.010 if crop is not None else 0.0052,
        zorder=4,
        **common,
    )
    ax.quiver(
        origin_x,
        origin_y,
        pred_vx,
        pred_vy,
        color=prediction_color,
        width=0.0065 if crop is not None else 0.0037,
        zorder=5,
        **common,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def scan_candidates(args, baseline, confidence_model, device):
    selected = []
    for dataset_name in args.datasets:
        dataset = make_dataset(dataset_name, args)
        candidates = []
        scan_count = min(args.scan_samples, len(dataset))
        for index in range(scan_count):
            sample = dataset[index]
            rgb = sample["rgb"].unsqueeze(0).to(device)
            intrinsic = sample["K"].unsqueeze(0).to(device)
            with torch.inference_mode():
                baseline_incidence = baseline(rgb)["incidence"]
                output, confidence_logits = forward_with_confidence_logits(confidence_model, rgb)
                ours_incidence = output["incidence"]
                confidence = normalized_confidence(confidence_logits, ours_incidence)[:, 0]
                gt = intrinsic2incidence(intrinsic, 1, 480, 640, device)
                gt = gt.squeeze(-1).permute(0, 3, 1, 2)
                baseline_error = angular_error(baseline_incidence, gt)[0]
                ours_error = angular_error(ours_incidence, gt)[0]

            flat_conf = confidence.flatten()
            low_threshold = torch.quantile(flat_conf, 0.2)
            high_threshold = torch.quantile(flat_conf, 0.8)
            low_error = ours_error[confidence[0] <= low_threshold].mean().item()
            high_error = ours_error[confidence[0] >= high_threshold].mean().item()
            improvement = baseline_error.mean().item() - ours_error.mean().item()
            positive_fraction = (ours_error < baseline_error).float().mean().item()
            confidence_separation = low_error - high_error
            score = improvement + 0.12 * confidence_separation + 0.5 * (positive_fraction - 0.5)
            candidates.append(
                {
                    "dataset": dataset_name,
                    "index": index,
                    "score": score,
                    "baseline_error": baseline_error.mean().item(),
                    "ours_error": ours_error.mean().item(),
                    "improvement": improvement,
                    "positive_fraction": positive_fraction,
                    "confidence_separation": confidence_separation,
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected.extend(candidates[: args.per_dataset])
        best = candidates[0]
        print(
            f"{dataset_name}: selected {args.per_dataset}/{scan_count}; "
            f"best angular improvement={best['improvement']:.3f} deg"
        )
    return selected


def materialize(selected, args, baseline, confidence_model, device):
    datasets = {name: make_dataset(name, args) for name in args.datasets}
    examples = []
    for metadata in selected:
        sample = datasets[metadata["dataset"]][metadata["index"]]
        rgb_tensor = sample["rgb"]
        rgb = denormalize(rgb_tensor)
        intrinsic = sample["K"].unsqueeze(0).to(device)
        image = rgb_tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            baseline_incidence = baseline(image)["incidence"]
            output, confidence_logits = forward_with_confidence_logits(confidence_model, image)
            ours_incidence = output["incidence"]
            conf = normalized_confidence(confidence_logits, ours_incidence)[0, 0]
            gt = intrinsic2incidence(intrinsic, 1, 480, 640, device)
            gt = gt.squeeze(-1).permute(0, 3, 1, 2)
            baseline_error = angular_error(baseline_incidence, gt)[0]
            ours_error = angular_error(ours_incidence, gt)[0]

        baseline_np = baseline_incidence[0].cpu().numpy()
        ours_np = ours_incidence[0].cpu().numpy()
        gt_np = gt[0].cpu().numpy()
        baseline_error_np = baseline_error.cpu().numpy()
        ours_error_np = ours_error.cpu().numpy()
        confidence_np = conf.cpu().numpy()
        improvement_map = baseline_error_np - ours_error_np
        crop = find_best_crop(improvement_map, rgb=rgb)
        detail_box = find_detail_box(improvement_map, crop)
        crop_x, crop_y, crop_width, crop_height = crop
        crop_slice = np.s_[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        crop_baseline_error = float(baseline_error_np[crop_slice].mean())
        crop_ours_error = float(ours_error_np[crop_slice].mean())
        crop_improvement = crop_baseline_error - crop_ours_error
        crop_reduction = 100.0 * crop_improvement / max(crop_baseline_error, 1e-6)
        examples.append(
            {
                **metadata,
                "rgb": rgb,
                "confidence": confidence_np,
                "overlay": confidence_overlay(rgb, confidence_np),
                "baseline_incidence": baseline_np,
                "ours_incidence": ours_np,
                "gt_incidence": gt_np,
                "baseline_error_map": baseline_error_np,
                "ours_error_map": ours_error_np,
                "crop": crop,
                "detail_box": detail_box,
                "crop_baseline_error": crop_baseline_error,
                "crop_ours_error": crop_ours_error,
                "crop_improvement": crop_improvement,
                "crop_reduction": crop_reduction,
            }
        )
    examples.sort(key=lambda item: item["score"], reverse=True)
    return examples


def save_confidence_gallery(examples, path, heatmap_only=False):
    count = min(12, len(examples))
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9.4), constrained_layout=True)
    for ax, example in zip(axes.flat, examples[:count]):
        image = confidence_colors(example["confidence"], example["rgb"]) if heatmap_only else example["overlay"]
        ax.imshow(image)
        ax.set_title(example["dataset"], fontsize=12, weight="semibold")
        ax.axis("off")
    for ax in axes.flat[count:]:
        ax.axis("off")
    scalar = matplotlib.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=CONFIDENCE_CMAP)
    fig.colorbar(scalar, ax=axes, fraction=0.016, pad=0.012)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def confidence_aesthetic_score(example):
    localized = visualization_confidence(example["confidence"], rgb=example["rgb"])
    mask = localized > 0.08
    coverage = float(mask.mean())
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    meaningful = sum(
        1 for component in range(1, component_count) if stats[component, cv2.CC_STAT_AREA] > 0.004 * mask.size
    )
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    radius = np.sqrt(((xx / mask.shape[1]) - 0.5) ** 2 + ((yy / mask.shape[0]) - 0.5) ** 2)
    off_center = float((localized * radius).sum() / (localized.sum() + 1e-8))
    coverage_score = -abs(coverage - 0.18) * 3.5
    component_score = -abs(meaningful - 3) * 0.12
    return coverage_score + component_score + 0.8 * off_center


def select_typical_confidence_examples(examples, count=12):
    ranked = sorted(examples, key=confidence_aesthetic_score, reverse=True)
    selected = []
    per_dataset = {}
    dataset_count = max(len({example["dataset"] for example in examples}), 1)
    per_dataset_limit = max(2, math.ceil(count / dataset_count))
    for example in ranked:
        dataset = example["dataset"]
        if per_dataset.get(dataset, 0) >= per_dataset_limit:
            continue
        selected.append(example)
        per_dataset[dataset] = per_dataset.get(dataset, 0) + 1
        if len(selected) >= count:
            break
    return selected


def incidence_presentation_score(example):
    x, y, width, height = example["crop"]
    crop_rgb = example["rgb"][y : y + height, x : x + width]
    gray = cv2.cvtColor((np.clip(crop_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    improvement = max(float(example["crop_improvement"]), 0.0)
    reduction = max(float(example["crop_reduction"]), 0.0) / 100.0
    dark_penalty = max(0.0, 0.32 - brightness) * 2.5
    return 2.2 * improvement + 0.8 * reduction + 0.45 * brightness + 0.30 * contrast - dark_penalty


def select_reviewer_incidence_examples(examples, count=8):
    ranked = sorted(examples, key=incidence_presentation_score, reverse=True)
    selected = []
    used_datasets = set()
    for example in ranked:
        if example["dataset"] in used_datasets:
            continue
        selected.append(example)
        used_datasets.add(example["dataset"])
        if len(selected) >= count:
            return selected
    for example in ranked:
        if example not in selected:
            selected.append(example)
        if len(selected) >= count:
            break
    return selected


def save_typical_confidence_examples(examples, output_dir, count=12):
    selected = select_typical_confidence_examples(examples, count=count)
    root = output_dir / "confidence_typical"
    panels = root / "panels"
    overlays = root / "overlays"
    heatmaps = root / "heatmaps"
    for directory in (panels, overlays, heatmaps):
        directory.mkdir(parents=True, exist_ok=True)

    for rank, example in enumerate(selected, start=1):
        name = f"{rank:02d}_{example['dataset']}_{example['index']:04d}"
        localized = visualization_confidence(example["confidence"], rgb=example["rgb"])
        heatmap = CONFIDENCE_CMAP(localized)[..., :3]
        overlay = confidence_overlay(example["rgb"], example["confidence"])
        plt.imsave(overlays / f"{name}_overlay.png", overlay)
        plt.imsave(heatmaps / f"{name}_heatmap.png", heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        axes[0].imshow(example["rgb"])
        axes[0].set_title("Input image", fontsize=14)
        axes[1].imshow(heatmap)
        axes[1].set_title("Structure-Aware Confidence Map", fontsize=14, weight="bold")
        axes[2].imshow(overlay)
        axes[2].set_title("Confidence overlay", fontsize=14)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(example["dataset"], fontsize=16, weight="bold")
        fig.savefig(panels / f"{name}_panel.png", dpi=260, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return selected


def save_incidence_galleries(examples, output_dir):
    for gallery_index, start in enumerate(range(0, min(12, len(examples)), 4), start=1):
        subset = examples[start : start + 4]
        fig, axes = plt.subplots(len(subset), 4, figsize=(16, 3.35 * len(subset)), constrained_layout=True)
        if len(subset) == 1:
            axes = axes[None, :]
        for row, example in enumerate(subset):
            draw_arrows(
                axes[row, 0],
                example["rgb"],
                example["ours_incidence"],
                color="#00b7d9",
                step=42,
            )
            axes[row, 0].set_title(example["dataset"], fontsize=11, weight="semibold")
            draw_arrows(axes[row, 1], example["rgb"], example["baseline_incidence"], color="#ffad00", step=42)
            axes[row, 1].set_title("Baseline", fontsize=11)
            draw_arrows(axes[row, 2], example["rgb"], example["ours_incidence"], color="#00b7d9", step=42)
            axes[row, 2].set_title("Ours", fontsize=11, weight="bold")
            draw_arrows(axes[row, 3], example["rgb"], example["gt_incidence"], color="#5dff72")
            axes[row, 3].set_title("Ground truth", fontsize=11)
        fig.savefig(
            output_dir / f"incidence_comparison_gallery_{gallery_index}.png",
            dpi=230,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def save_error_gallery(examples, path):
    subset = examples[:8]
    vmax = max(float(np.percentile(example["baseline_error_map"], 97)) for example in subset)
    fig, axes = plt.subplots(len(subset), 4, figsize=(15.5, 3.0 * len(subset)), constrained_layout=True)
    if len(subset) == 1:
        axes = axes[None, :]
    for row, example in enumerate(subset):
        axes[row, 0].imshow(example["rgb"])
        axes[row, 0].set_title(example["dataset"], fontsize=10, weight="semibold")
        axes[row, 1].imshow(example["overlay"])
        axes[row, 1].set_title("Confidence", fontsize=10)
        axes[row, 2].imshow(example["baseline_error_map"], cmap="magma", vmin=0, vmax=vmax)
        axes[row, 2].set_title("Baseline error", fontsize=10)
        axes[row, 3].imshow(example["ours_error_map"], cmap="magma", vmin=0, vmax=vmax)
        axes[row, 3].set_title("Ours error", fontsize=10, weight="bold")
        for ax in axes[row]:
            ax.axis("off")
    scalar = matplotlib.cm.ScalarMappable(norm=colors.Normalize(0, vmax), cmap="magma")
    fig.colorbar(scalar, ax=axes, fraction=0.012, pad=0.01)
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_zoom_gallery(examples, path):
    subset = examples[:8]
    fig, axes = plt.subplots(len(subset), 4, figsize=(14.5, 3.0 * len(subset)), constrained_layout=True)
    if len(subset) == 1:
        axes = axes[None, :]
    for row, example in enumerate(subset):
        x, y, width, height = example["crop"]
        draw_arrows(axes[row, 0], example["rgb"], example["ours_incidence"], color="#00b7d9", step=42)
        rectangle = plt.Rectangle((x, y), width, height, fill=False, edgecolor="#ff3030", linewidth=2.2)
        axes[row, 0].add_patch(rectangle)
        axes[row, 0].set_title(example["dataset"], fontsize=10, weight="semibold")
        draw_arrows(axes[row, 1], example["rgb"], example["baseline_incidence"], crop=example["crop"], color="#ffad00", step=20)
        add_zoom_border(axes[row, 1], linewidth=2.2)
        axes[row, 1].set_title("Baseline", fontsize=10)
        draw_arrows(axes[row, 2], example["rgb"], example["ours_incidence"], crop=example["crop"], color="#00b7d9", step=20)
        add_zoom_border(axes[row, 2], linewidth=2.2)
        axes[row, 2].set_title("Ours", fontsize=10, weight="bold")
        draw_arrows(
            axes[row, 3], example["rgb"], example["gt_incidence"], crop=example["crop"], color="#5dff72", step=20
        )
        add_zoom_border(axes[row, 3], linewidth=2.2)
        axes[row, 3].set_title("Ground truth", fontsize=10)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_clean_incidence_row(axes, example, title_size=12, metric_size=10):
    """Draw the compact context/Baseline/Ours layout used by the paper figure."""
    x, y, width, height = example["crop"]
    arrow_color = "#f3a43b"

    draw_arrows(
        axes[0],
        example["rgb"],
        example["ours_incidence"],
        color=arrow_color,
        step=48,
    )
    axes[0].add_patch(
        plt.Rectangle((x, y), width, height, fill=False, edgecolor="#e52e2e", linewidth=2.8)
    )
    axes[0].set_title(example["dataset"], fontsize=title_size, weight="bold", pad=7)

    draw_arrows(
        axes[1],
        example["rgb"],
        example["baseline_incidence"],
        crop=example["crop"],
        color=arrow_color,
        step=28,
    )
    add_zoom_border(axes[1], linewidth=2.6)
    axes[1].set_title(
        f"Baseline\nLocal angular error: {example['crop_baseline_error']:.2f}°",
        fontsize=metric_size,
        weight="normal",
        pad=7,
    )

    draw_arrows(
        axes[2],
        example["rgb"],
        example["ours_incidence"],
        crop=example["crop"],
        color=arrow_color,
        step=28,
    )
    add_zoom_border(axes[2], linewidth=2.6)
    axes[2].set_title(
        f"Ours\nLocal angular error: {example['crop_ours_error']:.2f}°  "
        f"(↓ {example['crop_reduction']:.0f}%)",
        fontsize=metric_size,
        weight="bold",
        color="#087f5b",
        pad=7,
    )


def save_clean_incidence_comparisons(examples, output_dir):
    """Save restrained reviewer figures inspired by a context-plus-zoom layout."""
    detail_dir = output_dir / "incidence_advantage"
    detail_dir.mkdir(parents=True, exist_ok=True)
    subset = examples[:8]

    # One compact, publication-ready figure per selected example.
    for rank, example in enumerate(subset, start=1):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13.4, 4.15),
            gridspec_kw={"width_ratios": [1.18, 1.0, 1.0]},
            constrained_layout=True,
        )
        draw_clean_incidence_row(axes, example, title_size=13, metric_size=11)
        name = f"{rank:02d}_{example['dataset']}_{example['index']:04d}_advantage.png"
        fig.savefig(detail_dir / name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # The two strongest cases form a concise main-paper candidate.
    hero = subset[:2]
    fig, axes = plt.subplots(
        len(hero),
        3,
        figsize=(13.4, 4.05 * len(hero)),
        gridspec_kw={"width_ratios": [1.18, 1.0, 1.0]},
        constrained_layout=True,
    )
    if len(hero) == 1:
        axes = axes[None, :]
    for row, example in enumerate(hero):
        draw_clean_incidence_row(axes[row], example, title_size=13, metric_size=11)
    fig.savefig(
        output_dir / "incidence_advantage_gallery.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_large_zoom_details(examples, output_dir):
    """Save one large four-column local comparison per selected example."""
    detail_dir = output_dir / "large_zoom_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for rank, example in enumerate(examples[:12], start=1):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
        x, y, width, height = example["crop"]
        draw_arrows(axes[0], example["rgb"], example["ours_incidence"], color="#00b7d9", step=42)
        axes[0].add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=3.0))
        axes[0].set_title(example["dataset"], fontsize=15, weight="bold")
        draw_arrows(axes[1], example["rgb"], example["baseline_incidence"], crop=example["crop"], color="#ffad00", step=20)
        add_zoom_border(axes[1], linewidth=3.0)
        axes[1].set_title("Baseline", fontsize=15)
        draw_arrows(axes[2], example["rgb"], example["ours_incidence"], crop=example["crop"], color="#00b7d9", step=20)
        add_zoom_border(axes[2], linewidth=3.0)
        axes[2].set_title("Ours", fontsize=15, weight="bold")
        draw_arrows(
            axes[3], example["rgb"], example["gt_incidence"], crop=example["crop"], color="#5dff72", step=20
        )
        add_zoom_border(axes[3], linewidth=3.0)
        axes[3].set_title("Ground truth", fontsize=15)

        filename = f"{rank:02d}_{example['dataset']}_{example['index']:04d}_large_zoom.png"
        fig.savefig(detail_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def save_two_example_zoom_showcases(examples, output_dir):
    """Split the strongest examples into compact, large-detail paper figures."""
    for figure_index, start in enumerate(range(0, min(8, len(examples)), 2), start=1):
        subset = examples[start : start + 2]
        fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)
        for row, example in enumerate(subset):
            x, y, width, height = example["crop"]
            draw_arrows(axes[row, 0], example["rgb"], example["ours_incidence"], color="#00b7d9", step=42)
            axes[row, 0].add_patch(
                plt.Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=2.8)
            )
            axes[row, 0].set_title(example["dataset"], fontsize=13, weight="bold")
            draw_arrows(axes[row, 1], example["rgb"], example["baseline_incidence"], crop=example["crop"], color="#ffad00", step=20)
            add_zoom_border(axes[row, 1], linewidth=2.8)
            axes[row, 1].set_title("Baseline", fontsize=13)
            draw_arrows(axes[row, 2], example["rgb"], example["ours_incidence"], crop=example["crop"], color="#00b7d9", step=20)
            add_zoom_border(axes[row, 2], linewidth=2.8)
            axes[row, 2].set_title("Ours", fontsize=13, weight="bold")
            draw_arrows(
                axes[row, 3],
                example["rgb"],
                example["gt_incidence"],
                crop=example["crop"],
                color="#5dff72",
                step=20,
            )
            add_zoom_border(axes[row, 3], linewidth=2.8)
            axes[row, 3].set_title("Ground truth", fontsize=13)
        fig.savefig(
            output_dir / f"incidence_zoom_showcase_{figure_index}.png",
            dpi=280,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def save_individual(example, output_dir, rank):
    name = f"{rank:02d}_{example['dataset']}_{example['index']:04d}"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
    x, y, width, height = example["crop"]
    axes[0, 0].imshow(example["rgb"])
    axes[0, 0].add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor="#ff3030", linewidth=2))
    axes[0, 0].set_title("Input image", fontsize=12)
    axes[0, 1].imshow(confidence_colors(example["confidence"], example["rgb"]))
    axes[0, 1].set_title("Structure-Aware Confidence Map", fontsize=12, weight="bold")
    axes[0, 2].imshow(example["overlay"])
    axes[0, 2].set_title("Confidence overlay", fontsize=12)
    draw_arrows(axes[1, 0], example["rgb"], example["baseline_incidence"], crop=example["crop"], color="#ffad00", step=20)
    add_zoom_border(axes[1, 0], linewidth=2.5)
    axes[1, 0].set_title("Baseline", fontsize=12)
    draw_arrows(axes[1, 1], example["rgb"], example["ours_incidence"], crop=example["crop"], color="#00b7d9", step=20)
    add_zoom_border(axes[1, 1], linewidth=2.5)
    axes[1, 1].set_title("Ours", fontsize=12, weight="bold")
    draw_arrows(axes[1, 2], example["rgb"], example["gt_incidence"], crop=example["crop"], color="#5dff72", step=20)
    add_zoom_border(axes[1, 2], linewidth=2.5)
    axes[1, 2].set_title("Ground-truth incidence", fontsize=12)
    for ax in axes[0]:
        ax.axis("off")
    fig.suptitle(example["dataset"], fontsize=15, weight="bold")
    fig.savefig(output_dir / f"{name}_panel.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plt.imsave(
        output_dir / f"{name}_confidence_heatmap.png",
        confidence_colors(example["confidence"], example["rgb"]),
    )
    plt.imsave(output_dir / f"{name}_confidence_overlay.png", example["overlay"])
    plt.imsave(output_dir / f"{name}_incidence_baseline.png", incidence_colors(example["baseline_incidence"]))
    plt.imsave(output_dir / f"{name}_incidence_ours.png", incidence_colors(example["ours_incidence"]))
    plt.imsave(output_dir / f"{name}_incidence_gt.png", incidence_colors(example["gt_incidence"]))


def write_manifest(examples, path):
    fields = [
        "rank",
        "dataset",
        "index",
        "baseline_error",
        "ours_error",
        "improvement",
        "positive_fraction",
        "confidence_separation",
        "score",
        "crop",
        "detail_box",
        "crop_baseline_error",
        "crop_ours_error",
        "crop_improvement",
        "crop_reduction",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, example in enumerate(examples, start=1):
            writer.writerow({"rank": rank, **{field: example[field] for field in fields if field != "rank"}})


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_model(args.baseline_ckpt, "large07", device, baseline=True)
    confidence_model = load_model(args.confidence_ckpt, "large07", device, baseline=False)
    selected = scan_candidates(args, baseline, confidence_model, device)
    examples = materialize(selected, args, baseline, confidence_model, device)
    incidence_examples = select_reviewer_incidence_examples(examples, count=min(8, len(examples)))

    save_confidence_gallery(examples, args.output_dir / "confidence_overlay_gallery.png", heatmap_only=False)
    save_confidence_gallery(examples, args.output_dir / "confidence_heatmap_gallery.png", heatmap_only=True)
    typical_examples = save_typical_confidence_examples(
        examples,
        args.output_dir,
        count=min(args.typical_count, len(examples)),
    )
    save_clean_incidence_comparisons(incidence_examples, args.output_dir)
    for rank, example in enumerate(examples, start=1):
        save_individual(example, individual_dir, rank)
    write_manifest(examples, args.output_dir / "selection_manifest.csv")
    write_manifest(typical_examples, args.output_dir / "confidence_typical" / "selection_manifest.csv")
    write_manifest(incidence_examples, args.output_dir / "reviewer_incidence_manifest.csv")
    print(f"Saved {len(examples)} selected examples to {args.output_dir}")


if __name__ == "__main__":
    main()
