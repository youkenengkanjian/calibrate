"""Run the confidence-component ablation on the WildCamera benchmark.

The five rows deliberately share the same samples, RANSAC seeds, and test
data.  Rows 2--5 also share one confidence-trained checkpoint, so the only
difference among them is whether confidence is used for sampling and/or
scoring.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
)
sys.path.insert(0, str(PROJECT_ROOT))

from WildCamera.datasets.IncdDataset import IncdDataset
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from tools.calibrator import MonocularCalibrator
from tools.evaluation import compute_intrinsic_measure


TRAIN_DATASETS = [
    "Nuscenes",
    "KITTI",
    "Cityscapes",
    "NYUv2",
    "ARKitScenes",
    "MegaDepth",
    "SUN3D",
    "MVImgNet",
    "Objectron",
]
TEST_DATASETS = ["Waymo", "BIWIRGBDID", "RGBD", "ScanNet", "MVS", "Scenes11"]
ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS
METRICS = ["error_fx", "error_fy", "error_f", "error_bx", "error_by", "error_b"]

VARIANTS = OrderedDict(
    [
        ("baseline_uniform", {"model": "baseline", "sampling": False, "scoring": False}),
        ("confidence_training", {"model": "confidence", "sampling": False, "scoring": False}),
        ("training_sampling", {"model": "confidence", "sampling": True, "scoring": False}),
        ("training_scoring", {"model": "confidence", "sampling": False, "scoring": True}),
        ("full", {"model": "confidence", "sampling": True, "scoring": True}),
    ]
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
        default=PROJECT_ROOT / "model_zoo" / "multi-modalitynewcrfs(42499)" / "step_49999.ckpt",
    )
    parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "WildCamera" / "data" / "MonoCalib")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "output" / "confidence_ablation")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--camera_model", choices=["4d", "1d", "both"], default="4d")
    parser.add_argument("--encoder", default="large07")
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--l1_th", type=float, default=0.02)
    parser.add_argument("--ransac_trials", type=int, default=2048)
    parser.add_argument("--ransac_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Per-dataset limit for smoke tests only")
    parser.add_argument("--confidence_transform", choices=["continuous", "top_p"], default="continuous")
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--confidence_gamma", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true", help="Disable per-sample progress bars")
    args = parser.parse_args()
    # GenericDataset changes the process working directory at import time.
    # Resolve user-supplied relative paths against the repository root so the
    # benchmark does not silently read from or write to WildCamera/WildCamera.
    for name in ("baseline_ckpt", "confidence_ckpt", "data_path", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, PROJECT_ROOT / value)
    return args


def _strip_module_prefix(state):
    return OrderedDict((key.removeprefix("module."), value) for key, value in state.items())


def load_model(checkpoint_path: Path, encoder: str, device: torch.device, baseline: bool):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    state = _strip_module_prefix(torch.load(checkpoint_path, map_location="cpu"))
    head_weight_key = "incidence_head.conv1.weight"
    head_bias_key = "incidence_head.conv1.bias"
    output_channels = state[head_weight_key].shape[0]

    if baseline:
        if output_channels != 3:
            raise ValueError(f"Baseline checkpoint must have a 3-channel head, got {output_channels}: {checkpoint_path}")
        # The current network class has a fourth confidence channel.  Padding
        # that channel with zeros preserves the baseline's first three outputs
        # exactly; the unused confidence map becomes a constant sigmoid(0).
        state[head_weight_key] = F.pad(state[head_weight_key], (0, 0, 0, 0, 0, 0, 0, 1))
        state[head_bias_key] = F.pad(state[head_bias_key], (0, 1))
    elif output_channels != 4:
        raise ValueError(f"Confidence checkpoint must have a 4-channel head, got {output_channels}: {checkpoint_path}")

    model = NEWCRFIF(version=encoder, pretrained=None)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def prepare_confidence(confidence, incidence, transform, top_p, gamma):
    if confidence.shape[-2:] != incidence.shape[-2:]:
        confidence = F.interpolate(confidence, size=incidence.shape[-2:], mode="bilinear", align_corners=False)

    minimum = confidence.amin(dim=(2, 3), keepdim=True)
    maximum = confidence.amax(dim=(2, 3), keepdim=True)
    confidence = ((confidence - minimum) / (maximum - minimum + 1e-6)).clamp(0.0, 1.0)
    confidence = confidence.pow(gamma)

    if transform == "top_p":
        if not 0.0 < top_p <= 1.0:
            raise ValueError("--top_p must be in (0, 1]")
        flat = confidence.flatten(1)
        keep = max(int(np.ceil(top_p * flat.shape[1])), 1)
        threshold = torch.topk(flat, k=keep, dim=1, sorted=False).values.amin(dim=1, keepdim=True)
        confidence = (flat >= threshold).reshape_as(confidence).to(confidence.dtype)
    return confidence


def make_dataset(name, args):
    return IncdDataset(
        data_root=str(args.data_path),
        datasets_included=[name],
        ht=args.input_height,
        wt=args.input_width,
        augmentation=name in TRAIN_DATASETS,
        shuffleseed=None,
        split="test",
        augscale=2.0,
        no_change_prob=0.0,
        transformcategory="transform_calibration",
    )


def calibrate(calibrator, incidence, restoration_ratio, variant, weights, camera_model, trials, seed):
    sampling_weights = weights if variant["sampling"] else None
    scoring_weights = weights if variant["scoring"] else None
    if camera_model == "4d":
        return calibrator.calibrate_camera_4DoF(
            incidence,
            RANSAC_trial=trials,
            sampling_weights=sampling_weights,
            scoring_weights=scoring_weights,
            seed=seed,
        )
    return calibrator.calibrate_camera_1DoF(
        incidence,
        r=restoration_ratio,
        RANSAC_trial=trials,
        sampling_weights=sampling_weights,
        scoring_weights=scoring_weights,
        seed=seed,
    )


def measure(k_est, k_gt, height, width):
    values = compute_intrinsic_measure(Kest=k_est.clone(), Kgt=k_gt.clone(), h=height, w=width)
    return {name: float(value) for name, value in zip(METRICS, values)}


def aggregate(results):
    summary = {}
    groups = {
        "all": list(results),
        "seen": [name for name in results if name in TRAIN_DATASETS],
        "unseen": [name for name in results if name in TEST_DATASETS],
    }
    for group_name, dataset_names in groups.items():
        if not dataset_names:
            continue
        summary[group_name] = {}
        for camera_model in next(iter(results.values())):
            summary[group_name][camera_model] = {}
            for variant_name in results[dataset_names[0]][camera_model]:
                summary[group_name][camera_model][variant_name] = {
                    metric: float(np.mean([results[name][camera_model][variant_name][metric] for name in dataset_names]))
                    for metric in METRICS
                }
    return summary


def write_outputs(args, results, metadata):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(results) if results else {}
    payload = {"metadata": metadata, "per_dataset": results, "summary": summary}
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with (args.output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "camera_model", "variant"] + METRICS)
        writer.writeheader()
        for dataset_name, camera_results in results.items():
            for camera_model, variant_results in camera_results.items():
                for variant_name, metrics in variant_results.items():
                    writer.writerow(
                        {
                            "dataset": dataset_name,
                            "split": "seen" if dataset_name in TRAIN_DATASETS else "unseen",
                            "camera_model": camera_model,
                            "variant": variant_name,
                            **metrics,
                        }
                    )

    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "camera_model", "variant"] + METRICS)
        writer.writeheader()
        for group_name, camera_results in summary.items():
            for camera_model, variant_results in camera_results.items():
                for variant_name, metrics in variant_results.items():
                    writer.writerow({"group": group_name, "camera_model": camera_model, "variant": variant_name, **metrics})


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full benchmark")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    variants = OrderedDict((name, VARIANTS[name]) for name in args.variants)
    needs_baseline = any(variant["model"] == "baseline" for variant in variants.values())
    baseline_model = load_model(args.baseline_ckpt, args.encoder, device, baseline=True) if needs_baseline else None
    confidence_model = load_model(args.confidence_ckpt, args.encoder, device, baseline=False)
    calibrator = MonocularCalibrator(l1_th=args.l1_th).to(device)
    calibrator.RANSAC_num = args.ransac_samples

    camera_models = ["4d", "1d"] if args.camera_model == "both" else [args.camera_model]
    metadata = {
        "baseline_checkpoint": str(args.baseline_ckpt.resolve()),
        "confidence_checkpoint": str(args.confidence_ckpt.resolve()),
        "data_path": str(args.data_path.resolve()),
        "datasets": args.datasets,
        "camera_models": camera_models,
        "confidence_transform": args.confidence_transform,
        "top_p": args.top_p if args.confidence_transform == "top_p" else None,
        "confidence_gamma": args.confidence_gamma,
        "l1_threshold": args.l1_th,
        "ransac_trials": args.ransac_trials,
        "ransac_samples": args.ransac_samples,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "quiet": args.quiet,
        "variants": variants,
    }
    results = {}

    for dataset_offset, dataset_name in enumerate(args.datasets):
        dataset = make_dataset(dataset_name, args)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
        sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
        sums = {
            camera_model: {variant: {metric: 0.0 for metric in METRICS} for variant in variants}
            for camera_model in camera_models
        }

        progress = tqdm(loader, total=sample_count, desc=dataset_name, dynamic_ncols=True, disable=args.quiet)
        for sample_index, batch in enumerate(progress):
            if sample_index >= sample_count:
                break
            rgb = batch["rgb"].to(device, non_blocking=True)
            k_gt = batch["K"].to(device, non_blocking=True).squeeze(0)
            restoration_ratio = float(batch["aspect_ratio_restoration"].item())

            baseline_out = baseline_model(rgb) if baseline_model is not None else None
            confidence_out = confidence_model(rgb)
            baseline_incidence = baseline_out["incidence"] if baseline_out is not None else None
            confidence_incidence = confidence_out["incidence"]
            weights = prepare_confidence(
                confidence_out["conf"],
                confidence_incidence,
                args.confidence_transform,
                args.top_p,
                args.confidence_gamma,
            )
            _, _, height, width = rgb.shape
            sample_seed = args.seed + dataset_offset * 1_000_000 + sample_index

            for camera_model in camera_models:
                for variant_name, variant in variants.items():
                    incidence = baseline_incidence if variant["model"] == "baseline" else confidence_incidence
                    variant_weights = None if variant["model"] == "baseline" else weights
                    k_est = calibrate(
                        calibrator,
                        incidence,
                        restoration_ratio,
                        variant,
                        variant_weights,
                        camera_model,
                        args.ransac_trials,
                        sample_seed,
                    )
                    metrics = measure(k_est, k_gt, height, width)
                    for metric, value in metrics.items():
                        sums[camera_model][variant_name][metric] += value

        results[dataset_name] = {
            camera_model: {
                variant_name: {metric: total / sample_count for metric, total in metrics.items()}
                for variant_name, metrics in variant_results.items()
            }
            for camera_model, variant_results in sums.items()
        }
        write_outputs(args, results, metadata)
        if not args.quiet:
            print(json.dumps({dataset_name: results[dataset_name]}, indent=2))

    write_outputs(args, results, metadata)
    print(f"Saved ablation results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
