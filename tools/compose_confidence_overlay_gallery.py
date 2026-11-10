"""Compose overlay-only confidence examples with one shared color bar."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--row_labels", nargs="*", default=None)
    parser.add_argument("--colorbar_fraction", type=float, default=0.85)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf_path", type=Path)
    parser.add_argument("--colormap", choices=["turbo", "confidence_soft"], default="turbo")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.colorbar_fraction <= 1.0:
        raise ValueError("colorbar_fraction must be in (0, 1]")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "overlay_only"
    individual_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for rank, path in enumerate(args.inputs, start=1):
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
        shutil.copy2(path, individual_dir / f"{rank:02d}_{path.name}")

    columns = args.columns
    rows = (len(images) + columns - 1) // columns
    figure = plt.figure(figsize=(4.15 * columns + 0.75, 3.32 * rows), facecolor="white")
    left_margin = 0.052 if args.row_labels else 0.008
    grid = figure.add_gridspec(
        rows,
        columns + 1,
        width_ratios=[1.0] * columns + [0.045],
        wspace=0.018,
        hspace=0.018,
        left=left_margin,
        right=0.952,
        top=0.965,
        bottom=0.035,
    )

    image_axes = []
    for index in range(rows * columns):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index], interpolation="lanczos")
            image_axes.append(axis)

    color_axis = figure.add_subplot(grid[:, -1])
    palette = matplotlib.colormaps["turbo"]
    if args.colormap == "confidence_soft":
        palette = colors.LinearSegmentedColormap.from_list(
            "confidence_turbo_soft", palette(np.linspace(0.05, 0.94, 256))
        )
    scalar = matplotlib.cm.ScalarMappable(
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        cmap=palette,
    )
    colorbar = figure.colorbar(scalar, cax=color_axis, orientation="vertical", ticks=[0.0, 1.0])
    colorbar.ax.tick_params(labelsize=15, length=0, pad=5)
    colorbar.outline.set_linewidth(0.8)

    # Match the visible image extent, including imshow's aspect-ratio adjustment.
    figure.canvas.draw()
    image_bottom = min(axis.get_position().y0 for axis in image_axes)
    image_top = max(axis.get_position().y1 for axis in image_axes)
    bar_height = (image_top - image_bottom) * args.colorbar_fraction
    bar_position = color_axis.get_position()
    color_axis.set_position(
        [bar_position.x0, (image_top + image_bottom - bar_height) / 2,
         bar_position.width, bar_height]
    )

    if args.row_labels:
        if len(args.row_labels) != rows:
            raise ValueError(f"Expected {rows} row labels, got {len(args.row_labels)}")
        usable_height = 0.965 - 0.035
        for row, label in enumerate(args.row_labels):
            y = 0.965 - usable_height * (row + 0.5) / rows
            figure.text(
                0.020,
                y,
                label,
                rotation=90,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="semibold",
            )

    output = args.output_dir / "confidence_overlay_gallery.png"
    figure.savefig(output, dpi=args.dpi, facecolor="white")
    if args.pdf_path:
        args.pdf_path.parent.mkdir(parents=True, exist_ok=True)
        matplotlib.rcParams["pdf.fonttype"] = 42
        colorbar.solids.set_rasterized(False)
        # Overlap adjacent vector cells to avoid white hairlines in PDF viewers.
        colorbar.solids.set_edgecolor("face")
        figure.savefig(args.pdf_path, dpi=args.dpi, facecolor="white")
    plt.close(figure)
    print(f"Saved {len(images)} overlay images and gallery to {args.output_dir}")


if __name__ == "__main__":
    main()
