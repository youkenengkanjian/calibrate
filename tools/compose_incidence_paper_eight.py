"""Recompose the eight user-selected examples with one shared bottom legend.

Uses the existing float32 prediction cache and existing ROI/A/B coordinates.
No inference, sample selection, or modification of predicted directions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("C:/Users/admin/Desktop/入射场图/改好")
CACHE_ROOT = ROOT / "output/incidence_gt_100_2026-11-08"
OUT = ROOT / "output/incidence_paper_eight_2026-11-08"
PDF_OUT = ROOT / "output/pdf/incidence_paper_eight_2026-11-08"
MM = 1 / 25.4
ORANGE, GREEN, RED, PINK = "#ed9a29", "#07825d", "#e73435", "#d92882"
INK = "#172437"
ORDER = [
    "006_SUN3D_0081.png", "012_NYUv2_0032.png", "020_NYUv2_0091.png",
    "015_MVImgNet_0019.png", "003_Cityscapes_0208.png",
    "011_Cityscapes_0163.png", "010_KITTI_0014.png", "008_Objectron_0015.png",
]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .4,
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directions(field, length=25.):
    xy = field[:2].astype(np.float64)
    norm = np.linalg.norm(xy, axis=0, keepdims=True)
    return np.divide(xy * length, norm, out=np.zeros_like(xy), where=norm > 1e-8)


def arrows(ax, xy, uv, color, width=.0055, zorder=5, edge=.08):
    return ax.quiver(*xy, *uv, color=color, angles="xy", scale_units="xy",
                     scale=1, width=width, headwidth=3.8, headlength=4.7,
                     headaxislength=4.2, edgecolor="#202833", linewidth=edge,
                     minlength=0, zorder=zorder)


def draw_sample(axes, meta, data, compact=False):
    rgb = data["rgb"]
    fields = [data[k] for k in ("baseline", "ours", "gt")]
    x, y, w, h = meta["crop"]
    xx, yy = np.meshgrid(np.arange(30, w - 29, 35), np.arange(30, h - 29, 30))
    xs, ys = xx + x, yy + y
    uv = [directions(f[:, ys, xs]) for f in fields]
    edge = .055 if compact else .09
    ax = axes[0]
    if "rgb_full" in data:
        ax.imshow(data["rgb_full"], interpolation="lanczos", extent=(-.5, 639.5, 479.5, -.5))
    else:
        ax.imshow(rgb, interpolation="none")
    gx, gy = np.meshgrid(np.arange(24, 640 - 23, 48), np.arange(24, 480 - 23, 48))
    arrows(ax, (gx, gy), directions(fields[1][:, gy, gx], 16.), GREEN, .003, edge=edge)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=RED,
                              lw=.55 if compact else .75, zorder=10))
    ax.set(xlim=(0, 640), ylim=(480, 0))
    ax.axis("off")
    for i, ax in enumerate(axes[1:]):
        if "rgb_crop" in data:
            ax.imshow(data["rgb_crop"], interpolation="lanczos", extent=(-.5, w-.5, h-.5, -.5))
        else:
            ax.imshow(rgb[y:y+h, x:x+w], interpolation="none")
        arrows(ax, (xx, yy), uv[2], "white", zorder=3, edge=edge)
        if i < 2:
            tips = [np.column_stack(((xx+v[0]).ravel(), (yy+v[1]).ravel())) for v in (uv[2], uv[i])]
            ax.add_collection(LineCollection(np.stack(tips, 1), colors=PINK,
                                            linewidths=.32 if compact else .48, zorder=4))
            arrows(ax, (xx, yy), uv[i], (ORANGE, GREEN)[i], edge=edge)
        for marker in meta["markers"]:
            mx, my = marker["x"], marker["y"]
            guv = directions(fields[2][:, my, mx])
            cx, cy = mx-x+guv[0]*.5, my-y+guv[1]*.5
            radius = float(np.linalg.norm(guv)*.5+4)
            for color, lw in (("#222b37", .9 if compact else 1.05),
                              ("white", .58 if compact else .72)):
                ax.add_patch(plt.Circle((cx, cy), radius, fill=False, ec=color,
                                       lw=lw, ls=(0, (2.5, 1.7)), zorder=7))
            tx, ty = np.clip(cx-radius+1, 8, w-8), np.clip(cy-radius-3, 8, h-8)
            ax.text(tx, ty, marker["label"], fontsize=4.4 if compact else 5.8,
                    weight="bold", color="white", ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#253244", ec="white", lw=.35))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                  ec=RED, lw=.4 if compact else .55, clip_on=False, zorder=10))
        ax.set(xlim=(0, w), ylim=(h, 0))
        ax.axis("off")


def shared_legend(fig, fontsize=6.5):
    handles = [
        Line2D([], [], color=ORANGE, lw=2.1, label="Baseline"),
        Line2D([], [], color=GREEN, lw=2.1, label="Ours"),
        Line2D([], [], color="white", lw=1.8, label="GT",
               path_effects=[pe.Stroke(linewidth=3, foreground="#526075"), pe.Normal()]),
        Line2D([], [], color=PINK, lw=.75, label="Projected direction discrepancy"),
        Line2D([], [], color="#546174", lw=.7, ls="--", label="A / B: matched rays"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .008),
               ncol=5, frameon=False, fontsize=fontsize, columnspacing=1.2,
               handlelength=1.65, handletextpad=.55, borderpad=.2)


def add_axes(fig, x, y, w, h, total_w, total_h):
    # Positions are millimetres from the top left of the canvas.
    return fig.add_axes((x/total_w, 1-(y+h)/total_h, w/total_w, h/total_h))


def validate(fig, sample_axes):
    assert len(fig.legends) == 1
    for axes in sample_axes:
        glyphs = [q for ax in axes[1:] for q in ax.collections if isinstance(q, Quiver)]
        assert len(glyphs) == 5
        for q in glyphs:
            length = np.hypot(q.U, q.V)
            assert np.allclose(length[length > 0], 25., atol=1e-9)
            assert (q.width, q.headwidth, q.headlength, q.headaxislength) == (.0055, 3.8, 4.7, 4.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        assert "error" not in obj.get_text().lower()
        box = obj.get_window_extent(renderer)
        assert bounds.x0-1 <= box.x0 and bounds.x1+1 >= box.x1, obj.get_text()
        assert bounds.y0-1 <= box.y0 and bounds.y1+1 >= box.y1, obj.get_text()
    legend_box = fig.legends[0].get_window_extent(renderer)
    bottom_image = min(ax.get_window_extent(renderer).y0 for row in sample_axes for ax in row)
    assert legend_box.y1 < bottom_image, "Legend overlaps images"


def gallery(metas, data, compact=False, output_dir=None, pdf_dir=None, png_dpi=600):
    output_dir = Path(output_dir) if output_dir is not None else OUT
    pdf_dir = Path(pdf_dir) if pdf_dir is not None else PDF_OUT
    if compact:
        width, left, right, gap, split, rows = 180., 7., 1.5, .8, 4., 4
        pw = (width-left-right-6*gap-split)/8
        rowgap, groupgap, top, bottom = 1.15, 3., 6., 9.5
    else:
        width, left, right, gap, split, rows = 160., 10.5, 1.5, 1.05, 0., 8
        pw = (width-left-right-3*gap)/4
        rowgap, groupgap, top, bottom = 1.05, 3., 6., 9.5
    ph = pw*.75
    height = top+rows*ph+(rows-2)*rowgap+groupgap+bottom
    fig = plt.figure(figsize=(width*MM, height*MM), dpi=120, facecolor="white")
    ys = [top+r*(ph+rowgap)+(groupgap-rowgap if r >= rows//2 else 0) for r in range(rows)]
    all_axes = []
    for i, meta in enumerate(metas):
        row, block = (i//2, i%2) if compact else (i, 0)
        bx = left + block*(4*pw+3*gap+split)
        axes = [add_axes(fig, bx+c*(pw+gap), ys[row], pw, ph, width, height) for c in range(4)]
        draw_sample(axes, meta, data[i], compact)
        all_axes.append(axes)
        fig.text((bx-1.8)/width, 1-(ys[row]+ph*.5)/height, f"({chr(97+i)})",
                 fontsize=6. if compact else 6.8, ha="center", va="center", color=INK,
                 rotation=0)
    for block in range(2 if compact else 1):
        bx = left+block*(4*pw+3*gap+split)
        for c, title in enumerate(("Full view", "Baseline + GT", "Ours + GT", "GT")):
            fig.text((bx+c*(pw+gap)+pw/2)/width, 1-2.5/height, title,
                     ha="center", va="center", fontsize=6.1 if compact else 7.8,
                     color=GREEN if c == 2 else INK, weight="bold" if c == 2 else "normal")
    for group, label in enumerate(("Indoor", "Outdoor")):
        a, b = group*(rows//2), (group+1)*(rows//2)-1
        yc = (ys[a]+ys[b]+ph)/2
        fig.text(2.1/width, 1-yc/height, label, ha="center", va="center",
                 rotation=90, fontsize=7.5 if compact else 8.8, weight="bold", color=INK)
    sep_y = (ys[rows//2-1]+ph+ys[rows//2])/2
    fig.add_artist(Line2D([left/width, (width-right)/width], [1-sep_y/height]*2,
                         transform=fig.transFigure, lw=.35, color="#bac2ca"))
    shared_legend(fig, 6.6 if compact else 6.4)
    validate(fig, all_axes)
    stem = "incidence_eight_compact" if compact else "incidence_eight_full"
    fig.savefig(output_dir/f"{stem}.png", dpi=png_dpi, facecolor="white")
    fig.savefig(output_dir/f"{stem}_preview.png", dpi=190, facecolor="white")
    # Embed the native raster assets without reducing them to the PDF DPI;
    # arrows, text, circles and rectangles remain vector objects.
    for axes in all_axes:
        for ax in axes:
            for image in ax.images:
                image.set_interpolation("none")
    fig.savefig(pdf_dir/f"{stem}.pdf", dpi=300, facecolor="white",
                metadata={"Title": "Qualitative incidence-field comparison", "Creator": "Matplotlib"})
    if not compact:
        fig.savefig(output_dir/f"{stem}.svg", dpi=300, facecolor="white")
    plt.close(fig)
    return dict(stem=stem, width_mm=width, height_mm=height, rows=rows,
                panels=32, shared_legends=1, png_dpi=png_dpi,
                image_interpolation="lanczos" if "rgb_full" in data[0] else "none",
                pdf_image_interpolation="none", vector_pdf_arrows=True)


def individual(meta, data, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else OUT
    width, left, gap, pw, top = 168., 1., 1.3, 40.525, 6.
    ph = pw*.75
    height = top+ph+10.
    fig = plt.figure(figsize=(width*MM, height*MM), dpi=120, facecolor="white")
    axes = [add_axes(fig, left+c*(pw+gap), top, pw, ph, width, height) for c in range(4)]
    draw_sample(axes, meta, data)
    for c, title in enumerate(("Full view", "Baseline + GT", "Ours + GT", "GT")):
        fig.text((left+c*(pw+gap)+pw/2)/width, 1-2.4/height, title,
                 ha="center", va="center", fontsize=8,
                 color=GREEN if c == 2 else INK, weight="bold" if c == 2 else "normal")
    shared_legend(fig)
    validate(fig, [axes])
    fig.savefig(output_dir/"individual"/meta["filename"], dpi=600, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"individual").mkdir(exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    available = {m["filename"]: m for m in json.loads((CACHE_ROOT/"selection.json").read_text())}
    assert set(ORDER) == {p.name for p in SOURCE.glob("*.png")}
    metas = [available[name] for name in ORDER]
    data, manifest = [], []
    for i, meta in enumerate(metas):
        desktop, canonical = SOURCE/meta["filename"], CACHE_ROOT/"individual"/meta["filename"]
        assert sha(desktop) == sha(canonical), f"Source differs: {desktop}"
        cache = CACHE_ROOT/"cache"/meta["cache"]
        with np.load(cache) as archive:
            data.append({k: archive[k] for k in ("rgb", "baseline", "ours", "gt")})
        manifest.append(dict(label=chr(97+i), group="Indoor" if i < 4 else "Outdoor",
                             source=str(desktop), source_sha256=sha(desktop), cache=str(cache),
                             cache_sha256=sha(cache), crop=meta["crop"], markers=meta["markers"]))
    layouts = [gallery(metas, data), gallery(metas, data, True)]
    for meta, sample in zip(metas, data):
        individual(meta, sample)
    # Verify originals/caches were not modified while rendering.
    for item in manifest:
        assert sha(Path(item["source"])) == item["source_sha256"]
        assert sha(Path(item["cache"])) == item["cache_sha256"]
    (OUT/"manifest.json").write_text(json.dumps(dict(layouts=layouts, samples=manifest),
                                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(layouts, indent=2), flush=True)


if __name__ == "__main__":
    main()
