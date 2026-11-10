"""Select real model predictions and render matched incidence/GT comparisons."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch

from WildCamera.benchmark.visualize_confidence_incidence import (
    DATASETS, angular_error, denormalize, load_model, make_dataset,
)
from tools.tools import intrinsic2incidence

ORANGE = "#ed9a29"
GREEN = "#07825d"
RED = "#e73435"
PINK = "#d92882"
DEFAULT_OUT = ROOT / "output/incidence_gt_100_2026-11-08"
ARROW_LENGTH = 25.0
ARROW_WIDTH = 0.0055


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--scan-min", type=int, default=100)
    p.add_argument("--scan-max", type=int, default=500)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()
    args.data_path = ROOT / "WildCamera/data/MonoCalib"
    args.baseline_ckpt = ROOT / "model_zoo/multi-modality newcrfs(8largecos)/step_49999.ckpt"
    args.confidence_ckpt = ROOT / "model_zoo/multi-modality newcrfs(train置信度)/step_29999.ckpt"
    args.output_dir = args.output_dir.resolve()
    return args


def window_mean(a, h=150, w=200):
    integral = cv2.integral(a.astype(np.float32), sdepth=cv2.CV_64F)
    return (integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]) / (h*w)


def projection(field):
    # Preserve projected direction AND magnitude. No per-method normalization.
    return field[:2] / np.maximum(field[2:3], 1e-5)


def direction_vectors(field, length=ARROW_LENGTH):
    """Display projected azimuth only; use identical glyph length for every ray."""
    xy = field[:2].astype(np.float64)
    norm = np.linalg.norm(xy, axis=0, keepdims=True)
    return np.divide(xy*length, norm, out=np.zeros_like(xy), where=norm > 1e-8)


def sample_grid(crop):
    x, y, w, h = crop
    xx, yy = np.meshgrid(np.arange(30, w-29, 35), np.arange(30, h-29, 30))
    return xx + x, yy + y


def choose_markers(fields, crop):
    xx, yy = sample_grid(crop)
    b, o, g = [f[:, yy, xx] for f in fields]
    angle_b = np.degrees(np.arccos(np.clip((b*g).sum(0), -1, 1)))
    angle_o = np.degrees(np.arccos(np.clip((o*g).sum(0), -1, 1)))
    pb, po, pg = [direction_vectors(f[:, yy, xx], 1.0) for f in fields]
    gap_b = np.linalg.norm(pb-pg, axis=0)
    gap_o = np.linalg.norm(po-pg, axis=0)
    score = np.where((angle_b > angle_o + 0.05) & (gap_b > gap_o + 0.001),
                     (angle_b-angle_o)*(gap_b-gap_o), -np.inf)
    markers = []
    for flat in np.argsort(score.ravel())[::-1]:
        row, col = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[row, col]):
            continue
        x, y = int(xx[row, col]), int(yy[row, col])
        if any(np.hypot(x-m["x"], y-m["y"]) < 65 for m in markers):
            continue
        markers.append(dict(label=chr(65+len(markers)), x=x, y=y,
                            baseline_deg=float(angle_b[row, col]), ours_deg=float(angle_o[row, col]),
                            baseline_unit_direction_gap=float(gap_b[row, col]),
                            ours_unit_direction_gap=float(gap_o[row, col])))
        if len(markers) == 2:
            return markers
    return []


def analyze(rgb, fields, eb, eo):
    gain = eb-eo
    global_gain = float(gain.mean())
    if global_gain <= 0.03:
        return None
    local_gain = window_mean(gain)[::10, ::10]
    win = window_mean((gain > 0).astype(np.float32))[::10, ::10]
    brightness = window_mean(rgb.mean(2))[::10, ::10]
    score = local_gain * (0.6 + 0.4*win) + 0.08*brightness
    score[(local_gain < 0.1) | (win < 0.65) | (brightness < 0.12)] = -np.inf
    # Prefer a visible region with two actual matching-ray improvements.
    for flat in np.argsort(score.ravel())[::-1]:
        y, x = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[y, x]):
            break
        crop = (int(x*10), int(y*10), 200, 150)
        markers = choose_markers(fields, crop)
        if not markers:
            continue
        x0, y0, w, h = crop
        region = np.s_[y0:y0+h, x0:x0+w]
        return dict(crop=list(crop), markers=markers,
                    global_baseline_deg=float(eb.mean()), global_ours_deg=float(eo.mean()),
                    global_gain_deg=global_gain, global_ours_closer_fraction=float((gain > 0).mean()),
                    crop_baseline_deg=float(eb[region].mean()), crop_ours_deg=float(eo[region].mean()),
                    crop_gain_deg=float(gain[region].mean()), crop_ours_closer_fraction=float(win[y, x]),
                    selection_score=float(score[y, x] + 0.15*global_gain))
    return None


def infer(sample, baseline, ours, device):
    with torch.inference_mode():
        image = sample["rgb"].unsqueeze(0).to(device)
        b = baseline(image)["incidence"]
        o = ours(image)["incidence"]
        gt = intrinsic2incidence(sample["K"].unsqueeze(0).to(device), 1, 480, 640, device)
        gt = gt.squeeze(-1).permute(0, 3, 1, 2)
        eb, eo = angular_error(b, gt)[0].cpu().numpy(), angular_error(o, gt)[0].cpu().numpy()
        fields = [f[0].cpu().numpy() for f in (b, o, gt)]
    return denormalize(sample["rgb"]), fields, eb, eo


def acquire(args):
    device = torch.device("cuda:0")
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)
    baseline = load_model(args.baseline_ckpt, "large07", device, baseline=True)
    ours = load_model(args.confidence_ckpt, "large07", device, baseline=False)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    selected = []
    audit = []
    start = time.monotonic()
    for di, name in enumerate(DATASETS):
        quota = args.count // len(DATASETS) + int(di < args.count % len(DATASETS))
        dataset = make_dataset(name, args)
        candidates = []
        seen_images = set()
        limit = min(args.scan_max, len(dataset))
        print(f"SCAN {name}: target={quota}, initial={args.scan_min}, maximum={limit}", flush=True)
        for index in range(limit):
            sample = dataset[index]
            rgb, fields, eb, eo = infer(sample, baseline, ours, device)
            digest = hashlib.sha256(np.round(rgb*255).astype(np.uint8).tobytes()).hexdigest()
            meta = analyze(rgb, fields, eb, eo)
            audit.append(dict(dataset=name, index=index, global_baseline_deg=float(eb.mean()),
                              global_ours_deg=float(eo.mean()), eligible=bool(meta), image_sha256=digest))
            if meta and digest not in seen_images:
                seen_images.add(digest)
                meta.update(dataset=name, index=index, image_sha256=digest,
                            source=dataset.datasets[0].data_names[index],
                            intrinsic=sample["K"].tolist())
                candidates.append(meta)
            if (index+1) % 25 == 0:
                print(f"  {name}: scanned={index+1}, eligible={len(candidates)}, elapsed={time.monotonic()-start:.0f}s", flush=True)
            if index+1 >= args.scan_min and len(candidates) >= quota:
                break
        if len(candidates) < quota:
            raise RuntimeError(f"{name}: only {len(candidates)} qualifying samples for quota {quota}; increase --scan-max")
        candidates.sort(key=lambda m: m["selection_score"], reverse=True)
        for meta in candidates[:quota]:
            sample = dataset[meta["index"]]
            rgb, fields, _, _ = infer(sample, baseline, ours, device)
            cache_name = f"{name}_{meta['index']:04d}.npz"
            np.savez_compressed(cache_dir / cache_name, rgb=rgb, baseline=fields[0], ours=fields[1], gt=fields[2], K=sample["K"].numpy())
            meta["cache"] = cache_name
            selected.append(meta)
        print(f"SELECTED {name}: {quota} samples", flush=True)
        (args.output_dir / "selection_in_progress.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    # Interleave datasets: each gallery page covers a range of scenes.
    groups = {name: [m for m in selected if m["dataset"] == name] for name in DATASETS}
    ordered = []
    for rank in range(max(map(len, groups.values()))):
        for name in DATASETS:
            if rank < len(groups[name]):
                ordered.append(groups[name][rank])
    for rank, meta in enumerate(ordered, 1):
        meta["rank"] = rank
        meta["filename"] = f"{rank:03d}_{meta['dataset']}_{meta['index']:04d}.png"
    (args.output_dir / "selection.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    with (args.output_dir / "candidate_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    return ordered


def quiver(ax, xy, uv, color, width, zorder):
    return ax.quiver(*xy, *uv, color=color, angles="xy", scale_units="xy", scale=1,
              width=width, headwidth=3.8, headlength=4.7, headaxislength=4.2,
              edgecolor="#202833", linewidth=0.25, minlength=0, zorder=zorder)


def border(ax):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                              edgecolor=RED, linewidth=2.2, clip_on=False, zorder=10))


def draw_row(axes, meta, data):
    rgb = data["rgb"]
    fields = [data[k] for k in ("baseline", "ours", "gt")]
    x, y, w, h = meta["crop"]
    xs, ys = sample_grid(meta["crop"])
    vectors = [direction_vectors(f[:, ys, xs]) for f in fields]
    ox, oy = xs-x, ys-y
    ax = axes[0]
    ax.imshow(rgb)
    xx, yy = np.meshgrid(np.arange(24, 640-23, 48), np.arange(24, 480-23, 48))
    uv = direction_vectors(fields[1][:, yy, xx], 16.0)
    quiver(ax, (xx, yy), uv, GREEN, 0.003, 3)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=RED, linewidth=2.4))
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.axis("off")
    for i, (title, color) in enumerate((("Baseline + GT", ORANGE), ("Ours + GT", GREEN), ("GT", "white"))):
        ax = axes[i+1]
        ax.imshow(rgb[y:y+h, x:x+w])
        quiver(ax, (ox, oy), vectors[2], "white", ARROW_WIDTH, 3)
        if i < 2:
            pred_tips = np.column_stack(((ox+vectors[i][0]).ravel(), (oy+vectors[i][1]).ravel()))
            gt_tips = np.column_stack(((ox+vectors[2][0]).ravel(), (oy+vectors[2][1]).ravel()))
            ax.add_collection(LineCollection(np.stack((gt_tips, pred_tips), 1), colors=PINK,
                                            linewidths=1.2, zorder=4))
            quiver(ax, (ox, oy), vectors[i], color, ARROW_WIDTH, 5)
        for marker in meta["markers"]:
            gx, gy = marker["x"], marker["y"]
            guv = direction_vectors(fields[2][:, gy, gx])
            cx, cy = gx-x+guv[0]*0.5, gy-y+guv[1]*0.5
            radius = float(np.linalg.norm(guv)*0.5+4)
            for edge, lw in (("#222b37", 2.5), ("white", 1.6)):
                ax.add_patch(plt.Circle((cx, cy), radius, fill=False, edgecolor=edge,
                                       linewidth=lw, linestyle=(0, (3, 2)), zorder=7))
            label_x = float(np.clip(cx-radius+1, 8, w-8))
            label_y = float(np.clip(cy-radius-3, 8, h-8))
            ax.text(label_x, label_y, marker["label"], fontsize=10, weight="bold",
                    color="white", ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.16", fc="#253244", ec="white", lw=0.7))
        border(ax)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title(title, fontsize=13, color=GREEN if i == 1 else "#172437",
                     weight="bold" if i == 1 else "normal", pad=10)


def legend(fig):
    handles = [Line2D([0], [0], color=ORANGE, lw=4, label="Baseline"),
               Line2D([0], [0], color=GREEN, lw=4, label="Ours"),
               Line2D([0], [0], color="white", lw=4, label="GT",
                      path_effects=[pe.Stroke(linewidth=6, foreground="#526075"), pe.Normal()]),
               Line2D([0], [0], color=PINK, lw=1.6, label="Projected direction discrepancy"),
               Line2D([0], [0], color="#546174", lw=1.3, ls="--", label="A / B: matched rays")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.012), columnspacing=2.0,
               handlelength=2.3, handletextpad=0.65)


def render_figure(metas, args, path, dpi):
    fig, axes = plt.subplots(len(metas), 4, figsize=(17.4, 3.75*len(metas)+0.45), squeeze=False)
    for row, meta in enumerate(metas):
        with np.load(args.output_dir / "cache" / meta["cache"]) as data:
            draw_row(axes[row], meta, data)
    fig.subplots_adjust(left=0.014, right=0.986, bottom=0.10 if len(metas) == 1 else 0.06,
                        top=0.90 if len(metas) == 1 else 0.94, wspace=0.07, hspace=0.20)
    legend(fig)
    # Inspect the actual plotted objects: all five crop fields use the same
    # length, shaft width, head dimensions, and axis display scale.
    for row in axes:
        glyphs = [q for ax in row[1:] for q in ax.collections if isinstance(q, matplotlib.quiver.Quiver)]
        assert len(glyphs) == 5
        for q in glyphs:
            lengths = np.hypot(q.U, q.V)
            assert np.allclose(lengths[lengths > 0], ARROW_LENGTH, atol=1e-9)
            assert (q.width, q.headwidth, q.headlength, q.headaxislength) == (ARROW_WIDTH, 3.8, 4.7, 4.2)
    # This assertion covers every rendered text label, including the legend.
    assert not any("error" in text.get_text().lower() for text in fig.findobj(matplotlib.text.Text))
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def make_browse_files(metas, args):
    thumbs = args.output_dir / "thumbnails"
    thumbs.mkdir(exist_ok=True)
    cards = []
    for meta in metas:
        with Image.open(args.output_dir / "individual" / meta["filename"]) as im:
            im.thumbnail((1305, 330))
            im.convert("RGB").save(thumbs / meta["filename"].replace(".png", ".jpg"), quality=88)
        thumb = "thumbnails/" + meta["filename"].replace(".png", ".jpg")
        label = f"{meta['rank']:03d} · {meta['dataset']} · #{meta['index']:04d}"
        cards.append(f'<article data-dataset="{meta["dataset"]}"><h2>{html.escape(label)}</h2>'
                     f'<a href="individual/{meta["filename"]}" target="_blank"><img loading="lazy" src="{thumb}" alt="{label}"></a></article>')
    buttons = '<button class="active" data-filter="all">全部 100 张</button>'
    for name, count in Counter(m["dataset"] for m in metas).items():
        buttons += f'<button data-filter="{name}">{name} · {count}</button>'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>入射场 · GT 对照图库</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#edf1f5;color:#1b2b40}header{padding:30px 4vw 20px;background:white;position:sticky;top:0;z-index:2;box-shadow:0 1px 9px #0001}
h1{font-size:27px;margin:0 0 12px}p{line-height:1.6;margin:8px 0;color:#526075}nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}button{border:1px solid #cbd3df;border-radius:20px;padding:8px 14px;background:white;cursor:pointer}button.active{background:#087f5b;color:white;border-color:#087f5b}main{max-width:1600px;margin:24px auto;padding:0 22px}article{background:white;border-radius:12px;padding:14px;margin:0 0 22px;box-shadow:0 2px 10px #15243a08}h2{font-size:15px;margin:0 0 10px;color:#526075}img{width:100%;height:auto;display:block}a{color:#087f5b}[hidden]{display:none}</style>
<header><h1>入射场对比 · 100 张 · 精简图例版</h1><p>从左到右：全图 / Baseline + GT / Ours + GT / GT。白色箭头为 GT，橙色为 Baseline，绿色为 Ours。<br>三种局部箭头长度、粗细、箭头头部尺寸完全一致，只比较二维投影方向。A、B 标记相同射线；紫色线段表示投影方向差异，越短越接近 GT。点击图片查看原尺寸。</p><nav>BUTTONS</nav></header><main>CARDS</main>
<script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>{document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('article').forEach(x=>x.hidden=b.dataset.filter!=='all'&&x.dataset.dataset!==b.dataset.filter);});</script></html>'''
    (args.output_dir / "index.html").write_text(page.replace("BUTTONS", buttons).replace("CARDS", "\n".join(cards)), encoding="utf-8")
    for page_index, start in enumerate(range(0, len(metas), 10), 1):
        subset = metas[start:start+10]
        sheet = Image.new("RGB", (1740, 420*len(subset)), "white")
        for i, meta in enumerate(subset):
            with Image.open(args.output_dir / "individual" / meta["filename"]) as im:
                panel = ImageOps.contain(im.convert("RGB"), (1740, 420))
                sheet.paste(panel, ((1740-panel.width)//2, i*420))
        sheet.save(args.output_dir / "galleries" / f"gallery_{page_index:02d}.jpg", quality=93)


def document(metas, args):
    counts = dict(Counter(m["dataset"] for m in metas))
    result = dict(count=len(metas), datasets=counts,
                  min_global_gain_deg=min(m["global_gain_deg"] for m in metas),
                  min_crop_gain_deg=min(m["crop_gain_deg"] for m in metas),
                  min_crop_ours_closer_fraction=min(m["crop_ours_closer_fraction"] for m in metas),
                  unique_source_images=len({m["image_sha256"] for m in metas}),
                  baseline_checkpoint=str(args.baseline_ckpt), ours_checkpoint=str(args.confidence_ckpt),
                  rendering="Direction-only: all crop arrows have length 25, width 0.0055, headwidth 3.8, headlength 4.7; real 2D azimuth preserved",
                  selection="Favorable qualitative examples; full-image and ROI mean angular distances both smaller for Ours")
    (args.output_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
             for k, v in m.items()} for m in metas]
    with (args.output_dir / "selection_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    readme = f'''# 入射场与 GT 对比图库

共 {len(metas)} 张独立样本：{json.dumps(counts, ensure_ascii=False)}。

- `index.html`：可按数据集筛选，点击查看高清原图。
- `individual/`：100 张四列高清对比图。
- `galleries/`：每页 10 张，共 10 页。
- `preview.png`：两张示例拼图。
- `selection_manifest.csv` / `selection.json`：真实指标、原始样本来源、相机内参、ROI 和 A/B 坐标。
- `candidate_audit.csv`：所有扫描样本的选取记录。
- `cache/`：未经修改的 float32 模型输出、GT、RGB 和内参，可直接重绘。

## 图的读法

四列依次为全图、Baseline + GT、Ours + GT、独立 GT。橙色为 Baseline，绿色为 Ours，带深色轮廓的白色箭头为 GT；全图上的绿色箭头表示 Ours 的上下文入射场。
图片顶部不显示数据集名称或样本编号，来源信息仍可在文件名、图库筛选及清单中查询。图例将真实场简写为 GT；紫色线段标为 Projected direction discrepancy，即等长投影方向向量端点间的欧氏距离，与二维方向夹角单调对应，不表示三维角误差。
A/B 虚线圆圈在三张局部图中对应完全相同的像素。三种局部箭头统一为 25 像素长度和完全相同的粗细、头部尺寸；粉色短线连接等长箭头端点，越短表示二维投影方向越接近。预测与 GT 方向重合时，白色 GT 可能被彩色箭头覆盖，可对照独立 GT 列。
图片中无 error 字样，也无角度、百分比或其他性能数字。

## 几何与选取

GT 由各样本经过数据集既定 resize/crop 变换后的真实内参 K，通过 normalize(K^-1 [u,v,1]) 得到。三列使用同一个 200×150 裁剪框、同一 5×4 采样网格。
箭头为二维投影方向的等长可视化：25 * (ray_x, ray_y) / sqrt(ray_x^2 + ray_y^2)。所有方法采用相同归一化，不旋转或夸大方向差异。它仅编码二维方位方向，不编码原始投影长度，也不能完整表达三维射线倾角。全图箭头同样等长，长度为 16 像素；零长度二维方向不绘制。
三维射线的角距离与二维方向差并不等价，A/B 处同时核对这两项均更接近 GT。改为等长版本后重新选择 A/B；必要时从同一张图重新挑选满足原筛选条件的 ROI。

这些是用于论文展示的优势样本，不代表随机抽样或总体评测。每个数据集至少扫描前 {args.scan_min} 个测试样本；不足时继续扫描，最多 {args.scan_max} 个。要求全图平均角距离改善 >0.03°，ROI 改善 >0.1°，ROI 至少 65% 像素更接近 GT；A/B 两点也分别通过三维和二维检查，再按局部改善排名选取。

Baseline：{args.baseline_ckpt}

Ours：{args.confidence_ckpt}

## 建议图注

入射场方向的定性对比。红框标出统一放大区域；白色、橙色和绿色箭头分别表示真实入射场、Baseline 和本文方法。局部箭头采用相同长度、粗细和头部尺寸，展示二维投影方向。A/B 对应相同像素处的射线，粉色短线表示等长箭头端点之间的间隔。在这些选取的样例中，本文方法的局部方向与 GT 更一致。
'''
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("individual", "galleries"):
        (args.output_dir / folder).mkdir(exist_ok=True)
    if args.render_only:
        metas = json.loads((args.output_dir / "selection.json").read_text(encoding="utf-8"))
        recropped = 0
        for meta in metas:
            with np.load(args.output_dir / "cache" / meta["cache"]) as data:
                fields = [data[k] for k in ("baseline", "ours", "gt")]
                markers = choose_markers(fields, meta["crop"])
                if markers:
                    meta["markers"] = markers
                else:
                    def angle(a, b):
                        return np.degrees(np.arccos(np.clip((a*b).sum(0), -1+1e-7, 1-1e-7)))
                    replacement = analyze(data["rgb"], fields, angle(fields[0], fields[2]), angle(fields[1], fields[2]))
                    if replacement is None:
                        raise RuntimeError(f"No direction-only ROI for {meta['dataset']} #{meta['index']}")
                    meta.update(replacement)
                    recropped += 1
        (args.output_dir / "selection.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
        print(f"Direction-only markers checked; {recropped} crops reselected", flush=True)
    else:
        metas = acquire(args)
    hero = []
    for name in ("MVImgNet", "Objectron"):
        matches = [m for m in metas if m["dataset"] == name]
        if matches:
            hero.append(matches[0])
    render_figure(hero or metas[:2], args, args.output_dir / "preview.png", args.dpi)
    if args.preview_only:
        print("PREVIEW READY", flush=True)
        return
    for i, meta in enumerate(metas, 1):
        render_figure([meta], args, args.output_dir / "individual" / meta["filename"], args.dpi)
        if i % 10 == 0:
            print(f"RENDERED {i}/{len(metas)}", flush=True)
    make_browse_files(metas, args)
    document(metas, args)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
