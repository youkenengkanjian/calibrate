"""Reproducible paper artwork; consumes frozen predictions, never trains models.

Run with D:/software/anaconda/python.exe -X utf8 tools/draw_paper_figures_20261110.py
All layout coordinates are millimetres at the intended 180 mm paper width.
FPN follows the author's supplied diagram/manuscript, as explicitly requested;
the surviving checkpoint implementation does not enable that neck. See README.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'WildCamera/newcrfs/newcrf_incidencefield.py').is_file())
OUT = ROOT / 'output/pdf/paper_figures_redrawn_2026-11-10'
RAW = ROOT / 'tmp/paper_figure_design_20261109/raw_confidence'
FROZEN = ROOT / 'output/incidence_paper_eight_2026-11-08/manifest.json'
NATIVE = ROOT / 'output/incidence_native_2026-11-09/assets'
BLUE, GREEN, ORANGE, RED = '#356DAD', '#087F73', '#D58C35', '#BE4B58'
INK, GRAY, LINE = '#233247', '#667383', '#CED6DF'
BP, GP, RP = '#EEF4FA', '#EDF7F4', '#FCF1F3'
ERROR_CMAP = colors.LinearSegmentedColormap.from_list(
    'ray_error', ['#FCFBF7', '#F8DCA1', '#ED9D62', '#CA5D59', '#813D63', '#392447'])
INCIDENCE = ['SUN3D_0081', 'NYUv2_0032', 'Cityscapes_0208', 'KITTI_0014']
CONFIDENCE = ['SUN3D_0081', 'NYUv2_0032', 'KITTI_0014', 'Objectron_0015']
OVERLAYS = {
    'SUN3D_0081': ROOT / 'output/paper_visualizations_confidence_100_final_v2/individual/41_SUN3D_0081_confidence_overlay.png',
    'NYUv2_0032': ROOT / 'output/paper_visualizations_confidence_100_final_v2/individual/28_NYUv2_0032_confidence_overlay.png',
    'KITTI_0014': ROOT / 'output/paper_visualizations_confidence_100_final_v2/individual/81_KITTI_0014_confidence_overlay.png',
    'Objectron_0015': ROOT / 'output/confidence_native_2026-11-09/overlays/06_Objectron_0015_overlay.png',
}
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7.2, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'mathtext.sf': 'Arial', 'mathtext.cal': 'Arial:italic',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angular(ray, gt):
    a, b = ray.astype(np.float64), gt.astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=0), -1, 1)))


def as_rgb(a):
    if isinstance(a, (str, Path)):
        return np.asarray(Image.open(a).convert('RGB'))
    a = np.asarray(a)
    return np.round(np.clip(a, 0, 1)*255).astype(np.uint8) if a.dtype != np.uint8 else a


def label(sample):
    name, index = sample.rsplit('_', 1)
    return f'{name} / {int(index):04d}'


class Artwork:
    def __init__(self, stem, height):
        self.stem, self.w, self.h = stem, 180., float(height)
        self.fig = plt.figure(figsize=(self.w/25.4, self.h/25.4), dpi=150)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, self.w), ylim=(self.h, 0), aspect='equal')
        self.ax.axis('off')
        self.texts, self.edges = [], []

    def text(self, x, y, text, size=7.2, color=INK, bold=False,
             ha='left', va='center', rotation=0, tag=None, **kw):
        artist = self.ax.text(x, y, text, fontsize=size, color=color,
                              fontweight='bold' if bold else 'normal',
                              ha=ha, va=va, rotation=rotation, linespacing=1.22, **kw)
        self.texts.append((tag or text, artist))
        return artist

    def rect(self, x, y, w, h, fill='white', edge=LINE, lw=.65, radius=1.0, dash=False):
        if radius:
            p = FancyBboxPatch((x, y), w, h,
                              boxstyle=f'round,pad=0,rounding_size={radius}',
                              facecolor=fill, edgecolor=edge, linewidth=lw,
                              linestyle=(0, (3, 2)) if dash else 'solid')
        else:
            p = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw,
                          linestyle=(0, (3, 2)) if dash else 'solid')
        self.ax.add_patch(p)
        return p

    def line(self, pts, color=GRAY, lw=.85, dash=False, arrow=False, tag=''):
        if len(pts) > 2:
            xs, ys = zip(*pts[:-1] if arrow else pts)
            self.ax.plot(xs, ys, color=color, lw=lw,
                         linestyle=(0, (3, 2)) if dash else 'solid', solid_capstyle='round')
        if arrow:
            p = FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>',
                                mutation_scale=6.5, color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0,
                                linestyle=(0, (3, 2)) if dash else 'solid')
            self.ax.add_patch(p)
        elif len(pts) == 2:
            self.ax.plot(*zip(*pts), color=color, lw=lw,
                         linestyle=(0, (3, 2)) if dash else 'solid')
        if tag:
            self.edges.append({'role': tag, 'points_mm': pts, 'dashed': dash, 'color': color})

    def box(self, x, y, w, h, title, subtitle=None, color=BLUE, fill=BP, size=7.2):
        self.rect(x, y, w, h, fill, color, .75)
        self.text(x+w/2, y+h/2-(1.7 if subtitle else 0), title, size, color,
                  True, ha='center')
        if subtitle:
            self.text(x+w/2, y+h/2+3.0, subtitle, 6.5, GRAY, ha='center')

    def image(self, a, x, y, w, h, border=True, interpolation='none'):
        self.ax.imshow(as_rgb(a), extent=(x, x+w, y+h, y), interpolation=interpolation, zorder=0)
        if border:
            self.rect(x, y, w, h, fill='none', lw=.5, radius=0)

    def header(self, x, y, letter, title, color=INK):
        self.text(x, y, f'({letter})', 8.6, color, True)
        self.text(x+7, y, title, 8.6, color, True)

    def colorbar(self, x, y, w, caption=True):
        strip = ERROR_CMAP(np.linspace(0, 1, 256))[None, :, :3]
        self.image(np.repeat(strip, 6, axis=0), x, y, w, 2.4)
        for v in (0, 1, 2, 3, 4, 5):
            self.text(x+w*v/5, y+4.6, str(v), 6.5, GRAY, ha='center')
        if caption:
            self.text(x+w/2, y-2.5, '3D angular error (deg)', 7, GRAY, ha='center')

    def export(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        boxes, out_of_bounds, overlaps = [], [], []
        for tag, artist in self.texts:
            b = artist.get_window_extent(renderer)
            coords = inverse.transform([[b.x0, b.y0], [b.x1, b.y1]])
            x0, x1 = sorted(coords[:, 0]); y0, y1 = sorted(coords[:, 1])
            record = dict(text=tag, bounds_mm=[float(x0), float(y0), float(x1), float(y1)],
                          point_size=artist.get_fontsize())
            boxes.append(record)
            if x0 < -.03 or y0 < -.03 or x1 > self.w+.03 or y1 > self.h+.03:
                out_of_bounds.append(record)
        for i, a in enumerate(boxes):
            for b in boxes[i+1:]:
                x0,y0,x1,y1 = a['bounds_mm']; u0,v0,u1,v1 = b['bounds_mm']
                ix, iy = min(x1,u1)-max(x0,u0), min(y1,v1)-max(y0,v0)
                if ix > .12 and iy > .12:
                    overlaps.append([a['text'], b['text'], round(ix*iy, 3)])
        # Fixed page size, with no tight crop: every export has the same mm geometry.
        for fmt in ('svg', 'pdf', 'png'):
            self.fig.savefig(OUT / f'{self.stem}.{fmt}', dpi=600,
                             metadata={'Creator': 'WildCamera reproducible paper artwork'}
                             if fmt == 'pdf' else None)
        self.fig.savefig(OUT / 'previews' / f'{self.stem}.png', dpi=180)
        result = dict(stem=self.stem, width_mm=self.w, height_mm=self.h,
                      png_dpi=600, min_font_pt=min(t['point_size'] for t in boxes),
                      text_out_of_bounds=out_of_bounds, text_overlaps=overlaps,
                      text_boxes=boxes, edges=self.edges)
        plt.close(self.fig)
        return result


def overall(data, meta):
    a = Artwork('01_overall_framework', 107)
    a.header(3, 5, 'a', 'Joint incidence and confidence prediction')
    # Network modules are schematic; the two output thumbnails are data-derived.
    k = 'KITTI_0014'
    a.rect(154.5, 5.2, 24.7, 48.5, 'white', LINE, .6)
    a.image(NATIVE / f"{meta[k]['label']}_{k}_full.png", 3, 22, 23, 17.25)
    a.text(14.5, 43, 'RGB image', 7.2, ha='center')
    a.line([(27, 30.6), (32, 30.6)], BLUE, arrow=True, tag='RGB to encoder')
    a.rect(33, 19, 26, 23, BP, BLUE, .8)
    a.text(46, 24, 'Swin', 8, BLUE, True, ha='center')
    a.text(46, 28, 'Transformer', 7.5, BLUE, True, ha='center')
    for j in range(4):
        a.rect(36+j*5.1, 33+j*.65, 4.1, 4.6-j*.4, 'white', BLUE, .55, .3)
        a.text(38.05+j*5.1, 39, f'$F_{j+1}$', 6.5, BLUE, ha='center')
    a.line([(59, 30.6), (65, 30.6)], BLUE, arrow=True, tag='multi-scale features to FPN')
    a.box(66, 19, 19, 23, 'FPN', 'Feature fusion', size=8)
    a.text(75.5, 15.1, 'Multi-scale', 6.5, GRAY, ha='center')
    # Deepest FPN level supplies global context; all scales guide CRFs.
    a.line([(85, 25), (90, 25), (90, 18), (96, 18)], BLUE, arrow=True, tag='FPN deepest scale to PSP')
    a.box(96, 12, 26, 12, 'PSP', 'Global context', size=7.8)
    a.line([(109, 24), (109, 32)], BLUE, arrow=True, tag='PSP context to coarse CRF')
    a.line([(85, 35), (94, 35)], BLUE, arrow=True, tag='FPN scales to CRFs')
    a.rect(95, 32, 30, 14, BP, BLUE, .8)
    a.text(110, 36, 'Cascaded CRFs', 7.2, BLUE, True, ha='center')
    for j, num in enumerate([3, 2, 1, 0]):
        x = 97+j*6.7
        a.rect(x, 39, 5.4, 4.7, 'white', BLUE, .5, .3)
        a.text(x+2.7, 41.3, f'$E_{num}$', 6.5, BLUE, ha='center')
        if j < 3:
            a.line([(x+5.5, 41.3), (x+6.55, 41.3)], BLUE, .6, arrow=True)
    a.line([(125, 39), (130, 39), (130, 30.6), (133, 30.6)], BLUE, arrow=True, tag='refined features to joint head')
    a.box(133, 22, 17, 17, 'Joint\nhead', 'Conv + up', size=7.3)
    a.text(141.5, 43, '3 + 1 channels', 6.5, GRAY, ha='center')
    a.line([(150, 30.5), (153, 30.5), (153, 17.5), (156, 17.5)], BLUE, arrow=True, tag='unit-normalized ray output')
    a.line([(150, 30.5), (153, 30.5), (153, 40), (156, 40)], GREEN, arrow=True, tag='bounded sigmoid confidence output')
    a.text(167, 8.5, r'Unit rays $\widehat{\mathbf{i}}$', 7.2, BLUE, True, ha='center')
    rays = (data[k]['ours'].transpose(1, 2, 0)+1)/2
    a.image(rays, 156, 11, 22, 16.5)
    a.text(167, 31.8, r'Confidence $c$', 7.1, GREEN, True, ha='center')
    a.image(OVERLAYS[k], 156, 34.5, 22, 16.5)
    # Shared predicted pair feeds supervision and a separate inference solver.
    a.line([(167, 53.8), (167, 56), (62, 56), (62, 74.5)], GRAY, .85, dash=True,
           arrow=True, tag='predictions to training loss')
    a.text(91, 53.5, r'Shared predictions: $\widehat{\mathbf{i}},\ c$', 6.8, GRAY, ha='center')
    a.line([(167, 56), (167, 59), (102, 59), (102, 74.5)], GREEN, .85,
           arrow=True, tag='confidence to inference weights')
    a.line([(144.5, 59), (144.5, 74.5)], BLUE, .85, arrow=True, tag='predicted rays to solver')
    a.rect(3, 63, 80, 36, 'white', RED, .7, 1.3, dash=True)
    a.rect(89, 63, 89, 36, 'white', GREEN, .7, 1.3)
    a.header(6, 67, 'b', 'Confidence-aware learning', RED)
    a.header(92, 67, 'c', 'Confidence-aware RANSAC', GREEN)
    a.box(7, 75, 16, 10, r'GT $K^\star$', color=BLUE, fill=BP)
    a.line([(23.5, 80), (27, 80)], BLUE, arrow=True)
    a.box(28, 75, 17, 10, r'GT $\mathbf{i}^\star$', color=BLUE, fill=BP)
    a.line([(45.5, 80), (49, 80)], RED, dash=True, arrow=True)
    a.box(50, 75, 29, 10, 'Weighted loss', color=RED, fill=RP, size=7.2)
    a.text(43, 91.5, r'$\mathcal{L}_{\mathrm{conf}}=\mathrm{mean}[c\,\ell_{\mathrm{geo}}-\alpha\log(c+\varepsilon)]$',
           8.2, INK, ha='center')
    a.line([(64.5, 85.5), (64.5, 96), (1.2, 96), (1.2, 48.8), (46, 48.8), (46, 42.5)],
           RED, .65, dash=True, arrow=True, tag='training gradient update to encoder')
    a.text(15, 49.2, 'Gradient update', 6.5, RED,
           bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    specs = [(93, 18, 'Confidence\nselection'), (115, 17, 'Guided\nsampling'),
             (136, 17, 'Two-point\nsolver'), (157, 17, 'Weighted\nscoring')]
    for x, w, title in specs:
        a.box(x, 75, w, 10, title, color=GREEN, fill=GP, size=6.7)
    for x1, x2 in [(111.2,114.2), (132.2,135.2), (153.2,156.2)]:
        a.line([(x1,80),(x2,80)], GREEN, arrow=True)
    a.line([(102,85.5),(102,88.8),(165.5,88.8),(165.5,85.5)], GREEN, .75,
           arrow=True, tag='same confidence weights reused in scoring')
    a.text(130, 88.7, r'Shared weights $w$', 6.5, GREEN, ha='center',
           bbox=dict(facecolor='white', edgecolor='none', pad=1))
    a.line([(165.5,89.5),(165.5,94.5),(156,94.5)], GREEN, arrow=True)
    a.text(127.5,94.6,r'$\widehat{K}\!:\ (f_x,f_y,b_x,b_y)$',8.1,GREEN,ha='center')
    a.text(3,103,'Dashed: training supervision / update',6.5,RED)
    a.text(95,103,'Solid: inference data flow',6.5,GREEN)
    return a.export()


def mechanism(data):
    a = Artwork('02_confidence_mechanism', 100)
    # A compact shared signal at the top makes reuse explicit without repeating a network.
    a.rect(49, 2, 82, 12, GP, GREEN, .8)
    a.text(90, 5.7, 'One learned confidence field', 8.4, GREEN, True, ha='center')
    a.text(90, 10.1, r'$c=c_{\min}+(c_{\max}-c_{\min})\,\sigma(r)$', 8.5, GREEN, ha='center')
    a.line([(80.5,14.4),(80.5,55),(20,55),(20,59.5)], RED,
           dash=True, arrow=True, tag='same c to geometric loss weighting')
    a.line([(80.5,55),(80.5,67.5),(78.3,67.5)], RED, dash=True, arrow=True,
           tag='same c to logarithmic regularizer')
    a.line([(131,8),(135,8),(135,25)], GREEN, arrow=True, tag='same c to inference mapping')
    a.header(3,20,'a','Training: weighted ray learning',RED)
    a.header(88,20,'b','Inference: weighted consensus',GREEN)
    a.box(4,28,28,12,r'GT rays $\mathbf{i}^\star$',r'from $K^\star$',BLUE,BP,7.1)
    a.box(45,28,32,12,r'Predicted $\widehat{\mathbf{i}}$',r'unit normalization',BLUE,BP,7.1)
    a.line([(18,40),(18,44),(39,44)],BLUE,arrow=True)
    a.line([(61,40),(61,44),(41,44)],BLUE,arrow=True)
    a.text(40,49.2,r'$\ell_{\mathrm{geo}}=1-\widehat{\mathbf{i}}^{T}\mathbf{i}^\star$',9.3,ha='center')
    a.line([(40,52),(40,53),(12,53),(12,59.5)],RED,dash=True,arrow=True)
    a.rect(3,60,34,15,RP,RED,.7)
    a.rect(44,60,34,15,RP,RED,.7)
    a.text(20,64,'Ray weighting',7.4,RED,True,ha='center')
    a.text(20,70,r'$c\,\ell_{\mathrm{geo}}$',10,RED,ha='center')
    a.text(61,64,'Log regularizer',7.4,RED,True,ha='center')
    a.text(61,70,r'$-\alpha\log(c+\varepsilon)$',8.4,RED,ha='center')
    a.text(40.5,67.4,'+',10.5,RED,ha='center')
    a.line([(20,75),(20,79),(40,79),(40,82)],RED,dash=True,arrow=True)
    a.line([(61,75),(61,79),(40,79)],RED,dash=True)
    a.box(12,83,56,10,r'$\mathcal{L}_{\mathrm{conf}}=\mathrm{mean}[\,\cdot\,]$',color=RED,fill='white',size=8.8)
    a.text(40,97,'Backpropagation to the joint predictor',6.8,RED,ha='center')
    # Solver detail follows calibrator.py: a small uniform sampling mixture,
    # ray-plane residuals on the sampled pool, and independent x/y winners.
    a.box(108,27,53,12,'Normalize + confidence selection',r'$w=\psi(c)$',GREEN,GP,7.0)
    a.line([(120,39.4),(120,44)],GREEN,arrow=True)
    a.box(89,45,43,16,'Guided sampling',r'$P_p=(1-\eta)\,w_p/\sum_q w_q+\eta/N$',GREEN,GP,7.3)
    a.text(110.5,65,'Small uniform mixture',6.5,GRAY,ha='center')
    a.line([(132.4,53),(139.5,53)],GREEN,arrow=True)
    a.box(141,45,36,16,'Two-point solver',r'$K_j=(f_x,f_y,b_x,b_y)$',BLUE,BP,7.3)
    a.text(159,40.7,r'$\widehat{\mathbf{i}}$, pixel coordinates',6.5,BLUE,ha='center')
    a.line([(159,41.8),(159,44)],BLUE,arrow=True)
    a.line([(159,61.4),(159,69)],BLUE,arrow=True)
    a.line([(161.5,33),(179,33),(179,79),(176.5,79)],GREEN,arrow=True,
           tag='w bypasses sampling and also enters consensus scoring')
    a.rect(89,70,88,16,GP,GREEN,.7)
    a.text(133,74,'Confidence-weighted per-axis scoring',7.5,GREEN,True,ha='center')
    a.text(133,81,r'$S_a(K_j)=\sum_{p\in\Omega_s} w_p\,\mathbf{1}[\delta_{a,p}(K_j)<\tau],\quad a\in\{x,y\}$',
           8.2,GREEN,ha='center')
    a.line([(133,86.4),(133,90)],GREEN,arrow=True)
    a.text(133,92.5,r'$\arg\max S_x\ \rightarrow\ (f_x,b_x),\qquad\arg\max S_y\ \rightarrow\ (f_y,b_y)$',
           8.3,GREEN,ha='center')
    a.text(133,97,'Assemble intrinsics and restore image coordinates',6.8,GREEN,ha='center')
    return a.export()


def confidence_figure(data, raw, meta, metrics):
    a=Artwork('03_confidence_visualization',147)
    a.text(3,5,'Confidence visualization and geometric evidence',9,INK,True)
    cols=[3,48,93,138]; w=39
    titles=['Input','Confidence visualization','Solver support','Ray error']
    subs=['RGB image','Existing display processing',r'Binary weights $w$',r'$\angle(\widehat{\mathbf{i}},\mathbf{i}^\star)$']
    for x,title,sub in zip(cols,titles,subs):
        a.text(x+w/2,12,title,7.1,ha='center',bold=True,
               color=GREEN if 'Confidence' in title or 'support' in title else INK)
        a.text(x+w/2,16,sub,6.5,GRAY,ha='center')
    for row,k in enumerate(CONFIDENCE):
        y=23+row*28.5
        # 4:3 content stays geometrically registered; no independent crops.
        h=24; ww=32
        a.text(3,y-2.5,f'({chr(97+row)})  {label(k)}',6.8,INK,True)
        full=NATIVE/f"{meta[k]['label']}_{k}_full.png"
        a.image(full,cols[0]+3.5,y,ww,h)
        a.image(OVERLAYS[k],cols[1]+3.5,y,ww,h)
        rgb=raw[k]['rgb'].astype(float); mask=raw[k]['mask']
        # Flat overlay of the ACTUAL binary mask, not the display heatmap.
        selected=np.array(colors.to_rgb(GREEN)); excluded=np.array([.18,.20,.24])
        support=.50*rgb+.50*np.where(mask[...,None],selected,excluded)
        a.image(support,cols[2]+3.5,y,ww,h)
        error=angular(raw[k]['rays'],data[k]['gt'])
        a.image(ERROR_CMAP(np.clip(error/5,0,1))[...,:3],cols[3]+3.5,y,ww,h)
        a.text(cols[3]+ww+3.0,y+20.9,f"{metrics[k]['raw_prediction_mean_deg']:.2f}"+r'$^\circ$',
               7.2,INK,True,ha='right',bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    a.rect(7,139,2.8,2.8,GREEN,GREEN,.5,0)
    a.text(11,140.5,'Selected',6.7,GRAY)
    a.rect(30,139,2.8,2.8,'#30343D','#30343D',.5,0)
    a.text(34,140.5,'Excluded',6.7,GRAY)
    a.text(65,140.5,'Display map and solver weights are distinct.',6.6,GRAY)
    a.colorbar(141.5,138.6,32,caption=False)
    a.text(157.5,135.7,'Angular error (deg)',6.5,GRAY,ha='center')
    return a.export()


def draw_rays(a, k, method, data, meta, x, y, w, h):
    crop=meta[k]['crop']; cx,cy,cw,ch=crop
    a.image(NATIVE/f"{meta[k]['label']}_{k}_crop.png",x,y,w,h)
    # Darkening is applied identically to the photographic background only.
    a.rect(x,y,w,h,fill='#FFFFFF',edge='none',radius=0,lw=0).set_alpha(.15)
    xx,yy=np.meshgrid(np.arange(30,cw-29,35),np.arange(30,ch-29,30))
    gx,gy=xx+cx,yy+cy
    for field,col,lw,z in [(data[k]['gt'],BLUE,1.05,4),
                           (data[k][method],ORANGE if method=='baseline' else GREEN,.58,5)]:
        v=field[:2,gy,gx].astype(float)
        norm=np.linalg.norm(v,axis=0,keepdims=True)
        v=np.divide(v,norm,out=np.zeros_like(v),where=norm>1e-8)
        # Equal 24 px glyphs, using genuine image-plane azimuth; no angle boost.
        dx=v[0]*24*w/cw; dy=v[1]*24*h/ch
        px=x+xx*w/cw; py=y+yy*h/ch
        for sx,sy,ux,uy in zip(px.ravel(),py.ravel(),dx.ravel(),dy.ravel()):
            p=FancyArrowPatch((sx-ux/2,sy-uy/2),(sx+ux/2,sy+uy/2),
                              arrowstyle='-|>',mutation_scale=7.0 if z==4 else 4.5,
                              linewidth=lw,edgecolor=col,facecolor='white' if z==4 else col,
                              shrinkA=0,shrinkB=0,zorder=z)
            a.ax.add_patch(p)


def incidence_figure(data, meta, metrics):
    a=Artwork('04_incidence_field_comparison',146)
    a.text(3,5,'Incidence fields: identical regions and shared error scale',9,INK,True)
    cols=[3,39,75,111,147];w=30;h=22.5
    titles=['Input','Baseline','Ours','Baseline error','Ours error']
    subs=['Shared ROI','ROI: predicted + GT','ROI: predicted + GT','3D angle to GT','3D angle to GT']
    for x,t,s,c in zip(cols,titles,subs,[INK,ORANGE,GREEN,ORANGE,GREEN]):
        a.text(x+w/2,12,t,7.7,c,True,ha='center')
        a.text(x+w/2,16,s,6.5,GRAY,ha='center')
    for row,k in enumerate(INCIDENCE):
        y=24+row*27.6
        cx,cy,cw,ch=meta[k]['crop']
        a.text(3,y-2.6,f'({chr(97+row)})  {label(k)}',6.8,INK,True)
        a.image(NATIVE/f"{meta[k]['label']}_{k}_full.png",cols[0],y,w,h)
        a.rect(cols[0]+cx*w/640,y+cy*h/480,cw*w/640,ch*h/480,'none',RED,1,0)
        draw_rays(a,k,'baseline',data,meta,cols[1],y,w,h)
        draw_rays(a,k,'ours',data,meta,cols[2],y,w,h)
        for method,x in [('baseline',cols[3]),('ours',cols[4])]:
            err=angular(data[k][method],data[k]['gt'])[cy:cy+ch,cx:cx+cw]
            a.image(ERROR_CMAP(np.clip(err/5,0,1))[...,:3],x,y,w,h)
            a.text(x+w-1.1,y+h-2.7,f"{metrics[k][method+'_roi_mean_deg']:.2f}"+r'$^\circ$',
                   7.3,INK,True,ha='right',bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    y=137
    a.line([(4,y),(10,y)],BLUE,1.3,arrow=True)
    a.text(12,y,'GT ray',6.8,BLUE)
    a.line([(34,y),(40,y)],ORANGE,1.3,arrow=True)
    a.text(42,y,'Baseline',6.8,ORANGE)
    a.line([(69,y),(75,y)],GREEN,1.3,arrow=True)
    a.text(77,y,'Ours',6.8,GREEN)
    a.colorbar(115,136.8,58,caption=False)
    a.text(144,133.8,'3D angular error (deg)',6.6,GRAY,ha='center')
    a.text(3,143,'Equal-length arrows show projected ray direction; labels report unclipped ROI means.',6.5,GRAY)
    return a.export()


CAPTIONS={
'01_overall_framework': (
'Overall framework of confidence-aware monocular camera calibration. '
'A hierarchical Swin Transformer and FPN fuse multi-scale features, followed by PSP context aggregation '
'and cascaded neural-window CRFs. A joint convolutional head predicts unit incidence rays '
'and bounded confidence. During training, intrinsics-derived ground-truth rays supervise a '
'confidence-weighted geometric loss with logarithmic regularization. During inference, confidence '
'selection produces weights reused for guided sampling and per-axis weighted consensus of two-point '
'hypotheses. Horizontal and vertical winners provide the four intrinsic parameters. '
'Dashed paths denote training supervision and parameter updates; solid paths denote inference flow. '
'The confidence thumbnail uses the existing processed visualization for display only.'),
'02_confidence_mechanism': (
'Reuse of learned confidence in ray learning and intrinsic estimation. '
'The same bounded confidence field weights the cosine ray error and enters a logarithmic '
'regularizer during training. At inference, image-wise normalization and confidence selection '
'map confidence to nonnegative weights. The sampling distribution includes a small uniform mixture '
'for numerical robustness. Two sampled correspondences generate an intrinsic hypothesis. '
'The same weights also enter horizontal and vertical inlier scores on the sampled pool '
'Omega_s, with ray-plane residual delta and threshold tau. The two axes select their winning '
'parameters independently before conversion to image coordinates. Confidence is clamped for '
'numerical stability inside the implemented training loss; the displayed loss follows the manuscript notation.'),
'03_confidence_visualization': (
'Confidence visualization, solver support, and ray accuracy on four real examples. '
'Each row shows the registered RGB input, an existing confidence display overlay, the actual binary '
'weights used by the inference selector, and the three-dimensional angular error of the corresponding '
'predicted rays relative to intrinsics-derived ground truth. The display overlays include local '
'normalization and image-structure processing and are not raw confidence, calibrated probabilities, '
'or the solver selection mask. Green and dark regions in the support column denote selected and '
'excluded pixels, respectively. Errors share a 0-5 degree scale; labels give unclipped full-image means. '
'These are qualitative examples and do not establish confidence calibration or dataset-wide reliability.'),
'04_incidence_field_comparison': (
'Qualitative incidence-field comparison on four previously selected examples. '
'The input marks a shared region of interest, enlarged in the Baseline and Ours columns. '
'Blue arrows represent ground-truth rays; orange and green arrows represent baseline and proposed '
'predictions at the same grid locations. All arrows have identical glyph lengths and show the '
'image-plane azimuth of the actual three-dimensional rays without angular exaggeration. '
'The two error columns show the full three-dimensional angle to ground truth within the identical ROI, '
'using one 0-5 degree scale; labels give unclipped ROI means. The baseline uses the local '
'baseline checkpoint. The examples and ROIs were selected for qualitative comparison, '
'and their means are not aggregate benchmark results.'),
}


def documents(qa,metrics,manifest):
    text='# English figure captions\n\n'+''.join(f'## Figure {i+1}\n\n{CAPTIONS[k]}\n\n' for i,k in enumerate(CAPTIONS))
    (OUT/'captions_en.md').write_text(text,encoding='utf-8')
    tex=[]
    for i,(k,c) in enumerate(CAPTIONS.items()):
        tex.append('\\begin{figure*}[t]\n\\centering\n'
                   f'\\includegraphics[width=\\textwidth]{{{k}.pdf}}\n'
                   f'\\caption{{{c.replace("Omega_s", "$\\Omega_s$").replace("delta and", "$\\delta$ and").replace("threshold tau", "threshold $\\tau$")}}}\n'
                   f'\\label{{fig:{k}}}\n\\end{{figure*}}\n')
    (OUT/'captions_en.tex').write_text('\n'.join(tex),encoding='utf-8')
    readme='''# 论文插图重绘交付（2026-11-10）

四张图均提供可编辑 SVG、矢量 PDF 和 600 DPI PNG；推荐以 PDF 插入英文论文。
图宽统一为 180 mm，适合双栏通栏。普通标签为 Arial；数学符号按论文记号排版。
SVG 保留文字对象、箭头、边框等矢量对象；照片与热图以位图嵌入。

## 文件

- 01_overall_framework：总体网络、训练监督和置信感知 RANSAC。
- 02_confidence_mechanism：同一置信度在损失、采样、评分中的复用。
- 03_confidence_visualization：输入、已有彩色可视化、实际二值权重、真实射线误差。
- 04_incidence_field_comparison：四个真实样例、冻结 ROI、真实方向箭头和统一角误差。
- captions_en.md / captions_en.tex：英文图注和 LaTeX 插图代码。
- previews/：低分辨率浏览图；figure_preview.jpg：总览；index.html：浏览入口。
- source/：绘图与验证代码；data/：来源、哈希、数值和排版检查。

## 本轮沿用的用户修订

FPN 保留；网络图不写具体置信筛选比例；展示已有彩色置信度可视化，原始置信度及 logits 不作为图中面板。

## 方法依据与需知

1. FPN 的依据是作者提供的“图片19.png”、第六版论文及作者在原聊天中的明确确认。
现存 WildCamera/newcrfs/newcrf_incidencefield.py 仍为 with_neck=False，无法凭现存代码验证丢失版本的 FPN 连接与权重。
因此 01 是作者确认的论文结构；当前缓存的可视化结果来自存留实现，不能据此证明启用 FPN 的效果。
这处差异必须在投稿前由作者核实，不能把结构示意当作现存 checkpoint 的逐层可验证结构。

2. 彩色置信可视化直接复用已有输出，不重新制造高响应区域。英文图注明确显示处理包含局部归一化和图像结构信息；它与原始置信度、实际 solver mask 是不同对象。
03 的第三列根据原始缓存中的实际 mask 绘制。由于浮点饱和并列值，实际支持较宽；不人为稀疏化或修改它。

3. 训练代码实际先 clamp 置信度再计算 c*l - alpha*log(c)。图内公式沿用论文的稳定项写法，英文图注注明实现的 clamp。
评分忠实现存代码：对采样池中射线平面坐标残差分别计算 x/y 分数并选择参数；不是把三维角误差作为 RANSAC 内点条件。
三维角误差仅用于实验可视化。采样中的均匀混合用符号 eta 表示，图内不出现具体筛选比例。

4. 四个入射场样例及 ROI 沿用已选结果，不代表全数据集平均；角误差均为重新归一化后的三维射线夹角。
热图 0-5 度饱和显示，数值标签用截断前均值。真实射线的投影箭头统一长度、同网格，不夸大角度。

5. 没有生成可靠性曲线；现有缓存并非独立完整可靠性验证实验，不能补造此类结论。

## 复现

在项目根目录运行：
`D:/software/anaconda/python.exe -X utf8 tools/draw_paper_figures_20261110.py`

也可以在包含 numpy、matplotlib、Pillow 的 Python 环境中执行 source/ 下副本；脚本自动向上查找项目根目录。
输入文件路径在脚本开头集中管理。查看 PDF 的 QA 渲染位于 tmp/pdfs/paper_figures_redrawn_20261110/。
源模型、训练代码、checkpoint 和原始实验缓存未修改，data/source_manifest.json 记录输入哈希。
'''
    (OUT/'README_zh.md').write_text(readme,encoding='utf-8')
    cards=''.join(f'<article><h2>{i+1:02d} &nbsp; {k[3:].replace("_"," ")}</h2>'
                  f'<a href="{k}.pdf"><img src="previews/{k}.png"></a>'
                  f'<p><a href="{k}.svg">Editable SVG</a> · <a href="{k}.pdf">Vector PDF</a> · '
                  f'<a href="{k}.png">600 DPI PNG</a></p><p>{c}</p></article>'
                  for i,(k,c) in enumerate(CAPTIONS.items()))
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper figures</title>
<style>body{font:16px Arial,sans-serif;color:#233247;background:#f3f5f7;margin:36px auto;max-width:1200px}
article{background:white;padding:24px 30px;margin:24px 0;border:1px solid #dbe0e7;border-radius:8px}
img{width:100%;height:auto}h1{font-size:26px}h2{font-size:20px}a{color:#087f73}p{line-height:1.6}</style>
<h1>Confidence-aware camera calibration / Paper figures</h1>
<p>180 mm width · editable SVG · vector PDF · 600 DPI PNG · Arial</p>
<p><a href="README_zh.md">中文交付说明与方法依据</a> · <a href="captions_en.md">English captions</a></p>'''+cards+'</html>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    (OUT/'data/layout_checks.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    (OUT/'data/metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (OUT/'data/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',26)
    sheet=Image.new('RGB',(1900,2650),'#F1F4F7'); d=ImageDraw.Draw(sheet)
    for idx,k in enumerate(CAPTIONS):
        x=30+(idx%2)*940; y=30+(idx//2)*1310
        im=Image.open(OUT/'previews'/f'{k}.png').convert('RGB')
        im.thumbnail((900,1210),Image.Resampling.LANCZOS)
        d.text((x,y),k.replace('_',' '),font=font,fill=INK)
        sheet.paste(im,(x+(900-im.width)//2,y+50))
    # Crop the unused lower margin, but keep both gallery rows aligned.
    sheet.save(OUT/'figure_preview.jpg',quality=94)


def main():
    for d in (OUT,OUT/'assets',OUT/'data',OUT/'source',OUT/'previews'):
        d.mkdir(parents=True,exist_ok=True)
    samples=json.loads(FROZEN.read_text('utf-8'))['samples']
    meta={Path(m['cache']).stem:m for m in samples}
    needed=list(dict.fromkeys(INCIDENCE+CONFIDENCE))
    data={k:dict(np.load(meta[k]['cache'])) for k in needed}
    raw={k:dict(np.load(RAW/f'{k}.npz')) for k in CONFIDENCE}
    files=[FROZEN,ROOT/'tools/calibrator.py',ROOT/'tools/tools.py',
           ROOT/'WildCamera/newcrfs/newcrf_incidencefield.py',
           ROOT/'WildCamera/train/train_calibrator.py',
           ROOT/'tmp/paper_figure_design_20261109/paper_v6/main.tex',
           Path('C:/Users/admin/Desktop/图片19.png')]
    metrics={}
    for k in needed:
        assert sha(meta[k]['cache'])==meta[k]['cache_sha256'],f'Changed prediction cache: {k}'
        cx,cy,cw,ch=meta[k]['crop']
        eb,eo=[angular(data[k][m],data[k]['gt']) for m in ('baseline','ours')]
        metrics[k]={
            'roi_xywh':meta[k]['crop'],
            'baseline_roi_mean_deg':float(eb[cy:cy+ch,cx:cx+cw].mean()),
            'ours_roi_mean_deg':float(eo[cy:cy+ch,cx:cx+cw].mean()),
            'baseline_full_mean_deg':float(eb.mean()),'ours_full_mean_deg':float(eo.mean())}
        files += [Path(meta[k]['cache']),NATIVE/f"{meta[k]['label']}_{k}_full.png",
                  NATIVE/f"{meta[k]['label']}_{k}_crop.png"]
        if k in raw:
            r=raw[k]
            assert np.allclose(r['rgb'],data[k]['rgb'],atol=1e-7)
            assert np.max(np.abs(r['rays']-data[k]['ours']))<1e-4
            c=r['confidence']; n=(c-c.min())/(c.max()-c.min()+1e-6)
            # torch.kthvalue is one-indexed; use the same threshold and ties.
            kth=max(int((1-.05)*c.size),1)
            mask=n>=np.partition(n.reshape(-1),kth-1)[kth-1]
            assert np.array_equal(mask,r['mask']),f'Mask disagrees with inference: {k}'
            metrics[k].update(selected_fraction=float(mask.mean()),
                              raw_prediction_mean_deg=float(angular(r['rays'],data[k]['gt']).mean()),
                              cached_vs_raw_ray_max_delta=float(np.max(np.abs(r['rays']-data[k]['ours']))))
            files += [RAW/f'{k}.npz',OVERLAYS[k]]
            shutil.copy2(OVERLAYS[k],OUT/'assets'/f'{k}_confidence_visualization.png')
    before={str(p):sha(p) for p in files}
    qa=[overall(data,meta),mechanism(data),confidence_figure(data,raw,meta,metrics),
        incidence_figure(data,meta,metrics)]
    assert all(sha(p)==s for p,s in before.items()),'An input changed during generation'
    manifest=dict(
        date='2026-11-10',font=font_manager.findfont('Arial'),
        baseline_checkpoint='model_zoo/multi-modality newcrfs(8largecos)/step_49999.ckpt',
        confidence_checkpoint='model_zoo/multi-modality newcrfs(train置信度)/step_29999.ckpt',
        FPN_basis='Author-provided network diagram, manuscript and explicit author confirmation; surviving code has with_neck=False.',
        confidence_display='Existing processed outputs reused; neither raw confidence nor a solver weight map.',
        angular_error='degrees(arccos(clip(dot(normalize(ray),normalize(gt)),-1,1)))',
        incidence_samples=INCIDENCE,confidence_samples=CONFIDENCE,
        selection='Existing favorable examples and frozen ROIs; not aggregate benchmark metrics.',
        input_sha256=before,source_files_unchanged=True)
    documents(qa,metrics,manifest)
    target=OUT/'source'/Path(__file__).name
    if Path(__file__).resolve()!=target.resolve():
        shutil.copy2(Path(__file__),target)
    verification=ROOT/'tools/verify_paper_figures_20261110.py'
    if verification.is_file():
        shutil.copy2(verification,OUT/'source'/verification.name)
        bundled=Path('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
        subprocess.run([str(bundled),'-X','utf8',str(verification)],check=True)
    brief=[{k:v for k,v in q.items() if k not in ('text_boxes','edges')} for q in qa]
    print(json.dumps(brief,indent=2))
    print(str(OUT))


if __name__=='__main__':
    main()
