# -*- coding: utf-8 -*-
"""BEV 质检渲染: occ 体素 / 点云 -> PNG (含 3D 框叠加), 用于 --inspect 人工复核。

朝向约定: 图像上方 = 自车系 +X(前), 左 = +Y(左); 自车标红点+朝前箭头。"""
import numpy as np
from PIL import Image, ImageDraw

# SUN RGB-D 13 类 (+others) 着色表
CLASS_COLORS = [
    (0, 0, 0),        # 0 others (用白灰, 见下 special-case)
    (255, 127, 80),   # 1 bed
    (0, 255, 255),    # 2 books
    (160, 120, 255),  # 3 ceiling
    (255, 127, 127),  # 4 chair
    (120, 180, 255),  # 5 floor
    (180, 255, 120),  # 6 furniture
    (200, 200, 60),   # 7 objects
    (255, 220, 90),   # 8 picture
    (90, 220, 90),    # 9 sofa
    (255, 160, 60),   # 10 table
    (255, 90, 200),   # 11 tv
    (180, 130, 255),  # 12 wall
    (60, 200, 200),   # 13 window
]


def _ego_marker(draw, cx, cy, scale):
    draw.ellipse([cx - 6 * scale, cy - 6 * scale, cx + 6 * scale, cy + 6 * scale], fill=(255, 0, 0))
    draw.polygon([(cx, cy - 14 * scale), (cx - 7 * scale, cy + 2 * scale),
                  (cx + 7 * scale, cy + 2 * scale)], fill=(255, 0, 0))


def render_occ_bev(semantics, mask_camera, path, size=600, title=None):
    """occ labels -> BEV PNG: 彩色=可见占据(沿 z 取最高类), 绿=可见 free, 深灰=未知。"""
    occ = (semantics < 17) & (mask_camera == 1)
    free = (semantics == 17) & (mask_camera == 1)
    H, W = semantics.shape[:2]
    rgb = np.full((H, W, 3), 24, np.uint8)
    rgb[free.any(2)] = (140, 185, 140)
    cls = np.zeros((H, W), np.uint8)
    for z in range(semantics.shape[2]):
        m = occ[:, :, z]
        cls[m] = semantics[:, :, z][m] + 1
    pal = np.array([[128, 128, 128]] + CLASS_COLORS, np.uint8)
    m2 = cls > 0
    rgb[m2] = pal[cls[m2]]
    img = Image.fromarray(rgb[::-1, ::-1]).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(img)
    _ego_marker(d, size // 2, size // 2, size / 400)
    if title:
        d.text((6, 4), title, fill=(255, 255, 0))
    img.save(path)
    return path


def render_points_bev(pts, path, size=600, extent=40, boxes=None, title=None):
    """点云 (N,5 ego 系) -> BEV PNG (按高度着色); boxes=[(translation,size,rotation_q),...]
    3D 框叠加为红框 (只投 x-y 平面)。"""
    img = Image.new("RGB", (size, size), (24, 24, 24))
    d = ImageDraw.Draw(img)
    if len(pts):
        h = np.clip(pts[:, 2], -1, 5.4)
        t = (h + 1) / 6.4
        xs = (pts[:, 0] / extent * 0.5 + 0.5) * size
        ys = (0.5 - pts[:, 1] / extent * 0.5) * size
        for x, y, tt in zip(xs, ys, t):
            c = (int(60 + 195 * tt), int(80 + 120 * tt), int(255 - 195 * tt))
            d.point((y, size - x), fill=c)  # 上=+X(前), 左=+Y(左)
    if boxes:
        for tr, sz, _q in boxes:
            cx = (0.5 - tr[1] / extent * 0.5) * size
            cy = size - (tr[0] / extent * 0.5 + 0.5) * size
            r = max(sz[0], sz[1]) / extent * 0.5 * size
            d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=(255, 60, 60), width=2)
    _ego_marker(d, size // 2, size // 2, size / 400)
    if title:
        d.text((6, 4), title, fill=(255, 255, 0))
    img.save(path)
    return path
