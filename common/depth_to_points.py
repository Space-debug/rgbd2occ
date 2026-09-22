# -*- coding: utf-8 -*-
"""点云全流程编排 (stage0-3): RGB+深度 -> 自车系点云+颜色。"""
import numpy as np
from .deproject_filtered import deproject_filtered
from .median_gradient import median_gradient
from .sor_radius import sor_radius
from .voxel_speckle import voxel_speckle


def depth_to_points(img, depth, fx, cx, cy, fy=None, raw=None, scale=None):
    """单帧 RGB+深度 -> 自车系点云与颜色 (stage0-3 全流程)。
    img/depth 同形 (H,W[,3]); raw/scale 为原始 uint16 深度与缩放系数,
    提供时启用 median_gradient 的 cv2 整数域快路径; 返回 (P, C)。"""
    fy = fx if fy is None else fy
    valid0 = (depth > 0.3) & (depth < 8)
    d1 = median_gradient(depth, valid0, raw=raw, scale=scale)
    P, m = deproject_filtered(d1, fx, cx, cy, fy)
    C = img[m].astype(np.uint8)
    if len(P) == 0:
        return P.astype(np.float32), C
    P2, C2 = sor_radius(P, C)
    if len(P2) == 0:
        return P2.astype(np.float32), C2
    # 斑点过滤在相机网格朝向 (X右, Y前, Z上) 下进行 —— gmin 默认值是该朝向的
    G = np.stack([-P2[:, 1], P2[:, 0], P2[:, 2]], 1)          # ego -> 网格: (X,Y,Z)->(-Y,X,Z)
    G3, C3 = voxel_speckle(G, C2)
    P3 = np.stack([G3[:, 1], -G3[:, 0], G3[:, 2]], 1)         # 网格 -> ego: 逆变换 (Y,-X,Z)
    # 注意: v2 在此处即舍入到 float32, 降采样质心是对舍入后的值求和 —— 必须
    # 保持这一舍入点, 否则输出与 v2 存在 1 ULP 差异
    return P3.astype(np.float32), C3
