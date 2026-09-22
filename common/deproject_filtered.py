# -*- coding: utf-8 -*-
"""v2 精确版反投影: 滤波后深度 -> 自车系点云 (与 v2 字节级等价)。"""
import numpy as np
from .cam_to_ego_axes import cam_to_ego_axes


def deproject_filtered(depth, fx, cx, cy, fy=None, dmin=0.3, dmax=8.0):
    """滤波后的深度 -> 自车系点云 (N,3) float64: X=前(深度z), Y=左(-x), Z=上(-y)。
    返回 (P, 有效像素掩膜 m) —— m 用于取该点对应的 RGB 颜色。"""
    if fy is None:
        fy = fx
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    m = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    z = depth[m].astype(np.float64)
    xd, yd = (u[m] - cx) / fx, (v[m] - cy) / fy
    # 相机系 -> 自车系 (共用轴变换; 纯取负/重排, 与 v2 逐位一致)
    P = cam_to_ego_axes(np.stack([xd * z, yd * z, z], 1))
    return P, m
