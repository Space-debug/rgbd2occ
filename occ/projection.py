# -*- coding: utf-8 -*-
"""几何投影: 深度图 -> 相机系点云。

相机系约定: x 向右, y 向下, z 向前(深度方向)。
支持布朗径向畸变 k1/k2: 像素先归一化为畸变坐标, 再去畸变得到直线射线方向
(仅影响射线方向, 深度值本身按 z 使用, 与逐像素全去畸变等价于一阶近似)。
"""
import numpy as np


def deproject(depth, fx, fy, cx, cy, k1=0.0, k2=0.0):
    """深度图 -> 相机系点云 (N,3), 以及每个点对应的像素坐标 u, v。"""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(depth) & (depth > 1e-4)
    z = depth[valid]
    xd = (u[valid] - cx) / fx
    yd = (v[valid] - cy) / fy
    if k1 or k2:
        r2 = xd * xd + yd * yd
        f = 1 + k1 * r2 + k2 * r2 * r2
        xd, yd = xd * f, yd * f
    return np.stack([xd * z, yd * z, z], 1), u[valid], v[valid]
