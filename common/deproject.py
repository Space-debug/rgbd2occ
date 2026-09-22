# -*- coding: utf-8 -*-
"""通用反投影: 深度图 -> 相机系点云 (含 k1/k2 径向去畸变), 返回像素坐标。"""
import numpy as np


def deproject(depth, fx, fy, cx, cy, k1=0.0, k2=0.0):
    """深度图 -> 相机系点云 (N,3), 以及每个点对应的像素坐标 u, v (供查语义标签)。
    支持布朗径向畸变 k1/k2: 像素先归一化为畸变坐标, 再去畸变得到直线射线方向。
    有效像素判据: 深度有限且 > 1e-4 (更严的区间过滤交给 common.depth_filter)。"""
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
