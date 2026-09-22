# -*- coding: utf-8 -*-
"""通用相机几何: 反投影与坐标轴变换 (occ 占据与 nuScenes 点云两条线共用)。

坐标约定:
- 相机系: x 右, y 下, z 前(深度方向)
- 自车系 (nuScenes/Occ3D 官方): X=前, Y=左, Z=上 —— 变换为 (z, -x, -y)
"""
import numpy as np


def cam_to_ego_axes(pts_cam):
    """(N,3) 相机系坐标 -> 自车系坐标 (只做轴重排/取反, 不平移, 逐位精确)。"""
    g = np.empty_like(pts_cam)
    g[:, 0] = pts_cam[:, 2]
    g[:, 1] = -pts_cam[:, 0]
    g[:, 2] = -pts_cam[:, 1]
    return g


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
