# -*- coding: utf-8 -*-
"""数据转换核心: 深度图(+可选语义标签) -> Occ3D-nuScenes 同构占据 GT。

三态编码 (官方约定, 已实证):
- semantics: 0=others, 1..16=语义类, 17=free; 未观测体素名义上保持 17
- mask_camera/mask_lidar: uint8 0/1, 本帧被射线穿过或打到的体素
- 未知 = mask==0 (语义值不作数), 训练/评测时被忽略
"""
import numpy as np

from common.projection import deproject
from .raycast import cast_rays
from .voxel_grid import make_grid, to_grid

FREE = 17    # 官方语义表的 free 类(空体素), 也是未观测体素的默认填充值
OTHERS = 0   # 官方: others/noise 类


def convert_frame(depth, fx, fy, cx, cy, label=None, voxel=0.4,
                  x_range=(-40, 40), y_range=(-40, 40), z_range=(-1, 5.4),
                  ray_stride=4, k1=0.0, k2=0.0):
    """单帧转换。depth: (H,W) float 米制(无效值 0); label: (H,W) uint8 像素类别 0..16。
    返回 dict(semantics, mask_lidar, mask_camera), 均为官方 uint8 规格。
    mask_lidar = mask_camera (深度相机即唯一射线源, 官方 lidar mask 的合理近似)。"""
    gmin, dims = make_grid(voxel, x_range, y_range, z_range)
    pts, u, v = deproject(depth, fx, fy, cx, cy, k1, k2)
    sem_px = label[v, u] if label is not None else np.full(len(pts), OTHERS, np.uint8)
    if sem_px.max(initial=0) > 16:
        raise ValueError(f"标签类别越界: 最大 {sem_px.max()} — 官方语义表仅 "
                         f"0(others)..16, 17 保留给 free")

    # ---- 占据体素 + 语义(同体素内取最后一个写入, 与 Occ3D 生成脚本一致) ----
    # 初始全填 FREE(17): 未观测体素的名义值, 之后语义只被观测证据(表面/射线)覆盖
    semantics = np.full(dims, FREE, np.uint8)
    idx, inside = to_grid(pts, gmin, voxel, dims)
    idx, sem_px = idx[inside], sem_px[inside]
    semantics[idx[:, 0], idx[:, 1], idx[:, 2]] = sem_px
    occupied = np.zeros(dims, bool)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # ---- ray casting 得 free 与可见掩膜 ----
    dirs = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
    dists = np.linalg.norm(pts, axis=1)
    free = cast_rays(np.zeros(3), dirs, dists, gmin, voxel, dims, ray_stride)
    free &= ~occupied
    semantics[free] = FREE   # 显式写一次: 射线确认的空(数值同默认填充, 语义来源不同)
    mask_camera = free | occupied
    return dict(semantics=semantics,
                mask_lidar=mask_camera.astype(np.uint8),
                mask_camera=mask_camera.astype(np.uint8))
