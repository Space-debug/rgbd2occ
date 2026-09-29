# -*- coding: utf-8 -*-
"""数据转换核心: 深度图(+可选语义标签) -> Occ3D-nuScenes 同构占据 GT。

三态编码 (官方约定, 已实证):
- semantics: 0=others, 1..16=语义类, 17=free; 未观测体素名义上保持 17
- mask_camera/mask_lidar: uint8 0/1, 本帧被射线穿过或打到的体素
- 未知 = mask==0 (语义值不作数), 训练/评测时被忽略
"""
import os

import numpy as np

from common import deproject
from .raycast import cast_rays
from .voxel_grid import make_grid, to_grid

FREE = 17    # 官方语义表的 free 类(空体素), 也是未观测体素的默认填充值
OTHERS = 0   # 官方: others/noise 类


def convert_frame(depth, fx, fy, cx, cy, label=None, voxel=0.4,
                  x_range=(-40, 40), y_range=(-40, 40), z_range=(-1, 5.4),
                  ray_stride=4, k1=0.0, k2=0.0, raw=None, scale=None,
                  label_vote=False):
    """单帧转换。depth: (H,W) float 米制(无效值 0); label: (H,W) uint8 像素类别 0..16。
    返回 dict(semantics, mask_lidar, mask_camera), 均为官方 uint8 规格。
    mask_lidar = mask_camera (深度相机即唯一射线源, 官方 lidar mask 的合理近似)。"""
    gmin, dims = make_grid(voxel, x_range, y_range, z_range)

    # ---- gpu 层全 GPU 路径: 体素化+射线追踪全部在卡上 (需 raw+label) ----
    if (raw is not None and scale is not None and label is not None
            and os.environ.get("RGBD2OCC_BACKEND") == "gpu"):
        try:
            from occ.gpu_raycast import voxelize_and_cast_gpu
            free, occ_cells, cell_lab = voxelize_and_cast_gpu(
                raw, scale, fx, fy, cx, cy, label, gmin, voxel, dims, ray_stride)
            semantics = np.full(dims, FREE, np.uint8)
            semantics[occ_cells] = cell_lab            # 未标签格=others(0), 其余 amax
            occupied = np.zeros(dims, bool)
            occupied.reshape(-1)[occ_cells] = True
            free &= ~occupied
            mask_camera = (free | occupied).astype(np.uint8)
            return dict(semantics=semantics,
                        mask_lidar=mask_camera,
                        mask_camera=mask_camera,
                        voxel=np.float32(voxel), gmin=gmin.astype(np.float32))
        except Exception:
            pass  # GPU 失败回退 CPU 精确路径

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
    if label is not None and label_vote:
        # 标签治理: 体素内多数投票 + 未标注(0)让位 (改变数据语义, 需重生成)
        d12 = dims[1] * dims[2]
        flat = idx[:, 0] * d12 + idx[:, 1] * dims[2] + idx[:, 2]
        key = flat * 17 + sem_px                      # sem_px 0..16
        uk, uc = np.unique(key, return_counts=True)
        order = np.lexsort((uk % 17, -uc, uk // 17))  # 每格: 票多优先, 平票取小 id
        first = np.ones(len(order), bool)
        first[1:] = uk[order][1:] // 17 != uk[order][:-1] // 17
        win_flat = uk[order][first] // 17
        win_lab = (uk[order][first] % 17).astype(np.uint8)
        semantics = np.full(dims, FREE, np.uint8)
        semantics.reshape(-1)[win_flat] = win_lab
    else:
        semantics[idx[:, 0], idx[:, 1], idx[:, 2]] = sem_px   # 末次写入(官方行为)
    occupied = np.zeros(dims, bool)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # ---- ray casting 得 free 与可见掩膜 ----
    # gpu 层 + 提供原始深度: 反投影/射线追踪全 GPU (上行 774KB 深度, 下行位压缩),
    # float32 近似; exact/fast 层或 GPU 失败时走 CPU numba 核 (逐位等价基线)
    free = None
    if raw is not None and scale is not None             and os.environ.get("RGBD2OCC_BACKEND") == "gpu":
        try:
            from occ.gpu_raycast import cast_rays_from_depth_gpu
            free = cast_rays_from_depth_gpu(raw, scale, fx, fy, cx, cy, gmin,
                                            voxel, dims, ray_stride,
                                            dmin=1e-4, dmax=1e9)
        except Exception:
            free = None
    if free is None:
        dirs = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
        dists = np.linalg.norm(pts, axis=1)
        free = cast_rays(np.zeros(3), dirs, dists, gmin, voxel, dims, ray_stride)
    free &= ~occupied
    semantics[free] = FREE   # 显式写一次: 射线确认的空(数值同默认填充, 语义来源不同)
    mask_camera = free | occupied
    return dict(semantics=semantics,
                mask_lidar=mask_camera.astype(np.uint8),
                mask_camera=mask_camera.astype(np.uint8),
                voxel=np.float32(voxel), gmin=gmin.astype(np.float32))
