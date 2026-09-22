# -*- coding: utf-8 -*-
"""点云滤波 (stage2/3, 逐行移植自 rebuild_sunrgbd_nuscenes.py, uniform_filter 的
float 版邻域计数 —— int 输入会被截断成 0, 必须先转 float32):
- sor_radius:   统计半径滤波, R 网格内 26 邻居数 < min_nbr 的点删除
- voxel_speckle: 体素斑点过滤, 5cm 体素 26 邻域占据数 < min_nbr 的点删除
两个函数均与坐标系无关 (邻居计数对轴置换不变), 传入点应为最终输出朝向。
"""
import numpy as np
from scipy.ndimage import uniform_filter


def _nbr_count(grid_f32):
    """3x3x3 邻域求和 (含自身), 输入 float32 计数网格。"""
    return np.rint(uniform_filter(grid_f32, size=3, mode="constant") * 27).astype(np.int32)


def sor_radius(P, C, R=0.03, min_nbr=6):
    """统计半径滤波: R 米网格内邻居数(含自身) >= min_nbr 的点保留。"""
    key = np.floor(P / R).astype(np.int64)
    kmin = key.min(0)
    ki = key - kmin
    dims = ki.max(0) + 3
    flat = (ki[:, 0] * dims[1] + ki[:, 1]) * dims[2] + ki[:, 2]
    grid = np.zeros(int(np.prod(dims)), np.int32)
    np.add.at(grid, flat, 1)
    nbr = _nbr_count(grid.reshape(dims).astype(np.float32)).ravel()
    keep = nbr[flat] >= min_nbr
    return P[keep], C[keep]


def voxel_speckle(P, C, vox=0.05, gmin=(-4.0, 0.3, -1.5), min_nbr=4):
    """体素斑点过滤: vox 米体素网格中占据体素的 26 邻居数 < min_nbr 的点删除。
    gmin 为网格下界, 默认 (X右-4m, Y前0.3m, Z上-1.5m) —— 相机位于网格内。"""
    idx = np.floor((P - np.asarray(gmin)) / vox).astype(np.int64)
    dims = idx.max(0) + 1
    occ = np.zeros(tuple(dims), bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    nbr = _nbr_count(occ.astype(np.float32))
    keep_vox = occ & (nbr >= min_nbr)
    sel = keep_vox[idx[:, 0], idx[:, 1], idx[:, 2]]
    return P[sel], C[sel]
