# -*- coding: utf-8 -*-
"""点云滤波 (stage2): 统计半径滤波, R 网格邻居数 < min_nbr 删除。"""
import numpy as np
from ._nbr_count import _nbr_count


def sor_radius(P, C, R=0.03, min_nbr=6):
    """统计半径滤波: R 米网格内邻居数(含自身) >= min_nbr 的点保留。"""
    key = np.floor(P / R).astype(np.int64)
    kmin = key.min(0)
    ki = key - kmin
    dims = ki.max(0) + 3
    flat = (ki[:, 0] * dims[1] + ki[:, 1]) * dims[2] + ki[:, 2]
    grid = np.zeros(int(np.prod(dims)), np.int32)
    np.add.at(grid, flat, 1)   # 实测比 bincount(minlength=大网格) 快: 分配主导
    nbr = _nbr_count(grid.reshape(dims).astype(np.float32)).ravel()
    keep = nbr[flat] >= min_nbr
    return P[keep], C[keep]
