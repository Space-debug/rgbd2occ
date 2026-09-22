# -*- coding: utf-8 -*-
"""点云滤波 (stage2): 统计半径滤波, R 网格邻居数 < min_nbr 删除。

双实现, 结果严格一致 (邻居计数是整数; 稠密路径的 float32 均值滤波 x27+rint
对小整数精确还原为整数和):
- numba 核 (可用时): 单遍散射计数 + 仅在占据格上做 27 邻域求和, ~3-5x。
- numpy 稠密路径 (回退): add.at + uniform_filter。"""
import os

import numpy as np

try:
    if os.environ.get("RGBD2OCC_NO_NUMBA"):
        raise ImportError
    from numba import njit

    @njit(cache=True)
    def _sor_kernel(flat, d0, d1, d2, min_nbr):
        # 散射计数与 27 邻域求和单遍完成 (此前散射走 np.add.at, 是阶段热点)
        grid = np.zeros(d0 * d1 * d2, np.int32)
        n = flat.shape[0]
        for i in range(n):
            grid[flat[i]] += 1
        keep = np.empty(n, np.bool_)
        d12 = d1 * d2
        for i in range(n):
            f = flat[i]
            x = f // d12
            y = (f - x * d12) // d2
            z = f - x * d12 - y * d2
            s = 0
            for xx in range(x - 1, x + 2):
                if xx < 0 or xx >= d0:
                    continue
                for yy in range(y - 1, y + 2):
                    if yy < 0 or yy >= d1:
                        continue
                    base = xx * d12 + yy * d2
                    for zz in range(z - 1, z + 2):
                        if zz < 0 or zz >= d2:
                            continue
                        s += grid[base + zz]
            keep[i] = s >= min_nbr
        return keep

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from ._nbr_count import _nbr_count  # noqa: E402 (回退路径用)


def sor_radius(P, C, R=0.03, min_nbr=6):
    """统计半径滤波: R 米网格内邻居数(含自身) >= min_nbr 的点保留。"""
    key = np.floor(P / R).astype(np.int64)
    kmin = key.min(0)
    ki = key - kmin
    dims = ki.max(0) + 3
    flat = (ki[:, 0] * dims[1] + ki[:, 1]) * dims[2] + ki[:, 2]
    if _HAS_NUMBA:
        keep = _sor_kernel(flat, int(dims[0]), int(dims[1]), int(dims[2]), min_nbr)
        return P[keep], C[keep]
    grid = np.zeros(int(np.prod(dims)), np.int32)
    np.add.at(grid, flat, 1)   # 实测比 bincount(minlength=大网格) 快: 分配主导
    nbr = _nbr_count(grid.reshape(dims).astype(np.float32)).ravel()
    keep = nbr[flat] >= min_nbr
    return P[keep], C[keep]
