# -*- coding: utf-8 -*-
"""点云滤波 (stage3): 体素斑点过滤, 26 邻域占据数 < min_nbr 删除。

双实现, 结果严格一致 (占据计数为整数, 见 sor_radius 的说明):
- numba 核 (可用时): 单遍散射占据 + 仅在占据体素上做 27 邻域计数, ~3-5x。
- numpy 稠密路径 (回退): occ 花式赋值 + uniform_filter。"""
import os

import numpy as np

try:
    if os.environ.get("RGBD2OCC_NO_NUMBA"):
        raise ImportError
    from numba import njit

    @njit(cache=True)
    def _speckle_kernel(idx, d0, d1, d2, min_nbr):
        occ = np.zeros(d0 * d1 * d2, np.uint8)
        d12 = d1 * d2
        for i in range(idx.shape[0]):
            # 负索引取模回绕: 与稠密路径 occ[负索引] 的 numpy 语义一致 (历史行为)
            x = idx[i, 0] % d0
            y = idx[i, 1] % d1
            z = idx[i, 2] % d2
            occ[x * d12 + y * d2 + z] = 1
        n = idx.shape[0]
        sel = np.empty(n, np.bool_)
        for i in range(n):
            x = idx[i, 0] % d0
            y = idx[i, 1] % d1
            z = idx[i, 2] % d2
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
                        s += occ[base + zz]
            sel[i] = s >= min_nbr
        return sel

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from ._nbr_count import _nbr_count  # noqa: E402 (回退路径用)


def voxel_speckle(P, C, vox=0.05, gmin=(-4.0, 0.3, -1.5), min_nbr=4):
    """体素斑点过滤: vox 米体素网格中占据体素的 26 邻居数 < min_nbr 的点删除。
    gmin 为网格下界, 默认 (X右-4m, Y前0.3m, Z上-1.5m) —— 相机位于网格内。"""
    idx = np.floor((P - np.asarray(gmin)) / vox).astype(np.int64)
    dims = idx.max(0) + 1
    if _HAS_NUMBA:
        sel = _speckle_kernel(idx, int(dims[0]), int(dims[1]), int(dims[2]), min_nbr)
        return P[sel], C[sel]
    occ = np.zeros(tuple(dims), bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    nbr = _nbr_count(occ.astype(np.float32))
    keep_vox = occ & (nbr >= min_nbr)
    sel = keep_vox[idx[:, 0], idx[:, 1], idx[:, 2]]
    return P[sel], C[sel]
