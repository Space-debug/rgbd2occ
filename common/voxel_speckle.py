# -*- coding: utf-8 -*-
"""点云滤波 (stage3): 体素斑点过滤, 26 邻域占据数 < min_nbr 删除。"""
import numpy as np
from ._nbr_count import _nbr_count


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
