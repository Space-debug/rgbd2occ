# -*- coding: utf-8 -*-
"""体素降采样: 每体素 P 质心 + C 均值(四舍五入)。"""
import numpy as np


def voxel_downsample(P, C, vox):
    """体素降采样: 每体素 P 取质心, C 取均值(四舍五入)。P:(N,3) C:(N,3)。"""
    idx = np.floor(P / vox).astype(np.int64)
    idx -= idx.min(0)
    dims = idx.max(0) + 1
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    Ps = np.zeros((len(cnt), 3))
    np.add.at(Ps, inv, P)
    Ps /= cnt[:, None]
    Cs = np.zeros((len(cnt), C.shape[1]), np.float64)
    np.add.at(Cs, inv, C)
    Cs = np.round(Cs / cnt[:, None]).astype(np.uint8)
    return Ps, Cs
