# -*- coding: utf-8 -*-
"""深度域滤波 (点云管线的 stage1, 逐行移植自 rebuild_sunrgbd_nuscenes.py 以保证字节级等价):
5x5 NaN 中值滤波 + 梯度剔除, 抑制深度图散斑噪声与深度不连续处的飞点。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_gradient(depth, valid, dmin=0.3, grad_thr=0.05):
    """depth: (H,W) 米制; valid: 有效像素掩膜(如 0.3<d<8)。
    返回滤波后的深度(无效/被剔除处置 0)。"""
    pad = np.pad(np.where(valid, depth, np.nan), 2, constant_values=np.nan)
    med = np.nanmedian(sliding_window_view(pad, (5, 5)), axis=(-1, -2))
    d1 = np.where(valid, np.where(np.isfinite(med), med, depth), 0)
    gy, gx = np.gradient(np.where(d1 > dmin, d1, np.nan))
    g = np.sqrt(gy * gy + gx * gx)
    return np.where((d1 > dmin) & (np.nan_to_num(g) < grad_thr), d1, 0)


def mask_depth(depth, valid_range=None, fill=0.0):
    """把有效区间外的深度(0=无效、超量程噪声)置为 fill, 返回新数组。
    valid_range=None 表示不过滤。典型: SUN RGB-D 管线用 (0.3, 8.0)。
    (自 occ/preprocess.py 移入 —— 通用深度预处理, 非 occ 专属。)"""
    if valid_range is None:
        return depth
    d = depth.copy()
    d[(d < valid_range[0]) | (d > valid_range[1])] = fill
    return d
