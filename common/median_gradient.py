# -*- coding: utf-8 -*-
"""深度域滤波 (stage1): 5x5 NaN 中值 + 梯度剔除。

性能: 窗口内无 NaN 的像素走 np.partition 快路径 (第 13 小, 与 nanmedian
逐位一致 —— 都是从同一组输入值里取同一个顺序统计量, 无任何算术);
仅含 NaN 的窗口 (图像边缘一圈 + 无效像素邻域) 走 nanmedian。
在 SUN RGB-D 上通常只占 ~5-15% 像素; NaN 窗口检测用积分图(布尔严格一致)。
所有路径输出逐位不变(golden 双相机字节级锁定)。"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_gradient(depth, valid, dmin=0.3, grad_thr=0.05):
    """depth: (H,W) 米制; valid: 有效像素掩膜(如 0.3<d<8)。
    返回滤波后的深度(无效/被剔除处置 0)。"""
    H, W = depth.shape
    # NaN 窗口检测走积分图: 窗内无效像素数>0 <=> 窗内含 NaN (含 pad 边缘, 布尔严格一致)
    inv = np.ones((H + 4, W + 4), np.int32)             # pad 边缘视为无效(NaN)
    inv[2:2 + H, 2:2 + W] = ~valid
    c = np.zeros((H + 5, W + 5), np.int64)
    c[1:, 1:] = inv.cumsum(0).cumsum(1)
    has_nan = (c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5]) > 0
    pad = np.pad(np.where(valid, depth, np.nan), 2, constant_values=np.nan)
    win = sliding_window_view(pad, (5, 5))
    med = np.empty(depth.shape, np.float64)
    ok = ~has_nan
    if ok.any():
        w = win[ok].reshape(-1, 25)
        med[ok] = np.partition(w, 12, axis=1)[:, 12]    # 25 个值的第 13 小 = 中值
    if has_nan.any():
        med[has_nan] = np.nanmedian(win[has_nan], axis=(-1, -2))
    d1 = np.where(valid, np.where(np.isfinite(med), med, depth), 0)
    gy, gx = np.gradient(np.where(d1 > dmin, d1, np.nan))
    g = np.sqrt(gy * gy + gx * gx)
    return np.where((d1 > dmin) & (np.nan_to_num(g) < grad_thr), d1, 0)
