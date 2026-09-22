# -*- coding: utf-8 -*-
"""深度域滤波 (stage1): 5x5 NaN 中值 + 梯度剔除。

三层实现, 全部逐位一致:
1. cv2 (可用且提供 raw/scale 时): 整数域 medianBlur —— 窗口全有效时
   25 个 uint16 的中值与 float64 域中值对应同一个原始值 (scale 为正常数,
   raw*scale 在 uint16 域内保序且单射), 再乘 scale 得到与逐像素缩放相同的
   float64 位型; SIMD 加速, ~1-3ms。
2. np.partition 快路径: 无 NaN 窗口取 25 值第 13 小 (同一顺序统计量)。
3. 含 NaN 窗口 (边缘一圈 + 无效像素邻域, ~5-15%) 走 nanmedian。
NaN 窗口检测用积分图 (布尔严格一致)。golden 双相机字节级锁定。"""
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _cv2_available():
    import os
    if os.environ.get("RGBD2OCC_NO_CV2"):
        return False
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def median_gradient(depth, valid, dmin=0.3, grad_thr=0.05, raw=None, scale=None):
    """depth: (H,W) 米制; valid: 有效像素掩膜(如 0.3<d<8)。
    raw/scale: 原始 uint16 深度与其缩放系数 —— 提供且 cv2 可用时启用整数域
    medianBlur 快路径 (与 nanmedian 逐位一致)。
    返回滤波后的深度(无效/被剔除处置 0)。"""
    H, W = depth.shape
    fast = os.environ.get("RGBD2OCC_BACKEND", "exact") == "fast"
    if fast and raw is not None and scale is not None and _cv2_available():
        import cv2
        # 近似: 全域 cv2 整数中值 (无效像素参与计数), 略过 NaN 窗口分支
        med = cv2.medianBlur(raw, 5).astype(np.float64) * scale
        d1 = np.where(valid, np.where(np.isfinite(med), med, depth), 0)
        gy, gx = np.gradient(np.where(d1 > dmin, d1, np.nan))
        g = np.sqrt(gy * gy + gx * gx)
        return np.where((d1 > dmin) & (np.nan_to_num(g) < grad_thr), d1, 0)
    # NaN 窗口检测走积分图: 窗内无效像素数>0 <=> 窗内含 NaN (含 pad 边缘, 布尔严格一致)
    inv = np.ones((H + 4, W + 4), np.int32)             # pad 边缘视为无效(NaN)
    inv[2:2 + H, 2:2 + W] = ~valid
    c = np.zeros((H + 5, W + 5), np.int64)
    c[1:, 1:] = inv.cumsum(0).cumsum(1)
    has_nan = (c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5]) > 0
    ok = ~has_nan
    med = np.empty(depth.shape, np.float64)
    use_cv2 = (raw is not None and scale is not None and _cv2_available()
               and ok.any())
    if use_cv2:
        import cv2
        # 整数域中值: ok 窗口不含 pad 边缘(边缘必 has_nan), cv2 的边界模式不影响取值
        med[ok] = cv2.medianBlur(raw, 5)[ok].astype(np.float64) * scale
    elif ok.any():
        pad = np.pad(np.where(valid, depth, np.nan), 2, constant_values=np.nan)
        win = sliding_window_view(pad, (5, 5))
        w = win[ok].reshape(-1, 25)
        med[ok] = np.partition(w, 12, axis=1)[:, 12]    # 25 个值的第 13 小 = 中值
    if has_nan.any():
        pad = np.pad(np.where(valid, depth, np.nan), 2, constant_values=np.nan)
        win = sliding_window_view(pad, (5, 5))
        med[has_nan] = np.nanmedian(win[has_nan], axis=(-1, -2))
    d1 = np.where(valid, np.where(np.isfinite(med), med, depth), 0)
    gy, gx = np.gradient(np.where(d1 > dmin, d1, np.nan))
    g = np.sqrt(gy * gy + gx * gx)
    return np.where((d1 > dmin) & (np.nan_to_num(g) < grad_thr), d1, 0)
