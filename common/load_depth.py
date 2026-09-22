# -*- coding: utf-8 -*-
"""数据读取: 深度图 -> float64 米制 (.npy 或 位深图×scale)。"""
import numpy as np
from ._imread import _load_image


def load_depth(path, depth_scale=1.0):
    """深度图 -> (H, W) float64 米制。"""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    return _load_image(path).astype(np.float64) * depth_scale


def load_depth_raw(path):
    """原始位深深度图 (uint16 ndarray, 未缩放) —— 供 GPU 管线小体积上传。"""
    return _load_image(path)
