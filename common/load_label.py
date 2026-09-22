# -*- coding: utf-8 -*-
"""数据读取: 语义标签图 -> uint8 (H,W)。"""
import numpy as np
from ._imread import _load_image


def load_label(path):
    """语义标签图 -> (H, W) uint8。"""
    if path.endswith(".npy"):
        return np.load(path).astype(np.uint8)
    return _load_image(path).astype(np.uint8)
