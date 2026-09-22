# -*- coding: utf-8 -*-
"""共用的 3x3x3 邻域计数 (scipy uniform_filter, float 版)。"""
import numpy as np
from scipy.ndimage import uniform_filter


def _nbr_count(grid_f32):
    """3x3x3 邻域求和 (含自身), 输入 float32 计数网格。"""
    return np.rint(uniform_filter(grid_f32, size=3, mode="constant") * 27).astype(np.int32)
