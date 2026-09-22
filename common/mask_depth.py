# -*- coding: utf-8 -*-
"""深度预处理: 有效区间外置 fill, None 表示不过滤。"""
import numpy as np


def mask_depth(depth, valid_range=None, fill=0.0):
    """把有效区间外的深度(0=无效、超量程噪声)置为 fill, 返回新数组。
    valid_range=None 表示不过滤。典型: SUN RGB-D 管线用 (0.3, 8.0)。
    (自 occ/preprocess.py 移入 —— 通用深度预处理, 非 occ 专属。)"""
    if valid_range is None:
        return depth
    d = depth.copy()
    d[(d < valid_range[0]) | (d > valid_range[1])] = fill
    return d
