# -*- coding: utf-8 -*-
"""数据预处理: 深度有效性掩膜与清洗。

职责边界: 几何转换之前的信号处理。各数据集管线的滤波/去噪都应挂在这里,
保持 convert 流程对"干净的米制深度"这一输入契约不变。
"""
import numpy as np


def mask_depth(depth, valid_range=None, fill=0.0):
    """把有效区间外的深度(0=无效、超量程噪声)置为 fill, 返回新数组。
    valid_range=None 表示不过滤。典型: SUN RGB-D 管线用 (0.3, 8.0)。"""
    if valid_range is None:
        return depth
    d = depth.copy()
    d[(d < valid_range[0]) | (d > valid_range[1])] = fill
    return d
