# -*- coding: utf-8 -*-
"""共用数据读取: 深度图 / 语义标签图 / 图像的加载与格式归一。

从 occ/data_io.py 上移为跨模块共用 (occ 标注与 nuScenes 点云两条管线都用)。
职责边界: 只管"文件 -> 数组", 不做任何几何或清洗逻辑。
- 深度统一输出 float64 米制 (.npy 视为已是米制, 位深图乘 depth_scale)
- 常见 depth_scale: RealSense 16bit = 0.001, SUN RGB-D 16bit = 1/6553.5
"""
import numpy as np


def _load_image(path):
    try:
        import cv2
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path))


def load_depth(path, depth_scale=1.0):
    """深度图 -> (H, W) float64 米制。"""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    return _load_image(path).astype(np.float64) * depth_scale


def load_label(path):
    """语义标签图 -> (H, W) uint8。"""
    if path.endswith(".npy"):
        return np.load(path).astype(np.uint8)
    return _load_image(path).astype(np.uint8)
