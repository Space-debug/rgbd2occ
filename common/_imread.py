# -*- coding: utf-8 -*-
"""共用的图像底层读取 (cv2 优先, 无则 PIL)。"""
import numpy as np


def _load_image(path):
    try:
        import cv2
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path))
