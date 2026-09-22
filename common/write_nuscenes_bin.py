# -*- coding: utf-8 -*-
"""nuScenes LIDAR_TOP bin 写出: float32 Nx5 (x,y,z,intensity,ring)。"""
import numpy as np


def write_nuscenes_bin(path, P, C):
    """nuScenes LIDAR_TOP bin: float32 Nx5 (x, y, z, intensity, ring)。
    intensity = 颜色亮度/255, ring 恒 0。"""
    intensity = (C.astype(np.float32).mean(1) / 255.0) if len(C) else np.zeros(0, np.float32)
    pts = np.concatenate([P, intensity[:, None],
                          np.zeros((len(P), 1), np.float32)], 1).astype(np.float32)
    pts.tofile(path)
    return len(pts)
