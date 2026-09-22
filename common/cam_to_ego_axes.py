# -*- coding: utf-8 -*-
"""相机系(x右,y下,z前) -> 自车系(X前,Y左,Z上) 轴变换, 全库唯一一份。"""
import numpy as np


def cam_to_ego_axes(pts_cam):
    """(N,3) 相机系坐标 -> 自车系坐标 (只做轴重排/取反, 不平移, 逐位精确)。"""
    g = np.empty_like(pts_cam)
    g[:, 0] = pts_cam[:, 2]
    g[:, 1] = -pts_cam[:, 0]
    g[:, 2] = -pts_cam[:, 1]
    return g
