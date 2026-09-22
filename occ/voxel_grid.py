# -*- coding: utf-8 -*-
"""体素网格: 网格构建与相机系 -> 官方自车系的轴变换。

官方 Occ3D-nuScenes 约定 (已用 mini 数据实证):
- 轴序 (X, Y, Z) = 自车系 (x=前, y=左, z=上), npz 第 0/1/2 轴与之对应
- 默认 0.4m, x/y∈[-40,40], z∈[-1,5.4] -> (200, 200, 16)
- 维度 = 范围整除截断 (80/0.4=200, 6.4/0.4=16), 上边界点由 to_grid 过滤
相机系(x右/y下/z前) -> 自车系: X=相机z(前), Y=-相机x(左), Z=-相机y(上)
"""
import numpy as np

from common import cam_to_ego_axes  # noqa: F401 (共用轴变换, 此处重导出)


def make_grid(voxel, x_range=(-40, 40), y_range=(-40, 40), z_range=(-1, 5.4)):
    """返回网格原点 gmin(=各轴下界) 与维度 dims。"""
    gmin = np.array([x_range[0], y_range[0], z_range[0]])
    dims = np.floor((np.array([x_range[1], y_range[1], z_range[1]]) - gmin)
                    / voxel).astype(int)
    return gmin, tuple(int(d) for d in dims)


def to_grid(pts_cam, gmin, voxel, dims):
    """(N,3) 相机系点 -> (N,3) 整数体素索引, 并给出网格内掩膜。"""
    idx = np.floor((cam_to_ego_axes(pts_cam) - gmin) / voxel).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(dims)), 1)
    return idx, inside
