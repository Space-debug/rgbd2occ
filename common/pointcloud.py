# -*- coding: utf-8 -*-
"""深度 -> 点云全流程编排 + 点云输出 (与 v2 管线字节级等价, 逐算子移植):
stage1 深度滤波 -> 反投影(自车系) -> stage2 SOR -> stage3 体素斑点
-> 体素降采样 -> nuScenes LIDAR_TOP bin (float32 Nx5)。

帧无关性说明: SOR/斑点过滤的邻居计数对坐标轴置换不变, 因此在自车系
(x前,y左,z上)下过滤与 v2 在 OpenGL 系下过滤选出完全相同的点集;
数值运算 (乘法/取负) 均为精确运算, 输出点值与 v2 逐位一致。
"""
import numpy as np

from .depth_filter import median_gradient
from .projection import cam_to_ego_axes
from .point_filter import sor_radius, voxel_speckle


def deproject_filtered(depth, fx, cx, cy, fy=None, dmin=0.3, dmax=8.0):
    """滤波后的深度 -> 自车系点云 (N,3) float64: X=前(深度z), Y=左(-x), Z=上(-y)。
    返回 (P, 有效像素掩膜 m) —— m 用于取该点对应的 RGB 颜色。"""
    if fy is None:
        fy = fx
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    m = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    z = depth[m].astype(np.float64)
    xd, yd = (u[m] - cx) / fx, (v[m] - cy) / fy
    # 相机系 -> 自车系 (共用轴变换; 纯取负/重排, 与 v2 逐位一致)
    P = cam_to_ego_axes(np.stack([xd * z, yd * z, z], 1))
    return P, m


def depth_to_points(img, depth, fx, cx, cy, fy=None):
    """单帧 RGB+深度 -> 自车系点云与颜色 (stage0-3 全流程)。
    img/depth 同形 (H,W[,3]); 返回 (P (N,3) float64, C (N,3) uint8)。"""
    fy = fx if fy is None else fy
    valid0 = (depth > 0.3) & (depth < 8)
    d1 = median_gradient(depth, valid0)
    P, m = deproject_filtered(d1, fx, cx, cy, fy)
    C = img[m].astype(np.uint8)
    if len(P) == 0:
        return P.astype(np.float32), C
    P2, C2 = sor_radius(P, C)
    if len(P2) == 0:
        return P2.astype(np.float32), C2
    # 斑点过滤在相机网格朝向 (X右, Y前, Z上) 下进行 —— gmin 默认值是该朝向的
    G = np.stack([-P2[:, 1], P2[:, 0], P2[:, 2]], 1)          # ego -> 网格: (X,Y,Z)->(-Y,X,Z)
    G3, C3 = voxel_speckle(G, C2)
    P3 = np.stack([G3[:, 1], -G3[:, 0], G3[:, 2]], 1)         # 网格 -> ego: 逆变换 (Y,-X,Z)
    # 注意: v2 在此处即舍入到 float32, 降采样质心是对舍入后的值求和 —— 必须
    # 保持这一舍入点, 否则输出与 v2 存在 1 ULP 差异
    return P3.astype(np.float32), C3


def voxel_downsample(P, C, vox):
    """体素降采样: 每体素 P 取质心, C 取均值(四舍五入)。P:(N,3) C:(N,3)。"""
    idx = np.floor(P / vox).astype(np.int64)
    idx -= idx.min(0)
    dims = idx.max(0) + 1
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    Ps = np.zeros((len(cnt), 3))
    np.add.at(Ps, inv, P)
    Ps /= cnt[:, None]
    Cs = np.zeros((len(cnt), C.shape[1]), np.float64)
    np.add.at(Cs, inv, C)
    Cs = np.round(Cs / cnt[:, None]).astype(np.uint8)
    return Ps, Cs


def write_nuscenes_bin(path, P, C):
    """nuScenes LIDAR_TOP bin: float32 Nx5 (x, y, z, intensity, ring)。
    intensity = 颜色亮度/255, ring 恒 0。"""
    intensity = (C.astype(np.float32).mean(1) / 255.0) if len(C) else np.zeros(0, np.float32)
    pts = np.concatenate([P, intensity[:, None],
                          np.zeros((len(P), 1), np.float32)], 1).astype(np.float32)
    pts.tofile(path)
    return len(pts)
