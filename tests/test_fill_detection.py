# -*- coding: utf-8 -*-
"""检测标注线单元测试: 类别映射 / 四元数 / 坐标链路。"""
import numpy as np

from datasets.sunrgbd.fill_detection import CLASS_MAP, M, rot_to_quat
from datasets.sunrgbd.labels import CLASSES_13


def test_class_map_targets_valid():
    """所有映射目标都必须是 13 类之一。"""
    bad = {k: v for k, v in CLASS_MAP.items() if v not in CLASSES_13}
    assert not bad, f"非法映射目标: {bad}"
    for name in ("chair", "bed", "dining_table", "tv_monitor", "sofa", "bookshelf"):
        assert name in CLASS_MAP, f"常见类别缺失: {name}"


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def test_rot_to_quat_identity_and_yaw():
    q = rot_to_quat(np.eye(3))
    assert np.allclose(q, [1, 0, 0, 0], atol=1e-9)
    th = np.deg2rad(90)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    q = rot_to_quat(Rz)
    assert np.allclose(np.abs(q), [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], atol=1e-9)
    # 往返: R -> q -> R
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.normal(size=(3, 3))
        Q, _r = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        assert np.allclose(_quat_to_R(rot_to_quat(Q)), Q, atol=1e-9)


def test_ego_toolbox_gravity_roundtrip():
    """ego -> 工具箱 -> (左乘 Rtilt) -> 重力, 逆变换应还原。"""
    Rtilt, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    p = np.array([2.0, -0.5, 0.3])           # ego: 前2m, 右0.5m, 上0.3m
    p_grav = Rtilt @ (M @ p)
    p_back = M.T @ Rtilt.T @ p_grav
    assert np.allclose(p_back, p, atol=1e-12)


def test_proper_rotation_mirrored_and_degenerate():
    """SUN RGB-D 实测 15.6% 框为镜像基 (det<0), 约半数非严格正交 ——
    proper_rotation 必须给出纯旋转, 且镜像情形盒体张成不变。"""
    from datasets.sunrgbd.fill_detection import proper_rotation

    # 1) 镜像基: 取反一列后应得 det=+1 正交旋转; 盒体 (列张成) 不变
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    B = Rz.copy()
    B[:, 1] *= -1                                   # det=-1
    assert np.linalg.det(B) < 0
    R = proper_rotation(B)
    assert np.linalg.det(R) > 1 - 1e-9
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    # 盒体不变性: 对每对 ±e_i 系数, 原基与净化基张成的平行六面体同集合
    for c in (np.array([1., 1, 1]), np.array([1., 2, .5])):
        pts = [B @ (c * t) for t in
               [(a, b, d) for a in (-.5, .5) for b in (-.5, .5) for d in (-.5, .5)]]
        pts2 = [R @ (c * t) for t in
                [(a, b, d) for a in (-.5, .5) for b in (-.5, .5) for d in (-.5, .5)]]
        hull1 = np.abs(np.array(pts)).max(0)
        hull2 = np.abs(np.array(pts2)).max(0)
        assert np.allclose(np.sort(hull1), np.sort(hull2), atol=1e-9)

    # 2) 非正交/近奇异基 (实测 clothing_rack det=0.02): 仍得纯旋转, 不抛异常
    D = Rz @ np.diag([1.0, 1.0, 0.02])
    R2 = proper_rotation(D)
    assert np.linalg.det(R2) > 1 - 1e-9
    assert np.allclose(R2.T @ R2, np.eye(3), atol=1e-9)
