# -*- coding: utf-8 -*-
"""字节级/逐元素 golden 回归: 输入与期望输出都签入仓库 (tests/data + tests/golden),
不依赖本机任何数据集 —— 换机器也能跑。

- img-000001 为 SUN RGB-D train 第 1 帧 (kv2, 730x530), 内参来自
  SUNRGBDMeta.mat[5050] 的真实 K_native (fx=fy=529.5, cx=365, cy=265)。
- golden bin 为 v2/v3 管线对该帧的产物 (已与 sunrgbd_nuscenes_v3 字节级一致)。
- golden occ npz 为 to_occ 单帧模式(无深度过滤)对该帧的产物。
任何破坏字节级等价的改动都会在这里被抓住。
"""
import os

import numpy as np
from PIL import Image

from common import (depth_to_points, load_depth, voxel_downsample,
                    write_nuscenes_bin)
from occ import convert_frame

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GOLDEN = os.path.join(HERE, "golden")
K = [[529.5, 0.0, 365.0], [0.0, 529.5, 265.0], [0.0, 0.0, 1.0]]  # kv2 帧1 真实内参


def _frame1():
    dep = load_depth(os.path.join(DATA, "1.png"), 1.0 / 6553.5)
    img = np.array(Image.open(os.path.join(DATA, "img-000001.jpg")).convert("RGB"))
    assert dep.shape == img.shape[:2], "fixture 形状不一致"
    return img, dep


def test_nuscenes_pointcloud_golden_bytes():
    """点云全流程 (滤波->降采样->bin) 输出与 v2/v3 管线逐字节一致。"""
    img, dep = _frame1()
    P, C = depth_to_points(img, dep, K[0][0], K[0][2], K[1][2])
    P, C = voxel_downsample(P.astype(np.float64), C.astype(np.float64), 0.03)
    out = os.path.join(GOLDEN, "_tmp_out.bin")
    try:
        write_nuscenes_bin(out, P, C)
        got = open(out, "rb").read()
        want = open(os.path.join(GOLDEN, "img-000001.pcd.bin"), "rb").read()
        assert got == want, "点云 bin 与 golden 不再字节级一致 —— 实现被改变!"
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_occ_labels_golden_arrays():
    """occ 转换 (无深度过滤) 输出与基准逐元素一致。"""
    _, dep = _frame1()
    res = convert_frame(dep, K[0][0], K[1][1], K[0][2], K[1][2], ray_stride=4)
    ref = np.load(os.path.join(GOLDEN, "occ_frame1.npz"))
    for k in ["semantics", "mask_lidar", "mask_camera"]:
        assert np.array_equal(res[k], ref[k]), f"{k} 与 golden 不再逐元素一致"


def test_nuscenes_pointcloud_golden_kv1():
    """kv1 传感器 (561x427) 的第二份 golden: 防住"只对一种相机正确"。"""
    img = np.array(Image.open(os.path.join(DATA, "img-001925.jpg")).convert("RGB"))
    dep = load_depth(os.path.join(DATA, "1925.png"), 1.0 / 6553.5)
    if dep.shape != img.shape[:2]:
        img = np.array(Image.open(os.path.join(DATA, "img-001925.jpg")).convert("RGB")
                       .resize((dep.shape[1], dep.shape[0])))
    K = [[518.857901, 0.0, 284.582449], [0.0, 519.469611, 208.736166], [0.0, 0.0, 1.0]]
    P, C = depth_to_points(img, dep, K[0][0], K[0][2], K[1][2])
    P, C = voxel_downsample(P.astype(np.float64), C.astype(np.float64), 0.03)
    out = os.path.join(GOLDEN, "_tmp_kv1.bin")
    try:
        write_nuscenes_bin(out, P, C)
        assert open(out, "rb").read() == open(
            os.path.join(GOLDEN, "img-001925.pcd.bin"), "rb").read()
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_occ_labels_golden_kv1():
    """kv1 带标签 occ golden: 锁定语义标签管线 (掩膜+标签+转换全链)。"""
    from occ import mask_depth
    dep = mask_depth(load_depth(os.path.join(DATA, "1925.png"), 1.0 / 6553.5), (0.3, 8.0))
    from common import load_label
    lab = load_label(os.path.join(DATA, "img13labels-001925.png"))
    res = convert_frame(dep, 518.857901, 519.469611, 284.582449, 208.736166, label=lab)
    ref = np.load(os.path.join(GOLDEN, "occ_kv1_labeled.npz"))
    for k in ["semantics", "mask_lidar", "mask_camera"]:
        assert np.array_equal(res[k], ref[k]), f"{k} 与 kv1 带标签 golden 不一致"


def test_median_cv2_path_bit_exact():
    """cv2 整数域快路径与 numpy 路径逐位一致 (cv2 大版本升级时此测试兜底)。"""
    import numpy as np
    from PIL import Image
    from common.median_gradient import _cv2_available, median_gradient
    if not _cv2_available():
        return
    raw = np.array(Image.open(os.path.join(DATA, "1.png")))
    scale = 1.0 / 6553.5
    dep = raw.astype(np.float64) * scale
    valid = (dep > 0.3) & (dep < 8)
    a = median_gradient(dep, valid)
    b = median_gradient(dep, valid, raw=raw, scale=scale)
    assert np.array_equal(a, b), "cv2 整数域路径与 numpy 路径出现偏差!"


def test_numba_kernels_bit_exact():
    """numba 核与 numpy 稠密回退路径逐位一致 (SOR + 体素斑点)。"""
    import importlib
    import numpy as np
    from PIL import Image
    from common.depth_to_points import depth_to_points
    sr = importlib.import_module("common.sor_radius")
    vs = importlib.import_module("common.voxel_speckle")
    if not sr._HAS_NUMBA:
        return
    raw = np.array(Image.open(os.path.join(DATA, "1925.png")))
    dep = raw.astype(np.float64) / 6553.5
    img = np.array(Image.open(os.path.join(DATA, "img-001925.jpg")).convert("RGB"))
    if dep.shape != img.shape[:2]:
        img = np.array(Image.open(os.path.join(DATA, "img-001925.jpg")).convert("RGB")
                       .resize((dep.shape[1], dep.shape[0])))
    kw = dict(raw=raw, scale=1.0 / 6553.5)
    sr._HAS_NUMBA = vs._HAS_NUMBA = False
    P0, C0 = depth_to_points(img, dep, 518.857901, 284.582449, 208.736166, **kw)
    sr._HAS_NUMBA = vs._HAS_NUMBA = True
    P1, C1 = depth_to_points(img, dep, 518.857901, 284.582449, 208.736166, **kw)
    assert np.array_equal(P0, P1) and np.array_equal(C0, C1), "numba 核出现数值偏差!"
