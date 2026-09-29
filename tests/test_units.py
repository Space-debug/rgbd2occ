# -*- coding: utf-8 -*-
"""单元测试: 坐标约定 / 网格 / 编码 / annotations 结构 —— 不依赖外部数据。"""
import os
import tempfile

import numpy as np

from common import cam_to_ego_axes, mask_depth, write_nuscenes_bin
from occ import FREE, OTHERS, convert_frame
from occ.annotations import OccAnnotations
from occ.voxel_grid import make_grid
from nustables import NU_TABLES, token, write_tables


def test_cam_to_ego_axes():
    """相机系(右,下,前) -> 自车系(前,左,上): (x,y,z) -> (z,-x,-y)。"""
    p = np.array([[1.0, 2.0, 3.0]])
    out = cam_to_ego_axes(p)
    assert np.array_equal(out, [[3.0, -1.0, -2.0]])


def test_make_grid_official():
    """官方规格: 0.4m, [-40,40]^2 x [-1,5.4] -> (200,200,16), 维度整除截断。"""
    gmin, dims = make_grid(0.4)
    assert np.array_equal(gmin, [-40.0, -40.0, -1.0])
    assert dims == (200, 200, 16)


def test_mask_depth():
    d = np.array([[0.0, 0.5, 9.0]])
    m = mask_depth(d, (0.3, 8.0))
    assert m.tolist() == [[0.0, 0.5, 0.0]]
    assert mask_depth(d, None) is d  # 不过滤时原样返回


def test_convert_frame_synthetic():
    """合成场景: 3m 处一面墙 -> 占据体素在轴0(X=前)索引~107, 三态编码正确。"""
    depth = np.full((240, 320), 3.0)
    res = convert_frame(depth, 300, 300, 160, 120, ray_stride=8)
    sem, mc, ml = res["semantics"], res["mask_camera"], res["mask_lidar"]
    assert sem.shape == (200, 200, 16) and sem.dtype == np.uint8
    assert mc.dtype == np.uint8 and np.array_equal(ml, mc)
    occ = np.argwhere((sem < 17) & (mc == 1))
    assert len(occ) > 0
    assert occ[:, 0].min() == 103 or occ[:, 0].max() == 107  # 3m/0.4-40=107 附近(取整)
    assert set(np.unique(sem).tolist()) <= set(range(18))
    # 未观测体素: 语义名义 FREE(17) 且 mask=0 —— "未知"仅由 mask 表达
    assert ((sem == 17) & (mc == 0)).sum() > 0


def test_occ_annotations_chain():
    """帧链 prev/next + EOF 边界 + 位姿置空 + gt_path。"""
    with tempfile.TemporaryDirectory() as td:
        ann = OccAnnotations(os.path.join(td, "annotations.json"))
        for i, tok in enumerate(["t1", "t2", "t3"]):
            ann.add("s1", tok, "train", 500.0, 500.0, 320.0, 240.0, img=f"{tok}.jpg", ts=i)
        ann.save()
        data = ann.data["scene_infos"]["s1"]
        ks = list(data)
        assert data[ks[0]]["prev"] == "EOF" and data[ks[-1]]["next"] == "EOF"
        assert data[ks[0]]["next"] == ks[1] and data[ks[1]]["prev"] == ks[0]
        e = data[ks[0]]
        assert sorted(e.keys()) == ["camera_sensor", "ego_pose", "gt_path",
                                     "next", "prev", "timestamp"]
        assert e["ego_pose"] is None                       # 无真值置空
        assert e["gt_path"] == "gts/s1/%s/labels.npz" % ks[0]
        cam = e["camera_sensor"]["CAM_FRONT"]
        assert cam["extrinsic"] is None and cam["intrinsics"][0][0] == 500.0


def test_tables_token_and_write():
    """token 为确定性 md5; write_tables 落盘 13 张表且可回读。"""
    assert token("abc") == token("abc") and len(token("abc")) == 32
    assert token("abc") != token("abd")
    with tempfile.TemporaryDirectory() as td:
        write_tables(td, {"sensor": [{"token": "x"}]})
        names = sorted(f[:-5] for f in os.listdir(td))
        assert names == sorted(NU_TABLES)
        import json
        assert json.load(open(os.path.join(td, "sample.json"))) == []


def test_write_nuscenes_bin():
    P = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    C = np.array([[255, 0, 0], [0, 255, 255]], np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.pcd.bin")
        n = write_nuscenes_bin(p, P, C)
        assert n == 2
        arr = np.fromfile(p, np.float32).reshape(-1, 5)
        assert arr.shape == (2, 5)
        assert np.allclose(arr[0, :3], [1, 2, 3])
        assert abs(arr[0, 3] - 255 / 3 / 255) < 1e-6       # intensity = 均值/255
        assert np.all(arr[:, 4] == 0)                       # ring 恒 0
