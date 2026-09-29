# -*- coding: utf-8 -*-
"""工具冒烟测试: view_pointcloud / view_occ --save 产出 PNG; doctor / BEV 裁剪。"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import doctor, view_occ, view_pointcloud  # noqa: E402
from common.render_bev import _visible_bounds  # noqa: E402


def test_view_pointcloud_saves_png():
    with tempfile.TemporaryDirectory() as td:
        bin_p = os.path.join(td, "t.pcd.bin")
        pts = np.array([[1, 0, 0, 0.5, 0], [2, 1, 1, 0.2, 0]], np.float32)
        pts.tofile(bin_p)
        out = os.path.join(td, "bev.png")
        view_pointcloud.main([bin_p, "--save", out])
        assert os.path.getsize(out) > 0


def _synth_occ():
    sem = np.full((200, 200, 16), 17, np.uint8)
    mc = np.zeros((200, 200, 16), np.uint8)
    sem[100, 100, 2] = 5          # floor
    sem[100, 101, 2] = 12         # wall
    mc[100, 100, 2] = mc[100, 101, 2] = 1
    return sem, mc


def test_view_occ_saves_png():
    with tempfile.TemporaryDirectory() as td:
        sem, mc = _synth_occ()
        npz_p = os.path.join(td, "labels.npz")
        np.savez_compressed(npz_p, semantics=sem, mask_lidar=mc, mask_camera=mc)
        out = os.path.join(td, "bev.png")
        view_occ.main([npz_p, "--save", out])
        assert os.path.getsize(out) > 0


def test_visible_bounds_crop_and_empty():
    vis = np.zeros((200, 200), bool)
    assert _visible_bounds(vis) is None
    vis[95:106, 90:111] = True
    b = _visible_bounds(vis, margin=10)
    assert b == (85, 116, 80, 121)


def test_doctor_reports_lines():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            doctor.run(None)
            code = 0
        except SystemExit as e:
            code = e.code
    out = buf.getvalue()
    assert "numpy" in out and "产物线可用后端" in out
    assert "sunrgbd occ" in out
    assert code == 0        # 测试环境必有基础依赖

