# -*- coding: utf-8 -*-
"""工具冒烟测试: view_pointcloud / view_occ 在 --save 模式下产出 PNG。"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import view_occ, view_pointcloud  # noqa: E402


def test_view_pointcloud_saves_png():
    with tempfile.TemporaryDirectory() as td:
        bin_p = os.path.join(td, "t.pcd.bin")
        pts = np.array([[1, 0, 0, 0.5, 0], [2, 1, 1, 0.2, 0]], np.float32)
        pts.tofile(bin_p)
        out = os.path.join(td, "bev.png")
        view_pointcloud.main([bin_p, "--save", out])
        assert os.path.getsize(out) > 0


def test_view_occ_saves_png():
    with tempfile.TemporaryDirectory() as td:
        sem = np.full((200, 200, 16), 17, np.uint8)
        mc = np.zeros((200, 200, 16), np.uint8)
        sem[100, 100, 2] = 5          # floor
        sem[100, 101, 2] = 12         # wall
        mc[100, 100, 2] = mc[100, 101, 2] = 1
        npz_p = os.path.join(td, "labels.npz")
        np.savez_compressed(npz_p, semantics=sem, mask_lidar=mc, mask_camera=mc)
        out = os.path.join(td, "bev.png")
        view_occ.main([npz_p, "--save", out])
        assert os.path.getsize(out) > 0
