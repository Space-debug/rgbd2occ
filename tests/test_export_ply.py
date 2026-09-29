# -*- coding: utf-8 -*-
"""export_ply 工具测试: PLY 写出/读回一致性 + 各模式冒烟 (零依赖可跑)。"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import export_ply  # noqa: E402
from common.render_bev import CLASS_COLORS  # noqa: E402


def _run(argv):
    """跑 CLI 并捕获 stdout, 返回输出文本。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_ply.main(argv)
    return buf.getvalue()


def _synth_npz(path):
    """合成一帧 occ: floor(5) + wall(12) 可见占据 + 可见 free + 未知。"""
    sem = np.full((200, 200, 16), 17, np.uint8)
    mc = np.zeros((200, 200, 16), np.uint8)
    sem[100, 100, 2] = 5                       # floor
    sem[100, 101, 2] = 12                      # wall
    sem[100, 102, 3] = 0                       # others (可见未标注)
    sem[42, 42, 2] = 17                        # 可见 free
    sem[7, 7, 1] = 4                           # 占据但不可见 -> 默认掩膜下应被剔除
    mc[100, 100, 2] = mc[100, 101, 2] = mc[100, 102, 3] = mc[42, 42, 2] = 1
    np.savez_compressed(path, semantics=sem, mask_lidar=mc, mask_camera=mc)
    return sem, mc


def test_ply_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.ply")
        xyz = np.array([[1.0, -2.0, 3.5], [0.0, 0.0, 0.0]], np.float64)
        rgb = np.array([[255, 0, 0], [10, 20, 30]], np.uint8)
        export_ply.write_ply(p, xyz, rgb, intensity=np.array([128, 0], np.uint8))
        d = export_ply.read_ply(p)
        assert np.allclose(d["x"], xyz[:, 0]) and np.allclose(d["z"], xyz[:, 2])
        assert list(d["red"]) == [255, 10] and list(d["blue"]) == [0, 30]
        assert "intensity" in d


def test_points_export():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        os.makedirs(os.path.join(td, "train"))
        pts = np.array([[1, 0, 0, 0.5, 0], [2, 1, 1, 0.25, 0]], np.float32)
        pts.tofile(os.path.join(td, "train", "img-000001.pcd.bin"))
        out = _run(["points", os.path.join(td, "train"), "--out", od])
        ply = os.path.join(od, "points_train_img-000001.ply")
        assert os.path.exists(ply)
        d = export_ply.read_ply(ply)
        assert np.allclose(d["x"], [1, 2])
        assert "intensity" not in d                     # 只 xyz+RGB, 无标量场
        assert list(d["red"]) == [127, 63]              # 亮度 -> 灰度着色
        assert "points_train_img-000001.ply" in out


def test_points_limit_and_names():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        os.makedirs(os.path.join(td, "train"))
        for i in (1, 2, 3):
            np.zeros((4, 5), np.float32).tofile(
                os.path.join(td, "train", "img-%06d.pcd.bin" % i))
        _run(["points", os.path.join(td, "train"), "--limit", "2", "--out", od])
        assert sorted(os.listdir(od)) == ["points_train_img-000001.ply",
                                          "points_train_img-000002.ply"]
        _run(["points", os.path.join(td, "train"), "--names", "3", "--out", od])
        assert sorted(os.listdir(od)) == ["points_train_img-000001.ply",
                                          "points_train_img-000002.ply",
                                          "points_train_img-000003.ply"]


def test_points_with_rgb():
    """--with-rgb: 重跑 CPU 反投影管线, PLY 顶点带真彩色 (非灰度)。"""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as raw, \
            tempfile.TemporaryDirectory() as od:
        root = os.path.join(td, "pkg")
        os.makedirs(os.path.join(root, "samples", "LIDAR_TOP", "train"))
        os.makedirs(os.path.join(root, "samples", "CAM_FRONT", "train"))
        np.zeros((12, 5), np.float32).tofile(
            os.path.join(root, "samples", "LIDAR_TOP", "train", "img-000001.pcd.bin"))
        json.dump({"train/img-000001": {
            "W": 20, "H": 20,
            "K_native": [[200.0, 0, 10.0], [0, 200.0, 10.0], [0, 0, 1]], "K_640": []}},
            open(os.path.join(root, "intrinsics_per_frame.json"), "w"))
        # 原始数据: 3m 均匀深度 + 四象限彩色图 -> 反投影点颜色应非灰度
        os.makedirs(os.path.join(raw, "sunrgbd_train_depth"))
        os.makedirs(os.path.join(raw, "SUNRGBD-train_images"))
        Image.fromarray(np.full((20, 20), int(3 * 6553.5), np.uint16)).save(
            os.path.join(raw, "sunrgbd_train_depth", "1.png"))
        quad = np.zeros((20, 20, 3), np.uint8)
        quad[:10, :10] = (255, 0, 0); quad[:10, 10:] = (0, 255, 0)
        quad[10:, :10] = (0, 0, 255); quad[10:, 10:] = (255, 255, 0)
        Image.fromarray(quad).save(os.path.join(raw, "SUNRGBD-train_images",
                                                "img-000001.jpg"))
        _run(["points", os.path.join(root, "samples", "LIDAR_TOP", "train"),
              "--with-rgb", "--raw-root", raw, "--out", od])
        d = export_ply.read_ply(os.path.join(od, "points_train_img-000001.ply"))
        assert len(d["x"]) > 1
        cols = np.stack([d["red"], d["green"], d["blue"]], 1)
        assert len(np.unique(cols, axis=0)) >= 2      # 真彩色: 至少两种颜色


def test_occ_export_legacy_default():
    """默认 legacy 表示: occupied 语义色散点 + free 白色 1/8 抽稀 + 未知不导出。"""
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        tok_dir = os.path.join(td, "sunrgbd-train-kv1", "00b1d48e")
        os.makedirs(tok_dir)
        _synth_npz(os.path.join(tok_dir, "labels.npz"))
        out = _run(["occ", td, "--out", od])
        po = os.path.join(od, "sunrgbd-train-kv1_00b1d48e_occupied_cubes.ply")
        pf = os.path.join(od, "sunrgbd-train-kv1_00b1d48e_free.ply")
        assert os.path.exists(po) and os.path.exists(pf)
        d = export_ply.read_ply(po, with_faces=True)
        # 小方格(立方体网格): 3 个可见占据体素 x (8 顶点+6 面)
        assert len(d["x"]) == 3 * 8 and len(d["faces"]) == 3 * 6
        cols = {tuple(c) for c in zip(d["red"], d["green"], d["blue"])}
        assert tuple(CLASS_COLORS[5]) in cols           # floor
        assert tuple(CLASS_COLORS[12]) in cols          # wall
        assert (128, 128, 128) in cols                  # others 灰
        f = export_ply.read_ply(pf)
        assert len(f["x"]) == 1                         # 1 个可见 free 体素
        assert list(f["red"]) == [245]                  # 白色
        assert "立方体网格" in out and "free" in out


def test_occ_point_style_still_available():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        tok_dir = os.path.join(td, "sc", "tok")
        os.makedirs(tok_dir)
        _synth_npz(os.path.join(tok_dir, "labels.npz"))
        _run(["occ", td, "--style", "point", "--out", od])
        d = export_ply.read_ply(os.path.join(od, "sc_tok_occupied.ply"))
        assert len(d["x"]) == 3                         # 散点: 每体素 1 顶点


def test_occ_free_mode_and_coords():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        npz = os.path.join(td, "labels.npz")
        _synth_npz(npz)
        _run(["occ", npz, "--free-sub", "1", "--out", od])
        (ply,) = [os.path.join(od, f) for f in os.listdir(od)
                  if f.endswith("_free.ply")]
        d = export_ply.read_ply(ply)
        assert len(d["x"]) == 1                       # 唯一可见 free 体素
        # 体素中心: idx(42,42,2) -> (42.5,42.5,2.5)*0.4 + (-40,-40,-1) = (-23,-23,0)
        assert np.allclose([d["x"][0], d["y"][0], d["z"][0]], [-23.0, -23.0, 0.0], atol=1e-4)


def test_occ_uses_npz_voxel_meta():
    """npz 自带 voxel/gmin (0.9.6+) 时优先于 --voxel/--zmin 假设。"""
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        sem, mc = _synth_npz(os.path.join(td, "labels.npz"))
        np.savez_compressed(os.path.join(td, "labels.npz"), semantics=sem,
                            mask_lidar=mc, mask_camera=mc,
                            voxel=np.float32(1.0),
                            gmin=np.array([-100, -100, -2], np.float32))
        _run(["occ", os.path.join(td, "labels.npz"), "--out", od])
        (ply,) = [os.path.join(od, f) for f in os.listdir(od)
                  if f.endswith("_occupied_cubes.ply")]
        d = export_ply.read_ply(ply)
        assert len(d["x"]) == 3 * 8                     # cube 默认: 每体素 8 顶点
        # 3 个体素 (100,100,2)/(100,101,2)/(100,102,3), voxel=1, cube_scale=0.95:
        # 中心±0.475 -> x [0.025,0.975] y [0.025,2.975] z [0.025,1.975]
        for v, lo, hi in ((d["x"], 0.025, 0.975), (d["y"], 0.025, 2.975),
                          (d["z"], 0.025, 1.975)):
            assert np.isclose(v.min(), lo, atol=1e-4)
            assert np.isclose(v.max(), hi, atol=1e-4)


def test_revox_frame_fine_voxels():
    """_revox_frame: 从原始深度按 0.05m 细体素重投影 (legacy 同款默认)。"""
    from PIL import Image
    with tempfile.TemporaryDirectory() as pkg, tempfile.TemporaryDirectory() as raw:
        json.dump({"train/img-000001": {
            "W": 20, "H": 20,
            "K_native": [[50.0, 0, 10.0], [0, 50.0, 10.0], [0, 0, 1]], "K_640": []}},
            open(os.path.join(pkg, "intrinsics_per_frame.json"), "w"))
        os.makedirs(os.path.join(raw, "sunrgbd_train_depth"))
        Image.fromarray(np.full((20, 20), int(3 * 6553.5), np.uint16)).save(
            os.path.join(raw, "sunrgbd_train_depth", "1.png"))
        res = export_ply._revox_frame(pkg, "train", "img-000001", raw, 0.05)
        assert res is not None
        occ = (res["semantics"] < 17) & (res["mask_camera"] > 0)
        assert occ.sum() > 100                       # 3m 墙面: 数百细体素
        assert abs(float(res["voxel"]) - 0.05) < 1e-6
        # 缺深度时优雅返回 None
        assert export_ply._revox_frame(pkg, "train", "img-000009", raw, 0.05) is None


def test_convert_frame_writes_voxel_meta():
    """convert_frame 返回 dict 自带 voxel/gmin (写入 labels.npz 的元数据来源)。"""
    from occ import convert_frame
    dep = np.zeros((8, 8), np.float64)
    dep[4, 4] = 2.0
    res = convert_frame(dep, 50.0, 50.0, 4.0, 4.0)
    assert abs(float(res["voxel"]) - 0.4) < 1e-6      # float32 存储有舍入
    assert np.allclose(res["gmin"], [-40, -40, -1])




def test_demo_generates_full_set():
    """demo: 单帧一键产出 点云/occ/框 PLY + 叠加 PNG + 两张 BEV。"""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        root = td
        os.makedirs(os.path.join(root, "samples", "LIDAR_TOP", "train"))
        os.makedirs(os.path.join(root, "samples", "CAM_FRONT", "train"))
        os.makedirs(os.path.join(root, "gts", "fake-scene", "samp1"))
        pts = np.array([[1, 0, 0, 0.5, 0], [2, 1, 1, 0.25, 0]], np.float32)
        pts.tofile(os.path.join(root, "samples", "LIDAR_TOP", "train",
                                "img-000001.pcd.bin"))
        Image.new("RGB", (200, 200), (90, 90, 90)).save(
            os.path.join(root, "samples", "CAM_FRONT", "train", "img-000001.jpg"))
        _synth_npz(os.path.join(root, "gts", "fake-scene", "samp1", "labels.npz"))
        ver = os.path.join(root, "v1.0-sunrgbd-train")
        os.makedirs(ver)
        json.dump([{"token": "cat1", "name": "chair.indoor", "description": ""}],
                  open(os.path.join(ver, "category.json"), "w"))
        json.dump([{"token": "samp1", "timestamp": 0, "prev": "", "next": "",
                    "scene_token": "sc1"}],
                  open(os.path.join(ver, "sample.json"), "w"))
        json.dump([{"token": "sc1", "log_token": "l1", "nbr_samples": 1,
                    "first_sample_token": "samp1", "last_sample_token": "samp1",
                    "name": "fake-scene", "description": ""}],
                  open(os.path.join(ver, "scene.json"), "w"))
        json.dump([{"token": "sd1", "sample_token": "samp1", "ego_pose_token": "sd1",
                    "calibrated_sensor_token": "cs1", "timestamp": 0,
                    "fileformat": "jpg", "is_key_frame": True, "height": 200,
                    "width": 200,
                    "filename": "samples/CAM_FRONT/train/img-000001.jpg",
                    "prev": "", "next": "", "sensor_modality": "camera"}],
                  open(os.path.join(ver, "sample_data.json"), "w"))
        json.dump([{"token": "inst1", "category_token": "cat1", "nbr_annotations": 1,
                    "first_annotation_token": "a1", "last_annotation_token": "a1"}],
                  open(os.path.join(ver, "instance.json"), "w"))
        json.dump([{"token": "a1", "sample_token": "samp1", "instance_token": "inst1",
                    "attribute_token": "", "translation": [1.0, 0.0, 1.5],
                    "size": [0.8, 1.2, 0.9], "rotation": [1.0, 0.0, 0.0, 0.0],
                    "prev": "", "next": "", "num_lidar_pts": 2, "num_radar_pts": 0}],
                  open(os.path.join(ver, "sample_annotation.json"), "w"))
        json.dump({"train/000001": {
            "Rtilt": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "boxes": [{"cls": "chair", "centroid": [0, 5, 1.5],
                       "coeffs": [1, 1, 1], "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       "bb2d": [50, 50, 40, 40]}]}},
            open(os.path.join(root, "detection_meta_cache.json"), "w"))
        json.dump({"train/img-000001": {
            "W": 200, "H": 200,
            "K_native": [[50, 0, 100], [0, 50, 100], [0, 0, 1]], "K_640": []}},
            open(os.path.join(root, "intrinsics_per_frame.json"), "w"))

        out = _run(["demo", root, "--names", "img-000001", "--revox", "0",
                    "--out", od])
        expect = ["points_train_img-000001.ply",
                  "occ_train_img-000001_fake-scene_samp1_occupied_cubes.ply",
                  "occ_train_img-000001_fake-scene_samp1_free.ply",
                  "preview_train_img-000001.png",
                  "bev_points_img-000001.png", "bev_occ_img-000001.png"]
        for f in expect:
            assert os.path.exists(os.path.join(od, f)), f
        assert "打开方式" in out


def _run_exit(argv):
    """跑 CLI, 返回 (输出, 退出码)。"""
    code = 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            export_ply.main(argv)
        except SystemExit as e:
            code = e.code or 0
    return buf.getvalue(), code


def test_diff_manifest():
    """manifest 对比: 相同退出 0; 有差异退出 1 并列出 changed/仅一侧。"""
    with tempfile.TemporaryDirectory() as td:
        base = {"generator": {"name": "rgbd2occ", "version": "1", "commit": "c",
                              "time": "t"},
                "entries": {"train/img-000001": {"md5": "aaa"},
                            "train/img-000002": {"md5": "bbb"},
                            "train/img-000003": {"md5": "ccc"}}}
        pa = os.path.join(td, "a.json")
        json.dump(base, open(pa, "w"))
        pb = os.path.join(td, "b.json")
        json.dump({"generator": base["generator"],
                   "entries": {"train/img-000001": {"md5": "aaa"},
                               "train/img-000002": {"md5": "XXX"},
                               "train/img-000004": {"md5": "ddd"}}}, open(pb, "w"))
        out, code = _run_exit(["diff", pa, pb])
        assert code == 1
        assert "一致 1, 不同 1" in out and "仅A 1, 仅B 1" in out
        assert "img-000002" in out and "img-000003" in out and "img-000004" in out
        out2, code2 = _run_exit(["diff", pa, pa])
        assert code2 == 0 and "不同 0" in out2


def test_info_and_list_smoke():
    with tempfile.TemporaryDirectory() as td:
        json.dump({"generator": {"name": "rgbd2occ", "version": "0.9.5",
                                 "commit": "d2861bb", "time": "t"},
                   "params": {"backend": "gpu"}, "frames": 1, "entries": {},
                   "product": "x"},
                  open(os.path.join(td, "manifest_nuscenes.json"), "w"))
        json.dump([{"token": "s"}], open(os.path.join(td, "v1.0-x.json"), "w"))
        out = _run(["info", td])
        assert "rgbd2occ" in out and "v1.0-x" in out
    out = _run(["list"])
    assert "sunrgbd" in out and "occ" in out


def test_convert_unknown_dataset_exits():
    try:
        _run(["convert", "nope", "occ"])
        assert False, "应退出"
    except SystemExit:
        pass


def test_preview_renders_2d3d():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        from PIL import Image
        os.makedirs(os.path.join(td, "samples", "CAM_FRONT", "train"))
        Image.new("RGB", (200, 200), (128, 128, 128)).save(
            os.path.join(td, "samples", "CAM_FRONT", "train", "img-000001.jpg"))
        json.dump({"train/000001": {
            "Rtilt": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "boxes": [{"cls": "chair", "centroid": [0.0, 5.0, 1.5],
                       "coeffs": [1.0, 1.0, 1.0],
                       "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       "bb2d": [50, 50, 40, 40]}]}},
            open(os.path.join(td, "detection_meta_cache.json"), "w"))
        json.dump({"train/img-000001": {
            "W": 200, "H": 200,
            "K_native": [[50, 0, 100], [0, 50, 100], [0, 0, 1]],
            "K_640": []}},
            open(os.path.join(td, "intrinsics_per_frame.json"), "w"))
        out = _run(["preview", td, "--split", "train",
                    "--names", "img-000001", "--out", od])
        png = os.path.join(od, "preview_train_img-000001.png")
        assert os.path.exists(png) and os.path.getsize(png) > 0
        assert "2D框 1, 3D框 0" in out            # 默认只画 2D
        out2 = _run(["preview", td, "--split", "train", "--with-3d",
                     "--out", od])
        assert "2D框 1, 3D框 1" in out2
