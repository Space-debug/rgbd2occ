# -*- coding: utf-8 -*-
"""语义标签接入 / manifest / QC 测试。"""
import json
import os
import tempfile

import numpy as np

from common import run_qc, write_manifest
from datasets.sunrgbd.labels import CLASSES_13, label_path, semantic_classes_doc
from occ import convert_frame


def test_label_path_resolution():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "train13labels")
        os.makedirs(d)
        open(os.path.join(d, "img13labels-000001.png"), "wb").close()
        assert label_path(td, "train", 1).endswith("img13labels-000001.png")
        assert label_path(td, "train", 2) is None


def test_semantic_classes_doc():
    doc = semantic_classes_doc()
    assert doc["mapping"] == "pixel value == semantic id"
    assert len([k for k in doc["classes"] if k not in ("0", "17")]) == 13
    assert doc["classes"]["1"] == CLASSES_13[0] == "bed"
    assert doc["classes"]["17"] .startswith("free")


def test_convert_frame_with_label():
    """带标签: 占据体素携带标签类 id; 标签越界(>16)报错。"""
    depth = np.full((240, 320), 3.0)
    label = np.full((240, 320), 7, np.uint8)          # 全图 = 类 7 (furniture)
    res = convert_frame(depth, 300, 300, 160, 120, label=label, ray_stride=8)
    occ = res["semantics"][(res["semantics"] < 17) & (res["mask_camera"] == 1)]
    assert len(occ) > 0 and set(np.unique(occ).tolist()) == {7}
    try:
        convert_frame(depth, 300, 300, 160, 120,
                      label=np.full((240, 320), 20, np.uint8), ray_stride=8)
        raise AssertionError("标签越界未被拦截")
    except ValueError:
        pass


def _build_occ_pkg(td, n=3, break_file=None, bad_k=False):
    root = os.path.join(td, "pkg")
    entries = {}
    ann = {"train_split": ["s1"], "val_split": [], "scene_infos": {"s1": {}}}
    for i in range(n):
        tok = "t%d" % i
        d = os.path.join(root, "gts", "s1", tok)
        os.makedirs(d)
        sem = np.full((200, 200, 16), 17, np.uint8)
        mc = np.zeros((200, 200, 16), np.uint8)
        mc.reshape(-1)[:100 + i] = 1
        np.savez_compressed(os.path.join(d, "labels.npz"),
                            semantics=sem, mask_lidar=mc, mask_camera=mc)
        entries[tok] = {"file": f"gts/s1/{tok}/labels.npz", "count": int(mc.sum())}
        ann["scene_infos"]["s1"][tok] = {
            "timestamp": i, "ego_pose": None,
            "gt_path": f"gts/s1/{tok}/labels.npz", "prev": "EOF", "next": "EOF",
            "camera_sensor": {"CAM_FRONT": {
                "intrinsics": [[-500.0 if bad_k else 500.0, 0, 320],
                               [0, 500.0, 240], [0, 0, 1]],
                "extrinsic": None, "ego_pose": None, "img_path": ""}}}
    json.dump(ann, open(os.path.join(root, "annotations.json"), "w"))
    write_manifest(root, "occ", params={"test": True}, entries=entries)
    if break_file:
        os.remove(os.path.join(root, break_file))
    return root


def test_manifest_and_qc_pass():
    with tempfile.TemporaryDirectory() as td:
        root = _build_occ_pkg(td)
        man = json.load(open(os.path.join(root, "manifest_occ.json")))
        assert man["generator"]["name"] == "rgbd2occ" and man["frames"] == 3
        e = man["entries"]["t0"]
        assert e["bytes"] > 0 and len(e["md5"]) == 32 and e["count"] == 100
        q = run_qc(root, "occ", sample_every=1)
        assert q == {"errors": 0, "warnings": 0}


def test_qc_catches_missing_file_and_bad_intrinsics():
    with tempfile.TemporaryDirectory() as td:
        root = _build_occ_pkg(td, break_file="gts/s1/t1/labels.npz")
        q = run_qc(root, "occ", sample_every=1)
        assert q["errors"] >= 1 and any("文件缺失" in e for e in
               json.load(open(os.path.join(root, "qc_report_occ.json")))["errors"])
    with tempfile.TemporaryDirectory() as td:
        root = _build_occ_pkg(td, bad_k=True)
        q = run_qc(root, "occ", sample_every=1)
        assert q["errors"] >= 1 and any("内参非法" in e for e in
               json.load(open(os.path.join(root, "qc_report_occ.json")))["errors"])
