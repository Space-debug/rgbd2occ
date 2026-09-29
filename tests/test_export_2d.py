# -*- coding: utf-8 -*-
"""2D 框导出产物线测试: 缓存 -> annotations2d.json 侧车。"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_export_2d_sidecar():
    from datasets.sunrgbd import export_2d
    with tempfile.TemporaryDirectory() as td:
        json.dump({
            "train/000001": {"Rtilt": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "boxes": [
                {"cls": "chair", "centroid": [0, 0, 0], "coeffs": [1, 1, 1],
                 "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "bb2d": [10, 20, 30, 40]},
                {"cls": "desk", "centroid": [1, 0, 0], "coeffs": [1, 1, 1],
                 "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "bb2d": None}]},
            "val/000001": {"Rtilt": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "boxes": []},
            "_stats": {"frames": 2},
        }, open(os.path.join(td, "detection_meta_cache.json"), "w"))
        assert export_2d.main(["--out", td]) == 0
        d = json.load(open(os.path.join(td, "annotations2d.json"), encoding="utf-8"))
        assert list(d) == ["train/000001"]            # 无 2D 框的帧不出现
        b = d["train/000001"]
        assert len(b) == 1 and b[0]["cls"] == "chair"
        assert b[0]["cls13"] == "chair" and b[0]["bbox"] == [10, 20, 30, 40]


def test_export_2d_limit_per_split():
    from datasets.sunrgbd import export_2d
    with tempfile.TemporaryDirectory() as td:
        cache = {}
        for i in (1, 2, 3):
            cache["train/%06d" % i] = {"Rtilt": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       "boxes": [{"cls": "chair", "centroid": [0, 0, 0],
                                                  "coeffs": [1, 1, 1],
                                                  "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                  "bb2d": [0, 0, 5, 5]}]}
        json.dump(cache, open(os.path.join(td, "detection_meta_cache.json"), "w"))
        assert export_2d.main(["--out", td, "--limit", "2"]) == 0
        d = json.load(open(os.path.join(td, "annotations2d.json"), encoding="utf-8"))
        assert sorted(d) == ["train/000001", "train/000002"]
