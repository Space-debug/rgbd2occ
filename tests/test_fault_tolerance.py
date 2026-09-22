# -*- coding: utf-8 -*-
"""容错测试: 单帧坏输入返回 error 状态(含堆栈), 不抛异常、不中断整批。"""
from datasets.sunrgbd.to_nuscenes import process_frame
from datasets.sunrgbd.to_occ import convert_one

K = [[529.5, 0.0, 365.0], [0.0, 529.5, 265.0], [0.0, 0.0, 1.0]]


def test_nuscenes_bad_frame_returns_error():
    r = dict(split="train", name="img-999999",
             img="no/such/img.jpg", dep="no/such/dep.png", K_native=K)
    key, st, err, _ = process_frame((r, "nowhere_out"))
    assert key == "train/img-999999"
    assert st == "error" and "Error" in err


def test_occ_bad_frame_returns_error():
    t = dict(scene="s", token="t", depth="no/such/dep.png",
             fx=529.5, fy=529.5, cx=365.0, cy=265.0)
    tok, st, err, _ = convert_one("nowhere_out", t)
    assert tok == "t"
    assert st == "error" and "Error" in err
