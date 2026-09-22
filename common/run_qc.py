# -*- coding: utf-8 -*-
"""自动质检 (基于 manifest + 抽样深检), 批量转换后自动运行。

错误(计为失败, 退出码 1):
- manifest 缺失 / 条目文件缺失 / 大小不符 / 点数为 0
- 内参非法 (fx<=0 / cx,cy 超出图像范围)
- occ 抽检: npz 键/dtype/shape 不符, semantics 越界(>17), mask_lidar != mask_camera
警告(只记录): 计数低于中位数 5% 的可疑帧 (疑似空场景或坏深度)。
报告写 <out_root>/qc_report.json, 返回 {"errors": n, "warnings": n}。"""
import json
import os

import numpy as np


def _intrinsics_errors(items):
    """items: 迭代 (key, W, H, fx, fy, cx, cy); W/H 为 None 时用宽松上界。"""
    errs = []
    for key, W, H, fx, fy, cx, cy in items:
        w = W or 10 ** 6
        h = H or 10 ** 6
        if not (fx > 0 and fy > 0 and 0 < cx <= w and 0 < cy <= h):
            errs.append(f"内参非法 {key}: fx={fx} fy={fy} cx={cx} cy={cy} W={W} H={H}")
    return errs


def _check_intrinsics(out_root, product):
    if product == "nuscenes":
        p = os.path.join(out_root, "intrinsics_per_frame.json")
        if not os.path.exists(p):
            return ["缺少 intrinsics_per_frame.json"]
        intr = json.load(open(p, encoding="utf-8"))
        return _intrinsics_errors(
            (k, v["W"], v["H"], v["K_native"][0][0], v["K_native"][1][1],
             v["K_native"][0][2], v["K_native"][1][2]) for k, v in intr.items())
    p = os.path.join(out_root, "annotations.json")
    if not os.path.exists(p):
        return ["缺少 annotations.json"]
    ann = json.load(open(p, encoding="utf-8"))
    items = []
    for scene, frames in ann["scene_infos"].items():
        for tok, e in frames.items():
            K = e["camera_sensor"]["CAM_FRONT"]["intrinsics"]
            items.append((f"{scene}/{tok}", None, None,
                          K[0][0], K[1][1], K[0][2], K[1][2]))
    return _intrinsics_errors(items)


def _deep_check(out_root, product, files):
    errs = []
    for rel in files:
        p = os.path.join(out_root, rel)
        if not os.path.exists(p):
            continue  # 缺失已由存在性检查报告
        if product == "nuscenes":
            size = os.path.getsize(p)
            if size == 0 or size % 20 != 0:
                errs.append(f"bin 异常 {rel}: {size} 字节 (应为 float32x5 的倍数)")
        else:
            with np.load(p) as d:
                if sorted(d.files) != ["mask_camera", "mask_lidar", "semantics"]:
                    errs.append(f"npz 键异常 {rel}: {sorted(d.files)}")
                    continue
                sem, ml, mc = d["semantics"], d["mask_lidar"], d["mask_camera"]
                if sem.shape != (200, 200, 16) or sem.dtype != np.uint8 \
                        or ml.dtype != np.uint8 or mc.dtype != np.uint8:
                    errs.append(f"npz 规格/形状异常 {rel}")
                if int(sem.max()) > 17:
                    errs.append(f"semantics 越界 {rel}: max={sem.max()}")
                if not np.array_equal(ml, mc):
                    errs.append(f"mask_lidar != mask_camera {rel}")
    return errs


def run_qc(out_root, product, sample_every=25):
    """执行质检, 写 qc_report.json, 返回 {"errors": n, "warnings": n}。"""
    errors, warnings = [], []
    counts, n_entries = [], 0
    mp = os.path.join(out_root, "manifest.json")
    if not os.path.exists(mp):
        errors.append("缺少 manifest.json (先完成转换)")
    else:
        entries = json.load(open(mp, encoding="utf-8"))["entries"]
        n_entries = len(entries)
        counts = [e["count"] for e in entries.values()]
        for key, e in entries.items():
            p = os.path.join(out_root, e["file"])
            if not os.path.exists(p):
                errors.append(f"文件缺失 {e['file']}")
            elif os.path.getsize(p) != e["bytes"]:
                errors.append(f"大小不符 {e['file']}")
            elif e["count"] == 0:
                errors.append(f"空帧 (count=0) {key}")
        med = float(np.median(counts)) if counts else 0.0
        if med > 0:
            warnings += [f"计数偏低 {key}: {e['count']} (中位数 {med:.0f})"
                         for key, e in entries.items() if e["count"] < 0.05 * med]
        errors += _deep_check(out_root, product,
                              [e["file"] for e in entries.values()][::sample_every])
    errors += _check_intrinsics(out_root, product)

    report = {
        "product": product,
        "frames": n_entries,
        "stats": {"count_median": float(np.median(counts)) if counts else 0,
                  "count_p5": float(np.percentile(counts, 5)) if counts else 0,
                  "count_p95": float(np.percentile(counts, 95)) if counts else 0},
        "errors": errors,
        "warnings": warnings,
    }
    with open(os.path.join(out_root, "qc_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return {"errors": len(errors), "warnings": len(warnings)}
