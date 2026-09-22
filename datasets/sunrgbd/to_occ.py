# -*- coding: utf-8 -*-
"""SUN RGB-D -> Occ3D-nuScenes 同构占据标注包 (主转换入口)。

两种模式:
  批量 (默认): 由 sunrgbd_nuscenes_v3 的 devkit 表逐帧驱动 ——
      真实内参(calibrated_sensor.json)、场景归属(sample/scene.json)、
      32位hex token、深度图(img-XXXXXX.jpg -> {N}.png, /6553.5, 掩膜[0.3,8]m)。
      多进程、断点续跑(已有 labels.npz 跳过)、annotations 按已完成帧重建。
  单帧 (--mode single): 转一帧深度图, 参数自给 (调试/自定义用)。

输出 (与 Occupancy3D-nuScenes-v1.0-mini 同构):
  <out-root>/annotations.json
  <out-root>/gts/<scene>/<token>/labels.npz
ego_pose/extrinsic 置空(null): v3 中本就是占位值, 不写假的。
容错: 单帧异常不中断整批, 失败帧(含堆栈)写入 <out-root>/failed.json, 重跑自动重试。
路径: 默认取 config.json, CLI --out-root/--raw-root/--v3-root 覆盖。

用法:
  python sunrgbd2occ.py                              # 全量 train+val
  python sunrgbd2occ.py --limit 3                    # 每场景前 3 帧(试跑)
  python sunrgbd2occ.py --splits train --workers 10
  python sunrgbd2occ.py --ann-only                   # 只重建 annotations.json
  python sunrgbd2occ.py --mode single D:/x/depth.png --fx 529.5 --fy 529.5 \
      --cx 365 --cy 265 --depth-scale 0.000152592 --scene s1 --token t1 --out-root out
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from occ import convert_frame, load_depth, load_label, mask_depth, OccAnnotations, pose
from common.get_logger import get_logger
from common.render_bev import render_occ_bev
from common.run_qc import run_qc
from common.write_manifest import write_manifest
from config import dataset_paths
from .labels import label_path, semantic_classes_doc

log = get_logger("rgbd2occ.sunrgbd.occ")

PATHS = dataset_paths("sunrgbd")    # raw_root / nuscenes_out / occ_out
DEPTH_SCALE = 1.0 / 6553.5          # 16bit png -> 米
VALID_RANGE = (0.3, 8.0)            # 与 branch1 清洗管线一致


# ---------------- 批量模式 ----------------

def build_tasks(splits, limit, v3_root=None, raw_root=None, labels=True):
    """从 v3 表构建有序任务列表 (场景内按 timestamp+token 排序, 即帧链顺序)。
    labels=True 时自动探测 train13labels/test13labels 逐帧标签路径。"""
    tasks = []
    for sp in splits:
        tb = os.path.join(v3_root or PATHS["nuscenes_out"], f"v1.0-sunrgbd-{sp}")
        samples = {s["token"]: s for s in json.load(open(os.path.join(tb, "sample.json")))}
        scenes = {s["token"]: s for s in json.load(open(os.path.join(tb, "scene.json")))}
        cs = {c["token"]: c["camera_intrinsic"]
              for c in json.load(open(os.path.join(tb, "calibrated_sensor.json")))}
        cam_frames = [e for e in json.load(open(os.path.join(tb, "sample_data.json")))
                      if "CAM_FRONT" in e["filename"]]
        depth_dir = os.path.join(raw_root or PATHS["raw_root"],
                                 "sunrgbd_train_depth" if sp == "train" else "sunrgbd_test_depth")
        rec = []
        for e in cam_frames:
            s = samples[e["sample_token"]]
            num = int(re.search(r"img-(\d+)\.jpg", e["filename"]).group(1))
            K = cs[e["calibrated_sensor_token"]]
            rec.append(dict(scene=scenes[s["scene_token"]]["name"],
                            token=e["sample_token"], split=sp, ts=s.get("timestamp", 0),
                            depth=os.path.join(depth_dir, f"{num}.png"), K=K,
                            img=e["filename"],
                            label=label_path(raw_root or PATHS["raw_root"], sp, num) if labels else None,
                            fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]))
        rec.sort(key=lambda r: (r["scene"], r["ts"], r["token"]))
        if limit:
            take, kept = {}, []
            for r in rec:
                take[r["scene"]] = take.get(r["scene"], 0) + 1
                if take[r["scene"]] <= limit:
                    kept.append(r)
            rec = kept
        tasks += rec
    return tasks


def convert_one(out_root, task):
    """转一帧; 异常捕获后返回 error 不中断整批。
    返回 (token, 状态, 错误堆栈, 可见体素数)。"""
    tok = task["token"]
    rel = os.path.join("gts", task["scene"], task["token"], "labels.npz")
    try:
        out = os.path.join(out_root, rel)
        if os.path.exists(out):
            with np.load(out) as d:
                return tok, "skip", "", int((d["mask_camera"] > 0).sum())
        dep = mask_depth(load_depth(task["depth"], DEPTH_SCALE), VALID_RANGE)
        label = load_label(task["label"]) if task.get("label") else None
        if label is not None and label.shape != dep.shape:
            raise ValueError(f"标签形状 {label.shape} != 深度 {dep.shape}")
        res = convert_frame(dep, task["fx"], task["fy"], task["cx"], task["cy"], label)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, **res)
        return tok, "ok", "", int((res["mask_camera"] > 0).sum())
    except Exception:
        err = traceback.format_exc()
        get_logger("rgbd2occ.sunrgbd.occ").error("%s 转换失败: %s", tok, err.replace("\n", " | "))
        return tok, "error", err, 0


def write_annotations(tasks, out_root):
    frames = []
    for t in tasks:
        if not os.path.exists(os.path.join(out_root, "gts", t["scene"], t["token"], "labels.npz")):
            continue
        frames.append(dict(scene=t["scene"], token=t["token"], split=t["split"],
                            fx=t["fx"], fy=t["fy"], cx=t["cx"], cy=t["cy"],
                            img=t["img"], ts=t["ts"],
                            gt_rel=f'gts/{t["scene"]}/{t["token"]}/labels.npz'))
    ann = OccAnnotations(os.path.join(out_root, "annotations.json"))
    ann.rebuild(frames)
    ann.save()
    return len(frames)


def save_failed(out_root, fails):
    p = os.path.join(out_root, "failed.json")
    if fails:
        json.dump(fails, open(p, "w", encoding="utf-8"), indent=1)
        log.error("失败 %d 帧, 明细 -> %s; 重跑同一命令将自动重试这些帧", len(fails), p)
    elif os.path.exists(p):
        os.remove(p)
        log.info("无失败帧, 清除旧 failed.json")


def run_batch(args):
    out_root = args.out_root or PATHS["occ_out"]
    use_labels = not args.no_labels
    tasks = build_tasks(args.splits, args.limit, args.v3_root, args.raw_root, use_labels)
    n_lab = sum(1 for t in tasks if t["label"])
    log.info("任务 %d 帧 (带标签 %d), 场景 %s, workers=%d, out=%s", len(tasks), n_lab,
             sorted(set(t["scene"] for t in tasks)), args.workers, out_root)
    if args.ann_only:
        n = write_annotations(tasks, out_root)
        log.info("annotations 重建登记帧 %d", n)
        return 0
    fails, counts, t0, stat = {}, {}, time.time(), {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (tok, st, err, cnt) in enumerate(
                ex.map(partial(convert_one, out_root), tasks, chunksize=8), 1):
            stat[st] = stat.get(st, 0) + 1
            counts[tok] = cnt
            if st == "error":
                fails[tok] = err
            if k % 200 == 0 or k == len(tasks):
                log.info("  %d/%d %s %.1f 帧/s", k, len(tasks), stat,
                         k / (time.time() - t0))
    save_failed(out_root, fails)
    n_ann = write_annotations(tasks, out_root)

    # ---- 数据集自描述侧车 + manifest (溯源) ----
    with open(os.path.join(out_root, "semantic_classes.json"), "w", encoding="utf-8") as f:
        json.dump(semantic_classes_doc(), f, indent=1, ensure_ascii=False)
    entries = {t["token"]: {"file": 'gts/%s/%s/labels.npz' % (t["scene"], t["token"]),
                            "count": counts.get(t["token"], 0)}
               for t in tasks
               if os.path.exists(os.path.join(out_root, "gts", t["scene"], t["token"], "labels.npz"))}
    write_manifest(out_root, "occ",
                   params=dict(splits=args.splits, limit=args.limit,
                               frames=getattr(args, "frames", None),
                               voxel=0.4, ranges=[[-40, 40], [-40, 40], [-1, 5.4]],
                               ray_stride=4, depth_scale=DEPTH_SCALE,
                               valid_range=list(VALID_RANGE), labels=use_labels,
                               label_mapping="pixel==semantic id (SUNRGBD-13)" if use_labels else None),
                   entries=entries)

    rc = 1 if fails else 0
    if not args.skip_qc:
        q = run_qc(out_root, "occ")
        log.info("质检: errors=%d warnings=%d (明细 -> qc_report.json)", q["errors"], q["warnings"])
        rc = rc or (1 if q["errors"] else 0)
    log.info("完成: labels.npz 共 %d, annotations 登记帧 %d, 耗时 %.0fs%s",
             stat.get("ok", 0) + stat.get("skip", 0), n_ann, time.time() - t0,
             " (有失败/质检错误, 退出码 1)" if rc else "")
    return rc




def run_inspect(args):
    """渲染指定/警告帧的 BEV 质检图 -> <out-root>/inspect/<token>.png"""
    import re as _re
    out_root = args.out_root or PATHS["occ_out"]
    targets = [args.inspect] if args.inspect else []
    if args.inspect_warned:
        qp = os.path.join(out_root, "qc_report.json")
        if not os.path.exists(qp):
            log.error("缺少 qc_report.json (先跑一次批量转换)")
            return 1
        for w in json.load(open(qp, encoding="utf-8"))["warnings"]:
            m = _re.search(r"([0-9a-f]{32})", w)
            if m:
                targets.append(m.group(1))
    ann = json.load(open(os.path.join(out_root, "annotations.json"), encoding="utf-8"))
    idx = {t: (s, e) for s, fr in ann["scene_infos"].items() for t, e in fr.items()}
    os.makedirs(os.path.join(out_root, "inspect"), exist_ok=True)
    n = 0
    for tok in targets:
        if tok not in idx:
            log.warning("token 未登记: %s", tok)
            continue
        s, e = idx[tok]
        p = os.path.join(out_root, e["gt_path"])
        with np.load(p) as d:
            vis = int((d["mask_camera"] > 0).sum())
            render_occ_bev(d["semantics"], d["mask_camera"],
                           os.path.join(out_root, "inspect", tok + ".png"),
                           title="%s/%s vis=%d" % (s, tok[:8], vis))
        n += 1
    log.info("inspect: %d 张 -> %s", n, os.path.join(out_root, "inspect"))
    return 0

# ---------------- 单帧模式 ----------------

def run_single(args):
    label = load_label(args.label) if args.label else None
    depth = load_depth(args.depth, args.depth_scale)
    if label is not None and label.shape != depth.shape:
        raise SystemExit(f"标签图形状 {label.shape} 与深度图 {depth.shape} 不一致")
    depth = mask_depth(depth, args.valid_range)
    res = convert_frame(depth, args.fx, args.fy, args.cx, args.cy, label,
                        args.voxel, args.xrange, args.yrange, args.zrange,
                        args.ray_stride, args.k1, args.k2)
    out = os.path.join(args.out_root, args.scene, args.token, "labels.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **res)
    root = os.path.basename(os.path.normpath(args.out_root))
    ann = OccAnnotations(args.annotations or os.path.normpath(
        os.path.join(args.out_root, "..", "annotations.json")))
    ann.add(args.scene, args.token, args.split, args.fx, args.fy, args.cx, args.cy,
            img=args.img or "", ts=args.timestamp,
            ego_pose=pose(args.ego_translation, args.ego_rotation),
            extrinsic=pose(args.extrinsic_translation, args.extrinsic_rotation),
            gt_rel=f"{root}/{args.scene}/{args.token}/labels.npz")
    ann.save()
    sem, mc = res["semantics"], res["mask_camera"]
    occ = int(((sem < 17) & (mc == 1)).sum())
    log.info("网格 %s 占据=%d free=%d 未知=%d -> %s", sem.shape, occ,
             int(((sem == 17) & (mc == 1)).sum()), int((mc == 0).sum()), out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["batch", "single"], default="batch")
    ap.add_argument("--out-root", default=None, help="输出根目录 (默认 config.json 的 occ_out)")
    ap.add_argument("--raw-root", default=None, help="SUN RGB-D 原始数据根目录 (默认 config)")
    ap.add_argument("--v3-root", default=None, help="nuScenes 格式数据根目录 (默认 config)")
    # 批量参数
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="每场景前 N 帧(试跑), 0=全量")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ann-only", action="store_true", help="只重建 annotations.json")
    ap.add_argument("--no-labels", action="store_true",
                    help="不接入 13 类语义标签 (占据记 others=0)")
    ap.add_argument("--skip-qc", action="store_true", help="跳过自动质检")
    ap.add_argument("--inspect", default=None, help="渲染指定 token 的 BEV 质检图后退出")
    ap.add_argument("--inspect-warned", action="store_true",
                    help="渲染 qc_report 中所有警告帧的 BEV 质检图后退出")
    # 单帧参数
    ap.add_argument("depth", nargs="?", help="[single] 深度图路径")
    ap.add_argument("--fx", type=float), ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float), ap.add_argument("--cy", type=float)
    ap.add_argument("--label", help="[single] 语义标签图 (像素值=类别id 0..16)")
    ap.add_argument("--scene", help="[single] 场景名"), ap.add_argument("--token", help="[single] 帧token")
    ap.add_argument("--img", help="[single] 登记用图像路径")
    ap.add_argument("--annotations", help="[single] annotations.json 路径 (默认 out-root 上一级)")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--timestamp", type=int, default=0)
    ap.add_argument("--ego-translation", type=float, nargs=3, default=None)
    ap.add_argument("--ego-rotation", type=float, nargs=4, default=None)
    ap.add_argument("--extrinsic-translation", type=float, nargs=3, default=None)
    ap.add_argument("--extrinsic-rotation", type=float, nargs=4, default=None)
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--xrange", type=float, nargs=2, default=[-40, 40])
    ap.add_argument("--yrange", type=float, nargs=2, default=[-40, 40])
    ap.add_argument("--zrange", type=float, nargs=2, default=[-1, 5.4])
    ap.add_argument("--depth-scale", type=float, default=0.001,
                    help="[single] png 深度比例(米/单位)")
    ap.add_argument("--valid-range", type=float, nargs=2, default=None,
                    help="[single] 深度有效区间, 如 0.3 8; 默认不过滤")
    ap.add_argument("--ray-stride", type=int, default=4)
    ap.add_argument("--k1", type=float, default=0.0)
    ap.add_argument("--k2", type=float, default=0.0)
    args = ap.parse_args(argv)

    if getattr(args, "inspect", None) or getattr(args, "inspect_warned", False):
        sys.exit(run_inspect(args))

    if args.mode == "single":
        need = ["depth", "fx", "fy", "cx", "cy", "scene", "token"]
        miss = [k for k in need if getattr(args, k) is None]
        if miss:
            ap.error(f"single 模式缺少参数: {', '.join('--' + k.replace('_', '-') for k in miss)}")
        args.out_root = args.out_root or "gts"
        run_single(args)
    else:
        sys.exit(run_batch(args))


if __name__ == "__main__":
    main()
