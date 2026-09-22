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
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from occ import convert_frame, load_depth, load_label, mask_depth, OccAnnotations, pose

V3_ROOT = r"D:\Datasets\sunrgbd_nuscenes_v3"
SUNRGBD_ROOT = r"D:\Datasets\sunrgbd"
OUT_ROOT = r"D:\Datasets\Occupancy3D-SUNRGBD"
DEPTH_SCALE = 1.0 / 6553.5          # 16bit png -> 米
VALID_RANGE = (0.3, 8.0)            # 与 branch1 清洗管线一致


# ---------------- 批量模式 ----------------

def build_tasks(splits, limit):
    """从 v3 表构建有序任务列表 (场景内按 timestamp+token 排序, 即帧链顺序)。"""
    tasks = []
    for sp in splits:
        tb = os.path.join(V3_ROOT, f"v1.0-sunrgbd-{sp}")
        samples = {s["token"]: s for s in json.load(open(os.path.join(tb, "sample.json")))}
        scenes = {s["token"]: s for s in json.load(open(os.path.join(tb, "scene.json")))}
        cs = {c["token"]: c["camera_intrinsic"]
              for c in json.load(open(os.path.join(tb, "calibrated_sensor.json")))}
        cam_frames = [e for e in json.load(open(os.path.join(tb, "sample_data.json")))
                      if "CAM_FRONT" in e["filename"]]
        depth_dir = os.path.join(SUNRGBD_ROOT,
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
    out = os.path.join(out_root, "gts", task["scene"], task["token"], "labels.npz")
    if os.path.exists(out):
        return "skip"
    dep = mask_depth(load_depth(task["depth"], DEPTH_SCALE), VALID_RANGE)
    res = convert_frame(dep, task["fx"], task["fy"], task["cx"], task["cy"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    return "ok"


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


def run_batch(args):
    tasks = build_tasks(args.splits, args.limit)
    print(f"任务: {len(tasks)} 帧, 场景 {sorted(set(t['scene'] for t in tasks))}, "
          f"workers={args.workers}, out={args.out_root}")
    if args.ann_only:
        n = write_annotations(tasks, args.out_root)
        print(f"annotations 重建登记帧 {n}")
        return
    t0, done, skip = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for st in ex.map(partial(convert_one, args.out_root), tasks, chunksize=8):
            done += st == "ok"
            skip += st == "skip"
            n = done + skip
            if n % 200 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} (新转 {done}, 跳过 {skip}) "
                      f"{n / (time.time() - t0):.1f} 帧/s", flush=True)
    n_ann = write_annotations(tasks, args.out_root)
    print(f"完成: labels.npz 共 {done + skip}, annotations 登记帧 {n_ann}, "
          f"耗时 {time.time() - t0:.0f}s")


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
    print(f"网格 {sem.shape} 占据={occ} free={int(((sem == 17) & (mc == 1)).sum())} "
          f"未知={int((mc == 0).sum())} -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["batch", "single"], default="batch")
    ap.add_argument("--out-root", default=OUT_ROOT)
    # 批量参数
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="每场景前 N 帧(试跑), 0=全量")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ann-only", action="store_true", help="只重建 annotations.json")
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

    if args.mode == "single":
        need = ["depth", "fx", "fy", "cx", "cy", "scene", "token"]
        miss = [k for k in need if getattr(args, k) is None]
        if miss:
            ap.error(f"single 模式缺少参数: {', '.join('--' + k.replace('_', '-') for k in miss)}")
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
