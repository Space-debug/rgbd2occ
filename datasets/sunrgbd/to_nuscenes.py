# -*- coding: utf-8 -*-
"""SUN RGB-D -> nuScenes devkit 格式 (主转换入口: 点云 + 图像 + 13 张表)。

合并原 v2/v3 两步管线 (rebuild_sunrgbd_nuscenes.py + build_v3.py), 输出等价:
- 点云: 每帧用 SUNRGBDMeta.mat 真实 K 在原始分辨率反投影,
  中值/梯度 -> SOR -> 体素斑点滤波 -> 30mm 体素降采样,
  float32 Nx5 (x,y,z,intensity=亮度/255,ring=0) -> samples/LIDAR_TOP/<split>/
- 图像: 原始分辨率 jpg 复制 -> samples/CAM_FRONT/<split>/
- 表: v3 风格 —— 按相机型号(kv1/kv2/realsense/xtion)分场景,
  calibrated_sensor 按唯一 K_native 建条目(相机->自车轴置换四元数)
- 逐帧内参 JSON: intrinsics_per_frame.json (v2 同格式)
映射: train 第 i 帧 (img-%06d, i 从 1) -> meta[5049+i]; val 第 i 帧 -> meta[i-1]

用法:
  python sunrgbd2nuscenes.py                        # 全量 train+val (断点续跑)
  python sunrgbd2nuscenes.py --limit 3              # 每 split 前 3 帧试跑
  python sunrgbd2nuscenes.py --frames 1,1925        # 指定帧号(各 split 都取)
  python sunrgbd2nuscenes.py --tables-only          # 只重建表(需已有缓存)
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from common.io import load_depth
from common.pointcloud import depth_to_points, voxel_downsample, write_nuscenes_bin
from nuscenes import token, write_tables
from .meta import load_meta

BASE = r"D:\Datasets\sunrgbd"
OUT_DEFAULT = r"D:\Datasets\sunrgbd_nuscenes_v3"
DEPTH_SCALE = 1.0 / 6553.5
DOWNSAMPLE_VOX = 0.03
TS_BASE = 1532402927647951
RES2SENSOR = {(730, 530): "kv2", (591, 441): "xtion",
              (561, 427): "kv1", (681, 531): "realsense"}
# 相机系(x右,y下,z前) -> 自车系(x前,y左,z上) 的轴置换四元数 (w,x,y,z)
CAM_AXIS_QUAT = [0.5, -0.5, 0.5, -0.5]
CLASSES_13 = ["bed", "books", "ceiling", "chair", "floor", "furniture",
              "objects", "picture", "sofa", "table", "tv", "wall", "window"]
SPLITS = {
    "train": {"imgs": "SUNRGBD-train_images", "depth": "sunrgbd_train_depth", "n": 5285},
    "val":   {"imgs": "SUNRGBD-test_images",  "depth": "sunrgbd_test_depth",  "n": 5050},
}
META_PATH = os.path.join(BASE, "SUNRGBDtoolbox", "Metadata", "SUNRGBDMeta.mat")


def prepare_frames(splits, limit, frames, out):
    """解析 meta(带缓存) -> 本次的帧记录列表。记录含真实 K 与图像尺寸。"""
    cache_path = os.path.join(out, "sunrgbd_meta_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    if not cache:
        print("解析 SUNRGBDMeta.mat ...")
        meta = load_meta(META_PATH)
        for split in SPLITS:
            for i in range(1, SPLITS[split]["n"] + 1):
                mi = 5049 + i if split == "train" else i - 1
                cache["%s/%06d" % (split, i)] = {
                    "meta_idx": mi,
                    "K_native": np.array(meta["K"][mi], np.float64).tolist(),
                }
        os.makedirs(out, exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
        print("  meta 缓存 ->", cache_path)

    recs = []
    for split in splits:
        ids = range(1, SPLITS[split]["n"] + 1)
        if frames:
            ids = [i for i in ids if i in frames]
        elif limit:
            ids = ids[:limit]
        for i in ids:
            name = "img-%06d" % i
            img_src = os.path.join(BASE, SPLITS[split]["imgs"], name + ".jpg")
            dep_src = os.path.join(BASE, SPLITS[split]["depth"], "%d.png" % i)
            if not (os.path.exists(img_src) and os.path.exists(dep_src)):
                continue
            W, H = Image.open(img_src).size
            K = cache["%s/%06d" % (split, i)]["K_native"]
            K640 = np.array(K, np.float64)
            K640[0] *= 640.0 / W
            K640[1] *= 480.0 / H
            recs.append(dict(split=split, i=i, name=name, W=W, H=H,
                             K_native=K, K_640=[[round(v, 3) for v in r] for r in K640.tolist()],
                             img=img_src, dep=dep_src))
    return recs


def process_frame(job):
    """生成一帧的 pcd.bin 与 jpg (已存在则跳过)。job 为 prepare_frames 的帧记录 + out。"""
    r, out = job
    bin_dst = os.path.join(out, "samples", "LIDAR_TOP", r["split"], r["name"] + ".pcd.bin")
    img_dst = os.path.join(out, "samples", "CAM_FRONT", r["split"], r["name"] + ".jpg")
    if not os.path.exists(img_dst):
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        shutil.copy2(r["img"], img_dst)
    if os.path.exists(bin_dst):
        return "skip"
    dep = load_depth(r["dep"], DEPTH_SCALE)
    img = np.array(Image.open(r["img"]).convert("RGB"))
    if dep.shape != img.shape[:2]:   # 深度网格为准, RGB 重采样对齐
        img = np.array(Image.open(r["img"]).convert("RGB")
                       .resize((dep.shape[1], dep.shape[0])))
    K = r["K_native"]
    P, C = depth_to_points(img, dep, K[0][0], K[0][2], K[1][2])
    if DOWNSAMPLE_VOX and len(P):
        P, C = voxel_downsample(P.astype(np.float64), C.astype(np.float64), DOWNSAMPLE_VOX)
    os.makedirs(os.path.dirname(bin_dst), exist_ok=True)
    write_nuscenes_bin(bin_dst, P, C)
    return "ok"


def build_tables(recs, split, out):
    """v3 表构建 (逐字段移植): 按相机型号分场景, 唯一 K_native 建标定条目。"""
    by_sensor = {}
    for r in recs:
        if r["split"] != split:
            continue
        if not os.path.exists(os.path.join(out, "samples", "LIDAR_TOP", split,
                                           r["name"] + ".pcd.bin")):
            continue
        by_sensor.setdefault(RES2SENSOR[(r["W"], r["H"])], []).append(r)
    for k in by_sensor:
        by_sensor[k].sort(key=lambda r: r["name"])

    sensor = [{"token": token("sensor:CAM_FRONT"), "channel": "CAM_FRONT", "modality": "camera"},
              {"token": token("sensor:LIDAR_TOP"), "channel": "LIDAR_TOP", "modality": "lidar"}]
    cam_sensor_tok = sensor[0]["token"]

    calib, calib_lookup = [], {}
    logs, scenes, samples, sds, eps = [], [], [], [], []
    sample_by_tok = {}
    for sens in sorted(by_sensor):
        lst = by_sensor[sens]
        log_tok = token("log:%s:%s" % (split, sens))
        logs.append({"token": log_tok, "logfile": "sunrgbd-%s-%s" % (split, sens),
                     "vehicle": sens, "date_captured": "2014-01-01", "location": "indoor"})
        prev_t, scene_samples = "", []
        for r in lst:
            st = TS_BASE + r["i"] * 33000
            t = token("sample:%s:%s" % (split, r["name"]))
            scene_samples.append(t)
            row = {"token": t, "timestamp": st, "prev": prev_t, "next": "", "scene_token": None}
            samples.append(row)
            sample_by_tok[t] = row
            if prev_t:
                sample_by_tok[prev_t]["next"] = t
            prev_t = t

            kk = tuple(round(v, 4) for row2 in r["K_native"] for v in row2)
            if kk not in calib_lookup:
                ct = token("calib:native:%s" % (kk,))
                calib_lookup[kk] = ct
                calib.append({"token": ct, "sensor_token": cam_sensor_tok,
                              "camera_intrinsic": r["K_native"],
                              "rotation": CAM_AXIS_QUAT,
                              "translation": [0.0, 0.0, 0.0]})
            cam_tok = token("sd:%s:%s:cam" % (split, r["name"]))
            sds.append({"token": cam_tok, "sample_token": t, "ego_pose_token": cam_tok,
                        "calibrated_sensor_token": calib_lookup[kk],
                        "timestamp": st, "fileformat": "jpg", "is_key_frame": True,
                        "height": r["H"], "width": r["W"],
                        "filename": "samples/CAM_FRONT/%s/%s.jpg" % (split, r["name"]),
                        "prev": "", "next": "", "sensor_modality": "camera"})
            eps.append({"token": cam_tok, "timestamp": st,
                        "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]})
            lid_tok = token("sd:%s:%s:lid" % (split, r["name"]))
            sds.append({"token": lid_tok, "sample_token": t, "ego_pose_token": lid_tok,
                        "calibrated_sensor_token": token("calib:LIDAR_TOP"),
                        "timestamp": st, "fileformat": "pcd.bin", "is_key_frame": True,
                        "height": 0, "width": 0,
                        "filename": "samples/LIDAR_TOP/%s/%s.pcd.bin" % (split, r["name"]),
                        "prev": "", "next": "", "sensor_modality": "lidar"})
            eps.append({"token": lid_tok, "timestamp": st,
                        "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]})

        scene_t = token("scene:%s:%s" % (split, sens))
        for t in scene_samples:
            sample_by_tok[t]["scene_token"] = scene_t
        scenes.append({"token": scene_t, "log_token": log_tok,
                       "nbr_samples": len(scene_samples),
                       "first_sample_token": scene_samples[0],
                       "last_sample_token": scene_samples[-1],
                       "name": "sunrgbd-%s-%s" % (split, sens),
                       "description": "SUN RGB-D %s (%s)" % (split, sens)})
    calib.append({"token": token("calib:LIDAR_TOP"), "sensor_token": token("sensor:LIDAR_TOP"),
                  "camera_intrinsic": [], "rotation": [1.0, 0.0, 0.0, 0.0],
                  "translation": [0.0, 0.0, 0.0]})

    tables = {
        "scene": scenes, "sample": samples, "sample_data": sds, "ego_pose": eps,
        "log": logs, "sensor": sensor, "calibrated_sensor": calib,
        "category": [{"token": token("cat/%s" % c), "name": "%s.indoor" % c,
                      "description": c} for c in CLASSES_13],
        "attribute": [], "instance": [], "sample_annotation": [],
        "map": [{"token": token("map/train"), "category": "indoor", "filename": "",
                 "log_tokens": [l["token"] for l in logs]}],
        "visibility": [{"token": token("vis/%d" % v), "level": "v%d" % v,
                        "description": "stub"} for v in range(1, 5)],
    }
    write_tables(os.path.join(out, "v1.0-sunrgbd-%s" % split), tables)
    return scenes, samples, sds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="每 split 前 N 帧(试跑)")
    ap.add_argument("--frames", default="", help="指定帧号(逗号分隔, 各 split 都取), 如 1,1925")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tables-only", action="store_true", help="只重建表(用已有 meta 缓存)")
    args = ap.parse_args(argv)
    frames = set(int(x) for x in args.frames.split(",") if x.strip()) if args.frames else None

    recs = prepare_frames(args.splits, args.limit, frames, args.out)
    print(f"帧记录: {len(recs)} (splits={args.splits}, out={args.out})")

    if not args.tables_only:
        t0, stat = time.time(), {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for k, st in enumerate(ex.map(process_frame, [(r, args.out) for r in recs],
                                          chunksize=4), 1):
                stat[st] = stat.get(st, 0) + 1
                if k % 200 == 0 or k == len(recs):
                    print(f"  {k}/{len(recs)} {stat} {k/(time.time()-t0):.1f} 帧/s", flush=True)

    for split in args.splits:
        scenes, samples, sds = build_tables(recs, split, args.out)
        print(f"  {split} 表: scene={len(scenes)} sample={len(samples)} sample_data={len(sds)}")

    intr = {"%s/%s" % (r["split"], r["name"]): {"W": r["W"], "H": r["H"],
                                                "K_native": r["K_native"], "K_640": r["K_640"]}
            for r in recs}
    json.dump(intr, open(os.path.join(args.out, "intrinsics_per_frame.json"), "w"))
    print("完成 ->", args.out)


if __name__ == "__main__":
    main()
