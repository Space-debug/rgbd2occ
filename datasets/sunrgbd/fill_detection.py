# -*- coding: utf-8 -*-
"""SUN RGB-D 3D 检测标注填充: groundtruth3DBB -> nuScenes sample_annotation/instance 表。

坐标链路 (已双重实证: 工具箱源码 + 椅高比值 1.00):
  框在 Rtilt 重力对齐系 (x右, y前, z上); 我们的自车系 X前/Y左/Z上。
  p_ego = M^T @ Rtilt^T @ p_grav,  M = [[0,-1,0],[1,0,0],[0,0,1]]
尺寸: coeffs = 框沿 basis 三轴的全长; nuScenes size=[w,l,h] (h=basis 第3轴, 重力上)。
深度尺度: /6553.5 (现行管线, 椅高检验中位 1.00; 官方 /8000 位运算解码不适用本分发版)。

产物: 在已有 nuScenes 格式包上原地填充
  <out>/v1.0-sunrgbd-<split>/sample_annotation.json + instance.json
  (attribute 留空; num_lidar_pts 由点云 bin 实际统计)
用法: python main.py sunrgbd detection [--limit N] [--skip-points]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from common.get_logger import get_logger
from common.write_manifest import write_manifest
from nuscenes import token
from .meta import load_meta
from .labels import CLASSES_13
from config import dataset_paths

log = get_logger("rgbd2occ.sunrgbd.detection")

PATHS = dataset_paths("sunrgbd")
# SUN RGB-D 37 类风格 classname -> 13 类 (与语义标签/类别表一致), 其余归 objects
CLASS_MAP = {
    "bed": "bed", "bed_(thin)": "bed", "bed_(thick)": "bed", "pillow": "bed",
    "books": "books", "book": "books", "bookshelf": "books",
    "ceiling": "ceiling", "ceilings": "ceiling",
    "chair": "chair", "chairs": "chair", "stool": "chair", "armchair": "sofa",
    "floor": "floor", "floors": "floor", "mat": "floor", "rug": "floor", "carpet": "floor",
    "furniture": "furniture", "nightstand": "furniture", "dresser": "furniture",
    "wardrobe": "furniture", "cabinet": "furniture", "shelf": "furniture",
    "cupboard": "furniture", "counter": "furniture", "desk": "table", "filecabinet": "furniture",
    "objects": "objects", "lamp": "objects", "plant": "objects", "computer": "objects",
    "printer": "objects", "trashbin": "objects", "box": "objects", "bag": "objects",
    "clothes": "objects", "toy": "objects", "radio": "objects", "phone": "objects",
    "refridgerator": "furniture", "stove": "furniture", "sink": "furniture",
    "bottle": "objects", "cup": "objects", "plate": "objects",
    "picture": "picture", "painting": "picture", "poster": "picture", "mirror": "picture",
    "sofa": "sofa", "couch": "sofa", "loveseat": "sofa",
    "table": "table", "dining_table": "table", "coffee_table": "table",
    "side_table": "table", "kitchen_table": "table", "conference_table": "table",
    "tv": "tv", "tv_monitor": "tv", "monitor": "tv", "keyboard": "objects",
    "wall": "wall", "door": "wall", "column": "wall", "radiator": "furniture",
    "window": "window", "curtain": "window", "blinds": "window",
    "bathtub": "furniture", "toilet": "furniture", "washing_machine": "furniture",
}
M = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])  # ego -> 工具箱(x右,y前,z上)


def _aslist(x):
    return x if isinstance(x, list) else [x]


def rot_to_quat(R):
    """旋转矩阵 -> 四元数 (w, x, y, z), 数值稳定的 Shepperd 法。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def load_boxes(meta_path, splits, out, limit=0, frames=None):
    """SUNRGBDMeta -> {split/num: {"Rtilt": 3x3, "boxes": [...]}} (带缓存)。"""
    cache_path = os.path.join(out, "detection_meta_cache.json")
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    else:
        log.info("解析 %s ...", meta_path)
        meta = load_meta(meta_path)
        cache = {}
        n_meta = len(meta["K"])
        for fi in range(n_meta):
            # 缓存永远全量提取 (train: fi>5049 -> 帧 fi-5049; val: fi<5050 -> 帧 fi+1)
            sp, num = ("train", fi - 5049) if fi > 5049 else ("val", fi + 1)
            if num < 1 or (sp == "val" and num > 5050):
                continue
            try:
                bb = meta["groundtruth3DBB"][fi]
                R = np.array(meta["Rtilt"][fi], np.float64)
                if R.shape != (3, 3):
                    continue
                boxes = []
                bcls = _aslist(bb["classname"])
                for j in range(len(bcls)):
                    try:
                        ctr = np.array(bb["centroid"][j], np.float64).ravel()
                        cf = np.array(bb["coeffs"][j], np.float64).ravel()
                        B = np.array(bb["basis"][j], np.float64)
                    except Exception:
                        continue
                    if ctr.shape != (3,) or cf.shape != (3,) or B.shape != (3, 3):
                        continue
                    boxes.append(dict(cls=str(bcls[j]), centroid=ctr.tolist(),
                                      coeffs=cf.tolist(), basis=B.tolist()))
                cache["%s/%06d" % (sp, num)] = {"Rtilt": R.tolist(), "boxes": boxes}
            except Exception:
                continue
        os.makedirs(out, exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
        log.info("detection meta 缓存 -> %s", cache_path)

    sel = {}
    for k, v in cache.items():
        sp, num = k.split("/")
        if frames and int(num) not in frames:
            continue
        sel[k] = v
    if limit:
        take = {}
        for sp in splits:
            ks = sorted(k for k in sel if k.startswith(sp + "/"))
            take.update({k: sel[k] for k in ks[:limit]})
        sel = take
    return sel


def process(split, frame_key, rec, samples_idx, cats, out, count_points=True):
    """一帧的框 -> sample_annotation/instance 行。"""
    sp, num = split, int(frame_key.split("/")[1])
    sample_tok = samples_idx.get(frame_key)
    if sample_tok is None:
        return [], []
    bin_p = os.path.join(out, "samples", "LIDAR_TOP", sp, "img-%06d.pcd.bin" % num)
    pts = None
    if count_points and os.path.exists(bin_p):
        pts = np.fromfile(bin_p, np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    Rt = M.T @ np.asarray(rec["Rtilt"]).T
    anns, insts = [], []
    for j, b in enumerate(rec["boxes"]):
        name13 = CLASS_MAP.get(b["cls"].lower(), "objects")
        ctr = np.asarray(b["centroid"], np.float64)
        B = np.asarray(b["basis"], np.float64)
        cf = np.asarray(b["coeffs"], np.float64)
        tr = Rt @ ctr
        Re = Rt @ B
        quat = rot_to_quat(Re).tolist()
        size = [float(cf[1]), float(cf[0]), float(cf[2])]   # nuScenes wlh: 长(basis1)宽(basis0)高(basis2)
        npts = 0
        if pts is not None:
            loc = (pts - tr) @ Re
            npts = int(np.all(np.abs(loc) <= cf / 2 + 0.05, 1).sum())
        a_tok = token("ann:%s:%d:%d" % (sp, num, j))
        i_tok = token("inst:%s:%d:%d" % (sp, num, j))
        anns.append({"token": a_tok, "sample_token": sample_tok,
                     "instance_token": i_tok,
                     "attribute_token": "",
                     "translation": tr.tolist(), "size": size, "rotation": quat,
                     "prev": "", "next": "",
                     "num_lidar_pts": npts, "num_radar_pts": 0})
        insts.append({"token": i_tok, "category_token": cats[name13],
                      "nbr_annotations": 1, "first_annotation_token": a_tok,
                      "last_annotation_token": a_tok})
    return anns, insts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="nuScenes 格式包根目录 (默认 config 的 nuscenes_out)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-points", action="store_true", help="跳过 num_lidar_pts 点云统计(更快)")
    args = ap.parse_args(argv)
    out = args.out or PATHS["nuscenes_out"]
    raw_root = PATHS["raw_root"]
    meta_path = os.path.join(raw_root, "SUNRGBDtoolbox", "Metadata", "SUNRGBDMeta.mat")

    frames = load_boxes(meta_path, args.splits, out, args.limit)
    log.info("待填充帧 %d, out=%s", len(frames), out)
    t0 = time.time()
    all_entries = {}
    for sp in args.splits:
        ver_dir = os.path.join(out, "v1.0-sunrgbd-%s" % sp)
        if not os.path.exists(os.path.join(ver_dir, "sample.json")):
            log.error("缺少 %s 的表 (先跑 nuscenes 产品线)", sp)
            return 1
        # sample.json 无帧号 -> 从 sample_data 文件名反查
        sds = json.load(open(os.path.join(ver_dir, "sample_data.json")))
        fn2sample = {}
        for e in sds:
            if "CAM_FRONT" in e["filename"]:
                num = int(e["filename"].split("img-")[1].split(".")[0])
                fn2sample["%s/%06d" % (sp, num)] = e["sample_token"]
        cats = {c["name"].split(".")[0]: c["token"]
                for c in json.load(open(os.path.join(ver_dir, "category.json")))}

        anns, insts, entries = [], [], {}
        n_bad = 0
        for key in sorted(k for k in frames if k.startswith(sp + "/")):
            rec = frames[key]
            a, i = process(sp, key, rec, fn2sample, cats, out, not args.skip_points)
            if not a:
                n_bad += 1
            anns += a
            insts += i
            num = int(key.split("/")[1])
            entries[key] = {"file": "v1.0-sunrgbd-%s/sample_annotation.json" % sp,
                            "count": len(a)}
        json.dump(anns, open(os.path.join(ver_dir, "sample_annotation.json"), "w"), indent=1)
        json.dump(insts, open(os.path.join(ver_dir, "instance.json"), "w"), indent=1)
        all_entries.update(entries)
        log.info("%s: 填充 %d 框 / %d 帧 (未匹配 %d) -> sample_annotation.json/instance.json",
                 sp, len(anns), len(entries), n_bad)

    # 简易质检: 尺寸/平移域 + 空框统计
    errs = []
    for sp in args.splits:
        for e in json.load(open(os.path.join(out, "v1.0-sunrgbd-%s" % sp,
                                             "sample_annotation.json"))):
            if not (0.02 < min(e["size"]) and max(e["size"]) < 8.0):
                errs.append("尺寸异常 %s: %s" % (e["token"][:8], e["size"]))
            if np.linalg.norm(e["translation"]) > 40:
                errs.append("平移越界 %s" % e["token"][:8])
    log.info("质检: %d 条异常", len(errs))
    for e in errs[:5]:
        log.warning("  %s", e)

    write_manifest(out, "detection",
                   params=dict(splits=args.splits, limit=args.limit,
                               frame="gravity(Rtilt)->ego", depth_scale=1.0 / 6553.5,
                               class_map="CLASS_MAP->SUNRGBD-13"),
                   entries=all_entries)
    log.info("完成, 耗时 %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
