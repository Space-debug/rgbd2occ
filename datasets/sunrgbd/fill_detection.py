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
QC: 尺寸/平移域检查 + 每框 3D->2D 投影与 gtBb2D 的 IoU 一致性统计 (进日志与 manifest)。
用法:
  python main.py sunrgbd detection [--limit N] [--skip-points] [--min-pts N]
  --min-pts N: 只保留点云内点数 >= N 的框 (清洗模式), 统计量进 manifest
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

from common.get_logger import attach_file, get_logger
from common.load_depth import load_depth
from common.load_label import load_label
from common.median_gradient import median_gradient
from common.deproject_filtered import deproject_filtered
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


def box_proj_iou(tr, Re, hf, K, bb2d, W, H):
    """3D 框角点投影到像素 vs gtBb2D [x y w h] 的 IoU。
    tr/Re 为自车系; 自车系 -> 工具箱相机系 (x右,y前,z上) 只需轴变换 M。"""
    cs = np.array([tr + Re @ np.array([dx, dy, dz]) * hf
                   for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    pc = cs @ M.T
    fwd = pc[:, 1]
    if (fwd <= 0.05).any():
        return None
    u = K[0, 0] * pc[:, 0] / fwd + K[0, 2]
    v = -K[1, 1] * pc[:, 2] / fwd + K[1, 2]
    x1, y1 = max(u.min(), 0), max(v.min(), 0)
    x2, y2 = min(u.max(), W), min(v.max(), H)
    gx, gy, gw, gh = bb2d
    ix1, iy1 = max(x1, gx), max(y1, gy)
    ix2, iy2 = min(x2, gx + gw), min(y2, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (x2 - x1) * (y2 - y1) + gw * gh - inter
    return inter / union if union > 0 else None


def load_boxes(meta_path, splits, out, limit=0, frames=None):
    """SUNRGBDMeta -> {split/num: {"Rtilt", "boxes"[, "bb2d"]}} (带缓存, 含 _stats 统计)。"""
    cache_path = os.path.join(out, "detection_meta_cache.json")
    stats = None
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        stats = cache.pop("_stats", None)
    else:
        log.info("解析 %s ...", meta_path)
        meta = load_meta(meta_path)
        cache = {}
        stats = {"frames": 0, "frames_dropped": 0, "boxes_kept": 0,
                 "boxes_dropped": 0, "boxes_no2d": 0}
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
                    stats["frames_dropped"] += 1
                    continue
                boxes = []
                bcls = _aslist(bb["classname"])
                b2dl = _aslist(bb["gtBb2D"])
                for j in range(len(bcls)):
                    try:
                        ctr = np.array(bb["centroid"][j], np.float64).ravel()
                        cf = np.array(bb["coeffs"][j], np.float64).ravel()
                        B = np.array(bb["basis"][j], np.float64)
                    except Exception:
                        continue
                    if ctr.shape != (3,) or cf.shape != (3,) or B.shape != (3, 3):
                        stats["boxes_dropped"] += 1
                        continue
                    bb2d = None
                    if j < len(b2dl):
                        g = np.array(b2dl[j]).ravel()
                        if g.shape == (4,) and g[2] > 0 and g[3] > 0:
                            bb2d = [float(v) for v in g]
                    if bb2d is None:
                        stats["boxes_no2d"] += 1
                    boxes.append(dict(cls=str(bcls[j]), centroid=ctr.tolist(),
                                      coeffs=cf.tolist(), basis=B.tolist(), bb2d=bb2d))
                stats["frames"] += 1
                stats["boxes_kept"] += len(boxes)
                cache["%s/%06d" % (sp, num)] = {"Rtilt": R.tolist(), "boxes": boxes}
            except Exception:
                stats["frames_dropped"] += 1
                continue
        cache["_stats"] = stats
        os.makedirs(out, exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
        log.info("detection meta 缓存 -> %s (帧 %d 丢帧 %d, 框保留 %d 丢弃 %d 无2D %d)",
                 cache_path, stats["frames"], stats["frames_dropped"],
                 stats["boxes_kept"], stats["boxes_dropped"], stats["boxes_no2d"])

    sel = {}
    for k, v in cache.items():
        if "/" not in k:              # 跳过 _stats 等元数据键
            continue
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
    return sel, stats


def process(split, frame_key, rec, samples_idx, cats, out, count_points=True, K=None, WH=None):
    """一帧的框 -> sample_annotation/instance 行 + 各框 2D 投影 IoU 列表。"""
    global _cur_Rtilt
    sp, num = split, int(frame_key.split("/")[1])
    sample_tok = samples_idx.get(frame_key)
    if sample_tok is None:
        return [], [], []
    bin_p = os.path.join(out, "samples", "LIDAR_TOP", sp, "img-%06d.pcd.bin" % num)
    pts = None
    if count_points and os.path.exists(bin_p):
        pts = np.fromfile(bin_p, np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    Rt = M.T @ np.asarray(rec["Rtilt"], np.float64).T
    anns, insts, ious = [], [], []
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
        if K is not None and WH is not None and b.get("bb2d"):
            ious.append(box_proj_iou(tr, Re, cf / 2, K, b["bb2d"], WH[0], WH[1]))
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
    return anns, insts, ious






def _gpu_ok():
    """gpu 层可用且 torch CUDA 在位 (label QC 的 GPU 快路径)。"""
    import os
    if os.environ.get("RGBD2OCC_BACKEND") != "gpu":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


LABEL_DIRS = {"train": "train13labels", "val": "test13labels"}
DEPTH_DIRS = {"train": "sunrgbd_train_depth", "val": "sunrgbd_test_depth"}


def label_consistency_qc(out, frames, splits, raw_root, sample_every, depth_scale=1.0 / 6553.5):
    """抽帧验证 3D 框与 13 类语义标签的一致率 (轴对齐近似, 保守值)。
    返回 stats dict; 抽检走 load_depth/median_gradient/deproject_filtered 全链,
    同时持续回归验证坐标链路。"""
    per_split = {}
    for sp in splits:
        ver_dir = os.path.join(out, "v1.0-sunrgbd-%s" % sp)
        sds = json.load(open(os.path.join(ver_dir, "sample_data.json"), encoding="utf-8"))
        cs = {c["token"]: c["camera_intrinsic"]
              for c in json.load(open(os.path.join(ver_dir, "calibrated_sensor.json"), encoding="utf-8"))}
        for e in sds:
            if "CAM_FRONT" in e.get("filename", ""):
                num = int(re.search(r"img-(\d+)", e["filename"]).group(1))
                per_split.setdefault(sp, {})[num] = cs[e["calibrated_sensor_token"]]
    rates = []
    for key in sorted(k for k in frames if int(k.split("/")[1]) % max(sample_every, 1) == 0):
        sp, num = key.split("/")
        num = int(num)
        rec = frames[key]
        if not rec["boxes"] or sp not in LABEL_DIRS:
            continue
        try:
            K = per_split.get(sp, {}).get(num)
            if K is None:
                continue
            lab = load_label(os.path.join(raw_root, LABEL_DIRS[sp], "img13labels-%06d.png" % num))
            if _gpu_ok():
                # gpu 层: 标签图作为"颜色"传给 GPU 全链 (median->梯度->反投影->SOR->斑点),
                # 每点的标签自动带出, 与生成 occ 的掩膜/滤波严格同源
                from common.gpu_points import _points_gpu_tensors
                from PIL import Image as _Im
                raw = np.array(_Im.open(os.path.join(raw_root, DEPTH_DIRS[sp], "%d.png" % num)))
                img_lab = np.stack([lab] * 3, axis=2)
                P_t, C_t = _points_gpu_tensors(raw, img_lab, depth_scale,
                                               K[0][0], K[1][1], K[0][2], K[1][2])
                P = P_t.cpu().numpy()
                C = C_t[:, 0].cpu().numpy()
            else:
                dep = load_depth(os.path.join(raw_root, DEPTH_DIRS[sp], "%d.png" % num), depth_scale)
                d1 = median_gradient(dep, (dep > 0.3) & (dep < 8))
                P, m = deproject_filtered(d1, K[0][0], K[0][2], K[1][2])
                C = lab[m]
        except Exception:
            continue
        Rt = M.T @ np.asarray(rec["Rtilt"], np.float64).T
        for b in rec["boxes"]:
            name13 = CLASS_MAP.get(b["cls"].lower(), "objects")
            if name13 not in CLASSES_13:
                continue
            cc = CLASSES_13.index(name13) + 1
            tr = Rt @ np.asarray(b["centroid"], np.float64)
            Re = Rt @ np.asarray(b["basis"], np.float64)
            hf = np.asarray(b["coeffs"], np.float64) / 2 + 0.1
            inside = np.all(np.abs((P - tr) @ Re) <= hf, 1)
            if inside.sum() < 5:
                continue
            rates.append(float(np.mean(C[inside] == cc)))
    if not rates:
        return {}
    r = np.array(rates)
    return {"frames_sampled": len(rates), "mean": round(float(r.mean()), 3),
            "median": round(float(np.median(r)), 3), "gt05": round(float(np.mean(r > 0.5)), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="nuScenes 格式包根目录 (默认 config 的 nuscenes_out)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-points", action="store_true", help="跳过 num_lidar_pts 点云统计(更快)")
    ap.add_argument("--min-pts", type=int, default=0,
                    help="只保留点云内点数 >= N 的框 (清洗模式, 0=全保留)")
    ap.add_argument("--label-qc-sample", type=int, default=50,
                    help="标签一致性抽检间隔帧 (0=关闭)")
    args = ap.parse_args(argv)
    out = args.out or PATHS["nuscenes_out"]
    raw_root = PATHS["raw_root"]
    meta_path = os.path.join(raw_root, "SUNRGBDtoolbox", "Metadata", "SUNRGBDMeta.mat")
    os.makedirs(out, exist_ok=True)
    attach_file(log, os.path.join(out, "logs", time.strftime("det_%Y%m%d_%H%M%S.log")))

    frames, cache_stats = load_boxes(meta_path, args.splits, out, args.limit)
    log.info("待填充帧 %d, out=%s, min_pts=%d", len(frames), out, args.min_pts)
    t0 = time.time()
    all_entries, all_ious, dropped_by_pts = {}, [], 0
    for sp in args.splits:
        ver_dir = os.path.join(out, "v1.0-sunrgbd-%s" % sp)
        if not os.path.exists(os.path.join(ver_dir, "sample.json")):
            log.error("缺少 %s 的表 (先跑 nuscenes 产品线)", sp)
            return 1
        # sample.json 无帧号 -> 从 sample_data 文件名反查, 并取该帧 K 与图像尺寸
        sds = json.load(open(os.path.join(ver_dir, "sample_data.json")))
        cs = {c["token"]: c for c in json.load(open(os.path.join(ver_dir, "calibrated_sensor.json")))}
        fn2sample, fn2K = {}, {}
        for e in sds:
            if "CAM_FRONT" in e["filename"]:
                num = int(e["filename"].split("img-")[1].split(".")[0])
                key = "%s/%06d" % (sp, num)
                fn2sample[key] = e["sample_token"]
                fn2K[key] = (np.array(cs[e["calibrated_sensor_token"]]["camera_intrinsic"]),
                             (e["width"], e["height"]))
        cats = {c["name"].split(".")[0]: c["token"]
                for c in json.load(open(os.path.join(ver_dir, "category.json")))}

        anns, insts, entries = [], [], {}
        n_bad = 0
        for key in sorted(k for k in frames if k.startswith(sp + "/")):
            rec = frames[key]
            K, WH = fn2K.get(key, (None, None))
            a, i, ious = process(sp, key, rec, fn2sample, cats, out,
                                 not args.skip_points, K, WH)
            if not a:
                n_bad += 1
            # --min-pts 清洗: 按框过滤 (ann 与对应 instance 同步删)
            if args.min_pts:
                keep = [j for j, x in enumerate(a) if x["num_lidar_pts"] >= args.min_pts]
                dropped_by_pts += len(a) - len(keep)
                a = [a[j] for j in keep]
                keep_set = {x["token"] for x in a}
                i = [x for x in i if x["first_annotation_token"] in keep_set]
            anns += a
            insts += i
            all_ious += [v for v in ious if v is not None]
            entries[key] = {"file": "v1.0-sunrgbd-%s/sample_annotation.json" % sp,
                            "count": len(a)}
        json.dump(anns, open(os.path.join(ver_dir, "sample_annotation.json"), "w"), indent=1)
        json.dump(insts, open(os.path.join(ver_dir, "instance.json"), "w"), indent=1)
        all_entries.update(entries)
        log.info("%s: 填充 %d 框 / %d 帧 (未匹配 %d)", sp, len(anns), len(entries), n_bad)

    # ---- 质检: 尺寸/平移域 + 2D 投影一致性统计 ----
    errs = []
    for sp in args.splits:
        for e in json.load(open(os.path.join(out, "v1.0-sunrgbd-%s" % sp,
                                             "sample_annotation.json"))):
            if not (0.02 < min(e["size"]) and max(e["size"]) < 8.0):
                errs.append("尺寸异常 %s: %s" % (e["token"][:8], e["size"]))
            if np.linalg.norm(e["translation"]) > 40:
                errs.append("平移越界 %s" % e["token"][:8])
    iou_stats = {}
    if all_ious:
        iou = np.array(all_ious)
        iou_stats = {"n": len(iou), "median": round(float(np.median(iou)), 3),
                     "mean": round(float(iou.mean()), 3), "gt05": round(float(np.mean(iou > 0.5)), 3)}
        log.info("2D 投影一致性: IoU 中位 %.2f 均值 %.2f >0.5 占比 %.0f%% (n=%d)",
                 iou_stats["median"], iou_stats["mean"], 100 * iou_stats["gt05"], len(iou))
    log.info("质检: %d 条尺寸/平移异常; min_pts 清洗丢弃 %d 框", len(errs), dropped_by_pts)

    # 类别映射侧车 (occ semantic_classes.json 的检测侧对应物)
    with open(os.path.join(out, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "SUNRGBD 37-style classname", "target": "SUNRGBD-13",
                   "map": CLASS_MAP, "fallback": "objects"}, f, indent=1,
                  ensure_ascii=False)
    for e in errs[:5]:
        log.warning("  %s", e)

    # ---- 语义标签一致性 QC (抽帧) ----
    label_qc = {}
    if args.label_qc_sample:
        label_qc = label_consistency_qc(out, frames, args.splits, raw_root,
                                        args.label_qc_sample)
        if label_qc:
            log.info("标签一致性: 一致率均值 %.0f%% 中位 %.0f%% (>0.5 占比 %.0f%%, 抽检 %d 框)",
                     label_qc["mean"] * 100, label_qc["median"] * 100,
                     label_qc["gt05"] * 100, label_qc["frames_sampled"])

    write_manifest(out, "detection",
                   params=dict(splits=args.splits, limit=args.limit, min_pts=args.min_pts,
                               dropped_by_min_pts=dropped_by_pts,
                               frame="gravity(Rtilt)->ego", depth_scale=1.0 / 6553.5,
                               class_map="CLASS_MAP->SUNRGBD-13",
                               meta_cache_stats=cache_stats, proj_iou=iou_stats,
                               label_qc=label_qc),
                   entries=all_entries)
    log.info("完成, 耗时 %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
