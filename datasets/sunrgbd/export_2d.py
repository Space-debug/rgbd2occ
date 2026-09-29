# -*- coding: utf-8 -*-
"""SUN RGB-D 2D 检测框导出: detection_meta_cache (gtBb2D) -> annotations2d.json 侧车。

背景: 2D gt 框只存在于内部缓存 detection_meta_cache.json, devkit 13 表没有
2D 框 —— 下游需要可训练的 2D 标注时从这里导出。

产物: <out>/annotations2d.json
  {"train/img-000001": [
      {"cls": "chair",          # SUNRGBD 37 风格原始类名
       "cls13": "chair",        # 检测线 CLASS_MAP 映射的 13 类
       "bbox": [x, y, w, h]},   # 原始分辨率像素坐标 (与 samples/CAM_FRONT 原图对齐)
      ...]}
无 2D 框的 3D 框不导出 (bbox 缺失); 帧内全部缺 2D 时该帧不出现。

用法:
  rgbd2occ convert sunrgbd 2d [--out ...] [--splits train val] [--limit N]
依赖: 先跑过检测线 (生成 detection_meta_cache.json)。
"""
import argparse
import json
import os
import sys

from common.get_logger import get_logger
from common.write_manifest import write_manifest
from config import dataset_paths
from .fill_detection import CLASS_MAP     # 37 风格 -> 13 类, 与检测线同一份映射

log = get_logger("rgbd2occ.sunrgbd.2d")

PATHS = dataset_paths("sunrgbd")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="数据包根目录 (默认 config 的 nuscenes_out)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="每 split 前 N 帧 (试跑)")
    args = ap.parse_args(argv)
    out = args.out or PATHS["nuscenes_out"]
    cache_p = os.path.join(out, "detection_meta_cache.json")
    if not os.path.exists(cache_p):
        sys.exit("缺 %s; 先跑: rgbd2occ convert sunrgbd detection" % cache_p)
    cache = json.load(open(cache_p, encoding="utf-8"))

    rows = {}
    for key in sorted(k for k in cache if "/" in k):
        sp = key.split("/")[0]
        if sp not in args.splits:
            continue
        boxes = [{"cls": b["cls"],
                  "cls13": CLASS_MAP.get(b["cls"].lower(), "objects"),
                  "bbox": b["bb2d"]}
                 for b in cache[key]["boxes"] if b.get("bb2d")]
        if boxes:
            rows[key] = boxes
    if args.limit:
        take, seen = {}, {}
        for key in sorted(rows):
            sp = key.split("/")[0]
            seen[sp] = seen.get(sp, 0) + 1
            if seen[sp] <= args.limit:
                take[key] = rows[key]
        rows = take
    entries = rows
    n_boxes = sum(len(v) for v in entries.values())

    dst = os.path.join(out, "annotations2d.json")
    json.dump(entries, open(dst, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    write_manifest(out, "2d",
                   params=dict(splits=args.splits, limit=args.limit,
                               source="detection_meta_cache.json (gtBb2D)",
                               bbox="原始分辨率像素 [x,y,w,h]",
                               class_fields=["cls (37 风格)", "cls13 (SUNRGBD-13)"]),
                   entries={k: {"file": "annotations2d.json", "count": len(v)}
                            for k, v in entries.items()})
    log.info("2D 框导出: %d 帧 / %d 框 -> %s", len(entries), n_boxes, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
