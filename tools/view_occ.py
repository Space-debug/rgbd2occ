# -*- coding: utf-8 -*-
"""Occ 查看工具: 查看 labels.npz 占据标注 (semantics + mask)。

用法:
  python tools/view_occ.py gts/<scene>/<token>/labels.npz
  python tools/view_occ.py <labels.npz> --mode 3d [--save out.png] [--mask none]
模式:
  bev (默认) : BEV 俯视图 (占据按类着色/可见free浅绿/未知深色), 零依赖
  3d         : 交互 3D —— 有 open3d 画体素点云(按类着色), 否则 matplotlib 3D
选项:
  --mask camera|lidar|none : bev/3d 采用的可见性掩膜 (默认 camera; none=显示全部占据)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.render_bev import CLASS_COLORS, render_occ_bev  # noqa: E402

CLS_NAMES = ["others", "bed", "books", "ceiling", "chair", "floor", "furniture",
             "objects", "picture", "sofa", "table", "tv", "wall", "window"]


def load_npz(path):
    with np.load(path) as d:
        return d["semantics"], d.get("mask_lidar"), d.get("mask_camera")


def stats(sem, mask):
    occ = (sem < 17) & (mask == 1)
    free = (sem == 17) & (mask == 1)
    u, c = np.unique(sem[occ], return_counts=True)
    dist = {CLS_NAMES[k] if k < len(CLS_NAMES) else str(k): int(v)
            for k, v in zip(u, c)}
    print(f"可见体素 {int((mask == 1).sum())} (占据 {int(occ.sum())}, free {int(free.sum())}), "
          f"未知(掩膜外) {int((mask == 0).sum())}")
    print("占据类别分布:", dist)


def show_3d(sem, mask, title):
    occ = (sem < 17) & (mask == 1)
    xyz = np.argwhere(occ) * 0.4          # 体素中心(近似, +0.2 偏移不影响观看)
    cls = sem[occ]
    pal = np.array(CLASS_COLORS, np.float32) / 255
    colors = pal[cls]
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:, ::-1] * [1, 1, 1])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=title)
        return
    except ImportError:
        pass
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 2], xyz[:, 1], xyz[:, 0], c=colors, s=8)
    ax.set_xlabel("z"); ax.set_ylabel("y(左)"); ax.set_zlabel("x(前)")
    ax.set_title(title)
    plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", help="labels.npz 路径")
    ap.add_argument("--mode", choices=["bev", "3d"], default="bev")
    ap.add_argument("--mask", choices=["camera", "lidar", "none"], default="camera")
    ap.add_argument("--save", default=None)
    args = ap.parse_args(argv)

    sem, ml, mc = load_npz(args.npz)
    if args.mask == "camera" and mc is not None:
        mask = mc
    elif args.mask == "lidar" and ml is not None:
        mask = ml
    else:
        mask = np.ones_like(sem, np.uint8)
    stats(sem, mask)

    if args.mode == "3d":
        show_3d(sem, mask, os.path.basename(os.path.dirname(args.npz)))
        return
    out = args.save or os.path.join(os.path.dirname(os.path.abspath(args.npz)),
                                    "bev_" + os.path.basename(os.path.dirname(args.npz)) + ".png")
    render_occ_bev(sem, mask, out, title=os.path.basename(args.npz))
    print("->", out)
    if not args.save:
        from PIL import Image
        Image.open(out).show()


if __name__ == "__main__":
    main()
