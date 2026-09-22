# -*- coding: utf-8 -*-
"""点云查看工具: 查看生成的 LIDAR_TOP .pcd.bin (float32 Nx5, 自车系 x前/y左/z上)。

用法:
  python tools/view_pointcloud.py samples/LIDAR_TOP/train/img-000001.pcd.bin
  python tools/view_pointcloud.py <bin> --mode 3d [--extent 8] [--save out.png]
模式:
  bev (默认) : BEV 俯视图 PNG (按高度着色), 零依赖
  3d         : 交互 3D —— 有 open3d 用 open3d, 否则 matplotlib 3D 散点
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.render_bev import render_points_bev  # noqa: E402


def load_bin(path):
    pts = np.fromfile(path, np.float32)
    assert pts.size % 5 == 0, "不是 float32 Nx5 的 pcd.bin"
    return pts.reshape(-1, 5)


def show_3d(pts, title):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        z = np.clip(pts[:, 2], -1, 5.4)
        t = (z + 1) / 6.4
        pcd.colors = o3d.utility.Vector3dVector(np.stack([t, 0.4 + 0.3 * t, 1 - 0.8 * t], 1))
        o3d.visualization.draw_geometries([pcd], window_name=title)
        return
    except ImportError:
        pass
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    c = pts[:, 2]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, cmap="viridis", s=1)
    ax.set_xlabel("x 前"); ax.set_ylabel("y 左"); ax.set_zlabel("z 上")
    ax.set_title("%s (%d 点)" % (title, len(pts)))
    plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bin", help=".pcd.bin 路径")
    ap.add_argument("--mode", choices=["bev", "3d"], default="bev")
    ap.add_argument("--extent", type=float, default=8, help="BEV 显示半径(米), 室内 8 / 官方范围 40")
    ap.add_argument("--save", default=None, help="保存 PNG 而不弹窗")
    args = ap.parse_args(argv)

    pts = load_bin(args.bin)
    print(f"{os.path.basename(args.bin)}: {len(pts)} 点, "
          f"x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] "
          f"z[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")
    if args.mode == "3d":
        show_3d(pts, os.path.basename(args.bin))
        return
    out = args.save or os.path.join(os.path.dirname(os.path.abspath(args.bin)),
                                    os.path.splitext(os.path.basename(args.bin))[0] + "_bev.png")
    render_points_bev(pts[:, :3], out, extent=args.extent,
                      title="%s  n=%d" % (os.path.basename(args.bin), len(pts)))
    print("->", out)
    if not args.save:
        from PIL import Image
        Image.open(out).show()


if __name__ == "__main__":
    main()
