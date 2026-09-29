# -*- coding: utf-8 -*-
"""示例: 读取 sunrgbd_nuscenes_v3 数据包 —— nuScenes devkit 与零依赖两种方式。

表目录名 v1.0-sunrgbd-{train,val} 遵循 nuScenes 的 "v1.0-<版本>" 目录约定,
devkit 的 NuScenes(version=..., dataroot=...) 按原样拼接表路径, 因此可直接挂载:

  pip install nuscenes-devkit
  from nuscenes.nuscenes import NuScenes
  nusc = NuScenes(version="v1.0-sunrgbd-train",
                  dataroot="D:/Datasets/sunrgbd_nuscenes_v3")

适用范围: 表导航 (scene -> sample 前后向帧链 -> sample_data -> 标定/位姿)、
LidarPointCloud / Box 加载、get_sample_data (点云自动变换到自车系, 本包平移
本就是 0) 均可直接用; 高精地图/雷达渲染类功能不适用 (室内数据无地图)。

本包特有约定 (与官方 nuScenes 的差异):
  - ego_pose / calibrated_sensor 平移恒 0 (无真值位姿, 不写假值)
  - sample_annotation.attribute_token 为空串 (devkit 渲染属性时跳过即可)
  - 点云 intensity=亮度/255, ring=0; 类别为 13 个 *.indoor

用法:
  python tools/read_devkit_example.py [数据包根目录] [--split train]
未安装 devkit 时自动退回零依赖模式 (纯 json+numpy, 走同样的读取链路)。
"""
import argparse
import json
import os
import sys

import numpy as np

DEFAULT_ROOT = "D:/Datasets/sunrgbd_nuscenes_v3"


def load_bin(path):
    """nuScenes LIDAR_TOP bin: float32 Nx5 (x,y,z,intensity,ring), 自车系。"""
    return np.fromfile(path, np.float32).reshape(-1, 5)


def demo_devkit(root, split):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-sunrgbd-%s" % split, dataroot=root, verbose=True)

    print("\n== scene 列表 ==")
    nusc.list_scenes()
    sc = nusc.scene[0]

    # 帧链: sample.prev/next 串起该 scene 的全部帧
    sample = nusc.get("sample", sc["first_sample_token"])
    print("\n== 前向帧链 (scene=%s, %d 帧) ==" % (sc["name"], sc["nbr_samples"]))
    for k in range(3):
        if sample is None:
            break
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        print("sample %s  ts=%d  lidar=%s" % (sample["token"][:8], sample["timestamp"],
                                              lidar_sd["filename"]))
        sample = nusc.get("sample", sample["next"]) if sample["next"] else None

    # 点云 + 3D 框 (get_sample_data 返回 (文件路径, Box 列表, 相机内参);
    # 不同 devkit 版本首元素为路径或点云对象, 统一用 LidarPointCloud 读文件最稳)
    from nuscenes.utils.data_classes import LidarPointCloud
    sample = nusc.get("sample", sc["first_sample_token"])
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    points = LidarPointCloud.from_file(os.path.join(root, lidar_sd["filename"]))
    print("\n== 点云/框 ==")
    print("points:", points.points.shape, "(x,y,z,intensity[,ring] x N, 自车系)")
    for b in boxes[:3]:
        print("Box %-12s center=%s size(wlh)=%s" % (
            b.name, np.round(b.center, 2), np.round(b.wlh, 2)))

    # 相机内参 (本包按唯一 K_native 建标定条目, 轴置换四元数)
    cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    print("\n== CAM_FRONT 内参 ==")
    for row in cs["camera_intrinsic"]:
        print("  ", [round(v, 1) for v in row])

    # 检测标注: sample -> sample_annotation -> instance -> category
    anns = [a for a in nusc.sample_annotation if a["sample_token"] == sample["token"]]
    print("\n== 检测标注 (%d 框) ==" % len(anns))
    for a in anns[:5]:
        inst = nusc.get("instance", a["instance_token"])
        cat = nusc.get("category", inst["category_token"])
        print("  %-12s size(wlh)=%s num_lidar_pts=%d" % (
            cat["name"], [round(v, 2) for v in a["size"]], a["num_lidar_pts"]))


def demo_plain(root, split):
    """零依赖: 直接读 13 张 json 表 + bin, 逻辑与 devkit 等价。"""
    tb = os.path.join(root, "v1.0-sunrgbd-%s" % split)
    rd = lambda n: json.load(open(os.path.join(tb, n + ".json"), encoding="utf-8"))
    scenes, samples = rd("scene"), {s["token"]: s for s in rd("sample")}
    inst = {i["token"]: i for i in rd("instance")}
    cats = {c["token"]: c["name"] for c in rd("category")}
    cs = {c["token"]: c for c in rd("calibrated_sensor")}

    print("scene 列表:")
    for sc in scenes:
        print("  %-24s %d 帧" % (sc["name"], sc["nbr_samples"]))

    sc = scenes[0]
    sample = samples[sc["first_sample_token"]]
    print("\n前向帧链 (scene=%s):" % sc["name"])
    for _ in range(3):
        if sample is None:
            break
        print("  sample %s  ts=%d" % (sample["token"][:8], sample["timestamp"]))
        sample = samples.get(sample["next"])

    sample = samples[sc["first_sample_token"]]
    sd = [e for e in rd("sample_data") if e["sample_token"] == sample["token"]]
    lid = next(e for e in sd if e["sensor_modality"] == "lidar")
    pts = load_bin(os.path.join(root, lid["filename"]))
    print("\n点云:", pts.shape, " x范围 [%.2f, %.2f] m" % (pts[:, 0].min(), pts[:, 0].max()))
    cam = next(e for e in sd if e["sensor_modality"] == "camera")
    print("CAM_FRONT 内参:", cs[cam["calibrated_sensor_token"]]["camera_intrinsic"][0])

    anns = [a for a in rd("sample_annotation")
            if a["sample_token"] == sample["token"]]
    print("检测标注 %d 框:" % len(anns))
    for a in anns[:5]:
        print("  %-16s size(wlh)=%s" % (
            cats[inst[a["instance_token"]]["category_token"]],
            [round(v, 2) for v in a["size"]]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="数据包根目录")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args(argv)
    try:
        import nuscenes  # noqa: F401
        print("== nuScenes devkit 模式 ==")
        demo_devkit(args.root, args.split)
    except ImportError:
        print("未安装 nuscenes-devkit (pip install nuscenes-devkit), 用零依赖模式\n")
        demo_plain(args.root, args.split)


if __name__ == "__main__":
    sys.exit(main())
