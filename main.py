# -*- coding: utf-8 -*-
"""rgbd2occ 统一命令行: 数据集转换 + 产物导出 (每层均有 --help)。

用法:
  rgbd2occ                                   # 列出已注册数据集与全局帮助
  rgbd2occ list                              # 注册表 + config 路径
  rgbd2occ doctor                            # 环境/依赖体检 (各线可用后端)
  rgbd2occ export --help                     # 导出组: points/occ/boxes/preview/info
  rgbd2occ export points <bins目录> --limit 5 # bin -> CloudCompare PLY
  rgbd2occ export occ <labels.npz|gts> ...    # 占据 GT -> PLY (按类着色)
  rgbd2occ export boxes <数据包> --names img-000001   # 3D 框 -> PLY 线框
  rgbd2occ export info <数据包>               # 数据包概览
  rgbd2occ convert <数据集> <产物> [参数...]   # 转换入口透传, 如:
  rgbd2occ convert sunrgbd nuscenes --limit 3
  rgbd2occ sunrgbd occ --limit 3              # 等价简写: 数据集名直接作子命令
  rgbd2occ sunrgbd occ --mode single <深度图> --fx 529.5 ...   # 产物层自带 --help
"""
import argparse
import importlib
import sys

from datasets import DATASETS
from tools import doctor, export_ply
from version import VERSION


def _dispatch_dataset(args):
    """`rgbd2occ <数据集> <产物> ...`: 数据集名即子命令, 转发 convert。"""
    args.dataset = args.cmd
    export_ply.mode_convert(args)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="rgbd2occ",
        description="RGBD 数据集 -> nuScenes devkit 格式 + Occ3D 占据标注; 产物导出 PLY。",
        epilog="转换参数原样透传给对应数据集入口, 各产物入口有自己的 --help "
               "(如: rgbd2occ sunrgbd occ --help)。")
    ap.add_argument("--version", action="version", version="rgbd2occ " + VERSION)
    sub = ap.add_subparsers(dest="cmd", metavar="{list,export,convert,<数据集>}")

    p = sub.add_parser("list", help="列出已注册数据集/产物与 config 路径")
    p.set_defaults(handler=export_ply.mode_list)

    p = sub.add_parser("doctor", help="环境/依赖体检: 各产物线可用后端与路径配置")
    p.set_defaults(handler=doctor.run)

    p = sub.add_parser("export", help="产物导出: CloudCompare PLY / 标注可视化 / 数据包信息")
    exp_sub = p.add_subparsers(dest="export_cmd", required=True,
                               metavar="{points,occ,boxes,preview,demo,info}")
    export_ply.add_subparsers(exp_sub, with_registry=False)

    p = sub.add_parser("convert", help="调用数据集转换入口 (参数原样透传)")
    p.add_argument("dataset", help="数据集名 (见 rgbd2occ list)")
    p.add_argument("product", help="产物名 (该数据集注册的产物)")
    p.add_argument("passthrough", nargs=argparse.REMAINDER,
                   help="原样转给产物入口的参数 (该入口 --help 查看)")
    p.set_defaults(handler=export_ply.mode_convert)

    for ds, products in DATASETS.items():
        p = sub.add_parser(ds, help="转换 %s (产物: %s; 等价于 convert %s ...)"
                           % (ds, ", ".join(products), ds))
        p.add_argument("product", choices=list(products), help="产物名")
        p.add_argument("passthrough", nargs=argparse.REMAINDER,
                       help="原样转给产物入口的参数")
        p.set_defaults(handler=_dispatch_dataset)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd is None:            # 无参数: 打印帮助 + 注册表 (旧行为, 退出码 1)
        ap.print_help()
        print("\n已注册数据集:")
        for ds, products in DATASETS.items():
            print("  %-12s -> %s" % (ds, ", ".join(products)))
        sys.exit(1)
    args.handler(args)


if __name__ == "__main__":
    main()
