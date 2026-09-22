# -*- coding: utf-8 -*-
"""主生成程序: 按数据集与目标产物调度转换。

用法:
  python main.py                          # 列出已注册的数据集与产物
  python main.py <数据集> <产物> [参数...]  # 其余参数原样转给对应入口, 例如:
  python main.py sunrgbd nuscenes --limit 3
  python main.py sunrgbd occ --limit 3
  python main.py sunrgbd occ --mode single <深度图> --fx 529.5 ...
"""
import importlib
import sys

from datasets import DATASETS


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("已注册:")
        for ds, products in DATASETS.items():
            print("  %-12s -> %s" % (ds, ", ".join(products)))
        sys.exit(0 if len(sys.argv) >= 2 else 1)
    ds, product = sys.argv[1], sys.argv[2]
    if ds not in DATASETS:
        sys.exit(f"未注册的数据集: {ds} (可选: {', '.join(DATASETS)})")
    if product not in DATASETS[ds]:
        sys.exit(f"{ds} 不支持产物 {product} (可选: {', '.join(DATASETS[ds])})")
    module_name, entry = DATASETS[ds][product]
    getattr(importlib.import_module(module_name), entry)(sys.argv[3:])


if __name__ == "__main__":
    main()
