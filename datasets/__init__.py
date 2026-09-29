# -*- coding: utf-8 -*-
"""数据集适配器注册表。

新增数据集: 在 datasets/ 下建 <name>/ 目录 (meta.py 读取原始数据,
to_nuscenes.py / to_occ.py 两条转换线), 然后在此登记一行。
主程序 main.py 按注册表调度, 核心库(common/occ/nuscenes)不需改动。
"""
from . import sunrgbd
from .sunrgbd import fill_detection
from .sunrgbd import export_2d
from .sunrgbd import run_all_lines

# {数据集: {产物: (模块, 入口函数)}}
DATASETS = {
    "sunrgbd": {
        "nuscenes": ("datasets.sunrgbd.to_nuscenes", "main"),
        "occ": ("datasets.sunrgbd.to_occ", "main"),
        "detection": ("datasets.sunrgbd.fill_detection", "main"),
        "2d": ("datasets.sunrgbd.export_2d", "main"),
        "all": ("datasets.sunrgbd.run_all_lines", "main"),
    },
}
