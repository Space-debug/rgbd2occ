# -*- coding: utf-8 -*-
"""nuScenes devkit 标注表工具: 稳定 token 生成与 13 张表的落盘。

nuScenes 数据库由 13 张 json 表描述 (attribute/calibrated_sensor/category/
ego_pose/instance/log/map/sample/sample_annotation/sample_data/scene/sensor/
visibility); 本模块只提供 token 与写出, 表的内容组装在各数据集入口
(如 sunrgbd2nuscenes.py) 中完成。
"""
import hashlib
import json
import os

NU_TABLES = ["attribute", "calibrated_sensor", "category", "ego_pose",
             "instance", "log", "map", "sample", "sample_annotation",
             "sample_data", "scene", "sensor", "visibility"]


def token(key):
    """稳定 token: 输入字符串的 md5 (32 hex, 与 nuScenes 格式一致)。"""
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def write_tables(ver_dir, tables):
    """把 {表名: 行列表} 写成 <ver_dir>/<表名>.json (indent=1, 同 v3)。"""
    os.makedirs(ver_dir, exist_ok=True)
    for name in NU_TABLES:
        with open(os.path.join(ver_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(tables.get(name, []), f, indent=1)
