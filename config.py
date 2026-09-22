# -*- coding: utf-8 -*-
"""路径配置: 机器相关路径一律来自 config.json (模板见 config.example.json)。

优先级: 环境变量 RGBD2OCC_CONFIG 指定的文件 > 仓库根 config.json > config.example.json。
数据集模块用 dataset_paths(name) 取路径; CLI 的 --out/--raw-root 等参数可临时覆盖。
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load():
    env = os.environ.get("RGBD2OCC_CONFIG")
    for p in ([env] if env else []) + [
            os.path.join(REPO_ROOT, "config.json"),
            os.path.join(REPO_ROOT, "config.example.json")]:
        if p and os.path.exists(p):
            return json.load(open(p, encoding="utf-8")), p
    raise FileNotFoundError(
        "找不到 config.json / config.example.json; 可复制模板为 config.json "
        "或用环境变量 RGBD2OCC_CONFIG 指定路径")


def dataset_paths(name):
    """取数据集的路径配置 dict (raw_root / nuscenes_out / occ_out ...)。"""
    data, path = _load()
    if name not in data:
        raise KeyError(f"配置文件 {path} 缺少数据集 {name}")
    return data[name]
