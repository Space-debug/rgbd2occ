# -*- coding: utf-8 -*-
"""SUN RGB-D 13 类语义标签: 路径解析与 occ 语义映射 (数据集侧唯一事实来源)。

train13labels/img13labels-XXXXXX.png (uint8, 与深度同分辨率):
  0=未标注, 1..13 按下表 CLASSES_13 顺序。

occ 语义映射决策 (保持 Occ3D-nuScenes 结构同构 + 最大化室内语义):
  semantic id == 像素值 —— 0=others(未标注), 1..13=SUN RGB-D 13 类,
  14..16 保留, 17=free。与 v3 category.json 的 13 个 *.indoor 类一一对应。
"""
import os

from common.load_label import load_label

CLASSES_13 = ["bed", "books", "ceiling", "chair", "floor", "furniture",
              "objects", "picture", "sofa", "table", "tv", "wall", "window"]

LABEL_DIRS = {"train": "train13labels", "val": "test13labels"}


def label_path(raw_root, split, num):
    """帧号 -> 标签图路径 (不存在返回 None)。"""
    p = os.path.join(raw_root, LABEL_DIRS[split], "img13labels-%06d.png" % num)
    return p if os.path.exists(p) else None


def semantic_classes_doc():
    """写入 occ 包的 semantic_classes.json 侧车 (数据集自描述)。"""
    doc = {"0": "others (未标注)",
           "17": "free (空体素; 未知=mask==0, 语义名义 17)"}
    for i, name in enumerate(CLASSES_13, 1):
        doc[str(i)] = name
    return {"mapping": "pixel value == semantic id",
            "source": "SUN RGB-D 13-class labels (train13labels)",
            "classes": doc}
