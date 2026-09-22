# -*- coding: utf-8 -*-
"""rgbd2occ 通用核心库: RGBD 数据 -> Occ3D-nuScenes 同构占据标注。

数据集无关。各数据集的专属入口为仓库根目录下的 xxx2occ.py
(如 sunrgbd2occ.py), 复用本包完成读取/预处理/转换/登记。
"""
from .convert import FREE, OTHERS, convert_frame
from .annotations import OccAnnotations, pose
from .data_io import load_depth, load_label
from .preprocess import mask_depth

__all__ = ["FREE", "OTHERS", "convert_frame", "OccAnnotations", "pose",
           "load_depth", "load_label", "mask_depth"]
