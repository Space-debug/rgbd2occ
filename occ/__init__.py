# -*- coding: utf-8 -*-
"""rgbd2occ 占据标注核心库: RGBD 数据 -> Occ3D-nuScenes 同构占据 GT。

数据集无关。各数据集的专属入口为仓库根目录下的 xxx2occ.py
(如 sunrgbd2occ.py), 复用本包完成读取/预处理/转换/登记。
深度/标签读取在 common.io (与 nuScenes 点云管线共用), 此处重导出保持 API。
"""
from common import load_depth, load_label, mask_depth
from .convert import FREE, OTHERS, convert_frame
from .annotations import OccAnnotations, pose

__all__ = ["FREE", "OTHERS", "convert_frame", "OccAnnotations", "pose",
           "load_depth", "load_label", "mask_depth"]
