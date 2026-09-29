# -*- coding: utf-8 -*-
"""批量进度显示: tqdm 可用时显示进度条, 否则原样返回迭代器
(调用方已有的定期日志行兜底, 如每 200 帧的 帧/s 统计)。"""
from __future__ import annotations


def progress_iter(iterable, total=None, desc=""):
    """包一层 tqdm (stderr, 不污染日志文件); 未安装 tqdm 时零开销透传。"""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc or None, unit="帧")
