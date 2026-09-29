# -*- coding: utf-8 -*-
"""progress_iter 测试: 有 tqdm 返回进度条、无 tqdm 透传, 两种路径都产出全部元素。"""
import builtins
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.progress import progress_iter  # noqa: E402


def test_progress_iter_yields_all():
    assert list(progress_iter([1, 2, 3], total=3, desc="t")) == [1, 2, 3]


def test_progress_iter_fallback_without_tqdm():
    """tqdm 导入失败时零依赖透传 (生成器语义不变)。"""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "tqdm":
            raise ImportError("no tqdm")
        return real_import(name, *a, **kw)

    builtins.__import__ = fake_import
    try:
        it = progress_iter([1, 2], total=2, desc="t")
        assert not hasattr(it, "format_meter")     # 不是 tqdm 对象
        assert list(it) == [1, 2]
    finally:
        builtins.__import__ = real_import
