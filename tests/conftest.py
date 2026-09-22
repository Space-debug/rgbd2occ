# -*- coding: utf-8 -*-
"""pytest 配置: 把仓库根加入 sys.path (零依赖运行器 run_all.py 依赖同样逻辑)。"""
import os
import sys

os.environ.setdefault("RGBD2OCC_BACKEND", "exact")   # 测试始终以 exact 断言
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
