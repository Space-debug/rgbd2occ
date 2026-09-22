# -*- coding: utf-8 -*-
"""生成器版本: 数据集溯源用, 写入 manifest.json。"""
import os
import subprocess

VERSION = "0.9.4"


def git_commit():
    """当前 git 短哈希 (无 git 环境时 "unknown")。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
