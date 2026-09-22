# -*- coding: utf-8 -*-
"""数据集清单 (manifest.json): 记录生成器版本/参数快照/逐帧文件 md5 与计数,
使生成的数据可溯源、可校验 ("这批数据是哪版代码+什么参数生成的")。"""
import hashlib
import json
import os
import time


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(out_root, product, params, entries):
    """写 <out_root>/manifest_<product>.json (多产物共用根目录时互不覆盖)。
    entries: {key: {"file": 相对路径, "count": 点数或可见体素数}} -> 自动补 bytes/md5。
    params: 本次运行的参数快照 dict (CLI args + 关键常量)。"""
    import version

    full = {}
    _cache = {}  # 多条目指向同一文件时只哈希一次
    for key, e in entries.items():
        p = os.path.join(out_root, e["file"])
        if p not in _cache:
            _cache[p] = (os.path.getsize(p) if os.path.exists(p) else 0,
                         _md5(p) if os.path.exists(p) else None)
        row = dict(e)
        row["file"] = e["file"].replace(chr(92), "/")
        row["bytes"], row["md5"] = _cache[p]
        full[key] = row
    manifest = {
        "generator": {"name": "rgbd2occ", "version": version.VERSION,
                      "commit": version.git_commit(),
                      "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "product": product,
        "params": params,
        "frames": len(full),
        "entries": full,
    }
    path = os.path.join(out_root, "manifest_%s.json" % product)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return path
