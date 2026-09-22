# -*- coding: utf-8 -*-
"""零依赖测试运行器 (无 pytest 环境时): python tests/run_all.py
有 pytest 时等价于: pytest tests/ -v
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # 仓库根


def main():
    import importlib
    failed, passed = [], 0
    for fname in sorted(os.listdir(HERE)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        mod = importlib.import_module("tests." + fname[:-3])
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {fname}::{name}")
            except Exception:
                failed.append((fname, name, traceback.format_exc()))
                print(f"FAIL {fname}::{name}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for f, n, tb in failed:
        print(f"\n==== {f}::{n}\n{tb}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
