# -*- coding: utf-8 -*-
"""环境/依赖体检: rgbd2occ doctor。

一条命令回答: 这个环境能跑哪些产物线、缺什么依赖、建议用什么命令。
检查项: Python / 基础依赖 (numpy/scipy/pillow, 缺失退出码 1) / 加速项
(cv2+numba 提速 exact-fast, torch+CUDA 开 gpu, triton 为 occ 批量 GPU 射线核
所必需) / config.json 路径有效性。
"""
import importlib
import os
import sys


def _mod(name):
    """模块 -> (版本, 错误); 不可用时错误取首行。"""
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "OK"), None
    except Exception as e:
        return None, str(e).splitlines()[0]


def run(_):
    ok_required = True
    print("== Python ==")
    print("  %s (%s)" % (sys.version.split()[0], sys.executable))

    print("== 基础依赖 (转换必需) ==")
    for name, mod in (("numpy", "numpy"), ("scipy", "scipy"), ("pillow", "PIL")):
        v, err = _mod(mod)
        print("  %-8s %-12s %s" % (name, v or "-", "OK" if v else "缺失 (%s)" % err))
        ok_required &= v is not None

    print("== 加速项 (缺失只变慢, 不报错) ==")
    cv, ce = _mod("cv2")
    nv, ne = _mod("numba")
    print("  %-10s %-12s %s" % ("cv2", cv or "-", "OK" if cv else "缺失 (%s)" % ce))
    print("  %-10s %-12s %s" % ("numba", nv or "-", "OK" if nv else "缺失 (%s)" % ne))

    tv, te = _mod("torch")
    cuda = False
    if tv:
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except Exception:
            pass
    trv, tre = _mod("triton")
    print("  %-10s %-12s %s" % ("torch", tv or "-", "CUDA=%s" % cuda if tv
                                else "缺失 (%s)" % te))
    print("  %-10s %-12s %s" % ("triton", trv or "-", "OK" if trv
                                else "缺失 (%s; occ 批量 GPU 射线核必需)" % tre))

    print("== 产物线可用后端 ==")
    gpu_nus = tv and cuda
    gpu_occ = gpu_nus and trv
    print("  sunrgbd nuscenes : exact / fast%s" % (" / gpu" if gpu_nus else ""))
    print("  sunrgbd occ      : exact / fast%s" % (" / gpu" if gpu_occ else
          "   (gpu 需 torch+CUDA+triton)"))
    print("  sunrgbd detection: exact / fast%s" % (" / gpu" if gpu_nus else ""))

    print("== 路径配置 ==")
    try:
        from config import dataset_paths
        for k, v in dataset_paths("sunrgbd").items():
            ok = bool(v) and os.path.isdir(v)
            print("  %-12s %-42s %s" % (k, v or "-", "OK" if ok else "不存在"))
    except Exception as e:
        print("  config 不可用: %s" % e)

    print("== 建议 ==")
    env = os.environ.get("RGBD2OCC_BACKEND")
    print("  当前 RGBD2OCC_BACKEND=%s" % (env or "未设置 (默认 exact)"))
    if gpu_occ:
        print("  全量最快: RGBD2OCC_BACKEND=gpu rgbd2occ sunrgbd all --gpu-batch 16")
    elif gpu_nus:
        print("  安装 triton 后 occ 线也能用 gpu;  当前仅 nuscenes/detection 可 gpu")
    else:
        print("  安装 torch(+CUDA)+triton 后可用 gpu (实测比 CPU 快一个量级以上)")
    sys.exit(0 if ok_required else 1)


if __name__ == "__main__":
    run(None)
