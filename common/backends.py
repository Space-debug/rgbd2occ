# -*- coding: utf-8 -*-
"""计算后端分层: exact (默认, 逐位等价) / fast (CPU 近似) / gpu (torch CUDA)。

选择: 环境变量 RGBD2OCC_BACKEND = exact | fast | gpu (默认 exact)。
- exact: numpy+cv2整数域+numba整数核, 输出与 golden 逐位一致 (数据兼容基线)。
- fast : 近似快路径 (median 直接 cv2 整数域全域, 不做 NaN 窗口分支)。
         输出可能与 exact 差异 —— 产物 manifest 记录 data_variant="fast"。
- gpu  : torch CUDA 优先 (raycast 等大并行核), 无 GPU/无 torch/无 triton 自动降级
         fast (occ 批量 GPU 射线核依赖 triton, gpu 层要求二者齐备)。
无论何种 tier, golden/单元测试始终以 exact 断言 (conftest 强制)。
"""
import os


def _torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _triton_ok():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def get_backend():
    """返回 ('exact'|'fast'|'gpu', 说明 dict)。gpu 不可用时自动降级 fast。"""
    want = os.environ.get("RGBD2OCC_BACKEND", "exact").lower()
    info = {"requested": want}
    if want == "gpu":
        if not _torch_cuda():
            info["note"] = "gpu 不可用 (无 torch/无 NVIDIA GPU), 降级 fast"
            return "fast", info
        if not _triton_ok():
            info["note"] = "gpu 不可用 (occ 批量射线核需要 triton), 降级 fast"
            return "fast", info
        info["note"] = "torch CUDA + triton available"
        return "gpu", info
    if want == "fast":
        info["note"] = "cpu 近似快路径"
        return "fast", info
    return "exact", info


def data_variant(backend):
    """exact 之外的 tier 会产生数值差异, 产物需记录 data_variant。"""
    return "exact" if backend == "exact" else backend
