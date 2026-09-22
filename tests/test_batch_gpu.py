# -*- coding: utf-8 -*-
"""批量 GPU 流水线测试 (无 torch/CUDA 时跳过)。"""
import os
import numpy as np

def _torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _synth_frames(n=3):
    """合成 n 帧: 5m 处一面墙 (uint16 深度) + 全图类别 4 标签。"""
    raw = np.full((240, 320), int(5.0 / (1 / 6553.5)), np.uint16)
    raw[:, :60] = int(1.5 * 6553.5)          # 左侧近墙
    lab = np.full((240, 320), 4, np.uint8)   # chair
    K = (529.5, 529.5, 320.0, 240.0)
    frames = [dict(raw=raw, lab=lab, fx=K[0], fy=K[1], cx=K[2], cy=K[3],
                   scale=1 / 6553.5) for _ in range(n)]
    return frames, K


def test_batch_matches_single_frame_gpu():
    if not _torch_cuda():
        return
    """批量调用与逐帧调用逐体素一致。"""
    from occ.gpu_raycast import voxelize_and_cast_gpu, voxelize_and_cast_batch_gpu
    from occ.voxel_grid import make_grid
    gmin, dims = make_grid(0.4)
    frames, _ = _synth_frames(3)
    os.environ["RGBD2OCC_BACKEND"] = "gpu"
    try:
        batch = voxelize_and_cast_batch_gpu(frames, gmin, 0.4, dims, 1)
        for i, fr in enumerate(frames):
            single = voxelize_and_cast_gpu(
                fr["raw"], fr["scale"], fr["fx"], fr["fy"], fr["cx"], fr["cy"],
                fr["lab"], gmin, 0.4, dims, 1)
            assert np.array_equal(batch[i][0], single[0]), f"帧{i} free 不一致"
            assert np.array_equal(batch[i][1], single[1]), f"帧{i} 占据格不一致"
            assert np.array_equal(batch[i][2], single[2]), f"帧{i} 标签不一致"
    finally:
        os.environ["RGBD2OCC_BACKEND"] = "exact"


def test_batch_vs_cpu_exact_small_diff():
    if not _torch_cuda():
        return
    """GPU 批量 vs CPU exact: 语义差异 < 0.5% (近似层容差)。"""
    from occ.gpu_raycast import voxelize_and_cast_batch_gpu
    from occ.voxel_grid import make_grid
    from occ.convert import convert_frame
    from common import mask_depth
    gmin, dims = make_grid(0.4)
    frames, K = _synth_frames(2)
    os.environ["RGBD2OCC_BACKEND"] = "exact"
    refs = [convert_frame(mask_depth(f["raw"].astype(np.float64) * f["scale"], (0.3, 8.0)),
                          K[0], K[1], K[2], K[3], label=f["lab"]) for f in frames]
    os.environ["RGBD2OCC_BACKEND"] = "gpu"
    batch = voxelize_and_cast_batch_gpu(frames, gmin, 0.4, dims, 1)
    os.environ["RGBD2OCC_BACKEND"] = "exact"
    tot = refs[0]["semantics"].size * len(frames)
    sd = md = 0
    for b, r in zip(batch, refs):
        sem = np.full(dims, 17, np.uint8)
        sem.reshape(-1)[b[1]] = b[2]                     # 占据格标签回填
        sd += int((sem != r["semantics"]).sum())
        md += int((b[0].astype(np.uint8) != r["mask_camera"]).sum())
    assert sd / tot < 0.005, f"语义差异 {sd/tot:.3%} 超容差"
    assert md / tot < 0.005, f"mask 差异 {md/tot:.3%} 超容差"
