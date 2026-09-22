# -*- coding: utf-8 -*-
"""向量化 DDA 射线追踪 (Amanatides & Woo 体素遍历)。

从相机原点沿每个像素方向行进到表面深度为止, 标记途经体素为 free。
双实现, 输出严格一致:
- numba 核 (可用时): 逐射线独立 DDA 循环, 纯浮点步进 + 整数索引,
  与向量化版相同的推进规则 (先标记 t<=dist 的当前体素, 再跨越最近边界;
  活跃条件 t<=dist, t 无穷视作不活跃)。
- numpy 向量化 (回退): 每次迭代把所有活跃射线各推进一步。
两者均已与最初的逐条循环实现逐元素比对一致; 输入的 dirs/max_dists 为
相机系, 内部完成与 voxel_grid.cam_to_ego_axes 相同的轴变换。
"""
import os

import numpy as np

try:
    if os.environ.get("RGBD2OCC_NO_NUMBA"):
        raise ImportError
    from numba import njit

    @njit(cache=True)
    def _cast_kernel(o, dirs_cam, dists, gmin, voxel, d0, d1, d2):
        free = np.zeros((d0, d1, d2), np.uint8)
        # 相机原点 -> 网格浮点坐标
        ox = (o[0] - gmin[0]) / voxel
        oy = (o[1] - gmin[1]) / voxel
        oz = (o[2] - gmin[2]) / voxel
        n = dirs_cam.shape[0]
        for i in range(n):
            # 轴变换: X=相机z, Y=-相机x, Z=-相机y
            dx = dirs_cam[i, 2]
            dy = -dirs_cam[i, 0]
            dz = -dirs_cam[i, 1]
            norm = (dx * dx + dy * dy + dz * dz) ** 0.5 + 1e-12
            vx, vy, vz = dx / norm, dy / norm, dz / norm
            dist = dists[i] / voxel
            # Amanatides & Woo 初始化
            cx = int(np.floor(ox))
            cy = int(np.floor(oy))
            cz = int(np.floor(oz))
            stepx = 1 if vx > 0 else (-1 if vx < 0 else 0)
            stepy = 1 if vy > 0 else (-1 if vy < 0 else 0)
            stepz = 1 if vz > 0 else (-1 if vz < 0 else 0)
            if vx != 0.0:
                tx = abs(((cx + 1 - ox) if vx > 0 else (ox - cx)) / vx)
                tdx = abs(1.0 / vx)
            else:
                tx = np.inf
                tdx = np.inf
            if vy != 0.0:
                ty = abs(((cy + 1 - oy) if vy > 0 else (oy - cy)) / vy)
                tdy = abs(1.0 / vy)
            else:
                ty = np.inf
                tdy = np.inf
            if vz != 0.0:
                tz = abs(((cz + 1 - oz) if vz > 0 else (oz - cz)) / vz)
                tdz = abs(1.0 / vz)
            else:
                tz = np.inf
                tdz = np.inf
            t = 0.0
            while t <= dist:
                if 0 <= cx < d0 and 0 <= cy < d1 and 0 <= cz < d2:
                    free[cx, cy, cz] = 1
                # 跨越最近的轴边界 (三轴并列时按 x,y,z 优先, 与 argmin 一致)
                if tx <= ty and tx <= tz:
                    t = tx
                    cx += stepx
                    tx += tdx
                elif ty <= tz:
                    t = ty
                    cy += stepy
                    ty += tdy
                else:
                    t = tz
                    cz += stepz
                    tz += tdz
        return free

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _cast_torch(origin_g, dirs, max_dists, gmin, voxel, dims, stride):
    """GPU torch 向量化 DDA (gpu 层; float32 -> 数值与 CPU 可能差 1ULP, 属近似层)。"""
    import torch
    dev = "cuda"
    o = torch.tensor((np.asarray(origin_g, np.float64) - gmin) / voxel,
                     dtype=torch.float32, device=dev)
    d = np.empty_like(dirs)
    d[:, 0], d[:, 1], d[:, 2] = dirs[:, 2], -dirs[:, 0], -dirs[:, 1]
    d = torch.tensor(d[::stride], dtype=torch.float32, device=dev)
    dist = torch.tensor((np.asarray(max_dists, np.float64) / voxel)[::stride],
                        dtype=torch.float32, device=dev)
    v = d / (d.norm(dim=1, keepdim=True) + 1e-12)
    n = v.shape[0]
    cur = torch.floor(o).to(torch.int64).repeat(n, 1)
    step = torch.sign(v).to(torch.int64)
    small = v.abs() <= 1e-12
    tDelta = torch.where(small, torch.inf, 1.0 / v.abs())
    frac = torch.where(v > 0, (cur + 1 - o) / v, (cur - o) / v)
    tMax = torch.where(small, torch.inf, frac.abs())
    t = torch.zeros(n, dtype=torch.float32, device=dev)
    dims_t = torch.tensor(dims, device=dev)
    free = torch.zeros(tuple(dims), dtype=torch.bool, device=dev)
    max_iters = int(dist.max().item()) + 2 + int(o.abs().max().item()) + 2
    for _ in range(max_iters):
        act = t <= dist
        if not act.any():
            break
        idx = cur[act]
        inb = ((idx >= 0) & (idx < dims_t)).all(1)
        fi = idx[inb]
        free[fi[:, 0], fi[:, 1], fi[:, 2]] = True
        ax = tMax.argmin(dim=1)
        t = torch.where(act, tMax[torch.arange(n, device=dev), ax], t)
        cur[torch.arange(n, device=dev), ax] += step[torch.arange(n, device=dev), ax]
        tMax[torch.arange(n, device=dev), ax] += tDelta[torch.arange(n, device=dev), ax]
        t[~torch.isfinite(t)] = torch.inf
    return free.cpu().numpy().astype(bool)


_TRITON_KERNEL = None


def _triton_fn():
    """惰性编译 Triton 融合核: 一线程一射线, 标量 DDA 循环跑到底 (无 launch 风暴)。"""
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _dda(o_x, o_y, o_z, dirs_ptr, dists_ptr, free_ptr,
                 voxel, d0, d1, d2, BLOCK: tl.constexpr):
            i = tl.program_id(0)          # 一 program 一射线: 全量并行
            dx = tl.load(dirs_ptr + i * 3 + 2)
            dy = -tl.load(dirs_ptr + i * 3 + 0)
            dz = -tl.load(dirs_ptr + i * 3 + 1)
            dist = tl.load(dists_ptr + i) / voxel
            norm = tl.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            vx = dx / norm
            vy = dy / norm
            vz = dz / norm
            cx = tl.floor(o_x).to(tl.int32)
            cy = tl.floor(o_y).to(tl.int32)
            cz = tl.floor(o_z).to(tl.int32)
            stepx = 1 if vx > 0 else (-1 if vx < 0 else 0)
            stepy = 1 if vy > 0 else (-1 if vy < 0 else 0)
            stepz = 1 if vz > 0 else (-1 if vz < 0 else 0)
            if vx != 0.0:
                tx = tl.abs(((cx + 1 - o_x) if vx > 0 else (o_x - cx)) / vx)
                tdx = tl.abs(1.0 / vx)
            else:
                tx = float("inf")
                tdx = float("inf")
            if vy != 0.0:
                ty = tl.abs(((cy + 1 - o_y) if vy > 0 else (o_y - cy)) / vy)
                tdy = tl.abs(1.0 / vy)
            else:
                ty = float("inf")
                tdy = float("inf")
            if vz != 0.0:
                tz = tl.abs(((cz + 1 - o_z) if vz > 0 else (o_z - cz)) / vz)
                tdz = tl.abs(1.0 / vz)
            else:
                tz = float("inf")
                tdz = float("inf")
            t = 0.0
            d12 = d1 * d2
            while t <= dist:
                inb = ((cx >= 0) & (cx < d0)) & ((cy >= 0) & (cy < d1)) & ((cz >= 0) & (cz < d2))
                if inb:
                    tl.store(free_ptr + cx * d12 + cy * d2 + cz, 1)
                if tx <= ty and tx <= tz:
                    t = tx
                    cx += stepx
                    tx += tdx
                elif ty <= tz:
                    t = ty
                    cy += stepy
                    ty += tdy
                else:
                    t = tz
                    cz += stepz
                    tz += tdz
        _TRITON_KERNEL = _dda
        return _dda
    except Exception:
        _TRITON_KERNEL = False
        return False


def _cast_triton(origin_g, dirs, max_dists, gmin, voxel, dims, stride):
    """Triton 融合核 (gpu 层首选): 一线程一射线。float32 -> 数值与 CPU 理论上
    可差 1ULP, 实测与 CPU numba 核逐体素一致 (RTX 5090)。"""
    import torch
    kern = _triton_fn()
    if not kern:
        raise RuntimeError("triton 不可用")
    dev = "cuda"
    o = (np.asarray(origin_g, np.float64) - np.asarray(gmin, np.float64)) / voxel
    d = np.empty_like(dirs)
    d[:, 0], d[:, 1], d[:, 2] = dirs[:, 2], -dirs[:, 0], -dirs[:, 1]
    d = np.ascontiguousarray(d[::stride], np.float32)
    dists = np.ascontiguousarray((np.asarray(max_dists, np.float64) / voxel)[::stride], np.float32)
    n = d.shape[0]
    free = torch.zeros(int(np.prod(dims)), dtype=torch.int32, device=dev)
    BLOCK = 64
    grid = ((n + BLOCK - 1) // BLOCK,)
    kern[grid](float(o[0]), float(o[1]), float(o[2]),
               torch.tensor(d, device=dev), torch.tensor(dists, device=dev),
               free, float(voxel), int(dims[0]), int(dims[1]), int(dims[2]),
               BLOCK=BLOCK)
    torch.cuda.synchronize()
    return free.view(*dims).bool().cpu().numpy()


def cast_rays(origin_g, dirs, max_dists, gmin, voxel, dims, stride):
    """返回 bool 网格 free (dims 形状), True = 射线穿过的体素。
    origin_g: 相机原点(相机系, 通常零向量); dirs/max_dists: 相机系方向与截断距离。
    RGBD2OCC_BACKEND=gpu 且 torch CUDA 可用时走 GPU (triton 融合核首选,
    退化 torch 向量化, 再退化 CPU 核)。"""
    if os.environ.get("RGBD2OCC_BACKEND", "exact") == "gpu":
        try:
            import torch
            if torch.cuda.is_available():
                if _triton_fn():
                    return _cast_triton(origin_g, dirs, max_dists, gmin, voxel, dims, stride)
                return _cast_torch(origin_g, dirs, max_dists, gmin, voxel, dims, stride)
        except Exception:
            pass  # GPU 失败自动回退 CPU 核
    if _HAS_NUMBA:
        dists = np.asarray(max_dists, np.float64)
        raw = _cast_kernel(np.asarray(origin_g, np.float64),
                           np.ascontiguousarray(dirs[::stride], np.float64),
                           dists[::stride], np.asarray(gmin, np.float64),
                           float(voxel), int(dims[0]), int(dims[1]), int(dims[2]))
        return raw.astype(bool)

    # ---- numpy 向量化回退 ----
    free = np.zeros(tuple(dims), bool)
    o = (np.asarray(origin_g, np.float64) - gmin) / voxel
    d = np.empty_like(dirs)
    d[:, 0], d[:, 1], d[:, 2] = dirs[:, 2], -dirs[:, 0], -dirs[:, 1]
    d = np.ascontiguousarray(d[::stride])
    dist = (np.asarray(max_dists) / voxel)[::stride]
    v = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    n = len(v)
    cur = np.floor(o).astype(np.int64)[None, :] + np.zeros((n, 1), np.int64)
    step = np.sign(v).astype(np.int64)
    small = np.abs(v) <= 1e-12
    tDelta = np.where(small, np.inf, 1.0 / np.abs(v))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(v > 0, (cur + 1 - o) / v, (cur - o) / v)
    tMax = np.where(small, np.inf, np.abs(frac))
    t = np.zeros(n)
    ar = np.arange(n)
    dims_a = np.array(dims)
    max_iters = int(dist.max()) + 2 + int(np.abs(o).max()) + 2
    for _ in range(max_iters):
        act = t <= dist
        if not act.any():
            break
        idx = cur[act]
        inb = np.all((idx >= 0) & (idx < dims_a), 1)
        fi = idx[inb]
        free[fi[:, 0], fi[:, 1], fi[:, 2]] = True
        ax = np.argmin(tMax, axis=1)
        t = np.where(act, tMax[ar, ax], t)
        cur[ar, ax] += step[ar, ax]
        tMax[ar, ax] += tDelta[ar, ax]
        t[~np.isfinite(t)] = np.inf
    return free
