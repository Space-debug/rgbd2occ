# -*- coding: utf-8 -*-
"""GPU 占据射线追踪 v2 (gpu 层; float32 + GPU 反投影, 属近似层)。

传输优化 (对比"CPU 反投影后上传射线"的 v1):
- 上行: 原始 uint16 深度图 (~774KB) 代替射线数组 (1.2-4.6MB) —— 反投影在 GPU 做
- 下行: free 网格按位压缩 (640KB -> 80KB), CPU 端 np.unpackbits 还原
流程: 深度上传 -> GPU 反投影/归一化/stride 抽样 -> Triton DDA -> GPU 位压缩。
"""
import numpy as np

_uvs = {}        # (H, W) -> (u, v) 网格缓存
_KERNEL = None   # triton 核缓存


def _kernel():
    """惰性定义/编译 Triton DDA 核: 输入为已在 GPU 上的自车系方向 (N,3) 展平。"""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _dda(dirs_ptr, dists_ptr, free_ptr, o_x, o_y, o_z,
             voxel, d0, d1, d2):
        i = tl.program_id(0)          # 一 program 一射线
        dx = tl.load(dirs_ptr + i * 3 + 0)
        dy = tl.load(dirs_ptr + i * 3 + 1)
        dz = tl.load(dirs_ptr + i * 3 + 2)
        dist = tl.load(dists_ptr + i)
        cx = tl.floor(o_x).to(tl.int32)
        cy = tl.floor(o_y).to(tl.int32)
        cz = tl.floor(o_z).to(tl.int32)
        stepx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        stepy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        stepz = 1 if dz > 0 else (-1 if dz < 0 else 0)
        if dx != 0.0:
            tx = tl.abs(((cx + 1 - o_x) if dx > 0 else (o_x - cx)) / dx)
            tdx = tl.abs(1.0 / dx)
        else:
            tx = float("inf")
            tdx = float("inf")
        if dy != 0.0:
            ty = tl.abs(((cy + 1 - o_y) if dy > 0 else (o_y - cy)) / dy)
            tdy = tl.abs(1.0 / dy)
        else:
            ty = float("inf")
            tdy = float("inf")
        if dz != 0.0:
            tz = tl.abs(((cz + 1 - o_z) if dz > 0 else (o_z - cz)) / dz)
            tdz = tl.abs(1.0 / dz)
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

    _KERNEL = _dda
    return _dda


def _bitpack_download(free_gpu):
    """GPU bool 网格 -> 位压缩 uint8 下载 (体积 /8)。"""
    import torch
    flat = free_gpu.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.bool, device=flat.device)])
    bits = (flat.reshape(-1, 8).to(torch.uint8) * torch.tensor(
        [128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=flat.device)).sum(1)
    return bits.cpu().numpy()


def cast_rays_from_depth_gpu(raw, scale, fx, fy, cx, cy, gmin, voxel, dims,
                             stride, dmin=0.3, dmax=8.0):
    """原始 uint16 深度 -> free bool 网格 (全 GPU: 一次上传一次位压缩下载)。
    采样语义与 CPU 一致: 有效像素按行主序排列后 ::stride。"""
    import torch
    H, W = raw.shape
    dev = "cuda"
    raw_t = torch.tensor(raw, device=dev)
    dep = raw_t.to(torch.float32) * scale
    if (H, W) not in _uvs:
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        _uvs[(H, W)] = (torch.tensor(u, dtype=torch.float32, device=dev),
                        torch.tensor(v, dtype=torch.float32, device=dev))
    u, v = _uvs[(H, W)]
    valid = (dep > dmin) & (dep < dmax)
    ii = torch.flatnonzero(valid.reshape(-1))[::stride]
    z = dep.reshape(-1)[ii].to(torch.float64)
    uu, vv = (ii % W).to(torch.float64), (ii // W).to(torch.float64)
    xd = (uu - cx) / fx
    yd = (vv - cy) / fy
    # 相机系 (xd*z, yd*z, z) -> 自车系方向 (X=z, Y=-xd*z, Z=-yd*z), 归一化
    # float64 归一化后再转 float32 上 GPU: 步进决策与 CPU 高位一致
    X, Y, Z = z, -xd * z, -yd * z
    norm = torch.sqrt(X * X + Y * Y + Z * Z) + 1e-12
    dirs = torch.stack([X / norm, Y / norm, Z / norm], 1).reshape(-1).to(torch.float32)
    dists = norm.to(torch.float32)

    free = torch.zeros(int(np.prod(dims)), dtype=torch.int32, device=dev)
    n = int(dirs.numel() // 3)
    o_gx = (0.0 - gmin[0]) / voxel
    o_gy = (0.0 - gmin[1]) / voxel
    o_gz = (0.0 - gmin[2]) / voxel
    _kernel()[(n,)](dirs, dists / voxel, free, o_gx, o_gy, o_gz,
                    float(voxel), int(dims[0]), int(dims[1]), int(dims[2]))
    packed = _bitpack_download(free.bool())
    bits = np.unpackbits(packed)[:int(np.prod(dims))]
    return bits.reshape(dims).astype(bool)


def voxelize_and_cast_gpu(raw, scale, fx, fy, cx, cy, lab, gmin, voxel, dims,
                          stride, dmin=0.3, dmax=8.0):
    """全 GPU: 反投影 -> 表面体素化(占据格 + 标签 amax 散射) -> DDA 射线追踪。
    返回 (free bool 网格, 占据格子 flat 索引 int64[cpu], 格子标签 uint8[cpu])。
    传输: 上行 深度+标签 两张 uint8/16 图 (~1.5MB), 下行 仅占据格 (~50k*(8+1)B)。
    近似说明 (gpu 层): 标签冲突取最大类 id (CPU exact 为末次写入); 其余一致。"""
    import torch
    H, W = raw.shape
    dev = "cuda"
    raw_t = torch.tensor(raw, device=dev)
    dep = raw_t.to(torch.float32) * scale
    if (H, W) not in _uvs:
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        _uvs[(H, W)] = (torch.tensor(u, dtype=torch.float32, device=dev),
                        torch.tensor(v, dtype=torch.float32, device=dev))
    u, v = _uvs[(H, W)]
    valid = (dep > dmin) & (dep < dmax)
    ii = torch.nonzero(valid.reshape(-1)).reshape(-1)    # 表面体素化: 全部有效像素
    z = dep.reshape(-1)[ii].to(torch.float64)
    uu, vv = (ii % W).to(torch.float64), (ii // W).to(torch.float64)
    xd = (uu - cx) / fx
    yd = (vv - cy) / fy
    X, Y, Z = z, -xd * z, -yd * z                        # 自车系
    inb = ((X >= gmin[0]) & (X < gmin[0] + dims[0] * voxel) &
           (Y >= gmin[1]) & (Y < gmin[1] + dims[1] * voxel) &
           (Z >= gmin[2]) & (Z < gmin[2] + dims[2] * voxel))
    X, Y, Z, ii, z = X[inb], Y[inb], Z[inb], ii[inb], z[inb]
    ix = torch.floor((X - gmin[0]) / voxel).to(torch.int64)
    iy = torch.floor((Y - gmin[1]) / voxel).to(torch.int64)
    iz = torch.floor((Z - gmin[2]) / voxel).to(torch.int64)
    d12 = dims[1] * dims[2]
    occ_flat = ix * d12 + iy * dims[2] + iz

    # 射线 (stride 抽样, 与 CPU 采样序一致; z 已随 inb 过滤, 与 ii 对齐)
    ii_s = ii[::stride]
    z_s = z[::stride]
    xd_s = ((ii_s % W).to(torch.float64) - cx) / fx
    yd_s = ((ii_s // W).to(torch.float64) - cy) / fy
    Xs, Ys, Zs = z_s, -xd_s * z_s, -yd_s * z_s
    norm = torch.sqrt(Xs * Xs + Ys * Ys + Zs * Zs) + 1e-12

    free = torch.zeros(int(np.prod(dims)), dtype=torch.int32, device=dev)
    n_rays = int(Xs.numel())
    o_gx = (0.0 - gmin[0]) / voxel
    o_gy = (0.0 - gmin[1]) / voxel
    o_gz = (0.0 - gmin[2]) / voxel
    _kernel()[(n_rays,)](
        torch.stack([Xs / norm, Ys / norm, Zs / norm], 1).reshape(-1).to(torch.float32),
        (norm / voxel).to(torch.float32), free,
        o_gx, o_gy, o_gz,
        float(voxel), int(dims[0]), int(dims[1]), int(dims[2]))

    # 标签散射: 非零标签 amax (未标注 0 天然让位); 占据格去重
    lab_t = torch.tensor(lab, device=dev).reshape(-1)[ii]
    sem_cells = torch.zeros(int(np.prod(dims)), dtype=torch.uint8, device=dev)
    labeled = lab_t > 0
    sem_cells.scatter_reduce_(0, occ_flat[labeled],
                              lab_t[labeled].to(torch.uint8), reduce="amax")
    occ_u_all = torch.unique(occ_flat)
    torch.cuda.synchronize()
    # 下行只传占据格 (flat 索引 + 该格标签, ~50k 条)
    return (free.bool().cpu().numpy().reshape(dims),
            occ_u_all.cpu().numpy(),
            sem_cells[occ_u_all].cpu().numpy())


_KERNEL_BATCH = None


def _kernel_batch():
    """惰性编译批量 DDA 核: 一 program 一射线, fid 索引各自帧的网格切片。"""
    global _KERNEL_BATCH
    if _KERNEL_BATCH is not None:
        return _KERNEL_BATCH
    import triton
    import triton.language as tl

    @triton.jit
    def _dda_b(dirs_ptr, dists_ptr, free_ptr, fid_ptr,
               o_x, o_y, o_z, voxel, d0, d1, d2, prod):
        i = tl.program_id(0)
        fid = tl.load(fid_ptr + i)
        dx = tl.load(dirs_ptr + i * 3 + 0)
        dy = tl.load(dirs_ptr + i * 3 + 1)
        dz = tl.load(dirs_ptr + i * 3 + 2)
        dist = tl.load(dists_ptr + i)
        cx = tl.floor(o_x).to(tl.int32)
        cy = tl.floor(o_y).to(tl.int32)
        cz = tl.floor(o_z).to(tl.int32)
        stepx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        stepy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        stepz = 1 if dz > 0 else (-1 if dz < 0 else 0)
        if dx != 0.0:
            tx = tl.abs(((cx + 1 - o_x) if dx > 0 else (o_x - cx)) / dx)
            tdx = tl.abs(1.0 / dx)
        else:
            tx = float("inf")
            tdx = float("inf")
        if dy != 0.0:
            ty = tl.abs(((cy + 1 - o_y) if dy > 0 else (o_y - cy)) / dy)
            tdy = tl.abs(1.0 / dy)
        else:
            ty = float("inf")
            tdy = float("inf")
        if dz != 0.0:
            tz = tl.abs(((cz + 1 - o_z) if dz > 0 else (o_z - cz)) / dz)
            tdz = tl.abs(1.0 / dz)
        else:
            tz = float("inf")
            tdz = float("inf")
        t = 0.0
        d12 = d1 * d2
        while t <= dist:
            inb = ((cx >= 0) & (cx < d0)) & ((cy >= 0) & (cy < d1)) & ((cz >= 0) & (cz < d2))
            if inb:
                tl.store(free_ptr + fid * prod + cx * d12 + cy * d2 + cz, 1)
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

    _KERNEL_BATCH = _dda_b
    return _dda_b


def _bitpack_download_b(free_gpu):
    import torch
    flat = free_gpu.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.bool, device=flat.device)])
    bits = (flat.reshape(-1, 8).to(torch.uint8) * torch.tensor(
        [128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=flat.device)).sum(1)
    return bits.cpu().numpy().astype(np.uint8)


def voxelize_and_cast_batch_gpu(frames, gmin, voxel, dims, stride,
                                dmin=0.3, dmax=8.0):
    """多帧批量 GPU: B 帧射线拼一次 DDA launch (fid 索引各自网格切片)。
    frames: [{raw, lab, fx, fy, cx, cy, scale}, ...]
    返回 [(free bool, occ_cells int64, cell_labels uint8), ...], 与逐帧调用等价。"""
    import torch
    dev = "cuda"
    d0, d1, d2 = int(dims[0]), int(dims[1]), int(dims[2])
    prod = d0 * d1 * d2
    B = len(frames)

    dirs_l, dist_l, fid_l = [], [], []
    occflat_l, lab_l, fidlab_l = [], [], []
    for f, fr in enumerate(frames):
        raw = fr["raw"]
        H, W = raw.shape
        dep = torch.tensor(raw, device=dev).to(torch.float32) * fr["scale"]
        if (H, W) not in _uvs:
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            _uvs[(H, W)] = (torch.tensor(u, dtype=torch.float32, device=dev),
                            torch.tensor(v, dtype=torch.float32, device=dev))
        u, v = _uvs[(H, W)]
        valid = (dep > dmin) & (dep < dmax)
        ii = torch.nonzero(valid.reshape(-1)).reshape(-1)
        z_all = dep.reshape(-1)[ii].to(torch.float64)
        uu, vv = (ii % W).to(torch.float64), (ii // W).to(torch.float64)
        xd = (uu - fr["cx"]) / fr["fx"]
        yd = (vv - fr["cy"]) / fr["fy"]
        X, Y, Z = z_all, -xd * z_all, -yd * z_all
        inb = ((X >= gmin[0]) & (X < gmin[0] + d0 * voxel) &
               (Y >= gmin[1]) & (Y < gmin[1] + d1 * voxel) &
               (Z >= gmin[2]) & (Z < gmin[2] + d2 * voxel))
        X, Y, Z, ii, z_all = X[inb], Y[inb], Z[inb], ii[inb], z_all[inb]
        ix = torch.floor((X - gmin[0]) / voxel).to(torch.int64)
        iy = torch.floor((Y - gmin[1]) / voxel).to(torch.int64)
        iz = torch.floor((Z - gmin[2]) / voxel).to(torch.int64)
        d12 = d1 * d2
        occflat_l.append(ix * d12 + iy * d2 + iz)     # 局部 flat (无帧偏移)
        lab_l.append(torch.tensor(fr["lab"], device=dev).reshape(-1)[ii].to(torch.uint8))
        fidlab_l.append(torch.full((ii.numel(),), f, dtype=torch.int64, device=dev))
        ii_s = ii[::stride]
        z_s = z_all[::stride]
        xd_s = ((ii_s % W).to(torch.float64) - fr["cx"]) / fr["fx"]
        yd_s = ((ii_s // W).to(torch.float64) - fr["cy"]) / fr["fy"]
        Xs, Ys, Zs = z_s, -xd_s * z_s, -yd_s * z_s
        norm = torch.sqrt(Xs * Xs + Ys * Ys + Zs * Zs) + 1e-12
        dirs_l.append(torch.stack([Xs / norm, Ys / norm, Zs / norm], 1)
                      .reshape(-1).to(torch.float32))
        dist_l.append(norm.to(torch.float32))
        fid_l.append(torch.full((Xs.numel(),), f, dtype=torch.int64, device=dev))

    dirs = torch.cat(dirs_l)
    dists = torch.cat(dist_l)
    fids = torch.cat(fid_l)
    n_rays = int(dirs.numel() // 3)
    occ_flat = torch.cat(occflat_l)
    labs = torch.cat(lab_l)
    fids_lab = torch.cat(fidlab_l)

    free = torch.zeros(B * prod, dtype=torch.int32, device=dev)
    o_gx = (0.0 - gmin[0]) / voxel
    o_gy = (0.0 - gmin[1]) / voxel
    o_gz = (0.0 - gmin[2]) / voxel
    _kernel_batch()[(n_rays,)](dirs, dists / voxel, free, fids,
                               o_gx, o_gy, o_gz, float(voxel), d0, d1, d2, prod)
    torch.cuda.synchronize()

    # 标签散射: 13 次按类 index_put (同格同值 -> 确定性, 高 id 后写 = amax 语义)
    sem_cells = torch.zeros(B * prod, dtype=torch.uint8, device=dev)
    for L in range(1, 14):
        m = labs == L
        if m.any():
            sem_cells[occ_flat[m] + fids_lab[m] * prod] = L
    occ_u = torch.unique(occ_flat + fids_lab * prod)
    torch.cuda.synchronize()

    out = []
    for f in range(B):
        packed = _bitpack_download_b(free[f * prod:(f + 1) * prod].bool())
        bits = np.unpackbits(packed)[:prod]
        free_f = bits.reshape(dims).astype(bool)
        g = occ_u.cpu().numpy()
        cells_local = g - f * prod
        m = (cells_local >= 0) & (cells_local < prod)
        out.append((free_f, cells_local[m], sem_cells.cpu().numpy()[g[m]]))
    return out
