# -*- coding: utf-8 -*-
"""GPU 点云线 (gpu 层, 近似): 深度 -> 点云各阶段的 torch 实现。

与 CPU exact 路径的差异 (文档化):
- median 含无效像素 (0 参与计数) 而非 NaN 剔除 -> 边缘窗口中值略降
- downsample 的浮点求和顺序不同 -> 质心/颜色可有 ULP 级差异
- SOR/斑点为整数邻域计数, 与 CPU 精确一致
"""
import numpy as np
import torch

_grids = {}   # (H, W) -> (u, v) float64 网格缓存


def _uv(H, W, dev):
    if (H, W) not in _grids:
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        _grids[(H, W)] = (torch.tensor(u, dtype=torch.float64, device=dev),
                          torch.tensor(v, dtype=torch.float64, device=dev))
    return _grids[(H, W)]


def median5x5_gpu(raw_u16, dev="cuda"):
    """5x5 中值 (含无效 0 像素, 近似) -> float64 米制。"""
    import torch
    import torch.nn.functional as F
    raw = torch.tensor(raw_u16, dtype=torch.float32, device=dev)
    w = raw.unsqueeze(0).unsqueeze(0)
    win = F.unfold(w, kernel_size=5, padding=2)             # (1, 25, H*W)
    med = win.median(dim=1).values.reshape(raw.shape)
    return med.double()


def gradient_filter_gpu(depth, dmin=0.3, grad_thr=0.05):
    """梯度剔除 (与 CPU np.gradient 同式: 中心差分+边缘单侧)。"""
    gy, gx = torch.gradient(torch.where(depth > dmin, depth,
                                        torch.full_like(depth, float("nan"))))
    g = torch.sqrt(gy * gy + gx * gx)
    ok = (depth > dmin) & torch.nan_to_num(g, nan=0.0) < grad_thr
    return torch.where(ok, depth, torch.zeros_like(depth))


def deproject_gpu(depth, fx, fy, cx, cy, dmin=0.3, dmax=8.0):
    """米制深度 -> 自车系点云 (N,3) float64 + 有效掩膜 (bool HxW)。"""
    H, W = depth.shape
    u, v = _uv(H, W, depth.device)
    m = torch.isfinite(depth) & (depth > dmin) & (depth < dmax)
    z = depth[m]
    xd = (u[m] - cx) / fx
    yd = (v[m] - cy) / fy
    P = torch.stack([z, -xd * z, -yd * z], 1)               # 自车系 X前/Y左/Z上
    return P, m




def _nbr27_gather(cnt_grid, cells, d0, d1, d2):
    """占据格 27 邻域计数和(含自身), 单次 (M,27) gather。
    要求 cells 距各维边界 >=1 (由调用方 ki+1 保证), 越界读不到 —— 网格
    稀疏(占据 ~6%)时比稠密 conv3d 快数倍; 整数求和与 conv3d+零padding 精确一致。"""
    d12 = d1 * d2
    ix = cells // d12
    r = cells - ix * d12
    iy = r // d2
    iz = r - iy * d2
    offs = torch.tensor([(dx, dy, dz) for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                        dtype=torch.int64, device=cells.device)       # (27,3)
    shifted = (torch.stack([ix, iy, iz], 1)[:, None, :]
               + offs[None, :, :])                                     # (M,27,3)
    sflat = shifted[..., 0] * d12 + shifted[..., 1] * d2 + shifted[..., 2]
    return cnt_grid[sflat.T].sum(0)


def sor_gpu(P, R=0.03, min_nbr=6):
    """SOR: 30mm 网格计数 + 3x3 邻域卷积 (整数计数, float32 精确)。"""
    import torch.nn.functional as F
    key = torch.floor(P / R).to(torch.int64)
    mn = key.min(0).values
    ki = key - mn + 1                                   # +1 margin: ±1 邻域不出界
    d0, d1, d2 = (ki.max(0).values + 2).tolist()
    d12 = d1 * d2
    flat = ki[:, 0] * d12 + ki[:, 1] * d2 + ki[:, 2]
    grid = torch.bincount(flat, minlength=d0 * d1 * d2)  # int64 稠密计数
    cells, inv = torch.unique(flat, return_inverse=True) # 占据格 ~6%
    nbr = _nbr27_gather(grid, cells, d0, d1, d2)
    keep_cells = nbr >= min_nbr
    return keep_cells[inv]


def speckle_gpu(P, vox=0.05, gmin=(-4.0, 0.3, -1.5), min_nbr=4):
    """体素斑点过滤: 占据体素 27 邻域计数 >= min_nbr 保留。"""
    import torch.nn.functional as F
    idx = torch.floor((P - torch.tensor(gmin, dtype=torch.float64, device=P.device)) / vox).to(torch.int64)
    d0, d1, d2 = (idx.max(0).values + 3).tolist()       # +1 现有 +1 margin +1 上界
    # 负索引取模回绕: 复现 numpy occ[负索引] 的历史语义 (CPU 路径依赖此行为)
    idx = torch.stack([(idx[:, 0] % (d0 - 2)) + 1,
                       (idx[:, 1] % (d1 - 2)) + 1,
                       (idx[:, 2] % (d2 - 2)) + 1], 1)   # 回绕到 [1, d-2], margin 保证
    d12 = d1 * d2
    flat = idx[:, 0] * d12 + idx[:, 1] * d2 + idx[:, 2]
    occ = torch.bincount(flat, minlength=d0 * d1 * d2)
    cells, inv = torch.unique(flat, return_inverse=True)
    nbr = _nbr27_gather(occ, cells, d0, d1, d2)
    keep_cells = nbr >= min_nbr                          # cells 全部占据, occ>0 恒真
    return keep_cells[inv]


def downsample_gpu(P, C, vox):
    """体素降采样: 质心 + 颜色均值 (GPU 求和顺序与 CPU 不同, 近似)。
    实现: unique 压缩到实际格数(~4万)后一次 index_add —— 此前对全网格槽
    (~8M) 做 6 次 float64 bincount 是点云线最大单点瓶颈(5.6ms/帧)。"""
    idx = torch.floor(P / vox).to(torch.int64)
    idx -= idx.min(0).values
    dims = (idx.max(0).values + 1).tolist()
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    uniq, inv = torch.unique(flat, return_inverse=True)
    n = uniq.numel()
    cnt = torch.bincount(inv, minlength=n).to(P.dtype)
    Ps = torch.zeros((n, 3), dtype=P.dtype, device=P.device)
    Ps.index_add_(0, inv, P)
    Ps /= cnt[:, None]
    Cs = torch.zeros((n, C.shape[1]), dtype=P.dtype, device=P.device)
    Cs.index_add_(0, inv, C.to(P.dtype))
    Cs = Cs / cnt[:, None]
    return Ps, torch.round(Cs).to(torch.uint8)


def depth_to_points_gpu(raw_u16, img, scale, fx, fy, cx, cy, dmin=0.3, dmax=8.0):
    """GPU 全链 (含 SOR/斑点), 返回 CPU numpy。"""
    P, C = _points_gpu_tensors(raw_u16, img, scale, fx, fy, cx, cy, dmin, dmax)
    return P.double().cpu().numpy(), C.cpu().numpy()


def depth_to_points_downsampled_gpu(raw_u16, img, scale, fx, fy, cx, cy, vox,
                                    dmin=0.3, dmax=8.0):
    """GPU 全链 + 体素降采样, 一步到位 (点云线主入口)。返回 CPU numpy P/C。"""
    P, C = _points_gpu_tensors(raw_u16, img, scale, fx, fy, cx, cy, dmin, dmax)
    P, C = downsample_gpu(P.double(), C.double(), vox)
    # f32 下载: write_bin 落盘本就 cast float32, f64->f32 舍入在 GPU/CPU 端一致
    return P.to(torch.float32).cpu().numpy(), C.cpu().numpy()


def _points_gpu_tensors(raw_u16, img, scale, fx, fy, cx, cy, dmin=0.3, dmax=8.0):
    """GPU 全链 (保持 GPU 张量): median -> 梯度滤波 -> 反投影 -> SOR -> 斑点。"""
    import torch
    dev = "cuda"
    H, W = raw_u16.shape
    raw = torch.tensor(raw_u16, device=dev)
    dep = raw.to(torch.float64) * scale
    valid = (dep > dmin) & (dep < dmax)
    dep32 = dep.to(torch.float32)                    # median 用 FP32: 输出与 FP64 差 0.000mm,
    pad = torch.where(valid, dep32,                   # FP64 nanmedian 在消费级卡上是 64x 减速
                      torch.full_like(dep32, float("nan")))
    pad = torch.nn.functional.pad(pad.unsqueeze(0).unsqueeze(0), (2, 2, 2, 2),
                                  mode="constant", value=float("nan")).reshape(H + 4, W + 4)
    win = pad.unfold(0, 5, 1).unfold(1, 5, 1)                # (H, W, 5, 5)
    med = win.reshape(*win.shape[:2], -1).nanmedian(dim=-1).values.to(torch.float64)
    d1 = torch.where(valid, torch.where(torch.isfinite(med), med, dep),
                     torch.zeros_like(dep))   # 后续梯度/反投影仍走 float64
    gy, gx = torch.gradient(torch.where(d1 > dmin, d1, torch.full_like(d1, float("nan"))))
    g = torch.sqrt(gy * gy + gx * gx)
    d1 = torch.where((d1 > dmin) & (torch.nan_to_num(g, nan=0.0) < 0.05), d1,
                     torch.zeros_like(d1))
    P, m = deproject_gpu(d1, fx, fy, cx, cy, dmin, dmax)
    C = torch.tensor(img.reshape(-1, 3), device=dev)[m.reshape(-1)]
    keep = sor_gpu(P)                                      # C 不参与 SOR 决策
    P, C = P[keep], C[keep]
    G = torch.stack([-P[:, 1], P[:, 0], P[:, 2]], 1)      # ego -> 网格(X右,Y前,Z上)
    keep = speckle_gpu(G)
    G3 = G[keep]
    P = torch.stack([G3[:, 1], -G3[:, 0], G3[:, 2]], 1)   # 网格 -> ego (逆变换)
    C = C[keep]
    return P, C
