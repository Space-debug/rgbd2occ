# -*- coding: utf-8 -*-
"""CloudCompare 导出: bin / labels.npz / 3D 框 -> PLY (rgbd2occ export 子命令)。

把转换产物导出为 CloudCompare 可直接打开的 .ply (binary_little_endian):
  点云   samples/LIDAR_TOP/<split>/img-XXXXXX.pcd.bin -> 带亮度着色的点云
  占据GT gts/<scene>/<token>/labels.npz               -> 按语义类着色的体素点云
  3D框   v1.0-sunrgbd-<split>/sample_annotation.json  -> 逐帧彩色线框点云
坐标系 = 自车系 X前/Y左/Z上, 与 CloudCompare 默认 Z 轴向上一致, 打开即是正视角。

统一入口 (推荐):
  rgbd2occ export points D:/Datasets/sunrgbd_nuscenes_v3/samples/LIDAR_TOP/train --limit 5
  rgbd2occ export occ D:/Datasets/sunrgbd_nuscenes_v3/gts --limit 4 --what all
  rgbd2occ export boxes D:/Datasets/sunrgbd_nuscenes_v3 --split train --names img-000001
  rgbd2occ export preview D:/Datasets/sunrgbd_nuscenes_v3 --names img-000001  (2D+3D框叠加图)
  rgbd2occ export info D:/Datasets/sunrgbd_nuscenes_v3
本文件也可独立运行 (等价): python tools/export_ply.py points|occ|boxes|info ...
输出默认写到当前目录新建的 rgbd2occ_export/ (--out 可改)。
"""
import argparse
import colorsys
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.render_bev import CLASS_COLORS  # noqa: E402

# 语义着色: 0=others 灰, 1..13 类, 17=free 浅绿 (与 BEV 质检图同源同约定)
SEM_COLORS = [(128, 128, 128)] + list(CLASS_COLORS[1:])
FREE_COLOR = (140, 185, 140)
DEFAULT_OUT = "rgbd2occ_export"     # 相对当前工作目录, 运行时新建


# ---------------- PLY 写出/读回 ----------------

def write_ply(path, xyz, rgb, intensity=None):
    """binary_little_endian PLY: float32 x,y,z + uchar red,green,blue
    (+ 点云线附带 uchar intensity 标量场, 供 CloudCompare 按标量重新着色)。"""
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if intensity is not None:
        fields.append(("intensity", "u1"))
    hdr = ["ply", "format binary_little_endian 1.0", "element vertex %d" % n,
           "property float x", "property float y", "property float z",
           "property uchar red", "property uchar green", "property uchar blue"]
    if intensity is not None:
        hdr.append("property uchar intensity")
    hdr.append("end_header")
    arr = np.zeros(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if intensity is not None:
        arr["intensity"] = intensity
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        arr.tofile(f)
    return n


def write_ply_mesh(path, xyz, rgb, faces):
    """顶点+面 PLY (quad/triangle 混合): CloudCompare 直接渲染为带色网格。
    faces: 顶点索引列表的列表 (长度 3 或 4)。"""
    n, m = len(xyz), len(faces)
    hdr = ["ply", "format binary_little_endian 1.0", "element vertex %d" % n,
           "property float x", "property float y", "property float z",
           "property uchar red", "property uchar green", "property uchar blue",
           "element face %d" % m, "property list uchar int vertex_indices",
           "end_header"]
    arr = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        arr.tofile(f)
        for face in faces:
            f.write(bytes([len(face)]))
            f.write(np.asarray(face, "<i4").tobytes())
    return n


def read_ply(path, with_faces=False):
    """读回本工具写出的 PLY (测试/校验用)。
    返回 {属性名: 数组}; with_faces=True 时附带 "faces" (list[int] 列表)。"""
    props, faces, n = [], [], 0
    elem = None
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                elem = line.split()[1]
                if elem != "vertex":
                    continue
            if elem == "vertex" and line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif elem == "vertex" and line.startswith("property"):
                props.append(line.split()[1:])          # [类型, 名]
            elif line == "end_header":
                break
        dt = np.dtype([(name, {"float": "<f4", "uchar": "u1"}[t]) for t, name in props])
        arr = np.frombuffer(f.read(dt.itemsize * n), dt)
        if with_faces:
            raw = f.read()
    out = {name: arr[name].copy() for _, name in props}
    if with_faces:
        pos = 0
        while pos < len(raw):
            k = raw[pos]
            pos += 1
            faces.append(list(np.frombuffer(raw, "<i4", k, pos)))
            pos += 4 * k
        out["faces"] = faces
    return out


def _hash_color(name):
    """类别名 -> 稳定区分色 (黄金比例散布色相; 13 类内的走 CLASS_COLORS)。"""
    if name in CLS_13:
        return CLASS_COLORS[CLS_13.index(name) + 1]
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    r, g, b = colorsys.hsv_to_rgb((h * 0.61803398875) % 1.0, 0.75, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


CLS_13 = ["bed", "books", "ceiling", "chair", "floor", "furniture", "objects",
          "picture", "sofa", "table", "tv", "wall", "window"]


# ---------------- 输入收集 ----------------

def _split_names(names):
    """--names 归一: 兼容逗号分隔的单词 ("img-001,img-002" 在部分 shell 下是单参数)。"""
    return [t for arg in names for t in arg.split(",") if t]


def _want(name, names):
    """--names 过滤: 接受 img-000123 / 123 / 文件名子串 / token 前缀。"""
    if not names:
        return True
    for t in names:
        if t.isdigit():
            if name == "img-%06d" % int(t):
                return True
        elif t in name:
            return True
    return False


def _collect_bins(src, limit, names):
    """points 输入: bin 文件或含 bin 的目录 -> 排序后的路径列表。"""
    if os.path.isfile(src):
        return [src]
    bins = sorted(f for f in os.listdir(src) if f.endswith(".pcd.bin"))
    bins = [os.path.join(src, f) for f in bins if _want(f[:-8], names)]
    return bins[:limit] if limit else bins


def _collect_npz(src, limit, names):
    """occ 输入: labels.npz 文件 / token 目录 / scene 目录 / gts 根 -> 路径列表。"""
    if os.path.isfile(src):
        return [src]
    direct = os.path.join(src, "labels.npz")
    if os.path.exists(direct):
        return [direct]
    found = []
    for sub in sorted(os.listdir(src)):
        d = os.path.join(src, sub)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "labels.npz")
        if os.path.exists(p):
            if _want(sub, names):
                found.append(p)
            continue
        for tok in sorted(os.listdir(d)):          # gts 根: 再下钻一层 scene/token
            p2 = os.path.join(d, tok, "labels.npz")
            if os.path.exists(p2) and _want(tok, names):
                found.append(p2)
    return found[:limit] if limit else found


# ---------------- 导出模式 ----------------

DEPTH_SCALE = 1.0 / 6553.5      # 与 to_nuscenes 一致的深度解码尺度
DOWNSAMPLE_VOX = 0.03           # 与点云线一致的降采样体素


def _find_pkg_root(path):
    """从产物文件向上找数据包根 (含 intrinsics_per_frame.json 的目录)。"""
    p = os.path.dirname(os.path.abspath(path))
    for _ in range(5):
        if os.path.exists(os.path.join(p, "intrinsics_per_frame.json")):
            return p
        p = os.path.dirname(p)
    return None


def _regen_rgb(bin_p, raw_root):
    """重跑 CPU 反投影管线为点云着真彩色 (bin 只存亮度)。
    返回 (P float32, C uint8); 原始数据/内参不可得时返回 None。"""
    from PIL import Image
    from common import depth_to_points, voxel_downsample
    root = _find_pkg_root(bin_p)
    if root is None:
        print("跳过着色: 未找到 intrinsics_per_frame.json (需在数据包内)", bin_p)
        return None
    name = os.path.basename(bin_p)[:-8]
    split = os.path.basename(os.path.dirname(bin_p))
    intr = json.load(open(os.path.join(root, "intrinsics_per_frame.json"),
                          encoding="utf-8"))
    meta = intr.get("%s/%s" % (split, name))
    dep_dir = ("sunrgbd_train_depth" if split == "train" else "sunrgbd_test_depth")
    img_dir = ("SUNRGBD-train_images" if split == "train" else "SUNRGBD-test_images")
    dep_p = os.path.join(raw_root, dep_dir, "%d.png" % int(name.split("-")[1]))
    jpg_p = os.path.join(raw_root, img_dir, name + ".jpg")
    if meta is None or not (os.path.exists(dep_p) and os.path.exists(jpg_p)):
        print("跳过着色: 缺 %s / %s / 内参" % (dep_p, jpg_p))
        return None
    raw = np.array(Image.open(dep_p))
    img = np.array(Image.open(jpg_p).convert("RGB"))
    if raw.shape != img.shape[:2]:
        img = np.array(Image.fromarray(img).resize((raw.shape[1], raw.shape[0])))
    K = meta["K_native"]
    P, C = depth_to_points(img, raw.astype(np.float64) * DEPTH_SCALE,
                           K[0][0], K[0][2], K[1][2], raw=raw, scale=DEPTH_SCALE)
    if DOWNSAMPLE_VOX and len(P):
        P, C = voxel_downsample(P.astype(np.float64), C.astype(np.float64),
                                DOWNSAMPLE_VOX)
    return P.astype(np.float32), C.astype(np.uint8)


def mode_points(args):
    out_dir = args.out or os.path.join(os.getcwd(), DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)
    names = _split_names(args.names)
    bins = _collect_bins(args.src, args.limit, names)
    total = 0
    if args.with_rgb and not args.raw_root:
        try:
            from config import dataset_paths
            args.raw_root = dataset_paths("sunrgbd")["raw_root"]
        except Exception:
            pass
    if args.with_rgb and not args.raw_root:
        print("--with-rgb 需要 --raw-root 指向 SUN RGB-D 原始数据 (深度 png + 原图 jpg)")
    for bin_p in bins:
        pts = np.fromfile(bin_p, np.float32).reshape(-1, 5)
        if not len(pts):
            print("跳过空点云", bin_p)
            continue
        xyz, rgb = pts[:, :3], None
        inten = np.clip(pts[:, 3] * 255.0, 0, 255).astype(np.uint8)
        if args.with_rgb and args.raw_root:
            got = _regen_rgb(bin_p, args.raw_root)
            if got is not None:
                xyz, rgb = got
                if len(xyz) != len(pts):
                    print("注: %s 重投影 %d 点 vs bin %d 点 (后端变体差异, 仅影响可视化)"
                          % (os.path.basename(bin_p), len(xyz), len(pts)))
                inten = np.clip(rgb.mean(1), 0, 255).astype(np.uint8)
        if rgb is None:
            rgb = (np.repeat(inten[:, None], 3, 1) if args.color == "intensity"
                   else np.full((len(pts), 3), 255, np.uint8))
        dst = os.path.join(out_dir, os.path.basename(bin_p)[:-8] + ".ply")
        n = write_ply(dst, xyz, rgb, intensity=inten)
        total += n
        print("%s  %d 点" % (dst, n))
    print("共 %d 文件 %d 点 -> %s" % (len(bins), total, out_dir))


def mode_occ(args):
    out_dir = args.out or os.path.join(os.getcwd(), DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)
    npzs = _collect_npz(args.src, args.limit, _split_names(args.names))
    total = 0
    for npz_p in npzs:
        with np.load(npz_p) as d:
            sem = d["semantics"]
            mask = (d.get("mask_camera") if args.mask == "camera" else d.get("mask_lidar"))
            mask = np.ones_like(sem, np.uint8) if mask is None else mask
            # 体素元数据 (0.9.6 起 npz 自带); 旧 npz 无键时按约定假设
            if "voxel" in d and "gmin" in d:
                voxel = float(d["voxel"])
                gmin = np.asarray(d["gmin"], np.float64)
            else:
                voxel = args.voxel
                H0, W0 = sem.shape[:2]
                gmin = np.array([-W0 * voxel / 2, -H0 * voxel / 2, args.zmin])
        H, W, D = sem.shape
        sel = np.zeros(sem.shape, bool)
        if args.what in ("occupied", "all"):
            sel |= (sem < 17) & (mask > 0)
        if args.what in ("free", "all"):
            sel |= (sem == 17) & (mask > 0)
        if not sel.any():
            print("跳过空占据", npz_p)
            continue
        idx = np.argwhere(sel)
        xyz = (idx + 0.5) * voxel + gmin
        cls = sem[sel]
        # 调色板: 0..13 语义, 14=free (语义 17 归并到 free 行)
        pal = np.array(SEM_COLORS + [FREE_COLOR], np.uint8)
        rgb = pal[np.minimum(cls, 14)]
        tok = os.path.basename(os.path.dirname(npz_p))            # token 目录名
        scene = os.path.basename(os.path.dirname(os.path.dirname(npz_p)))
        dst = os.path.join(out_dir, "%s_%s.ply" % (scene or "occ", tok))
        if args.style == "cube":
            verts, vcols, faces = _voxel_cubes(xyz, rgb, voxel * args.cube_scale)
            n = write_ply_mesh(dst, verts, vcols, faces)
            print("%s  %d 体素 (立方体网格, %d 顶点 %d 面)" % (dst, len(xyz), n, len(faces)))
            total += len(xyz)
        else:
            n = write_ply(dst, xyz, rgb)
            total += n
            print("%s  %d 体素" % (dst, n))
    print("共 %d 文件 %d 体素 -> %s" % (len(npzs), total, out_dir))


def _box_corners(tr, quat, size):
    """nuScenes 框 -> 8 角点 (自车系)。size=[w,l,h], 角序同 devkit Box.corners。"""
    w, l, h = size
    q = np.array(quat, np.float64)          # (w, x, y, z)
    q = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]),
         2 * (q[1] * q[3] + q[0] * q[2])],
        [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2),
         2 * (q[2] * q[3] - q[0] * q[1])],
        [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
         1 - 2 * (q[1] ** 2 + q[2] ** 2)]])
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return np.asarray(tr, np.float64) + loc @ R.T


_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _norm_frame(t):
    """帧号归一: 123 / img-123 -> img-000123。"""
    return ("img-%06d" % int(t)) if t.isdigit() else t


def _edge_tube(A, B, r):
    """边 (A,B) -> 细四棱管: 8 顶点 + 4 个 quad (局部索引), 连续线框用。"""
    d = B - A
    L = np.linalg.norm(d)
    if L < 1e-9:
        return A.reshape(1, 3), []
    d = d / L
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, up)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    ring_u, ring_v = u * r, v * r
    offs = [ring_u, ring_v, -ring_u, -ring_v]
    verts = np.array([A + o for o in offs] + [B + o for o in offs])
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return verts, faces


def mode_boxes(args):
    ver = args.src
    if not os.path.basename(ver).startswith("v1.0-"):
        ver = os.path.join(ver, "v1.0-sunrgbd-%s" % args.split)
    rd = lambda n: json.load(open(os.path.join(ver, n), encoding="utf-8"))
    anns = rd("sample_annotation.json")
    inst = {i["token"]: i for i in rd("instance.json")}
    cats = {c["token"]: c["name"].split(".")[-1] for c in rd("category.json")}
    samples = {s["token"]: s for s in rd("sample.json")}
    # sample_token -> 帧号 (CAM_FRONT sample_data 文件名反查)
    tok2num = {}
    for e in rd("sample_data.json"):
        if "CAM_FRONT" in e.get("filename", ""):
            tok2num[e["sample_token"]] = int(
                e["filename"].split("img-")[1].split(".")[0])

    names = _split_names(args.names)
    if names:
        want = {_norm_frame(t) for t in names}
        sel = [a for a in anns
               if ("img-%06d" % tok2num.get(a["sample_token"], -1)) in want]
    else:
        nums = sorted({tok2num[a["sample_token"]] for a in anns})
        keep = set(nums[:args.limit] if args.limit else nums)
        sel = [a for a in anns if tok2num.get(a["sample_token"]) in keep]
    if not sel:
        print("无匹配框 (ver=%s, names=%s)" % (ver, args.names))
        return

    out_dir = args.out or os.path.join(os.getcwd(), DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)
    by_frame = {}
    for a in sel:
        by_frame.setdefault(a["sample_token"], []).append(a)
    cand = os.path.dirname(ver)
    root = cand if os.path.isdir(os.path.join(cand, "samples")) else ver
    for st, alist in sorted(by_frame.items(), key=lambda kv: tok2num.get(kv[0], 0)):
        num = tok2num.get(st)
        dst = os.path.join(out_dir, "boxes_points_%s_img-%06d.ply" % (
            os.path.basename(ver).replace("v1.0-sunrgbd-", ""), num))
        bin_p = os.path.join(root, "samples", "LIDAR_TOP", args.split,
                             "img-%06d.pcd.bin" % num)
        if not os.path.exists(bin_p):
            print("跳过 (无 bin, 框必须并入点云):", bin_p)
            continue
        pts = np.fromfile(bin_p, np.float32).reshape(-1, 5)
        inten = np.clip(pts[:, 3] * 255.0, 0, 255).astype(np.uint8)
        p_xyz, p_rgb = pts[:, :3], np.repeat(inten[:, None], 3, 1)
        if getattr(args, "rgb", False) and getattr(args, "raw_root", None):
            got = _regen_rgb(bin_p, args.raw_root)
            if got is not None:
                p_xyz, p_rgb = got
        xyz, rgb = [p_xyz], [p_rgb]
        faces, base = [], len(xyz[0])
        n_pts = base
        for a in alist:
            cname = cats.get(inst.get(a["instance_token"], {}).get("category_token", ""),
                             "objects")
            col = _hash_color(cname)
            corners = _box_corners(a["translation"], a["rotation"], a["size"])
            for i, j in _EDGES:
                verts, tube = _edge_tube(corners[i], corners[j], args.radius)
                xyz.append(verts)
                rgb.append(np.tile(np.array(col, np.uint8), (len(verts), 1)))
                faces.extend([[a0 + base, a1 + base, a2 + base, a3 + base]
                              for a0, a1, a2, a3 in tube])
            base += len(verts)
        n = write_ply_mesh(dst, np.concatenate(xyz), np.concatenate(rgb), faces)
        print("%s  %d 框 (含原始点云 %d 点), %d 顶点 %d 面" % (
            dst, len(alist), n_pts, n, len(faces)))


def _rows(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []


# ---------------- preview: 2D/3D 标注叠加到相机图像 ----------------

# 投影约定同 datasets/sunrgbd/fill_detection.box_proj_iou:
# 自车系 -> 工具箱相机系 (x右,y前,z上); 图像 v 向下故 z 取负
_M_ego2cam = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def _project_corners(tr, Re, hf, K):
    """3D 框 8 角点 (ego 系) -> 像素 (u,v); 任一角点在相机后方时返回 None。"""
    cs = np.array([tr + Re @ np.array([dx, dy, dz]) * hf
                   for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    pc = cs @ _M_ego2cam.T
    fwd = pc[:, 1]
    if (fwd <= 0.05).any():
        return None
    u = K[0][0] * pc[:, 0] / fwd + K[0][2]
    v = -K[1][1] * pc[:, 2] / fwd + K[1][2]
    return np.stack([u, v], 1)


def mode_preview(args):
    """相机图 + 2D gt 框(绿) + 3D 框投影(按类着色) -> PNG。数据全部来自
    数据包内文件: detection_meta_cache.json(框/2D) + intrinsics_per_frame.json(K)
    + samples/CAM_FRONT 原图 —— 不依赖原始 SUN RGB-D 数据集。"""
    from PIL import Image, ImageDraw
    root = args.src
    cache_p = os.path.join(root, "detection_meta_cache.json")
    intr_p = os.path.join(root, "intrinsics_per_frame.json")
    if not (os.path.exists(cache_p) and os.path.exists(intr_p)):
        sys.exit("缺 detection_meta_cache.json / intrinsics_per_frame.json;"
                 " 先跑: rgbd2occ convert sunrgbd detection")
    cache = json.load(open(cache_p, encoding="utf-8"))
    intr = json.load(open(intr_p, encoding="utf-8"))
    out_dir = args.out or os.path.join(os.getcwd(), DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)

    nums = sorted(int(k.split("/")[1]) for k in cache
                  if k.startswith(args.split + "/"))
    names = _split_names(args.names)
    if names:
        want = {_norm_frame(t) for t in names}
        nums = [n for n in nums if ("img-%06d" % n) in want]
    elif args.limit:
        nums = nums[:args.limit]

    n_draw = 0
    for num in nums:
        key = "%s/%06d" % (args.split, num)
        rec, meta = cache[key], intr.get("%s/img-%06d" % (args.split, num))
        jpg = os.path.join(root, "samples", "CAM_FRONT", args.split,
                           "img-%06d.jpg" % num)
        if meta is None or not os.path.exists(jpg):
            print("跳过 (缺内参或图像):", key)
            continue
        img = Image.open(jpg).convert("RGB")
        d = ImageDraw.Draw(img)
        Rt = _M_ego2cam.T @ np.asarray(rec["Rtilt"], np.float64).T  # 重力系->ego
        K = meta["K_native"]
        n2 = n3 = 0
        for b in rec["boxes"]:
            if not b.get("bb2d"):
                continue
            x, y, w, h = b["bb2d"]
            d.rectangle([x, y, x + w, y + h], outline=(0, 220, 0), width=2)
            d.text((x + 2, y + 2), b["cls"], fill=(0, 220, 0))
            n2 += 1
        if args.with_3d:
            for b in rec["boxes"]:
                tr = Rt @ np.asarray(b["centroid"], np.float64)
                Re = Rt @ np.asarray(b["basis"], np.float64)
                uv = _project_corners(tr, Re, np.asarray(b["coeffs"], np.float64) / 2, K)
                if uv is None:
                    continue
                col = _hash_color(b["cls"].lower())
                for i, j in _EDGES:
                    d.line([tuple(uv[i]), tuple(uv[j])], fill=col, width=2)
                d.text((float(uv[:, 0].min()) + 2,
                        max(0.0, float(uv[:, 1].min()) - 12)), b["cls"], fill=col)
                n3 += 1
        dst = os.path.join(out_dir, "preview_%s_img-%06d.png" % (args.split, num))
        img.save(dst)
        n_draw += 1
        print("%s  2D框 %d, 3D框 %d" % (dst, n2, n3))
    print("共 %d 帧 -> %s" % (n_draw, out_dir))


def _token_of(root, split, name):
    """帧号 -> (sample_token, gts scene 目录名); 表缺失返回 (None, None)。"""
    tb = os.path.join(root, "v1.0-sunrgbd-%s" % split)
    try:
        scenes = {s["token"]: s.get("name", "gts")
                  for s in json.load(open(os.path.join(tb, "scene.json")))}
        sm = {s["token"]: s["scene_token"]
              for s in json.load(open(os.path.join(tb, "sample.json")))}
        for e in json.load(open(os.path.join(tb, "sample_data.json"))):
            if "CAM_FRONT" in e.get("filename", "") \
                    and e["filename"].endswith(name + ".jpg"):
                st = e["sample_token"]
                return st, scenes.get(sm.get(st), "gts")
    except OSError:
        pass
    return None, None


def mode_demo(args):
    """一键样例可视化: 单帧产出 点云 PLY / occ PLY / 3D 框 PLY / 标注叠加 PNG /
    点云+occ BEV PNG —— 手工跑一整套导出命令的等价快捷方式。"""
    from common.render_bev import render_occ_bev, render_points_bev
    root = args.src
    out_dir = args.out or os.path.join(os.getcwd(), DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)
    names = _split_names(args.names) or ["img-000001"]
    for name in names:
        num = int(name.split("-")[1])
        bin_p = os.path.join(root, "samples", "LIDAR_TOP", args.split,
                             name + ".pcd.bin")
        if not os.path.exists(bin_p):
            print("跳过 (无 bin):", bin_p)
            continue
        tok, scene = _token_of(root, args.split, name)
        mode_points(argparse.Namespace(
            src=bin_p, limit=0, names=[], out=out_dir, color="intensity",
            with_rgb=args.rgb, raw_root=args.raw_root))
        mode_boxes(argparse.Namespace(
            src=root, split=args.split, names=[name], limit=0,
            radius=0.015, rgb=args.rgb,
            raw_root=args.raw_root, out=out_dir))
        if tok:
            mode_occ(argparse.Namespace(
                src=os.path.join(root, "gts", scene, tok), limit=0, names=[],
                out=out_dir, what="all" if args.what_all else "occupied",
                mask="camera", voxel=0.4, zmin=-1.0, style="cube",
                cube_scale=0.95))
        mode_preview(argparse.Namespace(
            src=root, split=args.split, names=[name], limit=0, out=out_dir,
            with_3d=False))
        pts = np.fromfile(bin_p, np.float32).reshape(-1, 5)
        render_points_bev(pts, os.path.join(out_dir, "bev_points_%s.png" % name),
                          title="%s n=%d" % (name, len(pts)))
        if tok:
            with np.load(os.path.join(root, "gts", scene, tok, "labels.npz")) as d:
                render_occ_bev(d["semantics"], d["mask_camera"],
                               os.path.join(out_dir, "bev_occ_%s.png" % name),
                               title="%s %s" % (name, tok[:8]))
        print("-- %s 完成 (token=%s)" % (name, (tok or "无表")[:8]))
    print("打开方式: *.ply 拖入 CloudCompare; *.png 直接看 "
          "(bev_*=俯视, preview_*=原图+2D/3D框叠加)")


def mode_diff(args):
    """两份 manifest 逐帧对比: 一致/不同/仅一侧; md5 缺失时回退 count+bytes。
    退出码: 全一致 0, 有差异 1 (便于脚本化回归判断)。"""
    ma = json.load(open(args.a, encoding="utf-8"))
    mb = json.load(open(args.b, encoding="utf-8"))

    def _hdr(m):
        g = m.get("generator", {})
        return "%s %s @ %s (%s)" % (g.get("name", "?"), g.get("version", "?"),
                                    g.get("commit", "?"), g.get("time", "?"))

    def _sig(e):
        return e.get("md5") or "%s:%s" % (e.get("count"), e.get("bytes"))

    ea, eb = ma.get("entries", {}), mb.get("entries", {})
    ka, kb = set(ea), set(eb)
    same, changed = [], []
    for k in sorted(ka & kb):
        (changed if _sig(ea[k]) != _sig(eb[k]) else same).append(k)
    print("A: %s\nB: %s" % (_hdr(ma), _hdr(mb)))
    print("共有 %d 帧: 一致 %d, 不同 %d (%.2f%%); 仅A %d, 仅B %d"
          % (len(ka & kb), len(same), len(changed),
             100.0 * len(changed) / max(len(ka & kb), 1), len(ka - kb), len(kb - ka)))
    for k in changed[:args.max_show]:
        print("  ~ %s: %s -> %s" % (k, _sig(ea[k])[:12], _sig(eb[k])[:12]))
    if len(changed) > args.max_show:
        print("  ... 共 %d 帧不同 (仅展示前 %d)" % (len(changed), args.max_show))
    for k in sorted(ka - kb)[:args.max_show]:
        print("  - 仅A: %s" % k)
    for k in sorted(kb - ka)[:args.max_show]:
        print("  + 仅B: %s" % k)
    sys.exit(1 if (changed or ka != kb) else 0)


def _voxel_cubes(xyz, rgb, size):
    """体素中心+颜色 -> 立方体网格 (每体素 8 顶点 + 6 quad), Occup3D 风格。"""
    h = size / 2.0
    offs = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h)
                     for sz in (-h, h)], np.float64)
    verts = (xyz[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    vcols = np.repeat(rgb, 8, axis=0)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = []
    for b in range(0, len(verts), 8):
        faces.extend([[a0 + b, a1 + b, a2 + b, a3 + b] for a0, a1, a2, a3 in quads])
    return verts, vcols, faces


def mode_info(args):
    root = args.src
    print("数据包:", os.path.abspath(root))
    for name in ("manifest_nuscenes", "manifest_occ", "manifest_detection"):
        m = _rows(os.path.join(root, name + ".json"))
        if m:
            g, p = m.get("generator", {}), m.get("params", {})
            print("%s: %s %s @ %s (%s), backend=%s" % (
                name, g.get("name", "?"), g.get("version", "?"),
                g.get("commit", "?"), g.get("time", "?"), p.get("backend", "?")))
    for name in ("qc_report_nuscenes", "qc_report_occ"):
        q = _rows(os.path.join(root, name + ".json"))
        if q:
            print("%s: errors=%s warnings=%s frames=%s" % (
                name, len(q.get("errors", [])), len(q.get("warnings", [])),
                q.get("frames", "?")))
    for ver in sorted(d for d in os.listdir(root) if d.startswith("v1.0-")):
        vd = os.path.join(root, ver)
        scenes = _rows(os.path.join(vd, "scene.json"))
        print("%s: sample=%d scene=%d ann=%d" % (
            ver, len(_rows(os.path.join(vd, "sample.json"))), len(scenes),
            len(_rows(os.path.join(vd, "sample_annotation.json")))))
    samples = os.path.join(root, "samples", "LIDAR_TOP")
    if os.path.isdir(samples):
        for sp in sorted(os.listdir(samples)):
            sd = os.path.join(samples, sp)
            if os.path.isdir(sd):
                print("samples/LIDAR_TOP/%s: %d bin" % (sp, len(os.listdir(sd))))
    gts = os.path.join(root, "gts")
    if os.path.isdir(gts):
        n_scenes, n_npz = 0, 0
        for sub in os.listdir(gts):
            sd = os.path.join(gts, sub)
            if os.path.isdir(sd):
                n_scenes += 1
                n_npz += sum(os.path.exists(os.path.join(sd, t, "labels.npz"))
                             for t in os.listdir(sd))
        print("gts: %d scene, %d labels.npz" % (n_scenes, n_npz))


# ---------------- 数据集注册表 ----------------

def mode_list(_):
    from datasets import DATASETS
    from config import dataset_paths
    print("已注册数据集/产物 (datasets/__init__.py):")
    for ds, products in DATASETS.items():
        print("  %-12s -> %s" % (ds, ", ".join(products)))
    try:
        print("config 路径:", dataset_paths("sunrgbd"))
    except Exception as e:
        print("config 不可用:", e)
    print("转换:   rgbd2occ convert <数据集> <产物> [参数原样透传]")
    print("        (等价简写: rgbd2occ <数据集> <产物> ...)")
    print("导出:   rgbd2occ export points|occ|boxes|info (--help 查看)")


def mode_convert(args):
    from datasets import DATASETS
    import importlib
    if args.dataset not in DATASETS:
        sys.exit("未注册的数据集: %s (可选: %s)" % (args.dataset, ", ".join(DATASETS)))
    if args.product not in DATASETS[args.dataset]:
        sys.exit("%s 不支持产物 %s (可选: %s)"
                 % (args.dataset, args.product, ", ".join(DATASETS[args.dataset])))
    module_name, entry = DATASETS[args.dataset][args.product]
    getattr(importlib.import_module(module_name), entry)(args.passthrough)


def add_subparsers(sub, with_registry=False):
    """把导出/信息子命令挂到给定 subparsers (叶子经 set_defaults(handler=...) 分发)。
    with_registry=True 时附带 list/convert (本文件独立运行时用; rgbd2occ 顶层
    自带这两者, export 组只挂 points/occ/boxes/info)。"""
    if with_registry:
        p = sub.add_parser("list", help="列举已注册数据集/产物/路径")
        p.set_defaults(handler=mode_list)

        p = sub.add_parser("convert", help="调用数据集转换入口 (参数原样透传)")
        p.add_argument("dataset")
        p.add_argument("product")
        p.add_argument("passthrough", nargs=argparse.REMAINDER)
        p.set_defaults(handler=mode_convert)

    p = sub.add_parser("points", help="LIDAR_TOP bin -> PLY 点云")
    p.add_argument("src", help="bin 文件或含 bin 的目录")
    p.add_argument("--limit", type=int, default=0, help="只取排序后前 N 个")
    p.add_argument("--names", nargs="+", default=[], help="帧选择: img-000123 或 123")
    p.add_argument("--out", default=None, help="输出目录 (默认 ./rgbd2occ_export)")
    p.add_argument("--color", choices=["intensity", "none"], default="intensity",
                   help="intensity=按亮度灰度着色 (默认), none=白色")
    p.add_argument("--with-rgb", action="store_true",
                   help="重跑 CPU 反投影为点云着真彩色 (需包根内参 + --raw-root 原始数据)")
    p.add_argument("--raw-root", default=None,
                   help="SUN RGB-D 原始数据根 (默认取 config 的 raw_root)")
    p.set_defaults(handler=mode_points)

    p = sub.add_parser("occ", help="labels.npz -> PLY 体素点云 (按类着色)")
    p.add_argument("src", help="npz 文件 / token 目录 / scene 目录 / gts 根")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--names", nargs="+", default=[], help="token / token 前缀")
    p.add_argument("--out", default=None)
    p.add_argument("--what", choices=["occupied", "free", "all"], default="occupied",
                   help="occupied=占据体素(默认), free=可见空体素")
    p.add_argument("--mask", choices=["camera", "lidar"], default="camera",
                   help="可见性掩膜 (默认 camera)")
    p.add_argument("--voxel", type=float, default=0.4,
                   help="体素边长 (仅旧 npz 无元数据时生效, 0.9.6+ npz 自带)")
    p.add_argument("--zmin", type=float, default=-1.0,
                   help="Z 轴下界 (仅旧 npz 无元数据时生效)")
    p.add_argument("--style", choices=["cube", "point"], default="cube",
                   help="cube=按类着色小立方体网格 (默认, Occup3D 风格), point=散点")
    p.add_argument("--cube-scale", type=float, default=0.95,
                   help="立方体边长缩放 (<1 露出格间缝, 默认 0.95)")
    p.set_defaults(handler=mode_occ)

    p = sub.add_parser("boxes",
                       help="3D 框 -> 连续线框网格并入原始点云, 单文件输出")
    p.add_argument("src", help="数据包根目录或 v1.0-sunrgbd-<split> 目录")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--names", nargs="+", default=[], help="帧选择: img-000123 或 123")
    p.add_argument("--limit", type=int, default=0, help="前 N 帧")
    p.add_argument("--radius", type=float, default=0.015,
                   help="线框管半径, 米 (默认 0.015)")
    p.add_argument("--rgb", action="store_true",
                   help="并入的点云着真彩色 (需 --raw-root 原始数据)")
    p.add_argument("--raw-root", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(handler=mode_boxes)

    p = sub.add_parser("info", help="数据包概览统计")
    p.add_argument("src", help="数据包根目录")
    p.set_defaults(handler=mode_info)

    p = sub.add_parser("preview",
                       help="2D gt 框叠加相机图 -> PNG (--with-3d 可加 3D 框投影)")
    p.add_argument("src", help="数据包根目录 (需已跑 convert sunrgbd detection)")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--names", nargs="+", default=[], help="帧选择: img-000123 或 123")
    p.add_argument("--limit", type=int, default=0, help="前 N 帧")
    p.add_argument("--out", default=None)
    p.add_argument("--with-3d", action="store_true",
                   help="附加 3D 框投影线框 (默认只画 2D gt 框)")
    p.set_defaults(handler=mode_preview)

    p = sub.add_parser("demo",
                       help="一键样例可视化: 单帧产出 点云/occ/3D框 PLY + 叠加图 + BEV")
    p.add_argument("src", help="数据包根目录")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--names", nargs="+", default=[], help="帧选择 (默认 img-000001)")
    p.add_argument("--out", default=None)
    p.add_argument("--rgb", action="store_true",
                   help="点云 PLY 着真彩色 (需 --raw-root 原始数据)")
    p.add_argument("--raw-root", default=None)
    p.add_argument("--what-all", action="store_true", help="occ PLY 含 free 体素")
    p.set_defaults(handler=mode_demo)

    p = sub.add_parser("diff", help="对比两份 manifest 逐帧差异 (换后端/升级转换器后回归)")
    p.add_argument("a", help="manifest_<产物>.json 路径 (旧)")
    p.add_argument("b", help="manifest_<产物>.json 路径 (新)")
    p.add_argument("--max-show", type=int, default=20, help="最多展示的差异行数")
    p.set_defaults(handler=mode_diff)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_subparsers(sub, with_registry=True)
    args = ap.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
