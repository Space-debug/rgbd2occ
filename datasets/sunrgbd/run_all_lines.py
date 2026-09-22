# -*- coding: utf-8 -*-
"""三线统一编排: 一条命令按依赖顺序跑完 nuScenes 点云 -> occ 标注 -> 检测填充。

依赖关系: occ 的 build_tasks 读 nuScenes 表; 检测填充读点云 bin 与表 ——
顺序必须是 nuscenes -> occ -> detection。
参数: 本入口解析通用参数并按线翻译 (--root 同时映射到 nuscenes 的 --out
与 occ 的 --out-root); 线专属参数原样转发。

用法:
  python main.py sunrgbd all                          # 全量三线 (CPU exact)
  python main.py sunrgbd all --root D:/out --limit 5  # 试跑
  RGBD2OCC_BACKEND=gpu python main.py sunrgbd all --gpu-batch 16 --ray-stride 1
"""
import argparse
import importlib
import sys

from common.get_logger import get_logger

log = get_logger("rgbd2occ.sunrgbd.all")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="输出数据根 (= nuscenes --out = occ --out-root)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--gpu-batch", type=int, default=0, help="GPU 批量 (nuscenes+occ)")
    ap.add_argument("--ray-stride", type=int, default=4, help="occ 射线采样")
    ap.add_argument("--label-vote", action="store_true", help="occ 标签投票治理")
    ap.add_argument("--with-images", action="store_true", help="occ 随包图像")
    ap.add_argument("--min-pts", type=int, default=0, help="检测清洗")
    ap.add_argument("--label-qc-sample", type=int, default=50, help="检测标签QC抽样间隔")
    ap.add_argument("--skip-qc", action="store_true")
    args = ap.parse_args(argv)

    common = ["--splits"] + args.splits + ["--workers", str(args.workers)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.skip_qc:
        common += ["--skip-qc"]

    nus, occ = common[:], common[:]
    det = ['--splits'] + args.splits            # detection 无 --workers/--skip-qc
    if args.limit:
        det += ['--limit', str(args.limit)]
    if args.root:
        nus += ["--out", args.root]
        occ += ["--out-root", args.root]
        det += ["--out", args.root]
    if args.gpu_batch:
        nus += ["--gpu-batch", str(args.gpu_batch)]
        occ += ["--gpu-batch", str(args.gpu_batch)]
    if args.ray_stride != 4:
        occ += ["--ray-stride", str(args.ray_stride)]
    if args.label_vote:
        occ += ["--label-vote"]
    if args.with_images:
        occ += ["--with-images"]
    if args.min_pts:
        det += ["--min-pts", str(args.min_pts)]
    if args.label_qc_sample != 50:
        det += ["--label-qc-sample", str(args.label_qc_sample)]
    lines = [("nuscenes", "datasets.sunrgbd.to_nuscenes", nus),
             ("occ", "datasets.sunrgbd.to_occ", occ),
             ("detection", "datasets.sunrgbd.fill_detection", det)]

    rc = 0

    # ---- 阶段0: 表先行 (秒级; occ 线依赖表而非 bin) ----
    import time
    t0 = time.time()
    log.info("===== 表先行 (供 occ 并行启动) =====")
    try:
        importlib.import_module("datasets.sunrgbd.to_nuscenes").main(
            nus + ["--tables-first"] + (["--skip-qc"] if args.skip_qc else []))
    except SystemExit as e:
        rc = rc or (e.code or 0)
    log.info("===== 表先行完成 %.0fs =====", time.time() - t0)

    # ---- 阶段1: 点云线(GPU) 与 occ 线(GPU) 并行 (依赖仅为表, 计算独立) ----
    import threading
    t1 = time.time()
    results = {}

    def _run(name, mod_name, line_argv):
        try:
            results[name] = importlib.import_module(mod_name).main(line_argv) or 0
        except SystemExit as e:
            results[name] = e.code or 0
        except Exception:
            import traceback
            log.error("%s 线异常: %s", name, traceback.format_exc()[-300:])
            results[name] = 1

    th_nus = threading.Thread(target=_run, args=("nuscenes", lines[0][1], lines[0][2]))
    th_occ = threading.Thread(target=_run, args=("occ", lines[1][1], lines[1][2]))
    th_nus.start(); th_occ.start(); th_nus.join(); th_occ.join()
    rc = rc or (results.get("nuscenes", 1) or results.get("occ", 1))
    log.info("===== 并行阶段完成 %.0fs (nuscenes=%s, occ=%s) =====",
             time.time() - t1, results.get("nuscenes"), results.get("occ"))

    # ---- 阶段2: 检测线 (依赖点云 bin, CPU 为主) ----
    t2 = time.time()
    name, mod_name, line_argv = lines[2]
    log.info("===== %s 线开始 =====", name)
    try:
        rc_line = importlib.import_module(mod_name).main(line_argv)
        rc = rc or (rc_line or 0)
    except SystemExit as e:
        rc = rc or (e.code or 0)
    log.info("===== %s 线结束 %.0fs (累计退出码 %s) =====", name, time.time() - t2, rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
