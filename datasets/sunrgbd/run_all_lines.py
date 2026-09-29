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
    ap.add_argument("--voxel", type=float, default=0.4,
                    help="occ 体素边长, 米 (默认 0.4=官方 Occ3D schema; 细体素如 "
                         "0.05 自动切室内局部范围, 偏离官方维度)")
    ap.add_argument("--xrange", type=float, nargs=2, default=None,
                    help="occ X 范围 (默认随体素: 0.4->[-40,40], <0.1->[0.2,8])")
    ap.add_argument("--yrange", type=float, nargs=2, default=None)
    ap.add_argument("--zrange", type=float, nargs=2, default=None)
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
    if args.voxel != 0.4:
        occ += ["--voxel", str(args.voxel)]
    for flag, val in (("--xrange", args.xrange), ("--yrange", args.yrange),
                      ("--zrange", args.zrange)):
        if val:
            occ += [flag, str(val[0]), str(val[1])]
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
    rc_tables = 0
    try:
        importlib.import_module("datasets.sunrgbd.to_nuscenes").main(
            nus + ["--tables-first"] + (["--skip-qc"] if args.skip_qc else []))
    except SystemExit as e:
        rc_tables = e.code or 0
    rc = rc or rc_tables
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

    det_pre = det + ["--skip-points"] + (["--label-qc-sample", str(args.label_qc_sample)]
                                          if args.label_qc_sample != 50 else [])
    det_post = det + ["--points-only"]
    th_nus = threading.Thread(target=_run, args=("nuscenes", lines[0][1], lines[0][2]))
    th_occ = threading.Thread(target=_run, args=("occ", lines[1][1], lines[1][2]))
    th_det = threading.Thread(target=_run, args=("detection-pre", lines[2][1], det_pre))
    th_nus.start(); th_occ.start(); th_det.start()
    th_nus.join(); th_occ.join(); th_det.join()
    rc = rc or (results.get("nuscenes", 1) or results.get("occ", 1) or results.get("detection-pre", 1))
    log.info("===== 并行阶段完成 %.0fs (nuscenes=%s, occ=%s, det预填充=%s) =====",
             time.time() - t1, results.get("nuscenes"), results.get("occ"),
             results.get("detection-pre"))

    # ---- 阶段2: 检测线收尾 (依赖点云 bin, 只补 num_lidar_pts) ----
    t2 = time.time()
    log.info("===== detection points-only 开始 =====")
    rc_det_post = 0
    try:
        rc_det_post = importlib.import_module(lines[2][1]).main(det_post) or 0
        rc = rc or rc_det_post
    except SystemExit as e:
        rc_det_post = e.code or 0
        rc = rc or rc_det_post
    log.info("===== detection points-only 结束 %.0fs =====", time.time() - t2)

    # ---- 结束摘要: 逐线退出码 + 续跑提示 (阶段码归并到产物线) ----
    def _agg(*vals):
        return max((v or 0) for v in vals)

    line_rc = {"nuscenes": _agg(rc_tables, results.get("nuscenes")),
               "occ": _agg(results.get("occ")),
               "detection": _agg(results.get("detection-pre"), rc_det_post)}
    log.info("===== 汇总: 总耗时 %.0fs =====", time.time() - t0)
    for name, code in line_rc.items():
        log.info("  %-10s %s", name, "OK" if code == 0 else "失败 (退出码 %s)" % code)
    bad = [n for n, c in line_rc.items() if c]
    if bad:
        log.warning("失败线: %s; 重跑同一命令可续跑 (已完成产物自动跳过);"
                    " 疑似环境问题先跑: rgbd2occ doctor", ", ".join(bad))
    sys.exit(rc)


if __name__ == "__main__":
    main()
