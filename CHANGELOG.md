# Changelog

## 0.9.6 (2026-09-29)
- **统一命令行 rgbd2occ**: main.py 改 argparse 子命令树 —— list / doctor /
  export (points/occ/boxes/preview/info) / convert / 数据集名直接作子命令
  (rgbd2occ sunrgbd occ 向后兼容), 每一层均有 --help; 新增 --version
- rgbd2occ doctor: 环境/依赖体检 —— 基础依赖 / 加速项 (cv2/numba/torch/triton) /
  各产物线可用后端 / config 路径, 基础依赖缺失退出码 1 (实测可提前识别
  "有 torch 无 triton -> occ 线不可 gpu" 这类环境坑)
- run_all_lines 结束汇总: 逐线退出码 (nuscenes/occ/detection) + 失败续跑提示,
  取代原先只有"累计退出码"的结尾
- render_occ_bev 自动裁剪到可见 bbox(+10 格): 0.4m 网格 80x80m 画进 600px 时
  室内场景只占中心一小块, 裁剪后等效放大 (自车标记同步重映射)
- export points --with-rgb: 重跑 CPU 反投影管线为点云着真彩色 (bin 仅存亮度;
  需包根内参 + --raw-root 原始数据, GPU/CPU 变体点数差异已文档化)
- 2D 框导出产物线 sunrgbd 2d: detection_meta_cache 的 gtBb2D ->
  annotations2d.json 侧车 (cls 37 风格 + cls13 + 原图像素 bbox;
  实测 9758 帧/64062 框 = 3D 框数 - 无 2D 数)
- labels.npz 自带体素元数据: convert_frame 落盘 voxel/gmin (float32),
  export occ 优先读取, 旧 npz 无键回退 --voxel/--zmin 假设
- 批量进度条 common/progress.progress_iter: tqdm 可用显示进度条, 缺失回退
  定期日志; 接入点云/occ/检测三线批量循环; doctor 加速项报告 tqdm
- occ 导出重构 (反馈修正): 占据体素由散点改为按类着色的小立方体网格
  (--style cube 默认, Occup3D 风格, --cube-scale 控制格间缝; point 散点保留可选)
- **修复 3D 框朝向真 bug**: SUN RGB-D 64258 框实测 15.6% basis 为镜像基
  (det<0)、约半数非严格正交、1 例 NaN —— 直接 rot_to_quat 产出错误姿态
  (img-000001 主 table 框与基准 R 差 1.69)。新增 fill_detection.proper_rotation:
  取反一列消镜像 (盒体张成不变) + SVD 极分解投影; 检测线三处 basis 消费点
  统一走净化路径后重建 sample_annotation, 复验 9/9 框 R 差 3.8e-07、
  框内点数与统计一致; 附四项数值诊断 (bin 投影回图 99.8% / 独立重推占据
  IoU=1.0000 / 占据-点云中位距 0.19m) 证明点云与占据链路本身无误
- occ 导出文件名嵌入帧号 (occ_<split>_img-XXXXXX_...): 消除与 points/boxes
  跨帧误配对 (此前 kv1 token 文件实为 img-002202, 易与 img-000001 点云错配)
- 导出表示按 _legacy 验证过的方案重构 (反馈): points=纯 RGB 点云单文件
  (无标量场, --with-rgb 真彩); boxes=3D 框连续细管线框网格单文件 (与点云
  分文件, CC 中同载叠加 —— 含 face 的 PLY 在 CC 按网格导入不渲染孤立顶点,
  合并方案弃用); occ=占据体素中心语义色散点 + free 白色 1/8 抽稀分文件
  (未知不导出, --style cube 可选立方体) —— 采纳 _legacy/export_occ_pointcloud.py
- boxes_points 可见性修正 (反馈: "没有原始点云"): 含 face 的 PLY 在
  CloudCompare 中按网格导入、孤立顶点默认不渲染 —— 合并文件里的点云改为
  小八面体 surfel 实体化 (--point-size 0.02, 0=关闭), 真彩 RGB 回退补齐
  (此前 --rgb 在 boxes 路径因 raw_root 缺失静默变灰度); 检测附证: 框内点数
  与检测线 num_lidar_pts 9/9 逐框一致, 管顶点到边线距离 max=0.015m=半径
- preview 改为默认只画 2D gt 框 (反馈: 3D 框投影对 2D 图像无必要),
  --with-3d 可选叠加; 2D 框边缘准确性实测: 原始坐标 vs MATLAB 1-based 修正
  (-1px) 放大 3x 无可分辨差异, 边缘松散源自官方 2D 标注本身 (诊断图见
  sunrgbd_viz/diag_2d)
- boxes 再修正 (反馈): 删除"纯线框"独立导出路径 —— 框必须并入原始点云,
  单文件输出 boxes_points_*; demo 同帧产出 纯点云 PLY + 点云加框 PLY 两件
- boxes 线框重构 (反馈修正): 12 条边由离散采样点改为连续细四棱管网格
  (PLY face, CloudCompare 直渲, --radius 定粗细), --with-points 把原始点云
  (可选 --rgb 真彩) 与线框并入同一 PLY —— 框不再悬空、不再虚点
- export demo: 单帧一键全套可视化 (点云/occ/3D框 PLY + 叠加 PNG + 双 BEV)
- export diff: 两份 manifest 逐帧对比 (md5 缺失回退 count+bytes;
  退出码一致 0/有差异 1), 升级转换器/换后端后的回归验证
- 命令行导出 tools/export_ply.py (rgbd2occ export ...):
  points (LIDAR_TOP bin -> PLY 点云, 亮度标量场+灰度着色) /
  occ (labels.npz -> PLY 体素点云, 13 类语义着色, --what occupied|free|all) /
  boxes (sample_annotation -> PLY 彩色线框, nuScenes wlh+四元数) /
  info (数据包概览);
  二进制 PLY 零第三方依赖, CloudCompare 直接打开 (自车系 Z 向上即正视角);
  输出默认当前目录 rgbd2occ_export/, --limit/--names 支持逗号分隔多帧
- export preview: 2D gt 框 + 3D 框投影叠加相机图 -> PNG (数据全来自数据包内缓存)
- gpu 后端可用性检查补 triton (occ 批量射线核依赖 triton; 此前只查 torch,
  缺 triton 时三线编排在 occ 阶段中途崩溃而非整体降级 fast)
- **内部包改名 nuscenes/ -> nustables/**: 消除与 pip 的 nuscenes-devkit 的顶层包名
  冲突 (曾致 pip 安装后 rgbd2occ 入口在装有 devkit 的环境里 import 错包;
  非 editable 安装甚至会覆盖 devkit 的 __init__.py)。改动仅 3 处 import +
  pyproject/README; devkit 示例的 `import nuscenes` 现在无歧义指向官方包
- tools/read_devkit_example.py: nuScenes devkit 读取示例 (实测 v1.0-sunrgbd-train
  可直接 NuScenes(version=..., dataroot=...) 挂载), 未装 devkit 自动退回零依赖
  模式; 两条路径输出一致
- 修复 to_occ 单帧模式 (run_single 调用了未导入的 load_depth)
- 新增 tests/test_export_ply.py (8 项)
- tools/read_devkit_example.py: nuScenes devkit 读取示例 (实测 v1.0-sunrgbd-train
  可直接 NuScenes(version=..., dataroot=...) 挂载), 未装 devkit 自动退回零依赖
  模式; 两条路径输出一致
- 新增 tests/test_export_ply.py (8 项)

## 0.9.5 (2026-09-23)
- 检测线 label QC GPU 化 (标签图作为颜色传 GPU 全链, 与 occ 掩膜严格同源):
  检测线 328s -> ~200s (label QC 282s -> ~30s); 一致率数字与 CPU 版吻合 (38%)
- 三线统一编排: python main.py sunrgbd all [--root ... --gpu-batch 16 ...]
  (nuscenes -> occ -> detection 按依赖序, 参数自动按线翻译)
- occ GPU 批量线 4000 帧稳态实测: 138 帧/s, QC 0 错误, 抽查 mask 0.000% 差
- 10 万帧终版预估: occ ~12 分钟 / 点云 ~21 分钟 / 检测 ~30 分钟 (三线串行 <65 分钟)

## 0.9.4 (2026-09-23)
- nuscenes 线 GPU 批量流水线 (--gpu-batch B): 解码进程池 -> 主进程单一 CUDA
  上下文批量消费 -> 写盘线程池 (多进程逐帧 30 帧/s -> 77.4 帧/s, 2.6x)
- 500 帧稳态实测: 77.4 帧/s, QC 0 错误; 5 帧抽查 vs 生产 bin 最近邻全部 0.00cm,
  点数差 +0.07~0.29%
- 10 万帧外推: ~21 分钟 (点云线)

## 0.9.3 (2026-09-22)
- 点云线 GPU 化 (common/gpu_points.py, gpu 层近似): median(NaN-aware unfold)/梯度/
  反投影/SOR(卷积整数计数)/斑点(负索引取模回绕)/降采样 全链在卡上
- 实测 RTX 5090: 全链含降采样 10ms/帧 (CPU numba 144-177ms, ~15x);
  与生产 bin 的最近邻: 中位 0.00cm / 95分位 0.00cm (KDTree, 41757 vs 41473 点 +0.68%)
- to_nuscenes: RGBD2OCC_BACKEND=gpu 时自动启用 GPU 路径 (torch 惰性导入,
  无 torch/CUDA 自动回退 CPU exact)
- 检测线导出 class_map.json (37->13 类映射侧车)

## 0.9.2 (2026-09-22)
- GPU 批量流水线接入 to_occ (--gpu-batch B): 解码进程池 -> GPU 批量消费 -> 写盘线程池,
  含 --with-images 硬链接/stride/label-vote 全参数支持, manifest 记录 backend/data_variant
- 重大修复: gpu_raycast 单帧/批量核的原点参数错误 (gmin 被当作网格坐标传入,
  射线从未进格 -> free 恒空被 try/except 静默吞掉回退 CPU); dists 单位错误
  (米 vs 体素); 单帧返回未 reshape; 批量帧偏移双重相加; torch.flatnonzero 兼容
- 三方验证 (单帧 GPU vs 批量 GPU vs CPU exact): 单/批完全一致, 与 CPU mask 差
  0.05%/语义 0 差异 (float32 近似层文档化行为); 测试 28 项

## 0.9.1 (2026-09-22)
- voxelize_and_cast_batch_gpu: 多帧批量 GPU (B 帧射线一次 launch, fid 索引网格切片,
  标签 13 次按类 index_put 确定性 amax); 实测 2 帧 11.6ms (5.8ms/帧), 与 CPU exact
  差异 0.004-0.05% (gpu 近似层)
- 修复: 单帧/批量 GPU 函数的 z-ii 未同步过滤 (inb 过滤后 z 未跟随, 射线长度错位)
- 语义标签治理: --label-vote (体素内多数投票 + 未标注让位, opt-in)

## 0.9.1 (2026-09-22)
- 语义标签治理: --label-vote (体素内多数投票 + 未标注让位, opt-in, 改变数据语义;
  实测每帧治理 17-61 个冲突体素, mask 不变)
- GPU 层全链体素化: voxelize_and_cast_gpu (反投影/体素化/散射/射线全在卡上,
  下行仅占据格), 全帧 43-71ms (其中含 ~25ms PNG 解码+掩膜 CPU 时间)

## 0.9.0 (2026-09-22)
- GPU v2 传输优化: 上行原始 uint16 深度 (774KB, 反投影移至 GPU) 替代射线数组
  (1.2-4.6MB); 下行 free 网格位压缩 (640KB->80KB, np.unpackbits 还原)
- 实测 RTX 5090: occ 全帧 44-69ms (CPU 118-360ms, 2-3x); 语义 100% 一致,
  mask 差异 0.03-0.2% (float32 边界射线, gpu 层为近似层的文档化行为)
- GPU 管线使用: RGBD2OCC_BACKEND=gpu + Lss env (torch 2.7.1+cu128)

## 0.8.0 (2026-09-22)
- 检测线新增语义标签一致性 QC (--label-qc-sample, 抽帧验证 3D 框 vs 13 类标签,
  轴对齐保守口径; 固化坐标链路回归监控, 结果进 manifest)
- 正式数据以 --ray-stride 1 重生成 (free 标注更密, v0.8.0 manifest)


- occ 批量暴露 --ray-stride (射线采样间隔, 批量/单帧共用): 1=全像素, free 标注
  更密 (~+2% 体素, 集中在物体轮廓), 耗时几乎不变; 默认仍 4 (数据兼容)


- 计算后端分层: RGBD2OCC_BACKEND=exact|fast|gpu (默认 exact, 数据兼容基线)
  - fast: CPU 近似 (median 全域 cv2 整数中值); 实测仅 ~7% 收益, 近似空间已被
    精确核挤占
  - gpu: torch CUDA raycast (逐位规则与 CPU 一致, float32 属近似层);
    torch 不可用时自动降级 fast
  - manifest 记录 backend/data_variant; 测试始终强制 exact (conftest)
- 本机实测: RTX 5090 在位但 pytorch CUDA 轮子源不可达, GPU 层以优雅降级交付

## 0.7.0 (2026-09-22)
- 输出合并为单一数据根 (官方 Occ3D 布局): gts/+annotations.json 并入 nuScenes 根,
  manifest/qc_report 按产品命名 (_nuscenes/_occ/_detection)
- median_gradient NaN 窗口检测积分图化; cv2 整数域 medianBlur 快路径
  (uint16 中值 x scale, 与 nanmedian 逐位一致); raw/scale 贯穿调用链
- SOR/体素斑点/射线追踪 numba 整数核 (全链逐位一致, 10 帧盘上抽查 10/10)
  - 修复 speckle numba 核负索引语义 (取模回绕, 复现 numpy 历史行为)
- --with-images 改同盘硬链接 (零拷贝)
- 环境: conda Library/bin 入 PATH 后 ssl/pip 恢复, numba 0.8s->全链 79-144ms/帧
  (累计 ~5-10x); 测试 25 项
- 旧散装脚本归档 D:/Datasets/_legacy (附新旧对照)

## 0.6.0 (2026-09-22)
- occ 包图像收进管线: `--with-images` 随包拷贝 jpg (此前需手动)
- manifest 条目新增 `median_depth`; QC 新增"近距特写帧"警告 (中位深度 < 1m)
- 批量运行日志落盘: `<out>/logs/run_*.log` (可回溯)
- MAT 解析器修复 int16/uint16/int32 字段 (gtBb2D 此前被按字节拆坏)
- 检测线增强: `--min-pts` 清洗; 3D->2D 投影 IoU 一致性统计 (进日志与 manifest);
  meta 缓存统计 (丢帧/丢框/无2D 明细)
- golden 扩到 kv1 相机 (点云字节级 + 带标签 occ 逐元素)
- LICENSE (MIT) / CHANGELOG; 旧散装脚本归档到 D:\Datasets\_legacy

## 0.5.0 (2026-09-22)
- 语义标签接入: SUN RGB-D 13 类 (semantic id == 像素值), semantic_classes.json 侧车
- manifest 溯源 (版本/commit/参数快照/逐帧 md5), 自动质检 qc_report.json
- QC 可视化: --inspect / --inspect-warned (BEV 渲染)
- 3D 检测标注线: groundtruth3DBB -> sample_annotation/instance (Rtilt 坐标链路
  源码+椅高双重实证; 深度尺度 /6553.5 实证确认, 官方 /8000 不适用本分发版)
- pyproject.toml 打包 (rgbd2occ 命令); tools/ 查看工具 (点云/occ, BEV+3D)

## 0.4.0 (2026-09-22)
- P0 工程化: config.json 路径外置, 统一日志, 逐帧容错 + failed.json, 测试 11 项

## 0.3.0 (2026-09-22)
- common/ 拆分为每功能一文件 (独立优化/依赖隔离)

## 0.2.0 (2026-09-22)
- 主程序 + 数据集适配器架构 (main.py 调度, datasets/<name>/)

## 0.1.0 (2026-09-22)
- 首版: SUN RGB-D -> nuScenes devkit (v2 点云+v3 表, 字节级等价) 与 Occ3D 占据标注
