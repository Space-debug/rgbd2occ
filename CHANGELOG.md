# Changelog

## 0.8.0 (2026-09-22)
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
