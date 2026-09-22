# Changelog

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
