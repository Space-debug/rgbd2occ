# rgbd2occ

RGBD 数据集 → (nuScenes devkit 格式 + Occ3D 占据标注) 转换工具集。
**一个主程序 + 每种数据集一个适配目录**，核心库共用；新增数据集只需加目录并登记注册表。

## 目录结构

```
rgbd2occ/
├── main.py                  # 主生成程序: python main.py <数据集> <产物> [参数...]
├── common/                  # 工具类库: 每个通用功能一个文件 (可独立优化/替换, 依赖随文件隔离)
│   ├── load_depth.py        #   深度图 -> float64 米制
│   ├── load_label.py        #   语义标签图 -> uint8
│   ├── cam_to_ego_axes.py   #   相机系->自车系 (X前,Y左,Z上) 轴变换 (全库唯一一份)
│   ├── deproject.py         #   通用反投影 (含 k1/k2 去畸变, 返回像素坐标)
│   ├── median_gradient.py   #   深度滤波 stage1: 5x5 中值 + 梯度剔除
│   ├── mask_depth.py        #   深度有效性掩膜
│   ├── sor_radius.py        #   点云滤波 stage2: SOR 半径滤波
│   ├── voxel_speckle.py     #   点云滤波 stage3: 体素斑点过滤
│   ├── deproject_filtered.py#   v2 精确版反投影 (与 v2 字节级等价)
│   ├── depth_to_points.py   #   点云全流程编排 (stage0-3)
│   ├── voxel_downsample.py  #   体素降采样 (质心+颜色均值)
│   ├── write_nuscenes_bin.py#   nuScenes bin 写出 (float32 Nx5)
│   ├── _imread.py           #   (私有) 图像底层读取, load_depth/load_label 共用
│   └── _nbr_count.py        #   (私有) 3x3x3 邻域计数, sor_radius/voxel_speckle 共用
├── nustables/               # nuScenes devkit 格式核心
│   └── tables.py            #   稳定 token(md5) + 13 张表落盘
├── occ/                     # Occ3D 占据标注核心
│   ├── voxel_grid.py        #   官方自车系网格 X前/Y左/Z上 (整除截断, 200x200x16)
│   ├── raycast.py           #   向量化 DDA 射线追踪
│   ├── convert.py           #   convert_frame 三态编码
│   └── annotations.py       #   annotations.json (mini 同构)
│   (几何投影/深度预处理在 common/ —— 两条产物线共用; 相机->自车轴变换全库仅 common.projection 一份)
├── datasets/                # ★ 数据集适配器 (每种数据集一个目录)
│   ├── __init__.py          #   注册表 DATASETS = {名: {产物: (模块, 入口)}}
│   └── sunrgbd/
│       ├── meta.py          #   原始数据读取: SUNRGBDMeta.mat -> 逐帧真实内参
│       ├── labels.py        #   13 类语义标签 (路径/映射/occ 语义约定)
│       ├── to_nuscenes.py   #   → nuScenes devkit (点云+图像+13张表)
│       ├── to_occ.py        #   → Occ3D 占据标注 (批量+单帧)
│       └── fill_detection.py#   → 3D 检测标注 (groundtruth3DBB 填 sample_annotation)
└── README.md
```

## 用法 (统一命令行, 每层均有 --help)

```bash
rgbd2occ                                   # 全局帮助 + 已注册数据集/产物
rgbd2occ list                              # 注册表 + config 路径
rgbd2occ doctor                            # 环境/依赖体检: 各产物线可用后端
rgbd2occ convert sunrgbd nuscenes          # 全量 -> nuScenes 格式 (断点续跑)
rgbd2occ convert sunrgbd nuscenes --limit 3    # 每 split 前 3 帧试跑
rgbd2occ convert sunrgbd nuscenes --frames 1,1925
rgbd2occ convert sunrgbd nuscenes --tables-only
rgbd2occ sunrgbd occ                       # 等价简写: 数据集名直接作子命令
rgbd2occ convert sunrgbd 2d                # 2D gt 框 -> annotations2d.json 侧车
rgbd2occ sunrgbd occ --help                # 产物层参数 (透传给转换入口)
rgbd2occ sunrgbd occ --mode single <深度图> --fx 529.5 --fy 529.5 \
    --cx 365 --cy 265 --depth-scale 0.000152592 --scene s1 --token t1 --out-root out
```

`sunrgbd nuscenes` 首次运行解析 SUNRGBDMeta.mat 后缓存到
`<out>/sunrgbd_meta_cache.json`；逐帧真实 K 记录在 `<out>/intrinsics_per_frame.json`。

## 查看 / 导出 (CloudCompare) / devkit 读取

```bash
rgbd2occ export info   <out>                              # 数据包概览
rgbd2occ export points <out>/samples/LIDAR_TOP/train --limit 5 --with-rgb  # RGB点云
rgbd2occ export occ    <out>/gts/sunrgbd-train-kv1/<token>  # 0.05m 细方格+free
rgbd2occ export occ    <out>/gts --limit 4 --style point   # 散点可选
rgbd2occ export preview <out> --names img-000001   # 原图 + 2D gt 框 (--with-3d 加投影)
rgbd2occ export demo   <out> --names img-000001    # 一键全套可视化
rgbd2occ export diff   <旧manifest.json> <新manifest.json>   # 回归对比 (一致0/有差异1)
rgbd2occ export --help          # 导出组帮助; 每个子命令亦有 --help
python tools/read_devkit_example.py <out>   # nuScenes devkit 读取示例 (无 devkit 自动零依赖)
```

- 产物导出为**二进制 PLY**（零第三方依赖），CloudCompare 直接拖入即可：
  points=纯 RGB 点云（--with-rgb 重投影真彩）; occ=占据小方格（默认 0.05m 细体素
  重投影, legacy 同款; --revox 0 用数据集 0.4m 体素）+ free 白色 1/8 抽稀,
  文件名含帧号; 3D 框
  可视化用 `export preview --with-3d`（投影到原图）。坐标系=自车系 X前/Y左/Z上，
  与 CloudCompare 默认 Z 轴向上一致；输出默认当前目录 `rgbd2occ_export/`。
- `pip install .` 后以上均以 `rgbd2occ` 命令调起（未安装时 `python main.py ...` 等价）。
- devkit 挂载：表目录名 `v1.0-sunrgbd-<split>` 遵循 nuScenes `v1.0-*` 约定，
  `NuScenes(version="v1.0-sunrgbd-train", dataroot=<out>)` 可直接加载
  （`pip install nuscenes-devkit`；差异点见 tools/read_devkit_example.py 文档串）。

## 工程化

- **配置**: 机器路径集中在根目录 `config.json`（模板 `config.example.json`；
  环境变量 `RGBD2OCC_CONFIG` 可指定其他文件）。CLI 的 `--out/--raw-root/--v3-root` 可临时覆盖。
- **日志**: 统一 logging（时间戳/级别/进程名，多进程可区分 worker）；
  级别用环境变量 `RGBD2OCC_LOGLEVEL` 或 config 的 `log_level` 调整。
- **容错**: 单帧异常不中断整批；失败帧（含堆栈）写入 `<out>/failed.json`，
  重跑同一命令自动重试（已生成的产物跳过）；有失败帧时进程退出码为 1。
- **测试**: `python tests/run_all.py`（零依赖运行器）或 `pytest tests/`。
  golden 测试签入 1 帧输入与期望产物（点云 bin 与 v2/v3 管线字节级一致、
  occ 输出逐元素一致），任何破坏等价性的实现改动都会被立即抓住。
- **语义标签**: occ 批量转换自动接入 SUN RGB-D 13 类标签
  （`train13labels`/`test13labels`，映射为 **semantic id == 像素值**：
  0=others，1..13=bed…window，17=free），`--no-labels` 关闭；
  包内 `semantic_classes.json` 侧车记录映射，数据集自描述。
- **manifest 溯源**: 每次批量转换写 `<out>/manifest.json` ——
  生成器版本/git 哈希/时间 + 参数快照 + 逐帧文件 md5/大小/点数(或可见体素数)。
- **自动质检**: 转换后自动跑 QC 并写 `qc_report.json`（`--skip-qc` 关闭）：
  文件缺失/大小不符/空帧/内参非法为**错误**（退出码 1）；计数偏低帧为警告；
  另抽样深检 npz/bin 规格。 
- **打包**: `pyproject.toml`；`pip install .` 后可用 `rgbd2occ sunrgbd occ ...`
  命令（依赖 numpy/scipy/pillow）。
- **occ 包图像**: `--with-images` 随包拷贝对应 jpg（否则包内只有标注）。
- **检测清洗/一致性**: `sunrgbd detection --min-pts N` 丢弃点云内点不足的框；
  每次填充自动统计 3D->2D 投影 IoU（监控用，进 manifest）。
- **日志落盘**: 批量运行同时写 `<out>/logs/*.log`。
- 许可 MIT（LICENSE）；版本历史见 CHANGELOG.md。
- **计算后端**: `RGBD2OCC_BACKEND=exact|fast|gpu`（默认 exact=逐位等价/数据兼容基线）。
  fast=CPU 近似（~7% 收益）；gpu=torch CUDA raycast（无 torch 自动降级 fast）。
  非 exact 会在 manifest 记录 `data_variant`；测试始终以 exact 断言。

## 新数据集接入（只需三步）

1. 新建 `datasets/<name>/` 目录：`meta.py`（读该数据集的原始数据/内参/帧清单）、
   `to_nuscenes.py` 与 `to_occ.py`（复制 sunrgbd 对应文件，替换
   `prepare_frames()`/`build_tasks()` 中"数据集组织方式 -> 统一帧记录"的部分：
   split/序号/深度路径/内参/图像路径/时间戳）；
2. 在 `datasets/__init__.py` 注册一行；
3. 需要新滤波算子时加到 `common/point_filter.py`，核心其余不动。
   有真实位姿时传入 ego_pose（没有则置空 null，不写假的）。

## 回归验证状态

- `main.py sunrgbd nuscenes`: 4 种相机(kv1/kv2/realsense/xtion)各 1 帧 —— 点云 bin
  与 v3 数据集**字节级一致**；jpg 字节一致；8 张表逐行一致（sample/scene 的
  prev/next 链在部分帧试跑下为子集链，属预期，全量跑一致）。
- `main.py sunrgbd occ`: 单帧输出与旧 single_frame_occ.py 逐元素一致；批量输出与
  Occupancy3D-SUNRGBD 大包同 token 逐元素一致。
- `occ.raycast` 向量化与逐条 DDA 逐元素一致；轴向/语义约定/annotations 结构
  均与官方 mini 包逐项比对通过（含用 mini 自身自车位移做的轴序实证）。

## 输出格式

nuScenes devkit 格式（to_nuscenes）:
```
<out>/samples/{CAM_FRONT,LIDAR_TOP}/<split>/img-XXXXXX.{jpg,pcd.bin}
<out>/v1.0-sunrgbd-<split>/  13 张 json 表 (按相机型号分场景, 唯一 K 建标定)
```
点云为 float32 Nx5 (x,y,z,intensity=亮度/255,ring=0), 自车系 x前/y左/z上。

输出采用**合并根目录**（官方 Occ3D 布局: occ 产物与 nuScenes 数据同根）,
本机即 `D:/Datasets/sunrgbd_nuscenes/v3`（config 的 occ_out == nuscenes_out）:
```
<root>/samples/{CAM_FRONT,LIDAR_TOP}/<split>/    # 图像 + 点云
<root>/v1.0-sunrgbd-<split>/                     # 13 张 devkit 表 (含检测标注)
<root>/gts/<scene>/<token>/labels.npz            # occ: semantics(0=others,1..13=语义,17=free)+双 mask
<root>/annotations.json                          # occ 帧链/内外参 (img_path 直接指向 samples/)
<root>/semantic_classes.json                     # occ 语义映射侧车
<root>/manifest_{nuscenes,occ,detection}.json    # 各产物溯源 (版本/参数/逐帧 md5)
<root>/qc_report_{nuscenes,occ}.json             # 各产物质检报告
```
性能: median_gradient 对无 NaN 窗口走 np.partition 快路径 (与 nanmedian 逐位一致,
golden 锁定), 点云线 ~10 帧/s(10核) -> 实测新帧率 ~40+/s(16核)。
关键约定: 轴序=自车系 X前/Y左/Z上; 未知=mask==0(语义名义 17); 位姿无真值置空 null;
帧链 prev/next 首尾 "EOF"; 体素 0.4m, (200,200,16) —— 与官方 Occupancy3D-nuScenes-v1.0-mini 实测逐维一致 (0.4m/±40m/[-1,5.4m] 为 Occ3D 官方规格, 导出 PLY 的立方体尺寸即体素真尺寸; 室内如需更细分辨率可 convert sunrgbd occ --voxel 0.2, 但将偏离官方 schema 维度)。

深度尺度: **/6553.5** (已用椅高物理检验钉死: 实测高/框高中位 1.00;
官方 toolbox 的 /8000 位运算解码不适用于本数据分发版, 勿改)。
3D 框坐标链路: groundtruth3DBB 在 Rtilt 重力对齐系,
p_ego = M^T Rtilt^T p_grav (M 见 fill_detection.py, 源码+椅高双重验证)。
