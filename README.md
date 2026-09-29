# rgbd2occ

RGBD 数据集 → (nuScenes devkit 格式 + Occ3D 占据标注) 转换工具集。
**一个主程序 + 每种数据集一个适配目录**，核心库共用；新增数据集只需加目录并登记注册表。

## 安装

### 1. 前置要求

- Python ≥ 3.8（本机为 conda 环境）
- 基础依赖（安装时自动装）：`numpy` / `scipy` / `pillow`
- 可选依赖（缺失只降速/降级，不报错，`rgbd2occ doctor` 可体检）：

| 可选依赖 | 作用 |
|---|---|
| `opencv-python` + `numba` | exact/fast 后端加速（中值滤波整数域快路径、numba 核） |
| `torch`（CUDA 版） | gpu 后端（点云线 ~95 帧/s、occ 射线追踪） |
| `triton` | **occ 批量 GPU 射线核必需**（缺它 occ 线自动降级 fast） |
| `tqdm` | 批量转换进度条（缺失回退定期日志） |
| `nuscenes-devkit` | 用官方 devkit 读取产物（不装可用零依赖示例） |
| `open3d` | `tools/view_occ.py --mode 3d` 交互查看（可选） |

### 2. 安装本体（editable，改代码即时生效）

```bash
# 在目标 conda 环境中（示例为 base；GPU 生成建议装到带 torch+triton 的环境）
conda activate <env>
pip install -e D:\Code\rgbd2occ        # 或 cd D:\Code\rgbd2occ && pip install -e .
rgbd2occ --version                     # 验证: rgbd2occ 0.9.6
```

- 安装后 `rgbd2occ` 命令在该环境的 `Scripts/` 下，**激活环境后任意目录可直接调用**
  （未激活的裸终端需把 `<env>\Scripts` 加入 PATH）。
- 不装包时 `python main.py ...` / `python tools/export_ply.py ...` 等价可用。

### 3. 配置路径

复制模板 `config.example.json` 为 `config.json`（机器相关路径，已 gitignore），填三项：

```json
{ "sunrgbd": {
    "raw_root":     "D:/Datasets/sunrgbd",            // SUN RGB-D 原始数据
    "nuscenes_out": "D:/Datasets/sunrgbd_nuscenes_v3", // nuScenes 格式输出
    "occ_out":      "D:/Datasets/sunrgbd_nuscenes_v3"  // occ 输出 (与上同根=官方合并布局)
} }
```

- 环境变量 `RGBD2OCC_CONFIG` 可指定其他配置文件；CLI 的
  `--out/--out-root/--raw-root/--v3-root` 可临时覆盖。

### 4. 安装自检

```bash
rgbd2occ doctor     # Python/基础依赖/加速项/各产物线可用后端/config 路径有效性
                    # gpu 需要 torch+CUDA+triton 三者齐备; 缺失会降级并说明原因
```

## 命令用法

统一入口 `rgbd2occ`，**每一层都有 `--help`**（顶层 / 子命令组 / 叶子命令 / 产物入口）：

```
rgbd2occ [--version]
  list                    列出已注册数据集/产物 + config 路径
  doctor                  环境/依赖体检
  convert <数据集> <产物> [参数...]   调转换入口, 参数原样透传
  <数据集> <产物> [参数...]          等价简写 (如 rgbd2occ sunrgbd occ ...)
  export {points|occ|preview|demo|info|diff} ...   产物导出/可视化
```

### 生成数据集（convert / 数据集子命令）

已注册产物：`sunrgbd nuscenes | occ | detection | 2d | all`（`all` = 按依赖序跑三线+汇总）。

```bash
# 0) 体检 + 试跑 (每 split 前 3 帧, 秒级验证链路)
rgbd2occ doctor
rgbd2occ sunrgbd all --limit 3

# 1) 全量生成 (官方 Occ3D schema: 0.4m 体素, 200x200x16) —— GPU 最快路径
RGBD2OCC_BACKEND=gpu rgbd2occ sunrgbd all --gpu-batch 16 --ray-stride 1 --with-images
#    同上 CPU 精确基线: 去掉环境变量与 --gpu-batch (逐位可复现, manifest data_variant=exact)

# 2) 细体素数据集 (0.05m 方格, 自动切室内局部范围, 偏离官方 schema 维度)
rgbd2occ sunrgbd all --voxel 0.05

# 3) 单线重跑 (断点续跑: 已有产物自动跳过; 失败帧自动重试)
rgbd2occ convert sunrgbd nuscenes --frames 1,1925    # 指定帧
rgbd2occ convert sunrgbd nuscenes --tables-only      # 只重建 13 张表
rgbd2occ convert sunrgbd occ    --voxel 0.05 --splits train
rgbd2occ convert sunrgbd detection --min-pts 30      # 清洗: 丢点云内点不足的框
rgbd2occ convert sunrgbd 2d                          # 2D gt 框 -> annotations2d.json
```

常用参数（完整见各产物 `--help`，如 `rgbd2occ sunrgbd occ --help`）：

| 参数 | 产品线 | 说明 |
|---|---|---|
| `--limit N` / `--frames 1,1925` | 全部 | 试跑前 N 帧 / 指定帧号 |
| `--splits train val` / `--workers N` | 全部 | split 选择 / 进程数 |
| `--out` `--out-root` `--raw-root` `--v3-root` | 对应线 | 路径临时覆盖 config |
| `--gpu-batch B` | nuscenes/occ/all | GPU 批量流水线（需 `RGBD2OCC_BACKEND=gpu`） |
| `--voxel` `--xrange` `--yrange` `--zrange` | occ/all | 体素与范围（<0.1 自动局部范围; >5 亿格/帧拒绝） |
| `--ray-stride` `--label-vote` `--no-labels` | occ | 射线采样 / 标签投票治理 / 关闭语义 |
| `--with-images` | occ/all | jpg 随包硬链接 |
| `--tables-first` `--tables-only` | nuscenes | 表先行(供 occ 并行) / 只重建表 |
| `--min-pts` `--skip-points` `--points-only` | detection | 清洗 / 跳过点统计 / 只补 num_lidar_pts |
| `--mode single <深度图> --fx .. --fy .. --cx .. --cy ..` | occ | 单帧调试模式（见 --help 全参数） |

### 导出与可视化（export 组）

输出默认写到**当前目录**新建的 `rgbd2occ_export/`（`--out` 可改）；同帧文件名均含
`img-XXXXXX`，CloudCompare 里同帧文件一起拖入即叠加。

```bash
rgbd2occ export info    <数据包根>                        # 数据包概览 (manifest/QC/帧数)
rgbd2occ export points  <数据包>/samples/LIDAR_TOP/train --limit 5 --with-rgb   # 纯 RGB 点云
rgbd2occ export points  <数据包>/samples/LIDAR_TOP/train/img-000123.pcd.bin     # 单文件
rgbd2occ export occ     <数据包>/gts/sunrgbd-train-kv1/<token>   # 占据小方格 + free 两件
rgbd2occ export occ     <数据包>/gts --limit 4                   # 整个 gts 批量
rgbd2occ export occ     <数据包>/gts --revox 0                  # 用数据集 0.4m 体素(默认 0.05 细方格)
rgbd2occ export occ     <数据包>/gts --style point              # 散点可选
rgbd2occ export preview <数据包> --names img-000001             # 原图 + 2D gt 框 PNG
rgbd2occ export preview <数据包> --names img-000001 --with-3d   # 附加 3D 框投影
rgbd2occ export demo    <数据包> --names img-000001 --rgb       # 一键全套可视化 (推荐)
rgbd2occ export diff    <旧manifest.json> <新manifest.json>     # 两次生成回归对比 (一致0/差异1)
```

- `points`：`--with-rgb` 重投影真彩（需原始数据/config）；默认亮度灰度。
- `occ`：占据=按语义类着色小方格（默认 0.05m 细体素重投影，legacy 同款；`--revox`
  调尺寸、`--cube-scale` 格缝）+ free 白色 1/8 抽稀（`--free-sub`），未知不导出。
- `demo`：单帧一键产出 点云/占据 PLY + 2D 框叠加 PNG + 双 BEV 俯视图。
- 3D 框可视化：`preview --with-3d`（投影到原图）。
- 坐标系=自车系 X前/Y左/Z上，与 CloudCompare 默认 Z 向上一致，打开即正视角。

### 用 nuScenes devkit 读取产物

```bash
python tools/read_devkit_example.py <数据包根>   # 未装 devkit 自动回退零依赖模式
```

表目录名 `v1.0-sunrgbd-<split>` 遵循 nuScenes `v1.0-*` 约定，
`NuScenes(version="v1.0-sunrgbd-train", dataroot=<out>)` 可直接加载（实测通过）；
与官方的差异点（ego_pose 平移恒 0 等）见该脚本文档串。

### 环境变量

| 变量 | 作用 |
|---|---|
| `RGBD2OCC_BACKEND` | `exact`(默认,逐位可复现) / `fast`(CPU 近似) / `gpu`(torch+triton)；非 exact 记入 manifest `data_variant` |
| `RGBD2OCC_CONFIG` | 指定其他 config 文件 |
| `RGBD2OCC_LOGLEVEL` | 日志级别 (默认 INFO) |

## 目录结构

```
rgbd2occ/
├── main.py                  # 统一 CLI (argparse 子命令树: list/doctor/export/convert/数据集)
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
│   ├── backends.py          #   exact/fast/gpu 后端选择与降级
│   ├── progress.py          #   tqdm 进度条 (缺失回退)
│   ├── run_qc.py            #   自动质检
│   ├── write_manifest.py    #   溯源 manifest
│   └── render_bev.py        #   BEV 质检渲染 (自动裁剪)
├── nustables/               # nuScenes devkit 格式核心 (改名避免与 pip nuscenes-devkit 冲突)
│   └── tables.py            #   稳定 token(md5) + 13 张表落盘
├── occ/                     # Occ3D 占据标注核心
│   ├── voxel_grid.py        #   官方自车系网格 X前/Y左/Z上 (整除截断, 200x200x16)
│   ├── raycast.py           #   向量化 DDA 射线追踪
│   ├── convert.py           #   convert_frame 三态编码 (+voxel/gmin 元数据)
│   └── annotations.py       #   annotations.json (mini 同构)
├── datasets/                # ★ 数据集适配器 (每种数据集一个目录)
│   ├── __init__.py          #   注册表 DATASETS = {名: {产物: (模块, 入口)}}
│   └── sunrgbd/
│       ├── meta.py          #   原始数据读取: SUNRGBDMeta.mat -> 逐帧真实内参
│       ├── labels.py        #   13 类语义标签 (路径/映射/occ 语义约定)
│       ├── to_nuscenes.py   #   → nuScenes devkit (点云+图像+13张表)
│       ├── to_occ.py        #   → Occ3D 占据标注 (批量+单帧, --voxel 可调)
│       ├── fill_detection.py#   → 3D 检测标注 (groundtruth3DBB 填 sample_annotation)
│       ├── export_2d.py     #   → 2D gt 框侧车 annotations2d.json
│       └── run_all_lines.py #   三线统一编排 (结束逐线汇总+续跑提示)
├── tools/                   # 导出/体检/查看: export_ply(PLY+demo+diff) doctor
│                            #   read_devkit_example view_occ view_pointcloud
└── tests/                   # 52 项测试 (run_all.py 零依赖运行器 / pytest 均可)
```

## 工程化

- **配置**: 机器路径集中在根目录 `config.json`（模板 `config.example.json`；
  环境变量 `RGBD2OCC_CONFIG` 可指定其他文件）。CLI 的 `--out/--raw-root/--v3-root` 可临时覆盖。
- **日志**: 统一 logging（时间戳/级别/进程名，多进程可区分 worker），批量运行同时写
  `<out>/logs/*.log`；级别用 `RGBD2OCC_LOGLEVEL` 或 config 的 `log_level` 调整。
- **容错**: 单帧异常不中断整批；失败帧（含堆栈）写入 `<out>/failed.json`，
  重跑同一命令自动重试（已生成的产物跳过）；有失败帧时进程退出码为 1。
  `sunrgbd all` 结束逐线汇总 (nuscenes/occ/detection 各自退出码) + 续跑提示。
- **测试**: `python tests/run_all.py`（零依赖运行器）或 `pytest tests/`。
  golden 测试签入 1 帧输入与期望产物（点云 bin 与 v2/v3 管线字节级一致、
  occ 输出逐元素一致），任何破坏等价性的实现改动都会被立即抓住。
- **语义标签**: occ 批量转换自动接入 SUN RGB-D 13 类标签
  （`train13labels`/`test13labels`，映射为 **semantic id == 像素值**：
  0=others，1..13=bed…window，17=free），`--no-labels` 关闭；
  包内 `semantic_classes.json` 侧车记录映射，数据集自描述。
- **manifest 溯源**: 每次批量转换写 `manifest_<产物>.json` ——
  生成器版本/git 哈希/时间 + 参数快照(含 voxel/ranges/backend) + 逐帧 md5/点数。
- **自动质检**: 转换后自动跑 QC 并写 `qc_report_<产物>.json`（`--skip-qc` 关闭）：
  文件缺失/大小不符/空帧/内参非法为**错误**（退出码 1）；计数偏低帧为警告；
  另抽样深检 npz/bin 规格（适配自描述网格）。
- **计算后端**: `RGBD2OCC_BACKEND=exact|fast|gpu`（默认 exact=逐位等价/数据兼容基线）。
  fast=CPU 近似（~7% 收益）；gpu=torch CUDA（occ 批量核需 triton，缺失自动降级 fast）。
  非 exact 会在 manifest 记录 `data_variant`；测试始终以 exact 断言。
- **检测清洗/一致性**: `sunrgbd detection --min-pts N` 丢弃点云内点不足的框；
  每次填充自动统计 3D->2D 投影 IoU（监控用，进 manifest）。
  3D 框 basis 镜像基/非正交经 `proper_rotation` 净化（SUN RGB-D 实测 15.6% 镜像基）。
- 许可 MIT（LICENSE）；版本历史见 CHANGELOG.md。

## 新数据集接入（只需三步）

1. 新建 `datasets/<name>/` 目录：`meta.py`（读该数据集的原始数据/内参/帧清单）、
   `to_nuscenes.py` 与 `to_occ.py`（复制 sunrgbd 对应文件，替换
   `prepare_frames()`/`build_tasks()` 中"数据集组织方式 -> 统一帧记录"的部分：
   split/序号/深度路径/内参/图像路径/时间戳）；
2. 在 `datasets/__init__.py` 注册一行；
3. 需要新滤波算子时加到 `common/`，核心其余不动。
   有真实位姿时传入 ego_pose（没有则置空 null，不写假的）。

## 回归验证状态

- 点云线: 4 种相机(kv1/kv2/realsense/xtion)各 1 帧 —— 点云 bin 与 v3 数据集
  **字节级一致**；jpg 字节一致；8 张表逐行一致。
- occ 线: 单帧输出与旧 single_frame_occ.py 逐元素一致；批量输出与
  Occupancy3D-SUNRGBD 大包同 token 逐元素一致；从原始深度独立重推占据 IoU=1.0000。
- `occ.raycast` 向量化与逐条 DDA 逐元素一致；轴向/语义约定/annotations 结构
  均与官方 mini 包逐项比对通过（含用 mini 自身自车位移做的轴序实证）。

## 输出格式

nuScenes devkit 格式（to_nuscenes）:
```
<out>/samples/{CAM_FRONT,LIDAR_TOP}/<split>/img-XXXXXX.{jpg,pcd.bin}
<out>/v1.0-sunrgbd-<split>/  13 张 json 表 (按相机型号分场景, 唯一 K 建标定)
```
点云为 float32 Nx5 (x,y,z,intensity=亮度/255,ring=0), 自车系 x前/y左/z上。

输出采用**合并根目录**（官方 Occ3D 布局: occ 产物与 nuScenes 数据同根）:
```
<root>/samples/{CAM_FRONT,LIDAR_TOP}/<split>/    # 图像 + 点云
<root>/v1.0-sunrgbd-<split>/                     # 13 张 devkit 表 (含检测标注)
<root>/gts/<scene>/<token>/labels.npz            # semantics(0=others,1..13=语义,17=free)+双mask+voxel/gmin
<root>/annotations.json                          # occ 帧链/内外参 (img_path 直接指向 samples/)
<root>/annotations2d.json                        # 2D gt 框侧车 (cls+cls13+像素 bbox)
<root>/semantic_classes.json                     # occ 语义映射侧车
<root>/manifest_{nuscenes,occ,detection,2d}.json # 各产物溯源 (版本/参数/逐帧 md5)
<root>/qc_report_{nuscenes,occ}.json             # 各产物质检报告
```
性能 (RTX 5090, 10335 帧): 点云 ~95 帧/s / occ ~135 帧/s / 检测 ~40s, 三线 <4 分钟。
关键约定: 轴序=自车系 X前/Y左/Z上; 未知=mask==0(语义名义 17); 位姿无真值置空 null;
帧链 prev/next 首尾 "EOF"; 体素 0.4m, (200,200,16) —— 与官方 Occupancy3D mini
实测逐维一致 (0.4m 为官方规格; 室内更细分辨率可 --voxel 0.05, 自动切局部范围,
但偏离官方 schema 维度)。

深度尺度: **/6553.5** (已用椅高物理检验钉死: 实测高/框高中位 1.00;
官方 toolbox 的 /8000 位运算解码不适用于本数据分发版, 勿改)。
3D 框坐标链路: groundtruth3DBB 在 Rtilt 重力对齐系,
p_ego = M^T Rtilt^T p_grav (M 见 fill_detection.py, 源码+椅高双重验证);
镜像基/非正交 basis 先经 proper_rotation 净化 (盒体张成不变)。
