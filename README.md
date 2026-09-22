# rgbd2occ

RGBD 数据集 → (nuScenes devkit 格式 + Occ3D 占据标注) 转换工具集。
**一个主程序 + 每种数据集一个适配目录**，核心库共用；新增数据集只需加目录并登记注册表。

## 目录结构

```
rgbd2occ/
├── main.py                  # 主生成程序: python main.py <数据集> <产物> [参数...]
├── common/                  # 工具类库: 跨数据集、跨产物共用
│   ├── io.py                #   数据读取: 深度/标签加载与格式归一 (米制/uint8)
│   ├── depth_filter.py      #   深度域滤波: 5x5 中值 + 梯度剔除 (stage1)
│   ├── point_filter.py      #   点云滤波: SOR 半径 + 体素斑点 (stage2/3)
│   └── pointcloud.py        #   depth_to_points 编排 + 体素降采样 + nuScenes bin 写出
├── nuscenes/                # nuScenes devkit 格式核心
│   └── tables.py            #   稳定 token(md5) + 13 张表落盘
├── occ/                     # Occ3D 占据标注核心
│   ├── preprocess.py        #   深度有效性掩膜
│   ├── projection.py        #   几何投影 (含 k1/k2 径向畸变)
│   ├── voxel_grid.py        #   官方自车系 X前/Y左/Z上 网格
│   ├── raycast.py           #   向量化 DDA 射线追踪
│   ├── convert.py           #   convert_frame 三态编码
│   └── annotations.py       #   annotations.json (mini 同构)
├── datasets/                # ★ 数据集适配器 (每种数据集一个目录)
│   ├── __init__.py          #   注册表 DATASETS = {名: {产物: (模块, 入口)}}
│   └── sunrgbd/
│       ├── meta.py          #   原始数据读取: SUNRGBDMeta.mat -> 逐帧真实内参
│       ├── to_nuscenes.py   #   → nuScenes devkit (点云+图像+13张表)
│       └── to_occ.py        #   → Occ3D 占据标注 (批量+单帧)
└── README.md
```

## 用法

```bash
python main.py                                    # 列出已注册数据集/产物
python main.py sunrgbd nuscenes                   # 全量 -> nuScenes 格式 (断点续跑)
python main.py sunrgbd nuscenes --limit 3         # 每 split 前 3 帧试跑
python main.py sunrgbd nuscenes --frames 1,1925   # 指定帧号
python main.py sunrgbd nuscenes --tables-only     # 只重建表
python main.py sunrgbd occ                        # 全量 -> Occ 标注 (v3 表驱动)
python main.py sunrgbd occ --mode single <深度图> --fx 529.5 --fy 529.5 \
    --cx 365 --cy 265 --depth-scale 0.000152592 --scene s1 --token t1 --out-root out
```

`sunrgbd nuscenes` 首次运行解析 SUNRGBDMeta.mat 后缓存到
`<out>/sunrgbd_meta_cache.json`；逐帧真实 K 记录在 `<out>/intrinsics_per_frame.json`。

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

Occ3D 占据标注（to_occ, 与 Occupancy3D-nuScenes-v1.0-mini 同构）:
```
<out>/annotations.json
<out>/gts/<scene>/<token>/labels.npz   # semantics(0=others..16类,17=free) + mask_lidar + mask_camera
```
关键约定: 轴序=自车系 X前/Y左/Z上; 未知=mask==0(语义名义 17); 位姿无真值置空 null;
帧链 prev/next 首尾 "EOF"; 体素 0.4m, (200,200,16)。
