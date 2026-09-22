# rgbd2occ

RGBD 数据集 → (nuScenes devkit 格式 + Occ3D 占据标注) 转换工具集。
每种 RGBD 数据集一个主入口（`xxx2nuscenes.py` / `xxx2occ.py`），共用 `common/`、`occ/`、`nuscenes/` 核心库。

## 目录结构

```
rgbd2occ/
├── common/                  # 工具类目录: 跨数据集、跨产物(点云/occ)共用
│   ├── io.py                #   数据读取: 深度/标签加载与格式归一 (米制/uint8)
│   ├── depth_filter.py      #   深度域滤波: 5x5 中值 + 梯度剔除 (stage1)
│   ├── point_filter.py      #   点云滤波: SOR 半径 + 体素斑点 (stage2/3)
│   └── pointcloud.py        #   编排 depth_to_points + 体素降采样 + nuScenes bin 写出
├── nuscenes/                # nuScenes devkit 格式核心
│   └── tables.py            #   稳定 token(md5) + 13 张表落盘
├── occ/                     # Occ3D 占据标注核心
│   ├── preprocess.py        #   深度有效性掩膜
│   ├── projection.py        #   几何投影 (含 k1/k2 径向畸变)
│   ├── voxel_grid.py        #   官方自车系 X前/Y左/Z上 网格
│   ├── raycast.py           #   向量化 DDA 射线追踪
│   ├── convert.py           #   convert_frame 三态编码
│   └── annotations.py       #   annotations.json (mini 同构)
├── sunrgbd_meta.py          # SUNRGBDMeta.mat 解析 (逐帧真实内参)
├── sunrgbd2nuscenes.py      # 主入口: SUN RGB-D -> nuScenes devkit 格式 (v2点云+v3表合一)
├── sunrgbd2occ.py           # 主入口: SUN RGB-D -> Occ3D 占据标注 (批量+单帧)
└── README.md
```

## 回归验证状态

- `sunrgbd2nuscenes.py`: 4 种相机(kv1/kv2/realsense/xtion)各 1 帧 —— 点云 bin 与
  v3 数据集**字节级一致**；jpg 字节一致；sample_data/ego_pose/calibrated_sensor/
  log/sensor/category/visibility/map 表逐行一致（sample/scene 的 prev/next 链在
  部分帧试跑下为子集链，属预期）。
- `sunrgbd2occ.py`: 单帧输出与旧 single_frame_occ.py 逐元素一致；批量输出与
  Occupancy3D-SUNRGBD 大包同 token 逐元素一致。
- `occ.raycast` 向量化与逐条 DDA 逐元素一致；轴向/语义约定/annotations 结构
  均与官方 mini 包逐项比对通过。

## 用法

```bash
# SUN RGB-D -> nuScenes devkit (点云+图像+13张表, 断点续跑)
python sunrgbd2nuscenes.py                       # 全量 train+val
python sunrgbd2nuscenes.py --limit 3             # 每 split 前 3 帧试跑
python sunrgbd2nuscenes.py --frames 1,1925       # 指定帧号
python sunrgbd2nuscenes.py --tables-only         # 只重建表

# SUN RGB-D -> Occ3D 占据标注
python sunrgbd2occ.py                            # 全量 (v3 表驱动, 断点续跑)
python sunrgbd2occ.py --limit 3
python sunrgbd2occ.py --mode single <depth.png> --fx ... --scene s1 --token t1 --out-root out
```

`sunrgbd2nuscenes.py` 首次运行解析 SUNRGBDMeta.mat 后缓存到
`<out>/sunrgbd_meta_cache.json`；逐帧真实 K 记录在 `<out>/intrinsics_per_frame.json`。
v2 目录(sunrgbd_nuscenes_v2)已被清理也无妨 —— v3 的 bin 即 v2 的字节副本，
新管线独立从原始深度图重建。

## 输出格式

nuScenes devkit 格式（sunrgbd2nuscenes）:
```
<out>/samples/{CAM_FRONT,LIDAR_TOP}/<split>/img-XXXXXX.{jpg,pcd.bin}
<out>/v1.0-sunrgbd-<split>/  13 张 json 表 (按相机型号分场景, 唯一 K 建标定)
```
点云为 float32 Nx5 (x,y,z,intensity=亮度/255,ring=0), 自车系 x前/y左/z上。

Occ3D 占据标注（sunrgbd2occ, 与 Occupancy3D-nuScenes-v1.0-mini 同构）:
```
<out>/annotations.json
<out>/gts/<scene>/<token>/labels.npz   # semantics(0=others..16类,17=free) + mask_lidar + mask_camera
```
关键约定: 轴序=自车系 X前/Y左/Z上; 未知=mask==0(语义名义 17); 位姿无真值置空 null;
帧链 prev/next 首尾 "EOF"; 体素 0.4m, (200,200,16)。

## 新数据集接入指南

1. 新建 `xxx2nuscenes.py` / `xxx2occ.py`, 复制 SUN RGB-D 入口骨架;
2. 实现 `prepare_frames()`/`build_tasks()`: 数据集组织方式 -> 统一帧记录
   (split/序号/深度路径/内参/图像路径/时间戳);
3. 深度换算与滤波挂 `common` (需要新滤波算子时加在 common/point_filter.py);
4. 有真实位姿时传入 ego_pose (没有则置空, 不写假的)。
