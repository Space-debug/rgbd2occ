# rgbd2occ

RGBD 数据集 → Occ3D-nuScenes 同构 3D 占据标注转换工具集。
每种 RGBD 数据集一个主入口（`xxx2occ.py`），共用 `occ/` 核心库。

## 目录结构

```
rgbd2occ/
├── occ/                    # 数据集无关的核心库
│   ├── data_io.py          # 数据读取: 深度/标签加载与格式归一 (米制/uint8)
│   ├── preprocess.py       # 数据预处理: 深度有效性掩膜 (各管线滤波的接入点)
│   ├── projection.py       # 几何投影: 深度图->相机系点云 (含 k1/k2 径向畸变)
│   ├── voxel_grid.py       # 体素网格: 官方自车系 X前/Y左/Z上, 整除截断
│   ├── raycast.py          # 向量化 DDA 射线追踪 (Amanatides & Woo)
│   ├── convert.py          # 转换核心 convert_frame: 三态编码 (Occ3D 官方约定)
│   └── annotations.py      # annotations.json 读写 (mini 同构, 帧链/置空位姿)
├── sunrgbd2occ.py          # 主转换: SUN RGB-D (v3 表驱动批量 + 单帧)
└── README.md
```

## 输出格式 (与 Occupancy3D-nuScenes-v1.0-mini 同构, 已逐项验证)

```
<out-root>/
├── annotations.json          # train/val split + scene_infos(6键帧条目+prev/next帧链)
└── gts/<scene>/<token>/labels.npz
    ├── semantics   uint8 (200,200,16)  0=others, 1..16=语义类, 17=free
    ├── mask_lidar  uint8 0/1
    └── mask_camera uint8 0/1
```

关键约定：
- 轴序 = 官方自车系：**X=前, Y=左, Z=上**（相机系 x右/y下/z前 经 `cam_to_ego_axes` 变换）
- 体素 0.4m，范围 x/y∈[-40,40]、z∈[-1,5.4]，网格 (200,200,16)，维度整除截断
- **三态编码**：占据=semantics∈[0,16]；真 free=sem==17 且 mask==1；
  **未知=mask==0**（语义名义上保持 17，训练/评测时被忽略）
- 位姿：没有真实值时 `ego_pose`/`extrinsic` 置空（null），不写假的单位阵
- 帧链：场景内按 timestamp+token 排序 prev/next，首尾为 `"EOF"`

## 用法 (SUN RGB-D)

```bash
python sunrgbd2occ.py                    # 全量 train+val (10335 帧, ~7 分钟/10 核)
python sunrgbd2occ.py --limit 3          # 每场景前 3 帧试跑
python sunrgbd2occ.py --splits train     # 只转 train
python sunrgbd2occ.py --ann-only         # 只重建 annotations.json
python sunrgbd2occ.py --mode single <depth.png> --fx 529.5 --fy 529.5 --cx 365 --cy 265 \
    --depth-scale 0.000152592 --valid-range 0.3 8 --scene s1 --token t1 --out-root out
```

批量模式数据来源：`sunrgbd_nuscenes_v3`（真实逐帧内参/场景/token），
深度 `/6553.5` 换算并掩膜 [0.3,8]m。断点续跑：已存在的 labels.npz 自动跳过。

## 新数据集接入指南

1. 在仓库根目录新建 `xxx2occ.py`（如 `scannet2occ.py`），复制 `sunrgbd2occ.py` 骨架；
2. 实现 `build_tasks()`：把该数据集的组织方式映射为任务列表
   （每帧: scene/token/split/depth 路径/内参 fx fy cx cy/img 登记路径/ts）；
3. 深度换算与预处理放进 `occ.preprocess`（或入口内调用），几何与编码全部复用 `occ`；
4. 有真实位姿时经 `--ego-translation/--ego-rotation` 传入（null 语义见上）。

## 验证情况

- 向量化 raycast 与逐条 DDA 循环版输出逐元素一致；
- 轴序/语义约定/annotations 结构均与官方 mini 包逐项比对通过
  （含用 mini 自身自车位移做的轴序实证、404 帧 npz 规格统计）。
