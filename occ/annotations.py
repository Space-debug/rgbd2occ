# -*- coding: utf-8 -*-
"""annotations.json 的构建与写入 (与 Occupancy3D-nuScenes-v1.0-mini 同构)。

结构:
  {train_split: [scene...], val_split: [scene...],
   scene_infos: {scene: {token: 帧条目}}}
帧条目 6 键: timestamp(int), ego_pose, gt_path, prev, next, camera_sensor
  - prev/next: 场景内前后帧 token, 首帧 prev / 末帧 next = "EOF" (官方约定)
  - gt_path: 相对包根的 labels.npz 路径
  - camera_sensor.CAM_FRONT: intrinsics(3x3), extrinsic, ego_pose, img_path
位姿约定: 没有真实位姿时置空(null), 不写假的单位阵。
"""
import json
import os


def pose(translation, rotation):
    """由命令行参数构造位姿 dict; 两者都缺省时返回 None(置空)。"""
    if translation is None and rotation is None:
        return None
    return {"translation": list(translation) if translation is not None else [0.0] * 3,
            "rotation": list(rotation) if rotation is not None else [1.0, 0.0, 0.0, 0.0]}


class OccAnnotations:
    """增量写入(单帧)与批量重建(批转)两种用法共用一套 schema。"""

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"train_split": [], "val_split": [], "scene_infos": {}}
        self.data.setdefault("train_split", [])
        self.data.setdefault("val_split", [])
        self.data.setdefault("scene_infos", {})

    def _frame_entry(self, token, fx, fy, cx, cy, img, ts, ego_pose, extrinsic, gt_rel, scene):
        cam_ego = None if ego_pose is None else {"token": token, "timestamp": ts, **ego_pose}
        return {
            "timestamp": ts,
            "ego_pose": ego_pose,
            "gt_path": gt_rel or f"gts/{scene}/{token}/labels.npz",
            "prev": "EOF", "next": "EOF",
            "camera_sensor": {"CAM_FRONT": {
                "intrinsics": [[float(fx), 0.0, float(cx)],
                               [0.0, float(fy), float(cy)],
                               [0.0, 0.0, 1.0]],
                "extrinsic": extrinsic,
                "ego_pose": cam_ego,
                "img_path": img or "",
            }},
        }

    def add(self, scene, token, split, fx, fy, cx, cy, img="", ts=0,
            ego_pose=None, extrinsic=None, gt_rel=None, link=True):
        """登记一帧; link=True 时按插入顺序维护 prev/next 帧链。"""
        if scene not in self.data[split + "_split"]:
            self.data[split + "_split"].append(scene)
        toks = self.data["scene_infos"].setdefault(scene, {})
        entry = self._frame_entry(token, fx, fy, cx, cy, img, ts,
                                  ego_pose, extrinsic, gt_rel, scene)
        if token in toks:                       # 重跑同一帧: 保留原链位
            entry["prev"] = toks[token].get("prev", "EOF")
            entry["next"] = toks[token].get("next", "EOF")
        elif link and toks:
            last = list(toks)[-1]
            entry["prev"] = last
            toks[last]["next"] = token
        toks[token] = entry

    def rebuild(self, frames):
        """批量重建: frames 为有序帧 dict 列表
        (scene/token/split/fx/fy/cx/cy/img/ts, 可选 ego_pose/extrinsic/gt_rel)。
        顺序即帧链顺序 (建议场景内按 timestamp+token 排序)。"""
        self.data = {"train_split": [], "val_split": [], "scene_infos": {}}
        for f in frames:
            self.add(**f)
        for toks in self.data["scene_infos"].values():
            ks = list(toks)
            for i, k in enumerate(ks):
                toks[k]["prev"] = ks[i - 1] if i > 0 else "EOF"
                toks[k]["next"] = ks[i + 1] if i < len(ks) - 1 else "EOF"

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
