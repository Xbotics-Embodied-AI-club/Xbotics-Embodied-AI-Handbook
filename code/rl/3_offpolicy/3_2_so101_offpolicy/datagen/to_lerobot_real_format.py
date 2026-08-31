"""把仿真轨迹导成**与真机 task1 逐字段同定义**的 LeRobotDataset。

同目录 `to_lerobot.py` 走 ManiSkill 官方转换器，出来的是"ManiSkill 口径"：弧度、归一化
动作、`joint_0`/`action_0` 这类占位字段名、`robot_type=unknown`。真机 task1 是另一套口径。
逐字段比对下来共七处不同，本文件全部对齐：

| 项 | 真机 task1 | 官方转换器出的仿真 | 本文件 |
|---|---|---|---|
| `robot_type` | `so_follower` | `unknown` | `so_follower` |
| 单位 | **度** | 弧度 | **度** |
| action 语义 | **绝对位置**（下一帧要到的角） | 归一化 `[-1,1]` 增量 | **绝对位置** |
| state/action `names` | `shoulder_pan.pos` … | `joint_0` / `action_0` | `shoulder_pan.pos` … |
| 视频编码 | h264 | mp4v | h264 |
| 多出的字段 | — | `task`（列） | 去掉 |
| fps | 30 | 20 | 见下方说明 |

**动作语义怎么来的**：真机 action 实测满足 `|action[:-1] − state[1:]| = 1.59°`，即"下一帧要
到达的绝对关节角"。仿真侧同含义的量是控制器累加的目标 `obs/agent/controller/target_qpos`
（squint 的 `pd_joint_target_delta_pos` 内部维护它），实测 `|tq[:-1] − qpos[1:]| = 0.98°`
——与真机同定义且更精确。**不是**用归一化 action 反推。

**fps**：真机 30、仿真控制频率 20。这里**不改标签冒充 30**（那会让轨迹回放快 1.5×）。
要真正 30fps 必须用 `control_freq=30` 重跑仿真——见 `FPS` 常量处的说明。
"""

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

# 真机 task1 的字段名与机器人类型（唯一真相源 = 真机数据集自己的 meta/info.json）
JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
ROBOT_TYPE = "so_follower"
CAMERAS = ["top", "wrist"]

# 仿真控制频率，现已与真机对齐 = 30（`*Slow-v1` 任务把 `SimConfig.control_freq` 设成 30、
# `sim_freq` 设 120，因为 sim_freq 必须整除 control_freq）。
# **这个值必须等于环境真实的 control_freq** —— 只改标签不改环境会让轨迹按 fps 回放时倍速失真。
FPS = 30


def _episode_arrays(h5_path):
    """从 h5 逐集取出真机口径的 state / action（度、绝对位置）与两路 RGB。"""
    out = []
    with h5py.File(h5_path) as f:
        # 用字典序而非数字序：官方 convert_to_lerobot 就是这么排的，两条导出线保持
        # 同一个 episode_index -> 同一条轨迹，才能拿两个口径互相对照。
        for key in sorted(f.keys()):
            g = f[key]
            qpos = g["obs/agent/noisy_qpos"][:]                      # (T+1,6) rad
            target = g["obs/agent/controller/target_qpos"][:]        # (T+1,6) rad 绝对目标
            n_act = g["actions"].shape[0]
            # state = 当前关节角；action = 该帧下发的绝对目标（下一帧到达）
            state = np.degrees(qpos[:n_act]).astype(np.float32)
            action = np.degrees(target[:n_act]).astype(np.float32)
            rgb = {c: g[f"obs/sensor_data/{c}/rgb"][:n_act] for c in CAMERAS}
            out.append(dict(state=state, action=action, rgb=rgb, length=n_act))
    return out


def _write_video(frames, path, fps):
    """h264 写盘（与真机同编码）。cv2 的 mp4v 会出 0 字节，走 ffmpeg。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"_{path.stem}_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for i, fr in enumerate(frames):
        cv2.imwrite(str(tmp / f"{i:05d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    shutil.rmtree(tmp)


def _stats(values):
    """LeRobot v3.0 的 stats 结构（逐维 min/max/mean/std/count）。"""
    a = np.asarray(values, np.float64)
    flat = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(-1, 1)
    return dict(min=flat.min(0).tolist(), max=flat.max(0).tolist(),
                mean=flat.mean(0).tolist(), std=flat.std(0).tolist(),
                count=[int(flat.shape[0])])


def _build_info(n_episodes, n_frames, fps):
    """照真机 task1 的 info.json 结构写，字段名/dtype/shape 逐项对齐。"""
    def feat(shape, names=None, dtype="float32"):
        return {"dtype": dtype, "shape": list(shape), "names": names, "fps": float(fps)}

    features = {
        "action": feat([6], JOINT_NAMES),
        "observation.state": feat([6], JOINT_NAMES),
        "timestamp": feat([1]),
        "frame_index": feat([1], dtype="int64"),
        "episode_index": feat([1], dtype="int64"),
        "index": feat([1], dtype="int64"),
        "task_index": feat([1], dtype="int64"),
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(fps), "video.height": 480, "video.width": 640,
                     "video.channels": 3, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "has_audio": False},
        }
    return {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": 1,
        "total_videos": n_episodes * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "data_files_size_in_mb": 0,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }


def export(h5_path: Path, out_dir: Path, task_name: str, fps: int = FPS) -> Path:
    """把 `h5_path` 导成真机口径的 LeRobotDataset 到 `out_dir`。"""
    episodes = _episode_arrays(h5_path)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data" / "chunk-000").mkdir(parents=True)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    rows, ep_meta, global_index = [], [], 0
    for ep_idx, ep in enumerate(episodes):
        n = ep["length"]
        for i in range(n):
            rows.append({
                "action": ep["action"][i],
                "observation.state": ep["state"][i],
                "timestamp": np.float32(i / fps),
                "frame_index": np.int64(i),
                "episode_index": np.int64(ep_idx),
                "index": np.int64(global_index + i),
                "task_index": np.int64(0),
            })
        for cam in CAMERAS:
            _write_video(ep["rgb"][cam],
                         out_dir / "videos" / f"observation.images.{cam}" / "chunk-000"
                         / f"file-{ep_idx:03d}.mp4", fps)
        meta = {
            "episode_index": ep_idx,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": global_index, "dataset_to_index": global_index + n,
            "tasks": [task_name], "length": n,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        }
        for cam in CAMERAS:
            meta[f"videos/observation.images.{cam}/chunk_index"] = 0
            meta[f"videos/observation.images.{cam}/file_index"] = ep_idx
            meta[f"videos/observation.images.{cam}/from_timestamp"] = 0.0
            meta[f"videos/observation.images.{cam}/to_timestamp"] = (n - 1) / fps
        for key, arr in (("action", ep["action"]), ("observation.state", ep["state"])):
            for stat, val in _stats(arr).items():
                meta[f"stats/{key}/{stat}"] = val
        ep_meta.append(meta)
        global_index += n

    pd.DataFrame(rows).to_parquet(out_dir / "data" / "chunk-000" / "file-000.parquet",
                                  index=False)
    pd.DataFrame(ep_meta).to_parquet(
        out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task_name], name="task")).to_parquet(
        out_dir / "meta" / "tasks.parquet")

    info = _build_info(len(episodes), global_index, fps)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    all_state = np.concatenate([e["state"] for e in episodes])
    all_action = np.concatenate([e["action"] for e in episodes])
    (out_dir / "meta" / "stats.json").write_text(json.dumps({
        "action": _stats(all_action), "observation.state": _stats(all_state),
    }, indent=4))

    print(f"真机口径数据集 -> {out_dir}  ({len(episodes)} 集 / {global_index} 帧 @ {fps}fps)")
    print(f"  robot_type={ROBOT_TYPE}  单位=度  action=绝对位置  字段名={JOINT_NAMES[0]}…")
    return out_dir
