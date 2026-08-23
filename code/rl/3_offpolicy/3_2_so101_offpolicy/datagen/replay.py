"""回放 env_states，把同一套成功动作**重渲染成目标外观/相机**。

这是 sim=real 的关键一招：策略只在训练外观下工作（容易），而数据集要贴近真机（黑臂、
白底、真机相机位姿/FOV）。做法不是重训——rollout 存了每帧的 `env_states`，这里把它
`set_state_dict` 回目标外观的环境里重新渲染一帧，动作/状态原样不动，只有 RGB 变了。
所以同一批轨迹能渲出任意多套外观做增广。

**相机安装位置 / FOV / 外观**都在下面内联，是"后置可改"的旋钮——真机相机标定确认后，
改这里再重跑 replay + to_lerobot 即可，无需重训 v6 策略。
"""

import os
import shutil
from pathlib import Path

import numpy as np
import cv2
import torch
import gymnasium as gym

import so101_sim  # 注册任务

# ── 后置可改的旋钮（默认=环境原始外观/相机；真机标定后改这里）─────────────────
APPEARANCE = "black_white"          # "default"（原样）/ "black_white"（黑臂+白底，贴近真机）
ROBOT_COLOR = (0.0, 0.0, 0.0)       # 黑臂
BACKGROUND = 255                    # 白底（overlay 灰度值）
# 相机位姿 / FOV 留作真机标定接口：None=沿用环境默认（tune_camera 对齐后在此填真机值）
CAMERA_EYE = None
CAMERA_TARGET = None
CAMERA_FOV = None


def _make_target_env(task: str, work: Path):
    """按目标外观造一个 num_envs=1 的环境，供重渲。"""
    dr_config = dict(robot_color=ROBOT_COLOR, apply_overlay=True)
    overlay = work / "overlay.png"
    cv2.imwrite(str(overlay), np.full((128, 128, 3), BACKGROUND, np.uint8))
    dr_config["rgb_overlay_path"] = str(overlay)
    return gym.make(task, num_envs=1, obs_mode="rgb+segmentation", render_mode="all",
                    sim_backend="gpu", domain_randomization=False,
                    domain_randomization_config=dr_config,
                    sensor_configs=dict(width=128, height=128))


def replay(task: str, in_h5: Path, out_h5: Path, work: Path) -> Path:
    import h5py

    if APPEARANCE == "default":
        shutil.copy(in_h5, out_h5)   # 原样外观：不用重渲
        return out_h5

    shutil.copy(in_h5, out_h5)
    env = _make_target_env(task, work)
    env.reset(seed=0)
    # 机器人 articulation 键名 / 相机名不硬编码：随任务变（vanilla 是 so101，KIT 系是
    # so101_kit_slow；单相机环境叫 base_camera，双相机环境是 top/wrist），直接从环境自己
    # 声明的状态字典里取，环境换了也不用回来改这里。
    art_key = next(iter(env.unwrapped.get_state_dict()["articulations"]))
    with h5py.File(out_h5, "a") as f:
        for traj in f:
            g = f[traj]
            src_art_key = next(iter(g["env_states/articulations"]))
            art_qpos = g[f"env_states/articulations/{src_art_key}"][:]
            actors = {k: g[f"env_states/actors/{k}"][:] for k in g["env_states/actors"].keys()}
            cam_name = next(iter(g["obs/sensor_data"]))
            rgb_ds = g[f"obs/sensor_data/{cam_name}/rgb"]
            for i in range(rgb_ds.shape[0]):
                sd = env.unwrapped.get_state_dict()
                sd["articulations"][art_key] = torch.tensor(art_qpos[i:i + 1], device="cuda")
                for k, v in actors.items():
                    sd["actors"][k] = torch.tensor(v[i:i + 1], device="cuda")
                env.unwrapped.set_state_dict(sd)
                rgb = env.unwrapped.get_obs()["sensor_data"][cam_name]["rgb"][0].cpu().numpy()
                rgb_ds[i] = rgb.astype(np.uint8)
    env.close()
    print(f"replay done (APPEARANCE={APPEARANCE}) -> {out_h5}")
    return out_h5


TASK = "SO101PickPlaceCube40-v1"

if __name__ == "__main__":
    work = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_gen" / TASK
    replay(TASK, work / "rollout.h5", work / "replay.h5", work)
