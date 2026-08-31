"""给已有的 replay h5 补 `infos/jaw_gap` + `infos/is_item_grasped` + `infos/is_true_pinch`。

为什么单独一步：这两个量是**验收判据的数据源**（`tests/test_grasp_geometry.py`），
而 replay 阶段很长（128 集 × 400 帧 × 双相机），把标注耦合在里面意味着一旦标注逻辑
要改就得重跑整个 replay。拆开后可对已有 h5 反复标注，不重渲。

原理：逐帧把 `env_states` 灌回 KIT 环境，`scene.step()` 后读 `evaluate()`。
★必须 step —— `is_item_grasped` 依赖接触力，只 `set_state_dict` 读出来全是 0。
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import gymnasium as gym

import so101_sim  # noqa: F401

KIT_TASK = "SO101KitPlaceCube4RealSlow-v1"


def annotate(h5_path: Path, kit_task: str = KIT_TASK) -> tuple[int, int]:
    """就地给每条轨迹补两个 infos 字段。返回 (标注集数, 总帧数)。

    Args:
        h5_path: 待标注的 h5。
        kit_task: 对应的 KIT 任务 id，用来取物体几何。

    Returns:
        标注写回后的 h5 路径。
    """
    env = gym.make(kit_task, num_envs=1, obs_mode="state", render_mode="all",
                   sim_backend="gpu", domain_randomization=False, reconfiguration_freq=1)
    env.reset(seed=0)
    u = env.unwrapped
    art_key = next(iter(u.get_state_dict()["articulations"]))

    n_traj = n_frames_total = 0
    with h5py.File(h5_path, "a") as f:
        keys = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
        for ti, key in enumerate(keys):
            g = f[key]
            src = next(iter(g["env_states/articulations"]))
            qpos = g[f"env_states/articulations/{src}"][:]
            actors = {a: g[f"env_states/actors/{a}"][:] for a in g["env_states/actors"]}
            n = qpos.shape[0]

            gap = np.empty(n, np.float32)
            grasped = np.zeros(n, bool)
            # 真·两侧夹持：压入 ∧ 落在一对相对面 ∧ 都在面中部（见 envs._pinch_facts）
            pinch = np.zeros(n, bool)
            center_off = np.empty(n, np.float32)
            # ★进 success 的那个门判据也必须落盘。少了它，下游按名字取指标会全拿到
            # 缺字段 → 被当成"没有数据" → 报出"一集都不达标"的假阴性（v12 踩过）。
            center_grasp = np.zeros(n, bool)
            for i in range(n):
                sd = u.get_state_dict()
                sd["articulations"][art_key] = torch.as_tensor(qpos[i:i + 1], device="cuda")
                for a, v in actors.items():
                    if a in sd["actors"]:
                        sd["actors"][a] = torch.as_tensor(v[i:i + 1], device="cuda")
                u.set_state_dict(sd)
                sc = u.scene
                sc._gpu_apply_all()
                sc.px.gpu_update_articulation_kinematics()
                sc._gpu_fetch_all()
                sc.step()
                info = u.evaluate()
                gap[i] = float(info["jaw_gap"][0])
                grasped[i] = bool(info["is_item_grasped"][0])
                pinch[i] = bool(info["is_true_pinch"][0])
                center_off[i] = float(info["pinch_center_off"][0])
                center_grasp[i] = bool(info["is_center_grasp"][0])

            for name, arr in (("jaw_gap", gap), ("is_item_grasped", grasped),
                              ("is_true_pinch", pinch), ("pinch_center_off", center_off),
                              ("is_center_grasp", center_grasp)):
                path = f"infos/{name}"
                if path in g:
                    del g[path]
                g.create_dataset(path, data=arr)
            n_traj += 1
            n_frames_total += n
            if (ti + 1) % 16 == 0:
                print(f"  标注 {ti + 1}/{len(keys)} 集", flush=True)

    env.close()
    print(f"annotate done: {n_traj} 集 / {n_frames_total} 帧 -> {h5_path}")
    return n_traj, n_frames_total


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_grasp"
        / "v6_place_cube_4cm_in_bin" / "replay_kit.h5")
    annotate(target)
