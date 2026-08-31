"""把 vanilla 里录的成功抓取轨迹重渲成 **KIT 双相机 + 真机套件 mesh/颜色**。

同目录 `replay.py` 是单 `base_camera` 版（只换外观）。这里是 KIT 版：目标环境有 top/wrist
两路相机（绑 URDF 标定光学系）、物体是 STEP 转出的 mesh 并带 STEP 文件名里的真实颜色。

为什么要 replay 而不是直接在 KIT 环境里跑策略：专家的 CNN 编码器是 **3 通道**（单相机），
KIT 环境发 top+wrist **6 通道**，直接跑 shape 就不对。而 `env_states` 与相机无关——把
vanilla 那条成功轨迹的关节角与物体位姿灌进 KIT 环境逐帧重渲即可，动作/状态原样不动。
两边碰撞尺寸逐字一致（`SO101Lift*Real-v1` 与 `SO101KitLift*Real-v1` 共用同一份 STEP 真值），
所以灌进去是同一套物理，不是"换了个世界"。

★GPU 后端换状态必走全序列：`set_state_dict` 只写物理状态，**不重算 link 位姿**。缺了
`_gpu_apply_all → gpu_update_articulation_kinematics → _gpu_fetch_all` 就会出现
「wrist 相机定格在 reset 帧」「整幅全黑未打光」两种废帧。
"""

import shutil
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import h5py

import so101_sim  # noqa: F401  注册 KIT 任务

RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# ★关掉 squint 的 greenscreen 叠底。squint 默认 `apply_overlay=True` +
# `rgb_overlay_path=black_overlay.png`：除机器人与 item 之外的一切（**包括白台面**）都被
# 那张黑图顶替 —— 渲出来就是"黑底悬空的臂和方块"（实测 top 均值 7.2）。这是它给 sim2real
# 做背景随机化的手段，不是我们要的画面。关掉后拿到真实场景几何：白台面 + 台沿（top 均值
# 189.9 / wrist 200.0），与真机 task1 的白台面一致。
DISABLE_BACKGROUND_OVERLAY = dict(apply_overlay=False)


def _blit_gpu_state(env, articulation_qpos, actor_poses, articulation_key):
    """把一帧 env_states 灌进 GPU 环境，并重算 link 位姿（相机才跟得上夹爪）。"""
    sd = env.unwrapped.get_state_dict()
    sd["articulations"][articulation_key] = torch.as_tensor(
        articulation_qpos[None], device="cuda")
    for name, poses in actor_poses.items():
        if name in sd["actors"]:
            sd["actors"][name] = torch.as_tensor(poses[None], device="cuda")
    env.unwrapped.set_state_dict(sd)

    scene = env.unwrapped.scene
    scene._gpu_apply_all()
    scene.px.gpu_update_articulation_kinematics()
    scene._gpu_fetch_all()
    scene.update_render()


def replay_kit(kit_task: str, in_h5: Path, out_h5: Path,
               source_articulation: str | None = None) -> Path:
    """逐帧把 `in_h5`（vanilla 录的）重渲成 KIT 双相机，写成新的 h5。

    输出 h5 保留原 `actions`/`success`/`env_states` 等，把 `obs/sensor_data` 换成
    top/wrist 两路 RGB（原来的 base_camera 整组删掉）。
    """
    shutil.copy(in_h5, out_h5)
    env = gym.make(kit_task, num_envs=1, obs_mode="rgb+segmentation", render_mode="all",
                   sim_backend="gpu", domain_randomization=False, reconfiguration_freq=1,
                   domain_randomization_config=DISABLE_BACKGROUND_OVERLAY,
                   sensor_configs=dict(width=RENDER_WIDTH, height=RENDER_HEIGHT))
    env.reset(seed=0)
    # KIT 环境里机器人的 articulation 键名是 so101_kit（vanilla 是 so101），两者同构 25 维。
    kit_articulation = next(iter(env.unwrapped.get_state_dict()["articulations"]))

    with h5py.File(out_h5, "a") as f:
        for traj_name in list(f.keys()):
            g = f[traj_name]
            # ★源 articulation 键名随 vanilla 侧的 agent uid 变（`so101` / `so101_slow` / …），
            # 不能硬编码成 "so101"；h5 里机器人是唯一的 articulation，直接取。
            src_key = source_articulation or next(iter(g["env_states/articulations"]))
            qpos_seq = g[f"env_states/articulations/{src_key}"][:]
            actor_seq = {k: g[f"env_states/actors/{k}"][:]
                         for k in g["env_states/actors"].keys()}
            n_frames = qpos_seq.shape[0]

            sensor = g["obs/sensor_data"]
            for cam in list(sensor.keys()):
                del sensor[cam]

            buffers = {}
            # ★同时把几何夹持量 `jaw_gap` 存进 h5：常驻断言 tests/test_grasp_geometry.py 读它。
            # 不存则断言 skip、门是空的。这里在 blit 完状态后调 evaluate() 取（几何量与
            # 是否 step 物理无关，不像接触力那样必须 step）。
            jaw_gap = np.empty(n_frames, np.float32)
            # 同时存 `is_item_grasped`：断言要用它划"该夹着的帧"窗口。
            # 用几何代理（离台 + gap<contact_offset）划窗口太松——实测同一批数据
            # 代理窗口贴合率 7.5%，而 env 自己的 grasped 窗口是 49.3%。
            # ★`is_item_grasped` 依赖接触力，接触力必须 step 物理才算得出来
            #   （只 set_state_dict 读出来全是 0，踩过）。
            is_grasped = np.zeros(n_frames, bool)
            for i in range(n_frames):
                _blit_gpu_state(env, qpos_seq[i],
                                {k: v[i] for k, v in actor_seq.items()},
                                kit_articulation)
                env.unwrapped.scene.step()
                info = env.unwrapped.evaluate()
                jaw_gap[i] = float(info["jaw_gap"][0]) if "jaw_gap" in info else np.nan
                is_grasped[i] = bool(info["is_item_grasped"][0])
                for cam, cam_obs in env.unwrapped.get_obs()["sensor_data"].items():
                    rgb = cam_obs["rgb"][0].cpu().numpy().astype(np.uint8)
                    if cam not in buffers:
                        buffers[cam] = np.empty((n_frames, *rgb.shape), np.uint8)
                    buffers[cam][i] = rgb

            # 上面已经把这一集原有的相机整组删掉了，此时若一张图都没渲出来，写回去的就是
            # 一条只有状态、没有画面的轨迹——而 jaw_gap 仍会照常写入，从数据侧完全看不出
            # 缺了什么。宁可在这里停下，也不要让无图的集悄悄进库。
            if not buffers:
                raise RuntimeError(f"{traj_name}: 相机没有返回任何图像，重渲无效")

            for cam, frames in buffers.items():
                sensor.create_dataset(f"{cam}/rgb", data=frames, compression="gzip",
                                      compression_opts=4)
            if not np.isnan(jaw_gap).all():
                for name, values in (("jaw_gap", jaw_gap), ("is_item_grasped", is_grasped)):
                    if "infos" in g and name in g["infos"]:
                        del g[f"infos/{name}"]
                    g.create_dataset(f"infos/{name}", data=values)
            height, width = next(iter(buffers.values())).shape[1:3]
            print(f"  {traj_name}: {n_frames} 帧 × {sorted(buffers)} @{height}×{width}")

    env.close()
    print(f"replay_kit done -> {out_h5}")
    return out_h5
