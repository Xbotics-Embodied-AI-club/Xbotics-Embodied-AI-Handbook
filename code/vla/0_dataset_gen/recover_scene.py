"""从已交付的数据集反解一集的初始场景，写成 `--robot.initial_state_path` 能吃的 json。

## 为什么需要它

要用 `lerobot-record` 把已交付的动作重录一遍（编码与真机一致、字段逐字相同），
就得先把物体摆回原处 —— 同一串动作在物体不同的场景里跑，结果不同。而生成源 h5
（里面存着 `env_states`）已经不在了，物体位置只能从数据本身反解。

## 怎么反解

物体被夹住的那一刻，**夹持口袋**就在物体中心。已交付的 `observation.state` 是逐帧
关节角，喂给正运动学就得到那一刻的口袋位置。

★ 口袋不是"两指中点"。`finger1_tip` / `finger2_tip` 是纯 frame、**没有碰撞体**
  （碰撞体只挂在 `gripper_link` 与 `moving_jaw_so101_v1_link` 上），而且两指中点随
  张开角在 x 与 z 两个方向同时移动，不是刚体点。实测对 9 份真值：中点法均值 13.3mm、
  口袋法均值 **1.0mm**（最大 2.7mm）。口袋是标定值，逐场景声明在 `recipe.py`。

★ 夹持帧按**张开块结束**认，不按夹爪的绝对值。绝对值两头都堵：回家位形的夹爪是
  1.90°（行程 10.8%），比 cube40 合拢阶梯停的 10°（18.2%）**还小** ⇒ 阈值法会选中
  集尾的回家段（实测差 50~163mm）；而阶梯又是接触即停的，cube20 实际没走到最紧那档
  ⇒ 「降到最紧档」永不成立。夹爪先张到进场角、再降到下降角，这一整段是「张开块」，
  块结束的下一帧就是合拢起点 —— 那时物体刚坐进爪里、手臂还没抬起。

## 只覆盖两个 actor

其余 actor（台面、相机挂点）与机器人 articulation 一律取环境自己复位后的状态。
自己编那些数就会编错，而 `_apply_initial_state` 又要求所有会动的 actor 都在。

料箱只覆盖 xy、朝向沿用环境自己撒的：松手点离箱心实测 0.3~1.1cm，朝向对"落不落进去"
影响很小，而从数据里反解朝向没有可靠依据。落不进去的集由成功判据剔掉，不靠猜。

## 进程顺序：先 GPU 环境，后 CPU 副本

`create_pinocchio_model()` 只在 CPU 场景上成立，而 sapien 的 `physx.enable_gpu()`
要求「在任何其它 PhysX 代码之前」调用。**先起 CPU 场景再 make GPU 环境会直接抛**
`GPU PhysX can only be enabled once before any other code involving PhysX`。反过来可以。

## 一个进程反解整个场景

建 GPU 环境要几秒，而反解本身是纯 CPU 的正运动学（每集毫秒级）。所以**一个场景
只建一次环境**，把该场景全部集一次反完；录制阶段才按集并行。

用法：`python recover_scene.py <场景> <源数据集根> <输出目录> [<集号,集号,...>]`
      不给集号则反解全部集。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import recipe

# 夹爪那一维；六维里恒为最后一维。
GRIP = 5
# 夹爪 0~100 行程的两端，就是 URDF 里夹爪关节的限位。
GRIPPER_LO_DEG = -10.0
GRIPPER_HI_DEG = 100.0
# 判「张开块」的余量（行程百分比），吸收控制器跟随误差。
OPEN_MARGIN_PCT = 3.0
# 料箱的台面高度（m）。箱底贴着台面放，这个值是环境自己的常量。
BIN_Z = 0.0025


def load_all(root):
    """一次把整份数据集按集读进内存。

    逐集去扫全部 parquet 是 O(集数 × 分片数)；一次读完再按集分组是 O(分片数)。
    整份仿真数据集才几百 MB 的数值列（不含视频），装得下。

    Args:
        root: 源数据集根目录。

    Returns:
        `{集号: (action, state)}`，数组形状都是 `(帧数, 6)`。
    """
    per_ep = {}
    for f in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(f, columns=["episode_index", "action", "observation.state"])
        idx = np.asarray(table.column("episode_index").to_pylist())
        act = np.stack(table.column("action").to_pylist()).astype(float)
        sta = np.stack(table.column("observation.state").to_pylist()).astype(float)
        for ep in np.unique(idx):
            hit = idx == ep
            prev = per_ep.get(int(ep))
            if prev is None:
                per_ep[int(ep)] = (act[hit], sta[hit])
            else:
                per_ep[int(ep)] = (np.vstack([prev[0], act[hit]]),
                                   np.vstack([prev[1], sta[hit]]))
    if not per_ep:
        raise FileNotFoundError(f"{root} 下没读到任何集")
    return per_ep


def to_sim_rad(values):
    """真机口径 → URDF 弧度。臂关节度→弧度，夹爪行程百分比→弧度。

    Args:
        values: 六维真机口径值。

    Returns:
        六维弧度。
    """
    out = np.deg2rad(np.asarray(values, float))
    out[GRIP] = np.deg2rad(GRIPPER_LO_DEG
                           + values[GRIP] / 100.0 * (GRIPPER_HI_DEG - GRIPPER_LO_DEG))
    return out


def open_block(action, descent_pct):
    """第一段「张开块」的起止帧。

    Args:
        action: `(帧数, 6)`，夹爪在最后一维、单位是行程百分比。
        descent_pct: 下降相位的夹爪张开量（行程百分比）。

    Returns:
        `(块首帧, 块尾帧)`；整集没张开过、或张开块延到集尾没有合拢段时给 `None`。
        用返回 `None` 而不是抛错，是因为整批反解时**一集坏不该把整批打断** ——
        调用方跳过它并如实计数即可。
    """
    wide = action[:, GRIP] >= descent_pct - OPEN_MARGIN_PCT
    hit = np.flatnonzero(wide)
    if len(hit) == 0:
        return None
    start = int(hit[0])
    end = start
    while end + 1 < len(wide) and wide[end + 1]:
        end += 1
    if end + 1 >= len(wide):
        return None
    return start, end


def release_frame(action, descent_pct):
    """松手帧 —— 合拢之后夹爪再次张开的第一帧。

    Args:
        action: `(帧数, 6)`，夹爪在最后一维。
        descent_pct: 下降相位的夹爪张开量（行程百分比）。

    Returns:
        帧号；合拢之后再没张开过（那一集没有松手动作）时给 `None`。
    """
    block = open_block(action, descent_pct)
    if block is None:
        return None
    tail = action[block[1] + 1:, GRIP]
    hit = np.flatnonzero(tail >= descent_pct - OPEN_MARGIN_PCT)
    if len(hit) == 0:
        return None
    return block[1] + 1 + int(hit[0])


def main(argv) -> int:
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    scene, source_root, out_dir = argv[0], argv[1], Path(argv[2])
    spec = recipe.SCENES[scene]

    import gymnasium as gym
    import so101_sim  # noqa: F401  导入即注册场景

    # ★先 GPU 环境、后 CPU 副本 —— 顺序反了会抛 PhysX 的一次性启用错误。
    env = gym.make(spec["env_id"], num_envs=1, obs_mode="state", render_mode="all",
                   sim_backend="gpu", domain_randomization=False, reconfiguration_freq=1)
    env.reset(seed=0)
    base_state = {group: {name: [float(v) for v in tensor[0].cpu()]
                          for name, tensor in items.items()}
                  for group, items in env.unwrapped.get_state_dict().items()}
    robot_pose = env.unwrapped.agent.robot.pose
    base_p = [float(v) for v in robot_pose.p[0].cpu()]
    base_q = [float(v) for v in robot_pose.q[0].cpu()]
    env.close()

    from servo import ArmKinematics, BaseFrame
    from so101_sim.robots.so101_base.so101 import SO101

    kin = ArmKinematics(SO101.urdf_path)
    frame = BaseFrame(base_p, base_q)
    pocket = recipe.pocket(scene)
    descent_pct = ((spec["open_descent_deg"] - GRIPPER_LO_DEG)
                   / (GRIPPER_HI_DEG - GRIPPER_LO_DEG) * 100.0)

    def pocket_world(qpos_real):
        """某一帧的夹持口袋在世界系下的位置。"""
        p, rot = kin.ee_pose_local(to_sim_rad(qpos_real))
        return frame.to_world(p + rot @ pocket)

    episodes = load_all(source_root)
    wanted = ([int(x) for x in argv[3].split(",")] if len(argv) == 4
              else sorted(episodes))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)

    done, skipped = 0, []
    for ep in wanted:
        action, state = episodes[ep]
        # 反解不出来的集**跳过并如实计数**，不写一个猜的场景出去 —— 那会让整条
        # 重录链在这一集上安静地产出错数据。
        block = open_block(action, descent_pct)
        release_at = release_frame(action, descent_pct)
        if block is None or release_at is None:
            skipped.append(ep)
            continue
        grasp_at = block[1] + 1
        item_xyz = pocket_world(state[grasp_at])
        bin_xy = pocket_world(state[release_at])[:2]

        out = {group: dict(items) for group, items in base_state.items()}
        out["actors"]["item"] = ([float(item_xyz[0]), float(item_xyz[1]), spec["item_half"]]
                                 + [1.0, 0.0, 0.0, 0.0] + [0.0] * 6)
        bin_prev = out["actors"]["bin"]
        out["actors"]["bin"] = ([float(bin_xy[0]), float(bin_xy[1]), BIN_Z]
                                + list(bin_prev[3:7]) + [0.0] * 6)
        # 机器人从这一集的第 0 帧位形起步，与源集同起点。
        art_name = next(iter(out["articulations"]))
        art = list(out["articulations"][art_name])
        art[13:19] = [float(v) for v in to_sim_rad(state[0])]
        art[19:25] = [0.0] * 6
        out["articulations"][art_name] = art

        (out_dir / f"ep{ep}.json").write_text(json.dumps(out, ensure_ascii=False))
        # ★元信息**另存一个文件**，不塞进状态 json：`_apply_initial_state` 会把状态里
        #   每一组都往 tensor 转，多一个字符串键就是 `could not convert string to float`。
        #   状态文件必须就是 ManiSkill `get_state_dict()` 那个形状，不多不少。
        #   录制要按帧数算 `episode_time_s`（lerobot-record 只认秒数、不认帧数）。
        (out_dir / "meta" / f"ep{ep}.json").write_text(json.dumps(
            {"scene": scene, "episode": ep, "n_frames": len(action),
             "grasp_frame": grasp_at, "release_frame": int(release_at)},
            ensure_ascii=False))
        done += 1

    print(f"  {scene}: 反解 {done}/{len(wanted)} 集 → {out_dir}")
    if skipped:
        print(f"  跳过 {len(skipped)} 集（张开块或松手帧认不出来）：{skipped[:10]}")
    print("RECOVER_SCENE_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
