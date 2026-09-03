#!/usr/bin/env python
"""量夹住那一段的抖动：夹爪关节与方块在爪里各自抖多少。

## 为什么要量它

目审读出「物体被抓住的时候，夹爪一直在抖」。而**穿模本身是可接受的** —— 老版就是靠
碰撞体内缩换来的过盈，用户明确「和老版一样，那种穿模可以接受，就是稳定的，夹爪包裹住
物体的是可以的」。所以靶子只有「稳」，不是「不叠」。

机制上抖是这么来的：`pd_joint_pos` 的目标位在物理止点之外约 0.4 rad，PD 算出的力矩
（stiffness 1e3 × 0.4）远超夹爪力矩上限 2.0，于是**一旦饱和，阻尼项也在同一个被截的
和里面，速度反馈就失去权限**——爪子被一个恒定 2 N·m 顶着，接触求解器的回弹没人压制。
真机伺服堵转不嗡嗡响，是因为减速箱本身有摩擦，那一路与控制器输出无关。

## 判据从哪来

老版那批是用户认可的「稳定」，所以它的抖动量就是上限。本工具对新旧两批用同一段窗口、
同一组统计量，直接给出对比 —— 不自己编阈值。

## 窗口怎么取

从夹爪指令首次到底那一帧，到它再次张开之前一帧。这一段里手臂在动（抬起、平移、下降），
所以**方块的抖动要在夹爪坐标系里量** —— 在世界系里量会把正常搬运也算成抖。

用法：`python measure_grip_jitter.py <场景> <动作来源> [<集号>]`
      动作来源可以是准备目录（含 `plans/`）或已录数据集根（含 `data/`）。
      不给集号就逐集都量并汇总。
"""

import sys
from pathlib import Path

import numpy as np
import recipe
from transforms3d.quaternions import quat2mat

EE_LINK = "gripper_frame_link"


def _rot_angle_deg(rot):
    """一个旋转矩阵对应的转角（度）。

    Args:
        rot: `(3, 3)` 旋转矩阵。

    Returns:
        转角，落在 `[0, 180]`。

    用迹反解：`trace(R) = 1 + 2cos θ`。夹到 `[-1, 1]` 是因为浮点会让它略微越界。
    """
    cos = (float(np.trace(rot)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def load_actions(source):
    """从准备目录或数据集根取逐集动作。

    Args:
        source: 准备目录（含 `plans/`）或数据集根（含 `data/`）。

    Returns:
        `{集号: (帧数, 6)}`。
    """
    source = Path(source)
    if (source / "plans").is_dir():
        return {int(p.stem.removeprefix("ep")): np.load(p)
                for p in sorted((source / "plans").glob("ep*.npy"))}
    if (source / "data").is_dir():
        from check_success import episode_actions
        return episode_actions(source)
    sys.exit(f"★ {source} 既没有 plans/ 也没有 data/ —— 认不出这是什么来源")


def held_window(actions):
    """夹住那一段的帧区间 `[起, 止)`。

    Args:
        actions: `(帧数, 6)` 真机口径动作，第 6 列是夹爪行程百分比。

    Returns:
        `(起, 止)`；这一集没有"合到底"就给 `None`。
    """
    start = recipe.close_frame(actions)
    if start is None:
        return None
    after = np.flatnonzero(actions[start:, 5] > recipe.CLOSE_PCT + 1e-6)
    return start, (start + int(after[0]) if len(after) else len(actions))


def trace(env_id, state_path, actions):
    """复跑一集，逐帧记夹爪关节实际角度与方块在夹爪系里的位姿。

    Args:
        env_id: 已注册的环境 id。
        state_path: 强制初始状态 json。
        actions: `(帧数, 6)` 动作。

    Returns:
        `(夹爪实际角 (帧数,), 方块在夹爪系的位置 (帧数, 3), 方块相对夹爪的旋转 (帧数, 3, 3))`。

    ★ 位置与朝向都在**夹爪坐标系**里量。早先朝向量的是方块自身 z 轴与**世界** z 的夹角，
      而搬运段腕本身在转、刚性握住的方块跟着转也会让它变（量出 ep0 69°、ep2 90°、
      ep7 89°，看着像方块在爪里翻了个面）—— 那测不出打滑。参考系选错的度量会给出
      看似显著、实则无关的数（同型教训见 bd `xb-ro2i`）。
    """
    from so101_sim.config_lerobot_robot import SO101SimRobotConfig
    from so101_sim.lerobot_robot import JOINT_NAMES, SO101SimRobot

    robot = SO101SimRobot(SO101SimRobotConfig(
        task=env_id, episode_length=len(actions) + 10,
        initial_state_path=str(state_path)))
    robot.connect()
    inner = robot._env._env.unwrapped
    link = inner.agent.robot.links_map[EE_LINK]
    grip, local, rel = [], [], []
    for row in actions:
        obs = robot.get_observation()
        grip.append(float(obs["gripper.pos"]))
        p_ee = np.asarray(link.pose.p[0].cpu(), float)
        rot = quat2mat(np.asarray(link.pose.q[0].cpu(), float))
        state = np.asarray(inner.get_state_dict()["actors"]["item"][0][:7].cpu(), float)
        local.append(rot.T @ (state[:3] - p_ee))
        rel.append(rot.T @ quat2mat(state[3:7]))
        robot.send_action({f"{n}.pos": float(row[i]) for i, n in enumerate(JOINT_NAMES)})
    robot.disconnect()
    return np.asarray(grip), np.stack(local), np.stack(rel)


def main(argv) -> int:
    if len(argv) not in (2, 3):
        sys.exit(__doc__)
    scene, source = argv[0], Path(argv[1])
    only = int(argv[2]) if len(argv) == 3 else None
    spec = recipe.SCENES[scene]
    prep = source if (source / "states").is_dir() else None
    if prep is None:
        sys.exit(f"★ {source} 里没有 states/ —— 复跑必须有强制初始状态，否则场景对不上")

    actions = load_actions(source)
    if only is not None:
        actions = {only: actions[only]}
    print("  集   夹住帧数  夹爪逐帧|Δ|中位/p95(%)  峰峰(%)   方块在爪里逐帧|Δ|p95(mm)  转动p95(°/帧)")
    rows = []
    for ep in sorted(actions):
        window = held_window(actions[ep])
        if window is None:
            print(f"  ep{ep}: 没有合到底的段，跳过")
            continue
        lo, hi = window
        grip, local, rel = trace(spec["env_id"], prep / "states" / f"ep{ep}.json", actions[ep])
        g, pos, rot_rel = grip[lo:hi], local[lo:hi], rel[lo:hi]
        dg = np.abs(np.diff(g))
        dp = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0
        dr = np.array([_rot_angle_deg(rot_rel[i].T @ rot_rel[i + 1])
                       for i in range(len(rot_rel) - 1)])
        row = (float(np.median(dg)), float(np.percentile(dg, 95)), float(g.max() - g.min()),
               float(np.percentile(dp, 95)), float(np.percentile(dr, 95)))
        rows.append(row)
        print(f"  ep{ep}  {hi - lo:8d}  {row[0]:10.4f}/{row[1]:.4f}  {row[2]:8.4f}  "
              f"{row[3]:22.3f}  {row[4]:12.2f}")

    if not rows:
        sys.exit("★ 一集都没量到 —— 空结果不是「不抖」")
    arr = np.asarray(rows)
    print(f"\n  {len(arr)} 集中位：夹爪逐帧|Δ| {np.median(arr[:, 0]):.4f}% / "
          f"p95 {np.median(arr[:, 1]):.4f}%　峰峰 {np.median(arr[:, 2]):.4f}%　"
          f"方块|Δ|p95 {np.median(arr[:, 3]):.3f}mm　转动p95 {np.median(arr[:, 4]):.2f}°/帧")
    print("MEASURE_GRIP_JITTER_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
