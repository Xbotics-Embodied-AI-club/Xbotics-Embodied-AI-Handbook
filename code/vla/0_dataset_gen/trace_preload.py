#!/usr/bin/env python
"""抓取之前发生了什么：方块和料箱各自在第几帧被推动。

空抓那一集（两指合到底而中间无物）只有两种可能：瞄偏了，或者合拢前方块已经被碰走。
瞄点残差门已经排除前者 ⇒ 必然是后者。本工具定位「谁在第几帧被碰的」：

  · 料箱被推动 ⇒ 手臂撞上了箱体（该集料箱离物体只有 11cm）
  · 方块先动而料箱不动 ⇒ 是夹爪自己在下降途中蹭到了方块

两个量都直接读物理状态，不依赖接触 API 的可用性。

用法：`python trace_preload.py <场景> <准备目录> <集号>`
"""

import sys
from pathlib import Path

import numpy as np

import recipe

# 判"被推动"的位移门槛（m）。复位后物体从半空落定会有毫米级沉降，取 3mm 避开它。
MOVED_M = 0.003


def main(argv) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    scene, prep, ep = argv[0], Path(argv[1]), int(argv[2])
    spec = recipe.SCENES[scene]
    actions = np.load(prep / "plans" / f"ep{ep}.npy")
    grasp = recipe.close_frame(actions, recipe.CLOSE_PCT[spec["real_task"]])
    if grasp is None:
        sys.exit(f"★ ep{ep} 里找不到「张开后首次合到底」的帧")

    from so101_sim.config_lerobot_robot import SO101SimRobotConfig
    from so101_sim.lerobot_robot import JOINT_NAMES, SO101SimRobot

    robot = SO101SimRobot(SO101SimRobotConfig(
        task=spec["env_id"], episode_length=len(actions) + 10,
        initial_state_path=str(prep / "states" / f"ep{ep}.json")))
    robot.connect()
    inner = robot._env._env.unwrapped
    item, bin_p = [], []
    for row in actions[:grasp + 1]:
        robot.get_observation()
        item.append(np.asarray(inner.get_state_dict()["actors"]["item"][0][:3].cpu(), float))
        bin_p.append(np.asarray(inner.bin.pose.p[0].cpu(), float))
        robot.send_action({f"{n}.pos": float(row[i]) for i, n in enumerate(JOINT_NAMES)})
    robot.disconnect()

    item, bin_p = np.stack(item), np.stack(bin_p)
    for name, track in (("方块", item), ("料箱", bin_p)):
        # 与**落定之后**的位置比：复位那几帧的沉降不算被推动。
        base = track[5]
        moved = np.flatnonzero(np.linalg.norm(track[5:] - base, axis=1) > MOVED_M)
        total = float(np.linalg.norm(track[-1] - base)) * 1000.0
        when = f"第 {int(moved[0]) + 5} 帧" if len(moved) else "全程没动"
        print(f"  {name}：{when}起被推动，到抓取帧共移动 {total:.1f}mm")
    print(f"  抓取帧 {grasp} / 共 {len(actions)} 帧")
    print("TRACE_PRELOAD_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
