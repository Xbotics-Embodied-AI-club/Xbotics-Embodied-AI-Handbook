#!/usr/bin/env python
"""按当前口径重标夹持口袋：捏住那一刻，物体实际落在夹爪坐标系的哪里。

## 为什么必须重标

`recipe` 里的 `pocket_x` / `pocket_z` 是**旧口径**下标定的：合拢走阶梯（停在 13° 那一档）、
夹爪力矩上限 100。现在改成捏到底、力矩 2.0，两指停的位置变了（实测堵在 23~26% 而不是 21%），
方块被卡住的地方跟着变 —— 口袋没跟着重标，抓取瞄点就是错的。

实账：口袋没重标时 10 集里 5 集失败，且失败全是**抓不牢**而不是放不准 ——
掉落帧分别是抬起后第 7 帧、22 帧、66 帧，还有一集从第 2 帧起就没抓起来过。

## 怎么量

按已有规划复跑到「捏到底并保持」那一段的最后一帧（规划里夹爪指令首次为 `CLOSE_PCT`
之后的第 `CLOSE_FRAMES` 帧），此时两指已经堵转、方块被卡住。那一刻量

    口袋 = R_ee^T · (物体世界位置 − 夹爪 frame 世界位置)

即物体中心在 `gripper_frame_link` 局部系里的坐标。逐集取中位就是该写回配方的值。
直接读仿真里那个 link 的世界位姿，不再走 CPU 运动学副本 —— 少一层坐标约定就少一处能错的地方。

★ 这是**一次迭代**：瞄点用旧口袋，量出来的是"用旧口袋瞄时方块实际停在哪"。把它写回配方
  再跑一轮，瞄点就对了。收敛与否看逐集散布 —— 散布大说明方块每次停的位置本来就不稳定，
  那是别的病，不能靠改口袋治。

用法：`python calibrate_pocket.py <场景> <准备目录>`
"""

import sys
from pathlib import Path

import numpy as np
import recipe
from expert import CLOSE_FRAMES
from transforms3d.quaternions import quat2mat

EE_LINK = "gripper_frame_link"


def held_pocket(env_id, state_path, actions):
    """捏住那一刻，物体在夹爪局部系里的坐标。

    Args:
        env_id: 已注册的环境 id。
        state_path: 强制初始状态 json。
        actions: `(帧数, 6)` 该集的规划动作。

    Returns:
        `(3,)` 口袋坐标（m）；这一集的规划里没有"捏到底"段时给 `None`。
    """
    from so101_sim.config_lerobot_robot import SO101SimRobotConfig
    from so101_sim.lerobot_robot import JOINT_NAMES, SO101SimRobot

    closing = recipe.close_frame(actions)
    if closing is None:
        return None
    stop = min(closing + CLOSE_FRAMES, len(actions))

    robot = SO101SimRobot(SO101SimRobotConfig(
        task=env_id, episode_length=len(actions) + 10,
        initial_state_path=str(state_path)))
    robot.connect()
    inner = robot._env._env.unwrapped
    for row in actions[:stop]:
        robot.get_observation()
        robot.send_action({f"{n}.pos": float(row[i]) for i, n in enumerate(JOINT_NAMES)})
    link = inner.agent.robot.links_map[EE_LINK]
    p_ee = np.asarray(link.pose.p[0].cpu(), float)
    rot = quat2mat(np.asarray(link.pose.q[0].cpu(), float))
    p_item = np.asarray(inner.get_state_dict()["actors"]["item"][0][:3].cpu(), float)
    robot.disconnect()
    return rot.T @ (p_item - p_ee)


def main(argv) -> int:
    if len(argv) != 2:
        sys.exit(__doc__)
    scene, prep = argv[0], Path(argv[1])
    spec = recipe.SCENES[scene]

    plans = sorted((prep / "plans").glob("ep*.npy"),
                   key=lambda p: int(p.stem.removeprefix("ep")))
    if not plans:
        sys.exit(f"★ {prep}/plans 里没有规划文件")
    rows = []
    for path in plans:
        ep = path.stem
        actions = np.load(path)
        pocket = held_pocket(spec["env_id"], prep / "states" / f"{ep}.json", actions)
        if pocket is None:
            print(f"    {ep}: 规划里没有捏到底段，跳过")
            continue
        rows.append(pocket)
        print(f"    {ep}: 口袋 ({pocket[0] * 1000:+7.2f}, {pocket[1] * 1000:+7.2f}, "
              f"{pocket[2] * 1000:+7.2f}) mm")

    if not rows:
        sys.exit("★ 一集都没量到 —— 空结果不是结论")
    arr = np.stack(rows)
    median = np.median(arr, axis=0)
    spread = arr.max(axis=0) - arr.min(axis=0)
    print(f"\n  {len(rows)} 集：中位 ({median[0] * 1000:+.2f}, {median[1] * 1000:+.2f}, "
          f"{median[2] * 1000:+.2f}) mm　逐轴散布 "
          f"({spread[0] * 1000:.2f}, {spread[1] * 1000:.2f}, {spread[2] * 1000:.2f}) mm")
    print(f"  配方现值 pocket_x={spec['pocket_x'] * 1000:+.2f} "
          f"POCKET_Y={recipe.POCKET_Y * 1000:+.2f} pocket_z={spec['pocket_z'] * 1000:+.2f} mm")
    print("CALIBRATE_POCKET_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
