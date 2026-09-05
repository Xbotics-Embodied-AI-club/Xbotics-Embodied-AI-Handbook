#!/usr/bin/env python
"""腕滚等价解为什么没被采用：逐候选报「重解收不收敛」与「对齐残余多少」。

实测三集失败集本有落在稳定区的等价解（−43.5~−49.9°）、在限位内、按偏好也该胜出，
却没被 `solve_aligned_grasp` 的择优循环采用。否掉它们的只可能是那两道后置检查：
候选要 `solve_approach_and_grasp` 重解收敛、且 `|jaw_yaw_error| ≤ 2°`。

**两个子成因必须分开计数**，不能合成一句「解不出来」—— 前者要改重解的种子，
后者要放宽门，改错了方向就是白跑一轮。

本工具只读不写，不改产线。

用法：`python diagnose_roll_candidates.py <场景> <准备目录> <集号>`
"""

import json
import sys
from pathlib import Path

import numpy as np

import recipe
from grasp_ik import PREFERRED_WROLL, aim_point, jaw_yaw_error, solve_approach_and_grasp
from servo import ArmKinematics, BaseFrame

APPROACH_H = 0.06


def main(argv) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    scene, prep, ep = argv[0], Path(argv[1]), int(argv[2])
    spec = recipe.SCENES[scene]
    meta = json.loads((prep / "meta" / f"ep{ep}.json").read_text())
    state = json.loads((prep / "states" / f"ep{ep}.json").read_text())
    actions = np.load(prep / "plans" / f"ep{ep}.npy")
    grasp = recipe.close_frame(actions, recipe.CLOSE_PCT[spec["real_task"]])

    import gymnasium as gym
    import so101_sim  # noqa: F401
    env = gym.make(spec["env_id"], num_envs=1, obs_mode="state",
                   sim_backend="physx_cpu", domain_randomization=False)
    env.reset(seed=0)
    inner = env.unwrapped
    pose = inner.agent.robot.pose
    base = BaseFrame(np.asarray(pose.p[0].cpu(), float), np.asarray(pose.q[0].cpu(), float))
    kin = ArmKinematics(inner.agent.urdf_path)
    env.close()

    from so101_sim.robots.so101_base.so101 import (
        grip_rad_from_pct,
        gripper_limit_rad,
    )
    low, high = gripper_limit_rad()
    lo = np.array([-1.9199, -1.9199, -1.69, -1.6581, -1.1731, low])
    hi = np.array([1.9199, 1.7453, 1.69, 1.8326, 4.4120, high])
    grip = grip_rad_from_pct(spec["open_approach_pct"])
    pocket = aim_point(kin, scene)
    item_xy = np.asarray(meta["item_xy"], float)
    item_yaw = np.radians(meta["item_yaw_deg"])
    seed_q = np.asarray(state["articulations"]["so101"], float)[:6] \
        if "so101" in state.get("articulations", {}) else np.radians(actions[0, :5]).tolist() + [grip]
    seed_q = np.asarray(seed_q, float)[:6]

    used = float(actions[grasp, 4])
    print(f"  ep{ep}：规划采用腕滚 {used:+.1f}°，偏好 {np.degrees(PREFERRED_WROLL):+.0f}°")
    # 与产线同构：产线的候选重解用 q_seed 作种子（换成已收敛的 qa 已被实测否证、已退回）。
    # 这里仍先解一次，只为确认"采用的那个腕滚本身解得出来"这个诊断前提成立。
    _, _, _, ok0 = solve_approach_and_grasp(
        kin, base, seed_q, item_xy, spec["item_half"], APPROACH_H, grip, lo, hi,
        pocket=pocket, wroll=float(np.radians(used)))
    if not ok0:
        sys.exit("★ 连采用的那个腕滚都解不出来 —— 诊断前提不成立")
    print("  候选腕滚   离偏好   重解收敛   对齐残余     结论")
    for j in (-2, -1, 0, 1, 2):
        cand = np.radians(used) + j * np.pi / 2.0
        if not (lo[4] + 0.02 <= cand <= hi[4] - 0.02):
            continue
        _, qg, _, ok = solve_approach_and_grasp(
            kin, base, seed_q, item_xy, spec["item_half"], APPROACH_H, grip, lo, hi,
            pocket=pocket, wroll=float(cand))
        err = abs(np.degrees(jaw_yaw_error(kin, qg, item_yaw))) if ok else float("nan")
        verdict = ("★ 重解不收敛" if not ok else
                   ("★ 对齐门挡掉" if err > 2.0 else "可用"))
        print(f"  {np.degrees(cand):+8.1f}°  {abs(np.degrees(cand) - np.degrees(PREFERRED_WROLL)):7.1f}"
              f"   {'是' if ok else '否':^8s}  {err:8.2f}°   {verdict}")
    print("DIAGNOSE_ROLL_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
