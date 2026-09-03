#!/usr/bin/env python
"""逐集量抓取瞄点的残差：规划让夹持口袋落到哪，与方块中心差多少。

## 为什么要把「瞄点」和「结果」分开量

`calibrate_pocket.py` 量的是**结果** —— 跑完物理后方块实际停在夹爪局部系的哪里。
它把两件事混在一个数里：瞄偏了，和瞄准了但夹持时被挤动了。这份只做**运动学**：
按规划里抓取那一帧的关节角做正解，算口袋点落在世界的哪，与方块中心比。

  · 残差小、结果散 ⇒ 瞄准了，是夹持在挤动方块 —— 该查夹爪几何/接触
  · 残差大 ⇒ 逆解没把口袋送到方块上 —— 该在准备阶段弃掉这些场景，而不是调物性

实账：10 集里 5 集失败，其中 3 集**从没抬起来过**（最高 z 只比静置高 3.5mm，方块被碰走）。
那三集不可能是夹持力的问题 —— 夹爪压根没把方块围住。而俯仰角在成败两组完全相同
（都是 80~85°），所以也不是姿态。剩下能解释"碰走"的就是瞄点。

用法：`python check_aim.py <场景> <准备目录> [<失败集号,逗号分隔>]`
      给了失败集号就在每行右边标出成败，便于看残差与成败是否一一对应。
"""

import json
import sys
from pathlib import Path

import numpy as np
import recipe
from grasp_ik import jaw_yaw_error
from servo import ArmKinematics, BaseFrame


def _gripper_limits():
    """夹爪关节的上下限（弧度），从 URDF 现读。"""
    from so101_sim.robots.so101_base.so101 import gripper_limit_rad
    return gripper_limit_rad()


def base_and_urdf(env_id):
    """建一次环境，取机器人基座位姿与 URDF 路径。

    Args:
        env_id: 已注册的环境 id。

    Returns:
        `(BaseFrame, urdf 路径)`。

    基座在这个环境里是固定的，逐集不变，所以只建一次。走 CPU PhysX：这里只要运动学，
    起 GPU 后端反而会踩「`physx.enable_gpu()` 必须最先调用」那条限制。
    """
    import gymnasium as gym
    import so101_sim  # noqa: F401  导入即注册
    env = gym.make(env_id, num_envs=1, obs_mode="state", sim_backend="physx_cpu",
                   domain_randomization=False)
    env.reset(seed=0)
    inner = env.unwrapped
    pose = inner.agent.robot.pose
    frame = BaseFrame(np.asarray(pose.p[0].cpu(), float), np.asarray(pose.q[0].cpu(), float))
    urdf = inner.agent.urdf_path
    env.close()
    return frame, urdf


def main(argv) -> int:
    if len(argv) not in (2, 3):
        sys.exit(__doc__)
    scene, prep = argv[0], Path(argv[1])
    bad = {int(x) for x in argv[2].split(",")} if len(argv) == 3 else set()
    spec = recipe.SCENES[scene]
    pocket = np.asarray(recipe.pocket(scene), float)

    plans = sorted((prep / "plans").glob("ep*.npy"),
                   key=lambda p: int(p.stem.removeprefix("ep")))
    if not plans:
        sys.exit(f"★ {prep}/plans 里没有规划 —— 空结果不是结论")

    frame, urdf = base_and_urdf(spec["env_id"])
    kin = ArmKinematics(urdf)

    # 下降那一段的指尖间距：两指要跨的宽度超过它就夹不住。
    open_pct = spec["open_descent_pct"]
    low, high = _gripper_limits()
    tips = kin.tips_local(np.r_[np.zeros(5), low + open_pct / 100.0 * (high - low)])
    span = float(np.linalg.norm(tips["finger2_tip"] - tips["finger1_tip"])) * 1000.0

    print(f"  下降开度 {open_pct:.1f}% 时指尖间距 {span:.2f}mm，"
          f"方块边长 {spec['item_half'] * 2000:.0f}mm\n")
    print("  集    口袋瞄点残差 (dx, dy, dz) mm       模长   对齐残余  要跨   成败")
    rows = []
    for path in plans:
        ep = int(path.stem.removeprefix("ep"))
        meta = json.loads((prep / "meta" / f"{path.stem}.json").read_text())
        actions = np.load(path)
        grasp = int(np.flatnonzero(actions[:, 5] <= recipe.CLOSE_PCT + 1e-6)[0])
        qpos = np.radians(actions[grasp, :5])
        p_local, rot_local = kin.ee_pose_local(qpos)
        aim = frame.to_world(p_local + rot_local @ pocket)
        want = np.array([*meta["item_xy"], spec["item_half"]])
        d = (aim - want) * 1000.0
        norm = float(np.linalg.norm(d))
        resid = abs(np.degrees(jaw_yaw_error(
            kin, np.r_[qpos, 0.0], np.radians(meta["item_yaw_deg"]))))
        need = spec["item_half"] * 2000.0 / np.cos(np.radians(resid))
        rows.append((ep, norm, ep not in bad))
        tag = "" if not bad else ("成功" if ep not in bad else "★失败")
        print(f"  ep{ep}  ({d[0]:+7.2f}, {d[1]:+7.2f}, {d[2]:+7.2f})   {norm:6.2f}mm  "
              f"{resid:6.2f}°  {need:6.2f}mm  {tag}")

    arr = np.array([r[1] for r in rows])
    print(f"\n  {len(arr)} 集残差模长：中位 {np.median(arr):.2f}mm  最大 {arr.max():.2f}mm")
    if bad:
        good = np.array([n for _, n, s in rows if s])
        ng = np.array([n for _, n, s in rows if not s])
        print(f"  成功 {len(good)} 集 {good.min():.2f}~{good.max():.2f}mm　"
              f"失败 {len(ng)} 集 {ng.min():.2f}~{ng.max():.2f}mm")
    print("CHECK_AIM_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
