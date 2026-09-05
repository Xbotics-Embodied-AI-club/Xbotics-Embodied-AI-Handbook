#!/usr/bin/env python
"""下降段谁先碰到方块：逐帧算夹爪各连杆的最低点，看哪一块先进入方块的高度带。

## 为什么要按连杆分开量

「没抬起来」的主因是下降途中方块被蹭偏 8~10mm。但「下降开度太窄」这条已被单变量试验
否证 —— 把两指间距放宽 10 个百分点后，方块被蹭的时刻与幅度逐位不变 ⇒ 碰它的**不是两指
内侧面**。那就得让几何自己指出是哪一块：把每个连杆的碰撞网格顶点变换到世界系，逐帧取
最低点，看谁先降到方块顶面以下、且水平上落在方块的footprint 里。

纯运动学，不跑物理 —— 这里问的是"按规划的位形，谁会先撞上"，与接触求解无关。

用法：`python measure_descent_clearance.py <场景> <准备目录> <集号>`
"""

import json
import sys
from pathlib import Path

import numpy as np

import recipe
from servo import ArmKinematics, BaseFrame

# 只看夹爪这几块 —— 上臂在下降段离方块很远。
LINKS = ("gripper_link", "moving_jaw_so101_v1_link", "wrist_link")
# 判"落在方块上方"的水平余量（m）：方块半宽 + 这个余量之内就算会撞上。
FOOTPRINT_PAD = 0.005


def link_local_points(kin):
    """各连杆碰撞网格的顶点（连杆自身坐标系）。

    Args:
        kin: `ArmKinematics`，其 `_art` 是一份 CPU sapien articulation。

    Returns:
        `{连杆名: (点数, 3)}`；没有碰撞几何的连杆不出现。
    """
    out = {}
    for link in kin._art.get_links():
        if link.name not in LINKS:
            continue
        pts = []
        for shape in link.get_collision_shapes():
            # sapien 3 的凸网格 shape 直接带 vertices / scale，没有 .geometry 这一层。
            if not hasattr(shape, "vertices"):
                continue
            local = np.asarray(shape.vertices, float) * np.asarray(shape.scale, float)
            pose = shape.get_local_pose()
            rot = np.asarray(pose.to_transformation_matrix(), float)[:3, :3]
            pts.append(local @ rot.T + np.asarray(pose.p, float))
        if pts:
            out[link.name] = np.vstack(pts)
    return out


def main(argv) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    scene, prep, ep = argv[0], Path(argv[1]), int(argv[2])
    spec = recipe.SCENES[scene]
    meta = json.loads((prep / "meta" / f"ep{ep}.json").read_text())
    actions = np.load(prep / "plans" / f"ep{ep}.npy")
    grasp = recipe.close_frame(actions, recipe.CLOSE_PCT[spec["real_task"]])
    if grasp is None:
        sys.exit(f"★ ep{ep} 里找不到「张开后首次合到底」的帧")

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

    local = link_local_points(kin)
    if not local:
        sys.exit("★ 一个连杆的碰撞几何都没取到 —— 空结果不是结论")
    names = [ln.name for ln in kin._art.get_links()]
    item_xy = np.asarray(meta["item_xy"], float)
    top = spec["item_half"] * 2.0
    reach = spec["item_half"] + FOOTPRINT_PAD

    print(f"  ep{ep}：方块顶面 z={top * 1000:.1f}mm，抓取帧 {grasp}")
    print("  连杆                     首次侵入方块上方  离方块中心(mm)  该点高度(mm)")
    for name, pts in local.items():
        idx = names.index(name)
        first = None
        for f in range(grasp + 1):
            kin._pm.compute_forward_kinematics(
                np.r_[np.radians(actions[f, :5]),
                      np.zeros(max(0, kin.n - 5))][:kin.n])
            lp = kin._pm.get_link_pose(idx)
            rot = np.asarray(lp.to_transformation_matrix(), float)[:3, :3]
            # 顶点 → 连杆系 → 基座系 → 世界系。批量做，逐点调 `to_world` 会慢一个量级。
            in_base = pts @ rot.T + np.asarray(lp.p, float)
            world = in_base @ base.R.T + base.p
            # 判据要**两条同时**成立：低于方块顶面、且水平落在方块 footprint 内。
            # 只判"低于顶面"是没有信息的 —— home 位形本来就低于顶面，第 0 帧即成立。
            low = world[world[:, 2] < top]
            if not len(low):
                continue
            dists = np.linalg.norm(low[:, :2] - item_xy, axis=1)
            k = int(np.argmin(dists))
            if dists[k] < reach:
                first = (f, float(dists[k]) * 1000.0, float(low[k, 2]) * 1000.0)
                break
        if first is None:
            print(f"  {name:24s} 全程没有侵入方块上方")
        else:
            print(f"  {name:24s} 第 {first[0]:3d} 帧      {first[1]:10.1f}   {first[2]:10.1f}")
    print("MEASURE_DESCENT_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
