"""量一批已产出数据的实际咬合深度：搬运时方块顶面比两爪最低点高多少。

只读 h5 里逐帧存下来的关节角与方块位姿，不重跑仿真——爪子几何从 URDF 的碰撞网格
顶点算，方块半高固定在 `ITEM_HALF`。歪斜角量的是搬运时方块自身竖直轴与世界 z 轴
的夹角，衡量它有没有在爪里被转动。

用法：改下面的 `H5_FILES` 指向要量的一批 h5，直接跑 `python measure_engagement.py`。
"""

import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from transforms3d.quaternions import quat2mat

import so101_sim  # noqa: F401
from kinematics import ArmKinematics, BaseFrame

TASK = "SO101PickPlaceCube40-v1"
ITEM_HALF = 0.02
S_CARRY = 5


def main():
    env = gym.make(TASK, num_envs=1, control_mode="pd_joint_pos", obs_mode="state",
                   render_mode="all", sim_backend="gpu", domain_randomization=False,
                   max_episode_steps=100)
    u = env.unwrapped
    env.reset(seed=0)
    rp = u.agent.robot.pose
    base = BaseFrame(rp.p[0].cpu().numpy(), rp.q[0].cpu().numpy())
    kin = ArmKinematics(u.agent.urdf_path)
    names = [ln.name for ln in kin._art.get_links()]

    def jaw_low(q):
        kin._pm.compute_forward_kinematics(np.r_[q, np.zeros(max(0, kin.n - 6))][:kin.n])
        z = []
        for nm in ("gripper_link", "moving_jaw_so101_v1_link"):
            ln = kin._art.get_links()[names.index(nm)]
            lp = kin._pm.get_link_pose(names.index(nm))
            Lp, LR = np.asarray(lp.p), quat2mat(np.asarray(lp.q))
            for c in ln.get_collision_shapes():
                v = np.asarray(c.get_vertices())
                cp = c.get_local_pose()
                W = (LR @ (quat2mat(np.asarray(cp.q)) @ v.T + np.asarray(cp.p)[:, None])
                     + Lp[:, None]).T
                z += [base.to_world(x)[2] for x in W]
        return min(z)

    allv = []
    for path in H5_FILES:
        vals = []
        with h5py.File(path, "r") as f:
            for k in f:
                g = f[k]
                art = g["env_states/articulations"]
                art = art[list(art.keys())[0]][:]
                it = g["env_states/actors/item"][:]
                ph = g["phase"][:]
                sv = np.nonzero(ph == S_CARRY)[0]
                if len(sv) < 5:
                    continue
                seg = sv[len(sv) // 4:len(sv) * 3 // 4]           # 搬运中段，抬稳之后
                e = [(it[i, 2] + ITEM_HALF - jaw_low(art[i, 13:19].astype(float))) * 1000
                     for i in seg]
                # 方块歪了多少：搬运时它自身竖直轴与世界 z 的夹角
                tilt = []
                for i in seg:
                    R = quat2mat(it[i, 3:7])
                    tilt.append(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))
                vals.append((float(np.median(e)), float(np.median(tilt))))
        if vals:
            v = np.array(vals)
            print(f"{Path(path).name:24s} n={len(v):3d}  咬合中位 {np.median(v[:,0]):5.1f}mm  "
                  f"最大 {v[:,0].max():5.1f}｜方块歪斜中位 {np.median(v[:,1]):5.1f}°  "
                  f"最大 {v[:,1].max():5.1f}°")
            allv += vals
    if allv:
        a = np.array(allv)
        print(f"\n合计 n={len(a)}  咬合中位 {np.median(a[:,0]):5.1f}mm 最大 {a[:,0].max():5.1f}"
              f"｜歪斜中位 {np.median(a[:,1]):5.1f}° 最大 {a[:,1].max():5.1f}°")
    env.close()


_BATCH_DIR = Path(os.environ["DATASETS_ROOT"]) / "datasets" / "private" / "so101_sim" / "_grasp" / "ds5_cube40"
H5_FILES = sorted(_BATCH_DIR.glob("batch_*.h5"))          # 改这里指向要量的一批 h5


if __name__ == "__main__":
    main()
