"""SO-101 position reach —— 第 5 讲 §3.3 最小闭环。

本文件把讲义三个片段按顺序组装成一个可运行、可复现的最小 reach 闭环：
  1. load_model()：加载官方 scene.xml，运行时加入目标 site，并校验 timestep；
  2. ReachController：从末端位置误差经阻尼最小二乘得到关节位置目标；
  3. move_to()：用 5 ms 物理步 + 20 ms 外层 IK 两个时钟推进，留下判定证据。

运行（在任意 cwd 下均可，路径相对本文件解析）：

    .venv/bin/python reach.py

成功时终端打印 `reach success=True, final_error=... m`，并在 results/ 下生成 reach.csv。
本文件同时是后续实验（单变量、Sim2Real、pick-place）复用的核心模块，
可通过 `from reach import load_model, ReachController, move_to` 导入。
"""

from pathlib import Path
import csv

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# 常量与路径：资产、关节名与两个时钟，全部显式声明（讲义片段一）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR / "mujoco_menagerie" / "robotstudio_so101"
RESULT_DIR = SCRIPT_DIR / "results"

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER = "gripper"
EE_SITE = "gripperframe"
PHYSICS_DT = 0.005
CONTROL_DT = 0.020
DECIMATION = round(CONTROL_DT / PHYSICS_DT)


def load_model(target: np.ndarray) -> mujoco.MjModel:
    """加载官方 scene.xml，运行时加入目标 site，不修改官方 XML。"""
    scene = MODEL_ROOT / "scene.xml"
    if not scene.exists():
        raise FileNotFoundError(scene)

    spec = mujoco.MjSpec.from_file(str(scene))
    spec.worldbody.add_site(
        name="book_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=target,
        size=[0.012, 0.012, 0.012],
        rgba=[0.10, 0.75, 0.95, 0.9],
    )
    model = spec.compile()
    if not np.isclose(model.opt.timestep, PHYSICS_DT):
        raise ValueError(
            f"Expected timestep {PHYSICS_DT}, "
            f"got {model.opt.timestep}"
        )
    return model


class ReachController:
    """一次外层 IK 更新：不推进物理，也不负责宣布任务成功。

    按名称查找关节/执行器/site 接口，避免假定“前五个关节就是手臂”。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ):
        self.model = model
        self.data = data
        self.site_id = model.site(EE_SITE).id
        self.joint_ids = np.array(
            [model.joint(name).id for name in ARM_JOINTS]
        )
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.act_ids = np.array(
            [model.actuator(name).id for name in ARM_JOINTS]
        )
        self.gripper_id = model.actuator(GRIPPER).id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        # 实验指标观测（command 中更新，供 §3.7 单变量实验读取）
        self.last_min_singular = float("inf")
        self.last_clipped = False

    def command(
        self,
        target: np.ndarray,
        damping: float = 0.02,
    ) -> float:
        """target → error → Jacobian → 阻尼最小二乘 IK → 关节位置目标。"""
        error = target - self.data.site_xpos[self.site_id]
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self.jacp,
            self.jacr,
            self.site_id,
        )
        J = self.jacp[:, self.dof_ids]
        # 观测：最小奇异值（衡量接近奇异的程度）
        self.last_min_singular = float(np.linalg.svd(J)[1][-1])
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            0.5 * error,
        )

        norm = np.linalg.norm(dq)
        if norm > 0.04:
            dq *= 0.04 / norm

        q_cmd = self.data.qpos[self.qpos_ids] + dq
        low, high = self.model.actuator_ctrlrange[
            self.act_ids
        ].T
        clipped = np.clip(q_cmd, low, high)
        # 观测：是否触发关节限幅
        self.last_clipped = bool(np.any(~np.isclose(clipped, q_cmd)))
        self.data.ctrl[self.act_ids] = clipped
        return float(np.linalg.norm(error))


def move_to(
    controller: ReachController,
    target: np.ndarray,
    tolerance: float = 0.005,
    timeout_s: float = 5.0,
) -> tuple[bool, list[list[float]]]:
    """外层控制每 20 ms 更新一次，物理按 5 ms 推进。

    成功要求误差连续 8 个控制周期低于阈值，而不是某一帧偶然进入目标球。
    """
    model, data = controller.model, controller.data
    consecutive = 0
    error = float("inf")
    records: list[list[float]] = []

    for step in range(int(timeout_s / PHYSICS_DT)):
        if step % DECIMATION == 0:
            error = controller.command(target)
            consecutive = (
                consecutive + 1 if error < tolerance else 0
            )

        mujoco.mj_step(model, data)
        ee = data.site(EE_SITE).xpos.copy()
        records.append(
            [float(data.time), *ee.tolist(), error]
        )

        if consecutive >= 8:
            return True, records

    return False, records


def main() -> None:
    target = np.array([0.30, 0.10, 0.20])
    model = load_model(target)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = ReachController(model, data)

    success, records = move_to(controller, target)

    RESULT_DIR.mkdir(exist_ok=True)
    csv_path = RESULT_DIR / "reach.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "ee_x", "ee_y", "ee_z", "error_m"])
        writer.writerows(records)

    final_error = np.linalg.norm(
        target - data.site(EE_SITE).xpos
    )
    print(
        f"reach success={success}, "
        f"final_error={final_error:.4f} m"
    )
    print(f"records written to {csv_path}")


if __name__ == "__main__":
    main()
