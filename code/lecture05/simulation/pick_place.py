"""SO-101 pick-place（第 5 讲 §3.5）—— 可复用的阶段调度 + 单次运行入口。

从 scene_box.xml 的 pickup 关键帧开始（抓取就绪：夹爪张开、两指支在方块两侧、
方块放在地面），完成阶段链：

    闭合 → 确认抓取 → 抬升 → 搬运 → 下放 → 释放 → 稳定性检查

抓取确认使用四类证据（§3.5）：
  1. 方块分别与固定指（gripper body）、活动指（moving_jaw body）接触；
  2. mj_contactForce 读到两侧法向接触力；
  3. 抬升后方块高度确实增加；
  4. 方块相对夹爪（gripperframe）无明显漂移。

放置检查（§3.5）：方块离开夹爪、平面落点误差、高度接近支撑面、线速度足够小。

核心逻辑抽成 run_pick_place(model, data, ...)，供 experiments/mass.py 与
experiments/friction.py 切换场景做单变量实验复用。

运行：.venv/bin/python pick_place.py
输出：results/pick_place_trace.csv 与终端阶段汇总。
"""

from pathlib import Path
import csv

import mujoco
import numpy as np

from reach import (
    MODEL_ROOT,
    RESULT_DIR,
    GRIPPER,
    EE_SITE,
    PHYSICS_DT,
    CONTROL_DT,
    DECIMATION,
    ReachController,
)

# ---------------------------------------------------------------------------
# 实验参数（默认值；单变量实验按需覆盖）
# ---------------------------------------------------------------------------
GRASP_FORCE_TH = 3.0          # 双侧夹持力阈值 [N]（确认“已夹住”）
LIFT_HEIGHT = 0.12            # 抬升高度 [m]
PLACE_XY = np.array([0.32, 0.05])   # 放置目标（方块中心在桌面上的 x,y）
PLACE_Z = 0.03                # 方块中心放置高度（底部接触 z=0 桌面）
PLACE_XY_TOL = 0.02           # 落点平面误差阈值 [m]
STABLE_VEL = 0.05             # 稳定判定线速度阈值 [m/s]

# 搬运/抬升/下放用前 3 关节做位置 IK，固定手腕姿态以保持夹爪朝向（防滑脱）
ARM3 = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
WRIST = ["wrist_flex", "wrist_roll"]


class PickPlaceController:
    """手臂 IK + 夹爪开合 + 接触/漂移证据读取。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.arm = ReachController(model, data)
        self.gripper_act = model.actuator(GRIPPER).id
        self.gripper_qposadr = model.joint(GRIPPER).qposadr[0]

        self.ee_site_id = model.site(EE_SITE).id
        self.box_id = model.body("box").id
        self.box_dofadr = int(model.body("box").dofadr[0])
        self.gripper_body_id = model.body("gripper").id
        self.moving_body_id = model.body("moving_jaw_so101_v1").id

        # 3 关节 IK（固定手腕）的接口
        self.arm3_joint_ids = np.array([model.joint(n).id for n in ARM3])
        self.arm3_dof = model.jnt_dofadr[self.arm3_joint_ids]
        self.arm3_act = np.array([model.actuator(n).id for n in ARM3])
        self.wrist_act = np.array([model.actuator(n).id for n in WRIST])
        self.wrist_ctrl = data.ctrl[self.wrist_act].copy()

        # 可选：被动 viewer（--viewer 时注入，实时渲染）
        self.viewer = None

        # 抓取确认后记录的几何关系（用于漂移证据）
        self.grasp_offset = None          # box - gripperframe 偏移
        self.grasp_box_z = None           # 抓取时方块高度
        self.grasp_gripper_target = None  # 保持闭合的夹爪目标

    # ---- 动作 ---------------------------------------------------------
    def arm_to(self, target: np.ndarray) -> float:
        """手臂 IK：把 gripperframe 移向 target，返回末端位置误差。"""
        return self.arm.command(target)

    def arm_ik3(self, target: np.ndarray, damping: float = 0.02) -> float:
        """前 3 关节位置 IK，固定手腕姿态（保持夹爪朝向，防搬运时滑脱）。"""
        err = target - self.data.site_xpos[self.ee_site_id]
        mujoco.mj_jacSite(
            self.model, self.data, self.arm.jacp, self.arm.jacr, self.ee_site_id
        )
        J = self.arm.jacp[:, self.arm3_dof]
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3), 0.5 * err
        )
        norm = np.linalg.norm(dq)
        if norm > 0.04:
            dq *= 0.04 / norm
        q = self.data.qpos[self.model.jnt_qposadr[self.arm3_joint_ids]] + dq
        lo, hi = self.model.actuator_ctrlrange[self.arm3_act].T
        self.data.ctrl[self.arm3_act] = np.clip(q, lo, hi)
        self.data.ctrl[self.wrist_act] = self.wrist_ctrl
        return float(np.linalg.norm(err))

    def set_gripper(self, target_q: float) -> None:
        low, high = self.model.actuator_ctrlrange[self.gripper_act]
        self.data.ctrl[self.gripper_act] = float(np.clip(target_q, low, high))

    def keep_grasp(self) -> None:
        self.set_gripper(self.grasp_gripper_target)

    # ---- 证据读取 -----------------------------------------------------
    def gripper_contacts(self) -> tuple[float, int, float, int]:
        """返回 (固定指法向力和, 固定指接触数, 活动指法向力和, 活动指接触数)。"""
        ff = mf = 0.0
        fn = mn = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if self.box_id not in pair:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            fnorm = abs(f[0])
            if self.gripper_body_id in pair:
                ff += fnorm
                fn += 1
            elif self.moving_body_id in pair:
                mf += fnorm
                mn += 1
        return ff, fn, mf, mn

    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.box_id].copy()

    def box_vel_norm(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self.box_dofadr:self.box_dofadr + 3]))

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def step(self) -> None:
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

    def snapshot(self, phase: str, box_z0: float | None) -> list:
        ee = self.ee_pos()
        box = self.box_pos()
        ff, fn, mf, mn = self.gripper_contacts()
        drift = float("nan")
        if self.grasp_offset is not None:
            drift = float(np.linalg.norm((box - ee) - self.grasp_offset))
        box_lift = float("nan")
        if box_z0 is not None:
            box_lift = float(box[2] - box_z0)
        return [
            round(float(self.data.time), 3), phase,
            *[round(v, 5) for v in ee], *[round(v, 5) for v in box],
            round(float(self.data.qpos[self.gripper_qposadr]), 5),
            round(ff, 4), round(mf, 4), fn, mn,
            round(self.box_vel_norm(), 5), round(drift, 5), round(box_lift, 5),
        ]


HEADER = ["time_s", "phase", "ee_x", "ee_y", "ee_z",
          "box_x", "box_y", "box_z", "gripper_q",
          "fixed_force_N", "moving_force_N", "fixed_n", "moving_n",
          "box_vel", "drift_m", "box_lift_m"]


def load_box_model(scene_name: str = "scene_box.xml") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载 scene_box 系列场景并复位到 pickup 关键帧。"""
    scene = MODEL_ROOT / scene_name
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = model.key("pickup").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return model, data


def run_pick_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    place_xy: np.ndarray = PLACE_XY,
    grasp_force_th: float = GRASP_FORCE_TH,
    lift_height: float = LIFT_HEIGHT,
    viewer=None,
    frame_callback=None,
) -> dict:
    """跑一次完整 pick-place，返回阶段结果、四类抓取证据与放置检查指标。

    model/data 应已复位到 pickup 关键帧（见 load_box_model）。
    """
    ctrl = PickPlaceController(model, data)
    ctrl.viewer = viewer
    arm_ctrl = data.ctrl[ctrl.arm.act_ids].copy()
    gripper_open = float(data.qpos[ctrl.gripper_qposadr])
    initial_ee = ctrl.ee_pos()
    initial_box = ctrl.box_pos()
    trace: list[list] = []

    def push(phase):
        trace.append(ctrl.snapshot(phase, initial_box[2]))
        if frame_callback is not None:
            frame_callback()

    result = {"grasp": False, "lift": False, "carry": False,
              "lower": False, "release": False, "stable": False}
    grasp_ff = grasp_fn = grasp_mf = grasp_mn = 0
    lift_delta = 0.0
    carry_drift = 0.0

    # ---- 阶段 1：闭合 + 确认抓取 -------------------------------------
    close_target = gripper_open
    grasp_ok = False
    for _ in range(int(3.0 / CONTROL_DT)):
        data.ctrl[ctrl.arm.act_ids] = arm_ctrl
        close_target -= 0.01
        ctrl.set_gripper(close_target)
        ctrl.step()
        push("grasp")
        ff, fn, mf, mn = ctrl.gripper_contacts()
        if ff > grasp_force_th and mf > grasp_force_th:
            grasp_ok = True
            break
        if close_target <= -0.174:
            break

    if grasp_ok:
        # 持续夹紧目标设为 0.0：方块会阻挡 gripper 关节，位置执行器持续施加
        # 闭合力；抬升/搬运中方块下滑时活动指自动跟进，避免夹持力衰减滑脱。
        ctrl.grasp_gripper_target = 0.0
        ctrl.grasp_box_z = ctrl.box_pos()[2]
        ctrl.grasp_offset = ctrl.box_pos() - ctrl.ee_pos()
        grasp_ff, grasp_fn, grasp_mf, grasp_mn = ctrl.gripper_contacts()
        result["grasp"] = True

    # ---- 阶段 2：抬升 -------------------------------------------------
    lift_ok = False
    if grasp_ok:
        lift_target = initial_ee + np.array([0.0, 0.0, lift_height])
        for _ in range(int(4.0 / CONTROL_DT)):
            ctrl.arm_ik3(lift_target)
            ctrl.keep_grasp()
            ctrl.step()
            push("lift")
            if (np.linalg.norm(lift_target - ctrl.ee_pos()) < 0.005
                    and ctrl.box_pos()[2] - ctrl.grasp_box_z > 0.05):
                lift_ok = True
                break
        if lift_ok:
            lift_delta = ctrl.box_pos()[2] - ctrl.grasp_box_z
        result["lift"] = lift_ok

    # ---- 阶段 3：搬运 -------------------------------------------------
    carry_ok = False
    if lift_ok:
        carry_ee = np.array([place_xy[0], place_xy[1], ctrl.ee_pos()[2]]) \
            - ctrl.grasp_offset
        carry_ee[2] = ctrl.ee_pos()[2]  # 高度保持抬升高度
        for _ in range(int(4.0 / CONTROL_DT)):
            ctrl.arm_ik3(carry_ee)
            ctrl.keep_grasp()
            ctrl.step()
            push("carry")
            if np.linalg.norm(carry_ee - ctrl.ee_pos()) < 0.005:
                carry_ok = True
                break
        if carry_ok:
            carry_drift = float(np.linalg.norm(
                (ctrl.box_pos() - ctrl.ee_pos()) - ctrl.grasp_offset))
        result["carry"] = carry_ok

    # ---- 阶段 4：下放 -------------------------------------------------
    lower_ok = False
    if carry_ok:
        lower_ee = np.array([place_xy[0], place_xy[1], PLACE_Z]) \
            - ctrl.grasp_offset
        for _ in range(int(4.0 / CONTROL_DT)):
            ctrl.arm_ik3(lower_ee)
            ctrl.keep_grasp()
            ctrl.step()
            push("lower")
            if (np.linalg.norm(lower_ee - ctrl.ee_pos()) < 0.005
                    and abs(ctrl.box_pos()[2] - PLACE_Z) < 0.01):
                lower_ok = True
                break
        result["lower"] = lower_ok

    # ---- 阶段 5：释放 -------------------------------------------------
    release_ok = False
    release_box = None
    if lower_ok:
        for _ in range(int(2.0 / CONTROL_DT)):
            ctrl.set_gripper(gripper_open)  # 手臂保持在下放位置，仅张开夹爪
            ctrl.step()
            push("release")
            ff, fn, mf, mn = ctrl.gripper_contacts()
            if fn == 0 and mn == 0:
                release_ok = True
                release_box = ctrl.box_pos()
                break
        result["release"] = release_ok

    # ---- 阶段 6：稳定性检查 ------------------------------------------
    stable_ok = False
    if release_ok:
        for _ in range(int(2.0 / CONTROL_DT)):
            ctrl.set_gripper(gripper_open)  # 手臂保持不动，等待方块稳定
            ctrl.step()
            push("stable")
            if (ctrl.box_vel_norm() < STABLE_VEL
                    and ctrl.box_pos()[2] < 0.035):
                stable_ok = True
                break
        result["stable"] = stable_ok

    # ---- 放置检查（§3.5）--------------------------------------------
    box_final = ctrl.box_pos()
    place_err = float(np.linalg.norm(box_final[:2] - place_xy))
    placed = (release_ok and place_err < PLACE_XY_TOL
              and abs(box_final[2] - PLACE_Z) < 0.015
              and ctrl.box_vel_norm() < STABLE_VEL)

    return {
        "result": result,
        "grasp_ff": grasp_ff, "grasp_fn": grasp_fn,
        "grasp_mf": grasp_mf, "grasp_mn": grasp_mn,
        "lift_delta": lift_delta,
        "carry_drift": carry_drift,
        "place_err": place_err,
        "box_final_z": float(box_final[2]),
        "box_final_xy": box_final[:2].copy(),
        "box_vel": ctrl.box_vel_norm(),
        "placed": placed,
        "release_box_xy": None if release_box is None else release_box[:2].copy(),
        "trace": trace,
    }


def print_summary(r: dict) -> None:
    print("=== pick-place 阶段结果 ===")
    for k, v in r["result"].items():
        print(f"  {k:<8}: {'OK' if v else 'FAIL'}")
    print("=== 四类抓取证据（抓取确认时记录）===")
    print(f"  双侧接触:  固定指 {r['grasp_fn']} 点 / 活动指 {r['grasp_mn']} 点")
    print(f"  双侧法向力: 固定指 {r['grasp_ff']:.2f} N / 活动指 {r['grasp_mf']:.2f} N")
    print(f"  抬升后方块高度增量: {r['lift_delta']:.3f} m")
    print(f"  搬运后方块相对夹爪漂移: {r['carry_drift']:.4f} m")
    print("=== 放置检查 ===")
    print(f"  落点误差: {r['place_err']:.4f} m (阈值 {PLACE_XY_TOL})")
    print(f"  方块最终高度: {r['box_final_z']:.4f} m (目标 {PLACE_Z})")
    print(f"  方块线速度: {r['box_vel']:.4f} m/s (阈值 {STABLE_VEL})")
    print(f"  综合判定: {'PLACED' if r['placed'] else 'NOT PLACED'}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SO-101 pick-place")
    parser.add_argument("--viewer", action="store_true",
                        help="启动被动 viewer 实时渲染（需 X server/远程桌面）")
    args = parser.parse_args()

    model, data = load_box_model()
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
    try:
        r = run_pick_place(model, data, viewer=viewer)
    finally:
        if viewer is not None:
            viewer.close()
    print_summary(r)

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / "pick_place_trace.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(r["trace"])
    print(f"\ntrace -> {out}")


if __name__ == "__main__":
    main()
