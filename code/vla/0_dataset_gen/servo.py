"""位置-only 微分 IK 伺服器：把**被夹着的物体**送到指定的世界坐标点。

为什么要有这个东西——产线此前每一处「手臂没走到该去的地方」都是用**筛**解决的：
松手高度在已录轨迹里找一帧（找不到弃集）、料箱事后摆到物体碰巧经过的地方、
物体位置二十五点网格搜索。良率于是乘法坍缩（抓起 71% × 有松手窗口 54% × 过硬判据 32% ≈ 11%），
候选池 3300 条全跑完也只有约 253 条，够不到 500。**筛子的良率不可能靠加筛子提高，得改成控。**

与两条已证伪的老路的区别：
- 不是「给定固定末端姿态的解析 IK」（那条 6/6 无解：真机姿态随位置连续变化，
  死钉一个四元数把可解域掐死了）。这里**只约束位置，姿态自由漂移**，
  且从「已经夹着物体、物理上确实到达过」的当前构型出发做增量。
- 不是 `hybrid/make_dataset.py` 那套**手工标定的方向不对称增益**
  （pan +y 用 11.0 / −y 用 30.0，注释自己记着「单一线性增益必然一边推不动、一边冲飞」）。
  那些常数标定于资产修复之前（腕部限位死区 + 26% 假自碰撞 + 物体 80g + 34mm 碰撞盒），
  所谓「负向效率 2.7×/5.3×」多半是坏资产的产物而非运动学性质。雅可比本来就是构型相关的，
  各向异性由它自己处理，不需要人去标。

输出的是**绝对关节角**，交给 `pd_joint_pos` 控制器真跑出来——不 `set_qpos` 灌状态，
所以 action 与 state 同源，用 action 直接 replay 仍然成立。
"""

import numpy as np
import sapien
from transforms3d.quaternions import quat2mat

EE_LINK = "gripper_frame_link"
# 夹持中心要用两指尖与掌心算（沿夹爪中轴线回退到方块质心高度）。
# 这三个 link 的 FK 让「摆位搜索」不再需要先跑一遍空跑 rollout ——
# 而"录制必须是本进程第一次 rollout"正是动作可重放的前提。
TIP_LINKS = ("finger1_tip", "finger2_tip", "gripper_link")
N_ARM = 5                    # 前五个是手臂关节，第六个是夹爪，伺服只动手臂


class ArmKinematics:
    """挂在 GPU 环境旁边的一份 CPU-only URDF 副本，只做运动学，不参与仿真。

    `create_pinocchio_model()` 在 GPU 后端下直接抛 NotImplementedError，
    而物理与渲染又必须在 GPU 上；实测两者可在同一进程内共存。
    """

    def __init__(self, urdf_path):
        self._scene = sapien.Scene()
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._art = loader.load(str(urdf_path))
        self._pm = self._art.create_pinocchio_model()
        names = [l.name for l in self._art.get_links()]
        self._ee = names.index(EE_LINK)
        self._tips = {k: names.index(k) for k in TIP_LINKS if k in names}
        missing = [k for k in TIP_LINKS if k not in names]
        if missing:
            raise RuntimeError(f"URDF 里找不到这些 link：{missing}（夹持中心算不出来）")
        self.n = len(self._art.get_active_joints())

    def fk_local(self, q):
        """末端在**机器人基座系**下的位置。"""
        qq = np.zeros(self.n)
        qq[: min(len(q), self.n)] = np.asarray(q, float)[: self.n]
        self._pm.compute_forward_kinematics(qq)
        return np.asarray(self._pm.get_link_pose(self._ee).p, float)

    def ee_pose_local(self, q):
        """末端 link 在**基座系**下的 (位置, 旋转矩阵)。

        比 `fk_local` 多给姿态，用来把"刚体附着在腕上的点"（例如实测标定出来的夹持口袋）
        变换到基座系。
        """
        qq = np.zeros(self.n)
        qq[: min(len(q), self.n)] = np.asarray(q, float)[: self.n]
        self._pm.compute_forward_kinematics(qq)
        pose = self._pm.get_link_pose(self._ee)
        return np.asarray(pose.p, float), quat2mat(np.asarray(pose.q, float))

    def tips_local(self, q):
        """两指尖与掌心在**基座系**下的位置（一次 FK 全取）。"""
        qq = np.zeros(self.n)
        qq[: min(len(q), self.n)] = np.asarray(q, float)[: self.n]
        self._pm.compute_forward_kinematics(qq)
        return {k: np.asarray(self._pm.get_link_pose(i).p, float)
                for k, i in self._tips.items()}


class BaseFrame:
    """机器人根连杆在世界系下的位姿，用来把世界系与基座系互转。

    机器人不一定摆在世界原点；不转换的话雅可比方向会整体转一个角度，
    表现就是「越伺服越远」，而且极像控制律写错。
    """

    def __init__(self, pose_p, pose_q):
        self.p = np.asarray(pose_p, float)
        self.R = quat2mat(np.asarray(pose_q, float))

    def to_world(self, p_local):
        return self.R @ np.asarray(p_local, float) + self.p

    def vec_to_local(self, v_world):
        return self.R.T @ np.asarray(v_world, float)


def dls_delta(J, e_local, lam=0.05, max_dq=None):
    """阻尼最小二乘：dq = Jᵀ(JJᵀ + λ²I)⁻¹ e。

    阻尼项让接近奇异构型时不会解出巨大的关节增量（纯伪逆在那里会炸）。

    ★**不要**在这个结果上按关节做缩放来"少动腕部"：伪逆之后逐关节乘系数会**改变解的
    笛卡尔方向**，位置任务随之失效 —— 实测残差直接炸到 xy 39.6cm / z −25.9cm，
    手臂把方块甩飞。要偏好某些关节只能用**加权伪逆**（把权重放进度量里），
    或者干脆减少参与的关节数（见 `SERVO_JOINTS`：只用前三个关节解 3 维位置，
    3 自由度对 3 维目标是精确解，没有零空间可漂，腕部保持抓取时的角度）。
    """
    J = np.asarray(J, float)
    JT = J.T
    A = J @ JT + (lam ** 2) * np.eye(3)
    dq = JT @ np.linalg.solve(A, np.asarray(e_local, float))
    if max_dq is not None:
        m = np.abs(dq).max()
        if m > max_dq:
            dq = dq * (max_dq / m)
    return dq


def self_check(kin: ArmKinematics, base: BaseFrame, q_sim, p_ee_sim, tol=2e-3):
    """CPU 副本与仿真里的真机器人必须给出同一个末端位置，否则一切伺服都是错的。

    这是**可证伪**的自检：同一组关节角、同一个 link，两边算出来差多少毫米。
    历史上「关节顺序对不上」「基座系没转」都能在这里当场暴露，而不是等到伺服发散
    再去猜控制律。差超过 `tol` 直接抛错，不带着错前提往下跑。
    """
    p_cpu = base.to_world(kin.fk_local(q_sim))
    err = float(np.linalg.norm(p_cpu - np.asarray(p_ee_sim, float)))
    if err > tol:
        raise RuntimeError(
            f"运动学自检失败：CPU 副本算出 {p_cpu} / 仿真实际 {p_ee_sim}，差 {err*1000:.2f}mm。"
            f"关节顺序或基座变换不对，伺服不可用。")
    return err
