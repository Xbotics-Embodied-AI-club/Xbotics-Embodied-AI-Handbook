"""SO101KitSlow —— 把每步关节增量上限压到**真机实测速度包线**内的 KIT 机器人。

为什么要它：squint 默认控制器 `pd_joint_target_delta_pos` 的上限是臂 ±0.1 rad/step、
夹爪 ±0.2 rad/step。20fps 下饱和速度 = 臂 114.6°/s、夹爪 229°/s，而 reward 里**没有任何
速度代价** ⇒ 策略学成一路顶着上限跑。与用户真机 task1 实测对照（30fps、单位是度）：

    关节            仿真 p95    真机 p95   倍数
    shoulder_pan     105.1       29.0     3.6×
    shoulder_lift    112.0       63.3     1.8×
    elbow_flex       103.7       65.9     1.6×
    wrist_flex       121.5       41.5     2.9×
    wrist_roll        94.6       31.6     3.0×
    gripper          192.2       84.4     2.3×

于是真机跟不上。修法不是改渲染帧率（那只是纸面变慢、单帧跳变照旧），而是**把动作空间本身
压进真机包线再重训**——生成的轨迹才是"真机能执行的速度下的真解"。

每步上限按 `真机 p95 (deg/s) / 仿真 fps` 逐关节反算（不是一刀切同一个系数：pan 要压 4×、
elbow 只需 1.7×，一刀切会过度约束慢关节）。不改 vendored SO101，只在子类里覆盖控制器配置。
"""

import copy
import os
from pathlib import Path

import numpy as np
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent

from so101_sim.robots.so101_base.so101 import SO101  # 裸臂 SO101
from so101_sim.robots.so101_kit import KitFalseSelfCollisionMixin

# 控制频率 = 数据集帧率（一步一帧）。**对齐真机 task1 的 30fps**。
# ★注意 delta 上限是"每步"的量：帧率从 20 升到 30，若上限不变，
# 每秒步数多 1.5× ⇒ 角速度也涨 1.5×。所以下面反算时用的就是这个 SIM_FPS，
# 分母变大、上限自动缩小，速度不随帧率漂。
SIM_FPS = 30

# 在真机 p95 之上再整体放慢的系数。用户 2026-07-29 反馈"1.0–1.2× 仍偏快、要更慢"，
# 曾取 0.6 ⇒ 落到真机 p50–p95 之间。
# ★2026-08-13 曾降到 0.2 以求原生真机节奏，但每步位移只剩 1/3、episode 预算涨 2.25 倍，
# v13 实测训不出来（末帧成功恒 0）。先退回 0.6 把「轻放」行为训出来，节奏留到第二轮再压。
# 原始动机保留如下：用户要求"RL 轨迹能直接用、不要事后重参数化"。
# 0.6 时策略取放只用 **2.7s**，而 ModelScope 真机 300 集取放段（松手在进度 0.66）约 **7.8s**、
# 整集中位 11.87s。差约 3 倍 ⇒ 系数同比例下调，让轨迹**原生**就是真机节奏，
# 不必再靠弧长重参数化去凑（那会破坏落体物理、还得插值四元数）。
# 代价：每步位移只剩原来 1/3，同样动作需要约 3 倍步数，episode 上限与训练预算都要跟着提。
SPEED_SCALE = 0.6

# 真机 task1 实测 p95 角速度（deg/s，仅运动帧；见 scratch 的 measure_motion_speed.py）。
# ★注意真机数据单位是**度**（feature 名 `shoulder_pan.pos`，范围 −102…96），
#   仿真是弧度——两边混算会虚高 57.3 倍。
REAL_P95_DEG_PER_S = {
    "shoulder_pan": 29.0,
    "shoulder_lift": 63.3,
    "elbow_flex": 65.9,
    "wrist_flex": 41.5,
    "wrist_roll": 31.6,
    "gripper": 84.4,
}


def per_step_limits(joint_names):
    """按真机 p95 × SPEED_SCALE 反算每步增量上限（rad/step），顺序与 `joint_names` 一致。"""
    return [float(np.deg2rad(REAL_P95_DEG_PER_S[name] * SPEED_SCALE / SIM_FPS))
            for name in joint_names]


class SlowControllerMixin:
    """把 `pd_joint_target_delta_pos` / `pd_joint_delta_pos` 的上下限换成真机包线值。

    只动这两个 delta 控制器（训练与数据生产用的就是 `pd_joint_target_delta_pos`）；
    `pd_joint_pos` / `pd_joint_vel` 原样保留，便于对照与回放。
    """

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        names = [j.name for j in self.robot.active_joints]
        limits = per_step_limits(names)
        for key in ("pd_joint_delta_pos", "pd_joint_target_delta_pos"):
            if key not in configs:
                continue
            cfg = copy.deepcopy(configs[key])
            cfg.lower = [-v for v in limits]
            cfg.upper = list(limits)
            configs[key] = cfg
        return deepcopy_dict(configs)


# ── 夹得更稳 ────────────────────────────────────────────────────────────
#
# 实测基线（用 gripper 角→指尖间距的实测映射量出来的）：
#   Place  抓持期指尖间距 4.39cm vs 方块 4.00cm ⇒ **压入 −0.39cm，指尖比方块还宽=托着走**
#          方块姿态抖动（四元数逐帧 max|Δ|）中位 0.27
#   Lift   指尖间距 3.83cm ⇒ 压入 +0.17cm（真夹紧），抖动仅 0.022
# 同一套配方两个任务出相反结果，原因是 Place 的 reward 在 `is_item_grasped` 为真后就切到
# 搬运项——**夹多紧不影响得分**，而判据门槛又极低。
#
# 两个旋钮（都不碰 vendored）：
# 1. **物体摩擦**：squint 默认 `item_friction_range=(0.1,0.5)`，对 3D 打印件配橡胶指垫偏低
#    太多；靠摩擦不够就只能靠几何卡住，策略学不到"捏紧"。提到 0.8–1.0。
#    ⚠️ 这是**物理假设、非实测**——若已知真机件材质配对的实测值，换成它。
# 2. **抓取判据 `min_force`**：`is_grasping` 默认 0.5N，轻轻贴上就算抓住。提到 3.0N，
#    让"松夹"判不过，reward 逼策略收紧才拿得到搬运分。
# ★`min_force` 消融结论：
# `is_item_grasped` **只进 reward、不进 Place 的 success 判据**（success = 物体在箱内 ∧
# 已松手 ∧ 物体静止 ∧ 机器人静止）。所以提高 min_force **不会**收紧成功门槛，
# 只会让"抓住"这个 reward 闸门更难触发 —— 实测 3.0N 时 500k 步 success 仅 0.06
# （旧配方 864k 到 1.00）。故保持默认 0.5，"夹得更稳"靠摩擦而不是靠力门槛。
ITEM_FRICTION_RANGE = tuple(
    float(x) for x in os.environ.get("SO101_ITEM_FRICTION", "0.8,1.0").split(","))
GRASP_MIN_FORCE = float(os.environ.get("SO101_GRASP_MIN_FORCE", "0.5"))


class FirmGraspMixin:
    """把 `is_grasping` 的力门槛提高，使"轻贴"不再算抓住。"""

    def is_grasping(self, object, min_force=None, max_angle=110):
        return super().is_grasping(object,
                                   min_force=GRASP_MIN_FORCE if min_force is None else min_force,
                                   max_angle=max_angle)


@register_agent()
class SO101Slow(FirmGraspMixin, SlowControllerMixin, SO101):
    """裸臂 SO101 + 真机速度包线 + 严格抓取判据（vanilla 训练/rollout，单 base_camera）。"""

    uid = "so101_slow"


@register_agent()
class SO101KitSlow(KitFalseSelfCollisionMixin, FirmGraspMixin, SlowControllerMixin, SO101):
    """KIT 几何 + 真机速度包线 + 严格抓取判据（KIT 双相机重渲用）。

    与 `SO101Kit` 一样要关掉 KIT 碰撞几何造成的 4 对假自碰撞——它们是几何自带的，
    与速度包线无关，两个 KIT agent 都受影响。
    """

    uid = "so101_kit_slow"
    urdf_path = str(Path(__file__).parent / "kit_assets" / "kit_v1_so101.urdf")
