"""SO101Kit —— 我们真机的 SO101（KIT 版）机器人 agent。

与 vendored squint 的裸臂 SO101 相比，这里加载的是**真机几何** `kit_v1_so101.urdf`：
同样的 6 个活动关节（shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper），
但 URDF 里多了 kit 底板、支架（frame_extrusion），以及两个相机的安装座与光学系
（top_camera_optical_frame / wrist_camera_optical_frame，都是 fixed 关节、不增加自由度）。

因为活动关节名与 vendored SO101 完全一致，控制器配置、keyframes、抓取/接触判定全部原样
复用——直接继承 :class:`SO101`，只把 uid 与 urdf 换成 KIT 版。相机 sensor 不在这里定义，
而在环境（envs.py）里用 CameraConfig 绑到上面两个光学系上。
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from so101_sim.robots.so101_base.so101 import SO101  # 裸臂 SO101（复用其控制器/keyframe）


# ── 关掉 4 对**假**自碰撞 ──────────────────────────────────────────────────
#
# URDF 的碰撞几何是 STL 的凸包，比真件胖。后果实测：命令机器人保持真机 home 位姿时，
# `shoulder_link↔lower_arm_link` 接触力 130N 起步、几步内飙到 **18000N**，手臂被弹飞
# （shoulder_pan −5.8°→+115°），shoulder_lift 被顶到关节限位 —— 于是仿真**回不到真机 home**。
#
# 哪几对是"假"的，用可证伪的判据定：真机 106085 帧里出现过的关节角，物理上一定不自碰
# （真臂就那么摆着）。把这些关节角喂进仿真碰撞几何，重叠的就是假碰撞。
# 实测（从真机轨迹等距抽 400 帧逐对检测重叠）：
#
#     lower_arm_link  ~ shoulder_link            105/400 帧(26%)  最大穿透 10.25mm
#     gripper_link    ~ shoulder_link             14/400 帧       最大穿透  2.95mm
#     wrist_camera_mount_link ~ wrist_link        12/400 帧       最大穿透  3.42mm
#     shoulder_link   ~ wrist_link                 5/400 帧       最大穿透  1.12mm
#
# **四分之一的真机位姿在仿真里自碰** —— 此前"重放真机轨迹总不对劲"的物理来源。
#
# 只关这 4 对，不用 ManiSkill 的 `disable_self_collisions`（那是一刀切关掉整条 articulation
# 内部所有自碰，手臂会能穿过 KIT 支架与相机柱，画面上要露馅）。
# 机制：SAPIEN 的 `collision_groups[2]` 是"忽略掩码"，两个 shape 有公共 bit 就互不碰撞。
# 每对分配一个独立 bit，互不串扰（已核对：任意非指定对的掩码交集为 0）。
FALSE_SELF_COLLISION_PAIRS = [
    ("lower_arm_link", "shoulder_link"),
    ("gripper_link", "shoulder_link"),
    ("wrist_camera_mount_link", "wrist_link"),
    ("shoulder_link", "wrist_link"),
]
# 从 20 起，避开 ManiSkill 自己用的 bit 29（`disable_self_collisions`）
_IGNORE_BIT_BASE = 20


class KitFalseSelfCollisionMixin:
    """把 KIT 碰撞几何造成的假自碰撞逐对关掉（真碰撞与外部碰撞照旧生效）。"""

    def _after_init(self):
        super()._after_init()
        links = self.robot.links_map
        for k, (a, b) in enumerate(FALSE_SELF_COLLISION_PAIRS):
            bit = _IGNORE_BIT_BASE + k
            for name in (a, b):
                if name in links:
                    links[name].set_collision_group_bit(group=2, bit_idx=bit, bit=1)


@register_agent()
class SO101Kit(KitFalseSelfCollisionMixin, SO101):
    uid = "so101_kit"

    # 真机 KIT 几何（含支架与两相机光学系）；随包自包含，见同目录 kit_assets/。
    urdf_path = str(Path(__file__).parent / "kit_assets" / "kit_v1_so101.urdf")
