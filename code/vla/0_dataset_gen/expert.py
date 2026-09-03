"""脚本化专家：给定一个场景状态，离线规划出一串真机口径的绝对关节角。

## 为什么是离线规划 + 播放

`Teleoperator.get_action()` 不接受观测（lerobot 基类签名如此），看着物体现算的专家放不进
那个接口。而物体位置在这条路上**本来就是已知的**（环境撒的点，写在强制初始状态里），
所以抓取与放置都是**确定的几何问题**，用运动学算，不用看图、也不用学。
规划落一份 `.npy`，由 `so101_dataset_player --teleop.actions_path=` 播出去。

## 七个阶段，模拟真机遥操的行为

    ① 张开到进场量，从 home 走到物体正上方
    ② 保持张开量下降到抓取位（下降量比进场量小一点，与真机一致）
    ③ **捏到底**：夹爪指令一路给 `recipe.CLOSE_PCT`，靠物体把两指卡死
    ④ **原地垂直抬起**回到物体正上方那个位姿（仍然捏着）
    ⑤ 高位平移到料箱开口正上方
    ⑥ 垂直下降到投放高度
    ⑦ 张开松手
    ⑧ 回到 home

★ 第 ④ 步不能省。省掉它、从抓取位姿直接插值到料箱上方，关节空间的直线在笛卡尔空间里是
  一条**弦** —— 手臂贴着台面斜切过去，路上把料箱撞歪、方块卡在箱沿上。实账：第一批 10 集
  就是这么产的，三集目审全部"没放进去"。真机遥操是先原地抬起、再平移，这一步就是它。

★ 第 ③ 步是这份专家与旧产线唯一的本质区别，也是整轮返工的起点：旧版下发一个"合拢阶梯"
  （停在某个开度），于是仿真数据里夹爪 action 的含义是"停在多宽"，而真机数据里它的含义是
  "我要多用力合"。策略输出的是 action —— 同一个值在两份数据里含义不同就是两套映射。
  真机遥操是人把扳机捏到底、抓紧靠物体卡死（真机 action 中位 1.98% / state 中位 14.35%，
  那 12.4 点落差就是堵转），所以这里也必须下发到底。

★ 阶段之间一律**关节空间线性插值 + 每步限幅**，不做笛卡尔直线：这条产线要的是
  "能被 `pd_joint_pos` 真跑出来的连续小步"，而不是最短路径。

★ **限幅分段给，且取自真机实测**。用一个恒定限幅会把整集顶着上限走满 —— 实测第一批
  逐帧最大关节变化中位 1.976°、p99 2.000°（几乎每一帧都贴着 2.0° 的限幅），而真机是
  中位 0.527°、p90 3.516°、p99 5.451°、最大 9.32°：真机**大部分帧走得很慢、少数帧很快**，
  这就是"变速"。恒速的后果是整集时长只有 6.67 秒而真机中位 11.85 秒（且落在真机 p10 之下），
  目审直接看得出"快了、质量不如以前"。
  所以：进场与回家段给快的限幅（接近真机 p90），下降/抬起/投放段给慢的（接近真机中位）。

## 单位

输出恒为真机口径：臂五关节是度，夹爪是 0~100 行程百分比。换算标度从 URDF 现读
（`so101.gripper_limit_rad`），不在这里抄一份数。
"""

import numpy as np
import recipe
from grasp_ik import jaw_yaw_error, solve_aligned_grasp
from servo import ArmKinematics, BaseFrame

# 逐段的每步限幅（度）。取自真机 `pick_up_a_cube` 300 集的逐帧最大关节变化：
# 中位 0.527° / p90 3.516° / p99 5.451°。真机是变速的 —— 大部分帧慢、少数帧快。
#
# 空载移动（进场、回家）给 p90 那一档：那是真机赶路时的速度。
FAST_STEP_DEG = 3.5
# 带载或贴近物体的段（下降、抬起、平移、投放）给中位那一档：真机在这些段明显慢下来，
# 而且慢下来还有物理上的必要 —— 60°/s 恒速搬运时惯性会把刚体方块从两指间甩出去，
# 实测第一批 10 集里有 3 集"抬起来过但没落进箱口"，末位置偏 75~105mm，就是甩掉了。
SLOW_STEP_DEG = 0.6

# 两指连线与方块侧面的最大容许偏角（度）。
#
# ★ `solve_aligned_grasp` 的 `ok` **不代表对齐收敛**：它迭代 `iters` 次调腕滚，
#   跑完就 `return ..., True`，哪怕偏角还很大。而偏角大意味着两指斜着面对方块 ——
#   40mm 方块的对角是 56.6mm，而 36.4% 开度只有 52mm 间距，指头必然撞在角上把方块推走。
#   实账：不查这一条时 10 集里 5 集失败，且失败集与场景一一绑定、参数怎么调都是那几集
#   （口袋改 3~4mm、投放高度改 5cm、速度减半，失败集的末位置逐位不变）。
#   所以这里自己再量一次偏角，超了就**弃掉这个场景**，而不是拿一个抓不住的位姿去录。
MAX_JAW_YAW_DEG = 1.5

# 抓取前悬停高度（m，物体中心之上）。真机进场是从物体正上方压下来的。
APPROACH_H = 0.06
# 料箱内底的高度（m）。箱底贴着台面放，这是环境自己的常量。
BIN_FLOOR_Z = 0.0025
# 松手时物体底面离箱底留多少（m）。**不能从搬运高度直接松手** —— 那样物体从箱沿上方
# 六七厘米自由落下，40mm 刚体会在 8×10cm 的箱口里弹出去：实测投放目标取 9cm 时
# 10 集里 6 集"抬起来过但没落进箱口"，末位置偏 28~114mm，正是弹飞的量级。
# 真机是放低到接近箱底再松手，这个值就是那个"接近"。
DROP_CLEARANCE = 0.015
# 「捏到底」保持几帧。堵转要几帧才建立起来（实测 40 步内就稳），给 12 帧够抓实又不拖长集长。
CLOSE_FRAMES = 12
# 松手后停几帧再回家，让物体落稳。
RELEASE_FRAMES = 6


def _interp(start, end, step_deg):
    """两个关节位形之间的线性插值，每步不超过 `max_step`（弧度）。

    Args:
        start: `(6,)` 起始关节角（弧度）。
        end: `(6,)` 终止关节角（弧度）。
        step_deg: 每步各关节的最大增量（度）。

    Returns:
        `(步数, 6)`，不含起点、含终点；两者已经很近时给一个空数组。

    步数按**最大那一维**定：各维同步走完，轨迹在关节空间里是直的，
    不会出现某一维先到、另一维还在动的折线。
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    span = float(np.abs(end - start).max())
    steps = int(np.ceil(span / np.radians(step_deg)))
    if steps <= 0:
        return np.zeros((0, len(start)))
    ratios = np.arange(1, steps + 1)[:, None] / steps
    return start + (end - start) * ratios


def _with_grip(qpos, grip_pct, low, high):
    """把一个位形的夹爪那一维换成给定的行程百分比。

    Args:
        qpos: `(6,)` 关节角（弧度）。
        grip_pct: 夹爪行程百分比。
        low: 夹爪关节下限（弧度）。
        high: 夹爪关节上限（弧度）。

    Returns:
        `(6,)` 新的关节角（弧度）。
    """
    out = np.asarray(qpos, float).copy()
    out[5] = low + grip_pct / 100.0 * (high - low)
    return out


def plan(scene, item_xy, item_yaw, bin_xy, home_qpos, base_p, base_q, urdf_path):
    """规划一集的绝对关节角序列。

    Args:
        scene: `recipe.SCENES` 的键。
        item_xy: 物体中心的世界系 xy（m）。
        item_yaw: 物体绕 z 的自旋角（弧度）。两指要对齐到**物体的面**而不是世界轴 ——
            复位时物体带随机自旋，对错了就是指尖撞角把它推走。
        bin_xy: 料箱开口中心的世界系 xy（m）。
        home_qpos: `(6,)` 起始位形（弧度），取自环境复位后的实际 qpos。
        base_p: 机器人根连杆的世界系位置。
        base_q: 机器人根连杆的世界系四元数。
        urdf_path: 机器人 URDF，用来建 CPU 运动学副本。

    Returns:
        `(帧数, 6)` 真机口径动作（臂五关节度、夹爪行程百分比）；解不出抓取位姿、
        或两指对不齐方块侧面时给 `None`。

    ★ 解不出来就**如实返回 None**，让调用方弃掉这一集，而不是拿一个没收敛的位姿往下跑。
    """
    from so101_sim.robots.so101_base.so101 import gripper_limit_rad

    spec = recipe.SCENES[scene]
    low, high = gripper_limit_rad()
    kin = ArmKinematics(urdf_path)
    frame = BaseFrame(base_p, base_q)
    limits_lo = np.array([-1.9199, -1.9199, -1.69, -1.6581, -1.1731, low])
    limits_hi = np.array([1.9199, 1.7453, 1.69, 1.8326, 4.4120, high])

    approach_rad = low + spec["open_approach_pct"] / 100.0 * (high - low)
    q_above, q_grasp, _, _, ok = solve_aligned_grasp(
        kin, frame, np.asarray(home_qpos, float), item_xy, spec["item_half"],
        APPROACH_H, approach_rad, limits_lo, limits_hi, pocket=recipe.pocket(scene),
        item_yaw=item_yaw)
    if not ok:
        return None
    yaw_err = abs(np.degrees(jaw_yaw_error(kin, q_grasp, item_yaw)))
    if yaw_err > MAX_JAW_YAW_DEG:
        return None

    # 放置位姿：物体中心要落到「箱底 + 物体半高 + 余量」，而不是搬运高度 ——
    # `solve_aligned_grasp` 解的是"夹持口袋落在目标点"，而物体就握在口袋上，
    # 所以这个目标点就是松手瞬间物体中心该在的地方。
    # 同时它的 `approach_h` 给出高位过渡姿态，搬运走高位、只在箱口正上方才下来。
    drop_z = BIN_FLOOR_Z + spec["item_half"] + DROP_CLEARANCE
    q_carry, q_drop, _, _, ok = solve_aligned_grasp(
        kin, frame, q_grasp, bin_xy, drop_z,
        APPROACH_H, low, limits_lo, limits_hi, pocket=recipe.pocket(scene),
        item_yaw=item_yaw)
    if not ok:
        return None

    home = np.asarray(home_qpos, float)
    open_a, open_d = spec["open_approach_pct"], spec["open_descent_pct"]
    closed_grasp = _with_grip(q_grasp, recipe.CLOSE_PCT, low, high)
    closed_above = _with_grip(q_above, recipe.CLOSE_PCT, low, high)
    closed_carry = _with_grip(q_carry, recipe.CLOSE_PCT, low, high)
    closed_drop = _with_grip(q_drop, recipe.CLOSE_PCT, low, high)
    segments = [
        _interp(home, _with_grip(q_above, open_a, low, high), FAST_STEP_DEG),   # ① 进场（空载）
        _interp(_with_grip(q_above, open_a, low, high),
                _with_grip(q_grasp, open_d, low, high), SLOW_STEP_DEG),          # ② 下降
        np.repeat(closed_grasp[None], CLOSE_FRAMES, axis=0),                     # ③ 捏到底
        _interp(closed_grasp, closed_above, SLOW_STEP_DEG),                      # ④ 原地抬起
        _interp(closed_above, closed_carry, SLOW_STEP_DEG),                      # ⑤ 带载平移
        _interp(closed_carry, closed_drop, SLOW_STEP_DEG),                       # ⑥ 下降到投放高度
        np.repeat(_with_grip(q_drop, open_a, low, high)[None],
                  RELEASE_FRAMES, axis=0),                                       # ⑦ 松手
        _interp(_with_grip(q_drop, open_a, low, high),
                _with_grip(home, open_a, low, high), FAST_STEP_DEG),             # ⑧ 回家（空载）
    ]
    qpos = np.vstack([s for s in segments if len(s)])
    out = np.degrees(qpos)
    out[:, 5] = (qpos[:, 5] - low) / (high - low) * 100.0
    return out
