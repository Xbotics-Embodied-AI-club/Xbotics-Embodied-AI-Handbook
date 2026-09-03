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

# 巡航速度（度/帧）与全局加速度上限（度/帧²）。取自真机 `pick_up_a_cube` 300 集的
# 逐帧最大关节变化：中位 0.527° / p90 3.516° / p99 5.451°。
#
# ★ 早先这里是「每段一个恒定限幅、段间直接切换」：空载 3.5、带载 0.6。后果是段界上
#   速度**突变 6 倍**，而段内速度恒定 —— 整条轨迹的速度成了分段常值函数，处处不连续。
#   目审逐条读出来的就是这个：「抓的时候很抖，末端速度不连续」「回收的速度很快，也感觉
#   不连续」「轨迹的流畅性比老版差不少」。
#   真机不是这样：慢帧居多、快帧少数，而且快慢之间是**渐变**的 —— 人推摇杆有加减速。
#
# 现在给每段一个巡航上限，再用一个全局加速度上限把整条路径重新计时：速度沿路径连续，
# 只在真正需要停的地方（合拢前、松手前、回到 home）为零。
CRUISE_FREE_DEG = 2.0
# 空载赶路（进场、回家）。落在真机中位与 p90 之间，不再顶着 p90 走满 ——
# 顶着 p90 恒速走的后果就是「回收很快」。
CRUISE_LOAD_DEG = 0.8
# 带载或贴近物体（下降、抬起、平移、投放）。接近真机中位，且慢下来有物理必要：
# 恒速 60°/s 搬运时惯性会把刚体方块从两指间甩出去（实测第一批有 3 集"抬起来过但没落进
# 箱口"，末位置偏 75~105mm，就是甩掉的）。
MAX_ACCEL_DEG = 0.25
# 加速度上限 ⇒ 从静止到空载巡航要 8 帧（0.27 秒），与人推摇杆的起步量级相当。
DENSE_STEP_DEG = 0.05
# 重采样用的细撒点步长，远小于任何巡航速度 —— 它只决定路径的几何分辨率，不决定速度。

# 夹爪每帧最多合/张多少行程百分比。真机逐帧夹爪变化 p95 为 9.8~16.2%，所以 36.4% 的
# 开度合到底至少要三帧。
#
# ★ 早先是**一帧**从 36.4% 砸到 0%。那一下把刚体方块猛地夹住，画面上就是「被吸上来」。
GRIP_RATE_PCT = 12.0

# 两指连线与方块侧面的最大容许偏角（度）。
#
# ★ `solve_aligned_grasp` 的 `ok` **不代表对齐收敛**：它迭代 `iters` 次调腕滚，
#   跑完就 `return ..., True`，哪怕偏角还很大。而偏角大意味着两指斜着面对方块 ——
#   40mm 方块的对角是 56.6mm，而 36.4% 开度只有 52mm 间距，指头必然撞在角上把方块推走。
#   实账：不查这一条时 10 集里 5 集失败，且失败集与场景一一绑定、参数怎么调都是那几集
#   （口袋改 3~4mm、投放高度改 5cm、速度减半，失败集的末位置逐位不变）。
#   所以这里自己再量一次偏角，超了就**弃掉这个场景**，而不是拿一个抓不住的位姿去录。
MAX_JAW_YAW_DEG = 1.5

# 瞄点最大容许残差（毫米）：解出来的抓取位姿要真把夹持口袋送到物体中心。
#
# ★ `solve_aligned_grasp` 的 `ok` 也**不代表逆解收敛到目标点**。实账：某一集的瞄点残差
#   45.21mm（y 方向偏 40.16mm）、对齐残余 14.57°，却一路通过、还被判成「成功」——
#   那是蒙进箱口的。正常集的残差是 0.01~0.09mm，所以 1mm 的门槛离两边都很远。
MAX_AIM_MM = 1.0

# 抓取前悬停高度（m，物体中心之上）。真机进场是从物体正上方压下来的。
APPROACH_H = 0.06
# 料箱内底的高度（m）。箱底贴着台面放，这是环境自己的常量。
BIN_FLOOR_Z = 0.0025
# 松手时物体底面离箱底留多少（m）。用户 2026-09-03 明确放开：「你可以在更高一点的地方把
# 物体放下来，让他落入盒子就可以」「垂直下落 10cm 左右是可以的」⇒ 不必把爪伸进箱内。
#
# ★ 早先这里是 0.015，理由是"实测投放目标取 9cm 时 6/10 弹出箱口"。那次的前提已经不成立：
#   当时轨迹是分段常值速度（段界突变 6 倍）、夹爪一帧砸到底，物体是被甩着松手的。
#   现在速度连续、合拢渐变，10cm 自由落体按用户口径可接受 —— 但仍要按落点判据验。
DROP_CLEARANCE = 0.10
# 「捏到底」保持几帧。堵转要几帧才建立起来（实测 40 步内就稳），给 12 帧够抓实又不拖长集长。
CLOSE_FRAMES = 12
# 松手后停几帧再回家，让物体落稳。
RELEASE_FRAMES = 6


def _densify(start, end, step_deg):
    """两个关节位形之间按细步长撒点（不含起点、含终点）。

    Args:
        start: `(6,)` 起始关节角（弧度）。
        end: `(6,)` 终止关节角（弧度）。
        step_deg: 相邻点之间各关节的最大增量（度）。

    Returns:
        `(点数, 6)`；两端已经重合时给一个空数组。

    点数按**最大那一维**定：各维同步走完，路径在关节空间里是直的，不会出现某一维先到、
    另一维还在动的折线。这里只管**路径几何**，速度由 `_retime` 决定。
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    span = float(np.abs(end - start).max())
    count = int(np.ceil(span / np.radians(step_deg)))
    if count <= 0:
        return np.zeros((0, len(start)))
    ratios = np.arange(1, count + 1)[:, None] / count
    return start + (end - start) * ratios


def _retime(points, cruise_deg, accel_deg):
    """按巡航上限与加速度上限给一串密点重新计时，两端速度为零。

    Args:
        points: `(点数, 6)` 路径上的密点（弧度），首点是起始位形。
        cruise_deg: `(点数,)` 各点处的速度上限（度/帧）。
        accel_deg: 加速度上限（度/帧²）。

    Returns:
        `(帧数, 6)` 逐帧位形（弧度），不含起点、含终点。

    做法是轨迹时间标定的标准两步：先反向扫一遍算出「为了在终点停住，各点最多能有多快」
    的减速包络，再正向按加速度上限逐帧积分，速度取「上一帧速度 + 加速度」与包络的较小者。
    于是速度沿路径连续、加速度有界，而路径几何一点没变。

    距离度量用**最大关节增量**，与 `_densify` 同一把尺 —— 换尺子会让限速的含义悄悄改变。
    """
    points = np.asarray(points, float)
    gaps = np.degrees(np.abs(np.diff(points, axis=0)).max(axis=1))
    arc = np.r_[0.0, np.cumsum(gaps)]
    total = float(arc[-1])
    if total <= 0.0:
        return np.zeros((0, points.shape[1]))

    envelope = np.asarray(cruise_deg, float).copy()
    envelope[-1] = 0.0
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = min(envelope[i],
                          float(np.sqrt(envelope[i + 1] ** 2 + 2.0 * accel_deg * gaps[i])))

    out, walked, speed = [], 0.0, 0.0
    while walked < total:
        speed = min(speed + accel_deg, float(np.interp(walked, arc, envelope)))
        # 起步那一帧包络本身就是 0 之外的情形不存在（终点才为 0），但浮点上仍要保证前进。
        walked = min(walked + max(speed, 1e-6), total)
        out.append(np.array([np.interp(walked, arc, points[:, j])
                             for j in range(points.shape[1])]))
    return np.asarray(out)


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
    # 瞄点用**实测**的夹持口袋：它是老版那批（9/9、用户认可为稳定）稳定夹住时方块在
    # 夹爪局部系的位置中位。
    #
    # ★ 「张开后两指尖中点」这个纯几何定义**已被实测否掉**：换上去后成功率 8/10 → 5/10，
    #   解不出来的槽位从 3 个涨到 8 个。原因是动指绕轴摆动，张开时的中点不是方块最终待的
    #   地方 —— 合拢过程本身会把方块推走。居中与否改成量出来再修（见 `check_aim.py` 报的
    #   「方块中心 vs 两指尖中点」那一列），不靠换定义猜。
    pocket = recipe.pocket(scene)
    q_above, q_grasp, _, _, ok = solve_aligned_grasp(
        kin, frame, np.asarray(home_qpos, float), item_xy, spec["item_half"],
        APPROACH_H, approach_rad, limits_lo, limits_hi, pocket=pocket, item_yaw=item_yaw)
    if not ok:
        return None
    yaw_err = abs(np.degrees(jaw_yaw_error(kin, q_grasp, item_yaw)))
    if yaw_err > MAX_JAW_YAW_DEG:
        return None
    # 逆解真把口袋送到物体中心了吗 —— 自己再量一次，别信 `ok`。
    p_local, rot_local = kin.ee_pose_local(q_grasp[:5])
    aim = frame.to_world(p_local + rot_local @ pocket)
    if np.linalg.norm(aim - np.array([*item_xy, spec["item_half"]])) * 1000.0 > MAX_AIM_MM:
        return None

    # 放置位姿：物体中心要落到「箱底 + 物体半高 + 余量」，而不是搬运高度 ——
    # `solve_aligned_grasp` 解的是"夹持口袋落在目标点"，而物体就握在口袋上，
    # 所以这个目标点就是松手瞬间物体中心该在的地方。
    # 同时它的 `approach_h` 给出高位过渡姿态，搬运走高位、只在箱口正上方才下来。
    drop_z = BIN_FLOOR_Z + spec["item_half"] + DROP_CLEARANCE
    q_carry, q_drop, _, _, ok = solve_aligned_grasp(
        kin, frame, q_grasp, bin_xy, drop_z,
        APPROACH_H, low, limits_lo, limits_hi, pocket=pocket, item_yaw=item_yaw)
    if not ok:
        return None

    home = np.asarray(home_qpos, float)
    open_a, open_d = spec["open_approach_pct"], spec["open_descent_pct"]
    closed_grasp = _with_grip(q_grasp, recipe.CLOSE_PCT, low, high)
    closed_above = _with_grip(q_above, recipe.CLOSE_PCT, low, high)
    closed_carry = _with_grip(q_carry, recipe.CLOSE_PCT, low, high)
    closed_drop = _with_grip(q_drop, recipe.CLOSE_PCT, low, high)
    # 脚本：`move` 是移动到某位形（带巡航上限），`grip` 是原地改夹爪开度（带速率上限），
    # `hold` 是原地保持若干帧。连续的 `move` 会被串成**一条**路径一起重新计时，
    # 所以段界上速度是连续的；`grip` 与 `hold` 之间手臂静止，天然是停点。
    script = [
        ("move", _with_grip(q_above, open_a, low, high), CRUISE_FREE_DEG),   # ① 进场（空载）
        ("move", _with_grip(q_grasp, open_d, low, high), CRUISE_LOAD_DEG),   # ② 下降到抓取位
        ("grip", closed_grasp, GRIP_RATE_PCT),                               # ③ 合拢到底（渐变）
        ("hold", closed_grasp, CLOSE_FRAMES),                                # ④ 保持，等堵转建立
        ("move", closed_above, CRUISE_LOAD_DEG),                             # ⑤ 原地垂直抬起
        ("move", closed_carry, CRUISE_LOAD_DEG),                             # ⑥ 高位平移
        ("move", closed_drop, CRUISE_LOAD_DEG),                              # ⑦ 下降到投放高度
        ("grip", _with_grip(q_drop, open_a, low, high), GRIP_RATE_PCT),      # ⑧ 松手（渐变）
        ("hold", _with_grip(q_drop, open_a, low, high), RELEASE_FRAMES),     # ⑨ 等物体落稳
        ("move", _with_grip(home, open_a, low, high), CRUISE_FREE_DEG),      # ⑩ 回家（空载）
    ]

    frames, cursor = [], home
    run_points, run_cruise = [home], [CRUISE_FREE_DEG]

    def flush_run():
        """把已积累的连续移动段作为一条路径重新计时并落帧。"""
        if len(run_points) > 1:
            timed = _retime(np.stack(run_points), np.asarray(run_cruise), MAX_ACCEL_DEG)
            if len(timed):
                frames.append(timed)

    for kind, target, rate in script:
        if kind == "move":
            dense = _densify(cursor, target, DENSE_STEP_DEG)
            if len(dense):
                run_points.extend(dense)
                run_cruise.extend([rate] * len(dense))
                cursor = target
            continue
        flush_run()
        if kind == "grip":
            # 夹爪单独一路渐变，手臂不动 —— 与真机遥操一致：人先把臂停稳再捏扳机。
            span = abs(target[5] - cursor[5]) / (high - low) * 100.0
            count = max(1, int(np.ceil(span / rate)))
            ratios = np.arange(1, count + 1)[:, None] / count
            frames.append(cursor + (target - cursor) * ratios)
        else:
            frames.append(np.repeat(np.asarray(target, float)[None], rate, axis=0))
        cursor = np.asarray(target, float)
        run_points, run_cruise = [cursor], [CRUISE_FREE_DEG]
    flush_run()

    qpos = np.vstack([f for f in frames if len(f)])
    out = np.degrees(qpos)
    out[:, 5] = (qpos[:, 5] - low) / (high - low) * 100.0
    return out
