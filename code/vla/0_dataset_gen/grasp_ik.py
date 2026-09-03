"""脚本化抓取需要的运动学：抓取点、锁俯仰的雅可比、以及解预抓取位姿的 IK。

## 为什么抓取也不必交给策略

v2（真机轨迹开环重放）之所以必须靠猜方块位置，是因为**真机画面里方块的真实位置无从得知**，
只能从关节角反推、反推有 1cm 级误差 ⇒ 只能在 ±1.6cm 网格里试，端到端良率 0.06%。
v3 把抓取交给策略正是为了绕开这一点（策略看图像，闭环，抓稳率 40.6%）。

但 v3 里**方块是环境自己撒的点，位置在 recipe 里逐槽记着**（`item_xy`）——
"位置不可知"这个前提在 v3 已经不成立了。而策略的代价是运动观感：实测策略段
lift/elbow 的**行程/净位移 = 12–144 倍**（来回摸索），方向翻转 1.5–3.0 次/秒/关节，
真机只有 0.46。这不是高频噪声，低通滤不掉（β=0.9、9 帧滞后也只降到 1.1）。

位置已知 + 目标是"漂亮" ⇒ 抓取和放置一样是**确定的几何问题**，用伺服算，不用学。

## 三个几何量都标定自真机 300 集（不是拍脑袋）

用产线自己的 `key_frames` 取 pinch 帧，对 300 集做 FK：

| 量 | 真机中位 | p10–p90 | 本模块取值 |
|---|---|---|---|
| lift+elbow+wflex（决定夹爪俯仰） | 79.96° | 55.0 ~ 98.4° | `SUM_GRASP=80.0` |
| wrist_roll | 1.01° | −21.8 ~ 39.1° | `WROLL_GRASP=0.0` |

夹持口袋 `POCKET_LOCAL` 不在这张表里 —— 它标定自**仿真里实际抓住的那一刻**
（见 `grasp_point` 的说明），因为真机数据只能告诉我们手臂在哪，告诉不了夹爪的碰撞体在哪。

**俯仰只由 lift+elbow+wflex 之和决定**（平面链，已数值验证：三个关节各 +10° 都让俯仰
+9.10°，和相同则俯仰相同）。所以"锁俯仰"＝在伺服时让 wflex 吃掉 lift/elbow 的变化，
落到雅可比上就是把 wflex 那一列从 lift/elbow 两列里减掉。
"""

import numpy as np
from recipe import PITCH_CANDIDATES_DEG, SUM_GRASP_DEG, WROLL_GRASP_DEG
from servo import dls_delta

# 夹持口袋（方块中心相对 `gripper_frame_link` 的局部偏移）**逐场景不同**，
# 声明在 `recipe.py` 那张表里，本模块一律由调用方显式传进来。
# 原产线用 `POCKET_X` / `POCKET_Z` 环境变量传它，代价是"这批数据按哪套参数产的"
# 只存在于当时那条命令里 —— 实账：混着旧配置产了 4276 集，咬合 11.3mm（定稿 38.4mm），
# 全部作废重产。
#
# ★口袋标定的一条重要否证：曾想沿"掌心→指尖"轴把口袋往爪腔里挪来加深咬合，
#   实测**这个旋钮救不了夹得浅**。无论瞄多深，方块最终都停在指尖外 19.7~20.1mm
#   （夹深 0 / 6 / 10mm 三档分别 20.1 / 19.9 / 19.7mm），只是收成从 46 掉到 33 再到 8 集。
#   原因是两指**朝指尖收拢**：合拢 18.5° 时指尖间距 37.8mm、内 1cm 41.3mm、内 2cm 52.2mm，
#   40mm 的刚体方块只有在指尖那一档才被夹住。真机看着夹得实是因为真机夹的是**海绵**
#   （压扁 4.72mm，指头陷进去），仿真里是刚体 —— 材质差异，不是控制问题。该旋钮已删。
SUM_GRASP = np.radians(SUM_GRASP_DEG)    # lift+elbow+wflex 锁定值 ⇒ 夹爪俯仰
WROLL_GRASP = np.radians(WROLL_GRASP_DEG)
PITCH_CANDIDATES = np.radians(PITCH_CANDIDATES_DEG)
# 伺服每步的关节增量上限（弧度）。限步是为了让解出来的轨迹是连续小步，而不是一跳到位。
MAX_STEP = np.radians(2.0)


def grasp_point(kin, q, pocket):
    """夹持口袋在**基座系**下的位置：`gripper_frame_link` 位姿 + 一个刚体偏移。

    Args:
        kin: `servo.ArmKinematics`。
        q: 六维关节角（弧度）。
        pocket: 口袋在 `gripper_frame_link` 局部系的偏移（m），逐场景不同，
            由调用方从 `recipe.pocket(scene)` 取 —— 本模块不留默认值，
            默认值会让"这批数据按哪套标定产的"变成看不出来的事。

    Returns:
        `(3,)` 基座系坐标。

    口袋是**实测标定**值，不是从 URDF 几何推的：`finger1_tip` / `finger2_tip` 是纯 frame、
    没有碰撞体（碰撞体只挂在 `gripper_link` 与 `moving_jaw_so101_v1_link` 上），
    而且两指中点随张开角在夹爪开到 48° 时平移 31.6mm，不是刚体点。
    """
    p, R = kin.ee_pose_local(q)
    return p + R @ np.asarray(pocket, float)


def pitch_locked_jacobian(kin, q, *, pocket, delta=1e-4):
    """抓取点对 (pan, lift, elbow) 的雅可比，**且 wflex 同步补偿以锁住俯仰**。

    数值差分而不是解析式：抓取点是腕上的一个偏移点、不是 link 原点，解析式还要额外补
    姿态那一项；而且本仓库在坐标约定上栽过多次，差分只依赖已被自检验过的 FK。
    lift/elbow 每扰动 +δ，就给 wflex 扰动 −δ —— 这正是实际控制时要做的事，
    所以差分出来的就是**受约束运动**的真实雅可比，不需要再做零空间投影。
    """
    q = np.asarray(q, float).copy()
    J = np.zeros((3, 3))
    for col, j in enumerate((0, 1, 2)):
        qp, qm = q.copy(), q.copy()
        qp[j] += delta
        qm[j] -= delta
        if j in (1, 2):                      # lift / elbow 动 ⇒ wflex 反向吃掉，俯仰不变
            qp[3] -= delta
            qm[3] += delta
        J[:, col] = (grasp_point(kin, qp, pocket) - grasp_point(kin, qm, pocket)) / (2 * delta)
    return J


def apply_pitch_lock(q, grip=None, sum_pitch=SUM_GRASP, wroll=None):
    """把 wflex/wroll 按锁定关系写回：wflex = sum_pitch − lift − elbow。"""
    q = np.asarray(q, float).copy()
    q[3] = sum_pitch - q[1] - q[2]
    q[4] = WROLL_GRASP if wroll is None else float(wroll)
    if grip is not None:
        q[5] = grip
    return q


def jaw_yaw_error(kin, q):
    """两指连线在水平面内相对**方块的面**偏了多少（弧度，取到最近的 90° 倍数）。

    方块是世界轴对齐的 40mm 立方，而两指连线随 shoulder_pan 一起转 —— 偏 θ 角时，
    两指要跨越的宽度是 `40/cos θ`：偏 31.9° 就要跨 47.1mm，恰好等于张 26° 时的指尖间距
    ⇒ **夹不住**。实测撒点区 y≥10cm（pan≤−25°）的方块几乎全部因此抓失败，
    数据里 y 的覆盖被截在 8.1cm，而撒点区到 14.5cm。
    """
    t = kin.tips_local(q)
    d = t["finger2_tip"] - t["finger1_tip"]
    a = np.degrees(np.arctan2(d[1], d[0]))
    return np.radians(((a + 45.0) % 90.0) - 45.0)


def solve_grasp_pose(kin, base, q_seed, target_world, grip, lo, hi,
                     sum_pitch=SUM_GRASP, iters=300, tol=1e-4,
                     max_dq=MAX_STEP, *, pocket, wroll=None):
    """解「抓取点落在 target_world」的关节角，俯仰与 wroll 按锁定关系走。

    只有 pan/lift/elbow 三个自由度对三维位置 —— 精确定解，没有零空间可漂。
    返回 (q, 残差 m, 是否收敛)。**不收敛就如实返回 False**，让调用方弃掉这个槽位，
    而不是拿一个没解出来的位姿继续往下跑。
    """
    q = apply_pitch_lock(np.asarray(q_seed, float).copy(), grip, sum_pitch, wroll)
    err = np.inf
    for _ in range(iters):
        e = np.asarray(target_world, float) - base.to_world(grasp_point(kin, q, pocket))
        err = float(np.linalg.norm(e))
        if err < tol:
            return q, err, True
        dq = dls_delta(pitch_locked_jacobian(kin, q, pocket=pocket), base.vec_to_local(e),
                       lam=0.02, max_dq=max_dq)
        q[:3] = np.clip(q[:3] + dq, lo[:3], hi[:3])
        q = apply_pitch_lock(q, grip, sum_pitch, wroll)
        q[3] = np.clip(q[3], lo[3], hi[3])
    return q, err, False


def solve_approach_and_grasp(kin, base, q_seed, item_xy, item_half, approach_h,
                             grip, lo, hi, *, pocket, wroll=None):
    """一次解出「预抓取（正上方）」与「抓取」两个位姿，**共用同一个俯仰**。

    两个位姿必须同俯仰，否则下降段还要一边平移一边转腕，既不好看也容易把方块推走。
    俯仰按 `PITCH_CANDIDATES` 依次试，取第一个两处都收敛的。
    返回 (q_above, q_grasp, 俯仰, 是否成功)。
    """
    for sp in PITCH_CANDIDATES:
        tgt_g = np.array([item_xy[0], item_xy[1], item_half])
        tgt_a = tgt_g + np.array([0.0, 0.0, approach_h])
        q_a, _, ok_a = solve_grasp_pose(kin, base, q_seed, tgt_a, grip, lo, hi, sp,
                                       pocket=pocket, wroll=wroll)
        if not ok_a:
            continue
        q_g, _, ok_g = solve_grasp_pose(kin, base, q_a, tgt_g, grip, lo, hi, sp,
                                       pocket=pocket, wroll=wroll)
        if ok_g:
            return q_a, q_g, sp, True
    return None, None, None, False


def solve_aligned_grasp(kin, base, q_seed, item_xy, item_half, approach_h, grip, lo, hi,
                        *, pocket, iters=4):
    """解预抓取/抓取位姿，并用 wrist_roll 把两指连线**对齐到方块的面**。

    不对齐的代价是硬的：偏 θ 角时两指要跨 `40/cos θ` mm，撒点区远端（pan≈−35°）
    偏到 31.9° ⇒ 要跨 47.1mm，而张 26° 只有 47mm。迭代把偏角打到 0 之后，
    整个撒点区都只需要跨 40mm。返回 (q_above, q_grasp, pitch, wroll, ok)。
    """
    wr = 0.0
    out = None
    for _ in range(iters):
        qa, qg, sp, c = solve_approach_and_grasp(kin, base, q_seed, item_xy, item_half,
                                                 approach_h, grip, lo, hi,
                                                 pocket=pocket, wroll=wr)
        if not c:
            return None, None, None, None, False
        out = (qa, qg, sp)
        e = jaw_yaw_error(kin, qg)
        if abs(e) < np.radians(1.0):
            break
        wr = float(np.clip(wr - e, lo[4] + 0.02, hi[4] - 0.02))
    qa, qg, sp = out
    return qa, qg, sp, wr, True
