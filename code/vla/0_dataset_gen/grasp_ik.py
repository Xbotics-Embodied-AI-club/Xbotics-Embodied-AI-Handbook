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
# 夹持口袋的 y 分量（m）：三个场景共用，实测自捏住那一刻物体在夹爪局部系的 y 中位。
POCKET_Y_M = 0.00075


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


def jaw_yaw_error(kin, q, item_yaw=0.0):
    """两指连线在水平面内相对**方块的面**偏了多少（弧度，取到最近的 90° 倍数）。

    两指连线随 shoulder_pan 一起转 —— 相对方块的面偏 θ 角时，两指要跨越的宽度是
    `40/cos θ`：偏 31.9° 就要跨 47.1mm，恰好等于张 26° 时的指尖间距 ⇒ **夹不住**。

    ★ `item_yaw` 不可省。早先这里默认"方块是世界轴对齐的"，于是本函数量的是两指连线
      相对**世界 x/y 轴**的偏角 —— 而方块复位时是带随机自旋的。对齐世界轴等于对方块的面
      偏了方块自身那个自旋角，越接近 45° 越夹不住：实测 10 集里「离面对齐」>28° 的四集
      全部失败（方块被指尖撞角推走、从没抬起来过），<10.3° 的全部成功。
      而那五集的瞄点残差只有 0.01~0.09mm、俯仰角与成功组逐位相同 —— 错的只有朝向。
    """
    t = kin.tips_local(q)
    d = t["finger2_tip"] - t["finger1_tip"]
    a = np.degrees(np.arctan2(d[1], d[0])) - np.degrees(item_yaw)
    return np.radians(((a + 45.0) % 90.0) - 45.0)


# 腕滚的偏好值（弧度）。方块 90° 对称 ⇒ 腕滚 ±90° 抓的是同一组面，等价解里离它最近的胜出。
#
# 取 0°：**判据是"落进真机的腕滚分布"**。对 `pick_up_a_cube` 299 集取抓取那一刻的
# `wrist_roll`：中位 **+1.4°**，p10 −19.4 / p90 +34.5，最小 −59.1 / 最大 +96.2。
# 同一批 16 个种子上扫偏好值（腕滚中位 / p10 / p90 ｜ 可规划 ｜ 成功）：
#     −20° → −25.4 / −59.9 / +14.0 ｜ 14/16 ｜ 14/14
#       0° → −11.8 / −30.0 / +17.9 ｜ 11/16 ｜ 11/11
#     +10° → −10.9 / −28.2 / +18.5 ｜ 10/16 ｜ 10/10
# −20° 那档的**中位落在真机 p10 以下**、p10 顶到真机的最小值 —— 整批挤在真机分布的尾巴上。
# 0° 之后落进真机的 p10~p90。成功率各档都是 100%，代价只是可规划 14→11（超采 1.35 倍吸收）。
#
# ★ 旧值 −20° 的理由**已经失效，别照抄**：它取自"打滑集腕滚在 +26~+30°、稳定集在
#   −40~+24°"，那是**旧夹爪口径**（力矩上限 100 / 2.0 + 合拢阶梯）下的观察。
#   机制是两指尖沿插入方向错开形成力偶、腕滚决定它相对重力的朝向；力矩上限降到 0.20
#   之后这个力偶弱了一个量级（堵转段物体在爪内逐帧位移 0.565 → 0.041mm），
#   "躲开正腕滚"这条约束不再成立。
#
# ★ 取 0 曾被记为"无效实验"（理由：+29.3° 的 90° 等价解是 −60.7/+119.3/+209.3，
#   +29.3 本身离 0 最近 ⇒ 原解恒胜）。那条只说明**在那一集上**它不改变结果，
#   不等于在整批上不改变分布 —— 实测中位从 −25.4 变到 −11.8，是能证伪的。
PREFERRED_WROLL = np.radians(0.0)

# 夹持面与方块的面允许差多少（度，三维夹角）。俯仰候选要过了它才录取。
#
# 取 5：实测同一批十集里九集是 1.3~3.8°（俯仰 85.0~86.5°），落到俯仰 73.2° 的那一集
# 是 10.7°，腕部画面上方块角对着镜头。5 在两群之间，离两边都不近。
MAX_JAW_FACE_DEG = 5.0

# 两指跨过物体所需的富余（m）。判据是**物理的**：偏 θ 角时两指要跨 `宽度/cos θ`，
# 它必须落在该开度下的指尖间距之内、再留这么多余量。
#
# ★ 早先这里是一个随手取的角度阈值（2.0°）。它挡错过：某集的负腕滚候选残余 2.37°、
#   要跨 40.03mm，而下降开度给出的指尖间距是 52.19mm —— 余量 12mm 却被判不可用，
#   于是产线只能采用落在危险侧的正腕滚。角度阈值与"夹不夹得住"之间没有直接关系，
#   真正决定它的是宽度与间距之比。
SPAN_MARGIN_M = 0.006


def span_fits(kin, q, item_yaw, item_half, grip_rad):
    """这个位形下两指跨得过物体吗。

    Args:
        kin: `ArmKinematics`。
        q: `(6,)` 抓取位形（弧度）。
        item_yaw: 物体绕 z 的自旋角（弧度）。
        item_half: 物体半宽（m）。
        grip_rad: 判间距用的夹爪开度（弧度）。

    Returns:
        `bool`。
    """
    err = abs(jaw_yaw_error(kin, q, item_yaw))
    need = 2.0 * item_half / max(np.cos(err), 1e-6)
    tips = kin.tips_local(np.r_[np.zeros(5), grip_rad])
    have = float(np.linalg.norm(tips["finger2_tip"] - tips["finger1_tip"]))
    return need <= have - SPAN_MARGIN_M


def jaw_midpoint(kin, grip_rad):
    """两指尖中点在**夹爪坐标系**里的位置（米）。

    Args:
        kin: `ArmKinematics`。
        grip_rad: 夹爪关节角（弧度）—— 取**下降段的开度**，那才是两指围住物体时的宽度。

    Returns:
        `(3,)` 中点在 `gripper_frame_link` 局部系的坐标。

    结果只与夹爪开度有关、与手臂位形无关（从腕到指是一条固定链），所以手臂那五维给零即可。

    ★ **这是诊断量，不是产线的瞄点。** 产线瞄 `recipe.pocket(scene)` 那对实测值；
      `check_aim.py` 用本函数报「实测口袋离几何中点差多少」，好让人看出瞄点偏在哪一侧。

    ★ 「拿两指中点当瞄点」曾经上过产线，**被实测否掉**：成功率 8/10 → 5/10，解不出来的
      槽位 3 → 8。理由是动指绕轴摆动，张开时的中点不是物体最终待的地方 —— 合拢过程
      本身会把物体推走。当时支持它的论证是「实测口袋量的是『堵转之后物体实际停在哪』，
      那个位置被楔形指挤偏的缺陷污染了」；论证听起来成立，但换上去成功率就掉，
      ⇒ 以实测为准。本轮 405 集上实测口袋给出 93.3% 的成功率，这条结论继续成立。
    """
    q = np.zeros(6)
    q[5] = grip_rad
    tips = kin.tips_local(q)
    mid = 0.5 * (tips["finger1_tip"] + tips["finger2_tip"])
    p_ee, rot = kin.ee_pose_local(q)
    return rot.T @ (mid - p_ee)


def aim_point(kin, scene):
    """这条产线抓取时瞄的那个点（`gripper_frame_link` 局部系，米）——**唯一定义处**。

    Args:
        kin: `ArmKinematics`。
        scene: `recipe.SCENES` 的键。

    Returns:
        `(3,)` 该场景下降开度下的两指尖中点。

    ★ 产线与所有诊断工具都必须调这一个函数。瞄点是"规划把物体送到爪里的哪儿"，
      而观感指标（末端速度 / 加速度 / 逐帧转角）量的必须是**同一个点** ——
      两边取不同的点时，报出来的曲线不是被规划的那一条，而且看不出来。

    「居中」= 物体中心落在**两个夹持面的正中间**，于是两侧各留同样的缝，然后合拢夹住。
    夹持面从两块指头的**碰撞网格**现算，只取物体所占的那一小段 y/z 邻域里的顶点：
    固定指取它最靠内（x 最小）的顶点，动指取它最靠内（x 最大）的顶点。

    实测（cube40，40mm，`gripper_frame_link` 局部系，mm）：

        开度 26.6%  固定指面 0.00  动指面 −41.21  中线 −20.60  两侧各空  0.60
        开度 36.4%  固定指面 0.00  动指面 −54.44  中线 −27.22  两侧各空  7.22
        开度 46.4%  固定指面 0.00  动指面 −65.83  中线 −32.91  两侧各空 12.91

    ★ **这个量前后错过三次，三次都因为拿错了"指头在哪"：**
      · `jaw_midpoint`（两指尖中点）—— `finger1_tip` / `finger2_tip` 是**纯 frame、
        没有碰撞体**，随开度越张越往爪外跑（60% 开度下已到 −46mm）。拿它当瞄点时
        物体落在动指外侧 17.8mm，画面上整个挂在爪外。
      · `recipe.pocket` 那组实测值（−18.63）—— 它量的是"堵转之后物体停在哪"，
        偏向固定指 8.6mm，画面上物体贴着左指。
      · link **原点**（固定指 −7.9 / 动指 −28.1）—— 原点不是面。固定指的面其实在 0.00。
      ⇒ 只有碰撞网格上朝内的那两个面才定义得了"居中"。

    ★ 它是**推导量**：跟着该场景的 `open_descent_pct` 走，换场景、换开度都自动对。
    """
    from so101_sim.robots.so101_base.so101 import grip_rad_from_pct
    from transforms3d.quaternions import quat2mat

    import recipe
    from measure_descent_clearance import link_local_points

    spec = recipe.SCENES[scene]
    q = np.zeros(6)
    q[5] = grip_rad_from_pct(spec["open_descent_pct"])
    names = [link.name for link in kin._art.get_links()]
    kin._pm.compute_forward_kinematics(np.r_[q, np.zeros(max(0, kin.n - 6))][:kin.n])
    ee = kin._pm.get_link_pose(names.index("gripper_frame_link"))
    origin, rot = np.asarray(ee.p, float), quat2mat(np.asarray(ee.q, float))

    points = link_local_points(kin)
    half = spec["item_half"]
    faces = []
    for link, inner in (("gripper_link", "min"), ("moving_jaw_so101_v1_link", "max")):
        pose = kin._pm.get_link_pose(names.index(link))
        world = np.asarray(pose.p, float) + points[link] @ quat2mat(np.asarray(pose.q, float)).T
        local = (world - origin) @ rot
        # 只看物体真正占住的那一小段：更远处的机身不参与"夹持面"这件事。
        band = local[(np.abs(local[:, 1] - POCKET_Y_M) < half)
                     & (np.abs(local[:, 2] - spec["pocket_z"]) < half)]
        if not len(band):
            raise RuntimeError(f"{link} 在物体高度带里没有碰撞顶点 —— 夹持面无从算起")
        faces.append(band[:, 0].min() if inner == "min" else band[:, 0].max())
    return np.array([0.5 * (faces[0] + faces[1]), POCKET_Y_M, spec["pocket_z"]])


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


def jaw_face_angle_deg(kin, base, q, item_yaw, symmetry="square"):
    """「抓得正不正」的三维残差（度）—— 判据随物体的对称性变。

    Args:
        kin: `ArmKinematics`。
        base: `BaseFrame`。
        q: 抓取位形（前五个是手臂关节，弧度）。
        item_yaw: 物体绕 z 的自旋角（弧度）；`round` 时不参与。
        symmetry: `recipe.SCENES[场景]["item_symmetry"]`。

    Returns:
        残差（度），0 最好。

    · `square`（方块）：夹持轴与最近那个面法向的夹角。两个夹持面要与方块的一对面平行。
    · `round`（圆柱）：**圆柱没有面**，绕自身轴连续对称 ⇒ 任何水平朝向都等价，
      唯一的要求是夹持轴垂直于柱轴。所以判的是「夹持轴离水平面差多少」。
      拿 `square` 那套去卡圆柱是错的：它会按一个**虚构的自旋角**算出 0~45° 的残差，
      于是同样好的抓取有的被放行、有的被否，而且否得毫无规律。

    ★ **和 `jaw_yaw_error` 不是一回事，两个都要判。** 那一个把夹持轴投影到水平面再比，
      于是"轴本身翘起来了"它一律看不见 —— 俯仰 90° 时两者等价，俯仰越浅差得越多。
      实账：某一集俯仰落到 73.2°（其余九集 85.0~86.5°），水平投影残余 1.97° 全绿，
      而三维夹角是 10.7°（其余九集 1.3~3.8°）；腕部画面里方块是**角对着镜头**的，
      两个面都看得见，目审一眼就读出来了。
    """
    _, rot = kin.ee_pose_local(np.asarray(q, float)[:5])
    axis = base.R @ rot @ np.array([1.0, 0.0, 0.0])
    if symmetry == "round":
        # 夹持轴与柱轴（世界 z）的夹角该是 90°，差多少就是残差。
        return abs(90.0 - float(np.degrees(np.arccos(
            np.clip(abs(axis @ np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))))
    c, s = np.cos(item_yaw), np.sin(item_yaw)
    normals = (np.array([c, s, 0.0]), np.array([-s, c, 0.0]), np.array([0.0, 0.0, 1.0]))
    return float(min(np.degrees(np.arccos(np.clip(abs(axis @ n), -1.0, 1.0)))
                     for n in normals))


def solve_approach_and_grasp(kin, base, q_seed, item_xy, item_half, approach_h,
                             grip, lo, hi, *, pocket, wroll=None, pitch=None):
    """一次解出「预抓取（正上方）」与「抓取」两个位姿，**共用同一个俯仰**。

    两个位姿必须同俯仰，否则下降段还要一边平移一边转腕，既不好看也容易把方块推走。
    俯仰按 `PITCH_CANDIDATES` 依次试，取第一个两处都收敛的；`pitch` 给了就只试它
    （由 `solve_aligned_grasp` 在外层逐档试，好让平行度也参与选俯仰）。

    ★ **不要在这里判夹持面平行度。** 本函数是腕滚对齐迭代的内层：第一轮进来时腕滚还是
      初值，两指连线离方块的面可以差到 45°，平行度当然不合格 —— 于是每一档俯仰都被否，
      整个场景被弃。实测那样做 127 个种子里弃掉 117 个（不加时弃 0 个）。
      平行度是**收敛之后**才有意义的量，判它的地方在 `expert.plan` 的落盘门那一排。

    返回 (q_above, q_grasp, 俯仰, 是否成功)。
    """
    for sp in (PITCH_CANDIDATES if pitch is None else (pitch,)):
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


def _align_roll(kin, base, q_seed, item_xy, item_half, approach_h, grip, lo, hi,
                *, pocket, item_yaw, wr0, pitch=None, iters=4):
    """从 `wr0` 起迭代 wrist_roll，把两指连线对齐到方块的面。

    ★ 这个迭代对**主解和 ±90° 等价解一样都要跑**。早先只有主解跑、等价解直接照抄
      `wr ± 90°` 单解一次，于是等价解带着 6~15° 的对齐残余出场，被 `span_fits` 全数否掉
      （实测 ep0/ep4/ep9：主解残余 0.15/0.03/0.45°，+90° 候选 6.08/5.95/6.33°，
      +180° 候选 15.20/11.14/8.93° —— 每转 90° 长约 6° 且单调累加）。
      成因是抓取姿态下腕滚轴并非铅垂（俯仰 85~90°，不是正 90°），绕它转 90°
      在水平投影里就不是正好 90°；差的这一点只要再迭代一轮就归零。

    Returns:
        `(q_above, q_grasp, pitch, wroll, ok)`。
    """
    wr = float(wr0)
    out = None
    for _ in range(iters):
        qa, qg, sp, c = solve_approach_and_grasp(kin, base, q_seed, item_xy, item_half,
                                                 approach_h, grip, lo, hi,
                                                 pocket=pocket, wroll=wr, pitch=pitch)
        if not c:
            return None, None, None, None, False
        out = (qa, qg, sp)
        e = jaw_yaw_error(kin, qg, item_yaw)
        if abs(e) < np.radians(1.0):
            break
        wr = float(np.clip(wr - e, lo[4] + 0.02, hi[4] - 0.02))
    return out[0], out[1], out[2], wr, True


def solve_aligned_grasp(kin, base, q_seed, item_xy, item_half, approach_h, grip, lo, hi,
                        *, pocket, item_yaw=0.0, symmetry="square", iters=4):
    """解预抓取/抓取位姿，并用 wrist_roll 把两指连线**对齐到方块的面**。

    不对齐的代价是硬的：偏 θ 角时两指要跨 `40/cos θ` mm，撒点区远端（pan≈−35°）
    偏到 31.9° ⇒ 要跨 47.1mm，而张 26° 只有 47mm。迭代把偏角打到 0 之后，
    整个撒点区都只需要跨 40mm。返回 (q_above, q_grasp, pitch, wroll, ok)。

    `item_yaw` 是方块绕 z 的自旋角（弧度）；对齐的目标是**方块的面**，不是世界轴。

    方块 90° 对称 ⇒ 腕滚加减 90° 抓的是同一组面，是**真正的等价解**。在它们里挑离
    `PREFERRED_WROLL` 最近的那个（见该常量处：判据是落进真机的腕滚分布），
    等价于挑腕部最不别扭的那个方向。

    ★ **俯仰在外层逐档试，平行度参与选档。** 每一档俯仰都跑完整的腕滚对齐（主解 ＋
      ±90° 等价解），收敛之后再判 `jaw_face_angle_deg`，不合格就换下一档俯仰。
      两种更省事的写法都试过、都不对：
        · 把平行度放进 `solve_approach_and_grasp` 的内层 —— 那里腕滚还没对齐，
          离方块的面能差到 45°，每一档都被否 ⇒ 127 个种子弃 117 个。
        · 只在最后当一道弃用门 —— 收敛了但不平行的场景整个丢掉，实测 35 个种子弃 7 个
          （占全部弃用的 88%）。而且**弃得不随机**：被弃的是只有浅俯仰才解得出来的位姿，
          集中在工作区边缘 ⇒ 数据集系统性缺那一带，评测时方块照样撒到那里。
    """
    kw = {"pocket": pocket, "item_yaw": item_yaw, "iters": iters}
    args = (kin, base, q_seed, item_xy, item_half, approach_h, grip, lo, hi)
    for sp_try in PITCH_CANDIDATES:
        qa, qg, sp, wr, ok = _align_roll(*args, wr0=0.0, pitch=sp_try, **kw)
        if not ok:
            continue
        best = (abs(wr - PREFERRED_WROLL), wr, qa, qg, sp)
        for k in (-2, -1, 1, 2):
            seed = wr + k * np.pi / 2.0
            if not (lo[4] + 0.02 <= seed <= hi[4] - 0.02):
                continue
            # ★ 种子换成已收敛的 qa **已被实测否证**：ep12/ep27 的负腕滚候选照样不收敛，
            #   逐位不变。那两个位姿是真的解不出来 —— 腕滚 −49.6/−43.5° 时锁俯仰的三自由度
            #   逆解在限位内无法把口袋送到方块中心，是运动学硬限制，不是种子问题。
            ca, cg, csp, cwr, ok2 = _align_roll(*args, wr0=seed, pitch=sp_try, **kw)
            if not ok2 or not span_fits(kin, cg, item_yaw, item_half, grip):
                continue
            score = abs(cwr - PREFERRED_WROLL)
            if score < best[0]:
                best = (score, cwr, ca, cg, csp)
        _, wr, qa, qg, sp = best
        if jaw_face_angle_deg(kin, base, qg, item_yaw, symmetry) <= MAX_JAW_FACE_DEG:
            return qa, qg, sp, wr, True
    return None, None, None, None, False
