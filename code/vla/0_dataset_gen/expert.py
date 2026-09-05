"""脚本化专家：给定一个场景状态，离线规划出一串真机口径的绝对关节角。

## 为什么是离线规划 + 播放

`Teleoperator.get_action()` 不接受观测（lerobot 基类签名如此），看着物体现算的专家放不进
那个接口。而物体位置在这条路上**本来就是已知的**（环境撒的点，写在强制初始状态里），
所以抓取与放置都是**确定的几何问题**，用运动学算，不用看图、也不用学。
规划落一份 `.npy`，由 `so101_dataset_player --teleop.actions_path=` 播出去。

## 形状：两条连贯的曲线，不是八段直角

目审逐条否掉过直角版：「抓东西为什么一定要抬起来然后垂直下落，再松开夹爪？和真机轨迹
完全不一样」「末端依然有很多无意义的旋转」「轨迹还是不连续、不像人的、再抖动」。
病灶是把动作写成了「到位 → 换方向 → 再到位」的阶段串：段界上末端速度方向突变，
眼睛看到的就是一顿一顿。

现在只有两条一气呵成的曲线，中间只在真正该停的地方停（合拢前、松手前、回家后）：

    ① 取物 —— 一条**落地曲线**：从 home 斜着下来，下降与平移重叠着走，
       最后 5cm 才转成垂直下插（那 5cm 是为了竖直下来不蹭到物体侧面）
    ② 合拢 —— 夹爪按真机速率渐变到底，手臂不动，保持若干帧让堵转建立
    ③ 搬运 —— 一条**弧线**：抬起与平移融成一段，末端从物体上方 8cm 平滑滑到料箱上方
    ④ 松手 —— 渐变张开，物体自由落体进箱
    ⑤ 回家 —— 一段回到 home

拐点不停车：结点之间段内匀速，再用 Hann 窗滑动平均把拐角抹圆，于是速度处处连续
（滑动平均后的速度是原速度的加权平均 ⇒ 峰值仍不超过逐关节上限）。
最后按**末端笛卡尔速度**上限重新计时 —— 关节匀速 ≠ 末端匀速，手臂伸出去之后同样的
关节角速度对应大得多的末端线速度，那正是目审读到的「中间突然飞快一下」。

## 捏到底

夹爪指令一路给 `recipe.CLOSE_PCT[该场景的真机任务]`（真机实测的"扳机到底"值，
cube 1.27 / can 1.735），抓紧靠物体把两指卡死。旧产线下发的是一个
「合拢阶梯」（停在某个开度），于是仿真数据里夹爪 action 的含义是"停在多宽"，而真机
数据里它的含义是"我要多用力合"。策略输出的是 action —— 同一个值在两份数据里含义
不同就是两套映射。真机遥操是人把扳机捏到底，所以这里也必须下发到底。
"到底"取真机实测值而不是 0：真机 action 恰好等于 0.00 的帧只占 0.11%，
下发 0 会让 29% 的仿真帧压在 0 上、且 30.4% 的夹爪读数落到真机从未出现过的区间。

## 单位

输出恒为真机口径：臂五关节是度，夹爪是 0~100 行程百分比。换算标度从 URDF 现读
（`so101.gripper_limit_rad`），不在这里抄一份数。
"""

import numpy as np

import recipe
from grasp_ik import (
    MAX_JAW_FACE_DEG,
    aim_point,
    jaw_face_angle_deg,
    solve_aligned_grasp,
    solve_grasp_pose,
    span_fits,
)
from servo import ArmKinematics, BaseFrame

FPS = 30

# 每步关节增量上限（度/帧），臂五关节。**取自真机 `pick_up_a_cube` 300 集逐帧
# `|Δq|·30` 的 p95**（34.3 / 113.4 / 110.8 / 47.5 / 31.6 °/s ÷ 30）。
#
# ★ 沿用了很久的那组 [0.58, 1.27, 1.32, 0.83, 0.63] 名义上也叫"真机 p95"，实测只有
#   真机的 1/2~1/3。手臂因此爬行 —— 末端速度中位 3.5cm/s 而真机 8.7，观感就是
#   "磨磨蹭蹭、一顿一顿"。这里按实测重设；再快就超过真机了。
STEP_MAX_DEG = np.array([1.14, 3.78, 3.69, 1.58, 1.05])

# 舵机编码器的一个计数（度）。STS3215 是 12 位绝对编码器，**分母是 4095 不是 4096**
# —— lerobot 自己就取 `max_res = 分辨率 − 1`（`motors/motors_bus.py:858`）。
#
# 真机的关节读数落在这个栅格上：最小二乘从 300 集直接反解得 0.08791209，
# 两任务 × action/state × 5 关节共 20 组八位小数全一致。判据要看**绝对值**才分得开：
# 对逐帧差两种分母几乎无差别（差只有几个计数），对绝对值则是 4095 命中 100.0%、
# 4096 只有 47.97~57.59%（±110° 上最大错位 0.0269°，约 0.3 个计数）。
#
# 不量化的话一行 `np.diff(action) / 0.087912` 就能把两份数据分开：仿真原本是连续量，
# `min|Δ|` 只有 7.5e-05~0.0023°。量化之后仿真也才会出现真机那种"读数停住"的台阶
# （唯一取值数占帧数的比例，真机 0.73~2.04%、未量化的仿真 39.7~83.6%）。
# ★ 格点的**相位**不必对齐真机：`_normalize` 里 `mid = (min_+max_)/2`，标定端点和为奇数时
#   落在半格、为偶数时落在整格 —— 那是每台机标定端点的奇偶，没有物理含义，实测三台真机
#   （cube / can ep<200 / can ep>=200）的相位各不相同。仿真是又一台机，相位 0 是它自己的
#   合法标定。要对齐的是**量化本身**（12 位编码器，每台真机都这样），不是相位。
ENCODER_STEP_DEG = 360.0 / 4095.0


def _snap_into(v, lo, hi, step):
    """量化到编码器格点，并保证落在 `[lo, hi]` **界内**的格点上。

    Args:
        v: `(帧数, 通道数)` 待处理的值。
        lo: `(通道数,)` 下界。
        hi: `(通道数,)` 上界。
        step: 格点间距（标量）。

    Returns:
        与 `v` 同形，元素都在格点 `k·step` 上且落在 `[lo, hi]` 内。

    先量化后夹，且夹的两端各自吸附到**界内**最近的格点（下界向上取整、上界向下取整）——
    反过来「先夹后量化」会把刚好夹在边界上的值再推出去半格，实测 −103.47 → −103.4725、
    −58.77 → −58.8132，于是"落在真机范围内"那条审计被这一步的顺序弄红，红的还是假问题。
    """
    return np.clip(np.round(v / step), np.ceil(lo / step), np.floor(hi / step)) * step


def _still_frames(quantiles, rng):
    """按真机那条经验分位曲线采一个静止段长度（帧）。

    Args:
        quantiles: `(p10, p25, p50, p75, p90)`，来自 `recipe.REAL_*_STILL_FRAMES`。
        rng: `numpy` 随机数发生器，由集的种子派生 ⇒ 同一集永远采到同一个值。

    Returns:
        帧数（int，至少 1）。

    在 p10~p90 之间按分位线性插值，两端各留 10% 的概率外推到相邻区间 ——
    要的是"有这么宽的一条分布"，不是把真机那条曲线拟合到小数点后。
    """
    q = np.asarray(quantiles, float)
    u = float(rng.uniform(0.10, 0.90))
    return max(1, round(float(np.interp(u, [0.10, 0.25, 0.50, 0.75, 0.90], q))))

# 臂五关节的容许区间（度），逐关节 [下界, 上界]。
#
# 取**真机 `pick_up_a_cube` 300 集里 `action` 与 `observation.state` 两列各自实际
# 出现过的范围的交集** —— 我们下发的是 action，而仿真的 state 会跟着它走，
# 两列都要落在真机对应列的范围内，所以取交集才两头都不越。
#
# ★ 这里原先直接抄 URDF 限位 `[±110, −110~100, ±96.8, −95~105, −67.2~252.8]`。
#   URDF 说的是"关节能转到哪"，真机数据说的是"人实际转到过哪"，两者差出来的部分
#   就是分布外样本。实测 89 集仿真对 300 集真机，四个通道越界：
#     shoulder_lift 下越 6.15°（−110.98 vs −104.84）
#     wrist_flex    上越 4.87°（105.00 vs 100.13，正好顶在 URDF 限位上）
#     wrist_roll    下越 5.62°（−64.91 vs −59.30）
#     elbow_flex    state 上越 0.52°
#   声明层与官方 merge 全过，只有把数值域摆在一起才看得见。
# 包线本身按**真机对标任务**声明在 `recipe.REAL_ENVELOPE_DEG` 里（cube 与 can 两份，
# 差别不小）。这里只负责把它减去过冲余量。

# 逐关节再往里收这么多度，给**跟随器的过冲**留地方（[下界要抬高, 上界要压低]）。
#
# ★ **lift 这一格必须留 0。** 曾按「实测 state 冲出 action 区间 6.73°」给它 10.5，
#   结果把 home 位形本身夹掉了：仿真静置时手臂在重力下沉到 lift ≈ −103.9（最差 −110.2），
#   而钳制把第 0 帧的指令抬到 −92.97 ⇒ 开集第一帧指令与实际差 11°，手臂窜一下。
#   真机第 0 帧 action ≈ state（−103.16 vs −102.68），从不这样。
#   ⇒ 那 6.73 根本不是过冲，是**静置下沉**，病灶在 home 位形，不在钳制。
#     钳制的下界永远不能高于机器人静置时实际所在的位置。
TRACK_MARGIN_DEG = np.array([
    [0.0, 1.0],     # shoulder_pan   上冲 0.38
    [0.0, 0.0],     # shoulder_lift  见上：那 6.73 是静置下沉，不是过冲
    [0.0, 1.0],     # elbow_flex     上冲 0.46
    [0.5, 2.0],     # wrist_flex     上冲 1.14 / 下冲 0.31
    [0.0, 1.5],     # wrist_roll     上冲 0.88
])



def plan_joint_range_deg(scene):
    """该场景规划实际可用的关节区间（度）= 它对标的真机包线往里收掉过冲余量。

    Args:
        scene: `recipe.SCENES` 的键。

    Returns:
        `(5, 2)` 的 [下界, 上界]。
    """
    real = np.asarray(recipe.REAL_ENVELOPE_DEG[recipe.SCENES[scene]["real_task"]], float)
    return real + TRACK_MARGIN_DEG * np.array([1.0, -1.0])

# Hann 窗宽（帧）：既把拐角抹圆，也充当首尾的加减速。
#
# ★ 9 抹不够（逐帧最大转角 46.7°）、29 抹过头（出现掉头 179.6°），19 是实测选的。
BLEND = 19

# 末端线速度上限（m/s）。真机指尖速度中位 8.7cm/s、p95 22.5~23.7、最大 32.6~33.6，
# 取 0.12 明显在真机包线内，同时把"冲一下再急刹"的峰值削掉；搬运段慢下来还顺带压住
# 物体在爪内被甩歪。
TIP_V_MAX = 0.12

# 取物落地曲线的结点：(沿来向后退 m, 高于物体中心 m, 该段速度占上限的比例)。
#
# ★ 为什么不再"先到正上方再竖直降"：实测那样取物段末端走 25.7cm 而直线只有 14cm
#   （绕路 1.83 倍），多出来的全是纯升降 —— 眼睛看到的就是"手臂抬老高再放下来"。
#   人操作是把升降和平移混着走一条斜线的。保留最后 5cm 垂直，是为了守住"竖直下来
#   绝不蹭到物体侧面"这条安全性。
# ★ 速度比例逐段递减：靠近物体时慢下来是人的节奏，也是不把物体撞飞的前提。
#
# ★ 最后两段的比例从 0.35 / 0.20 提到 0.50 / 0.35。那两个是老仓 v5 传下来的保守值，
#   目审读出「下降这个过程看着比正常的慢一点」。查证：从开集到抓住那一刻，
#   仿真中位 4.95s、真机 300 集中位 7.17s —— **绝对时长比真机快 2.2 秒**，
#   所以「慢」不是总时长的问题，是**取物段末端速度中位只有 2.87cm/s**（真机整集 8.79）：
#   前面赶路那一截快、最后 7cm 磨，快慢反差让那一段显得拖。提这两个比例正是提那一截。
#
# ★ **「横向滑入」在本仓的爪几何下已被实测否证，不要再试。** 老仓 v5 的做法是：前几个结点
#   整体沿爪局部 y 让开 45mm（物体在爪外、爪腔是空的），最后一个结点收回偏移、两指沿 y
#   平移滑到物体两侧，全程没有任何面从物体上方扫下来。照搬过来两个方向都**更差**
#   （同一批 10 个种子，判据「抓取前物体被推多少」）：
#     side +45mm → 被推 中位 16.82 最大 32.38mm，可规划 6/10，成功 3/6
#     side −45mm → 被推 中位 30.99 最大 31.28mm，可规划 7/10，成功 2/7
#     side −30mm → 被推 中位 15.20 最大 24.36mm，可规划 9/10，成功 9/9
#     不滑入      → 被推 中位  3.93 最大 15.24mm，可规划 10/10，成功 10/10
#   ⇒ 滑入时反而有指头从物体身上扫过去，说明这里爪子的"C 形口"并不在局部 y 方向。
#     v5 的这条依赖它自己那版爪几何，不可移植。
APPROACH_KNOTS = ((0.055, 0.090, 1.00),
                  (0.022, 0.068, 0.60),
                  (0.000, 0.050, 0.50),
                  (0.000, 0.000, 0.35))

# 直达路径会撞到物体时改走的中转位形（臂五关节，度）：手臂立起来，从物体上方绕过去。
VIA_POSE_DEG = np.array([0.0, 0.0, 0.0, 90.0, 0.0])
# 判"会撞到物体"时，末端各点与物体外廓的最小净空（m）。
CLEARANCE_M = 0.002

# 搬运时物体中心抬到多高（m，物体中心之上）。抬起与平移融成一段，这只是那条弧的顶点。
LIFT_H = 0.08
# 料箱内底的高度（m）。箱底贴着台面放，这是环境自己的常量。
BIN_FLOOR_Z = 0.0025
# 松手时物体底面离箱底留多少（m）。用户 2026-09-03 明确放开：「你可以在更高一点的地方把
# 物体放下来，让他落入盒子就可以」「垂直下落 10cm 左右是可以的」⇒ 不必把爪伸进箱内。
# 取 6cm 而不是 10cm：搬运弧顶在物体上方 8cm，松手点比它略低，弧线才是往下走的；
# 取 10cm 会让末端在箱口上方再抬一次，那就又变成"举起来再放"。
DROP_CLEARANCE = 0.06

# 夹爪每帧最多合/张多少行程百分比。
#
# ★ 早先是**一帧**从 36.4% 砸到 0%。那一下把刚体物体猛地夹住，画面上就是「被吸上来」。
#
# ★ 取值必须落在**要混进来的那两份真机数据**的包线内，不能用跨 9 个任务的宽区间。
#   旧值 12.0 的依据写的是「真机逐帧夹爪变化 p95 为 9.8~16.2%」，而对 `pick_up_a_cube`
#   300 集与 `pick_up_a_can` 250 集逐集实测 `|Δaction[5]|` 的最大值：
#     · cube 中位 6.43% / p95 9.21% / **最大 10.87%**
#     · can  中位 4.65% / p95 7.03% / **最大  7.97%**
#   12.0 会让合拢那一帧走 46.4/4 = 11.60%，**比 550 集真机里出现过的任何一帧都快**。
#   策略输出的是 action —— 数据里出现真人从没做出过的夹爪跳变，就是分布外的取值。
#   取 7.0（两份任务 p95 里较紧的那个），cube40 合拢变成 7 帧、6.63%/帧，两份都落在 p95 内。
GRIP_RATE_PCT = 7.0
# 「捏到底」保持几帧。堵转要几帧才建立起来（实测 40 步内就稳），给 12 帧够抓实又不拖长集长。
CLOSE_FRAMES = 12
# 松手后停几帧再回家，让物体落稳。
RELEASE_FRAMES = 6
# 每集**首尾各有一段静止**，长度按真机的经验分布逐集采样
# （分位表在 `recipe.REAL_LEAD_STILL_FRAMES` / `REAL_TAIL_STILL_FRAMES`）。
#
# ★ 开头那一段早先**完全没有**（恒 1 帧），而真机 cube 中位 41 帧、p10 13 / p90 70 ——
#   一个 δ 分布对一条宽分布，只拿到数据就能认出来。它同时是集长离散度的主要来源：
#   真机集长 σ 90.6，仿真原本只有 19.6，支撑集都不重叠。
# ★ 结尾那一段早先是定值 18，实测落在 13~17 之间；真机 cube 中位 32、p10 21 / p90 47。
# ★ 尾部静止还有一个机械作用：`lerobot-record` 按墙钟停表，尾部偶尔少几帧；
#   少掉的若是**还在运动**的帧，保真核对会判「差的不只是静止段」。留一段静止之后，
#   少掉的那几帧必然落在静止段里。

# 抓取前悬停高度（m，物体中心之上）—— 只用来给 `solve_aligned_grasp` 一个同俯仰的
# 预抓取位姿做对齐迭代，不再作为轨迹上的一个"必经点"。
APPROACH_H = 0.06

# 搬运与松手那两个结点允许的俯仰范围与扫描步长（度）。
#
# ★ 这里**不能**沿用抓取那张候选表（`recipe.PITCH_CANDIDATES_DEG`，70~95°）。
#   实测 30 个种子里 9 个「放置逆解不收敛」，而把俯仰从 40° 扫到 120° 时**这 9 个
#   全部在 40~45° 上残差 0.0mm 收敛** —— 料箱离基座 35~38cm，手臂伸到那么远时腕部
#   自然就放平了，逼它维持抓取时的 85° 等于把可达域掐掉一大块。
#   抓取那张表窄是有道理的（要从正上方压下去），放置没有这个约束：物体已经在手里，
#   松开就落进箱子。
# ★ 扫描**从抓取俯仰起、按远近排序**，取第一个收敛的 ⇒ 能不转腕就不转腕，
#   真要转也只转到够得着为止。搬运段腕部总行程另有质量门兜着。
CARRY_PITCH_RANGE_DEG = (40.0, 110.0)
CARRY_PITCH_STEP_DEG = 2.5

# 瞄点最大容许残差（毫米）：解出来的抓取位姿要真把夹持口袋送到物体中心。
#
# ★ `solve_aligned_grasp` 的 `ok` 也**不代表逆解收敛到目标点**。实账：某一集的瞄点残差
#   45.21mm（y 方向偏 40.16mm）、对齐残余 14.57°，却一路通过、还被判成「成功」——
#   那是蒙进箱口的。正常集的残差是 0.01~0.09mm，所以 1mm 的门槛离两边都很远。
MAX_AIM_MM = 1.0

# 腕滚的容许区间按**真机对标任务**声明在 `recipe.REAL_WROLL_RANGE_DEG` 里
# （cube [−59.1, +96.2]，中位 +1.4 / p10 −19.4 / p90 +34.5；can [−39.16, +116.53]）。
# 落在外面的场景弃掉 —— 判据与本产线其它每一处一样：不产出真人没做出过的取值。
#
# ★ 这里原先是 `MAX_SAFE_WROLL_DEG = 25.0`，一道**打滑安全门**，理由是"两指尖沿插入
#   方向错开 7.6mm、合拢形成力偶，正腕滚过 +25° 后力偶与重力同向把物体推出去"，
#   证据是 n=40 的成败分界（成功组 [−63.95, +24.94]、失败组中位 +40.09）。
#   那批数据出自**旧夹爪口径**（力矩上限 100 / 2.0 ＋ 合拢阶梯）。力矩上限降到 0.20 后
#   按同一把尺子重测（`probe_grasp_direction.py`，cube40，偏好值扫 −40~+40、
#   安全门关掉、去重 31 集）：
#       [−90,−40) 6/6   [−40,−25) 3/3   [−25,−10) 3/3   [−10,0) 2/2
#       [0,+10) 1/1   [+10,+25) 3/3   [+25,+40) 3/3   [+40,+90) 10/10
#   **全区间 31/31**，+25° 那道坎在新口径下不存在了 ⇒ 换成域约束。
#   顺带：这道门与 `PREFERRED_WROLL = 0` 本来就打架 —— 偏好 0 时腕滚 p90 是 +27.6°、
#   最大 +34.9°，旧门会把那些场景整个丢掉（可规划 18/20 → 11/16）。

# ── 落盘质量门 ────────────────────────────────────────────────────────────
# 判的是**观感**，量的是用户逐条抱怨的那几件事，全部在末端（口袋点）上逐帧算，不抽样：
# 每 3 帧抽样会把一帧之内的转向抹平，而那正是「相邻帧还是抖动」指的那一类。
#
# 末端加速度矢量上限（cm/s²，含切向 + 法向 ⇒ 既拦"忽快忽慢"也拦"忽然转向"）。
# 用它而不是"逐帧转角"：转角在低速时天然会大，按转角设门会误杀。真机末端加速度
# p95 174、最大 315~361；本产线典型 p95 45 —— 门收到 120 是为了筛掉工作区边缘那些
# 位形差的集（同样的末端速度要用更大的关节速度，画面上就是"抖"）。
MAX_TIP_ACC_CMS2 = 120.0
# 最长一次停顿（s），只在运动段上算。真机中位 1.55s。
MAX_DEAD_S = 1.2
# 爬行兜底：运动段末端速度中位（cm/s）。真机 8.7。
MIN_TIP_SPEED_CMS = 2.0
# 搬运段腕部（wflex + wroll）总行程上限（度）—— 直接量「无意义的旋转」。
#
# 60° 落在真机的 **p82**：对 `pick_up_a_cube` 146 集用同一个算法实测，搬运段腕部总行程
# 中位 34.0° / p90 69.4° / p95 76.2° / 最大 168.9°。**这道门比真机严** —— 有意为之：
# 「末端依然有很多无意义的旋转」是被目审点名否掉的，宁可少收 2~3% 的场景。
MAX_WRIST_TRAVEL_DEG = 60.0
# 搬运段末端路径的迂曲度上限（实际弧长 / 直线距离）。
MAX_TORTUOSITY = 2.0
# 判"这一帧末端在动"的速度门槛（m/s）。
MOVING_EPS = 0.005


def _seg_frames(a, b, speed, step_max):
    """两个位形之间按逐关节上限匀速走完要多少帧。

    Args:
        a: `(6,)` 起点（弧度）。
        b: `(6,)` 终点（弧度）。
        speed: 占上限的比例，`(0, 1]`。
        step_max: `(6,)` 逐关节每帧上限（弧度）。

    Returns:
        帧数，至少 2。
    """
    delta = np.abs(np.asarray(b, float) - np.asarray(a, float))
    return int(max(2, np.ceil((delta / (np.asarray(step_max, float) * speed)).max())))


def _tip_track(traj_rad, kin, base, pocket):
    """逐帧的夹持口袋在世界系的位置。

    Args:
        traj_rad: `(帧数, 6)` 轨迹（弧度）。
        kin: `ArmKinematics`。
        base: `BaseFrame`。
        pocket: `(3,)` 口袋在夹爪局部系的偏移（m）。

    Returns:
        `(帧数, 3)` 世界坐标（m）。

    量的是**两指之间那一点**而不是腕部原点：用户看的是夹爪走得顺不顺，
    「因为人操作的也是末端」。
    """
    out = np.empty((len(traj_rad), 3))
    for i, row in enumerate(traj_rad):
        p_local, rot = kin.ee_pose_local(row[:5])
        out[i] = base.to_world(p_local + rot @ np.asarray(pocket, float))
    return out


def _retime_tip(traj_rad, kin, base, pocket, v_max):
    """按末端笛卡尔速度上限给整段轨迹重新计时。

    Args:
        traj_rad: `(帧数, 6)` 轨迹（弧度）。
        kin: `ArmKinematics`。
        base: `BaseFrame`。
        pocket: `(3,)` 口袋偏移（m）。
        v_max: 末端速度上限（m/帧）。

    Returns:
        `(新帧数, 6)` 重新计时后的轨迹。

    ★ 关节匀速 ≠ 末端匀速：同样的关节角速度，手臂伸出去之后末端线速度大得多。
      实测取物段末端从 11cm/s 一路冲到 29cm/s 再急刹进下降段 —— 目审读到的
      「2 秒的时候有个飞快的速度」就是这个几何增益，不是指令变快了。

    做法：算出每帧末端走了多远，超速的帧按比例把时间轴拉长，再插值回整数帧。
    **只会变慢不会变快** ⇒ 逐关节上限自动仍然成立；路径一点不变，只改走它的快慢。
    """
    pts = _tip_track(traj_rad, kin, base, pocket)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    clock = np.r_[0.0, np.cumsum(np.maximum(1.0, step / max(v_max, 1e-9)))]
    count = max(2, int(np.ceil(clock[-1])))
    ticks = np.linspace(0.0, clock[-1], count + 1)
    return np.stack([np.interp(ticks, clock, traj_rad[:, j])
                     for j in range(traj_rad.shape[1])], axis=1)


def _plan_path(knots, speeds, kin, base, pocket, step_max):
    """把一串关节结点变成一段速度处处连续的指令序列。

    Args:
        knots: `[(6,)]` 关节结点（弧度），至少两个。
        speeds: `[float]` 每段占上限的比例，长度 = 结点数 − 1。
        kin: `ArmKinematics`。
        base: `BaseFrame`。
        pocket: `(3,)` 口袋偏移（m）。
        step_max: `(6,)` 逐关节每帧上限（弧度）。

    Returns:
        `(帧数, 6)` 指令序列（弧度），不含起点、含终点。

    段内匀速 → 拐点处速度会突变 → 用 Hann 窗滑动平均抹圆。首尾各垫 `BLEND` 帧常量，
    保证起止都精确落在结点上且速度为零（不垫的话滑动平均会把端点也平均掉，到不了位）。

    ★ 垫帧必须再剪掉。它们是为了让滑动平均能精确落在结点上，但放出去就是**画面里的
      一次停顿** —— 每段两头各约 9 帧，几段加起来一秒多站着不动。剪掉之后首帧与结点
      相差不到 0.02°（按定义那几帧就没动），不产生跳变。
    """
    knots = [np.asarray(k, float) for k in knots]
    raw = [np.repeat(knots[0][None], BLEND, axis=0)]
    # `strict=True`：速度表与段数对不上时当场抛。不加的话 zip 会**静默少走一段** ——
    # 轨迹照样生成、照样过门，只是少了一个结点，而这在画面上看不出来。
    for start, end, speed in zip(knots[:-1], knots[1:], speeds, strict=True):
        count = _seg_frames(start, end, speed, step_max)
        ratios = np.linspace(0.0, 1.0, count + 1)[1:, None]
        raw.append(start[None] + (end - start)[None] * ratios)
    raw.append(np.repeat(knots[-1][None], BLEND, axis=0))
    raw = np.concatenate(raw, axis=0)

    window = np.hanning(BLEND + 2)[1:-1]
    window = window / window.sum()
    padded = np.concatenate([np.repeat(raw[:1], BLEND, axis=0), raw,
                             np.repeat(raw[-1:], BLEND, axis=0)])
    smooth = np.stack([np.convolve(padded[:, j], window, mode="same")
                       for j in range(raw.shape[1])], axis=1)[BLEND:-BLEND]

    step = np.abs(np.diff(smooth, axis=0)).max(axis=1)
    live = np.flatnonzero(step > np.radians(0.02))
    if len(live):
        smooth = smooth[live[0]:live[-1] + 2]
    return _retime_tip(smooth, kin, base, pocket, TIP_V_MAX / FPS)[1:]


def _with_grip(qpos, grip_pct):
    """把一个位形的夹爪那一维换成给定的行程百分比。

    Args:
        qpos: `(6,)` 关节角（弧度）。
        grip_pct: 夹爪行程百分比，**真机口径**（`so101.grip_rad_from_pct` 那把尺子）。

    Returns:
        `(6,)` 新的关节角（弧度）。

    ★ 这里曾经写死 `low + pct/100*(high-low)`，也就是 URDF 区间那把尺子。夹爪标度
      改成标定到真机之后，同一个数字在两把尺子下差一倍：松手目标 24.83% 被解成物理
      行程的 24.8%（应为 46.4%），爪子根本没张开，物体一路被夹回 home ——
      判据报的是「没落进箱口 / 始终没落回台面」，成功率 10/10 → 1/3。
    """
    from so101_sim.robots.so101_base.so101 import grip_rad_from_pct

    out = np.asarray(qpos, float).copy()
    out[5] = grip_rad_from_pct(grip_pct)
    return out


def _grip_ramp(cursor, grip_pct):
    """只动夹爪的一段：手臂不动，开度按真机速率渐变。

    Args:
        cursor: `(6,)` 当前位形（弧度）。
        grip_pct: 目标行程百分比，**真机口径**。

    Returns:
        `(帧数, 6)`，不含起点、含终点。
    """
    from so101_sim.robots.so101_base.so101 import grip_pct_from_rad

    target = _with_grip(cursor, grip_pct)
    # 渐变速率也要按真机那把尺子算帧数，否则实际速率会随标度整体缩放。
    span = abs(grip_pct_from_rad(target[5]) - grip_pct_from_rad(cursor[5]))
    count = max(1, int(np.ceil(span / GRIP_RATE_PCT)))
    ratios = np.arange(1, count + 1)[:, None] / count
    return cursor + (target - cursor) * ratios


def _path_clear(kin, base, q_from, q_to, item_xy, item_half, samples=40):
    """这条关节空间直线上，夹爪会不会蹭到物体。

    Args:
        kin: `ArmKinematics`。
        base: `BaseFrame`。
        q_from: `(6,)` 起点（弧度）。
        q_to: `(6,)` 终点（弧度）。
        item_xy: 物体中心 xy（m）。
        item_half: 物体半宽（m）。
        samples: 沿路采样点数。

    Returns:
        `bool`；两指尖与掌心任意一点侵入物体外廓（再留 `CLEARANCE_M`）就给 `False`。
    """
    box = np.array([item_xy[0], item_xy[1], item_half], float)
    for ratio in np.linspace(0.0, 1.0, samples):
        q = np.asarray(q_from, float) + (np.asarray(q_to, float)
                                         - np.asarray(q_from, float)) * ratio
        for point in kin.tips_local(q).values():
            gap = np.abs(base.to_world(point) - box) - item_half
            if float(gap.max()) < CLEARANCE_M:
                return False
    return True


def quality_faults(traj_rad, carry_slice, moving_mask, kin, base, pocket):
    """轨迹观感有没有毛病 —— 逐帧在末端上量，不抽样。

    Args:
        traj_rad: `(帧数, 6)` 整条轨迹（弧度）。
        carry_slice: 搬运段在 `traj_rad` 里的下标切片。
        moving_mask: `(帧数,)` 布尔，True 表示这一帧手臂**本来就该在动**
            （取物 / 搬运 / 回家），False 是合拢、保持、松手、落稳那几段。
        kin: `ArmKinematics`。
        base: `BaseFrame`。
        pocket: `(3,)` 口袋偏移（m）。

    Returns:
        `[str]` 毛病清单，空表示这一集观感合格。

    ★ 停顿**只在该动的帧上算**（`moving_mask`）。合拢 + 保持 + 松手 + 落稳一共二十几帧
      手臂本来就不动，把它们算进去会把好集全枪毙；而反过来"把它们的长度当成允许的静止
      额度"同样不对 —— 那等于给整条轨迹放宽了近一秒，一集若在搬运途中真停了 1.7 秒也
      照样过门。按掩码分开算，两个问题一起没有。
    """
    tip = _tip_track(traj_rad, kin, base, pocket)
    vel = np.diff(tip, axis=0) * FPS
    speed = np.linalg.norm(vel, axis=1)
    acc = np.linalg.norm(np.diff(vel, axis=0), axis=1) * FPS * 100.0

    faults = []
    if len(acc) and float(acc.max()) > MAX_TIP_ACC_CMS2:
        faults.append(f"末端加速度突变 {acc.max():.0f}cm/s²")
    moving = speed[speed > MOVING_EPS]
    median_cms = float(np.median(moving)) * 100.0 if len(moving) else 0.0
    if median_cms < MIN_TIP_SPEED_CMS:
        faults.append(f"末端太慢 {median_cms:.1f}cm/s")

    runs, run = [], 0
    for value, should_move in zip(speed, moving_mask[:len(speed)], strict=True):
        if should_move and value < MOVING_EPS:
            run += 1
        else:
            runs.append(run)
            run = 0
    runs.append(run)
    dead_s = max(runs) / FPS
    if dead_s > MAX_DEAD_S:
        faults.append(f"停顿 {dead_s:.1f}s")

    carry = traj_rad[carry_slice]
    if len(carry) >= 3:
        wrist = float(np.abs(np.diff(np.degrees(carry[:, 3:5]), axis=0)).sum())
        if wrist > MAX_WRIST_TRAVEL_DEG:
            faults.append(f"搬运时腕部乱转 {wrist:.0f}°")
        path = tip[carry_slice]
        arc = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        straight = float(np.linalg.norm(path[-1] - path[0]))
        tortuosity = arc / max(straight, 1e-6)
        if tortuosity > MAX_TORTUOSITY:
            faults.append(f"搬运路径迂曲 {tortuosity:.1f} 倍")
    return faults


def plan(scene, item_xy, item_yaw, bin_xy, home_qpos, base_p, base_q, urdf_path,
         seed=0):
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
        seed: 这一集的种子。只用来采两段静止的长度 —— 轨迹本身仍是确定的几何解，
            同一个种子永远给出逐位相同的一集。

    Returns:
        `(动作 (帧数, 6), 弃用理由)`。成功时理由为 `None`；弃用时动作为 `None`、
        理由是那道门的名字。

    ★ 解不出来就**如实弃掉**，不拿一个没收敛的位姿往下跑。理由必须带出来 —— 只知道
      "跳过了"而不知道是哪道门挡的，就无从判断该放宽哪一道；而放宽错的那道会直接把
      成功率拉低。
    """
    from so101_sim.robots.so101_base.so101 import (
        grip_pct_from_rad,
        grip_rad_from_pct,
        gripper_limit_rad,
    )

    spec = recipe.SCENES[scene]
    low, high = gripper_limit_rad()
    kin = ArmKinematics(urdf_path)
    frame = BaseFrame(base_p, base_q)
    plan_range = plan_joint_range_deg(scene)
    # 落盘量化用的夹爪量子，唯一声明处见 `recipe.grip_step_pct`（一个数，不按场景查表）。
    grip_step = recipe.grip_step_pct()
    limits_lo = np.r_[np.radians(plan_range[:, 0]), low]
    limits_hi = np.r_[np.radians(plan_range[:, 1]), high]
    # 夹爪那一维的步长上限要换回弧度：速率是真机口径的百分点/帧。
    step_max = np.r_[np.radians(STEP_MAX_DEG),
                     abs(grip_rad_from_pct(GRIP_RATE_PCT) - grip_rad_from_pct(0.0))]

    approach_rad = grip_rad_from_pct(spec["open_approach_pct"])
    # 判"跨得过"要用**下降段**的开度 —— 两指是在那个宽度上围住物体的。
    descent_rad = grip_rad_from_pct(spec["open_descent_pct"])
    # 瞄点 = **下降开度下两指尖的中点**，由运动学现算，不是一组手调常量。
    #
    # ★ 这条曾经被否掉过，是在**旧夹爪口径**下（力矩上限 100 + 合拢阶梯）：当时换上中点
    #   成功率 8/10 → 5/10、解不出的槽位 3 → 8。此后夹爪那一侧改了两次（捏到底取代阶梯、
    #   力矩上限 100 → 2.0 → 0.20），前提已经不成立。在新口径下按同一把尺子重测
    #   （cube40，40 个种子，力矩 0.20）：
    #     实测口袋 (−18.63, −12.17)  可规划 35，收 34　97.1%
    #     中间点   (−25.00,  −8.00)  可规划 38，收 37　97.4%
    #     两指中点 (−31.94,  −4.60)  可规划 37，收 37　**100%**
    #   中点那组**既解得出更多场景、又一集没失败**。⇒ 旧结论已被推翻（bd 里已 supersede）。
    #
    # ★ 用推导而不是再放一组常量：中点只由该场景的下降开度决定，跟着 `open_descent_pct`
    #   自动走。手维护的 `pocket_x` / `pocket_z` 与开度是两个会各自漂的数，
    #   一旦有人只改了开度，那对口袋就悄悄指到别处去了。
    pocket = aim_point(kin, scene)
    home = np.asarray(home_qpos, float)
    rng = np.random.default_rng(seed)
    lead_still = _still_frames(recipe.REAL_LEAD_STILL_FRAMES[spec["real_task"]], rng)
    tail_still = _still_frames(recipe.REAL_TAIL_STILL_FRAMES[spec["real_task"]], rng)
    q_above, q_grasp, sum_pitch, wroll, ok = solve_aligned_grasp(
        kin, frame, home, item_xy, spec["item_half"],
        APPROACH_H, approach_rad, limits_lo, limits_hi, pocket=pocket,
        item_yaw=item_yaw, symmetry=spec["item_symmetry"])
    if not ok:
        return None, "抓取逆解不收敛"
    if not span_fits(kin, q_grasp, item_yaw, spec["item_half"], descent_rad):
        return None, "两指跨不过物体（宽度/cosθ 超出该开度的指尖间距）"
    # 腕滚跑出真机见过的范围就弃掉这个场景 —— 产一集真人做不出来的姿态没有意义。
    wroll_deg = float(np.degrees(wroll))
    lo_w, hi_w = recipe.REAL_WROLL_RANGE_DEG[spec["real_task"]]
    if not lo_w <= wroll_deg <= hi_w:
        return None, f"腕滚解出了真机的范围（{wroll_deg:+.1f}° ∉ [{lo_w}, {hi_w}]）"
    # 两个夹持面与方块的那对面平行吗。**这一项与"两指连线的水平投影对齐"是两回事**：
    # 投影把"夹持轴自己翘起来"那一部分抹掉了，俯仰越浅抹掉得越多。实账：某集俯仰落到
    # 73.2°（同批其余九集 85.0~86.5°），投影残余 1.97° 全绿，三维夹角却是 10.7°，
    # 腕部画面里方块角对着镜头、两个面都看得见 —— 目审一眼读出「这个明显不对」。
    face_deg = jaw_face_angle_deg(kin, frame, q_grasp, item_yaw,
                                  spec["item_symmetry"])
    if face_deg > MAX_JAW_FACE_DEG:
        return None, f"夹持面与物体的面不平行（{face_deg:.1f}° > {MAX_JAW_FACE_DEG}°）"
    # 逆解真把口袋送到物体中心了吗 —— 自己再量一次，别信 `ok`。
    p_local, rot_local = kin.ee_pose_local(q_grasp[:5])
    aim = frame.to_world(p_local + rot_local @ pocket)
    aim_mm = float(np.linalg.norm(aim - np.array([*item_xy, spec["item_half"]]))) * 1000.0
    if aim_mm > MAX_AIM_MM:
        return None, f"瞄点残差过大（{aim_mm:.2f}mm > {MAX_AIM_MM}mm）"

    # ── 取物：一条落地曲线 ────────────────────────────────────────────────
    # 来向 = 从 home 处的口袋水平位置指向物体。落地曲线沿这个方向后退着抬高，
    # 于是"下降"与"平移"是重叠着走完的，不是两段直角。
    p_home, rot_home = kin.ee_pose_local(home[:5])
    toward = np.asarray(item_xy, float) - frame.to_world(p_home + rot_home @ pocket)[:2]
    toward = toward / max(float(np.linalg.norm(toward)), 1e-9)

    knots, speeds = [], []
    seed = q_above
    for index, (back, height, speed) in enumerate(APPROACH_KNOTS):
        # 进场张到进场量，**在指尖降到物体顶面之前收回到下降量**：最后两个结点都用
        # 下降开度 —— 倒数第二点离物体中心 5cm、指尖还在物体顶面之上；若拖到最后一点
        # 才收，动指要在物体旁边扫一个大弧，会把它打翻。
        grip = descent_rad if index >= len(APPROACH_KNOTS) - 2 else approach_rad
        target = np.array([item_xy[0] - toward[0] * back,
                           item_xy[1] - toward[1] * back,
                           spec["item_half"] + height])
        q_knot, _, converged = solve_grasp_pose(
            kin, frame, seed, target, grip, limits_lo, limits_hi, sum_pitch,
            pocket=pocket, wroll=wroll)
        if not converged:
            return None, "落地曲线某个结点逆解不收敛"
        knots.append(q_knot)
        speeds.append(speed)
        seed = q_knot

    # ★ 起手那一帧**不许把爪张开**。夹爪在 home 位形下就停在物体附近的空间里，
    #   而张开是动指绕轴甩出去 —— 物体离得近时那一甩直接把它扫飞。实测两集在**第 1、2 帧**
    #   物体就被推了 37.8mm / 43.4mm，而那时手臂还在 home、执行误差只有 0.1°：
    #   规划一点没错，是"开爪"这个动作本身撞的。
    #   改成从 home 的实际开度起、在飞向第一个结点的途中渐渐张开：到结点时已经张好，
    #   而张开发生在离物体 9cm 高、5.5cm 远的空中。人也是这么做的 —— 手在路上才张开。
    #   这样 `_path_clear` 也才量得准：它逐点插值整个六维位形，开爪过程因此在检查之内。
    start = home.copy()
    lead = [start]
    lead_speeds = []
    if not _path_clear(kin, frame, start, knots[0], item_xy, spec["item_half"]):
        via = _with_grip(np.r_[np.radians(VIA_POSE_DEG), 0.0],
                         spec["open_approach_pct"])
        if not (_path_clear(kin, frame, start, via, item_xy, spec["item_half"])
                and _path_clear(kin, frame, via, knots[0], item_xy, spec["item_half"])):
            return None, "起手路径会撞到物体"
        lead.append(via)
        lead_speeds.append(1.0)
    approach = _plan_path(lead + knots, lead_speeds + speeds, kin, frame, pocket, step_max)

    # `moving` 与 `frames` 一一对应，标出这一段手臂**本来就该在动**吗 ——
    # 停顿那道门只在该动的帧上算，不然合拢与松手那二十几帧静止会顶掉整条轨迹的额度。
    # 开头先静止一段 —— 真机每集都有一段人还没动手的时间，见 `REAL_LEAD_STILL_FRAMES`。
    lead = np.repeat(approach[0][None], lead_still, axis=0)
    frames, moving = [lead, approach], [False, True]
    cursor = approach[-1]

    # ── 合拢：捏到底，手臂不动 ──────────────────────────────────────────
    close_pct = recipe.CLOSE_PCT[spec["real_task"]]
    closing = _grip_ramp(cursor, close_pct)
    frames.append(closing)
    moving.append(False)
    cursor = closing[-1]
    frames.append(np.repeat(cursor[None], CLOSE_FRAMES, axis=0))
    moving.append(False)

    # ── 搬运：抬起与平移融成一条弧 ────────────────────────────────────
    # 俯仰要允许换：手里夹着物体之后可达域变了，硬钉抓取时那个俯仰会让料箱上方解不出来。
    # 扫描按「离抓取俯仰多远」排序，取第一个收敛的 ⇒ 能不转腕就不转腕。
    drop_z = BIN_FLOOR_Z + spec["item_half"] + DROP_CLEARANCE
    grid = np.arange(CARRY_PITCH_RANGE_DEG[0],
                     CARRY_PITCH_RANGE_DEG[1] + CARRY_PITCH_STEP_DEG / 2.0,
                     CARRY_PITCH_STEP_DEG)
    ladder = np.radians(grid[np.argsort(np.abs(grid - np.degrees(sum_pitch)))])
    closed_rad = grip_rad_from_pct(close_pct)
    # ★ 搬运与松手仍用**抓取那个瞄点**，不要换成 `recipe.pocket()` 那组"实测握持点"。
    #   换过，更差：同一批 10 集，离箱心中位从 3.6mm 涨到 13.2mm。
    #   原因是那组实测值是在"下降途中物体被推歪"的条件下标出来的，本身被污染；
    #   把下降不碰物体这件事修好之后，物体就待在瞄点上，瞄点才是它的真实位置。
    #   ⇒ 先修因（别碰物体），不要拿一个被污染的测量去补偿它。
    carry_knots, seed = [], cursor
    for target in (np.array([item_xy[0], item_xy[1], spec["item_half"] + LIFT_H]),
                   np.array([bin_xy[0], bin_xy[1], drop_z])):
        solved = None
        for pitch in ladder:
            candidate, _, converged = solve_grasp_pose(
                kin, frame, seed, target, closed_rad,
                limits_lo, limits_hi, pitch, pocket=pocket, wroll=wroll)
            if converged:
                solved = candidate
                break
        if solved is None:
            return None, "放置逆解不收敛"
        carry_knots.append(solved)
        seed = solved
    carry = _plan_path([cursor, *carry_knots], [0.8, 0.8], kin, frame, pocket, step_max)
    carry_slice = slice(sum(len(f) for f in frames), sum(len(f) for f in frames) + len(carry))
    frames.append(carry)
    moving.append(True)
    cursor = carry[-1]

    # ── 松手与回家 ────────────────────────────────────────────────────
    opening = _grip_ramp(cursor, spec["open_approach_pct"])
    frames.append(opening)
    moving.append(False)
    cursor = opening[-1]
    frames.append(np.repeat(cursor[None], RELEASE_FRAMES, axis=0))
    moving.append(False)
    # 回家时把爪**合回 home 的开度**。早先这里给的是 `open_approach_pct`，于是每集末帧
    # 都停在 46.40% 张开（20/20 集，标准差 0），而真机末帧夹爪是合着的：
    # cube 中位 1.896 ± 0.709、can 中位 2.897 ± 0.732。它还连带制造出"集与集之间夹爪
    # 跳 45 点"——真机相邻两集的衔接处逐关节中位差精确为 0（人把臂留在原地接着录）。
    home_grip_pct = float(grip_pct_from_rad(home[5]))
    closing_home = _grip_ramp(cursor, home_grip_pct)
    frames.append(closing_home)
    moving.append(False)
    cursor = closing_home[-1]
    going_home = _plan_path([cursor, _with_grip(home, home_grip_pct)],
                            [1.0], kin, frame, pocket, step_max)
    frames.append(going_home)
    moving.append(True)
    frames.append(np.repeat(going_home[-1][None], tail_still, axis=0))
    moving.append(False)

    qpos = np.vstack([np.asarray(f, float) for f in frames if len(f)])
    moving_mask = np.concatenate([np.full(len(f), m, bool)
                                  for f, m in zip(frames, moving, strict=True) if len(f)])
    faults = quality_faults(qpos, carry_slice, moving_mask, kin, frame, pocket)
    if faults:
        return None, "观感不合格：" + "、".join(faults)

    out = np.degrees(qpos)
    # 臂五关节也夹一次真机包线。逆解本来就在这个区间里解，但**抹圆与重新计时是在
    # 逆解之后做的**，Hann 平滑会让端点越过一点点：实测 shoulder_lift 逆解限位是
    # −110.00° 而落盘值到了 −110.98°。差的这 1° 在物理上无所谓，在"有没有真人
    # 没做出过的取值"这条判据上是硬伤。
    # ★ 夹与量化的**顺序**：先量化到格点、再把格点夹进包线，不能反过来。
    #   反过来（先夹后量化）时，量化会把刚好夹在边界上的值又推出去半格 —— 实测
    #   −103.47 → −103.4725、−58.77 → −58.8132，于是审计里那条"落在真机范围内"
    #   反被这一步的顺序弄红，红的还是个假问题。
    #   `_snap` 保证输出仍在格点上，`np.clip` 之后再 `_snap` 一次把边界拉回**界内**
    #   最近的格点（`np.ceil` / `np.floor` 方向由 clip 的哪一侧决定，见 `_snap_into`）。
    out[:, :5] = _snap_into(out[:, :5], plan_range[:, 0], plan_range[:, 1], ENCODER_STEP_DEG)
    # 夹爪这一列夹进**真机 action 那一列**的实测区间。★ 不能拿 `REAL_GRIP_RANGE_PCT`
    # 来夹：那是 follower（`observation.state`）的范围，两条通道的端点本来就不同
    # （cube action [0.00, 65.08] / state [1.22, 64.25]）。同时它也顺手消掉负零 ——
    # 捏到底那几帧换算回百分比会落在 −1e-14 量级、打印成 `-0.00`，那是"仿真里有
    # 真机没有的取值"的最小实例。
    # 夹爪那一路的量子取真机 **leader**（也就是 action 列）那一侧的实测值（上面已查出
    # `grip_step`）；用臂的计数换算是错的，两边的标定跨度不一样。
    # 夹爪相位实测为 0 —— 真机 leader 与 follower 两列都 100% 落在整数格上，所以不带相位。
    grip_lo, grip_hi = recipe.REAL_GRIP_ACTION_RANGE_PCT[spec["real_task"]]
    out[:, 5] = _snap_into(grip_pct_from_rad(qpos[:, 5])[:, None],
                           np.array([grip_lo]), np.array([grip_hi]), grip_step)[:, 0]
    return out, None
