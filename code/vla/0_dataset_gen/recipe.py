"""三个场景的抓放标定表 —— **全部标定值的唯一声明处**。

★ 这张表里的每个数都是实测标定的，**照抄别的场景会整批 0 收**。
  cube20 沿用 cube40 的口袋 x（−21.7mm）时，20mm 方块的中心被摆进动指面里，
  下降途中就被顶走 —— 收成 7/64，弃因全是"抓前被推 1.1cm"；改成 −12mm 后 49/64。

★ 为什么写成表而不是环境变量：原产线用 `POCKET_X` / `LADDER` 这类环境变量传标定值，
  于是"这批数据是按哪套参数产的"只存在于当时那条命令里。实账：混着旧配置产了 4276 集，
  抽样咬合只有 11.3mm（定稿是 38.4mm），全部作废重产。

## 各量怎么来的

| 量 | 怎么算 |
|---|---|
| `source_name` | 已交付数据集在 `$HF_LEROBOT_HOME/so101_sim/` 下的目录名 |
| `pocket_x` / `pocket_z` | **实测**：捏住那一刻物体在 `gripper_frame_link` 局部系的坐标中位 |
| `open_descent_pct` | **按仿真里物体的宽度反查**：指尖间距 ≥ 物体宽 + 12mm（单边 6mm 净空） |
| `open_approach_pct` | 下降量 + 10 点（真机 `pick_up_a_cube` 的进场−下降差就是 10.4 点） |

★张开量有三个坑，都实际付过代价：

  · **数值必须由仿真里物体的宽度定，不能照搬真机的百分比。** 真机 `pick_up_a_cube` 的
    下降中位是 26.6%，换成关节角 19.3° ⇒ 指尖间距只有约 38.9mm，**比仿真的 40mm 方块还窄**
    —— 直接搬过来手指会在下降途中顶到方块把它推走。真机那个数低，是因为真机画面里的
    方块比 40mm 小（`xb-5xol` 记过"任务目录名与画面里的物体并不一致"）。
    真机数据能给的只是**风格**（不要张到限位、进场比下降宽约 10 点），绝对值得自己算。
  · **单位是行程百分比，不是角度。** 旧版把真机实测的百分比当角度存：`open_approach = 42`
    本意是真机进场的 42%，被当成 42° 下发。换算是 `pct = (deg + 10) / 1.1`，那两个数就是
    夹爪限位 [−10°, +100°]。对证很硬：重新统计 modelscope 9 源 265 集，进场 p90 = 42.52%，
    与那个 42 几乎一模一样；旧注释里"真机进场中位 36.2°、p90 42°"也正好是同一个值的
    两种单位（`-10 + 1.1×42 = 36.2`）。
    ⚠️ **但把它读成百分比之后必须重新按物体宽度定值** —— 32 当 32° 用时间距约 54mm、
    宽于方块，所以旧产线抓得上；改读成 32% 则只有 45mm，26.6% 更只有 38.9mm。
    "单位改对"与"数值仍成立"是两件事。
  · **不能张到夹爪限位上限**。曾做成 100°（限位上限），那不是真机的样子。

★**没有"合拢阶梯"这个东西了。** 真机遥操是人把扳机**捏到底**，抓紧靠物体把两指卡死
  （真机夹爪 action 中位 1.98% / state 中位 14.35%，那 12.4 点落差就是堵转）。
  旧配方用一个阶梯把"捏"写成了"停在某个开度"，两侧语义因此不同 —— 策略输出的是 action，
  同一个值在两份数据里含义不同就是两套映射。现在合拢一律下发 `CLOSE_PCT`，
  由 `so101.GRIPPER_FORCE_LIMIT = 2.0` 保证它表现为堵转而不是把刚体挤飞
  （力矩 10 与 5 都会把 40mm 刚体方块挤出 79~115mm；2.0 时堵在 23~26%、物体只挪 8~12mm）。

`POCKET_Y` 与俯仰 / wroll 三个场景共用，与物体尺寸无关。它们的值重新反解自
**modelscope 9 源 265 集的 15080 个 pinch 帧**（pinch 帧按"指令 <5% 且 state−action >5 点"
认，即人捏到底而两指被物体卡住）：`lift+elbow+wflex` 中位 75.87°（p10~p90 51.1~94.1）、
`wrist_roll` 中位 +2.59°（p10~p90 −9.8~+29.0）—— 现值 80.0 / 0.0 都落在带内。

★ 旧版这里有一张「单批 64 槽位的实测收成」表（cube40 收成 28~32、咬合 39.0mm 等）。
  那批数是在**旧夹爪口径 + 力矩上限 100 + 合拢阶梯**下得的，三个前提全变了 ⇒ 已删，
  不留一张会被下一轮当基线的失效表。重录后按新口径重测。

圆柱沿用方块 40mm 的横向参数（跨距相同）；`jaw_yaw_error` 那套"对齐方块侧面"
对圆柱无意义但无害 —— 旋转对称，任何朝向都行，立着的圆柱也不会滚。
"""

import numpy as np

# 视频编码。真机 9 份实测全是 h264；`lerobot-record` 不给 `--dataset.vcodec` 时录出 av1，
# 而官方合并逐字比 `features`（`video.codec` 在其中）⇒ 不显式声明就合不了。
VIDEO_CODEC = "h264"

# 允许比源集少几帧。
#
# 重录要的是**逐集 1:1 还原源数据集**：窗口给正好 `源帧数/30`，动作逐位等于源集。
# `lerobot-record` 按墙钟停表，只要有循环踩过 33.3ms 尾部就少几帧 —— 真机采集同样会
# 偶尔差一两帧没跟上，所以少几帧是可接受的，**多出来则不接受**：多的全是静止帧，
# 会把末尾静止段撑长（余量 90 帧时实测撑到 68 帧、占集长 15.9%，而真机是 16 帧 / 3.9%）。
FRAME_TOLERANCE = 5

# 一集最多录几次。超差就重录，而不是放宽判据把它收下。
# 上限存在的意义是**不让它挂死**：始终录不进容差的集会耗尽次数、被标记作废，
# 收口时用官方 `delete_episodes` 删掉，并在日志里留下计数 —— 而不是安静地重试到天亮。
MAX_RECORD_RETRIES = 5

# 合拢时下发的夹爪指令（行程百分比）。0 = 捏到底，抓紧靠物体把两指卡死 ——
# 真机遥操就是这个语义。不设成 >0 的某个开度：那会把"捏"写成"停在多宽"。
CLOSE_PCT = 0.0

# 夹持口袋的 y 分量（m）：三个场景共用。实测自 cube40 十集"捏住那一刻"物体在夹爪局部系的
# y 坐标中位 +0.75mm（`calibrate_pocket.py`）。旧值 +2.00mm 是阶梯口径下标的。
POCKET_Y = 0.00075
# 俯仰锁定值（lift+elbow+wflex 之和）与腕滚，度。重新反解自 modelscope 9 源 265 集的
# 15080 个 pinch 帧：之和中位 75.87°（p10~p90 51.1~94.1）、wrist_roll 中位 +2.59°
# （p10~p90 −9.8~+29.0）。现值 80.0 / 0.0 都落在带内，故不动。
# 工具：experiment_main_v1/scripts/EAI-exp-002/review_grasp_calib.py
SUM_GRASP_DEG = 80.0
WROLL_GRASP_DEG = 0.0
# 俯仰候选：先试真机中位，够不到再往真机 p10–p90（55~98°）里退。
# 单一定值在撒点区远端（x≥0.30 且要求抬高 6cm）解不出来 —— 那不是"抓不到"，
# 是"用这一个俯仰抓不到"，真机本来就不是每次都用同一个俯仰。
PITCH_CANDIDATES_DEG = (85.0, 90.0, 80.0, 95.0, 75.0, 70.0)

# 三个对外分发场景。录制与渲染用同一个环境，一个场景一个环境。
# 每个键的含义见模块 docstring 的那张表。
#
# ★ `task_text` **逐字抄真机数据集的 tasks.parquet**，不自己拟措辞。
#   语言指令是模型的输入，措辞不同就是不同的任务：仿真写 "Pick up a cylinder…"
#   而真机写 "Pick up a can…" 时，两份数据永远混不成同一条指令，
#   模型还得额外学一层同义。真机三句实测（`$DATASETS_ROOT/datasets/public/
#   so101-pick-place-tasks/*/meta/tasks.parquet`）：
#     · cube      → "Pick up a cube and place in the bin"
#     · can       → "Pick up a can and place in the bin"
#     · 小方块    → 真机侧**没有**抓放任务，只有 "Stack the smaller cube on the
#                   larger one"（那是堆叠，不是抓放）⇒ cube20 无真机对应指令
#
#   ⇒ cube20 是唯一的例外：没有可抄的真机指令，沿用仿真源数据集自己那句
#     "Pick up a small cube and place in the bin"。它与真机任何一句都不同名，
#     合并后是独立的一条指令，不会与真机 cube 那条混成同一个 task_index。
# `real_task` 是**验收时拿来当尺子的那份真机数据**，不是"混训要配哪一份"。
# 真机 9 份的规格签名实测完全相同（distinct-signatures = 1），所以拿哪一份当尺子都一样；
# 有同名任务的就用同名那份，cube20 真机侧没有抓放小方块的任务，借 cube 那份当尺子。
SCENES = {
    "cube40": {
        "env_id": "SO101PickPlaceCube40-v1",
        "source_name": "pick_place_cube_40mm",
        "task_text": "Pick up a cube and place in the bin",
        "real_task": "pick_up_a_cube_and_place_in_the_bin",
        "item_half": 0.020,
        # 实测（捏到底 + 力矩 2.0 下十集的中位）：x −18.63mm、z −12.17mm。
        # 旧值 −21.70 / −16.00 是"合拢阶梯 + 力矩 100"口径下标的，瞄点差 3~4mm ——
        # 而 40mm 刚体只能被指尖那 37.8mm 的一档夹住，3~4mm 就是抓牢与抓不牢的分界。
        "pocket_x": -0.01863,
        "pocket_z": -0.01217,
        # 40mm 方块 → 需指尖间距 52mm → 张到 +30.0°，即行程 36.4%（实测曲线反查）
        "open_descent_pct": 36.4,
        "open_approach_pct": 46.4,
    },
    "cube20": {
        "env_id": "SO101PickPlaceCube20-v1",
        "source_name": "pick_place_cube_20mm",
        "task_text": "Pick up a small cube and place in the bin",
        "real_task": "pick_up_a_cube_and_place_in_the_bin",
        "item_half": 0.010,
        "pocket_x": -0.0120,
        "pocket_z": 0.0,
        # 20mm 方块 → 需指尖间距 32mm → 张到 +14.0°，即行程 21.8%（实测曲线反查）
        "open_descent_pct": 21.8,
        "open_approach_pct": 31.8,
    },
    "cylinder40": {
        "env_id": "SO101PickPlaceCylinder40-v1",
        "source_name": "pick_place_cylinder_40mm",
        "task_text": "Pick up a can and place in the bin",
        "real_task": "pick_up_a_can_and_place_in_the_bin",
        "item_half": 0.020,
        "pocket_x": -0.0217,
        "pocket_z": -0.012,
        # 直径 40mm 圆柱，跨距与 cube40 相同 ⇒ 同一组值
        "open_descent_pct": 36.4,
        "open_approach_pct": 46.4,
    },
}


def pocket(scene):
    """某场景的夹持口袋（`gripper_frame_link` 局部系，米）。

    Args:
        scene: `SCENES` 的键。

    Returns:
        `(3,)` 数组。
    """
    s = SCENES[scene]
    return np.array([s["pocket_x"], POCKET_Y, s["pocket_z"]], float)
