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
| `pocket_x` | −(固定指内侧面 2mm + 物体半宽) |
| `pocket_z` | 指尖不能低于台面：最低点 ≈ 物体中心 − 6.3cm + z |
| `open_approach_deg` / `open_descent_deg` | **按真机 57 条实测的 p90 取**，不按物体尺寸算死 |
| `close_ladder_deg` | 从"刚接触"往紧排。阶梯而不是一步合拢，让方块被两侧同时夹住 |

★张开角有两个坑，都实际付过代价：

  · **不能张到夹爪限位上限**。曾把用户说的"靠近时要张开一点"做成 100°（限位上限），
    那不是真机的样子 —— 真机进场中位 36.2°、p90 42°。
  · **不能贴着物体尺寸算死**。原来按单边净空 0.6mm 算，仿真里够用，但真机位置/尺寸
    一有偏差余量就没了，手指会蹭到物体把它推走。放宽到真机 p90 后咬合反而更好
    （38.1 → 40.6mm），歪斜不变。

`POCKET_Y` 与俯仰 / wroll 三个场景共用：它们标定自真机 300 集的 pinch 帧
（lift+elbow+wflex 中位 79.96°、wrist_roll 中位 1.01°），与物体尺寸无关。

## 单批 64 槽位的实测收成（歪斜门 6°）

| 场景 | 收成 | 咬合 | 占物体高度 | 歪斜 |
|---|---|---|---|---|
| cube40 | 28~32 | 39.0mm | 98% | 2.5° |
| cube20 | 49 | 17.9mm | 90% | 0.6° |
| cylinder40 | 42 | 37.7mm | 94% | 3.4° |

圆柱直接沿用方块 40mm 的横向与合拢参数（跨距相同）；`jaw_yaw_error` 那套"对齐方块侧面"
对圆柱无意义但无害 —— 旋转对称，任何朝向都行，立着的圆柱也不会滚。
"""

import numpy as np

# 夹持口袋的 y 分量（m）：三个场景共用，标定自策略成功抓取的中位数。
POCKET_Y = 0.0020
# 俯仰锁定值（lift+elbow+wflex 之和）与腕滚，度。取自真机 300 集 pinch 帧的中位数。
SUM_GRASP_DEG = 80.0
WROLL_GRASP_DEG = 0.0
# 俯仰候选：先试真机中位，够不到再往真机 p10–p90（55~98°）里退。
# 单一定值在撒点区远端（x≥0.30 且要求抬高 6cm）解不出来 —— 那不是"抓不到"，
# 是"用这一个俯仰抓不到"，真机本来就不是每次都用同一个俯仰。
PITCH_CANDIDATES_DEG = (85.0, 90.0, 80.0, 95.0, 75.0, 70.0)

# 三个对外分发场景。录制与渲染用同一个环境，一个场景一个环境。
# 每个键的含义见模块 docstring 的那张表。
SCENES = {
    "cube40": {
        "env_id": "SO101PickPlaceCube40-v1",
        "source_name": "pick_place_cube_40mm",
        "task_text": "Pick up a cube and place in the bin",
        "item_half": 0.020,
        "pocket_x": -0.0217,
        "pocket_z": -0.016,
        "open_approach_deg": 42.0,
        "open_descent_deg": 32.0,
        "close_ladder_deg": (13.0, 12.0, 11.0, 10.0),
    },
    "cube20": {
        "env_id": "SO101PickPlaceCube20-v1",
        "source_name": "pick_place_cube_20mm",
        "task_text": "Pick up a small cube and place in the bin",
        "item_half": 0.010,
        "pocket_x": -0.0120,
        "pocket_z": 0.0,
        "open_approach_deg": 42.0,
        "open_descent_deg": 19.0,
        "close_ladder_deg": (2.0, 0.0, -2.0, -4.0, -6.0),
    },
    "cylinder40": {
        "env_id": "SO101PickPlaceCylinder40-v1",
        "source_name": "pick_place_cylinder_40mm",
        "task_text": "Pick up a cylinder and place in the bin",
        "item_half": 0.020,
        "pocket_x": -0.0217,
        "pocket_z": -0.012,
        "open_approach_deg": 42.0,
        "open_descent_deg": 32.0,
        "close_ladder_deg": (13.0, 12.0, 11.0, 10.0),
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
