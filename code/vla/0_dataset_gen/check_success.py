#!/usr/bin/env python
"""任务成没成：把每集录下来的动作复跑一遍，看物体最后在不在料箱开口内。

## 为什么必须单独有这道门

帧数容差、12 路数值一致性、31/38 项结构一致性 —— **一条都不看任务成不成**。
实账：第一批 10 集全部落进帧数容差、12 路无仿真侧缺陷、38/38 结构项通过，
而目审三集**全部"方块没放进去"**（料箱被撞歪、方块卡在箱沿）。
数字全绿而任务全败，是因为没有一条判据看物体去了哪。

## 判据：抬起来过，且末帧落在箱口内

与 `verify_cross_backend.py` 同一套判据（那份还要跨两个后端，这份只判本后端，
用于产出后立即自查）：

  · 抬起来过：最高 z > 物体半高 + `LIFT_MARGIN`
  · 落进箱口：末帧位置在**料箱自己的坐标系**里落在开口半尺寸内 —— 料箱每集带一个绕 z
    的偏航，按世界轴比 xy 会在边界附近判反

不看环境给的 `success`：那是四条的合取、含"手臂静止"，收尾停稳前后会反复翻转。

用法：`python check_success.py <场景> <数据集根> <准备目录>`
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import recipe

# 判"抬起来过"的余量（m）：高于静置高度这么多才算真离台。
LIFT_MARGIN = 0.01


def episode_actions(root):
    """数据集里逐集的动作。

    Args:
        root: 数据集根目录。

    Returns:
        `{集号: (帧数, 6) float32}`。
    """
    per_ep = {}
    for f in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(f, columns=["episode_index", "action"])
        idx = np.asarray(table.column("episode_index").to_pylist())
        act = np.stack(table.column("action").to_pylist()).astype(np.float32)
        for ep in np.unique(idx):
            rows = act[idx == ep]
            prev = per_ep.get(int(ep))
            per_ep[int(ep)] = rows if prev is None else np.vstack([prev, rows])
    return per_ep


def replay(env_id, state_path, actions):
    """按这串动作复跑一遍，报物体轨迹与料箱几何。

    Args:
        env_id: 已注册的环境 id。
        state_path: 强制初始状态 json。
        actions: `(帧数, 6)` 真机口径动作。

    Returns:
        `(最高 z, 末帧 xy, 料箱 xy, 料箱偏航弧度, 开口半尺寸 xy, 逐帧 z)`。

    逐帧 z 是用来定位**什么时候掉的**：只看末位置分不出"搬运途中掉了"与"松手时弹出去了"，
    而这两件事要改的地方完全不同。实账：两轮改了投放高度而失败集的末位置**逐位相同**，
    那就说明物体压根没到投放那一步。
    """
    from so101_sim.config_lerobot_robot import SO101SimRobotConfig
    from so101_sim.lerobot_robot import JOINT_NAMES, SO101SimRobot

    robot = SO101SimRobot(SO101SimRobotConfig(
        task=env_id, episode_length=len(actions) + 10,
        initial_state_path=str(state_path)))
    robot.connect()
    inner = robot._env._env.unwrapped
    bin_xy = np.asarray(inner.bin.pose.p[0][:2].cpu(), float)
    quat = np.asarray(inner.bin.pose.q[0].cpu(), float)
    # 偏航只锁在 z 轴上，所以从 (w, z) 就能解出来。
    bin_yaw = 2.0 * np.arctan2(quat[3], quat[0])
    half = np.array([float(inner.bin_half_sizes_x[0]), float(inner.bin_half_sizes_y[0])])
    item = np.empty((len(actions), 3))
    for i, row in enumerate(actions):
        robot.get_observation()
        robot.send_action({f"{n}.pos": float(row[j]) for j, n in enumerate(JOINT_NAMES)})
        item[i] = np.asarray(inner.get_state_dict()["actors"]["item"][0][:3].cpu(), float)
    robot.disconnect()
    return float(item[:, 2].max()), item[-1, :2], bin_xy, bin_yaw, half, item[:, 2]


def judge(spec, max_z, end_xy, bin_xy, bin_yaw, half, z_track):
    """一集成没成。

    Args:
        spec: `recipe.SCENES` 里那一项。
        max_z: 物体最高高度（m）。
        end_xy: 末帧物体 xy（m，世界系）。
        bin_xy: 料箱中心 xy（m，世界系）。
        bin_yaw: 料箱绕 z 的偏航（弧度）。
        half: 开口半尺寸 xy（m）。
        z_track: 逐帧的物体高度（m）。

    Returns:
        `(是否成功, 说明)`。
    """
    if max_z <= spec["item_half"] + LIFT_MARGIN:
        return False, f"没抬起来（最高 z {max_z:.4f}m）"
    offset = np.asarray(end_xy, float) - bin_xy
    cos, sin = np.cos(-bin_yaw), np.sin(-bin_yaw)
    local = np.array([cos * offset[0] - sin * offset[1],
                      sin * offset[0] + cos * offset[1]])
    if np.any(np.abs(local) > half):
        # 定位什么时候掉的：峰值之后 z 首次回落到"贴台面"的那一帧。
        peak = int(np.argmax(z_track))
        floor = spec["item_half"] + LIFT_MARGIN
        after = np.flatnonzero(z_track[peak:] < floor)
        lost = f"第 {peak + int(after[0])} 帧" if len(after) else "始终没落回台面"
        return False, (f"没落进箱口（料箱系里偏 {local[0] * 1000:.1f},{local[1] * 1000:.1f}mm，"
                       f"开口半尺寸 {half[0] * 1000:.0f},{half[1] * 1000:.0f}mm；"
                       f"最高点在第 {peak} 帧 / 共 {len(z_track)} 帧，掉落于 {lost}）")
    return True, "成功"


def main(argv) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    scene, root, prep = argv[0], Path(argv[1]), Path(argv[2])
    spec = recipe.SCENES[scene]

    actions = episode_actions(root)
    if not actions:
        sys.exit(f"★ {root} 里没有动作 —— 空结果不是「全成功」")
    ok = []
    for ep in sorted(actions):
        state_path = prep / "states" / f"ep{ep}.json"
        if not state_path.is_file():
            sys.exit(f"★ 没有 ep{ep} 的强制初始状态 {state_path}")
        good, why = judge(spec, *replay(spec["env_id"], state_path, actions[ep]))
        ok.append(good)
        print(f"    ep{ep}: {'成功' if good else '★ ' + why}")

    rate = sum(ok) / len(ok)
    print(f"\n  {scene}：{sum(ok)}/{len(ok)} 成功（{rate * 100:.0f}%）")
    print("CHECK_SUCCESS_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
