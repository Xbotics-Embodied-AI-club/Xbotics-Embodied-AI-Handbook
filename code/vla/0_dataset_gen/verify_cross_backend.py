"""跨后端一致门：录出来的每一集，在 CPU 与 GPU 两个物理后端上复跑都要完成任务。

## 为什么这道门必须有

数据集的内容不能依赖跑它的机器。两笔实账：同一批动作在 w2 录、拿到 w1 只过 16/46，
回 w2 又是 46/46；`contact_offset` 用 ManiSkill 默认的 20mm 时，同一强制初始状态下
GPU 把方块抬到 0.0987m 放进箱，CPU 把它侧向弹开 27.6mm、8 集一次也没抬起来。
第二笔已经修在源头（声明成 5mm），但修好一个参数不等于每一集都稳 —— 余量薄的集
该被筛掉，而不是靠继续调参数去救。

## 判据：物体抬起来过，且落在料箱开口内

不看环境给的 `success`：那是四条的合取、含"手臂静止"，收尾停稳前后会反复翻转，
源集自己的末帧也常是 `False`。看**物体去了哪**才是任务本身。

  · 抬起来过：最高 z > 物体半高 + 1cm
  · 落进箱口：末帧位置落在料箱开口内

★ **"落进箱口"必须在料箱自己的坐标系里判。** 料箱每集带一个绕 z 的偏航
  （`place.py:382-383` 给它随机四元数），而开口是 8×10cm 的长方形、两边差 25% ——
  按世界轴比 xy 会在边界附近判反：偏航 6° 时角点误差约 5mm。半尺寸也从环境现读
  （`bin_half_sizes_x` / `_y`），不在这里抄一份数 —— 抄的那份不会随环境改。

★ **源集号 ≠ 数据集内集号。** `--resume` 按录制顺序发号，而分片是隔片取集，
  所以要复跑哪一集的动作得先查 `logs/shard<N>.order` 这本账。不查账会拿别的集的
  动作去跑，结论错而不报错。

★ **不合格集号按「哪一片的第几集」打印，不折算成合并后的编号。** 删必须在合并**之前**
  逐片做：官方 merge 的产物 `meta/episodes` 指针会大面积悬空（实测 243/344），之后
  任何 lerobot 数据编辑工具都不能再用。删除仍走官方
  `lerobot-edit-dataset --operation.type delete_episodes`，输入是各分片自己。

★ 这道门每集要在两个后端各回放一遍，单进程跑一个场景是小时量级，所以支持切片并行：
  第五个参数写 `<第几片>/<共几片>`，各片各跑各的集，输出各自的待删集号，最后合起来。

用法：`python verify_cross_backend.py <场景> <状态目录> <录制输出目录> <分片数> [<第几片>/<共几片>]`
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import recipe
from check_regen import read_order

# 判"抬起来过"的余量（m）：高于静置高度这么多才算真离台。
LIFT_MARGIN = 0.01


def shard_episodes(out_dir, shards):
    """按账本摊平成 `(分片号, 片内集号, 源集号, 录制判词)`。

    Args:
        out_dir: 录制输出目录（`logs/shard<N>.order` 在它下面）。
        shards: 分片数。

    Returns:
        `[(分片号, 片内集号, 源集号, "ok" 或 "reject")]`。

    片内集号就是该集在那份分片数据集里的 `episode_index` —— 删除要按它下手，因为删必须
    在合并之前逐片做。录制时已判作废的那几份也占着片内集号，所以一并列出、由调用方删掉。
    """
    rows = []
    for shard in range(shards):
        for index, (source_ep, verdict) in enumerate(
                read_order(out_dir / "logs" / f"shard{shard}.order")):
            rows.append((shard, index, source_ep, verdict))
    return rows


def recorded_actions(root, episode_index):
    """从重录的分片数据集里取某一集的动作。

    Args:
        root: 分片数据集根目录。
        episode_index: **数据集内**的集号，不是源集号。

    Returns:
        `(帧数, 6)`；这一集不在里面时给 `None`。
    """
    out = []
    for f in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(f, columns=["episode_index", "action"])
        idx = np.asarray(table.column("episode_index").to_pylist())
        hit = np.flatnonzero(idx == episode_index)
        if len(hit):
            out.append(np.stack(table.column("action").to_pylist()).astype(np.float32)[hit])
    return np.concatenate(out) if out else None


def replay_on(backend, spec, state_path, actions):
    """在给定后端上按这串动作跑一遍，报物体轨迹与料箱几何。

    Args:
        backend: `sim_backend` 取值。
        spec: `recipe.SCENES` 里那一项。
        state_path: 强制初始状态 json。
        actions: `(帧数, 6)` 真机口径动作。

    Returns:
        `(最高 z, 末帧 xy, 料箱 xy, 料箱偏航弧度, 料箱开口半尺寸 xy)`。
    """
    from so101_sim.config_lerobot_robot import SO101SimRobotConfig
    from so101_sim.lerobot_robot import JOINT_NAMES, SO101SimRobot

    robot = SO101SimRobot(SO101SimRobotConfig(
        task=spec["env_id"], episode_length=1200,
        initial_state_path=str(state_path), sim_backend=backend))
    robot.connect()
    inner = robot._env._env.unwrapped
    bin_pose = inner.bin.pose
    bin_xy = np.asarray(bin_pose.p[0][:2].cpu(), float)
    quat = np.asarray(bin_pose.q[0].cpu(), float)
    # 偏航只锁在 z 轴上（`place.py:382` 的 `lock_x` / `lock_y`），所以从 (w, z) 就能解出来。
    bin_yaw = 2.0 * np.arctan2(quat[3], quat[0])
    half = np.array([float(inner.bin_half_sizes_x[0]), float(inner.bin_half_sizes_y[0])])

    item = np.empty((len(actions), 3))
    for i, row in enumerate(actions):
        robot.get_observation()
        robot.send_action({f"{n}.pos": float(row[j]) for j, n in enumerate(JOINT_NAMES)})
        item[i] = np.asarray(inner.get_state_dict()["actors"]["item"][0][:3].cpu(), float)
    robot.disconnect()
    return float(item[:, 2].max()), item[-1, :2], bin_xy, bin_yaw, half


def judge(spec, max_z, end_xy, bin_xy, bin_yaw, half):
    """一次运行合不合格：抬起来过，且末帧落在料箱开口内。

    Args:
        spec: `recipe.SCENES` 里那一项。
        max_z: 物体最高高度（m）。
        end_xy: 末帧物体的 xy（m，世界系）。
        bin_xy: 料箱中心的 xy（m，世界系）。
        bin_yaw: 料箱绕 z 的偏航（弧度）。
        half: 料箱开口的半尺寸 xy（m）。

    Returns:
        `(是否合格, 说明)`。
    """
    if max_z <= spec["item_half"] + LIFT_MARGIN:
        return False, f"没抬起来（最高 z {max_z:.4f}m）"
    offset = np.asarray(end_xy, float) - bin_xy
    cos, sin = np.cos(-bin_yaw), np.sin(-bin_yaw)
    local = np.array([cos * offset[0] - sin * offset[1],
                      sin * offset[0] + cos * offset[1]])
    if np.any(np.abs(local) > half):
        return False, (f"没落进箱口（料箱系里偏 {local[0] * 1000:.1f},{local[1] * 1000:.1f}mm，"
                       f"开口半尺寸 {half[0] * 1000:.0f},{half[1] * 1000:.0f}mm）")
    return True, "合格"


def main(argv) -> int:
    if len(argv) not in (4, 5):
        sys.exit(__doc__)
    scene, states_dir, out_dir = argv[0], Path(argv[1]), Path(argv[2])
    spec = recipe.SCENES[scene]

    ledger = shard_episodes(out_dir, int(argv[3]))
    rejected = [r[:3] for r in ledger if r[3] != "ok"]
    rows = [r[:3] for r in ledger if r[3] == "ok"]
    if len(argv) == 5:
        which, count = (int(x) for x in argv[4].split("/"))
        rows = [r for i, r in enumerate(rows) if i % count == which]
        # 作废件只由第 0 片报一次，免得各片各报一遍、合起来重复。
        rejected = rejected if which == 0 else []
        print(f"  本片 {which}/{count}：{len(rows)} 集")
    ok = []
    bad = [(shard, index, source_ep, "录制时已判超差并重录，这一份是作废件")
           for shard, index, source_ep in rejected]
    for shard, index, source_ep in rows:
        actions = recorded_actions(out_dir / f"shard{shard}", index)
        if actions is None:
            bad.append((shard, index, source_ep, "重录数据里没有这一集"))
            continue
        state_path = states_dir / f"ep{source_ep}.json"
        verdicts = {}
        # ★顺序不能反：sapien 的 `physx.enable_gpu()` 要求在任何其它 PhysX 代码之前调用，
        #   先建 CPU 场景再建 GPU 环境会直接抛
        #   `GPU PhysX can only be enabled once before any other code involving PhysX`。
        for backend in ("gpu", "physx_cpu"):
            verdicts[backend] = judge(spec, *replay_on(backend, spec, state_path, actions))
        if all(v[0] for v in verdicts.values()):
            ok.append((shard, index))
        else:
            bad.append((shard, index, source_ep,
                        "；".join(f"{b}: {v[1]}" for b, v in verdicts.items() if not v[0])))
        print(f"  shard{shard} 第{index}集（源集 ep{source_ep}）: "
              + "  ".join(f"{b}={'过' if v[0] else v[1]}" for b, v in verdicts.items()))

    print(f"\n  {scene}: 两后端都过 {len(ok)}/{len(rows)}")
    if bad:
        print("  不合格（编号是**片内**集号，删要在合并之前逐片做）：")
        for shard, index, source_ep, why in bad[:10]:
            print(f"    shard{shard} 第{index}集（源集 ep{source_ep}）: {why}")
        if len(bad) > 10:
            print(f"    …… 共 {len(bad)} 集")
        per_shard = {}
        for shard, index, _, _ in bad:
            per_shard.setdefault(shard, []).append(index)
        for shard in sorted(per_shard):
            print(f"  待删片内集号 shard{shard}："
                  f"[{','.join(str(i) for i in sorted(per_shard[shard]))}]")
    print("CROSS_BACKEND_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
