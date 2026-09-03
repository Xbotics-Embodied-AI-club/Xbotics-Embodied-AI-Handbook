#!/usr/bin/env python
"""为若干集各准备两样东西：强制初始状态 json 与专家规划出来的动作 `.npy`。

## 为什么要先落盘再录

规划要知道物体与料箱在哪，而那是环境复位时撒的点。若规划与录制各自复位一次，两次撒的点
不同，规划出来的轨迹就对不上录制时的场景 —— 手臂会抓向一个空位，而**全程不报错**。
所以这里复位一次、把整个状态字典落成 json，录制时用 `--robot.initial_state_path` 强制成
同一个场景。两边看到的是同一份状态，这件事由文件保证，不靠"应该一样"。

## 一集一个种子

`reset(seed=k)` 决定这一集的撒点。种子写进文件名与 `meta`，于是"这一集的场景是怎么来的"
可以复现 —— 不是"跑一次看看撒到哪"。

## 解不出来就跳过，并把跳过的记下来

抓取位姿解不出来的槽位如实跳过（`expert.plan` 返回 None），并在 `meta/skipped.json` 里
留下种子与原因。**不重试、不放宽判据** —— 那会把"这批数据里有几集是勉强凑的"变成看不出来的事。

用法：`python prepare_episodes.py <场景> <输出目录> <要几集> [<起始种子>]`
      输出：`<输出目录>/states/ep<N>.json` · `plans/ep<N>.npy` · `meta/ep<N>.json`
"""

import json
import sys
from pathlib import Path

import numpy as np
import recipe
from expert import plan

# 等物体落定最多空转几帧，以及判"停住了"的逐帧位移门槛（m）。
# 30 帧 = 1 秒，足够 40mm 刚体在台面上停稳；停不下来的场景会带着残余速度进规划，
# 那种场景本来就不该产。
SETTLE_MAX_STEPS = 30
SETTLE_TOL_M = 1e-5


def sample_scene(env_id, seed):
    """复位一次，取整份状态、物体的 xy 与自旋角、料箱 xy、home 位形、机器人基座位姿。

    Args:
        env_id: 已注册的环境 id。
        seed: 这一集的种子。

    Returns:
        `(状态字典, 物体 xy, 物体自旋角, 料箱 xy, home 位形 (6,), 基座位置, 基座四元数)`；
        物体或料箱在 `SETTLE_MAX_STEPS` 帧内没停稳时给 `None`。

    物体的自旋角必须取出来传给规划：复位时它是随机的，而两指要对齐的是**物体的面**。
    偏航只锁在 z 轴上，所以从 (w, z) 就能解出来。

    用 CPU PhysX：录制本来也走 CPU，而且专家的运动学副本也是 CPU 的 sapien 场景 ——
    全程不碰 GPU 就绕开了 `physx.enable_gpu()` 那条"必须在任何其它 PhysX 代码之前"的限制。
    """
    import gymnasium as gym
    import so101_sim  # noqa: F401  导入即注册
    env = gym.make(env_id, num_envs=1, obs_mode="state", sim_backend="physx_cpu",
                   domain_randomization=False, reconfiguration_freq=1)
    env.reset(seed=seed)
    inner = env.unwrapped

    # 先让物体落定再取状态。复位那一瞬物体还在沉降/滑动：实测 ep1 到抓取帧自己漂了
    # 7.8mm（那时手臂还在离开 home 的路上、没碰到它），而下降开度 52mm、方块 40mm，
    # 两侧净空各 6mm ⇒ 这点漂移足以让指尖撞在角上，表现为两指合到底而中间无物。
    # **落定之后**才拍强制初始状态，回放的起点因此就是规划瞄的那个点，漂移从源头消失。
    #
    # ★ 料箱也要等。实测有一集料箱从第 7 帧起被推动 177.7mm —— 那不是沉降、是被弹开
    #   （生成时与别的几何相交）。早先只判物体，于是带着穿模的料箱被拍进强制初始状态。
    home_action = np.asarray(inner.agent.robot.get_qpos()[0].cpu(), float)

    def _poses():
        return np.stack([np.asarray(inner.item.pose.p[0].cpu(), float),
                         np.asarray(inner.bin.pose.p[0].cpu(), float)])

    last, settled = _poses(), False
    for _ in range(SETTLE_MAX_STEPS):
        env.step(home_action)
        now = _poses()
        if float(np.abs(now - last).max()) < SETTLE_TOL_M:
            settled = True
            break
        last = now
    if not settled:
        env.close()
        return None
    state = {group: {name: [float(v) for v in tensor[0].cpu()]
                     for name, tensor in items.items()}
             for group, items in inner.get_state_dict().items()}
    item_xy = np.asarray(inner.item.pose.p[0, :2].cpu(), float)
    item_q = np.asarray(inner.item.pose.q[0].cpu(), float)
    item_yaw = 2.0 * float(np.arctan2(item_q[3], item_q[0]))
    bin_xy = np.asarray(inner.bin.pose.p[0, :2].cpu(), float)
    home = np.asarray(inner.agent.robot.get_qpos()[0].cpu(), float)
    pose = inner.agent.robot.pose
    base_p = [float(v) for v in pose.p[0].cpu()]
    base_q = [float(v) for v in pose.q[0].cpu()]
    env.close()
    return state, item_xy, item_yaw, bin_xy, home, base_p, base_q


def main(argv) -> int:
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    scene, out_dir, count = argv[0], Path(argv[1]), int(argv[2])
    first_seed = int(argv[3]) if len(argv) == 4 else 0

    spec = recipe.SCENES[scene]
    from so101_sim.robots.so101_base.so101 import SO101

    for sub in ("states", "plans", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    made, skipped = 0, []
    seed = first_seed
    while made < count:
        sampled = sample_scene(spec["env_id"], seed)
        if sampled is None:
            skipped.append({"seed": seed, "why": "场景没停稳（物体或料箱仍在动）"})
            print(f"    种子 {seed}：场景没停稳，跳过")
            seed += 1
            continue
        state, item_xy, item_yaw, bin_xy, home, base_p, base_q = sampled
        actions, why = plan(scene, item_xy, item_yaw, bin_xy, home, base_p, base_q,
                            SO101.urdf_path)
        if actions is None:
            skipped.append({"seed": seed, "why": why,
                            "item_xy": item_xy.tolist(), "bin_xy": bin_xy.tolist()})
            print(f"    种子 {seed}：{why}，跳过（物体 {item_xy.round(4).tolist()}）")
            seed += 1
            continue
        episode = made
        (out_dir / "states" / f"ep{episode}.json").write_text(
            json.dumps(state, ensure_ascii=False))
        np.save(out_dir / "plans" / f"ep{episode}.npy", actions)
        (out_dir / "meta" / f"ep{episode}.json").write_text(json.dumps(
            {"seed": seed, "n_frames": len(actions),
             "item_xy": item_xy.tolist(), "item_yaw_deg": round(np.degrees(item_yaw), 3),
             "bin_xy": bin_xy.tolist(),
             "gripper_min_pct": round(float(actions[:, 5].min()), 4),
             "gripper_max_pct": round(float(actions[:, 5].max()), 4)},
            ensure_ascii=False))
        print(f"    ep{episode}（种子 {seed}）：{len(actions)} 帧，夹爪 "
              f"{actions[:, 5].min():.2f}~{actions[:, 5].max():.2f}%")
        made += 1
        seed += 1

    (out_dir / "meta" / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False))
    print(f"\n  {scene}：备好 {made} 集，跳过 {len(skipped)} 个槽位（种子 "
          f"{first_seed}~{seed - 1}）")
    tally = {}
    for row in skipped:
        key = str(row["why"]).split("（")[0]
        tally[key] = tally.get(key, 0) + 1
    for key, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {key}: {n} 个（{n / max(1, len(skipped)) * 100:.0f}% 的弃用）")
    print(f"  {out_dir}")
    print("PREPARE_EPISODES_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
