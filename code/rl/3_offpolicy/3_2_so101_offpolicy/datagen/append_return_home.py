"""给每条轨迹接一段「收回起始位姿」的收尾。

真机采的 `task1` 每集都是从起始位姿出发、干完活再回到起始位姿收尾（首末关节角最大只差
11.8°）。而策略 rollout 截到「首次成功 + 收尾几帧」就停了，机械臂停在料箱正上方，首末差
中位 48.9°。要跟真机同口径，得把这段收回补上。

策略没学过回家，所以这段是合成的：关节空间余弦缓入缓出插值，从末位姿收回该集自己的首帧
位姿。物体位姿保持末帧不动（它已经躺在箱底了）。

**动作怎么算**：控制器是 `pd_joint_target_delta_pos`，
`target[i+1] = target[i] + action[i] * scale`。所以先插出目标序列，再逐帧反解出动作。
`scale` 不硬编码——从这批数据自己的 (action, target) 对反解，残差不够小就直接报错，
免得换了控制器还照旧算、悄悄写出一批错动作。

**回程给多长**：由动作上限倒推。余弦插值的峰值步长是 `(pi/2)/n * 总位移`，它必须小于
`scale`，否则反解出来的动作会越出 [-1, 1] 被截断——机械臂就回不到位。所以逐集按需要的
最少帧数算，再留一点余量，而不是所有集强行等长。
"""

from pathlib import Path

import h5py
import numpy as np

# articulation state 排布：root_pose 7 + root_linvel 3 + root_angvel 3 + qpos 6 + qvel 6
QPOS = slice(13, 19)
QVEL = slice(19, 25)

FRAME_MARGIN = 1.2      # 回程帧数余量，避免动作贴着 ±1 饱和
MIN_RETURN_FRAMES = 45  # 再短的收回看着像瞬移
CONTROL_FREQ = 30

# 整段轨迹（取放 + 收回）的目标时长区间。真机 task1 是 19.27s，v7 策略段平均只有 6.03s，
# 光靠策略段凑不到；回程帧数由本区间反推补足。余弦插值的峰值步长随帧数**单调下降**，
# 所以把回程拉长只会更平缓、绝不会越出动作上限，安全方向是"够长"。
TARGET_TOTAL_SECONDS = (10.0, 17.0)
TARGET_TOTAL_FRAMES = tuple(int(s * CONTROL_FREQ) for s in TARGET_TOTAL_SECONDS)


def _solve_delta_scale(f) -> np.ndarray:
    """从 (action, target) 对反解每个关节的动作尺度，并验证控制器确实是 target-delta 式。"""
    actions, deltas = [], []
    for traj in f:
        g = f[traj]
        actions.append(g["actions"][:])
        deltas.append(np.diff(g["obs/agent/controller/target_qpos"][:], axis=0))
    actions = np.concatenate(actions)
    deltas = np.concatenate(deltas)

    scale = np.empty(actions.shape[1], np.float64)
    for j in range(actions.shape[1]):
        s, offset = np.polyfit(actions[:, j], deltas[:, j], 1)
        residual = np.abs(actions[:, j] * s + offset - deltas[:, j]).max()
        if residual > 1e-5:
            raise SystemExit(
                f"关节 {j} 对不上 target-delta 模型（残差 {residual:.2e}）——"
                "控制器不是 pd_joint_target_delta_pos，动作反解会写出错数据")
        scale[j] = s
    return scale


def _smoothstep(n: int) -> np.ndarray:
    """0→1 的余弦缓入缓出，两端速度为 0，收回不会有速度阶跃。"""
    return 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))


def _return_length(target_gap: np.ndarray, scale: np.ndarray, n_policy: int) -> int:
    """这一集的收回给几帧：既要够慢（不越动作上限），又要把整段凑进目标时长区间。

    下界＝动作上限倒推的最少帧数（余弦峰值步长 `(pi/2)/n × 位移` 必须 < `scale`）再留余量；
    在此之上按 `TARGET_TOTAL_FRAMES` 补足，让「取放 + 收回」落进 10–17s。
    策略段已经超过区间上限时不再拉长，只给下界（该集本身就够长，不该再加）。
    """
    least = np.ceil((np.pi / 2) * np.abs(target_gap) / scale).max()
    floor = int(max(MIN_RETURN_FRAMES, np.ceil(least * FRAME_MARGIN)))
    lo, hi = TARGET_TOTAL_FRAMES
    if n_policy >= hi:
        return floor
    # 往区间中点靠，既不贴下限也不贴上限
    want = int(np.clip((lo + hi) // 2 - n_policy, floor, hi - n_policy))
    return max(floor, want)


def append_return_home(in_h5: Path, out_h5: Path) -> tuple[int, int]:
    """给 `in_h5` 每集接上收回段，写到 `out_h5`。返回 (集数, 追加的总帧数)。

    Args:
        in_h5: 待处理的 h5。
        out_h5: 输出路径。

    Returns:
        `(集数, 追加的总帧数)`。
    """
    added_total, kept = 0, 0
    with h5py.File(in_h5, "r") as src:
        scale = _solve_delta_scale(src)
        with h5py.File(out_h5, "w") as dst:
            for attr, value in src.attrs.items():
                dst.attrs[attr] = value

            for traj_name in sorted(src.keys(), key=lambda s: int(s.split("_")[1])):
                g = src[traj_name]
                art_key = next(iter(g["env_states/articulations"]))
                states = g[f"env_states/articulations/{art_key}"][:]
                targets = g["obs/agent/controller/target_qpos"][:]

                n_ret = _return_length(targets[0] - targets[-1], scale, len(targets))
                w = _smoothstep(n_ret)[:, None]

                # 关节角与控制目标一起插回首帧，两者同步才自洽
                ret_states = np.repeat(states[-1:], n_ret, axis=0)
                ret_states[:, QPOS] = states[-1, QPOS] * (1 - w) + states[0, QPOS] * w
                ret_states[:, QVEL] = np.gradient(ret_states[:, QPOS], axis=0) * CONTROL_FREQ
                ret_targets = targets[-1] * (1 - w) + targets[0] * w

                # 动作＝目标的逐帧增量除以尺度；首个增量接的是原轨迹最后一个目标
                full_targets = np.concatenate([targets, ret_targets])
                ret_actions = np.diff(full_targets[len(targets) - 1:], axis=0) / scale
                ret_actions = ret_actions.astype(np.float32)

                out = dst.create_group(traj_name)
                n_obs = states.shape[0]

                def extend(name, obj):
                    """把一个数据集沿时间维接上收回段。

                    观测比动作多一帧，所以两类数据集要按各自的长度分别补齐，不能一刀切。

                    Args:
                        name: h5 里的数据集名。
                        obj: 对应的 h5 对象（只处理 Dataset，忽略 Group）。
                    """
                    if not isinstance(obj, h5py.Dataset):
                        return
                    data = obj[:]
                    if name == f"env_states/articulations/{art_key}":
                        data = np.concatenate([data, ret_states.astype(data.dtype)])
                    elif name in ("obs/agent/controller/target_qpos",
                                  "env_states/controller/target_qpos"):
                        data = np.concatenate([data, ret_targets.astype(data.dtype)])
                    elif name == "obs/agent/noisy_qpos":
                        # 这批 noisy_qpos 与真实 qpos 逐位相同（未加噪），保持同一口径
                        data = np.concatenate([data, ret_states[:, QPOS].astype(data.dtype)])
                    elif name == "actions":
                        data = np.concatenate([data, ret_actions])
                    elif data.shape[0] == n_obs:
                        # 其余逐帧量（物体位姿、相机参数、占位图像）保持末帧不变
                        data = np.concatenate([data, np.repeat(data[-1:], n_ret, axis=0)])
                    elif data.shape[0] == n_obs - 1:
                        data = np.concatenate([data, np.repeat(data[-1:], n_ret, axis=0)])
                    out.create_dataset(name, data=data)

                g.visititems(extend)
                # 让下游能认出哪几帧是合成的收回段，不必靠帧数相减去猜
                out.attrs["return_start"] = n_obs - 1
                out.attrs["return_frames"] = n_ret
                added_total += n_ret
                kept += 1

    print(f"append_return_home: {kept} 集，共追加 {added_total} 帧收回 "
          f"(每集 {added_total / kept:.0f} 帧均值) -> {out_h5}")
    return kept, added_total
