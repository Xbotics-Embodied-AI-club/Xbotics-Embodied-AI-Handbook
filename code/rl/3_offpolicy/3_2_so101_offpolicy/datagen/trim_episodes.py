"""裁掉两类废帧，并只保留成功轨迹。

上一批 ReachCube 数据集暴露了两个问题（2026-07-29 核账），这一步在转 LeRobot 之前修掉：

1. **首帧是坏渲染**：`RecordEpisode` 录的第 0 帧，相机位姿还没初始化（`cam2world_gl` 的
   平移恰为 (0,0,0)），画面与其余帧明显不是同一场景，而它照样进了数据集。→ 丢首帧。
2. **成功后大段静止**：episode 定长 50 帧、成功后不早停，上一批 800 帧里有 679 帧
   （85%）发生在 success 之后、动作幅度几乎为 0。策略吃这种数据容易学成"不动"。
   → 成功后只留 `HOLD_FRAMES` 帧作收尾，其余截掉。

另外只保留**真正成功**的 episode（`success` 数组里出现过 True），失败轨迹不进数据集。

h5 里逐帧的数据有两种长度：观测/env_states 是 N+1 帧，`actions`/`success`/`rewards` 是 N 帧
（`action[i]` 把 `obs[i]` 推到 `obs[i+1]`）。丢首帧要**同时**丢 `action[0]`，否则错位一帧。
"""

from pathlib import Path

import h5py
import numpy as np

DROP_FIRST_FRAMES = 1   # 坏渲染首帧
HOLD_FRAMES = 5         # 成功后保留几帧收尾（0=成功即切）


def trim(in_h5: Path, out_h5: Path) -> tuple[int, int]:
    """裁剪 + 只留成功轨迹，写到 `out_h5`。返回 (保留集数, 保留总帧数)。

    Args:
        in_h5: 待裁剪的 h5。
        out_h5: 输出路径。

    Returns:
        `(保留集数, 保留总帧数)`。
    """
    kept, total_frames = 0, 0
    with h5py.File(in_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        for attr, value in src.attrs.items():
            dst.attrs[attr] = value
        for traj_name in sorted(src.keys(), key=lambda s: int(s.split("_")[1])):
            g = src[traj_name]
            success = g["success"][:]
            if not success.any():
                continue  # 失败轨迹不要

            first_success = int(np.argmax(success))
            # 动作序列保留到 first_success + HOLD_FRAMES（含），再丢掉开头的坏帧
            act_end = min(first_success + 1 + HOLD_FRAMES, success.shape[0])
            act_slice = slice(DROP_FIRST_FRAMES, act_end)
            # 观测比动作多一帧：动作切 [a, b) 对应观测切 [a, b+1)
            obs_slice = slice(DROP_FIRST_FRAMES, act_end + 1)
            if act_slice.stop - act_slice.start <= 0:
                continue

            out = dst.create_group(f"traj_{kept}")

            # 逐数据集拷贝再切片（避免整组拷完再删的双倍占用）
            def copy_and_slice(name, obj):
                """把一个数据集拷进输出组并按帧切片。

                逐个数据集拷完就切，而不是整组拷完再删——后者在磁盘上会短暂占用双倍空间。

                Args:
                    name: h5 里的数据集名。
                    obj: 对应的 h5 对象（只处理 Dataset，忽略 Group）。
                """
                if isinstance(obj, h5py.Dataset):
                    data = obj[:]
                    if data.shape[0] == success.shape[0] + 1:
                        data = data[obs_slice]
                    elif data.shape[0] == success.shape[0]:
                        data = data[act_slice]
                    out.create_dataset(name, data=data)

            g.visititems(copy_and_slice)
            n = out["actions"].shape[0]
            total_frames += n
            kept += 1
    print(f"trim: 保留 {kept} 集 / {total_frames} 帧 "
          f"(丢首 {DROP_FIRST_FRAMES} 帧, 成功后留 {HOLD_FRAMES} 帧) -> {out_h5}")
    return kept, total_frames
