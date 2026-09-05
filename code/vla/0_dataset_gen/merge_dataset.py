#!/usr/bin/env python
"""把重录的仿真数据集与真机数据集合成一份训练集 —— 全程官方命令。

## 为什么这一步能成立

它此前不成立：仿真那四份出自已退役的手工转换路径，`video.codec` 是 `mpeg4` 而真机
那 9 份实测全是 `h264`，`robot_type` 也各写各的；而官方合并的第一步
`validate_all_metadata`（`aggregate.py:66-80`）**逐字比 fps / robot_type / features**，
任一不等就抛错。先前是拿 `video_files_size_in_mb=1` 绕开跨编码拼接、再做一份真机
工作副本抹平 `features` —— 那是绕开，不是消除。

现在三样都在源头对齐了，所以这里只剩一条官方命令：

  · **编码**：仿真改用 `lerobot-record` 录，并显式给 `--dataset.vcodec`，取值来自
    `0_dataset_gen/recipe.py` 的 `VIDEO_CODEC`（= 真机那个 `h264`）。
    **不能用默认值** —— `lerobot-record` 不给这个参数时录出来是 av1。
  · **robot_type**：机器人插件的 `name` 取 `so_follower`（与真机同值）。选择器仍是
    `--robot.type=so101_sim` —— lerobot 自己就这么分（`config_so_follower.py` 把同一个
    配置类注册成 `so101_follower` 与 `so100_follower` 两个选择器，共用一个 robot_type）。
  · **features**：关节名、单位、分辨率本来就同源；相机形状取自声明常量，不依赖运行时。

## 为什么必须先物理合并，不能训练时挂多份

`lerobot.datasets.factory.make_dataset` 对非 str 的 `repo_id` 直接
`raise NotImplementedError`，多数据集那条分支是死代码 ⇒ 训练入口只吃一份数据集。

用法：`python merge_dataset.py <输出根> <数据集根> [<数据集根> ...]`
"""

import json
import subprocess
import sys
from pathlib import Path

# 合并后的数据集标识。目录名与它都标着这批含仿真数据，`robot_type` 不必再承担这件事。
MERGED_REPO_ID = "xbotics/so101-sim-real"


def features_of(root):
    """一份数据集的 `features` 字典与 fps / robot_type。

    Args:
        root: 数据集根目录。

    Returns:
        `(features, fps, robot_type)`。
    """
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    return info["features"], info["fps"], info["robot_type"]


def main(argv) -> int:
    if len(argv) < 3:
        sys.exit(__doc__)
    out_root, roots = Path(argv[0]), [Path(r) for r in argv[1:]]

    # 合并前先逐项报出来。官方那步只会抛一个「不等」，不告诉你差在哪一项。
    base = features_of(roots[0])
    print(f"  基准 {roots[0].name}: fps={base[1]} robot_type={base[2]} "
          f"features {len(base[0])} 项")
    for root in roots[1:]:
        feats, fps, robot = features_of(root)
        diff = [k for k in set(base[0]) | set(feats) if base[0].get(k) != feats.get(k)]
        print(f"  {root.name}: fps={fps} robot_type={robot} "
              f"features 差 {len(diff)} 项" + (f" → {diff}" if diff else ""))

    repo_ids = [f"local/{root.name}" for root in roots]
    # 用**跑本脚本的那个解释器**旁边的 CLI，不靠 PATH —— worker 上 PATH 里没有它，
    # 而靠 PATH 找会在不同机器上悄悄取到另一个环境的版本。
    edit_cli = Path(sys.executable).parent / "lerobot-edit-dataset"
    cmd = [str(edit_cli),
           "--operation.type=merge",
           f"--operation.repo_ids=[{','.join(repo_ids)}]",
           f"--operation.roots=[{','.join(str(r) for r in roots)}]",
           f"--new_repo_id={MERGED_REPO_ID}",
           f"--new_root={out_root}"]
    print("\n  " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)

    info = json.loads((out_root / "meta" / "info.json").read_text())
    print(f"\n  合并结果 {info['total_episodes']} 集 / {info['total_frames']} 帧 / "
          f"{info['total_tasks']} 个任务 / fps={info['fps']}")
    # 训练侧读 meta/stats.json 做归一化，读不到不报错、只静默不归一化。
    if not (out_root / "meta" / "stats.json").is_file():
        sys.exit("★ 缺 meta/stats.json —— 训练会静默用错归一化")
    print("MERGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
