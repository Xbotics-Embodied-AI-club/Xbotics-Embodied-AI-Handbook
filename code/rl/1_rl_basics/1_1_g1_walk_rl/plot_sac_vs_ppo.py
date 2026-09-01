"""把 SAC 与 PPO 的评测奖励画成两张对照图：一张按环境步数、一张按墙钟时间。

数据来自 result/1_1_g1_walk_rl/compare.json（评测协议：256 并行环境 × 300 步、确定性动作），
这里只负责画图，不重跑实验。

原来这两张图是有的，但出图脚本没跟着进仓 —— 图上带 Matplotlib 元数据、全仓却查不到脚本。
本文件按留存的 compare.json 把它们补回来，同时接上全书统一字体。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 全书统一字体：西文 Times New Roman、中文宋体。字体路径走环境变量，取不到就报错停下。
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "assets" / "figures"))
import figstyle  # noqa: E402  —— 必须在 sys.path 补好之后再导入
FONT_NAME = figstyle.apply()

here = Path(__file__).parent
result_dir = here.parent / "result" / "1_1_g1_walk_rl"
compare = json.loads((result_dir / "compare.json").read_text())

# 与 plot_reward_curves.py 同一套版本配色，跨图读起来才是同一个 v3 / 同一个 v4。
SERIES = [("ppo", "v3 PPO (on-policy)", "#333333", "o"),
          ("sac", "v4 SAC (off-policy)", "#1F77B4", "s")]

# (x 轴取哪个字段, x 轴标题, 要不要对数轴, 输出文件名)
PANELS = [("env_steps", "environment steps", True, "sac-vs-ppo-envsteps.png"),
          ("wall", "wall-clock time (s)", False, "sac-vs-ppo-walltime.png")]


def draw(x_key, x_label, log_x, filename):
    """画一张对照图并存盘。"""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for key, label, color, marker in SERIES:
        points = compare[key]
        ax.plot([p[x_key] for p in points], [p["mean_reward"] for p in points],
                color=color, marker=marker, markersize=4, linewidth=1.2, label=label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("deterministic eval reward / step")
    ax.set_title("G1 walk: SAC vs PPO")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = result_dir / filename
    fig.savefig(out, dpi=130, metadata={"Font": FONT_NAME})
    plt.close(fig)
    print(f"图已保存到 {out}")


def main():
    for panel in PANELS:
        draw(*panel)


if __name__ == "__main__":
    main()
