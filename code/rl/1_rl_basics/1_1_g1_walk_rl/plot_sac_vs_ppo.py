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

# 存到讲义直接引用的那个路径，与 plot_returns.py 同一套约定。图只能有一个落点：
# 早先本脚本存进 result/，讲义读的却是 assets/ 下另一份，于是这里接上的统一字体
# 从来没体现在讲义用的那张图上（PDF 批注「图依然要用 timesnewromen 和宋体」就是它）。
fig_dir = here.parents[3] / "assets" / "figures" / "lecture16" / "ref"
fig_dir.mkdir(parents=True, exist_ok=True)

# 与 plot_reward_curves.py 同一套版本配色，跨图读起来才是同一个 v3 / 同一个 v4。
SERIES = [("ppo", "v3 PPO (on-policy)", "#333333", "o"),
          ("sac", "v4 SAC (off-policy)", "#1F77B4", "s")]

# (x 轴取哪个字段, x 轴单位换算, x 轴标题, 要不要对数轴, 输出文件名)
# 墙钟按分钟出图：讲义正文、图注和对照表都用分钟，图上再用秒读者就得自己换算。
PANELS = [("env_steps", 1.0, "environment steps", True, "fig-v4-sac-envsteps.png"),
          ("wall", 1 / 60, "wall-clock time (min)", False, "fig-v4-sac-walltime.png")]


def draw(x_key, x_scale, x_label, log_x, filename):
    """画一张对照图并存盘。"""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for key, label, color, marker in SERIES:
        points = compare[key]
        ax.plot([p[x_key] * x_scale for p in points], [p["mean_reward"] for p in points],
                color=color, marker=marker, markersize=4, linewidth=1.2, label=label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("deterministic eval reward / step")
    ax.set_title("G1 walk: SAC vs PPO")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=130, metadata={"Font": FONT_NAME})
    plt.close(fig)
    print(f"图已保存到 {out}")


def main():
    for panel in PANELS:
        draw(*panel)


if __name__ == "__main__":
    main()
