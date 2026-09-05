"""把三个版本的训练奖励曲线画成一张对照图，作为「三级阶梯」的证据图。

数据来自训练时记的 wandb run（`rl_class` 项目里的 g1-walk-reinforce / -a2c / -ppo，
各 3000 次迭代），已导出成 result/1_1_g1_walk_rl/train-reward-curves.json，
这里只负责画图，不再联网。
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
curves = json.loads((here.parent / "result" / "1_1_g1_walk_rl" / "train-reward-curves.json").read_text())
out_path = here.parents[3] / "assets" / "figures" / "lecture14" / "ref" / "fig146-train-reward-curves.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

# 三条曲线的抖动程度差很多（A2C 那条尤其毛），只画原始值会互相盖住。
# 叠一条滑动平均把趋势拎出来，原始曲线用淡色留在底下，
# 这样"抖不抖"和"涨不涨"能在同一张图上分开读。
WINDOW = 50


def moving_average(values, window):
    """算滑动平均，用来从逐次迭代的原始曲线里把趋势拉出来。

    Args:
        values: 逐次迭代的原始数值。
        window: 窗口长度。开头不足一窗时按已有点数取平均，曲线才从第一个点就画得出来。

    Returns:
        与输入等长的滑动平均序列。
    """
    out, running = [], 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


# 按讲解顺序排：v1 → v2 → v3
ORDER = ["g1-walk-reinforce", "g1-walk-a2c", "g1-walk-ppo"]
COLORS = {"g1-walk-reinforce": "#1F6FB2", "g1-walk-a2c": "#D68910", "g1-walk-ppo": "#C0392B"}

fig, ax = plt.subplots(figsize=(9, 5))
for name in ORDER:
    run = curves["runs"][name]
    steps = [p[0] for p in run["points"]]
    rewards = [p[1] for p in run["points"]]
    ax.plot(steps, rewards, color=COLORS[name], alpha=0.18, linewidth=0.8)
    ax.plot(steps, moving_average(rewards, WINDOW), color=COLORS[name],
            linewidth=2.0, label=run["label"])

ax.set_xlabel("训练迭代")
ax.set_ylabel("训练回合平均奖励")
# 图内不写标题：标题由讲义正文的图注给出，图里再写一遍 Quarto 会渲成「图 N: 图：…」。
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_path, dpi=200, metadata={"Font": FONT_NAME})
print(f"图已保存到 {out_path}")
