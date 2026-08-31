"""把三个版本的训练奖励曲线画成一张对照图，作为「三级阶梯」的证据图。

数据来自训练时记的 wandb run（`rl_class` 项目里的 g1-walk-reinforce / -a2c / -ppo，
各 3000 次迭代），已导出成 result/1_1_g1_walk_rl/train-reward-curves.json，
这里只负责画图，不再联网。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

here = Path(__file__).parent
curves = json.loads((here.parent / "result" / "1_1_g1_walk_rl" / "train-reward-curves.json").read_text())
out_path = here.parents[3] / "assets" / "figures" / "lecture14" / "ref" / "fig146-train-reward-curves.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

# 训练奖励每次迭代抖得厉害，直接画成一团毛线；叠一条滑动平均把趋势显出来，
# 原始曲线用淡色留在底下，读者能同时看到"抖"和"涨"。
WINDOW = 50


def moving_average(values, window):
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
ax.set_title(f"G1 行走：三个算法的训练奖励曲线（细线为原始值，粗线为 {WINDOW} 次迭代滑动平均）")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_path, dpi=200)
print(f"图已保存到 {out_path}")
