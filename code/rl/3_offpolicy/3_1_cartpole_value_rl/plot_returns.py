"""把两级值学习的回合回报画成一张对照图：手工分桶的表格 Q vs 吃原始状态的 DQN。

两个训练脚本跑完各自把整条回报曲线存进 result/，这里只负责画图。
横轴统一用环境步数（而不是回合数）——两级的回合长度差很多，按回合对齐会误导。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

here = Path(__file__).parent
result_dir = here.parent / "result"
out_path = here.parents[3] / "assets" / "figures" / "lecture16" / "ref" / "fig162-cartpole-returns.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

# 单回合回报抖得厉害，叠一条滑动平均看趋势；原始值用淡色留在底下。
WINDOW = 20


def moving_average(values, window):
    """滑动平均，开头不足一窗时按已有的点数取均值（不丢前几个点）。

    单回合回报本身抖得厉害，直接画成一团毛线看不出趋势；叠一条滑动平均才读得出
    "有没有在涨"。开头不补 NaN 是有意的——曲线从第一个回合就画得出来。

    Args:
        values: 按回合顺序排列的回报序列。
        window: 窗口长度（回合数）。

    Returns:
        与 values 等长的滑动平均列表。
    """
    out, running = [], 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


PANELS = [
    ("cartpole-qlearning.json", "v1 表格 Q-learning（手工分桶）", "#1F6FB2"),
    ("cartpole-dqn.json", "v2 DQN（网络吃原始状态）", "#C0392B"),
]

fig, ax = plt.subplots(figsize=(9, 5))
for filename, label, color in PANELS:
    data = json.loads((result_dir / filename).read_text())
    returns = data["returns"]
    # 每个回合的长度就是它的回报（CartPole 每活一步得 1 分），累加即到该回合为止的环境步数。
    env_steps, total = [], 0
    for r in returns:
        total += r
        env_steps.append(total)
    ax.plot(env_steps, returns, color=color, alpha=0.15, linewidth=0.8)
    ax.plot(env_steps, moving_average(returns, WINDOW), color=color, linewidth=2.0,
            label=f"{label}　末50回合均值 {sum(returns[-50:]) / 50:.0f}")

ax.axhline(500, color="#666", linestyle="--", linewidth=1.2)
ax.text(198000, 508, "CartPole-v1 满分 500", fontsize=10, color="#666", ha="right")

ax.set_xlabel("环境步数")
ax.set_ylabel("单回合回报（杆子立住的步数）")
ax.set_ylim(0, 560)
ax.set_title(f"CartPole：从查表到用网络（细线为单回合，粗线为 {WINDOW} 回合滑动平均）")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_path, dpi=200)
print(f"图已保存到 {out_path}")
