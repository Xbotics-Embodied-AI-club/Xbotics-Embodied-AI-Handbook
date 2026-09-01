"""把三个版本的训练 loss 画成对照图，看清楚"三级阶梯"各自在优化什么。

数据来自训练时记的 wandb run（`rl_class` 项目里的 g1-walk-reinforce / -a2c / -ppo，
各 3000 次迭代），已导出成 result/1_1_g1_walk_rl/train-loss-curves.json，
这里只负责画图，不再联网。

三个版本记录的 loss 项并不相同——这本身就是版本差异的直接体现：
  v1 REINFORCE  只有 policy_loss 和 entropy（没有 critic）
  v2 A2C        多出 value_loss（critic 要回归 returns）
  v3 PPO        再多出 kl 与 lr（按 KL 自适应调学习率）
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
# 全书统一字体：西文 Times New Roman、中文宋体。字体路径走环境变量，取不到就报错停下。
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "assets" / "figures"))
import figstyle  # noqa: E402  —— 必须在 sys.path 补好之后再导入
FONT_NAME = figstyle.apply()

here = Path(__file__).parent
curves = json.loads((here.parent / "result" / "1_1_g1_walk_rl" / "train-loss-curves.json").read_text())
out_path = here.parents[3] / "assets" / "figures" / "lecture14" / "ref" / "fig146-train-loss-curves.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

WINDOW = 50
COLORS = {"g1-walk-reinforce": "#1F6FB2", "g1-walk-a2c": "#D68910", "g1-walk-ppo": "#C0392B"}
ORDER = ["g1-walk-reinforce", "g1-walk-a2c", "g1-walk-ppo"]


def moving_average(values, window):
    """算滑动平均，用来从逐次迭代的原始曲线里把趋势拉出来。

    Args:
        values: 逐次迭代的原始数值。
        window: 窗口长度。开头不足一窗时按已有点数取平均。

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


PANELS = [
    ("policy_loss", "策略损失 policy_loss", "三版都有：$-\\log\\pi\\cdot A$ 的均值"),
    ("entropy", "策略熵 entropy", "熵越大动作分布越散，探索越充分"),
    ("value_loss", "价值损失 value_loss", "只有 v2/v3 有——critic 回归 returns"),
    ("kl", "新旧策略 KL", "只有 v3 有——步长管家据此调学习率"),
]

def robust_ylim(series):
    """按分位数给纵轴定范围。

    训练曲线常有个别尖峰，直接用最大最小值会让一根尖刺把其余部分全压成一条平线。

    Args:
        series: 若干条曲线的数值列表。

    Returns:
        (下界, 上界)；没有任何数据时返回 None。
    """
    flat = sorted(v for s in series for v in s)
    if not flat:
        return None
    lo, hi = flat[int(len(flat) * 0.02)], flat[int(len(flat) * 0.98) - 1]
    pad = (hi - lo) * 0.15 or 1.0
    return lo - pad, hi + pad


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (key, title, note) in zip(axes.flat, PANELS):
    drawn = False
    collected = []
    for name in ORDER:
        pts = curves["runs"].get(name, {}).get(key)
        if not pts:
            continue
        steps = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        ax.plot(steps, vals, color=COLORS[name], alpha=0.15, linewidth=0.7)
        ax.plot(steps, moving_average(vals, WINDOW), color=COLORS[name], linewidth=1.8,
                label=curves["runs"][name]["label"])
        collected.append(vals)
        drawn = True
    ax.set_title(f"{title}\n{note}", fontsize=11)
    lim = robust_ylim(collected)
    if lim:
        ax.set_ylim(*lim)
    ax.set_xlabel("训练迭代")
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(fontsize=9)

fig.suptitle(f"G1 行走：三个算法的训练损失（细线为原始值，粗线为 {WINDOW} 次迭代滑动平均）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(out_path, dpi=200, metadata={"Font": FONT_NAME})
print(f"图已保存到 {out_path}")
