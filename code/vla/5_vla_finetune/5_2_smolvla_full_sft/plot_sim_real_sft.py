"""把仿真+真机混训那一轮的收敛曲线与成功率画成两张图。

在讲12 2.3 节被引用。数据来自 `scratch/` 下两个文本文件，都从训练控制台日志抽出：

  train-loss.txt   每行 `epch:2.24 loss:0.051 grdn:0.951`
  eval-points.txt  每行一个 pc_success，按评测顺序（每 5000 步一个）

★ 横轴用 **epoch 不用 step**：`lerobot-train` 过了 999 步就把 step 印成 `5K` 这种
  圆整值，同一个 `5K` 会对应上千步、画出来是一根竖线。`epch` 带两位小数
  （分辨率约 133 步），既准又比步数更好读。这不是审美选择，是日志格式逼出来的。

★ 成功率那张图**按场景分开画、并标出 95% 置信区间**，不画三场景平均：
  平均会把某一项的失败藏起来。验收判据本来就是"最低那一项 ≥70%"。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import beta

# 字体按**本机实际装了什么**挑，不写死一个名字。
# 实测 w1 上一个中文字体都没有（总共 22 个字体），写死 "Noto Sans CJK SC" 的后果是
# 图里每个汉字变成一个方框 —— 而 matplotlib 只在 stderr 印一行 findfont 警告，
# 脚本照常退 0、图照常存盘。**这是会安静产出废图的那类失败。**
_WANT = ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
         "Source Han Sans SC", "Droid Sans Fallback")
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
_PICK = [n for n in _WANT if n in _HAVE]
if not _PICK:
    raise SystemExit(
        f"★ 本机没有可用的中文字体，画出来的图汉字会是方框。已找候选 {_WANT}。\n"
        f"  装一个：apt-get install -y fonts-noto-cjk，或在有字体的机器上跑本脚本。")
plt.rcParams["font.sans-serif"] = _PICK
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
OUT = HERE / "result"
OUT.mkdir(parents=True, exist_ok=True)

# 三场景最终验收：(指令, 成功数, 总局数)
FINAL = [
    ("Pick up a cube", 47, 50),
    ("Pick up a small cube", 45, 50),
    ("Pick up a can", 48, 50),
]
TARGET = 70.0  # 验收线（%）


def clopper_pearson(k, n, alpha=0.05):
    """成功率的精确二项区间。

    Args:
        k: 成功数。
        n: 总局数。
        alpha: 显著性水平。

    Returns:
        `(下界, 上界)`，单位是比例不是百分比。

    用精确区间而不是正态近似：n=50、p 接近 1 时正态近似的上界会超过 1，
    而"成功率 103%"这种数出现在讲义图里是硬伤。
    """
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def plot_loss():
    """收敛曲线，并把六个评测点标在对应的 epoch 上。"""
    rows = (HERE / "scratch" / "train-loss.txt").read_text().split()
    epch, loss = [], []
    for i in range(0, len(rows), 3):
        epch.append(float(rows[i].removeprefix("epch:")))
        loss.append(float(rows[i + 1].removeprefix("loss:")))
    evals = [float(x) for x in
             (HERE / "scratch" / "eval-points.txt").read_text().split()]

    _, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.plot(epch, loss, color="#c0392b", linewidth=1.2, label="flow-matching 损失")
    ax1.set_yscale("log")
    ax1.set_xlabel("训练轮数 epoch")
    ax1.set_ylabel("flow-matching 损失（对数轴）", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(True, which="both", alpha=0.25)

    # 评测每 5000 步一次，总 30000 步 = 2.25 epoch ⇒ 每 0.375 epoch 一个点。
    step_per_epoch = 30000 / max(epch)
    ax2 = ax1.twinx()
    xs = [(i + 1) * 5000 / step_per_epoch for i in range(len(evals))]
    ax2.plot(xs, evals, "o-", color="#2471a3", linewidth=1.6, markersize=6,
             label="训练途中成功率（cube40，各 10 局）")
    ax2.axhline(TARGET, color="#7f8c8d", linestyle="--", linewidth=1)
    ax2.text(max(epch) * 0.99, TARGET + 2, "验收线 70%", ha="right",
             fontsize=9, color="#7f8c8d")
    ax2.set_ylabel("成功率 pc_success (%)", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")
    ax2.set_ylim(0, 108)

    lines = ax1.get_lines() + ax2.get_lines()[:1]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    plt.title("SmolVLA 全模型全参数微调 · 仿真+真机混训 3698 集")
    plt.tight_layout()
    p = OUT / "fig12-2-sim-real-sft-convergence.png"
    plt.savefig(p, dpi=150)
    print(f"  收敛曲线 -> {p}  （末端 loss {loss[-1]:.3f}，评测点 {evals}）")


def plot_success():
    """三场景最终成功率，带 95% 精确二项区间。"""
    _, ax = plt.subplots(figsize=(7.5, 3.6))
    names = [f"{n}\n({k}/{tot})" for n, k, tot in FINAL]
    pcts = [k / tot * 100 for _, k, tot in FINAL]
    errs = np.array([[p - clopper_pearson(k, tot)[0] * 100,
                      clopper_pearson(k, tot)[1] * 100 - p]
                     for p, (_, k, tot) in zip(pcts, FINAL)]).T

    bars = ax.barh(names, pcts, color="#2471a3", height=0.5, zorder=2)
    ax.errorbar(pcts, range(len(FINAL)), xerr=errs, fmt="none",
                ecolor="#1b2631", capsize=5, linewidth=1.4, zorder=3)
    ax.axvline(TARGET, color="#c0392b", linestyle="--", linewidth=1.4, zorder=1)
    # 标签放在绘图区**内部**、贴着虚线。放在轴上方会与标题挤在一行，
    # 读起来像标题的一部分（第一版就是这样）。
    ax.text(TARGET - 1.5, -0.42, "验收线 70%", ha="right", va="center",
            color="#c0392b", fontsize=9)
    for bar, p, (_, k, tot) in zip(bars, pcts, FINAL):
        lo = clopper_pearson(k, tot)[0] * 100
        ax.text(2, bar.get_y() + bar.get_height() / 2,
                f"{p:.0f}%　95% 下界 {lo:.0f}%", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("成功率 pc_success (%)　·　每场景 50 局")
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    plt.title("最终 checkpoint 在三个仿真任务上的成功率（误差棒为 95% 精确二项区间）")
    plt.tight_layout()
    p = OUT / "fig12-2-sim-real-sft-success.png"
    plt.savefig(p, dpi=150)
    print(f"  成功率 -> {p}")


if __name__ == "__main__":
    plot_loss()
    plot_success()
