"""把仿真+真机混训那一轮的收敛曲线与成功率画成两张图。

在讲12 2.3 节被引用。数据来自 `scratch/` 下两个文本文件，都从训练控制台日志抽出：

  train-loss.txt   每行 `epch:2.24 loss:0.051 grdn:0.951`

★ 横轴用 **epoch 不用 step**：`lerobot-train` 过了 999 步就把 step 印成 `5K` 这种
  圆整值，同一个 `5K` 会对应上千步、画出来是一根竖线。`epch` 带两位小数
  （分辨率约 133 步），既准又比步数更好读。这不是审美选择，是日志格式逼出来的。

★ 成功率那张图**按场景分开画、并标出 95% 置信区间**，不画三场景平均：
  平均会把某一项的失败藏起来。验收判据本来就是"最低那一项 ≥70%"。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 统一字体走 assets/figures/figstyle.py（TimesSong：西文 Times New Roman + 中文宋体）。
# 不自己挑字体：本仓所有讲义配图共用这一套，各图各挑会出一批字体不一致的图。
# figstyle 取不到字体时**直接中止**，不回落系统字体 —— 回落只会渲出看不出错的错图。
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "assets" / "figures"))
import figstyle

FONT = figstyle.apply()
# 图按**放置尺寸**出（版心约 5.6 英寸），不出大图再让 LaTeX 缩 —— 缩放会把字一起缩小。
# 第一版按 7.5 英寸出、放到 5.3 英寸，字小了 30%，单看 PNG 不明显、进 PDF 就糊。
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

HERE = Path(__file__).parent
# ★ 存盘路径**就是讲义 markdown 里引用的那个路径**，不存别处再拷。
#   xb-q1ut：plot_sac_vs_ppo.py 存进 result/、讲义读 assets/，于是它接好的统一字体
#   从没体现在讲义用的那两张图上 —— 拷贝这一步会让"改了脚本"与"讲义里换了图"脱钩。
OUT = HERE.resolve().parents[3] / "assets" / "figures" / "lecture12" / "ref"
OUT.mkdir(parents=True, exist_ok=True)

def save(path):
    """存盘：写字体元数据，并裁掉四周白边。

    Args:
        path: 目标 PNG 路径。

    元数据里的 `Font` 让这张图**自己证明**是用统一字体渲的（`exiftool` 或
    `PIL.Image.open(p).info` 都读得到），不用回头翻脚本。

    裁白边是 xb-h26t：matplotlib 的 `bbox_inches="tight"` 只裁到画布边界，
    条形图上下仍会留一大条，进讲义后表现为「图与图注之间空一大块」。
    """
    plt.savefig(path, dpi=200, bbox_inches="tight",
                metadata={"Software": "Matplotlib", "Font": FONT})
    plt.close()
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    ink = np.where((arr < 248).any(axis=2))          # 非白像素
    if len(ink[0]):
        pad = 12
        top, bot = max(0, ink[0].min() - pad), min(arr.shape[0], ink[0].max() + pad + 1)
        left, right = max(0, ink[1].min() - pad), min(arr.shape[1], ink[1].max() + pad + 1)
        im.crop((left, top, right, bot)).save(path)
    print(f"  {path.name}  {Image.open(path).size[0]}x{Image.open(path).size[1]}  Font={FONT}")


def plot_loss():
    """收敛曲线，并把六个评测点标在对应的 epoch 上。"""
    figstyle.assert_covered(
        "flow-matching 损失（对数轴）训练轮数 epoch成功率 pc_success (%)"
        "训练途中成功率（cube40，各 10 局）0.05.1.2.4", "收敛曲线")
    rows = (HERE / "scratch" / "train-loss.txt").read_text().split()
    epch, loss = [], []
    for i in range(0, len(rows), 3):
        epch.append(float(rows[i].removeprefix("epch:")))
        loss.append(float(rows[i + 1].removeprefix("loss:")))
    # 途中评测每 5000 步一次，值来自 eval-points.txt（每行一个 pc_success，按评测顺序）
    evals = [float(x) for x in
             (HERE / "scratch" / "eval-points.txt").read_text().split()]

    _, ax1 = plt.subplots(figsize=(5.6, 3.1))
    ax1.plot(epch, loss, color="#c0392b", linewidth=1.2, label="flow-matching 损失")
    ax1.set_yscale("log")
    # ★ 对数轴的默认刻度标签是 $10^{-1}$，里面的减号是 U+2212 —— **TimesSong 没有这个字形**
    #   （figstyle 的文档里点名过它），渲出来是一个 ¤ 方框，而 matplotlib 只印一行
    #   findfont 警告、图照存。改成写死的普通小数标签，既避开缺字又比指数记法好读。
    ax1.set_yticks([0.05, 0.1, 0.2, 0.4])
    ax1.set_yticklabels(["0.05", "0.1", "0.2", "0.4"])
    ax1.minorticks_off()
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
    ax2.set_ylabel("成功率 pc_success (%)", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")
    ax2.set_ylim(0, 108)

    lines = ax1.get_lines() + ax2.get_lines()[:1]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    plt.tight_layout()
    save(OUT / "fig12-2-sim-real-sft-convergence.png")


if __name__ == "__main__":
    plot_loss()
