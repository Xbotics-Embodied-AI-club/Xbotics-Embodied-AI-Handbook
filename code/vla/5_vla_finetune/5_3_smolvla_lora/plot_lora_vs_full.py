"""把 LoRA 与全参微调的两条 loss 曲线、两组途中成功率画在一起。

在讲12 3.10 节被引用。两轮除了「怎么改权重」之外逐字相同：同一份 3698 集的训练集、
同样 30000 步、同样 warmup、同样六个图像增强、同样六卡 effective batch 96、
同样的评测环境参数。**只有这样，两条曲线的差才只归因于 LoRA 本身。**

图要说明的事：LoRA 的 loss 全程比全参高一截且不收敛到同一处（末点 0.099 对 0.052），
但**成功率追平**。损失差不等于能力差 —— 这正是 2.4.2 节那条「训练损失不是验收指标」
在另一个场景下的复现。

★ 存盘路径就是讲义引用的那个路径，不存别处再拷（xb-q1ut）。

用法：`python plot_lora_vs_full.py`
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[3] / "assets" / "figures"))
import figstyle
import matplotlib.pyplot as plt

FONT = figstyle.apply()
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
OUT = HERE.resolve().parents[3] / "assets" / "figures" / "lecture12" / "ref"

# 两轮的 loss 轨迹各存一份：全参那轮在 5_2 的 scratch 下，LoRA 这轮在本目录 scratch 下。
FULL_LOSS = HERE.parents[0] / "5_2_smolvla_full_sft" / "scratch" / "train-loss.txt"
LORA_LOSS = HERE / "scratch" / "lora-loss.txt"
FULL_EVAL = HERE.parents[0] / "5_2_smolvla_full_sft" / "scratch" / "eval-points.txt"
LORA_EVAL = HERE / "scratch" / "lora-eval-points.txt"


def read_loss(path: Path) -> tuple[list[float], list[float]]:
    """读一条 `epch:… loss:…` 的轨迹。

    Args:
        path: 轨迹文件。

    Returns:
        `(epoch 列表, loss 列表)`。
    """
    if not path.is_file():
        sys.exit(f"★ 缺 {path}")
    ep, ls = [], []
    for tok in path.read_text().split():
        if tok.startswith("epch:"):
            ep.append(float(tok[5:]))
        elif tok.startswith("loss:"):
            ls.append(float(tok[5:]))
    if len(ep) != len(ls):
        sys.exit(f"★ {path} 里 epch 与 loss 数量对不上（{len(ep)} vs {len(ls)}）")
    return ep, ls


def read_eval(path: Path) -> list[float]:
    """读一列途中评测成功率（每行一个，按评测顺序）。"""
    if not path.is_file():
        sys.exit(f"★ 缺 {path}")
    return [float(x) for x in path.read_text().split()]


def main() -> int:
    figstyle.assert_covered(
        "flow-matching 损失（对数轴）训练轮数 epoch 成功率 pc_success (%) "
        "全参 403M LoRA 10M 途中成功率 0.05.1.2.4", "LoRA 与全参对读图")

    fe, fl = read_loss(FULL_LOSS)
    le, ll = read_loss(LORA_LOSS)
    fev, lev = read_eval(FULL_EVAL), read_eval(LORA_EVAL)

    _, ax1 = plt.subplots(figsize=(5.6, 3.1))
    ax1.plot(fe, fl, color="#c0392b", linewidth=1.1, label="全参 403M")
    ax1.plot(le, ll, color="#e59866", linewidth=1.1, label="LoRA 10M")
    ax1.set_yscale("log")
    # 对数轴默认刻度里的减号是 U+2212，TimesSong 没有这个字形，渲出来是方框。
    ax1.set_yticks([0.05, 0.1, 0.2, 0.4])
    ax1.set_yticklabels(["0.05", "0.1", "0.2", "0.4"])
    ax1.minorticks_off()
    ax1.set_xlabel("训练轮数 epoch")
    ax1.set_ylabel("flow-matching 损失（对数轴）")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    # 两轮都是每 5000 步评一次、共 30000 步 ⇒ 评测点按 epoch 等距落在 1/6 … 6/6 上。
    span = max(max(fe), max(le))
    for vals, color, mark, lab in ((fev, "#2471a3", "o", "全参 · 途中成功率"),
                                   (lev, "#5dade2", "s", "LoRA · 途中成功率")):
        xs = [(i + 1) * span / len(vals) for i in range(len(vals))]
        ax2.plot(xs, vals, mark + "-", color=color, linewidth=1.4, markersize=5, label=lab)
    ax2.set_ylabel("成功率 pc_success (%)", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")
    ax2.set_ylim(0, 108)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    plt.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig12-3-lora-vs-full.png"
    plt.savefig(path, dpi=200, bbox_inches="tight",
                metadata={"Software": "Matplotlib", "Font": FONT})
    plt.close()
    from PIL import Image
    w, h = Image.open(path).size
    print(f"  {path.name}  {w}x{h}  比 {w / h:.2f}  Font={FONT}")
    print(f"  全参末 loss {fl[-1]:.3f} · LoRA 末 loss {ll[-1]:.3f}；评测点 全参{fev} LoRA{lev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
