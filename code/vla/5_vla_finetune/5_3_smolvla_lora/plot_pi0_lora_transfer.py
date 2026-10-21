"""把同一套 LoRA 配方在 SmolVLA 与 $\\pi_0$ 上的三场成功率画成一张对读图。

在讲12 3.10.3 节被引用。四轮共用同一份 3698 集训练集、同样 30000 步、同样六个图像增强、
同样的评测环境参数，每个任务各评 50 局 —— 差别只有基座和改权重的方式。

图要说明的事：**参数占比对齐不等于能力对齐**。SmolVLA 的 2.1% 与 $\\pi_0$ 的 2.06% 是
照着对齐的，两个大任务上四根柱子几乎一样高，20mm 小方块那一组却塌下去一半。
右边两根（$\\pi_0$ 的 1.53 与 2.25 epoch）几乎等高，是这张图的第二个用途：
多训的那 10000 步一个点都没抬起来 ⇒ 塌的原因不是欠拟合。

数字来源是四轮各自的三场终评 `eval_info.json`（w4 `/work/EAI-exp-002/outputs/eval-*/`），
共十二个数，直接写在下面的表里；轨迹类数据才走文件（见 plot_lora_vs_full.py）。

★ 存盘路径就是讲义引用的那个路径，不存别处再拷（xb-q1ut）。

用法：`python plot_pi0_lora_transfer.py`
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[3] / "assets" / "figures"))
import figstyle
import matplotlib.pyplot as plt

FONT = figstyle.apply()
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 9, "ytick.labelsize": 8})
OUT = HERE.resolve().parents[3] / "assets" / "figures" / "lecture12" / "ref"

TASKS = ("方块 40mm", "小方块 20mm", "圆柱体")
# (图例, 颜色, 三场成功率 %)。顺序 = 图上从左到右。
RUNS = (
    ("SmolVLA 全参 403M", "#c0392b", (94.0, 90.0, 96.0)),
    ("SmolVLA LoRA 10M", "#e59866", (94.0, 84.0, 98.0)),
    ("π0 LoRA 85M（1.53 轮）", "#5dade2", (92.0, 34.0, 94.0)),
    ("π0 LoRA 85M（2.25 轮）", "#2471a3", (92.0, 32.0, 92.0)),
)


def main() -> int:
    for lab, _, vals in RUNS:
        assert len(vals) == len(TASKS), f"★ {lab} 给了 {len(vals)} 个数，要 {len(TASKS)} 个"
        assert all(0.0 <= v <= 100.0 for v in vals), f"★ {lab} 里有不是百分数的值：{vals}"

    figstyle.assert_covered(
        "".join(TASKS) + "".join(lab for lab, _, _ in RUNS)
        + "成功率 pc_success (%) 每场 50 局 LoRA 全参 轮 0123456789.（）",
        "pi0 与 SmolVLA 的 LoRA 对读图")

    _, ax = plt.subplots(figsize=(5.8, 3.2))
    n = len(RUNS)
    width = 0.78 / n
    for i, (lab, color, vals) in enumerate(RUNS):
        xs = [t + (i - (n - 1) / 2) * width for t in range(len(TASKS))]
        ax.bar(xs, vals, width * 0.92, color=color, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("成功率 pc_success (%)　每场 50 局")
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    # 图例压在坐标区外面：四组柱子里有三组顶到 90 以上，放在区内一定会盖住柱顶那排数字。
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8,
              frameon=False)
    plt.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig12-3-pi0-lora-transfer.png"
    plt.savefig(path, dpi=200, bbox_inches="tight",
                metadata={"Software": "Matplotlib", "Font": FONT})
    plt.close()
    from PIL import Image
    w, h = Image.open(path).size
    print(f"  {path.name}  {w}x{h}  比 {w / h:.2f}  Font={FONT}")
    for lab, _, vals in RUNS:
        print(f"  {lab:26s} {vals}  最低 {min(vals):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
