"""画真机部署的两侧分工图：板子只管现场，模型推理在 x86 GPU 机上。

在讲12 2.5 节被引用。这张图原先是正文里的一段 ASCII 框图，进 PDF 之后垮掉了 ——
制表符（U+2500 一族）在讲义正文字体里没有字形，箭头和横线整条消失、文字错位。
**用字符拼出来的图在换字体时会静默变形**，所以改成真图。

★ 存盘路径就是讲义引用的那个路径，不存别处再拷（xb-q1ut）。

用法：`python plot_deploy_split.py`
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "assets" / "figures"))
import figstyle

FONT = figstyle.apply()
OUT = HERE.resolve().parents[3] / "assets" / "figures" / "lecture12" / "ref"

BOARD = ["读舵机、开相机", "把观测打包发出去", "收到动作块下发舵机"]
GPU = ["加载 checkpoint", "跑策略推理", "回一个动作块"]


def box(ax, x, w, title, lines, face):
    """画一侧的方框与它的职责清单。

    Args:
        ax: 画布。
        x: 方框左边缘（0~1 轴坐标）。
        w: 方框宽度。
        title: 框头文字。
        lines: 框内逐条职责。
        face: 填充色。

    ★ 宽度是参数不是常数：左框标题带机型名、比右框长一半，两侧同宽时标题会**溢出框外**
      —— 而 matplotlib 不会为此报错，只是画出去。
    """
    ax.add_patch(FancyBboxPatch((x, 0.14), w, 0.72, boxstyle="round,pad=0.012",
                                linewidth=1.0, edgecolor="#555", facecolor=face))
    ax.text(x + w / 2, 0.76, title, ha="center", va="center", fontsize=9, fontweight="bold")
    for i, t in enumerate(lines):
        ax.text(x + 0.03, 0.58 - i * 0.14, f"· {t}", ha="left", va="center", fontsize=8.5)


def main() -> int:
    figstyle.assert_covered(
        "边缘端部署板 x86 GPU 机 读舵机、开相机 把观测打包发出去 收到动作块下发舵机"
        "加载 checkpoint 跑策略推理 回一个动作块 观测 动作块 权重 多数边缘端设备装不下也跑不动",
        "真机部署分工图")

    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.36, "边缘端部署板", BOARD, "#eef3f8")
    box(ax, 0.62, 0.36, "x86 GPU 机", GPU, "#f6efe8")

    # 两条箭头分开画：一来一回是两个方向，画成双箭头就看不出谁先谁后。
    ax.add_patch(FancyArrowPatch((0.39, 0.62), (0.61, 0.62), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color="#2471a3"))
    ax.text(0.50, 0.68, "观测（gRPC）", ha="center", va="bottom", fontsize=9, color="#2471a3")
    ax.add_patch(FancyArrowPatch((0.61, 0.34), (0.39, 0.34), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color="#c0392b"))
    ax.text(0.50, 0.26, "动作块", ha="center", va="top", fontsize=9, color="#c0392b")

    # 这一行是**为什么要拆**，不是装饰：板子的内存装不下权重，才有这张图。
    ax.text(0.5, 0.045, "$\\pi_0$ 权重 3 GB 多、SmolVLA 900 MB —— 多数边缘端设备装不下也跑不动",
            ha="center", va="center", fontsize=8.5, color="#444")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig12-2-real-deploy-split.png"
    plt.savefig(path, dpi=200, bbox_inches="tight",
                metadata={"Software": "Matplotlib", "Font": FONT})
    plt.close()
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    ink = np.where((arr < 248).any(axis=2))
    pad = 10
    im.crop((max(0, ink[1].min() - pad), max(0, ink[0].min() - pad),
             min(arr.shape[1], ink[1].max() + pad + 1),
             min(arr.shape[0], ink[0].max() + pad + 1))).save(path)
    w, h = Image.open(path).size
    print(f"  {path.name}  {w}x{h}  比 {w/h:.2f}  Font={FONT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
