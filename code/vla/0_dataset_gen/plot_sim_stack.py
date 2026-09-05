"""画 SO-101 仿真器这一摞：物理引擎 → 场景包 → lerobot 封装 → 三条命令。

在讲10 3.2.1 节被引用。读者要能一眼看出 `so101_sim` 与 lerobot 之间是什么关系 ——
它不是另一套训练框架，而是一个**把 ManiSkill 场景包成 lerobot 认的 gym 环境**的薄层，
所以录数据、训练、评测三条官方命令原样可用。

那一层真正在做的事只有一件：**把口径换成真机的**。ManiSkill 内部一律弧度，而真机
（`lerobot-record` 走 `so_follower`）是混着的 —— 五个臂关节是度，夹爪是 0~100 的行程
百分比。所以「统一到真机」不是一个单位换算，是逐通道换算。图里把这句话放在那一层下面。

★ 存盘路径就是讲义引用的那个路径，不存别处再拷（xb-q1ut）。

用法：`python plot_sim_stack.py`
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[2] / "assets" / "figures"))
import figstyle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT = figstyle.apply()
OUT = HERE.resolve().parents[2] / "assets" / "figures" / "lecture10"

# 竖排四层。★ **不横排**：四个框挤在 5.6 英寸的版心里，标题一律溢出框外，
# 而 matplotlib 不会为此报错，只是把字画到框外面去。
LAYERS = [
    ("SAPIEN / ManiSkill", "刚体物理与渲染，并行环境", "#eef3f8"),
    ("so101_sim 场景包", "与实物同一副 SO-101 的 URDF；三个抓放场景与成功判据", "#f0f4ec"),
    ("So101SimEnv（lerobot 封装）", "gym 接口；把观测与动作换成真机口径", "#f6efe8"),
    ("lerobot 官方命令", "record 录数据 · train 训练 · eval 评测", "#f2eef4"),
]


def main() -> int:
    figstyle.assert_covered(
        "SAPIEN / ManiSkill so101_sim 场景包 So101SimEnv（lerobot 封装）lerobot 官方命令 "
        "刚体物理与渲染 并行环境 与实物同一副 SO-101 的 URDF 三个抓放场景与成功判据 "
        "gym 接口；把观测与动作换成真机口径 record 录数据 · train 训练 · eval 评测 "
        "刚体物理与渲染，并行环境 与实物同一副 SO-101 的 URDF；三个抓放场景与成功判据 "
        "臂关节用度、夹爪用 0~100 的行程百分比", "仿真器结构图")

    _, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(LAYERS)
    h, gap = 0.195, 0.062
    for i, (title, body, face) in enumerate(LAYERS):
        y = 1.0 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.02, y), 0.96, h, boxstyle="round,pad=0.008",
                                    linewidth=1.0, edgecolor="#555", facecolor=face))
        ax.text(0.05, y + h * 0.68, title, ha="left", va="center", fontsize=8.5, fontweight="bold")
        ax.text(0.05, y + h * 0.26, body, ha="left", va="center", fontsize=8, color="#333")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.006), (0.5, y - gap + 0.006),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.0, color="#555"))

    # 这一行是那层封装的全部意义，不是脚注：口径不对，数据就混不进真机那份。
    y3 = 1.0 - 3 * h - 2 * gap
    ax.text(0.955, y3 + h * 0.26, "臂关节用度、夹爪用 0~100 的行程百分比",
            ha="right", va="center", fontsize=7.5, color="#c0392b")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig10-3-sim-stack.png"
    plt.savefig(path, dpi=200, bbox_inches="tight",
                metadata={"Software": "Matplotlib", "Font": FONT})
    plt.close()
    from PIL import Image
    w_px, h_px = Image.open(path).size
    print(f"  {path.name}  {w_px}x{h_px}  比 {w_px / h_px:.2f}  Font={FONT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
