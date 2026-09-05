"""画一集脚本化示教的关节轨迹，把五个阶段标出来。

在讲10 3.2.2 节被引用：读者要能看见「脚本化专家产出的一集」到底长什么样 ——
六条曲线怎么走、五个阶段各占多长、夹爪在哪两处动。

阶段边界从**夹爪指令**解出来，不硬编帧号：夹爪张开过之后首次合到底 = 抓住，
此后首次再张开 = 松手。起手那一帧夹爪本来就是合的（张开会把物体推走），
所以必须先等它张开过，否则边界会落在第 0 帧。

★ 存盘路径就是讲义引用的那个路径，不存别处再拷（xb-q1ut）。

用法：`python plot_demo_phases.py <数据集根> [<集号>]`
"""

import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[2] / "assets" / "figures"))
import figstyle
import matplotlib.pyplot as plt

FONT = figstyle.apply()
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
OUT = HERE.resolve().parents[2] / "assets" / "figures" / "lecture10"

JOINTS = ["底座回转", "大臂抬升", "小臂屈伸", "腕俯仰", "腕滚转"]
PHASES = [("取物", "#dbe8f4"), ("合拢", "#f6e3c9"), ("搬运", "#dcecdc"),
          ("松手", "#f4dcdc"), ("回家", "#e8e4f2")]


def load_episode(root: Path, ep: int) -> np.ndarray:
    """取一集的动作序列。

    Args:
        root: 数据集根目录。
        ep: 集号。

    Returns:
        `(帧数, 6)` 的动作数组，前五列是臂关节角（度），末列是夹爪行程百分比。
    """
    import pandas as pd

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"★ {root}/data 下没有 parquet")
    for f in files:
        df = pd.read_parquet(f)
        sel = df[df["episode_index"] == ep]
        if len(sel):
            return np.stack(sel.sort_values("frame_index")["action"].to_numpy())
    sys.exit(f"★ 没有第 {ep} 集")


def phase_bounds(grip: np.ndarray) -> list[int]:
    """按夹爪指令解出五个阶段的分界帧。

    Args:
        grip: `(帧数,)` 夹爪指令，小 = 合拢。

    Returns:
        长度 6 的分界点（含首尾）。
    """
    mid = (grip.max() + grip.min()) / 2
    opened = grip > mid
    if not opened.any():
        sys.exit("★ 这一集夹爪从没张开过，定位不了阶段")
    open_at = int(np.argmax(opened))
    closed_after = ~opened[open_at:]
    close_at = open_at + int(np.argmax(closed_after))
    open_again = opened[close_at:]
    release_at = close_at + int(np.argmax(open_again))
    last = len(grip) - 1
    # 合拢段 = 夹爪一直在往小走的那几帧；松手段 = 一直在往大走的那几帧。
    # ★ 别写成「首次到达最小值」：夹爪合到底之后会在底部维持很久，最小值可能出现在
    #   搬运段末尾，于是边界跑到松手之后去 —— 实测得到过 [0,155,359,286,287,376] 这种逆序，
    #   而 matplotlib 画逆序区间**不报错**，只是把色块画反。
    def settle(start: int, sign: int) -> int:
        d = np.diff(grip[start:])
        k = int(np.argmax(sign * d >= 0)) if (sign * d >= 0).any() else len(d)
        return min(start + max(k, 1), last)

    return [0, close_at, settle(close_at, +1), release_at, settle(release_at, -1), last]


def main(argv) -> int:
    if len(argv) not in (1, 2):
        sys.exit(__doc__)
    root = Path(argv[0])
    ep = int(argv[1]) if len(argv) == 2 else 0
    act = load_episode(root, ep)
    arm, grip = act[:, :5], act[:, 5]
    b = phase_bounds(grip)
    t = np.arange(len(act))

    figstyle.assert_covered(
        "关节角（度）夹爪行程（%）帧 取物 合拢 搬运 松手 回家 " + " ".join(JOINTS),
        "示教轨迹五段图")

    _, ax1 = plt.subplots(figsize=(5.6, 2.4))
    for i in range(5):
        lo, hi = b[i], b[i + 1]
        ax1.axvspan(lo, hi, color=PHASES[i][1], zorder=0)
        ax1.text((lo + hi) / 2, 1.02, PHASES[i][0], transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8)

    for j, name in enumerate(JOINTS):
        ax1.plot(t, arm[:, j], linewidth=1.0, label=name)
    ax1.set_ylabel("关节角（度）")
    ax1.legend(ncol=5, loc="lower center", frameon=False, columnspacing=1.0, handlelength=1.2)
    axg = ax1.twinx()
    axg.plot(t, grip, color="#666", linewidth=1.0, linestyle="--")
    axg.set_ylabel("夹爪行程（%）", color="#666")
    axg.tick_params(axis="y", labelcolor="#666")

    ax1.set_xlabel("帧")

    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig10-3-demo-phases.png"
    plt.savefig(path, dpi=200, bbox_inches="tight",
                metadata={"Software": "Matplotlib", "Font": FONT})
    plt.close()
    from PIL import Image
    w, h = Image.open(path).size
    print(f"  {path.name}  {w}x{h}  比 {w / h:.2f}  Font={FONT}")
    print(f"  阶段分界帧 {b}（必须递增）")
    assert b == sorted(b), f"★ 阶段边界逆序：{b}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
