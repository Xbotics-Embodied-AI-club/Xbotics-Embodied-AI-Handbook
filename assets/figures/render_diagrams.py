"""用 matplotlib 画本课程自绘的示意图，产出讲义引用的 PNG。

图与代码放在一起，改文字改配色都在这里改，重跑即可再生——不依赖任何外部绘图工具。
同名的 `lectureNN/src/*.svg` 是等价的手写源，二者择一即可。

跑法：python assets/figures/render_diagrams.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent

# 写进 PNG 元数据，交付自检据此确认"该自绘的图确实已自绘"，而不是还留着外部截图。
OWNER_TAG = "xbotics-render_diagrams"

BLUE, GREEN, ORANGE, RED, GREY = "#1F6FB2", "#2E8B57", "#D68910", "#C0392B", "#777777"
FILL = {BLUE: "#E8F1FA", GREEN: "#E7F4EC", ORANGE: "#FDF2E0", RED: "#FBEAE8", GREY: "#F4F4F4"}


def box(ax, x, y, w, h, color, lines):
    """在 (x,y) 画一个圆角框；lines 是 (文字, 字号, 是否加粗) 的列表，纵向居中均分。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=2, edgecolor=color, facecolor=FILL[color]))
    n = len(lines)
    gap = h / (n + 1)
    for i, (text, size, bold) in enumerate(lines, 1):
        ax.text(x + w / 2, y + h - i * gap, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=color if bold else "#444444")


def arrow(ax, a, b, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.7, color=color))


def label(ax, x, y, text, color, size=10, bold=False, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            fontweight="bold" if bold else "normal")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, lecture, name):
    out = HERE / lecture / "ref" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white",
                metadata={"Software": OWNER_TAG})
    plt.close(fig)
    print(f"  {out.relative_to(HERE.parents[1])}")


# ── 第14讲 3.4：on-policy 与 off-policy 的数据流对比 ────────────────────
def onoff_dataflow():
    fig, ax = canvas(13, 5.6)
    label(ax, 0.22, 0.955, "on-policy（第14讲、第15讲）", BLUE, 15, True)
    label(ax, 0.75, 0.955, "off-policy（第16讲）", RED, 15, True)
    ax.plot([0.46, 0.46], [0.02, 0.90], color="#DDDDDD", linewidth=1.5, linestyle="--")

    # 左：采样 → 更新 → 作废
    box(ax, 0.09, 0.68, 0.26, 0.13, BLUE, [("当前策略", 14, True)])
    box(ax, 0.07, 0.36, 0.30, 0.17, BLUE, [("新鲜数据", 13, True), ("（只来自当前策略）", 10, False)])
    arrow(ax, (0.17, 0.675), (0.17, 0.535), BLUE)
    label(ax, 0.135, 0.605, "采样", BLUE, 10, ha="right")
    arrow(ax, (0.27, 0.535), (0.27, 0.675), BLUE)
    label(ax, 0.305, 0.605, "更新一次", BLUE, 10, ha="left")
    arrow(ax, (0.22, 0.35), (0.22, 0.24), GREY)
    label(ax, 0.22, 0.185, "更新完立即作废", "#666666", 11)
    label(ax, 0.22, 0.10, "策略一变，全部重采", "#999999", 10)

    # 右：三来源 → 经验池 → 目标策略
    pool_top = 0.635
    for cx, text, color in [(0.585, "旧策略经验", BLUE), (0.745, "人类示范", GREEN),
                            (0.905, "人工干预", ORANGE)]:
        box(ax, cx - 0.068, 0.75, 0.136, 0.10, color, [(text, 11, True)])
        arrow(ax, (cx, 0.745), (cx * 0.35 + 0.745 * 0.65, pool_top + 0.005), color)

    box(ax, 0.52, 0.45, 0.44, 0.185, RED,
        [("replay buffer（经验池）", 14, True), ("长期保存，反复使用", 11, True)])
    arrow(ax, (0.63, 0.445), (0.63, 0.305), RED)
    label(ax, 0.595, 0.375, "随机抽 batch", RED, 10, ha="right")
    arrow(ax, (0.86, 0.305), (0.86, 0.445), GREY)
    label(ax, 0.895, 0.375, "新经验入池", "#666666", 10, ha="left")
    box(ax, 0.575, 0.16, 0.31, 0.145, RED, [("目标策略", 14, True)])
    label(ax, 0.73, 0.09, "采数据的手 与 被训练的脑 解耦", "#666666", 10)
    save(fig, "lecture14", "fig-onoff-dataflow")


# ── 第14讲 5.3：A2C 的 Actor-Critic 分工 ───────────────────────────────
def actor_critic_a2c():
    fig, ax = canvas(12, 5.6)
    label(ax, 0.5, 0.965, "A2C：critic 只是给策略梯度当参照物", "#333333", 15, True)

    box(ax, 0.04, 0.56, 0.26, 0.17, BLUE, [("Actor（策略网络）", 13, True), ("π(a | s)　出动作", 11, False)])
    box(ax, 0.04, 0.14, 0.26, 0.17, GREEN, [("Critic（价值网络）", 13, True), ("V(s)　估状态价值", 11, False)])
    box(ax, 0.40, 0.56, 0.17, 0.17, GREY, [("环境", 13, True), ("给出 r 与 s′", 10, False)])
    box(ax, 0.66, 0.33, 0.30, 0.22, ORANGE,
        [("优势 A", 13, True), ("A = r + γV(s′) − V(s)", 11, False), ("「比平均水平好多少」", 10, False)])

    arrow(ax, (0.305, 0.645), (0.395, 0.645), "#555555")
    label(ax, 0.35, 0.685, "a", "#555555", 10)
    arrow(ax, (0.575, 0.62), (0.655, 0.50), "#555555")
    label(ax, 0.645, 0.60, "r, s′", "#555555", 10)
    arrow(ax, (0.305, 0.245), (0.655, 0.39), GREEN, rad=-0.10)
    label(ax, 0.47, 0.245, "V(s), V(s′)", GREEN, 10)

    arrow(ax, (0.81, 0.555), (0.17, 0.74), RED, rad=0.14)
    label(ax, 0.49, 0.885, "用 A 加权策略梯度，更新 actor", RED, 12, True)
    arrow(ax, (0.70, 0.325), (0.17, 0.13), GREEN, rad=0.12)
    label(ax, 0.44, 0.045, "critic 回归 returns，把 V 估得更准", GREEN, 12, True)
    save(fig, "lecture14", "fig-actor-critic-a2c")


# ── 第16讲 3.2：DDPG 的 Actor-Critic 分工 ──────────────────────────────
def actor_critic_ddpg():
    fig, ax = canvas(12.5, 5.8)
    label(ax, 0.5, 0.965, "DDPG：Actor 直接吐连续动作，Critic 给它打分", "#333333", 15, True)

    box(ax, 0.03, 0.58, 0.29, 0.20, BLUE,
        [("Actor（确定性策略）", 13, True), ("a = μ(s)", 11, False), ("不做 argmax，一步吐出连续动作", 9, False)])
    box(ax, 0.03, 0.17, 0.29, 0.20, GREEN,
        [("Critic（动作价值）", 13, True), ("Q(s, a)", 11, False), ("学习主体，评价「在 s 做了 a」", 9, False)])
    box(ax, 0.40, 0.60, 0.22, 0.18, ORANGE,
        [("replay buffer", 12, True), ("(s, a, r, s′) 反复使用", 10, False)])
    box(ax, 0.66, 0.20, 0.31, 0.22, RED,
        [("贝尔曼回归目标", 12, True), ("y = r + γ Q′(s′, μ′(s′))", 11, False), ("目标网络软更新，稳住目标", 9, False)])

    arrow(ax, (0.325, 0.685), (0.395, 0.685), "#555555")
    label(ax, 0.36, 0.80, "新经验入池", "#555555", 9)
    arrow(ax, (0.47, 0.595), (0.33, 0.335), ORANGE, rad=-0.12)
    label(ax, 0.44, 0.44, "抽 batch", ORANGE, 10)
    arrow(ax, (0.325, 0.25), (0.655, 0.29), GREEN)
    label(ax, 0.49, 0.20, "Q(s,a) 向 y 回归", GREEN, 10)

    arrow(ax, (0.76, 0.425), (0.17, 0.79), RED, rad=0.14)
    label(ax, 0.47, 0.895, "Actor 沿 ∇a Q(s, a) 的方向更新——把 Q 推高", RED, 12, True)
    label(ax, 0.5, 0.05, "与 A2C 的分工正好相反：这里 critic 是学习主体，"
                         "actor 更像「帮连续动作找 Q 最大值」的执行器。", "#555555", 10)
    save(fig, "lecture16", "fig162-actor-critic-ddpg")


# ── 第14讲 2.2：动作空间分两类 ──────────────────────────────────────────
def action_space():
    fig, ax = canvas(12, 5.0)
    label(ax, 0.5, 0.955, "动作空间分两类", "#333333", 15, True)
    ax.plot([0.5, 0.5], [0.04, 0.88], color="#CCCCCC", linewidth=1.5, linestyle="--")

    label(ax, 0.25, 0.845, "离散动作空间", BLUE, 13, True)
    label(ax, 0.25, 0.775, "从有限个选项里挑一个", "#4A6A85", 10)
    for (bx, by, text) in [(0.09, 0.575, "向左推"), (0.27, 0.575, "向右推"),
                           (0.09, 0.425, "不动"), (0.27, 0.425, "跳")]:
        box(ax, bx, by, 0.14, 0.11, BLUE, [(text, 11, False)])
    label(ax, 0.25, 0.32, "动作个数有限，可以逐个比较好坏", "#555555", 10)
    label(ax, 0.25, 0.20, "例：游戏手柄按键、", "#888888", 10)
    label(ax, 0.25, 0.13, "语言模型从词表里选下一个 token", "#888888", 10)

    label(ax, 0.75, 0.845, "连续动作空间", RED, 13, True)
    label(ax, 0.75, 0.775, "动作是一个实数区间里的取值", "#4A6A85", 10)
    ax.annotate("", xy=(0.95, 0.53), xytext=(0.58, 0.53),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=2.2))
    for tick in (0.60, 0.92):
        ax.plot([tick, tick], [0.505, 0.555], color=ORANGE, linewidth=2)
    ax.plot(0.76, 0.53, "o", color=RED, markersize=9)
    label(ax, 0.60, 0.455, "−1.0", "#8A5A08", 10)
    label(ax, 0.92, 0.455, "+1.0", "#8A5A08", 10)
    label(ax, 0.76, 0.60, "任一实数", RED, 10)
    label(ax, 0.75, 0.32, "取值有无穷多个，没法逐个枚举", "#555555", 10)
    label(ax, 0.75, 0.20, "例：方向盘转角、", "#888888", 10)
    label(ax, 0.75, 0.13, "G1 的 29 个关节目标量", "#888888", 10)
    save(fig, "lecture14", "fig-action-space")


# ── 第14讲 2.4：策略把观测映射成动作 ────────────────────────────────────
def policy_brain():
    fig, ax = canvas(12, 4.0)
    label(ax, 0.5, 0.94, "策略：把观测映射成动作", "#333333", 15, True)
    box(ax, 0.04, 0.42, 0.24, 0.32, BLUE,
        [("观测 s", 13, True), ("关节角度、躯干姿态", 10, False), ("目标速度指令", 10, False)])
    box(ax, 0.37, 0.38, 0.26, 0.40, ORANGE,
        [("策略网络 π", 13, True), ("神经网络", 10, False),
         ("输出一个动作分布", 10, False), ("而不是一个写死的动作", 10, False)])
    box(ax, 0.72, 0.42, 0.24, 0.32, RED,
        [("动作分布 π(a | s)", 12, True), ("从中采样一个动作 a", 10, False)])
    arrow(ax, (0.285, 0.58), (0.365, 0.58), "#666666")
    arrow(ax, (0.635, 0.58), (0.715, 0.58), "#666666")
    label(ax, 0.5, 0.22, "同一个观测下合理的动作往往不止一个，所以策略输出的是分布；"
                         "随机性正是探索的来源。", "#555555", 10)
    save(fig, "lecture14", "fig-policy-brain")


# ── 第14讲 2.1：智能体与环境的交互闭环 ──────────────────────────────────
def agent_env_loop():
    fig, ax = canvas(11, 4.6)
    box(ax, 0.36, 0.70, 0.28, 0.19, BLUE,
        [("智能体", 14, True), ("策略 π：看到什么 → 做什么", 10, False)])
    box(ax, 0.36, 0.11, 0.28, 0.19, GREEN,
        [("环境", 14, True), ("按物理规律推进一步", 10, False)])
    arrow(ax, (0.645, 0.795), (0.645, 0.205), RED, rad=-0.55)
    label(ax, 0.90, 0.50, "动作 $a_t$", RED, 12, True)
    arrow(ax, (0.355, 0.205), (0.355, 0.795), BLUE, rad=-0.55)
    label(ax, 0.10, 0.545, "新状态 $s_{t+1}$", BLUE, 12, True)
    label(ax, 0.10, 0.455, "奖励 $r_t$", BLUE, 12, True)
    label(ax, 0.5, 0.50, "如此循环，直到回合结束", "#888888", 10)
    save(fig, "lecture14", "fig141-agent-env-loop")


print("渲染示意图：")
action_space()
policy_brain()
agent_env_loop()
onoff_dataflow()
actor_critic_a2c()
actor_critic_ddpg()
