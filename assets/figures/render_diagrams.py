"""用 matplotlib 画本课程自绘的示意图，产出讲义引用的 PNG。

图与代码放在一起，改文字改配色都在这里改，重跑即可再生——不依赖任何外部绘图工具。
**本文件是自绘图的唯一真相源**：不再维护同名 `lectureNN/src/*.svg`，改图只改这里。

跑法：python assets/figures/render_diagrams.py

## 字号规则（style-guide §17 的 D6，全书统一判据）

印在纸上的字号由两件事决定，缺一不可：

    印出字号 = 源码 fontsize × 缩放比
    缩放比   = 版心宽度 × 排版 width% ÷ 作图宽度

版心按 A4 加 1 英寸边距算，约 6.5 英寸（`BODY_WIDTH_IN`）。判据一句话：
**图内最小的那个字，印在纸上不低于 8 pt**（`MIN_PRINT_PT`）。

由此本文件一律按"所见即所得"作图——**画布宽度不超过 8 英寸、图内任何文字不小于
10 pt、排版一律 `width=100%`**。8 英寸时缩放比 6.5/8 = 0.81，10 pt 落到纸上 8.1 pt，
刚好过线；画布越窄越宽裕。

`save()` 会读回 PNG 的真实像素宽度重算一遍并逐张打印，不达标直接终止。所以**不能靠调
排版的 `width%` 蒙混**：`width` 把留白连同文字一起放大，字相对于画面还是那么小，纸上
该看不清还是看不清。要救只有三条作图侧的路——放大字号、收窄画布、把信息拆成两张图。
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.text import Text

import figstyle

matplotlib.use("Agg")

# 全书统一字体：西文 Times New Roman、中文宋体（同目录 figstyle.py，字体路径走环境变量）。
FONT_NAME = figstyle.apply()

HERE = Path(__file__).parent

# 写进 PNG 元数据，交付自检据此确认"该自绘的图确实已自绘"，而不是还留着外部截图。
OWNER_TAG = "xbotics-render_diagrams"

DPI = 200
BODY_WIDTH_IN = 6.5    # A4 加 1 英寸边距后的版心宽度
MIN_PRINT_PT = 8.0     # 图内最小字印在纸上的下限

BLUE, GREEN, ORANGE, RED, GREY = "#1F6FB2", "#2E8B57", "#D68910", "#C0392B", "#777777"
FILL = {BLUE: "#DCE9F7", GREEN: "#DDEEE3", ORANGE: "#FBEBCE", RED: "#F7DDDA", GREY: "#EDEDED"}

# 流程图的线条与文字统一走这三个常量，颜色只留在框的填充上。
# 参照 fig142-rl-taxonomy（draw.io 出品）：细深灰框线 + 黑字 + 淡填充，
# 而不是"每个元素各自上色"——后者会让整张图花得刺眼、框线粗过内容。
EDGE = "#5A5A5A"        # 框线与箭头
INK = "#1A1A1A"         # 框内文字
BOX_LW = 1.1


def box(ax, x, y, w, h, color, lines):
    """在 (x,y) 画一个圆角框；lines 是 (文字, 字号, 是否加粗) 的列表，纵向居中均分。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.014",
                                linewidth=BOX_LW, edgecolor=EDGE, facecolor=FILL[color]))
    n = len(lines)
    gap = h / (n + 1)
    for i, (text, size, bold) in enumerate(lines, 1):
        ax.text(x + w / 2, y + h - i * gap, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=INK)


def side(x, y, w, h, where, t=0.5):
    """取框某条边上的一点，默认边中点。

    箭头接在边中点上，视觉上才是"连到这个框"；手填坐标容易落在靠近框角的位置，
    看起来像从框身上长出来的。t 可在 0–1 之间挪动接点，用于同一条边接多根线。

    Args:
        x, y, w, h: 框的左下角与宽高，与 box() 同一套参数。
        where: "top" / "bottom" / "left" / "right"。
    """
    if where == "top":
        return (x + w * t, y + h)
    if where == "bottom":
        return (x + w * t, y)
    if where == "left":
        return (x, y + h * t)
    return (x + w, y + h * t)


def plain_box(ax, x, y, w, h, color, fill=None, style="solid", lw=None):
    """只画框不写字，供需要自己排版内部元素的图使用。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                                linewidth=BOX_LW, edgecolor=EDGE, linestyle=style,
                                facecolor=FILL[color] if fill is None else fill))


def arrow(ax, a, b, color=None, rad=0.0, lw=1.4, style="-|>"):
    color = EDGE if color is None else color
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color))


_NORMAL = {"top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0)}


def connect(ax, src, src_side, dst, dst_side, color=None, lw=1.4, stub=0.055,
            src_t=0.5, dst_t=0.5, lane=None, radius=0.016):
    """把两个框用正交折线连起来，线从边中点**垂直离开**框，不贴任何框边跑。

    这是流程图连线的基本要求：贴着框边走的线会和框线叠在一起，读者分不清
    哪一段是框、哪一段是箭头（本图早先就是这个毛病）。所以两端各先垂直伸出
    一小段 stub，进到空白区再拐弯。

    Args:
        src, dst: 两个框的 (x, y, w, h)，与 box() 同一套参数。
        src_side, dst_side: 各自从哪条边出发/进入。
        stub: 垂直离开框的那一小段有多长。
        lane: 中间那条公共通道的位置；两端同向时用它指定绕行的横带/竖带，
              避免两条线挤在同一条通道上。
    """
    color = EDGE if color is None else color
    p0 = side(*src, src_side, src_t)
    p1 = side(*dst, dst_side, dst_t)
    n0, n1 = _NORMAL[src_side], _NORMAL[dst_side]
    a = (p0[0] + n0[0] * stub, p0[1] + n0[1] * stub)
    b = (p1[0] + n1[0] * stub, p1[1] + n1[1] * stub)

    mid = []
    horiz0, horiz1 = n0[0] != 0, n1[0] != 0
    if horiz0 and horiz1:                       # 两端都横着出来 → 中间走一条竖带
        xm = lane if lane is not None else (a[0] + b[0]) / 2
        mid = [(xm, a[1]), (xm, b[1])]
    elif not horiz0 and not horiz1:             # 两端都竖着出来 → 中间走一条横带
        ym = lane if lane is not None else (a[1] + b[1]) / 2
        mid = [(a[0], ym), (b[0], ym)]
    elif horiz0:                                # 先横后竖
        mid = [(b[0], a[1])]
    else:                                       # 先竖后横
        mid = [(a[0], b[1])]

    _polyline(ax, [p0, a] + mid + [b, p1], color, lw, radius)


def _polyline(ax, pts, color, lw, radius):
    """按给定折点画一条圆角正交折线，末端带箭头。"""
    # 去掉重复点，否则圆角计算会除以零
    clean = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - clean[-1][0]) > 1e-9 or abs(q[1] - clean[-1][1]) > 1e-9:
            clean.append(q)
    verts, codes = [clean[0]], [MplPath.MOVETO]
    for i in range(1, len(clean) - 1):
        prev, cur, nxt = np.array(clean[i - 1]), np.array(clean[i]), np.array(clean[i + 1])
        din, dout = cur - prev, nxt - cur
        din = din / (np.hypot(*din) or 1)
        dout = dout / (np.hypot(*dout) or 1)
        r = min(radius, np.hypot(*(cur - prev)) / 2, np.hypot(*(nxt - cur)) / 2)
        verts += [tuple(cur - din * r), tuple(cur), tuple(cur + dout * r)]
        codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    verts.append(clean[-1])
    codes.append(MplPath.LINETO)
    ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw, color=color,
                                 joinstyle="round", capstyle="round"))


def elbow(ax, a, b, color, via="v", lw=1.7, style="-|>", detour=None, radius=0.018):
    """正交折线箭头：只走横平竖直，拐角是圆角直角。

    跨半张图的长回环用弧线会在画面中间划出一道大圆弧，压住别的元素也看不出走向；
    折线沿着版面的横竖骨架走，读者一眼能跟着拐弯看到终点。

    Args:
        a, b: 起点、终点。
        via: "v" 先竖后横，"h" 先横后竖，"hvh" 先横到 detour 再竖再横（绕开中间的框）。
        detour: via="hvh" 时那条竖直中段所在的 x。
    """
    (x0, y0), (x1, y1) = a, b
    if via == "ring":
        # 走外圈：先横到 detour 那条竖带，沿它竖直走到终点高度所在的横带，再横回终点。
        # 用于两条回流互不相交时——内圈让给短的那条。
        xm, ym = detour
        pts = [(x0, y0), (xm, y0), (xm, ym), (x1, ym), (x1, y1)]
    elif via == "vhv":
        # 先竖到 detour 那条横带，沿它横过去，再竖进终点——用来绕开中间的框走上下空带。
        ym = detour if detour is not None else (y0 + y1) / 2
        pts = [(x0, y0), (x0, ym), (x1, ym), (x1, y1)]
    elif via == "hvh":
        xm = detour if detour is not None else (x0 + x1) / 2
        pts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]
    elif via == "h":
        pts = [(x0, y0), (x1, y0), (x1, y1)]
    else:
        pts = [(x0, y0), (x0, y1), (x1, y1)]

    # 直角处切掉一小段、用圆角接上，避免尖角在小尺寸下显得毛躁。
    verts, codes = [pts[0]], [MplPath.MOVETO]
    for i in range(1, len(pts) - 1):
        prev, cur, nxt = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[i + 1])
        din = cur - prev
        dout = nxt - cur
        din = din / (np.hypot(*din) or 1)
        dout = dout / (np.hypot(*dout) or 1)
        r = min(radius, np.hypot(*(cur - prev)) / 2, np.hypot(*(nxt - cur)) / 2)
        verts += [tuple(cur - din * r), tuple(cur), tuple(cur + dout * r)]
        codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    verts.append(pts[-1])
    codes.append(MplPath.LINETO)

    ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color,
                                 joinstyle="round", capstyle="round"))


def label(ax, x, y, text, color, size=10, bold=False, ha="center", va="center", bg=False):
    """写一行字；bg=True 时给字加白底，用来压住穿过它的箭头而不必留大片空白。"""
    kw = {}
    if bg:
        kw["bbox"] = dict(facecolor="white", edgecolor="none", pad=1.5)
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight="bold" if bold else "normal", **kw)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    return fig, ax


def save(fig, lecture, name):
    """存图，并按模块开头那条字号规则核算这一张印出来够不够大。"""
    out = HERE / lecture / "ref" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 缺字会渲成空方框而不报错，所以存盘前先把这张图上的字逐个对一遍字体覆盖。
    figstyle.assert_covered("".join(t.get_text() for t in fig.findobj(Text)), name)
    smallest = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white",
                metadata={"Software": OWNER_TAG, "Font": FONT_NAME})
    plt.close(fig)

    # PNG 的 IHDR 就在文件头，宽度是第 17–20 字节；除以 dpi 得到落到纸上的真实作图宽度。
    width_in = int.from_bytes(out.read_bytes()[16:20], "big") / DPI
    scale = BODY_WIDTH_IN / width_in
    printed = smallest * scale
    print(f"  {out.relative_to(HERE.parents[1])}\n"
          f"      作图 {width_in:.2f} in ×{scale:.2f} → 最小字 {smallest:g} pt 印出 {printed:.1f} pt")
    if printed < MIN_PRINT_PT:
        raise SystemExit(f"D6 不达标：{name} 最小的字印出来只有 {printed:.1f} pt，"
                         f"低于 {MIN_PRINT_PT} pt。请收窄画布或加大字号，不要去调排版 width%。")


# ── 第14讲 2.1：智能体与环境的交互闭环 ──────────────────────────────────
# 标签直接压在箭头弧顶上（白底），不再靠两侧留白摆放——留白一放大，字反而更小。
def agent_env_loop():
    fig, ax = canvas(6.0, 3.0)
    agent = (0.30, 0.70, 0.40, 0.22)
    env = (0.30, 0.06, 0.40, 0.22)
    box(ax, *agent, BLUE, [("智能体", 13, True), ("策略 $\\pi$：看到什么 → 做什么", 10, False)])
    box(ax, *env, GREEN, [("环境", 13, True), ("按物理规律推进一步", 10, False)])

    # 闭环走正交折线，两端都接在框的边中点上：右侧下行送动作，左侧上行回状态。
    elbow(ax, side(*agent, "right"), side(*env, "right"), RED, via="hvh", detour=0.87)
    label(ax, 0.87, 0.49, "动作 $a_t$", RED, 11, True, bg=True)
    elbow(ax, side(*env, "left"), side(*agent, "left"), BLUE, via="hvh", detour=0.13)
    label(ax, 0.13, 0.49, "新状态 $s_{t+1}$\n奖励 $r_{t+1}$", BLUE, 11, True, bg=True)
    label(ax, 0.5, 0.49, "如此循环，直到回合结束", "#888888", 10)
    save(fig, "lecture14", "fig141-agent-env-loop")


# ── 第14讲 2.2：动作空间分两类 ──────────────────────────────────────────
def action_space():
    fig, ax = canvas(6.8, 3.2)
    ax.plot([0.5, 0.5], [0.03, 0.89], color="#CCCCCC", linewidth=1.4, linestyle="--")

    label(ax, 0.25, 0.855, "离散动作空间", BLUE, 12, True)
    label(ax, 0.25, 0.765, "从有限个选项里挑一个", "#4A6A85", 10)
    for (bx, by, text) in [(0.10, 0.545, "←"), (0.28, 0.545, "→"),
                           (0.10, 0.395, "A"), (0.28, 0.395, "B")]:
        box(ax, bx, by, 0.12, 0.115, BLUE, [(text, 12, False)])
    label(ax, 0.25, 0.29, "动作个数有限，可以逐个比较好坏", "#555555", 10)
    label(ax, 0.25, 0.15, "例：手柄的四个键；", "#888888", 10)
    label(ax, 0.25, 0.06, "语言模型选下一个 token", "#888888", 10)

    label(ax, 0.75, 0.855, "连续动作空间", RED, 12, True)
    label(ax, 0.75, 0.765, "动作是一个实数区间里的取值", "#4A6A85", 10)
    ax.annotate("", xy=(0.94, 0.47), xytext=(0.57, 0.47),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=2.0))
    for tick in (0.60, 0.90):
        ax.plot([tick, tick], [0.445, 0.495], color=ORANGE, linewidth=2)
    ax.plot(0.74, 0.47, "o", color=RED, markersize=8)
    label(ax, 0.60, 0.395, "-1.0", INK, 10)
    label(ax, 0.90, 0.395, "+1.0", INK, 10)
    label(ax, 0.74, 0.555, "任一实数", RED, 10)
    label(ax, 0.75, 0.29, "取值有无穷多个，没法逐个枚举", "#555555", 10)
    label(ax, 0.75, 0.15, "例：方向盘转角；", "#888888", 10)
    label(ax, 0.75, 0.06, "G1 的 29 个关节目标量", "#888888", 10)
    save(fig, "lecture14", "fig-action-space")


# ── 第14讲 2.3：折扣因子决定"看多远" ────────────────────────────────────
# 这张图存在的理由：回报的求和公式给出的是代数，看不出 γ 到底把视野截在哪里。
# 左图画权重衰减曲线，把"越接近 1 越看长远"变成可看的三条线；右图把 γ=0.99 的
# 有效视野贴到一整个回合上，说明它只覆盖前十分之一——正文那句"不是几乎
# 一视同仁地看完整段"的依据就在这里。公式与图各说各的，不重复。
def discount_horizon():
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.0),
                                      gridspec_kw={"width_ratios": [1.1, 1]})

    # 三条曲线直接在自己旁边标名字，不用图例——图例摆在哪都会压住下面那两处刻度标注。
    steps = range(0, 301)
    for gamma, color, note, tx, ty in ((0.90, ORANGE, "只顾眼前", 45, 0.13),
                                       (0.99, BLUE, "本讲取值", 150, 0.30),
                                       (0.999, GREEN, "极重长期", 88, 0.925)):
        weights = [gamma ** k for k in steps]
        left.plot(steps, weights, color=color, linewidth=2.0)
        left.text(tx, ty, f"$\\gamma={gamma}$（{note}）", fontsize=10, color=color, ha="left")

    # 两个好记的刻度：半衰期 ln2/(1-γ)≈69 步，有效时间尺度 1/(1-γ)=100 步。
    for k, w, text in ((69, 0.5, "半衰期 69 步"), (100, 0.99 ** 100, "有效视野 100 步")):
        left.plot([k, k], [0, w], color=BLUE, linewidth=1.0, linestyle=":")
        left.plot([0, k], [w, w], color=BLUE, linewidth=1.0, linestyle=":")
        left.plot(k, w, "o", color=BLUE, markersize=4)
        left.annotate(text, (k, w), textcoords="offset points", xytext=(6, 7),
                      fontsize=10, color=BLUE)

    left.set_xlabel("往后第 $k$ 步", fontsize=10)
    left.set_ylabel("这一步奖励在回报里的权重 $\\gamma^{k}$", fontsize=10)
    left.set_title("$\\gamma$ 决定未来的奖励还剩多少分量", fontsize=11, color="#444444")
    left.set_xlim(0, 300); left.set_ylim(0, 1.02)
    left.set_xticks([0, 100, 200, 300])
    left.tick_params(labelsize=10)
    left.spines["top"].set_visible(False); left.spines["right"].set_visible(False)
    left.grid(alpha=0.25, linewidth=0.6)

    # 右图：把左图那三条曲线的有效视野 1/(1-γ) 贴到同一条回合轴上，一一对应。
    right.axis("off")
    right.set_xticks([]); right.set_yticks([])
    right.set_xlim(0, 1); right.set_ylim(0, 1)
    right.set_title("同样三个 $\\gamma$，视野贴到一整个回合上", fontsize=11, color="#444444")

    x0, span = 0.30, 0.68                       # 轴左端与全长，全长 = 1000 步
    right.annotate("", xy=(x0, 0.90), xytext=(x0 + span, 0.90),
                   arrowprops=dict(arrowstyle="<->", color=GREY, linewidth=1.2))
    right.text(x0 + span / 2, 0.99, "一个回合：1000 步 = 20 秒（50 Hz）",
               ha="center", va="top", fontsize=10, color="#555555")

    for i, (gamma, color, horizon) in enumerate(((0.90, ORANGE, 10),
                                                 (0.99, BLUE, 100),
                                                 (0.999, GREEN, 1000))):
        y = 0.60 - i * 0.20
        h = 0.14
        right.add_patch(FancyBboxPatch((x0, y), span, h,
                                       boxstyle="round,pad=0.003,rounding_size=0.008",
                                       linewidth=1.1, edgecolor="#CCCCCC", facecolor="#F7F7F7"))
        w = span * horizon / 1000
        right.add_patch(FancyBboxPatch((x0, y), w, h,
                                       boxstyle="round,pad=0.003,rounding_size=0.008",
                                       linewidth=1.5, edgecolor=color, facecolor=FILL[color]))
        right.text(x0 - 0.02, y + h / 2, f"$\\gamma={gamma}$", ha="right", va="center",
                   fontsize=10, color=color, fontweight="bold")
        # 视野占满整条轴时，字写到条里面，免得溢出画面。
        seconds = horizon / 50
        note = f"{horizon} 步 = {seconds:g} 秒"
        if w > 0.7 * span:
            right.text(x0 + w - 0.02, y + h / 2, note, ha="right", va="center",
                       fontsize=10, color=color)
        else:
            right.text(x0 + w + 0.02, y + h / 2, note, ha="left", va="center",
                       fontsize=10, color=color)

    right.text(x0 + span / 2, 0.05,
               "有效视野 $1/(1-\\gamma)$ 给的是步数，不是秒",
               ha="center", va="center", fontsize=10, color="#444444")

    fig.tight_layout()
    save(fig, "lecture14", "fig14-3-discount-horizon")


# ── 第14讲 2.4：策略把状态映射成动作分布 ────────────────────────────────
# 右端画出真正的分布形状而不是写"动作分布"四个字——"策略输出的是分布、动作是采出来的"
# 这件事，光靠 π(a|s) 这个记号看不出来，画出来就一目了然。
def policy_distribution():
    fig, ax = canvas(7.2, 3.0)
    label(ax, 0.30, 0.95, "策略：把状态映射成一个动作分布", "#333333", 13, True)
    box(ax, 0.02, 0.46, 0.24, 0.34, BLUE,
        [("状态 $s$", 12, True), ("关节角度、躯干姿态", 10, False), ("目标速度指令", 10, False)])
    box(ax, 0.32, 0.46, 0.20, 0.34, ORANGE,
        [("策略网络 $\\pi$", 12, True), ("参数 $\\theta$", 10, False)])
    arrow(ax, (0.265, 0.63), (0.315, 0.63), EDGE)
    arrow(ax, (0.525, 0.63), (0.575, 0.63), EDGE)

    # 右端：高斯密度 + 一个采样点。均值 μ 是网络算出来的，宽度 σ 就是探索强度。
    inset = fig.add_axes([0.585, 0.30, 0.40, 0.44])
    xs = [-3 + 6 * i / 400 for i in range(401)]
    mu, sigma = 0.2, 0.85
    ys = [2.718281828 ** (-((x - mu) ** 2) / (2 * sigma ** 2)) for x in xs]
    inset.fill_between(xs, ys, color=FILL[RED], alpha=0.9)
    inset.plot(xs, ys, color=RED, linewidth=2.0)
    inset.plot([mu, mu], [0, 1.0], color=RED, linewidth=1.2, linestyle=":")

    # σ 画成从均值往右量出去的单向跨度，且贴着曲线高度画，免得被读成 ±σ 或悬空。
    half = 2.718281828 ** -0.5
    inset.annotate("", xy=(mu - sigma, half), xytext=(mu, half),
                   arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.2))
    inset.text(mu - sigma / 2, half + 0.07, "$\\sigma$", ha="center", va="bottom",
               fontsize=10, color="#666666")
    inset.text(mu - 0.12, 1.04, "均值 $\\mu_\\theta(s)$", ha="right", va="center",
               fontsize=10, color=RED)

    sampled = 1.25
    inset.plot(sampled, 2.718281828 ** (-((sampled - mu) ** 2) / (2 * sigma ** 2)),
               "o", color="#333333", markersize=6, zorder=5)
    inset.annotate("采出来的动作 $a$", xy=(sampled + 0.06, 0.50), xytext=(1.85, 0.72),
                   fontsize=10, color="#333333", va="center",
                   arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))
    inset.set_title("动作分布 $\\pi_\\theta(a \\mid s)$", fontsize=11, color=RED, pad=16)
    inset.set_xlim(-3, 4.0); inset.set_ylim(0, 1.20)
    inset.set_xlabel("动作取值", fontsize=10, labelpad=1)
    inset.set_xticks([-2, 0, 2]); inset.tick_params(labelsize=10)
    inset.set_yticks([])
    for side in ("top", "right", "left"):
        inset.spines[side].set_visible(False)

    # 说明文字只占左半幅，避免横穿到分布图下方压住它的 x 轴标签。
    save(fig, "lecture14", "fig14-2-policy-distribution")


# ── 第14讲 3.4：on-policy 与 off-policy 的数据流对比 ────────────────────
def onoff_dataflow():
    fig, ax = canvas(7.8, 3.4)
    label(ax, 0.22, 0.955, "on-policy（第14讲、第15讲）", BLUE, 13, True)
    label(ax, 0.75, 0.955, "off-policy（第16讲）", RED, 13, True)
    ax.plot([0.46, 0.46], [0.02, 0.90], color="#DDDDDD", linewidth=1.4, linestyle="--")

    # 左：采样 → 更新 → 作废
    box(ax, 0.09, 0.68, 0.26, 0.13, BLUE, [("当前策略", 12, True)])
    box(ax, 0.07, 0.36, 0.30, 0.17, BLUE, [("新鲜数据", 12, True), ("（只来自当前策略）", 10, False)])
    arrow(ax, (0.17, 0.675), (0.17, 0.535), EDGE)
    label(ax, 0.135, 0.605, "采样", BLUE, 10, ha="right")
    arrow(ax, (0.27, 0.535), (0.27, 0.675), EDGE)
    label(ax, 0.305, 0.605, "更新一次", BLUE, 10, ha="left")
    arrow(ax, (0.22, 0.35), (0.22, 0.24), EDGE)
    label(ax, 0.22, 0.185, "更新完立即作废", INK, 10)
    label(ax, 0.22, 0.09, "策略一变，全部重采", "#999999", 10)

    # 右：三来源 → 经验池 → 目标策略。三条入池线各自竖直落到经验池顶边的三个等分点上，
    # 不用斜线汇聚——斜线在这种"多对一"里最容易看成随手连的。
    pool = (0.52, 0.45, 0.44, 0.185)
    sources = [(0.585, "旧策略经验", BLUE), (0.745, "人类示教", GREEN),
               (0.905, "人工干预", ORANGE)]
    for i, (cx, text, color) in enumerate(sources):
        src = (cx - 0.070, 0.75, 0.140, 0.10)
        box(ax, *src, color, [(text, 10, True)])
        # 落点与源框中心同一条竖线 ⇒ 三条都是直上直下，视觉上并列
        dst_t = (cx - pool[0]) / pool[2]
        connect(ax, src, "bottom", pool, "top", color, dst_t=dst_t, stub=0.03)

    box(ax, *pool, RED,
        [("replay buffer（经验池）", 12, True), ("长期保存，反复使用", 10, True)])
    policy = (0.575, 0.16, 0.31, 0.145)
    # 两条竖线落在同一对 x 上（pool 比 policy 宽，t 要各自换算），才是直上直下。
    for x, color, src_first in ((0.665, RED, True), (0.815, "#999999", False)):
        tp = (x - pool[0]) / pool[2]
        tq = (x - policy[0]) / policy[2]
        if src_first:
            connect(ax, pool, "bottom", policy, "top", color, src_t=tp, dst_t=tq, stub=0.03)
        else:
            connect(ax, policy, "top", pool, "bottom", color, src_t=tq, dst_t=tp, stub=0.03)
    label(ax, 0.645, 0.375, "随机抽 batch", RED, 10, ha="right")
    label(ax, 0.90, 0.375, "新经验入池", INK, 10, ha="left")
    box(ax, *policy, RED, [("目标策略", 12, True)])
    label(ax, 0.73, 0.075, "采数据的手 与 被训练的脑 解耦", INK, 10)
    save(fig, "lecture14", "fig-onoff-dataflow")


# ── 第14讲 5.3：A2C 的 Actor-Critic 分工 ───────────────────────────────
def actor_critic_a2c():
    """A2C 的四个角色与四条通路。

    版面按网格排：左列 Actor / Critic 上下对齐、等宽等高，中列环境与优势各占一行，
    行高与列宽都从同一组常量算出来，避免逐个框手填坐标排不齐。
    """
    fig, ax = canvas(7.8, 3.4)

    COL_L, COL_M, COL_R = 0.045, 0.375, 0.655
    W_SIDE, W_MID, W_ADV = 0.28, 0.20, 0.30
    ROW_TOP, ROW_BOT = 0.62, 0.10
    H = 0.19

    actor = (COL_L, ROW_TOP, W_SIDE, H)
    critic = (COL_L, ROW_BOT, W_SIDE, H)
    envb = (COL_M, ROW_TOP, W_MID, H)
    adv = (COL_R, ROW_BOT, W_ADV, 0.30)

    box(ax, *actor, BLUE,
        [("Actor（策略网络）", 11, True), ("$\\pi(a\\,|\\,s)$　出动作", 10, False)])
    box(ax, *envb, GREY, [("环境", 11, True), ("给出 $r$、$s'$", 10, False)])
    box(ax, *critic, GREEN,
        [("Critic（价值网络）", 11, True), ("$V(s)$　估状态价值", 10, False)])
    box(ax, *adv, ORANGE,
        [("优势 A：比平均水平好多少", 11, True),
         ("多步平滑版 $\\hat{A}^{\\mathrm{GAE}}$", 10, False),
         ("一步版 $A=r+\\gamma V(s')-V(s)$", 10, False)])

    # 紫色只用于这一条箭头与它的标签，不进 FILL，所以不加进模块级调色板。
    PURPLE = "#7D3C98"

    # 前向：actor 出动作进环境；环境的 r、s′ 进优势计算。
    connect(ax, actor, "right", envb, "left", "#555555")
    label(ax, (COL_L + W_SIDE + COL_M) / 2, ROW_TOP + H * 0.5 + 0.055, "$a$",
          "#555555", 10)
    connect(ax, envb, "right", adv, "top", "#555555", dst_t=0.3)
    label(ax, 0.635, ROW_TOP + 0.10, "$r,\; s'$", "#555555", 10, bg=True)

    # Critic 把 V 供给优势计算，走底行横带。
    connect(ax, critic, "right", adv, "left", GREEN, dst_t=0.4)
    label(ax, 0.50, ROW_BOT + H + 0.045, "$V(s),\; V(s')$", GREEN, 10)

    # 两条训练回流（图注：红的更新 actor，紫的更新 critic）各走一条外围通道：
    # 红的沿顶部横带回 Actor，紫的沿左列与中列之间的竖带下到 Critic。
    connect(ax, adv, "top", actor, "top", RED, src_t=0.7, lane=0.94)
    label(ax, 0.45, 0.965, "用 $A$ 加权策略梯度，更新 actor", RED, 11, True)
    connect(ax, envb, "bottom", critic, "top", PURPLE,
            src_t=0.3, dst_t=0.6, lane=0.345)
    label(ax, 0.175, 0.435, "critic 回归 returns，\n把 $V$ 估得更准", PURPLE, 11, True)

    save(fig, "lecture14", "fig-actor-critic-a2c")


# ── 第16讲 3.2：DDPG 的 Actor-Critic 分工 ──────────────────────────────
# 四个角色围成一个环：actor 的更新方向取决于 critic 此刻的打分，critic 的回归目标又
# 取决于 actor 此刻给出的 μ′(s′)。红箭头必须从 Critic 出发——∇a Q 是 critic 给的。
def actor_critic_ddpg():
    """DDPG 的四个角色与四条通路。

    与 A2C 同一套排法：左右两列、上下两行对齐，线一律用 connect() 从边中点垂直引出，
    通道位置在这里一次定死，不各自即兴挑。
    """
    fig, ax = canvas(7.8, 4.0)

    COL_L, COL_M, COL_R = 0.03, 0.24, 0.66
    ROW_TOP, ROW_BOT = 0.60, 0.12
    H = 0.26

    envb = (COL_L, ROW_TOP + 0.03, 0.14, 0.20)
    actor = (COL_M, ROW_TOP, 0.30, H)
    buf = (COL_R, ROW_TOP, 0.31, H)
    critic = (COL_M, ROW_BOT, 0.30, H)
    target = (COL_R, ROW_BOT, 0.31, H)

    box(ax, *envb, GREY, [("环境", 11, True), ("给 $s$、$r$", 10, False)])
    box(ax, *actor, BLUE,
        [("Actor（确定性策略）", 11, True), ("$a=\\mu(s)$", 10, False),
         ("一步吐出连续动作", 10, False)])
    box(ax, *buf, ORANGE,
        [("replay buffer", 11, True), ("$(s,a,r,s')$ 反复使用", 10, False)])
    box(ax, *critic, GREEN,
        [("Critic（动作价值）", 11, True), ("$Q(s,a)$", 10, False),
         ("学习主体，给 $(s,a)$ 打分", 10, False)])
    box(ax, *target, RED,
        [("贝尔曼回归目标", 11, True), ("$y = r + \\gamma\\,Q'(s',\\mu'(s'))$", 10, False),
         ("目标网络软更新，稳住目标", 10, False)])

    connect(ax, envb, "right", actor, "left", "#555555")
    label(ax, 0.205, ROW_TOP + H * 0.5 + 0.055, "$s$", "#555555", 10)
    connect(ax, actor, "right", buf, "left", "#555555")
    label(ax, 0.595, ROW_TOP + H * 0.5 + 0.05, "新经验入池", "#555555", 10, bg=True)

    # 经验池 → Critic：走右列与左列之间的竖带下来，不走斜线。
    # 走两行之间那条横带，从 Critic 顶边偏右进——不穿过下排任何框。
    connect(ax, buf, "bottom", critic, "top", ORANGE,
            src_t=0.25, dst_t=0.75, lane=0.505)
    label(ax, 0.72, 0.565, "抽 batch", ORANGE, 10)

    connect(ax, critic, "right", target, "left", GREEN)
    label(ax, 0.58, ROW_BOT + H * 0.5 + 0.055, "$Q$ 向 $y$ 回归", GREEN, 10)

    # actor 的更新回流：从 Critic 顶边垂直上去进 Actor 底边，走两框之间的空带。
    connect(ax, critic, "top", actor, "bottom", RED, src_t=0.25, dst_t=0.25, lw=2.2)
    label(ax, 0.135, 0.50, "沿 $\\nabla_a Q(s,a)$\n更新 actor，把 $Q$ 推高",
          RED, 11, True)

    save(fig, "lecture16", "fig162-actor-critic-ddpg")


# ── 第16讲 2.2：经验池的存与取 ─────────────────────────────────────────
# 存的顺序和取的顺序是两回事——上排按时间成片、下排随机打散，这个对照公式写不出来。
def replay_buffer():
    fig, ax = canvas(7.8, 3.1)
    runs = [(9, BLUE), (6, GREEN), (7, BLUE), (8, ORANGE)]
    cells = [color for n, color in runs for _ in range(n)]
    n = len(cells)

    x0, x1, cw = 0.02, 0.98, (0.98 - 0.02) / n
    label(ax, 0.02, 0.955, "池内经验（按时间存放：相邻转移几乎一样，来源成片）",
          "#444444", 10, ha="left")
    for i, color in enumerate(cells):
        ax.add_patch(FancyBboxPatch((x0 + i * cw + cw * 0.12, 0.72), cw * 0.76, 0.15,
                                    boxstyle="round,pad=0.001,rounding_size=0.004",
                                    linewidth=0.8, edgecolor=color, facecolor=color))

    # 随机抽出的一个 batch：来源与时刻都打散，下排位置与上排一一连线。
    picked = [1, 6, 10, 13, 16, 19, 23, 27]
    bw = 0.062
    bx0 = (1.0 - len(picked) * bw - (len(picked) - 1) * 0.020) / 2
    for j, i in enumerate(picked):
        bx = bx0 + j * (bw + 0.020)
        ax.add_patch(FancyBboxPatch((bx, 0.20), bw, 0.15,
                                    boxstyle="round,pad=0.001,rounding_size=0.004",
                                    linewidth=0.8, edgecolor=cells[i], facecolor=cells[i]))
        ax.add_patch(FancyArrowPatch((x0 + (i + 0.5) * cw, 0.705), (bx + bw / 2, 0.36),
                                     connectionstyle="arc3,rad=0.18", arrowstyle="-",
                                     linewidth=0.8, color="#C9C9C9"))
    label(ax, 0.5, 0.115, "随机抽出的一个 batch（时刻、回合、来源全打散）", INK, 10)

    # 图例放在最下面一行，不再贴着右侧的柱子。
    lx = 0.20
    for text, color in (("旧策略经验", BLUE), ("人类示教", GREEN), ("人工干预", ORANGE)):
        ax.add_patch(FancyBboxPatch((lx, 0.015), 0.022, 0.045,
                                    boxstyle="round,pad=0.001,rounding_size=0.004",
                                    linewidth=0.8, edgecolor=color, facecolor=color))
        label(ax, lx + 0.032, 0.038, text, "#555555", 10, ha="left")
        lx += 0.22
    save(fig, "lecture16", "fig162-replay-buffer")


# ── 第16讲 2.3：稀疏奖励沿转移链回传 ───────────────────────────────────
def q_propagation():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    groups = [("$Q(s_0,a_0)$\n甲：伸向方块", [0.0, 0.0, 0.81]),
              ("$Q(s_1,a_1)$\n乙：抵住方块推", [0.0, 0.9, 0.9]),
              ("$Q(s_2,a_2)$\n丙：方块进目标区", [1.0, 1.0, 1.0])]
    rounds = ["抽到丙之后", "再抽到乙之后", "再抽到甲之后"]
    alphas = [0.30, 0.62, 1.0]

    width = 0.24
    for gi, (name, values) in enumerate(groups):
        for ri, v in enumerate(values):
            ax.bar(gi + (ri - 1) * width, v, width * 0.88, color=RED, alpha=alphas[ri],
                   label=rounds[ri] if gi == 0 else None, edgecolor="none")

    # 每往左传一格乘一次 γ；箭头统一走柱子上方的空带，不压住柱顶的数字。
    for x_from, x_to in ((2 - width, 1.0), (1.0, 0 + width)):
        ax.annotate("", xy=(x_to, 1.12), xytext=(x_from, 1.12),
                    arrowprops=dict(arrowstyle="-|>", color="#9B3626", linewidth=1.4,
                                    connectionstyle="arc3,rad=0.30"))
        ax.text((x_from + x_to) / 2, 1.30, "$\\times\\,\\gamma$", ha="center", fontsize=10, color="#9B3626")
    for x, v in ((0 + width, 0.81), (1.0, 0.9), (2 - width, 1.0)):
        ax.text(x, v + 0.03, f"{v:g}", ha="center", fontsize=10, color="#9B3626",
                fontweight="bold")

    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("Q 值", fontsize=10)
    ax.set_ylim(0, 1.85); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=10)
    ax.set_title("一个终点奖励沿转移链回传成价值坡道（$\\gamma=0.9$；起点三个 $Q$ 全为 0）",
                 fontsize=11, color="#444444")
    ax.legend(fontsize=10, frameon=False, ncol=3, loc="upper center", borderpad=0.2,
              handlelength=1.2, columnspacing=1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "lecture16", "fig162-q-propagation")


# ── 第16讲 2.4：CartPole 任务长什么样 ──────────────────────────────────
# 正文已经把"小车 + 杆子 + 左右推"说清楚了，所以图要给的是文字给不了的两样东西：
# (1) 两个终止阈值（±2.4 出界、±12° 倾角）在空间上各管哪一块——文字只能分开报数字，
#     图能让读者一眼看出"角度那道线比位置那道线先撞上"；
# (2) 推车方向与杆子倒向的反直觉对应——右半两个姿态并排，杆往右倒时车也往右追。
# 数字口径：CartPole-v1 的 ±2.4 与 ±0.21 rad(≈12°)，与脚本 OBS_LOW/HIGH 一致；
# 分桶数 8 取自同一脚本的 NUM_BINS。
def cartpole_task():
    fig, ax = canvas(7.8, 4.4)
    W_IN, H_IN = 7.8, 4.4          # 画角度要按画布长宽比折算，否则 12° 画出来不是 12°

    def tip(px, py, deg, inches):
        """从 (px,py) 出发、与竖直方向成 deg 度、视觉长度 inches 的那个端点。"""
        rad = math.radians(deg)
        return px + inches * math.sin(rad) / W_IN, py + inches * math.cos(rad) / H_IN

    def draw_pole(px, py, deg, inches, color, lw=3.0):
        ex, ey = tip(px, py, deg, inches)
        ax.plot([px, ex], [py, ey], color=color, linewidth=lw, solid_capstyle="round")
        return ex, ey

    ax.plot([0.595, 0.595], [0.03, 0.90], color="#CCCCCC", linewidth=1.4, linestyle="--")

    # ── 左半：把两个阈值画在同一张场景里 ──────────────────────────────
    TRACK_Y, X0, XC, X1 = 0.35, 0.03, 0.26, 0.50      # 轨道；XC 是位置 0，两端是 ±2.4
    CART_X, CART_W, CART_H = 0.36, 0.072, 0.058
    PIVOT = (CART_X, TRACK_Y + CART_H)

    ax.plot([X0, X1], [TRACK_Y, TRACK_Y], color="#666666", linewidth=2.4)
    for bx, name in ((X0, "-2.4"), (X1, "+2.4")):
        ax.plot([bx, bx], [TRACK_Y - 0.06, TRACK_Y + 0.17], color=RED,
                linewidth=1.8, linestyle="--")
        label(ax, bx, TRACK_Y + 0.205, name, RED, 10, True)
    ax.plot([XC, XC], [TRACK_Y - 0.022, TRACK_Y + 0.022], color="#999999", linewidth=1.6)
    label(ax, XC, TRACK_Y - 0.052, "0", "#888888", 10)

    # ±12° 的扇形：杆尖只要越出这片浅橙，回合当场结束
    wedge = [PIVOT] + [tip(*PIVOT, d, 1.42) for d in range(-12, 13, 2)]
    ax.add_patch(Polygon(wedge, closed=True, facecolor="#FDF2E0",
                         edgecolor="none", zorder=0))
    for d in (-12, 12):
        ex, ey = tip(*PIVOT, d, 1.42)
        ax.plot([PIVOT[0], ex], [PIVOT[1], ey], color=ORANGE,
                linewidth=1.6, linestyle="--")
    ax.plot([PIVOT[0], PIVOT[0]], [PIVOT[1], tip(*PIVOT, 0, 1.42)[1]],
            color="#BBBBBB", linewidth=1.2, linestyle=":")

    plain_box(ax, CART_X - CART_W / 2, TRACK_Y, CART_W, CART_H, GREY)
    pole_x, pole_y = draw_pole(*PIVOT, 6, 1.28, "#8B5A2B")
    ax.plot(*PIVOT, "o", color="#333333", markersize=9, markerfacecolor="white",
            markeredgewidth=1.8, zorder=6)

    label(ax, 0.30, 0.815, "(3) 杆角度 $\\theta$ 只有 $\\pm 12^\\circ$ 可用", ORANGE, 10, True)
    label(ax, 0.30, 0.768, "（$\\approx 0.21$ rad）越出即回合结束", ORANGE, 10)

    # ④ 角速度画成杆尖上的一小段转向弧，落在它描述的那个量上
    arrow(ax, (pole_x + 0.008, pole_y + 0.030), (pole_x + 0.042, pole_y - 0.030),
          EDGE, rad=-0.5, lw=1.4)
    label(ax, pole_x + 0.050, pole_y - 0.048, "(4) 杆角速度 $\\dot{\\theta}$",
          "#8B5A2B", 10, ha="left")

    # 自由转轴是这个任务的题眼：没有任何电机去扶那根杆
    arrow(ax, (0.075, 0.585), (CART_X - 0.020, PIVOT[1] + 0.014), EDGE, rad=-0.22, lw=1.4)
    label(ax, 0.030, 0.615, "自由转轴：没有电机", "#333333", 10, True, ha="left")

    ax.annotate("", xy=(CART_X, TRACK_Y - 0.115), xytext=(XC, TRACK_Y - 0.115),
                arrowprops=dict(arrowstyle="<|-|>", color=BLUE, linewidth=1.5))
    label(ax, 0.31, TRACK_Y - 0.175, "(1) 车位置 x", BLUE, 10, True)
    arrow(ax, (CART_X + 0.045, TRACK_Y + 0.028), (CART_X + 0.108, TRACK_Y + 0.028), EDGE, lw=1.5)
    label(ax, CART_X + 0.118, TRACK_Y + 0.028, "(2) 车速度 $\\dot{x}$", BLUE, 10,
          ha="left", bg=True)


    # ── 右半上：动作只有两档 ──────────────────────────────────────────
    label(ax, 0.80, 0.845, "动作只有两档，没有第三个选项", "#333333", 11, True)
    for ax_x, txt, color in ((0.70, "0：左推 ←", BLUE), (0.90, "1：右推 →", RED)):
        box(ax, ax_x - 0.085, 0.735, 0.17, 0.075, color, [(txt, 11, True)])

    # ── 右半下：反直觉的那一下 ────────────────────────────────────────
    label(ax, 0.80, 0.605, "杆往右倒，车要往哪边推？", "#333333", 11, True)
    for cx, deg, color, cap1, cap2 in (
            (0.685, 11, RED, "倒到 11°", "再偏一点就出界"),
            (0.905, 4, GREEN, "回到 4°", "车追到了杆下面")):
        ax.plot([cx - 0.055, cx + 0.055], [0.36, 0.36], color="#999999", linewidth=1.8)
        plain_box(ax, cx - 0.030, 0.36, 0.060, 0.048, GREY)
        draw_pole(cx, 0.408, deg, 1.05, color, lw=2.6)
        label(ax, cx, 0.315, cap1, color, 10, True)
        label(ax, cx, 0.268, cap2, INK, 10)
    arrow(ax, (0.762, 0.435), (0.828, 0.435), EDGE, lw=2.0)
    label(ax, 0.795, 0.478, "往右推", RED, 10, True)

    save(fig, "lecture16", "fig16-cartpole-task")


# ── 第13讲 2.6：长程操作的四种范式 ─────────────────────────────────────
# 按参考文献 9（Long-VLA）图 1 的四类范式重画。原图是论文整页截图、连图题一起截了进来，
# 分辨率也不够印；重画一版把三项勾叉与"掩码闸门到底掩掉了什么"讲清楚。
def long_horizon_paradigms():
    fig, ax = canvas(7.8, 5.6)

    def marks(cx, y, flags):
        """一格底部的三项勾叉，是这张图真正要对照的东西。"""
        items = list(zip(("统一模型", "长程", "技能串接"), flags))
        widths = [0.150, 0.100, 0.150]
        x = cx - sum(widths) / 2
        for (name, ok), w in zip(items, widths):
            color = GREEN if ok else RED
            label(ax, x, y, name, color, 10, ha="left")
            # 对勾用线段画，不用 ✓ 字形：TimesSong 没有 U+2713，用字形会渲成豆腐块。
            mark_x = x + 0.034 + 0.019 * len(name)
            if ok:
                ax.plot([mark_x, mark_x + 0.007, mark_x + 0.020],
                        [y + 0.001, y - 0.007, y + 0.013],
                        color=color, linewidth=1.6, solid_capstyle="round", zorder=5)
            else:
                for dx in (0.0, 0.017):
                    ax.plot([mark_x + dx, mark_x + 0.017 - dx],
                            [y - 0.007, y + 0.011],
                            color=color, linewidth=1.6, solid_capstyle="round", zorder=5)
            x += w

    def feed(cx, y0, with_gate):
        """从观测落到策略的一条输入通路；带闸门的多插一道分阶段掩码。"""
        if with_gate:
            arrow(ax, (cx, y0 + 0.278), (cx, y0 + 0.250), EDGE)
            plain_box(ax, cx - 0.0375, y0 + 0.195, 0.075, 0.048, ORANGE)
            label(ax, cx, y0 + 0.219, "掩码闸门", INK, 10)
            arrow(ax, (cx, y0 + 0.190), (cx, y0 + 0.166), EDGE)
        else:
            arrow(ax, (cx, y0 + 0.278), (cx, y0 + 0.166), EDGE)

    def cell(x0, y0, tag, flags, draw):
        w, h = 0.475, 0.38
        plain_box(ax, x0, y0, w, h, GREY, fill="#FCFCFC", lw=1.0)
        label(ax, x0 + w / 2, y0 + 0.345, tag, "#333333", 10, True)
        label(ax, x0 + w / 2, y0 + 0.295, "一整条长任务的观测", INK, 10)
        draw(x0 + w / 2, y0)
        marks(x0 + w / 2, y0 + 0.025, flags)

    def one_policy(cx, y0, with_gate):
        feed(cx, y0, with_gate)
        plain_box(ax, cx - 0.16, y0 + 0.105, 0.32, 0.055, BLUE)
        label(ax, cx, y0 + 0.1325, "一个 VLA 策略", INK, 10)
        label(ax, cx, y0 + 0.070, "$a^{t-1},\\; a^{t},\\; a^{t+1}$", INK, 10)

    def two_policies(cx, y0, with_gate):
        for dx, name in ((-0.115, "移动策略"), (0.115, "交互策略")):
            feed(cx + dx, y0, with_gate)
            plain_box(ax, cx + dx - 0.09, y0 + 0.105, 0.18, 0.055, BLUE)
            label(ax, cx + dx, y0 + 0.1325, name, INK, 10)
        # 两个策略之间是一道接缝：不是同一个模型，交接处要单独想办法。
        ax.plot([cx, cx], [y0 + 0.098, y0 + 0.168], color=RED, linewidth=1.4, linestyle=(0, (3, 2)))
        label(ax, cx, y0 + 0.070, "两个模型，中间有接缝" if not with_gate else "接缝靠输入级适配接住",
              RED, 10)

    cell(0.010, 0.575, "(a) 朴素端到端 VLA", (True, False, False),
         lambda cx, y0: one_policy(cx, y0, with_gate=False))
    cell(0.515, 0.575, "(b) 两段式：拆成两个策略", (False, True, False),
         lambda cx, y0: two_policies(cx, y0, with_gate=False))
    cell(0.010, 0.160, "(c) 两段式 + 输入级适配", (False, True, True),
         lambda cx, y0: two_policies(cx, y0, with_gate=True))
    cell(0.515, 0.160, "(d) 端到端 + 输入级适配（Long-VLA）", (True, True, True),
         lambda cx, y0: one_policy(cx, y0, with_gate=True))

    label(ax, 0.5, 0.095, "掩码闸门 = 分阶段输入掩码（phase-aware masking）",
          "#8A5A08", 10, True)
    save(fig, "lecture13", "fig13-2-long-horizon-paradigms")


# ── 第13讲 4.1：三条线切的是同一刀，刀口位置各不相同 ───────────────────
# 三条 bullet 用文字说不清的是**几何关系**：同一个"切开"的招式，落在三条链路的
# 不同维度上（时间 / 信息 / 职责）。三格共用同一套视觉语法，让"同一个招式"由图形自己说。
def one_cut_three_lines():
    fig, ax = canvas(7.8, 3.6)

    def cut(x0, y0, x1, y1):
        """统一的刀口符号：红色虚线 + 中点一个红菱形。"""
        ax.plot([x0, x1], [y0, y1], color=RED, linewidth=1.5, linestyle=(0, (3, 2)))
        ax.plot((x0 + x1) / 2, (y0 + y1) / 2, marker="D", color=RED, markersize=5)

    # 实时线：刀切在时间上。
    label(ax, 0.16, 0.885, "实时线：切在时间上", BLUE, 11, True)
    label(ax, 0.16, 0.705, "正在执行的动作块", "#999999", 10)
    plain_box(ax, 0.03, 0.545, 0.115, 0.085, GREY, fill="#EDEDED")
    label(ax, 0.0875, 0.5875, "已冻结", INK, 10)
    plain_box(ax, 0.175, 0.545, 0.115, 0.085, BLUE)
    label(ax, 0.2325, 0.5875, "可重画", INK, 10)
    cut(0.16, 0.50, 0.16, 0.675)
    label(ax, 0.16, 0.465, "时间轴 →", "#999999", 10)
    label(ax, 0.16, 0.335, "刀口：软过渡段", RED, 10)
    label(ax, 0.16, 0.245, "左边来不及改，", INK, 10)
    label(ax, 0.16, 0.165, "右边重新生成", INK, 10)

    # 记忆线：刀切在信息上。
    label(ax, 0.5, 0.885, "记忆线：切在信息上", GREEN, 11, True)
    label(ax, 0.435, 0.775, "每帧刷新", "#999999", 10)
    label(ax, 0.625, 0.775, "写入与巩固", "#999999", 10)
    plain_box(ax, 0.375, 0.665, 0.12, 0.075, BLUE)
    label(ax, 0.435, 0.7025, "当下单帧", INK, 10)
    plain_box(ax, 0.565, 0.665, 0.12, 0.075, GREEN)
    label(ax, 0.625, 0.7025, "记忆库", INK, 10)
    cut(0.53, 0.615, 0.53, 0.79)
    # 两条汇入线走正交：竖直下来，再横到决策框顶边的两个点上。
    for x_from, x_to in ((0.435, 0.500), (0.625, 0.560)):
        _polyline(ax, [(x_from, 0.660), (x_from, 0.605), (x_to, 0.605), (x_to, 0.565)],
                  EDGE, 1.4, 0.012)
    plain_box(ax, 0.465, 0.485, 0.13, 0.075, GREY, fill="#F0F0F0")
    label(ax, 0.53, 0.5225, "决策", INK, 10)
    label(ax, 0.53, 0.335, "刀口：检索 + 门控融合", RED, 10)
    label(ax, 0.53, 0.245, "两边各存各的，", INK, 10)
    label(ax, 0.53, 0.165, "各有各的更新节奏", INK, 10)

    # 人形线：刀切在职责上。
    label(ax, 0.845, 0.885, "人形线：切在职责上", ORANGE, 11, True)
    plain_box(ax, 0.775, 0.735, 0.14, 0.070, GREY, fill="#F0F0F0")
    label(ax, 0.845, 0.770, "语言指令", INK, 10)
    arrow(ax, (0.845, 0.730), (0.845, 0.705), EDGE)
    plain_box(ax, 0.735, 0.625, 0.22, 0.075, ORANGE)
    label(ax, 0.845, 0.6625, "S2：懂任务的大脑", INK, 10)
    cut(0.715, 0.575, 0.975, 0.575)
    plain_box(ax, 0.735, 0.485, 0.22, 0.075, ORANGE)
    label(ax, 0.845, 0.5225, "S1：会走路的身体", INK, 10)
    arrow(ax, (0.845, 0.480), (0.845, 0.450), EDGE)
    label(ax, 0.845, 0.415, "电机指令", INK, 10)
    label(ax, 0.845, 0.335, "刀口：潜在 verb / 分块命令", RED, 10)
    label(ax, 0.845, 0.245, "上面不学走路，", INK, 10)
    label(ax, 0.845, 0.165, "下面不懂任务", INK, 10)

    save(fig, "lecture13", "fig13-4-one-cut-three-lines")


# ── 第8讲 1.1：生成机器人运动的两支 ────────────────────────────────────
# 原来那张英文截图只是把"分成两支"这件事画了出来，看不出这一讲站在哪一支上。
# 重画时把走向标出来：右支加粗、底部一条"本讲与第9–13讲全在这一支上"的落点。
def motion_taxonomy():
    fig, ax = canvas(7.0, 3.8)
    root = (0.355, 0.865, 0.29, 0.105)
    left = (0.045, 0.645, 0.42, 0.145)
    right = (0.535, 0.645, 0.42, 0.145)
    box(ax, *root, GREY, [("生成机器人运动", 12, True)])
    box(ax, *left, BLUE,
        [("显式建模 explicit modeling", 11, True), ("人写规则一步步算出来", 10, False)])
    box(ax, *right, GREEN,
        [("隐式建模 implicit modeling", 11, True), ("从数据里学出来", 10, False)])

    # 分叉走正交：从根框底边中点下来，沿一条横带分左右，再垂直进两个子框顶边。
    for child in (left, right):
        connect(ax, root, "bottom", child, "top", EDGE, stub=0.035, lane=0.825)

    # 每组的竖干落在子框外侧的空白里（早先取在框内 0.085，线就贴着框左边缘跑）。
    for parent, xbox, color, items in (
            (left, 0.130, BLUE, ["正运动学", "逆运动学", "规划与控制（RRT、MPC）"]),
            (right, 0.620, GREEN, ["深度强化学习", "从专家示教中学习"])):
        xline = xbox - 0.055          # 落在子框左侧的空白，不贴任何框边
        for i, text in enumerate(items):
            y = 0.500 - i * 0.110
            child = (xbox, y, 0.335, 0.086)
            plain_box(ax, *child, color)
            label(ax, xbox + 0.1675, y + 0.043, text, INK, 10)
            ax.plot([xline, xline], [parent[1] - 0.030, y + 0.043],
                    color=EDGE, linewidth=BOX_LW)
            arrow(ax, (xline, y + 0.043), (xbox - 0.008, y + 0.043), lw=BOX_LW)
        ax.plot([xline, xline], [parent[1], parent[1] - 0.030], color=EDGE, linewidth=BOX_LW)


    plain_box(ax, 0.535, 0.115, 0.42, 0.175, GREEN, fill="#D8EDE1", lw=2.0)
    label(ax, 0.745, 0.235, "本讲起走这一支", INK, 11, True)
    label(ax, 0.745, 0.155, "观测直接进、动作直接出", INK, 10)
    save(fig, "lecture08", "fig08-1-motion-taxonomy")


# ── 第8讲 2.3：策略的四档谱系 ──────────────────────────────────────────
# 正文用整段描述了一条谱系却没有图。图要多给的是**递进关系**：每往右一档，
# 观测里多进来一样东西，能听懂的指令就宽一层——这条"加法"是文字列举给不出的。
def policy_spectrum():
    fig, ax = canvas(7.8, 3.4)

    stages = [(0.015, BLUE, "状态策略", "state\npolicy", "只有本体数字", "看不见画面", "—"),
              (0.262, GREEN, "视觉运动策略", "visuomotor\npolicy", "＋相机图像", "于是看得见",
               "ACT、\nDiffusion Policy"),
              (0.509, ORANGE, "语言条件策略", "language-\nconditioned", "＋语言指令", "于是听得懂",
               "RT-1"),
              (0.756, RED, "视觉-语言-动作", "VLA", "＋大模型统一表征", "于是会推理",
               "$\\pi_0$")]
    w = 0.229
    for x0, color, name, en, add, gain, cases in stages:
        plain_box(ax, x0, 0.325, w, 0.535, color)
        label(ax, x0 + w / 2, 0.805, name, color, 12, True)
        label(ax, x0 + w / 2, 0.752, en, "#888888", 10, va="top")
        label(ax, x0 + w / 2, 0.575, add, INK, 11, True)
        label(ax, x0 + w / 2, 0.505, gain, INK, 10)
        label(ax, x0 + w / 2, 0.440, "代表：" + cases if cases != "—" else "",
              "#666666", 10, va="top")

    ax.annotate("", xy=(0.985, 0.245), xytext=(0.015, 0.245),
                arrowprops=dict(arrowstyle="-|>", color="#999999", linewidth=1.6))
    label(ax, 0.13, 0.175, "只看数字", "#999999", 10)
    label(ax, 0.87, 0.175, "看图、听话、会推理", "#999999", 10)

    ax.plot([0.756, 0.756], [0.075, 0.235], color=RED, linewidth=1.4, linestyle=(0, (3, 2)))
    label(ax, 0.985, 0.100, "第11讲起从这里展开", RED, 11, True, ha="right")
    save(fig, "lecture08", "fig08-2-policy-spectrum")


# ── 第8讲 3 开场：训练到部署的来回路 ────────────────────────────────────
# 回收进自绘源，顺手修掉左侧竖排标签贴着画布边缘、"时"字被裁掉一角的老毛病：
# 标签不再竖排，改成写在每一排上方的行首。
def train_deploy_roundtrip():
    fig, ax = canvas(7.8, 3.4)
    upper = ["示教数据", "统一动作口径\n（绝对 / 相对）", "归一化\n（各维压到 [-1, 1]）", "网络\n（训练）"]
    lower = ["机器人执行\n（裁剪 / 限位兜底）", "还原动作口径\n（相对口径需累加）",
             "反归一化\n（乘回 $\\sigma$、加回 $\\mu$）", "网络\n（推理输出）"]

    label(ax, 0.015, 0.935, "训练时：数据一路进网络", BLUE, 12, True, ha="left")
    label(ax, 0.015, 0.235, "部署时：输出必须原路逆着走回来", RED, 12, True, ha="left")

    w, gap = 0.213, 0.031
    for i, text in enumerate(upper):
        x0 = 0.015 + i * (w + gap)
        plain_box(ax, x0, 0.665, w, 0.225, BLUE)
        label(ax, x0 + w / 2, 0.7775, text, INK, 10)
        if i:                       # 上排从左往右：数据进网络
            arrow(ax, (x0 - gap + 0.003, 0.7775), (x0 - 0.003, 0.7775), EDGE)
    for i, text in enumerate(lower):
        x0 = 0.015 + i * (w + gap)
        plain_box(ax, x0, 0.315, w, 0.225, RED)
        label(ax, x0 + w / 2, 0.4275, text, "#8E3226", 10)
        if i:                       # 下排从右往左：输出原路走回来
            arrow(ax, (x0 - 0.003, 0.4275), (x0 - gap + 0.003, 0.4275), EDGE)

    # 两条虚线就是本节要拆的两处衔接：断在这里，网络再好也没用。
    for i, text in enumerate(("两端必须用\n同一套口径", "两端必须用\n同一套统计量")):
        x = 0.015 + (i + 1) * (w + gap) + w / 2
        ax.plot([x, x], [0.545, 0.660], color="#888888", linewidth=1.4, linestyle=(0, (4, 3)))
        label(ax, x + 0.012, 0.6025, text, INK, 10, ha="left")

    save(fig, "lecture08", "fig08-3-train-deploy-roundtrip")


# ── 第8讲 3.4：各维量纲差异与归一化后的统一区间 ──────────────────────────
# 回收进自绘源，顺手把原来那根紫色条形换回全书五色。
def normalization_ranges():
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    rows = [("夹爪开合", 0.0, 1.0, ORANGE), ("关节角", -3.14, 3.14, BLUE),
            ("相机像素", 0.0, 255.0, GREEN)]

    for i, (name, lo, hi, color) in enumerate(rows):
        left.barh(i, hi - lo, left=lo, height=0.42, color=FILL[color], edgecolor=color, linewidth=2)
        right.barh(i, 2.0, left=-1.0, height=0.42, color=FILL[color], edgecolor=color, linewidth=2)
    for axis, title, xlabel in ((left, "原始值：各维尺度天差地别", "原始数值"),
                                (right, "各维各自归一化后：统一落到 [-1, +1]", "归一化后的数值")):
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([r[0] for r in rows], fontsize=10)
        axis.set_title(title, fontsize=11, color="#444444")
        axis.set_xlabel(xlabel, fontsize=10)
        axis.tick_params(labelsize=10)
        axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    left.set_xlim(-20, 275)
    right.set_xlim(-1.6, 1.6)
    right.set_xticks([-1, 0, 1]); right.set_xticklabels(["-1", "0", "+1"])
    right.axvline(0, color="#CCCCCC", linewidth=1.0, linestyle=":")
    fig.tight_layout()
    save(fig, "lecture08", "fig08-3-normalization-ranges")


# ── 第8讲 3.4：尺度错位的两种症状 ───────────────────────────────────────
# 回收进自绘源，顺手修掉左图那处必修缺陷：说明文字原来被图例框整个压住。
# 现在两条线各自旁标、图例删掉，说明文字落在曲线下方的空白区。
def scale_error_symptoms():
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    target, start, steps = 1.2, 0.3, 40

    def rollout(gain):
        pos, track = start, [start]
        for _ in range(steps - 1):
            pos += gain * (target - pos)
            track.append(pos)
        return track

    for axis, gain, title in ((left, 1.99, "动作被系统性放大"),
                              (right, 0.008, "动作被系统性缩小")):
        axis.axhline(target, color=GREEN, linewidth=2.0, linestyle="--")
        # 左图那条过冲曲线会从这行字上穿过去，所以给它加白底压住——原来的缺陷是被图例遮住，
        # 只把图例删掉还不够，字仍旧糊在曲线里。
        axis.text(39, target + 0.13, "本该到达的目标", ha="right", fontsize=10, color=GREEN,
                  zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        axis.plot(range(steps), rollout(gain), color=RED, linewidth=2.0)
        axis.set_title(title, fontsize=11, color="#444444")
        axis.set_xlabel("时间步", fontsize=10)
        axis.set_ylabel("关节角（rad）", fontsize=10)
        axis.set_xlim(-1, 40); axis.set_ylim(-0.05, 2.5)
        axis.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
        axis.tick_params(labelsize=10)
        axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    left.text(20, 2.30, "实际执行", ha="center", fontsize=10, color=RED)
    right.text(20, 0.62, "实际执行", ha="center", fontsize=10, color=RED)
    fig.tight_layout()
    save(fig, "lecture08", "fig08-3-scale-error-symptoms")


# ── 第9讲 2.1：采集方法的权衡地图 ───────────────────────────────────────
# 2.1 那张表给的是逐字段属性；图要给的是**相对位置**——二十来个方法落在平面上，
# 从左上到右下连成一条带，"越可执行越不可规模化"这件事才看得见。
# 两条轴都取自那张表自己的列（机器人是否在回路 / 便携·可规模化）。
def teleop_tradeoff_map():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # (x=机器人在回路的程度, y=可规模化程度, 名字, 标签相对点的偏移)
    groups = [
        (BLUE, "真机数据", [
            (9.45, 1.5, "KineDex", (0.16, 0.00)), (9.05, 0.6, "KineSoft", (0.16, 0.00)),
            (9.30, 2.6, "Mobile ALOHA", (0.16, 0.00)), (8.35, 3.2, "U-Arm（异构主从）", (0.16, 0.00)),
            (7.95, 2.1, "TeleMoMa", (-0.16, 0.00))]),
        (ORANGE, "通用接口遥操作", [
            (7.35, 3.9, "Bunny-VisionPro", (-0.16, 0.00)),
            (7.05, 4.8, "Open-TeleVision", (-0.16, 0.00)),
            (6.60, 5.5, "OpenTeach", (0.16, 0.00)), (6.15, 4.4, "手柄 joycon", (-0.16, 0.00)),
            (5.75, 6.4, "RoboTurk 手机众包", (0.16, 0.00))]),
        (GREEN, "人类代理数据", [
            (5.05, 5.6, "PIKA", (0.16, 0.00)), (4.75, 7.2, "UMI", (-0.16, 0.00)),
            (4.30, 7.9, "FastUMI", (0.16, 0.00)), (4.10, 6.4, "AnyTeleop", (0.16, 0.00)),
            (3.70, 7.0, "ARCap", (-0.16, 0.00)), (3.95, 5.2, "DOGlove", (-0.16, 0.00)),
            (3.30, 5.9, "U-Arm Humanoid", (-0.16, 0.00)),
            (2.95, 7.6, "HumanPlus", (-0.16, 0.00))]),
        (GREY, "无机器人数据", [
            (1.75, 8.9, "In-N-On", (0.16, 0.00)), (1.15, 9.6, "Ctrl-World", (0.16, 0.00))]),
    ]

    # 那条负相关带：从左上（轻、可规模化）到右下（重、可执行）。
    ax.fill_between([0.4, 11.8], [10.6, 0.6], [8.4, -1.6], color="#F2F2F2", zorder=0)
    ax.text(2.6, 3.1, "越往右下：轨迹越能直接执行，\n但越贵、越不便携\n\n"
                      "越往左上：越轻越能规模化，\n但越需要重定向与后处理",
            fontsize=10, color="#888888", ha="center", va="center", zorder=1)

    for color, name, points in groups:
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=46,
                   facecolor=FILL[color], edgecolor=color, linewidth=1.6, zorder=3)
        for x, y, text, (dx, dy) in points:
            ax.text(x + dx, y + dy, text, fontsize=10, color=color, zorder=4,
                    ha="left" if dx > 0 else "right", va="center")
        ax.scatter([], [], s=46, facecolor=FILL[color], edgecolor=color, linewidth=1.6, label=name)

    # 本课入口单独高亮：同一张图上标出"我们从哪儿进"。
    ax.scatter([8.75], [1.9], s=190, facecolor=FILL[RED], edgecolor=RED, linewidth=2.2,
               marker="*", zorder=5)
    ax.annotate("SO-101 主从臂（本课入口）", xy=(8.62, 1.9), xytext=(6.5, 1.1),
                fontsize=10, color=RED, ha="right", va="center", zorder=5,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2))

    ax.set_xlabel("机器人在回路的程度（左：完全不在　→　右：全程在回路）", fontsize=10)
    ax.set_ylabel("便携 / 可规模化程度", fontsize=10)
    ax.set_xlim(0.3, 11.8); ax.set_ylim(-0.4, 10.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=10, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.10), handletextpad=0.3, columnspacing=1.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "lecture09", "fig09-2-teleop-tradeoff-map")


# ── 第9讲 3.1：LeRobot 的分层与 lerobot-record 穿过哪几层 ─────────────────
# 正文已经把六层逐层描述过一遍了，图要多给两件文字说不清的事：**调用方向**
# （上层调下层，箭头统一朝下）与**本讲那条路径**（record 命令实际穿过哪几层）。
def lerobot_layers():
    fig, ax = canvas(7.6, 4.8)
    label(ax, 0.03, 0.965, "LeRobot 自上而下的分层（箭头＝调用方向）",
          "#333333", 12, True, ha="left")
    layers = [("CLI 命令入口层", "scripts/", False),
              ("AI 策略模型层", "policies/：ACT、$\\pi_0$、SmolVLA", False),
              ("数据处理管线", "processor/ + datasets/", True),
              ("实物与仿真抽象层", "robots/、envs/", False),
              ("遥操作器", "teleoperators/：主臂、手柄", True),
              ("基础设施层", "configs/ + utils/ + transport/", False)]

    x0, w, h, gap = 0.03, 0.60, 0.105, 0.030
    tops = []
    for i, (name, detail, mine) in enumerate(layers):
        y = 0.795 - i * (h + gap)
        tops.append(y + h / 2)
        color = GREEN if mine else GREY
        plain_box(ax, x0, y, w, h, color, fill=FILL[GREEN] if mine else "#F6F6F6",
                  lw=2.0 if mine else 1.2)
        label(ax, x0 + 0.02, y + h / 2, name, GREEN if mine else "#333333", 11, mine, ha="left")
        label(ax, x0 + w - 0.02, y + h / 2, detail, INK, 10, ha="right")
        if i:
            arrow(ax, (x0 + 0.10, y + h + gap - 0.003), (x0 + 0.10, y + h + 0.003), EDGE, lw=1.2)

    # lerobot-record 实际穿过的那条路：入口 → 遥操作器读动作 → 实物执行 → 数据管线落盘。
    px = x0 + w + 0.045
    label(ax, px, 0.955, "lerobot-record 穿过的路", RED, 11, True, ha="left")
    order = [0, 4, 3, 2]
    # 三段跳各占一条竖带（越靠外的跳得越远），走正交折线。
    # 早先三段同挤在 px 上、又都是大圆弧，四步的先后顺序在图上根本读不出来。
    for k, (a, b) in enumerate(zip(order, order[1:])):
        lane = px + 0.030 + k * 0.026
        _polyline(ax, [(px, tops[a]), (lane, tops[a]), (lane, tops[b]), (px, tops[b])],
                  RED, 1.4, 0.012)
    for i, note in ((0, "(1) 起于命令"), (4, "(2) 读主臂动作"), (3, "(3) 从臂执行、相机出图"),
                    (2, "(4) 拼成一帧写进数据集")):
        ax.plot([x0 + w + 0.004, px], [tops[i], tops[i]], color=RED, linewidth=1.0,
                linestyle=(0, (2, 2)))
        # 标签让到最外一条竖带右侧，否则跳线会从字上穿过去
        label(ax, px + 0.030 + 2 * 0.026 + 0.018, tops[i], note, RED, 10, ha="left")
    label(ax, 0.03, 0.045, "绿色那两层是本讲要动的：teleoperators/ 读动作、datasets/ 落数据集",
          GREEN, 10, ha="left")
    save(fig, "lecture09", "fig09-3-lerobot-layers")


# ── 第9讲 4.4：一圈 33.3 ms 里的时间占用与超期 ───────────────────────────
# 那五行伪代码已经给了顺序，图要给的是**时间轴上的占用**：正常圈靠 sleep 补足余量，
# 超期圈把余量吃光、帧间不再等距——"sleep 不是硬实时保证"那句话的证据。
def record_loop_timing():
    fig, ax = canvas(7.6, 3.4)

    colors = [BLUE, GREEN, ORANGE, GREY]
    x0, span = 0.055, 0.72          # span = 一圈 33.3 ms 的预算

    def draw_loop(y, widths, sleep_w):
        x = x0
        for color, wfrac in zip(colors, widths):
            plain_box(ax, x, y, wfrac * span - 0.004, 0.10, color)
            x += wfrac * span
        label(ax, x0 + widths[0] * span / 2, y + 0.05, "get_observation", INK, 10)
        if sleep_w > 0:
            plain_box(ax, x, y, sleep_w * span - 0.004, 0.10, GREY, fill="#FAFAFA",
                      style=(0, (3, 2)), lw=1.2)
            label(ax, x + sleep_w * span / 2, y + 0.05, "precise_sleep 补足余量", "#888888", 10)
        return x + max(sleep_w, 0) * span

    # 正常的一圈：四段活加 sleep 正好凑满 33.3 ms。
    label(ax, 0.055, 0.855, "正常的一圈", BLUE, 11, True, ha="left")
    end_ok = draw_loop(0.700, [0.30, 0.07, 0.08, 0.09], 0.46)
    label(ax, 0.055, 0.635, "蓝段含两路相机解码；后面三小段依次是 "
                            "get_action、send_action、add_frame", "#666666", 10, ha="left", bg=True)

    # 超期的一圈：相机解码变慢把余量吃光，本圈拖到 40 ms（= 1.2 圈预算）。
    label(ax, 0.055, 0.505, "相机解码变慢的一圈", RED, 11, True, ha="left")
    end_bad = draw_loop(0.350, [0.96, 0.07, 0.08, 0.09], 0.0)
    label(ax, 0.055 + 0.48 * span, 0.295, "解码把余量整个吃光，连 sleep 都没得睡", RED, 10)

    # 两条竖线：一条是本该收圈的 33.3 ms，一条是这一圈实际收圈的时刻。
    ax.plot([end_ok, end_ok], [0.255, 0.830], color="#999999", linewidth=1.2,
            linestyle=(0, (4, 3)))
    label(ax, end_ok, 0.870, "33.3 ms", INK, 10)
    ax.plot([end_bad, end_bad], [0.255, 0.470], color=RED, linewidth=1.4, linestyle=(0, (4, 3)))
    label(ax, end_bad, 0.505, "约 40 ms", RED, 10)
    ax.annotate("", xy=(end_bad, 0.215), xytext=(end_ok, 0.215),
                arrowprops=dict(arrowstyle="<->", color=RED, linewidth=1.2))
    label(ax, (end_ok + end_bad) / 2, 0.170, "这一段就是帧间的抖动", RED, 10)

    save(fig, "lecture09", "fig09-4-record-loop-timing")


# ── 第10讲 2.4.2：ACT 主干 Step 3 的数据流 ──────────────────────────────
# 替换外部素材 `02_training3.png`。那张图上 decoder 标 "7x cross-attention blocks"、
# 图像特征标 728，都与本讲正文（`n_decoder_layers = 1`、ResNet18 出 512 维）正面冲突。
# 重画一版同构的数据流，两处标注按 LeRobot 默认改正。
def act_step3_dataflow():
    fig, ax = canvas(7.8, 3.5)

    plain_box(ax, 0.02, 0.30, 0.115, 0.50, GREY, fill="#F6F6F6")
    label(ax, 0.077, 0.72, "四路相机", "#333333", 11, True)
    label(ax, 0.077, 0.58, "480×640\n的 RGB 图", INK, 10)
    label(ax, 0.077, 0.38, "机器人关节\n状态 + $z$", INK, 10)

    plain_box(ax, 0.165, 0.46, 0.145, 0.34, BLUE)
    label(ax, 0.2375, 0.70, "ResNet18", BLUE, 11, True)
    label(ax, 0.2375, 0.555, "每路出\n15×20×512", "#4A6A85", 10)
    arrow(ax, (0.138, 0.63), (0.162, 0.63), EDGE, lw=1.4)

    plain_box(ax, 0.34, 0.46, 0.145, 0.34, BLUE)
    label(ax, 0.4125, 0.70, "展平 + 位置编码", BLUE, 11, True)
    label(ax, 0.4125, 0.555, "每路 300 个\n512 维 token", "#4A6A85", 10)
    arrow(ax, (0.313, 0.63), (0.337, 0.63), EDGE, lw=1.4)

    plain_box(ax, 0.515, 0.46, 0.15, 0.34, GREEN)
    label(ax, 0.59, 0.70, "Transformer\nEncoder", GREEN, 11, True)
    label(ax, 0.59, 0.535, "4 层自注意力", "#3D6B51", 10)
    arrow(ax, (0.488, 0.63), (0.512, 0.63), EDGE, lw=1.4)

    plain_box(ax, 0.695, 0.46, 0.15, 0.34, ORANGE)
    label(ax, 0.77, 0.70, "Transformer\nDecoder", ORANGE, 11, True)
    label(ax, 0.77, 0.535, "1 层交叉注意力", "#8A6212", 10)
    arrow(ax, (0.668, 0.63), (0.692, 0.63), EDGE, lw=1.4)

    plain_box(ax, 0.875, 0.46, 0.108, 0.34, RED)
    label(ax, 0.929, 0.70, "预测的\n动作序列", RED, 11, True)
    label(ax, 0.929, 0.535, "$k$ 步动作块", "#8B3A2E", 10)
    arrow(ax, (0.848, 0.63), (0.872, 0.63), EDGE, lw=1.4)

    # 关节状态与 z 走的是旁路：不过 ResNet，各自一个线性层直接投到 512 维进 encoder。
    ax.add_patch(FancyArrowPatch((0.138, 0.40), (0.560, 0.44), connectionstyle="arc3,rad=0.16",
                                 arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=GREY))
    label(ax, 0.42, 0.315, "关节状态与 $z$ 各过一个线性层，直接投到 512 维",
          "#888888", 10, bg=True)

    label(ax, 0.5, 0.045, "配置项就是 2.5 节那张表里的 $d_{\\mathrm{model}}=512$ 与 "
                          "n_decoder_layers = 1", "#888888", 10)
    save(fig, "lecture10", "fig10-2-act-step3-dataflow")


# ── 第10讲 2.6.2：动作队列 vs temporal ensemble ─────────────────────────
# 替换外部素材。原图把权重标成 [0.5, 0.3, 0.2, 0.1]，可正文的 $w_i\propto\exp(-mi)$ 在
# $m=0.01$ 时几个权重几乎相等——学生照公式推会推不出图上那组数。这里直接标真实量级。
def chunking_vs_ensemble():
    fig, ax = canvas(7.4, 4.6)
    x0, dx, bw = 0.075, 0.088, 0.072
    weights = np.exp(-0.01 * np.arange(4))
    weights = weights / weights.sum()

    def cell(step, y, color, h=0.072):
        plain_box(ax, x0 + step * dx, y, bw, h, color)

    # 时间刻度：两半共用同一条时间轴，位置一一对应。
    for s in range(8):
        label(ax, x0 + s * dx + bw / 2, 0.955, str(s), INK, 10)
    label(ax, x0 - 0.012, 0.955, "时间步", INK, 10, ha="right")

    label(ax, x0 - 0.012, 0.855, "动作队列", BLUE, 11, True, ha="right")
    for s in range(4):
        cell(s, 0.820, BLUE)
    label(ax, x0 + 4 * dx + 0.012, 0.856, "$t=0$ 预测的一段，依次执行完", BLUE, 10, ha="left")
    for s in range(4, 8):
        cell(s, 0.715, BLUE)
    label(ax, x0 + 4 * dx - 0.012, 0.751, "$t=4$ 才重新查询策略", BLUE, 10, ha="right")

    label(ax, x0 - 0.012, 0.545, "temporal\nensemble", GREEN, 11, True, ha="right")
    for row in range(4):
        y = 0.520 - row * 0.088
        for s in range(row, row + 4):
            cell(s, y, GREEN if s != 3 else ORANGE, h=0.070)
        label(ax, x0 + (row + 4) * dx + 0.012, y + 0.035, f"$t={row}$ 预测的一段",
              GREEN, 10, ha="left")

    # 第 3 步这一列被四段预测同时覆盖——图要给的就是这个重叠。
    ax.add_patch(FancyBboxPatch((x0 + 3 * dx - 0.006, 0.250), bw + 0.012, 0.340,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=2.0, edgecolor=ORANGE, facecolor="none"))
    label(ax, x0 + 3 * dx + bw / 2, 0.212, "第 3 步这一列被四段预测同时覆盖", ORANGE, 10)

    txt = "、".join(f"{w:.3f}" for w in weights)
    label(ax, 0.5, 0.120, f"执行时把这一列加权平均：$w_i \\propto \\exp(-m\\,i)$，"
                          f"$m=0.01$ 时归一化权重是 {txt}", "#555555", 10)
    save(fig, "lecture10", "fig10-2-chunking-vs-temporal-ensemble")


# ── 第8讲 3.2：误差滚雪球 ────────────────────────────────────────────────
# 曲线是示意用的合成数据，不是某次真实 rollout：这张图讲的是"走出示教覆盖范围之后
# 误差加速放大"这个机制。图内只标图元，解读由正文图注给出。
def error_snowball():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    t = np.linspace(0, 60, 400)
    demo = 0.5 + 0.28 * np.sin(t / 9.5)
    band = 0.27 + 0.02 * np.cos(t / 13.0)
    ax.fill_between(t, demo - band, demo + band, color=FILL[GREEN], zorder=0,
                    label="示教数据覆盖的状态范围")
    ax.plot(t, demo, color="#555555", linewidth=2.0, linestyle="--", label="示教轨迹")

    # 越界之前贴着示教走；越界之后锁住当时的值再指数发散——
    # 若继续叠 demo 的正弦，下行段会盖过发散项，画出「先回落再上扬」的错觉。
    leave = 29.0
    drift = 0.02 * (t / leave) ** 2
    at_leave = 0.5 + 0.28 * np.sin(leave / 9.5) + 0.02
    blow = 0.055 * (np.exp((t - leave) / 6.5) - 1.0)
    policy = np.where(t <= leave, demo + drift, at_leave + blow)
    keep = policy <= 2.75
    ax.plot(t[keep], policy[keep], color=RED, linewidth=2.4, label="策略闭环轨迹")
    ax.axvline(leave, color=RED, linewidth=1.2, linestyle=":", zorder=1)
    ax.text(leave + 1.2, 2.45, "走出示教覆盖范围", ha="left", va="top",
            fontsize=10.5, color=RED)

    ax.set_xlabel("时间步", fontsize=11)
    ax.set_ylabel("状态（如某个关节角，rad）", fontsize=11)
    ax.set_xlim(0, 62); ax.set_ylim(-0.15, 2.8)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "lecture08", "832_error-snowball")


# ── 第8讲 3.3：绝对口径与相对口径的同一步运动 ───────────────────────────
# 三个数就是正文那个算例：当前 1.18、下一帧 1.20。
def abs_vs_delta_axis():
    fig, ax = canvas(7.4, 2.9)
    y = 0.50
    x_now, x_next = 0.34, 0.52          # 1.18 与 1.20 两个刻度的位置
    ticks = {1.14: 0.10, 1.16: 0.22, 1.18: x_now, 1.20: x_next, 1.22: 0.64, 1.24: 0.76}

    ax.annotate("", xy=(0.94, y), xytext=(0.05, y),
                arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.6))
    for value, x in ticks.items():
        ax.plot([x, x], [y - 0.035, y + 0.035], color="#222222", linewidth=1.4)
        ax.text(x, y - 0.085, f"{value:.2f}", ha="center", va="top",
                fontsize=11, color="#333333")
    ax.text(0.955, y - 0.085, "关节角 (rad)", ha="right", va="top",
            fontsize=11, color="#333333")

    ax.plot(x_now, y, "o", markersize=11, color=GREY)
    ax.plot(x_next, y, "o", markersize=11, color=GREEN)
    # 两个点标签共用一条基线，各自锚在自己的点正下方（刻度数字下面一行）。
    POINT_LABEL_Y = y - 0.235
    ax.text(x_now, POINT_LABEL_Y, "当前位置", ha="center", va="top",
            fontsize=11, color=GREY)
    ax.text(x_next, POINT_LABEL_Y, "下一帧", ha="center", va="top",
            fontsize=11, color=GREEN)

    # 两块口径标注共用一条基线，各自锚在自己那根引线的上端。
    CALLOUT_Y = 0.88
    ax.annotate("", xy=(x_next - 0.005, y + 0.045), xytext=(x_now + 0.005, y + 0.045),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=2.2,
                                connectionstyle="arc3,rad=-0.62"))
    ax.annotate("", xy=((x_now + x_next) / 2, y + 0.175), xytext=((x_now + x_next) / 2, CALLOUT_Y - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=1.6))
    ax.text((x_now + x_next) / 2, CALLOUT_Y, "相对口径：动多少\n动作 = +0.02",
            ha="center", va="bottom", fontsize=12, color=BLUE,
            fontweight="bold", linespacing=1.6)

    ax.annotate("", xy=(x_next + 0.018, y + 0.02), xytext=(0.78, CALLOUT_Y - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.6))
    ax.text(0.78, CALLOUT_Y, "绝对口径：去哪儿\n动作 = 1.20",
            ha="center", va="bottom", fontsize=12, color=RED,
            fontweight="bold", linespacing=1.6)

    fig.tight_layout()
    save(fig, "lecture08", "833_abs-vs-delta-axis")


# ── 第8讲 3.3：口径错配的两种失败 ───────────────────────────────────────
# 左右两栏是两个方向的错配，曲线是示意用的合成数据。
def mismatch_failures():
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.0))
    t = np.linspace(0, 39, 200)

    ax = axes[0]
    intended = 1.18 + 0.22 * np.sin(t / 7.0)
    actual = np.where(t < 5, 1.18 - 1.16 * (t / 5) ** 1.4, 0.02 + 0.012 * np.sin(t))
    ax.plot(t, intended, color=GREEN, linewidth=2.0, linestyle="--", label="本该执行的轨迹")
    ax.plot(t, actual, color=RED, linewidth=2.2, label="实际执行")
    ax.set_title("相对数据 错当 绝对目标（训练侧弄错）", fontsize=11)
    ax.set_ylim(-0.12, 1.78)

    ax = axes[1]
    ax.axhspan(-0.3, 3.14, color=FILL[GREEN], zorder=0)
    ax.plot(t, np.full_like(t, 1.18), color=GREEN, linewidth=2.0, linestyle="--",
            label="本该执行的轨迹")
    runaway = 2.36 + 1.18 * t
    keep = runaway <= 7.0
    ax.plot(t[keep], runaway[keep], color=RED, linewidth=2.2, label="实际执行")
    ax.text(38.0, 0.30, "关节可达范围", ha="right", fontsize=10, color=GREEN)
    ax.set_title("绝对目标 错当 相对增量（部署侧弄错）", fontsize=11)
    ax.set_ylim(-0.3, 7.6)

    for ax in axes:
        ax.set_xlabel("时间步", fontsize=11)
        ax.set_ylabel("关节角 (rad)", fontsize=11)
        ax.set_xlim(-1, 40)
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    save(fig, "lecture08", "833_mismatch-failures")


# ── 第8讲 3.4：为什么用 q01/q99 而不是 min/max 归一化 ────────────────────
# 左边直方图是示意用的合成分布（主体高斯 + 几个离群毛刺），右边是映射示意。
def q99_normalization():
    rng = np.random.default_rng(8)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    bulk = rng.normal(0.5, 0.13, 4000)
    spikes = np.array([2.76, 2.90, 3.05])
    ax.hist(np.concatenate([bulk, spikes]), bins=70, color="#AEC6E4", edgecolor="none")
    q01, q99 = np.quantile(bulk, [0.01, 0.99])
    ax.set_ylim(0, 860)

    # 竖线的标签放线的一侧、不骑在线上；三个标签共用一条基线。
    LINE_LABEL_Y = 790
    ax.axvline(q01, color=GREEN, linewidth=2.2)
    ax.text(q01 + 0.07, LINE_LABEL_Y, "q01", ha="left", va="center",
            fontsize=10.5, color=GREEN)
    ax.axvline(q99, color=GREEN, linewidth=2.2)
    ax.text(q99 + 0.09, LINE_LABEL_Y, "q99", ha="left", va="center",
            fontsize=10.5, color=GREEN)
    ax.axvline(spikes.max(), color=RED, linewidth=2.0, linestyle="--")
    ax.text(spikes.max() - 0.07, LINE_LABEL_Y, "max", ha="right", va="center",
            fontsize=10.5, color=RED)

    # 离群点标签从左侧横着指过来，避免竖引线跟 max 虚线并排看着像第二条线。
    ax.plot(spikes, np.full_like(spikes, 20), marker="v", linestyle="none",
            markersize=9, color=RED)
    ax.annotate("离群点", xy=(spikes.min() - 0.04, 26), xytext=(1.72, 150),
                ha="right", va="center", fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.1))

    ax.set_xlabel("某一维动作的原始值（rad）", fontsize=11)
    ax.set_ylabel("样本数", fontsize=11)
    ax.set_xlim(-0.05, 3.4)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    TOP, BOT = 0.70, 0.26
    for yy in (TOP, BOT):
        ax.annotate("", xy=(0.99, yy), xytext=(0.06, yy),
                    arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.5))
    ax.text(0.06, TOP + 0.19, "原始值", ha="left", va="center",
            fontsize=10.5, color="#333333")
    ax.text(0.06, BOT - 0.19, "归一化后", ha="left", va="center",
            fontsize=10.5, color="#333333")

    # 三列竖直对齐；毛刺那列往左收，别贴到画布边。
    for x, top_text, bot_text, color in ((0.26, "q01", "-1", GREEN),
                                         (0.48, "0.5", "0", GREY),
                                         (0.70, "q99", "+1", GREEN)):
        ax.plot(x, TOP, "o", markersize=9, color=color)
        ax.text(x, TOP + 0.10, top_text, ha="center", va="center",
                fontsize=10.5, color=color)
        ax.annotate("", xy=(x, BOT + 0.045), xytext=(x, TOP - 0.045),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6))
        ax.text(x, BOT - 0.10, bot_text, ha="center", va="center",
                fontsize=10.5, color="#333333")

    ax.plot(0.88, TOP, "o", markersize=9, color=RED)
    ax.text(0.88, TOP + 0.10, "毛刺", ha="center", va="center",
            fontsize=10.5, color=RED)
    ax.annotate("", xy=(0.715, BOT + 0.045), xytext=(0.873, TOP - 0.045),
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.6,
                                linestyle="--"))

    fig.tight_layout()
    save(fig, "lecture08", "834_q99-normalization")


# ── 第10讲 3.1：LIBERO 上 eval/pc_success 的三段形状 ─────────────────────
# 数据不是编的，也不是重新跑的：直接从仓库里那张 wandb 面板截图上把曲线像素提出来，
# 按坐标轴定标还原成 (步数, 成功率)，再用全书五色重画。定标点见下面两行常量。
def libero_success_curve():
    panel = HERE.parents[1] / "code/vla/3_imitation_learning/result/3_1_act/eval_result1.png"
    img = plt.imread(panel)[:, :, :3] * 255
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    pink = (r > 200) & (g < 130) & (b > 150) & (b < 230)
    pink[:770] = False; pink[1100:] = False; pink[:, :70] = False; pink[:, 1000:] = False

    steps, lo, hi = [], [], []
    for x in range(70, 1000):
        col = np.nonzero(pink[:, x])[0]
        if col.size == 0:
            continue
        steps.append(200 + (x - 266.5) / 0.959)      # x 轴定标：刻度 200 在 266.5 px，每 100 步 95.9 px
        lo.append((1096 - col.max()) / 330 * 100)    # y 轴定标：0% 在 1096 px，100% 在 766 px
        hi.append((1096 - col.min()) / 330 * 100)
    steps, lo, hi = np.array(steps), np.array(lo), np.array(hi)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.fill_between(steps, lo, hi, color=BLUE, linewidth=0.0, alpha=0.85)
    plateau = steps >= 400
    median = np.median(((lo + hi) / 2)[plateau])
    ax.plot([400, 950], [median, median], color=RED, linewidth=2.0, linestyle="--")
    # 平台期整条带子都是曲线，这行字必须带白底，否则会糊在震荡里。
    ax.text(945, median - 26, f"平台期中位水平 {median:.0f}%", ha="right", fontsize=10, color=RED,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    for xa, xb, color, text in ((0, 184, GREY, "长时间贴 0"),
                                (184, 400, ORANGE, "很短的区间里抬起来"),
                                (400, 950, GREEN, "高位震荡的平台")):
        ax.axvspan(xa, xb, color=FILL[color], alpha=0.55, zorder=0)
        ax.text((xa + xb) / 2, 112, text, ha="center", fontsize=10, color=color)

    ax.set_xlabel("训练步数", fontsize=10)
    ax.set_ylabel("eval/pc_success（%）", fontsize=10)
    ax.set_xlim(0, 950); ax.set_ylim(-3, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, "lecture10", "fig10-3-libero-success-curve")


# ── 第10讲 5.2：2D 棋盘格上的概率路径 ────────────────────────────────────
# 回收 `2d_flow_matching_visualize_path.png`（原图 viridis 配色、英文轴标、10 格排版，
# 字太小）。这里画的是**解析的**概率路径 $z_t=(1-t)z_0+t z_1$ 的边际分布，
# 不需要训练模型——它正是 Flow Matching 要去拟合的那条路径本身。
def flow_matching_path():
    rng = np.random.default_rng(0)
    n = 9000
    # 棋盘格：在 [-2,2]^2 里只保留黑格，作为数据分布 z1。
    pts = rng.uniform(-2, 2, size=(n * 4, 2))
    keep = (np.floor(pts[:, 0]) + np.floor(pts[:, 1])) % 2 == 0
    z1 = pts[keep][:n]
    z0 = rng.normal(size=(n, 2))

    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(ts), figsize=(7.6, 1.95))
    for axis, t in zip(axes, ts):
        zt = (1 - t) * z0 + t * z1
        axis.scatter(zt[:, 0], zt[:, 1], s=1.0, color=BLUE, alpha=0.16, linewidths=0)
        axis.set_title(f"$t = {t:.2f}$", fontsize=10, color="#444444", pad=4)
        axis.set_xlim(-3, 3); axis.set_ylim(-3, 3)
        axis.set_xticks([]); axis.set_yticks([])
        for side in axis.spines.values():
            side.set_edgecolor("#CCCCCC")
    axes[0].set_xlabel("噪声", fontsize=10, color=GREY, labelpad=3)
    axes[-1].set_xlabel("数据分布（棋盘格）", fontsize=10, color=GREEN, labelpad=3)
    fig.tight_layout()
    save(fig, "lecture10", "fig10-5-flow-matching-path")


# ── 第10讲 5.3.1：条件路径是直的，边际流不是 ─────────────────────────────
# 原图只有一根线段加一个方框公式，而那个公式正是它上方正文刚给过的同一个式子（D2）。
# 重画要给公式给不了的东西：**多对端点**——每个噪声点各自连向一个数据点，这些直线彼此交叉。
def interpolation_paths():
    fig, ax = plt.subplots(figsize=(6.8, 3.3))

    # 端点手工摆开，不用随机种子——要的是"配对是乱的、所以连线交叉"这件事看得清楚。
    # 左列是噪声样本，右列是数据样本（双峰），同一下标的两个点配成一条条件路径。
    noise = [1.55, 0.85, 0.25, -0.40, -1.05, -1.60, 0.55]
    data = [-1.30, 1.75, -1.60, 1.30, 2.00, -1.85, 1.55]

    for i, (a, b) in enumerate(zip(noise, data)):
        if i == 0:
            continue
        ax.plot([0, 1], [a, b], color=GREY, linewidth=1.3, alpha=0.85, zorder=2)
    ax.scatter([0] * 7, noise, s=44, facecolor=FILL[BLUE], edgecolor=BLUE, linewidth=1.6, zorder=4)
    ax.scatter([1] * 7, data, s=44, facecolor=FILL[GREEN], edgecolor=GREEN, linewidth=1.6, zorder=4)

    # 高亮其中一对，把 z_0 / z_t / z_1 三个记号安在同一条路径上。
    a, b = noise[0], data[0]
    ax.plot([0, 1], [a, b], color=RED, linewidth=2.4, zorder=5)
    ax.scatter([0, 1], [a, b], s=58, facecolor=FILL[RED], edgecolor=RED, linewidth=2.2, zorder=6)
    t = 0.55
    ax.scatter([t], [(1 - t) * a + t * b], s=58, color=RED, zorder=6)
    ax.text(-0.04, a, "$z_0$", ha="right", va="center", fontsize=11, color=RED)
    ax.text(1.04, b, "$z_1$", ha="left", va="center", fontsize=11, color=RED)
    ax.text(t - 0.02, (1 - t) * a + t * b + 0.34, "$z_t=(1-t)\\,z_0+t\\,z_1$",
            ha="center", fontsize=10, color=RED, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.text(0, 2.62, "噪声 $\\mathcal{N}(0, I)$", ha="center", fontsize=10, color=BLUE)
    ax.text(1, 2.62, "数据分布（双峰）", ha="center", fontsize=10, color=GREEN)
    ax.set_xlim(-0.24, 1.24); ax.set_ylim(-3.4, 2.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    fig.tight_layout()
    save(fig, "lecture10", "fig10-5-linear-interpolation-path")


# ── 第11讲 2.3.2：连续动作值怎么变成词表里的一个槽位 ─────────────────────
# §2 有 122 行连续无图，而且正压在核心机制上。图要把正文分散在三小节、始终没放在一起的
# 三件事摆到同一条流水线上：分位数为什么挡得住异常值、一个具体值落到哪一格、代价落在词表哪一段。
def action_tokenization():
    rng = np.random.default_rng(11)
    fig, (hist, bins, vocab) = plt.subplots(
        1, 3, figsize=(7.8, 2.9), gridspec_kw={"width_ratios": [1.05, 0.95, 1.05]})
    q1, q99, a_i = -0.20, 0.30, 0.05

    # (a) 该维动作的分布：两端各 1% 的尾巴被剔掉，min-max 会被这些点拉垮。
    main = rng.normal(0.05, 0.107, 4000)
    tail = np.concatenate([rng.uniform(-0.62, -0.30, 26), rng.uniform(0.40, 0.70, 26)])
    counts, edges = np.histogram(np.concatenate([main, tail]), bins=54, range=(-0.7, 0.75))
    centers = (edges[:-1] + edges[1:]) / 2
    inside = (centers >= q1) & (centers <= q99)
    hist.bar(centers[inside], counts[inside], width=0.026, color=FILL[BLUE], edgecolor=BLUE, lw=0.6)
    hist.bar(centers[~inside], counts[~inside], width=0.026, color="#DDDDDD", edgecolor="#BBBBBB",
             lw=0.6)
    # 两条分位线离得近，标注各自朝外侧排，不要都居中——居中会撞在一起。
    for x, name, side in ((q1, "$q_1=-0.20$", "right"), (q99, "$q_{99}=0.30$", "left")):
        hist.axvline(x, color=RED, linewidth=1.6, linestyle="--")
        hist.text(x + (-0.03 if side == "right" else 0.03), 500, name, ha=side,
                  fontsize=10, color=RED)
    hist.text(-0.52, 150, "灰色是前后\n各 1% 的尾巴", ha="center", fontsize=10, color="#888888")
    hist.set_title("(a) 该维动作在训练集上的分布", fontsize=10, color="#444444")
    hist.set_xlim(-0.75, 0.8); hist.set_ylim(0, 560)
    hist.set_yticks([]); hist.tick_params(labelsize=10)
    for side in ("top", "right", "left"):
        hist.spines[side].set_visible(False)

    # (b) [q1, q99] 均分 256 格，正文那个算例落在第 128 格。
    bins.add_patch(FancyBboxPatch((0.06, 0.42), 0.88, 0.17,
                                  boxstyle="round,pad=0.004,rounding_size=0.02",
                                  linewidth=1.6, edgecolor=BLUE, facecolor=FILL[BLUE]))
    for k in range(1, 32):
        bins.plot([0.06 + 0.88 * k / 32] * 2, [0.42, 0.59], color=BLUE, linewidth=0.4, alpha=0.5)
    frac = (a_i - q1) / (q99 - q1)
    bins.plot([0.06 + 0.88 * frac] * 2, [0.36, 0.66], color=RED, linewidth=2.2)
    bins.text(0.06 + 0.88 * frac, 0.72, f"$a_i={a_i}$ 落在正中间\n→ 第 128 格",
              ha="center", fontsize=10, color=RED)
    bins.text(0.06, 0.32, "$q_1$\n第 0 格", ha="center", fontsize=10, color="#666666")
    bins.text(0.94, 0.32, "$q_{99}$\n第 255 格", ha="center", fontsize=10, color="#666666")
    bins.text(0.5, 0.10, "整个区间均分成 256 格", ha="center", fontsize=10, color="#555555")
    bins.set_title("(b) 均分成 256 个 bin", fontsize=10, color="#444444")
    bins.set_xlim(0, 1); bins.set_ylim(0, 1)
    bins.set_xticks([]); bins.set_yticks([]); bins.axis("off")

    # (c) 代价：词表最低频的 256 个槽位被动作 bin 顶掉，那批词从此不可用。
    vocab.add_patch(FancyBboxPatch((0.30, 0.34), 0.40, 0.62,
                                   boxstyle="round,pad=0.004,rounding_size=0.02",
                                   linewidth=1.6, edgecolor=GREY, facecolor="#F6F6F6"))
    vocab.add_patch(FancyBboxPatch((0.30, 0.34), 0.40, 0.12,
                                   boxstyle="round,pad=0.004,rounding_size=0.02",
                                   linewidth=1.8, edgecolor=RED, facecolor=FILL[RED]))
    vocab.text(0.50, 0.72, "Llama 2 词表\n常用词", ha="center", va="center", fontsize=10,
               color="#666666")
    # 说明排在色块正下方，不再往窄格子里硬塞——原来那版文字、箭头、色块糊成一团。
    vocab.text(0.50, 0.20, "红色＝最低频的 256 个槽位", ha="center", fontsize=10, color=RED)
    vocab.text(0.50, 0.06, "被动作 bin 顶掉，微调后不再可用", ha="center", fontsize=10,
               color="#666666")
    vocab.set_title("(c) 代价落在词表哪一段", fontsize=10, color="#444444")
    vocab.set_xlim(0, 1); vocab.set_ylim(0, 1); vocab.axis("off")

    fig.tight_layout()
    save(fig, "lecture11", "fig11-2-action-tokenization")


# ── 第11讲 7.2.3（2）：三种 VLM–动作模块的连接方式 ───────────────────────
# 7.2.1 那张对照表里"交替 CA + 因果 SA / 共享 Transformer + MoE 路由 / Flamingo 风格 CA"
# 三个短语根本讲不明白结构上到底差在哪，三张消融表给的又只是成绩。图给的是**结构**。
def alternating_ca_sa():
    fig, ax = canvas(7.8, 4.6)

    # 左侧：被冻结的 VLM，只取前 N 层的输出。三个方案共用同一个来源。
    plain_box(ax, 0.010, 0.30, 0.135, 0.53, GREY, fill="#F0F0F0")
    label(ax, 0.0775, 0.735, "VLM\n（冻结）", INK, 11, True)
    label(ax, 0.0775, 0.435, "只取前 $N$ 层\n作 key/value", INK, 10)

    # 标题与脚注都拆成两行——三格并排时，单行长标题一定会横向撞在一起。
    panels = [
        (0.175, GREEN, "SmolVLA\n交替 CA / SA",
         ["SA（因果）", "CA", "SA（因果）", "CA"],
         "每个块只放一种\n注意力，交替堆叠"),
        (0.455, BLUE, "$\\pi_0$\n共享 Transformer",
         ["SA", "SA", "SA", "SA"],
         "动作与 VLM token\n同处一个序列"),
        (0.735, ORANGE, "GR00T N1\nFlamingo 式",
         ["CA + SA", "CA + SA", "CA + SA", "CA + SA"],
         "CA 与 SA 挤在\n同一个块里"),
    ]
    w = 0.245
    for x0, color, title, blocks, note in panels:
        label(ax, x0 + w / 2, 0.888, title, color, 11, True)
        for i, name in enumerate(blocks):
            y = 0.355 + i * 0.125
            is_ca = name.startswith("CA")
            plain_box(ax, x0 + 0.030, y, w - 0.06, 0.100, color,
                      fill=FILL[color] if is_ca else "#FFFFFF")
            label(ax, x0 + w / 2, y + 0.050, name, color, 10)
            # CA 块的 key/value 来自 VLM：从左侧引一条横向虚线加箭头进来。
            if is_ca:
                ax.plot([0.147, x0 + 0.026], [y + 0.050, y + 0.050], color=GREY,
                        linewidth=1.0, linestyle=(0, (2, 2)), zorder=0)
                arrow(ax, (x0 + 0.008, y + 0.050), (x0 + 0.028, y + 0.050), EDGE, lw=1.3)
        label(ax, x0 + w / 2, 0.272, note, INK, 10)

    # 因果掩码画成下三角，说明 SA 只往回看。
    mx, my, cell = 0.400, 0.025, 0.020
    for i in range(4):
        for j in range(4):
            ax.add_patch(FancyBboxPatch((mx + j * cell, my + (3 - i) * cell), cell * 0.86,
                                        cell * 0.86, boxstyle="square,pad=0",
                                        linewidth=0.6, edgecolor="#BBBBBB",
                                        facecolor=FILL[GREEN] if j <= i else "#FFFFFF"))
    label(ax, mx - 0.014, my + 2 * cell, "查询", "#888888", 10, ha="right")
    label(ax, mx + 4 * cell + 0.016, my + 2 * cell, "绿色＝可以看到（下三角）", "#888888", 10,
          ha="left")
    save(fig, "lecture11", "fig11-7-alternating-ca-sa")


# ── 第11讲 8.1：六个模型的动作头地图 ────────────────────────────────────
# 8.1 白纸黑字承诺了"一张地图"却整节零图。两段 bullet 已经说了谁属于哪一路，
# 图必须多给**位置关系与演进方向**，尤其是 π0.5 横跨两侧这件文字最难说清的事。
def action_head_map():
    fig, ax = plt.subplots(figsize=(7.5, 4.3))

    # (x=动作表示, y=参数量 B, 名字, 标签方向)
    points = [
        (0.10, 7.0, "OpenVLA", BLUE, "right"),
        (0.16, 3.3, "$\\pi_0$-FAST", BLUE, "left"),
        (0.27, 2.35, "VLA-0", BLUE, "right"),
        (0.86, 3.3, "$\\pi_0$", GREEN, "right"),
        (0.90, 0.45, "SmolVLA", GREEN, "right"),
        (0.94, 0.08, "ACT（参照）", GREY, "left"),
        (0.74, 0.12, "Diffusion Policy（参照）", GREY, "left"),
    ]
    for x, y, name, color, side in points:
        ax.scatter([x], [y], s=70, facecolor=FILL[color], edgecolor=color, linewidth=2.0, zorder=4)
        ax.annotate(name, (x, y), xytext=(9 if side == "right" else -9, 0),
                    textcoords="offset points", fontsize=10, color=color,
                    ha="left" if side == "right" else "right", va="center", zorder=5)

    # π0.5 横跨两侧：预训练用离散 FAST，后训练换连续 flow。这是文字最难说清的一件事。
    ax.add_patch(FancyBboxPatch((0.42, 2.75), 0.38, 1.05,
                                boxstyle="round,pad=0.0,rounding_size=0.04",
                                linewidth=2.0, edgecolor=ORANGE, facecolor=FILL[ORANGE],
                                zorder=3, mutation_aspect=0.32))
    ax.text(0.61, 3.28, "$\\pi_{0.5}$：预训练离散、后训练连续", ha="center", va="center",
            fontsize=10, color=ORANGE, zorder=5)

    for (xa, ya), (xb, yb), color in (((0.10, 7.0), (0.16, 3.3), BLUE),
                                      ((0.16, 3.3), (0.27, 2.35), BLUE),
                                      ((0.86, 3.3), (0.90, 0.45), GREEN)):
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.8,
                                    shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.18"))
    ax.text(0.185, 5.0, "离散一路", fontsize=10, color=BLUE, ha="left")
    ax.text(0.845, 1.15, "连续一路", fontsize=10, color=GREEN, ha="right")

    ax.set_yscale("log")
    ax.set_xlabel("动作表示：离散 token　←→　连续量", fontsize=10)
    ax.set_ylabel("参数量（B，对数轴）", fontsize=10)
    ax.set_xlim(-0.02, 1.12); ax.set_ylim(0.05, 13)
    ax.set_xticks([]); ax.set_yticks([0.1, 0.45, 1, 3.3, 7])
    ax.set_yticklabels(["0.1", "0.45", "1", "3.3", "7"])
    ax.tick_params(labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    save(fig, "lecture11", "fig11-8-action-head-map")


# ── 第12讲 1.2：两条轴张成的平面 ────────────────────────────────────────
# 1.2 把"正交、可组合"当结论说了出来，却从头到尾只有文字。1.3.4 与 1.4.5 那两张表
# 都是**沿单条轴**的一维清单，排不出"同一个 LoRA 可以配三种不同信号"这件事。
def two_axes_map():
    fig, ax = canvas(7.4, 4.2)
    cols = ["全模型全参数", "模块级全参数", "PEFT（LoRA / QLoRA）"]
    rows = ["SFT", "离线偏好\nDPO 等", "在线奖励\nPPO / GRPO"]
    x0, y0, cw, ch = 0.215, 0.185, 0.257, 0.222

    for i in range(4):
        ax.plot([x0, x0 + 3 * cw], [y0 + i * ch] * 2, color="#E4E4E4", linewidth=1.0, zorder=0)
        ax.plot([x0 + i * cw] * 2, [y0, y0 + 3 * ch], color="#E4E4E4", linewidth=1.0, zorder=0)
    for j, name in enumerate(cols):
        label(ax, x0 + (j + 0.5) * cw, 0.905, name, INK, 11, True)
    for i, name in enumerate(rows):
        label(ax, x0 - 0.018, y0 + (i + 0.5) * ch, name, INK, 11, True, ha="right")

    # 只有真正在本讲（或第15讲）出现过的组合才摆点，不为了填满格子编组合。
    # 格子只有 0.23 宽，长名字一律拆两行——单行写满会顶出框外。
    chips = [
        (0, 0, "全量 SFT", BLUE, False), (0, 1, "全量 DPO", BLUE, False),
        (1, 0, "冻视觉编码器\n＋全参训语言骨干", GREEN, False),
        (2, 0, "LoRA SFT\n（本讲主线）", RED, True),
        (2, 0, "QLoRA SFT", ORANGE, False),
        (2, 1, "LoRA DPO", ORANGE, False),
        (2, 2, "LoRA GRPO\n（第15讲）", ORANGE, False),
    ]
    seen = {}
    for j, i, text, color, main in chips:
        k = seen.get((j, i), 0)
        seen[(j, i)] = k + 1
        cy = y0 + (i + 0.5) * ch + (0.055 if (j, i) == (2, 0) else 0) - k * 0.110
        plain_box(ax, x0 + (j + 0.5) * cw - 0.115, cy - 0.047, 0.23, 0.094, color,
                  lw=2.2 if main else 1.6)
        label(ax, x0 + (j + 0.5) * cw, cy, text, color, 10, main)

    label(ax, 0.5, 0.075, "参数更新范围：越往右，被更新的参数越少", "#555555", 10)
    label(ax, 0.012, 0.905, "训练信号\n↑", "#888888", 10, ha="left")
    save(fig, "lecture12", "fig12-1-two-axes-map")


# ── 第12讲 4.6.5：单卡显存柱在五种方案下怎么变 ───────────────────────────
# 4.4.6 那张表已经把 (4+K/N_d)Ψ 这些数给全了，图要给的是**决策顺序与卡容量线的关系**：
# 为什么 DDP 那一格柱子高度一点不变、为什么四件套要排在分片之前。
def vram_under_each_strategy():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    # 7B、混合精度 + Adam，按 4.1.2 的 16 字节/参数账本：参数 14、梯度 14、优化器 84 GB。
    # 分片按 N=8 卡，与 4.4.6 表里的 Stage 2 / Stage 3 公式一致。
    n = 8
    plans = [
        ("(1) 单卡\n基线", 14, 14, 84, 12),
        ("(2) 省显存\n四件套", 14, 14, 0, 3),
        ("(3) DDP\n(8 卡)", 14, 14, 84, 12),
        ("(4) ZeRO-2\n(8 卡)", 14, 14 / n, 84 / n, 12),
        ("⑤ZeRO-3\n(8 卡)", 14 / n, 14 / n, 84 / n, 12),
    ]
    segs = [("参数 fp16", BLUE), ("梯度 fp16", GREEN), ("优化器状态 fp32", ORANGE),
            ("激活值", GREY)]
    xs = np.arange(len(plans))
    bottoms = np.zeros(len(plans))
    for k, (name, color) in enumerate(segs):
        vals = np.array([p[k + 1] for p in plans])
        ax.bar(xs, vals, bottom=bottoms, width=0.52, color=FILL[color], edgecolor=color,
               linewidth=1.6, label=name, hatch="///" if k == 3 else None, zorder=3)
        bottoms += vals
    for x, total in zip(xs, bottoms):
        ax.text(x, total + 3, f"{total:.0f} GB", ha="center", fontsize=10, color="#444444",
                zorder=5)

    ax.axhline(40, color=RED, linewidth=1.8, linestyle="--", zorder=4)
    ax.text(4.45, 44, "单卡容量线 40 GB", ha="right", fontsize=10, color=RED, zorder=5)

    # DDP 那一格：柱子一点没变，多出来的是旁边一模一样的另外 7 根。
    for k in range(3):
        ax.bar([2.30 + k * 0.10], [124], width=0.06, color="#FFFFFF", edgecolor="#BBBBBB",
               linewidth=1.0, zorder=2)
    ax.text(2.62, 100, "…另外 7 张卡\n各存一模一样的一份", ha="left", fontsize=10,
            color="#888888", zorder=5)
    ax.annotate("", xy=(2.0, 132), xytext=(0.0, 132), zorder=5,
                arrowprops=dict(arrowstyle="<->", color="#888888", linewidth=1.2))
    ax.text(1.0, 136, "加卡只提速，单卡显存一点没降", ha="center", fontsize=10, color="#888888")

    ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in plans], fontsize=10)
    ax.set_ylabel("单卡显存（GB，7B 模型）", fontsize=10)
    ax.set_ylim(0, 152); ax.set_yticks([0, 40, 80, 120])
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handletextpad=0.4, columnspacing=1.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "lecture12", "fig12-4-vram-under-each-strategy")


print("渲染示意图：")
two_axes_map()
vram_under_each_strategy()
action_tokenization()
alternating_ca_sa()
action_head_map()
act_step3_dataflow()
chunking_vs_ensemble()
libero_success_curve()
flow_matching_path()
interpolation_paths()
teleop_tradeoff_map()
lerobot_layers()
record_loop_timing()
motion_taxonomy()
policy_spectrum()
train_deploy_roundtrip()
normalization_ranges()
scale_error_symptoms()
error_snowball()
abs_vs_delta_axis()
mismatch_failures()
q99_normalization()
long_horizon_paradigms()
one_cut_three_lines()
agent_env_loop()
action_space()
discount_horizon()
policy_distribution()
onoff_dataflow()
actor_critic_a2c()
actor_critic_ddpg()
replay_buffer()
q_propagation()
cartpole_task()
