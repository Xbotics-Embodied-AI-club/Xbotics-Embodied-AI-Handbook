# 插图源文件与渲染

自绘插图的**唯一真相源**是 `assets/figures/render_diagrams.py`。改图、改文字、改配色都在那个文件里改，然后重跑：

```bash
python3 assets/figures/render_diagrams.py
```

它会把全部自绘图重新渲染到各自的 `lectureNN/ref/*.png`，并逐张核算字号（见下）。
`lectureNN/ref/` 下其余的位图是外部截图或照片，不由本脚本产出。

## 字号规则

脚本会按下面这条规则核算每一张图，不达标直接终止，不能靠调排版 `width%` 绕过：

```
印出字号 = 源码 fontsize × 缩放比
缩放比   = 版心宽度（6.5 英寸）× 排版 width% ÷ 作图宽度
判据     ：图内最小的那个字，印在纸上不低于 8 pt
```

落到作图约定就是三句话：**画布宽度 ≤ 8 英寸、图内任何文字 ≥ 10 pt、排版一律 `width=100%`**。
细节见 `render_diagrams.py` 的模块 docstring。

## 配色

全书统一五色，不要引入新配色：

| 颜色 | 值 |
|---|---|
| 蓝 | `#1F6FB2` |
| 绿 | `#2E8B57` |
| 橙 | `#D68910` |
| 红 | `#C0392B` |
| 灰 | `#777777` |

对应的浅色填充在脚本的 `FILL` 字典里。

## 字体

脚本按 `Noto Sans CJK SC` / `Noto Sans CJK JP` / `WenQuanYi Zen Hei` 的顺序找中文字体，
渲染机需装其中之一，否则中文会掉字。Debian/Ubuntu：

```bash
sudo apt install fonts-noto-cjk
```

## 已失效的 SVG 源（待清理）

`lecture14/src/`、`lecture16/src/` 下还留着 5 个 `.svg`：

`fig141-agent-env-loop.svg`、`fig-action-space.svg`、`fig-actor-critic-a2c.svg`、
`fig-onoff-dataflow.svg`（以上 lecture14）、`fig162-actor-critic-ddpg.svg`（lecture16）。

这几张对应的 PNG **已全部改由 `render_diagrams.py` 重画**，SVG 不再是它们的等价源。
不要再从这些 SVG 渲染 PNG——那会用旧版本覆盖掉现在讲义引用的图。它们可以直接删除，
本轮因不在授权路径内未动。

（`lecture05/original/*.svg` 是另一回事，与本脚本无关。）
