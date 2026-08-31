# 插图源文件与渲染

`lectureNN/src/*.svg` 是本课程自绘插图的**源文件**（可版本控制、可改文字改配色）。
`lectureNN/ref/*.png` 是讲义实际引用的位图，由 src 渲染得到。

## 渲染

任选一种（输出宽度按讲义排版取 2 倍图，保证 PDF 清晰）：

```bash
# rsvg-convert（推荐，最快）
rsvg-convert -w 2000 lecture14/src/fig-action-space.svg -o lecture14/ref/fig-action-space.png

# Inkscape
inkscape lecture14/src/fig-action-space.svg --export-type=png --export-width=2000 \
         --export-filename=lecture14/ref/fig-action-space.png

# Chrome/Chromium 无头
chromium --headless --screenshot=lecture14/ref/fig-action-space.png \
         --window-size=2000,860 lecture14/src/fig-action-space.svg
```

批量：

```bash
for f in lecture*/src/*.svg; do
  out="${f/\/src\//\/ref\/}"; out="${out%.svg}.png"
  rsvg-convert -w 2000 "$f" -o "$out"
done
```

## 字体

SVG 里指定的字体族为 `Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, sans-serif`。
渲染机需装其中之一，否则中文会掉字。Debian/Ubuntu：

```bash
sudo apt install fonts-noto-cjk
```

## 当前 src 清单

| 源文件 | 渲染到 | 讲义位置 |
|---|---|---|
| `lecture14/src/fig141-agent-env-loop.svg` | `lecture14/ref/fig141-agent-env-loop.png` | 第14讲 2.1 |
| `lecture14/src/fig-action-space.svg` | `lecture14/ref/fig-action-space.png` | 第14讲 2.2 |
| `lecture14/src/fig-policy-brain.svg` | `lecture14/ref/fig-policy-brain.png` | 第14讲 2.4 |
| `lecture14/src/fig-onoff-dataflow.svg` | `lecture14/ref/fig-onoff-dataflow.png` | 第14讲 3.4 |
| `lecture14/src/fig-actor-critic-a2c.svg` | `lecture14/ref/fig-actor-critic-a2c.png` | 第14讲 5.3 |
| `lecture16/src/fig162-actor-critic-ddpg.svg` | `lecture16/ref/fig162-actor-critic-ddpg.png` | 第16讲 3.2 |

渲染后可删除的旧位图（已被替换，不再被讲义引用）：
`lecture14/ref/fig-actor-critic.png`、`lecture16/ref/fig162-actor-critic.png`、
`lecture16/ref/fig162-onoff-dataflow.png`、`lecture16/ref/fig162-action-value.png`、
`lecture14/ref/fig-reward-return.png`、`lecture14/ref/fig-value-function.png`。
