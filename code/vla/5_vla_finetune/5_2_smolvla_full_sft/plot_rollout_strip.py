"""把三个仿真任务各一集成功回合，拼成一张「策略在跑」的连拍图。

在讲12 2.4 节被引用。曲线只说明"数对不对"，读者还要看见**它究竟在做什么** ——
一条 94% 的柱子不告诉人机械臂是怎么把方块放进料箱的。

每行一个任务，八帧走完示教定义的五个阶段（`0_dataset_gen/expert.py:19`）：
取物 → 合拢 → 搬运 → 松手 → 回家。

★ 关键时刻按**夹爪动作**定位，不按帧号硬编 —— 硬编的帧号换一集就指向别处
  （本项目在"取错时刻"上栽过四次）。合拢帧与松手帧从录像同名的 `.tsv`
  （每步的夹爪指令）里解出来，其余六帧按阶段插在它们之间。

★ 录像必须是 `record_full_rollout.py` 产的：`lerobot-eval` 的录像在判成功那一刻
  就断了，拍不到第五段回家。

用法：`python plot_rollout_strip.py <录像目录>`
      目录里要有 SO101PickPlace{Cube40,Cube20,Cylinder40}-v1.mp4 与同名 .tsv
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
OUT = HERE.resolve().parents[3] / "assets" / "figures" / "lecture12" / "ref"

# 每行一个任务：(录像文件名主干, 行首标签)
ROWS = [
    ("SO101PickPlaceCube40-v1", "Pick up a cube"),
    ("SO101PickPlaceCube20-v1", "Pick up a small cube"),
    ("SO101PickPlaceCylinder40-v1", "Pick up a can"),
]
N_FRAMES = 8
CROP_PAD = 12


def read_frames(video: Path) -> np.ndarray:
    """把整段录像解成帧数组。

    Args:
        video: mp4 路径。

    Returns:
        `(帧数, 高, 宽, 3)` 的 uint8 数组。
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def content_box(arr: np.ndarray) -> tuple[int, int, int, int]:
    """整段里"有内容"的区域：动过的像素 ∪ 暗色物件（料箱、机械臂）。

    Args:
        arr: `(帧数, 高, 宽, 3)`。

    Returns:
        `(left, top, right, bottom)`。

    只按"动过"定框会漏掉全程不动的料箱 —— 而这张图要说明的正是物体最后进了箱。
    只按"暗色"定框又会把远处的地砖网格圈进来。两者取并集才是这段回合真正的舞台。
    """
    g = arr.astype(np.int16).mean(axis=3)
    mask = ((g.max(axis=0) - g.min(axis=0)) > 12) | (g.mean(axis=0) < 110)
    ys, xs = np.where(mask)
    h, w = mask.shape
    top = max(0, ys.min() - CROP_PAD)
    # 再往下收一点：上缘那条空地板不含任何信息，而**图高决定它能不能排进当页** ——
    # 每张图排版高要压到 3.4 英寸以内（xb-1zh），超了 LaTeX 就把整张推到次页、原地留白。
    top += int((ys.max() - top) * 0.10)
    return (max(0, xs.min() - CROP_PAD), top,
            min(w, xs.max() + CROP_PAD + 1), min(h, ys.max() + CROP_PAD + 1))


def moments(grip: np.ndarray, n_steps: int) -> list[int]:
    """按夹爪指令解出八个关键时刻的帧号。

    Args:
        grip: `(步数,)` 每步的夹爪指令（行程百分比，小 = 合拢）。
        n_steps: 录像总帧数。

    Returns:
        长度 8 的帧号列表，按时间递增。

    合拢帧 = 夹爪**张开过之后**首次越回开合中线以下；松手帧 = 此后首次再越上去。
    两者把轨迹分成取物 / 搬运 / 回家三段，八帧按段插进去，于是换一集、换一个任务
    都指得准。

    ★ "张开过之后"这个前提不能省：示教的起手那一帧**不许把爪张开**
      （`expert.py:678`，张开会把物体推走），所以第 0 帧夹爪就是合的。
      直接找"首次低于中线"会命中第 0 帧，八个时刻全挤在开头 —— 第一版就是这样，
      而它**不报错**，只是产出一张八格几乎一样的图。
    """
    mid = (grip.max() + grip.min()) / 2
    opened = grip > mid
    if not opened.any():
        sys.exit("★ 这段录像里夹爪从没张开过，定位不了合拢/松手")
    open_at = int(np.argmax(opened))
    closed_after = ~opened[open_at:]
    close_at = open_at + int(np.argmax(closed_after)) if closed_after.any() else n_steps // 2
    open_again = opened[close_at:]
    release_at = close_at + int(np.argmax(open_again)) if open_again.any() else n_steps - 1
    last = n_steps - 1
    # 搬运段按比例取两帧而不是取中点：夹住之后策略先原地稳一会儿才抬，取中点时物体
    # 还停在原位，与前一格几乎一样。"即将合拢"与"已夹住"也不各占一格 —— 夹爪在整幅
    # 画面里只有几十像素，两帧看不出差别，白费八格里的一格。
    span = max(1, release_at - close_at)
    picks = [
        0,                                       # 初始位姿
        close_at // 2,                           # 落地曲线下降途中
        min(last, close_at + 6),                 # 合拢夹住
        close_at + int(0.45 * span),             # 抬起，进入搬运弧线
        close_at + int(0.85 * span),             # 停在料箱上方
        min(last, release_at + 8),               # 松手，物体落进箱
        (release_at + last) // 2,                # 退回途中
        last,                                    # 回到 home
    ]
    assert len(picks) == N_FRAMES, f"取了 {len(picks)} 帧，画布按 {N_FRAMES} 帧排"
    return [min(max(p, 0), last) for p in picks]


def main(argv) -> int:
    if len(argv) != 1:
        sys.exit(__doc__)
    src = Path(argv[0])
    OUT.mkdir(parents=True, exist_ok=True)

    font_path = os.environ.get("XBOTICS_FIG_FONT", "").strip()
    if not font_path or not Path(font_path).is_file():
        sys.exit("★ XBOTICS_FIG_FONT 没配好，行标签画不出中文。见 assets/figures/figstyle.py")

    tiles, labels = [], []
    for stem, label in ROWS:
        video, table = src / f"{stem}.mp4", src / f"{stem}.tsv"
        for f in (video, table):
            if not f.is_file():
                sys.exit(f"★ 缺 {f} —— 录像要用 record_full_rollout.py 产")
        arr = read_frames(video)
        grip = np.array([float(l.split("\t")[1]) for l in
                         table.read_text().strip().splitlines()])
        box = content_box(arr)
        idx = moments(grip[:len(arr)], len(arr))
        tiles.append([Image.fromarray(arr[i]).crop(box) for i in idx])
        labels.append(f"{label}（{len(arr)} 步）")
        print(f"  {label}: {len(arr)} 帧，取 {idx}")

    fw, fh = tiles[0][0].size
    gap = 6
    # 字号按**成品在页面上的物理尺寸**定，不写死像素：像素宽 W 放到 6 英寸上是 W/6 dpi，
    # 要印出来约 9pt，像素高就得是 9/72 × (W/6)。写死 22px 时印出来只有 4.6pt，看不清。
    px_per_pt = (fw * N_FRAMES + gap * (N_FRAMES - 1)) / 6 / 72
    font = ImageFont.truetype(font_path, max(12, round(9 * px_per_pt)))
    lab_h = font.size + 8
    canvas = Image.new("RGB", (fw * N_FRAMES + gap * (N_FRAMES - 1),
                               (fh + lab_h + gap) * len(ROWS)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (row, lab) in enumerate(zip(tiles, labels)):
        y = r * (fh + lab_h + gap)
        draw.text((2, y + 4), lab, fill="black", font=font)
        for c, im in enumerate(row):
            canvas.paste(im, (c * (fw + gap), y + lab_h))
    path = OUT / "fig12-2-rollout-strip.png"
    canvas.save(path)
    print(f"  {path.name}  {canvas.size[0]}x{canvas.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
