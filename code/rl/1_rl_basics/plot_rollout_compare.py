"""把三个版本的 rollout 视频抽成关键帧对照图：每行一个版本，标题印该版本的评测数字。

两张图共用一套排版：3 行 × 5 帧，每行上面一条深色标题条，写「版本名 + 评测奖励 + 摔倒率」。
奖励与摔倒率不是手抄的，是从同目录的评测 json 里读出来的 —— 数字和图永远对得上。

  · 行走对照   → assets/figures/lecture14/ref/fig144-walk-compare.png
  · 动作跟随对照 → assets/figures/lecture14/ref/fig144-track-compare.png

素材是留存的 rollout mp4，不重跑实验。帧靠 ffmpeg 从视频里按时间点取，走管道不落临时文件。

原来这两张图是有的，但出图脚本没跟着进仓（图上没有任何元数据、全仓也查不到脚本）。
本文件按留存素材把它们补回来，同时接上全书统一字体。
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# 全书统一字体：西文 Times New Roman、中文宋体。字体路径走环境变量，取不到就报错停下。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "figures"))
import figstyle  # noqa: E402  —— 必须在 sys.path 补好之后再导入
FONT_NAME = figstyle.apply()

HERE = Path(__file__).parent
FIGURES = HERE.parents[2] / "assets" / "figures" / "lecture14" / "ref"

N_FRAMES = 5
# 掐头去尾取帧：开头几帧常常还没进入状态，结尾可能已经停住。
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

BAR_COLOR = "#1E2A38"
BAR_TEXT = "#FFFFFF"

PANELS = {
    "fig144-walk-compare.png": {
        "result_dir": "result/1_1_g1_walk_rl",
        "rows": [("g1-walk-reinforce", "v1  REINFORCE"),
                 ("g1-walk-a2c", "v2  A2C  (+critic +GAE)"),
                 ("g1-walk-ppo", "v3  PPO  (+clip +KL +reuse)")],
    },
    "fig144-track-compare.png": {
        "result_dir": "result/1_3_g1_motion_tracking",
        "rows": [("track-v1-reinforce", "v1  REINFORCE"),
                 ("track-v2-a2c", "v2  A2C  (+critic +GAE)"),
                 ("track-v3-ppo-original", "v3  PPO  (original, 10000 it)")],
    },
}


def video_duration(path):
    """问 ffprobe 要视频时长（秒）。"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def grab_frame(path, when):
    """从视频里取 when 秒处的那一帧，走管道返回 PIL 图像，不落临时文件。"""
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{when:.3f}", "-i", str(path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout
    if not png:
        raise SystemExit(f"从 {path} 的 {when:.3f}s 处没取到帧 —— 视频比预期短？")
    return Image.open(io.BytesIO(png)).convert("RGB")


def row_title(label, metrics):
    """标题条上那一行字：版本名 + 评测奖励 + 摔倒率。"""
    return (f"{label}    reward {metrics['mean_reward']:.3f}"
            f"    falls {metrics['done_fraction'] * 100:.1f}%")


def build(filename, spec):
    """拼出一张对照图并存盘。"""
    result_dir = HERE / spec["result_dir"]
    rows = spec["rows"]

    # 每行 = 一条窄标题条 + 一排帧；高度比按现有图的观感取。
    fig, axes = plt.subplots(
        len(rows) * 2, N_FRAMES, figsize=(16.32, 8.46),
        gridspec_kw={"height_ratios": [1, 9] * len(rows), "wspace": 0.008, "hspace": 0.0})

    for r, (stem, label) in enumerate(rows):
        video = result_dir / f"{stem}.mp4"
        metrics = json.loads((result_dir / f"{stem}.json").read_text())
        duration = video_duration(video)

        # 标题条：把这一行的第一个格子撑到整行宽，其余格子隐藏，文字用这个格子自己的坐标摆。
        # （早先用 transFigure 摆过，三行标题会叠在同一个高度上 —— 那是整张图的坐标，不分行。）
        pos_first = axes[r * 2][0].get_position()
        pos_last = axes[r * 2][-1].get_position()
        for ax in axes[r * 2][1:]:
            ax.set_visible(False)
        bar = axes[r * 2][0]
        bar.set_position([pos_first.x0, pos_first.y0,
                          pos_last.x1 - pos_first.x0, pos_first.height])
        bar.set_facecolor(BAR_COLOR)
        bar.set_xticks([]); bar.set_yticks([])
        for side in bar.spines.values():
            side.set_visible(False)
        bar.text(0.004, 0.42, row_title(label, metrics), transform=bar.transAxes,
                 ha="left", va="center", color=BAR_TEXT, fontsize=15, fontweight="bold")

        for c, frac in enumerate(FRACTIONS[:N_FRAMES]):
            ax = axes[r * 2 + 1][c]
            ax.imshow(grab_frame(video, duration * frac))
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)

    # 标题条的文字用 transFigure 定位，得在 tight_layout 之后再摆一次才准，
    # 所以这里不用 tight_layout，直接留窄边距。
    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002)
    out = FIGURES / filename
    fig.savefig(out, dpi=100, facecolor=BAR_COLOR, metadata={"Font": FONT_NAME})
    plt.close(fig)
    print(f"图已保存到 {out}")


def main():
    for filename, spec in PANELS.items():
        build(filename, spec)


if __name__ == "__main__":
    main()
