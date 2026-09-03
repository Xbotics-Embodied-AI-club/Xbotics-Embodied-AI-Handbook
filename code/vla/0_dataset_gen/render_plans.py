#!/usr/bin/env python
"""把准备好的规划逐集复跑并渲成一段视频：上视图与腕部并排，每集打上集号。

## 为什么要有这个

数字全绿而任务全败是本轮已经发生过的事（帧数容差、12 路一致性、38 项结构项全过，
目审三集全部"方块没放进去"）。**成没成只有看画面才作数。**
`check_plans.py` 给的是判据结论，这份给的是判据背后的画面 —— 两者一起看才知道
"成功"是怎么成功的：夹爪有没有真包住方块、抬起时方块有没有在爪里转、松手是不是在箱口内。

腕部画面不能省：上视图看不出咬合深浅，而"被挤出去"最早的迹象就出现在腕部那一路。

## 编码

h264、30fps，与数据集两侧一致（真机 9 份实测都是 h264）。用 PyAV 直接写，
不落中间图片 —— 那会多出一个"图片怎么编回视频"的口径。

用法：`python render_plans.py <场景> <准备目录> <输出mp4> [<集号,逗号分隔>]`
      不给集号就渲全部。
"""

import sys
from pathlib import Path

import av
import numpy as np
import recipe
from PIL import Image, ImageDraw

FPS = 30
# 每集之间插几帧黑场，读的人才分得清集与集的边界。
GAP_FRAMES = 8
CAMERAS = ("top", "wrist")


def episode_frames(env_id, state_path, actions):
    """按这串动作复跑一集，逐帧收各路相机画面并横向拼起来。

    Args:
        env_id: 已注册的环境 id。
        state_path: 强制初始状态 json —— 场景必须与规划当初解算时逐位相同。
        actions: `(帧数, 6)` 真机口径动作。

    Returns:
        `[(H, W, 3) uint8]`，长度与 `actions` 相同。
    """
    from so101_sim.config_lerobot_robot import SO101SimRobotConfig
    from so101_sim.lerobot_robot import JOINT_NAMES, SO101SimRobot

    robot = SO101SimRobot(SO101SimRobotConfig(
        task=env_id, episode_length=len(actions) + 10,
        initial_state_path=str(state_path)))
    robot.connect()
    out = []
    for row in actions:
        obs = robot.get_observation()
        missing = [c for c in CAMERAS if c not in obs]
        if missing:
            robot.disconnect()
            sys.exit(f"★ 观测里没有这几路相机 {missing}；现有的是 {sorted(obs)}")
        out.append(np.hstack([np.asarray(obs[c], np.uint8) for c in CAMERAS]))
        robot.send_action({f"{n}.pos": float(row[i]) for i, n in enumerate(JOINT_NAMES)})
    robot.disconnect()
    return out


def label(frame, text):
    """在画面左上角写一行标签。

    Args:
        frame: `(H, W, 3) uint8` 画面。
        text: 标签文字（ASCII —— 容器里不保证有中文字体，缺字形会静默画成方框）。

    Returns:
        `(H, W, 3) uint8` 带标签的画面。
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 8 + 7 * len(text), 20), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.asarray(img)


def main(argv) -> int:
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    scene, prep, out_path = argv[0], Path(argv[1]), Path(argv[2])
    want = {int(x) for x in argv[3].split(",")} if len(argv) == 4 else None
    spec = recipe.SCENES[scene]

    plans = sorted((prep / "plans").glob("ep*.npy"),
                   key=lambda p: int(p.stem.removeprefix("ep")))
    if want is not None:
        plans = [p for p in plans if int(p.stem.removeprefix("ep")) in want]
    if not plans:
        sys.exit(f"★ {prep}/plans 里没有要渲的集 —— 空结果不是结论")

    clips = []
    for path in plans:
        ep = int(path.stem.removeprefix("ep"))
        state = prep / "states" / f"{path.stem}.json"
        if not state.is_file():
            sys.exit(f"★ 没有 {path.stem} 的强制初始状态 {state}")
        frames = episode_frames(spec["env_id"], state, np.load(path))
        clips.append([label(f, f"ep{ep}  {i + 1}/{len(frames)}")
                      for i, f in enumerate(frames)])
        print(f"    ep{ep}: {len(frames)} 帧")

    height, width = clips[0][0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    black = np.zeros((height, width, 3), np.uint8)
    for i, clip in enumerate(clips):
        for frame in clip:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        if i < len(clips) - 1:
            for _ in range(GAP_FRAMES):
                container.mux(stream.encode(av.VideoFrame.from_ndarray(black, format="rgb24")))
    container.mux(stream.encode())
    container.close()

    total = sum(len(c) for c in clips) + GAP_FRAMES * (len(clips) - 1)
    print(f"\n  {len(clips)} 集、{total} 帧、{width}×{height}、{total / FPS:.1f} 秒")
    print(f"  {out_path}")
    print("RENDER_PLANS_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
