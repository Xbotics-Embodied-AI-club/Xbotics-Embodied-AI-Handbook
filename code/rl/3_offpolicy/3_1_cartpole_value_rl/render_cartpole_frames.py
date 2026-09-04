"""渲三帧真实的 CartPole 画面，拼成讲义 2.4 节那张实景图。

为什么要有这张图：2.4 节原先只有一张自绘示意图，读者看不到这个环境真实长什么样。
示意图讲的是「两道终止线」和「反直觉的动作方向」，这里就按那两道线各取一帧真实状态，
让示意图上的红虚线、橙扇形在实景里有对应物。

三帧不是训练轨迹里截的，是直接把环境状态设成想要的那三个姿态再渲——CartPole 的
state 就是那四个连续量，允许直接赋值，比跑到某个姿态再抓帧可靠得多。

跑法（gymnasium 只在 gpu_x86 那个大 extra 里，为渲一张图不值得装，用临时环境）：
    export XBOTICS_FIG_FONT=<TimesSong.ttf>
    uv run --with gymnasium --with pygame --with matplotlib \
        python rl/3_offpolicy/3_1_cartpole_value_rl/render_cartpole_frames.py
"""

import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# 全书统一字体：西文 Times New Roman、中文宋体。字体路径走环境变量，取不到就报错停下。
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "assets" / "figures"))
import figstyle  # noqa: E402  —— 必须在 sys.path 补好之后再导入
FONT_NAME = figstyle.apply()

out_path = (Path(__file__).resolve().parents[4] / "assets" / "figures"
            / "lecture16" / "ref" / "fig16-cartpole-real.png")

# (小车位置, 小车速度, 杆角 rad, 杆角速度)，配一句说明。
# 终止线：杆角 ±0.2095 rad（约 12°）、小车位置 ±2.4。三帧分别是「站住」和贴着两道线。
FRAMES = [
    ((0.00, 0.0, 0.02, 0.0), "杆基本竖直，回合继续"),
    ((0.00, 0.0, 0.20, 0.0), "杆偏 11.5°，再歪一点就到 12° 终止线"),
    ((2.20, 0.0, 0.03, 0.0), "小车滑到 2.2，再滑一点就出 ±2.4 轨道"),
]


def frames():
    """把三个指定状态各渲成一张 RGB 图。"""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    shots = []
    for state, caption in FRAMES:
        env.reset(seed=0)
        # CartPole 的观测就是内部状态本身，直接赋值即可摆出想要的姿态。
        env.unwrapped.state = np.array(state, dtype=np.float64)
        shots.append((env.render(), caption))
    env.close()
    return shots


def main():
    shots = frames()
    for _, caption in FRAMES:
        figstyle.assert_covered(caption, where="CartPole 实景图分题")
    fig, axes = plt.subplots(1, len(shots), figsize=(12.0, 3.2))
    for ax, (image, caption) in zip(axes, shots):
        ax.imshow(image)
        ax.set_xlabel(caption, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor("#cccccc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Font": FONT_NAME})
    plt.close(fig)
    print(f"图已保存到 {out_path}")


def demo():
    """自检：三帧确实渲出了不同画面，且姿态按指定状态变化。

    只断言「渲出来了、而且三张不一样」——如果状态赋值被环境忽略（比如 reset 之后
    又被覆盖），三张图会一模一样，这个断言就会红。
    """
    shots = frames()
    assert len(shots) == 3, shots
    arrays = [s[0] for s in shots]
    assert all(a.ndim == 3 and a.shape[2] == 3 for a in arrays), [a.shape for a in arrays]
    assert not np.array_equal(arrays[0], arrays[1]), "杆角不同却渲出同一张图"
    assert not np.array_equal(arrays[0], arrays[2]), "车位不同却渲出同一张图"
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
