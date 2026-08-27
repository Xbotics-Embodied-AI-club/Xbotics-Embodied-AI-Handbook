"""离屏渲染 pick-place 全程为 mp4 视频（不依赖远程桌面 / GPU）。

用 mujoco.Renderer 逐帧离屏渲染（软件渲染即可），帧写为 PPM，再用系统 ffmpeg
合成 mp4。整个抓取过程约几百帧，几十秒内完成。

运行：
  .venv/bin/python render_video.py
  .venv/bin/python render_video.py --output results/pick_place.mp4 --height 720 --width 1280
"""

import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco

from reach import RESULT_DIR
from pick_place import load_box_model, run_pick_place, print_summary


def save_ppm(path: Path, rgb: np.ndarray) -> None:
    """把 RGB uint8 (H, W, 3) 写为 PPM P6（二进制，零第三方库依赖）。"""
    h, w = rgb.shape[:2]
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.ascontiguousarray(rgb).tobytes())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="离屏渲染 pick-place 全程为 mp4")
    parser.add_argument("--output", default=str(RESULT_DIR / "pick_place.mp4"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--keep-frames", action="store_true",
                        help="保留中间 PPM 帧（调试用）")
    args = parser.parse_args()

    model, data = load_box_model()
    try:
        renderer = mujoco.Renderer(model, args.height, args.width)
    except Exception as e:
        print(f"离屏渲染初始化失败：{e}")
        print("Renderer 需要 OpenGL 上下文。可选：")
        print("  1) 在有 GPU 或 X11 显示的环境运行（默认 glfw 后端）；")
        print("  2) 无显示时尝试软件渲染：MUJOCO_GL=egl .venv/bin/python render_video.py")
        sys.exit(1)

    # 设置 free camera 视角（斜俯视搬运路径）
    cam = renderer.scene.camera
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.27, 0.03, 0.06]
    cam.distance = 1.0
    cam.azimuth = 150.0
    cam.elevation = -25.0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / (out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    counter = [0]

    def frame_callback():
        renderer.update_scene(data, camera=-1)
        rgb = renderer.render()
        save_ppm(frames_dir / f"frame_{counter[0]:05d}.ppm", rgb)
        counter[0] += 1

    r = run_pick_place(model, data, frame_callback=frame_callback)
    n_frames = counter[0]

    if n_frames == 0:
        print("无帧可渲染")
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%05d.ppm"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print(f"渲染 {n_frames} 帧 -> {out}  ({n_frames / args.fps:.2f} s @ {args.fps} fps)")
    print()
    print_summary(r)


if __name__ == "__main__":
    main()
