"""管线第一步：从一段普通视频里恢复人体动作。

调用 GVHMR 把单目视频还原成世界坐标系下的三维人体动作（SMPL-X 参数）。
渲染很吃时间也用不上，所以走的是不渲染的那条分支。

讲义对应：第14讲 4.4 节。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


def required_packages() -> list[str]:
    """列出这一步额外需要的第三方包。

    Returns:
        包名列表，供调用方在缺依赖时给出可读的提示。
    """
    return ["hmr4d"]


def default_checkpoint_root() -> Path:
    """给出 GVHMR 预训练权重的默认存放目录。

    Returns:
        下载权重所在目录。
    """
    return Path(os.environ["DATASETS_ROOT"]) / "models" / "downloaded" / "gvhmr" / "inputs" / "checkpoints"


def recover_human_motion(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    static_camera: bool = True,
    runner: Callable[..., object] = subprocess.run,
) -> Path:
    """对一段视频跑 GVHMR，得到世界坐标系下的人体动作。

    Args:
        video_path: 输入视频。
        output_dir: 预测结果落盘目录。

    Returns:
        GVHMR 预测文件的路径。
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"input video not found: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_script = Path(__file__).resolve().parent / "run_gvhmr_no_render.py"
    checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else default_checkpoint_root()
    command = [
        str(python_executable),
        str(runner_script),
        "--video",
        str(video_path),
        "--output-root",
        str(output_dir),
        "--checkpoint-root",
        str(checkpoint_root),
    ]
    if static_camera:
        command.append("-s")

    runner(command, check=True, env=os.environ.copy())

    expected = output_dir / video_path.stem / "hmr4d_results.pt"
    if expected.is_file():
        return expected

    candidates = sorted(output_dir.rglob("hmr4d_results.pt"))
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "GVHMR command finished but hmr4d_results.pt was not found. "
        f"Expected {expected}; found {len(candidates)} candidates under {output_dir}."
    )


def main() -> None:
    # 主要修改这一段：输入视频与 GVHMR 输出目录。
    """对课程自带的那段武术视频跑一次动作恢复。"""
    video = Path("input_video.mp4")
    output_dir = Path("gvhmr_output")
    checkpoint_root = None
    static_camera = True

    result = recover_human_motion(
        video,
        output_dir,
        checkpoint_root=checkpoint_root,
        static_camera=static_camera,
    )
    print(result)


if __name__ == "__main__":
    main()
