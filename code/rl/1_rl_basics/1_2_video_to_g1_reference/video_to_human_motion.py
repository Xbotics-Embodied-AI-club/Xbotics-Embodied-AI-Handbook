from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


def required_packages() -> list[str]:
    return ["hmr4d"]


def default_checkpoint_root() -> Path:
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
