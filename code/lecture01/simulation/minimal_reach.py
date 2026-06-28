"""Lecture 01 — 最小 reaching 闭环（纯 Python 2D 示意，无硬件依赖）."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ReachConfig:
    target_eef: tuple[float, float] = (0.20, 0.05)
    max_steps: int = 50
    step_size: float = 0.02
    threshold: float = 0.01


def run_reach(cfg: ReachConfig) -> dict:
    current = np.array([0.0, 0.0], dtype=float)
    target = np.array(cfg.target_eef, dtype=float)
    trajectory = [current.copy()]

    for _ in range(cfg.max_steps):
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist < cfg.threshold:
            break
        step = delta / (dist + 1e-8) * min(cfg.step_size, dist)
        current = current + step
        trajectory.append(current.copy())

    trajectory_arr = np.stack(trajectory)
    final_error = float(np.linalg.norm(target - trajectory_arr[-1]))
    return {
        "target_eef": list(cfg.target_eef),
        "final_eef": trajectory_arr[-1].tolist(),
        "final_error": final_error,
        "success": final_error < cfg.threshold,
        "num_steps": len(trajectory) - 1,
        "trajectory": trajectory_arr.tolist(),
    }


def plot_trajectory(result: dict, out: Path) -> None:
    traj = np.array(result["trajectory"])
    target = np.array(result["target_eef"])
    plt.figure(figsize=(5, 5))
    plt.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, label="eef path")
    plt.scatter(*target, c="red", s=80, label="target")
    plt.scatter(traj[0, 0], traj[0, 1], c="green", s=80, label="start")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"L01 minimal reach (error={result['final_error']:.4f})")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    cfg = ReachConfig()
    result = run_reach(cfg)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_trajectory(result, out_dir / "trajectory.png")
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
