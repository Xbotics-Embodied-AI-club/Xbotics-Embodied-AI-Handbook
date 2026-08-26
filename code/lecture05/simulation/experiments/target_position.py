"""实验一：目标位置与工作空间（第 5 讲 §3.7）。

基线目标 [0.30, 0.10, 0.20]。每次只改变 x / y / z 中的一个分量，其余保持基线。

预测（运行前写下）：
  * 越接近关节限位或奇异姿态，收敛时间可能增加；
  * 超出工作空间时会超时，或停在误差不再下降的最近可达状态。

指标：最终误差、达标时间、是否触发限幅、最小奇异值、末端路径长度。

运行：.venv/bin/python experiments/target_position.py
输出：results/exp1_target_position.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np

from reach import RESULT_DIR
from metrics import run_reach_metrics


def main() -> None:
    baseline = np.array([0.30, 0.10, 0.20])
    axis_idx = {"x": 0, "y": 1, "z": 2}
    sweeps = {
        "x": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        "y": [-0.25, -0.15, -0.05, 0.00, 0.10, 0.20, 0.30],
        "z": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    }

    rows = []
    for axis, values in sweeps.items():
        for v in values:
            target = baseline.copy()
            target[axis_idx[axis]] = v
            r = run_reach_metrics(target)
            rows.append(
                [
                    axis, f"{v:.2f}", int(r["success"]),
                    "" if r["reached_time"] is None
                    else f"{r['reached_time']:.3f}",
                    f"{r['final_error']:.5f}",
                    f"{r['min_singular']:.5f}",
                    r["clipped_updates"],
                    f"{r['path_length']:.4f}",
                ]
            )

    out = RESULT_DIR / "exp1_target_position.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["axis", "value", "success", "reached_time_s",
             "final_error_m", "min_singular", "clipped_updates",
             "path_length_m"]
        )
        w.writerows(rows)

    print(f"{'axis':<5}{'value':>7}{'success':>9}{'time_s':>9}"
          f"{'final_err':>11}{'clipped':>9}{'min_sing':>11}")
    for r in rows:
        print(f"{r[0]:<5}{r[1]:>7}{r[2]:>9}{r[3]:>9}"
              f"{r[4]:>11}{r[6]:>9}{r[5]:>11}")

    print(f"\noutput -> {out}")


if __name__ == "__main__":
    main()
