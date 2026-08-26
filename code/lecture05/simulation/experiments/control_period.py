"""实验四：控制周期（第 5 讲 §3.7）。

固定目标与物理步长，只改外层 IK 周期 CONTROL_DT：
  physics timestep 恒为 5 ms（不改 model.opt.timestep）；
  外层 IK 周期依次取 10 / 20 / 40 / 80 / 160 ms（DECIMATION=2/4/8/16/32）。

预测（运行前写下）：
  控制命令保持更久，末端路径可能变抖、最终误差增大、收敛变慢；
  但物理积分误差本身不变（timestep 未动）。

指标：末端路径长度、最终误差、达标时间、最小奇异值。

运行：.venv/bin/python experiments/control_period.py
输出：results/exp4_control_period.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np

from reach import RESULT_DIR, PHYSICS_DT
from metrics import run_reach_metrics


def main() -> None:
    target = np.array([0.30, 0.10, 0.20])
    control_dts = [0.010, 0.020, 0.040, 0.080, 0.160]

    rows = []
    for dt in control_dts:
        r = run_reach_metrics(target, control_dt=dt)
        decimation = round(dt / PHYSICS_DT)
        rows.append(
            [
                f"{dt * 1000:.0f}", decimation, int(r["success"]),
                "" if r["reached_time"] is None
                else f"{r['reached_time']:.3f}",
                f"{r['final_error']:.5f}",
                f"{r['path_length']:.4f}",
                f"{r['min_singular']:.5f}",
            ]
        )

    out = RESULT_DIR / "exp4_control_period.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["control_period_ms", "decimation", "success", "reached_time_s",
             "final_error_m", "path_length_m", "min_singular"]
        )
        w.writerows(rows)

    print(f"{'ctrl_ms':>9}{'decim':>7}{'success':>9}{'time_s':>9}"
          f"{'final_err':>11}{'path_len':>10}{'min_sing':>11}")
    for r in rows:
        print(f"{r[0]:>9}{r[1]:>7}{r[2]:>9}{r[3]:>9}"
              f"{r[4]:>11}{r[5]:>10}{r[6]:>11}")

    print(f"\noutput -> {out}")


if __name__ == "__main__":
    main()
