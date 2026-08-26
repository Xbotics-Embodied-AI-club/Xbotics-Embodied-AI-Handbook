"""实验二：方块质量（第 5 讲 §3.7）。

基线：scene_box.xml（方块未显式设质量，density=1000 约 0.096 kg）。
改动：scene_box_mass.xml（方块显式 mass="0.20"）。

预测（运行前写下）：质量增加会提高抬升/加速阶段的负载，可能扩大相对漂移或导致滑脱；
但夹持力足够时，质量增加也可能不改变成功结果。

指标：双侧法向力、抬升高度、方块-末端相对漂移、是否完成放置。

运行：.venv/bin/python experiments/mass.py
输出：results/exp2_mass.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from reach import RESULT_DIR
from pick_place import load_box_model, run_pick_place


def main() -> None:
    cases = [
        ("baseline", "scene_box.xml"),        # 默认方块（约 0.096 kg）
        ("mass_0.20", "scene_box_mass.xml"),  # 显式 mass=0.20 kg
    ]
    rows = []
    for label, scene in cases:
        for rep in range(3):
            model, data = load_box_model(scene)
            r = run_pick_place(model, data)
            rows.append([
                label, rep, int(r["placed"]),
                f"{r['grasp_ff']:.2f}", f"{r['grasp_mf']:.2f}",
                f"{r['lift_delta']:.4f}", f"{r['carry_drift']:.4f}",
                f"{r['place_err']:.4f}",
            ])

    out = RESULT_DIR / "exp2_mass.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "rep", "placed", "grasp_ff_N", "grasp_mf_N",
                    "lift_delta_m", "carry_drift_m", "place_err_m"])
        w.writerows(rows)

    print(f"{'case':<12}{'rep':>4}{'placed':>8}{'ff_N':>8}{'mf_N':>8}"
          f"{'lift':>9}{'drift':>9}{'place_err':>11}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:>4}{r[2]:>8}{r[3]:>8}{r[4]:>8}"
              f"{r[5]:>9}{r[6]:>9}{r[7]:>11}")
    print(f"\noutput -> {out}")


if __name__ == "__main__":
    main()
