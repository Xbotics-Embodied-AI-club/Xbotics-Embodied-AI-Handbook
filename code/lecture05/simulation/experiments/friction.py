"""实验三：摩擦（第 5 讲 §3.7）—— 研究方块—桌面界面（释放后滑动）。

基线：scene_box.xml（方块 sliding friction=1）。
改动：scene_box_lowfriction.xml（方块 sliding friction=0.1）。

说明：本实验只改「方块—桌面」界面（方块自身的 friction），未改夹爪 collision_gripper
的高优先级接触参数；夹爪—方块界面不受本改动影响。

预测（运行前写下）：方块—桌面摩擦降低后，释放后方块滑动距离增大，落点误差可能增大。

指标：是否放置、落点误差、释放后滑动距离（释放瞬间 → 稳定落点的平面位移）。

运行：.venv/bin/python experiments/friction.py
输出：results/exp3_friction.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np

from reach import RESULT_DIR
from pick_place import load_box_model, run_pick_place


def main() -> None:
    cases = [
        ("mu_1.0", "scene_box.xml"),
        ("mu_0.1", "scene_box_lowfriction.xml"),
    ]
    rows = []
    for label, scene in cases:
        for rep in range(3):
            model, data = load_box_model(scene)
            r = run_pick_place(model, data)
            slide = float("nan")
            if r["release_box_xy"] is not None:
                slide = float(np.linalg.norm(
                    r["release_box_xy"] - r["box_final_xy"]))
            rows.append([
                label, rep, int(r["placed"]),
                f"{r['place_err']:.4f}",
                f"{slide:.4f}" if slide == slide else "",
            ])

    out = RESULT_DIR / "exp3_friction.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "rep", "placed", "place_err_m", "slide_dist_m"])
        w.writerows(rows)

    print(f"{'case':<10}{'rep':>4}{'placed':>8}{'place_err':>11}{'slide_dist':>12}")
    for r in rows:
        print(f"{r[0]:<10}{r[1]:>4}{r[2]:>8}{r[3]:>11}{r[4]:>12}")
    print(f"\noutput -> {out}")


if __name__ == "__main__":
    main()
