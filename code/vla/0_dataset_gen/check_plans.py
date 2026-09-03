#!/usr/bin/env python
"""直接拿准备好的规划复跑判成败 —— 不经过录制，用来扫夹爪几何这类物性旋钮。

与 `check_success.py` 同一套判据（抬起来过 + 末帧落在料箱自己坐标系的开口内），
区别只在动作来源：那份读已录出的数据集，这份读 `plans/ep*.npy`。录一轮要几十分钟，
而扫 δ 需要同一批场景反复跑十几次，所以这里跳过录制。

★ 判据必须与录制后那道门**同一个函数** —— 各写一套的话，扫出来的"最优 δ"可能在
  另一套判据下并不最优，而那才是最终验收用的那套。这里 import 而不抄。

用法：`python check_plans.py <场景> <准备目录>`
"""

import sys
from pathlib import Path

import numpy as np
import recipe
from check_success import judge, replay


def main(argv) -> int:
    if len(argv) != 2:
        sys.exit(__doc__)
    scene, prep = argv[0], Path(argv[1])
    spec = recipe.SCENES[scene]

    plans = sorted((prep / "plans").glob("ep*.npy"),
                   key=lambda p: int(p.stem.removeprefix("ep")))
    if not plans:
        sys.exit(f"★ {prep}/plans 里没有规划 —— 空结果不是「全成功」")

    ok = []
    for path in plans:
        ep = path.stem
        state = prep / "states" / f"{ep}.json"
        if not state.is_file():
            sys.exit(f"★ 没有 {ep} 的强制初始状态 {state}")
        good, why = judge(spec, *replay(spec["env_id"], state, np.load(path)))
        ok.append(good)
        print(f"    {ep}: {'成功' if good else '★ ' + why}")

    print(f"\n  {scene}：{sum(ok)}/{len(ok)} 成功（{sum(ok) / len(ok) * 100:.0f}%）")
    print("CHECK_PLANS_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
