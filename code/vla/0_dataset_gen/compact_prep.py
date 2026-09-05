#!/usr/bin/env python
"""把过了成功门的那些集挑出来、重新编号，作为正式录制的输入。

## 为什么要有这一步

规划出来的集里有极少数在物理上抓不住（撒点落在工作区死角，手臂够得到但压不下去）。
**这种集不能进数据集** —— 模仿学习会把"伸过去、夹空、把空气搬到料箱"照单学走。
产线此前是录完再判，于是不合格的集已经在数据集里了，只能事后 `delete_episodes`；
而官方 merge 之后集号指针会大面积悬空，删就必须赶在合并之前，链条因此很脆。

改成**录之前就筛**：`check_plans.py` 用与录制**同一个后端、同一份强制初始状态**复跑一遍，
所以它的判定与录制结果一致，不是近似。筛完重新编号 0..N−1，录制那一段就不必再关心空号。

## 编号变了，来源不能丢

新集号写进 `meta/ep<新>.json` 的同时保留原 `seed` 与 `source_ep` —— "这一集的场景是怎么
来的"必须可复现，重新编号不能把它抹掉。

用法：`python compact_prep.py <准备目录> <输出目录> <成功清单文件> [<要几集>]`
      成功清单是 `check_plans.py` 的输出，逐行含 `ep<N>: 成功`。
"""

import json
import re
import shutil
import sys
from pathlib import Path

PASS = re.compile(r"\bep(\d+):\s*成功")


def main(argv) -> int:
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    prep, out_dir, verdicts = Path(argv[0]), Path(argv[1]), Path(argv[2])
    want = int(argv[3]) if len(argv) == 4 else None

    text = verdicts.read_text()
    kept = sorted(int(m.group(1)) for m in PASS.finditer(text))
    total = len(re.findall(r"\bep(\d+):", text))
    if not kept:
        sys.exit(f"★ {verdicts} 里没有一集判成功 —— 空结果不是「全成功」")
    if want is not None and len(kept) < want:
        sys.exit(f"★ 只有 {len(kept)} 集通过成功门，要 {want} 集。"
                 f"提高准备集数再来，不要凑数下发。")
    if want is not None:
        kept = kept[:want]

    for sub in ("states", "plans", "meta"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for new, old in enumerate(kept):
        shutil.copy2(prep / "states" / f"ep{old}.json", out_dir / "states" / f"ep{new}.json")
        shutil.copy2(prep / "plans" / f"ep{old}.npy", out_dir / "plans" / f"ep{new}.npy")
        meta = json.loads((prep / "meta" / f"ep{old}.json").read_text())
        meta["source_ep"] = old
        (out_dir / "meta" / f"ep{new}.json").write_text(
            json.dumps(meta, ensure_ascii=False))

    print(f"  成功门：{total} 集判过，收 {len(kept)} 集"
          f"（{len(kept) / max(total, 1) * 100:.0f}%），重新编号 0..{len(kept) - 1}")
    print(f"  {out_dir}")
    print("COMPACT_PREP_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
