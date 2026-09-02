"""重录之后的收尾核对：有没有集被墙钟截断，有没有集没完成任务。

两件事都会静默入库，所以必须核：

★ **截断**。`lerobot-record` 是墙钟驱动的，仿真跑不满 30 Hz。窗口给小了，
  轨迹就在中途停下 —— 数据看起来正常（帧数、字段、编码全对），只是手臂还没
  把物体放进箱子。判据：录到的帧数 ≥ 源集帧数。

★ **没完成**。回放不是逐帧复现原结果（GPU PhysX 的接触求解跨 reset 不可复现），
  所以有的集会失败。判据用**成功判据在末段是否成立**，不看单帧：源集自己的末帧
  也常是 `success=False`（物体已放进去、只差手臂还在微动）。

不合格的集**列出集号**，交给 `lerobot-edit-dataset --operation.type delete_episodes`
官方删除，不在这里动数据。

用法：`python check_regen.py <场景> <状态目录> <录制输出目录>`
"""

import json
import sys
from pathlib import Path

import numpy as np

# 末段窗口（帧）。成功判据在这段里成立过就算完成 —— 手臂停稳前后会反复翻转。
TAIL_FRAMES = 30
# 末段里至少要有这么多帧成立。取 1 是因为 `is_robot_static` 在停稳过程中抖动，
# 而"物体已在箱口内"一旦成立就不会自己变回去。
TAIL_MIN_TRUE = 1


def main(argv) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    scene, states_dir, out_dir = argv[0], Path(argv[1]), Path(argv[2])

    truncated, failed, ok = [], [], []
    for meta_path in sorted((states_dir / "meta").glob("ep*.json")):
        ep = int(meta_path.stem[2:])
        want = json.loads(meta_path.read_text())["n_frames"]
        log_path = out_dir / "logs" / f"{scene}-ep{ep}.npz"
        if not log_path.exists():
            failed.append((ep, "没有状态日志 —— 这一集没录成"))
            continue
        log = np.load(log_path)
        got = len(log["success"])
        if got < want:
            truncated.append((ep, f"{got}/{want} 帧"))
            continue
        tail = np.asarray(log["success"], bool)[-TAIL_FRAMES:]
        if int(tail.sum()) < TAIL_MIN_TRUE:
            failed.append((ep, f"末 {TAIL_FRAMES} 帧成功 {int(tail.sum())} 帧"))
            continue
        ok.append(ep)

    total = len(ok) + len(truncated) + len(failed)
    print(f"  {scene}: 合格 {len(ok)}/{total} · 被截断 {len(truncated)} · 未完成 {len(failed)}")
    for label, rows in (("被截断（窗口给小了，调大 RATE_FLOOR 重录）", truncated),
                        ("未完成任务（交给 delete_episodes 删）", failed)):
        if rows:
            print(f"  {label}：")
            for ep, why in rows[:10]:
                print(f"    ep{ep}: {why}")
            if len(rows) > 10:
                print(f"    …… 共 {len(rows)} 集")
    bad = [ep for ep, _ in failed]
    if bad:
        print(f"  待删集号：[{','.join(str(e) for e in bad)}]")
    print("CHECK_REGEN_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
