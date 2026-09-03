"""重录之后的保真核对：录出来的每一集是不是源集的 1:1 还原。

★ **判据**：动作逐位等于源集，帧数与源集相差不超过 `recipe.FRAME_TOLERANCE` 帧（双向），
  且差掉的那几帧必须落在源集自己的静止尾巴里。
  为什么容许差几帧：`lerobot-record` 按墙钟停表，循环踩过 33.3ms 就少一帧，
  真机采集同样会偶尔差一两帧没跟上。
  为什么只容许**几**帧：多出来的帧全是静止帧，会把末尾静止段撑长 ——
  实测给 90 帧余量时撑到 68 帧、占集长 15.9%，而真机是 16 帧 / 3.9%，
  仿真被撑成真机的四倍，那正是"仿真=真机"要消灭的分布差异。

★ **任务成没成不在这里判**，在 `verify_cross_backend.py`：那件事要在两个物理后端
  各跑一遍才算数，而且要按料箱自己的坐标系判"落进箱口"。一个问题一道门。

★ **源集号 ↔ 数据集内集号靠 `logs/shard<N>.order` 对账。**
  每录一集追加一行 `<源集号> <ok|reject>`，**行号（从 0 起）就是这一集在本分片里的
  `episode_index`**。`--resume` 按录制顺序发号，而分片是隔片取集，两者不相等；
  超差重录时那一集也已经写进数据集了，所以它也占一行、标成 `reject`。
  不查账就会核错集 —— 而且核错了不报错，只是结论错。

不合格的集号按**合并后**的编号打印，交给
`lerobot-edit-dataset --operation.type delete_episodes` 官方删除，不在这里动数据。

用法：`python check_regen.py <场景> <录制输出目录> <源数据集根> <分片数>`
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from recipe import FRAME_TOLERANCE


def action_by_episode(root):
    """一份数据集里每一集的动作，列序按它自己声明的关节名。

    Args:
        root: 数据集根目录。

    Returns:
        `({集号: (帧数, 6) float32}, 关节名列表)`。
    """
    names = json.loads((Path(root) / "meta" / "info.json").read_text())["features"]["action"]["names"]
    per_ep = {}
    for f in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(f, columns=["episode_index", "action"])
        idx = np.asarray(table.column("episode_index").to_pylist())
        act = np.stack(table.column("action").to_pylist()).astype(np.float32)
        for ep in np.unique(idx):
            hit = act[idx == ep]
            prev = per_ep.get(int(ep))
            per_ep[int(ep)] = hit if prev is None else np.vstack([prev, hit])
    return per_ep, names


def read_order(path):
    """读一份分片账本。

    Args:
        path: `logs/shard<N>.order`。

    Returns:
        `[(源集号, "ok" 或 "reject")]`，下标即该集在本分片里的 `episode_index`。
    """
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        source_ep, verdict = line.split()
        rows.append((int(source_ep), verdict))
    return rows


def compare_actions(recorded, source):
    """录出来的一集是不是源集的 1:1 还原。

    Args:
        recorded: `(帧数, 6)`，重录数据集里的动作，列序已对齐源集。
        source: `(源帧数, 6)`，源数据集里的动作。

    Returns:
        `(是否合格, 说明)`。
    """
    delta = len(recorded) - len(source)
    if abs(delta) > FRAME_TOLERANCE:
        return False, f"帧数与源集差 {delta:+d}，超过容差 ±{FRAME_TOLERANCE} 帧"
    n = min(len(recorded), len(source))
    if not np.array_equal(recorded[:n], source[:n]):
        differs = (recorded[:n] != source[:n]).any(axis=1)
        return False, (f"动作与源集不符：{int(differs.sum())} 帧不同，"
                       f"最早在第 {int(np.flatnonzero(differs)[0])} 帧")
    # 差的那几帧必须落在源集自己的静止尾巴里 —— 少掉或多补的都得是"手臂已经停住"的帧。
    # 多补的来自遥操器保持源集末帧，少掉的取自源集末尾，两边都拿源集末帧当基准比。
    rest = recorded[n:] if len(recorded) > n else source[n:]
    if len(rest) and not np.array_equal(rest, np.repeat(source[-1:], len(rest), axis=0)):
        kind = "多补" if len(recorded) > n else "少掉"
        return False, f"{kind}的 {len(rest)} 帧不全等于源集末帧 —— 差的不只是静止段"
    return True, f"录 {len(recorded)} 帧 / 源 {len(source)} 帧（{delta:+d}）"


def main(argv) -> int:
    if len(argv) != 4:
        sys.exit(__doc__)
    scene, out_dir = argv[0], Path(argv[1])
    source_root, shards = Path(argv[2]), int(argv[3])

    source_acts, source_names = action_by_episode(source_root)

    bad, ok = [], []
    merged_base = 0
    for shard in range(shards):
        order = read_order(out_dir / "logs" / f"shard{shard}.order")
        rec_acts, rec_names = action_by_episode(out_dir / f"shard{shard}")
        if sorted(rec_names) != sorted(source_names):
            sys.exit(f"★ 关节名对不上：源 {source_names} / 重录 {rec_names}")
        # 按名字取列，不按列序 —— 列序若不同，按序比会把肩转的值和肘弯的值对着比，
        # 结果是"动作与源集不符"，而真正的原因是列没对齐。
        take = [rec_names.index(n) for n in source_names]
        if len(rec_acts) != len(order):
            sys.exit(f"★ 片 {shard} 有 {len(rec_acts)} 集，账上记了 {len(order)} 集 —— "
                     "对不上就无法判断哪一集是哪一集")
        for idx, (src_ep, verdict) in enumerate(order):
            merged_idx = merged_base + idx
            if verdict != "ok":
                bad.append((merged_idx, src_ep, "录制时已判超差并重录，这一份是作废件"))
                continue
            good, why = compare_actions(rec_acts[idx][:, take], source_acts[src_ep])
            if good:
                ok.append(merged_idx)
            else:
                bad.append((merged_idx, src_ep, why))
        merged_base += len(order)

    # 重录会让账上的行数多于源集数（作废的那几份也占行），所以"覆盖了几集"要按
    # **合格行覆盖到的源集号**去数，不能拿总行数比。
    covered = {src_ep for shard in range(shards)
               for src_ep, verdict in read_order(out_dir / "logs" / f"shard{shard}.order")
               if verdict == "ok"}
    print(f"  {scene}: 合格 {len(ok)} 行 / 共 {len(ok) + len(bad)} 行，"
          f"覆盖源集 {len(covered)}/{len(source_acts)} 集")
    if len(covered) != len(source_acts):
        print(f"  ★ 有 {len(source_acts) - len(covered)} 个源集没有合格的录制")
    if bad:
        print("  不合格（编号是**合并后**的集号）：")
        for merged_idx, src_ep, why in bad[:10]:
            print(f"    合并后 ep{merged_idx}（源集 ep{src_ep}）: {why}")
        if len(bad) > 10:
            print(f"    …… 共 {len(bad)} 集")
        print(f"  待删集号：[{','.join(str(m) for m, _, _ in bad)}]")
    print("CHECK_REGEN_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
