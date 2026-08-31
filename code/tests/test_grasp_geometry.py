"""常驻断言：抓取数据集里「真夹住」必须是**几何贴合**，不是接触力。

为什么要这条：PhysX `contact_offset = 0.02` ⇒ 两体相距 2cm 以内就报接触力。
早先曾用"两指各 16–18N"宣布夹住；实测两爪碰撞网格到方块盒的最小距离，
`moving_jaw` **有缝帧占 19.8%**、两爪同时贴合仅 **78.5%**。
**基于力的判据在这个设定下天然分不清「夹住」和「靠近」**，所以这里只认几何量。

★窗口必须是「该夹着的帧」，不是「物体离台的帧」。实测 v5：
    held    (z > z0+1cm) 353 帧 → 贴合 6.2%、gap 中位 20.00mm
    grasped (env 判 grasped) 67 帧 → 贴合 49.3%、gap 中位 0.03mm
    held 里有 **299/353 帧根本不是 grasped**（放手后物体已在箱内静置）。
用 held 窗口的话，策略再好也过不了阈值 —— 那是我第一版断言的错。
几何代理（离台 + gap<contact_offset）也太松——同一批数据代理窗口贴合率 7.5% vs
env 自己的 grasped 窗口 49.3%。故 `replay_kit` 现在把逐帧 `is_item_grasped` 一并存进
h5 `infos/`，断言直接用它划窗口。
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

GRASP_ROOT = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_grasp"
MIN_FIRM_FRAC = 0.90        # 该夹着的帧里，两爪都贴合须占 ≥90%
CONTACT_OFFSET = 0.02       # PhysX contact_offset：gap 超过它连力都不报，显然不在夹持中
DATASETS = ["v5_place_cube_4cm_in_bin", "v6_place_cube_4cm_in_bin"]


def _grasp_window_gaps(h5_path):
    """返回「该夹着」的帧的 jaw_gap（米）。无 jaw_gap 字段则 None。"""
    gaps = []
    with h5py.File(h5_path) as f:
        for key in f:
            g = f[key]
            if "infos/jaw_gap" not in g:
                return None
            jg = g["infos/jaw_gap"][:]
            if "infos/is_item_grasped" in g:
                window = g["infos/is_item_grasped"][:].astype(bool)
            else:
                # 回落到几何代理（偏松，仅用于旧数据）
                item_z = g["env_states/actors/item"][:, 2]
                n = len(jg)
                window = (item_z[:n] > item_z[0] + 0.01) & (jg < CONTACT_OFFSET)
            gaps.append(jg[window])
    return np.concatenate(gaps) if gaps else np.array([])


@pytest.mark.parametrize("name", DATASETS)
def test_jaws_geometrically_closed(name):
    h5_path = GRASP_ROOT / name / "trimmed.h5"
    if not h5_path.exists():
        pytest.skip(f"{name}: 无 trimmed.h5")
    gaps = _grasp_window_gaps(h5_path)
    if gaps is None:
        pytest.skip(f"{name}: h5 无 infos/jaw_gap（生成时未导出几何量）")
    if len(gaps) < 20:
        pytest.skip(f"{name}: 夹持窗口帧数过少 ({len(gaps)})")

    firm_frac = float((gaps <= 0).mean())
    assert firm_frac >= MIN_FIRM_FRAC, (
        f"{name}: 夹持窗口内两爪都贴合的帧只占 {firm_frac:.1%} "
        f"(要求 ≥{MIN_FIRM_FRAC:.0%})；缝隙中位 {np.median(gaps) * 1000:+.2f}mm、"
        f"p90 {np.percentile(gaps, 90) * 1000:+.2f}mm。"
        f"接触力大不代表夹住——contact_offset=2cm 内就报力。"
    )
