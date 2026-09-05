"""观感门的自检：它必须**能红**。

一道永远不会失败的门等于没有门，而这类缺陷不报错、只是安静地放行。所以每条判据都要
成对地测：合格的样子放行、不合格的样子被拦下。这里只测那条最容易写错的 —— 停顿。

停顿门的难点在于「哪些帧本来就该静止」：合拢、保持、松手、落稳一共二十几帧手臂不动，
算进去会把好集全枪毙；而把它们的长度当成"允许的静止额度"加给整条轨迹，又会让搬运途中
真停一秒多的集照样过门。两种写错法都不会报错。

跑：`pytest test_expert.py`（需要 worker 上那个 venv —— expert 间接依赖 sapien）
"""

import numpy as np

import expert

FPS = 30


def _straight_then_still(move_a, still, move_b):
    """一条「动 → 静 → 动」的末端轨迹，以及与之对应的帧数。"""
    tip = np.concatenate([np.linspace(0.0, 0.4, move_a),
                          np.full(still, 0.4),
                          np.linspace(0.4, 0.8, move_b)])
    return np.stack([tip, np.zeros(len(tip)), np.zeros(len(tip))], axis=1)


def test_dead_time_gate_ignores_frames_that_should_be_still(monkeypatch):
    """合拢那几十帧静止不该被算成停顿。"""
    track = _straight_then_still(40, 40, 40)
    monkeypatch.setattr(expert, "_tip_track", lambda *a, **k: track)
    mask = np.r_[np.ones(40, bool), np.zeros(40, bool), np.ones(40, bool)]

    faults = expert.quality_faults(np.zeros((120, 6)), slice(0, 40), mask, None, None, None)

    assert not any("停顿" in f for f in faults), faults


def test_dead_time_gate_catches_a_stall_while_the_arm_should_move(monkeypatch):
    """同一条轨迹，若那 40 帧本来该动，就必须被拦下（40/30 = 1.33s > 1.2s）。"""
    track = _straight_then_still(40, 40, 40)
    monkeypatch.setattr(expert, "_tip_track", lambda *a, **k: track)

    faults = expert.quality_faults(np.zeros((120, 6)), slice(0, 40),
                                   np.ones(120, bool), None, None, None)

    assert any("停顿" in f for f in faults), faults


def test_moving_mask_length_matches_the_trajectory():
    """掩码与轨迹逐帧对齐 —— 对不齐时 `zip(..., strict=True)` 会当场抛，不静默截断。

    这条看着琐碎，但它守的是一个会**静默错**的东西：掩码短一截时非 strict 的 zip 会
    悄悄只判前半条轨迹，后半段的停顿永远发现不了。
    """
    track = _straight_then_still(10, 10, 10)
    expert._tip_track = lambda *a, **k: track
    short = np.ones(5, bool)

    try:
        expert.quality_faults(np.zeros((30, 6)), slice(0, 10), short, None, None, None)
    except ValueError:
        return
    raise AssertionError("掩码比轨迹短却没抛 —— 这一段判据会静默只覆盖前几帧")
