"""无硬件 Sim2Real：延迟 + 观测噪声注入（第 5 讲 §6.3）。

在 SO-101 reach 闭环里加入两类明确的错误假设，检验控制闭环的敏感性：
  * 动作延迟：关节位置目标延迟若干「外层控制更新」（20 ms 为一档）；
  * 观测噪声：末端位置读数叠加高斯噪声。

成功判定使用「带噪观测」误差（接近部署接口），同时记录「仿真真值」误差
（独立评测）。二者同时保留，用以揭示观测误差与真实任务误差的区别。

实验矩阵（见讲义 §6.3 表）：

    延迟更新数  位置噪声标准差  重复 seed
    0          0.000           5 组（基线）
    1          0.000           5 组（延迟敏感性）
    2          0.000           5 组（延迟敏感性）
    0          0.001           5 组（抖动与保持判定）
    0          0.002           5 组（抖动与保持判定）
    2          0.002           5 组（组合效应）

运行：.venv/bin/python sim2real.py
输出：results/sim2real_summary.csv（汇总）与 results/sim2real/*.csv（逐条轨迹）。
"""

from collections import deque
from pathlib import Path
import csv

import mujoco
import numpy as np

from reach import (
    load_model,
    ReachController,
    ARM_JOINTS,
    EE_SITE,
    PHYSICS_DT,
    DECIMATION,
    RESULT_DIR,
)


class DelayedNoisyChannel:
    """延迟关节目标，并给末端位置观测加高斯噪声（讲义 §6.3 原样）。"""

    def __init__(
        self,
        initial_ctrl: np.ndarray,
        delay_updates: int,
        position_noise_std: float,
        seed: int,
    ):
        if delay_updates < 0:
            raise ValueError("delay_updates must be non-negative")
        self.queue = deque(
            [initial_ctrl.copy() for _ in range(delay_updates)]
        )
        self.position_noise_std = position_noise_std
        self.rng = np.random.default_rng(seed)

    def observe_position(self, true_position: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(
            loc=0.0,
            scale=self.position_noise_std,
            size=3,
        )
        return true_position + noise

    def delayed_ctrl(self, new_ctrl: np.ndarray) -> np.ndarray:
        self.queue.append(new_ctrl.copy())
        return self.queue.popleft()


class ChanneledController(ReachController):
    """在 ReachController 上接入延迟 + 噪声通道，只改写 command。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        delay_updates: int,
        position_noise_std: float,
        seed: int,
    ):
        super().__init__(model, data)
        self.channel = DelayedNoisyChannel(
            initial_ctrl=data.ctrl[self.act_ids],
            delay_updates=delay_updates,
            position_noise_std=position_noise_std,
            seed=seed,
        )

    def command(self, target: np.ndarray, damping: float = 0.02) -> float:
        # 观测带噪：误差来自带噪末端位置，而非仿真真值。
        measured_ee = self.channel.observe_position(
            self.data.site_xpos[self.site_id]
        )
        error = target - measured_ee

        mujoco.mj_jacSite(
            self.model, self.data, self.jacp, self.jacr, self.site_id
        )
        J = self.jacp[:, self.dof_ids]
        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3), 0.5 * error
        )
        norm = np.linalg.norm(dq)
        if norm > 0.04:
            dq *= 0.04 / norm

        q_cmd = self.data.qpos[self.qpos_ids] + dq
        low, high = self.model.actuator_ctrlrange[self.act_ids].T
        new_ctrl = np.clip(q_cmd, low, high)
        # 动作延迟：写入执行器的是被延迟后的关节目标。
        self.data.ctrl[self.act_ids] = self.channel.delayed_ctrl(new_ctrl)
        return float(np.linalg.norm(error))


def run_once(
    target: np.ndarray,
    delay_updates: int,
    position_noise_std: float,
    seed: int,
    tolerance: float = 0.005,
    timeout_s: float = 5.0,
) -> dict:
    """跑一次带扰动 reach，返回结果与逐物理步轨迹。"""
    model = load_model(target)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = ChanneledController(
        model, data, delay_updates, position_noise_std, seed
    )

    consecutive = 0
    error_measured = float("inf")
    rows: list[list[float]] = []
    reached_time = None
    min_true_error = float("inf")

    for step in range(int(timeout_s / PHYSICS_DT)):
        if step % DECIMATION == 0:
            error_measured = controller.command(target)
            consecutive = (
                consecutive + 1 if error_measured < tolerance else 0
            )

        mujoco.mj_step(model, data)
        ee_true = data.site(EE_SITE).xpos.copy()
        error_true = float(np.linalg.norm(target - ee_true))
        min_true_error = min(min_true_error, error_true)
        rows.append(
            [float(data.time), *ee_true.tolist(), error_true, error_measured]
        )

        if consecutive >= 8:
            reached_time = float(data.time)
            break

    final_true_error = float(
        np.linalg.norm(target - data.site(EE_SITE).xpos)
    )
    return {
        "success": reached_time is not None,  # 基于带噪观测的连续保持判定
        "reached_time": reached_time,
        "final_true_error": final_true_error,
        "min_true_error": min_true_error,
        "rows": rows,
    }


def main() -> None:
    target = np.array([0.30, 0.10, 0.20])
    # (delay_updates, position_noise_std_m, label)
    matrix = [
        (0, 0.000, "baseline"),
        (1, 0.000, "delay1"),
        (2, 0.000, "delay2"),
        (0, 0.001, "noise1mm"),
        (0, 0.002, "noise2mm"),
        (2, 0.002, "delay2_noise2mm"),
    ]
    seeds = [0, 1, 2, 3, 4]

    trace_dir = RESULT_DIR / "sim2real"
    trace_dir.mkdir(exist_ok=True)
    summary_path = RESULT_DIR / "sim2real_summary.csv"

    summary_rows = []
    for delay, noise_std, label in matrix:
        for seed in seeds:
            result = run_once(target, delay, noise_std, seed)

            trace_path = trace_dir / f"{label}_s{seed}.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    ["time_s", "ee_x", "ee_y", "ee_z",
                     "error_true_m", "error_measured_m"]
                )
                w.writerows(result["rows"])

            summary_rows.append(
                [
                    label, delay, f"{noise_std * 1000:.1f}", seed,
                    int(result["success"]),
                    "" if result["reached_time"] is None
                    else f"{result['reached_time']:.3f}",
                    f"{result['final_true_error']:.5f}",
                    f"{result['min_true_error']:.5f}",
                ]
            )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "delay_updates", "noise_std_mm", "seed",
             "success_measured", "reached_time_s",
             "final_true_error_m", "min_true_error_m"]
        )
        w.writerows(summary_rows)

    # 终端汇总：每个档位的成功率
    print(f"{'label':<18}{'success':>8}{'mean_final_true_err':>20}")
    labels = [m[2] for m in matrix]
    for label in labels:
        subset = [r for r in summary_rows if r[0] == label]
        n_ok = sum(r[4] for r in subset)
        mean_err = np.mean([float(r[6]) for r in subset])
        print(f"{label:<18}{f'{n_ok}/{len(subset)}':>8}{mean_err:>20.5f}")

    print(f"\nsummary -> {summary_path}")
    print(f"traces  -> {trace_dir}/")


if __name__ == "__main__":
    main()
