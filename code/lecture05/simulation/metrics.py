"""单变量实验共享工具：跑一次带增强指标的 reach（§3.7）。

在 ReachController 之外补充记录实验所需指标：
最终误差、达标时间、最小奇异值、触发限幅次数、末端路径长度。
"""

import numpy as np
import mujoco

from reach import (
    load_model,
    ReachController,
    EE_SITE,
    PHYSICS_DT,
)


def run_reach_metrics(
    target: np.ndarray,
    control_dt: float = 0.020,
    tolerance: float = 0.005,
    timeout_s: float = 5.0,
) -> dict:
    """跑一次 reach，返回 success 与各项指标。

    control_dt 只改变外层 IK 周期（DECIMATION 随之变化），
    不改动 model.opt.timestep —— 这是实验四研究“命令保持更久”的前提。
    """
    model = load_model(target)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = ReachController(model, data)

    decimation = round(control_dt / PHYSICS_DT)
    consecutive = 0
    error = float("inf")
    reached_time = None
    min_singular = float("inf")
    clipped_updates = 0
    n_control = 0
    path_length = 0.0
    prev_ee = None

    for step in range(int(timeout_s / PHYSICS_DT)):
        if step % decimation == 0:
            error = controller.command(target)
            n_control += 1
            min_singular = min(min_singular, controller.last_min_singular)
            if controller.last_clipped:
                clipped_updates += 1
            consecutive = consecutive + 1 if error < tolerance else 0

        mujoco.mj_step(model, data)
        ee = data.site(EE_SITE).xpos.copy()
        if prev_ee is not None:
            path_length += float(np.linalg.norm(ee - prev_ee))
        prev_ee = ee

        if consecutive >= 8 and reached_time is None:
            reached_time = float(data.time)
            break

    final_error = float(np.linalg.norm(target - data.site(EE_SITE).xpos))
    return {
        "success": reached_time is not None,
        "reached_time": reached_time,
        "final_error": final_error,
        "min_singular": min_singular,
        "clipped_updates": clipped_updates,
        "n_control": n_control,
        "path_length": path_length,
    }
