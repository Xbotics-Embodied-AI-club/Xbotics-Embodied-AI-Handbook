"""so101_sim — 把 squint 的 ManiSkill3 SO101 仿真接入 lerobot。

import 本包即完成两件事：

1. 把 vendored squint 的 `envs` 包放上 `sys.path` 并导入，向 ManiSkill 注册 8 个 SO101 任务
   （SO101{Reach,Lift,Place,Stack}{Cube,Can}-v1）。
2. 向 gymnasium 注册 lerobot 评测入口 `SO101Sim-v1`（→ :class:`So101SimEnv`）。

于是 `lerobot-eval --env.type=so101_sim --env.task=SO101ReachCube-v1` 时，lerobot 的
`make_env` 会先 `import so101_sim`（触发上面两步）再 `gym.make("SO101Sim-v1", task=...)`。
"""

import os
import sys

# vendored squint 的 envs 用仓根绝对导入（`import envs.robot...`），
# 把它的根目录放到 sys.path 最前，`import envs` 即命中这份 vendored 副本。
_SQUINT_ROOT = os.path.join(os.path.dirname(__file__), "vendor", "squint")
if _SQUINT_ROOT not in sys.path:
    sys.path.insert(0, _SQUINT_ROOT)

import envs as _squint_envs  # noqa: E402,F401  导入即注册 8 个 SO101 任务

from gymnasium.envs.registration import register  # noqa: E402

from so101_sim.so101_sim_env import So101SimEnv  # noqa: E402
from so101_sim.train_env import make_train_env  # noqa: E402

register(id="SO101Sim-v1", entry_point="so101_sim.so101_sim_env:So101SimEnv")

__all__ = ["So101SimEnv", "make_train_env"]
