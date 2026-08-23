# so101_sim —— SO101 机械臂仿真环境

基于 ManiSkill3 / SAPIEN（PhysX GPU 后端）的 SO-101 仿真，物体尺寸、机器人几何与
运动速度都按真机 KIT 标定。**独立包**：不依赖 lerobot 也能用。

## 三个场景

| 环境 id | 场景 |
|---|---|
| `SO101PickPlaceCube40-v1` | 抓 4cm 方块放进料盒 |
| `SO101PickPlaceCube20-v1` | 抓 2cm 方块放进料盒 |
| `SO101PickPlaceCylinder40-v1` | 抓 4cm 圆柱（立在桌面，抓圆面）放进料盒 |

一个场景一个环境，不做参数化派生。

## 两个入口

### 1. 原生（ManiSkill 批量环境）

数据生产与 RL 都走这个。观测是 **GPU 上的 torch tensor**，首维恒为 `num_envs`：

```python
import gymnasium as gym
import so101_sim  # import 即注册

env = gym.make("SO101PickPlaceCube40-v1", num_envs=64, obs_mode="state",
               sim_backend="gpu", render_mode="all")
obs, _ = env.reset(seed=0)      # obs.shape == (64, ...)，在 cuda 上
```

视觉 RL 另有一个便利构造器（降采样 16px + 颜色抖动 + 向量化），返回的仍是
ManiSkill 标准的 `ManiSkillVectorEnv`：

```python
from so101_sim import visual_rl_env
env = visual_rl_env("SO101PickPlaceCube40-v1", num_envs=1024, image_size=16)
```

### 2. lerobot 评测（标准单环境 gym.Env）

`So101SimEnv` 把批量 tensor 观测转成 lerobot 约定的 numpy 格式
（`{"agent_pos": ..., "pixels": {"top": ..., "wrist": ...}}`，两路相机都给）：

```bash
lerobot-eval --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1 --eval.n_episodes=20
```

`--env.type=so101_sim` 这个选项由 `platform/lerobot/0004-so101-sim-env.patch` 注册
（上游 lerobot 没有）。**依赖方向是单向的**：本包不 import lerobot，是 lerobot 通过补丁
认识本包——所以不打补丁时入口 1 照样能用。

## 结构

```
so101_sim/
├── envs.py                 三个分发场景（mixin 组合：双相机 / 真机尺寸 / 可达生成 / 速度包线）
├── tasks/                  任务基类与成功判定（place.py + base_random_env.py）
├── robots/
│   ├── so101_kit.py        KIT 版机器人（含底板、型材、两个相机支架）
│   ├── so101_kit_slow.py   真机速度包线版
│   ├── so101_base/         裸臂 SO101 本体与网格
│   └── kit_assets/         KIT URDF + 网格 + 物体
├── lerobot_env.py          入口 2：lerobot 评测口
├── wrappers.py              RL 观测包装 + visual_rl_env / state_rl_env
└── _core.py                两个入口共享的 gym.make 内核（防止环境定义漂移）
```

`tasks/` 与 `robots/so101_base/` 最初取自 squint（MIT），现已分叉自维护，
来源与分叉声明见 [`so101_sim/tasks/UPSTREAM.md`](so101_sim/tasks/UPSTREAM.md)。

## 装

由 `code/pyproject.toml` 以 editable 装入统一 uv 环境（`so101_sim = { path = ... }`），
`import so101_sim` 直接可用，不需要设 `PYTHONPATH`。
