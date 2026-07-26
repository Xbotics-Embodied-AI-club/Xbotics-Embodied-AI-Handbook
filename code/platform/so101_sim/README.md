# so101_sim — SO101 仿真闭环（接入 lerobot）

把 **squint** 的 ManiSkill3 SO101 任务接进 lerobot：**无机械臂**也能在仿真里跑 policy 评测，
并作为后续「RL 大量生成数据 → VLA 微调」的地基。定位对标 LIBERO——课程训练任务的统一落脚点。

## 8 个任务

squint 官方 8 任务（4 类 × {Cube, Can}，配 4 个 3D 打印件，sim 物体与真机一致）：

| 类别 | 任务 id |
|---|---|
| Reach | `SO101ReachCube-v1` / `SO101ReachCan-v1` |
| Lift | `SO101LiftCube-v1` / `SO101LiftCan-v1` |
| Place | `SO101PlaceCube-v1` / `SO101PlaceCan-v1` |
| Stack | `SO101StackCube-v1` / `SO101StackCan-v1` |

## 环境

在统一 uv 环境的 `gpu_x86` extra 里（`mani_skill==3.0.1` 已含），并先打好 lerobot 补丁：

```bash
cd experiments
bash lerobot/fetch_lerobot.sh     # 含 0004-so101-sim-env.patch（注册 env.type=so101_sim）
uv sync --extra gpu_x86
```

## 入口

policy 仿真评测走 lerobot 原生 CLI（真机/仿真同一条命令，只换 `--env.type`）：

```bash
# so101_sim 需在 PYTHONPATH 上（从 code/ 根运行即可）
PYTHONPATH=. lerobot-eval \
  --env.type=so101_sim \
  --env.task=SO101ReachCube-v1 \
  --policy.path=<训练好的 policy> \
  --eval.n_episodes=10 --eval.batch_size=1
```

输出 `pc_success` + 每集录像。`--policy.type=act`（不给 `--policy.path`）会用随机初始化的 policy
跑通全流程，可用于自检管线。

> RL 训练（`train_rl.py`）、数据集生成（`gen_dataset.py`）、回放换色（`replay.py`）属后续阶段，接入本模块的 `So101SimEnv`。

## 结构与边界

```
so101_sim/
├── __init__.py         # import 即注册 8 个 SO101 任务 + lerobot 评测入口 SO101Sim-v1
├── so101_sim_env.py    # So101SimEnv(gym.Env)：ManiSkill 观测 → lerobot 格式（唯一依赖 squint 的适配器）
└── vendor/squint/      # vendored squint（MIT）：8 任务定义 + SO101 机器人资产（见其 README）
```

关键边界：**只有 `so101_sim_env.py` 依赖 vendored squint / ManiSkill**；其余一切（评测、后续 RL
数据生成、机器人伪装）都只依赖 `So101SimEnv`。换仿真器或升级 squint 只动这一处。lerobot 侧的
`0004` 补丁只加一个 `EnvConfig` 注册，走 `make_env` 的通用 `package_name/gym_id` 分支。
