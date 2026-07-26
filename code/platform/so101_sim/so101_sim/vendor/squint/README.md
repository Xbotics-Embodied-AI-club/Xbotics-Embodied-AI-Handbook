# vendor/squint — SO101 ManiSkill3 任务与机器人资产

本目录是 **squint** 上游仓库的 **sim 子集**，原样收录、未改动，供 `so101_sim` 适配器构建 SO101 仿真环境。

## 上游

- 仓库：https://github.com/aalmuzairee/squint
- 提交：`2a1f6e894e2a4cfd97a18dbe43b1570dde65fa42`（2026-03-04）
- 论文：*Fast Visual Reinforcement Learning for Sim-to-Real Manipulation*（arXiv:2602.21203）
- 许可：MIT（见 `LICENSE`，© 2026 Abdulaziz Almuzairee）；与 ManiSkill3(Apache-2.0)/SAPIEN(MIT)/SO101 mesh(TheRobotStudio Apache) 全兼容。

## 收录内容

只收 SO101 仿真必需的部分：

```
envs/
├── __init__.py                 # import 本包即向 ManiSkill 注册 8 个 SO101 任务
├── reach.py / lift.py /        # 4 类任务 × {Cube, Can}=8 个：
│   place.py / stack.py         #   SO101{Reach,Lift,Place,Stack}{Cube,Can}-v1
├── base_random_env.py          # 相机/域随机化基类（含 rgb_overlay 换外观钩子）
├── black_overlay.png           # 黑臂/白底外观贴图
└── robot/                      # SO101 本体：so101.py + so101.urdf/srdf + meshes/（27 个）
train_squint.py                 # SAC 训练器（ManiSkill 千级并行；CNNEncoder/Actor + Args）
utils.py                        # obs 包装（FlattenRGBD 后 DownsampleObs 到 16px 等）
```

## 未收录（真机部分，仿真闭环不需要）

`deploy.py`、`deploy_utils/`（真机部署 + Joy-Con + hidapi）、`examples/`、`results/`、`environment.yaml`（conda）。

## 用法

不直接 import 本目录。由 `so101_sim`（上上级包）在导入时把本目录加入 `sys.path` 并 `import envs`，完成 8 个任务的注册；随后：

- 仿真环境：`so101_sim.So101SimEnv` 或 `gym.make("SO101ReachCube-v1")`。
- RL 数据生产（`rl/3_offpolicy/3_1_so101_visual_sac/`）：`import so101_sim` 后 `import train_squint, utils` 取 `CNNEncoder`/`Actor`/`Args`/`DownsampleObsWrapper` 复用其 SAC 训练器与 obs 包装。

8 个任务对应 4 个 3D 打印件（cube / can / bin / large_cube），sim 物体与真机一致。
