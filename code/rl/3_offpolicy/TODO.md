# TODO · 组3 Off-policy：值学习地基 → 连续控制 → 真机落地（讲16）

off-policy 六级由简到繁依次提升，编号即教学序。详细规划见
`docs/superpowers/plans/RL-exp-009-plan.md` + `docs/superpowers/specs/RL-exp-009-design.md`。

## 3_1_cartpole_value_rl/（值学习入门：Q-learning → DQN，CartPole 贯穿）

- [ ] `train_v1_qlearning.py`：CartPole 状态分桶 + 表格 Q（故意无网络/无 Lightning——"没有网络"即教学点）。
- [ ] `train_v2_dqn.py`：网络 Q + 经验回放 + 目标网络（Lightning 四件套，自包含）。
- 两级统一报回合回报，直接看结果对照；无 model.py、无 env.py、无 rollout.py。

## 3_2_so101_offpolicy/（SO101 连续控制：DDPG → TD3 → SAC → squint 分布式 SAC）

squint 与 BeyondMimic 同地位的 RL 教学案例，顶点 v6 顺手产 VLA 仿真数据（callback）。

- [ ] `train_v3_ddpg.py` / `train_v4_td3.py` / `train_v5_sac.py`：连续控制阶梯，**各自包含**（本算法的模型 `nn.Module` + `LightningModule` 更新 + 回放，无共享 model.py——核心在每个 model）；单代表任务做阶梯对照。
- [ ] `train_v6_squint.py`：C51 分布式 SAC（现已验证 0.99），换 16px 视觉，铺 8 任务。
- [ ] `datagen/`（数据生成独立文件夹）：`rollout.py`（采 env_states）→ `replay.py`（换外观重渲）→ `to_lerobot.py`（convert）→ `gen_dataset.py`（编排），产 `so101_sim/<task>` LeRobotDataset。
- SO101 环境不在本模块定义——统一从 `platform/so101_sim`（`make_train_env`）消费；无 env.py。

## 3_3_hilserl_so101/（HIL-SERL 真机；讲16，讲义与实验双缺口，待建）

- [ ] SO-101 接触型任务（插拔/抓放）整套 HIL-SERL：示范打底 + 人工干预 + 视觉奖励分类器 + actor-learner 解耦；产出「失败→成功」视频与干预次数下降曲线。
- [ ] 仿真兜底：无真机时在 `so101_sim` 接触任务跑同样 off-policy + 人在环流程。

## 代码风格（后续实现必须完全匹配既有风格）

1. **课堂演示取向**：常量就近内联（不集中 config 块），阅读顺序=讲解顺序，不堆 try/except/抽象层。
2. **禁止**：argparse/args、mock、monkeypatch、改第三方库、`os.environ.get`/默认值/存在性检查、机器名/绝对路径/内部任务号。
3. **自写训练四件套**：普通 `Dataset`+`LightningDataModule`、普通 `nn.Module`+`LightningModule`，入口 `trainer.fit(model, data)`（参照 rl/1_1、rl/2_2）。数据生产线薄包 vendored 训练器不受此约束（vendor/ 例外）。
4. **路径**：只直读 `DATASETS_ROOT`/`HF_HOME`；组内数据用 `parents[1]/"data"/...` 相对路径；产物落 `DATASETS_ROOT/models/trained/`。
5. **notebook**：先 `.py` 跑通再转同名 `.ipynb`；中文编号分节，开篇讲「做什么/为什么/与前后模块关系」，每节讲动机+关键行+与上一版 diff；代码 cell 与 .py 逐行一致、无输出（参照 rl/1_1/train_v1_reinforce.ipynb）。每个 train_v* 配同名 notebook；`datagen/` 脚本只 `.py`。
6. **模块必备 README**（定位/文件表/运行/结果），环境走统一 `code/pyproject.toml` extra。
