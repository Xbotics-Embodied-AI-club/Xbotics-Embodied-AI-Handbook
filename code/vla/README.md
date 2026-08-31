# VLA 实验代码

VLA（视觉-语言-动作）课程段的演示代码。按**主题组**组织，每个组是一讲的完整材料
（代码 + 结果），组目录整体打包即可阅读与运行。

## 目录架构

```
vla/
├── 1_policy_rollout/          # 组1 端到端策略闭环
│   ├── 1_1_libero_env/        #   LIBERO 环境入门
│   └── 1_2_pi0_libero_rollout/#   π0 当黑盒跑通第一个闭环（+白噪声对照）
├── 2_data_collection/         # 组2 操作数据闭环
│   ├── 2_2_so101_setup/       #   SO-101 串口/相机绑定
│   ├── 2_3_teleop_record/     #   主从臂遥操作 + 数据录制
│   └── 2_4_lerobot_dataset/   #   LeRobot 数据集格式走读
└── 3_imitation_learning/      # 组3 模仿学习
    ├── 3_1_act/               #   ACT：训练 / 本地推理 / server-client 异步推理
    └── result/3_1_act/        #   结果样例
```

约定与 `../rl/` 完全一致：双编号 `<组>_<序>_<名字>` 只增不重排；讲次映射只在本表维护；
面向课堂的入口配同名中文分节 `.ipynb`；大产物（checkpoint/数据集）不入库。

## 课程讲次映射（含旧课模块号对照）

| 组 / 模块 | 内容 | 课程对应 | 旧 vla_class 模块号 |
|---|---|---|---|
| `1_policy_rollout/1_1_libero_env` | LIBERO 环境入门 | 讲8 | 3_4_libero |
| `1_policy_rollout/1_2_pi0_libero_rollout` | π0 第一个策略闭环 + 白噪声对照 | 讲8（讲11 π0 推理复用） | 6_3_pi0 |
| `2_data_collection/2_2_so101_setup` | SO-101 硬件绑定脚本 | 讲9 | 3_5_SO101 |
| `2_data_collection/2_3_teleop_record` | 遥操作与录制 | 讲9 | 4_3_tele_so101 |
| `2_data_collection/2_4_lerobot_dataset` | 数据集格式走读 | 讲9 | 4_2_lerobot_dataset |
| `3_imitation_learning/3_1_act` | ACT 训练/部署/解析全套 | 讲10（讲11 异步推理复用其 server/client） | 5_2_ACT |
| `4_vla_inference/` | OpenVLA / π0-FAST / π0.5 / VLA-0 / SmolVLA 推理 demo（各闭环一次 + lerobot-eval 入口） | 讲11 | — |
| `5_vla_finetune/`（待建，见组内 TODO.md） | 全量 / LoRA / 多卡微调 | 讲12 | — |

> 讲次调整只改本表，不动目录。编号只增不重排：`2_1` 已随统一 uv 环境废弃（原 LeRobot 环境修复说明），空号保留。RL 段（讲14-16）见 `../rl/README.md`。

## 环境

统一 uv 环境见 `code/pyproject.toml`：

```bash
cd code
bash platform/lerobot/fetch_lerobot.sh    # 拉 lerobot 源并打补丁（组2/3 需要）
uv sync --extra gpu_x86
```
