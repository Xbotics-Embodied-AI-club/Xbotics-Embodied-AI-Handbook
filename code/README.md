# code/ — 演示代码 + 统一 uv 环境

> **本目录（讲 8–16 的代码）**：来自 [Xbotics-One](https://github.com/Xbotics-Embodied-AI-club/Xbotics-One)（已归档）的最终版，按主题组组织而非按讲组织——`vla/`（讲 8–12）与 `rl/`（讲 14–16），讲次映射见 [vla/README.md](vla/README.md) 与 [rl/README.md](rl/README.md)；`platform/` 为共享底座（LeRobot 获取与补丁、SO101 仿真环境、RDK 板端）。
> 环境为**统一 uv 管理**（本目录 `pyproject.toml`，用法见下文），不使用每讲独立 venv。讲 1–7、17–21 的 `lectureNN/` 目录不属本环境。

Xbotics 教学环境的**可跑演示代码树**：VLA（视觉-语言-动作）课程的 notebook / demo 脚本，以及配套的 RDK 板端部署工具。所有课程共用 `code/` 根的**一套 uv 管理 Python 环境**。

## 目录

```
code/
├── pyproject.toml / uv.lock      # 统一环境（按机器选一个 extra：gpu_x86 / nogpu_x86 / rdk_s100）
├── vla/                          # VLA 课程演示代码（按主题组组织，见 vla/README.md）
│   ├── 1_policy_rollout/         # 端到端策略闭环：LIBERO + π0 第一个闭环
│   ├── 2_data_collection/        # 操作数据闭环：SO-101 遥操采集 + LeRobot 数据集
│   ├── 3_imitation_learning/     # 模仿学习：ACT 训练/部署/解析
│   └── 4_vla_inference/          # VLA 推理导览：OpenVLA/π0-FAST/π0.5/VLA-0/SmolVLA 各闭环一次
├── rl/                           # RL 课程演示代码（按主题组组织，见 rl/README.md）
│   ├── 1_rl_basics/              # G1 行走 + 动作跟随，REINFORCE→A2C→PPO 三算法对照
│   ├── 2_grpo_posttraining/      # GRPO 后训练：VLM 数数 + VLA-0 自我提升
│   └── 3_offpolicy/              # Off-policy：SO101 视觉 RL（squint）+ HIL-SERL 真机（讲16）
└── platform/                     # 上游框架 + 硬件 + 部署（非课程 demo，被 import/fetch）
    ├── lerobot/                  #   lerobot 本地补丁 + fetch + 上游源（见 platform/lerobot/…）
    ├── rdk/                      #   地瓜 RDK 板端部署（ACT 上板，见 platform/rdk/README.md）
    └── so101_sim/                #   SO101 仿真环境包（editable；squint/ManiSkill3，入 lerobot-eval + RL 数据生成）
```

> 演示代码面向课堂：常量就近内联、自上而下按讲解顺序读，不用命令行参数层。

## 一、环境（uv）

改 `pyproject.toml` 后用 `uv sync`，不要 `pip install`。只有三种部署形态，每台机器按硬件选一个 extra（彼此互斥）：

| extra | 用途 | torch | lerobot |
| --- | --- | --- | --- |
| `gpu_x86` | x86 GPU 工作站：VLA + RL 全部训练/推理/演示（含 GRPO、GVHMR、ACT BPU 导出、SO101 仿真 ManiSkill3） | cu128（GPU，CUDA 12.8） | `lerobot[all]` |
| `nogpu_x86` | x86 无 GPU：SO-101 数据采集 + ACT ONNX/BPU 导出 | CPU | `lerobot[feetech]` |
| `rdk_s100` | 地瓜 RDK S100/S600 板端：SO-101 数据采集（aarch64，`--no-editable`） | PyPI aarch64 | `lerobot[feetech]` |

```bash
cd experiments
bash platform/lerobot/fetch_lerobot.sh   # 拉取 lerobot 0.5.1 源并打补丁（不入库）
uv sync --extra gpu_x86
```

### platform/lerobot（本地补丁，不分发源树）

lerobot 不提交源码：`platform/lerobot/fetch_lerobot.sh` 从 git 拉取 v0.5.1 到 `platform/lerobot/lerobot/`（gitignore）并打 `0001`/`0002`/`0003`/`0004` 补丁，再以 editable 安装。补丁内容见 `platform/lerobot/*.patch`（`0004` 注册 `so101_sim` 仿真 env）。`so101_sim` 本身是 `platform/so101_sim/` 的 editable 包（`import so101_sim` 直接可用，无需 PYTHONPATH）。

## 二、环境变量

代码直接读取环境变量、不设默认值。复制模板并填值：

```bash
cp .env.example .env          # 填 HF_TOKEN / WANDB_API_KEY / DATASETS_ROOT
cp .envrc.example .envrc && direnv allow   # 由 DATASETS_ROOT 派生 HF_HOME / HF_LEROBOT_HOME
```

- `DATASETS_ROOT` —— 数据集 / 模型产物根（默认仓内 `code/datasets/`，开箱即用，可改挂载点）
- `HF_HOME = $DATASETS_ROOT/hfcache`（HF 下载缓存，默认仓库本地）、`HF_LEROBOT_HOME = $HF_HOME/lerobot`
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `WANDB_API_KEY`

## 三、机器约定

- **x86 训练机**：带 GPU，CUDA 锁定 12.8，用于训练 / 导出 / 编译。
- **x86 数采机**：无显卡，用于 SO-101 遥操作数据采集。
- **RDK S100 / S600**：地瓜开发板，板端 BPU 推理；ACT 上板流程见 `platform/rdk/`。
