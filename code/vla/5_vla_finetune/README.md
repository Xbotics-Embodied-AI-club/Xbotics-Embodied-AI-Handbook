# 组5 · VLA 微调实战（讲12）

组4 跑的都是别人训好的权重；这一组开始，权重由我们自己训出来。两条路线各占一个模块，
差别只在**数据从哪来**——一条完全不碰真机，一条用公开的真机遥操数据。

| 模块 | 数据来源 | 微调对象 | 评测方式 |
|---|---|---|---|
| `5_2_smolvla_full_sft/` | SO-101 仿真：已有的仿真演示数据集 | SmolVLA，全参数微调 | `lerobot-eval` 在同一仿真里报 `pc_success` |
| `5_4_so101_real_sft/` | 公开的 SO-101 真机遥操数据 | π0 / SmolVLA / ACT，全参数微调 | 离线验收（`verify_checkpoint.py`）+ 可选真机部署（`deploy/`） |

两条路线都走 lerobot 官方的 `lerobot-train` 入口，不自写训练循环——全参微调、图像增强、
学习率调度这些都是官方 config 里现成的开关，命令行传进去就够了。

## 这一组真正的难点不是训练

微调命令本身只有一行，会把人挡在门外的是另外两件事：

1. **观测特征对齐。** 预训练权重按它自己那套相机命名训出来（`smolvla_base` 是
   `camera1/2/3`），你的数据集叫别的名字（仿真那份只有一路 `base_camera`），不改名直接跑
   会抛 `Feature mismatch`。`--rename_map` 就是干这件事的，`5_2` 的 README 有完整说明。
2. **归一化伴随文件。** π0 这类策略把状态和动作按数据集统计量标准化，真正的 mean/std 不在
   模型权重里，而在两个几 KB 的伴随文件里。少了它们，加载时会**静默**退回恒等归一化——
   loss 照样好看，输出动作的尺度整体是错的。`5_4/verify_checkpoint.py` 第一件事就是验这个。

## 跑之前

环境统一走 `code/pyproject.toml` 的 uv 环境：

```bash
cd code
uv sync --extra gpu_x86
```

大文件（数据集、checkpoint）都落共享数据根 `$DATASETS_ROOT`，不进 git。各模块的具体命令
见各自的 README。

## 与讲义的对应

- 讲12 2.3 节：全量 SFT 的两条路线，就是上面那张表；
- 讲12 3.9 节：在同一条 `lerobot-train` 后面加 `--policy.use_peft=true --peft.method_type=LORA`
  即可切成 LoRA 微调，注入位置由 lerobot 的策略默认值给出；
- 讲12 4.6 节：多卡训练。`5_4` 的 `train_so101_real.sh` 里那个 `NUM_GPUS` 开关，走的就是
  `accelerate launch --multi_gpu`。
