# 5_4 全量微调 → SO-101 真机遥操数据（讲12）

用一批公开的 SO-101 遥操作数据，对 **π0 / SmolVLA / ACT** 三个策略做全参数微调，
再离线验收 checkpoint；有真机时可以接着走 `deploy/` 把策略跑到臂上。

和 `5_2` 的区别只在数据源：`5_2` 用仿真里 RL 专家自动产出的演示，这里用人真实遥操录下来的
九个任务。数据是别人机位采的，光照和白平衡跟自己的臂必然不同，所以训练脚本把图像增强的
光度扰动开得比官方默认更宽。

## 文件

| 文件 | 作用 |
|---|---|
| `train_so101_real.sh` | 微调入口。改脚本里的 `MODEL` 选 `pi0` / `smolvla` / `act`；`NUM_GPUS` 大于 1 时自动走 `accelerate launch --multi_gpu` |
| `verify_checkpoint.py` | 逐个 checkpoint 验收：文件齐不齐、归一化是不是真统计量、离线预测有没有赢过"照抄当前关节角"这条基线 |
| `infer_demo.py` | 拿某个 checkpoint 回放一集数据，把预测动作和遥操真值画在一起逐关节对比 |
| `deploy/` | 把策略接到真机：推理在 x86 GPU 机上，板子只负责读舵机、开相机、下发动作。见 `deploy/README.md` |

## 运行

```bash
cd code
uv sync --extra gpu_x86

# 1) 微调（先在脚本里选 MODEL）
bash vla/5_vla_finetune/5_4_so101_real_sft/train_so101_real.sh

# 2) 验收：先验一个欠训的存点当"及格地板"，再拿末点跟它比
python vla/5_vla_finetune/5_4_so101_real_sft/verify_checkpoint.py \
    $DATASETS_ROOT/so101/outputs/pi0_9task/checkpoints/002500/pretrained_model
python vla/5_vla_finetune/5_4_so101_real_sft/verify_checkpoint.py \
    $DATASETS_ROOT/so101/outputs/pi0_9task/checkpoints/last/pretrained_model \
    $DATASETS_ROOT/so101/outputs/pi0_9task/checkpoints/002500/metrics.json

# 3) 单集回放对比
python vla/5_vla_finetune/5_4_so101_real_sft/infer_demo.py
```

产物落 `$DATASETS_ROOT/so101/outputs/<model>_9task/`，不进 git。

## 验收判据为什么不是相关系数

这批数据里 `action` 是 Leader 臂目标位姿、`observation.state` 是 Follower 臂实测位姿，两者天然贴在一起。
实测过：什么都不学、原样照抄 `state`，逐关节 `corr(pred, action)` 就有 0.936–0.995，
全部能过 0.9——那条门槛没有任何区分力。

`verify_checkpoint.py` 换成三个相对量，基线由数据自己定义，不用外部拍阈值：

| 指标 | 定义 | 及格线 |
|---|---|---|
| `ratio` | `MAE(pred, action) / MAE(state, action)` | 照抄基线恒为 1，模型要明显低于 1 |
| `delta_corr` | `corr(pred − state, action − state)` | 只看"下一步往哪挪"学到没有，要大于 0 |
| `delta_std` | `std(pred − state) / std(action − state)` | 抓"输出≈state"这种坍缩 |

给第二个参数（一个欠训 checkpoint 的 `metrics.json`）时，还会额外要求每个关节都比它更好。

## 一条边界

`verify_checkpoint.py` 和 `infer_demo.py` 做的都是**开环逐帧比对**：每帧 `policy.reset()`
后独立预测。真机上动作是连续下发的，误差会逐步累积，所以这两个脚本过了只说明模型学到了
非平凡的东西，**不等于真机可用**。本仓库没有 SO-ARM101 真机，闭环那一步要自己补。
