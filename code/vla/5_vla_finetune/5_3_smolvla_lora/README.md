# 5_3 LoRA 微调 SmolVLA（讲12 第 3 节）

数据、步数、warmup、图像增强、卡数、effective batch、评测环境参数全部与
`5_2_smolvla_full_sft/train_smolvla_sim_real.sh` 逐字相同 ——
**两轮之间唯一的差别就是「怎么改权重」**，这样两条 loss 曲线才对得起来。

| 文件 | 作用 |
|---|---|
| `train_lora.sh` | LoRA 微调入口。`bash train_lora.sh <smolvla\|pi0>` |

评测复用 `../5_2_smolvla_full_sft/eval_3scene.sh`，判据与全参那轮同一套。

## 一条踩过的坑：默认的 `target_modules` 对换本体这件事开得太窄

上游 `SmolVLAPolicy._get_default_peft_targets` 只挂**动作专家**的 q/v 投影
加五个具身投影，`r=16` 时可训练参数 742,656（占全模型 0.16%）。
照默认跑到 8000 步，三个评测点全是 0.0%，而全参微调在第一个点就 60%。

决定性的对读不是成功率而是 loss：**LoRA 在 7K 步是 0.198，全参在 1K 步已经 0.194** ——
不是学歪了，是欠拟合到差一个数量级的进度。

基座里可挂 q/v 的一共三族（读权重名单数出来的）：

| 位置 | q/v 数 | 默认挂了吗 |
|---|---|---|
| 动作专家 `lm_expert` | 32 | 挂了 |
| 文本塔 `vlm.model.text_model` | 32 | **没有** |
| 视觉塔 `vlm.model.vision_model` | 48 | **没有** |

而 2.4 节的结论正是「解冻视觉塔才跨得过新机器人的视觉差异」。三族一起挂、
`r` 提到 64、五个具身投影改成 `full_training_modules` 全训（低秩近似一个本来就不大的
投影没有意义）、学习率随之提到 3e-4，可训练参数变成 9,851,728（10M），
loss 立刻贴上全参。

## `--policy.use_peft` 不是这个开关

它是**加载一个已经训好的适配器**用的（见讲义 3.9 节）。开启 LoRA 训练靠 `--peft.*`。
