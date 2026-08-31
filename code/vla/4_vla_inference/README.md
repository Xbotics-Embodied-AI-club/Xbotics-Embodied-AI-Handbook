# 组4 · VLA 推理导览（讲11）

讲 11 介绍的模型，这里挑四个在 LIBERO 里真跑一次：同一套「observation 进 → action 出」
闭环模板（组1 的 `pi0_demo.py`），只换模型。跑完四个 demo，两条技术路线就不再是
概念——**自回归离散动作**（OpenVLA / π0-FAST）与**连续动作头**
（π0 / π0.5 / SmolVLA）在代码上的差别，全部藏在 `select_action` 内部。

| 模块 | 模型 | 路线 | checkpoint |
|---|---|---|---|
| `4_1_openvla_infer/` | OpenVLA（7B） | 自回归离散（256 bin/维，逐 token） | `openvla/openvla-7b-finetuned-libero-10` |
| `4_2_pi0fast_pi05_infer/` | π0-FAST | 自回归离散（DCT+BPE 压缩） | `lerobot/pi0fast-libero` |
| 〃 | π0.5 | 连续动作头（flow matching+分层） | `lerobot/pi05_libero_finetuned_v044` |
| `4_4_smolvla_infer/` | SmolVLA（0.45B） | 连续动作头（小模型） | `lerobot/smolvla_libero` |

> 编号只增不重排：`4_3` 为空号，保留不复用。

π0 本体的闭环在 `../1_policy_rollout/1_2_pi0_libero_rollout/`；异步推理（讲11 §7.4）
是部署层改造，server/client 实战复用 `../3_imitation_learning/3_1_act/`。

每个模块都是三件套：demo `.py`（逐行走读，固定成功局）+ 同名 `.ipynb`（分节讲解）
+ `*_eval.sh`（官方 `lerobot-eval` 标准评测：多 episode pc_success 统计 + 自动录像）。

环境：`cd code && uv sync --extra gpu_x86`。4_1 用的 openvla policy 在我们维护的
lerobot fork 里（`xbotics` 分支），`uv sync` 一并装好；权重自动下载落 `$HF_HOME`。
