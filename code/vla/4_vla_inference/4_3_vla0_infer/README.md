# VLA-0 推理：动作就是一串数字

VLA-0 把自回归离散推到极简：不加动作头、不扩词表，动作块直接离散成整数打印成
数字串，模型像聊天一样"说"出来，xgrammar 约束解码保证输出合法。demo 会把第一个
动作块的原始数字串打印出来——那就是模型的全部输出。

- `vla0_demo.ipynb` — 课堂走读版（分节讲解）
- `vla0_demo.py` — 脚本版，跑通即产出 `output/vla0_libero_success.mp4`
- `vla0_eval.sh` — 官方 `lerobot-eval` 标准评测入口（多 episode pc_success + 自动录像）

checkpoint 来自讲 15 的 GRPO 后训练实验（`rl/2_grpo_posttraining/2_2_grpo_vla0_libero/`，
数字串解码也改装自那里的 `model.py`），放在
`$DATASETS_ROOT/models/trained/xbotics_rl_grpo_vla0/grpo_runs/iter002/`。

运行：`cd code && uv sync --extra gpu_x86 && bash platform/lerobot/fetch_lerobot.sh`
（vla0_smol policy 由 `lerobot/0002-vla0-smol-policy.patch` 提供），然后
`uv run python vla/4_vla_inference/4_3_vla0_infer/vla0_demo.py`。

结果：libero_object task0「pick up the alphabet soup and place it in the basket」，
初始状态 0，154 步成功；首个动作块打印为 56 个整数（8 步 × 7 维，512 bin）。
