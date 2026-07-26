# OpenVLA 推理

7B 开源 VLA（自回归离散动作路线的代表）：Llama 2 底座 + 双视觉编码器，每维动作
离散成 256 个 bin，模型逐个"说出" 7 个动作 token。加载 LIBERO-10 官方微调
checkpoint，在 LIBERO 里闭环推理一次并录像。

- `openvla_demo.ipynb` — 课堂走读版（分节讲解）
- `openvla_demo.py` — 脚本版，跑通即产出 `output/openvla_libero_success.mp4`
- `openvla_eval.sh` — 官方 `lerobot-eval` 标准评测入口（多 episode pc_success + 自动录像）

运行：`cd experiments && uv sync --extra gpu_x86 && bash lerobot/fetch_lerobot.sh`
（openvla policy 由 `lerobot/0003-openvla-policy.patch` 提供），然后
`uv run python vla/4_vla_inference/4_1_openvla_infer/openvla_demo.py`。

结果：libero_10 task0「put both the alphabet soup and the tomato sauce in the
basket」，初始状态 0，247 步成功。每个仿真步做一次 7-token 贪心解码（7 次 7B
前向），推理明显慢于动作块模型——自回归离散动作的代价在这能直接体感到。
