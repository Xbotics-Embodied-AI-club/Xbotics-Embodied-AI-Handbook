# SmolVLA 推理

0.45B 的小模型路线代表：SmolVLM2 底座 + flow matching 动作专家，消费级显卡可跑。
加载 LIBERO 微调 checkpoint 闭环推理一次并录像；与 4_1 的 7B OpenVLA 对比，加载
和每步推理的速度差一个数量级。

- `smolvla_demo.ipynb` — 课堂走读版（分节讲解）
- `smolvla_demo.py` — 脚本版，跑通即产出 `output/smolvla_libero_success.mp4`
- `smolvla_eval.sh` — 官方 `lerobot-eval` 标准评测入口（rename_map 把 image/image2 映射到 checkpoint 的 camera1/2；实测 2 episode pc_success=50%）

运行：`cd experiments && uv sync --extra gpu_x86`，然后
`uv run python vla/4_vla_inference/4_4_smolvla_infer/smolvla_demo.py`。

结果：libero_goal task5「push the plate to the front of the stove」，初始状态 2，
126 步成功。异步推理（讲义 §7.4）是部署层改造：把同一个 `select_action` 搬进
PolicyServer、与机器人端解耦，模型本身不用改——server/client 实战见
`../3_imitation_learning/3_1_act/` 的部署脚本。
