# π0-FAST 与 π0.5 推理

π 系两个后续版本各推理一次，与组1 的 π0 对照：

- **π0-FAST**（自回归离散）：动作块先 DCT 频域压缩再 BPE 合并成 FAST token，
  一次生成一串 token 解码回整个动作块——同为离散路线，比 OpenVLA 的逐步解码快得多。
- **π0.5**（连续动作头）：flow matching 出动作，训练配方引入离散 token 与分层推理，
  面向开放世界泛化。

文件：

- `pi0fast_demo.py` / `pi0fast_demo.ipynb` — π0-FAST 闭环一次（`lerobot/pi0fast-libero`）
- `pi05_demo.py` / `pi05_demo.ipynb` — π0.5 闭环一次（`lerobot/pi05_libero_finetuned_v044`）
- `pi0fast_eval.sh` / `pi05_eval.sh` — 官方 `lerobot-eval` 标准评测入口
  （多 episode pc_success 统计 + 自动录像进 `output/eval_*/`）

运行：`cd code && uv sync --extra gpu_x86`，然后
`uv run python vla/4_vla_inference/4_2_pi0fast_pi05_infer/pi0fast_demo.py`（或 pi05）。

结果：libero_goal task5「push the plate to the front of the stove」，初始状态 2，
π0-FAST 129 步 / π0.5 125 步成功，录像在 `output/`。
