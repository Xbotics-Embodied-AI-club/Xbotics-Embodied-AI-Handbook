# ACT：训练 / 推理 / 异步部署全套

| 文件 | 干什么 |
|---|---|
| `train_act_libero.py` | LIBERO 短程单任务（bowl→plate）筛数据 + 训练 |
| `train_act_libero.sh` | 同一件事的官方 CLI 版：先切数据集子集，再 `lerobot-train` |
| `train_act_cuboid.sh` | SO-101 真机 cuboid 数据训练（`RESUME=true` 续训） |
| `infer_act_libero.py/.ipynb` | 本地加载 checkpoint 闭环推理 + 录像 + 时延统计 |
| `infer_act_libero_server.py` / `start_act_policy_server.sh` | 异步推理 server（策略跑在 GPU 机器上） |
| `infer_act_libero_client.py/.ipynb` / `run_act_cuboid_client.sh` | 异步推理 client（仿真 / 真机两种） |

第10讲 3.1 与 3.2 节用前三个训练入口，3.3 节逐层拆的就是 `train_act_libero.py` 串起来的那条链；
server-client 形态是第11讲的异步推理实验。

全部在 `code/` 下用 `uv run` 启动，例如：

```bash
uv run python vla/3_imitation_learning/3_1_act/train_act_libero.py
```

训练输出落 `outputs/`（不入库），样例结果见 `../result/3_1_act/`。推理脚本默认读
`outputs/act_libero_goal_plate/checkpoints/last/pretrained_model`，用 `train_act_libero.py`
训练时输出目录带时间戳，记得把路径改成自己那一次的。
