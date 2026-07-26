# ACT：训练 / 推理 / 异步部署全套

| 文件 | 干什么 |
|---|---|
| `train_act_libero.py` | LIBERO 短程单任务（bowl→plate）筛数据 + 训练 |
| `train_act_cuboid.sh` | SO-101 真机 cuboid 数据训练（支持 RESUME 续训） |
| `infer_act_libero.py/.ipynb` | 本地加载 checkpoint 闭环推理 + 录像 |
| `infer_act_libero_server.py` / `start_act_policy_server.sh` | 异步推理 server（脚本形态，GPU 机上跑） |
| `infer_act_libero_client.py/.ipynb` / `run_act_cuboid_client.sh` | 异步推理 client（仿真 / 真机两种） |

server-client 形态即讲11 的异步推理实验。全部在 `code/` 下用 `uv run` 启动；训练输出落 `outputs/`（不入库），样例结果见 `../result/3_1_act/`。
