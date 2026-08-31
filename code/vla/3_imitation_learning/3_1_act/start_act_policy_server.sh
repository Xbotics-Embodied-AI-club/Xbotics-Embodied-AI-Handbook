#!/usr/bin/env bash
# 用 LeRobot 官方 CLI 起异步推理 server，等价于 infer_act_libero_server.py。
# server 本身不指定模型：加载哪个 checkpoint 是 client 握手时告诉它的，
# 所以同一个 server 可以先后服务不同的策略。
#
# 在 code/ 目录下运行：bash vla/3_imitation_learning/3_1_act/start_act_policy_server.sh
set -euo pipefail

# 监听所有网卡，这样另一台机器上的机器人 client 也连得进来。
uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080
