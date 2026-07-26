#!/usr/bin/env bash
# 用 bash 运行这个脚本。
set -euo pipefail
# 任一命令失败就退出。

uv run python \
`# 统一 uv 环境启动（在 code/ 下运行）。` \
  -m lerobot.async_inference.policy_server \
`# 启动 LeRobot 的 async inference policy server。` \
  --host=0.0.0.0 \
`# 监听所有网卡，方便机器人 client 连接。` \
  --port=8080
`# 远程推理服务端口写死为 8080。`
# 这个 server 本身不指定模型，模型会在 client 握手时发给远程 GPU 机加载。
