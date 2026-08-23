#!/usr/bin/env bash
# 在 x86 GPU 机上起策略服务端。板子那侧跑 client。
#
# 为什么要拆成两边：Orin Nano 只有 7.4G 统一内存，π0 光权重就 3G 多、SmolVLA 900M，
# 板上加载完再推理，显存和延迟都撑不住。所以让板子只干它必须在现场干的事
# （读舵机、开相机、下发动作），策略放 GPU 机上跑，中间过 gRPC。
#
# server 本身不指定模型 —— 加载哪个 checkpoint 是 client 握手时告诉它的，
# 所以这一个 server 可以先后接 π0 / ACT / SmolVLA 三种 client，不用重启。
#
#   bash vla/5_vla_finetune/5_4_so101_real_sft/deploy/start_policy_server.sh
#
# 起好之后确认防火墙放开 8080，然后在板子上跑对应的 run_*_client.sh。

set -euo pipefail

PORT=8080
FPS=30

uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port="$PORT" \
  --fps="$FPS"
