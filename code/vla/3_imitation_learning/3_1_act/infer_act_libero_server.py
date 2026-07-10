"""启动 LeRobot async inference server，给 LIBERO client 远程调用。

运行方式：
1. 先运行这个 server。
2. 再运行 infer_act_libero_client.py。
3. client 会把 observation 发过来，server 加载 policy 并返回 action chunk。
"""
from concurrent import futures
import logging

import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc


# 课堂演示配置：只需要改这里，不需要命令行参数。
# HOST/PORT 是 client 连接的地址；本机测试用 127.0.0.1 即可。
HOST = "127.0.0.1"
PORT = 8080

# FPS 需要和 client 控制频率一致。LIBERO/robosuite 这里按 20Hz 做实时控制。
FPS = 20

# server 人为等待的推理延迟。课堂本地测试设为 0，让结果尽快返回。
INFERENCE_LATENCY = 0.0  # 0 表示 action chunk 一算好就返回。

# 如果 server 长时间收不到 observation，就会 timeout；本地仿真可以给宽一点。
OBS_QUEUE_TIMEOUT = 10.0

# gRPC 线程池大小；这个 demo 只有一个 client，4 个 worker 足够。
WORKERS = 4


def main() -> None:
    # 打印 LeRobot server 的连接、加载 policy、推理耗时等日志。
    logging.basicConfig(level=logging.INFO)

    # 这里直接使用 LeRobot 自带的 PolicyServer。
    # 它负责接收 observation、跑 policy、再把 action chunk 发回 client。
    cfg = PolicyServerConfig(
        host=HOST,
        port=PORT,
        fps=FPS,
        inference_latency=INFERENCE_LATENCY,
        obs_queue_timeout=OBS_QUEUE_TIMEOUT,
    )
    policy_server = PolicyServer(cfg)

    # LeRobot 的 async inference 通信走 gRPC。
    # services_pb2_grpc.add_AsyncInferenceServicer_to_server 会把 PolicyServer 注册成 RPC 服务。
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=WORKERS))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    # start() 之后 server 开始监听；wait_for_termination() 会一直阻塞。
    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        # Ctrl+C 时关闭 gRPC server，同时通知 LeRobot PolicyServer 停止内部线程。
        policy_server.logger.info("KeyboardInterrupt received, shutting down server.")
        server.stop(grace=0)
    finally:
        policy_server.stop()


if __name__ == "__main__":
    main()
