"""启动 LeRobot 的异步推理 server，让策略跑在 GPU 机器上、由远端 client 调用。

对应第11讲的异步推理实验：策略推理和机器人控制拆到两台机器上，server 一次返回一段
action chunk，client 按自己的控制频率消费，这样单次推理慢一点也不会把控制循环卡住。

用法：先起这个 server，再运行 `infer_act_libero_client.py`。
client 会在握手时把 checkpoint 路径发过来，server 据此加载策略。

在 `code/` 目录下运行：`uv run python vla/3_imitation_learning/3_1_act/infer_act_libero_server.py`
"""
from concurrent import futures
import logging

import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc


def main() -> None:
    """建好 gRPC server、注册 LeRobot 的 PolicyServer，然后一直监听。"""
    # 打开 INFO 日志才看得到连接建立、策略加载、每次推理耗时这些关键信息。
    logging.basicConfig(level=logging.INFO)

    cfg = PolicyServerConfig(
        # 监听回环地址表示只接受本机 client；跨机部署时改成 0.0.0.0。
        host="127.0.0.1",
        port=8080,
        # 要和 client 的控制频率一致，server 靠它推算每个动作对应的时间戳。
        fps=20,
        # 人为注入的推理延迟，用来模拟慢网络或慢模型；本地实验设 0，算完立刻返回。
        inference_latency=0.0,
        # 超过这个秒数没收到新观测就判定 client 掉线。仿真跑得慢，给宽一点。
        obs_queue_timeout=10.0,
    )
    policy_server = PolicyServer(cfg)

    # LeRobot 的异步推理走 gRPC；这一步把 PolicyServer 注册成 RPC 服务。
    # 只有一个 client，4 个 worker 线程足够。
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        # 除了停 gRPC，还要让 PolicyServer 收掉自己的内部线程，否则进程退不干净。
        policy_server.logger.info("KeyboardInterrupt received, shutting down server.")
        server.stop(grace=0)
    finally:
        policy_server.stop()


if __name__ == "__main__":
    main()
