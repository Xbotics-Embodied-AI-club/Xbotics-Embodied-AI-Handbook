"""最小版 LIBERO 异步推理 client：仿真在本地跑，策略在远端 server 上跑。

对应第11讲的异步推理实验。一圈数据流是：

1. client 在本地推进 LIBERO 仿真，拿到 observation；
2. 把 observation 发给远端 server；
3. server 在 GPU 上跑一次策略，一次返回一整段 action chunk；
4. client 按固定控制频率从本地队列里取动作，连续控制仿真。

关键在第 3、4 步之间：因为 server 一次给回一整段动作，client 不必等每一次推理返回，
控制循环的节奏就和推理的节奏解耦了。

用法：先运行 `infer_act_libero_server.py`，再在 `code/` 目录下运行
`uv run python vla/3_imitation_learning/3_1_act/infer_act_libero_client.py`
"""

import os
import pickle  # nosec: 只在可信的本地连接上传输，不接受外部来源的数据。
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace

import grpc
import numpy as np
import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.robot_client import RobotClient
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR


# LIBERO 底层是 MuJoCo，要离屏渲染出图像；EGL 后端不需要桌面环境。
os.environ["MUJOCO_GL"] = "egl"

# 相机名要和训练时存进数据集的 image key 一致，否则 server 那边对不上号。
CAMERAS = ("image", "image2")

# 观测图像尺寸，同样要和训练时一致。
OBS_HEIGHT, OBS_WIDTH = 256, 256


def raw_from_obs(env, env_pre, obs):
    """把 LIBERO 的观测转成异步 server 认识的原始字典。

    server 接收的是「原始机器人观测」格式（state 拆成一个个标量、图像是 HWC uint8），
    而不是 policy 的张量输入——因为归一化那一步要在 server 侧用它自己那份统计量来做。

    Args:
        env: 向量化的 LIBERO 环境。
        env_pre: LeRobot 的环境预处理器。
        obs: `env.reset()` / `env.step()` 返回的观测。

    Returns:
        `(raw, features)` 二元组。`raw` 是待发送的观测字典，`features` 描述这些键
        应该怎么还原成 LeRobot observation，server 靠它解读 `raw`。
    """
    batch = env_pre(add_envs_task(env, preprocess_observation(obs)))

    # state 拆成 state_0、state_1……是 LeRobot 原始机器人观测的惯用格式。
    state = batch["observation.state"][0].detach().cpu().float().numpy()
    raw = {f"state_{i}": float(v) for i, v in enumerate(state)}
    features = {f"state_{i}": float for i in range(len(state))}

    for cam in CAMERAS:
        img = batch[f"observation.images.{cam}"][0].detach().cpu()

        # env_pre 输出通常是 CHW float，而 server 要的是 HWC uint8，这里补上转换。
        img = img.permute(1, 2, 0) if img.shape[0] in (1, 3) else img

        # 有的路径下图像已经是 0~255，有的还是 0~1；按最大值判断，避免把已经是
        # 0~255 的图再乘一遍。
        img = img * 255 if img.dtype.is_floating_point and float(img.max()) <= 1.5 else img

        raw[cam] = img.clamp(0, 255).byte().numpy()
        features[cam] = (OBS_HEIGHT, OBS_WIDTH, 3)

    raw["task"] = batch.get("task", "")
    return raw, hw_to_dataset_features(features, OBS_STR, use_video=False)


# server 的地址，要和 infer_act_libero_server.py 里的 host/port 对上。
SERVER_ADDRESS = "127.0.0.1:8080"

# 控制频率，同样要和 server 的 fps 一致。
CONTROL_HZ = 20

# 一次向 server 要多少步未来动作。20Hz 下 100 步大约覆盖 5 秒，
# 这段余量就是 server 偶尔变慢时不会断动作的本钱。
ACTIONS_PER_CHUNK = 100


def make_client(policy_cfg, features):
    """搭一个只有异步队列、不接真实机器人的精简 client。

    这里绕开 `RobotClient.__init__`（它会去连真实硬件），只手工填上
    `receive_actions()` 需要的那几个字段。这样既不碰真机，又能直接复用 LeRobot
    自带的动作队列和多段 chunk 的聚合逻辑，不必自己再写一套缓冲策略。

    Args:
        policy_cfg: 从 checkpoint 读出的策略配置，只用它的 `type` 字段告诉 server 加载哪类策略。
        features: `raw_from_obs()` 返回的 feature 描述。

    Returns:
        已经连上 server、后台线程正在拉取动作的 client 对象。
    """
    c = object.__new__(RobotClient)

    # client 端不做推理，动作留在 CPU 上直接交给仿真执行。
    # 多段 chunk 覆盖同一时刻时，用 LeRobot 自带的加权平均去聚合。
    c.config = SimpleNamespace(
        client_device="cpu",
        aggregate_fn=get_aggregate_function("weighted_average"),
    )

    # 初始重连间隔设成一个控制周期，断线后能在下一拍就重试。
    c.channel = grpc.insecure_channel(
        SERVER_ADDRESS, grpc_channel_options(initial_backoff=f"{1 / CONTROL_HZ:.4f}s")
    )
    c.stub = services_pb2_grpc.AsyncInferenceStub(c.channel)

    c.shutdown_event = threading.Event()
    c.latest_action_lock = threading.Lock()
    c.action_queue_lock = threading.Lock()
    c.latest_action = -1
    c.action_chunk_size = ACTIONS_PER_CHUNK
    c.action_queue = Queue()
    c.action_queue_size = []
    c.start_barrier, c.must_go = threading.Barrier(1), threading.Event()

    # 握手：告诉 server 加载哪个 checkpoint、放在哪个设备上、观测该怎么解读。
    # 注意 checkpoint 路径是在 server 那台机器上解析的。
    policy_path = Path(
        "vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate/checkpoints/last/pretrained_model"
    )
    policy = RemotePolicyConfig(policy_cfg.type, str(policy_path), features, ACTIONS_PER_CHUNK, "cuda")
    c.stub.Ready(services_pb2.Empty())
    c.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy)))

    # 后台线程持续拉取 action chunk 并写进队列，控制循环只管从队列取。
    threading.Thread(target=c.receive_actions, daemon=True).start()
    return c


def send_obs(c, raw, step, must_go=False):
    """把一帧观测发给 server。

    Args:
        c: `make_client()` 建好的 client。
        raw: `raw_from_obs()` 产出的观测字典。
        step: 这帧观测对应的时间步，server 返回的动作从这个步号往后编号。
        must_go: 队列空了或是第一帧时置 True，表示 server 不许跳过这一帧。
    """
    obs = TimedObservation(time.time(), step, raw, must_go=must_go)
    c.stub.SendObservations(send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation, silent=True))


def pop_action(c):
    """从队列取一个动作，取不到就等，超时就报错。

    控制循环不能无限期等下去：等超过这个时限还没有动作，说明 server 或链路出了问题，
    这时候报错停下来，比让机器人拿着过期动作继续动要安全。

    Args:
        c: `make_client()` 建好的 client。

    Returns:
        一步动作张量（已搬到 CPU）。

    Raises:
        TimeoutError: 超时仍未收到远端动作。
    """
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            with c.action_queue_lock:
                a = c.action_queue.get_nowait()

            # 记下已执行到哪一步，聚合逻辑据此丢弃 server 送来的过期动作。
            with c.latest_action_lock:
                c.latest_action = a.get_timestep()

            return a.get_action().detach().cpu()
        except Empty:
            # 短睡一下，避免空转把 CPU 占满。
            time.sleep(0.001)
    raise TimeoutError("No remote action received.")


def main():
    """在本地跑一个 LIBERO episode，动作全部来自远端 server。"""
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_steps = 300
    task_suite, task_id = "libero_goal", 8

    policy_cfg = PreTrainedConfig.from_pretrained(
        Path("vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate/checkpoints/last/pretrained_model")
    )

    # 环境建法和本地推理那份完全一样，区别只在策略跑在哪。
    env_cfg = LiberoEnvConfig(
        task=task_suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=OBS_HEIGHT,
        observation_width=OBS_WIDTH,
        episode_length=max_steps,
    )
    env = make_env(env_cfg, n_envs=1)[task_suite][task_id]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)

    # 固定初始状态便于复现；设成 None 就用 LIBERO 默认的轮换顺序。
    episode_index = None
    if episode_index is not None:
        env.envs[0].episode_index = env.envs[0].init_state_id = episode_index

    obs, _ = env.reset(seed=[seed + (episode_index or 0)])
    raw, features = raw_from_obs(env, env_pre, obs)
    c = make_client(policy_cfg, features)
    success = False
    dt = 1 / CONTROL_HZ

    try:
        # 先发第 0 帧并等第一段动作回来。这次等待躲不掉——队列是空的。
        send_obs(c, raw, 0, must_go=True)
        action, t0 = pop_action(c), time.perf_counter()

        for step in range(max_steps):
            # 每步对齐到真实控制频率。server 慢没关系，只要队列里还有动作。
            target = t0 + step * dt
            if time.perf_counter() < target:
                time.sleep(target - time.perf_counter())

            # 第 0 步用上面等回来的那个动作，之后每步从队列取。
            if step:
                action = pop_action(c)

            action = env_post({"action": action.unsqueeze(0)})["action"]
            obs, _, terminated, truncated, info = env.step(
                action.cpu().numpy() if torch.is_tensor(action) else action
            )
            success = success or bool(info.get("final_info", {}).get("is_success", False))

            raw, _ = raw_from_obs(env, env_pre, obs)

            # 队列剩不到一半就提前发新观测，让 server 抢在动作用完之前算好下一段。
            # 这一步是整个异步方案能连续控制的关键：补货要早于断货。
            with c.action_queue_lock:
                q = c.action_queue.qsize()
            if q / ACTIONS_PER_CHUNK <= 0.5:
                send_obs(c, raw, step + 1, must_go=(q == 0))

            if bool(terminated[0]) or bool(truncated[0]):
                break

        # 实际频率若明显低于 CONTROL_HZ，说明有步在等动作，需要调大 chunk 或提前补货的阈值。
        print({"success": success, "steps": step + 1, "hz": (step + 1) / (time.perf_counter() - t0)})
    finally:
        c.shutdown_event.set()
        c.channel.close()
        env.close()


if __name__ == "__main__":
    main()
