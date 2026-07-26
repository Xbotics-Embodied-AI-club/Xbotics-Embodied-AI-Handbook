"""最小版 LIBERO realtime client，通过 LeRobot async inference 远程推理。

整体数据流：
1. client 在本地跑 LIBERO 仿真环境，拿到 observation。
2. client 把 observation 发给远端 server。
3. server 在 GPU 上跑 policy，一次返回一段 action chunk。
4. client 按 20Hz 从 action queue 里取动作，连续控制 LIBERO。
"""

import os
import pickle  # nosec: 本课堂 demo 默认是可信本地连接。
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


os.environ.setdefault("MUJOCO_GL", "egl")

# 课堂演示配置：只需要改这里，不需要命令行参数。
# POLICY_PATH 需要和训练输出目录里的 pretrained_model 对齐。
# 这个路径会发给 server，由 server 负责加载这个 checkpoint。
POLICY_PATH = Path("vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate_20260409_141037/checkpoints/000954/pretrained_model")

# SERVER_ADDRESS 要和 infer_act_libero_server.py 里的 HOST/PORT 一致。
SERVER_ADDRESS = "127.0.0.1:8080"

# POLICY_DEVICE 指的是“远端 server 加载 policy 的设备”，不是 client 本机设备。
# 如果 server 在 GPU 机器上运行，就写 "cuda"；如果 server 只有 CPU，就改成 "cpu"。
POLICY_DEVICE = "cuda"

# CLIENT_DEVICE 指的是 client 收到 action 后放在哪个设备上。
# LIBERO env.step() 最终使用 numpy action，所以这里保持 cpu 最简单。
CLIENT_DEVICE = "cpu"

# LIBERO 任务配置；EPISODE_INDEX=None 表示使用 LIBERO 默认初始状态顺序。
TASK_SUITE, TASK_ID, EPISODE_INDEX = "libero_goal", 8, None

# CONTROL_HZ 是真实控制频率。LIBERO/robosuite 这里按 20Hz 执行动作。
MAX_STEPS, H, W, CONTROL_HZ, SEED = 300, 256, 256, 20, 7

# 这里的 camera 名字要和训练时保存进 LeRobot dataset 的 image key 对齐。
CAMERAS = ("image", "image2")

# ACT 一次会输出一段 action chunk，而不是只输出一个动作。
# ACTIONS_PER_CHUNK=100 表示每次让 server 最多返回未来 100 个动作。
# 20Hz 控制时，100 个动作大约覆盖 5 秒；这样 server 慢一点也不容易断动作。
ACTIONS_PER_CHUNK = 100

# 当本地 action queue 剩余比例低于 0.5 时，就提前发下一帧 observation 给 server。
# 例如 100 个动作的 chunk，队列少于约 50 个动作时开始补货。
CHUNK_SIZE_THRESHOLD = 0.5

# 多个 action chunk 可能会覆盖同一个未来 timestep。
# weighted_average 是 LeRobot 自带聚合方式：旧动作占 0.3，新动作占 0.7。
# 也可以改成 latest_only / average / conservative。
AGGREGATE_FN = "weighted_average"

# 等待远端 action 的最长时间。超过 30 秒还没有动作，说明 server/client 通路可能有问题。
ACTION_TIMEOUT = 30.0


def raw_from_obs(env, env_pre, obs):
    # LIBERO observation 先走 LeRobot 的 env_pre，再整理成 async server 需要的 raw dict。
    batch = env_pre(add_envs_task(env, preprocess_observation(obs)))

    # state 被拆成 state_0, state_1 ...，这是 LeRobot raw robot observation 常用格式。
    state = batch["observation.state"][0].detach().cpu().float().numpy()
    raw = {f"state_{i}": float(v) for i, v in enumerate(state)}
    features = {f"state_{i}": float for i in range(len(state))}

    for cam in CAMERAS:
        # LeRobot async server 需要 HWC uint8 图片；env_pre 输出通常是 CHW float。
        img = batch[f"observation.images.{cam}"][0].detach().cpu()

        # CHW -> HWC，和 LeRobot dataset/raw observation 的图片格式保持一致。
        img = img.permute(1, 2, 0) if img.shape[0] in (1, 3) else img

        # 如果图片是 0~1 的 float，就转成 0~255；如果已经是 0~255，就保持原值。
        img = img * 255 if img.dtype.is_floating_point and float(img.max()) <= 1.5 else img

        raw[cam] = img.clamp(0, 255).byte().numpy()
        features[cam] = (H, W, 3)

    # features 告诉 server：raw dict 里的 state/image 应该如何还原成 LeRobot observation。
    raw["task"] = batch.get("task", "")
    return raw, hw_to_dataset_features(features, OBS_STR, use_video=False)


def make_client(policy_cfg, features):
    c = object.__new__(RobotClient)  # 不连接真实机器人，只复用 RobotClient 的 async 队列。

    # RobotClient.receive_actions() 会读取 c.config.client_device 和 c.config.aggregate_fn。
    # aggregate_fn 使用 LeRobot 自带实现，不在这个 demo 里手写。
    c.config = SimpleNamespace(client_device=CLIENT_DEVICE, aggregate_fn=get_aggregate_function(AGGREGATE_FN))

    # 建立到 server 的 gRPC 连接。
    c.channel = grpc.insecure_channel(SERVER_ADDRESS, grpc_channel_options(initial_backoff=f"{1 / CONTROL_HZ:.4f}s"))
    c.stub = services_pb2_grpc.AsyncInferenceStub(c.channel)

    # 下面这些字段是 RobotClient.receive_actions() 需要的最小状态。
    # 这样可以复用 LeRobot 自带的 action_queue 和 aggregate_fn，不手写 buffer 策略。
    c.shutdown_event = threading.Event()
    c.latest_action_lock = threading.Lock()
    c.action_queue_lock = threading.Lock()
    c.latest_action = -1
    c.action_chunk_size = ACTIONS_PER_CHUNK
    c.action_queue = Queue()
    c.action_queue_size = []
    c.start_barrier, c.must_go = threading.Barrier(1), threading.Event()

    # 第一次连接时，client 把 policy 类型、checkpoint 路径、feature schema 发给 server。
    # server 收到 RemotePolicyConfig 后才知道该加载哪个 policy，以及 observation 怎么解释。
    policy = RemotePolicyConfig(policy_cfg.type, str(POLICY_PATH), features, ACTIONS_PER_CHUNK, POLICY_DEVICE)
    c.stub.Ready(services_pb2.Empty())
    c.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy)))

    # 后台线程持续从 server 拉 action chunk，并写入 RobotClient.action_queue。
    threading.Thread(target=c.receive_actions, daemon=True).start()
    return c


def send_obs(c, raw, step, must_go=False):
    # must_go=True 用在第一帧或队列空了的时候，表示 server 不能跳过这帧 observation。
    # timestep 用来对齐：server 返回的 action chunk 会从这个 step 往后编号。
    obs = TimedObservation(time.time(), step, raw, must_go=must_go)
    c.stub.SendObservations(send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation, silent=True))


def pop_action(c):
    # 控制循环不能等太久；如果 ACTION_TIMEOUT 内没有 action，就说明 server/client 流水线断了。
    deadline = time.monotonic() + ACTION_TIMEOUT
    while time.monotonic() < deadline:
        try:
            with c.action_queue_lock:
                a = c.action_queue.get_nowait()

            # latest_action 告诉 LeRobot 聚合逻辑：已经执行过的旧 timestep 不要再放回队列。
            with c.latest_action_lock:
                c.latest_action = a.get_timestep()

            return a.get_action().detach().cpu()
        except Empty:
            # 这里短 sleep 是为了避免 while 空转占满 CPU。
            time.sleep(0.001)
    raise TimeoutError("No remote action received.")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)

    # 创建和本地推理一致的 LIBERO 环境；这里只把 policy 放在远端 server 跑。
    env_cfg = LiberoEnvConfig(
        task=TASK_SUITE,
        task_ids=[TASK_ID],
        obs_type="pixels_agent_pos",
        observation_height=H,
        observation_width=W,
        episode_length=MAX_STEPS,
    )
    env = make_env(env_cfg, n_envs=1)[TASK_SUITE][TASK_ID]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)

    # 固定 episode index 可以复现实验；不需要时保持 None。
    if EPISODE_INDEX is not None:
        env.envs[0].episode_index = env.envs[0].init_state_id = EPISODE_INDEX

    obs, _ = env.reset(seed=[SEED + (EPISODE_INDEX or 0)])
    raw, features = raw_from_obs(env, env_pre, obs)
    c = make_client(policy_cfg, features)
    success = False
    dt = 1 / CONTROL_HZ

    try:
        # 先发第 0 帧 observation，并等待第一段 action chunk 回来；这一步叫 warmup。
        send_obs(c, raw, 0, must_go=True)
        action, t0 = pop_action(c), time.perf_counter()

        for step in range(MAX_STEPS):
            # 每一步都对齐到真实控制频率。server 慢没关系，只要 action queue 里还有动作即可。
            target = t0 + step * dt
            if time.perf_counter() < target:
                time.sleep(target - time.perf_counter())

            # 第 0 步使用 warmup 拿到的第一帧动作；之后每步从 LeRobot action_queue 取一个新动作。
            if step:
                action = pop_action(c)

            # policy 输出仍然要经过 env_post，变成 LIBERO env.step() 能接受的 action。
            action = env_post({"action": action.unsqueeze(0)})["action"]
            obs, _, terminated, truncated, info = env.step(action.cpu().numpy() if torch.is_tensor(action) else action)
            success = success or bool(info.get("final_info", {}).get("is_success", False))

            # step 后立刻把新 observation 转成 raw 格式，准备发给 server 计算未来动作。
            raw, _ = raw_from_obs(env, env_pre, obs)

            # 队列低于阈值时就发新的 observation，让 server 提前计算下一段 action chunk。
            with c.action_queue_lock:
                q = c.action_queue.qsize()
            if q / ACTIONS_PER_CHUNK <= CHUNK_SIZE_THRESHOLD:
                send_obs(c, raw, step + 1, must_go=(q == 0))

            if bool(terminated[0]) or bool(truncated[0]):
                break

        print({"success": success, "steps": step + 1, "hz": (step + 1) / (time.perf_counter() - t0)})
    finally:
        c.shutdown_event.set()
        c.channel.close()
        env.close()


if __name__ == "__main__":
    main()
