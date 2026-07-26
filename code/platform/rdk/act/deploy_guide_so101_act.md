# SO-101 ACT 部署到 RDK 板（S600 / S100）对着敲指南

> 把已训好的 SO-101 ACT（`so101_act_cuboid`，act/20000步/chunk100/6-DOF）量化编译成的 BPU 模型，部署到 **RDK S600（nash-p）或 RDK S100（nash-e）**，驱动 SO-101 机械臂做「黑方块入筐」pick-and-place。
> 走 `rdk_LeRobot_tools` s600 分支（官方 2026-06-12 流程）。

## 0. 全流程已完成的部分（在 x86 训练机上）

| 阶段 | 产物 | 位置 |
|---|---|---|
| 训练（历史） | SO-101 ACT checkpoint（front/side→已改 top/wrist） | `$DATASETS_ROOT/models/so101_act_cuboid_topwrist/pretrained_model` |
| 导出 ONNX+校准 | 2 个 ONNX + 校准数据 + build 脚本 | `$DATASETS_ROOT/models/bpu_export_act_so101_s600/` |
| 编译 S600（nash-p） | 2 个 .hbm + 归一化 npy | `…/bpu_export_act_so101_s600/bpu_output_s600/` |
| 编译 S100（nash-e） | 2 个 .hbm + 归一化 npy | `…/bpu_export_act_so101_s600/bpu_output_s100/` |

`bpu_output_sXXX/` 内含：
- `BPU_ACTPolicy_VisionEncoder.hbm`（ResNet18 吃图，输出 [1,512,15,20]）
- `BPU_ACTPolicy_TransformerLayers.hbm`（吃 states[1,6]+top_features+wrist_features，输出 Actions[1,100,6]，100步 chunk）
- `top_mean/std.npy`、`wrist_mean.npy/std.npy`、`action_mean/std.npy`(+unnormalize)、`new_actions.npy`

> 注：相机名已统一 **top / wrist**（语义与物理一致：数据集 put_black 已 relabel）。

## 1. 板端环境（S600 或 S100 板上，一次）

SSH 登录板子（S600 `sunrise@<BOARD_IP>` / S100 `sunrise@<BOARD_IP>`），装 lerobot + rdk_LeRobot_tools s600 + BPU 运行时：

```bash
git clone https://github.com/huggingface/lerobot.git && cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s600 && cd ..
pip install -e ".[feetech]"          # 板端 Python 环境
pip install hbm-runtime              # BPU 推理运行时（仅板端需要）
```

> S100 旧经验：板端用 `~/xbotics/experiments` 本地 mirror + `uv sync --extra rdk_s100 --no-editable`（NAS 慢）。S600 若也慢，照此。

## 2. 把 bpu_output 拷到板子

从开发机（能读 NAS）把对应板的 `bpu_output` 拷到板子（例如 `~/bpu_output`）：

```bash
# 在开发机上（S600 用 bpu_output_s600，S100 用 bpu_output_s100）
scp -r $DATASETS_ROOT/models/bpu_export_act_so101_s600/bpu_output_s600 sunrise@<BOARD_IP>:~/bpu_output
# 板端把目录名统一成 bpu_output（脚本默认找这个名）
```

板端确认：`ls ~/bpu_output` 应有 2 个 `.hbm` + `top_mean.npy`/`wrist_mean.npy`/`action_*.npy` + `new_actions.npy`。

## 3. 接线（SO-101 从手 + 双 USB 相机）

- **从手机械臂**（SO-101 follower，6 个飞特舵机）USB 接板子。`lerobot-find-port` 记录端口（如 `/dev/ttyACM0`）。
- **两个 USB 相机**：一个俯视（top）、一个腕部（wrist）。`lerobot-find-cameras` 看各自 `index_or_path`（如 0、1）。
- ⚠️ **top/wrist 物理 index 可能互换**（这台机器人上 top 和 wrist 的 USB 枚举顺序是反的）。先用 `index_or_path` 各拍一帧确认哪个是 top、哪个是 wrist，再填 `--camera-index`。**录视频前务必核对**，否则画面喂反。

## 4. 跑 BPU 推理（板端）

> ⚠️ **先看这条（重要）**：`bpu_control_robot.py` 这版在 `SO100FollowerConfig.cameras` 里**只配一个相机**（`--camera-name/--camera-index` 那个），是**单相机**脚本。而 cuboid 模型是**双相机**（top+wrist），直接跑会缺一个相机输入。两条出路：
> - **性能视频的主脊不受影响**：6.20ms/161.2inf/s 来自**独立 BPU 性能基准**（官方 §8，纯模型前向、不要机械臂/相机）——视频头部数字用这个基准，无需解决双相机。
> - **真机抓放镜头**需要把脚本改成双相机：policy 和 `build_policy_batch` 本就遍历 `camera_names`（双相机就绪），**只有 `main()` 的 `SO100FollowerConfig(cameras={...})` 配了单相机**。精确改法（在 `bpu_control_robot.py` 的 `main()` 里）：
>   1. `parse_args()` 加一个参数：
>      ```python
>      parser.add_argument("--camera-indices", type=str, default='{"top":0,"wrist":1}',
>          help='JSON: 每个相机的 USB index，键名=模型相机名')
>      ```
>   2. 把那段 `robot = SO100Follower(SO100FollowerConfig(... cameras={opt.camera_name: OpenCVCameraConfig(index_or_path=opt.camera_index, ...)}))` 换成按 `camera_names` 全配：
>      ```python
>      import json as _json
>      _idx = _json.loads(opt.camera_indices)   # {"top":0,"wrist":1}
>      cameras_cfg = {
>          name: OpenCVCameraConfig(index_or_path=_idx[name], width=opt.camera_width,
>              height=opt.camera_height, fps=opt.fps, warmup_s=10, fourcc="MJPG")
>          for name in camera_names
>      }
>      robot = SO100Follower(SO100FollowerConfig(
>          port=opt.robot_port, id="so100_follower",
>          max_relative_target=opt.max_relative_target, cameras=cameras_cfg))
>      ```
>   改完跑：`python bpu_control_robot.py --bpu-act-path ~/bpu_output --robot-port /dev/ttyACM0 --camera-indices '{"top":0,"wrist":1}' --fps 30`（top/wrist 的 0/1 按你机器实际 USB 枚举填，可能要互换）。
>   本目录已提供改好的双相机版 `bpu_control_robot_dual.py`（用前需在板上实测）。

单相机原版命令（参考；双相机需按上条改）：
```bash
cd ~/lerobot/rdk_LeRobot_tools
python bpu_control_robot.py \
  --bpu-act-path ~/bpu_output \
  --robot-port /dev/ttyACM0 \
  --camera-index 0 --camera-name top \
  --fps 30 --inference-time 60
```

`--camera-name` 必须与 `bpu_output/*_mean.npy` 前缀（top / wrist）一致；ACT 一次出 100 步 chunk，30fps 下每 3.33 秒 BPU 推理一次（6.20ms）。

## 5. 录性能视频

录视频要的镜头：
- 真机流畅抓放长镜头（开场 hook）。
- `hrut_somstatus`（S600）/ 板端 BPU 利用率——ACT 推理时 BPU 几乎空闲（99.8% 占空比≈0）。
- 性能基准数字（6.20ms / 161.2inf/s）来自官方 §8 基准（纯 BPU 前向，20 warmup+200 采样）；若要在板上复现，用 rdk_LeRobot_tools s600 分支里的 perf 脚本或 hbm-runtime profiling（见 §6）。

## 6. 排查 & 已知坑

- **机械臂不动**：`ls /dev/ttyACM*` 确认端口；`--robot-port` 对不对。
- **相机报错 / 画面反**：核对 `--camera-index` 对应的物理相机（top/wrist 在这台机互换）；`--camera-name` 与 `*_mean.npy` 前缀一致。
- **SO100 vs SO-101**：`bpu_control_robot.py` 默认连 **SO100Follower**；用 SO-101 要确认 lerobot robot 类型与机械臂一致（可能需改默认或传参）。
- **n_action_steps**：脚本从 `new_actions.npy` 自动推断，**勿传 `--n-action-steps 1`**（会改变 ACT 语义）。
- **机型**：S600=nash-p、S100=nash-e，`bpu_output_s600` 与 `bpu_output_s100` 的 .hbm **不可混用**；归一化 npy 两板相同。
- **性能基准复现**：官方 §8 给了 6.20ms/161.2inf/s（nash-p，S600）。板上跑 perf：在 rdk_LeRobot_tools s600 分支找 benchmark 脚本，或用 `hbm-runtime` 的模型 profiling（20 warmup+200 采样）。S100(nash-e) 数字会不同。
- **6.20ms 是纯模型前向**，非端到端回路延迟（相机/USB/Python/舵机开销另算）——视频口径以此为准。

## 7. 改回去 / 重训

- 重训 ACT：开发机 `lerobot-train --policy.type=act ...`（数据集 `so101/put_black_cuboid_into_basket`，top/wrist）。
- 重导出+编译：开发机跑 `code/lerobot/rdk/fetch_rdk_tools.sh` 拉的 rdk_LeRobot_tools，用 `bpu_export_config_so101_cuboid_demo.yaml`（act_path 指新 checkpoint，type=nash-p/nash-e），再 `docker run … bash build_all.sh`。
