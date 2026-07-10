# RDK S100 / S600 上 ACT 板端验证与性能实测 — 对着敲指南

> **目的**：把已量化的 SO-101 ACT（`so101_act_cuboid`，6-DOF / 双相机 top+wrist / 100 步 chunk）在 **RDK S100（nash-e）和 RDK S600（nash-p）** 上跑起来，**不接触实物机械臂**，用数据集回放验证「BPU 推理是否正常、与数据集是否一致」，并采集**性能测评视频脚本**要的那些数（每模块 BPU 前向 ms、完整推理 ms、占空比/空闲率、`hrut_somstatus`）。
>
> 配套：部署/真机指南 `deploy_guide_so101_act.md`。
>
> 验证脚本：`code/lerobot/rdk/act/validate_act_dataset_replay.py`（推理走官方 `rdk_LeRobot_tools` 的 `BPUACTPolicy`，本指南只在外面套数据集读取 + 比对）。

## 0. 三个必须先知道的坑

1. **板端 Python 与 lerobot 版本绑死**（这是 S100/S600 唯一的关键差异）：
   - **S100**：系统 Python **3.10**；公网 `hbm-runtime` 只有 **cp310** wheel；而 lerobot **0.5.x 要 Python ≥3.12**（源码 `motors_bus.py` 用 PEP 695 `type` 语法）。两者无法共存 → **S100 只能用 lerobot 0.4.4**（py3.10 上能装的最高版，仍有新 robots API）。
   - **S600**：系统 Python **3.12**，且镜像**自带 cp312 的 `hbm-runtime`** → 直接用文档要求的 **lerobot 0.5.1**（PyPI 最高 0.5.x；README 写的 v0.5.2 是 git 版）。
2. **数据集视频是 AV1 编码，板子解不了**（`Missing Sequence Header / no AV1 support`）→ 回放用的帧必须**在开发机用 ffmpeg 解成 PNG** 再拷到板上。
3. **top / wrist 相机在这个模型上是交叉的**：模型的 `top` 输入要喂数据集的 `wrist` 画面、`wrist` 输入喂数据集 `top`（实测交叉后 pred-vs-真值 MAE 16.8°→10°）。真机接相机时同理要核对（对应 `deploy_guide §3/§6` 的 top/wrist 互换告警）。脚本里已写死 `MODEL_CAM_FROM_DATASET={"top":"wrist","wrist":"top"}`。

## 1. 板子访问

| 板 | 架构 | SSH | 系统 Python | lerobot |
|---|---|---|---|---|
| RDK S100 | nash-e | `sunrise@<BOARD_IP>`（密码 `sunrise`） | 3.10.12 | **0.4.4** |
| RDK S600 | nash-p | `sunrise@<BOARD_IP>`（密码 `sunrise`） | 3.12.3 | **0.5.1** |

两块板都已挂 NAS `$DATASETS_ROOT`。模型量化产物在 NAS：
`$DATASETS_ROOT/models/bpu_export_act_so101_s600/{bpu_output_s100,bpu_output_s600}/`（各含两个 `.hbm` + 归一化 `.npy`）。

## 2. 开发机：从数据集抽回放帧（AV1 → PNG）

在能读 NAS、有 ffmpeg(libdav1d) 的开发机上跑。数据集 = `…/hf-hub/lerobot/so101/put_black_cuboid_into_basket`（v3.0，episode 0 = 全局帧 0..288）。抽 episode 0 的 5 帧（top+wrist）：

```bash
DS=$DATASETS_ROOT/hf-hub/lerobot/so101/put_black_cuboid_into_basket
OUT=code/.result/rdk/replay_frames; mkdir -p "$OUT"   # 仓库内任务绑定 results（勿用 /tmp）
for IDX in 0 47 94 141 188; do
  for CAM in top wrist; do
    ffmpeg -hide_banner -loglevel error -y \
      -i "$DS/videos/observation.images.$CAM/chunk-000/file-000.mp4" \
      -vf "select=eq(n\,$IDX)" -vframes 1 "$OUT/${CAM}_${IDX}.png"
  done
done
ls "$OUT"   # 应有 top_0.png … wrist_188.png 共 10 张
```

> 帧号 `0 47 94 141 188` 必须与脚本里的 `REPLAY_FRAMES` 一致。

## 3. 板端环境（每块板做一次）

> 用隔离 venv，不污染系统 Python。两块板的工作目录都叫 `~/act_sXXX_bench/`。

### 3a. S100（lerobot 0.4.4）

```bash
ssh sunrise@<BOARD_IP>
mkdir -p ~/act_s100_bench && cd ~/act_s100_bench
python3 -m venv --system-site-packages .venv      # --system-site-packages 取系统 /usr/hobot 库
. .venv/bin/activate
pip install hbm_runtime                             # cp310
pip install "lerobot[feetech]==0.4.4"              # py3.10 上能装的最高版；torchcodec 被 aarch64 marker 自动避开
python -c "from hbm_runtime import HB_HBMRuntime; from lerobot.robots.so_follower import SO100Follower; print('OK')"
```

### 3b. S600（lerobot 0.5.1，文档原版）

```bash
ssh sunrise@<BOARD_IP>
mkdir -p ~/act_s600_bench && cd ~/act_s600_bench
python3 -m venv --system-site-packages .venv      # 继承系统预装的 cp312 hbm-runtime
. .venv/bin/activate
pip install "lerobot[feetech]==0.5.1"             # py3.12 原生，无需 --ignore-requires-python
python -c "from hbm_runtime import HB_HBMRuntime; from lerobot.robots.so_follower import SO100Follower; print('OK')"
```

## 4. 拷资产到板（每块板）

从开发机拷：对应板的 `bpu_output`、官方工具 s600 分支、回放帧、验证脚本。**模型目录在板上统一命名 `bpu_output`**（脚本相对自身目录找 `bpu_output/`）。

```bash
# —— S100 ——（开发机上跑）
B=sunrise@<BOARD_IP>; D='~/act_s100_bench'
rsync -az $DATASETS_ROOT/models/bpu_export_act_so101_s600/bpu_output_s100/ $B:$D/bpu_output/
rsync -az --exclude='.git' code/lerobot/rdk/rdk_LeRobot_tools/ $B:$D/rdk_tools_s600/
rsync -az code/.result/rdk/replay_frames/ $B:$D/replay_frames/
rsync -az code/lerobot/rdk/act/validate_act_dataset_replay.py $B:$D/

# —— S600 ——（开发机上跑，模型换成 bpu_output_s600）
B=sunrise@<BOARD_IP>; D='~/act_s600_bench'
rsync -az $DATASETS_ROOT/models/bpu_export_act_so101_s600/bpu_output_s600/ $B:$D/bpu_output/
rsync -az --exclude='.git' code/lerobot/rdk/rdk_LeRobot_tools/ $B:$D/rdk_tools_s600/
rsync -az code/.result/rdk/replay_frames/ $B:$D/replay_frames/
rsync -az code/lerobot/rdk/act/validate_act_dataset_replay.py $B:$D/
```

板上每个工作目录最终长这样：`bpu_output/`（该板 .hbm + npy）、`rdk_tools_s600/`、`replay_frames/`、`validate_act_dataset_replay.py`、`.venv/`。

## 5. 跑验证（每块板）

```bash
# 板上：
cd ~/act_s100_bench   # S600 则 ~/act_s600_bench
./.venv/bin/python validate_act_dataset_replay.py
hrut_somstatus | grep -iE 'bpu|temperature'     # 视频要的 BPU 利用率/温度
```

脚本会打印两块：**① 数据集回放一致性**（BPU 预测 vs 数据集真值动作，1 步 / 10 步 / 整段 100 步 MAE）；**② 纯 BPU 前向基准**（20 warmup + 200 采样，分别计 VisionEncoder / TransformerLayers / 完整 ACT，并由 100 步 chunk 推占空比）。

期望（S600 实测，节选）：

```
=== 数据集回放一致性 ===
  >> 平均: 1步 8.40°   10步 9.33°   整段 14.55°
=== 纯 BPU 前向基准（20 warmup + 200 采样）===
  VisionEncoder (单相机)            2.466 ms   405.51 inf/s
  TransformerLayers                3.698 ms   270.41 inf/s
  完整 ACT (2 相机视觉 + 1 transformer) 8.656 ms   115.53 inf/s
     -> 100 步 chunk @ 30fps 覆盖 3.33s；占空比 0.260%  空闲 99.74%
```

## 6. 结果与对照（喂视频脚本）

两块板实测（本指南方法：板上 `hbm_runtime.run()`，20 warmup + 200 采样）：

| 模块 | S100 (nash-e) | S600 (nash-p) | 官方帖 §8 (S600) |
|---|---|---|---|
| VisionEncoder（单相机） | 4.73 ms / 211 inf/s | **2.47 ms / 405 inf/s** | 3.92 ms / 255 inf/s |
| TransformerLayers | 7.86 ms / 127 inf/s | **3.70 ms / 270 inf/s** | 2.29 ms / 436 inf/s |
| 完整 ACT（1 视觉+1 tf，对齐官方口径） | ~12.6 ms | **6.17 ms** | **6.20 ms** |
| 完整 ACT（2 视觉+1 tf，**本模型真实双相机**） | **17.5 ms / 57 inf/s** | **8.66 ms / 116 inf/s** | — |
| 100 步 chunk @30fps 占空比 | 0.52% | 0.26% | ≈0.19% |
| **BPU 空闲率** | **99.47%** | **99.74%** | 99.8% |

**结论**：
- ACT 在两板上都**正常推理**，输出有效 100 步 chunk；**与数据集一致**（即时动作 1 步 MAE 8.4°，量程 ~200°）——量化没破坏模型。整段 100 步开环 MAE 偏大是 ACT 闭环重规划的正常开环发散，不是错。
- 视频脚本要的参数**全部可获取并上屏**：每模块 ms、完整 ms、占空比、空闲率、`hrut_somstatus`。
- **口径校正（重要，录视频别说错）**：官方招牌 **6.20ms 是单相机**（1 视觉+1 tf）。S600 实测单相机口径 6.17ms ≈ 官方，吻合；但**这个 cuboid 模型是双相机**，真实每决策 BPU = **S600 8.66ms / S100 17.5ms**。无论哪种口径，**100 步 chunk → 每 3.33 秒才推一次 → BPU 99%+ 时间空闲**，视频「算力远没到瓶颈」的核心结论两板都成立。
- S600 比 S100 约 **2×** 快（完整双相机 8.66 vs 17.5ms）。

## 7. 相机语义校准 —— 改进模型本身（S100 实测，2026-06-19）

S100 上实测两个相机 + 比对数据集 + 数据集回放，结论分两层：

**物理相机命名是对的**（两个不同型号，udev 已正确区分）：

| udev | 设备 | 型号 | 实拍视角 | 对应数据集 |
|---|---|---|---|---|
| `/dev/top_camera` | `/dev/video0` | Realtek `0bda:3035` | **俯视工作区** | dataset `top`（俯视） |
| `/dev/wrist_camera` | `/dev/video2` | Sonix `05a3:9230`（USB2.0_CAM1） | **夹爪视角**（画面下方两根夹爪指） | dataset `wrist`（夹爪） |

**但模型包里的 top/wrist 归一化标反了**（训练/relabel 时交叉）。数据集回放实测：

```
原模型 + 交叉喂法 :    1步MAE 8.41°   (基准/正确)
原模型 + 正常喂法 :    1步MAE 16.08°  (错)
```

**修法：改模型本身，不在部署时记交叉。** 经实验证明——transformer 的 `top_features`/`wrist_features` 两个视觉通道**对称**，交叉纯粹是归一化 npy 的标签问题，所以**只要把模型包里 `top↔wrist` 的归一化 npy 对调**即可，逐帧 MAE 与交叉喂法**完全一致（8.41°）**：

```
换npy模型(top/wrist 归一化对调) + 正常喂法 : 1步MAE 8.41°  (= 基准，完全等价)
```

**修正版模型（已生成，拷贝原模型 + 仅对调 4 个 npy，`.hbm`/动作 npy 不动）**：

```
$DATASETS_ROOT/models/bpu_export_act_so101_s600/bpu_output_s100_fixed   # nash-e
$DATASETS_ROOT/models/bpu_export_act_so101_s600/bpu_output_s600_fixed   # nash-p
# 内容 = 原 bpu_output_sXXX，但 top_mean/std.npy 与 wrist_mean/std.npy 互换
```

用修正版模型后，**部署按正常物理映射**（直观）：

```python
cameras = {
    "top":   OpenCVCameraConfig(index_or_path="/dev/video0", ...),  # 物理俯视 Realtek
    "wrist": OpenCVCameraConfig(index_or_path="/dev/video2", ...),  # 物理夹爪 Sonix
}
```

> 已实测两相机都可读（各 480×640×3）；用正常映射把两路实时帧喂修正版模型，BPU 正常出 `[100,6]` chunk（通路通过；精度需真实方块场景才有意义）。

## 8.（可选，后续）真机抓放推理

本指南只做数据集回放 + 相机校准、不动实物闭环。真机闭环（机械臂动）走 `rdk_tools_s600/bpu_control_robot.py`，另需：
1. **标定**：新格式标定文件（lerobot 0.5.x 录的）拷成板上 `{HF_LEROBOT_HOME}/calibration/robots/so_follower/so100_follower.json`（脚本里 `id="so100_follower"`）。
2. **双相机改法**：stock 脚本 `main()` 只配单相机 → 按 `deploy_guide §4` 配齐双相机，相机映射用上面第 7 节的交叉表。

## 8. 已知坑速查

- `hbm_runtime not found` / 装不上：S100 必须 py3.10（cp310 wheel）；S600 用系统 py3.12 自带的。
- `SyntaxError motors_bus.py`：在 py3.10 上装了 lerobot 0.5.x，降到 0.4.4。
- 板上读 mp4 报 AV1 / 黑帧：板子解不了 AV1，帧要在开发机抽好拷过去。
- 回放 MAE 异常大（shoulder_lift 几十度）：多半是 top/wrist 喂反了，核对第 0 节交叉映射。
- `.hbm` 不能混用：`bpu_output_s100`(nash-e) 与 `bpu_output_s600`(nash-p) 不可互换；归一化 npy 两板相同。
