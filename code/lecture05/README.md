# SO-101 simulation —— 仿真环境与任务搭建

## 项目概述

本目录提供第 5 讲配套的仿真教学代码，对应讲义 `docs/part2-vision-manipulation/05-simulation.md`（「仿真环境与任务搭建——从真实任务到可复盘闭环」）。其目标不仅是给出一个“能跑起来的仿真”，而是把**任务契约定义、控制闭环实现、抓取证据采集、单变量实验与 Sim2Real 退化分析**这一完整链路尽量透明地呈现出来。

本讲用两类任务贯穿，分属两个平台：

| 平台 | 任务 | 算力要求 | 本目录状态 |
|---|---|---|---|
| MuJoCo | SO-101 reach / pick-place / Sim2Real / 单变量实验 | 纯 CPU 即可 | **含全部自写代码，本机实测可复现** |
| Isaac Lab | Franka Reach（§4）、G1 locomotion（§5） | 需 NVIDIA GPU | 无自写代码，仅给换环境后的官方命令 |

仿真实验的核心方法论，是**每次只改一个变量，并用连续指标而非“看起来能动”来判定结果**：

```text
定义契约 → 跑通闭环 → 记录连续指标 → 单变量扰动 → 对照预测复盘
```

## 环境准备

MuJoCo 的 PyPI wheel 自包含 C 引擎，无需系统级 C 库，venv 即可满足：

```bash
cd code/lecture05/simulation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # mujoco==3.10.0 + numpy
```

如果默认镜像下载较慢或失败，可以临时使用清华 PyPI 镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

本目录已 clone 好机器人资产。从零复现时执行：

```bash
git clone --filter=blob:none --sparse --no-checkout \
  https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set robotstudio_so101
git -C mujoco_menagerie checkout 71f066ad0be9cd271f7ed58c030243ef157af9f4
```

验证安装：

```bash
.venv/bin/python -c "import mujoco; print(mujoco.__version__)"   # 3.10.0
```

### 软件与资产基线

对齐讲义 §1.3，核验日期为讲义标注的 **2026-07-25**：

| 组件 | 基线 | 用途 |
|---|---|---|
| Python（MuJoCo） | ≥ 3.10 | 本目录所有 MuJoCo 实验 |
| MuJoCo | 3.10.0 | SO-101 核心实验 |
| SO-101 资产 | MuJoCo Menagerie commit `71f066ad…` | 官方 MJCF（`robotstudio_so101`） |
| Python（Isaac Lab） | 3.11 | Isaac Lab 2.3.2 环境 |
| Isaac Lab | 2.3.2 | 环境接口与并行评测 |
| Isaac Sim | 5.1.0 | Isaac Lab 2.3.2 的对应基线 |

### 仓库不包含的资源

下列内容体积过大或本身是独立 git 仓库，**已在 `.gitignore` 中排除，clone 后不存在**，需按对应命令自行获取：

| 资源 | 体积 | 获取方式 |
|---|---|---|
| `mujoco_menagerie/` | ~28 MB | 本节上方的 sparse clone 三行命令，**必须 checkout 到指定 commit** |
| `.venv/` | — | `python3 -m venv .venv && pip install -r requirements.txt` |
| `results/*.mp4` | — | `render_video.py` 重新渲染 |
| `__pycache__/` | — | 运行时自动生成 |

`results/` 下的 CSV（约 1.5 MB）**随仓库分发**，作为本机基线证据，便于和你自己跑出来的结果逐列对照。

## 目录说明

本 README 位于 `code/lecture05/`，全部代码与数据在同级的 `simulation/` 下；真机路径见 [`hardware/README.md`](hardware/README.md)。

```text
code/lecture05/
├── README.md          # 本文件
├── hardware/          # 有硬件版路径（待补充）
└── simulation/        # 无硬件可运行路径，下表文件均在此目录内
```

| 文件 | 作用 |
|---|---|
| `reach.py` | 基础必做：SO-101 position reach 最小闭环，同时作为可导入的核心模块（`load_model` / `ReachController`） |
| `sim2real.py` | 进阶四：无硬件 Sim2Real，注入动作延迟与观测噪声，跑 6 档 × 5 seed 实验矩阵 |
| `pick_place.py` | 进阶一：SO-101 pick-place 阶段调度，核心 `run_pick_place()` 供实验二/三复用 |
| `render_video.py` | 离屏渲染 pick-place 全程为 mp4（逐帧 PPM + 系统 ffmpeg 合成） |
| `metrics.py` | 单变量实验共享工具，提供带增强指标的 `run_reach_metrics()` |
| `experiments/target_position.py` | 实验一：扫描目标位置与工作空间边界 |
| `experiments/control_period.py` | 实验四：扫描外层 IK 控制周期 |
| `experiments/mass.py` | 实验二：方块质量对比 |
| `experiments/friction.py` | 实验三：方块—桌面摩擦对比 |
| `mujoco_menagerie/` | Menagerie 稀疏 clone，只含 `robotstudio_so101` |
| `results/` | 全部 CSV 输出与录像 |
| `requirements.txt` | pip 依赖清单 |

场景文件（`mujoco_menagerie/robotstudio_so101/`）全部 `include` 同一份 `so101.xml`，彼此只差一个受控变量：

| 文件 | 相对基线的差异 |
|---|---|
| `so101.xml` | 机器人本体、关节、碰撞体、位置执行器；`timestep="0.005"`，`integrator="implicitfast"` |
| `scene.xml` | reach 场景，只有地面，无方块 |
| `scene_box.xml` | pick-place 基线场景，方块 `size="0.02 0.02 0.03"`、`friction="1 .03 .003"`、无 `mass` 属性（默认密度 1000 → 约 0.096 kg），含 `pickup` 关键帧 |
| `scene_box_mass.xml` | 与 `scene_box.xml` **仅差一行**：方块 `mass="0.20"` |
| `scene_box_lowfriction.xml` | 与 `scene_box.xml` **仅差一行**：sliding friction `1 → 0.1` |

## 推荐实操流程

建议按照下面 6 步完成一次教学或实验演示：

1. **准备环境与资产**
   - 建 venv 并安装 `mujoco==3.10.0`；
   - 确认 Menagerie commit 与讲义一致；
   - 先跑版本验证命令，再跑实验。

2. **跑通基础 reach**
   - 运行 `reach.py`，确认成功且末端误差在毫米级；
   - 回看任务契约：目标是什么、动作是什么、两个时钟分别是多少。

3. **跑通 pick-place**
   - 运行 `pick_place.py`，确认 6 个阶段全部 OK；
   - 重点看**四类抓取证据**，而不是“看起来夹住了”。

4. **做单变量实验**
   - 每个实验只改一个变量，其余保持基线；
   - 运行前**先写下预测**，再和 CSV 结果对照；
   - 结果与预测不符时，先怀疑变量选错了界面，再怀疑代码。

5. **做无硬件 Sim2Real**
   - 运行 `sim2real.py`，观察延迟、噪声的单独效应与组合效应；
   - 区分「带噪观测判定的成功」与「仿真真值误差」。

6. **按统一格式记录**
   - 每个实验按讲义 §7.2 的记录表填写；
   - 保留 `results/` 下 CSV 作为原始证据。

## 最小可运行自检

如果只想确认 MuJoCo 装好、资产路径正确、IK 方向没写反，先跑基础 reach。它不依赖 GPU、显示环境和任何外部数据：

```text
加载 scene.xml → 运行时注入目标 site → 阻尼最小二乘 IK → 双时钟推进 → 连续 8 周期误差 < 5 mm 判成功
```

```bash
cd code/lecture05/simulation
.venv/bin/python reach.py
```

本机基线输出（与讲义 §3.3 一致）：

```text
reach success=True, final_error=0.0005 m
```

`results/reach.csv` 记录逐物理步的 `time_s, ee_x, ee_y, ee_z, error_m`（本机 105 数据行，约 0.525 s 达标，末端误差 0.00048 m）。

## 基础必做：SO-101 position reach（§3.3）

`reach.py` 把讲义三个片段组装成一个最小 reach 闭环：加载官方 `scene.xml` → 运行时加入目标 site → 末端位置误差经**阻尼最小二乘**变成关节位置目标 → 5 ms 物理步与 20 ms 外层 IK 两个时钟推进 → 误差连续 8 个控制周期低于 5 mm 判成功。

关键常量：

| 项目 | 取值 |
|---|---|
| 目标位置 | `[0.30, 0.10, 0.20]`（世界坐标，米） |
| 末端 site | `gripperframe` |
| 手臂关节 | `shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_flex` / `wrist_roll` |
| 物理步长 | 5 ms（`PHYSICS_DT = 0.005`） |
| 控制周期 | 20 ms（`CONTROL_DT = 0.020`，`DECIMATION = 4`） |
| 成功判定 | 误差 < 5 mm 且连续保持 8 个控制周期 |
| 超时 | 5.0 s |
| IK 阻尼 | `0.02`，单步关节增量限幅 `0.04` rad |

**契约回看（§2.2）**：目标 `gripperframe` 到达世界坐标 `[0.30, 0.10, 0.20]`；动作是 5 个手臂关节的**位置目标**（`data.ctrl`），不是力矩；物理步长 5 ms 与外层 IK 周期 20 ms 是两个不同的时钟。

## 进阶一：SO-101 pick-place（§3.5）

`pick_place.py` 从 `scene_box.xml` 的 `pickup` 关键帧（抓取就绪）开始，完成阶段链：

```text
闭合 → 确认抓取 → 抬升 → 搬运 → 下放 → 释放 → 稳定性检查
```

实现要点（对应 §3.5 的接触证据与 §2.3 的单变量原则）：

- **闭合**：夹爪目标每控制步减小 0.01（下限 `-0.174`），检测到双侧法向力超过 3.0 N 即「确认夹住」；之后用**持续夹紧目标 0.0**，方块阻挡 gripper 关节、位置执行器持续施力，搬运中活动指自动跟进（否则方块会缓慢滑脱）。
- **搬运 / 抬升 / 下放**：固定手腕姿态（`wrist_flex` / `wrist_roll`），只用前 3 关节 `shoulder_pan` / `shoulder_lift` / `elbow_flex` 做位置 IK（`arm_ik3`）。这是关键：若用 5 关节位置 IK，冗余自由度会让夹爪朝向在搬运中漂移，方块滑脱。
- **四类抓取证据**：双侧接触点、双侧法向力、抬升后方块高度增量、方块相对夹爪漂移。

判定阈值：

| 项目 | 取值 |
|---|---|
| 夹持力阈值 | 双侧均 > 3.0 N |
| 抬升高度 | 目标 0.12 m，判定需高度增量 > 0.05 m |
| 放置位置 | `xy = [0.32, 0.05]`，`z = 0.03`，落点误差 < 0.02 m |
| 放置高度 | `abs(box_z - 0.03) < 0.015` m |
| 稳定判定 | 方块线速度 < 0.05 m/s |

```bash
cd code/lecture05/simulation
.venv/bin/python pick_place.py
```

本机基线输出：

```text
=== pick-place 阶段结果 ===
  grasp/lift/carry/lower/release/stable : OK
=== 四类抓取证据（抓取确认时记录）===
  双侧接触:  固定指 1 点 / 活动指 2 点
  双侧法向力: 固定指 4.41 N / 活动指 25.72 N
  抬升后方块高度增量: 0.124 m
  搬运后方块相对夹爪漂移: 0.0050 m
=== 放置检查 ===
  落点误差: 0.0115 m (阈值 0.02)
  方块最终高度: 0.0194 m (目标 0.03)
  方块线速度: 0.0190 m/s (阈值 0.05)
  综合判定: PLACED
```

`results/pick_place_trace.csv` 记录逐控制步的末端 / 方块位姿、双侧力、漂移、抬升量（本机 112 数据行，6 个阶段依次为 grasp 34 / lift 30 / carry 32 / lower 11 / release 2 / stable 3 行）。

### 实时观看与录像

物理计算与渲染解耦，无头模式不依赖任何显示环境：

| 方式 | 命令 | 依赖 |
|---|---|---|
| 无头（默认） | `.venv/bin/python pick_place.py` | 无 |
| 实时 viewer | `.venv/bin/python pick_place.py --viewer` | 可用的窗口化 OpenGL 上下文 |
| 离屏录像 | `.venv/bin/python render_video.py` | 离屏 OpenGL 上下文 |

- `--viewer`：启动 MuJoCo 被动 viewer（`mujoco.viewer.launch_passive`）实时渲染抓取过程；默认按**真实时间节流**（约 6 s 物理时间），可交互拖视角。
- `render_video.py`：用 `mujoco.Renderer` 逐帧离屏渲染，写 PPM 后由系统 ffmpeg 合成 `results/pick_place.mp4`。默认 640×480 @ 50 fps，可用 `--width` / `--height` / `--fps` / `--output` 覆盖，`--keep-frames` 保留中间 PPM 帧。无 GPU 时可用软件渲染：`MUJOCO_GL=egl .venv/bin/python render_video.py`。

## 进阶四：无硬件 Sim2Real（§6.3）

`sim2real.py` 在 reach 闭环中接入 `DelayedNoisyChannel`：**动作延迟**（关节目标延迟若干个「控制更新」）+ **观测噪声**（末端位置加高斯噪声）。成功判定用带噪观测的误差，同时记录仿真真值误差。实验矩阵为 6 档 × 5 个 seed（`seeds = [0, 1, 2, 3, 4]`）。

```bash
cd code/lecture05/simulation
.venv/bin/python sim2real.py
```

本机基线汇总：

| 档位 | 延迟 / 噪声 | 成功率（带噪判定） | 平均最终真值误差 |
|---|---|---:|---:|
| `baseline` | 0 / 0 | 5/5 | 0.00048 m |
| `delay1` | 1 个控制更新 / 0 | 5/5 | 0.00124 m |
| `delay2` | 2 个控制更新 / 0 | 5/5 | 0.00343 m |
| `noise1mm` | 0 / 1 mm | 5/5 | 0.00122 m |
| `noise2mm` | 0 / 2 mm | 5/5 | 0.00214 m |
| `delay2_noise2mm` | 2 个控制更新 / 2 mm | **0/5** | 0.00646 m |

观察：单独的延迟或噪声闭环仍能成功，但真值误差随之增大；**组合效应**（2 个更新延迟 + 2 mm 噪声）使带噪判定无法连续保持 8 个周期，真值误差也超过 5 mm 阈值，二者同时失守。

## 四个单变量实验（§3.7）

每个实验只改一个变量，输出 CSV 到 `results/`。下表为本机基线观察：

| 实验 | 命令 | 关键观察 |
|---|---|---|
| 一：目标位置 | `.venv/bin/python experiments/target_position.py` | x ≥ 0.50 超时失败；`min_singular` 从 0.077 骤降到 0.0004 以下，失败机制是**接近奇异**而非触发限幅（`clipped_updates = 0`） |
| 四：控制周期 | `.venv/bin/python experiments/control_period.py` | 10→160 ms，达标时间 0.395→4.965 s 单调增大；路径长度在 20 ms 时最短（0.163 m），此后增至 0.417 m，即「命令保持更久、路径变抖」 |
| 二：方块质量 | `.venv/bin/python experiments/mass.py` | 0.096→0.20 kg，双侧力略增（4.41/25.72 N → 5.06/27.61 N）、漂移 5.0→4.0 mm，均 PLACED —— 证据**否定**「质量增加必滑脱」 |
| 三：摩擦 | `.venv/bin/python experiments/friction.py` | 方块—桌面 friction 1.0→0.1，落点误差与滑动距离完全不变（0.0115 m / 0.0095 m） |

扫描取值：

| 实验 | 扫描变量与取值 | 重复次数 |
|---|---|---|
| 一：目标位置 | x ∈ {0.15…0.60}（10 点）、y ∈ {-0.25…0.30}（7 点）、z ∈ {0.05…0.40}（8 点），共 25 行 | 各 1 次 |
| 四：控制周期 | 10 / 20 / 40 / 80 / 160 ms（decimation 2 / 4 / 8 / 16 / 32） | 各 1 次 |
| 二：方块质量 | `baseline`（≈0.096 kg）vs `mass_0.20` | 各 3 次 |
| 三：摩擦 | `mu_1.0` vs `mu_0.1` | 各 3 次 |

> 实验三是「先分清研究哪个界面」（§3.7）的实例：改方块—桌面摩擦对「垂直放下」几乎无影响，因为释放时水平速度≈0，桌面摩擦不是主导变量；要观察摩擦的显著作用，应改**夹爪—方块**界面的 `collision_gripper` 摩擦（`so101.xml` 中 `friction="1 5e-3 5e-4"`），而非只改方块。

## 输出数据格式

所有实验证据都落在 `results/` 下：

| 文件 | 产生脚本 | 列名 |
|---|---|---|
| `reach.csv` | `reach.py` | `time_s, ee_x, ee_y, ee_z, error_m` |
| `pick_place_trace.csv` | `pick_place.py` | `time_s, phase, ee_x, ee_y, ee_z, box_x, box_y, box_z, gripper_q, fixed_force_N, moving_force_N, fixed_n, moving_n, box_vel, drift_m, box_lift_m` |
| `sim2real_summary.csv` | `sim2real.py` | `label, delay_updates, noise_std_mm, seed, success_measured, reached_time_s, final_true_error_m, min_true_error_m` |
| `sim2real/{label}_s{seed}.csv` | `sim2real.py` | `time_s, ee_x, ee_y, ee_z, error_true_m, error_measured_m`（6 档 × 5 seed = 30 个文件） |
| `exp1_target_position.csv` | `experiments/target_position.py` | `axis, value, success, reached_time_s, final_error_m, min_singular, clipped_updates, path_length_m` |
| `exp2_mass.csv` | `experiments/mass.py` | `case, rep, placed, grasp_ff_N, grasp_mf_N, lift_delta_m, carry_drift_m, place_err_m` |
| `exp3_friction.csv` | `experiments/friction.py` | `case, rep, placed, place_err_m, slide_dist_m` |
| `exp4_control_period.csv` | `experiments/control_period.py` | `control_period_ms, decimation, success, reached_time_s, final_error_m, path_length_m, min_singular` |
| `pick_place.mp4` | `render_video.py` | 视频，默认 640×480 @ 50 fps |

`metrics.py` 的 `run_reach_metrics(target, control_dt=0.020, tolerance=0.005, timeout_s=5.0)` 是实验一 / 四的共享入口，返回 `success`、`reached_time`、`final_error`、`min_singular`、`clipped_updates`、`n_control`、`path_length` 七个字段，其中 `min_singular` 与 `clipped_updates` 就是区分「奇异」和「限幅」两种失败机制的判据。

## 实验设计建议

| 项目 | 建议 |
|---|---|
| 变量数 | 每次只改一个变量，其余全部保持基线 |
| 界面选择 | 先分清要研究哪个接触界面，再决定改哪个参数 |
| 预测在先 | 运行前写下预测，再和结果对照，否则实验退化为“看现象” |
| 判定标准 | 用连续指标（误差、力、漂移、奇异值），不用“看起来成功” |
| 重复次数 | 涉及接触的实验至少重复 3 次，接触求解本身有随机性 |
| 三个量分开 | `command`（任务条件）≠ `policy action`（策略输出）≠ `actuator input`（执行器量） |

如果实验结果与预测不符，先检查改的变量是否落在主导那个物理机制的界面上，再怀疑代码——实验三就是典型例子。

## 常用命令

### 1. 基础 reach

```bash
cd code/lecture05/simulation
.venv/bin/python reach.py
```

### 2. pick-place

```bash
cd code/lecture05/simulation
.venv/bin/python pick_place.py            # 无头
.venv/bin/python pick_place.py --viewer   # 实时 viewer
```

### 3. 离屏录像

```bash
cd code/lecture05/simulation
.venv/bin/python render_video.py \
  --output results/pick_place.mp4 \
  --width 640 \
  --height 480 \
  --fps 50
```

### 4. 无硬件 Sim2Real

```bash
cd code/lecture05/simulation
.venv/bin/python sim2real.py
```

### 5. 实验一：目标位置

```bash
cd code/lecture05/simulation
.venv/bin/python experiments/target_position.py
```

### 6. 实验二 / 三：质量与摩擦

```bash
cd code/lecture05/simulation
.venv/bin/python experiments/mass.py
.venv/bin/python experiments/friction.py
```

### 7. 实验四：控制周期

```bash
cd code/lecture05/simulation
.venv/bin/python experiments/control_period.py
```

## Isaac Lab 部分（§4 Franka + §5 G1）—— 换环境复现

本部分无自写代码，均为官方脚本 / 环境入口，只需在满足下列条件的机器上照命令执行。

### 环境要求

- NVIDIA GPU（Isaac Sim 需 GPU；本讲实验至少数 GB 显存）
- Python **3.11**
- **Isaac Lab 2.3.2** + **Isaac Sim 5.1.0**（固定版本，见讲义 §1.3）

安装按固定版本的官方页面进行（不复制易过期的长步骤）：

- <https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html>

安装完成后在 **Isaac Lab 仓库根目录**运行下列命令（`./isaaclab.sh` 为其自带入口脚本）。

### §4.4 Franka Reach 三种运行方式

**方式一：检查环境、观测和动作形状（不依赖 RL 训练）**

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Reach-Franka-v0 --num_envs 32
```

记录：observation 字典键与张量形状、action 维数与范围、环境数量、episode 何时 reset、目标如何随机化。「随机 agent 能跑」只证明环境可创建 / step / reset，**不等于 reach 已学会**。

**方式二：差分 IK（不依赖 RL）**

```bash
./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py \
  --robot franka_panda --num_envs 128
```

用于对比 MuJoCo 的 Jacobian IK 与 Isaac Lab 控制器接口；它不是 `Isaac-Reach-Franka-v0` 的 RL reward 示例。

**方式三：官方 RL 训练 / 播放**

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Franka-v0 --headless

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-Franka-Play-v0 --num_envs 32 \
  --checkpoint /ABSOLUTE/PATH/TO/model.pt
```

> 官方脚本提供 checkpoint 获取机制，但**并非每个任务都保证存在可下载预训练权重**（讲义 §4.4）。无 checkpoint 时，方式一、二足以完成环境阅读实验。

**环境阅读对照点（§4.3）**：在配置中标出 Scene / Articulation / Observation / Action / Termination 的代码位置；区分 `terminated`（MDP 内终态）与 `truncated`（时间上限等 MDP 外截断）；确认 reward 高不等于任务成功；核对 success predicate 是教材约定而非官方默认行为。

### §5 G1 工况比较

官方注册名（Isaac Lab 2.3.2）：

```text
Isaac-Velocity-Flat-G1-v0   Isaac-Velocity-Flat-G1-Play-v0
Isaac-Velocity-Rough-G1-v0  Isaac-Velocity-Rough-G1-Play-v0
```

训练 / 播放：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 32 \
  --checkpoint /ABSOLUTE/PATH/TO/model.pt
```

**Standing / Walking / Turning 不是三个独立任务**（§5.3），而是同一 velocity-tracking 环境的不同 command 工况：`c_t = (v_x*, v_y*, ω_z*)` 分别为 `(0,0,0)`、非零线速度、非零 yaw 角速度。

记录指标（§5.4–5.6）：

| 工况 | 记录指标 |
|---|---|
| Standing | base roll/pitch 的 RMSE、base 线 / 角速度、两足接触序列、动作变化率、固定时域存活时间 |
| Walking | 速度跟踪 RMSE（`RMSE_v`）、base 高度 / 姿态漂移、足底接触交替、足滑、episode 是否摔倒终止 |
| Turning | base 轨迹弧线、实际 yaw rate 跟踪、内外侧足交叉 / 自碰撞、髋 / 膝 / 踝是否接近限位 |

### 统一实验记录表（§7.2）

每个实验按同一格式记录：实验 ID、任务、平台与版本、资产与提交、随机种子、初始状态、观测、动作、physics timestep、control period、decimation、改动变量、运行前预测、成功判定、连续指标、重复次数、结果统计、最早异常、下一步。

## 常见错误与排查建议

### 1. 把接近奇异误判为限幅

- 表现：目标稍微放远就再也够不到，误差停在几厘米；
- 原因：雅可比接近奇异，阻尼最小二乘给不出有效方向；
- 建议：看 `min_singular`（从 0.077 骤降到 0.0004）而不是猜，同时确认 `clipped_updates = 0` 排除限幅。

### 2. 搬运中方块缓慢滑脱

- 表现：抓取确认成功，但 carry 阶段漂移持续增大；
- 原因：闭合后没有维持夹紧目标，或用了 5 关节 IK 导致夹爪朝向漂移；
- 建议：保持夹紧目标 0.0，搬运只用前 3 关节做位置 IK。

### 3. 改了变量却什么都没变

- 表现：摩擦从 1.0 改到 0.1，落点误差与滑动距离一模一样；
- 原因：改的界面不是主导该现象的界面；
- 建议：先判断力是从哪个接触面传递的，再决定改哪个参数。

### 4. 用「看起来成功」代替判定

- 表现：viewer 里动作很像抓取，但 CSV 中双侧法向力接近 0；
- 原因：没有采集接触证据，只看渲染；
- 建议：坚持四类抓取证据 + 连续指标，渲染只用于定性观察。

### 5. 渲染相关报错

- 表现：`--viewer` 或 `render_video.py` 报 OpenGL / GLFW 错误；
- 原因：无显示环境或缺少离屏上下文；
- 建议：默认用无头模式；录像时改用 `MUJOCO_GL=egl`；确认系统已安装 ffmpeg。

### 6. 资产版本不一致

- 表现：本机结果与讲义基线数值对不上；
- 原因：Menagerie clone 到了其他 commit，模型参数已变；
- 建议：`git -C mujoco_menagerie log -1 --format=%H` 应为 `71f066ad0be9cd271f7ed58c030243ef157af9f4`。

## 关键文件与讲义对照

| 讲义节 | 本目录落点 |
|---|---|
| §3.3 基础 reach | `reach.py` |
| §3.5 pick-place | `pick_place.py`、`render_video.py` |
| §3.7 实验一 / 四 | `experiments/target_position.py`、`experiments/control_period.py`、`metrics.py` |
| §3.7 实验二 / 三 | `experiments/mass.py`、`experiments/friction.py` |
| §6.3 无硬件 Sim2Real | `sim2real.py` |
| §4 / §5 Isaac Lab | 本 README「Isaac Lab 部分」（换环境） |

## 一句话理解本目录

- `reach.py`：把「末端到达一个点」写成最小可验证闭环；
- `pick_place.py`：把「抓起来并放好」拆成阶段链，并用接触证据证明它真的抓住了；
- `sim2real.py`：把延迟和噪声加进来，看闭环在什么组合下失守；
- `metrics.py` + `experiments/`：把单变量实验的指标口径统一，让四组实验可比；
- `render_video.py`：把过程录下来，用于讲解而非用于判定；
- `mujoco_menagerie/`：锁定资产版本，保证结果可复现。
