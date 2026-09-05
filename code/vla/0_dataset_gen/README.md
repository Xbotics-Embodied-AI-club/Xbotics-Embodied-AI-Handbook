# 0_dataset_gen —— 生成与真机同口径的仿真数据集

这一讲产出三份仿真数据集，它们与真机数据集**可以直接合并成一份训练集**。
"可以合并"不是形容词：官方合并的第一步 `validate_all_metadata` 逐字比
`fps` / `robot_type` / 整个 `features` 字典，任一不等就抛 `ValueError`。

三个场景，指令逐字抄真机（cube20 除外，真机没有对应任务）：

| 场景 | 环境 | 语言指令 | 真机对应任务 |
| --- | --- | --- | --- |
| `cube40` | `SO101PickPlaceCube40-v1` | `Pick up a cube and place in the bin` | `pick_up_a_cube_and_place_in_the_bin` |
| `cube20` | `SO101PickPlaceCube20-v1` | `Pick up a small cube and place in the bin` | 无 |
| `cylinder40` | `SO101PickPlaceCylinder40-v1` | `Pick up a can and place in the bin` | `pick_up_a_can_and_place_in_the_bin` |

## 怎么跑

```sh
bash build_dataset.sh cube40 300 2 0        # 场景 · 要几集 · 分片数 · 可用 GPU 列表
bash build_dataset.sh cube20 300 2 0
bash build_dataset.sh cylinder40 300 2 0
```

跑完每个场景得到 `$DATASETS_ROOT/datasets/private/so101_sim_gen/<场景>/shards/shard*`，
收口（跨后端门 → 删不合格 → 合并 → 对口径）交给
`experiment_main_v1/scripts/EAI-exp-002/close_scene.sh`。

改产线之后自查用短链，不录制、直接拿规划在仿真里复跑：

```sh
RESULT_DIR=<本轮 attempt 目录> bash run_all.sh 40 cube40
```

## 动作从哪来：脚本化专家，不是回放

物体与料箱的位置是环境自己撒的点，写在强制初始状态里 ⇒ **抓取与放置是确定的几何问题**，
用运动学算，不用看图也不用学。专家离线规划出一整集的绝对关节角，落成 `.npy`，
由 `so101_dataset_player --teleop.actions_path=` 播出去。

轨迹的形状是这一讲最难的部分。目审逐条否掉过"八段直角"那一版：「抓东西为什么一定要
抬起来然后垂直下落，再松开夹爪？和真机轨迹完全不一样」「末端依然有很多无意义的旋转」。
现在只有两条一气呵成的曲线：

    取物 —— 一条落地曲线：从 home 斜着下来，下降与平移重叠着走完，最后 5cm 才转成垂直
    搬运 —— 一条弧线：抬起与平移融成一段，末端从物体上方 8cm 平滑滑到料箱上方

拐点不停车：结点之间段内匀速，再用 Hann 窗滑动平均把拐角抹圆，速度处处连续；
最后按**末端笛卡尔**速度上限重新计时（关节匀速 ≠ 末端匀速，手臂伸出去之后同样的关节
角速度对应大得多的末端线速度 —— 那正是目审读到的「中间突然飞快一下」）。

## 为什么是"生成 + 录制"而不是"转换"

已交付的那四份仿真数据集出自手工转换路径，视频编码是 mpeg4，而真机实测是 h264 ——
`features` 里的 `video.codec` 不等，合不了。正确的解法不是再写一个转换器把编码抹平，
而是**把仿真注册成一台 lerobot 机器人，用 `lerobot-record` 原样录一遍**：
编码、字段、目录布局全部由官方命令给出，不需要事后修补任何一项。

全链只用两条官方命令：`lerobot-record` 与 `lerobot-edit-dataset`。

## 四步链路

```
prepare_episodes.py → check_plans.py → compact_prep.py → lerobot-record（分片并行）
   撒场景 + 规划        录之前先判成败      只留过门的、重编号        录制
```

**① `prepare_episodes.py`** 一集一个种子复位环境、等物体与料箱落定、拍下整份状态，
再让 `expert.plan` 算出这一集的动作。解不出来就如实跳过并记下理由，不重试、不放宽判据。

**② `check_plans.py`** **录之前**就把每集在仿真里复跑一遍判成败，一集一个进程并行。
不合格的集一旦进了数据集就只能事后 `delete_episodes`，而官方 merge 之后集号指针会
大面积悬空（实测 243/344），删必须赶在合并之前 —— 链条很脆。这一步用的是与录制
**同一个后端、同一份强制初始状态**，判定与录制结果一致，不是近似。

**③ `compact_prep.py`** 只留判成功的，重新编号 0..N−1。原始种子写进 meta，来源不丢。

**④ `lerobot-record`** 一集一个进程。一个进程只能录一集，因为它按墙钟停表，
而一次调用只认一个窗口值、每集的窗口又得按各自的规划长度给。

## 收货判据

| 判据 | 谁来判 | 判什么 |
| --- | --- | --- |
| 观感 | `expert.quality_faults`（规划时就判） | 末端加速度 ≤120cm/s²、最长停顿、爬行兜底、搬运腕部行程 ≤60°、路径迂曲 ≤2 倍 |
| 任务完成 | `check_plans.py`（录之前） | 抬起来过 ∧ 落进料箱开口内 |
| 保真 | `check_recorded.py` | 录到的动作前 N 帧逐位等于规划，其后每帧等于规划末帧 |
| 任务完成 ∧ 不依赖机器 | `verify_cross_backend.py` | 每集在 CPU 与 GPU 两个后端都"抬起来过 ∧ 落进箱口" |
| 与真机同口径 | `experiment_main_v1/scripts/EAI-exp-002/verify_same_as_real.py` | 38 项，从声明一路核到视频字节 |

### 帧数不是判据，动作才是

`lerobot-record` 按墙钟停表，循环跑不满 30 Hz 时尾部会少帧，**少多少与机器快慢有关**。
窗口按 `(规划帧数 + 热身补偿)/30` 给，热身补偿只抵消循环第一帧建管线的固定开销 ——
**不是"留余量"**：留余量会多出一串静止帧，把末尾静止段撑长（实测余量 90 帧时撑到
68 帧、占集长 15.9%，而真机是 16 帧 / 3.9%）。

⇒ 每集的**总帧数不是跨机器可复现的量**，**动作序列是**。判据只用后者。

### 规划集号 ≠ 数据集内集号

`--resume` 按录制顺序发号（0,1,2,…），而分片是隔片取集（0,2,4,… / 1,3,5,…），
两者不相等。这本账记在 `logs/shard<N>.order`，每录成一集追加一行规划集号，
行号就是该集在这一分片里的 `episode_index`。

**不查这本账，跨后端门会拿别的集的动作去复跑、`delete_episodes` 会删错集，
而这两件事都不报错。**

## 文件

| 文件 | 干什么 |
| --- | --- |
| `recipe.py` | 三个场景的标定表与全部声明常量（编码、帧数容差、夹持口袋、张开量、俯仰候选） |
| `expert.py` | 脚本化专家：落地曲线、搬运弧线、Hann 抹圆、末端限速重计时、四道观感门 |
| `grasp_ik.py` | 抓取几何：抓取点、锁俯仰的雅可比、对齐到物体的面、跨度判据、腕滚等价解择优 |
| `servo.py` | 正运动学、基座系换算、阻尼最小二乘微分 IK |
| `prepare_episodes.py` | 撒场景 + 规划，逐集落强制初始状态与 `.npy` |
| `check_plans.py` / `check_success.py` | 成功门（前者读规划、后者读已录数据集，判据是同一个函数） |
| `compact_prep.py` | 过门的集重新编号，作为录制输入 |
| `build_dataset.sh` | 编排器：规划 → 成功门 → 重编号 → 分片录制 → 保真核对 |
| `check_recorded.py` | 保真核对（参照物是 `prep/plans/*.npy`） |
| `verify_cross_backend.py` | 跨后端一致门 |
| `render_plans.py` | 把规划复跑并渲成目审视频（上视图 + 腕部并排） |
| `check_aim.py` · `measure_grip_jitter.py` · `measure_descent_clearance.py` · `calibrate_pocket.py` · `diagnose_roll_candidates.py` · `trace_preload.py` | 诊断，不在产线链路上 |

## 硬约束

- **装环境只能 `uv sync`**，不用 `pip` / `uv pip`。
- **不设 `PYTHONPATH`**。脚本按 `.envrc` 声明的关系从 `HF_HOME` 推出数据集根，不写死路径。
- **业务与物理常量写在 py 文件里，不走环境变量** —— 走环境变量时"这批数据是按哪套参数产的"
  只存在于当时那条命令里，同一份代码在两台机器上会产出不同的数据，且两边都不报错。
- **不给 lerobot 打补丁、不改上游代码**。仿真侧只做两件事：把自己注册成一台机器人
  （`--robot.type=so101_sim`）、把回放注册成一个遥操器（`--teleop.type=so101_dataset_player`）。
