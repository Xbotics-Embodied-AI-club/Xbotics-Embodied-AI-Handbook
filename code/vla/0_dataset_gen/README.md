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
bash regen_dataset.sh cube40 4 0        # 场景 · 分片数 · 可用 GPU 列表
bash regen_dataset.sh cube20 4 0
bash regen_dataset.sh cylinder40 4 0
```

跑完每个场景得到 `$DATASETS_ROOT/datasets/private/so101_sim_regen/<场景>/merged/`。

分片数按可用 GPU 给。限制在**渲染**而不是核数：物理跑在 CPU（3.2ms/步），
但每帧仍要在 GPU 上渲两路 640×480（9ms/帧）。分片抢同一块卡时循环会变慢，
靠录制窗口的余量兜（`recipe.TAIL_PAD_FRAMES`）。

## 为什么是"重录"而不是"转换"

已交付的那四份仿真数据集出自手工转换路径，视频编码是 mpeg4，而真机实测是 h264 ——
`features` 里的 `video.codec` 不等，合不了。正确的解法不是再写一个转换器把编码抹平，
而是**把仿真注册成一台 lerobot 机器人，用 `lerobot-record` 原样录一遍**：
编码、字段、目录布局全部由官方命令给出，不需要事后修补任何一项。

全链只用两条官方命令：`lerobot-record` 与 `lerobot-edit-dataset`。

## 三步链路

```
recover_scene.py   → lerobot-record（每集一个进程，分片并行） → lerobot-edit-dataset merge
   反解初始场景            录制                                    合片
```

**① `recover_scene.py`** 把每一集的物体与料箱摆回原处。生成源数据集的 h5
（里面存着 `env_states`）已经不在了，所以位置只能从数据本身反解：物体被夹住的那一刻，
夹持口袋就在物体中心，把那一帧的关节角喂给正运动学就得到它。
口袋不是"两指中点"——两指中点随张开角移动、不是刚体点，实测均值误差 13.3mm，
而标定出来的口袋是 1.0mm。标定值逐场景声明在 `recipe.py`。

**② `lerobot-record`** 一集一个进程。一个进程只能录一集，因为它按墙钟停表，
而一次调用只认一个窗口值、每集的窗口又得按各自的源集长度给。
遥操器 `so101_dataset_player` 把源集的动作逐帧播出去，机器人 `so101_sim` 在仿真里执行。

**③ `lerobot-edit-dataset --operation.type merge`** 把分片合成该场景的成品。

## 收货判据

| 判据 | 谁来判 | 判什么 |
| --- | --- | --- |
| 保真 | `check_regen.py` | 录到的动作前 N 帧逐位等于源集，其后每帧等于源集末帧 |
| 任务完成 ∧ 不依赖机器 | `verify_cross_backend.py` | 每集在 CPU 与 GPU 两个后端都"抬起来过 ∧ 落进箱口" |
| 与真机同口径 | `experiment_main_v1/scripts/EAI-exp-002/verify_same_as_real.py` | 30 项，从声明一路核到视频字节 |

不合格的集号由前两者打印成**合并后**的编号，交给
`lerobot-edit-dataset --operation.type delete_episodes` 官方删除。

### 帧数不是判据，动作才是

`lerobot-record` 按墙钟停表，循环跑不满 30 Hz 时尾部会少帧，**少多少与机器快慢有关**。
所以窗口按 `(源帧数 + TAIL_PAD_FRAMES)/30` 给，让回放一定走完整条源轨迹；
多出来的帧是遥操器保持源集最后一帧动作的结果，而源集末尾本来就静止。

⇒ 每集的**总帧数不是跨机器可复现的量**，**动作序列是**。判据只用后者。

### 源集号 ≠ 数据集内集号

`--resume` 按录制顺序发号（0,1,2,…），而分片是隔片取集（0,2,4,… / 1,3,5,…），
两者不相等。这本账记在 `logs/shard<N>.order`，每录成一集追加一行源集号，
行号就是该集在这一分片里的 `episode_index`。

**不查这本账，跨后端门会拿别的集的动作去复跑、`delete_episodes` 会删错集，
而这两件事都不报错。**

## 文件

| 文件 | 干什么 |
| --- | --- |
| `recipe.py` | 三个场景的标定表与全部声明常量（编码、窗口余量、抓取口袋、张开角阶梯） |
| `recover_scene.py` | 从源数据集反解每一集的初始场景 |
| `servo.py` | 正运动学、基座系换算、阻尼最小二乘微分 IK。`recover_scene.py` 靠它反解 |
| `grasp_ik.py` | 纯运动学抓取专家：抓取点、锁俯仰的雅可比、预抓取与抓取位姿的 IK。**重录链路不调用它** —— 轨迹来自源数据集的回放；它是从零生成新轨迹时用的那一支 |
| `regen_dataset.sh` | 编排器：反解 → 分片录制 → 保真核对 → 合片 |
| `check_regen.py` | 保真核对 |
| `verify_cross_backend.py` | 跨后端一致门 |

## 硬约束

- **装环境只能 `uv sync`**，不用 `pip` / `uv pip`。
- **不设 `PYTHONPATH`**。脚本按 `.envrc` 声明的关系从 `HF_HOME` 推出数据集根，不写死路径。
- **业务与物理常量写在 py 文件里，不走环境变量** —— 走环境变量时"这批数据是按哪套参数产的"
  只存在于当时那条命令里，同一份代码在两台机器上会产出不同的数据，且两边都不报错。
- **不给 lerobot 打补丁、不改上游代码**。仿真侧只做两件事：把自己注册成一台机器人
  （`--robot.type=so101_sim`）、把回放注册成一个遥操器（`--teleop.type=so101_dataset_player`）。
