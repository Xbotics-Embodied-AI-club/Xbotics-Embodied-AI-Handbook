# RL 基础：同一个任务，三个算法

用「REINFORCE → A2C → PPO」一条演进链讲清强化学习为什么长成今天这样。两条任务线
共用同一套算法阶梯：

1. **行走**（`1_1_g1_walk_rl`）：宇树 G1 按速度指令走路——入门任务，三个算法都
   能学出点样子，但稳定性差距肉眼可见。
2. **动作跟随**（`1_3_g1_motion_tracking`）：G1 逐帧贴住一段武术参考动作——更难
   的任务把差距放大：v1/v2 明显跟不住，只有完整 PPO 能干净完成。

同一版本号的文件在两条线上一一对应（`train_v1_reinforce.py` / `train_v2_a2c.py` /
`train_v3_ppo.py` / `rollout.py`），建议横向对照着读：算法不变，任务变难。

## 两条线到底有多同构

"同一套代码搬去更难的任务"这句话得站得住，否则横向对照没有意义。逐文件量一遍
（去掉 docstring 之后的纯代码行差异）：

| 文件 | 两条线的纯代码差异 | 差在哪 |
|---|---|---|
| `model.py` | **0 行** | 逐字相同 |
| `train_v1_reinforce.py` | 25 行 | 环境类名、权重目录、多一个 `motion_file` 参数 |
| `train_v2_a2c.py` | 25 行 | 同上 |
| `train_v3_ppo.py` | 28 行 | 同上 |
| `rollout.py` | 22 行 | 同上，外加参考动作加载与 ghost 开关 |
| `env.py` | 25 行 | **这是唯一被设计成不同的东西**：mjlab 的速度任务 vs 动作跟踪任务 |

剩下的每一行都是任务本身要求的差异。这个数字是维护出来的，不是自然形成的：
`1_3/train_v3_ppo.py` 曾经内联抄过一份 `ActorCritic` 和 `compute_gae`（同目录
`model.py` 里就有），单这一项就让它和行走线差了 248 行，而 `rollout.py` 还从
**训练脚本**里 import 网络类、把三个观测维度写死成 `160 / 286 / 29`。

## 数据交接

```
1_0_video_to_g1_reference  ──产出──▶  data/g1_reference_motions/*.npz  ──消费──▶  1_3_g1_motion_tracking
（视频→GVHMR→GMR→npz 工具链）          （682帧 G1 参考动作）                 （动作跟随训练）
```

`1_1` 不需要数据（奖励由环境在线给出）；`1_3` 的参考动作已生成好放在
`data/g1_reference_motions/`，想换自己的视频再走一遍 `1_0` 工具链即可。

`1_0` 编号在 `1_1` 之前，是因为它只产数据、不产结论 —— 它是 `1_3` 的前置，
不是算法阶梯上的一级。讲义把它单独放在第 4 节，也是同一个道理。

## 运行

```bash
cd code
uv sync --extra gpu_x86
uv run python rl/1_rl_basics/1_1_g1_walk_rl/train_v1_reinforce.py   # 或打开同名 .ipynb
```

训练产物（checkpoint）落 `DATASETS_ROOT/models/trained/` 下；两条线各自的
`rollout.py` 录制对照视频，样例结果已在 `result/` 里。
