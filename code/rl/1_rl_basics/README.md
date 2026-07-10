# 组1 · RL 基础：同一个任务，三个算法

用「REINFORCE → A2C → PPO」一条演进链讲清强化学习为什么长成今天这样。两条任务线
共用同一套算法阶梯：

1. **行走**（`1_1_g1_walk_rl`）：宇树 G1 按速度指令走路——入门任务，三个算法都
   能学出点样子，但稳定性差距肉眼可见。
2. **动作跟随**（`1_3_g1_motion_tracking`）：G1 逐帧贴住一段武术参考动作——更难
   的任务把差距放大：v1/v2 明显跟不住，只有完整 PPO 能干净完成。

同一版本号的文件在两条线上一一对应（`train_v1_reinforce.py` / `train_v2_a2c.py` /
`train_v3_ppo.py` / `rollout.py`），建议横向对照着读：算法不变，任务变难。

## 数据交接

```
1_2_video_to_g1_reference  ──产出──▶  data/g1_reference_motions/*.npz  ──消费──▶  1_3_g1_motion_tracking
（视频→GVHMR→GMR→npz 工具链）          （682帧 G1 参考动作）                 （动作跟随训练）
```

`1_1` 不需要数据（奖励由环境在线给出）；`1_3` 的参考动作已生成好放在
`data/g1_reference_motions/`，想换自己的视频再走一遍 `1_2` 工具链即可。

## 运行

```bash
cd experiments
uv sync --extra gpu_x86
uv run python rl/1_rl_basics/1_1_g1_walk_rl/train_v1_reinforce.py   # 或打开同名 .ipynb
```

训练产物（checkpoint）落 `DATASETS_ROOT/models/trained/` 下；两条线各自的
`rollout.py` 录制对照视频，样例结果已在 `result/` 里。
