# 仿真遥操采集（无真机兜底线）

没有真机械臂时，用 SO-101 **仿真器**顶替 `2_3` 的真机主从臂：人用键盘直接驱动仿真里的从臂，
逐帧记录相机图 / 关节位置 / 动作，转成与实物线**完全同格式**的 `LeRobotDataset`
（128px 图 + 6 维 state + 6 维 action）。下游 ACT / VLA 训练读进来，分不出数据来自真机还是仿真。

和 `rl/3_offpolicy` 的「RL 专家自动采数据」生产线同一套骨架——录制器、h5 结构、官方格式转换三步
全复用，只把**动作来源**从策略网络换成键盘。

## 文件

- `teleop_record.py` — 主脚本：键盘监听 → 仿真采集 → h5 → 官方 `convert_to_lerobot`
- `teleop_record.ipynb` — 同一份代码的中文分节讲解版
- `README.md` — 本文件

## 按键

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| `a` / `d` | 底座左转 / 右转 | `r` / `f` | 腕部上仰 / 下俯 |
| `w` / `s` | 大臂抬起 / 落下 | `t` / `g` | 腕部左滚 / 右滚 |
| `e` / `q` | 小臂前伸 / 收回 | `c` / `v` | 夹爪张开 / 闭合 |
| `回车` | 存下当前这一集 | `Esc` | 结束整轮采集 |

## 运行

在 `code/` 下（环境走统一 `pyproject.toml` 的 `gpu_x86` extra，`import so101_sim` 直接可用）：

```bash
uv run python vla/2_data_collection/2_5_sim_teleop_record/teleop_record.py
```

采集方式由脚本末尾的 `RUN` 决定：

- `RUN = "teleop"` —— 接键盘实采，弹出相机窗口，采满 `N_EPISODES` 集后自动转数据集。
- `RUN = "selfcheck"` —— 无人值守：用内置的确定性动作序列跑通「采集 → 转换 → 加载」整条链，
  再用 `LeRobotDataset` 加载并打印 episodes / frames / features 验证格式。没有键盘 / 显示器也能跑。

想换场景（3 个 SO-101 场景之一）改脚本里的 `TASK`；想调遥操快慢改 `STEP`。

## 输出

数据集落 `DATASETS_ROOT/so101_sim/_teleop/<TASK>/dataset`（自检落 `_teleop_selfcheck/`），不入代码仓。
目录即标准 `LeRobotDataset`（`data/` parquet + `videos/` mp4 + `meta/` info/stats/episodes），
逐字段与 `rl/3_offpolicy` 的 `datagen/` 产出、与实物线一致：`action` f32×6、`observation.state` f32×6、
`observation.images.top` 与 `observation.images.wrist` 各一路 128×128 视频（三个场景都是双相机，
对齐真机），`fps=20`。
