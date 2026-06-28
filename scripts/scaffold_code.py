#!/usr/bin/env python3
"""Scaffold code/README.md for each lecture."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

LECTURES = {
    1: ("具身智能导论", "最小 reaching / 末端接近目标点", "simulation/minimal_reach.py"),
    2: ("ROS2 与 LeRobot 闭环", "ROS2 节点：状态读取、目标发布、episode 记录", "simulation/mock_ros2_loop.py"),
    3: ("本体与控制", "关节/eef 动作空间对比实验", "simulation/action_space_demo.py"),
    4: ("传感器与坐标系", "RGB-D 反投影与 camera_to_base", "simulation/rgbd_backproject.py"),
    5: ("仿真导论", "MuJoCo / Isaac Lab reach 入口", "simulation/README.md"),
    6: ("位姿估计", "YOLO + SAM + grasp pose 生成", "simulation/pose_grasp_pipeline.py"),
    7: ("操作技能", "规则版 pick-place 状态机", "simulation/pick_place_fsm.py"),
    8: ("端到端导论", "预训练 policy 闭环推理", "simulation/policy_rollout.py"),
    9: ("数据采集", "episode 录制与 LeRobot 格式转换", "simulation/record_episode.py"),
    10: ("模仿学习", "BC / ACT 训练脚本", "simulation/train_bc_act.py"),
    11: ("Diffusion Policy", "DP 推理与 receding horizon", "simulation/dp_inference.py"),
    12: ("VLA 理论", "构造 VLA sample", "simulation/build_vla_sample.py"),
    13: ("VLA 实操", "action normalization 与部署调试", "simulation/action_scale_experiment.py"),
    14: ("RL 基础", "reaching PPO/SAC 环境", "simulation/reaching_rl_env.py"),
    15: ("RL 运控 G1", "G1 locomotion 仿真入口说明", "simulation/g1_locomotion_notes.md"),
    16: ("RL 后训练", "recovery policy 仿真", "simulation/recovery_rl.py"),
    17: ("Robot Agent", "LangGraph skill 调用 mock", "simulation/agent_skills.py"),
    18: ("世界模型", "综合任务 + episode + 失败库", "simulation/full_pipeline_demo.py"),
}

TEMPLATE = """# Lecture {n:02d} — {title}

> 对应文稿：见 `docs/` 中第 {n} 讲

## 本讲 Demo

{demo}

## 目录结构

```
lecture{n:02d}/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 快速开始（仿真）

```bash
cd code/lecture{n:02d}
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python {entry}
```

## 真机路径

见 `hardware/README.md`（待补充）。

## 状态

- [ ] 仿真 Demo 可运行
- [ ] 真机 Demo 可运行
- [ ] 与文稿实验步骤一致
- [ ] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture {n:02d}] ...`
"""


def main() -> None:
    for n, (title, demo, entry) in LECTURES.items():
        d = CODE / f"lecture{n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "hardware").mkdir(exist_ok=True)
        (d / "simulation").mkdir(exist_ok=True)
        readme = d / "README.md"
        readme.write_text(TEMPLATE.format(n=n, title=title, demo=demo, entry=entry), encoding="utf-8")
        hw = d / "hardware" / "README.md"
        if not hw.exists():
            hw.write_text("# 真机 Demo\n\n待补充：SO101 / xLeRobot / G1 运行步骤。\n", encoding="utf-8")
        print(f"Scaffolded lecture{n:02d}")


if __name__ == "__main__":
    main()
