# act/ — SO-101 ACT 在 RDK 板上的部署

把训练好的 SO-101 ACT 策略（6-DOF / 双相机 top+wrist / 100 步 chunk）量化编译成 BPU 模型（`.hbm`），部署到 RDK S600（`nash-p`）/ S100（`nash-e`），做「黑方块入筐」抓放。

推理走官方 `rdk_LeRobot_tools` 的 `BPUACTPolicy`（用 `../fetch_rdk_tools.sh` 拉取）。

## 链路（按顺序看）

| 阶段 | 文档 | 配套脚本 / 配置 |
|---|---|---|
| 1. 导出 + 量化编译 | [`convert-act-to-bpu-walkthrough.md`](convert-act-to-bpu-walkthrough.md) | `bpu_export_config_so101_cuboid_demo.yaml` |
| 2. 部署到板 / 真机抓放 | [`deploy_guide_so101_act.md`](deploy_guide_so101_act.md) | `bpu_control_robot_dual.py` |
| 3. 板端验证 + 性能实测 | [`board_act_validate_benchmark_guide.md`](board_act_validate_benchmark_guide.md) | `validate_act_dataset_replay.py` |

## 关键约定

- **导出只认 config YAML**：`export_bpu_actpolicy.py` 会丢弃命令行覆盖参数，所有设置改 YAML。
- **换板子只改一行**：YAML 里 `type: nash-p`（S600）/ `nash-e`（S100）决定 `.hbm` 指令集，两板不可混用；归一化 `.npy` 两板相同。
- **相机 top/wrist**：本模型两路视觉通道对称，归一化标签曾交叉；修正版把模型包里 `top↔wrist` 归一化 `.npy` 对调后即可按正常物理映射部署（详见验证文档 §7）。
- 路径用 `$DATASETS_ROOT/...` 占位，按自己机器设置。
