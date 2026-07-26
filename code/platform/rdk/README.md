# rdk/ — 地瓜 RDK 开发板上的 LeRobot 部署

把在 x86 训练机上训好的 LeRobot 策略，量化编译后部署到地瓜 RDK 开发板（S100 / S600）的 BPU 上运行。

## 目录

- `act/` —— **SO-101 ACT** 策略的完整上板链路：导出 → 编译 → 部署 → 验证 → 录像。详见 [`act/README.md`](act/README.md)。
- `fetch_rdk_tools.sh` —— 拉取官方 `rdk_LeRobot_tools`（BPU 导出/推理工具，按需 clone，不入库）。

> 后续其他策略（如 SmolVLA / Pi0 等）的上板流程会作为 `act/` 的平级子目录加入。

## 约定

- 训练 / 导出 / 编译在 **x86 训练机**（带 GPU，CUDA 锁定 12.8）上完成；板端只做 BPU 推理。
- 数据集与模型产物路径用 `$DATASETS_ROOT/...` 占位，按自己机器的挂载点设置。
- 板型对应 BPU 平台：**S600 = `nash-p`**、**S100 = `nash-e`**，两者的 `.hbm` 指令集不通用。
