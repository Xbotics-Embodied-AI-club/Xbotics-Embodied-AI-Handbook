# 遥操作与数据录制

- `tele.sh` — SO-101 主从臂遥操（双相机 + rerun 可视化），先用它练手感
- `record.sh` — 遥操 + `lerobot-record` 录数据集（任务：把 cuboid 放进篮子；50 条 episode，数据落 `$HF_LEROBOT_HOME/so101/cuboid`）
- `rerun-S100-说明.md` — RDK S100 上 rerun 窗口起不来的修复说明

前置：`../2_2_so101_setup/` 的 udev 绑定已生效；统一 uv 环境（`uv sync --extra nogpu_x86`）。
