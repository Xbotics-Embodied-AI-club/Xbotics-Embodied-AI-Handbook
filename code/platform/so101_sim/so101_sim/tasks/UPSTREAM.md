# tasks/ 与 robots/so101_base/ 的来源

本目录的 `place.py`、`base_random_env.py`、`black_overlay.png`，以及
`../robots/so101_base/`（`so101.py` + `so101.urdf`/`.srdf` + `meshes/`），
最初取自 **squint** 项目：

- 仓库：https://github.com/aalmuzairee/squint
- 提交：`2a1f6e894e2a4cfd97a18dbe43b1570dde65fa42`（2026-03-04）
- 论文：*Fast Visual Reinforcement Learning for Sim-to-Real Manipulation*（arXiv:2602.21203）
- 许可：MIT（见 `LICENSE`，© 2026 Abdulaziz Almuzairee）

## 分叉声明

引入时逐字节未改动。**自 2026-08-23 起本目录转为本项目自维护，与上游分叉、不再跟随上游更新。**
分叉的实际原因是行为已经不同：我们的三个分发环境通过 mixin 覆盖了任务的相机、
物体尺寸、生成范围与机器人速度包线，`Place` 事实上已被改写。

未收录上游的部分（我们不需要）：`train_squint.py`（SAC 训练器）、
`stack.py` / `lift.py` / `reach.py`（未使用的任务）、`robot/so100.py`（SO100 本体）、
`deploy.py` / `deploy_utils/` / `examples/`（上游真机侧）。
