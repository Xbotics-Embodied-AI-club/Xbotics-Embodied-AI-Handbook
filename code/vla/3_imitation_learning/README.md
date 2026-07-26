# 组3 · 模仿学习

用组2 采来的数据训第一个自己的策略。`3_1_act/` 是 ACT 的完整闭环：

- `train_act_libero.py` / `train_act_cuboid.sh`：LIBERO 子集与 SO-101 真机数据两条训练线；
- `infer_act_libero.py`：本地加载 checkpoint 闭环推理 + 录像；
- `infer_act_libero_server/client.py`：LeRobot async inference 的 server-client 部署形态
  ——策略在 GPU 机器上服务，仿真/真机侧按控制频率消费 action chunk。

训练输出落 `outputs/`（不入库）；结果样例见 `result/3_1_act/`。
