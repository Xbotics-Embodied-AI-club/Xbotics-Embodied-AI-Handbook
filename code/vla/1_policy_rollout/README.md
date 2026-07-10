# 组1 · 端到端策略闭环

端到端方法的第一课不是训练，而是**跑通一个闭环**：observation 进、action 出、
env.step、再来一遍。本组两个模块：

1. `1_1_libero_env/`：先认识 LIBERO——任务套件、observation/action 接口、初始状态机制。
2. `1_2_pi0_libero_rollout/`：把预训练 π0 当黑盒策略，在 LIBERO 上跑通第一个闭环并录像；
   附一个「白噪声换图」的对照实验，验证策略确实在利用视觉。

π0 内部结构（VLM + flow matching）此处不展开——先把「policy 就是一个函数」的直觉建立起来。
