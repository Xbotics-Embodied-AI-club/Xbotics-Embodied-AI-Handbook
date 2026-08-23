# deploy/ —— 把训好的策略接到真机

推理和执行分家：部署板（Jetson Orin Nano / RDK S100）只做「读舵机 + 开相机 → 发观测 →
收动作 → 下发舵机」，模型推理放 x86 GPU 机上。

```
部署板                              x86 GPU 机
─────────────────                  ─────────────
读舵机 / 开相机   ──观测(gRPC)──>   加载 checkpoint
下发动作到舵机    <──动作块────     推理
```

这么拆的原因是内存：Orin Nano 只有 7.4G 统一内存，π0 权重 3G 多、SmolVLA 900M，
板上加载完再推理，内存和延迟都撑不住。板上那份 torch 只用来搬张量，取 PyPI 的
aarch64 轮子就够，不需要 Jetson CUDA 版。

| 文件 | 作用 |
|---|---|
| `start_policy_server.sh` | x86 机上起策略服务端（不指定模型） |
| `run_pi0_client.sh` | 板上跑 π0 |
| `run_act_client.sh` | 板上跑 ACT |
| `run_smolvla_client.sh` | 板上跑 SmolVLA |

一个 server 可以先后接三种 client——加载哪个模型是 client 握手时告诉它的，换模型
不用重启 server。

设备名（`/dev/topcam` 等）由 [`platform/so101_real/setup/`](../../../../platform/so101_real/setup/)
的 udev 规则固定，跑本目录脚本前先在板上绑好。
