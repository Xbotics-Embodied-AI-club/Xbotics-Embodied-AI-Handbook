# so101_real —— SO-101 真机设备绑定

真机上的 USB 设备（两路相机、从臂串口）每次插拔都可能换设备号。这里的脚本把它们
按 USB **口位置**固定成稳定的名字，之后所有代码都用固定名字，不再关心 `/dev/video*`
这次排到了几号。

## 目录

| 文件 | 作用 |
|---|---|
| `setup/find_devices.sh` | 打印当前相机/串口落在哪个 USB 口位置（换口后抄 `ID_PATH` 用） |
| `setup/bind_devices.sh` | 写 udev 规则，把设备固定成 `/dev/topcam`、`/dev/wristcam`、`/dev/follower` |
| `setup/bind_camera_s100.sh` | RDK S100 板上的相机绑定（口位置常量与 x86 不同） |
| `setup/bind_uarm_serial_port.sh` | 绑从臂串口 |
| `setup/bind_uarm_serial_port_s100.sh` | RDK S100 板上的从臂串口绑定 |

## 三条真机/仿真线怎么分

| 目录 | 干什么 |
|---|---|
| `so101_sim` 包 | 仿真（ManiSkill3），独立仓 Xbotics-SO101-Sim |
| `platform/so101_real/` | **本目录**：真机设备绑定 |
| `platform/rdk/` | 地瓜 RDK 开发板上的 BPU 上板部署（量化编译 → 板端推理） |

标定（`calibrate_follower.sh`）与相机自检（`check_cameras.py`）属于课程流程，
在 [`vla/2_data_collection/2_2_so101_setup/`](../../vla/2_data_collection/2_2_so101_setup/)。
把训好的策略接到真机上跑（分离式推理：板上取观测、x86 上推理）在
[`vla/5_vla_finetune/5_4_so101_real_sft/deploy/`](../../vla/5_vla_finetune/5_4_so101_real_sft/deploy/)。
