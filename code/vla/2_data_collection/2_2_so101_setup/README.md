# SO-101 硬件标定与自检

> 设备绑定（udev 固定相机与串口名）已上移到平台层
> [`platform/so101_real/setup/`](../../../platform/so101_real/setup/)，本节只讲标定与自检。

标定从臂、检查相机、回放录制的 episode 验证硬件是否正常：

- `calibrate_follower.sh` — 标定 SO101 从臂，结果落 `$HF_LEROBOT_HOME/calibration/`
- `check_cameras.py` — 两路相机同时取图测实际帧率，各存一张样张确认没装反
- `replay_episode.sh` — 把某条 episode 的动作逐帧原样回放到真机，验证标定和硬件是否正常
