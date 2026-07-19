# Hand-eye calibration

## 项目概述

本目录提供第 6 讲配套的手眼标定示例，覆盖两种常见拓扑，并与讲义第 5.4.4 / 5.4.5 小节对应：

| 拓扑 | 相机位置 | 目标外参 |
|---|---|---|
| Eye-in-Hand | 相机固定在末端 | `T_end_camera` |
| Eye-to-Hand | 相机固定在工作空间外 | `T_base_camera` |

## 目录说明

| 文件 | 作用 |
|---|---|
| `common_transforms.py` | 齐次矩阵、误差统计、JSON 读写等公共工具 |
| `calibrate_eye_in_hand.py` | Eye-in-Hand 标定求解脚本（对应 5.4.4） |
| `calibrate_eye_to_hand.py` | Eye-to-Hand 标定求解脚本（对应 5.4.5） |
| `validate_hand_eye.py` | 标定结果验证脚本 |

## 输入数据格式

建议使用统一 JSON 记录，每个样本都要有同一个 `id`，例如：

```json
{
  "poses": [
    {
      "id": "sample_001",
      "T_base_end": [[...], [...], [...], [...]]
    }
  ]
}
```

相机侧样本建议写成：

```json
{
  "poses": [
    {
      "id": "sample_001",
      "T_cam_board": [[...], [...], [...], [...]]
    }
  ]
}
```

## 运行示例

```bash
cd "code/lecture06/Hand-eye calibration"
python calibrate_eye_in_hand.py \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --output outputs/eye_in_hand.json
```

```bash
cd "code/lecture06/Hand-eye calibration"
python calibrate_eye_to_hand.py \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --output outputs/eye_to_hand.json
```

```bash
cd "code/lecture06/Hand-eye calibration"
python validate_hand_eye.py \
  --transform outputs/eye_to_hand.json \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --report outputs/validation_report.json
```

## 输出内容

脚本会保存：
- 4×4 齐次变换矩阵
- 样本数与运动对数量
- 平移 / 旋转残差统计
- 验证报告 JSON

## 说明

- 统一使用米制单位，标定板尺寸、机器人位姿、深度尺度必须保持一致。
- 请显式区分 `T_end_camera` 与 `T_base_camera`，不要混用方向命名。
- `calibrate_eye_in_hand.py` 和 `calibrate_eye_to_hand.py` 采用 OpenCV `calibrateHandEye` 的教学封装，适合课堂样例和小规模真机数据。
- 教学版脚本优先保证可读性和可验证性。
