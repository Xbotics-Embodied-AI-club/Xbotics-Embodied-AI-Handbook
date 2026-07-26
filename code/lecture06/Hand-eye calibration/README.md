# Hand-eye calibration

## 项目概述

本目录提供第 6 讲配套的手眼标定教学代码。其目标不仅是给出一个“可运行的求解器”，而是把**相机观测生成、样本整理、数据筛查、手眼求解与闭环验证**这一完整链路尽量透明地呈现出来。

覆盖两种常见拓扑，并与讲义第 5.4.4 / 5.4.5 小节对应：

| 拓扑 | 相机位置 | 目标外参 | 运行时核心链路 |
|---|---|---|---|
| Eye-in-Hand | 相机固定在末端 | `T_end_camera` | `T_base_board = T_base_end · T_end_camera · T_cam_board` |
| Eye-to-Hand | 相机固定在工作空间外 | `T_base_camera` | `T_base_end · T_end_board = T_base_camera · T_cam_board` |

手眼标定的本质，是求一个固定外参矩阵 `X`，使多组相对运动满足：

$$
AX = XB
$$

本目录同时保留了两类脚本：

- **最小 demo 脚本**：适合快速跑通求解与验证；
- **教学支撑脚本**：适合展示 `AX=XB` 的逻辑、检查样本质量、补齐相机侧位姿生成环节。

## 环境准备

```bash
pip install numpy opencv-python
```

如果默认镜像下载较慢或失败，可以临时使用清华 PyPI 镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy opencv-python
```

## 目录说明

| 文件 | 作用 |
|---|---|
| `common_transforms.py` | 齐次矩阵、位姿误差、相对运动构造、教学版求解等公共工具 |
| `estimate_board_pose_aruco.py` | 从 RGB 图像与相机内参提取 ArUco 位姿，输出 `camera_poses.json` |
| `calibrate_eye_in_hand.py` | Eye-in-Hand 主求解脚本（OpenCV / 教学版求解器二选一） |
| `calibrate_eye_to_hand.py` | Eye-to-Hand 主求解脚本（OpenCV / 教学版求解器二选一） |
| `validate_hand_eye.py` | 拓扑感知验证脚本，检查固定关系是否闭环一致 |
| `inspect_hand_eye_dataset.py` | 数据集检查脚本，统计运动激励、弱样本、缺失样本 |
| `solve_hand_eye_teaching.py` | 教学版透明求解器，显式展示 `AX=XB` 的求解逻辑 |
| `generate_synthetic_hand_eye_data.py` | 合成样本生成器，便于课堂演示与回归测试 |
| `pipeline_hand_eye_demo.py` | 一键教学流水线：生成合成样本、筛查、求解、验证并汇总输出 |

## 推荐实操流程

建议按照下面 6 步完成一次教学或实验演示：

1. **采集机器人侧样本**
   - 记录多组 `T_base_end`；
   - 样本应同时包含明显平移与旋转变化；
   - 所有长度单位统一为米。

2. **生成相机侧样本**
   - 采集标定板图像；
   - 使用 `estimate_board_pose_aruco.py` 从图像中提取 `T_cam_board`；
   - 以图像名或自定义规则生成与机器人侧对齐的 `id`。

3. **整理为统一 JSON**
   - 机器人侧写成 `robot_poses.json`；
   - 相机侧写成 `camera_poses.json`；
   - 两边样本用统一 `id` 配对。

4. **先做筛样与激励检查**
   - 用 `inspect_hand_eye_dataset.py` 查看运动对数量是否足够；
   - 查看平移 / 旋转变化是否过小；
   - 查看是否存在缺失配对和近重复样本。

5. **执行求解**
   - 使用 `calibrate_eye_in_hand.py` 或 `calibrate_eye_to_hand.py`；
   - 默认采用 OpenCV 工程解；
   - 若需要教学展示，可切换 `--solver park_teaching`。

6. **做闭环验证**
   - Eye-in-Hand：检查恢复出的 `T_base_board` 是否近似恒定；
   - Eye-to-Hand：检查恢复出的 `T_end_board` 是否近似恒定；
   - 用 `validate_hand_eye.py` 输出逐样本残差和最差样本。

## 最小可运行自检

如果还没有真实机器人和相机数据，可以先用合成样本验证代码链路。下面命令不会依赖外部硬件，适合确认 Python 环境、矩阵方向和 JSON 输出是否正常。`pipeline_hand_eye_demo.py` 会依次完成：

```text
生成合成样本 → 样本配对与运动激励检查 → 求解固定外参 → 闭环一致性验证 → 输出 pipeline_summary.json
```

### Eye-in-Hand 自检

```bash
cd "code/lecture06/Hand-eye calibration"

python pipeline_hand_eye_demo.py --topology eye_in_hand
```

默认输出目录为 `data/hand_eye_demo/eye_in_hand/`。核心结果是 `eye_in_hand.json`，其中的 `matrix` 字段就是 `T_end_camera`。

### Eye-to-Hand 自检

```bash
cd "code/lecture06/Hand-eye calibration"

python pipeline_hand_eye_demo.py --topology eye_to_hand
```

默认输出目录为 `data/hand_eye_demo/eye_to_hand/`。核心结果是 `eye_to_hand.json`，其中的 `matrix` 字段就是 `T_base_camera`。

两个 demo 目录都会包含：

| 文件 | 说明 |
|---|---|
| `robot_poses.json` | 合成机器人侧样本，字段为 `T_base_end` |
| `camera_poses.json` | 合成相机侧样本，字段为 `T_cam_board` |
| `ground_truth.json` | 合成数据使用的真实外参，仅用于教学自检 |
| `inspect_report.json` | 样本配对、运动激励和弱运动对检查 |
| `eye_in_hand.json` / `eye_to_hand.json` | 求解得到的手眼外参 |
| `*_calibration_report.json` | 求解阶段详细报告 |
| `validation_report.json` | 闭环一致性验证报告 |
| `pipeline_summary.json` | 一键流水线摘要，包含残差和输出路径 |

`inspect`、`calibrate` 和 `validate` 的 JSON 报告都会包含 `pairing` 字段，用于检查机器人侧和相机侧样本是否按 `id` 成功配对；同时包含 `diagnostics` 字段，用于提示样本数量不足、平移 / 旋转激励不足或弱运动对过多等问题。真实标定时，建议先让 `inspect` 报告没有明显警告，再执行求解。

### 分步展开版

一键脚本内部调用的仍是下面这些独立脚本。需要替换为真实数据时，可以保留 `inspect → calibrate → validate` 这个顺序，只把 `robot_poses.json` 和 `camera_poses.json` 换成实采文件。

```bash
cd "code/lecture06/Hand-eye calibration"

python inspect_hand_eye_dataset.py \
  --topology eye_in_hand \
  --robot-poses data/hand_eye_demo/eye_in_hand/robot_poses.json \
  --camera-poses data/hand_eye_demo/eye_in_hand/camera_poses.json \
  --report data/hand_eye_demo/eye_in_hand/inspect_report.json

python calibrate_eye_in_hand.py \
  --robot-poses data/hand_eye_demo/eye_in_hand/robot_poses.json \
  --camera-poses data/hand_eye_demo/eye_in_hand/camera_poses.json \
  --output data/hand_eye_demo/eye_in_hand/eye_in_hand.json \
  --report-output data/hand_eye_demo/eye_in_hand/eye_in_hand_calibration_report.json

python validate_hand_eye.py \
  --transform data/hand_eye_demo/eye_in_hand/eye_in_hand.json \
  --robot-poses data/hand_eye_demo/eye_in_hand/robot_poses.json \
  --camera-poses data/hand_eye_demo/eye_in_hand/camera_poses.json \
  --report data/hand_eye_demo/eye_in_hand/validation_report.json
```

## 输入数据格式

### 机器人侧样本

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

### 相机侧样本

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

### 建议附加字段

如果你已有更完整的相机侧结果，也可以在同一条记录中附加元数据，例如：

- `reprojection_error_px`
- `timestamp`
- `board_type`
- `marker_id`
- `marker_size_m`
- `image_path`

当前求解脚本会保留这些字段，但核心求解仍只依赖 `id` 与对应 4×4 矩阵。

## ArUco 相机侧位姿生成

`estimate_board_pose_aruco.py` 用于把标定板图像转换成手眼标定所需的 `camera_poses.json`。其输出字段与当前 `calibrate_*` 脚本保持兼容。

### 输入约定

- 单张图像：`--image path/to/rgb.png`
- 批处理目录：`--input-dir path/to/images/`
- 相机内参：`intrinsics.json`，至少包含：

```json
{
  "fx": 615.0,
  "fy": 615.0,
  "cx": 320.0,
  "cy": 240.0
}
```

可选畸变参数支持两种写法：

```json
{
  "dist_coeffs": [k1, k2, p1, p2, k3]
}
```

或

```json
{
  "dist": [k1, k2, p1, p2, k3]
}
```

若未提供畸变参数，脚本默认使用零畸变模型。

### 典型命令

```bash
cd "code/lecture06/Hand-eye calibration"
python estimate_board_pose_aruco.py \
  --input-dir data/aruco_images \
  --intrinsics data/intrinsics.json \
  --marker-size-m 0.04 \
  --marker-id 0 \
  --output data/camera_poses.json \
  --report-output outputs/aruco_report.json
```

### 脚本输出

- `camera_poses.json`：可直接供 `calibrate_eye_in_hand.py` / `calibrate_eye_to_hand.py` 使用；
- 可选详细报告：包含每张图是否检测成功、重投影误差、`rvec/tvec`、拒绝原因等。

### ArUco 使用建议

| 项目 | 建议 |
|---|---|
| 字典 | 教学默认 `DICT_4X4_50` |
| 标定板形式 | 优先单 marker 教学版，便于解释 `T_cam_board` |
| PnP 求解 | 单方形 marker 优先 `SOLVEPNP_IPPE_SQUARE` |
| marker 尺寸 | 必须使用米制并与真实打印尺寸一致 |
| 误差阈值 | 可先以 `2.0 px` 左右作为重投影误差筛选阈值 |

### ArUco 常见问题

- **尺寸单位写错**：例如把 `40 mm` 写成 `0.4 m`，会直接导致尺度错误；
- **目标 marker id 不一致**：会导致检测到 marker 但无法生成目标位姿；
- **图像畸变未处理**：在广角或边缘区域会显著恶化重投影误差；
- **图像命名与机器人样本 id 不一致**：后续无法正确配对。

## 采样建议

| 项目 | 建议 |
|---|---|
| 样本数 | 建议 15–30 组 |
| 姿态覆盖 | 同时包含明显平移与明显旋转变化 |
| 视角变化 | 避免所有样本都只在一个平面内小幅移动 |
| 单位 | 全部统一为米 |
| 支架刚性 | 相机支架、标定板支架必须足够刚性 |

如果样本只做了很小的平移或几乎没有转动，手眼标定很容易退化，即使求解器给出结果，结果也可能不稳定。

## 常用命令

### 1. ArUco 位姿提取

```bash
cd "code/lecture06/Hand-eye calibration"
python estimate_board_pose_aruco.py \
  --input-dir data/aruco_images \
  --intrinsics data/intrinsics.json \
  --marker-size-m 0.04 \
  --marker-id 0 \
  --output data/camera_poses.json \
  --report-output outputs/aruco_report.json
```

### 2. 数据检查

```bash
cd "code/lecture06/Hand-eye calibration"
python inspect_hand_eye_dataset.py \
  --topology eye_in_hand \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --report outputs/inspect_report.json
```

### 3. Eye-in-Hand 标定

```bash
cd "code/lecture06/Hand-eye calibration"
python calibrate_eye_in_hand.py \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --output outputs/eye_in_hand.json \
  --report-output outputs/eye_in_hand_report.json
```

### 4. Eye-to-Hand 标定

```bash
cd "code/lecture06/Hand-eye calibration"
python calibrate_eye_to_hand.py \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --output outputs/eye_to_hand.json \
  --report-output outputs/eye_to_hand_report.json
```

### 5. 教学版透明求解器

```bash
cd "code/lecture06/Hand-eye calibration"
python solve_hand_eye_teaching.py \
  --topology eye_in_hand \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --output outputs/teaching_solver_report.json
```

### 6. 闭环验证

```bash
cd "code/lecture06/Hand-eye calibration"
python validate_hand_eye.py \
  --transform outputs/eye_in_hand.json \
  --robot-poses data/robot_poses.json \
  --camera-poses data/camera_poses.json \
  --report outputs/validation_report.json
```

### 7. 生成合成数据

```bash
cd "code/lecture06/Hand-eye calibration"
python generate_synthetic_hand_eye_data.py \
  --topology eye_in_hand \
  --output-dir synthetic/eye_in_hand_case \
  --sample-count 16 \
  --noise-translation-mm 0.5 \
  --noise-rotation-deg 0.2
```

## OpenCV 工程解 vs 教学版求解器

| 方式 | 适合场景 | 特点 |
|---|---|---|
| OpenCV `calibrateHandEye` | 真机数据、快速求解 | 稳定、开箱即用、工程中最常用 |
| `park_teaching` 教学解 | 课堂讲解、理解 `AX=XB` | 能看到相对运动构造、旋转 / 平移分步求解逻辑 |

建议：

- **先用 OpenCV 工程解获得稳定结果**；
- **再用教学版求解器解释“为什么能够求解”**。

## 输出内容

求解脚本会输出：

- 4×4 齐次变换矩阵；
- 样本数与运动对数量；
- 平移 / 旋转残差统计；
- 弱运动对提示；
- 可选详细报告 JSON。

验证脚本会输出：

- 闭环一致性残差；
- 最差样本；
- 逐样本误差；
- 参考固定变换（如 `T_base_board` 或 `T_end_board`）。

数据检查脚本会根据 `--topology` 选择正确的相对运动构造方式：

| 拓扑 | 检查时使用的机器人侧运动 | 检查目标 |
|---|---|---|
| `eye_in_hand` | `T_base_end` 的相对运动 | 判断末端运动与相机观测是否共同提供足够激励 |
| `eye_to_hand` | `inv(T_base_end)` 的相对运动 | 判断固定相机拓扑下的 `AX=XB` 运动对是否充分 |

## 常见错误与排查建议

### 1. 单位不统一

- 表现：结果尺度明显不对，平移残差大；
- 原因：机器人是米，标定板尺寸或相机端是毫米；
- 建议：统一全部长度单位为米。

### 2. 欧拉角顺序弄错

- 表现：矩阵数值看起来正常，但真机复现完全偏；
- 原因：机器人 SDK 的 RPY 顺序与自己构造矩阵时不一致；
- 建议：明确是 `XYZ` 还是 `ZYX`，不要凭感觉写。

### 3. 矩阵方向写反

- 表现：旋转方向、坐标链完全相反；
- 原因：把 `T_a_b` 和 `T_b_a` 混用；
- 建议：统一采用“把 b 坐标系下的点变到 a 坐标系”的记号。

### 4. 样本激励不足

- 表现：不同方法结果漂移大、验证残差不稳定；
- 原因：采样几乎都在小范围挪动，没有足够旋转变化；
- 建议：增加大角度姿态变化，覆盖近中远不同位置。

### 5. 支架不刚性

- 表现：理论上闭环成立，但逐样本残差忽大忽小；
- 原因：相机支架或标定板夹具存在微小晃动；
- 建议：先排机械刚性，再怀疑求解器。

### 6. TCP / 工具坐标未统一

- 表现：标定矩阵验证正常，但抓取点复现仍偏；
- 原因：机器人末端法兰系与夹爪 TCP 没有统一；
- 建议：补齐 `T_end_tcp`，并与抓取链路一致使用。

## 一句话理解本目录

- `estimate_board_pose_aruco.py`：把图像观测变成 `T_cam_board`；
- `inspect_hand_eye_dataset.py`：看样本是否足够好；
- `calibrate_*.py`：把样本解成手眼外参；
- `validate_hand_eye.py`：看结果是否真的闭环一致；
- `solve_hand_eye_teaching.py`：看 `AX=XB` 到底是怎么求的；
- `generate_synthetic_hand_eye_data.py`：给课堂和回归测试提供标准输入。
