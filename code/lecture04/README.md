# Sensors and coordinates —— 传感器与坐标转换（无硬件仿真版）

## 项目概述

本目录提供第 4 讲配套的传感器与坐标转换教学代码，对应讲义 `docs/part1-system-basics/04-sensors-coordinates.md` 的 **4.10.3 无硬件仿真版 Demo**。其目标不仅是给出一个“能算出坐标的脚本”，而是把**深度读取、邻域鲁棒采样、反投影、外参变换、可视化、误差扰动与目标位姿生成**这一完整链路尽量透明地呈现出来。

贯穿案例是一条单向链路：

```text
点击目标像素 → 邻域鲁棒深度 → 反投影到 camera 系 → 外参变换到 base 系 → 点云可视化 → 扰动实验 → 可执行目标位姿
```

坐标转换的本质，是把同一个物理点在两个坐标系下的表达用一个固定齐次矩阵联系起来：

$$
p_{base} = T_{base\_camera} \cdot p_{camera}
$$

本目录覆盖两种数据源，在 Notebook 中用 `SOURCE` 变量一键切换：

| 数据源 | 图像来源 | 深度 scale | 真值对照 |
|---|---|---|---|
| `tum`（默认） | 真实 TUM RGB-D fr1/desk 序列 | `/5000` 米 | 无逐点真值，按量级 / 内参 / scale 合理性验收 |
| `synthetic` | `make_sample_data.py` 程序合成 | `/1000` 米 | 有逐点真值（讲义 4.6.6 例子） |

纯 Python + Jupyter Notebook 实现，**不依赖 ROS 2 运行时**。讲义 4.10.3 约定无硬件版的输入只有「样例 RGB、深度图、相机内参、虚拟 `T_base_camera` + 可视化环境」——这里的虚拟外参矩阵即等价于 TF 树中 `base_link → camera_optical_frame` 的 static transform，无需真正运行 `tf2`。

## 环境准备

推荐用仓库自带脚本一键创建隔离环境并注册 Jupyter kernel：

```bash
cd code/lecture04/simulation
bash setup_env.sh              # 默认 venv；--conda 走 environment.yml；--venv 显式指定
```

也可以手工安装依赖：

```bash
pip install numpy scipy matplotlib PyYAML Pillow jupyter ipykernel
```

如果默认镜像下载较慢或失败，可以临时使用清华 PyPI 镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

环境基线：

| 项目 | 基线 |
|---|---|
| Python | 3.10+（本机 3.10.12 实测通过） |
| 核心依赖 | `numpy>=1.21`、`scipy>=1.8`、`matplotlib>=3.5`、`PyYAML>=6.0`、`Pillow>=9.0` |
| Notebook | `jupyter` + `ipykernel`（kernel 名 `lecture4-sensors`） |
| ROS 2 | **无运行时依赖**；接真机走 TF 见讲义 4.10.2 |

### 仓库不包含的资源

下列内容体积过大或可本地生成，**已在 `.gitignore` 中排除，clone 后不存在**，需按对应命令自行获取：

| 资源 | 体积 | 获取方式 |
|---|---|---|
| `rgbd_dataset_freiburg1_desk.tgz` | ~328 MB | TUM 官网下载，见「完整 TUM 路径」 |
| `rgbd_dataset_freiburg1_desk/` | ~353 MB | 上者解压产物，提取完可删 |
| `.venv/` | — | `bash setup_env.sh` |
| `__pycache__/` | — | 运行时自动生成 |

`data/` 下已提取好的 5 个小文件（合成 + TUM 各一套，共约 2 MB）**随仓库分发**，因此只想跑 demo 的人无需下载 TUM 原始序列。

## 目录说明

本 README 位于 `code/lecture04/`，全部代码与数据在同级的 `simulation/` 下；真机路径见 [`hardware/README.md`](hardware/README.md)。

```text
code/lecture04/
├── README.md          # 本文件
├── hardware/          # 有硬件版路径（待补充）
└── simulation/        # 无硬件仿真版路径，下表文件均在此目录内
```

| 文件 | 作用 |
|---|---|
| `sensors_coordinates_demo.ipynb` | 主交付 Notebook，26 个 cell（13 markdown + 13 code），已保存一次完整执行结果 |
| `make_sample_data.py` | 合成样例数据生成器，输出 RGB 图与 uint16 毫米深度图，含已知逐点真值 |
| `prepare_tum_data.py` | 从 TUM 序列按时间戳最近邻配对（等价官方 `associate.py`），提取一对 RGB+depth 与内参 |
| `setup_env.sh` | 一键创建隔离环境（venv / conda）并注册 kernel `lecture4-sensors` |
| `camera_info.yaml` | 合成相机内参，ROS `CameraInfo` 字段布局 |
| `environment.yml` | conda 环境定义（`name: lecture4-sensors`，conda-forge，Python 3.10） |
| `requirements.txt` | pip 依赖清单（venv 路径使用） |
| `data/` | 样例数据，合成与 TUM 各一套 |
| `output/` | 运行产物，`results.json` 与可视化图 |

`data/` 目录内容：

| 文件 | 说明 |
|---|---|
| `rgb.png` | 合成 RGB，640×480 |
| `depth.npy` | 合成深度，numpy `uint16`，`(480, 640)`，值域 0–1200 毫米 |
| `tum_rgb.png` | TUM 真实 RGB，640×480 |
| `tum_depth.png` | TUM 真实深度，16 bit PNG，`0` 表示无效 |
| `tum_camera_info.yaml` | TUM fr1 的 RGB / depth 内参与深度 scale |

`output/` 目录内容：

| 文件 | 说明 |
|---|---|
| `results.json` | Notebook 汇总结果，见「输出数据格式」 |
| `step7_visualization.png` | 点云 + 坐标轴 + 目标点可视化 |

## 推荐实操流程

建议按照下面 5 步完成一次教学或实验演示：

1. **准备隔离环境**
   - 运行 `bash setup_env.sh`；
   - 确认 kernel `lecture4-sensors` 注册成功；
   - 避免系统 Python 里 `matplotlib` 与 `matplotlib-inline` 版本错配。

2. **准备样例数据**
   - 合成路径：运行 `make_sample_data.py`，或让 Notebook 在数据缺失时自动现场合成；
   - TUM 路径：下载并解压 fr1/desk 序列，用 `prepare_tum_data.py` 提取一对图；
   - 确认 `data/` 下目标文件存在。

3. **确认数据源与内参**
   - 在 Notebook 中设置 `SOURCE`，默认 `"tum"`；
   - 检查 RGB / depth 的尺寸、`dtype`、深度值域；
   - **反投影必须用 depth 相机内参**，不要误用 RGB 内参。

4. **跑通主链路**
   - `Kernel → Restart & Run All`；
   - 关注 `point_camera`、`point_base`、`target_pose_base` 三个数值；
   - 合成数据可直接与逐点真值对照。

5. **做扰动实验与复盘**
   - 依次运行深度 scale 错误、外参旋转 1°、外参取逆三组扰动；
   - 对照现象量级判断“错在哪一环”；
   - 检查 `output/results.json` 是否落盘。

## 最小可运行自检

如果只想确认 Python 环境、数组方向和内参配置是否正常，可以先跑合成数据路径，它不依赖 TUM 原始序列，也不依赖任何硬件：

```text
合成 RGB+depth → 邻域鲁棒深度 → 反投影 → 外参变换 → 与逐点真值比对
```

```bash
cd code/lecture04/simulation
source .venv/bin/activate
python make_sample_data.py
```

脚本会生成 `data/rgb.png` 与 `data/depth.npy`。随后在 Notebook 中把 `SOURCE` 改为 `"synthetic"` 并全量运行，预期链路数值为：

```text
pixel (420, 220) → point_camera [0.10, -0.02, 0.60] → point_base [0.80, -0.05, 0.42]
→ target_pose_base [0.80, -0.05, 0.47, 1, 0, 0, 0]
```

合成内参为 `fx=fy=600, cx=320, cy=240 @ 640×480`（无畸变），深度为 `uint16 毫米 / 1000 = 米`。若上述数值能对上，说明反投影公式、矩阵方向与单位换算都正确。

### 完整 TUM 路径

默认路径 `SOURCE="tum"` 需要 `data/` 下已有 TUM 图。从零准备：

```bash
cd code/lecture04/simulation

# 1. 下载序列（约 328 MB）
# https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz

# 2. 解压并提取一对对齐的 RGB+depth
tar -xzf rgbd_dataset_freiburg1_desk.tgz
python prepare_tum_data.py rgbd_dataset_freiburg1_desk --index 0
```

生成 `data/tum_rgb.png`、`data/tum_depth.png`、`data/tum_camera_info.yaml`。

> TUM 的 RGB 与 depth 时间戳不同步，必须用 associate（最近邻时间戳匹配）配对；`prepare_tum_data.py` 已实现该逻辑，`--index` 用于换其他帧。

TUM 路径点击图像中心 `(320, 240)`（该点深度有效且接近 depth 主点），反投影点接近光轴：

```text
pixel (320, 240) → point_camera ≈ [0.0006, 0.0006, 0.627] → point_base ≈ [0.827, 0.049, 0.399]
```

## 输入数据格式

### 相机内参（合成，`camera_info.yaml`）

```yaml
image_width: 640
image_height: 480
frame_id: camera_optical_frame
camera_matrix:
  data: [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  data: [0.0, 0.0, 0.0, 0.0, 0.0]
```

### 相机内参（TUM，`data/tum_camera_info.yaml`）

```yaml
source: TUM RGB-D fr1
rgb:   {fx: 517.3, fy: 516.5, cx: 318.6, cy: 255.3}
depth: {fx: 525.0, fy: 525.0, cx: 319.5, cy: 239.5}
depth_scale: 5000.0
depth_invalid: 0
```

TUM 的 depth 已配准到 RGB 平面，像素一一对应；但**反投影必须使用 depth 内参**，RGB 内参仅用于显示。

### 深度图约定

| 数据源 | dtype | 单位 | 换算 | 无效值 |
|---|---|---|---|---|
| `synthetic` | `uint16`（`.npy`） | 毫米 | `/1000` | `0` |
| `tum` | 16 bit PNG | TUM 自定标 | `/5000` | `0` |

## Notebook 步骤与讲义映射

| Notebook 步骤 | 讲义 | 验收证据 |
|---|---|---|
| 加载并检查 RGB/depth 尺寸、dtype、深度范围、scale | 4.10.3 步骤 1 / 4.6.4 | 终端输出 |
| 数据源切换（合成 / TUM）与内参 | 4.6.3 | 终端输出 |
| 选择目标像素、邻域中位数 | 4.10.3 步骤 3/4 / 4.6.4 | 有效样本数 |
| 反投影 `point_camera` | 4.10.3 步骤 5 / 4.6.5 | 数值（合成可对照真值） |
| `T_base_camera` 转 `point_base` | 4.10.3 步骤 6 / 4.6.6 | 数值 |
| 点云 + 坐标轴 + 目标点可视化 | 4.10.3 步骤 7 | `output/step7_visualization.png` |
| 深度 scale 错误 / +10 mm 扰动 | 4.10.3 步骤 8 / 4.10.5 | 位移量级 |
| 外参旋转 1° 扰动 | 4.10.3 步骤 9 | 远点误差放大 |
| 外参取逆（方向写反） | 4.10.3 步骤 10 / 4.10.5 | 目标落到地下 |
| 安全偏移 + 姿态 → `target_pose_base` | 4.6.7 | 位姿数值 |
| 汇总保存 | 4.10.3 步骤 11 | `output/results.json` |

Notebook 中的核心函数：

| 函数 | 作用 |
|---|---|
| `robust_depth(depth_raw, u, v, depth_scale, radius=3)` | 邻域中位数深度，剔除 `0` 无效值 |
| `deproject(u, v, Z)` | 用内参把像素 + 深度反投影为 camera 系三维点 |
| `apply_transform(T, p)` | 4×4 齐次变换作用于三维点 |
| `build_pointcloud_base(depth_raw, depth_scale, step=8)` | 整张深度图反投影并变换到 base 系，用于可视化 |
| `draw_frame(ax, R, t, length=0.12, label="")` | 在 3D 图上绘制坐标系三轴 |

## 三组扰动实验

每组只改一个变量，用于复现典型失败并建立“现象 → 原因”的对应关系：

| 扰动 | 现象（合成） | 现象（TUM） |
|---|---|---|
| 忘记除以 depth scale | `point_base` 飞到 `[600.2, -99.95, 20.4]` | 飞到 `[3136.2, -2.94, -2.59]` |
| 外参旋转 1° | 近点误差 0.011 m / 远点 0.053 m | 近点 0.011 m / 远点 0.052 m |
| `T_base_camera` 取逆 | `z = -0.10`，目标落到地下 | `z = -0.199`，目标落到地下 |

三个现象的量级本身就是诊断依据：**尺度错误是百倍级偏移，旋转错误随距离线性放大，方向写反则整条链路符号翻转**。

## 输出数据格式

`output/results.json`（下例为默认 `SOURCE="tum"` 的一次实际运行）：

```json
{
  "source": "tum",
  "click_pixel": [320, 240],
  "depth_scale": 5000.0,
  "depth_m": 0.6272,
  "depth_raw_median": 3136.0,
  "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
  "point_camera": [0.000597, 0.000597, 0.6272],
  "point_base": [0.8272, 0.049403, 0.399403],
  "target_pose_base": [0.8272, 0.049403, 0.449403, 1.0, 0.0, 0.0, 0.0],
  "depth_plus10mm_delta_m": [0.01, -1e-05, -1e-05],
  "forgot_scale_point_base": [3136.2, -2.937, -2.587],
  "rotation_1deg_near_err_m": 0.010947,
  "rotation_1deg_far_err_m": 0.052359,
  "inverse_extrinsic_point_base": [0.0494, -0.2272, -0.1994]
}
```

说明：

- `target_pose_base` 前 3 位是位置，后 4 位是四元数 `[x, y, z, w] = [1, 0, 0, 0]`，表示夹爪朝下（绕 base 的 x 轴旋转 180°）；
- 位置比 `point_base` 高 0.05 m，是抓取前的**安全偏移**；
- `SOURCE="synthetic"` 时会额外写入 `point_camera_gt`、`point_base_gt`、`reprojection_error_m`、`transform_error_m` 等真值对照字段。

## 使用建议

| 项目 | 建议 |
|---|---|
| 内参选择 | 反投影一律用 depth 内参，RGB 内参只用于显示 |
| 深度采样 | 用邻域中位数而非单像素，先剔除 `0` 无效值 |
| 单位 | 全部统一为米，深度先除 scale 再进入任何几何计算 |
| 目标像素 | 选深度有效、纹理稳定的点，避免物体边缘的深度跳变 |
| 矩阵记号 | 统一采用 `T_a_b` 表示“把 b 系下的点变到 a 系” |
| 环境 | 用 `setup_env.sh` 的隔离环境，避免系统包版本冲突 |

如果目标像素落在深度空洞或物体边缘，即使公式完全正确，反投影结果也会出现跳变，这类问题应先查数据而非查代码。

## 常用命令

### 1. 创建隔离环境

```bash
cd code/lecture04/simulation
bash setup_env.sh                  # venv（默认）
bash setup_env.sh --conda          # conda，走 environment.yml
```

### 2. 生成合成样例数据

```bash
cd code/lecture04/simulation
python make_sample_data.py         # 输出 data/rgb.png、data/depth.npy
python make_sample_data.py --show  # 额外弹窗显示（需 GUI）
```

### 3. 提取 TUM 数据对

```bash
cd code/lecture04/simulation
python prepare_tum_data.py rgbd_dataset_freiburg1_desk \
  --index 0 \
  --out data
```

### 4. 运行 Notebook

```bash
cd code/lecture04/simulation
source .venv/bin/activate          # conda 方式：conda activate lecture4-sensors
jupyter notebook sensors_coordinates_demo.ipynb
```

打开后执行 **Kernel → Restart & Run All**，并确认右上角 kernel 为 `Python (lecture4-sensors)`。

## 常见错误与排查建议

### 1. 忘记除以 depth scale

- 表现：`point_base` 数值达到百米甚至千米量级；
- 原因：把原始 `uint16` 深度当成米直接参与计算；
- 建议：合成数据除以 1000，TUM 除以 5000，换算写在读取环节而不是散落各处。

### 2. 用错内参

- 表现：结果量级正常但横向偏移持续存在；
- 原因：TUM 的 RGB 内参（`517.3/516.5`）被用来做深度反投影；
- 建议：反投影固定用 depth 内参 `525.0/525.0/319.5/239.5`。

### 3. 外参方向写反

- 表现：目标点 `z` 为负，落到地面以下；
- 原因：把 `T_base_camera` 和 `T_camera_base` 混用；
- 建议：统一记号为“把 b 系下的点变到 a 系”，并用一个已知点做方向自检。

### 4. 深度空洞或边缘跳变

- 表现：换一个相邻像素结果差别很大；
- 原因：点击到深度为 `0` 的空洞或物体边缘；
- 建议：用 `robust_depth` 的邻域中位数，并检查有效样本数是否足够。

### 5. `%matplotlib inline` 报 `'RcParams' object has no attribute '_get'`

- 表现：Notebook 首个绘图 cell 直接报错；
- 原因：`matplotlib-inline` 与旧版 `matplotlib` 版本错配；
- 建议：使用 `setup_env.sh` 创建的隔离环境，不要用系统 Python。

### 6. Notebook 找不到 kernel `lecture4-sensors`

- 表现：内核列表中无对应条目；
- 原因：未执行环境脚本，kernel 未注册；
- 建议：先运行 `bash setup_env.sh` 完成 `ipykernel install`。

### 7. TUM 数据缺失

- 表现：默认 `SOURCE="tum"` 时读取图像失败；
- 原因：`data/tum_rgb.png` / `data/tum_depth.png` 不存在；
- 建议：按「完整 TUM 路径」重新提取，或把 `SOURCE` 改为 `"synthetic"`。

## 与真机 ROS 2 版的关系

无硬件版与真机版共享同一套数学与验收标准（讲义 4.10.4）。接真机时只需替换两个边界：

| 环节 | 无硬件版 | 真机 ROS 2 版 |
|---|---|---|
| 外参来源 | 硬编码虚拟 `T_base_camera` | `tf_buffer.lookup_transform("base_link", msg.header.frame_id, msg.header.stamp)`（讲义 4.9.8） |
| 数据来源 | 样例 RGB + depth 文件 | 真实 RGB-D 驱动的话题订阅 |

反投影、坐标变换、鲁棒深度与误差检查逻辑完全不变。

## 一句话理解本目录

- `make_sample_data.py`：造一份带真值的可控输入；
- `prepare_tum_data.py`：把真实 TUM 序列变成一对可用的 RGB+depth；
- `setup_env.sh`：保证环境干净、kernel 可用；
- `camera_info.yaml` / `data/tum_camera_info.yaml`：告诉你该用哪组内参和 scale；
- `sensors_coordinates_demo.ipynb`：把像素一路算到可执行目标位姿，并用三组扰动说明它为什么会算错。
