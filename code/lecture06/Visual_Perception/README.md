# xbotics —— 深度图 → 点云处理流水线

## 项目概述

本项目提供从**深度图 (depth image)** 到**干净目标点云 (clean point cloud)** 的完整处理链路，支持多种输入源和交互方式。

### 课堂配套脚本（5 步法）

[`pipeline_from_syllabus.py`](pipeline_from_syllabus.py) 是一个**完全独立、可直接运行**的教学脚本，严格对应课堂大纲的 5 个步骤：

```
步骤 1 ── 掩膜筛选：       depth × mask → masked_depth
步骤 2 ── 反投影生成：      利用相机内参 K，将过滤后的深度图映射为 3D 坐标 (X, Y, Z)
步骤 3 ── 空间粗裁（直通）： numpy 数组切片，切除 Z < 0（桌面以下）或距离过远的无效空间
步骤 4 ── 精洗去噪：        统计滤波（剔除孤立飞点）→ 体素降采样（网格合并，完成瘦身）
步骤 5 ── 目标点云提取：    导出最终 .pcd 文件，用于下一讲的位姿估计
```

相比其他脚本，该脚本使用 **Open3D 内置 `create_from_depth_image()`** 进行反投影（大纲推荐方式），并使用 **numpy 数组切片** 做直通滤波。脚本会保存 masked depth、各阶段点云、阶段截图和 JSON 摘要，便于和教材中的图 6.8 至图 6.12 对照阅读。

```
深度图 + mask + 相机内参 ──→ 掩膜筛选深度 ──→ 反投影目标点云 ──→ 直通滤波 ──→ 统计滤波去噪 ──→ 体素降采样 ──→ 干净点云
```

---

## Python 脚本说明

| 脚本 | 功能描述 |
|------|----------|
| [`depth_sources.py`](depth_sources.py) | **深度数据输入源抽象层**。定义 `DepthFrame` 数据结构和三种输入源：`FileDepthSource`（本地文件）、`UsbDepthCameraSource`（RealSense USB 相机）、`NetworkDepthSource`（网络 JSON manifest）。工厂函数 `create_depth_source()` 统一创建。 |
| [`Open3D_PointCloud_Pipeline.py`](Open3D_PointCloud_Pipeline.py) | **Open3D 点云预处理核心库**。包含深度图→点云反投影、直通滤波、体素降采样、统计滤波、并排对比可视化、2×2 拼图等核心函数。可独立运行（默认使用斯坦福兔子数据），也被其他脚本调用。 |
| [`DepthMask_PointCloud_Pipeline.py`](DepthMask_PointCloud_Pipeline.py) | **深度图+掩膜→点云处理管道**。加载 depth.png + mask.png + intrinsics.json，完成场景反投影→掩膜提取→三步滤波，输出 3 个 `.pcd` 文件及截图/拼图。 |
| [`interactive_mask_ui.py`](interactive_mask_ui.py) | **基于 OpenCV 的交互式掩膜选择 UI**。提供鼠标单击（点提示）和拖拽（框提示）两种交互方式，以及分割结果预览确认/重选/取消功能。 |
| [`sam2_segmenter.py`](sam2_segmenter.py) | **SAM2 分割包装层**。提供统一的 `segment()` 接口（point/box 提示），当前含圆形/矩形回退掩膜逻辑，预留真实 SAM2 后端扩展接口。 |
| [`interactive_depth_pipeline.py`](interactive_depth_pipeline.py) | **交互式深度管道统一入口**。整合输入源 + 交互选择 + SAM2 分割 + 点云管道，支持 `--source file|usb|network` 三种模式。 |
| [`PointCloud_Sandbox.py`](PointCloud_Sandbox.py) | **交互式点云滤波沙盒**。基于 Open3D GUI，实时滑块控制滤波参数并即时更新 3D 渲染，适合教学演示和参数探索。 |
| [`pipeline_from_syllabus.py`](pipeline_from_syllabus.py) | **课堂 5 步法独立脚本**。严格对应大纲"掩膜筛选→反投影→直通滤波→统计+体素→导出"五步骤，使用 Open3D 内置 `create_from_depth_image()` 和 numpy 切片，每步打印点数变化，适合学生自学和课堂演示。 |

### 脚本调用关系

```
interactive_depth_pipeline.py          (用户入口 - 交互式全流程)
  ├── depth_sources.py                 (输入源选择：文件/USB/网络)
  ├── interactive_mask_ui.py           (OpenCV 交互窗口)
  ├── sam2_segmenter.py                (SAM2 分割)
  └── DepthMask_PointCloud_Pipeline.py (点云处理管道)
        └── Open3D_PointCloud_Pipeline.py (核心函数库)

PointCloud_Sandbox.py                  (独立运行 - 滤波参数探索)
  └── Open3D_PointCloud_Pipeline.py    (复用核心函数)
```

---

## data/ 目录分析

### data/bunny/ —— 斯坦福兔子数据集（**纯输入**）

原始扫描数据来自 Stanford Computer Graphics Laboratory 的 Stanford 3D Scanning Repository：<https://graphics.stanford.edu/data/3Dscanrep/>。这是三维重建和点云处理领域的经典 Stanford Bunny 模型，常用于 Open3D / PCL / 图形学教学示例，并不是 OpenCV 专属数据。

下载入口：

```bash
cd code/lecture06/Visual_Perception
mkdir -p data
curl -L -o data/bunny.tar.gz https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
tar -xzf data/bunny.tar.gz -C data
```

| 文件 | 类型 | 说明 |
|------|------|------|
| `data/bunny/data/bun.conf` / `bun.conf~` | 输入 | 各视角点云的配准变换配置文件 |
| `data/bunny/data/bun000.ply` ~ `bun315.ply` (8个) | 输入 | 8 个不同视角的原始距离扫描点云 |
| `data/bunny/data/chin.ply` | 输入 | 下巴局部扫描 |
| `data/bunny/data/ear_back.ply` | 输入 | 耳背局部扫描 |
| `data/bunny/data/top2.ply` / `top3.ply` | 输入 | 顶部局部扫描 |
| `data/bunny/data/README` | 元数据 | 官方说明文档 |
| `data/bunny/reconstruction/bun_zipper.ply` | 输入 | 高分辨率重建网格模型 |
| `data/bunny/reconstruction/bun_zipper_res2.ply` ~ `res4.ply` | 输入 | 降采样版重建网格（不同分辨率） |
| `data/bunny/reconstruction/README` | 元数据 | 官方说明文档 |

### data/rgbd_object_demo/ —— UW RGB-D Object Dataset 示例（**输入 + 输出混合**）

> ⚠️ 此目录用于存放从公开数据集整理出的示例输入，通常不随 Git 提交二进制文件。
> 推荐从 University of Washington RGB-D Object Dataset 下载 full 640×480 RGB-D 帧并转换成本目录格式。
> 下载页：<https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/>。可选择 `coffee_mug_1.tar`、`food_box_1.tar`、`bowl_1.tar` 等对象实例包；解压后取同一帧的 RGB、`*_depth.png` 和 `*_depthmask.png` / `*_mask.png`，分别重命名为 `rgb.png`、`depth.png`、`mask.png`，再手写 `intrinsics.json`。
> 也可以使用 `interactive_depth_pipeline.py` 的 USB / network 模式，或用 RealSense 等 RGB-D 相机采集自己的数据。

最小下载示例：

```bash
cd code/lecture06/Visual_Perception
mkdir -p data/raw_rgbd data/rgbd_object_demo
curl -L -o data/raw_rgbd/coffee_mug_1.tar \
  https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/coffee_mug_1.tar
tar -xf data/raw_rgbd/coffee_mug_1.tar -C data/raw_rgbd
```

用于演示"从 RGB-D 深度图和目标掩膜中提取目标点云"的完整链路。

#### 📥 输入数据

| 文件 | 说明 |
|------|------|
| `depth.png` | **16-bit 单通道深度图**，单位毫米，尺寸 640×480 |
| `mask.png` | **目标物体二值掩膜**，白色区域为目标 |
| `intrinsics.json` | **相机内参**：fx/fy/cx/cy + depth_scale |
| `rgb.png` | **与深度图对齐的 RGB 预览图**（当前示例使用预览图占位） |
| `depth_preview.png` | **深度预览图**（伪彩色/灰度，便于肉眼查看） |
| `network_manifest.json` | **网络输入模式清单文件**，指向同一目录下的 rgb/depth 和内参 |

#### 📤 输出数据（运行后生成到 `output/` 子目录）

| 相对路径 | 说明 |
|----------|------|
| `output/raw_scene_from_depth.pcd` | 整幅深度图反投影得到的**完整场景点云** |
| `output/masked_target_raw.pcd` | 掩膜提取的**目标物体原始点云**（未经滤波） |
| `output/masked_target_clean.pcd` | 经直通+体素+统计滤波后的**目标干净点云**（最终成果） |
| `output/mask_overlay_preview.png` | 掩膜叠加在预览图上的**可视化预览图** |
| `output/screenshots/00_scene_raw.png` ~ `03_target_clean.png` | 各阶段截图 |
| `output/screenshots/04_pipeline_comparison.png` | 四阶段并排对比图 |
| `output/screenshots/05_pipeline_montage_2x2.png` | 四张截图 2×2 拼图 |
| `output/syllabus_masked_depth.png` | 课堂 5 步法第 1 步输出：mask 筛选后的深度图 |
| `output/syllabus_target_raw.pcd` | 课堂 5 步法第 2 步输出：反投影后的目标原始点云 |
| `output/syllabus_target_passthrough.pcd` | 课堂 5 步法第 3 步输出：直通滤波后的目标点云 |
| `output/syllabus_target_statistical.pcd` | 课堂 5 步法第 4 步中间输出：统计滤波后的目标点云 |
| `output/syllabus_target_clean.pcd` | 课堂 5 步法第 5 步输出：最终清洗点云 |
| `output/syllabus_pipeline_summary.json` | 课堂 5 步法运行摘要：输入路径、参数、像素统计、点数变化和 XYZ 范围 |
| `output/screenshots_syllabus/` | 课堂 5 步法各阶段截图与 2×2 拼图 |

### 其他目录 / 文件

| 路径 | 类型 | 说明 |
|------|------|------|
| `data/output/` | 输出 | `Open3D_PointCloud_Pipeline.py` 独立运行时的输出目录（`.pcd` + `screenshots/`） |

---

## 快速开始

### 1. 安装依赖

```bash
pip install numpy open3d Pillow opencv-python
# USB 相机模式：pip install pyrealsense2
# SAM2 真实分割：pip install torch sam2
```

如果默认镜像下载 `open3d` 失败，可以临时改用清华 PyPI 镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy open3d Pillow opencv-python
# USB 相机模式：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyrealsense2
# SAM2 真实分割：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch sam2
```

也可以把清华镜像设为当前 Python 环境的默认源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 运行 Open3D 点云预处理（教学示例 — 斯坦福兔子）

```bash
python Open3D_PointCloud_Pipeline.py --no-vis
```

### 3. 运行深度图+掩膜管道（合成桌面场景）

```bash
python DepthMask_PointCloud_Pipeline.py --no-vis
```

### 4. 运行交互式管道（单击/框选分割目标）

```bash
python interactive_depth_pipeline.py --source file --input-dir data/rgbd_object_demo --output-dir data/rgbd_object_demo
```

### 5. 运行交互式滤波沙盒

```bash
python PointCloud_Sandbox.py
```

### 6. 运行课堂 5 步法独立脚本

```bash
python pipeline_from_syllabus.py --demo-dir data/rgbd_object_demo --no-vis
```

不带 `--no-vis` 会在最终弹出 3D 窗口显示清洗后的目标点云。

### 7. 最小自检命令（推荐先跑）

无头服务器或远程终端中，建议先跳过截图和窗口，只验证核心几何链路是否能跑通：

```bash
python pipeline_from_syllabus.py \
  --demo-dir data/rgbd_object_demo \
  --no-vis \
  --skip-screenshots
```

成功后重点查看两个文件：

| 文件 | 用途 |
|---|---|
| `data/rgbd_object_demo/output/syllabus_target_clean.pcd` | 第 4 讲/第 6 讲后续位姿估计可读取的目标点云 |
| `data/rgbd_object_demo/output/syllabus_pipeline_summary.json` | 输入路径、相机内参、滤波参数、像素统计、各阶段点数和 XYZ 范围 |

如果运行失败，优先根据报错检查三件事：`mask.png` 是否有前景、`depth.png` 在 mask 区域是否有有效深度、`intrinsics.json` 中的 `fx/fy/cx/cy/depth_scale` 是否与当前图像分辨率和单位一致。

---

## 数据流总览

```
                         ┌──────────────────┐
                         │   depth_sources   │  ← File / USB / Network
                         └────────┬─────────┘
                                  │ DepthFrame(rgb, depth, intrinsics)
                                  ▼
┌──────────────────┐   ┌──────────────────┐
│ interactive_mask │   │  sam2_segmenter  │
│      _ui.py      │──▶│                  │  ← point / box prompt
│ (OpenCV 交互)    │   │ (SAM2 分割包装)   │
└──────────────────┘   └────────┬─────────┘
                               │ mask
                               ▼
                  ┌────────────────────────────┐
                  │ DepthMask_PointCloud_      │
                  │      Pipeline.py           │
                  │  ┌──────────────────────┐  │
                  │  │ 反投影 → 场景点云      │  │
                  │  │ 掩膜 → 目标点云        │  │
                  │  │ 直通滤波              │  │
                  │  │ 体素降采样            │  │
                  │  │ 统计滤波去噪          │  │
                  │  └──────────────────────┘  │
                  └───────────┬────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  点云 .pcd 文件   │
                    │  截图 / 拼图     │
                    └──────────────────┘

         ┌───────────────────────────────┐
         │   PointCloud_Sandbox.py       │
         │   (独立沙盒，实时参数调优)      │
         └───────────────────────────────┘
```

---

## 滤波参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `z_min` / `z_max` | 0.20 / 1.60 m | 直通滤波：保留指定深度范围内的点，剔除过近/过远噪点 |
| `voxel_size` | 0.01 m | 体素降采样：每个体素内只保留一个点，减少数据量 |
| `nb_neighbors` | 20 | 统计滤波：检查每个点最近的 K 个邻居 |
| `std_ratio` | 1.5 | 统计滤波：标准差倍数阈值，剔除离群飞点 |

课堂 5 步法脚本会在启动时检查这些参数：`z_min` 必须小于 `z_max`，`voxel_size` 不能为负，`nb_neighbors` 至少为 1，`std_ratio` 必须为正数。这样可以避免参数写错后仍然生成一个看似正常、实际不可用的点云。

---

## 输出文件命名约定

| 命名模式 | 含义 |
|----------|------|
| `*_raw.pcd` / `*_scene_from_depth.pcd` | 原始反投影点云（未滤波） |
| `*_passthrough.*` | 直通滤波后结果 |
| `*_downsampled.*` | 体素降采样后结果 |
| `*_filtered.pcd` / `*_clean.pcd` | 最终清洗后点云 |
| `*_comparison.png` | 多阶段并排对比图 |
| `*_montage_2x2.png` | 四阶段 2×2 拼图 |
| `*_overlay_preview.png` | 掩膜叠加预览图 |
