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

相比其他脚本，该脚本使用 **Open3D 内置 `create_from_depth_image()`** 进行反投影（大纲推荐方式），并使用 **numpy 数组切片** 做直通滤波，每步打印详细点数变化，适合课堂演示和学生自学。

```
深度图 + 相机内参 ──→ 反投影为场景点云 ──→ 掩膜提取目标 ──→ 直通滤波 ──→ 体素降采样 ──→ 统计滤波去噪 ──→ 干净点云
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

原始扫描数据，来自 [Stanford Computer Graphics Laboratory](http://www-graphics.stanford.edu)。

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

### data/clutter_depth_demo/ —— 杂乱桌面示例（**输入 + 输出混合**）

> ⚠️ 此目录为**程序合成**的演示数据，**不是从任何外部网站下载的**。
> 它由项目作者预先制作，包含一个中心目标盒体、左侧杯状物、右侧书本、后方小盒子及桌面深度渐变。
> 如需自己生成此类数据，可使用 Blender + Python 渲染深度图，或使用 RealSense 相机采集。

用于演示"从杂乱场景中提取目标点云"的完整链路。

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
python interactive_depth_pipeline.py --source file --input-dir data/clutter_depth_demo --output-dir data/clutter_depth_demo
```

### 5. 运行交互式滤波沙盒

```bash
python PointCloud_Sandbox.py
```

### 6. 运行课堂 5 步法独立脚本

```bash
python pipeline_from_syllabus.py --demo-dir data/clutter_depth_demo --no-vis
```

不带 `--no-vis` 会在最终弹出 3D 窗口显示清洗后的目标点云。

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
