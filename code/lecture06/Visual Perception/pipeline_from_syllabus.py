# pipeline_from_syllabus.py
# ─────────────────────────────────────────────────────────────
# 课堂配套脚本：SAM 掩膜 → 深度图 → 干净目标点云（5 步法）
# ─────────────────────────────────────────────────────────────
# 对应教学大纲：
#   步骤 1 ─ 掩膜筛选：    depth × mask → masked_depth
#   步骤 2 ─ 反投影生成：  利用相机内参 K，将过滤后的深度图映射为 3D 坐标 (X, Y, Z)
#   步骤 3 ─ 空间粗裁：    直通滤波，切除 Z < 0（桌面以下）或距离过远的无效空间
#   步骤 4 ─ 精洗去噪：    统计滤波（剔除孤立飞点）→ 体素降采样（网格合并，完成瘦身）
#   步骤 5 ─ 目标点云提取：导出最终 .pcd 文件
#
# 本脚本特点：
#   - 完全独立，不依赖其他项目模块
#   - 每步都有明确标号和中英文注释
#   - 使用 Open3D 内置 create_from_depth_image（大纲推荐方式）
#   - 使用 numpy 数组切片做直通滤波（大纲推荐方式）
#   - 打印每步前后点数变化，方便课堂观察
#
# 用法：
#   python pipeline_from_syllabus.py
#   python pipeline_from_syllabus.py --demo-dir data/clutter_depth_demo --no-vis
#   python pipeline_from_syllabus.py --demo-dir data/clutter_depth_demo
#
# 依赖：numpy, open3d, Pillow
# ─────────────────────────────────────────────────────────────

import argparse
import copy
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import open3d as o3d
except ModuleNotFoundError:
    raise SystemExit(
        "未安装 open3d，请先执行 `pip install open3d` "
        "或在已安装 Open3D 的 Python 环境中运行。"
    )


REQUIRED_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")


# ============================================================
# 辅助函数：加载数据
# ============================================================

def load_depth(path: str) -> np.ndarray:
    """加载 16-bit 单通道深度图（单位：毫米）"""
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError("深度图必须是单通道 16-bit 图像。")
    return arr


def load_mask(path: str, expected_shape) -> np.ndarray:
    """加载二值掩膜，返回 bool 数组"""
    arr = np.asarray(Image.open(path).convert("L"))
    if arr.shape != expected_shape:
        raise ValueError(
            f"掩膜尺寸 {arr.shape} 与深度图尺寸 {expected_shape} 不一致。"
        )
    return arr > 0  # 转为 bool


def load_intrinsics(path: str) -> dict:
    """加载相机内参 JSON"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_rgb(path: str) -> np.ndarray:
    """加载 RGB 预览图（用于给点云上色）"""
    return np.asarray(Image.open(path).convert("RGB"))


def validate_intrinsics(intrinsics: dict, image_shape, depth_scale: float) -> dict:
    """检查相机内参与深度尺度，避免静默生成错误尺度的点云。"""
    missing = [key for key in REQUIRED_INTRINSIC_KEYS if key not in intrinsics]
    if missing:
        raise KeyError(f"intrinsics.json 缺少字段: {', '.join(missing)}")

    height, width = image_shape
    checked = {}
    for key in REQUIRED_INTRINSIC_KEYS:
        value = float(intrinsics[key])
        if not np.isfinite(value):
            raise ValueError(f"相机内参 {key} 不是有限数值: {value}")
        checked[key] = value

    if checked["fx"] <= 0 or checked["fy"] <= 0:
        raise ValueError("fx/fy 必须为正数。")
    if depth_scale <= 0 or not np.isfinite(depth_scale):
        raise ValueError("depth_scale 必须为正数。")

    warnings = []
    if not (0 <= checked["cx"] < width):
        warnings.append(f"cx={checked['cx']} 超出图像宽度范围 [0, {width})")
    if not (0 <= checked["cy"] < height):
        warnings.append(f"cy={checked['cy']} 超出图像高度范围 [0, {height})")
    if warnings:
        print("  ⚠️ 相机内参检查提示：")
        for item in warnings:
            print(f"    - {item}")

    return checked


def validate_filter_params(z_min: float, z_max: float, voxel_size: float, nb_neighbors: int, std_ratio: float) -> None:
    """检查滤波参数是否处在可执行范围内。"""
    if z_min >= z_max:
        raise ValueError(f"z_min 必须小于 z_max，当前为 {z_min} >= {z_max}。")
    if voxel_size < 0:
        raise ValueError("voxel_size 不能为负数。")
    if nb_neighbors < 1:
        raise ValueError("nb_neighbors 至少为 1。")
    if std_ratio <= 0:
        raise ValueError("std_ratio 必须为正数。")


def save_depth(path: str, depth_mm: np.ndarray) -> None:
    """保存 16-bit 深度图，便于检查 mask 筛选后的深度区域。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_uint16 = np.asarray(depth_mm)
    if depth_uint16.dtype != np.uint16:
        depth_uint16 = np.clip(depth_uint16, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(depth_uint16, mode="I;16").save(path)


def summarize_point_cloud(pcd: o3d.geometry.PointCloud) -> dict:
    """返回点云点数和 XYZ 范围，用于生成可复现实验摘要。"""
    points = np.asarray(pcd.points)
    summary = {"points": int(len(points))}
    if len(points) == 0:
        summary["bounds_m"] = None
        return summary

    summary["bounds_m"] = {
        "x": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y": [float(points[:, 1].min()), float(points[:, 1].max())],
        "z": [float(points[:, 2].min()), float(points[:, 2].max())],
    }
    return summary


def save_json(path: str, data: dict) -> None:
    """保存 JSON 文件，记录本次实验的输入、参数、点数和输出路径。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


# ============================================================
# 可视化辅助函数（截图、并排对比、拼图）
# ============================================================

def _save_screenshot(geometries, image_path, window_name, width=1280, height=720):
    """将当前几何体渲染结果保存为 PNG 截图"""
    non_empty = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.PointCloud) and len(geometry.points) == 0:
            continue
        non_empty.append(geometry)
    if not non_empty:
        print(f"  ⚠️ 跳过截图，当前阶段没有可渲染几何体: {window_name}")
        return False

    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name=window_name, width=width, height=height, visible=False)
        for geometry in non_empty:
            vis.add_geometry(geometry)
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])
        render_option.point_size = 3.0
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path, do_render=True)
        print(f"  🖼️ 截图已保存: {image_path}")
        return True
    except Exception as exc:
        print(f"  ⚠️ 截图保存失败（可能不支持离屏渲染）: {exc}")
        return False
    finally:
        vis.destroy_window()


COMPARISON_COLORS = [
    [0.45, 0.45, 0.45],  # 原始点云：灰色
    [0.90, 0.30, 0.30],  # 阶段 1：红色
    [0.20, 0.65, 0.20],  # 阶段 2：绿色
    [0.20, 0.45, 0.95],  # 阶段 3：蓝色
    [0.80, 0.55, 0.15],  # 额外阶段：橙色
]


def _build_colored_comparison(stage_items):
    """将不同阶段染成不同颜色，并沿 X 轴平移后用于并排对比显示"""
    comparison_geometries = []
    legend_lines = []
    offset_x = 0.0

    for index, (stage_name, pcd) in enumerate(stage_items):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        colored_pcd = copy.deepcopy(pcd)
        colored_pcd.paint_uniform_color(color)

        if len(colored_pcd.points) == 0:
            legend_lines.append(f"{stage_name} -> RGB={color}（空点云）")
            continue

        bbox = colored_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = max(float(extent[0]), float(extent[1]), float(extent[2]), 0.05)

        colored_pcd.translate((offset_x - center[0], -center[1], -center[2]))
        comparison_geometries.append(colored_pcd)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max_extent * 0.25,
            origin=[offset_x, 0.0, 0.0]
        )
        comparison_geometries.append(axis)

        legend_lines.append(f"{stage_name} -> RGB={color}")
        offset_x += max_extent * 1.8

    return comparison_geometries, legend_lines


def save_screenshots_and_comparison(stage_items, screenshots_dir, montage_path):
    """保存各阶段截图 + 并排对比图 + 2×2 拼图"""
    os.makedirs(screenshots_dir, exist_ok=True)

    # ── 各阶段截图 ──
    image_paths = []
    for i, (stage_name, pcd) in enumerate(stage_items):
        fname = f"stage_{i:02d}_{stage_name}.png".replace(" ", "_")
        fpath = os.path.join(screenshots_dir, fname)
        _save_screenshot([pcd], fpath, stage_name)
        image_paths.append((stage_name, fpath))

    # ── 并排对比图（染色后同一张图内对比） ──
    comparison_path = os.path.join(screenshots_dir, "comparison_all.png")
    comparison_geoms, legend_lines = _build_colored_comparison(stage_items)

    print("\n  并排对比颜色说明：")
    for line in legend_lines:
        print(f"    - {line}")

    _save_screenshot(
        comparison_geoms,
        comparison_path,
        "多阶段并排对比",
        width=1600,
        height=900,
    )

    # ── 2×2 拼图（仅前 4 个阶段） ──
    if len(image_paths) >= 4:
        _create_montage_2x2(image_paths[:4], montage_path)
    else:
        print("  ⚠️ 阶段数不足 4，跳过 2×2 拼图。")

    return comparison_path


def _create_montage_2x2(image_items, output_path, tile_width=900, tile_height=520):
    """将前 4 张阶段截图拼成 2x2 大图"""
    if len(image_items) != 4:
        print("  ⚠️ 仅支持 4 张图片的 2x2 拼图，已跳过。")
        return

    margin = 30
    title_height = 48
    border_width = 2

    tiles = []
    for title, image_path in image_items:
        if not os.path.exists(image_path):
            print(f"  ⚠️ 拼图跳过，未找到截图: {image_path}")
            return

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = ImageOps.contain(img, (tile_width, tile_height))

            tile = Image.new("RGB", (tile_width, tile_height + title_height), "white")
            draw = ImageDraw.Draw(tile)
            draw.text((18, 14), title, fill=(0, 0, 0))

            paste_x = (tile_width - img.width) // 2
            paste_y = title_height + (tile_height - img.height) // 2
            tile.paste(img, (paste_x, paste_y))

        tile = ImageOps.expand(tile, border=border_width, fill=(180, 180, 180))
        tiles.append(tile)

    canvas_width = tiles[0].width * 2 + margin * 3
    canvas_height = tiles[0].height * 2 + margin * 3
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    positions = [
        (margin, margin),
        (margin * 2 + tiles[0].width, margin),
        (margin, margin * 2 + tiles[0].height),
        (margin * 2 + tiles[0].width, margin * 2 + tiles[0].height),
    ]
    for tile, pos in zip(tiles, positions):
        canvas.paste(tile, pos)

    canvas.save(output_path)
    print(f"  🧩 2x2 拼图已保存: {output_path}")


# ============================================================
# 步骤 1 ── 掩膜筛选
# ============================================================

def step1_mask_filtering(depth_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    步骤 1：掩膜筛选
    ─────────────────
    将深度图与 Mask 相乘，把非目标像素的深度值置为 0。
    这样后续反投影时，只有目标区域会生成 3D 点。
    """
    print("=" * 60)
    print("步骤 1：掩膜筛选 ── depth × mask → masked_depth")
    print("=" * 60)

    total_pixels = depth_mm.size
    foreground_pixels = int(mask.sum())
    ratio = foreground_pixels / total_pixels * 100

    print(f"  深度图总像素数:  {total_pixels:,}")
    print(f"  掩膜前景像素数:  {foreground_pixels:,} ({ratio:.1f}%)")

    # 核心操作：逐元素乘法，掩膜外区域深度变为 0
    masked_depth = depth_mm * mask

    nonzero_before = np.count_nonzero(depth_mm)
    nonzero_after = np.count_nonzero(masked_depth)
    print(f"  掩膜前有效深度像素: {nonzero_before:,}")
    print(f"  掩膜后有效深度像素: {nonzero_after:,} (减少了 {nonzero_before - nonzero_after:,})")
    if foreground_pixels == 0:
        raise ValueError("mask 中没有前景像素，无法生成目标点云。")
    if nonzero_after == 0:
        raise ValueError("mask 区域内没有有效深度，请检查 RGB-D 对齐、深度单位或 mask 位置。")
    print()

    return masked_depth


# ============================================================
# 步骤 2 ── 反投影生成（深度图 → 3D 点云）
# ============================================================

def step2_backprojection(
    masked_depth: np.ndarray,
    intrinsics: dict,
    depth_scale: float = 1000.0,
    rgb_image: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """
    步骤 2：反投影生成
    ─────────────────
    利用相机内参矩阵 K，将过滤后的深度图映射为 3D 坐标 (X, Y, Z)。

    使用 Open3D 内置的 create_from_depth_image() 函数，
    该函数内部完成：
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth(u, v) / depth_scale

    参数:
        masked_depth:  步骤 1 输出的掩膜过滤深度图（单位：毫米）
        intrinsics:    相机内参字典 {fx, fy, cx, cy}
        depth_scale:   深度图单位转换因子（毫米→米，默认 1000.0）
        rgb_image:     可选的 RGB 图，用于给点云上色
    """
    print("=" * 60)
    print("步骤 2：反投影生成 ── 深度图 → 3D 点云 (X, Y, Z)")
    print("=" * 60)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    print(f"  相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"  depth_scale = {depth_scale} (毫米→米)")

    if rgb_image is not None and rgb_image.shape[:2] != masked_depth.shape:
        raise ValueError(
            f"RGB 尺寸 {rgb_image.shape[:2]} 与深度图尺寸 {masked_depth.shape} 不一致，"
            "请先完成 RGB-D 对齐或不要给点云上色。"
        )

    # ── 构造 Open3D 深度图像对象 ──
    depth_m = masked_depth.astype(np.float32) / depth_scale
    o3d_depth = o3d.geometry.Image(depth_m)

    # ── 构造 Open3D 相机内参对象 ──
    height, width = masked_depth.shape
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )

    # ── 反投影：Open3D 内置函数 ──
    # depth_scale=1.0 因为我们已经手动转成了米
    # depth_trunc 设为一个较大值，避免截断
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d_depth,
        intrinsic=o3d_intrinsics,
        extrinsic=np.eye(4),  # 单位矩阵，不额外旋转/平移
        depth_scale=1.0,       # 深度值已为米
        depth_trunc=100.0,     # 不截断
        stride=1,              # 不跳像素
    )

    # ── 点云着色（如有 RGB 图） ──
    if rgb_image is not None:
        rgb_o3d = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d,
            depth=o3d_depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )
        pcd_rgb = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd,
            intrinsic=o3d_intrinsics,
            extrinsic=np.eye(4),
        )
        pcd.colors = pcd_rgb.colors

    point_count = len(pcd.points)
    print(f"  反投影生成点数: {point_count:,}")
    print()

    return pcd


# ============================================================
# 步骤 3 ── 空间粗裁（直通滤波）
# ============================================================

def step3_passthrough(
    pcd: o3d.geometry.PointCloud,
    z_min: float = 0.20,
    z_max: float = 1.50,
) -> o3d.geometry.PointCloud:
    """
    步骤 3：空间粗裁（直通滤波）
    ──────────────────────────
    使用 numpy 数组切片，移除 Z 轴小于 z_min（桌面以下）或
    大于 z_max（距离过远）的点。

    大纲要求：用 numpy 数组切片实现。
    """
    print("=" * 60)
    print(f"步骤 3：空间粗裁（直通滤波）── {z_min}m ≤ Z ≤ {z_max}m")
    print("=" * 60)

    points = np.asarray(pcd.points)
    before = len(points)

    if before == 0:
        print("  ⚠️ 点云为空，跳过直通滤波。")
        return o3d.geometry.PointCloud()

    # ── numpy 布尔切片：核心操作 ──
    z = points[:, 2]
    mask = (z >= z_min) & (z <= z_max)

    cropped_points = points[mask]
    after = len(cropped_points)

    print(f"  滤波前点数: {before:,}")
    print(f"  滤波后点数: {after:,}")
    print(f"  剔除点数:   {before - after:,} ({(before - after) / before * 100:.1f}%)")
    print(f"  Z 范围:     [{z.min():.3f}, {z.max():.3f}] m")

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(cropped_points)

    # 保留颜色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd_out.colors = o3d.utility.Vector3dVector(colors[mask])

    print()

    return pcd_out


# ============================================================
# 步骤 4 ── 精洗去噪（统计滤波 + 体素降采样）
# ============================================================

def step4_denoise_and_downsample(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    voxel_size: float = 0.005,
) -> dict:
    """
    步骤 4：精洗去噪
    ──────────────
    大纲要求：
      (a) 体素降采样：调用 pcd.voxel_down_sample(voxel_size=0.005)，
          观察点数从 30 万降到几千。
      (b) 统计滤波去噪：调用 pcd.remove_statistical_outlier(
              nb_neighbors=20, std_ratio=2.0)，剔除离群点。

    注意：大纲中先体素再统计，但实际工程中通常先统计滤波再体素降采样，
    因为先降采样会丢失小簇离群点的几何信息，降低统计滤波效果。
    这里提供两种顺序，默认采用"先统计、后体素"的工程实践顺序，
    并在输出中同时报告另一种顺序供课堂对比。
    """
    print("=" * 60)
    print("步骤 4：精洗去噪")
    print("  (a) 统计滤波 ── 计算平均距离，剔除孤立飞点")
    print(f"      参数: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    print("  (b) 体素降采样 ── 按网格合并相近点，完成瘦身")
    print(f"      参数: voxel_size={voxel_size} m")
    print("=" * 60)

    before_all = len(pcd.points)
    if before_all == 0:
        print("  ⚠️ 输入点云为空，跳过统计滤波和体素降采样。")
        empty = o3d.geometry.PointCloud()
        return {
            "statistical": empty,
            "downsampled": empty,
            "count_before": 0,
            "count_after_stat": 0,
            "count_after_voxel": 0,
        }

    # ── (a) 先执行统计滤波（剔除离群飞点） ──
    print(f"\n  ┌─ (a) 统计滤波去噪 ──")
    print(f"  │  输入点数: {len(pcd.points):,}")

    if len(pcd.points) < nb_neighbors + 1:
        print("  │  ⚠️ 点数不足，跳过统计滤波。")
        pcd_stat = pcd
    else:
        pcd_stat, outlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors),
            std_ratio=float(std_ratio),
        )

    n_stat = len(pcd_stat.points)
    print(f"  │  输出点数: {n_stat:,}")
    print(f"  │  剔除飞点: {len(pcd.points) - n_stat:,} "
          f"({(len(pcd.points) - n_stat) / len(pcd.points) * 100:.1f}%)")

    # ── (b) 再执行体素降采样（瘦身） ──
    print(f"\n  └─ (b) 体素降采样 ──")
    print(f"     输入点数: {len(pcd_stat.points):,}")
    print(f"     体素尺寸: {voxel_size} m")

    if voxel_size is None or voxel_size <= 0 or len(pcd_stat.points) == 0:
        pcd_voxel = pcd_stat
    else:
        pcd_voxel = pcd_stat.voxel_down_sample(voxel_size=float(voxel_size))

    n_voxel = len(pcd_voxel.points)
    print(f"     输出点数: {n_voxel:,}")
    if len(pcd_stat.points) > 0:
        print(f"     压缩比:   {n_voxel / len(pcd_stat.points) * 100:.2f}%")
    print()

    # ── 总览 ──
    after_all = n_voxel
    print(f"  ★ 步骤 4 总览: {before_all:,} → {after_all:,} 点 "
          f"(减少了 {before_all - after_all:,}, "
          f"保留 {after_all / before_all * 100:.2f}%)")
    print()

    return {
        "statistical": pcd_stat,
        "downsampled": pcd_voxel,
        "count_before": before_all,
        "count_after_stat": n_stat,
        "count_after_voxel": n_voxel,
    }


# ============================================================
# 步骤 5 ── 目标点云提取与导出
# ============================================================

def step5_export(
    pcd_clean: o3d.geometry.PointCloud,
    output_path: str,
) -> None:
    """
    步骤 5：目标点云提取
    ─────────────────
    将清洗后的干净目标点云导出为 .pcd 文件，
    用于下一讲的位姿估计。
    """
    print("=" * 60)
    print("步骤 5：目标点云提取 ── 导出 .pcd 文件")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    o3d.io.write_point_cloud(output_path, pcd_clean)
    print(f"  ✅ 已保存: {output_path}")
    print(f"  最终点数: {len(pcd_clean.points):,}")
    print()

    # 打印点云空间范围
    if len(pcd_clean.points) > 0:
        points = np.asarray(pcd_clean.points)
        print(f"  空间范围:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")


# ============================================================
# 一键运行：完整的 5 步流水线
# ============================================================

def run_syllabus_pipeline(
    demo_dir: str,
    z_min: float = 0.20,
    z_max: float = 1.50,
    voxel_size: float = 0.005,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    show_window: bool = True,
    skip_screenshots: bool = False,
) -> dict:
    """
    按课堂大纲顺序执行完整的"掩膜→深度→点云"5 步流水线。
    返回每步的中间结果，方便课堂分析和调试。
    """
    validate_filter_params(z_min, z_max, voxel_size, nb_neighbors, std_ratio)

    # ── 路径准备 ──
    rgb_path = os.path.join(demo_dir, "rgb.png")
    depth_path = os.path.join(demo_dir, "depth.png")
    mask_path = os.path.join(demo_dir, "mask.png")
    intrinsics_path = os.path.join(demo_dir, "intrinsics.json")
    output_dir = os.path.join(demo_dir, "output")
    masked_depth_path = os.path.join(output_dir, "syllabus_masked_depth.png")
    raw_pcd_path = os.path.join(output_dir, "syllabus_target_raw.pcd")
    passthrough_pcd_path = os.path.join(output_dir, "syllabus_target_passthrough.pcd")
    statistical_pcd_path = os.path.join(output_dir, "syllabus_target_statistical.pcd")
    output_pcd = os.path.join(output_dir, "syllabus_target_clean.pcd")
    summary_path = os.path.join(output_dir, "syllabus_pipeline_summary.json")
    screenshots_dir = os.path.join(output_dir, "screenshots_syllabus")
    montage_path = os.path.join(screenshots_dir, "montage_2x2.png")

    print("\n" + "█" * 60)
    print("  深度掩膜 → 点云清洗流水线（课堂 5 步法）")
    print("█" * 60 + "\n")

    # ── 加载数据 ──
    print("▶ 加载数据 ...")
    depth_mm = load_depth(depth_path)
    mask = load_mask(mask_path, depth_mm.shape)
    intrinsics = load_intrinsics(intrinsics_path)
    depth_scale = float(intrinsics.get("depth_scale", 1000.0))
    intrinsics_checked = validate_intrinsics(intrinsics, depth_mm.shape, depth_scale)

    rgb_image = None
    if os.path.exists(rgb_path):
        rgb_image = load_rgb(rgb_path)

    print(f"  深度图:       {depth_path}")
    print(f"  尺寸:         {depth_mm.shape[1]} × {depth_mm.shape[0]}")
    print(f"  掩膜:         {mask_path}")
    print(f"  相机内参:     {intrinsics_path}")
    print(f"  深度缩放:     {depth_scale} (毫米→米)")
    print()

    # ── 步骤 1：掩膜筛选 ──
    masked_depth = step1_mask_filtering(depth_mm, mask)
    save_depth(masked_depth_path, masked_depth)
    print(f"  掩膜筛选深度图已保存: {masked_depth_path}\n")

    # ── 步骤 2：反投影生成 ──
    pcd_raw = step2_backprojection(masked_depth, intrinsics_checked, depth_scale, rgb_image)
    o3d.io.write_point_cloud(raw_pcd_path, pcd_raw)

    # ── 步骤 3：空间粗裁（直通滤波） ──
    pcd_passthrough = step3_passthrough(pcd_raw, z_min=z_min, z_max=z_max)
    o3d.io.write_point_cloud(passthrough_pcd_path, pcd_passthrough)

    # ── 步骤 4：精洗去噪（统计 + 体素） ──
    denoise_result = step4_denoise_and_downsample(
        pcd_passthrough,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
        voxel_size=voxel_size,
    )
    pcd_clean = denoise_result["downsampled"]
    o3d.io.write_point_cloud(statistical_pcd_path, denoise_result["statistical"])

    # ── 步骤 5：目标点云提取与导出 ──
    step5_export(pcd_clean, output_pcd)

    output_paths = {
        "masked_depth": masked_depth_path,
        "raw_pcd": raw_pcd_path,
        "passthrough_pcd": passthrough_pcd_path,
        "statistical_pcd": statistical_pcd_path,
        "clean_pcd": output_pcd,
        "summary_json": summary_path,
        "screenshots_dir": screenshots_dir,
        "montage_2x2": montage_path,
    }

    summary = {
        "inputs": {
            "rgb": rgb_path if os.path.exists(rgb_path) else None,
            "depth": depth_path,
            "mask": mask_path,
            "intrinsics": intrinsics_path,
        },
        "parameters": {
            "depth_scale": depth_scale,
            "z_min": z_min,
            "z_max": z_max,
            "voxel_size": voxel_size,
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio,
        },
        "pixel_counts": {
            "depth_total": int(depth_mm.size),
            "mask_foreground": int(mask.sum()),
            "depth_nonzero_before_mask": int(np.count_nonzero(depth_mm)),
            "depth_nonzero_after_mask": int(np.count_nonzero(masked_depth)),
        },
        "point_clouds": {
            "raw": summarize_point_cloud(pcd_raw),
            "passthrough": summarize_point_cloud(pcd_passthrough),
            "statistical": summarize_point_cloud(denoise_result["statistical"]),
            "clean": summarize_point_cloud(pcd_clean),
        },
        "sanity_checks": {
            "mask_depth_coverage": float(np.count_nonzero(masked_depth) / max(int(mask.sum()), 1)),
            "clean_retention_from_raw": float(len(pcd_clean.points) / max(len(pcd_raw.points), 1)),
            "rgb_depth_shape_aligned": bool(rgb_image is None or rgb_image.shape[:2] == depth_mm.shape),
        },
        "outputs": output_paths,
    }
    save_json(summary_path, summary)
    print(f"  流水线摘要已保存: {summary_path}\n")

    # ── 保存各阶段截图 + 并排对比 + 2×2 拼图 ──
    if not skip_screenshots:
        stage_items = [
            ("02-反投影原始点云", pcd_raw),
            ("03-直通滤波后", pcd_passthrough),
            ("04a-统计滤波后", denoise_result["statistical"]),
            ("04b-体素降采样后", pcd_clean),
        ]
        save_screenshots_and_comparison(stage_items, screenshots_dir, montage_path)
        print(f"  📁 阶段截图目录: {screenshots_dir}")

    # ── 可视化（可选） ──
    if show_window and len(pcd_clean.points) > 0:
        print("▶ 显示最终清洗结果（关闭窗口后结束）...")
        o3d.visualization.draw_geometries(
            [pcd_clean],
            window_name="步骤 5：清洗后的目标点云",
            width=1280,
            height=720,
        )

    # ── 返回所有中间结果 ──
    return {
        "depth_mm": depth_mm,
        "mask": mask,
        "masked_depth": masked_depth,
        "pcd_raw": pcd_raw,
        "pcd_passthrough": pcd_passthrough,
        "pcd_statistical": denoise_result["statistical"],
        "pcd_clean": pcd_clean,
        "output_pcd": output_pcd,
        "output_paths": output_paths,
        "summary_path": summary_path,
        "counts": {
            "raw": len(pcd_raw.points),
            "passthrough": len(pcd_passthrough.points),
            "statistical": denoise_result["count_after_stat"],
            "clean": denoise_result["count_after_voxel"],
        },
    }


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="课堂 5 步法：SAM 掩膜 → 深度图 → 干净目标点云"
    )
    default_demo = os.path.join(
        os.path.dirname(__file__), "data", "clutter_depth_demo"
    )
    parser.add_argument("--demo-dir", default=default_demo, help="示例数据目录")
    parser.add_argument("--z-min", type=float, default=0.20, help="直通滤波 Z 最小值（米）")
    parser.add_argument("--z-max", type=float, default=1.50, help="直通滤波 Z 最大值（米）")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="体素降采样尺寸（米）")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="统计滤波近邻数")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="统计滤波标准差倍数")
    parser.add_argument("--no-vis", action="store_true", help="不弹出 3D 可视化窗口")
    parser.add_argument("--skip-screenshots", action="store_true", help="跳过阶段截图与拼图")
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查必要文件
    required = ["depth.png", "mask.png", "intrinsics.json"]
    for fname in required:
        fpath = os.path.join(args.demo_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"缺少必要文件: {fpath}\n"
                f"请确认 --demo-dir 指向包含 {', '.join(required)} 的目录。"
            )

    result = run_syllabus_pipeline(
        demo_dir=args.demo_dir,
        z_min=args.z_min,
        z_max=args.z_max,
        voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        show_window=not args.no_vis,
        skip_screenshots=args.skip_screenshots,
    )

    # 打印最终汇总
    print("\n" + "=" * 60)
    print("  流水线完成！点数变化汇总")
    print("=" * 60)
    c = result["counts"]
    print(f"  步骤 2（反投影）:        {c['raw']:>8,} 点")
    print(f"  步骤 3（直通滤波）:      {c['passthrough']:>8,} 点")
    print(f"  步骤 4a（统计滤波）:     {c['statistical']:>8,} 点")
    print(f"  步骤 4b（体素降采样）:   {c['clean']:>8,} 点")
    print(f"  ─────────────────────────────────────")
    print(f"  压缩率: {c['clean'] / max(c['raw'], 1) * 100:.2f}%")
    print(f"  最终输出: {result['output_pcd']}")
    print(f"  阶段摘要: {result['summary_path']}")
    print(f"  阶段截图: {result['output_paths']['screenshots_dir']}")
    print()


if __name__ == "__main__":
    main()
