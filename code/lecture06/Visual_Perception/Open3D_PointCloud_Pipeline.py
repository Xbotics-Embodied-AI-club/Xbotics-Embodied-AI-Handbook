# Open3D_PointCloud_Pipeline.py
# 功能：Open3D 点云预处理流水线（核心库 + 独立教学脚本）
# - 点云基础操作：读取 .ply/.pcd、保存、统计点数
# - 深度图 → 点云反投影（针孔相机模型）
# - 预处理三步：直通滤波 → 体素降采样 → 统计滤波去噪
# - 可视化工具：单阶段显示、多阶段并排对比、2×2 拼图
# - 支持命令行独立运行（默认使用 data/bunny/ 斯坦福兔子）
# - 被 DepthMask_PointCloud_Pipeline.py 和 PointCloud_Sandbox.py 调用
#
# 依赖：numpy, open3d, Pillow

import argparse
import copy
import os

try:
    import open3d as o3d
except ModuleNotFoundError as exc:
    raise SystemExit(
        "未安装 open3d，请先执行 `pip install open3d` 或在已安装 Open3D 的 Python 环境中运行。"
    ) from exc

import numpy as np
from PIL import Image, ImageDraw, ImageOps


COMPARISON_COLORS = [
    [0.45, 0.45, 0.45],  # 原始点云：灰色
    [0.90, 0.30, 0.30],  # 阶段 1：红色
    [0.20, 0.65, 0.20],  # 阶段 2：绿色
    [0.20, 0.45, 0.95],  # 阶段 3：蓝色
    [0.80, 0.55, 0.15],  # 额外阶段：橙色
]


def print_point_count(stage_name, pcd):
    """辅助函数：打印当前点云阶段名称及点数"""
    count = len(pcd.points)
    print(f"[{stage_name}] 当前点云数量 (Points count): {count:,}")


def clone_point_cloud_with_mask(pcd, mask):
    """按布尔掩膜拷贝点云，同时保留颜色/法线属性"""
    points = np.asarray(pcd.points)
    clipped = o3d.geometry.PointCloud()

    if len(points) == 0:
        return clipped

    clipped.points = o3d.utility.Vector3dVector(points[mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        clipped.colors = o3d.utility.Vector3dVector(colors[mask])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        clipped.normals = o3d.utility.Vector3dVector(normals[mask])

    return clipped


def save_screenshot(geometries, image_path, window_name, width=1280, height=720):
    """辅助函数：将当前几何体渲染结果保存为截图"""
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name=window_name, width=width, height=height, visible=False)

        for geometry in geometries:
            vis.add_geometry(geometry)

        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])
        render_option.point_size = 3.0

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path, do_render=True)
        print(f"🖼️ 截图已保存: {image_path}")
        return True
    except Exception as exc:
        print(f"⚠️ 截图保存失败（可能是当前环境不支持离屏渲染）: {exc}")
        return False
    finally:
        vis.destroy_window()


def visualize_stage(stage_name, pcd, screenshot_path=None, show_window=True):
    """辅助函数：按阶段依次显示点云结果，并可选保存截图"""
    if screenshot_path is not None:
        save_screenshot([pcd], screenshot_path, f"{stage_name} 截图")

    if not show_window:
        return

    print(f"正在显示：{stage_name}（关闭当前窗口后继续下一步）...")
    try:
        o3d.visualization.draw_geometries([pcd], window_name=stage_name, width=1280, height=720)
    except Exception as exc:
        print(f"⚠️ 交互式可视化失败: {exc}")


def build_colored_comparison(stage_items):
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


def visualize_comparison(stage_items, screenshot_path=None, show_window=True):
    """显示多个阶段的染色并排对比图，并可选保存截图"""
    comparison_geometries, legend_lines = build_colored_comparison(stage_items)

    print("\n并排对比颜色说明：")
    for line in legend_lines:
        print(f"  - {line}")

    if screenshot_path is not None:
        save_screenshot(
            comparison_geometries,
            screenshot_path,
            "多阶段并排对比截图",
            width=1600,
            height=900
        )

    if not show_window:
        return

    print("正在显示：多阶段并排对比（关闭窗口后程序结束）...")
    try:
        o3d.visualization.draw_geometries(
            comparison_geometries,
            window_name="多阶段并排对比（不同颜色）",
            width=1600,
            height=900
        )
    except Exception as exc:
        print(f"⚠️ 并排对比显示失败: {exc}")


def create_stage_montage(image_items, output_path, tile_width=900, tile_height=520):
    """将 4 张阶段截图拼成 2x2 大图，方便整体对比"""
    if len(image_items) != 4:
        print("⚠️ 仅支持 4 张图片的 2x2 拼图，当前已跳过拼接。")
        return

    margin = 30
    title_height = 48
    border_width = 2

    tiles = []
    for title, image_path in image_items:
        if not os.path.exists(image_path):
            print(f"⚠️ 拼图跳过，未找到截图: {image_path}")
            return

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = ImageOps.contain(image, (tile_width, tile_height))

            tile = Image.new("RGB", (tile_width, tile_height + title_height), "white")
            draw = ImageDraw.Draw(tile)
            draw.text((18, 14), title, fill=(0, 0, 0))

            paste_x = (tile_width - image.width) // 2
            paste_y = title_height + (tile_height - image.height) // 2
            tile.paste(image, (paste_x, paste_y))

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

    for tile, position in zip(tiles, positions):
        canvas.paste(tile, position)

    canvas.save(output_path)
    print(f"🧩 2x2 拼图大图已保存: {output_path}")


def point_cloud_from_depth(depth_image, intrinsics, mask=None, depth_scale=1000.0, depth_min=None, depth_max=None):
    """将深度图按针孔相机模型反投影为点云，可选使用掩膜筛选目标像素"""
    depth_array = np.asarray(depth_image)
    if depth_array.ndim != 2:
        raise ValueError("深度图必须是单通道二维数组。")

    depth_m = depth_array.astype(np.float32)
    if depth_scale is not None and depth_scale > 0:
        depth_m = depth_m / float(depth_scale)

    if mask is None:
        valid_mask = np.ones(depth_m.shape, dtype=bool)
    else:
        mask_array = np.asarray(mask)
        if mask_array.shape != depth_m.shape:
            raise ValueError("掩膜尺寸必须与深度图一致。")
        valid_mask = mask_array.astype(bool)

    valid_mask &= np.isfinite(depth_m)
    valid_mask &= depth_m > 0

    if depth_min is not None:
        valid_mask &= depth_m >= depth_min
    if depth_max is not None:
        valid_mask &= depth_m <= depth_max

    fy = float(intrinsics["fy"])
    fx = float(intrinsics["fx"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    v_coords, u_coords = np.nonzero(valid_mask)
    if len(u_coords) == 0:
        return o3d.geometry.PointCloud(), valid_mask

    z = depth_m[v_coords, u_coords]
    x = (u_coords.astype(np.float32) - cx) * z / fx
    y = (v_coords.astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, valid_mask


def apply_passthrough_filter(pcd, z_min=None, z_max=None, x_range=None, y_range=None):
    """直通滤波：按 X/Y/Z 范围裁剪点云"""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return o3d.geometry.PointCloud()

    mask = np.ones(len(points), dtype=bool)

    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max

    if x_range is not None:
        x_min, x_max = x_range
        if x_min is not None:
            mask &= points[:, 0] >= x_min
        if x_max is not None:
            mask &= points[:, 0] <= x_max

    if y_range is not None:
        y_min, y_max = y_range
        if y_min is not None:
            mask &= points[:, 1] >= y_min
        if y_max is not None:
            mask &= points[:, 1] <= y_max

    return clone_point_cloud_with_mask(pcd, mask)


def apply_voxel_downsample(pcd, voxel_size):
    """体素降采样：精简点云密度"""
    if len(pcd.points) == 0:
        return o3d.geometry.PointCloud()

    if voxel_size is None or voxel_size <= 0:
        return copy.deepcopy(pcd)

    return pcd.voxel_down_sample(voxel_size=float(voxel_size))


def apply_statistical_filter(pcd, nb_neighbors, std_ratio):
    """统计滤波：剔除离群点和飞点"""
    if len(pcd.points) == 0:
        return o3d.geometry.PointCloud()

    if nb_neighbors is None or nb_neighbors < 2 or len(pcd.points) <= nb_neighbors:
        return copy.deepcopy(pcd)

    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio)
    )
    return filtered


def run_preprocessing_pipeline(pcd, z_min=None, z_max=1.0, x_range=None, y_range=None,
                               voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    """按“直通 -> 体素 -> 统计”顺序执行点云预处理"""
    pcd_passthrough = apply_passthrough_filter(
        pcd,
        z_min=z_min,
        z_max=z_max,
        x_range=x_range,
        y_range=y_range
    )
    pcd_downsampled = apply_voxel_downsample(pcd_passthrough, voxel_size=voxel_size)
    pcd_filtered = apply_statistical_filter(
        pcd_downsampled,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    stage_items = [
        ("原始点云", pcd),
        ("直通滤波后", pcd_passthrough),
        ("体素降采样后", pcd_downsampled),
        ("统计滤波后", pcd_filtered),
    ]
    return {
        "passthrough": pcd_passthrough,
        "downsampled": pcd_downsampled,
        "filtered": pcd_filtered,
        "stage_items": stage_items,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Open3D 点云预处理教学脚本")
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    default_input = os.path.join(default_data_dir, "bunny", "reconstruction", "bun_zipper_res3.ply")
    default_output = os.path.join(default_data_dir, "Open3D_output", "cup_filtered_final.pcd")

    parser.add_argument("--input", default=default_input, help="输入点云文件路径 (.ply/.pcd)")
    parser.add_argument("--output", default=default_output, help="输出点云文件路径 (.pcd)")
    parser.add_argument("--z-min", type=float, default=None, help="Z 轴最小保留值（米）")
    parser.add_argument("--z-max", type=float, default=1.0, help="Z 轴最大保留值（米）")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="体素降采样尺寸（米）")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="统计滤波近邻点数量")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="统计滤波标准差倍数")
    parser.add_argument("--no-vis", action="store_true", help="只保存结果，不弹出交互式窗口")
    parser.add_argument("--skip-screenshots", action="store_true", help="跳过阶段截图与拼图输出")
    return parser.parse_args()


def main():
    args = parse_args()

    # ==========================================
    # 步骤 0：准备输出目录与截图目录（输出到 data/output/）
    # ==========================================
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    output_dir = os.path.join(data_dir, "Open3D_output")
    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    file_path = args.input
    print(f"点云文件路径: {file_path}")

    print("正在加载本地点云文件...")
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as exc:
        print(f"读取文件失败，请检查路径是否正确: {exc}")
        return

    if len(pcd.points) == 0:
        print("读取到的点云为空，程序结束。")
        return

    raw_image_path = None if args.skip_screenshots else os.path.join(screenshot_dir, "00_raw.png")
    passthrough_image_path = None if args.skip_screenshots else os.path.join(screenshot_dir, "01_passthrough.png")
    downsampled_image_path = None if args.skip_screenshots else os.path.join(screenshot_dir, "02_downsampled.png")
    filtered_image_path = None if args.skip_screenshots else os.path.join(screenshot_dir, "03_filtered.png")
    comparison_image_path = None if args.skip_screenshots else os.path.join(screenshot_dir, "04_comparison.png")
    montage_image_path = None if args.skip_screenshots else os.path.join(screenshot_dir, "05_montage_2x2.png")

    print_point_count("0. 原始输入点云", pcd)
    visualize_stage("0. 原始点云", pcd, raw_image_path, show_window=not args.no_vis)

    pipeline_result = run_preprocessing_pipeline(
        pcd,
        z_min=args.z_min,
        z_max=args.z_max,
        voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio
    )

    pcd_passthrough = pipeline_result["passthrough"]
    pcd_downsampled = pipeline_result["downsampled"]
    pcd_filtered = pipeline_result["filtered"]
    stage_items = pipeline_result["stage_items"]

    print_point_count(
        f"1. 直通滤波后 ({args.z_min if args.z_min is not None else '-∞'} <= Z <= {args.z_max}m)",
        pcd_passthrough
    )
    visualize_stage("1. 直通滤波后", pcd_passthrough, passthrough_image_path, show_window=not args.no_vis)

    print_point_count(f"2. 体素降采样后 (Voxel={args.voxel_size}m)", pcd_downsampled)
    visualize_stage(
        f"2. 体素降采样后 (Voxel={args.voxel_size}m)",
        pcd_downsampled,
        downsampled_image_path,
        show_window=not args.no_vis
    )

    print_point_count(
        f"3. 统计滤波去噪后 (K={args.nb_neighbors}, Std={args.std_ratio})",
        pcd_filtered
    )
    visualize_stage(
        f"3. 统计滤波去噪后 (K={args.nb_neighbors}, Std={args.std_ratio})",
        pcd_filtered,
        filtered_image_path,
        show_window=not args.no_vis
    )

    # ==========================================
    # 步骤 4：保存最终结果、并排对比图和 2x2 拼图
    # ==========================================
    o3d.io.write_point_cloud(args.output, pcd_filtered)
    print(f"\n✅ 处理完成！干净的目标点云已保存至: {args.output}")

    visualize_comparison(stage_items, comparison_image_path, show_window=not args.no_vis)

    if not args.skip_screenshots:
        montage_items = [
            ("原始点云", raw_image_path),
            ("直通滤波后", passthrough_image_path),
            ("体素降采样后", downsampled_image_path),
            ("统计滤波后", filtered_image_path),
        ]
        create_stage_montage(montage_items, montage_image_path)
        print(f"\n📁 所有截图均已保存到: {screenshot_dir}")
        print(f"🧩 2x2 拼图文件: {montage_image_path}")


if __name__ == "__main__":
    main()
