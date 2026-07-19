# DepthMask_PointCloud_Pipeline.py
# 功能：深度图 + 掩膜 → 点云的核心处理管道
# - 加载深度图、掩膜、相机内参
# - 将深度图反投影为整幅场景点云
# - 用掩膜从场景中提取目标物体点云
# - 依次执行：直通滤波 → 体素降采样 → 统计滤波去噪
# - 输出三个点云文件（场景/目标原始/目标清洗）
# - 保存阶段截图、并排对比图、2×2 拼图、掩膜叠加预览图
# - 支持命令行参数配置各滤波参数
#
# 依赖：numpy, open3d, Pillow

import argparse
import json
import os
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError as exc:
    raise SystemExit(
        "未安装 open3d，请先执行 `pip install open3d` 或在已安装 Open3D 的 Python 环境中运行。"
    ) from exc

from PIL import Image

from Open3D_PointCloud_Pipeline import (
    create_stage_montage,
    point_cloud_from_depth,
    print_point_count,
    run_preprocessing_pipeline,
    save_screenshot,
    visualize_comparison,
)


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(json_path, data):
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_depth_image(depth_path):
    depth_image = Image.open(depth_path)
    depth_array = np.asarray(depth_image)

    if depth_array.ndim != 2:
        raise ValueError("depth.png 必须是单通道 16-bit 深度图。")

    return depth_array


def load_rgb_image(rgb_path):
    rgb_image = Image.open(rgb_path).convert("RGB")
    return np.asarray(rgb_image)


def load_mask_image(mask_path, expected_shape):
    mask_image = Image.open(mask_path).convert("L")
    mask_array = np.asarray(mask_image)

    if mask_array.shape != expected_shape:
        raise ValueError(
            f"mask.png 尺寸 {mask_array.shape} 与深度图尺寸 {expected_shape} 不一致。"
        )

    return mask_array > 0


def save_depth_png(depth_path, depth_array):
    depth_uint16 = np.asarray(depth_array)
    if depth_uint16.dtype != np.uint16:
        depth_uint16 = np.clip(depth_uint16, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(depth_uint16, mode="I;16").save(depth_path)


def save_rgb_png(rgb_path, rgb_array):
    rgb_uint8 = np.asarray(rgb_array)
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8, mode="RGB").save(rgb_path)


def save_mask_png(mask_path, mask_array):
    mask_uint8 = np.where(mask_array, 255, 0).astype(np.uint8)
    Image.fromarray(mask_uint8, mode="L").save(mask_path)


def colorize_depth_preview(depth_array, mask=None):
    depth_m = depth_array.astype(np.float32)
    nonzero = depth_m[depth_m > 0]
    if nonzero.size == 0:
        preview = np.zeros((*depth_array.shape, 3), dtype=np.uint8)
        return preview

    min_depth = np.percentile(nonzero, 2)
    max_depth = np.percentile(nonzero, 98)
    normalized = (depth_m - min_depth) / max(max_depth - min_depth, 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)
    gray = (255.0 * (1.0 - normalized)).astype(np.uint8)
    preview = np.stack([gray, gray, gray], axis=-1)
    preview[depth_array == 0] = [0, 0, 0]

    if mask is not None:
        preview[np.asarray(mask).astype(bool)] = [255, 96, 96]

    return preview


def ensure_demo_paths(base_dir):
    # 输入文件目录：base_dir/（用户提供的原始数据）
    # 输出文件目录：base_dir/output/（管道运行产生的所有产物）
    out = os.path.join(base_dir, "output")
    return {
        # --- 输入路径（指向 base_dir） ---
        "rgb": os.path.join(base_dir, "rgb.png"),
        "depth": os.path.join(base_dir, "depth.png"),
        "mask": os.path.join(base_dir, "mask.png"),
        "intrinsics": os.path.join(base_dir, "intrinsics.json"),
        "preview": os.path.join(base_dir, "depth_preview.png"),
        "manifest": os.path.join(base_dir, "network_manifest.json"),
        # --- 输出路径（指向 base_dir/output/） ---
        "scene_pcd": os.path.join(out, "raw_scene_from_depth.pcd"),
        "target_raw_pcd": os.path.join(out, "masked_target_raw.pcd"),
        "target_clean_pcd": os.path.join(out, "masked_target_clean.pcd"),
        "screenshots_dir": os.path.join(out, "screenshots"),
        "comparison": os.path.join(out, "screenshots", "04_pipeline_comparison.png"),
        "montage": os.path.join(out, "screenshots", "05_pipeline_montage_2x2.png"),
        "mask_overlay": os.path.join(out, "mask_overlay_preview.png"),
    }


def process_depth_and_mask(depth_array, mask_array, intrinsics, depth_scale=1000.0,
                           z_min=0.20, z_max=1.60, voxel_size=0.01,
                           nb_neighbors=20, std_ratio=1.5):
    scene_pcd, scene_valid_mask = point_cloud_from_depth(
        depth_array,
        intrinsics,
        depth_scale=depth_scale,
        depth_min=z_min,
        depth_max=z_max
    )
    target_raw_pcd, target_valid_mask = point_cloud_from_depth(
        depth_array,
        intrinsics,
        mask=mask_array,
        depth_scale=depth_scale,
        depth_min=z_min,
        depth_max=z_max
    )

    pipeline_result = run_preprocessing_pipeline(
        target_raw_pcd,
        z_min=z_min,
        z_max=z_max,
        voxel_size=voxel_size,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    target_passthrough = pipeline_result["passthrough"]
    target_downsampled = pipeline_result["downsampled"]
    target_clean = pipeline_result["filtered"]

    stage_items = [
        ("整幅场景原始点云", scene_pcd),
        ("掩膜目标原始点云", target_raw_pcd),
        ("目标直通滤波后", target_passthrough),
        ("目标体素降采样后", target_downsampled),
        ("目标统计滤波后", target_clean),
    ]

    return {
        "scene_pcd": scene_pcd,
        "target_raw_pcd": target_raw_pcd,
        "target_passthrough": target_passthrough,
        "target_downsampled": target_downsampled,
        "target_clean_pcd": target_clean,
        "scene_valid_mask": scene_valid_mask,
        "target_valid_mask": target_valid_mask,
        "stage_items": stage_items,
    }


def print_processing_summary(mask_array, result):
    print(f"有效深度像素数（整幅场景）: {int(result['scene_valid_mask'].sum()):,}")
    print(f"掩膜前景像素数: {int(np.asarray(mask_array).astype(bool).sum()):,}")
    print(f"掩膜内有效深度像素数: {int(result['target_valid_mask'].sum()):,}")

    print_point_count("0. 整幅深度图反投影", result["scene_pcd"])
    print_point_count("1. 掩膜提取目标原始点云", result["target_raw_pcd"])
    print_point_count("2. 目标点云直通滤波后", result["target_passthrough"])
    print_point_count("3. 目标点云体素降采样后", result["target_downsampled"])
    print_point_count("4. 目标点云统计滤波后", result["target_clean_pcd"])


def save_point_cloud_outputs(result, paths):
    os.makedirs(os.path.dirname(paths["scene_pcd"]), exist_ok=True)
    o3d.io.write_point_cloud(paths["scene_pcd"], result["scene_pcd"])
    o3d.io.write_point_cloud(paths["target_raw_pcd"], result["target_raw_pcd"])
    o3d.io.write_point_cloud(paths["target_clean_pcd"], result["target_clean_pcd"])

    print("\n✅ 点云文件已输出：")
    print(f"  - 整幅场景点云: {paths['scene_pcd']}")
    print(f"  - 掩膜目标原始点云: {paths['target_raw_pcd']}")
    print(f"  - 掩膜目标干净点云: {paths['target_clean_pcd']}")


def save_mask_overlay_preview(output_path, preview_rgb, mask_array):
    overlay = np.asarray(preview_rgb).copy()
    mask_bool = np.asarray(mask_array).astype(bool)
    overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * np.array([255, 64, 64])).astype(np.uint8)
    save_rgb_png(output_path, overlay)


def save_pipeline_screenshots(stage_items, paths, show_window):
    if not os.path.exists(paths["screenshots_dir"]):
        os.makedirs(paths["screenshots_dir"], exist_ok=True)

    raw_scene_image = os.path.join(paths["screenshots_dir"], "00_scene_raw.png")
    target_raw_image = os.path.join(paths["screenshots_dir"], "01_target_raw.png")
    target_passthrough_image = os.path.join(paths["screenshots_dir"], "02_target_passthrough.png")
    target_clean_image = os.path.join(paths["screenshots_dir"], "03_target_clean.png")

    save_screenshot([stage_items[0][1]], raw_scene_image, "整幅深度图反投影点云")
    save_screenshot([stage_items[1][1]], target_raw_image, "掩膜目标原始点云")
    save_screenshot([stage_items[2][1]], target_passthrough_image, "直通滤波后目标点云")
    save_screenshot([stage_items[4][1]], target_clean_image, "清洗后目标点云")

    comparison_items = [
        stage_items[0],
        stage_items[1],
        stage_items[2],
        stage_items[4],
    ]
    visualize_comparison(comparison_items, paths["comparison"], show_window=show_window)

    montage_items = [
        ("整幅深度图反投影", raw_scene_image),
        ("掩膜目标原始点云", target_raw_image),
        ("目标直通滤波后", target_passthrough_image),
        ("目标清洗后", target_clean_image),
    ]
    create_stage_montage(montage_items, paths["montage"])


def run_depth_mask_pipeline(depth_array, mask_array, intrinsics, output_dir,
                            depth_scale=1000.0, rgb_image: Optional[np.ndarray] = None,
                            preview_rgb: Optional[np.ndarray] = None,
                            z_min=0.20, z_max=1.60, voxel_size=0.01,
                            nb_neighbors=20, std_ratio=1.5,
                            save_intermediate_inputs=True,
                            skip_screenshots=False,
                            show_window=False):
    paths = ensure_demo_paths(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    result = process_depth_and_mask(
        depth_array,
        mask_array,
        intrinsics,
        depth_scale=depth_scale,
        z_min=z_min,
        z_max=z_max,
        voxel_size=voxel_size,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    print_processing_summary(mask_array, result)
    save_point_cloud_outputs(result, paths)

    if save_intermediate_inputs:
        save_depth_png(paths["depth"], depth_array)
        save_mask_png(paths["mask"], mask_array)
        save_json(paths["intrinsics"], {
            **intrinsics,
            "depth_scale": depth_scale,
            "image_width": int(depth_array.shape[1]),
            "image_height": int(depth_array.shape[0]),
        })

        if preview_rgb is None:
            preview_rgb = colorize_depth_preview(depth_array, mask=mask_array)
        if rgb_image is not None:
            save_rgb_png(paths["rgb"], rgb_image)
        save_rgb_png(paths["preview"], preview_rgb)
        save_mask_overlay_preview(paths["mask_overlay"], preview_rgb, mask_array)

    if not skip_screenshots:
        save_pipeline_screenshots(result["stage_items"], paths, show_window=show_window)
        print(f"📁 阶段截图目录: {paths['screenshots_dir']}")
        print(f"🧩 拼图文件: {paths['montage']}")

    return {
        **result,
        "paths": paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="从深度图和掩膜生成目标干净点云")
    default_demo_dir = os.path.join(os.path.dirname(__file__), "data", "clutter_depth_demo")

    parser.add_argument("--demo-dir", default=default_demo_dir, help="示例数据目录")
    parser.add_argument("--z-min", type=float, default=0.20, help="直通滤波 Z 最小值（米）")
    parser.add_argument("--z-max", type=float, default=1.60, help="直通滤波 Z 最大值（米）")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="体素降采样尺寸（米）")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="统计滤波近邻数")
    parser.add_argument("--std-ratio", type=float, default=1.5, help="统计滤波标准差倍数")
    parser.add_argument("--no-vis", action="store_true", help="只保存文件，不弹出交互式窗口")
    parser.add_argument("--skip-screenshots", action="store_true", help="跳过阶段截图与拼图")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ensure_demo_paths(args.demo_dir)

    required_files = [paths["depth"], paths["mask"], paths["intrinsics"]]
    for required_file in required_files:
        if not os.path.exists(required_file):
            raise FileNotFoundError(f"未找到必需输入文件: {required_file}")

    intrinsics_config = load_json(paths["intrinsics"])
    depth_scale = float(intrinsics_config.get("depth_scale", 1000.0))
    intrinsics = {
        "fx": intrinsics_config["fx"],
        "fy": intrinsics_config["fy"],
        "cx": intrinsics_config["cx"],
        "cy": intrinsics_config["cy"],
    }

    depth_array = load_depth_image(paths["depth"])
    mask_array = load_mask_image(paths["mask"], depth_array.shape)

    preview_rgb = None
    if os.path.exists(paths["rgb"]):
        preview_rgb = load_rgb_image(paths["rgb"])
    elif os.path.exists(paths["preview"]):
        preview_rgb = load_rgb_image(paths["preview"])

    print(f"深度图路径: {paths['depth']}")
    print(f"掩膜路径: {paths['mask']}")
    print(f"相机内参路径: {paths['intrinsics']}")
    print(f"深度图尺寸: {depth_array.shape[1]} x {depth_array.shape[0]}")

    run_depth_mask_pipeline(
        depth_array,
        mask_array,
        intrinsics,
        output_dir=args.demo_dir,
        depth_scale=depth_scale,
        preview_rgb=preview_rgb,
        z_min=args.z_min,
        z_max=args.z_max,
        voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        save_intermediate_inputs=False,
        skip_screenshots=args.skip_screenshots,
        show_window=not args.no_vis,
    )


if __name__ == "__main__":
    main()
