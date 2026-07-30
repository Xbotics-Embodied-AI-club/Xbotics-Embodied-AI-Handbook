# interactive_depth_pipeline.py
# 功能：交互式深度管道统一入口脚本
# - 整合四种输入源：文件(file)、USB深度相机(usb)、网络(network)
# - 通过 interactive_mask_ui 提供的 OpenCV 窗口进行单击/框选交互
# - 通过 sam2_segmenter 进行分割（含 fallback 回退掩膜）
# - 调用 DepthMask_PointCloud_Pipeline 完成点云提取全流程
# - 自动保存中间输入和网络 manifest（network 模式）
#
# 依赖：numpy, opencv-python, open3d, Pillow, 可选 sam2/torch

import argparse
import os
from contextlib import suppress

import numpy as np

from DepthMask_PointCloud_Pipeline import (
    colorize_depth_preview,
    ensure_demo_paths,
    run_depth_mask_pipeline,
    save_json,
)
from depth_sources import create_depth_source
from interactive_mask_ui import InteractiveMaskSelector, preview_mask_overlay
from sam2_segmenter import Sam2Segmenter


def ensure_preview_image(frame):
    if frame.rgb is not None:
        return frame.rgb
    return colorize_depth_preview(frame.depth)


def save_network_manifest(output_dir, paths, intrinsics, depth_scale):
    manifest = {
        "rgb": os.path.basename(paths["rgb"]) if os.path.exists(paths["rgb"]) else None,
        "depth": os.path.basename(paths["depth"]),
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
    }
    save_json(paths["manifest"], manifest)


def run_interactive_pipeline(source_type, output_dir, selector, segmenter,
                             base_dir=None, manifest_path=None, device="auto",
                             z_min=0.20, z_max=1.60, voxel_size=0.01,
                             nb_neighbors=20, std_ratio=1.5,
                             skip_screenshots=False, show_window=True):
    source = create_depth_source(
        source_type,
        base_dir=base_dir,
        manifest_path=manifest_path,
        device=device,
    )

    try:
        while True:
            frame = source.get_frame()
            preview_rgb = ensure_preview_image(frame)

            interaction = selector.select(preview_rgb)
            if interaction is None:
                print("已取消当前选择。")
                return None

            if interaction.mode == "point":
                segmentation = segmenter.segment(preview_rgb, point_xy=interaction.point)
            elif interaction.mode == "box":
                segmentation = segmenter.segment(preview_rgb, box_xyxy=interaction.box_xyxy)
            else:
                raise ValueError(f"未知交互模式: {interaction.mode}")

            decision = preview_mask_overlay(preview_rgb, segmentation.mask)
            if decision == "reselect":
                print("用户选择重新选目标，返回选择界面。")
                continue
            if decision == "cancel":
                print("用户取消本次分割结果。")
                return None

            result = run_depth_mask_pipeline(
                frame.depth,
                segmentation.mask,
                frame.intrinsics,
                output_dir=output_dir,
                depth_scale=frame.depth_scale,
                rgb_image=frame.rgb,
                preview_rgb=preview_rgb,
                z_min=z_min,
                z_max=z_max,
                voxel_size=voxel_size,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
                save_intermediate_inputs=True,
                skip_screenshots=skip_screenshots,
                show_window=show_window,
            )

            paths = ensure_demo_paths(output_dir)
            if source_type == "network":
                save_network_manifest(output_dir, paths, frame.intrinsics, frame.depth_scale)

            print(f"\n✅ 交互式目标点云提取完成，输出目录: {output_dir}")
            return result
    finally:
        with suppress(Exception):
            source.close()


def parse_args():
    parser = argparse.ArgumentParser(description="交互式深度图/SAM2/点云提取统一入口")
    default_demo_dir = os.path.join(os.path.dirname(__file__), "data", "rgbd_object_demo")

    parser.add_argument("--source", choices=["file", "usb", "network"], default="file", help="输入源类型")
    parser.add_argument("--input-dir", default=default_demo_dir, help="文件输入源目录")
    parser.add_argument("--output-dir", default=default_demo_dir, help="输出目录")
    parser.add_argument("--manifest", default=None, help="网络输入 manifest 路径")
    parser.add_argument("--device", default="auto", help="USB 深度相机设备序列号或 auto")
    parser.add_argument("--sam2-checkpoint", default=None, help="SAM2 checkpoint 路径")
    parser.add_argument("--sam2-config", default=None, help="SAM2 config 路径")
    parser.add_argument("--sam2-device", default="cpu", help="SAM2 推理设备，例如 cpu/cuda")
    parser.add_argument("--z-min", type=float, default=0.20, help="直通滤波 Z 最小值（米）")
    parser.add_argument("--z-max", type=float, default=1.60, help="直通滤波 Z 最大值（米）")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="体素降采样尺寸（米）")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="统计滤波近邻数")
    parser.add_argument("--std-ratio", type=float, default=1.5, help="统计滤波标准差倍数")
    parser.add_argument("--skip-screenshots", action="store_true", help="跳过阶段截图与拼图")
    parser.add_argument("--no-vis", action="store_true", help="不显示 Open3D 3D 结果窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    selector = InteractiveMaskSelector()
    segmenter = Sam2Segmenter(
        checkpoint_path=args.sam2_checkpoint,
        config_path=args.sam2_config,
        device=args.sam2_device,
    )

    run_interactive_pipeline(
        source_type=args.source,
        output_dir=args.output_dir,
        selector=selector,
        segmenter=segmenter,
        base_dir=args.input_dir,
        manifest_path=args.manifest,
        device=args.device,
        z_min=args.z_min,
        z_max=args.z_max,
        voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        skip_screenshots=args.skip_screenshots,
        show_window=not args.no_vis,
    )


if __name__ == "__main__":
    main()
